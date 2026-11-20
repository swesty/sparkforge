# Design

## Context

See proposal.md (Why). Sources for every number:
- The checkpoint's `config.json` at `local-inference-lab/Qwen3.8-Flash-Next-NVFP4@6909a5bed089`, checked in verbatim.
- Its `model.safetensors.index.json` (tensor names).
- The safetensors header sums recorded in the Linear plan document "Qwen3.8 Flash Next engine plan (2026-11-20)".

## Decisions

1. **`text_config` wins; the top level supplies only `quantization_config` and `architectures`.** The vision tower's fields are ignored. Dense configs without `text_config` parse exactly as before.
2. **Hybrid is opt-in by `model_type`.** A `layer_types` schedule containing `linear_attention` is accepted only for `qwen4_exp(_text)`. Qwen3-Next shares the name but lacks HyperConnection, PLE and the indexer, so it is rejected rather than planned with missing math.
   - A schedule of only `full_attention` is treated as dense.
   - The schedule must match `num_hidden_layers` and, when present, `full_attention_interval`.
   - `mrope_section` must cover the rotated dims (`2 × Σ = head_dim × partial_rotary_factor`).
3. **`MIXED_PRECISION` becomes a per-module map, not a single format.** `ModelSpec.weights` is the default (BF16), and `ModelSpec.formats` holds the 435 listed modules. Accepted schemes:

   | Scheme | Mapped to | Group size |
   |---|---|---|
   | `NVFP4` | `Nvfp4` | 16 |
   | `W4A16_NVFP4` | `Nvfp4WeightOnly` (same bytes; activations stay BF16) | 16 |
   | `MXFP8` | `Mxfp8` (1 + 1/32 B per element) | 32 |

   Any other scheme or group size is an error naming the module. `--format` still replaces detection entirely, map included.
4. **The planner looks up formats by path suffix and reports what it never used.**
   - Main-model lookups match `layers.{i}.{module}` outside `mtp.`. MTP and model-level modules match by suffix.
   - Every map entry a lookup touches is marked used. The checkpoint lists the MTP experts under two paths, and both count.
   - A remaining entry becomes a plan note. A checkpoint with an unfamiliar path layout therefore can't fall back to BF16 silently.
5. **One op class per distinct kernel.** The checkpoint has about 900 per-layer op instances per decode token, but they collapse into 34 classes with identical shape and format, so one generated kernel serves each class.
   - **HyperConnection:** `hc_down_inject` (N = 324: the 320 low-rank rows plus the 4 injection logits for the next combine, ×96), `hc_down_final` (N = 320, ×1, because the final mixer has no `block_inject_weight`), and `hc_up` (×97).
   - **PLE projections:** run once per token. `key_proj` widens to the residual width (`streams × hidden` = 10,240, matching the checkpoint's `[10240, 2560]`).
6. **Budget terms.**
   - **Context-free:** every linear's active bytes, the FP32 Gated DeltaNet state read and written, the conv1d state and weights, HC norm weights, the embedding row and the PLE rows.
   - **Per context token:** FP8 KV gathered for `min(ctx, budget + compress_ratio − 1)` positions, plus the indexer scan of `ceil(ctx / compress_ratio)` FP8 keys, in each sparse-attention layer.
   - **Resident:** all expert copies, the PLE table (`heads × ngram_vocab_size_base` rows, which is within a few ppm of the checkpoint's prime-sized tables), the MTP layer and the small parameters. Gated DeltaNet state is fixed memory.
   - **KV:** planned as FP8 (the checkpoint sets no `kv_cache_quant_algo`, and the fleet engine runs FP8 KV).
7. **Nothing is generated for the hybrid layout yet.**
   - Codegen's dense kernels would be wrong for it: full-head RoPE, dense attention, BF16-only GEMVs. So every op class is listed as `planned` with no source.
   - `verify` refuses the model, and `bind` refuses hybrid or mixed-precision checkpoints.
   - Dense plans and generated code are unchanged: the Llama and Qwen3-30B-A3B cache keys equal `main`'s.

## Risks / Trade-offs

- **[The PLE tensors live under `layers.1.ple` while the config says `ple_layer_ids [2]`]** → This is recorded here for the loader ticket (SHA-767). The planner only needs the per-token cost.
- **[Unread model math: router normalization, indexer compression, PLE apply, HC SiLU scale]** → These don't affect shapes or bytes. They are noted in the plan and resolved by the kernel tickets.
- **[The indexer cache dtype is assumed FP8]** → The indexer scan is 12.6 MB of 7,375 MB at 32K, so a BF16 cache would move the ceiling by 0.2%.
