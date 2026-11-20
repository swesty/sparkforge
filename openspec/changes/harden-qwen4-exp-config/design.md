# Design

## Context

See proposal.md (Why). Each decision below answers one codex-review finding on swesty/sparkforge#9 at `6157a77`.

## Decisions

1. **Keep `quantized_layers` as raw JSON** (`fp:46f9e2b58e8a`). Each entry is decoded only inside the MIXED_PRECISION branch of format detection, and the error names the entry. With `--format`, detection doesn't run, so a map this parser can't read no longer blocks an override. This matches the behavior before #9.
2. **`declared_mixed` records what the checkpoint is, separately from what the planner uses** (`fp:e3726e909d4d`). It is set from `quant_method`/`quant_algo` alone, whatever the override. `bind` refuses when it is set. It is omitted from JSON when false, so dense plans don't change.
3. **The output head has fixed paths** (`fp:dba130190267`). `lm_head` matches only a language-model prefix plus `lm_head` (`lm_head`, `model.lm_head`, …). Other model-level lookups (the PLE table and projections) match the module path on a `.` boundary, so `aux_ple.key_proj` is not `ple.key_proj`.
4. **The MTP key and value projections are separate lookups** (`fp:a8a58ba6665f`). This replaces doubling the key term.
5. **The family discriminator falls back to the top level** (`fp:e1830456f788`). The nested `model_type` is more specific (`qwen4_exp_text` under `qwen4_exp`), so the top-level value only fills a missing or null nested one, with no conflict check.
6. **MoE top-k range** (`fp:1246d9fbf482`). `1 ≤ num_experts_per_tok ≤ num_experts` applies to every MoE family. It holds for the two example configs.
7. **Refuse before CUDA** (`fp:6cddef73fc82`). The host-side `codegen::require_generated` runs in `main` before `Engine::load`, and again inside `verify::run` for library callers. Checked on a Mac with no CUDA: Flash Next `verify` prints the planned-only error.

8. **Round 1 of codex-review on this change.**
   - The whole `quantization_config` is raw JSON, decoded into `RawQuant` only during format detection. `declared_mixed` reads the raw `quant_method`/`quant_algo` strings, so it holds even when other fields are malformed.
   - PLE lookups (`Formats::ple`) match only `ple.<m>` or `layers.<n>.ple.<m>` under a text prefix. MTP lookups match only `mtp.layers.<n>.<m>`. The suffix-based `Formats::model` is removed.
   - The MTP shared expert's gate, up and down, and its router, are priced at their own formats.

9. **Round 2 of codex-review on this change.**
   - The PLE table also matches at its model-level path (`ple_embedding.ngram_embedding` under a text prefix).
   - Layer-qualified PLE entries match only at a configured PLE layer `p`, or `p − 1`, the layer the checkpoint files the PLE applied before layer 2 under (`layers.1.ple`). A stale index such as `layers.999` stays unused.
   - The MTP shared-expert projections are looked up only when the shared expert exists, so stale entries for an absent one stay reported. The router is counted either way.

## Risks / Trade-offs

- **[Lazy decoding means a bad map entry surfaces later]** → It still surfaces at parse time whenever no override is given, naming the entry.
