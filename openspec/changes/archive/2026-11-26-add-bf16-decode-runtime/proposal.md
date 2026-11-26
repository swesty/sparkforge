# Proposal

## Why

Every throughput number so far was per kernel: the sum of isolated kernel times predicted 12.2 tok/s for Llama 3.1 8B at 32K context. The real figure depends on launch gaps and wiring, which only an end-to-end run measures (SHA-755).

## What Changes

- **`sparkforge generate`:** greedy decode of a dense BF16 checkpoint from pre-tokenized prompt ids. Each step is one CUDA Graph, prefill runs as repeated decode steps, and `--tokenizer tokenizer.json` decodes the output.
  - `--check N` compares the first N steps' logits with the CPU reference streamed from the checkpoint.
  - `--bench-ctx` times graph replays at given context lengths.
- **A static step graph:**
  - `rope_kv_append` and `attn_decode_split` read the position from device memory.
  - New `embed_gather` looks up the token from device memory.
  - New `argmax_advance` writes the greedy token, records it and increments the position.
- **`forward::Runtime`:** owns the KV caches, activations, device token and position, and the captured graph. It takes weight addresses (`WeightPtrs`), so checkpoint weights (`device_weights::upload_dense`, now also copying norms and the embedding) and random test weights share it. The KV capacity is private, and every step is checked against it.
- **Admission:** `weights::admit_runtime` covers every tensor plus the KV caches, activations and token state.
- **CPU reference:** reads weights through a `Weights` trait, either random `DenseWeights` or `CheckpointWeights`, which streams via `weights::reference_gemv`. GB10's unified memory can't hold a second host copy of a model resident on the GPU.
- **`tokenizer::Decoder`:** ids to text for byte-level BPE `tokenizer.json`.
- **`verify`:** adds `embed_gather` and `argmax_advance` checks, and replays the decode step from a captured graph (bitwise against plain launches).
- **codex-review follow-ups from #25:**
  - `fp:a3b68a20e8a6`: the KV capacity is private.
  - `fp:29a5f1998955`: `reference::decode_step` validates before writing, and returns errors.
  - `fp:bd3fa010f065`: `f64_to_bf16` rounds bf16 subnormals once.

- **codex-review cycle 1 on this change:**
  - `fp:b4f0ef0952f4`: the prompt's last step gives the first output token, so N tokens take prompt + N - 1 replays (a run used to make one extra replay and discard it).
  - `fp:83bcb9e0f559`: a step whose token comes from the previous step needs a full embedding table.
  - `fp:c7921f52e588`: `WeightPtrs` is built only by the owners of the memory, and the `Runtime` borrows the owner.
  - `fp:96a508102cf4`: a reference step that fails partway restores the slots it wrote.
  - `fp:207ec0ac8dbc`: each benchmark starts from a zeroed cache and the same token.
  - `fp:dc612d7ba3fc`: the tokenizer stores entries in a map and refuses ids past `u32`.
  - `fp:0f8e3bef027c`: argmax lets negative infinity win over NaN.
  - `fp:36f5a8a8b697`: `CheckpointWeights` checks every bound shape against the model.

**Expected GB10 effect:** Llama 3.1 8B decodes at 15.5, 15.0 and 12.1 tok/s at 1K, 4K and 32K, within 1% of the planner ceiling.

## Non-goals

- Prefill GEMMs (SHA-751), kernel fusion and PDL (SHA-750), sampling other than greedy, and tokenizing text.

## Capabilities

### Modified Capabilities
- `gpu-runtime`: greedy decode with one graph per step, its admission, and throughput measurement.
- `gpu-verification`: the token kernels, graph replay, and logits against the reference on real weights.
- `kernel-codegen`: the position comes from device memory; `embed_gather` and `argmax_advance`.

## Impact

- New: `src/tokenizer.rs`, `tests/tokenizer.rs`.
- Changed: `src/forward.rs`, `src/reference.rs`, `src/codegen.rs`, `src/verify.rs`, `src/device_weights.rs`, `src/weights.rs`, `src/main.rs`, `src/lib.rs`, `tests/reference.rs`.
