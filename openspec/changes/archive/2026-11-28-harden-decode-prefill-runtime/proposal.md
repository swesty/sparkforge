# Proposal

## Why

codex-review approved the decode runtime (#26), the BF16 prefill GEMM (#27) and dense prefill attention (#28) with fifteen follow-ups between them (SHA-824). None of them changed a measured result, because the CLI always passes matching values. But each one is a way the runtime's library API, its options or an unusual input can silently give wrong memory accounting, wrong reads or wrong output:

- A runtime accepts kernels generated for another model, or a split count other than the one compiled into the attention kernel.
- A failed step can leave the host's copy of the position behind the device's.
- `prefill::gemm` can launch one projection's kernel with another's checks, drops cudarc's cross-stream tracking, and has a safe raw-address variant.
- A NaN in a future value can reach an earlier row of prefill attention through 0 × NaN.
- `generate --check` allocates its host reference cache outside admission.
- The CPU reference can read checkpoint slices with a different model than it validated.
- Undefined token ids decode to nothing.
- The decode benchmark compares its timed positions with the ceiling at another context length.
- `verify` passes a planned prefill kernel that is missing, and its one-row misalignment probe never reaches the alignment check.
- The MoE prefill estimate charges every expert for a prompt too short to reach them.

## What Changes

- **Runtime binding:**
  - `forward::Kernels::new(plan, generated, engine)` and `PrefillKernels` record the model they were generated for and the CUDA context they were loaded in.
  - `Runtime::new`, `Runtime::from_host` and `Runtime::enable_prefill` refuse a bundle for another model, or one loaded in another context, before allocating anything.
  - The attention scratch is sized from the split count compiled into `attn_decode_split` (its grid), and the `splits` parameter is gone.
- **Step state:** the host's record of the device position and token moves to `cursor::Cursor`, which has host-only tests. A step, traced step or prefill that fails after its launches began marks the runtime stale, and it refuses every step until `set`.
- **Prefill GEMM launch:**
  - `prefill::Gemm::new(plan, generated, engine, linear)` binds a GEMM's tile, manifest entry and loaded function by one name.
  - `prefill::gemm` passes its views to the launch, so cudarc records the write.
  - The raw-address launcher is `unsafe fn gemm_raw`, with its obligations documented.
  - Both check the stream's context.
- **Prefill attention NaN guard:**
  - In a tile holding keys some row of the block doesn't attend, the barrier that makes the tile resident also asks whether any of its V values isn't finite (`__syncthreads_or`).
  - If one is, the block zeroes those values in shared memory and records per dimension the first key that had one.
  - At the end, each row is set to NaN at the dimensions where it attends such a key.
  - Shared memory grows by 4 bytes per head dimension.
  - The guard costs about 2–3% of attention throughput.
- **`generate`:**
  - Admission counts the host reference cache `--check` keeps for the whole prompt (1 GiB for 8192 tokens). The cache is allocated, with every page touched, before admission is released.
  - `--bench-ctx` compares its timed steps with the harmonic mean of the planner's ceilings over the same positions (`plan::decode_ceiling_over`).
  - Undefined token ids print as `<unknown token ID>`, with a warning.
- **CPU reference:** `CheckpointWeights::check` requires the model it reads with to have the step model's dimensions.
- **`verify`:**
  - It fails, naming the kernel, when a planned prefill kernel is missing from the manifest (`codegen::missing_prefill_kernels`), or when `PrefillKernels` refuses the bundle.
  - The misalignment probe uses one row of a buffer with room for it, and requires the A-alignment error.
  - The largest GEMM runs with event tracking on, and a copy on a second stream must see its finished output.
  - Prefill attention keeps earlier rows bitwise when V, or K and V, past a row are NaN, and makes every later row NaN.
  - Kernels for a model with half the heads and twice the head dimension are refused.
- **Estimate:** the MoE prefill estimate charges min(experts, tokens × `experts_per_tok`) expert GEMM pairs.

## Impact

- Code: `src/forward.rs`, `src/prefill.rs`, `src/cursor.rs` (new), `src/codegen.rs`, `src/plan.rs`, `src/estimate.rs`, `src/reference.rs`, `src/tokenizer.rs`, `src/verify.rs`, `src/main.rs`.
- Tests:
  - `tests/cursor.rs` (new);
  - estimate, span ceiling and missing-kernel tests in `tests/plan.rs`;
  - the reference cache price in `tests/reference.rs`;
  - the reference model check in `tests/weights.rs`;
  - unknown ids in `tests/tokenizer.rs`;
  - the golden plans (`prefill_attn.smem_bytes` 69,632 to 70,144).
- API: `Runtime::new` and `from_host` lose `splits`. `Kernels::new` takes the plan. `prefill::gemm` takes a `Gemm`. `gemm_at` became `unsafe fn gemm_raw`.
