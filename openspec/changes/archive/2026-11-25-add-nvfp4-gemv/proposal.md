# Proposal

## Why

Batch-1 decode on GB10 is limited by DRAM bandwidth, so the bytes read per token set the speed. NVFP4 weights take 0.5625 bytes per element against BF16's 2, but sparkforge has no kernel that reads them. Today every NVFP4 GEMV is `planned`:
- the Qwen3-30B-A3B NVFP4 plan (5.40 GB per token, a 43.5 tok/s ceiling) cannot run;
- in Qwen3.8 Flash Next, the routed experts are 1,327 MB of the 7,338 MB read per decode token, and nothing can run them.

SHA-791 now places the Flash Next experts in GPU memory in the tile layout this GEMV reads, so the GEMV is the next step on milestone M2.

## What Changes

- **NVFP4 decode GEMV (W4A16).** A generated `gemv_<linear>` for NVFP4 and NVFP4 weight-only linears.
  - It reads the load-time tile layout (`src/repack.rs`): 16 K x 64 N tiles of 512 bytes, with per-16 scales as S0E5M3.
  - It decodes E2M1 pairs with `cvt.rn.f16x2.e2m1x2` and accumulates in f32 against BF16 activations.
  - It applies the per-16 scale once per K-tile and row, then a per-row global scale.
  - It uses no shared memory, so K has no staging limit.
  - The design is FlashInfer's `GemvBf16Fp4Sm12x` (flashinfer@db23734, Apache-2.0), re-derived for GB10's 48 SMs and 1,536 threads per SM, with every shape a `constexpr`.
- **Grouped variant.** One launch covers every active expert of a layer:
  - expert ids are read from a device array;
  - the weights come from the per-layer grouped buffers SHA-791 builds;
  - the global scales come from its `[E, 3]` `weight_scale_2` buffer.
  This is the base for SHA-758 (router top-k and the full MoE launch).
- **Deterministic split-K in one launch.** K is split across `grid.y` until about 20 warps per SM are busy.
  - Partials go to an f32 workspace.
  - The last warp to arrive for each N tile sums them in a fixed split order and resets its arrival counter.
  - The output is therefore bitwise reproducible, and the kernel can be replayed in a CUDA graph.
- **Planner.** An NVFP4 GEMV gets a split count, a 3-D grid, and its workspace and counter sizes. These rules derive from `target.rs` and the shape alone. BF16 GEMV plans and their JSON are unchanged.
- **Codegen.** NVFP4 GEMVs whose shape tiles (N a multiple of 64, K of 16) are `generated`.
  - A Flash Next plan now generates its two routed-expert GEMVs. Every other hybrid op stays `planned`.
- **Verification.** `sparkforge verify` checks each NVFP4 GEMV against an f64 reference.
  - The reference dequantizes the checkpoint-format bytes, not the repacked ones.
  - The random scales include zero and subnormal values.
  - Each global-scale segment gets a distinct value.
  - The grouped kernel is given unordered, non-contiguous expert ids.
  - A second launch must match the first bit for bit.
  - With `--bench`, the grouped kernel is timed on a fresh random expert set per launch, drawn from all experts. GB/s counts weight and scale bytes.

**Expected GB10 effect.** M2's bar is at least 225 GB/s for GEMVs of 10 MB or more, at an error of at most 1/64.
- **Flash Next, `gemv_expert_gate_up`:** 18.4 MB per layer for 10 experts, so 82 µs at 225 GB/s.
- **Flash Next, `gemv_expert_down`:** 9.2 MB per layer, so about 46 µs at 200 GB/s. It is below 10 MB, so the measured number is recorded but not gated.
- **Flash Next total:** at 225 GB/s, the routed experts cost 5.9 ms per token of the 33 tok/s ceiling's 30 ms.
- **Qwen3-30B-A3B NVFP4:** `verify --bench` can now print a per-token estimate. It is compared with the planner's 43.5 tok/s ceiling.

All of these numbers are measured with `cargo run --release --features cuda -- verify <config> --bench` on spark-5 and recorded in the PR with the host and CUDA version.

## Non-goals

- **PDL and loads before `griddepcontrol.wait`:** SHA-750.
- **Fusions:** a SiLU-and-mul epilogue and an RMSNorm prologue (M5).
- **W4A4:** activation quantization and FP4 tensor cores. The decode GEMV keeps activations in BF16 and does not use `input_scale`.
- **M ≥ 2 decode:** MTP verify steps and batches (SHA-753, SHA-775).
- **Routing kernels:** the router top-k kernel and the MoE step (SHA-758).
- **Real-checkpoint expert checks on the GPU:** these come with the golden diff (SHA-773). Loading dense NVFP4 checkpoints is also out; `verify-weights` stays BF16.
- **MXFP8 GEMVs:** SHA-768.
- **Engine memory budget:** this change sizes the workspace but leaves the engine-wide budget to the inference-engine change (task 2.2).

## Capabilities

### New Capabilities
- None.

### Modified Capabilities
- `kernel-codegen`: NVFP4 GEMVs become `generated` with a defined layout, math, split-K merge and grouped variant. Hybrid plans generate their NVFP4 expert GEMVs.
- `kernel-planning`: NVFP4 decode GEMVs get a split-K and grid rule, and the hybrid plan note no longer says nothing is generated.
- `gpu-verification`: NVFP4 GEMVs get their inputs, reference, determinism check and timing. The planned-only refusal scenario moves to a hybrid config with no NVFP4 experts.

## Impact

- **Source:**
  - `src/codegen.rs`: FP4 GEMV template and generated status;
  - `src/plan.rs`: FP4 GEMV plan fields and split rule;
  - `src/verify.rs`: FP4 checks and timing;
  - `src/repack.rs`: the lane map shared by the kernel and its host model.
- **Tests:** `tests/plan.rs`, `tests/flash_next.rs`, `tests/repack.rs`, and a new host-side emulation of the kernel's indexing and split-K order.
- **Plans and manifests:** BF16 plan JSON is unchanged (`tests/golden/*.plan.json`). Flash Next and NVFP4 manifests gain `generated` entries and new argument lists.
- **Dependencies:** none new. Kernels compile with NVRTC for `sm_121a` as today. ptxas in CUDA 13.0.88 on spark-5 accepts `cvt.rn.f16x2.e2m1x2` for `sm_121a`; each byte compiles to one `F2FP.F16.E2M1.UNPACK_B` instruction.
