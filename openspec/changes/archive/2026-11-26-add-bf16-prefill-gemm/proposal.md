# Proposal

## Why

The planner tiled every prefill GEMM, but each `gemm_<linear>` was `planned`: prompt processing had no tensor-core path, and `generate` prefills a 2048-token prompt one decode step at a time, at about 65 ms per token (SHA-751).

## What Changes

- **Kernel** (`codegen.rs`): `gemm_<linear>(A, W, C, M)`, C = A · Wᵀ in bf16 with f32 accumulation.
  - N, K and the tile are `constexpr`; the prompt length M is an argument, and rows past M are zero-filled on load and not stored.
  - A `cp.async` pipeline of the planned stages feeds shared memory, with rows padded by 16 bytes so 8-row `ldmatrix` reads are conflict-free at BK 32 and 64.
  - Each warp does 4 x (WN/8) `mma.sync m16n8k16`.
  - Generated for BF16 linears whose tile divides N and K; NVFP4 and MXFP8 prefill stay `planned`.
- **Planner:**
  - Shared memory is priced with the row pad.
  - The warp tile is 64x64 on 256-wide tiles and 64x32 otherwise, with a register budget to match, and the tile-intensity cap is raised to a 128x256 tile's.
  - Measured on GB10 at 128x256x32 with 64x64 warps: 64-70 TFLOP/s, against 54-58 for the previous 128x128x32 with 64x32 warps.
  - `lm_head` gets no prefill GEMM: prefill needs the last position's logits, which `gemv_lm_head` computes.
- **Tile order** (`group_n`): CTAs walk M down one weight tile at a time when the activation matrix fits in three quarters of L2; otherwise they use L2 groups of N tiles. Measured on Llama 3.1 8B:

  | Order | TFLOP/s |
  |---|---|
  | Walk M | 66-75 |
  | L2 groups | 64-72 |
  | Pure N-major | 20-50 |

  The issue proposed N-major from b12x; the survey notes N-major wins when A exceeds L2. Here A is 16 MiB for every Llama GEMM except `down_proj`.
- **`verify`:**
  - Checks each generated GEMM against f64 at the planned M and 37 rows fewer, on sampled rows and on all rows of sampled columns, and requires rows past M to be untouched.
  - `--bench` reports TFLOP/s and prints a GEMM-only prefill estimate (`estimate::prefill_estimate_us`).
- **Goldens:** `tests/golden/*.plan.json` are regenerated for the deliberate plan change: tiles, notes, no `lm_head` GEMM.

- **codex-review cycle 1 on this change:**
  - `fp:e2719f559050` (P1): a grid sized for the planned M launched for a shorter prompt made the extra blocks read past W. The kernel now exits blocks past the last tile, and `prefill::gemm` sizes the grid from M. Without the exit, that launch faulted on GB10.
  - `fp:ce42d0643802`: `prefill::gemm` checks the alignment of A, W and C, and the buffer lengths, before launching.
  - `fp:25ef53094238`: the planner prefers a tile whose BN divides N.
  - `fp:01f50afe3619`: the prefill estimate includes a shared expert's GEMMs and names any untimed kernel.

**Expected GB10 effect:**
- Llama 3.1 8B prefill GEMMs run at 65.6-75.1 TFLOP/s, 70-92% of cuBLAS on the same GPU (81-101 TFLOP/s, PyTorch 2.13 in the vLLM image).
- The GEMMs of a 2048-token prompt take about 403 ms (about 5,080 tok/s). Decode is unchanged.

## Non-goals

- Wiring the GEMMs into `generate`, which needs prefill attention (SHA-757).
- Grouped MoE prefill (SHA-777), NVFP4 and MXFP8 prefill (SHA-759, SHA-776), and closing the remaining gap to cuBLAS.

## Capabilities

### Modified Capabilities
- `kernel-planning`: the padded tile search, warp width, no `lm_head` GEMM, and L2 tile order.
- `kernel-codegen`: generated BF16 prefill GEMMs.
- `gpu-verification`: GEMM checks, TFLOP/s and the prefill estimate.

## Impact

- Changed: `src/plan.rs`, `src/codegen.rs`, `src/verify.rs`, `src/estimate.rs`, `src/main.rs`, `tests/plan.rs`, `tests/flash_next.rs`, `tests/golden/*.plan.json`.
