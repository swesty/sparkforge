# Tasks

## 1. Lane map and host emulation

- [x] 1.1 Add `repack::lane_elem(lane, j, b) -> (row_slot, k_pair)` with the design's formulas (decision 2). Verify: a `tests/repack.rs` test shows that, over all 32 x 4 x 4 (lane, j, b), each of a tile's 64 rows x 8 K pairs is visited exactly once, at the byte position `tile_byte` assigns it.
- [x] 1.2 Add a host emulation of the NVFP4 GEMV in `tests/repack.rs`: it walks lanes with `lane_elem`, decodes scales from S0E5M3, sums split partials in split order, and applies per-segment global scales. Verify against an f64 reference that dequantizes the checkpoint-format bytes:
  - for N 128, K 80 (5 K tiles, so S 3 has a short last split) with S 1 and S 3;
  - for a grouped case with 3 slots, unordered expert ids and distinct gate and up scales;
  - with scale bytes including 0x00, 0x01 and 0x07.

## 2. Planner

- [x] 2.1 Add `plan_gemv_fp4`. It applies the split-K rule and fills the grid (x, S, slots), the workspace and counter bytes, and the global-scale segments (q, k, v; gate, up; or one segment, with grouped columns into `[E, 3]`). Its new `GemvPlan` fields are optional and skipped in JSON. Verify: `tests/plan.rs` asserts the spec scenarios:
  - Flash Next `expert_gate_up` S 5, grid (3, 5, 10), workspace 256,000 B, counters 800 B;
  - Flash Next `expert_down` S 3, grid (5, 3, 10);
  - Qwen3-30B-A3B NVFP4 `qkv_proj` S 12, grid (10, 12, 1); `o_proj` S 29; `expert_gate_up` S 5, grid (3, 5, 8); `expert_down` S 4, grid (4, 4, 8);
  - the golden BF16 plans in `tests/golden/*.plan.json` unchanged.
- [x] 2.2 Replace the hybrid plan note "every decode op ... is planned, not generated" with one that names the generated NVFP4 GEMVs. Verify: the `tests/flash_next.rs` note assertion is updated and passes.

## 3. Codegen

- [x] 3.1 Add the `GEMV_FP4` template (design decisions 1 and 3 to 7), with arguments `W, SF, gscale, experts, x, y, partial, count`, and emit it for every NVFP4 or NVFP4 weight-only GEMV that tiles. It is `generated` with no dynamic shared memory, and a GEMV that does not tile stays `planned`. Verify in `tests/plan.rs`:
  - Qwen3-30B-A3B NVFP4 generates `gemv_qkv_proj`, `gemv_o_proj`, `gemv_expert_gate_up` and `gemv_expert_down` with source;
  - an NVFP4 config with `hidden_size` 2016 leaves `gemv_o_proj` and `gemv_down_proj` planned without source (an `intermediate_size` of 1000 has no legal prefill tile, a separate planner limit);
  - an NVFP4 linear with K 65536 is generated with no dynamic shared memory;
  - every generated block is a multiple of 32 and at most 1024.
- [x] 3.2 Hybrid codegen: generate the NVFP4 GEMVs that tile and list every other op class as `planned`, keying the cache on the plan as before. Verify in `tests/flash_next.rs`:
  - Flash Next's manifest has exactly `gemv_expert_gate_up` and `gemv_expert_down` generated, and its source has exactly two `__global__` functions;
  - a hybrid config with MXFP8 experts still has none, and `require_generated` refuses it.
- [x] 3.3 Compile the generated `model.cu` for Flash Next and Qwen3-30B-A3B NVFP4 with NVRTC for `sm_121a` on spark-5. Verify: `sparkforge verify` gets past `Engine::load`, and `cuobjdump -sass` of the emitted module shows `F2FP.F16.E2M1.UNPACK_B`, no local-memory spills (`-Xptxas -v` or the NVRTC log), and the register count. The register count is recorded in the PR.

## 4. Verification on the GPU

- [x] 4.1 Add the NVFP4 GEMV checks to `verify::gemvs` (design decision 9):
  - checkpoint-format random inputs, including the forced zero and subnormal scales;
  - repack for the GPU, and an independent f64 reference;
  - distinct segment scales;
  - unordered grouped ids;
  - the NaN-sentinel double launch with a bitwise comparison;
  - a zeroed workspace and counters per GEMV.

  Verify on spark-5 (`verify`, without `--bench`): every NVFP4 check passes below 1/64 for `configs/qwen3.8-flash-next-nvfp4.json` and for `configs/qwen3-30b-a3b.json --format nvfp4`, and the BF16 checks for Llama 3.1 8B and Qwen3-30B-A3B still pass.
- [x] 4.2 Add `--bench` timing:
  - dense NVFP4 GEMVs rotate weight copies to 3x L2;
  - grouped ones allocate all `num_experts` experts and use a fresh random id set per launch;
  - GB/s counts weight and scale bytes;
  - `token_estimate_us` counts a grouped GEMV once per layer.

  Verify: `tests/plan.rs` (host) covers the estimate arithmetic with a fixed set of timing rows.
- [x] 4.3 Run `cargo run --release --features cuda -- verify <config> --bench` on spark-5 for:
  - `configs/qwen3.8-flash-next-nvfp4.json`;
  - `configs/qwen3-30b-a3b.json --format nvfp4`;
  - `configs/llama-3.1-8b.json` (a BF16 regression check).

  Verify: record in the PR the host, CUDA version, the worst errors, each NVFP4 GEMV's µs and GB/s against the 225 GB/s bar (gated for GEMVs of 10 MB or more), and the Qwen3-30B-A3B NVFP4 per-token estimate against the 43.5 tok/s ceiling. The spark-5 serving lane stays up for this, because `verify` allocates about 2 GB.

## 5. Docs and checks

- [x] 5.1 Update `README.md` where it says NVFP4 GEMVs are planned, and the CLI help where it names generated kernels. Verify: `grep -n -i nvfp4 README.md` shows no claim that NVFP4 decode GEMVs are unimplemented.
- [x] 5.2 Run `tools/code-factory/verify.sh` on macOS, and `cargo test --release` plus `cargo check --features cuda` on spark-5. Verify: all pass, with the counts recorded in the PR.

## Workflow follow-up

- Archive with `openspec archive add-nvfp4-gemv` after the PR's codex-review passes and before merge, as for earlier changes.
