# Proposal

## Why

`plan::build` picks a prefill GEMM tile for every linear of a dense model. A tile is legal only when its BK (32 or 64) divides K. When no BK divides K, `plan_gemm` called `.expect("no legal GEMM tile for this shape")`. So `sparkforge plan`, `emit`, `verify` and `generate` all aborted for any dense model with a K that isn't a multiple of 32: for example Qwen3-30B-A3B with `moe_intermediate_size` 1000 or 1008, or Llama 3.1 8B with `intermediate_size` 14340 or `hidden_size` 4080. Decode planning never ran, even though the NVFP4 decode GEMV accepts K = 1008 (it needs only K % 16 == 0).

The code that reads the tiles already handles a missing one. `forward::Prefill::new` refuses with "prefill needs kernels that were not generated". `codegen` emits only the tiles it is given. `prefill_estimate_us` errors and names the missing GEMM.

## What Changes

- A linear with no legal prefill tile gets no `GemmTile`. The planner adds a note that names the linear, its K and the BK rule, and plans everything else as before.
- A new kernel-planning requirement with two scenarios: K = 1000 is left unplanned with a note, and K = 1056 (a multiple of 32 but not 64) still plans with BK 32.
- `cargo test` covers both scenarios in `tests/plan.rs`.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `kernel-planning`: adds "A prefill GEMM no tile can cover is noted, not planned".

## Impact

- Code: `src/plan.rs` (`plan_gemm` returns `Option`, new `plan_gemms` helper, `GEMM_BK` constant).
- Tests: `tests/plan.rs`.
- Spec: `openspec/specs/kernel-planning/spec.md`, merged by `openspec archive` after merge.
- GB10 performance: none. Every shipped config has K divisible by 64, so their plans, generated kernels and timings are unchanged and there is nothing to measure with `sparkforge verify --bench`.

## Non-goals

- A K-tail prefill GEMM (zero-filled `cp.async` on the last K tile). That would let these shapes run prefill; it is its own change.
- Changing the tile search, scores or legality rules.
