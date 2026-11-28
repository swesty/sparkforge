# Tasks

## 1. Spec

- [x] 1.1 Add "A prefill GEMM no tile can cover is noted, not planned" to the kernel-planning delta, and verify with `openspec validate --all --strict`

## 2. Planner

- [x] 2.1 `plan_gemm` in `src/plan.rs` returns `Option<GemmTile>`; `build` plans dense and grouped GEMMs through `plan_gemms`, which notes each linear that gets no tile
- [x] 2.2 Confirm the readers of `Plan::gemm` (`forward::Prefill::new`, `codegen::generate`, `estimate::prefill_estimate_us`, `verify` GEMM checks, `generate`'s chunk length) tolerate a missing tile

## 3. Tests

- [x] 3.1 Add `cargo test` coverage in `tests/plan.rs`: Qwen3-30B-A3B with `moe_intermediate_size` 1000 plans with no `expert_down` tile, keeps `expert_gate_up` and the decode GEMV, notes `expert_down` with K=1000, and emits no `gemm_expert_down`; 1056 plans `expert_down` with BK 32. Verify with `tools/code-factory/verify.sh`
- [x] 3.2 codex-review cycle 1 (`fp:9a0ceeeeb2d8`): the note blames K only when neither BK divides it, and names the target's limits otherwise; `cargo test` plans K = 1056 against a 16 KiB shared-memory target
- [x] 3.3 codex-review cycle 2 (`fp:15613a925a0b`): the MoE prefill note counts the grouped GEMMs that got a tile when any is missing (text unchanged when all do, so the Qwen3 golden plan is unchanged); `cargo test` checks 1 of 2 at K = 1000 and 0 of 2 on the 16 KiB target

No shipped config changes its plan, so there is no `sparkforge verify --bench` task.

## Workflow follow-up

- After the PR merges, run `openspec archive skip-untileable-prefill-gemm` to merge the delta into `openspec/specs/kernel-planning/spec.md`, then validate again.
