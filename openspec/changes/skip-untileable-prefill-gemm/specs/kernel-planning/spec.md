## ADDED Requirements

### Requirement: A prefill GEMM no tile can cover is noted, not planned

When no prefill tile is legal for a linear (no BK of 32 or 64 divides K, or none fits the target's limits), the planner SHALL plan no GEMM for it and add a note naming the linear and K, and SHALL still plan everything else; planning SHALL NOT abort. The note SHALL call for a K-tail GEMM only when neither BK divides K, and SHALL otherwise name the target's limits.

#### Scenario: K = 1000
- **WHEN** Qwen3-30B-A3B is planned with `moe_intermediate_size` 1000
- **THEN** no `expert_down` prefill tile is planned and no `gemm_expert_down` kernel is emitted, `expert_gate_up` keeps its tile, the `expert_down` decode GEMV is planned, and a note names `expert_down` and K=1000

#### Scenario: K = 1056
- **WHEN** Qwen3-30B-A3B is planned with `moe_intermediate_size` 1056 (a multiple of 32 but not 64)
- **THEN** `expert_down` gets a prefill tile with BK 32

#### Scenario: Limits, not K, empty the search
- **WHEN** Qwen3-30B-A3B with `moe_intermediate_size` 1056 is planned for a target whose per-block shared-memory limit is 16 KiB (below every candidate tile)
- **THEN** the `expert_down` note names K=1056 and the target's limits, and does not call for a K-tail GEMM
