## ADDED Requirements

### Requirement: A prefill GEMM no tile can cover is noted, not planned

A prefill tile SHALL be legal only when its BK (32 or 64) divides K. When no tile is legal for a linear, the planner SHALL plan no prefill GEMM for that linear, SHALL add a note that names the linear and its K, and SHALL still plan every other GEMM and every decode operation. Planning SHALL NOT abort.

#### Scenario: K = 1000
- **WHEN** Qwen3-30B-A3B is planned with `moe_intermediate_size` 1000
- **THEN** no `expert_down` prefill tile is planned and no `gemm_expert_down` kernel is emitted, `expert_gate_up` keeps its tile, the `expert_down` decode GEMV is planned, and a note names `expert_down` and K=1000

#### Scenario: K = 1056
- **WHEN** Qwen3-30B-A3B is planned with `moe_intermediate_size` 1056 (a multiple of 32 but not 64)
- **THEN** `expert_down` gets a prefill tile with BK 32
