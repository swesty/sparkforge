## MODIFIED Requirements

### Requirement: Decode GEMV grids keep DRAM busy across all SMs

Each BF16 decode GEMV SHALL use 8 warps per block and choose rows-per-warp as the first of 4, 2, 1
that divides N and still yields at least 4 blocks per SM (192 blocks); otherwise 1. The activation
vector SHALL be staged in shared memory (2*K bytes). When 2*K exceeds the 101,376 B per-block limit
(K above 50,688) the plan SHALL still record the GEMV but SHALL add a note naming it, its K and its
shared-memory need; codegen then leaves it `planned` (see kernel-codegen).

#### Scenario: lm_head
- **WHEN** Llama 3.1 8B's `lm_head` (N = 128256) is planned
- **THEN** it uses 4 rows per warp and a grid of 4008 blocks

#### Scenario: Activation vector too large to stage
- **WHEN** a dense config with `hidden_size` 16384 and no `intermediate_size` is planned (down_proj K = 65536)
- **THEN** the plan notes that `gemv down_proj` needs 131072 B, above the 101376 B limit

## ADDED Requirements

### Requirement: NVFP4 GEMVs split K until about 20 warps per SM are busy

An NVFP4 GEMV SHALL use 8 warps per block, one 64-row N tile per warp, and a grid of (ceil(N/64/8), S, slots), where slots is the experts per token for a routed-expert linear and 1 otherwise. S SHALL be the smaller of ceil(20 x SMs / (N/64 x slots)) and the cap, then reduced so no split is empty. The cap is the larger of 1 and the smaller of floor(0.95 x 6 x SMs / blocks per split) and floor(K/16/4).

#### Scenario: Flash Next experts
- **WHEN** `configs/qwen3.8-flash-next-nvfp4.json` is planned
- **THEN** `expert_gate_up` (N 1280, K 2560, 10 slots) has S = 5 and grid (3, 5, 10), and `expert_down` (N 2560, K 640) has S = 3 and grid (5, 3, 10)

#### Scenario: Qwen3-30B-A3B NVFP4
- **WHEN** `configs/qwen3-30b-a3b.json` is planned with NVFP4 weights
- **THEN** `qkv_proj` has S = 12 and grid (10, 12, 1), `o_proj` has S = 29, `expert_gate_up` has S = 5 and grid (3, 5, 8), and `expert_down` has S = 4 and grid (4, 4, 8)

### Requirement: NVFP4 GEMV plans record their workspace

When S is above 1, an NVFP4 GEMV plan SHALL record its split-K workspace as slots x S x N f32 values and its arrival counters as slots x N/64 u32 values. When S is 1 it SHALL record neither. The plan JSON of a BF16 GEMV SHALL NOT gain these fields.

#### Scenario: Flash Next gate/up workspace
- **WHEN** Flash Next's `expert_gate_up` is planned
- **THEN** its workspace is 10 x 5 x 1280 x 4 = 256,000 bytes and its counters 10 x 20 x 4 = 800 bytes

#### Scenario: Golden BF16 plans
- **WHEN** `sparkforge plan --json` runs on `configs/llama-3.1-8b.json` or `configs/qwen3-30b-a3b.json`
- **THEN** the output equals `tests/golden/*.plan.json` unchanged
