# kernel-planning Specification

## Purpose

Turns a `ModelSpec` plus the GB10 target model into concrete per-shape decisions (`src/plan.rs`,
`src/target.rs`): which linear layers exist, how each decode GEMV and prefill GEMM is launched,
how decode attention is split, how RMSNorm is threaded, and what the model costs in memory and
decode bandwidth. Every number comes from the config or `target.rs`; nothing is tuned per model.

## Requirements

### Requirement: The GB10 target model is the only source of hardware limits

The planner SHALL take hardware limits only from the `GB10` target: arch `sm_121a`, 48 SMs,
1024 threads per block, 1536 threads and 65,536 registers per SM, 101,376 B dynamic shared memory
per block after opt-in, 102,400 B per SM, 24 MiB L2, 273 GB/s nominal and 235 GB/s planning
bandwidth, and 114 GiB usable unified memory.

#### Scenario: No tile exceeds the shared-memory limit
- **WHEN** either example config is planned
- **THEN** every prefill tile's shared memory and the attention kernel's shared memory are at most 101,376 B

### Requirement: Linear layers are derived from the architecture

The planner SHALL emit a fused `qkv_proj` (N = q_dim + 2*kv_dim, K = hidden) and `o_proj`. Dense
models SHALL get a fused `gate_up_proj` (N = 2*intermediate) and `down_proj`. MoE models SHALL get
a BF16 `router` (N = experts), `expert_gate_up` and `expert_down` with copies = expert count and
active = experts per token, plus `shared_gate_up`/`shared_down` when shared experts exist. An
`lm_head` GEMV (N = vocab) is always planned in BF16.

#### Scenario: Llama gate/up
- **WHEN** Llama 3.1 8B is planned
- **THEN** `gate_up_proj` has N = 28672 and K = 4096

### Requirement: Decode GEMV grids keep DRAM busy across all SMs

Each decode GEMV SHALL use 8 warps per block and choose rows-per-warp as the first of 4, 2, 1 that
divides N and still yields at least 4 blocks per SM (192 blocks); otherwise 1. The activation
vector SHALL be staged in shared memory (2*K bytes). When 2*K exceeds the 101,376 B per-block limit
(K above 50,688) the plan SHALL still record the GEMV but SHALL add a note naming it, its K and its
shared-memory need; codegen then leaves it `planned` (see kernel-codegen).

#### Scenario: lm_head
- **WHEN** Llama 3.1 8B's `lm_head` (N = 128256) is planned
- **THEN** it uses 4 rows per warp and a grid of 4008 blocks

#### Scenario: Activation vector too large to stage
- **WHEN** a dense config with `hidden_size` 16384 and no `intermediate_size` is planned (down_proj K = 65536)
- **THEN** the plan notes that `gemv down_proj` needs 131072 B, above the 101376 B limit

### Requirement: Prefill tiles are chosen by a legal-configuration search

For each single-copy linear at the planned prompt length, the planner SHALL search BM in {64, 128},
BN in {64, 128, 256}, BK in {32, 64} and stages in {2, 3, 4}; discard any configuration whose
multi-buffered BF16 tiles exceed the shared-memory limit, whose thread count exceeds 1024, that fits
zero CTAs per SM, or whose BK does not divide K; and keep the highest score.

#### Scenario: Llama prefill tile
- **WHEN** Llama 3.1 8B is planned with a 2048-token prompt
- **THEN** `qkv_proj` gets a 128x128x32 tile with 3 stages and 49152 B of shared memory

### Requirement: MoE expert tiles use the average tokens per expert

MoE expert linears SHALL be tiled at M = prompt tokens * experts per token / experts (at least 1).

#### Scenario: MoE grouped GEMM M
- **WHEN** Qwen3-30B-A3B is planned with a 2048-token prompt
- **THEN** `expert_gate_up` is tiled for M = 128

### Requirement: Decode attention splits fill the GPU

Decode attention SHALL use 8 warps per CTA, one CTA per (kv_head, split), with
splits = ceil(2*48 / kv_heads) clamped to [1, max(1, floor(ctx/256))], so every context length,
including those below 256 tokens, gets at least one split; and shared memory
warps * group * (head_dim + 2) * 4 bytes.

#### Scenario: Short contexts
- **WHEN** Llama 3.1 8B (8 KV heads) is planned at contexts of 128, 255, 256 and 512 tokens
- **THEN** it gets 1, 1, 1 and 2 splits

#### Scenario: Grids for the examples
- **WHEN** both examples are planned at a 32768-token context
- **THEN** Llama gets 8x12 CTAs and Qwen3-30B-A3B gets 4x24, both at least 48

### Requirement: The budget reports memory fit and a decode bandwidth ceiling

The budget SHALL count weights as all linear copies per layer times layers, plus a BF16 embedding,
plus a BF16 `lm_head` unless embeddings are tied; the KV cache as BF16 at the planned context. It
SHALL report whether weights plus KV fit in usable memory, and a decode ceiling equal to the
planning bandwidth divided by bytes read per token (active weights, `lm_head`, and the whole KV
cache at the planned context).

#### Scenario: Llama 3.1 8B at 32K
- **WHEN** Llama 3.1 8B BF16 is planned at a 32768-token context
- **THEN** weights are about 15.0 GiB, decode reads about 19.3 GB per token, and the ceiling is about 12.2 tok/s

### Requirement: Correctness gaps are reported as plan notes

The plan SHALL include a note when `qk_norm` or `qkv_bias` is set (output is wrong until the feature
is generated), when head_dim is not a multiple of 32, when hidden is not a multiple of 8, when
weights are NVFP4, when weights plus KV exceed one Spark, when a GEMV's staged activation vector
exceeds the shared-memory limit, and the MoE tokens-per-expert used for grouped tiles.

#### Scenario: Qwen3 note
- **WHEN** Qwen3-30B-A3B is planned
- **THEN** the notes mention q_norm/k_norm, and Llama 3.1 8B's notes do not
