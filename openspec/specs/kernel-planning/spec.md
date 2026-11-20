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
zero CTAs per SM, or whose BK does not divide K; and keep the highest score, the product of wave
efficiency, padding efficiency, tile intensity (capped at 1) and a pipelining factor.

#### Scenario: Llama prefill tile
- **WHEN** Llama 3.1 8B is planned with a 2048-token prompt
- **THEN** `qkv_proj` gets a 128x128x32 tile with 3 stages and 49152 B of shared memory

#### Scenario: Wave efficiency outranks tile intensity
- **WHEN** Qwen3-30B-A3B's `expert_down` (M = 128, N = 2048) is planned
- **THEN** it gets a 64x64 tile, not the higher-intensity 128x128 that would fill only 16 of 48 SMs

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

### Requirement: Hybrid models are planned as one op per distinct kernel

For a hybrid model the planner SHALL emit one op class per distinct shape and format, each with its
shape, weight format, instances per decode token and bytes per token, instead of one entry per
layer. Layers of one module class in different formats SHALL be separate op classes, each counted
at its own size, and projections SHALL fuse only where their formats match. Prefill tiles SHALL NOT
be planned for hybrid models yet.

#### Scenario: One layer in a different format
- **WHEN** layer 0's `in_proj_qkv` is BF16 and the other 35 are MXFP8
- **THEN** `gdn_in_proj_qkv` is MXFP8 ×35 and `gdn_in_proj_qkv_bf16` is BF16 ×1

#### Scenario: Shared expert up projection left in BF16
- **WHEN** the 48 shared-expert `up_proj` entries are removed from the map
- **THEN** `shared_gate` (MXFP8 ×48) and `shared_up` (BF16 ×48) replace `shared_gate_up`, and both budgets grow by 76,185,600 bytes

#### Scenario: Flash Next op classes
- **WHEN** `configs/qwen3.8-flash-next-nvfp4.json` is planned
- **THEN** 34 uniquely named op classes cover HyperConnection, Gated DeltaNet, sparse attention, MoE, PLE and `lm_head`, for example `gdn_in_proj_qkv` MXFP8 ×36 and `qsa_q_proj` N=12288 K=2560 ×12

### Requirement: The hybrid decode budget counts formats, state and context terms

The hybrid budget SHALL count each linear at its module's format, Gated DeltaNet state read and
written, and per context token the FP8 KV of the selected positions plus the indexer key scan.
Resident weights SHALL include every expert copy, the PLE table, the MTP layer, norm and conv
filter parameters, and a tied embedding once.

#### Scenario: Flash Next at 32K
- **WHEN** `configs/qwen3.8-flash-next-nvfp4.json` is planned at a 32768-token context
- **THEN** decode reads 7.338 GB per token context-free and about 7.375 GB at 32K, and resident weights are about 105.4 GB, each within 1% of the checkpoint's header sums

### Requirement: Dense plan output is unchanged by hybrid support

The plan JSON of a dense model SHALL NOT gain fields from hybrid support: hybrid-only fields are
omitted when empty.

#### Scenario: Llama and Qwen3-30B-A3B
- **WHEN** `sparkforge plan --json` runs on `configs/llama-3.1-8b.json` or `configs/qwen3-30b-a3b.json`
- **THEN** the output equals the checked-in golden file from before this change

### Requirement: Format lookups are scoped to the language model

Per-layer format lookups SHALL match only language-model module paths, so an entry of another tower
(vision, MTP) with the same layer suffix never sets a text layer's format.

#### Scenario: Vision experts in the map
- **WHEN** the map also lists `model.visual.layers.0.mlp.experts` as MXFP8
- **THEN** the text experts stay NVFP4, the budget is unchanged, and the vision entry is reported as unused

### Requirement: Unused format-map entries are reported

Every mixed-precision map entry that no planner lookup consumes SHALL be reported as a plan note
naming it, so an unfamiliar module path layout never falls back to the default format silently.

#### Scenario: Unknown module path
- **WHEN** the Flash Next map gains an entry `model.decoder.blocks.0.ffn`
- **THEN** the plan notes that entry as not used by the plan
