## MODIFIED Requirements

### Requirement: Prefill tiles are chosen by a legal-configuration search

For each single-copy linear but `lm_head` at the planned prompt length, the planner SHALL search
BM in {64, 128}, BN in {64, 128, 256}, BK in {32, 64} and stages in {2, 3, 4}; discard any
configuration whose padded multi-buffered BF16 tiles exceed the shared-memory limit, whose thread
count exceeds 1024, that fits zero CTAs per SM, or whose BK does not divide K; and keep the highest
score, the product of wave efficiency, padding efficiency, tile intensity (capped at 4/3) and a
pipelining factor.

#### Scenario: Llama prefill tile
- **WHEN** Llama 3.1 8B is planned with a 2048-token prompt
- **THEN** `qkv_proj` gets a 128x256x32 tile with 3 stages, 8 warps of 64x64 and 92160 B of shared memory

#### Scenario: Wave efficiency outranks tile intensity
- **WHEN** Qwen3-30B-A3B's `expert_down` (M = 128, N = 2048) is planned
- **THEN** it gets a 64x64 tile, not the higher-intensity 128x128 that would fill only 16 of 48 SMs

## ADDED Requirements

### Requirement: Prefill tiles pad shared memory and size warps by tile width

A prefill tile SHALL pad each shared-memory row by 16 bytes. Its warps SHALL each cover 64 rows by
64 columns on 256-wide tiles and 64 by 32 otherwise, with a register budget of 255 and 168 per
thread respectively when counting CTAs per SM.

#### Scenario: Padded tiles
- **WHEN** either example model is planned
- **THEN** every tile's shared memory is stages x (BM + BN) x (BK + 8) x 2 bytes, at most 101,376 B, with (BM / 64) x (BN / warp width) warps

### Requirement: Prefill computes lm_head for the last position only

The planner SHALL NOT plan a prefill GEMM for `lm_head`, and SHALL note that the last prompt
position's logits come from `gemv_lm_head`.

#### Scenario: No lm_head GEMM
- **WHEN** either example model is planned
- **THEN** no `gemm_lm_head` is planned, and the note names `gemv_lm_head`

### Requirement: Prefill tiles are ordered for L2

Each prefill tile SHALL carry `group_n`, the number of N tiles CTAs walk before moving to the next M
tile. It SHALL be 1 (walk M down one weight tile at a time) when the M x K bf16 activations fit in
three quarters of L2; otherwise the most N tiles whose weight tiles plus one wave's A tiles fit
there, or else the group with the smallest such working set, from the target's SM count and L2 size.

#### Scenario: Llama on GB10
- **WHEN** Llama 3.1 8B is planned with a 2048-token prompt
- **THEN** `qkv_proj`, `o_proj` and `gate_up_proj` (16 MiB of activations) walk M, and `down_proj` (56 MiB) uses groups of 4 N tiles
