## ADDED Requirements

### Requirement: Prefill attention tiles maximize resident warps, then K/V reuse

A dense plan SHALL carry `prefill_attn`. Its candidates are BLOCK_M 64 or 128 (16 query rows per warp), BLOCK_N 32, 64 or 128, and 2 or 3 stages. A candidate is legal when max(Q tile, K and V stages) with 16-byte row pads fits 101,376 B, and the threads and estimated registers fit. The planner SHALL pick the most resident warps per SM, then the largest BLOCK_M × BLOCK_N / (BLOCK_M + BLOCK_N), then the most stages. A hybrid plan has none.

#### Scenario: head_dim 128
- **WHEN** Llama 3.1 8B or Qwen3-30B-A3B is planned for 2048 prompt tokens
- **THEN** `prefill_attn` is 128 query rows × 64 keys, 2 stages, 8 warps, 69,632 B, grid (32, 16)

#### Scenario: Hybrid model
- **WHEN** Qwen3.8 Flash Next is planned
- **THEN** the plan has no `prefill_attn` and no `attn_prefill` kernel
