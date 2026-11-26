## ADDED Requirements

### Requirement: Prefill attention is causal FlashAttention-2 on mma.sync

`attn_prefill(qkv, k_cache, v_cache, out, M, start, scale_log2)` SHALL compute causal attention for M prompt rows, row r at position start + r over cache positions 0..start + r. It SHALL read Q from the qkv rows and K/V from the decode cache. It SHALL use the planned tiles, Q fragments in registers, a `cp.async` K/V pipeline, `mma.sync m16n8k16` for Q·Kᵀ and P·V, and an fp32 exp2 online softmax.

#### Scenario: Generated for dense models
- **WHEN** a dense BF16 plan is generated
- **THEN** `attn_prefill` is `generated` with the plan's block and dynamic shared memory

#### Scenario: Later chunk
- **WHEN** 128 rows run from position 1000 over a cache that holds positions 0 to 1127
- **THEN** each row matches f64 causal attention over its own prefix

### Requirement: Row-wise kernels run a whole prompt

`rope_kv_append`, `residual_add`, `silu_mul` and `embed_gather` SHALL process row `blockIdx.y`, and `rmsnorm` row `blockIdx.x`. `rope_kv_append` SHALL place row r at position `*pos_p` + r. A decode step launches one row; prefill launches one per prompt token.

#### Scenario: Prompt rows
- **WHEN** `rope_kv_append` runs over M rows with `*pos_p` = s
- **THEN** cache slots s to s + M - 1 hold each row's rotated k and its v, rotated as a decode step at that position rotates them
