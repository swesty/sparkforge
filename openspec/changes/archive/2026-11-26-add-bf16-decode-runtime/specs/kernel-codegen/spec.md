## MODIFIED Requirements

### Requirement: Decode kernels are generated with shapes as constexpr

Generation SHALL emit `model.cu` containing `rmsnorm`, `rope_kv_append`, `attn_decode_split`,
`attn_decode_combine`, `residual_add`, `embed_gather`, `argmax_advance`, `silu_mul`, and one
`gemv_<linear>` for each decode GEMV whose status is `generated` (no source is emitted for a
`planned` GEMV), each in its own namespace of `constexpr` dimensions, with `static_assert`s for shape
preconditions (for example `HIDDEN == THREADS * ITERS * 8`, `HD % 32 == 0`, `K % 8 == 0`).

#### Scenario: Different models, different code
- **WHEN** Llama 3.1 8B and Qwen3-30B-A3B are generated
- **THEN** their cache keys differ

### Requirement: RoPE rotates q and k and appends K/V at the position

`rope_kv_append` SHALL rotate q and k in the fused QKV buffer using the NeoX half-split layout, and
append the rotated k and the unmodified v to the caches at the position it reads from device memory.
`attn_decode_split` SHALL attend over positions 0 through that same position.

#### Scenario: Cache slot
- **WHEN** `rope_kv_append` runs at position p
- **THEN** cache slot p holds the rotated k and the unmodified v for every KV head

## ADDED Requirements

### Requirement: The token is looked up and chosen on the device

`embed_gather` SHALL copy the embedding row named by a token id in device memory. `argmax_advance`
SHALL write the index of the greatest logit (the lowest index on ties; NaN never wins, so negative
infinity can; all NaN gives 0) as the next token, record it at the current position of an output
buffer, and increment the position. A step whose token comes from the previous step SHALL need an
embedding row for every vocabulary index.

#### Scenario: Ties and NaN
- **WHEN** the greatest logit appears twice and NaNs come before it
- **THEN** the lower index wins, it is recorded at the position, and the position advances by one

#### Scenario: NaN and negative infinity
- **WHEN** index 0 is NaN and every other logit is negative infinity
- **THEN** index 1 wins
