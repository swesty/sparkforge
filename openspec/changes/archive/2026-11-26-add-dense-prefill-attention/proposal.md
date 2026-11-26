# Proposal

## Why

sparkforge had tensor-core prefill GEMMs (SHA-751) but no prefill attention, so `generate` ran a prompt as decode steps at about 65 ms per token for Llama 3.1 8B: about 133 s for 2048 tokens. SHA-757, which originally covered dense prefill attention, was re-scoped to QSA sparse attention for Qwen3.8 Flash Next. Dense attention moved to SHA-828.

## What Changes

- **Kernel** (`codegen.rs`): `attn_prefill(qkv, k_cache, v_cache, out, M, start, scale_log2)`, causal attention in the style of FlashAttention-2.
  - Prompt row r is at position start + r and attends to positions 0 through start + r.
  - Each warp owns 16 query rows, with its Q fragments in registers.
  - K/V tiles are read from the decode KV cache through a `cp.async` pipeline that reuses the Q tile's shared memory.
  - Q·Kᵀ and P·V run on `mma.sync m16n8k16`, with `ldmatrix.trans` for V.
  - The online softmax is fp32 in the exp2 domain, and P is rounded to bf16 for the P·V product.
  - Grid: (heads, ceil(M / BLOCK_M)). The heads that share a KV head are adjacent, and the longest query tiles run first.
- **Row-wise kernels:** `rope_kv_append`, `residual_add`, `silu_mul` and `embed_gather` process row `blockIdx.y` (`rmsnorm` already used `blockIdx.x`).
  - A decode step launches them with one row, so its launches are unchanged.
  - Prefill launches one row per prompt token, so a prompt row is computed by the same code a decode step runs.
- **Planner:** `prefill_attn` (BLOCK_M, BLOCK_N, stages).
  - Candidates are BLOCK_M 64 or 128, BLOCK_N 32, 64 or 128, and 2 or 3 stages.
  - Each candidate must fit 101,376 B of shared memory (max of the Q tile and the K/V stages, rows padded by 16 bytes), the thread limit, and an estimated register count.
  - Ranking: resident warps per SM first, then K/V reuse (BLOCK_M × BLOCK_N / (BLOCK_M + BLOCK_N)), then stages.
  - For `head_dim` 128 it picks 128 × 64 with 2 stages (68 KB), the fastest of the 8 configurations measured at 2048 tokens. Hybrid plans have no dense prefill attention.
- **Runtime** (`forward.rs`): `Runtime::enable_prefill` allocates activations for chunks of up to N rows (`prefill_bytes`). `Runtime::prefill(tokens, start)` then:
  - runs each chunk through the row-wise kernels, the prefill GEMMs (`prefill::gemm_at`, the checked launch on raw addresses) and `attn_prefill`;
  - computes the last position's logits with `gemv_lm_head`, and its token with `argmax_advance`;
  - leaves the token and position where the next captured decode step reads them.
- **`generate`:**
  - The prompt is prefilled in chunks of the planned prompt length, and the run prints the time to the first token. `--bench-prefill L` times prefills of L tokens from an empty cache.
  - `--check N` compares the prefill's logits and the next N - 1 steps' with the CPU reference, which runs the whole prompt first.
  - Admission adds the prefill activations.
- **`verify`:**
  - `attn_prefill` is checked against f64 causal attention at prompt lengths 1, 128, the planned length, 2048 and 8192 from position 0, and at 128 rows from position 1000.
    - Long prompts are checked on sampled rows (both sides of each tile edge, the middle and the end) for every head and dimension.
    - The metric is scaled by the largest output value, because P is rounded to bf16 by design.
  - Causality is checked bitwise: rewriting the cache past row 300 must not change rows 0 to 300.
  - `Runtime::prefill` is checked against token-by-token decode steps on two layers of random weights, for a 100-token prompt in one chunk and in chunks of 48. The checks:
    - every layer's KV cache and the last logits agree;
    - the greedy token is the argmax;
    - the captured graph decodes on from the prefill;
    - both chunkings give bitwise the same logits.
  - `--bench` reports attention TFLOP/s, and the prefill estimate now includes attention.
- **Goldens:** `tests/golden/*.plan.json` gain `prefill_attn`.

**Expected GB10 effect (measured on spark-5, CUDA 13.0.3):**
- `attn_prefill`: 69.4 to 70.4 TFLOP/s at 2048 tokens and 79.3 to 82.2 at 8192.
- Llama 3.1 8B time to first token:
  - 490.5 ms for 2048 tokens (4,176 tok/s), against about 133 s before;
  - 2,122.9 ms for 8192 tokens;
  - 93.9 ms for a 20-token prompt.
- Decode is unchanged (15.4 tok/s at 1K against a ceiling of 15.5).

Tile sweep (TFLOP/s at 2048 / 8192 tokens, `head_dim` 128):

| BLOCK_M × BLOCK_N, stages | 2048 | 8192 |
|---|---|---|
| 128 × 64, 2 (planned) | 69.4 | 80.3 |
| 192 × 64, 2 | 67.3 | 85.3 |
| 192 × 32, 2 | 65.8 | 81.6 |
| 64 × 32, 2 | 65.5 | 73.9 |
| 128 × 32, 2 | 64.2 | 75.8 |
| 128 × 32, 3 | 61.9 | 72.5 |
| 64 × 64, 2 | 50.9 | 55.6 |
| 64 × 32, 3 | 41.9 | 45.0 |

## Non-goals

- QSA sparse prefill attention, FP8 KV and the output gate (SHA-757).
- Mixture-of-experts prefill (SHA-777), and q/k norm or q/k/v bias (Qwen3-30B-A3B's attention kernel is generated and verified, but its model doesn't run end to end).
- The codex-review follow-ups tracked in SHA-824.

## Capabilities

### Modified Capabilities
- `kernel-planning`: prefill attention tiles.
- `kernel-codegen`: `attn_prefill` and the row-wise kernels.
- `gpu-runtime`: prefill in `generate`, time to first token.
- `gpu-verification`: attention and prefill checks, and attention in the prefill estimate.

## Impact

- Changed: `src/codegen.rs`, `src/plan.rs`, `src/forward.rs`, `src/prefill.rs`, `src/verify.rs`, `src/estimate.rs`, `src/main.rs`, `tests/plan.rs`, `tests/flash_next.rs`, `tests/golden/*.plan.json`, `README.md`.
