## MODIFIED Requirements

### Requirement: Decode kernels are generated with shapes as constexpr

Generation SHALL emit `model.cu` containing `rmsnorm`, `rope_kv_append`, `attn_decode_split`,
`attn_decode_combine`, `residual_add`, `silu_mul`, and one `gemv_<linear>` for each decode GEMV
whose status is `generated` (no source is emitted for a `planned` GEMV), each in its own
namespace of `constexpr` dimensions, with `static_assert`s for shape preconditions (for example
`HIDDEN == THREADS * ITERS * 8`, `HD % 32 == 0`, `K % 8 == 0`).

#### Scenario: Different models, different code
- **WHEN** Llama 3.1 8B and Qwen3-30B-A3B are generated
- **THEN** their cache keys differ

## ADDED Requirements

### Requirement: RMSNorm and the residual add round like HF transformers

`rmsnorm` SHALL round `x * rsqrt(mean(x^2) + eps)` to bf16 before multiplying by the weight, as HF's
`LlamaRMSNorm` does. `residual_add` SHALL add a bf16 vector into the hidden state in place, summing
in f32 and rounding to bf16.

#### Scenario: Reference agreement
- **WHEN** `verify` checks `rmsnorm` against a reference that rounds before the weight
- **THEN** it passes (measured 3.80e-3 on Llama 3.1 8B)
