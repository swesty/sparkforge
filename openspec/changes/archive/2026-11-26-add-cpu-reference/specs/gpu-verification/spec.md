## MODIFIED Requirements

### Requirement: Kernels are compared to an f64 reference with a fixed tolerance

Inputs SHALL be bf16 values drawn uniformly from [-1, 1) by a seeded xorshift generator (default
seed 0x5eed). The reference SHALL be computed in f64 from the same bf16 inputs. The per-element
error SHALL be |gpu - ref| / (|ref| + 0.01 * max|ref|); a NaN or infinite GPU value SHALL count as
infinite error. A check passes when the worst element is below 1/64, and the command MUST exit
non-zero if any check fails.

#### Scenario: Current kernels pass
- **WHEN** `verify` runs on a GB10 for Llama 3.1 8B and for Qwen3-30B-A3B in BF16
- **THEN** every check passes with worst per-kernel error at most 3.9e-3 (measured 3.82e-3, from `rope_kv_append` at position 4095 and `residual_add`)

### Requirement: Each generated kernel is exercised where it is most likely to break

Verification SHALL check: `rmsnorm` on 4 rows; `rope_kv_append` at positions 1, 4095, 32767 and
`max_position_embeddings - 1` (those below the maximum); decode attention at sequence lengths 1, 37,
1000 and the planned context; `silu_mul`; `residual_add`; and every generated GEMV. Kernels whose
status is `planned` SHALL be listed as skipped rather than silently omitted.

#### Scenario: NVFP4 plan
- **WHEN** `verify` runs for Qwen3-30B-A3B with NVFP4 weights
- **THEN** the four NVFP4 GEMVs are checked, and no GEMV appears in the "not generated yet (skipped)" list

#### Scenario: Flash Next plan
- **WHEN** `verify` runs for `configs/qwen3.8-flash-next-nvfp4.json`
- **THEN** `gemv_expert_gate_up` and `gemv_expert_down` are checked with 10 slots each, and every other op class is listed as skipped

## ADDED Requirements

### Requirement: A dense decode step matches the CPU reference at every layer boundary

For a dense BF16 model, `verify` SHALL run a decode step of 2 layers of random weights wired from
the generated kernels against the CPU reference, and compare the hidden state after each layer's
attention and MLP residual adds, and the logits. Any other model SHALL print why the check was
skipped.

#### Scenario: Llama 3.1 8B
- **WHEN** `verify` runs for Llama 3.1 8B on a GB10
- **THEN** four steps from an empty cache and two steps after a random 1000-position prefix pass at every boundary, one row per layer boundary and case (measured 2.9e-3 to 6.6e-3)

#### Scenario: Mixture-of-experts model
- **WHEN** `verify` runs for Qwen3-30B-A3B
- **THEN** it prints that the decode step check was skipped because mixture-of-experts layers are not covered

### Requirement: Layer boundaries are measured against the tensor's scale

The decode-step error SHALL be max|gpu - ref| / max|ref| over the tensor, with the 1/64 tolerance.
Both sides round every op to bf16, so a residual add that cancels to near zero makes the
per-element error meaningless there.

#### Scenario: Correct step under the per-element metric
- **WHEN** the same Llama 3.1 8B step is measured per element instead
- **THEN** it would fail (measured 4.4e-2 at layer 0 to 4.2e-1 at the logits), which is why the scaled metric is used

#### Scenario: Wiring bug
- **WHEN** the step omits the newest token from attention, feeds `o_proj` the wrong buffer, or drops the MLP residual add
- **THEN** the check fails (measured 0.6 to 3.9)

### Requirement: The CPU reference rounds where HF transformers does in bf16

The reference (`src/reference.rs`) SHALL compute each op in f64 and round its output once to bf16
(round to nearest even, not through f32) wherever the GPU stores bf16: RMSNorm rounds `x * rsqrt(mean(x^2) + eps)` before the weight
multiply (as HF's `LlamaRMSNorm` does) and again after it; each GEMV, the RoPE output, the attention
output, SiLU-and-multiply and the residual sum are rounded; attention probabilities are not; the
logits are compared unrounded.

#### Scenario: RMSNorm rounding
- **WHEN** the reference normalizes 4096 random elements
- **THEN** each output equals bf16(w * bf16(x * r)), and some differ from rounding x * r * w once

#### Scenario: Rounding through f32 would tie the wrong way
- **WHEN** an f64 result is 1 + 2^-8 + 2^-25, just above a bf16 midpoint that f32 rounds onto
- **THEN** the reference gives 1 + 2^-7, where going through f32 gives 1
