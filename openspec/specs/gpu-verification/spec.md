# gpu-verification Specification

## Purpose

`sparkforge verify [--bench]` (`src/verify.rs`) is the definition of "a kernel works": it runs every
generated kernel on a GB10 against an f64 CPU reference and can time it. CI has no GPU, so this is
run by hand on a Spark and its results are recorded in the pull request.

## Requirements

### Requirement: Kernels are compared to an f64 reference with a fixed tolerance

Inputs SHALL be bf16 values drawn uniformly from [-1, 1) by a seeded xorshift generator (default
seed 0x5eed). The reference SHALL be computed in f64 from the same bf16 inputs. The per-element
error SHALL be |gpu - ref| / (|ref| + 0.01 * max|ref|); a NaN or infinite GPU value SHALL count as
infinite error. A check passes when the worst element is below 1/64, and the command MUST exit
non-zero if any check fails.

#### Scenario: Current kernels pass
- **WHEN** `verify` runs on a GB10 for Llama 3.1 8B and for Qwen3-30B-A3B in BF16
- **THEN** every check passes with worst error at most 3.8e-3

### Requirement: Each generated kernel is exercised where it is most likely to break

Verification SHALL check: `rmsnorm` on 4 rows; `rope_kv_append` at positions 1, 4095, 32767 and
`max_position_embeddings - 1` (those below the maximum); decode attention at sequence lengths 1, 37,
1000 and the planned context; `silu_mul`; and every generated GEMV. Kernels whose status is
`planned` SHALL be listed as skipped rather than silently omitted.

#### Scenario: NVFP4 plan
- **WHEN** `verify` runs for Qwen3-30B-A3B with NVFP4 weights
- **THEN** the NVFP4 GEMVs appear in the "not generated yet (skipped)" list

### Requirement: Timings measure DRAM, not L2

With `--bench`, each GEMV SHALL be timed while rotating through enough weight copies that the working
set is at least 3x the 24 MB L2; GB/s SHALL be weight bytes divided by kernel time. Attention SHALL
be timed at the planned context. The report SHALL include a per-token estimate summing isolated
kernel times per layer (and per active expert for MoE) plus `lm_head`, and SHALL omit the estimate
when a required kernel was not generated.

#### Scenario: BF16 GEMV bandwidth
- **WHEN** `verify --bench` runs for Llama 3.1 8B on a GB10
- **THEN** `gemv_lm_head` (1.05 GB of weights) reports about 245-251 GB/s

### Requirement: Verification refuses a model with no generated kernels

`sparkforge verify` MUST fail with a message naming the model when its plan generates no kernels,
rather than reporting an empty pass.

#### Scenario: Flash Next verify
- **WHEN** `sparkforge verify configs/qwen3.8-flash-next-nvfp4.json` runs
- **THEN** it fails stating that no kernels are generated for this model yet
