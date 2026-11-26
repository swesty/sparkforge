## ADDED Requirements

### Requirement: Prefill GEMMs are checked at the planned and a ragged length

`verify` SHALL check every generated prefill GEMM against an f64 reference at the planned M and at
37 rows fewer through `prefill::gemm`, and at half the planned M launched with the manifest's grid,
on 8 sampled rows across all columns and on every row of 16 sampled columns. It SHALL fail the GEMM
if any row past M is written or a misaligned A view is launched.

#### Scenario: Both example models
- **WHEN** `verify` runs for Llama 3.1 8B and Qwen3-30B-A3B in BF16 on a GB10
- **THEN** every generated GEMM passes, with worst error 3.83e-3

### Requirement: Prefill GEMMs report TFLOP/s and a prompt estimate

With `--bench`, `verify` SHALL time each generated prefill GEMM at the planned M and report TFLOP/s,
and SHALL print a prefill estimate for the planned prompt: the layers' GEMMs (a shared expert's
included) plus one `gemv_lm_head`, stating that attention, norms, RoPE and activations are not
included, or name the kernel whose missing timing prevents it.

#### Scenario: Llama 3.1 8B on GB10
- **WHEN** `verify --bench` runs for Llama 3.1 8B on spark-5
- **THEN** the GEMMs run at 65.6 to 75.1 TFLOP/s and the estimate is about 403 ms for 2048 tokens (about 5,080 tok/s)

#### Scenario: Qwen3-30B-A3B on GB10
- **WHEN** `verify --bench` runs for Qwen3-30B-A3B in BF16
- **THEN** the dense GEMMs run at 68 to 69 TFLOP/s and the M = 128 expert shapes at 25 to 33
