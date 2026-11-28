## MODIFIED Requirements

### Requirement: Prefill GEMMs are checked at the planned and a ragged length

`verify` SHALL check every generated prefill GEMM against an f64 reference at the planned M and at 37 rows fewer through `prefill::gemm`, and at half the planned M with the manifest's grid, on 8 sampled rows across all columns and every row of 16 sampled columns. It SHALL fail the GEMM if a row past M is written, if one misaligned row of A isn't refused by the alignment check, or if a copy on another stream misses the largest GEMM's tracked output.

#### Scenario: Both example models
- **WHEN** `verify` runs for Llama 3.1 8B and Qwen3-30B-A3B in BF16 on a GB10
- **THEN** every generated GEMM passes, with worst error 3.83e-3

#### Scenario: Planned M of 1
- **WHEN** `verify --prefill-tokens 1` runs for Llama 3.1 8B
- **THEN** the misaligned probe of one row is refused by the A alignment check and every GEMM passes

#### Scenario: Untracked launch
- **WHEN** `prefill::gemm` launches with raw addresses instead of its views
- **THEN** the second-stream copy reads stale output and `verify` fails

### Requirement: Prefill attention matches f64 causal attention

`verify` SHALL check `attn_prefill` against f64 causal attention at 1, 128, the planned length, 2048 and 8192 rows from position 0, and 128 rows from position 1000, on all rows up to 256, else sampled rows. The error SHALL be scaled by the largest output. Rewriting the cache past a row SHALL leave it bitwise unchanged; NaN in V, or K and V, past a row SHALL also leave it so, and make every later row NaN.

#### Scenario: Both example models on GB10
- **WHEN** `verify --bench` runs for Llama 3.1 8B and Qwen3-30B-A3B
- **THEN** every case passes (worst 3.68e-3), causality and the NaN cases hold, and attention runs at 67.7 to 69.6 TFLOP/s at 2048 rows and 78.1 to 81.9 at 8192

#### Scenario: Mutations are caught
- **WHEN** the kernel attends one future key, skips a tile a row needs, drops the O rescale, or never zeroes a V value that isn't finite
- **THEN** `verify` fails

## ADDED Requirements

### Requirement: A planned prefill kernel missing from the manifest fails verification

`verify` SHALL fail, naming the kernel, when the plan schedules a prefill GEMM with no manifest entry, or has prefill attention without a generated `attn_prefill` (`codegen::missing_prefill_kernels`). An entry that is present but planned SHALL stay an honest skip. The prefill step check SHALL fail when `PrefillKernels` refuses a dense plan's bundle for any other reason.

#### Scenario: Entry removed
- **WHEN** `attn_prefill` and `gemm_o_proj` are removed from a Llama 3.1 8B manifest
- **THEN** both are named as missing, while a W4A16 plan's planned GEMMs and a hybrid plan report none

### Requirement: The CPU reference reads with the step's model

`CheckpointWeights::check` SHALL require the model its reads index with to have the step model's hidden width, MLP width, layers, heads, KV heads, head dimension and vocabulary, before any tensor shape check or read.

#### Scenario: Narrower read model
- **WHEN** the step's model is 16 wide and the reference's read model 8 wide over the same checkpoint
- **THEN** the step is refused before the first read and the cache is unchanged

### Requirement: The MoE prefill estimate counts only the experts a prompt can reach

The prefill estimate SHALL charge min(experts, tokens × `experts_per_tok`) routed-expert GEMM pairs per layer.

#### Scenario: One-token Qwen3-30B-A3B prompt
- **WHEN** the estimate is made for a plan with 1 prompt token
- **THEN** it charges 8 expert pairs per layer, not 128, and 2048 tokens still charge all 128
