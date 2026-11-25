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
- **THEN** the four NVFP4 GEMVs are checked, and no GEMV appears in the "not generated yet (skipped)" list

#### Scenario: Flash Next plan
- **WHEN** `verify` runs for `configs/qwen3.8-flash-next-nvfp4.json`
- **THEN** `gemv_expert_gate_up` and `gemv_expert_down` are checked with 10 slots each, and every other op class is listed as skipped

### Requirement: Timings measure DRAM, not L2

With `--bench`, each GEMV SHALL be timed while rotating through enough weight copies that the working
set is at least 3x the 24 MB L2; GB/s SHALL be weight bytes divided by kernel time. Attention SHALL
be timed at the planned context. The report SHALL include a per-token estimate summing isolated
kernel times per layer (per active expert for MoE, or once for a grouped GEMV) plus `lm_head`, and
SHALL omit the estimate when a required kernel was not generated.

#### Scenario: BF16 GEMV bandwidth
- **WHEN** `verify --bench` runs for Llama 3.1 8B on a GB10
- **THEN** `gemv_lm_head` (1.05 GB of weights) reports about 245-251 GB/s

#### Scenario: NVFP4 per-token estimate
- **WHEN** `verify --bench` runs for Qwen3-30B-A3B with NVFP4 weights on a GB10
- **THEN** the report includes a per-token estimate that counts `gemv_expert_gate_up` and `gemv_expert_down` once per layer

#### Scenario: Shared expert without a fitting activation kernel
- **WHEN** a MoE plan's shared expert is 4096 wide while `silu_mul` is generated at the routed width of 768
- **THEN** the report omits the per-token estimate even when both shared GEMVs were timed

### Requirement: Verification refuses a model with no generated kernels

`sparkforge verify` MUST fail with a message naming the model when its plan generates no kernels,
rather than reporting an empty pass, and MUST do so before creating a CUDA context.

#### Scenario: Flash Next verify
- **WHEN** `sparkforge verify` runs on a Flash Next config whose routed experts are MXFP8, so no op class has a kernel
- **THEN** it fails stating that no kernels are generated for this model yet

#### Scenario: Host without a GPU
- **WHEN** the same command runs on a host where CUDA cannot initialize
- **THEN** it reports the same planned-only error instead of a CUDA failure

### Requirement: NVFP4 inputs are checkpoint-format bytes checked against an independent reference

NVFP4 GEMV inputs SHALL be drawn as checkpoint-format bytes: E2M1 code bytes, E4M3 scale bytes and f32 global scales from the seeded generator. The weight loader's repack SHALL turn them into the GPU operands. The f64 reference SHALL dequantize the checkpoint-format bytes, never the repacked ones, and the same 1/64 tolerance SHALL apply.

#### Scenario: NVFP4 kernels pass
- **WHEN** `verify` runs on a GB10 for Qwen3-30B-A3B with NVFP4 weights and for `configs/qwen3.8-flash-next-nvfp4.json`
- **THEN** every NVFP4 GEMV check passes below 1/64

### Requirement: NVFP4 checks cover scale edge cases, fused segments and expert order

Each NVFP4 GEMV check SHALL include scale bytes 0x00, 0x01 and 0x07 and a distinct global scale per fused segment. A grouped check SHALL use distinct expert ids that are neither contiguous nor sorted.

#### Scenario: Wrong segment scale
- **WHEN** a kernel applied the gate global scale to the up rows
- **THEN** the up rows' error exceeds 1/64, because the check's gate and up scales differ by a factor of 2

#### Scenario: Activation at the BF16 limit
- **WHEN** one row has code +6 with scale byte 0x01 at the only K position whose activation is nonzero, the largest finite BF16
- **THEN** that row's output is finite and within 1/64 of the reference, about 4e36 times its global scale

#### Scenario: Activation at the BF16 limit under a small global scale
- **WHEN** a row has code +6 with block scale 1.0 and global scale 1/8 at the only K position whose activation is nonzero, the largest finite BF16
- **THEN** that row's output is finite and within 1/64 of the reference, about 2.54e38

### Requirement: Split-K NVFP4 GEMVs are checked for bitwise replay

A split-K NVFP4 GEMV SHALL run on 4 activation vectors in turn, each checked against its own reference, on one workspace whose counters are never cleared. After 63 more launches, the first vector SHALL run again over a NaN-sentinel output, and the check SHALL fail unless that output is bitwise identical to its first.

#### Scenario: Stale counter
- **WHEN** a split-K GEMV's last-arriving split fails to reset its counter
- **THEN** no split writes the second launch's output, the NaN sentinel remains, and the check fails

### Requirement: Grouped NVFP4 GEMVs are timed without L2 reuse

With `--bench`, a grouped NVFP4 GEMV SHALL be timed over copies of the whole expert table, each launch reading distinct expert ids, with enough copies that the launches between two reads of one expert copy read at least 3 x L2 bytes. The GB/s of any NVFP4 GEMV SHALL count the weight and scale bytes it reads: for a grouped GEMV, those of the active experts.

#### Scenario: Flash Next gate/up bandwidth
- **WHEN** `verify --bench` runs for `configs/qwen3.8-flash-next-nvfp4.json` on a GB10
- **THEN** `gemv_expert_gate_up` reports GB/s over 10 x 1,843,200 bytes per launch from one copy of all 512 experts, with no expert reread within 3 x L2 bytes

#### Scenario: Small expert table
- **WHEN** a grouped GEMV has 8 experts of 1,990,656 bytes and 2 slots
- **THEN** the timing allocates 5 copies of the table and reads each expert copy again only after 19 other launches (75.6 MB)

#### Scenario: Batches that do not divide the table
- **WHEN** a grouped GEMV has 10 experts of 18,874,368 bytes and 6 slots
- **THEN** the timing allocates 2 copies of the table, and no expert is read by two consecutive launches
