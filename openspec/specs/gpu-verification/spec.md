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
- **THEN** every check passes with worst per-kernel error at most 3.9e-3 (measured 3.82e-3, from `rope_kv_append` at position 4095 and `residual_add`)

### Requirement: Each generated kernel is exercised where it is most likely to break

Verification SHALL check: `rmsnorm` on 4 rows; `rope_kv_append` at positions 1, 4095, 32767 and
`max_position_embeddings - 1` (those below the maximum); decode attention at sequence lengths 1, 37,
1000 and the planned context; `silu_mul`; `residual_add`; `embed_gather` (bitwise rows);
`argmax_advance` (planted ties and NaNs, the recorded token and the advanced position); and every
generated GEMV. Kernels whose status is `planned` SHALL be listed as skipped rather than silently
omitted.

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

### Requirement: A dense decode step matches the CPU reference at every layer boundary

For a dense BF16 model, `verify` SHALL run a decode step of 2 layers of random weights wired from
the generated kernels against the CPU reference, and compare the hidden state after each layer's
attention and MLP residual adds, and the logits. The same steps replayed from a captured CUDA Graph
SHALL give bitwise the same logits. Any other model SHALL print why the check was skipped.

#### Scenario: Llama 3.1 8B
- **WHEN** `verify` runs for Llama 3.1 8B on a GB10
- **THEN** four steps from an empty cache and two steps after a random 1000-position prefix pass at every boundary, one row per layer boundary and case (measured 7.2e-4 to 6.4e-3), and the graph replay matches bitwise with the argmax recorded as the next token

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

### Requirement: Decoded logits match the CPU reference on real weights

`generate --check N` SHALL compare the logits of the first N steps (prompt, then generated) with the
CPU reference, which streams the weights from the checkpoint, with the scaled layer-boundary metric
and the 1/64 tolerance, and SHALL fail if any step exceeds it. The reference SHALL refuse a model
whose tensor shapes differ from the checkpoint's binding, and a step that fails partway SHALL restore
every cache slot it wrote.

#### Scenario: Llama 3.1 8B
- **WHEN** 24 steps of Llama 3.1 8B are checked on a GB10
- **THEN** every step passes (worst 1.37e-2 at position 3; positions 5 to 23 measure 4.3e-3 to 8.2e-3) and the greedy token equals the reference's argmax in 24 of 24

### Requirement: Prefill GEMMs are checked at the planned and a ragged length

`verify` SHALL check every generated prefill GEMM against an f64 reference at the planned M and at
37 rows fewer, on 8 sampled rows across all columns and on every row of 16 sampled columns, and
SHALL fail it if any row past M is written.

#### Scenario: Both example models
- **WHEN** `verify` runs for Llama 3.1 8B and Qwen3-30B-A3B in BF16 on a GB10
- **THEN** every generated GEMM passes, with worst error 3.82e-3

### Requirement: Prefill GEMMs report TFLOP/s and a prompt estimate

With `--bench`, `verify` SHALL time each generated prefill GEMM at the planned M and report TFLOP/s,
and SHALL print a prefill estimate for the planned prompt: the layers' GEMMs plus one `gemv_lm_head`,
stating that attention, norms, RoPE and activations are not included.

#### Scenario: Llama 3.1 8B on GB10
- **WHEN** `verify --bench` runs for Llama 3.1 8B on spark-5
- **THEN** the GEMMs run at 65.6 to 75.1 TFLOP/s and the estimate is about 403 ms for 2048 tokens (about 5,080 tok/s)

#### Scenario: Qwen3-30B-A3B on GB10
- **WHEN** `verify --bench` runs for Qwen3-30B-A3B in BF16
- **THEN** the dense GEMMs run at 68 to 69 TFLOP/s and the M = 128 expert shapes at 25 to 33
