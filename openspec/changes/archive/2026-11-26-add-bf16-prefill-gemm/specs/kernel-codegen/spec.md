## MODIFIED Requirements

### Requirement: Kernel status separates generated code from planned work

Each manifest entry SHALL carry entry name, file, grid, block, dynamic shared memory, argument names
and a status. A GEMV SHALL be `generated` when its weights are BF16 (or the linear is `lm_head` or
`router`) and its staged activation vector (2*K bytes) fits the 101,376 B per-block shared-memory
limit, or when its weights are NVFP4 or NVFP4 weight-only and N is a multiple of 64 and K a multiple
of 16; otherwise it SHALL be `planned`. Prefill GEMMs follow their own rule.

#### Scenario: Shape too wide to stage
- **WHEN** a config gives `down_proj` K = 65536 (131072 B of staged activations)
- **THEN** `gemv_down_proj` is `planned`

#### Scenario: NVFP4 GEMVs are generated
- **WHEN** Qwen3-30B-A3B is generated with NVFP4 weights
- **THEN** `gemv_qkv_proj`, `gemv_o_proj`, `gemv_expert_gate_up` and `gemv_expert_down` are `generated` with source in `model.cu`, and request no dynamic shared memory

#### Scenario: NVFP4 shape that does not tile
- **WHEN** an NVFP4 linear has N = 2016 (not a multiple of 64)
- **THEN** its GEMV is `planned` with no source

## ADDED Requirements

### Requirement: BF16 prefill GEMMs are generated

A prefill GEMM SHALL be `generated` when its linear's weights are BF16 and its tile divides N and K,
and `planned` otherwise.

#### Scenario: Prefill GEMMs by format
- **WHEN** Llama 3.1 8B (BF16) and Qwen3-30B-A3B with NVFP4 weights are generated
- **THEN** Llama's four `gemm_<linear>` entries are `generated` with source in `model.cu`, and every NVFP4 linear's GEMM is `planned` with no source

### Requirement: The prefill GEMM computes A times W transposed with mma.sync

`gemm_<linear>(A, W, C, M)` SHALL compute C[M][N] = A[M][K] · W[N][K]ᵀ in bf16 with f32 accumulation,
N, K and the tile `constexpr` and the prompt length M an argument: a `cp.async` pipeline of the
planned stages into 16-byte-padded shared memory, `ldmatrix`, `mma.sync m16n8k16`, tiles in the
planned `group_n` order, rows past M zero-filled on load and not stored.

#### Scenario: Ragged prompt
- **WHEN** a GEMM runs with M 37 rows short of the planned length
- **THEN** the rows below M match the reference and the rows past M are left as they were
