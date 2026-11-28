## MODIFIED Requirements

### Requirement: Prefill GEMMs launch through a checked host call

`prefill::gemm` SHALL launch a GEMM through a `prefill::Gemm`, which binds one `gemm_<linear>`'s tile, manifest entry and function, for M rows with a grid of ceil(M / BM) x N / BN, after checking the stream's context, that A holds M x K elements, W exactly N x K, C M x N, A and W are 16-byte aligned and C 4-byte aligned. It SHALL pass its views to the launch. A larger grid SHALL leave extra blocks idle.

#### Scenario: Misaligned activations
- **WHEN** A is a view one element past an aligned allocation
- **THEN** `prefill::gemm` refuses it before launching

#### Scenario: Manifest grid for a shorter prompt
- **WHEN** a Llama GEMM is launched with the manifest's grid (for M = 2048) and M = 1024
- **THEN** the output matches the reference and rows past 1024 are untouched (without the in-kernel exit, the same launch faulted with an illegal address)

#### Scenario: Raw addresses
- **WHEN** a caller holds operands only as device addresses
- **THEN** it launches with `unsafe fn gemm_raw`, which makes the same checks and documents the lifetime, context and stream-ordering obligations

## ADDED Requirements

### Requirement: Prefill attention keeps a future value out of earlier rows

In a tile holding keys some row of its block doesn't attend, `attn_prefill` SHALL ask at the tile's barrier (`__syncthreads_or`) whether any V value isn't finite. If one is, it SHALL zero those values in shared memory, record per dimension the first key that had one, and at the end set each row to NaN at the dimensions where it attends such a key. Other tiles SHALL run unchanged.

#### Scenario: NaN in a future value
- **WHEN** two rows run from position 0 with a finite V at position 0 and NaN in V at position 1
- **THEN** row 0 is bitwise what it is with a finite value there, and row 1 is NaN
