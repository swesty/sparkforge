# gpu-runtime Specification

## Purpose

Compiles generated source for GB10 and loads its entry points (`src/runtime.rs`), behind the
`cuda` cargo feature. `cudarc` loads `libcuda` and `libnvrtc` dynamically, so the crate builds,
and CI checks the feature, on machines with no GPU or CUDA toolkit.

## Requirements

### Requirement: The runtime refuses anything but GB10

Loading SHALL open GPU 0 and MUST fail unless its compute capability is exactly 12.1.

#### Scenario: Wrong GPU
- **WHEN** the runtime starts on a device reporting compute capability 12.0
- **THEN** loading fails with an error naming the expected and found capability

### Requirement: Source is compiled for sm_121a with NVRTC

`model.cu` SHALL be compiled by NVRTC with arch `sm_121a`, `--std=c++17`, `--use_fast_math`, and
the CUDA include directory supplied by the caller (CLI default `/usr/local/cuda/include`). The `a`
suffix is required for block-scaled FP4 MMA.

#### Scenario: Both examples load on a GB10
- **WHEN** `sparkforge load` runs on a DGX Spark (CUDA 13.0)
- **THEN** Llama 3.1 8B loads 10 kernels and Qwen3-30B-A3B loads 11

### Requirement: Only generated kernels are loaded, with large shared memory opted in

The runtime SHALL load a function only for manifest entries whose status is `generated`. For any
kernel whose dynamic shared memory exceeds 48 KB it SHALL set
`CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` to that size at load time, because the launch
otherwise fails.

#### Scenario: Large activation vector
- **WHEN** a GEMV with K greater than 24576 (more than 48 KB of staged activations) is loaded
- **THEN** its function has the dynamic shared memory attribute set to 2*K bytes

### Requirement: The feature compiles without CUDA installed

`cargo check --features cuda` SHALL succeed on a machine with no CUDA toolkit or GPU.

#### Scenario: CI
- **WHEN** the Code Factory verify workflow runs on ubuntu-latest
- **THEN** `cargo check --locked --features cuda` passes
