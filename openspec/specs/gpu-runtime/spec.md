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
- **WHEN** a GEMV with K greater than 24576 and at most 50688 (more than 48 KB of staged activations, within the 101376 B limit) is loaded
- **THEN** its function has the dynamic shared memory attribute set to 2*K bytes

### Requirement: The feature compiles without CUDA installed

`cargo check --features cuda` SHALL succeed on a machine with no CUDA toolkit or GPU.

#### Scenario: CI
- **WHEN** the Code Factory verify workflow runs on ubuntu-latest
- **THEN** `cargo check --locked --features cuda` passes

### Requirement: Greedy decode replays one CUDA Graph per step

`sparkforge generate` SHALL decode a dense BF16 checkpoint greedily on a GB10 from a pre-tokenized prompt. The prefill gives the first output token. A step SHALL read its token and position from device memory and end by writing the greedy next token and advancing the position, so one captured CUDA Graph on a non-default stream replays every later step with no host work between tokens: N tokens take N - 1 replays.

#### Scenario: One-token prompt, one token
- **WHEN** `generate` runs with a one-token prompt and `--steps 1`
- **THEN** the prefill of that token returns one token, with no replay and a one-position cache

#### Scenario: Llama 3.1 8B
- **WHEN** `generate` runs Llama 3.1 8B on a 20-token prompt about general relativity for 32 steps
- **THEN** the output is coherent text ("the curvature of spacetime caused by the presence of mass and energy...")

#### Scenario: Graph replay
- **WHEN** a captured step and the same step as plain launches run from the same state
- **THEN** their logits are bitwise identical and both record the argmax as the next token

### Requirement: Decode is admitted for every tensor and its buffers before CUDA starts

Before any CUDA call, `generate` SHALL take admission for every tensor it copies plus the staging buffer, the KV caches for its capacity, the activations, the token state and, with `--check`, the host reference KV cache for the whole prompt. It SHALL confirm it before allocating, and allocate the reference cache, every page touched, before releasing it. The KV capacity SHALL be private to the runtime, and every step checked against it.

#### Scenario: Context that does not fit
- **WHEN** spark-5's vLLM engine holds about 100 GiB, so about 16.6 GiB can be admitted
- **THEN** 1K and 4K context fit beside it (14.96 GiB of weights plus up to 0.5 GiB of KV cache), and 32K (4 GiB of KV cache) needs the engine stopped

#### Scenario: Checked long prompt
- **WHEN** Llama 3.1 8B runs an 8192-token prompt with `--check 1` while 19.1 GiB is available
- **THEN** admission asks for 17.29 GiB instead of 16.29 (the 1 GiB reference cache) and refuses before any weight is copied

### Requirement: Decode throughput is measured against the planner

`generate --bench-ctx` SHALL time graph replays at each given context length, each from a zeroed
cache and the same input token so the order of the lengths doesn't matter, and print ms per token,
tok/s and the planner's ceiling over the timed positions (the harmonic mean of each step's ceiling).

#### Scenario: Llama 3.1 8B on GB10
- **WHEN** it runs at 1K, 4K and 32K on spark-5 with the GPU otherwise idle
- **THEN** it measures 15.5, 15.0 and 12.1 tok/s against ceilings of 15.5, 15.1 and 12.2

#### Scenario: Tiny context
- **WHEN** `--bench-ctx 1` times the steps at positions 3 to 34
- **THEN** the ceiling printed is the one over those positions (15.7 tok/s for Llama 3.1 8B), not the ceiling at a context of 1

### Requirement: Prompts are prefilled in one pass per chunk

`Runtime::prefill` SHALL run a prompt in chunks of the enabled rows (the planned prompt length in `generate`) through the row-wise kernels, the prefill GEMMs and `attn_prefill`. It SHALL fill the KV cache, then run `gemv_lm_head` and `argmax_advance` for the last position, so the next graph replay decodes that token at the next position. `generate` SHALL admit the prefill activations with the rest.

#### Scenario: Graph handoff
- **WHEN** a captured graph is replayed right after a prefill, with no host input
- **THEN** it decodes the prefill's greedy token at the position after the prompt

### Requirement: Time to first token is measured

`generate` SHALL print the prompt's time to the first token. `--bench-prefill L` SHALL time the prefill of L tokens (the prompt repeated) from a zeroed cache, after a warm-up, and print the time to the first token and tok/s.

#### Scenario: Llama 3.1 8B on GB10
- **WHEN** `--bench-prefill 128,2048,8192` runs on spark-5 with the GPU otherwise idle
- **THEN** the first token takes 99.8, 490.5 and 2,122.9 ms (1,283, 4,176 and 3,859 tok/s), against about 133 s for 2048 tokens as decode steps

### Requirement: A runtime runs only kernels generated for its model

`forward::Kernels` and `PrefillKernels` SHALL record the model they were generated for and the CUDA context they were loaded in. `Runtime::new`, `Runtime::from_host` and `Runtime::enable_prefill` SHALL refuse a bundle for another model or context before allocating. The decode attention scratch SHALL be sized from the split count compiled into `attn_decode_split`, not from the caller.

#### Scenario: Same projection widths, another head layout
- **WHEN** a Llama 3.1 8B runtime is given kernels from a plan with 16 heads of 256 instead of 32 of 128
- **THEN** both the decode and the prefill bundle are refused

### Requirement: A step that fails partway stops the runtime until the host sets it

The runtime SHALL keep its record of the device position and token in a `cursor::Cursor`. A step, a traced step or a prefill that fails after its launches began SHALL leave the cursor stale, and every later step SHALL be refused until `set` writes the position and token again.

#### Scenario: Readback fails at the last slot
- **WHEN** a traced step at the last cache position launches `argmax_advance` and then its logits readback fails
- **THEN** a retry is refused, instead of passing the host's capacity check while the device position is at capacity

### Requirement: Undefined token ids are shown

`tokenizer::Decoder::decode` SHALL show an id the file doesn't define as `<unknown token ID>`, and `generate` SHALL warn, listing such ids in the prompt or output.

#### Scenario: Mismatched tokenizer
- **WHEN** ids 99 and 7 are decoded with a file that defines neither
- **THEN** the text contains `<unknown token 99>` and `<unknown token 7>` where they were
