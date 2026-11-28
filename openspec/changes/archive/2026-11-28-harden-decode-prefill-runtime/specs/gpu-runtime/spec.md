## MODIFIED Requirements

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

## ADDED Requirements

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
