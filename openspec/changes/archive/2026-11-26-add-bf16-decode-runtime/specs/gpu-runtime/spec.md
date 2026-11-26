## ADDED Requirements

### Requirement: Greedy decode replays one CUDA Graph per step

`sparkforge generate` SHALL decode a dense BF16 checkpoint greedily on a GB10 from a pre-tokenized
prompt. A step SHALL read its token and position from device memory and end by writing the greedy
next token and advancing the position, so one captured CUDA Graph on a non-default stream replays
every step, prompt and generated, with no host work between tokens.

#### Scenario: Llama 3.1 8B
- **WHEN** `generate` runs Llama 3.1 8B on a 20-token prompt about general relativity for 48 steps
- **THEN** the output is coherent text ("the curvature of spacetime caused by the presence of mass and energy...")

#### Scenario: Graph replay
- **WHEN** a captured step and the same step as plain launches run from the same state
- **THEN** their logits are bitwise identical and both record the argmax as the next token

### Requirement: Decode is admitted for every tensor and its buffers before CUDA starts

Before any CUDA call, `generate` SHALL take admission for every tensor it copies (linears, norms
and the embedding) plus the staging buffer, the KV caches for its capacity, the activations and the
token state, and SHALL confirm it before allocating. The KV capacity SHALL be private to the runtime,
and every step SHALL be checked against it before launch.

#### Scenario: Context that does not fit
- **WHEN** spark-5's vLLM engine holds about 100 GiB, so about 16.6 GiB can be admitted
- **THEN** 1K and 4K context fit beside it (14.96 GiB of weights plus up to 0.5 GiB of KV cache), and 32K (4 GiB of KV cache) needs the engine stopped

### Requirement: Decode throughput is measured against the planner

`generate --bench-ctx` SHALL time graph replays at each given context length and print ms per token,
tok/s and the planner's ceiling at that context.

#### Scenario: Llama 3.1 8B on GB10
- **WHEN** it runs at 1K, 4K and 32K on spark-5 with the GPU otherwise idle
- **THEN** it measures 15.5, 15.0 and 12.1 tok/s against ceilings of 15.5, 15.1 and 12.2
