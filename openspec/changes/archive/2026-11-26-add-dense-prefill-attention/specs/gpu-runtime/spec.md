## MODIFIED Requirements

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

## ADDED Requirements

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
