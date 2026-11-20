# Spec Delta

## Purpose

Runs a planned model on one GB10 at batch 1 with sparkforge's generated kernels: prompt in, tokens
out, with measured speed reported against the model's byte ceiling and correctness checked against
reference activations.

## ADDED Requirements

### Requirement: The engine runs a planned model end to end

`sparkforge generate` SHALL take a supported model config, its checkpoint and a prompt, and SHALL
produce tokens using only kernels the planner generated for that model. It MUST report decode tok/s,
time to first token, and the byte-ceiling tok/s for the configuration it ran.

#### Scenario: Greedy decode
- **WHEN** `sparkforge generate` runs a supported checkpoint with a prompt and temperature 0
- **THEN** it prints the generated text, decode tok/s, time to first token and the ceiling tok/s

#### Scenario: Unsupported model
- **WHEN** the config describes an architecture or format the planner cannot compute correctly
- **THEN** the command fails before loading weights, naming what is unsupported

### Requirement: Engine memory is budgeted before any allocation

Before allocating, the engine SHALL sum resident weights, KV and recurrent state, speculative
snapshots, activations and workspaces from the plan, and MUST refuse to start if the sum exceeds
`MemAvailable` minus 4 GiB, stating both numbers. Admission SHALL follow the weight-loading
serialization rule.

#### Scenario: Model does not fit
- **WHEN** the planned total exceeds `MemAvailable` minus 4 GiB
- **THEN** the engine exits before any GPU allocation, printing the planned total and `MemAvailable`

### Requirement: Steady-state steps replay captured graphs

Each step shape (decode, draft, verify, prefill chunk) SHALL execute as one captured CUDA Graph.
Per-step values such as position, sequence length and token ids MUST be updated without
re-instantiating the graph, and a steady-state decode step MUST NOT launch kernels individually.

#### Scenario: Decode step launch count
- **WHEN** the engine decodes 256 tokens after warm-up
- **THEN** each step issues one graph launch per step shape it uses, and the reported kernels-per-step matches the step program

### Requirement: Greedy output is identical with and without speculation

With temperature 0, the token sequence produced with speculative decoding enabled MUST equal the
sequence produced with it disabled, for any number of draft tokens and any draft-head option.

#### Scenario: MTP on and off
- **WHEN** the same prompt is generated greedily with MTP disabled and with MTP k=4
- **THEN** the two token sequences are identical

### Requirement: Speculative sampling preserves the target distribution

With temperature above 0, speculative decoding SHALL accept or resample draft tokens by rejection
sampling against the target model's probabilities, so the output distribution equals non-speculative
sampling. Draft-only approximations MUST NOT change the target probabilities.

#### Scenario: Truncated draft head
- **WHEN** the draft head scores a truncated vocabulary
- **THEN** verification still uses the full-vocabulary target probabilities and draft probabilities normalized over the truncated vocabulary

### Requirement: Rejected drafts roll back all recurrent state

After a verify step accepts j of k draft tokens, every recurrent state (linear-attention state and
convolution state) and every KV entry MUST equal what non-speculative decoding would hold after the
same accepted tokens.

#### Scenario: Partial acceptance
- **WHEN** a verify step accepts 2 of 4 drafts
- **THEN** the next step's recurrent and KV state matches a non-speculative run over the same tokens within kernel tolerance

### Requirement: Correctness is gated before speed is claimed

A model's engine MUST pass a per-layer comparison against reference activations, and a
teacher-forced top-1 agreement check against a reference engine within that engine's own noise
band, before any tok/s result for it is recorded as a milestone.

#### Scenario: Golden diff
- **WHEN** the engine processes a teacher-forced reference sequence
- **THEN** it reports per-layer maximum and mean error against the reference activations and the top-1 agreement at the final layer

### Requirement: Numerics-changing options are opt-in and gated

An option that changes numerics or draft quality, such as lower-precision copies of checkpoint
weights or a truncated draft head, MUST be off by default. Enabling one MUST report its effect on
tok/s, on top-1 agreement with the reference engine and, for draft options, on acceptance rate.

#### Scenario: MXFP8 HyperConnection weights
- **WHEN** the user enables the MXFP8 HyperConnection weight option
- **THEN** the run reports tok/s and top-1 agreement with and without the option

### Requirement: Full-model runs on a shared node do not displace a serving engine silently

The engine SHALL NOT start when another process holds the memory it needs; the budget refusal names
the shortfall. Tooling that stops a serving engine to free a node MUST require an explicit operator
confirmation for each stop.

#### Scenario: Fleet engine resident
- **WHEN** a serving engine holds 80% of the GB10's memory and the engine needs 76 GiB
- **THEN** the engine refuses to start with the budget message and does not stop the other process
