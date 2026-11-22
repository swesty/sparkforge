## ADDED Requirements

### Requirement: The checked PLE table prices the reported budget

`sparkforge weights` SHALL report the plan's resident budget with the PLE table at the rows it checked. PLE row totals and padding that cannot be represented in 64 bits MUST fail the check, not wrap or panic.

#### Scenario: One more row per shard
- **WHEN** two checkpoints of one config have per-head sizes making 15- and 16-row shards in 4 NVFP4 shards of 16 columns
- **THEN** both pass, and the second's resident budget is 36 bytes larger

#### Scenario: Overflowing padding
- **WHEN** the per-head sizes sum within 64 bits but their padding to the configured multiple does not
- **THEN** the check fails saying the row count overflows

### Requirement: A tied embedding carries the head GEMV's tensors

With tied embeddings, the embedding matrix SHALL be checked as a GEMV operand, counted once: in NVFP4 it MUST have its activation scale, and in weight-only NVFP4 it MUST NOT.

#### Scenario: Tied NVFP4 and weight-only NVFP4
- **WHEN** a tied checkpoint's embedding is NVFP4, or W4A16 NVFP4
- **THEN** NVFP4 fails without `embed_tokens.input_scale` and passes with it, and W4A16 NVFP4 passes without it and reports it as unexpected if present
