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

Each step shape (decode, speculative step of drafts plus verify plus accept, prefill chunk) SHALL
execute as one captured CUDA Graph. Per-step values such as position, sequence length and token
ids MUST be updated without re-instantiating the graph, and a steady-state step MUST NOT launch
kernels individually.

#### Scenario: Decode step launch count
- **WHEN** the engine decodes 256 tokens after warm-up
- **THEN** each step reports one graph submission per step shape it uses, zero individual host kernel launches, and executed kernel nodes equal to the step program's kernel count

### Requirement: Prompt ingestion advances state by exactly the prompt length

Prefill SHALL consume a prompt of any length, including a final chunk shorter than the planned
chunk size, and MUST advance KV, indexer, recurrent and token-history state by exactly the number of
real prompt tokens; padded positions MUST NOT change any state.

#### Scenario: One token past a full chunk
- **WHEN** the prompt is 16,385 tokens long and the chunk size is 16,384
- **THEN** the committed length is 16,385 and the first generated token matches token-by-token ingestion

### Requirement: Overlapped launches do not change results

Overlapping a kernel with its predecessor (programmatic dependent launch) MUST NOT change any
output. A kernel MAY read before the dependency wait only data its predecessor does not produce,
such as weights at fixed addresses; data-dependent reads, including weights selected by routing,
MUST come after the wait.

#### Scenario: Changing expert routes
- **WHEN** consecutive tokens route to different experts and the step runs with overlapped launches and then serialized
- **THEN** the logits of every step are bitwise identical between the two runs

### Requirement: Speculation runs only with a resident PLE table

Speculative decoding SHALL run only when the PLE table is resident in GPU memory. When the memory
budget keeps the table on disk, the engine MUST run non-speculative decode and report that
speculation is disabled and why, instead of failing or staging table rows inside a step.

#### Scenario: Resident table denied
- **WHEN** a request selects MTP k=4 and the memory budget keeps the PLE table on disk
- **THEN** the request completes with non-speculative decode and the run reports speculation as disabled because the table is on disk

### Requirement: Streamed output keeps every step's tokens in order

Each step's accepted tokens and count SHALL stay paired in a result slot that is not reused until
the host has copied it, so output is complete and in order however late the host reads.

#### Scenario: Delayed reader
- **WHEN** the host reads results late while consecutive speculative steps accept different numbers of drafts
- **THEN** the streamed tokens equal the tokens accepted, in order, with none dropped or duplicated

### Requirement: Greedy output is identical with and without speculation

With temperature 0, the token sequence produced with speculative decoding enabled MUST equal the
sequence produced with it disabled, for any number of draft tokens and any draft-head option. Ties
between equal logits SHALL resolve to the lowest token id.

#### Scenario: MTP on and off
- **WHEN** the same prompt is generated greedily with MTP disabled and with MTP k=4
- **THEN** the two token sequences are identical

### Requirement: Target logits do not depend on step shape

For a given position and context, the target model's logits computed in a multi-token verify step
MUST be bitwise identical to those computed in a single-token decode step, so that rounding cannot
reorder near-tied tokens between the two shapes.

#### Scenario: Near tie
- **WHEN** a position's two highest logits differ by less than one unit in the last place
- **THEN** decode and verify produce bitwise-identical logits there and select the same token

### Requirement: Speculative sampling preserves the target distribution

With temperature above 0, speculative decoding SHALL accept or resample draft tokens by rejection
sampling against the target model's probabilities, so the output distribution equals non-speculative
sampling. Draft-only approximations MUST NOT change the target probabilities.

#### Scenario: Truncated draft head
- **WHEN** the draft head scores a truncated vocabulary
- **THEN** verification still uses the full-vocabulary target probabilities and draft probabilities normalized over the truncated vocabulary

### Requirement: Rejected drafts roll back all recurrent state

After a verify step accepts j of k draft tokens, every recurrent state (linear-attention state and
convolution state) MUST equal what non-speculative decoding would hold after the same accepted
tokens.

#### Scenario: Partial acceptance
- **WHEN** a verify step accepts 2 of 4 drafts
- **THEN** the next step's recurrent state matches a non-speculative run over the same tokens within kernel tolerance

### Requirement: Each verify query sees exactly its causal prefix

In a verify step whose block starts at committed length C, the query at block offset i SHALL see
positions below C + i + 1 and none after, in attention and in the sparse indexer alike. After
acceptance, only the accepted prefix SHALL become visible to later steps.

#### Scenario: Earlier position in the same block
- **WHEN** the indexer for the query at offset 2 would select the key at offset 1 of the same block
- **THEN** that key is eligible, keys at offsets 3 and later are not, and the logits are bitwise identical to sequential decode

#### Scenario: Rejected KV suffix
- **WHEN** a verify step appended KV for 4 drafts and accepted 2
- **THEN** the next step's attention and indexer select the same positions as non-speculative decoding and never a rejected one

### Requirement: Speculative writes never destroy committed state

A verify step's provisional writes MUST NOT overwrite any entry that remains live after the step,
and MUST stay inside their allocations at every committed length up to the context limit. Every
buffer that verify writes SHALL reserve room for a full verify block beyond its live contents, and
the memory budget SHALL include that room.

#### Scenario: Partial acceptance across a ring wrap
- **WHEN** a full raw-key ring receives a 5-position verify block across its wrap point and 2 positions are accepted
- **THEN** the next step's keys, selected positions and logits equal sequential decode

#### Scenario: Verify at the context limit
- **WHEN** MTP k=4 verifies with one to four positions left before the context limit
- **THEN** every write stays inside its allocation, nothing past the limit is committed, and the tokens match sequential decode

### Requirement: Token-history lookups follow the committed prefix

Any lookup keyed by preceding tokens, such as the PLE n-gram embedding, SHALL derive its history
from the committed tokens plus the causal portion of the current block, so a rejected draft never
appears in a later lookup.

#### Scenario: Rejected draft in the n-gram window
- **WHEN** 2 of 4 drafts are accepted and the next token's n-gram row differs depending on whether the last rejected or the last accepted draft precedes it
- **THEN** the lookup uses the accepted history and the logits equal sequential decode

### Requirement: Correctness is gated before speed is claimed

Before a model's tok/s is recorded as a milestone, its engine MUST pass a per-layer activation
comparison and a teacher-forced check against a reference engine on a fixed reference set. Each
error SHALL be at most 1.5 times the reference engine's own noise floor, and top-1 agreement SHALL
meet a minimum rather than fall inside a band.

#### Scenario: Golden diff
- **WHEN** the engine processes a teacher-forced reference sequence
- **THEN** it reports per-layer error, top-1 agreement, logprob differences and mean log-likelihood, each with a PASS or FAIL against its limit

#### Scenario: Closer than the band
- **WHEN** top-1 agreement is 97.5%, above the reference engine's 96.5–97.1% noise band
- **THEN** the agreement check passes

#### Scenario: Mis-scaled output head
- **WHEN** every final logit is doubled, leaving top-1 agreement unchanged
- **THEN** the logprob and log-likelihood checks fail

### Requirement: Numerics-changing options are opt-in and gated

An option that changes target numerics, such as lower-precision copies of checkpoint weights, MUST
be off by default and SHALL pass a gate on the fixed reference set: every correctness limit, and
mean and 99th-percentile absolute logprob difference at most 10% above the checkpoint-precision
engine's. A failing option MUST report FAIL and stay off.

#### Scenario: MXFP8 HyperConnection weights
- **WHEN** the user enables the MXFP8 HyperConnection weight option
- **THEN** the run reports tok/s, top-1 agreement and the logprob figures, with a PASS or FAIL verdict

#### Scenario: Logprobs drift with top-1 unchanged
- **WHEN** an option keeps every top-1 choice but its 99th-percentile logprob difference exceeds the limit
- **THEN** the gate reports FAIL and the option is not applied

### Requirement: Draft-only options report acceptance

An option that changes only the draft model, such as a truncated draft vocabulary, MUST be off by
default and MUST report tok/s and per-draft acceptance with and without it; greedy output stays
identical by the speculation requirement.

#### Scenario: Truncated draft head
- **WHEN** the truncated draft head is enabled
- **THEN** the run reports tok/s and acceptance with and without it, and greedy output is unchanged

### Requirement: Full-model runs on a shared node do not displace a serving engine silently

The engine SHALL NOT start when another process holds the memory it needs; the budget refusal names
the shortfall. Tooling that stops a serving engine to free a node MUST require an explicit operator
confirmation for each stop.

#### Scenario: Fleet engine resident
- **WHEN** a serving engine holds 80% of the GB10's memory and the engine needs 76 GiB
- **THEN** the engine refuses to start with the budget message and does not stop the other process
