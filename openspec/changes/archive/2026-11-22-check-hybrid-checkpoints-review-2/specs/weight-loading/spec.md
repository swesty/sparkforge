## MODIFIED Requirements

### Requirement: The checked PLE table prices the reported budget

`sparkforge weights` SHALL check a hybrid checkpoint's tensors before building any plan, and report the resident budget with the PLE table at the rows it checked. Resolving a spec onto another checkpoint MUST clear rows checked earlier. PLE row totals and padding that cannot be represented in 64 bits MUST fail the check, not wrap or panic.

#### Scenario: One more row per shard
- **WHEN** two checkpoints of one config have per-head sizes making 15- and 16-row shards in 4 NVFP4 shards of 16 columns
- **THEN** both pass, and the second's resident budget is 36 bytes larger

#### Scenario: Overflowing padding
- **WHEN** the per-head sizes sum within 64 bits but their padding to the configured multiple does not
- **THEN** the check fails saying the row count overflows

#### Scenario: Spec reused across checkpoints
- **WHEN** a spec checked against the 15-row checkpoint is resolved onto the 16-row one
- **THEN** its checked rows are cleared and its budget equals a fresh resolution's, until the 16-row checkpoint is checked
