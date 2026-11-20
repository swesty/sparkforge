# Spec Delta

## MODIFIED Requirements

### Requirement: Hybrid and mixed-precision checkpoints are refused

Binding MUST fail, before any GPU work, for a hybrid-layout checkpoint or one whose config declares
mixed precision, until the mixed-format loader exists, naming the architecture. A format override
used for planning MUST NOT lift this refusal.

#### Scenario: Flash Next checkpoint
- **WHEN** `sparkforge weights` opens a Qwen3.8 Flash Next checkpoint
- **THEN** it fails stating that hybrid or mixed-precision checkpoints do not load yet

#### Scenario: Override used for planning
- **WHEN** a dense MIXED_PRECISION config is parsed with a BF16 override and its checkpoint is bound
- **THEN** binding fails stating that mixed-precision checkpoints do not load yet
