# Spec Delta

## ADDED Requirements

### Requirement: Hybrid and mixed-precision checkpoints are refused

Binding MUST fail, before any GPU work, for a hybrid-layout or mixed-precision checkpoint until
the mixed-format loader exists, naming the architecture.

#### Scenario: Flash Next checkpoint
- **WHEN** `sparkforge weights` opens a Qwen3.8 Flash Next checkpoint
- **THEN** it fails stating that hybrid or mixed-precision checkpoints do not load yet
