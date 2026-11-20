# Spec Delta

## ADDED Requirements

### Requirement: Verification refuses a model with no generated kernels

`sparkforge verify` MUST fail with a message naming the model when its plan generates no kernels,
rather than reporting an empty pass.

#### Scenario: Flash Next verify
- **WHEN** `sparkforge verify configs/qwen3.8-flash-next-nvfp4.json` runs
- **THEN** it fails stating that no kernels are generated for this model yet
