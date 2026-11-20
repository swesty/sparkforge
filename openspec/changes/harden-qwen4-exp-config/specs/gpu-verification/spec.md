# Spec Delta

## MODIFIED Requirements

### Requirement: Verification refuses a model with no generated kernels

`sparkforge verify` MUST fail with a message naming the model when its plan generates no kernels,
rather than reporting an empty pass, and MUST do so before creating a CUDA context.

#### Scenario: Flash Next verify
- **WHEN** `sparkforge verify configs/qwen3.8-flash-next-nvfp4.json` runs
- **THEN** it fails stating that no kernels are generated for this model yet

#### Scenario: Host without a GPU
- **WHEN** the same command runs on a host where CUDA cannot initialize
- **THEN** it reports the same planned-only error instead of a CUDA failure
