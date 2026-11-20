# Spec Delta

## ADDED Requirements

### Requirement: Hybrid plans generate no kernels until their op classes exist

For a hybrid plan, code generation SHALL list every op class as `planned` and SHALL emit no CUDA
kernel source, because the dense kernels assume full-head RoPE, dense attention and uniform
weights. Its cache key SHALL include the plan, so two hybrid plans never share an output directory.

#### Scenario: Flash Next emit
- **WHEN** `sparkforge emit configs/qwen3.8-flash-next-nvfp4.json` runs
- **THEN** the manifest has one `planned` entry per op class and the source contains no `__global__` function
