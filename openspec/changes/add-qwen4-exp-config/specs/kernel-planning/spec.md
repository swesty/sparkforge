# Spec Delta

## ADDED Requirements

### Requirement: Hybrid models are planned as one op per distinct kernel

For a hybrid model the planner SHALL emit one op class per distinct shape and format, each with its
shape, weight format, instances per decode token and bytes per token, instead of one entry per
layer. Prefill tiles SHALL NOT be planned for hybrid models until their formats have tile models.

#### Scenario: Flash Next op classes
- **WHEN** `configs/qwen3.8-flash-next-nvfp4.json` is planned
- **THEN** 34 uniquely named op classes cover HyperConnection, Gated DeltaNet, sparse attention, MoE, PLE and `lm_head`, for example `gdn_in_proj_qkv` MXFP8 ×36 and `qsa_q_proj` N=12288 K=2560 ×12

### Requirement: The hybrid decode budget counts formats, state and context terms

The hybrid budget SHALL count each linear at its module's format, Gated DeltaNet state read and
written, and per context token the FP8 KV of the selected positions plus the indexer key scan.
Resident weights SHALL include every expert copy, the PLE table and the MTP layers.

#### Scenario: Flash Next at 32K
- **WHEN** `configs/qwen3.8-flash-next-nvfp4.json` is planned at a 32768-token context
- **THEN** decode reads 7.338 GB per token context-free and about 7.375 GB at 32K, and resident weights are about 105.4 GB, each within 1% of the checkpoint's header sums

### Requirement: Unused format-map entries are reported

Every mixed-precision map entry that no planner lookup consumes SHALL be reported as a plan note
naming it, so an unfamiliar module path layout never falls back to the default format silently.

#### Scenario: Unknown module path
- **WHEN** the Flash Next map gains an entry `model.decoder.blocks.0.ffn`
- **THEN** the plan notes that entry as not used by the plan
