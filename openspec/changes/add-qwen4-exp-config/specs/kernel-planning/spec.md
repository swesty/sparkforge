# Spec Delta

## ADDED Requirements

### Requirement: Hybrid models are planned as one op per distinct kernel

For a hybrid model the planner SHALL emit one op class per distinct shape and format, each with its
shape, weight format, instances per decode token and bytes per token, instead of one entry per
layer. Layers of one module class in different formats SHALL be separate op classes, each counted
at its own size. Prefill tiles SHALL NOT be planned for hybrid models yet.

#### Scenario: One layer in a different format
- **WHEN** layer 0's `in_proj_qkv` is BF16 and the other 35 are MXFP8
- **THEN** `gdn_in_proj_qkv` is MXFP8 ×35 and `gdn_in_proj_qkv_bf16` is BF16 ×1

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

### Requirement: Dense plan output is unchanged by hybrid support

The plan JSON of a dense model SHALL NOT gain fields from hybrid support: hybrid-only fields are
omitted when empty.

#### Scenario: Llama and Qwen3-30B-A3B
- **WHEN** `sparkforge plan --json` runs on `configs/llama-3.1-8b.json` or `configs/qwen3-30b-a3b.json`
- **THEN** the output equals the checked-in golden file from before this change

### Requirement: Unused format-map entries are reported

Every mixed-precision map entry that no planner lookup consumes SHALL be reported as a plan note
naming it, so an unfamiliar module path layout never falls back to the default format silently.

#### Scenario: Unknown module path
- **WHEN** the Flash Next map gains an entry `model.decoder.blocks.0.ffn`
- **THEN** the plan notes that entry as not used by the plan
