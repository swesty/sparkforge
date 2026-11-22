## MODIFIED Requirements

### Requirement: Groups hold exactly their planned members

On resolving a checkpoint, every module under a group (a layer's routed experts, the PLE table) SHALL be one the plan prices: experts `0..num_experts` with their three projections, and shards `shard_0..` numbered with no gap (exactly `split_ngram_parts` of them when configured, at least one otherwise). Any other module MUST fail resolution, naming it.

#### Scenario: Extra shard or expert
- **WHEN** the checkpoint adds `shard_128` to the 128-shard table with an agreeing NVFP4 entry, or expert 512 to a 512-expert layer
- **THEN** resolution fails naming the extra module

#### Scenario: Module inside an expert
- **WHEN** the checkpoint adds `layers.0.mlp.experts.0.observer` with an agreeing NVFP4 entry
- **THEN** resolution fails naming the observer

#### Scenario: No shard count and no shards
- **WHEN** `split_ngram_parts` is omitted and the table holds no shards, or shards with a gap in their numbering
- **THEN** resolution fails naming the table, while the complete checkpoint resolves

### Requirement: Each MTP layer has its own formats

Every MTP lookup SHALL be made at one layer index, and the MTP budget and tensor schema SHALL use each layer's own formats. The input projections (`mtp.fc_embedding`, `mtp.fc_hidden`) and the final mixer SHALL be counted once, however many draft layers there are.

#### Scenario: Two draft layers in different formats
- **WHEN** a spec with two MTP layers has layer 0's `v_proj` MXFP8 and layer 1's BF16, with tensors to match
- **THEN** the tensor check passes, and only layer 0's projection is priced at MXFP8, with and without the checkpoint

#### Scenario: Shared input projections
- **WHEN** the tiny spec is planned with zero, one and two MTP layers in the same formats
- **THEN** twice the one-layer MTP bytes minus the two-layer bytes equals the FC matrices plus the final mixer

## ADDED Requirements

### Requirement: Planning against a checkpoint checks modules, not storage formats

Resolving a checkpoint's tensor names for planning SHALL require each planned module to exist, whatever format it is priced at, so a planning override (for example W4A16) plans against the checkpoint. The tensor check SHALL still require the declared storage formats.

#### Scenario: W4A16 override
- **WHEN** `plan <config> --checkpoint <dir> --format w4a16` runs on a complete hybrid checkpoint
- **THEN** it plans at W4A16, `weights <dir>` passes on the formats the checkpoint declares, and a tensor check with the override spec fails stating that only BF16, MXFP8 and NVFP4 tensors load
