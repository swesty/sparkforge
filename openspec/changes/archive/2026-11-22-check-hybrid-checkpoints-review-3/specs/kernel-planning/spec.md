## MODIFIED Requirements

### Requirement: Planning against a checkpoint checks modules, not storage formats

Resolving a checkpoint's tensor names for planning SHALL require each planned module to exist, whatever format it is priced at and without format-specific metadata (such as a table's global scale), so a planning override (W4A16, NVFP4) plans against the checkpoint. The tensor check SHALL still require the declared storage formats.

#### Scenario: W4A16 override
- **WHEN** `plan <config> --checkpoint <dir> --format w4a16` runs on a complete hybrid checkpoint
- **THEN** it plans at W4A16, `weights <dir>` passes on the formats the checkpoint declares, and a tensor check with the override spec fails stating that only BF16, MXFP8 and NVFP4 tensors load

#### Scenario: NVFP4 override on a BF16 table
- **WHEN** `plan --checkpoint <dir> --format nvfp4` runs on a checkpoint whose PLE table is BF16 with no global scale
- **THEN** it plans, and `weights <dir>` checks the table as BF16

### Requirement: Groups hold exactly their planned members

On resolving a checkpoint, every module under a group SHALL be one the plan prices: experts `0..num_experts` with their three projections and nothing at the experts' own path, and table shards `shard_0..` with no gap, each a module at its exact path (exactly `split_ngram_parts` when configured, at least one otherwise). The shard count SHALL be compared before anything is built per declared shard. Other modules MUST fail, named.

#### Scenario: Extra shard or expert
- **WHEN** the checkpoint adds `shard_128` to the 128-shard table with an agreeing NVFP4 entry, or expert 512 to a 512-expert layer
- **THEN** resolution fails naming the extra module

#### Scenario: Module inside an expert
- **WHEN** the checkpoint adds `layers.0.mlp.experts.0.observer` with an agreeing NVFP4 entry
- **THEN** resolution fails naming the observer

#### Scenario: No shard count and no shards
- **WHEN** `split_ngram_parts` is omitted and the table holds no shards, or shards with a gap in their numbering
- **THEN** resolution fails naming the table, while the complete checkpoint resolves

#### Scenario: Hollow shard or experts-path weight
- **WHEN** a shard holds only an observer's tensor, or a weight sits at `layers.0.mlp.experts` itself
- **THEN** resolution fails naming the shard or the experts path

#### Scenario: Extreme shard count
- **WHEN** `split_ngram_parts` is 4294967295 and the checkpoint holds 4 shards
- **THEN** the check fails promptly saying how many shards are missing, without enumerating them

### Requirement: Each MTP layer has its own formats

Every MTP lookup SHALL be made at one layer index, and the MTP budget and tensor schema SHALL use each layer's own formats, including the shared-expert gate's. The input projections (`mtp.fc_embedding`, `mtp.fc_hidden`) and the final mixer SHALL be counted once, however many draft layers there are.

#### Scenario: Two draft layers in different formats
- **WHEN** a spec with two MTP layers has layer 0's `v_proj` MXFP8 and layer 1's BF16, with tensors to match
- **THEN** the tensor check passes, and only layer 0's projection is priced at MXFP8, with and without the checkpoint

#### Scenario: Shared input projections
- **WHEN** the tiny spec is planned with zero, one and two MTP layers in the same formats
- **THEN** twice the one-layer MTP bytes minus the two-layer bytes equals the FC matrices plus the final mixer

#### Scenario: MXFP8 shared-expert gate
- **WHEN** the tiny checkpoint's `mtp.layers.0.mlp.shared_expert_gate` is MXFP8 with tensors to match
- **THEN** the tensor check passes, the resident budget falls by 62 bytes, and the entry is used

## ADDED Requirements

### Requirement: The PLE module and its table are located independently

The PLE module (`layers.<n>.ple` or `ple`) and its n-gram embedding (under that module, or at the model-level `ple_embedding`) SHALL each be found at exactly one location; two locations for either MUST fail, naming both.

#### Scenario: Sole model-level table
- **WHEN** the projections are under `model.language_model.ple` and the only table is at `model.language_model.ple_embedding.ngram_embedding`
- **THEN** resolution and the tensor check pass, while a second table under `ple.ple_embedding` fails naming both
