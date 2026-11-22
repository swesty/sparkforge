## ADDED Requirements

### Requirement: Groups hold exactly their planned members

On resolving a checkpoint, each layer's routed experts SHALL be exactly indices `0..num_experts`, and the PLE table's shards exactly `shard_0..shard_{split_ngram_parts - 1}` when the count is configured. Any other member MUST fail resolution, naming it.

#### Scenario: Extra shard or expert
- **WHEN** the checkpoint adds `shard_128` to the 128-shard table with an agreeing NVFP4 entry, or expert 512 to a 512-expert layer
- **THEN** resolution fails naming the extra module

### Requirement: Each MTP layer has its own formats

Every MTP lookup SHALL be made at one layer index, and the MTP budget and tensor schema SHALL use each layer's own formats.

#### Scenario: Two draft layers in different formats
- **WHEN** a spec with two MTP layers has layer 0's `v_proj` MXFP8 and layer 1's BF16, with tensors to match
- **THEN** the tensor check passes, and only layer 0's projection is priced at MXFP8, with and without the checkpoint
