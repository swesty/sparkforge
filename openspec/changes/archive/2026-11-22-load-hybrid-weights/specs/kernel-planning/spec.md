## MODIFIED Requirements

### Requirement: Each MTP layer has its own formats

Every MTP lookup SHALL be made at one layer index, and the MTP budget and tensor schema SHALL use each layer's own formats, including the shared-expert gate's. The input projections (`mtp.fc_embedding`, `mtp.fc_hidden`) SHALL each be looked up at their exact path and checked and priced at their own format, and the final mixer SHALL be counted once, however many draft layers there are. MTP mixer norms SHALL be priced as BF16, as they are checked.

#### Scenario: Two draft layers in different formats
- **WHEN** a spec with two MTP layers has layer 0's `v_proj` MXFP8 and layer 1's BF16, with tensors to match
- **THEN** the tensor check passes, and only layer 0's projection is priced at MXFP8, with and without the checkpoint

#### Scenario: Shared input projections
- **WHEN** the tiny spec is planned with zero, one and two MTP layers in the same formats
- **THEN** twice the one-layer MTP bytes minus the two-layer bytes equals the FC matrices plus the final mixer, whose norm is 2 bytes per element

#### Scenario: MXFP8 shared-expert gate
- **WHEN** the tiny checkpoint's `mtp.layers.0.mlp.shared_expert_gate` is MXFP8 with tensors to match
- **THEN** the tensor check passes, the resident budget falls by 62 bytes, and the entry is used

#### Scenario: Declared MXFP8 input projection
- **WHEN** the map lists `mtp.fc_embedding` as MXFP8
- **THEN** a BF16 weight without a scale fails the check naming both tensors, and MXFP8 tensors pass and price 3,968 bytes below BF16

## ADDED Requirements

### Requirement: The table root and expert counts are checked from live names

On resolving a checkpoint, the PLE table's own path MAY hold only its shared `weight_scale_2`, which is never required there; any other tensor there MUST fail, named. Every layer's live expert count SHALL be compared with the declared count before anything is built per declared expert, and a shortfall MUST fail naming the layer and how many experts are missing.

#### Scenario: Matrix at the table root
- **WHEN** the tiny checkpoint adds a `weight` at the table's own path beside its `weight_scale_2`
- **THEN** resolution fails naming it, while a BF16 table with no scalar still plans under an NVFP4 override

#### Scenario: Extreme expert count
- **WHEN** the spec declares 4294967295 experts and the checkpoint holds 4 per layer
- **THEN** resolution fails promptly naming `layers.0.mlp.experts` and 4294967291 missing experts
