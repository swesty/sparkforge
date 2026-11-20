# Spec Delta

## MODIFIED Requirements

### Requirement: Format lookups are scoped to the language model

Per-layer format lookups SHALL match only language-model module paths, so an entry of another tower
(vision, MTP) with the same layer suffix never sets a text layer's format. `lm_head` SHALL match
only at its model-level paths, other model-level modules only on a module boundary, and each MTP
projection SHALL be looked up by its own path.

#### Scenario: Vision experts in the map
- **WHEN** the map also lists `model.visual.layers.0.mlp.experts` as MXFP8
- **THEN** the text experts stay NVFP4, the budget is unchanged, and the vision entry is reported as unused

#### Scenario: Auxiliary head in a layer
- **WHEN** the map lists `model.language_model.layers.0.aux_lm_head` as NVFP4
- **THEN** `lm_head` stays BF16, both budgets are unchanged, and the entry is reported as unused

#### Scenario: MTP value projection
- **WHEN** the map lists `mtp.layers.0.self_attn.v_proj` as MXFP8 and not `k_proj`
- **THEN** resident weights fall by 1,269,760 bytes and the entry is consumed
