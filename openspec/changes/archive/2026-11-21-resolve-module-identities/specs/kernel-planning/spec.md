## ADDED Requirements

### Requirement: Formats resolve onto a checkpoint's modules

Given a checkpoint's tensor names, the planner SHALL take each module's format from the map
entries that cover that module: an entry applies to the modules at its path or under it, entries
that cover one module MUST agree, and an entry that covers no module of the checkpoint SHALL be
reported as unused. Paths are compared after removing any language-model prefix.

#### Scenario: The checkpoint's own map
- **WHEN** the Flash Next plan is built with the checkpoint's tensor names
- **THEN** the budget and ops equal those of the config-only plan, and only `mtp.layers.48.mlp.experts` is unused

#### Scenario: Overlapping entries disagree
- **WHEN** `model.language_model.layers.1.ple.ple_embedding` is MXFP8 and `model.language_model.layers.1.ple.ple_embedding.ngram_embedding` is NVFP4
- **THEN** resolution fails naming both entries and the module they share, while the same entries in one format are accepted

#### Scenario: Alias that names no module
- **WHEN** the map adds `model.language_model.ple_embedding.ngram_embedding` as MXFP8 and the checkpoint's table lives under `layers.1.ple`
- **THEN** the entry is reported as unused and the table keeps its NVFP4 price

## MODIFIED Requirements

### Requirement: Format lookups are scoped to the language model

Per-layer format lookups SHALL match only language-model module paths, so an entry of another tower
(vision, MTP) with the same layer suffix never sets a text layer's format. `lm_head`, PLE modules
(`ple.<m>`, or `layers.<n>.ple.<m>` with n a configured PLE layer p or p−1) and MTP modules
(`mtp.layers.<n>.<m>`, n below the configured MTP layer count) SHALL match only at those paths, and
each projection by its own path. A tied `lm_head` SHALL take the embedding's format.

#### Scenario: Vision experts in the map
- **WHEN** the map also lists `model.visual.layers.0.mlp.experts` as MXFP8
- **THEN** the text experts stay NVFP4, the budget is unchanged, and the vision entry is reported as unused

#### Scenario: Auxiliary head in a layer
- **WHEN** the map lists `model.language_model.layers.0.aux_lm_head` as NVFP4
- **THEN** `lm_head` stays BF16, both budgets are unchanged, and the entry is reported as unused

#### Scenario: Auxiliary PLE or MTP path
- **WHEN** the map lists `model.language_model.layers.0.aux.ple.key_proj` or `mtp.layers.0.aux.self_attn.v_proj`
- **THEN** neither budget changes and the entry is reported as unused

#### Scenario: PLE entry at the wrong layer
- **WHEN** the map lists `model.language_model.layers.999.ple.key_proj` as MXFP8
- **THEN** neither budget changes and the entry is reported as unused

#### Scenario: MTP value projection
- **WHEN** the map lists `mtp.layers.0.self_attn.v_proj` as MXFP8 and not `k_proj`
- **THEN** resident weights fall by 1,269,760 bytes and the entry is consumed

#### Scenario: Stale MTP layer index
- **WHEN** the map lists `mtp.layers.999.self_attn.v_proj` as MXFP8
- **THEN** the budget is unchanged and the entry is reported as unused

#### Scenario: Tied embedding in another format
- **WHEN** a tied model's map lists `model.language_model.embed_tokens` as MXFP8
- **THEN** `lm_head` is planned as MXFP8 and the matrix is resident once at MXFP8

### Requirement: Unused format-map entries are reported

Every mixed-precision map entry that no planner lookup consumes SHALL be reported as a plan note
naming it, so an unfamiliar module path layout never falls back to the default format silently.

#### Scenario: Unknown module path
- **WHEN** the Flash Next map gains an entry `model.decoder.blocks.0.ffn`
- **THEN** the plan notes that entry as not used by the plan

#### Scenario: Second MTP expert entry
- **WHEN** the Flash Next map lists the MTP experts under `mtp.layers.0` and `mtp.layers.48`, and the model has one MTP layer
- **THEN** `mtp.layers.48.mlp.experts` is reported as unused and the budget is unchanged
