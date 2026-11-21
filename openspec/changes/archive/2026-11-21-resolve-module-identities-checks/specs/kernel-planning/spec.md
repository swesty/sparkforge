## MODIFIED Requirements

### Requirement: Formats resolve onto a checkpoint's modules

Given a checkpoint's tensor names, the planner SHALL take each module's format from the map
entries covering it (at or under the entry's path); entries covering one module MUST agree, and
entries covering nothing, or modules no lookup prices, SHALL be reported as unused. MTP indices,
single-format groups, a tied head and the config SHALL be checked against what the planner
models.

#### Scenario: The checkpoint's own map
- **WHEN** the Flash Next plan is built with the checkpoint's tensor names
- **THEN** the budget and ops equal those of the config-only plan, and only `mtp.layers.48.mlp.experts` is unused

#### Scenario: Overlapping entries disagree
- **WHEN** `model.language_model.layers.1.ple.ple_embedding` is MXFP8 and `model.language_model.layers.1.ple.ple_embedding.ngram_embedding` is NVFP4
- **THEN** resolution fails naming both entries and the module they share, while the same entries in one format are accepted

#### Scenario: Alias that names no module
- **WHEN** the map adds `model.language_model.ple_embedding.ngram_embedding` as MXFP8 and the checkpoint's table lives under `layers.1.ple`
- **THEN** the entry is reported as unused and the table keeps its NVFP4 price

#### Scenario: Table split across formats
- **WHEN** one PLE table shard is declared MXFP8 and another is left at the default
- **THEN** resolution fails naming the table and both formats instead of pricing the table at one of them

#### Scenario: Extra MTP index
- **WHEN** the checkpoint holds `mtp.layers.48` modules beside the one configured MTP layer
- **THEN** resolution fails naming the indices instead of letting index 48 reprice layer 0

#### Scenario: Entry for an unplanned tower
- **WHEN** an entry covers the vision tower's experts, which the checkpoint holds
- **THEN** the entry is reported as unused and the text budget is unchanged

#### Scenario: Config from another checkpoint
- **WHEN** the config given with `--checkpoint` differs from that checkpoint's own `config.json`
- **THEN** planning fails instead of resolving one generation's map against another's modules

#### Scenario: Invalid shard reference in the index
- **WHEN** the index names a shard by a parent or absolute path, or not by a string
- **THEN** reading its tensor names fails as the loader would, without opening any shard

### Requirement: Format lookups are scoped to the language model

Per-layer format lookups SHALL match only language-model module paths, so an entry of another tower
(vision, MTP) with the same layer suffix never sets a text layer's format. `lm_head`, PLE modules
(`ple.<m>`, or `layers.<n>.ple.<m>` with n a configured PLE layer p or p−1) and MTP modules
(`mtp.layers.<n>.<m>`, n below the MTP layer count) SHALL match only at those paths. A tied
`lm_head` SHALL take the embedding's format, which is BF16 unless an entry covers it.

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

#### Scenario: Tied head with its own entry
- **WHEN** a tied model's map lists `embed_tokens` as MXFP8 and `lm_head` as NVFP4
- **THEN** parsing fails naming both entries

#### Scenario: General weight format
- **WHEN** a hybrid plan uses `--format nvfp4` and no entry covers the embedding
- **THEN** the embedding gather stays BF16 at 2 bytes per element
