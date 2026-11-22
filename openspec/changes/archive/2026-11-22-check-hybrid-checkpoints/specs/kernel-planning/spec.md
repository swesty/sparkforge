## MODIFIED Requirements

### Requirement: Formats resolve onto a checkpoint's modules

Given a checkpoint's tensor names, the planner SHALL take each projection's format from the entries covering its exact module, and each group's (routed experts, the PLE table) from its members. Entries covering one module MUST agree; entries covering nothing, or modules no lookup prices, SHALL be reported as unused. MTP indices, one PLE location, single-format groups, every planned module, a tied head and the config SHALL be checked.

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
- **THEN** planning fails instead of resolving one generation's map against another's modules, and the spec is always parsed from the checkpoint's own config

#### Scenario: Invalid shard reference in the index
- **WHEN** the index names a shard by a parent or absolute path, or not by a string
- **THEN** reading its tensor names fails as the loader would, without opening any shard

#### Scenario: Model-level table split across formats
- **WHEN** the PLE lives at model level (`ple.`) and its table has one shard MXFP8 and the rest at the default
- **THEN** resolution fails naming the table

#### Scenario: Non-canonical MTP index
- **WHEN** the checkpoint holds `mtp.layers.00` modules beside `mtp.layers.0`
- **THEN** resolution fails naming `mtp.layers.00`

#### Scenario: Tensors under a projection
- **WHEN** layer 3's `q_proj` has no entry and an observer tensor under it is declared MXFP8
- **THEN** `q_proj` keeps its default price and the observer entry is reported as unused

#### Scenario: Agreeing parent and child entries
- **WHEN** `layers.1.ple.ple_embedding` and its `ngram_embedding` child are both NVFP4
- **THEN** neither entry is reported as unused

#### Scenario: Two PLE locations
- **WHEN** the checkpoint holds an NVFP4 table under `layers.1.ple` and an MXFP8 table at `ple_embedding.ngram_embedding`
- **THEN** resolution fails naming both locations

#### Scenario: Shard entries beside a table scalar
- **WHEN** the table keeps `weight_scale_2` at its own path and the map adds an agreeing NVFP4 entry for `shard_0`
- **THEN** both entries are used, and with per-shard entries only the table is still one format

#### Scenario: Projection only under an observer
- **WHEN** layer 3's `q_proj` tensors are missing and only an observer's tensor under it is declared MXFP8
- **THEN** resolution fails naming `layers.3.self_attn.q_proj`

#### Scenario: Missing module
- **WHEN** a GDN norm, one expert projection, one PLE table shard or an MTP mixer norm is missing
- **THEN** resolution fails naming that module

## ADDED Requirements

### Requirement: Lookups agree with and without a checkpoint

Without a checkpoint, an MTP entry SHALL match only a configured layer index spelled as a canonical decimal, as a checkpoint lookup does. A dense head and embedding SHALL take their resolved formats: a tied head is the embedding matrix, resident once.

#### Scenario: Non-canonical MTP index
- **WHEN** the map lists `mtp.layers.00.self_attn.v_proj` as MXFP8 and a plan is built from the config alone
- **THEN** the budget is unchanged and the entry is reported as unused, while `mtp.layers.0.self_attn.v_proj` lowers it

#### Scenario: Tied dense embedding
- **WHEN** a tied Llama 3.1 8B spec is resolved with `model.embed_tokens` as MXFP8
- **THEN** the head GEMV reads MXFP8 bytes and the matrix is resident once at MXFP8
