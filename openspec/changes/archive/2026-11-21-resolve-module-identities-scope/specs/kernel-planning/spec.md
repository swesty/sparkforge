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

#### Scenario: Model-level table split across formats
- **WHEN** the PLE table lives at `ple_embedding.ngram_embedding` with one shard MXFP8 and one at the default
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
