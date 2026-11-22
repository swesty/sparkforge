## ADDED Requirements

### Requirement: The PLE table's shard layout is read from the config

`split_ngram_parts` and `make_ngram_vocab_size_divisible_by` SHALL be parsed when present and MUST be positive. Planning SHALL NOT need them; checking a checkpoint's PLE table MUST fail when `split_ngram_parts` is absent.

#### Scenario: Flash Next config
- **WHEN** the Flash Next config is parsed
- **THEN** the PLE table has 128 shards and a size multiple of 128
