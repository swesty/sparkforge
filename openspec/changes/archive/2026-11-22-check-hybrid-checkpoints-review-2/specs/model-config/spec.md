## MODIFIED Requirements

### Requirement: The PLE table's shard layout is read from the config

`split_ngram_parts` and `make_ngram_vocab_size_divisible_by` SHALL be parsed when present and MUST be positive. Planning SHALL NOT need them; checking a checkpoint's PLE table MUST fail when `split_ngram_parts` is absent. A `ngram_vocab_size_base` whose product with the PLE head count overflows 64 bits MUST be rejected when parsing.

#### Scenario: Flash Next config
- **WHEN** the Flash Next config is parsed
- **THEN** the PLE table has 128 shards and a size multiple of 128

#### Scenario: Planning without the shard count
- **WHEN** a hybrid config without `split_ngram_parts` is planned with `--checkpoint`
- **THEN** planning succeeds, while `sparkforge weights` on the same checkpoint fails naming `split_ngram_parts`

#### Scenario: Overflowing vocab base
- **WHEN** `ngram_vocab_size_base` is i64::MAX with 4 PLE heads
- **THEN** parsing fails saying the row count overflows, and `sparkforge weights` exits with that error instead of panicking
