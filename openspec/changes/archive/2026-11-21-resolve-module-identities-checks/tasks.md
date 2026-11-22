# Tasks

- [x] 1.1 `resolve_modules`: MTP indices, single-format groups, tied head; tests for each
- [x] 1.2 `Formats` tracks resolved entries; a vision-only entry is reported unused
- [x] 1.3 A tied head takes the embedding's format; a conflicting entry is rejected at parse time; the embedding defaults to BF16
- [x] 1.4 `resolve_with_checkpoint` (config equality) and the shard-name rule in `tensor_names`; tests
- [x] 1.5 Blob mismatch error without the computed digest
- [x] 1.6 The 7 new tests fail against the cycle-1 head; `verify.sh`; spark-5 tests; the real plan is byte-identical
