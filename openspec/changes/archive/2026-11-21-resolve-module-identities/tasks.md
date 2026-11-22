# Tasks

## 1. Module identities
- [x] 1.1 `ModuleTable` from tensor names, and `resolve` (coverage, agreement, unused)
- [x] 1.2 `ModelSpec::resolve_modules`; `Formats` lookups through it (layer, MTP, PLE, head, embed, unused)
- [x] 1.3 Config-only: MTP indices below the configured count; a tied head takes the embedding's format; the embedding priced at its format

## 2. CLI and loader
- [x] 2.1 `weights::tensor_names`; `plan`/`emit --checkpoint`; `weights` resolves from the opened checkpoint
- [x] 2.2 Follow a Hugging Face blob's own link (shared store), with the bytes verified against the name

## 3. Tests
- [x] 3.1 The three findings: an overlapping-entry conflict names both entries while agreeing entries pass; a bare PLE alias is unused and the price is unchanged; index 999 is unused with the budget unchanged (checkpoint and config); a tied MXFP8 embedding prices `lm_head` at MXFP8, resident once
- [x] 3.2 A checkpoint-resolved plan equals the config plan for the real map; existing tests check the one unused MTP entry
- [x] 3.3 A blob chain into a shared store opens and verifies, and refuses other bytes
- [x] 3.4 The new tests fail against `main` (6 of 6)

## 4. Verification
- [x] 4.1 `verify.sh` on macOS; `cargo test` on spark-5; the dense golden plans unchanged
- [x] 4.2 Real checkpoint on spark-5: `plan --checkpoint` byte-identical to the config plan; same budget and ops as `main`
