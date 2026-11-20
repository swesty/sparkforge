# Tasks

## 1. Fixes

- [x] 1.1 Interpret `quantized_layers` lazily. Verify: `cargo test` checks that a `null` entry plans as BF16 with `--format` and fails naming the entry without it.
- [x] 1.2 Keep `declared_mixed` through an override, and refuse in `bind`. Verify: `cargo test` (`tests/weights.rs`) checks that the override lets a dense mixed config parse, while binding its checkpoint fails.
- [x] 1.3 Match `lm_head` at its model-level paths, and match model-level modules on a boundary. Verify: `cargo test` checks that an `aux_lm_head` entry is reported unused and changes neither budget, while a real `lm_head` entry is honored.
- [x] 1.4 Price the MTP `k_proj` and `v_proj` separately. Verify: `cargo test` checks that an MXFP8 MTP `v_proj` changes resident bytes by exactly 1,269,760 B and is consumed.
- [x] 1.5 Fall back to the top-level `model_type`. Verify: `cargo test` checks that the variant still plans as qwen4_exp with only planned entries.
- [x] 1.6 Validate MoE top-k. Verify: `cargo test` checks that 0 and 513 fail.
- [x] 1.7 Refuse a planned-only model before CUDA. Verify: `cargo test` covers `require_generated`, and `cargo run --features cuda -- verify configs/qwen3.8-flash-next-nvfp4.json` on a Mac without CUDA prints the planned-only error.

## 1b. codex-review round 1

- [x] 1b.1 Decode the whole `quantization_config` only during format detection. Verify: `cargo test` checks that a string `bits` plans as BF16 with `--format`, fails without it, and that a malformed mixed config still sets `declared_mixed`.
- [x] 1b.2 Price the MTP shared expert's gate, up and down separately. Verify: `cargo test` checks that quantizing any one changes resident bytes by exactly one matrix's delta, and the entry is consumed.
- [x] 1b.3 Match PLE and MTP modules at exact paths. Verify: `cargo test` checks that auxiliary `aux.ple.key_proj` and `aux.self_attn.v_proj` entries are reported unused and change no budget, while the real PLE key projection is honored.

## 1c. codex-review round 2

- [x] 1c.1 Accept the model-level PLE table path. Verify: `cargo test` checks that an MXFP8 table entry at `model.language_model.ple_embedding.ngram_embedding` is consumed and changes resident and decode bytes by the exact amounts.
- [x] 1c.2 Accept layer-qualified PLE entries only at layer p or p−1. Verify: `cargo test` checks that `layers.999` is unused with budgets unchanged, while layers 1 and 2 are honored.
- [x] 1c.3 Skip MTP shared-expert lookups when the width is zero. Verify: `cargo test` checks that a stale `up_proj` entry is reported unused.

## 2. Unchanged behavior

- [x] 2.1 The dense example plans stay byte-identical. Verify: the golden-file test passes, and `plan --json` diffs empty.
- [x] 2.2 No kernel changed, so `sparkforge verify --bench` on a GB10 for both example configs is unaffected. Verify: run it on spark-5 and record the numbers in the PR.

## Workflow follow-up

- Archive with `openspec archive harden-qwen4-exp-config` after merge.
