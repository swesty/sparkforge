# Tasks

## 1. Parse the config

- [x] 1.1 Read `text_config` with the top-level `quantization_config` and `architectures`. Verify: `cargo test` parses `configs/qwen3.8-flash-next-nvfp4.json` (checkpoint config, verbatim).
- [x] 1.2 Parse the qwen4_exp hybrid layout (schedule, Gated DeltaNet, sparse attention, HyperConnection, PLE, mRoPE, MTP), with errors for other hybrid models, inconsistent schedules, missing fields and mismatched `mrope_section`. Verify: `tests/flash_next.rs` covers each case.
- [x] 1.3 Map ModelOpt `MIXED_PRECISION` to a per-module format map (NVFP4, weight-only NVFP4, MXFP8) and reject unknown schemes and group sizes. Verify: `cargo test`, with 435 modules mapped for the checkpoint.
- [x] 1.4 Read `rope_parameters`, reject partial rotary on dense models, and honor `shared_expert_intermediate_size`. Verify: `cargo test`.

## 2. Plan it

- [x] 2.1 Plan the hybrid as one op class per distinct kernel, with shapes, formats and instances per token. Verify: `cargo test` checks 34 unique op classes and their instance counts.
- [x] 2.2 Build the hybrid decode budget and the resident-weight total. Verify: `cargo test` matches the header sums within 1%: 7.338 GB/token context-free, the per-group figures, and 105.4 GB resident.
- [x] 2.3 Report format-map entries the plan never uses. Verify: `cargo test` flags an injected unknown path, and none for the checkpoint itself.
- [x] 2.4 Print the op table and byte groups in `sparkforge plan`. Verify: `cargo run -- plan configs/qwen3.8-flash-next-nvfp4.json`.

## 3. Keep everything else safe

- [x] 3.1 Codegen lists hybrid op classes as planned and emits no source; `verify` and `bind` refuse the model. Verify: `cargo test` asserts no `__global__` in the generated files and every status `planned`.
- [x] 3.2 Dense plans and kernels are unchanged. Verify: Llama 3.1 8B and Qwen3-30B-A3B plan JSON and `emit` cache keys are identical to `main` (`a8d62dd6088f7ef1`, `36838cb28f25286c`).
- [x] 3.3 No kernel changed, so `sparkforge verify --bench` on a GB10 for both example configs is unaffected. Verify: run it on spark-5 and record the numbers in the PR.

## 4. codex-review round 1

- [x] 4.1 Reject dense MIXED_PRECISION configs. Verify: `cargo test` fails a dense map with "only for qwen4_exp".
- [x] 4.2 Key hybrid emit directories on the plan. Verify: `cargo test` gives different cache keys with and without the format map.
- [x] 4.3 Split format groups into separate op classes. Verify: `cargo test` checks that a BF16 layer 0 gives `gdn_in_proj_qkv` ×35 MXFP8 plus `gdn_in_proj_qkv_bf16` ×1, and the budget grows by exactly one matrix's difference.
- [x] 4.4 Require exactly one PLE placement, a positive attention interval, and agreeing MTP counts (read from either field). Verify: `cargo test`.
- [x] 4.5 Name every unused format-map entry. Verify: `cargo test` with four injected entries.
- [x] 4.6 Keep dense plan JSON unchanged. Verify: `cargo test` compares against golden files from `main` at `8aa1a73`, and the CLI output is byte-identical.

## Workflow follow-up

- Archive with `openspec archive add-qwen4-exp-config` after merge.
