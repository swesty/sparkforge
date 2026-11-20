# Tasks

## 1. Spec

- [x] 1.1 Add unsupported GPTQ/AWQ bit widths to "Unknown or unsupported quantization is rejected" in the model-config delta, and verify with `openspec validate --all --strict`

## 2. Tests

- [x] 2.1 Confirm `detect_format` in `src/config.rs` already rejects GPTQ/AWQ at bits other than 4 and with no `bits` (catch-all arm), so no parser change is needed
- [x] 2.2 Add `cargo test` coverage in `tests/plan.rs`: 8-bit GPTQ and AWQ, 3-bit GPTQ and a missing `bits` all fail without an override, the 8-bit error names the bit width and suggests `--format`, and a BF16 override still parses the 8-bit AWQ config. Verify with `cargo test --locked`

No kernel, plan or codegen output changes, so there is no `sparkforge verify --bench` task.

## Workflow follow-up

- After the PR merges, run `openspec archive reject-unsupported-quant-bit-widths` to merge the delta into `openspec/specs/model-config/spec.md`, then validate again.
