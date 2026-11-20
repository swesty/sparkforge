# Proposal

## Why

The model-config spec maps GPTQ/AWQ to W4A16 only at 4 bits and sends everything else to the requirement "Unknown or unsupported quantization is rejected". That requirement lists bad group sizes, non-NVFP4 modelopt and unknown methods, but not a recognized GPTQ/AWQ method at another bit width. A config such as `{"quant_method": "gptq", "bits": 8, "group_size": 128}` therefore has no acceptance criterion: a future parser could map it to W4A16 and misread 8-bit weights as 4-bit without breaking any requirement (codex-review on swesty/sparkforge#3).

`detect_format` in `src/config.rs` already rejects it today. Its only GPTQ/AWQ arm matches `bits == Some(4)`, so 8 bits, or a missing `bits`, falls through to the catch-all error, which suggests `--format`. This change makes that behavior a requirement and puts it under test.

## What Changes

- The rejection requirement in model-config also covers GPTQ/AWQ with a `bits` value other than 4, or with no `bits`, when the caller gives no format override.
- New scenarios: 8-bit GPTQ is rejected with a message that suggests `--format`, and an explicit override still wins over an 8-bit config.
- `cargo test` covers 8-bit GPTQ and AWQ rejection, a missing `bits`, and the override.
- No parser change. The behavior was already correct.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `model-config`: "Unknown or unsupported quantization is rejected" adds unsupported GPTQ/AWQ bit widths.

## Impact

- Spec: `openspec/specs/model-config/spec.md`, merged by `openspec archive` after merge.
- Tests: `tests/plan.rs`.
- Code: none. `src/config.rs` `detect_format` already behaves this way.
- GB10 performance: none. No kernel, plan or codegen output changes, so decode and prefill tok/s are unchanged and there is nothing to measure with `sparkforge verify --bench`.

## Non-goals

- Supporting 8-bit (W8A16) or 3-bit GPTQ/AWQ weights. That would be a new weight format and its own change.
- Changing how `--format` works.
