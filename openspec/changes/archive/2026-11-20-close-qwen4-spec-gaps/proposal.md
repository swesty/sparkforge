# Proposal

## Why

Archiving `harden-qwen4-exp-config` (swesty/sparkforge#12) showed three gaps in the resulting specs:
- the MoE requirement lost the explicit zero default for `n_shared_experts`;
- the rule that rejects differing top-level and nested `quantization_config` values contradicted `--format`. This was also a code bug: the conflict was rejected before the override applied;
- the binding refusal was worded only in terms of what the config declares.

## What Changes

- `model-config`:
  - `n_shared_experts` defaults to 0.
  - A top-level/nested `quantization_config` conflict is an error only without a format override.
  - A mixed-precision declaration at either level sets the declared-mixed status.
- `weight-loading`: binding also refuses, before GPU work, any planned tensor whose dtype is not the planned one, so a quantized checkpoint behind a BF16 config is refused.

**Expected GB10 effect:** none (host-side parsing and validation only). Measured by `cargo test`.

## Capabilities

### New Capabilities
None.

### Modified Capabilities
- `model-config`: the shared-expert default, and an override that bypasses a nested quantization conflict.
- `weight-loading`: refusal on undeclared quantized tensors.

## Impact

`src/config.rs` (`text_config` returns the conflicting top-level value; `from_json` errors only without an override). Tests: `tests/flash_next.rs`, `tests/weights.rs`.

## Non-goals

Any other parser or loader change.
