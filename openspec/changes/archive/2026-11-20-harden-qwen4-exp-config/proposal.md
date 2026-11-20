# Proposal

## Why

codex-review approved swesty/sparkforge#9 (the qwen4_exp config and plan) in its last round with seven follow-ups. One is a regression from that change: a malformed `quantized_layers` map now fails even with `--format`, although the parser used to ignore the map entirely. The others let edited or unusual configs produce a wrong budget, bypass a loader guard, or report the wrong error. The mixed-format loader builds on this code, so they are fixed first.

## What Changes

- `model-config`:
  - `quantized_layers` is interpreted only during format detection, so `--format` overrides a malformed map again.
  - The top-level `model_type` fills a missing nested one.
  - `num_experts_per_tok` must be between 1 and the expert count.
- `weight-loading`: a checkpoint that declares MIXED_PRECISION is refused even when `--format` overrode the planning format.
- `kernel-planning`:
  - `lm_head` is matched only at its model-level paths.
  - Model-level lookups match on a module boundary.
  - Each MTP projection is priced at its own format.
- `gpu-verification`: `verify` reports a planned-only model before creating any CUDA context.

**Expected GB10 effect.** None: no kernel changes. The Flash Next plan is unchanged at 7.338 GB/token context-free, and both dense example plans remain byte-identical (golden files). Measured by `cargo test`.

## Capabilities

### New Capabilities
None.

### Modified Capabilities
- `model-config`: lazy mixed-precision map, top-level `model_type` fallback, MoE top-k range.
- `weight-loading`: the declared mixed-precision status survives a format override.
- `kernel-planning`: exact `lm_head` lookup, boundary-matched model-level lookups, separate MTP projections.
- `gpu-verification`: refusal before CUDA initialization.

## Impact

Code: `src/config.rs`, `src/plan.rs`, `src/codegen.rs` (`require_generated`), `src/verify.rs`, `src/main.rs` and `src/weights.rs`. Tests: `tests/flash_next.rs` and `tests/weights.rs`.

## Non-goals

- Any kernel, loader or prefill work.
- Supporting more than one MTP layer or quantized HyperConnection weights. Both remain rejected.
