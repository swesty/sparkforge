# Proposal

## Why

`sparkforge weights` refused every hybrid checkpoint, so nothing checked the Qwen3.8 Flash Next checkpoint's 301,593 tensors against the plan. The GPU load (milestone M1, part 2) needs that check first: it must know, for every module, which tensors exist and what they hold. codex-review cycle 3 on #17 also left six narrow follow-ups about how a checkpoint's modules are priced; requiring every planned module to exist settles most of them.

## What Changes

- **New `src/inventory.rs`:** one list of every tensor a hybrid checkpoint must hold, derived from the spec and from the format each module is priced at. Per format: BF16 is one tensor; MXFP8 is E4M3 values plus a `U8` scale per 32 inputs; NVFP4 is packed `U8` pairs, an E4M3 scale per 16 inputs, an F32 `weight_scale_2` and, when activations are quantized, an F32 `input_scale`. Norm and parameter dtypes come from the reference checkpoint (F32 mixer norms in the main model, BF16 in the MTP layer).
- **`weights::check_hybrid`:** checks every tensor against that list. Each wanted tensor must exist once, with the expected dtype and shape. Every other tensor must belong to the vision tower, which is skipped. The PLE table's shard rows come from the checkpoint's per-head table sizes, padded to `make_ngram_vocab_size_divisible_by` and split into `split_ngram_parts` shards. `sparkforge weights` reports the checked tensors, bytes by group, and the plan's resident total beside them.
- **GPU binding** still refuses hybrid checkpoints, with a message saying they are checked but not loaded yet.
- **The six cycle-3 follow-ups on #17:**
  1. `resolve_modules` refuses a checkpoint with the PLE at more than one accepted path.
  2. Group lookups (routed experts, the PLE table) price their members and count every entry under the group as used, so a table scalar at the table's own path no longer hides agreeing shard entries.
  3. A dense head and embedding take their resolved formats.
  4. Config-only MTP lookups accept only canonical indices.
  5. `weights::spec_from_checkpoint` builds the spec from the checkpoint's own config (it replaces `resolve_with_checkpoint`).
  6. Projection lookups are exact, and `resolve_modules` requires every module the plan prices to exist at its exact path.

**Expected GB10 effect:** none on kernels. `sparkforge weights` on the reference checkpoint goes from a refusal to a pass.

## Capabilities

### Modified Capabilities
- `weight-loading`: hybrid checkpoints are checked tensor by tensor; GPU loading stays dense BF16 only.
- `kernel-planning`: projection and group lookups, one PLE location, every planned module present, canonical MTP indices without a checkpoint, dense head and embedding formats.
- `model-config`: the PLE table's shard layout is read from the config.

## Impact

- `src/inventory.rs` (new), `src/weights.rs`, `src/config.rs`, `src/modules.rs`, `src/plan.rs`, `src/main.rs`, `src/lib.rs`.
- `tests/common/mod.rs` (new, a hand-written copy of the reference layout), `tests/flash_next.rs`, `tests/weights.rs`, `tests/plan.rs`.
