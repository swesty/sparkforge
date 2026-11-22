# Proposal

## Why

codex-review cycle 1 on #18 found six gaps in the hybrid tensor check:

1. Checkpoint-backed planning required `split_ngram_parts`, which the spec makes optional for planning.
2. `weights` reported a resident budget priced from `ngram_vocab_size_base` instead of the rows it had just checked.
3. One MTP layer's format set the tensor schema of every draft layer.
4. An extra table shard or expert was accepted (and its entry counted as used) without being budgeted.
5. A tied NVFP4 embedding was inventoried only as a gather, without the head GEMV's activation scale.
6. Padding a checkpoint's PLE row count could overflow.

## What Changes

- **Optional shard count when planning:** the inventory lists table shards only when `split_ngram_parts` is known. Name-only resolution (planning) works without it, while the tensor check still requires it.
- **Budget at the checked size:** `Ple::checked_rows` records the checked table size. `weights::apply_checked_table` sets it from a `HybridCheck`, and `sparkforge weights` reports the resident budget with the table at that size.
- **Per-layer MTP formats:** `Formats::mtp` and `mtp_group` take the layer index. The inventory and the MTP budget use each layer's own formats.
- **Exact group members:** `resolve_modules` requires each group's members to be exactly the planned ones (experts `0..num_experts` per layer, table shards `0..split_ngram_parts`).
- **Tied embedding as a GEMV operand:** a tied embedding is inventoried as a GEMV operand, so NVFP4 requires `input_scale`.
- **Checked arithmetic:** the PLE row total and its padding use checked arithmetic.

**Expected GB10 effect:** none. On the reference checkpoint, `weights` still passes, and `plan --checkpoint` is byte-identical to config-only planning.

## Capabilities

### Modified Capabilities
- `weight-loading`, `kernel-planning`, `model-config`.

## Impact

- `src/inventory.rs`, `src/config.rs`, `src/plan.rs`, `src/weights.rs`, `src/main.rs`; `tests/weights.rs`, `tests/flash_next.rs`.
