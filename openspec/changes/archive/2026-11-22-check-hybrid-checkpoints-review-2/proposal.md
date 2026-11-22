# Proposal

## Why

codex-review cycle 2 on #18 found six more gaps:

1. Name-only resolution built the storage tensor inventory, so `plan --checkpoint --format w4a16` failed on a format the loader doesn't read.
2. Without `split_ngram_parts`, planning accepted a PLE table with no shards.
3. A module inside an expert (an observer) was covered by the experts group, and its entry counted as used, without being priced.
4. The MTP budget charged the shared input projections once per draft layer.
5. An extreme `ngram_vocab_size_base` overflowed while `weights` built its first plan, before the check ran.
6. Rows checked in one checkpoint survived re-resolving the spec onto another.

## What Changes

- **Modules-only inventory:** `Layout { table_rows: None }` lists modules only. Each matrix appears as its `.weight` alone, whatever its format, so resolution checks module existence and leaves storage formats to the tensor check.
- **Group modules:** every module under a group must be planned. Without a configured shard count, the table must hold shards `shard_0..` with no gap.
- **MTP budget:** the FC matrices and the final mixer are counted once.
- **Overflow:** parsing rejects a `ngram_vocab_size_base` whose product with the PLE head count overflows 64 bits.
- **Check before planning:** `sparkforge weights` checks hybrid tensors before building any plan.
- **Re-resolution:** `resolve_modules` clears `Ple::checked_rows`.

**Expected GB10 effect:** none. On the reference checkpoint, `weights` passes and `plan --checkpoint` is byte-identical to config-only planning.

## Capabilities

### Modified Capabilities
- `kernel-planning`, `weight-loading`, `model-config`.

## Impact

- `src/inventory.rs`, `src/config.rs`, `src/plan.rs`, `src/main.rs`; `tests/weights.rs`, `tests/flash_next.rs`.
