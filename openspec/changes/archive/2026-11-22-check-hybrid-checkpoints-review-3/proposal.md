# Proposal

## Why

codex-review cycle 3 on #18 (the last automatic cycle) found seven more gaps:

1. **NVFP4 override:** the names-only inventory still required an NVFP4 table's global scale, so `--format nvfp4` refused a BF16 table.
2. **Bare table:** a sole model-level table beside the PLE projections counted as two PLE locations.
3. **Group roots and hollow shards:** a hollow shard (only an observer under it) and a weight at the experts' own path passed resolution.
4. **MTP shared-expert gate:** the budget ignored its format.
5. **Head count:** the PLE head count was computed unchecked.
6. **Shard count:** an extreme `split_ngram_parts` made resolution allocate one entry per declared shard.
7. **Failed re-resolution:** it cleared the previous checkpoint's checked rows and formats.

## What Changes

- **Modules-only inventory:** it no longer lists format-specific table metadata.
- **Independent locations:** `inventory::ple_paths` finds the PLE module and its n-gram embedding separately (`PlePaths { module, embedding }`). The embedding may be the model-level `ple_embedding`.
- **Shard check first:** `check_table_shards` runs before the inventory is built. It requires the configured count (or at least one shard), numbered with no gap, each a module at its exact path. It names extra shards, and missing ones when few.
- **Group roots:** a group's own path is allowed only for the table's shared scalars, not for the experts.
- **MTP gate:** the MTP budget prices the shared-expert gate at its own format.
- **Overflow at parse:** parsing checks `(ngram_size - 1) * heads_per_ngram` before using it.
- **Restore on failure:** `resolve_modules` restores the previous formats and checked rows when resolution fails.

**Expected GB10 effect:** none on kernels. The reference budget gains the 5,120-byte BF16 MTP shared-expert gate (no change at the displayed precision). `plan --checkpoint` stays byte-identical to config-only planning, with and without `--format`.

## Capabilities

### Modified Capabilities
- `kernel-planning`, `weight-loading`, `model-config`.

## Impact

- `src/inventory.rs`, `src/config.rs`, `src/plan.rs`, `src/weights.rs`; `tests/weights.rs`, `tests/flash_next.rs`.
