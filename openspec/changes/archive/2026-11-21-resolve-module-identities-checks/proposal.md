# Proposal

## Why

codex-review cycle 1 on swesty/sparkforge#17 (`resolve-module-identities`) approved it with eight P2 follow-ups. Seven are about what the resolved planner accepts:
- a tied head with its own format entry;
- a PLE table split across formats;
- an extra live MTP index;
- an entry whose modules no op prices;
- the embedding taking the general weight format;
- a config from another checkpoint;
- an index with invalid shard references.

The eighth, a "digest oracle" through a blob link to a private file, needs another party to control the cache. That's out of scope with one account per Spark.

## What Changes

- **`resolve_modules` checks:** the MTP indices must be the configured ones; each group the planner prices as one format must have one format; a tied head must not disagree with the embedding.
- **Unused entries:** the planner records which resolved entries its lookups use, so an entry covering only unplanned modules is still reported.
- **Tied head:** it always takes the embedding's format, and a conflicting `lm_head` entry is rejected at parse time.
- **Embedding:** it defaults to BF16, not the general weight format.
- **`--checkpoint`:** `weights::resolve_with_checkpoint` requires the given config to equal the checkpoint's own `config.json`, read through the same opened directory. `tensor_names` applies the loader's shard-name rule.
- **Blob mismatch error:** it no longer prints the computed digest.

**Expected GB10 effect:** none. The real checkpoint's plan is unchanged.

## Capabilities

### Modified Capabilities
- `kernel-planning`: resolution checks, embedding default, tied-head rule.

## Impact

- `src/config.rs`, `src/plan.rs`, `src/modules.rs`, `src/weights.rs`, `src/main.rs`; tests.
