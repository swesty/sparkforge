# Proposal

## Why

codex-review cycle 2 on swesty/sparkforge#15 approved it with five P2 follow-ups:
- **Pair check:** the two identity checks on a Hugging Face snapshot's revision and blob store could each pass for entries that never coexisted (swap, then swap back, between the checks).
- **Symlink in the re-check:** the revision re-check followed a symlink swapped in for `snapshots`.
- **Repair procedure:** the repair took the old file's lock, which an outside account can keep through a descriptor opened under the ACL, so it could stall. A load from an older build already waiting on the old file wouldn't re-check after the replacement.
- **Test timing:** the replaced-lock test could pass without the waiting load ever opening the old file.

## What Changes

- **Blobs are verified against their names.** A 64-digit name is the SHA256 of the content, a 40-digit name the Git object SHA1. Shards are hashed in parallel, metadata files over exactly the bytes returned, and the verified stamp must still hold when headers are read. This replaces the pair check (and its symlink-following re-check). Whatever happens to the directories, a revision's links can only yield the bytes their names fix. `sha2` now uses its `asm` feature: 2.14 GB/s on GB10, against 0.57 GB/s without.
- **Repair is close, drain, replace:** `chmod 0600` fences new loads, which now fail saying a repair is under way. Draining waits for every sparkforge process, including waiters of any build. Then only the file is replaced. No lock on the old file is needed.
- **The replaced-lock test is synchronized** by a hidden `admit_observed` hook that reports when a load first finds a lock held.

**Expected GB10 effect:** loads from a Hugging Face cache hash every blob, about +1 s for Llama 3.1 8B (4 shards in parallel). Plain checkpoints are unchanged.

## Capabilities

### Modified Capabilities
- `weight-loading`: blob verification, closed-lock refusal.

## Impact

- `src/weights.rs`: `verify_blob`, `blob_digest`, parallel verification in `Checkpoint::open_dir`, `read_metadata`; the pair check, `ResolveStep` and `resolve_checkpoint_dir_between` are removed; `open_lock` closed-lock message; `admit_observed`.
- `Cargo.toml`: `sha1`; `sha2` with `asm`.
