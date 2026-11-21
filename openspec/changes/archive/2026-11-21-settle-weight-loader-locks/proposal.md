# Proposal

## Why

codex-review cycles 2 and 3 on swesty/sparkforge#14 (`tighten-weight-loading`) approved it with eight P2 follow-ups (SHA-788):
- **Old-loader locks:** three were about finding locks held by the loader before #13 during an upgrade. Two loads could take those locks in opposite orders and stall each other. An old load using another temp directory, or running in another PID namespace, can't be found at all.
- **Node lock:** the lock's owner, group, mode and ACL were checked by path, but a different file could then be opened and locked. Separately, the documented repair removed the lock while loads might be starting.
- **Hugging Face snapshots:** `../../blobs` was resolved from the snapshot's current location, so a revision moved between the two opens could pair with another repository's blobs. A broken `blobs` path also blocked snapshots that don't use it.
- **Portability:** the inode set added in #14 didn't compile on non-Unix targets.

## What Changes

- **Old-loader locks are no longer consulted.** No environment reveals every place such a load may hold one, so the partial detection (its runtime directories, the `/proc` scan, the inode deduplication) is removed. Upgrading from that loader drains running loads first, which the provisioning runbook already requires. Admission takes two locks, user then node, in the same order in every process.
- **The node lock is checked again on its descriptor** (`fstat` and `fgetxattr`) before `flock`.
- **Repair fails closed:** with `/run/sparkforge` present and the lock missing, loads refuse (tested). The error and the runbook say to remove only the file.
- **Snapshots resolve from a pinned repository:** `<repo>` is opened once, and `snapshots/<rev>` and `blobs` are opened downward from it. A `blobs` open error is kept and reported only when a blob link needs the store.
- **Non-Unix builds compile:** the inode set is gone, Unix-only link code is gated, and `pread` has a fallback for targets that are neither Unix nor Windows. Checked with `cargo check --lib --target wasm32-unknown-unknown`.

**Expected GB10 effect:** none on kernels or bandwidth.

## Capabilities

### New Capabilities
None.

### Modified Capabilities
- `weight-loading`: snapshot resolution, node-lock descriptor checks, fail-closed repair, no old-loader locks.

## Impact

- `src/weights.rs`: `resolve_checkpoint_dir`, `CheckpointDir::blobs`, `admit`, `LockKind`, `NodeLock`, `AdmissionLocks`, `admission_locks_in` (no `legacy_dirs`), `check_node_lock`, `path_acl` / `fd_acl`; `legacy_lock_dirs` and `runtime_dirs_of_processes` removed.
- Upgrades from the loader before #13 must drain running loads first.
