# Proposal

## Why

Every Spark runs sparkforge under the single account `spark`; no other accounts exist on any of them (user, 2026-11-21). The node-wide admission lock (`/run/sparkforge/admission.lock`, a `sparkforge` group, ACL checks, re-checks of the locked file, and a repair procedure) existed to coordinate loads across accounts. Three PRs of codex-review follow-ups (#13, #14, #15) were about that boundary between accounts, which doesn't exist here. The per-user lock already makes every load on a node take turns.

## What Changes

- **Removed:** the node lock and everything that existed only for it:
  - `NODE_LOCK_DIR`, `LockScope`, `LockKind`, `LockFile`, `NodeLock`, `AdmissionLocks`, `admission_locks_in`;
  - `check_node_lock` and the ACL parsing (`acl_named_grant`, `path_acl`, `fd_acl`);
  - the closed-lock message, the re-lock after a replacement, the per-inode deduplication, and `admit_observed`;
  - the "not provisioned" note.
- **Kept:**
  - the per-user lock under the password-database home (now `admission_lock` / `admission_lock_in`);
  - `admit(lock, …)` with the 30-minute wait limit;
  - the memory guard before any CUDA call and again before allocating;
  - the admission held through verification.
- **Fixed (the one cycle-3 follow-up on #15 unrelated to the node lock):** blob verification creates threads with `std::thread::Builder::spawn_scoped`, and verifies on the calling thread when a worker can't be created, instead of panicking.

**Expected GB10 effect:** none.

## Capabilities

### Modified Capabilities
- `weight-loading`: the memory requirement is replaced by a single-account one.

## Impact

- `src/weights.rs`, `tests/weights.rs`.
- homelab-build#35, which added the node-lock setup to the runbook, is closed without merging.
