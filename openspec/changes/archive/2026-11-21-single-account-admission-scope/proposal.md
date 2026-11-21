# Proposal

## Why

codex-review cycle 2 on swesty/sparkforge#16 approved it with three P2 follow-ups:
- Deleting the lock file while a load is admitted lets a later load lock a new file.
- The drain rule covered only the loader before #13, not a version that held only the node lock.
- The deadline test depended on thread scheduling.

## What Changes

- **Scope:** deleting the lock file or `~/.local/state/sparkforge` while a load is admitted is unsupported, stated in `admit`'s docs and the spec. No lock survives its own deletion, and nothing on the Sparks deletes that directory. The design note in `single-account-admission-cutover` that cited cleaning the state directory as an example is superseded.
- **Drain rule:** upgrading from any earlier version drains running loads first. No node ever had the node lock set up, so the old node-only case can't occur on the Sparks; the general rule covers it anyway.
- **Deterministic test:** `admit_with(…, &AdmitHooks { on_wait, sleep })` (hidden from docs) replaces `admit_observed`. The deadline test supplies its own pause, which releases the lock after the deadline and before the next attempt. With the late-acquisition check disabled it fails 5 of 5 runs; with it, it passes 5 of 5.

**Expected GB10 effect:** none.

## Capabilities

### Modified Capabilities
- `weight-loading`: the upgrade scenario is generalized, and a scenario for deleting the state directory is added.

## Impact

- `src/weights.rs` (`admit` docs, `AdmitHooks`, `admit_with`), `tests/weights.rs`.
