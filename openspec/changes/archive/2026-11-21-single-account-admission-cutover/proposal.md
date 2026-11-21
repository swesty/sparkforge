# Proposal

## Why

codex-review cycle 1 on swesty/sparkforge#16 (`single-account-admission`) approved it with three P2 follow-ups. Each applies to one account:
- The re-check that the lock's path still names the locked file was removed along with the node lock, though it also guarded this account's own lock.
- The migration note dropped the rule that an upgrade from the loader before #13 drains running loads.
- A lock acquired after the wait limit, once it had been found held, still led to an admission.

## What Changes

- **Path re-check:** after `flock`, the lock's path must still name the locked file. If it doesn't, the load locks the new file, within the same deadline.
- **Deadline:** once the lock has been found held, an acquisition at or after the deadline is a timeout.
- **Spec:** scenarios for a holder released after the wait, a lock replaced while waiting, and the drain rule for upgrades from the loader before #13.
- `admit_observed` (hidden from docs) reports the first time each opened lock file is found held, so the replaced-lock test is deterministic.

**Expected GB10 effect:** none.

## Capabilities

### Modified Capabilities
- `weight-loading`: three scenarios on the single-account memory requirement.

## Impact

- `src/weights.rs` (`admit`, `admit_observed`), `tests/weights.rs`.
