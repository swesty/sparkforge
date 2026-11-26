# Proposal

## Why

codex-review cycle 2 on swesty/sparkforge#23 approved the loader cutover rule with two follow-ups (SHA-820):
- `fp:10a6fa12ef96`: the cutover rule keeps the outgoing version disabled only until the switch finishes, so a pre-SHA-765 loader rebuilt afterwards from a kept checkout could run alongside a current load.
- `fp:1b92534be296`: two overlapping switches (an upgrade and a rollback to a pre-SHA-765 version) could each make a different version runnable.

Both need a pre-SHA-765 loader. Every loader from SHA-765 (#13, `6fe6934`) on takes the same lock, `<home>/.local/state/sparkforge/sparkforge-gpu-admission.lock` (checked at `6fe6934`, `d807f7c`, `ef0a031`, `10bfe78` and `main`), so any two of them already exclude each other. Loaders before #13 locked `$XDG_RUNTIME_DIR/sparkforge-gpu-admission.lock`, or a `sparkforge-<uid>` directory under the temp directory.

Detecting those old locks at run time was tried and removed in SHA-788 (#15, `ef0a031`): an old loader's environment is readable only through `/proc` within one PID namespace, two loads can deadlock unless every process sorts the locks, and each review cycle found another gap. Retiring the old loaders closes both findings without detection.

## What Changes

- **New requirement, "Loaders older than SHA-765 are retired":** no build whose loader predates SHA-765 may be built, run or rolled back to on a Spark, including from a kept checkout. Loads under any two later versions are serialized by the shared user lock, whatever order versions are switched in.
- **Scenarios:** "Kept checkout of a retired loader" (`fp:10a6fa12ef96`), "Overlapping switches" (`fp:1b92534be296`), and "Telling a retired loader apart" (the user-lock test; builds without the `cuda` feature have no loader).
- **codex-review cycle 1 on this change:**
  - `fp:525a1ca7c462`: the cutover rule's "Rollback" scenario still said to make the previous version runnable, even a retired one. It now applies only when that version isn't retired, and the new scenario "Rollback to a retired loader" refuses such a rollback before the current version is stopped.
  - `fp:68a8eb5b762e`: removing a retired copy doesn't stop a load already running from it. "Kept checkout of a retired loader" now stops its launchers, cancels pending invocations and waits for running ones to exit before removing it. (On spark-5, `pgrep -af sparkforge` was empty before the removal.)
  - `fp:8918c466aaf1` (two accounts with different homes on one Spark) is out of scope: the weight-loading spec already assumes one account per node and requires every sparkforge process on a node to see the same `~/.local/state/sparkforge`.
- **Fleet (done with this change):**
  - The only pre-SHA-765 copy on any Spark was spark-5 `~/experiments/sparkforge-probe` (Nov 20, source only). It is removed.
  - Its one file not in git history, `examples/probe_sm121a.rs`, is a standalone 84-line sm_121a capability probe that doesn't use the loader. It is kept as `~/experiments/probe_sm121a.rs`.
  - spark-1 to spark-4 hold no sparkforge copies.
  - spark-5's other trees all take the user lock. Of its four binaries, the two release builds take the user lock, and the two debug builds were built without `cuda`, so they have no `load-weights` or `verify-weights` command.

**Expected GB10 effect:** none. Only the spec changes.

## Non-goals

- Run-time detection of pre-SHA-765 locks (removed in SHA-788 for the reasons above).
- Changing the cutover rule. It still governs switches; the shared lock is what keeps overlapping switches between current versions safe.

## Capabilities

### Modified Capabilities
- `weight-loading`: adds the requirement "Loaders older than SHA-765 are retired", and limits the cutover rule's "Rollback" scenario to versions that aren't retired.

## Impact

- `openspec/specs/weight-loading/spec.md` only.
