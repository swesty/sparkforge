# Proposal

## Why

The scenario "Upgrade from a loader before SHA-765" listed upgrade steps in order, and codex-review on swesty/sparkforge#21 found a new gap in it in each of three cycles. Cycle 1 (`fp:86bffa01ccfd`) and cycle 2 (`fp:672103083fcc`) were fixed by adding steps. Cycle 3 found two more (SHA-815):
- `fp:027cee08a8f2`: the drain came before the old binary was disabled, so a run started by hand in between was still going when dispatch resumed. Removing execute permission doesn't stop a running process.
- `fp:b4ce68d7540a`: the rollback clause said only "drain". It didn't stop launchers, cancel pending runs or disable the newer binary, so a newer-version load could start alongside the restored older one.

Adding a step per finding doesn't converge. The hazard is always the same: two versions that lock different files can both be admitted before either allocates, and the node runs out of host memory.

## What Changes

- **One rule for both directions:** a new requirement, "Loader version switches follow one cutover rule", states that a version switch disables every way to start the outgoing version first, and makes the incoming version runnable only after every invocation of the outgoing one has exited.
- **The upgrade scenario points at the rule:** "Upgrade from a loader before SHA-765" keeps its name and trigger (OpenSpec 2.1.1 refuses to drop a scenario from a MODIFIED requirement), and its outcome becomes "the switch follows the cutover rule" instead of the step list.
- **Disable before waiting:** the new scenario "Disable before waiting" spells the rule out. Launchers stop, pending runs are cancelled and every runnable copy (installed binaries and build outputs) is disabled; only then does the switch wait for running invocations to exit. The incoming version stays unrunnable and nothing runs from a source checkout until the wait ends.
- **The two cycle-3 cases** each get a scenario: "Start during the cutover window" (a hand start in the window is one of the invocations the wait covers) and "Rollback" (the same rule with the roles swapped).

**Expected GB10 effect:** none. Only the spec changes; no code, kernel or admission decision does.

## Non-goals

- Detecting another version at run time, for example by also locking the pre-SHA-765 paths. The cutover rule covers switches.
- Enforcing the rule in tooling. Every Spark runs sparkforge under the single account `spark`, no scheduled sparkforge jobs exist, and no Spark keeps an old loader, so the rule binds operators only if one of those changes.

## Capabilities

### Modified Capabilities
- `weight-loading`: adds the requirement "Loader version switches follow one cutover rule" with three scenarios, and the upgrade scenario of "Loads never exhaust host memory on a single-account node" points at it.

## Impact

- `openspec/specs/weight-loading/spec.md` only.
