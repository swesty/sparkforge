# Proposal

## Why

codex-review cycle 3 of 3 on swesty/sparkforge#16 approved it with three follow-ups (SHA-790):
- The deadline test `a_holder_that_outlasts_the_wait_never_leads_to_an_admission` started its clock before `admit_with` computed its deadline. A test thread descheduled in between released the lock before admission's deadline, so admission correctly succeeded and the test failed (`fp:2090b792f620`).
- A container running as `spark` with a different home directory mounted at the same path would take a different lock file (`fp:3f3f53aed3e5`).
- A launcher still pointing at an old binary could start a load during an upgrade (`fp:664757fbf6bc`).

## What Changes

- **Deterministic deadline test:** the test's clock starts at the first pause call, which always follows admission's deadline computation, and waits past `wait` from there. With an injected 30 ms delay before `admit_with`, the old test fails and the new one passes. The new test passes 5 of 5 runs with the late-acquisition check and fails 5 of 5 with the check disabled.
- **Scope:** every sparkforge process on a node must see the same `~/.local/state/sparkforge`. sparkforge runs on the host as `spark`, and only vLLM runs in containers. Stated in `admission_lock_in`'s docs and a new spec scenario.
- **Drain rule:** the upgrade scenario also switches any launcher to the new binary before loads resume. No scheduled sparkforge jobs exist today.

**Expected GB10 effect:** none. No kernel, load path or admission decision changes; one test and two scope statements do.

## Non-goals

- Coordinating loads across different home mounts or containers. The Sparks run sparkforge only on the host, under one account.
- Detecting an old binary at run time. The drain procedure covers upgrades.

## Capabilities

### Modified Capabilities
- `weight-loading`: the upgrade scenario adds the launcher switch, and a scenario for a different home mount is added.

## Impact

- `tests/weights.rs` (the deadline test), `src/weights.rs` (`admission_lock_in` docs only).
