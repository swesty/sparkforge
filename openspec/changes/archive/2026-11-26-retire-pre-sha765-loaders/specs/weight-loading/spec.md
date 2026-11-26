## ADDED Requirements

### Requirement: Loaders older than SHA-765 are retired

No sparkforge build whose weight loader predates SHA-765 SHALL be built, run or rolled back to on a
Spark, including from a kept source checkout, because its admission lock doesn't exclude a current
load. Every loader from SHA-765 on locks `~/.local/state/sparkforge/sparkforge-gpu-admission.lock`,
so loads under any two of them are serialized by admission in whatever order versions are switched.

#### Scenario: Kept checkout of a retired loader
- **WHEN** a source checkout, build output or binary of a loader older than SHA-765 is still on a Spark, including one kept after a switch has finished
- **THEN** it is not built or run, and it is removed from the node, since it locks `$XDG_RUNTIME_DIR` or a temp-directory path that a current load never takes; retirement doesn't end when the switch does

#### Scenario: Overlapping switches
- **WHEN** two version switches overlap on one node, such as an upgrade and a rollback that each make a different version runnable
- **THEN** both versions take the shared user lock, so a load under one waits for a load under the other to allocate instead of being admitted alongside it, and neither switch may make a retired loader runnable

#### Scenario: Telling a retired loader apart
- **WHEN** deciding whether a sparkforge copy holds a retired loader
- **THEN** a copy whose loader doesn't take `~/.local/state/sparkforge/sparkforge-gpu-admission.lock` is retired; a build without the `cuda` feature has no `load-weights` or `verify-weights` command, so it takes no admission and isn't affected
