## MODIFIED Requirements

### Requirement: Loader version switches follow one cutover rule

Switching a node's sparkforge version, as an upgrade or a rollback, SHALL first disable every way
to start the outgoing version, and SHALL make the incoming version runnable only after every
invocation of the outgoing one has exited. An invocation includes a start still in progress: any
command begun before the outgoing version was disabled that may yet run it. Two versions that lock
different files can both be admitted before either allocates.

#### Scenario: Disable before waiting
- **WHEN** a node switches sparkforge versions in either direction
- **THEN** the outgoing version is disabled first: every launcher (a script or scheduled job) stops dispatching it, its pending invocations are cancelled, and no copy of it is left runnable on the node, counting installed binaries and build outputs (a kept copy is non-executable or off the node); then the switch waits until every command begun before the disable that may run it has exited or been cancelled, not only until no loader process is visible; until that wait ends the incoming version isn't runnable either and nothing is run from a source checkout, since a checkout can build either version; only then is the incoming version made runnable, every launcher that keeps dispatching is pointed at the incoming binary, and the launchers resume

#### Scenario: Start during the cutover window
- **WHEN** the outgoing binary is started by hand after its launchers stop, either before it is made non-executable or in an exec that has passed the permission check but not yet replaced the starting process
- **THEN** that start is one of the invocations the switch waits for, since removing execute permission stops neither a running process nor an exec already past its check, so no incoming-version load starts until the command that began it has exited or been cancelled

#### Scenario: Launcher naming a version-specific path
- **WHEN** a launcher names the outgoing binary by an absolute path, and the incoming binary is installed at a different path while the outgoing file is kept non-executable
- **THEN** the launcher is pointed at the incoming binary before it resumes, so it dispatches the incoming version instead of failing on the disabled copy

#### Scenario: Rollback
- **WHEN** a node rolls back to the version it ran before, and that version isn't retired (see "Loaders older than SHA-765 are retired")
- **THEN** the same rule applies with the roles swapped: the newer version's launchers stop, its pending invocations are cancelled and its copies are disabled, and the switch waits until every command that may run it has exited or been cancelled; then the older copy is made runnable again, and every launcher that keeps dispatching is pointed back at it before it resumes

## ADDED Requirements

### Requirement: Loaders older than SHA-765 are retired

No sparkforge build whose weight loader predates SHA-765 SHALL be built, run or rolled back to on a
Spark, including from a kept source checkout, because its admission lock doesn't exclude a current
load. Every loader from SHA-765 on locks `~/.local/state/sparkforge/sparkforge-gpu-admission.lock`,
so loads under any two of them are serialized by admission in whatever order versions are switched.

#### Scenario: Kept checkout of a retired loader
- **WHEN** a source checkout, build output or binary of a loader older than SHA-765 is still on a Spark, including one kept after a switch has finished
- **THEN** it is not built or run; its launchers stop, its pending invocations are cancelled, and once every running invocation of it has exited it is removed from the node, since removing its files doesn't stop a running process and it locks `$XDG_RUNTIME_DIR` or a temp-directory path that a current load never takes; retirement doesn't end when the switch does

#### Scenario: Rollback to a retired loader
- **WHEN** the version a rollback would restore is older than SHA-765, for example one restored from a snapshot
- **THEN** the rollback is refused before the current version is stopped, so the current version stays runnable; a rollback may only target a version from SHA-765 on

#### Scenario: Overlapping switches
- **WHEN** two version switches overlap on one node, such as an upgrade and a rollback that each make a different version runnable
- **THEN** both versions take the shared user lock, so a load under one waits for a load under the other to allocate instead of being admitted alongside it, and neither switch may make a retired loader runnable

#### Scenario: Telling a retired loader apart
- **WHEN** deciding whether a sparkforge copy holds a retired loader
- **THEN** a copy whose loader doesn't take `~/.local/state/sparkforge/sparkforge-gpu-admission.lock` is retired; a build without the `cuda` feature has no `load-weights` or `verify-weights` command, so it takes no admission and isn't affected
