## MODIFIED Requirements

### Requirement: Loads never exhaust host memory on a single-account node

A load SHALL be refused before any CUDA call if its copied bytes, staging buffer and, when
verifying, reference scratch exceed `MemAvailable` minus 4 GiB, and rechecked before allocating.
Admission SHALL be held until that memory is allocated, or freed after verification, and
serialized through one lock per account, under the home directory from the password database.
sparkforge assumes one account runs it per node.

#### Scenario: Too large to copy
- **WHEN** the bytes to copy plus overhead exceed `MemAvailable` minus 4 GiB
- **THEN** loading fails before any CUDA call, stating both numbers

#### Scenario: Concurrent loads
- **WHEN** two sparkforge loads of 14 GiB start together with 20 GiB available
- **THEN** the second waits until the first has allocated, then is refused

#### Scenario: Second load during verification
- **WHEN** a second `verify-weights` starts while the first is copying or verifying, with room for only one
- **THEN** it waits until the first has freed its memory, then is admitted

#### Scenario: Login during a load
- **WHEN** a load is admitted and a login then creates the account's runtime directory
- **THEN** the next load waits on the same lock, which depends only on the home directory

#### Scenario: Lock location from ownership
- **WHEN** the account has no password-database home, or its lock directory is writable by others
- **THEN** admission refuses instead of using an environment variable or a shared temp path

#### Scenario: Stuck holder
- **WHEN** another process holds the lock for 30 minutes
- **THEN** the waiting load fails naming the lock instead of hanging

#### Scenario: Holder released after the wait
- **WHEN** the lock was found held and is released only after the wait limit has passed
- **THEN** the waiting load times out instead of being admitted late

#### Scenario: Lock replaced while waiting
- **WHEN** a load waits on the lock file and the file is replaced before it gets the lock
- **THEN** it locks the new file within the same deadline, so it is never admitted alongside a load holding the new file

#### Scenario: Upgrade from a loader before SHA-765
- **WHEN** sparkforge is upgraded from any earlier version, including one that locked in `/run/user/<uid>`, `$XDG_RUNTIME_DIR`, the temp directory or `/run/sparkforge`
- **THEN** the switch follows "Loader version switches follow one cutover rule", since an earlier version may hold a lock this one doesn't take

#### Scenario: State directory deleted during a load
- **WHEN** the account deletes the lock file or `~/.local/state/sparkforge` while a load is admitted
- **THEN** that is unsupported: a load started afterwards locks a new file and isn't coordinated with the admitted one, so the directory must not be removed while sparkforge runs

#### Scenario: A different home mounted at the account's home path
- **WHEN** a sparkforge process runs as the account but sees a different directory at its home path, as in a container that mounts its own home
- **THEN** that is unsupported: it takes a different lock file and isn't coordinated with the host's loads, so every sparkforge process on a node must see the same `~/.local/state/sparkforge`

#### Scenario: Memory used after admission
- **WHEN** `MemAvailable` drops below the guard between admission and upload
- **THEN** the upload refuses before allocating

#### Scenario: Wide vocabulary
- **WHEN** `verify-weights` checks Llama 3.1 8B's `lm_head` (1.05 GB of weights)
- **THEN** the process's peak host memory stays bounded (measured 225 MB) instead of scaling with the tensor

## ADDED Requirements

### Requirement: Loader version switches follow one cutover rule

Switching a node's sparkforge version, as an upgrade or a rollback, SHALL first disable every way
to start the outgoing version, and SHALL make the incoming version runnable only after every
invocation of the outgoing one has exited. Two versions that lock different files can both be
admitted before either allocates.

#### Scenario: Disable before waiting
- **WHEN** a node switches sparkforge versions in either direction
- **THEN** the outgoing version is disabled first: every launcher (a script or scheduled job) stops dispatching it, its pending invocations are cancelled, and no copy of it is left runnable on the node, counting installed binaries and build outputs (a kept copy is non-executable or off the node); then the switch waits until no invocation of it is running; until that wait ends the incoming version isn't runnable either and nothing is run from a source checkout, since a checkout can build either version; only then is the incoming version made runnable and its launchers resumed

#### Scenario: Start during the cutover window
- **WHEN** the outgoing binary is started by hand after its launchers stop but before it is made non-executable
- **THEN** that run is one of the invocations the switch waits for, since the wait begins only after every copy is disabled and removing execute permission doesn't stop a running process, so no incoming-version load starts while it runs

#### Scenario: Rollback
- **WHEN** a node rolls back to the version it ran before
- **THEN** the same rule applies with the roles swapped: the newer version's launchers stop, its pending invocations are cancelled and its copies are disabled, and the switch waits for its running invocations to exit before the older copy is made runnable again and its launchers resume
