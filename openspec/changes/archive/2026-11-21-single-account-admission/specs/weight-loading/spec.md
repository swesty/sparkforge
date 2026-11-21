## REMOVED Requirements

### Requirement: Loads never exhaust host memory

**Reason**: It also required a node-wide lock shared between accounts, through a `sparkforge` group, with ACL checks and a repair procedure. Every Spark runs sparkforge under the single account `spark`, so there is no second account to coordinate with.
**Migration**: Replaced by "Loads never exhaust host memory on a single-account node", which keeps the memory guard, the per-user lock and every scenario that applies to one account. Remove `/etc/tmpfiles.d/sparkforge.conf` and `/run/sparkforge` if a node was provisioned (none were).

## ADDED Requirements

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

#### Scenario: Memory used after admission
- **WHEN** `MemAvailable` drops below the guard between admission and upload
- **THEN** the upload refuses before allocating

#### Scenario: Wide vocabulary
- **WHEN** `verify-weights` checks Llama 3.1 8B's `lm_head` (1.05 GB of weights)
- **THEN** the process's peak host memory stays bounded (measured 225 MB) instead of scaling with the tensor
