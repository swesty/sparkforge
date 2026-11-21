# Design

## Decisions

### 1. Drop old-loader lock detection

The loader before #13 put its lock in `$XDG_RUNTIME_DIR`, or else in `$TMPDIR/sparkforge-<uid>`, both read from its own environment. A new process can learn that environment only through `/proc`, and only for processes in its PID namespace. Even then, two loads that each find the other's directory first can deadlock unless every process sorts the locks. Each review cycle found another gap in the detection. The upgrade it protects is a one-time event on nodes that already drain loads before changing the lock setup (homelab-build RUNBOOK §Spark-5), and spark-5 is the only node that has run that loader. So the detection goes, and the drain is the rule. With only the user lock and the node lock left, every process takes them in the same order, so lock ordering can't deadlock.

### 2. Node-lock checks on the locked descriptor

`admission_locks_in` still checks the path first, which gives an early, clear error. `admit` then opens the lock and runs the same `check_node_lock` on `fstat` and `fgetxattr` of that descriptor before `flock`. `LockKind::Node` carries the owner and group to check against.

### 3. Fail closed during repair

With `/run/sparkforge` present and `admission.lock` missing, discovery already fails. The repair instructions now say to remove only the file and leave the directory, so a load started mid-repair refuses rather than running under only its user lock. Removing the whole directory (needed only for a path of the wrong type) still requires stopping new loads, as the runbook says.

### 4. Snapshots from a pinned repository (and checked as a pair)

For `<repo>/snapshots/<rev>`, `<repo>` is opened (`O_PATH | O_DIRECTORY | O_NOFOLLOW` on Linux), then `snapshots/<rev>` and `blobs` are walked down from it without following symlinks. A revision moved between the two opens is still the one opened, and its blob store is that repository's. `resolve_checkpoint_dir_between` runs a closure between the two opens, so a test can move the revision there. A `blobs` that is missing, a symlink, or not a directory is recorded as an error string and reported only by a blob link that needs it.

After both opens, `snapshots/<rev>` and `blobs` are looked up again from the repository descriptor and must still name the opened directories (device and inode). If they do, the repository held exactly that revision and that store at that moment, which is a consistent pair. If not, resolution fails and asks for a retry (codex-review cycle 1). A store swapped before it is opened is just the repository's state at the check, and is used whole. `ResolveStep` marks the two hook points for tests.

### 5. Locks follow file replacement; one lock per file

After `flock` succeeds, `admit` checks that the lock's path still names the locked file. If the file was replaced while the load waited, the load drops it and locks the new file. The runbook's repair replaces the file while holding the old lock. Together, these mean a load allocating under the old file finishes first, and every later load ends up on the new file, so two lock files never admit at once. `admit` also locks each file once by (device, inode), declared only on Unix, so duplicate entries don't wait on themselves (codex-review cycle 1).
