# Design

## Decisions

### 1. Deleting the lock while admitted is out of scope

A load holds its `flock` on the file it opened. If that file is unlinked, a new load creates and locks a different file, and nothing connects the two. Re-checking at admission covers a replacement before the lock is taken. It can't cover a deletion after, because the admitted load has nothing left to check against until it allocates. Locking a directory instead would only move the problem: the directory can be deleted too, and `flock` on a directory over NFS needs a writable descriptor, which directories don't have. One account owns `~/.local/state/sparkforge` and nothing on the Sparks cleans it, so the requirement is simply not to delete it while sparkforge runs.

### 2. An injectable pause

The deadline test needs an exact order of events: the lock found held, the deadline passing, the lock released, then the next attempt. Real sleeps can't guarantee that order on a loaded runner. `AdmitHooks::sleep` lets the test supply the pause between attempts. Production uses `std::thread::sleep`.
