# Design

## Decisions

### 1. One account per node

The node lock's checks only make sense if another local account might be hostile or careless: a lock any account can open, the wrong group, ACL grants, a file swapped in, a repair stalled by an outside holder. With one account (`spark`) on every Spark, the per-user lock is a lock shared by every sparkforge process on the node. What still matters is the memory race between that account's own loads, and the per-user lock plus the memory guard handle it. If a second account is ever added, the node lock comes back. #13 to #15 and the archived changes `harden-weight-loading`, `tighten-weight-loading`, `settle-weight-loader-locks` and `verify-hf-blobs` record its design.

### 2. Simpler admission

`admit` takes one lock path. Without other accounts or a root-provisioned file there is nothing to re-check on the descriptor, no file another party replaces, and no second entry to deduplicate. The lock directory must still be this account's and not writable by others, and the lock must be a regular file opened without following a symlink.

### 3. Blob verification without a panic

`Scope::spawn` panics if the OS refuses a thread (for example, at a container's task limit). `Builder::spawn_scoped` returns an error instead, and that blob is verified on the calling thread. `Checkpoint::open_dir_with(dir, 0)` (hidden from docs) takes the same path for every blob, which is how the fallback is tested.
