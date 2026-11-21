# Design

## Decisions

### 1. Which shard links are followed

A link is judged lexically from the link's own directory: its target must never leave the checkpoint directory, even to come back in, because a path through the parent can be replaced. The one exception is the Hugging Face cache layout, `<repo>/snapshots/<rev>/<file> -> ../../blobs/<hash>`: it is accepted only when the opened directory's parent is named `snapshots` and the blob name is 40 or 64 hex digits. Blobs are named by their content hash, so the name cannot come to mean another generation. The target is still walked with `O_NOFOLLOW` on every component, so a symlink on the way (an alias) is refused as before.

Alternative considered: accept outside targets and re-check them after the load. Rejected: a check after the fact can't tell a replaced directory from an unchanged one without pinning it, and pinning every external directory adds a second generation to reason about.

### 2. `O_PATH` for directories (Linux)

`O_PATH` descriptors serve as `openat`, `fstatat` and `readlinkat` bases and need only search permission on the directory. Other platforms keep `O_RDONLY` (macOS has no `O_PATH`); production is Linux on GB10.

### 3. Legacy lock in `$XDG_RUNTIME_DIR`

The environment is read only to find a lock an older binary may hold. It is taken only if it already exists, belongs to this user and sits in a private directory, and it is never created, so the user lock in the password-database home stays the one meeting point. A relative value is ignored (XDG Base Directory spec). Upgrades should still drain running loads first, as the runbook says.

### 4. ACL check on the node lock

On Linux, `lgetxattr(system.posix_acl_access)` is parsed (version 2, 8-byte entries). Any `ACL_USER` or `ACL_GROUP` entry with a nonzero permission is refused, regardless of the mask. `ENODATA` (no ACL) and `EOPNOTSUPP` (no ACL support) mean none. A value that doesn't parse is an error. The directory needs no ACL check: a named entry with write access raises the mask, which shows in the group bits the existing `mode & 0o022` check already refuses.

### 5. Admission through verification

`upload(&Admission)` confirms the admission and borrows it, and `verify-weights` drops it only after `verify_weights` returns. Freeing alone is not enough: cudarc frees with `cuMemFreeAsync` into the stream's pool, which returns memory to the system only at a synchronize. Measured on spark-5 before the synchronize was added: the queued load was admitted the moment the lock was released, read `MemAvailable` 11.27 GiB (the first load's 14 GiB still counted), and was refused. `verify_weights` therefore drops the weights and synchronizes the stream before returning.
