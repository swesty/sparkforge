# Design

## Decisions

### 1. Which shard links are followed

A link is judged lexically from the link's own directory: its target must never leave the checkpoint directory, even to come back in, because a path through the parent can be replaced. The one exception is the Hugging Face cache layout, `<repo>/snapshots/<rev>/<file> -> ../../blobs/<hash>`: it is accepted only when the opened directory's parent is named `snapshots` and the blob name is 40 or 64 hex digits. The target is still walked with `O_NOFOLLOW` on every component, so a symlink on the way (an alias) is refused as before.

The filesystem never resolves a link's `..` (codex-review cycle 1). An inside target is reduced to a path with no `..` and walked from the pinned checkpoint descriptor, so a subdirectory renamed between two opens can't carry `..` outside. For a snapshot, `../../blobs` is opened once, when the snapshot is resolved, and blob links are opened in that descriptor. Replacing `<repo>/blobs` mid-load therefore can't substitute other bytes under the same hash name, which puts the blob store on the same footing as the checkpoint directory. Rewriting a blob file in place, or renaming another file over it inside the pinned store, is the same case as doing that to a file inside the checkpoint directory: the per-shard stamps catch in-place writes, and the rest is outside the one-generation guarantee, which is about directory-level replacement.

Alternative considered: hash every blob against its name. Rejected for now: SHA-256 over a 14 to 50 GB checkpoint adds seconds to tens of seconds to every load, and git-object blob names (SHA-1 of a header plus the content) need a second scheme.

Alternative considered: accept outside targets and re-check them after the load. Rejected: a check after the fact can't tell a replaced directory from an unchanged one without pinning it, and pinning every external directory adds a second generation to reason about.

### 2. `O_PATH` for directories (Linux)

`O_PATH` descriptors serve as `openat`, `fstatat` and `readlinkat` bases and need only search permission on the directory. Other platforms keep `O_RDONLY` (macOS has no `O_PATH`); production is Linux on GB10.

### 3. Legacy lock in `$XDG_RUNTIME_DIR`

The environment is read only to find a lock an older binary may hold. On Linux the `XDG_RUNTIME_DIR` of every process this user runs is read from `/proc/<pid>/environ` (codex-review cycle 1). An old load started with a runtime directory this process wasn't started with is found through its own environment, which is the one place that records it. It is taken only if it already exists, belongs to this user and sits in a private directory, and it is never created, so the user lock in the password-database home stays the one meeting point. A relative value is ignored (XDG Base Directory spec). Upgrades should still drain running loads first, as the runbook says.

### 4. ACL check on the node lock

On Linux, `lgetxattr(system.posix_acl_access)` is parsed (version 2, 8-byte entries). Any `ACL_USER` or `ACL_GROUP` entry with a nonzero permission is refused, regardless of the mask. The repair is to replace the file, not to remove the ACL: a descriptor opened while the ACL allowed it keeps working after the ACL is removed, while a replaced file is a new inode that the old descriptor doesn't reach (codex-review cycle 1). The error message says so. `ENODATA` (no ACL) and `EOPNOTSUPP` (no ACL support) mean none. A value that doesn't parse is an error. The directory needs no ACL check: a named entry with write access raises the mask, which shows in the group bits the existing `mode & 0o022` check already refuses.

### 5. Admission through verification

`upload(&Admission)` confirms the admission and borrows it, and `verify-weights` drops it only after `verify_weights` returns. Freeing alone is not enough: cudarc frees with `cuMemFreeAsync` into the stream's pool, which returns memory to the system only at a synchronize. Measured on spark-5 before the synchronize was added: the queued load was admitted the moment the lock was released, read `MemAvailable` 11.27 GiB (the first load's 14 GiB still counted), and was refused. `verify_weights` therefore drops the weights and synchronizes the stream before returning.

### 6. One flock per file

`admit` skips a lock whose (device, inode) it already holds. Two `flock`s on one file through two descriptors conflict, so two spellings of one runtime directory (`/run/user/1000/../1000`) would make admission wait on its own first lock (codex-review cycle 1).
