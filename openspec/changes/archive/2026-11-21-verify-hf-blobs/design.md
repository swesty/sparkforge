# Design

## Decisions

### 1. Content verification instead of directory checks

No finite sequence of separate directory lookups can show that a revision and a blob store were published together: an entry can be swapped and swapped back between any two checks. What a snapshot actually promises is that each link names its bytes by hash. So each blob is verified against its name. The links are read from the revision opened at resolution (the generation of the config), so the bytes are fixed no matter which store or repository the blob is found in. This also catches corrupt downloads.

- **Names:** 64 hex digits is the SHA256 of the content (LFS files). 40 is the Git object SHA1 of `blob <len>\0<content>` (small files such as `config.json`).
- **Shards** are hashed with one thread per shard, chunked by `available_parallelism`. The stamp taken before hashing must still hold after hashing and when the header is read, and every later read is checked against it.
- **Metadata files** are hashed over exactly the bytes returned to the caller.
- **Cost on GB10:** `sha2` without `asm` runs at 0.57 GB/s, which made an 8B load about 11 s slower. With `asm` (hardware SHA256) it runs at 2.14 GB/s, and the 8B load is about 0.5 to 1.3 s slower than plain files (three runs each). `sha2-asm` is built only on x86, x86_64 and aarch64.

Alternative considered: a publication protocol (a generation file, or swapping a single symlink to a repository). Rejected: the Hugging Face cache doesn't follow one, and sparkforge reads caches it doesn't write.

### 2. Repair: close, drain, replace

Taking the old file's lock can stall on an outside holder, and an older build's waiter wouldn't re-check after the replacement. So the repair doesn't use the lock:
1. `chmod 0600` stops members opening the file, so new loads fail with a message saying a repair is under way. That message applies only to the read-only node lock, not the user lock.
2. Wait until `pgrep -af sparkforge` is empty. That covers holders and waiters of every build.
3. `rm -f` the file, never the directory, then `systemd-tmpfiles --create`.

The re-check after `flock` stays as a second line of defence.

### 3. A synchronized test

`admit_observed` calls a hook the first time each opened lock file is found held. The replaced-lock test waits for that signal before replacing the file, so load B certainly opened the old file. With the re-check disabled, the test fails in 3 of 3 runs.
