# Proposal

## Why

codex-review approved swesty/sparkforge#13 (`harden-weight-loading`) at round 3 of 3 with five follow-ups (SHA-787). Each one is a case where the loader's guarantees don't hold:
- **Direct shard links:** a shard linked straight to a file in a real outside directory (`/models/published/model.safetensors`) passes the no-symlink walk, yet that directory can be replaced by another generation between the config read and the shard open.
- **Upgrades:** the loader before SHA-765 locked whatever `XDG_RUNTIME_DIR` named; the new one checks only `/run/user/<uid>` and `<tmp>/sparkforge-<uid>`, so an old load running with a custom runtime directory doesn't block a new one.
- **ACLs:** a `0640 root:sparkforge` node lock can still be opened by an outside account through a POSIX ACL entry, which the mode bits don't show.
- **Verification scratch:** `upload` dropped the admission when the copy finished, before verification allocated its reference scratch.
- **Search-only directories:** the checkpoint directory was opened `O_RDONLY`, which needs list permission; path-based opens before SHA-765 needed only search permission, so a mode 0710 directory broke.

## What Changes

- **Shard links:** a shard symlink is followed only when its target stays inside the checkpoint directory, or is `../../blobs/<sha1 or sha256>` from a Hugging Face `snapshots/<revision>` directory (content-addressed, so the name can't come to mean other bytes). Absolute targets and targets that leave the directory are refused, naming the shard.
- **Upgrades:** an existing, user-owned lock in a private `$XDG_RUNTIME_DIR` is also taken as a legacy lock (never created).
- **ACLs:** on Linux the node lock's `system.posix_acl_access` is read, and an entry granting a named user or group any permission is refused.
- **Verification:** `upload` borrows the admission, and `verify-weights` holds it until verification has freed the weights and synchronized, so the memory is back in `MemAvailable` before the next load is admitted.
- **Directories:** on Linux the checkpoint directory and the directories on a shard's path are opened `O_PATH`, which needs only search permission.

**Expected GB10 effect:** no change in kernels or bandwidth. A second `verify-weights` started during a first one waits until the first has freed its memory and is then admitted, where it used to be admitted while the first's weights were resident and refused. Measured by `cargo test` and two concurrent `verify-weights` runs of Llama 3.1 8B on spark-5.

## Capabilities

### New Capabilities
None.

### Modified Capabilities
- `weight-loading`: shard link targets, search-only directories, ACL-aware node lock, custom runtime directory legacy lock, admission held through verification.

## Impact

- `src/weights.rs`: `CheckpointDir::open` link classification, `O_PATH` directory opens, `legacy_lock_dirs`, `acl_named_grant`, node-lock ACL check.
- `src/device_weights.rs`, `src/verify.rs`, `src/main.rs`: the admission is borrowed by `upload` and held by the caller; verification frees and synchronizes before returning.
- Checkpoints whose shards link to absolute paths or outside directories stop loading; they must be copied or linked inside the checkpoint directory.
