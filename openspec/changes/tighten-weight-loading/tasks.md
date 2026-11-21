# Tasks

## 1. Shard links (SHA-787 follow-up 1)
- [x] 1.1 Follow a shard link only when its target stays inside the checkpoint directory or is a Hugging Face snapshot's `../../blobs/<hash>`
- [x] 1.2 Test: a direct link to a real outside directory is refused after the config read (absolute and relative); a link that leaves and re-enters is refused; an inside link opens
- [x] 1.3 Test: a snapshot's blob link opens; a non-hash blob name and a non-snapshot directory are refused

## 2. Upgrades from a custom runtime directory (follow-up 2)
- [x] 2.1 `legacy_lock_dirs` adds an absolute `$XDG_RUNTIME_DIR`, deduplicated
- [x] 2.2 Test: an old holder in a custom runtime directory blocks a new load; nothing is created there when no old lock exists

## 3. Node lock ACL (follow-up 3)
- [x] 3.1 Parse `system.posix_acl_access` and refuse named-user or named-group grants
- [x] 3.2 Test: parser cases (plain, named user, named group, zero-permission entry, malformed, wrong version)
- [x] 3.3 Test (Linux, `setfacl`): a 0640 lock with `u:65534:r` is refused and accepted again after `setfacl -b`

## 4. Admission through verification (follow-up 4)
- [x] 4.1 `upload` borrows the admission; `verify-weights` holds it until `verify_weights` returns
- [x] 4.2 `verify_weights` frees the weights and synchronizes before returning
- [x] 4.3 GB10: two concurrent `verify-weights` runs of Llama 3.1 8B; the second is admitted after the first frees its memory and passes

## 5. Search-only directories (follow-up 5)
- [x] 5.1 Open the checkpoint directory and walked directories with `O_PATH` on Linux
- [x] 5.2 Test (Linux, non-root): an indexed checkpoint in a mode 0100 directory with a mode 0100 shard subdirectory loads

## 6. Verification
- [x] 6.1 `cargo test` on macOS and on spark-5; `cargo build --release --features cuda` on spark-5
- [x] 6.2 `verify-weights` on Llama 3.1 8B passes on spark-5
