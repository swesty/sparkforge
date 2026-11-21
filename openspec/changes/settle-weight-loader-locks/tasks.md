# Tasks

## 1. Old-loader locks (SHA-788 follow-ups 1 to 3)
- [x] 1.1 Remove `LockKind::Legacy`, `AdmissionLocks::legacy`, `legacy_lock_dirs`, `runtime_dirs_of_processes` and the inode deduplication; locks are user then node
- [x] 1.2 Remove the tests of the removed behaviour; the spec scenario states the drain rule

## 2. Node lock (follow-ups 4 and 5)
- [x] 2.1 `check_node_lock` on path metadata at discovery and on the descriptor in `admit`
- [x] 2.2 Test: a lock swapped for a mode 0644 file after discovery is refused at admission; on Linux with `setfacl`, one swapped for a file with a named ACL entry too
- [x] 2.3 Test: `/run/sparkforge` without `admission.lock` refuses; the error and the runbook say to remove only the file

## 3. Snapshots (follow-ups 6 and 7)
- [x] 3.1 Open `<repo>` once and open `snapshots/<rev>` and `blobs` from it
- [x] 3.2 Test: a revision moved to another repository between the two opens reads its own repository's blob
- [x] 3.3 Keep a `blobs` open error and report it only for blob links; test both sides with a symlinked `blobs`

## 4. Portability (follow-up 8)
- [x] 4.1 Gate Unix-only link code; `pread` fallback for targets that are neither Unix nor Windows
- [x] 4.2 `cargo check --lib --target wasm32-unknown-unknown` with no errors or warnings (spark-5)

## 5. Verification
- [x] 5.1 `tools/code-factory/verify.sh` on macOS; `cargo test` and `cargo build --release --features cuda` on spark-5
- [x] 5.2 `verify-weights` on Llama 3.1 8B on spark-5, from plain files and from a Hugging Face snapshot layout
