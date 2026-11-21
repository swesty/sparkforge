# Tasks

## 1. Remove the node lock
- [x] 1.1 Remove the node lock, ACL checks, re-lock, deduplication, `admit_observed` and the not-provisioned note
- [x] 1.2 `admission_lock` / `admission_lock_in` / `admit(lock, …)`; keep the ownership checks, the wait limit and the memory guard
- [x] 1.3 Remove the node-lock tests; adapt the user-lock tests; add a test that a lock directory others can write is refused

## 2. Blob verification threads
- [x] 2.1 `Builder::spawn_scoped` with verification on the calling thread when a worker can't be created
- [x] 2.2 Test: `open_dir_with(dir, 0)` verifies a blob and refuses a mismatch

## 3. Verification
- [x] 3.1 `verify.sh` on macOS; tests, wasm check and CUDA build on spark-5; `verify-weights` on Llama 3.1 8B
