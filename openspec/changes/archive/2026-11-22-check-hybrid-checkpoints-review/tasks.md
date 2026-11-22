# Tasks

## 1. codex-review cycle 1 follow-ups on #18
- [x] 1.1 Planning without `split_ngram_parts`; the tensor check still requires it (fp:3600da7c4e15)
- [x] 1.2 `apply_checked_table`; `weights` prices the table at its checked rows (fp:06e6fe0d9f78)
- [x] 1.3 Per-layer MTP formats in the inventory and the budget (fp:de9c37520703)
- [x] 1.4 Group members exactly the planned experts and shards (fp:a93bb1c2cb52)
- [x] 1.5 Tied NVFP4 embedding requires the head GEMV's activation scale (fp:7027c462f116)
- [x] 1.6 Checked arithmetic for the PLE row total and padding (fp:c89c702ebdf6)

## 2. Verification
- [x] 2.1 `verify.sh` on macOS; tests and CUDA build on spark-5; `weights` and `plan --checkpoint` on the reference checkpoint
