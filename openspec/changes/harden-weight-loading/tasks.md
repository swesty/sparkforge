# Tasks

- [x] 1.1 Resolve the lock location from ownership: node-wide when provisioned, else `/run/user/<uid>`, else a private directory under the password-database home directory. Verify: `cargo test` covers the shared node lock for two UIDs, the refusal of a misprovisioned node directory, and the user fallbacks; the old temp-dir fallback is never used.
- [x] 1.2 Decide admission on the host before CUDA and hold it through the upload. Verify: `cargo test` checks that a 1 GiB `MemAvailable` refuses through `admit_load_with` without any CUDA call.
- [x] 1.3 Resolve the checkpoint directory once. Verify: `cargo test` opens through a symlink, switches it to another generation, and still reads the first.
- [x] 1.4 Open the metadata files non-blocking and as regular files, and treat a dangling index link as an error. Verify: `cargo test` covers a FIFO at `config.json` and at the index (each fails within 5 s), and a dangling index symlink naming the index.
- [x] 1.5 Check presence on the resolved `lm_head` binding. Verify: `cargo test` checks that an untied checkpoint without `lm_head.weight` is refused, while the tied spec binds it.
- [x] 1.6 `verify-weights` on Llama 3.1 8B passes on a GB10 with GEMV numbers recorded in the PR. Verify: spark-5, all checks 3.63e-3 to 3.79e-3, 230–240 GB/s, 13.98 GiB in 9.1 s, 226 MB peak host RSS.

## Workflow follow-up

- Add the `/run/sparkforge` provisioning step to the fleet runbook (homelab-build) and apply it on the GB10 nodes (needs root).
- Archive with `openspec archive harden-weight-loading` after merge.
