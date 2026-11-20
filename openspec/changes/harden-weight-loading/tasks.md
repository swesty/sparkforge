# Tasks

- [x] 1.1 Resolve the lock location from ownership: node-wide when provisioned, and a private directory under the password-database home directory (superseded by 2.1–2.3). Verify: `cargo test` covers the shared node lock for two UIDs, the refusal of a misprovisioned node directory, and the user fallbacks; the old temp-dir fallback is never used.
- [x] 1.2 Decide admission on the host before CUDA and hold it through the upload. Verify: `cargo test` checks that a 1 GiB `MemAvailable` refuses through `admit_load_with` without any CUDA call.
- [x] 1.3 Resolve the checkpoint directory once. Verify: `cargo test` opens through a symlink, switches it to another generation, and still reads the first.
- [x] 1.4 Open the metadata files non-blocking and as regular files, and treat a dangling index link as an error. Verify: `cargo test` covers a FIFO at `config.json` and at the index (each fails within 5 s), and a dangling index symlink naming the index.
- [x] 1.5 Check presence on the resolved `lm_head` binding. Verify: `cargo test` checks that an untied checkpoint without `lm_head.weight` is refused, while the tied spec binds it.
- [x] 1.6 `verify-weights` on Llama 3.1 8B passes on a GB10 with GEMV numbers recorded in the PR. Verify: spark-5, all checks 3.63e-3 to 3.79e-3, 230–240 GB/s, 13.98 GiB in 9.1 s, 226 MB peak host RSS.

## 2. codex-review cycle 1

- [x] 2.1 Share the node lock through a group and bound the wait. Verify: `cargo test` refuses a lock that grants access to every account and times out a second admission after 200 ms.
- [x] 2.2 Keep one user lock under the home directory, take it before the node lock, and look up the home directory only for it. Verify: `cargo test` covers a login during a load, turning the node lock on during a load, and a uid with no password entry.
- [x] 2.3 Open the admission lock without blocking and require a regular file. Verify: `cargo test` fails a FIFO lock within 5 s.
- [x] 2.4 Read the config, index and shards relative to one opened directory. Verify: `cargo test` replaces the canonical directory after the config read and still reads the first generation.
- [x] 2.5 Reject metadata reads longer than the cap. Verify: `cargo test` refuses a 65-byte read under a 64-byte cap even though the first 64 bytes parse.
- [x] 2.6 Confirm the admission in `upload` against the bytes copied and re-read `MemAvailable`. Verify: `cargo test` refuses a larger binding and a drop below the guard after admission.
- [x] 2.7 `verify-weights` on Llama 3.1 8B still passes on a GB10. Verify: spark-5, 87 tests pass; checks 3.63e-3 to 3.79e-3 at 228–238 GB/s, 225 MB peak RSS; two concurrent 1B loads queue on the home-directory user lock.

## Workflow follow-up

- Add the `/run/sparkforge` provisioning step (group `sparkforge`, lock mode 0640, drain running loads first) to the fleet runbook (homelab-build), then apply it on the GB10 nodes (needs root).
- Archive with `openspec archive harden-weight-loading` after merge.
