# Proposal

## Why

The safetensors loader (`weight-loading`) has nine open edge cases from codex-review of its PRs (swesty/sparkforge#4 and the `add-weight-loading` archive), and the mixed-format Flash Next loader will build on this code:
- **Admission lock:** the lock path depends on `XDG_RUNTIME_DIR`; its `/tmp` fallback can be pre-claimed by another user; and per-user locks let two users pass the machine-wide memory guard together.
- **Generations:** a checkpoint symlink switched mid-load can mix two generations.
- **Metadata:** a FIFO at `config.json` or the index hangs the loader, and a dangling index symlink silently falls back to an old single file.
- **CUDA order:** `verify-weights` creates a CUDA context before the memory guard runs.
- **Spec wording:** the spec doesn't say that the tied `lm_head` binding satisfies the presence check.

## What Changes

- **Admission lock:**
  - A node-wide lock (`/run/sparkforge/admission.lock`, root-provisioned) coordinates every user's loads.
  - Without it, a per-user lock is chosen by ownership (`/run/user/<uid>`, else a private directory under the home directory from the password database), never from environment variables or a shared temp path, with a warning that cross-user coordination is off.
  - A node directory that exists but isn't trustworthy is an error.
- **Generations:** the checkpoint directory is resolved once (canonicalized), and the config, the index and every shard are read from that one generation.
- **Metadata files:** `config.json` and the index follow the shard rules (non-blocking open, regular file, size cap). The index's presence is its directory entry, so a dangling symlink is an error.
- **CUDA order:** admission (lock and memory guard) is decided on the host before any CUDA call and held through the upload.
- **Tied `lm_head`:** the presence check applies to the resolved `lm_head` binding.

**Expected GB10 effect:** none on kernels or bandwidth. Llama 3.1 8B `verify-weights` must still pass on a GB10 with unchanged GEMV numbers. Measured by `cargo test` and `verify-weights` on spark-5.

## Capabilities

### New Capabilities
None.

### Modified Capabilities
- `weight-loading`: checkpoint discovery and generation pinning, metadata open rules, resolved `lm_head` presence, and node-wide admission before CUDA.

## Impact

- **Code:** `src/weights.rs` (lock resolution, `admit_load`, `bytes_to_copy` moved here, metadata reads, directory resolution), `src/device_weights.rs` (`upload` takes the `Admission`), `src/verify.rs`, `src/main.rs`.
- **Deploy step on each GB10:** provision `/run/sparkforge/admission.lock` as root at boot, for node-wide coordination.

## Non-goals

- Mixed-format loading (SHA-767), and coordinating with programs other than sparkforge, which are still seen only through `MemAvailable`.
