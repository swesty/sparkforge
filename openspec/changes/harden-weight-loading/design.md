# Design

## Context

See proposal.md. GB10 device memory comes from the same unified pool as host memory and bypasses the OOM killer, so the memory guard has to hold across processes and across users.

## Decisions

1. **Lock location from ownership, not environment.** The order:
   - The node-wide lock, when `/run/sparkforge` is a root-owned directory not writable by others and holds a root-owned regular `admission.lock` (mode 0644; read access is enough for `flock`). Every user opens it read-only and takes `flock` on it.
   - Otherwise `/run/user/<uid>` if this UID owns it and others can't write it.
   - Otherwise `<home>/.local/state/sparkforge`, created 0700, with the home directory from `getpwuid_r`.

   A node directory that exists but fails those checks is an error, so a misprovisioned node isn't silently treated as unprovisioned. The resolver takes every root as a parameter (`admission_lock_path_in`), so tests cover each branch without root.
2. **Provisioning is a deploy step, not runtime code.** `/run` is tmpfs, so each GB10 recreates the lock at boot: `install -d -m 0755 -o root /run/sparkforge && install -m 0644 -o root /dev/null /run/sparkforge/admission.lock`. sparkforge never needs root.
3. **A per-user fallback warns.** Without the node lock, loads by different users can still both pass the machine-wide guard, so the loader prints that only this user's processes are serialized.
4. **One generation per load.** `resolve_checkpoint_dir` canonicalizes once. The CLI reads `config.json`, and `Checkpoint::open` reads the index and shards, from that path, so switching a symlink mid-load can't mix generations. The per-shard stamps from the earlier hardening still catch in-place rewrites.
5. **Metadata uses the shard open rules.** `config.json` and the index are opened non-blocking, must be regular files, and are capped at 256 MB (Flash Next's index is about 33 MB). Index presence uses `symlink_metadata`, so a dangling index link is an error naming it.
6. **Admission before CUDA.** `weights::admit_load` (host) takes the lock and checks the memory guard for the copied bytes plus overhead. `verify-weights` calls it before `Engine::load`, and `upload` takes the returned `Admission` and holds it until every allocation is filled.

## Risks / Trade-offs

- **[Any process that holds the node lock delays every other load on the node]** → `flock` is released when the process exits, and loads hold it only while allocating.
- **[Nodes without the deploy step coordinate per user only]** → The loader warns. The fleet runbook gets the provisioning step.
