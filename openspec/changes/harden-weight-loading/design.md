# Design

## Context

See proposal.md. GB10 device memory comes from the same unified pool as host memory and bypasses the OOM killer, so the memory guard has to hold across processes and across users.

## Decisions

1. **Lock locations from ownership, not environment.** An admission takes up to two `flock`s, always in this order:
   - **The user lock,** `<home>/.local/state/sparkforge/sparkforge-gpu-admission.lock` (directory created 0700), with the home directory from `getpwuid_r`. It doesn't depend on a login session, so a background job and a login shell of one user always meet there. (An earlier draft preferred `/run/user/<uid>`, but that directory comes and goes with logins, so one user's loads could split across two locks.)
   - **The node lock,** when `/run/sparkforge` is a root-owned directory not writable by others and holds a root-owned regular `admission.lock` that grants no access to every account (mode 0640, group `sparkforge`). Members open it read-only, which is enough for `flock`. A lock every account could open would let any local account hold it and stall every load, so the resolver refuses one.

   A node directory that exists but fails those checks is an error, not a silent fallback. When the node lock is available the user lock is optional, so a numeric uid without a password entry still loads. Without either lock, admission refuses. Taking the user lock first means a load started while the node lock is being turned on still waits for this user's earlier load. The resolver takes every root as a parameter (`admission_locks_in`), so tests cover each branch without root.
2. **Provisioning is a deploy step, not runtime code.** `/run` is tmpfs, so each GB10 recreates the node lock at boot from a `tmpfiles.d` entry: `d /run/sparkforge 0755 root root -` and `f /run/sparkforge/admission.lock 0640 root sparkforge -`. Users who load weights join the `sparkforge` group. Turning the lock on drains running loads first, because another user's load admitted under its own user lock cannot be seen from the node lock. sparkforge never needs root.
3. **A per-user fallback warns, and waiting is bounded.** Without the node lock, loads by different users can still both pass the machine-wide guard, so the loader prints that only this user's processes are serialized. A load waits at most 30 minutes for a lock (`ADMISSION_WAIT`), then fails naming it.
4. **One generation per load.** `resolve_checkpoint_dir` canonicalizes once and opens the directory (`O_DIRECTORY | O_NOFOLLOW`). The config, the index and every shard are opened relative to that descriptor (`openat`), so neither a switched symlink nor a replaced directory can mix generations. The per-shard stamps from the earlier hardening still catch in-place rewrites.
5. **Metadata uses the shard open rules.** `config.json` and the index are opened non-blocking, must be regular files, and are capped at 256 MB (Flash Next's index is about 33 MB). The cap is checked on the bytes actually read too, so a file that grows after its size check fails instead of yielding a truncated prefix. Index presence uses `fstatat(AT_SYMLINK_NOFOLLOW)`, so a dangling index link is an error naming it.
6. **Admission before CUDA, confirmed before allocating.** `weights::admit_load` (host) takes the locks and checks the memory guard for the copied bytes plus overhead, before `Engine::load`. `upload` takes the returned `Admission`, confirms it covers the bytes actually copied and re-reads `MemAvailable` under the same locks (CUDA startup uses memory too), and holds it until every allocation is filled.

## Risks / Trade-offs

- **[A `sparkforge` group member that holds the node lock delays every other load on the node]** → Only group members can open it, `flock` is released when the process exits, loads hold it only while allocating, and waiting gives up after 30 minutes.
- **[Nodes without the deploy step coordinate per user only]** → The loader warns. The fleet runbook gets the provisioning step.
