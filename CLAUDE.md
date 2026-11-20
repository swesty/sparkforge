# sparkforge — repo instructions

This repo uses **OpenSpec** for spec-driven development, layered with the global PAI
framework (`~/.claude/CLAUDE.md`) by **scope**. Read this before multi-step work here.

## This repo

- **Target:** NVIDIA DGX Spark only (GB10, cc 12.1, `sm_121a`). `openspec/config.yaml` lists the
  hardware limits and conventions every change must respect; the capability specs are in
  `openspec/specs/`.
- **Delivery:** Code Factory (automatic `codex-review`, advisory). Work from a Linear issue in team
  SHA, project "sparkforge": `wt switch --create <issue branch>`, `wt step commit`, `wt ready`.
  Merge on GitHub only.
- **"Done" for a kernel** means `cargo run --release --features cuda -- verify <config> --bench`
  passes on a GB10, with the host, CUDA version and numbers recorded in the PR. CI runs only
  `cargo test` and `cargo check --features cuda` (it has no GPU).

## Source of truth

- **OpenSpec owns the durable spec.** `openspec/specs/` is the source of truth for what
  each capability currently does. `openspec/changes/` holds in-flight proposals.
- **PAI owns execution + verification.** The PAI Algorithm (PRD + ISC in `MEMORY/WORK/`)
  drives *how* work gets done and verified.

## The loop (OpenSpec stage → PAI Algorithm phase)

| OpenSpec | PAI Algorithm |
|---|---|
| `/openspec-propose` (proposal + tasks + spec deltas) | OBSERVE — its `tasks.md` **seeds the ISC** |
| `/openspec-apply-change` | BUILD + EXECUTE |
| PAI VERIFY | ISC pass/fail in `PRD.md` |
| `openspec archive <change>` + git | LEARN — the CLI merges deltas into `openspec/specs/` |

## Rules

- **PRD references OpenSpec; it never duplicates it.** A PAI `PRD.md` for work here puts the
  change name + `openspec/changes/<name>/` path in its Context and derives ISC from
  `tasks.md`. Do not restate the spec in the PRD.
- **Bringing an OpenSpec change into a PAI run:** use the `_OPENSPEC` PAI skill
  (`~/.claude/skills/_OPENSPEC/`) to seed a PRD from a change.
- **Spec edits go through OpenSpec** (`/openspec-propose` → `/openspec-apply-change` → `openspec archive <change>`),
  not by hand-editing `openspec/specs/` — except for backfilling already-built capabilities.
- OpenSpec skills (`openspec-propose`, `-apply-change`, `-explore`, `-update-change`) live in
  `.agents/skills/`; `.claude/skills/` holds relative links to them. `skillsScope: project` in
  `openspec/config.yaml` keeps them in the repo. There are no `/opsx:*` commands in 2.x; invoke the
  skills (for example `/openspec-propose`) or just ask.
- **Merge specs and archive with the CLI: `openspec archive <change>`.** It applies deltas in code
  (an identical ADDED requirement is a no-op, a conflicting one aborts). The agent-driven
  `openspec-sync-specs` and `openspec-archive-change` skills are deliberately not installed here:
  codex-review found their 2.1.1 instructions can write outside the specs root through a symlink,
  leave a multi-capability sync half-applied, race the archive move, and overwrite a re-synced
  ADDED requirement (upstream Fission-AI/OpenSpec#2289).
- **Refresh the skills only with `tools/openspec-refresh.sh`**, never a bare `openspec update`. It
  refuses to run unless `openspec --version` is the pinned 2.1.1, then runs `openspec update` on
  the repo that contains the script, removes the two skills above (only their generated
  `SKILL.md`, even when the update fails), and checks that each `.claude/skills` link resolves to
  its own `.agents/skills` directory. Never run `openspec update --force` here: it also deletes the
  global `~/.codex/prompts/opsx-*` files other repos use. After changing the script, run
  `tools/test-openspec-refresh.sh` (stub `openspec`, no GPU or network needed).
