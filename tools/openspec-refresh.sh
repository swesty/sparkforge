#!/usr/bin/env bash
# Regenerate this repo's OpenSpec skills with the pinned OpenSpec version. Use this instead of a
# bare `openspec update`.
#
# Why a script:
# - A contributor whose PATH still resolves an older OpenSpec (1.x generated the removed
#   .claude/commands/opsx layout) must be stopped before anything is written.
# - Spec merging and archiving stay with the deterministic CLI (`openspec archive <change>`).
#   The agent-driven openspec-sync-specs and openspec-archive-change skills that `openspec update`
#   installs are removed: codex-review found their 2.1.1 instructions can write outside the specs
#   root through a symlink, leave a multi-capability sync half-applied, race the archive move, and
#   overwrite a re-synced ADDED requirement (upstream Fission-AI/OpenSpec#2289).
set -euo pipefail

PINNED="2.1.1"
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

if ! command -v openspec >/dev/null 2>&1; then
  echo "openspec-refresh: openspec is not on PATH; install it with: npm install -g @fission-ai/openspec@$PINNED" >&2
  exit 1
fi
FOUND="$(openspec --version 2>/dev/null | tr -d '[:space:]')"
if [ "$FOUND" != "$PINNED" ]; then
  echo "openspec-refresh: found OpenSpec '$FOUND' at $(command -v openspec), this repo pins $PINNED." >&2
  echo "openspec-refresh: nothing was changed. Install the pinned version: npm install -g @fission-ai/openspec@$PINNED" >&2
  exit 1
fi

# Non-interactive; openspec/config.yaml records skillsScope: project, so skills stay in the repo.
openspec update

# Spec merging and archiving belong to the CLI here, not to agent-followed instructions.
for skill in openspec-sync-specs openspec-archive-change; do
  rm -rf ".agents/skills/$skill"
  rm -f ".claude/skills/$skill"
done

# Check the layout this repo expects.
fail=0
for skill in openspec-apply-change openspec-explore openspec-propose openspec-update-change; do
  [ -f ".agents/skills/$skill/SKILL.md" ] || { echo "openspec-refresh: missing .agents/skills/$skill/SKILL.md" >&2; fail=1; }
  [ -L ".claude/skills/$skill" ] || { echo "openspec-refresh: .claude/skills/$skill is not a link into .agents/skills" >&2; fail=1; }
done
[ ! -e ".claude/commands/opsx" ] || { echo "openspec-refresh: legacy .claude/commands/opsx is present" >&2; fail=1; }
exit "$fail"
