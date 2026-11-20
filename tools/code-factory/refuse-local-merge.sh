#!/usr/bin/env bash
# GitHub owns the merge gate. `wt merge` would update local main without the GitHub PR/status gate.
set -euo pipefail

cat >&2 <<'EOF'
Code Factory does not merge canonical branches locally.

Use `wt ready` to verify, push, and open/update the GitHub PR. Merge in GitHub only after the
`codex-review` status passes. After GitHub merges, remove the worktree with `wt remove`.
EOF
exit 1
