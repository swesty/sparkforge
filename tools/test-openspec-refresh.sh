#!/usr/bin/env bash
# Tests for tools/openspec-refresh.sh. Needs bash and git only: no GPU, no network, and no real
# OpenSpec. A stub `openspec` on PATH stands in for the CLI, and every case runs a copy of the
# script inside a throwaway fixture repo, so this checkout is never touched.
#
#   tools/test-openspec-refresh.sh
#
# Stub controls (environment):
#   STUB_VERSION=<v>         what `openspec --version` prints (default 2.1.1)
#   STUB_FAIL_AFTER_SKILLS=1 `openspec update` exits 7 after writing every skill
#   STUB_TERM_PARENT=1       `openspec update` sends SIGTERM to the refresh script after writing
set -uo pipefail

HERE="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SCRIPT="${REFRESH_SCRIPT:-$HERE/openspec-refresh.sh}"  # REFRESH_SCRIPT: test another copy
TMP="$(mktemp -d)"
TMP="$(cd -P "$TMP" && pwd -P)"
trap 'rm -rf "$TMP"' EXIT

EXPECTED=(openspec-apply-change openspec-explore openspec-propose openspec-update-change)
EXCLUDED=(openspec-sync-specs openspec-archive-change)

# --- stub openspec -------------------------------------------------------------------------------
mkdir -p "$TMP/bin"
cat > "$TMP/bin/openspec" <<'STUB'
#!/usr/bin/env bash
# Stand-in for OpenSpec 2.1.1. `update [path]` writes all six skills into the target and links
# any missing .claude/skills entry; it leaves existing links alone, so a test can plant a bad one.
set -euo pipefail
case "${1:-}" in
  --version) echo "${STUB_VERSION:-2.1.1}" ;;
  update)
    target="${2:-$PWD}"
    echo "update cwd=$(pwd -P) target=$target" >> "$STUB_LOG"
    for s in openspec-apply-change openspec-explore openspec-propose openspec-update-change \
             openspec-sync-specs openspec-archive-change; do
      mkdir -p "$target/.agents/skills/$s" "$target/.claude/skills"
      echo "generated $s" > "$target/.agents/skills/$s/SKILL.md"
      l="$target/.claude/skills/$s"
      [ -e "$l" ] || [ -L "$l" ] || ln -s "../../.agents/skills/$s" "$l"
    done
    if [ "${STUB_TERM_PARENT:-}" = 1 ]; then kill -TERM "$PPID"; fi
    if [ "${STUB_FAIL_AFTER_SKILLS:-}" = 1 ]; then echo "stub: write failed" >&2; exit 7; fi
    ;;
  *) echo "stub openspec: unexpected arguments: $*" >&2; exit 64 ;;
esac
STUB
chmod +x "$TMP/bin/openspec"
export PATH="$TMP/bin:$PATH"

# --- helpers -------------------------------------------------------------------------------------
pass=0 failed=0
ok()   { echo "ok   - $1"; pass=$((pass + 1)); }
bad()  { echo "FAIL - $1" >&2; failed=$((failed + 1)); }
check() { if eval "$2"; then ok "$1"; else bad "$1"; fi; }

# A fresh sparkforge-shaped repo with the committed skill layout and a copy of the script.
fixture() {
  local d="$TMP/$1" s
  mkdir -p "$d/tools" "$d/openspec" "$d/.agents/skills" "$d/.claude/skills"
  git init -q "$d"
  cp "$SCRIPT" "$d/tools/openspec-refresh.sh"
  chmod +x "$d/tools/openspec-refresh.sh"
  echo "skillsScope: project" > "$d/openspec/config.yaml"
  for s in "${EXPECTED[@]}"; do
    mkdir -p "$d/.agents/skills/$s"
    echo "committed $s" > "$d/.agents/skills/$s/SKILL.md"
    ln -s "../../.agents/skills/$s" "$d/.claude/skills/$s"
  done
  echo "$d"
}

# Path, type, link target and content hash of everything outside .git.
snapshot() {
  (cd "$1" && find . -path ./.git -prune -o -print | LC_ALL=C sort | while IFS= read -r p; do
    if [ -L "$p" ]; then echo "L $p -> $(readlink "$p")"
    elif [ -f "$p" ]; then echo "F $p $(git hash-object "$p")"
    else echo "D $p"; fi
  done; git -C "$1" status --porcelain)
}

# Run a fixture's script by absolute path from directory $2. Sets RC and ERR.
run_from() {
  local fix="$1" cwd="$2"; shift 2
  export STUB_LOG="$TMP/stub.log"
  : > "$STUB_LOG"
  ERR="$(cd "$cwd" && env "$@" "$fix/tools/openspec-refresh.sh" 2>&1 >/dev/null)"
  RC=$?
}

no_excluded() {
  local fix="$1" s
  for s in "${EXCLUDED[@]}"; do
    [ ! -e "$fix/.agents/skills/$s/SKILL.md" ] && [ ! -L "$fix/.claude/skills/$s" ] \
      && [ ! -e "$fix/.claude/skills/$s" ] || return 1
  done
}

# --- 1. the repo comes from the script's location, not the caller's cwd ---------------------------
F="$(fixture item1)"
CALLER="$TMP/caller"
mkdir -p "$CALLER/openspec" "$CALLER/.agents/skills/openspec-sync-specs" "$CALLER/sub/dir"
git init -q "$CALLER"
echo "skillsScope: project" > "$CALLER/openspec/config.yaml"
echo "caller's own skill" > "$CALLER/.agents/skills/openspec-sync-specs/SKILL.md"
before="$(snapshot "$CALLER")"
run_from "$F" "$CALLER/sub/dir"
check "1: refresh from another repo succeeds (rc=$RC)" '[ "$RC" -eq 0 ]'
check "1: the caller repo is unchanged" '[ "$(snapshot "$CALLER")" = "$before" ]'
check "1: openspec update ran in and on the script's repo" \
  'grep -qx "update cwd=$F target=$F" "$STUB_LOG"'
check "1: the script's repo ends with no excluded skill" 'no_excluded "$F"'

# --- 2. a failed update still removes the excluded skills and keeps its status --------------------
F="$(fixture item2)"
run_from "$F" "$F" STUB_FAIL_AFTER_SKILLS=1
check "2: the refresh fails with the update's status (rc=$RC, want 7)" '[ "$RC" -eq 7 ]'
check "2: no excluded skill is left after the failure" 'no_excluded "$F"'

F="$(fixture item2-term)"
run_from "$F" "$F" STUB_TERM_PARENT=1
check "2: an interrupted refresh fails (rc=$RC, want 143)" '[ "$RC" -eq 143 ]'
check "2: no excluded skill is left after the interruption" 'no_excluded "$F"'

# --- 3. only the generated SKILL.md is deleted from an excluded skill directory -------------------
F="$(fixture item3)"
mkdir -p "$F/.agents/skills/openspec-sync-specs/notes"
echo "my notes" > "$F/.agents/skills/openspec-sync-specs/notes.md"
echo "more" > "$F/.agents/skills/openspec-sync-specs/notes/deep.txt"
run_from "$F" "$F"
check "3: the refresh succeeds (rc=$RC)" '[ "$RC" -eq 0 ]'
check "3: the untracked file survives with its content" \
  '[ "$(cat "$F/.agents/skills/openspec-sync-specs/notes.md" 2>/dev/null)" = "my notes" ]'
check "3: a nested untracked file survives" '[ -f "$F/.agents/skills/openspec-sync-specs/notes/deep.txt" ]'
check "3: the excluded skill is still disabled" 'no_excluded "$F"'
check "3: the kept files are reported" 'printf "%s" "$ERR" | grep -q "kept unexpected files.*notes.md"'

# --- 4. .claude/skills links must resolve to their own .agents/skills directory -------------------
F="$(fixture item4-dangling)"
rm "$F/.claude/skills/openspec-propose"
ln -s "../../.agents/skills/openspec-propose-gone" "$F/.claude/skills/openspec-propose"
run_from "$F" "$F"
check "4: a dangling link fails the refresh (rc=$RC)" '[ "$RC" -ne 0 ]'
check "4: the dangling link is named" 'printf "%s" "$ERR" | grep -q "openspec-propose is dangling"'
check "4: excluded skills are still removed" 'no_excluded "$F"'

F="$(fixture item4-stale)"
mkdir -p "$F/old-copy/openspec-propose"
echo "stale instructions" > "$F/old-copy/openspec-propose/SKILL.md"
rm "$F/.claude/skills/openspec-propose"
ln -s "../../old-copy/openspec-propose" "$F/.claude/skills/openspec-propose"
run_from "$F" "$F"
check "4: a link to a readable stale copy fails the refresh (rc=$RC)" '[ "$RC" -ne 0 ]'
check "4: the misdirected link is named" 'printf "%s" "$ERR" | grep -q "openspec-propose resolves to .*old-copy"'

F="$(fixture item4-wrong-skill)"
rm "$F/.claude/skills/openspec-propose"
ln -s "../../.agents/skills/openspec-explore" "$F/.claude/skills/openspec-propose"
run_from "$F" "$F"
check "4: a link to another skill's directory fails (rc=$RC)" '[ "$RC" -ne 0 ]'

F="$(fixture item4-copy)"
rm "$F/.claude/skills/openspec-propose"
cp -R "$F/.agents/skills/openspec-propose" "$F/.claude/skills/openspec-propose"
run_from "$F" "$F"
check "4: a copied directory instead of a link fails (rc=$RC)" '[ "$RC" -ne 0 ]'

# --- the version pin still writes nothing --------------------------------------------------------
F="$(fixture pin)"
before="$(snapshot "$F")"
run_from "$F" "$F" STUB_VERSION=1.6.0
check "pin: a different OpenSpec version is refused (rc=$RC)" '[ "$RC" -eq 1 ]'
check "pin: openspec update never ran" '! grep -q update "$STUB_LOG"'
check "pin: the repo is unchanged" '[ "$(snapshot "$F")" = "$before" ]'

echo "$pass passed, $failed failed"
[ "$failed" -eq 0 ]
