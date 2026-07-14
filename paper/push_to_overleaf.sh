#!/usr/bin/env bash
# Push this paper to an Overleaf project over Overleaf's Git bridge.
#
#   ./push_to_overleaf.sh https://git.overleaf.com/<project-id> ["commit message"]
#
# The Overleaf project must ALREADY EXIST — the Git bridge can push to a project
# but cannot create one. In Overleaf: New Project -> Blank Project, then
# Menu -> Git -> copy the URL; generate a token under Account Settings ->
# Git integration (Git access needs a plan that includes it).
#
# The token is read from $OVERLEAF_GIT_TOKEN, or from ~/.overleaf_token
# (chmod 600). It is passed via GIT_ASKPASS, so it never appears in a URL, in
# git config, in the process list, or in shell history.
#
# Only the paper's own files are copied into the Overleaf project's (unrelated)
# git history. This never pushes eco8-neaixt history anywhere.
set -euo pipefail

URL="${1:-}"
MSG="${2:-Update paper from eco8-neaixt}"
[ -n "$URL" ] || { echo "usage: $0 https://git.overleaf.com/<project-id> [msg]" >&2; exit 2; }

TOKEN="${OVERLEAF_GIT_TOKEN:-}"
if [ -z "$TOKEN" ] && [ -r "$HOME/.overleaf_token" ]; then
  TOKEN="$(tr -d '[:space:]' < "$HOME/.overleaf_token")"
fi
[ -n "$TOKEN" ] || {
  echo "ERROR: no token. Set OVERLEAF_GIT_TOKEN, or write it to ~/.overleaf_token (chmod 600)." >&2
  exit 2
}

PAPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Feed credentials to git without ever putting the token in a URL or on disk.
ASKPASS="$WORK/askpass.sh"
cat > "$ASKPASS" <<'EOS'
#!/bin/sh
case "$1" in
  Username*) echo git ;;
  *)         printf '%s' "$OVERLEAF_GIT_TOKEN" ;;
esac
EOS
chmod +x "$ASKPASS"
export OVERLEAF_GIT_TOKEN="$TOKEN" GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0

echo "[overleaf] cloning $URL"
git clone --quiet "$URL" "$WORK/proj"

echo "[overleaf] syncing paper files"
cd "$WORK/proj"
mkdir -p figures
cp "$PAPER_DIR"/main.tex "$PAPER_DIR"/refs.bib "$PAPER_DIR"/spconf.sty \
   "$PAPER_DIR"/IEEEbib.bst "$PAPER_DIR"/README.md .
cp "$PAPER_DIR"/figures/architecture.pdf "$PAPER_DIR"/figures/architecture.tex figures/
# A blank Overleaf project ships a placeholder main.tex we have just overwritten;
# nothing else of theirs is touched.

if git diff --quiet && git diff --cached --quiet && [ -z "$(git status --porcelain)" ]; then
  echo "[overleaf] already up to date — nothing to push."
  exit 0
fi

git add -A
git -c user.name="${GIT_AUTHOR_NAME:-$(git config --global user.name || echo paper-bot)}" \
    -c user.email="${GIT_AUTHOR_EMAIL:-$(git config --global user.email || echo paper@local)}" \
    commit --quiet -m "$MSG"
git push --quiet origin HEAD
echo "[overleaf] pushed. Open the project in Overleaf and hit Recompile."
