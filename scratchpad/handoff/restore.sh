#!/bin/bash
# Restore the stance5-replication working files from branch claude/handoff-stance5r into the scratchpad path
# the pinned tools expect, verify every file, and rebuild the judge clones at their pinned commits.
# Usage (from a checkout of this branch):  bash restore.sh [--force]
# Read only for the repository: it never writes into the main checkout. --force overwrites differing files.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SP=/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad
REMOTE=${REMOTE:-https://github.com/kuds/mesozoic-labs}
FORCE=${1:-}

cd "$HERE"
sha256sum -c --strict --quiet SHA256SUMS
echo "branch files verified ($(wc -l < SHA256SUMS) files)"

mkdir -p "$SP"
while read -r f; do
  src="$HERE/scratchpad/$f"; dst="$SP/$f"
  if [ -e "$dst" ] && ! cmp -s "$src" "$dst"; then
    if [ "$FORCE" != "--force" ]; then echo "refusing: $dst exists and differs (rerun with --force)"; exit 3; fi
  fi
  mkdir -p "$(dirname "$dst")"; cp -p "$src" "$dst"
done < FILELIST.txt
( cd "$SP" && sed 's#  scratchpad/#  #' "$HERE/SHA256SUMS" | grep -v '  patches/' | sha256sum -c --strict --quiet - )
echo "restored under $SP and re-verified"

# Judge clones: folder, branch, commit (each judge refuses to run unless its clone is clean at this commit).
while read -r dir branch commit; do
  d="$SP/fix/$dir"
  if [ ! -d "$d/.git" ]; then git clone -q --branch "$branch" "$REMOTE" "$d"; fi
  head=$(git -C "$d" rev-parse HEAD)
  if [ "$head" != "$commit" ]; then git -C "$d" fetch -q origin "$branch" && git -C "$d" checkout -q "$commit"; fi
  test "$(git -C "$d" rev-parse HEAD)" = "$commit" && test -z "$(git -C "$d" status --porcelain)" \
    && echo "clone $dir clean at ${commit:0:8}" || { echo "clone $dir NOT at $commit or dirty"; exit 4; }
done <<'EOF'
scratch_copband claude/screen-trex-r9-armlts-copband cf80ecb14d1630ce3b4dcdea28a05835d143470b
screen_s4_LTS claude/screen-trex-r9-armlt-stagger c1e146a82658112a9a1c0838250020b214ef303a
screen_s3_L claude/screen-trex-r9-arml-12m 7e6b557db999900bbc95df63ca651b1f2f4e2df1
screen_s3_LT claude/screen-trex-r9-armlt 1af2d4d767aebc89fa2cf73fc59a11004048f797
EOF
echo "done. Checkpoints are not restored: re-fetch from Drive (read-only) only when something must be re-judged."
