#!/usr/bin/env bash
# PC round (STANCE7) results durable copy, at the maintainer's word of 2026-11-02 ('Push it (Recommended)'): text only.
# The round's pinned files and fetch records (round/ without the fetched run files), the measurements' summaries
# (round/panels), F2 (scores, both readings, the independent checks, the measure logs), the F2 preparation (VP1, VP4,
# disk, deletions), the approved deviation reading (amend2), the final review (final/), COLAB_STEPS, and the run log.
# Left out: the fetched run files (round/fetch/F2, round/fetch/raw: they are on Drive), the measuring copy, every
# npz/zip/pkl/gz/png/mp4/pt/pth/pyc, tmp/ and caches, and any file over 5 MB (listed in skipped.txt).
set -euo pipefail
export LC_ALL=C
ST=/home/user/work/pc_study
RUNLOG=/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance4/live/run_log.txt
D=/home/user/work/pc_round_durable
STAGE=$D/stage
MAX=$((5 * 1024 * 1024))
rm -rf "$STAGE"; mkdir -p "$STAGE"; : > "$D/skipped.txt"
for top in round f2 f2prep amend2 final COLAB_STEPS.txt; do
  [ -e "$ST/$top" ] || continue
  while IFS= read -r -d '' f; do
    rel=${f#"$ST/"}
    n=$(stat -c %s "$f")
    if (( n > MAX )); then echo "$rel (over 5 MB, $n)" >> "$D/skipped.txt"; continue; fi
    mkdir -p "$STAGE/$(dirname "$rel")"; cp -p "$f" "$STAGE/$rel"
  done < <(find "$ST/$top" \( -path "$ST/round/fetch/F2" -o -path "$ST/round/fetch/raw" -o -name 'copy_*' -o -name '__pycache__' \
            -o -name 'tmp' -o -name 'pytest-of-*' -o -name 'site-packages' \) -prune \
            -o -type f ! -name '*.npz' ! -name '*.npy' ! -name '*.zip' ! -name '*.pkl' ! -name '*.gz' ! -name '*.tar' \
            ! -name '*.pt' ! -name '*.pth' ! -name '*.png' ! -name '*.mp4' ! -name '*.pdf' ! -name '*.pyc' ! -name '*.so' -print0 | sort -z)
done
cp -p "$RUNLOG" "$STAGE/run_log.txt"
