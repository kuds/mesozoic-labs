#!/bin/bash
# F2 (OFF s42, session 1): a response the harness saved to a file (large files) is copied unchanged into the raw
# folder as <PATH with / as __>.t<K>.json (never overwriting), checked with cmp, then recorded with --saved.
# usage: xfer_saved.sh PATH HARNESS_FILE K
RUN=stance7_r10LTSBoff_s42; P=$1; S=$2; K=$3
STEM=${P//\//__}
OUT=/home/user/work/pc_study/round/fetch/raw/F2/$RUN/$STEM.t$K.json
cd /tmp || exit 9
case "$S" in /root/.claude/projects/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/tool-results/*) ;; *) echo "REFUSED: not a tool-results file" >&2; exit 2;; esac
[ -e "$OUT" ] && { echo "REFUSED: $OUT exists" >&2; exit 3; }
cp --update=none "$S" "$OUT" || exit 4
cmp "$S" "$OUT" || exit 5
echo "copied unchanged: $OUT ($(stat -c %s "$OUT") bytes)"
/home/user/work/pc_study/round/fetch/xtools/ft.sh transfer --fetch F2 --run "$RUN" --path "$P" --saved "$OUT"
