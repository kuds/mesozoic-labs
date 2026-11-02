#!/bin/bash
# F2 (pc_s80): copies a harness-saved download response (tool-results file) unchanged into the raw folder under
# <PATH with / as __>.t<K>.json (refuses an existing name), then records it with the pinned fetch tool (transfer --saved)
# usage: xsaved.sh PATH SAVED_FILE K
RUN=stance7_r10LTSBpc_s80; P=$1; S=$2; K=$3
STEM=${P//\//__}
O=/home/user/work/pc_study/round/fetch/raw/F2/$RUN/$STEM.t$K.json
cd /tmp || exit 9
case "$S" in /root/.claude/projects/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/tool-results/*) ;; *) echo "REFUSED: not a tool-results file"; exit 2;; esac
[ -e "$O" ] && { echo "REFUSED: $O exists"; exit 2; }
cp --update=none "$S" "$O" && cmp "$S" "$O" || exit 3
echo "copied-equal $O"
/home/user/work/pc_study/f2/xtools_stance7_r10LTSBpc_s80/ft.sh transfer --fetch F2 --run "$RUN" --path "$P" --saved "$O"
