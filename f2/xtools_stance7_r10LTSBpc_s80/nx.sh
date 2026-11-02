#!/bin/bash
# F2 (pc_s80): run `next`, then print T0 (date -u) and, for a transfer action, the next transfer number K of its path
D=/home/user/work/pc_study/f2/xtools_stance7_r10LTSBpc_s80
cd /tmp || exit 9
OUT=$($D/ft.sh next --fetch F2 --run stance7_r10LTSBpc_s80 | grep -v "^r10round lane B")
echo "$OUT"
P=$(echo "$OUT" | sed -n 's/^ *transfer \([^ ]*\) (.*/\1/p' | head -1)
echo "T0=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
[ -n "$P" ] && echo "K=$(/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/venv/bin/python -I -B $D/nextk.py "$P")"
exit 0
