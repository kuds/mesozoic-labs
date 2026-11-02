#!/bin/bash
# F2 (OFF s42, session 1): the maintainer's exact-copy transfer method (stance6 RUNLOG line 709; COLAB_STEPS F2):
# extract_response.py copies the inline response from the session record (newest at or after T0) to a new file
# <PATH with / as __>.t<K>.json, then the pinned fetch tool records it with --saved.
# usage: xfer.sh PATH FILE_ID T0 K
RUN=stance7_r10LTSBoff_s42; P=$1; FID=$2; T0=$3; K=$4
STEM=${P//\//__}
OUT=/home/user/work/pc_study/round/fetch/raw/F2/$RUN/$STEM.t$K.json
cd /tmp || exit 9
export TMPDIR=/home/user/work/pc_study/round/fetch/tmp
/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/venv/bin/python -I /tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/f2/xfer/extract_response.py --transcripts /root/.claude/projects/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/subagents/workflows --file-id "$FID" --after-utc "$T0" --out "$OUT" || exit 3
/home/user/work/pc_study/round/fetch/xtools/ft.sh transfer --fetch F2 --run "$RUN" --path "$P" --saved "$OUT"
