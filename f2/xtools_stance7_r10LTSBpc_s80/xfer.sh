#!/bin/bash
# F2 (pc_s80): saves the download_file_content response for FILE_ID made at or after T0 exactly from the session record
# (extract_response.py), then records it with the pinned fetch tool (transfer --saved)
# usage: xfer.sh PATH FILE_ID T0 K
RUN=stance7_r10LTSBpc_s80; P=$1; FID=$2; T0=$3; K=$4
STEM=${P//\//__}
OUT=/home/user/work/pc_study/round/fetch/raw/F2/$RUN/$STEM.t$K.json
cd /tmp || exit 9
export TMPDIR=/home/user/work/pc_study/round/fetch/tmp
/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/venv/bin/python -I /tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/f2/xfer/extract_response.py --transcripts /root/.claude/projects/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/subagents/workflows --file-id "$FID" --after-utc "$T0" --out "$OUT" || exit 3
/home/user/work/pc_study/f2/xtools_stance7_r10LTSBpc_s80/ft.sh transfer --fetch F2 --run "$RUN" --path "$P" --saved "$OUT"
