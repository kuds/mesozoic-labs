#!/bin/bash
# usage: xfer.sh RUN_ID PATH FILE_ID -- saves the earliest unused download response for FILE_ID exactly from the
# session record, then gives it to the pinned fetch tool (transfer --fetch F0 --saved)
RUN=$1; P=$2; FID=$3
STEM=${P//\//__}
cd /tmp || exit 9
SAVED=$(/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/venv/bin/python -I -B /home/user/work/pc_study/round/fetch/xtools/save_download.py --transcript /root/.claude/projects/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/subagents/workflows/wf_2814e0f0-42c/agent-a8c9f5efb2f5ce769.jsonl --file-id "$FID" --outdir /home/user/work/pc_study/round/fetch/raw/F0/$RUN --stem "$STEM" --used /home/user/work/pc_study/round/fetch/xtools/used_download_ids.txt | tail -1) || exit 3
[ -n "$SAVED" ] || exit 4
/home/user/work/pc_study/round/fetch/xtools/ft.sh transfer --fetch F0 --run "$RUN" --path "$P" --saved "$SAVED"
