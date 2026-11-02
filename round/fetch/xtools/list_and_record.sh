#!/bin/bash
# usage: list_and_record.sh RUN_ID KEY FOLDER_ID  -- saves the newest listing pages exactly from the session record,
# then records them with the pinned fetch tool (listing --fetch F0)
RUN=$1; KEY=$2; FID=$3
STEM=${KEY//\//__}
cd /tmp || exit 9
PAGES=$(/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/venv/bin/python -I -B /home/user/work/pc_study/round/fetch/xtools/save_listing.py --transcript /root/.claude/projects/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/subagents/workflows/wf_2814e0f0-42c/agent-a8c9f5efb2f5ce769.jsonl --folder-id "$FID" --outdir /home/user/work/pc_study/round/fetch/raw/F0/$RUN --stem "$STEM" | tail -1) || exit 3
[ -n "$PAGES" ] || exit 4
/home/user/work/pc_study/round/fetch/xtools/ft.sh listing --fetch F0 --run "$RUN" --key "$KEY" --folder-id "$FID" --pages $PAGES
