#!/bin/bash
# F2 (pc_s60): saves the newest listing pages of one folder exactly from this agent's session record, then records
# them with the pinned fetch tool (listing)
# usage: list_and_record.sh KEY FOLDER_ID
RUN=stance7_r10LTSBpc_s60; KEY=$1; FID=$2
STEM=${KEY//\//__}
cd /tmp || exit 9
export TMPDIR=/home/user/work/pc_study/round/fetch/tmp
PAGES=$(/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/venv/bin/python -I -B /home/user/work/pc_study/round/fetch/xtools/save_listing.py --transcript /root/.claude/projects/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/subagents/workflows/wf_cd81846d-bc1/agent-afd38b7ffcd3efece.jsonl --folder-id "$FID" --outdir /home/user/work/pc_study/round/fetch/raw/F2/$RUN --stem "$STEM" | tail -1) || exit 3
[ -n "$PAGES" ] || exit 4
/home/user/work/pc_study/f2/xtools_stance7_r10LTSBpc_s60/ft.sh listing --fetch F2 --run "$RUN" --key "$KEY" --folder-id "$FID" --pages $PAGES
