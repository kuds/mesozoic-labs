#!/bin/bash
# F2 (OFF s60, session 1): as fetch/xtools_w2_s42/list_and_record.sh, for F2, reading this fetch agent's own session record
# usage: list_and_record.sh RUN_ID KEY FOLDER_ID
RUN=$1; KEY=$2; FID=$3
STEM=${KEY//\//__}
cd /tmp || exit 9
export TMPDIR=/home/user/work/pc_study/round/fetch/tmp
PAGES=$(/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/venv/bin/python -I -B /home/user/work/pc_study/round/fetch/xtools/save_listing.py --transcript /root/.claude/projects/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/subagents/workflows/wf_cd81846d-bc1/agent-ab147d00bac394e2b.jsonl --folder-id "$FID" --outdir /home/user/work/pc_study/round/fetch/raw/F2/$RUN --stem "$STEM" | tail -1) || exit 3
[ -n "$PAGES" ] || exit 4
/home/user/work/pc_study/round/fetch/xtools/ft.sh listing --fetch F2 --run "$RUN" --key "$KEY" --folder-id "$FID" --pages $PAGES
