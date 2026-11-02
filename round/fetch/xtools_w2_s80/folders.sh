#!/bin/bash
# wave 2 (OFF s80): saves the newest ppo folder listing pages exactly from this agent's session record, then records
# them with the pinned fetch tool (folders --ppo-id)
# usage: folders.sh PPO_ID
FID=$1
cd /tmp || exit 9
export TMPDIR=/home/user/work/pc_study/round/fetch/tmp
PAGES=$(/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/venv/bin/python -I -B /home/user/work/pc_study/round/fetch/xtools/save_listing.py --transcript /root/.claude/projects/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/subagents/workflows/wf_3dce4cff-414/agent-ad6cc67790bd98478.jsonl --folder-id "$FID" --outdir /home/user/work/pc_study/round/fetch/raw/folders --stem ppo | tail -1) || exit 3
[ -n "$PAGES" ] || exit 4
/home/user/work/pc_study/round/fetch/xtools/ft.sh folders --ppo-id "$FID" --pages $PAGES
