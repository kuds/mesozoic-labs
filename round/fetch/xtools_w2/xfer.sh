#!/bin/bash
# wave 2 (OFF runs): as fetch/xtools/xfer.sh, reading this fetch agent's own session record, own used-id ledger
# usage: xfer.sh RUN_ID PATH FILE_ID
RUN=$1; P=$2; FID=$3
STEM=${P//\//__}
cd /tmp || exit 9
export TMPDIR=/home/user/work/pc_study/round/fetch/tmp
SAVED=$(/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/venv/bin/python -I -B /home/user/work/pc_study/round/fetch/xtools/save_download.py --transcript /root/.claude/projects/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/subagents/workflows/wf_f9cada4f-1f5/agent-abe05a77788361bfd.jsonl --file-id "$FID" --outdir /home/user/work/pc_study/round/fetch/raw/F0/$RUN --stem "$STEM" --used /home/user/work/pc_study/round/fetch/xtools_w2/used_download_ids.txt | tail -1) || exit 3
[ -n "$SAVED" ] || exit 4
/home/user/work/pc_study/round/fetch/xtools/ft.sh transfer --fetch F0 --run "$RUN" --path "$P" --saved "$SAVED"
