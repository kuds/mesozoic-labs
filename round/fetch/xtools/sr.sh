#!/bin/bash
# session-record extractor wrapper (copies a Drive tool result exactly as the session's record holds it)
cd /tmp || exit 9
exec /tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/venv/bin/python -I -B /home/user/work/pc_study/round/fetch/xtools/session_record.py "$1" --transcript /root/.claude/projects/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/subagents/workflows/wf_2814e0f0-42c/agent-a8c9f5efb2f5ce769.jsonl "${@:2}"
