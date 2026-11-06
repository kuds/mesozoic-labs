#!/bin/bash
# Runs each argument as one bash command; logs it verbatim and the exit status of every pipeline element to CMDLOG.txt.
D=/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/pilot4/prelaunch/setup
LOG=$D/CMDLOG.txt
export TMPDIR=$D/tmp
export MUJOCO_GL=disable
export PYTHONDONTWRITEBYTECODE=1
export GIT_OPTIONAL_LOCKS=0
mkdir -p "$TMPDIR"
SF=$D/tmp/.status
for c in "$@"; do
  printf '$ %s\n' "$c" >> "$LOG"
  rm -f "$SF"
  bash -c "$c"$'\n''printf "%s " "${PIPESTATUS[@]}" > '"$SF"
  rc=$?
  if [ -f "$SF" ]; then st=$(cat "$SF"); else st="(no PIPESTATUS; bash exit $rc)"; fi
  printf '    exit: %s\n' "$st" >> "$LOG"
  echo "[exit: $st]"
done
