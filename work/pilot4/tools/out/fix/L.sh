#!/bin/bash
# Command logger of the fix lane. Each argument is one command, run in its own bash -c and
# logged verbatim. Logged: the command's exit status, the statuses of its last pipeline, and
# (through an ERR trap with errtrace) every simple command or pipeline inside it that exited
# nonzero outside a conditional, with its status. A command not listed as nonzero exited 0.
L=/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/pilot4/tools/out/fix
LOG=$L/CMDLOG.txt
export TMPDIR=$L/tmp MUJOCO_GL=disable PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0
export PSF=$L/tmp/.pipestatus.$$ __LOG=$LOG
rc_all=0
for c in "$@"; do
  n=$(grep -c '^\[' "$LOG")
  printf '[%d] %s %s\n' "$n" "$(date -u +%T)" "$c" >> "$LOG"
  rm -f "$PSF"
  bash -E -c 'trap '"'"'__s=$?; printf "    inner nonzero: status %d from: %s\n" "$__s" "$BASH_COMMAND" >> "$__LOG"'"'"' ERR'$'\n'"$c"$'\n''__p=("${PIPESTATUS[@]}"); echo "${__p[*]}" > "$PSF"; exit "${__p[-1]}"'
  rc=$?
  ps="(not reached: command exited early)"
  [ -f "$PSF" ] && ps=$(cat "$PSF")
  rm -f "$PSF"
  printf '    -> status %d (last pipeline statuses: %s)\n' "$rc" "$ps" >> "$LOG"
  echo "[rc=$rc ps=$ps]"
  [ $rc -ne 0 ] && rc_all=$rc
done
exit $rc_all
