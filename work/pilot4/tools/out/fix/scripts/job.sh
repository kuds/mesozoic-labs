#!/bin/bash
# fix lane job wrapper (copied from the e2e lane; an ERR trap logs every inner nonzero status): bash job.sh NAME 'COMMAND'
# Logs the command verbatim to CMDLOG.txt (tagged with NAME), runs it with the lane's environment, logs every
# pipeline element's exit status to CMDLOG.txt and "NAME rc=N" to e2e/status.txt, and exits with the last status.
# The environment: TMPDIR, MPLCONFIGDIR and XDG_CACHE_HOME inside the lane; MUJOCO_GL=disable; no bytecode;
# OMP_NUM_THREADS=1 (as pilot 3's judge ran); no git optional locks; PYTHONPATH unset.
LANE=/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/pilot4/tools/out/fix
LOG="$LANE/CMDLOG.txt"
NAME="$1"
CMD="$2"
export TMPDIR="$LANE/tmp" MUJOCO_GL=disable PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 GIT_OPTIONAL_LOCKS=0
export MPLCONFIGDIR="$LANE/tmp/mpl" XDG_CACHE_HOME="$LANE/tmp/xdg" MESA_SHADER_CACHE_DISABLE=true
unset PYTHONPATH
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$LANE/e2e"
printf '[%s job %s] $ %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$NAME" "$CMD" >> "$LOG"
start=$(date +%s)
set -E; trap '__s=$?; printf "    inner nonzero: status %d from: %s\n" "$__s" "$BASH_COMMAND" >> "$LOG"' ERR
eval "$CMD"
ps=("${PIPESTATUS[@]}")
end=$(date +%s)
printf '[%s job %s] exit: %s (%ss)\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$NAME" "${ps[*]}" "$((end - start))" >> "$LOG"
echo "$NAME rc=${ps[-1]} elapsed_s=$((end - start))" >> "$LANE/e2e/status.txt"
exit "${ps[-1]}"
