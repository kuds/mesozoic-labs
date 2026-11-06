#!/bin/bash
# Smoke lane job wrapper (the tolerance lane's job.sh with this lane's paths): bash job.sh NAME 'COMMAND'
# Logs the command verbatim to CMDLOG.txt (tagged with NAME), runs it with the lane's environment, logs every pipeline
# element's exit status (and, through an ERR trap, every inner nonzero status) to CMDLOG.txt and
# "NAME rc=N elapsed_s=S" to $JOB_STATUS (default judge/status.txt), and exits with the last status. The environment:
# TMPDIR, MPLCONFIGDIR and XDG_CACHE_HOME inside the lane; MUJOCO_GL=disable unless the caller set MUJOCO_GL; no
# bytecode; OMP_NUM_THREADS=1; no git optional locks; PYTHONPATH unset.
W=$(ls -d /tmp/*/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work)
LANE=$W/pilot4/prelaunch/smoke
LOG="$LANE/CMDLOG.txt"
NAME="$1"
CMD="$2"
STATUS="${JOB_STATUS:-$LANE/judge/status.txt}"
export TMPDIR="$LANE/tmp" MUJOCO_GL="${MUJOCO_GL:-disable}" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 GIT_OPTIONAL_LOCKS=0
export MPLCONFIGDIR="$LANE/tmp/mpl" XDG_CACHE_HOME="$LANE/tmp/xdg" MESA_SHADER_CACHE_DISABLE=true
unset PYTHONPATH
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$(dirname "$STATUS")"
printf '[%s job %s] $ %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$NAME" "$CMD" >> "$LOG"
start=$(date +%s)
set -E; trap '__s=$?; printf "    inner nonzero: status %d from: %s\n" "$__s" "$BASH_COMMAND" >> "$LOG"' ERR
eval "$CMD"
ps=("${PIPESTATUS[@]}")
end=$(date +%s)
printf '[%s job %s] exit: %s (%ss)\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$NAME" "${ps[*]}" "$((end - start))" >> "$LOG"
echo "$NAME rc=${ps[-1]} elapsed_s=$((end - start))" >> "$STATUS"
exit "${ps[-1]}"
