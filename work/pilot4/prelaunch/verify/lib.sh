# Command logger for the verify lane. Sourced at the start of every shell call as
#   S=$(ls -d /tmp/*/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work); K=$S/pilot4/prelaunch/verify; . $K/lib.sh
# L '<command>' appends the command verbatim to CMDLOG.txt, runs it, and appends the exit status of every element
# of its pipeline (PIPESTATUS). One simple command (or one pipeline) per L call, so every status is recorded.
# N '<text>' appends a [note] line (file-tool writes, read-only tool use, anything not run through L).
export TMPDIR="$K/tmp"
export MPLCONFIGDIR="$K/tmp/mpl" XDG_CACHE_HOME="$K/tmp/xdg"
export MUJOCO_GL=disable
export PYTHONDONTWRITEBYTECODE=1
export GIT_OPTIONAL_LOCKS=0
unset PYTHONPATH
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$K/logs" "$K/data" "$K/scripts"
PY="$S/pilot3/branch/venv_full/bin/python"
P4="$S/pilot4/tools"
P3="$S/pilot3"
PL="$S/pilot4/prelaunch"
printf '%s\n' "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] \$ S=\$(ls -d /tmp/*/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work); K=\$S/pilot4/prelaunch/verify; . \$K/lib.sh   [prefix of a new shell call]" >> "$K/CMDLOG.txt"
L() {
  printf '[%s] $ %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" >> "$K/CMDLOG.txt"
  __ps=()
  eval "$1"$'\n''__ps=("${PIPESTATUS[@]}")'
  local rc=${__ps[$((${#__ps[@]}-1))]}
  printf '[exit %s]\n' "${__ps[*]}" >> "$K/CMDLOG.txt"
  return $rc
}
N() { printf '[%s] [note] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" >> "$K/CMDLOG.txt"; }
