# Command logger for this lane. Sourced at the start of every shell call after
#   S=$(ls -d /tmp/*/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work); T=$S/pilot4/prelaunch/tol
# L '<command>' appends the command verbatim to CMDLOG.txt, runs it, and appends
# the exit status of every element of its last pipeline (PIPESTATUS).
# One simple command (or one pipeline) per L call, so every status is recorded.
export TMPDIR="$T/tmp"
export MUJOCO_GL=disable
export PYTHONDONTWRITEBYTECODE=1
export GIT_OPTIONAL_LOCKS=0
PY="$S/pilot3/branch/venv_full/bin/python"
P4="$S/pilot4/tools"
printf '%s\n' '$ S=$(ls -d /tmp/*/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work); T=$S/pilot4/prelaunch/tol; . $T/lib.sh   [prefix of a new shell call]' >> "$T/CMDLOG.txt"
L() {
  printf '$ %s\n' "$1" >> "$T/CMDLOG.txt"
  __ps=()
  eval "$1"$'\n''__ps=("${PIPESTATUS[@]}")'
  local rc=${__ps[$((${#__ps[@]}-1))]}
  printf '[exit %s]\n' "${__ps[*]}" >> "$T/CMDLOG.txt"
  return $rc
}
