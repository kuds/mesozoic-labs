#!/bin/bash
# usage: runrec.sh <name> <cmd...>  ; runs cmd from /tmp with stdout -> <D>/<name>.out, stderr -> <D>/<name>.err,
# exit code -> <D>/<name>.exit; appends start/end UTC, command, exit code, wall seconds to <D>/cmdlog.txt
D=/home/user/work/pc_study/f2/measure/stance7_r10LTSBoff_s60
name="$1"; shift
cd /tmp || exit 99
s_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ); s=$(date -u +%s)
"$@" > "$D/$name.out" 2> "$D/$name.err"
rc=$?
e_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ); e=$(date -u +%s)
echo "$rc" > "$D/$name.exit"
printf '%s  start %s  end %s  exit %s  wall %ss  cmd: %s  (stdout %s.out, stderr %s.err)\n' "$name" "$s_utc" "$e_utc" "$rc" "$((e-s))" "$*" "$name" "$name" >> "$D/cmdlog.txt"
exit $rc
