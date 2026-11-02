#!/bin/bash
# usage: runrec.sh NAME OUTFILE CMD...   (cwd /tmp; stdout -> OUTFILE, stderr -> NAME.err, exit -> NAME.exit; one line to cmdlog.txt)
D=/home/user/work/pc_study/f2/measure/stance7_r10LTSBoff_s80
NAME="$1"; shift
OUT="$1"; shift
cd /tmp || exit 99
S=$(date -u +%Y-%m-%dT%H:%M:%SZ); S0=$(date -u +%s)
"$@" > "$OUT" 2> "$D/$NAME.err"
RC=$?
E=$(date -u +%Y-%m-%dT%H:%M:%SZ); E0=$(date -u +%s)
echo "$RC" > "$D/$NAME.exit"
echo "$NAME  start $S  end $E  exit $RC  wall $((E0-S0))s  cmd: $*  (stdout $(basename "$OUT"), stderr $NAME.err)" >> "$D/cmdlog.txt"
exit $RC
