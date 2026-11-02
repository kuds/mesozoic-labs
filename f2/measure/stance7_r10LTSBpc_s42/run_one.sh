#!/bin/bash
# run_one.sh TAG CMD...   (cwd /tmp; stdout -> <M>/TAG.out, stderr -> <M>/TAG.err, exit -> <M>/TAG.exit)
# Appends one line to <M>/cmdlog.txt: start (date -u), rc, wall seconds, end, tag, the command.
M=/home/user/work/pc_study/f2/measure/stance7_r10LTSBpc_s42
TAG="$1"; shift
cd /tmp || exit 99
S=$(date -u +%Y-%m-%dT%H:%M:%SZ); s0=$(date -u +%s)
echo "$S" > "$M/$TAG.start"
"$@" > "$M/$TAG.out" 2> "$M/$TAG.err"
rc=$?
E=$(date -u +%Y-%m-%dT%H:%M:%SZ); s1=$(date -u +%s)
echo "$rc" > "$M/$TAG.exit"
echo "start=$S end=$E wall=$((s1-s0))s rc=$rc" > "$M/$TAG.time"
echo "$S rc=$rc wall=$((s1-s0))s end=$E [$TAG] CMD: $* > $M/$TAG.out 2> $M/$TAG.err" >> "$M/cmdlog.txt"
exit $rc
