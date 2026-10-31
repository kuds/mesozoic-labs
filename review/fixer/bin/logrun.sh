#!/bin/bash
# usage: logrun.sh <logname> <cmd...> ; runs cmd with stdout/stderr to logs/<logname>.out/.err, logs to notes.txt
F=/home/user/work/phase_screen/review/fixer
name="$1"; shift
t0=$(date -u +%FT%TZ)
"$@" > "$F/logs/$name.out" 2> "$F/logs/$name.err"
rc=$?
t1=$(date -u +%FT%TZ)
{
  echo "$t0 START [$name] $*"
  echo "$t1 END   [$name] exit $rc; stdout $(wc -l < "$F/logs/$name.out") lines, stderr $(wc -l < "$F/logs/$name.err") lines -> logs/$name.out/.err"
} >> "$F/notes.txt"
echo "exit $rc"
exit $rc
