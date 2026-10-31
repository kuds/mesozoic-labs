#!/bin/bash
# usage: run.sh <logname> <command string>
# Logs the command, start/end times (date -u) and exit status to the notes file;
# full stdout+stderr goes to logs/<logname>.out
R=/home/user/work/phase_screen/review/rereview
name="$1"; shift
cmd="$*"
t0=$(date -u +%FT%TZ)
bash -c "$cmd" > "$R/logs/$name.out" 2>&1
rc=$?
t1=$(date -u +%FT%TZ)
{
  echo "[$t0 -> $t1] rc=$rc log=logs/$name.out"
  echo "  CMD: $cmd"
  echo "  TAIL: $(tail -n 3 "$R/logs/$name.out" | tr '\n' '|' | cut -c1-400)"
} >> "$R/notes.txt"
cat "$R/logs/$name.out"
exit $rc
