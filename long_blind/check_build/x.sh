#!/bin/bash
# usage: x.sh LABEL 'command string'   -- runs the command with bash -c, logs times, exit code and output to check_build
CB=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/long_blind/check_build
label="$1"; shift
cmd="$*"
n=$(ls "$CB/logs" | wc -l)
out="$CB/logs/$(printf '%03d' "$n")_${label}.out"
t0=$(date -u +%FT%TZ)
bash -c "$cmd" > "$out" 2>&1
rc=$?
t1=$(date -u +%FT%TZ)
{
  echo "=== [$t0 -> $t1] label=$label rc=$rc out=$out"
  echo "CMD: $cmd"
  echo "RESULT (first 15 lines):"
  head -n 15 "$out" | sed 's/^/  | /'
} >> "$CB/NOTES.txt"
cat "$out"
echo "[x.sh] rc=$rc out=$out"
exit $rc
