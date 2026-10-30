#!/bin/bash
# Two lanes (at most 2 simulation processes), nice 10.  Each lane runs its conditions in series.
D=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/com_revisit/notes/termination_probe
PY=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/venv/bin/python
SEEDS=7001:7129
lane() {
  local name=$1; shift
  for spec in "$@"; do
    set -- $spec
    echo "$(date -u +%H:%M:%S) start $spec" >> $D/runs/lane_$name.log
    nice -n 10 $PY -I -B $D/run_probe.py --label $1 --seeds $SEEDS --out $D/runs ${@:2} >> $D/runs/lane_$name.log 2>> $D/runs/lane_$name.err
    echo "$(date -u +%H:%M:%S) exit $? $spec" >> $D/runs/lane_$name.log
  done
}
mkdir -p $D/runs
lane A "p1048576" "c2511_3145728" "c2511_E" "b2511_E" "p2097152 --hold 0 --lite" &
lane B "p2097152" "c2531_3145728" "c2531_E" "p2097152 --hold -1 --lite" "p2097152 --hold 1 --lite" &
wait
echo "$(date -u +%H:%M:%S) all done" >> $D/runs/done.txt
