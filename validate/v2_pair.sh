#!/usr/bin/env bash
# V2 driver: two no-update runs (train() a no-op) to S + 262,144, seed 9101, one per given arm, at the same time
# (at most two simulation processes), nice 10, OMP/MKL/OPENBLAS 1, cwd /tmp.  Writes only under <BS>/validate/.
set -u
S2=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad
BS=$S2/r10round/bias_screen
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
pids=()
for arm in "$@"; do
  (cd /tmp && exec nice -n 10 "$S2/venv/bin/python" -I -B "$BS/tools/run_bs.py" --arm "$arm" --seed 9101 --no-update \
     --until 1310720 --out "$BS/validate/v2") > "$BS/validate/logs/v2_$arm.log" 2>&1 &
  pids+=($!)
done
for i in "${!pids[@]}"; do
  wait "${pids[$i]}"; echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) ${*:$((i+1)):1} exit $?" >> "$BS/validate/logs/v2_driver.log"
done
