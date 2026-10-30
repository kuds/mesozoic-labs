#!/usr/bin/env bash
# V3 trial driver: the full read of the trial V2 control run and the reference read of the parent, at the same time
# (two simulation processes), nice 10, OMP 1, cwd /tmp.  Writes only under <BS>.
S2=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad
BS=$S2/r10round/bias_screen
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
(cd /tmp && exec nice -n 10 "$S2/venv/bin/python" -I -B "$BS/tools/read_bs.py" "$BS/validate/v2/control_9101") \
  > "$BS/validate/logs/v3_trial_control.log" 2>&1 &
p1=$!
(cd /tmp && exec nice -n 10 "$S2/venv/bin/python" -I -B "$BS/tools/read_bs.py" --reference) \
  > "$BS/validate/logs/v3_reference.log" 2>&1 &
p2=$!
wait $p1; echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) control read exit $?" >> "$BS/validate/logs/v3_driver.log"
wait $p2; echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) reference read exit $?" >> "$BS/validate/logs/v3_driver.log"
