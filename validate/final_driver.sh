#!/usr/bin/env bash
# Final validation driver (no optimizer step anywhere): V1 for the four arms (one at a time), then V2 in two pairs
# (at most two simulation processes), then CHECK on each V2 run.  nice 10, OMP/MKL/OPENBLAS 1, cwd /tmp, PY -I -B.
# Every output is under <BS>/validate/ (v1final/, v2final/, logs/).
set -u
S2=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad
BS=$S2/r10round/bias_screen
PY=$S2/venv/bin/python
L=$BS/validate/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$L/final_driver.log"; }
log "start; tools: $(sha256sum "$BS"/tools/run_bs.py "$BS"/tools/bs_hooks.py "$BS"/tools/bs_common.py | cut -c1-16 | tr '\n' ' ')"
for arm in control critic_blind_tail3 gae_lambda_1 frozen_obs_norm; do
  (cd /tmp && exec nice -n 10 "$PY" -I -B "$BS/tools/run_bs.py" --arm "$arm" --seed 9101 --save-loaded-only \
     --out "$BS/validate/v1final") > "$L/v1final_$arm.log" 2>&1
  log "V1 $arm exit $?"
done
for pair in "control critic_blind_tail3" "gae_lambda_1 frozen_obs_norm"; do
  pids=(); arms=()
  for arm in $pair; do
    (cd /tmp && exec nice -n 10 "$PY" -I -B "$BS/tools/run_bs.py" --arm "$arm" --seed 9101 --no-update \
       --until 1310720 --out "$BS/validate/v2final") > "$L/v2final_$arm.log" 2>&1 &
    pids+=($!); arms+=("$arm")
    log "V2 $arm started"
  done
  for i in 0 1; do
    wait "${pids[$i]}"; rc=$?
    log "V2 ${arms[$i]} exit $rc"
    (cd /tmp && exec "$PY" -I -B "$BS/tools/read_bs.py" --check "$BS/validate/v2final/${arms[$i]}_9101") \
      > "$L/v2final_check_${arms[$i]}.log" 2>&1
    log "CHECK ${arms[$i]} exit $?"
  done
done
log "done"
