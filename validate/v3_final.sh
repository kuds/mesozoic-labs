#!/usr/bin/env bash
# V3 driver (after final_driver.sh): the reader's full read of each final V2 run (two at a time: at most two
# simulation processes), the V2 pairing comparison (files only), and the parent reference read again (the reproduction
# check, with the truncated ledger).  nice 10, OMP 1, cwd /tmp, PY -I -B.  Writes only under <BS>.
set -u
S2=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad
BS=$S2/r10round/bias_screen
PY=$S2/venv/bin/python
L=$BS/validate/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$L/v3_final.log"; }
read_one() { (cd /tmp && exec nice -n 10 "$PY" -I -B "$BS/tools/read_bs.py" "$@") ; }
log "start; read_bs $(sha256sum "$BS/tools/read_bs.py" | cut -c1-16)"
for pair in "control critic_blind_tail3" "gae_lambda_1 frozen_obs_norm"; do
  pids=(); arms=()
  for arm in $pair; do
    read_one "$BS/validate/v2final/${arm}_9101" > "$L/v3final_$arm.log" 2>&1 &
    pids+=($!); arms+=("$arm")
  done
  for i in 0 1; do wait "${pids[$i]}"; log "read ${arms[$i]} exit $?"; done
done
(cd /tmp && exec "$PY" -I -B "$BS/validate/v2_compare.py" "$BS/validate/v2final" 9101) > "$L/v2_compare.out" 2>&1
log "v2_compare exit $?"
read_one --reference > "$L/v3final_reference.log" 2>&1
log "reference exit $?"
log "done"
