#!/usr/bin/env bash
# BS2 V3 driver (after final_driver.sh): the reader's full read of each final V2 run and the parent reference read (the
# reproduction check at the parent's F2 pairs 2,097,152 and 3,145,728), at most 3 simulation processes at once, then
# the V2 pairing comparison and the V2 summary (files only).  nice 10, OMP 1, cwd /tmp, PY -I -B.  Writes under
# <BS2>/validate/ (the reads' outputs: v2final/<run>/read/, v3final/parent_s42/; logs/), and, by read_bs.py's design,
# its loop cache <BS2>/reads/_cache/ and its TMPDIR <BS2>/tmp/read_<name>/.  Derived from BS's validate/v3_final.sh.
set -u
S2=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad
BS2=$S2/r10round/bias_screen2
PY=$S2/venv/bin/python
L=$BS2/validate/logs
SEED=9201
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$L/v3_final.log"; }
read_one() { (cd /tmp && exec nice -n 10 "$PY" -I -B "$BS2/tools/read_bs.py" "$@") ; }
log "start; read_bs $(sha256sum "$BS2/tools/read_bs.py" | cut -c1-16); load $(cat /proc/loadavg)"
# wave A: control and critic_blind (their start pair and end pair), and the parent reference (3 processes)
read_one "$BS2/validate/v2final/control_$SEED" > "$L/v3final_control.log" 2>&1 & pa=$!
read_one "$BS2/validate/v2final/critic_blind_tail3_$SEED" > "$L/v3final_critic_blind_tail3.log" 2>&1 & pb=$!
read_one --reference --out "$BS2/validate/v3final/parent_s42" > "$L/v3final_reference.log" 2>&1 & pc=$!
wait $pa; log "read control exit $?"
wait $pb; log "read critic_blind_tail3 exit $?"
wait $pc; log "reference exit $?"
# wave B: the other two arms (2 processes)
read_one "$BS2/validate/v2final/gae_lambda_1_$SEED" > "$L/v3final_gae_lambda_1.log" 2>&1 & pa=$!
read_one "$BS2/validate/v2final/frozen_obs_norm_$SEED" > "$L/v3final_frozen_obs_norm.log" 2>&1 & pb=$!
wait $pa; log "read gae_lambda_1 exit $?"
wait $pb; log "read frozen_obs_norm exit $?"
(cd /tmp && exec "$PY" -I -B "$BS2/validate/v2_compare.py" "$BS2/validate/v2final" "$SEED") > "$L/v2_compare.out" 2>&1
log "v2_compare exit $?"
(cd /tmp && exec "$PY" -I -B "$BS2/validate/v2_summary.py" "$BS2/validate/v2final" "$SEED") > "$L/v2_summary.out" 2>&1
log "v2_summary exit $?"
log "done; load $(cat /proc/loadavg)"
