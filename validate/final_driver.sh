#!/usr/bin/env bash
# BS2 final validation driver (no optimizer step anywhere): V1 for the four arms (one at a time, --save-loaded-only),
# then V2 for the four arms at once (--no-update --until bs_common.VALIDATE_UNTIL; 4 processes, the screen's own slot
# count, so the timing is measured at the screen's concurrency), then CHECK (read_bs.py --check) on each V2 run.
# nice 10, OMP/MKL/OPENBLAS 1, cwd /tmp, PY -I -B.  Every output is under <BS2>/validate/ (v1final/, v2final/, logs/);
# run_bs.py itself also uses <BS2>/tmp/<mode>_<name>/ as its TMPDIR (its design, unchanged from BS).
# Derived from BS's validate/final_driver.sh (seed 9101, --until 1310720, V2 in two pairs).
set -u
S2=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad
BS2=$S2/r10round/bias_screen2
PY=$S2/venv/bin/python
L=$BS2/validate/logs
SEED=9201
UNTIL=$("$PY" -I -B -c "import importlib.util,sys; s=importlib.util.spec_from_file_location('c',sys.argv[1]); c=importlib.util.module_from_spec(s); s.loader.exec_module(c); print(c.VALIDATE_UNTIL)" "$BS2/tools/bs_common.py")
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$L/final_driver.log"; }
log "start; until $UNTIL; tools: $(sha256sum "$BS2"/tools/run_bs.py "$BS2"/tools/bs_hooks.py "$BS2"/tools/bs_common.py "$BS2"/tools/read_bs.py | cut -c1-16 | tr '\n' ' ')"
log "load: $(cat /proc/loadavg)"
for arm in control critic_blind_tail3 gae_lambda_1 frozen_obs_norm; do
  (cd /tmp && exec nice -n 10 "$PY" -I -B "$BS2/tools/run_bs.py" --arm "$arm" --seed "$SEED" --save-loaded-only \
     --out "$BS2/validate/v1final") > "$L/v1final_$arm.log" 2>&1
  log "V1 $arm exit $?"
done
pids=(); arms=()
for arm in control critic_blind_tail3 gae_lambda_1 frozen_obs_norm; do
  (cd /tmp && exec nice -n 10 "$PY" -I -B "$BS2/tools/run_bs.py" --arm "$arm" --seed "$SEED" --no-update \
     --until "$UNTIL" --out "$BS2/validate/v2final") > "$L/v2final_$arm.log" 2>&1 &
  pids+=($!); arms+=("$arm")
  log "V2 $arm started"
done
log "load after start: $(cat /proc/loadavg)"
for i in 0 1 2 3; do
  wait "${pids[$i]}"; rc=$?
  log "V2 ${arms[$i]} exit $rc"
done
for arm in "${arms[@]}"; do
  (cd /tmp && exec "$PY" -I -B "$BS2/tools/read_bs.py" --check "$BS2/validate/v2final/${arm}_$SEED") \
    > "$L/v2final_check_$arm.log" 2>&1
  log "CHECK $arm exit $?"
done
log "done"
