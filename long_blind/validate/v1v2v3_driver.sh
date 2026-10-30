#!/usr/bin/env bash
# LB validation driver A (no optimizer step anywhere): V1 (the four arms one at a time, --save-loaded-only, seed 9301),
# then V2 (the four arms, --no-update --until bs_common.VALIDATE_UNTIL, seed 9301, ONE AT A TIME: driver B runs V4 beside
# it and the measurement builder's validation holds two more slots, so this workflow stays at 4 processes machine-wide),
# CHECK on each V2 run, then V3 (the reader's full read of each V2 run and the parent reference read restricted to the
# REPRO pairs 2,097,152 and 3,145,728; one at a time).
# nice 10, OMP/MKL/OPENBLAS 1, cwd /tmp, PY -I -B.  Writes only under <LB>/validate/ (v1/, v2/, v3/, logs/) and, by the
# tools' design, <LB>/tmp/ (TMPDIRs) and <LB>/reads/_cache/ (the reader's loop cache).  Derived from BS2's
# validate/final_driver.sh and v3_final.sh.
set -u
S2=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad
LB=$S2/r10round/long_blind
PY=$S2/venv/bin/python
L=$LB/validate/logs
SEED=9301
UNTIL=$("$PY" -I -B -c "import importlib.util,sys; s=importlib.util.spec_from_file_location('c',sys.argv[1]); c=importlib.util.module_from_spec(s); s.loader.exec_module(c); print(c.VALIDATE_UNTIL)" "$LB/tools/bs_common.py")
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$L/driverA.log"; }
log "start; until $UNTIL; tools: $(cd "$LB/tools" && sha256sum run_bs.py bs_hooks.py bs_common.py bs_resume.py read_bs.py | cut -c1-16 | tr '\n' ' ')"
log "load: $(cat /proc/loadavg)"
for arm in control critic_blind_tail3 gae_lambda_1 frozen_obs_norm; do
  (cd /tmp && exec nice -n 10 "$PY" -I -B "$LB/tools/run_bs.py" --arm "$arm" --seed "$SEED" --save-loaded-only \
     --out "$LB/validate/v1") > "$L/v1_$arm.log" 2>&1
  log "V1 $arm exit $?"
done
v2() { (cd /tmp && exec nice -n 10 "$PY" -I -B "$LB/tools/run_bs.py" --arm "$1" --seed "$SEED" --no-update \
          --until "$UNTIL" --out "$LB/validate/v2") > "$L/v2_$1.log" 2>&1; }
for arm in control critic_blind_tail3 gae_lambda_1 frozen_obs_norm; do
  log "V2 $arm start; load $(cat /proc/loadavg)"
  v2 "$arm"; log "V2 $arm exit $?"
done
for arm in control critic_blind_tail3 gae_lambda_1 frozen_obs_norm; do
  (cd /tmp && exec "$PY" -I -B "$LB/tools/read_bs.py" --check "$LB/validate/v2/${arm}_$SEED") > "$L/v2_check_$arm.log" 2>&1
  log "CHECK $arm exit $?"
done
read_one() { (cd /tmp && exec nice -n 10 "$PY" -I -B "$LB/tools/read_bs.py" "$@") ; }
read_one --reference --pairs 2097152,3145728 --out "$LB/validate/v3/parent_s42" > "$L/v3_reference.log" 2>&1
log "reference exit $?"
for arm in control critic_blind_tail3 gae_lambda_1 frozen_obs_norm; do
  read_one "$LB/validate/v2/${arm}_$SEED" > "$L/v3_$arm.log" 2>&1; log "read $arm exit $?"
done
log "done; load $(cat /proc/loadavg)"
