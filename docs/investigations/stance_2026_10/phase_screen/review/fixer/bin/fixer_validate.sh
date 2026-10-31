#!/usr/bin/env bash
# Fixer re-validation after the run_bs.py change (P-B1, P-i3); no optimizer step anywhere (V1: no learn(); V2:
# train() a no-op).  V1 both arms at 9401, then V2 four no-update runs at once (control / critic_phase at 9401 and
# 9411, until bs_common.VALIDATE_UNTIL, as the builder's v2_driver.sh), CHECK on each, then V3 (the reader's full
# read of each V2 run and the reference read, at most 3 at once), then the builder's v1/v2/v3 check scripts on these
# outputs.  nice 10, OMP/MKL/OPENBLAS 1, cwd /tmp, PY -I -B.  Outputs: <PS>/validate/fixer/{v1,v2,v3} (run_bs.py
# allows the no-optimizer-step modes only under <PS>/validate), logs: review/fixer/logs/val/.  run_bs.py / read_bs.py
# use <PS>/tmp/<mode>_<name>/ and <PS>/tmp/read_<name>/ as TMPDIR, and the reader its cache <PS>/reads/_cache/ (their
# design).
set -u
S2=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad
PS=/home/user/work/phase_screen
PY=$S2/venv/bin/python
F=$PS/review/fixer
L=$F/logs/val
O=$PS/validate/fixer
mkdir -p "$O"
UNTIL=$("$PY" -I -B -c "import importlib.util,sys; s=importlib.util.spec_from_file_location('c',sys.argv[1]); c=importlib.util.module_from_spec(s); s.loader.exec_module(c); print(c.VALIDATE_UNTIL)" "$PS/tools/bs_common.py")
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$L/driver.log"; }
run_bs() { (cd /tmp && exec nice -n 10 "$PY" -I -B "$PS/tools/run_bs.py" "$@"); }
read_one() { (cd /tmp && exec nice -n 10 "$PY" -I -B "$PS/tools/read_bs.py" "$@"); }
log "start; until $UNTIL; tools: $(sha256sum "$PS"/tools/run_bs.py "$PS"/tools/bs_hooks.py "$PS"/tools/bs_common.py "$PS"/tools/read_bs.py | cut -c1-16 | tr '\n' ' ')"
log "load: $(cat /proc/loadavg)"
# ---- V1
for arm in control critic_phase; do
  run_bs --arm "$arm" --seed 9401 --save-loaded-only --out "$O/v1" > "$L/v1_$arm.log" 2>&1 &
done
wait
log "V1 done: $(grep -h saved_loaded_only "$L"/v1_*.log | tr '\n' ' ')"
# ---- V2
pids=(); names=()
for seed in 9401 9411; do
  for arm in control critic_phase; do
    run_bs --arm "$arm" --seed "$seed" --no-update --until "$UNTIL" --out "$O/v2" > "$L/v2_${arm}_$seed.log" 2>&1 &
    pids+=($!); names+=("${arm}_$seed")
    log "V2 ${arm}_$seed started"
  done
done
for i in 0 1 2 3; do
  wait "${pids[$i]}"; rc=$?
  log "V2 ${names[$i]} exit $rc"
done
for name in "${names[@]}"; do
  (cd /tmp && exec "$PY" -I -B "$PS/tools/read_bs.py" --check "$O/v2/$name") > "$L/v2_check_$name.log" 2>&1
  log "CHECK $name exit $?"
done
# ---- V3
read_one "$O/v2/control_9401" > "$L/v3_control_9401.log" 2>&1 & pa=$!
read_one "$O/v2/critic_phase_9401" > "$L/v3_critic_phase_9401.log" 2>&1 & pb=$!
read_one --reference --out "$O/v3/parent_s42" > "$L/v3_reference.log" 2>&1 & pc=$!
wait $pa; log "read control_9401 exit $?"
wait $pb; log "read critic_phase_9401 exit $?"
wait $pc; log "reference exit $?"
read_one "$O/v2/control_9411" > "$L/v3_control_9411.log" 2>&1 & pa=$!
read_one "$O/v2/critic_phase_9411" > "$L/v3_critic_phase_9411.log" 2>&1 & pb=$!
wait $pa; log "read control_9411 exit $?"
wait $pb; log "read critic_phase_9411 exit $?"
# ---- the builder's check scripts on these outputs (files only)
(cd /tmp && exec "$PY" -I -B "$PS/validate/v1_independent.py" "$O/v1" 9401) > "$L/v1_independent.out" 2>&1
log "v1_independent exit $?"
(cd /tmp && exec "$PY" -I -B "$PS/validate/v2_check.py" "$O/v2" 9401 9411) > "$L/v2_check.out" 2>&1
log "v2_check exit $?"
(cd /tmp && exec "$PY" -I -B "$PS/validate/v3_check.py" "$O") > "$L/v3_check.out" 2>&1
log "v3_check exit $?"
log "done; load $(cat /proc/loadavg)"
