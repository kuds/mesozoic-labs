#!/usr/bin/env bash
# PS V2 driver (no optimizer step anywhere): four no-update runs at once (the screen's slot count, so the timing is
# measured at its concurrency): control and critic_phase at seeds 9401 and 9411 (--no-update --until
# bs_common.VALIDATE_UNTIL = 2,359,296), then CHECK (read_bs.py --check) on each.  nice 10, OMP/MKL/OPENBLAS 1,
# cwd /tmp, PY -I -B.  Outputs under <PS>/validate/v2/ and validate/logs/; run_bs.py also uses <PS>/tmp/<mode>_<name>/
# as its TMPDIR (its design).  Derived from BS2's validate/final_driver.sh.
set -u
S2=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad
PS=/home/user/work/phase_screen
PY=$S2/venv/bin/python
L=$PS/validate/logs
UNTIL=$("$PY" -I -B -c "import importlib.util,sys; s=importlib.util.spec_from_file_location('c',sys.argv[1]); c=importlib.util.module_from_spec(s); s.loader.exec_module(c); print(c.VALIDATE_UNTIL)" "$PS/tools/bs_common.py")
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$L/v2_driver.log"; }
log "start; until $UNTIL; tools: $(sha256sum "$PS"/tools/run_bs.py "$PS"/tools/bs_hooks.py "$PS"/tools/bs_common.py "$PS"/tools/read_bs.py | cut -c1-16 | tr '\n' ' ')"
log "load: $(cat /proc/loadavg)"
pids=(); names=()
for seed in 9401 9411; do
  for arm in control critic_phase; do
    (cd /tmp && exec nice -n 10 "$PY" -I -B "$PS/tools/run_bs.py" --arm "$arm" --seed "$seed" --no-update \
       --until "$UNTIL" --out "$PS/validate/v2") > "$L/v2_${arm}_$seed.log" 2>&1 &
    pids+=($!); names+=("${arm}_$seed")
    log "V2 ${arm}_$seed started"
  done
done
log "load after start: $(cat /proc/loadavg)"
for i in 0 1 2 3; do
  wait "${pids[$i]}"; rc=$?
  log "V2 ${names[$i]} exit $rc"
done
for name in "${names[@]}"; do
  (cd /tmp && exec "$PY" -I -B "$PS/tools/read_bs.py" --check "$PS/validate/v2/$name") > "$L/v2_check_$name.log" 2>&1
  log "CHECK $name exit $?"
done
log "done; load $(cat /proc/loadavg)"
