#!/usr/bin/env bash
# PS V3 driver (after v2_driver.sh): the reader's full read of each V2 run and the parent reference read (the
# reproduction check at the parent's F2 pairs 2,097,152 and 3,145,728), at most 3 simulation processes at once, then
# v3_check.py (files only).  nice 10, OMP 1, cwd /tmp, PY -I -B.  Writes under <PS>/validate/ (v2/<run>/read/,
# v3/parent_s42/, logs/) and, by read_bs.py's design, its loop cache <PS>/reads/_cache/ and its TMPDIR
# <PS>/tmp/read_<name>/.  Derived from BS2's validate/v3_final.sh.
set -u
S2=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad
PS=/home/user/work/phase_screen
PY=$S2/venv/bin/python
L=$PS/validate/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$L/v3_driver.log"; }
read_one() { (cd /tmp && exec nice -n 10 "$PY" -I -B "$PS/tools/read_bs.py" "$@") ; }
log "start; read_bs $(sha256sum "$PS/tools/read_bs.py" | cut -c1-16); load $(cat /proc/loadavg)"
read_one "$PS/validate/v2/control_9401" > "$L/v3_control_9401.log" 2>&1 & pa=$!
read_one "$PS/validate/v2/critic_phase_9401" > "$L/v3_critic_phase_9401.log" 2>&1 & pb=$!
read_one --reference --out "$PS/validate/v3/parent_s42" > "$L/v3_reference.log" 2>&1 & pc=$!
wait $pa; log "read control_9401 exit $?"
wait $pb; log "read critic_phase_9401 exit $?"
wait $pc; log "reference exit $?"
read_one "$PS/validate/v2/control_9411" > "$L/v3_control_9411.log" 2>&1 & pa=$!
read_one "$PS/validate/v2/critic_phase_9411" > "$L/v3_critic_phase_9411.log" 2>&1 & pb=$!
wait $pa; log "read control_9411 exit $?"
wait $pb; log "read critic_phase_9411 exit $?"
(cd /tmp && exec "$PY" -I -B "$PS/validate/v3_check.py" "$PS/validate") > "$L/v3_check.out" 2>&1
log "v3_check exit $?"
log "done; load $(cat /proc/loadavg)"
