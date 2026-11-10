#!/usr/bin/env bash
# Stage 1b arm K (STAGE1B_PLAN.md section 3, "The K bodies"; ARM_K.md): Stage 1's prepare (plant, references,
# transitions) and gates G1-G4 on the two K bodies, after their sizing (python -m designlab.refine --cases
# narrow_knee_60) and the carpet-lift pass that sets their carpet swing (refs.CARPET_LIFT).
#
# Run from hardware/trex_form_study:  [PY=PYTHON] [LOG=DIR] bash stage1/runs/2026-11-10-speed/run_arm_k.sh [COMMAND]
#   check          every git-ignored input arm K reads (arm_k_inputs.json: the case's 4 round-two model files and the
#                  K bodies' 56 reference and transition arrays) is here with its listed sha256; exit status 1 if not
#   copy-from DIR  copy those inputs from another checkout DIR (its root or its hardware/trex_form_study), each checked
#                  by its sha256; a file already here and right is left alone
#   run (default)  every product, each skipped only when it is here in full: the case's models (regenerated from the
#                  records if missing, about 3 min), plant.json, refs.json and transitions.json with every array at
#                  its recorded sha256 (missing or wrong arrays are regenerated in a scratch directory and put in
#                  place only if they match the record, which is never rewritten), then each gate gates.json lacks
#   g1             G1 again on both K bodies, alone and with one process per core as its rule asks, replacing their G1
#                  records: run once after the merge with the machine otherwise idle (ARM_K.md)
# The CLI's output goes to LOG (default stage1/runs/2026-11-10-speed/logs, not committed).  At most 3 processes at a
# time (PYTHON_CPU_COUNT=3: every pool of the prepare and the gates takes os.cpu_count() processes), except G1's
# per-core timing, which takes G1_PROCS processes (default: one per core).
# Resumable: every step checks its products first and each file is put in place whole; a transitions step cut short
# saves nothing, and running the script again redoes that body's transitions.  The old form "run_arm_k.sh PYTHON"
# still means PY=PYTHON run.
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1  # hardware/trex_form_study
PY=${PY:-python}
case "${1:-run}" in
    run|check|copy-from|g1) CMD=${1:-run}; [ $# -gt 0 ] && shift ;;
    *) if command -v "$1" > /dev/null 2>&1; then PY=$1; CMD=run; shift; else sed -n '2,22p' "$0"; exit 2; fi ;;
esac
OUT=stage1/runs/2026-11-10-speed
INPUTS="$OUT/arm_k_inputs.py"
BODIES=(60-N-T-10-K 60-N-H-10-K)
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHON_CPU_COUNT=3
LOG=${LOG:-$OUT/logs}
mkdir -p "$LOG"
say() { echo "$(date -u +%T) $*" | tee -a "$LOG/run_arm_k.log"; }

have_transitions() {  # transitions.json holds every transition of transitions.TRANSITIONS
    "$PY" - "$1" <<'EOF'
import json, sys
from pathlib import Path
from designlab.stage1 import transitions
p = Path(sys.argv[1]) / "transitions.json"
ok = p.exists() and set(json.loads(p.read_text())["transitions"]) >= set(transitions.TRANSITIONS)
sys.exit(0 if ok else 1)
EOF
}
have_gate() {  # gates.json holds gate $2 (G1-G4)
    "$PY" - "$1" "$2" <<'EOF'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "gates.json"
sys.exit(0 if p.exists() and sys.argv[2] in json.loads(p.read_text()) else 1)
EOF
}
gate() {  # gate $2 of body $1 into gates.json; G1 times one process per core (G1_PROCS) as its rule asks
    local b=$1 g=$2 status
    say "$b gates --only $g"
    if [ "$g" = g1 ] && [ -n "${G1_PROCS:-}" ]; then
        PYTHON_CPU_COUNT=$G1_PROCS "$PY" -m designlab.stage1 gates --body "$b" --out "$OUT" --only g1 \
            > "$LOG/${b}_gates_g1.log" 2>&1
    elif [ "$g" = g1 ]; then
        env -u PYTHON_CPU_COUNT "$PY" -m designlab.stage1 gates --body "$b" --out "$OUT" --only g1 \
            > "$LOG/${b}_gates_g1.log" 2>&1
    else
        "$PY" -m designlab.stage1 gates --body "$b" --out "$OUT" --only "$g" > "$LOG/${b}_gates_$g.log" 2>&1
    fi
    status=$?
    say "$b $g exit status $status (1: a gate recorded so far failed)"
}

run() {
    local b pid pids=() failed=0 rc
    # 1. the case's round-two models (git-ignored), regenerated exactly from the records if missing
    "$PY" "$INPUTS" models 2>&1 | tee -a "$LOG/run_arm_k.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "models step failed"; return 1; }
    # 2. plant and references, both bodies at once (one process each): a refs.json whose arrays are missing or wrong
    #    gets them regenerated against its own sha256
    for b in "${BODIES[@]}"; do
        rm -f "$LOG/${b}_arrays_refs.log"
        (
            if [ ! -f "$OUT/$b/refs.json" ]; then
                parts=refs; [ -f "$OUT/$b/plant.json" ] || parts=plant,refs
                say "$b prepare --parts $parts"
                exec "$PY" -m designlab.stage1 prepare --body "$b" --out "$OUT" --parts "$parts" --no-gates \
                    > "$LOG/${b}_prepare_$parts.log" 2>&1
            fi
            if [ ! -f "$OUT/$b/plant.json" ]; then
                say "$b prepare --parts plant"
                "$PY" -m designlab.stage1 prepare --body "$b" --out "$OUT" --parts plant --no-gates \
                    > "$LOG/${b}_prepare_plant.log" 2>&1 || exit 1
            fi
            exec "$PY" "$INPUTS" arrays "$b" refs > "$LOG/${b}_arrays_refs.log" 2>&1
        ) &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    for b in "${BODIES[@]}"; do [ -f "$LOG/${b}_arrays_refs.log" ] && tail -n 1 "$LOG/${b}_arrays_refs.log"; done
    [ "$failed" -eq 0 ] || { echo "plant/refs step failed (see $LOG)"; return 1; }
    # 3. transitions, one body at a time on 3 processes; a complete transitions.json gets its arrays checked the same way
    for b in "${BODIES[@]}"; do
        if ! have_transitions "$OUT/$b"; then
            say "$b prepare --parts transitions"
            "$PY" -m designlab.stage1 prepare --body "$b" --out "$OUT" --parts transitions --no-gates \
                > "$LOG/${b}_transitions.log" 2>&1 || { echo "$b transitions failed (see $LOG)"; return 1; }
        else
            "$PY" "$INPUTS" arrays "$b" transitions > "$LOG/${b}_arrays_transitions.log" 2>&1
            rc=$?
            tail -n 1 "$LOG/${b}_arrays_transitions.log"
            [ "$rc" -eq 0 ] || { echo "$b transition arrays failed (see $LOG)"; return 1; }
        fi
    done
    # 4. gates G1-G4, one gate at a time: gates.json keeps each gate as it finishes
    for b in "${BODIES[@]}"; do
        for g in g1 g2 g3 g4; do
            have_gate "$OUT/$b" "${g^^}" || gate "$b" "$g"
        done
    done
    "$PY" "$INPUTS" check && say "done"
}

case "$CMD" in
    check) "$PY" "$INPUTS" check ;;
    copy-from) "$PY" "$INPUTS" copy-from "${1:?copy-from DIR: the checkout to copy the arm K inputs from}" ;;
    run) run ;;
    g1) "$PY" "$INPUTS" check || exit 1
        for b in "${BODIES[@]}"; do gate "$b" g1; done ;;
esac
