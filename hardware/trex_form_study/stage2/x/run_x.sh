#!/usr/bin/env bash
# Stage 2's expressive head, the -X bodies (STAGE2_PLAN.md S2-2; DECISION_LOG P29: modelled in simulation beside
# today's head, nothing adopted, P2's deferral of a jaw stands for the build): 60-N-T-10-X and 75-M-H-10-X, each its
# pick with a neck yaw at the neck base (+-0.6 rad) and a hinged jaw (0-0.6 rad), each driven by an STS3215
# (designlab/build.py NECK_YAW_RANGE; refine cases head_x_60 and head_x_75).  The clearance screen cannot cut either
# range (it pairs no body with its parent: clearance.SEAM_RULE), so the plant keeps them; the case records hold the
# seam sweep (head_x.joint_ranges).  Notes: stage2/x/HEAD_X.md.
# Modelled on arm K's stage1/runs/2026-11-10-speed/run_arm_k.sh: the refine cases' build and screens, then Stage 1's
# prepare (plant, references, transitions) and gates G1-G3.  G4, the tracker's gate, does not apply.
#
# Run from hardware/trex_form_study:  [PY=PYTHON] [LOG=DIR] [PROCS=N] bash stage2/x/run_x.sh [COMMAND]
#   build          the refine cases head_x_60 and head_x_75 (built once from their picks' records, not re-sized, and
#                  screened: clearance, sensors, turning, carpet, envelope) and the carpet-lift pass on each case with
#                  its pick's lifts and walks; every product is skipped when its record is here, so no committed record
#                  is rewritten (refine.json and REFINE.md are regenerated from the records, with the cases' rows)
#   check          every git-ignored input the run reads (x_inputs.json: the cases' 4 model files and the -X bodies'
#                  reference and transition arrays) is here with its listed sha256; exit status 1 if not
#   copy-from DIR  copy those inputs from another checkout DIR (its root or its hardware/trex_form_study), each checked
#                  by its sha256; a file already here and right is left alone
#   run (default)  every product, each skipped only when it is here in full: the cases' models (regenerated from the
#                  records if missing, about 1 min), plant.json, refs.json and transitions.json with every array at
#                  its recorded sha256 (missing or wrong arrays are regenerated in a scratch directory and put in
#                  place only if they match the record, which is never rewritten), then each gate gates.json lacks
#                  (G1-G3); with no x_inputs.json yet, it is written at the end
#   g1             G1 again on both -X bodies, alone and with one process per core as its rule asks (G1_PROCS, if
#                  set, instead), replacing their G1 records: run once with the machine otherwise idle
#   gates          G1-G3 again on both -X bodies (G1 as g1 times it), replacing their gate records
#   records        the head_x cases' candidate records again from the code (x_inputs.py records: each case's build
#                  in a scratch directory, put in place only when its models match x_inputs.json; the screens are not
#                  re-run), then refine.json and REFINE.md from the records
# The CLI's output goes to LOG (default stage2/x/logs, not committed).  At most PROCS processes at a time (default 2:
# PYTHON_CPU_COUNT=PROCS, since every pool of the prepare and the gates takes os.cpu_count() processes), each at
# nice 10, so a build elsewhere on the machine keeps its cores, except G1's per-core timing, which takes one process
# per core as arm K's does (PYTHON_CPU_COUNT unset).  On a shared machine set G1_PROCS (e.g. G1_PROCS=$PROCS): G1 is
# then recorded as provisional, its step rate timed on G1_PROCS of the machine's cores (gates.PROVISIONAL_RATE_RULE),
# until g1 runs with one process per core on an otherwise idle machine.
# Resumable: every step checks its products first and each file is put in place whole; a transitions step cut short
# saves nothing, and running the script again redoes that body's transitions.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1  # hardware/trex_form_study
PY=${PY:-python}
PROCS=${PROCS:-2}
case "${1:-run}" in
    run|build|check|copy-from|g1|gates|records) CMD=${1:-run}; [ $# -gt 0 ] && shift ;;
    *) sed -n '2,38p' "$0"; exit 2 ;;
esac
OUT=stage2/x
INPUTS="$OUT/x_inputs.py"
BODIES=(60-N-T-10-X 75-M-H-10-X)
CASES=(head_x_60 head_x_75)
CIDS=(60-N-T-10 75-M-H-10)  # each case's round-two candidate id (its files' names)
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHON_CPU_COUNT=$PROCS
LOG=${LOG:-$OUT/logs}
mkdir -p "$LOG"
say() { echo "$(date -u +%T) $*" | tee -a "$LOG/run_x.log"; }
py() { nice -n 10 "$PY" "$@"; }

have_transitions() {  # transitions.json holds every transition of transitions.TRANSITIONS
    py - "$1" <<'EOF'
import json, sys
from pathlib import Path
from designlab.stage1 import transitions
p = Path(sys.argv[1]) / "transitions.json"
ok = p.exists() and set(json.loads(p.read_text())["transitions"]) >= set(transitions.TRANSITIONS)
sys.exit(0 if ok else 1)
EOF
}
have_gate() {  # gates.json holds gate $2 (G1-G3)
    py - "$1" "$2" <<'EOF'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "gates.json"
sys.exit(0 if p.exists() and sys.argv[2] in json.loads(p.read_text()) else 1)
EOF
}
gate() {  # gate $2 of body $1 into gates.json; G1 times one process per core as its rule asks (G1_PROCS, if set)
    local b=$1 g=$2 status
    say "$b gates --only $g"
    if [ "$g" = g1 ] && [ -n "${G1_PROCS:-}" ]; then
        # (nproc alone would print 1: it reads the OMP_NUM_THREADS=1 exported above)
        say "$b g1: step rate on $G1_PROCS of $(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc) cores (G1_PROCS);" \
            "recorded as provisional unless that is all"
        PYTHON_CPU_COUNT=$G1_PROCS py -m designlab.stage1 gates --body "$b" --out "$OUT" --only g1 \
            > "$LOG/${b}_gates_g1.log" 2>&1
    elif [ "$g" = g1 ]; then
        env -u PYTHON_CPU_COUNT nice -n 10 "$PY" -m designlab.stage1 gates --body "$b" --out "$OUT" --only g1 \
            > "$LOG/${b}_gates_g1.log" 2>&1
    else
        py -m designlab.stage1 gates --body "$b" --out "$OUT" --only "$g" > "$LOG/${b}_gates_$g.log" 2>&1
    fi
    status=$?
    say "$b $g exit status $status (1: a gate recorded so far failed or is provisional, or G1 and G2 are not both in)"
    tail -n 4 "$LOG/${b}_gates_$g.log" | tee -a "$LOG/run_x.log"
}

build() {
    local i c cid todo=()
    # 1. the refine cases: built from the picks' records and screened (refine --resume keeps what is here)
    for i in "${!CASES[@]}"; do
        c=${CASES[$i]} cid=${CIDS[$i]}
        if [ ! -f "round2/refine/$c/candidates/$cid.json" ] \
            || [ ! -f "round2/refine/$c/review/screening/${cid}_envelope.json" ]; then
            todo+=("$c")
        fi
    done
    if [ "${#todo[@]}" -gt 0 ]; then
        say "refine --cases ${todo[*]} --resume (build and screens)"
        py -m designlab.refine --out round2/refine --workers "$PROCS" --cases "${todo[@]}" --resume \
            > "$LOG/refine_cases.log" 2>&1 || { echo "refine failed (see $LOG)"; return 1; }
    fi
    # 2. the carpet-lift pass on each case, with its pick's lifts and walks (refs.CARPET_LIFT_RECORD)
    if [ ! -f round2/refine/carpet_lift_head_x_60_0.190_0.195_nominal_only.json ]; then
        say "refine --carpet-lift --lift-cases head_x_60 --lifts 0.19 0.195 --nominal-only"
        py -m designlab.refine --out round2/refine --workers "$PROCS" --cases --carpet-lift --lift-cases head_x_60 \
            --lifts 0.19 0.195 --nominal-only > "$LOG/carpet_lift_head_x_60.log" 2>&1 || { echo "carpet lift failed"; return 1; }
    fi
    if [ ! -f round2/refine/carpet_lift_head_x_75_0.175_0.180_0.190.json ]; then
        say "refine --carpet-lift --lift-cases head_x_75 --lifts 0.175 0.18 0.19"
        py -m designlab.refine --out round2/refine --workers "$PROCS" --cases --carpet-lift --lift-cases head_x_75 \
            --lifts 0.175 0.18 0.19 > "$LOG/carpet_lift_head_x_75.log" 2>&1 || { echo "carpet lift failed"; return 1; }
    fi
    say "build done"
}

run() {
    local b pid pids=() failed=0 rc
    # 1. the cases' models (git-ignored), regenerated exactly from the records if missing
    if [ -f "$OUT/x_inputs.json" ]; then
        py "$INPUTS" models 2>&1 | tee -a "$LOG/run_x.log"
        [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "models step failed"; return 1; }
    fi
    # 2. plant and references, both bodies at once (one process each): a refs.json whose arrays are missing or wrong
    #    gets them regenerated against its own sha256
    for b in "${BODIES[@]}"; do
        rm -f "$LOG/${b}_arrays_refs.log"
        (
            if [ ! -f "$OUT/$b/refs.json" ]; then
                parts=refs; [ -f "$OUT/$b/plant.json" ] || parts=plant,refs
                say "$b prepare --parts $parts"
                exec env PYTHON_CPU_COUNT=1 nice -n 10 "$PY" -m designlab.stage1 prepare --body "$b" --out "$OUT" \
                    --parts "$parts" --no-gates > "$LOG/${b}_prepare_$parts.log" 2>&1
            fi
            if [ ! -f "$OUT/$b/plant.json" ]; then
                say "$b prepare --parts plant"
                PYTHON_CPU_COUNT=1 py -m designlab.stage1 prepare --body "$b" --out "$OUT" --parts plant --no-gates \
                    > "$LOG/${b}_prepare_plant.log" 2>&1 || exit 1
            fi
            [ -f "$OUT/x_inputs.json" ] || exit 0
            exec nice -n 10 "$PY" "$INPUTS" arrays "$b" refs > "$LOG/${b}_arrays_refs.log" 2>&1
        ) &
        pids+=($!)
        [ "$PROCS" -ge 2 ] || wait "${pids[-1]}"
    done
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    for b in "${BODIES[@]}"; do [ -f "$LOG/${b}_arrays_refs.log" ] && tail -n 1 "$LOG/${b}_arrays_refs.log"; done
    [ "$failed" -eq 0 ] || { echo "plant/refs step failed (see $LOG)"; return 1; }
    # 3. transitions, one body at a time on PROCS processes; a complete transitions.json gets its arrays checked
    for b in "${BODIES[@]}"; do
        if ! have_transitions "$OUT/$b"; then
            say "$b prepare --parts transitions"
            py -m designlab.stage1 prepare --body "$b" --out "$OUT" --parts transitions --no-gates \
                > "$LOG/${b}_transitions.log" 2>&1 || { echo "$b transitions failed (see $LOG)"; return 1; }
        elif [ -f "$OUT/x_inputs.json" ]; then
            py "$INPUTS" arrays "$b" transitions > "$LOG/${b}_arrays_transitions.log" 2>&1
            rc=$?
            tail -n 1 "$LOG/${b}_arrays_transitions.log"
            [ "$rc" -eq 0 ] || { echo "$b transition arrays failed (see $LOG)"; return 1; }
        fi
    done
    # 4. gates G1-G3, one gate at a time: gates.json keeps each gate as it finishes (G4 does not apply)
    for b in "${BODIES[@]}"; do
        for g in g1 g2 g3; do
            have_gate "$OUT/$b" "${g^^}" || gate "$b" "$g"
        done
    done
    if [ ! -f "$OUT/x_inputs.json" ]; then
        py "$INPUTS" manifest | tee -a "$LOG/run_x.log"
    fi
    py "$INPUTS" check && say "done"
}

case "$CMD" in
    build) build ;;
    check) py "$INPUTS" check ;;
    copy-from) py "$INPUTS" copy-from "${1:?copy-from DIR: the checkout to copy the -X inputs from}" ;;
    run) run ;;
    g1) py "$INPUTS" check || exit 1
        for b in "${BODIES[@]}"; do gate "$b" g1; done ;;
    gates) py "$INPUTS" check || exit 1
        for b in "${BODIES[@]}"; do for g in g1 g2 g3; do gate "$b" "$g"; done; done ;;
    records) py "$INPUTS" check || exit 1
        py "$INPUTS" records 2>&1 | tee -a "$LOG/run_x.log"
        [ "${PIPESTATUS[0]}" -eq 0 ] || exit 1
        say "refine --report-only (refine.json and REFINE.md from the records)"
        py -m designlab.refine --out round2/refine --report-only > "$LOG/refine_report.log" 2>&1 || exit 1
        py "$INPUTS" check ;;
esac
