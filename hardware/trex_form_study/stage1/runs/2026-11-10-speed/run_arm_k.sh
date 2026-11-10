#!/usr/bin/env bash
# Stage 1b arm K (STAGE1B_PLAN.md section 3, "The K bodies"): Stage 1's prepare (plant, references, transitions) and
# gates G1-G4 on the two K bodies, after their sizing (python -m designlab.refine --cases narrow_knee_60) and the
# carpet-lift pass that sets their carpet swing (refs.CARPET_LIFT).  At most 3 processes at a time
# (PYTHON_CPU_COUNT=3: every pool of the prepare and the gates takes os.cpu_count() processes).
#
# Run from hardware/trex_form_study:  [LOG=DIR] bash stage1/runs/2026-11-10-speed/run_arm_k.sh [PYTHON]
# (the CLI's output goes to DIR, default stage1/runs/2026-11-10-speed/logs, not committed)
# Resumable: a step whose product exists is skipped (plant.json, refs.json, transitions.json with every transition,
# each gate in gates.json, written as each gate finishes); delete a product (or a gate's entry) to redo it.  A
# transitions step cut short saves nothing; rerun the script and it redoes that body's transitions in full.
set -uo pipefail
PY=${1:-python}
OUT=stage1/runs/2026-11-10-speed
BODIES=(60-N-T-10-K 60-N-H-10-K)
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHON_CPU_COUNT=3
LOG=${LOG:-$OUT/logs}
mkdir -p "$LOG"

have_transitions() {  # every transition of transitions.TRANSITIONS saved
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

# 1. plant and references: one process per body, both bodies at once
pids=()
for b in "${BODIES[@]}"; do
    parts=()
    [ -f "$OUT/$b/plant.json" ] || parts+=(plant)
    [ -f "$OUT/$b/refs.json" ] || parts+=(refs)
    if [ ${#parts[@]} -gt 0 ]; then
        p=$(IFS=,; echo "${parts[*]}")
        echo "$(date -u +%T) $b prepare --parts $p" | tee -a "$LOG/run_arm_k.log"
        "$PY" -m designlab.stage1 prepare --body "$b" --out "$OUT" --parts "$p" --no-gates > "$LOG/${b}_prepare_$p.log" 2>&1 &
        pids+=($!)
    fi
done
for pid in "${pids[@]}"; do wait "$pid" || { echo "plant/refs step failed (see $LOG)"; exit 1; }; done

# 2. transitions, then 3. gates G1-G4: one body at a time, 3 processes
for b in "${BODIES[@]}"; do
    if ! have_transitions "$OUT/$b"; then
        echo "$(date -u +%T) $b prepare --parts transitions" | tee -a "$LOG/run_arm_k.log"
        "$PY" -m designlab.stage1 prepare --body "$b" --out "$OUT" --parts transitions --no-gates \
            > "$LOG/${b}_transitions.log" 2>&1 || { echo "$b transitions failed (see $LOG)"; exit 1; }
    fi
done
for b in "${BODIES[@]}"; do  # one gate at a time: gates.json keeps each gate as it finishes
    for g in g1 g2 g3 g4; do
        if ! have_gate "$OUT/$b" "${g^^}"; then
            echo "$(date -u +%T) $b gates --only $g" | tee -a "$LOG/run_arm_k.log"
            "$PY" -m designlab.stage1 gates --body "$b" --out "$OUT" --only "$g" > "$LOG/${b}_gates_$g.log" 2>&1
            echo "$(date -u +%T) $b $g exit status $? (1: a gate recorded so far failed)" | tee -a "$LOG/run_arm_k.log"
        fi
    done
done
echo "$(date -u +%T) done" | tee -a "$LOG/run_arm_k.log"
