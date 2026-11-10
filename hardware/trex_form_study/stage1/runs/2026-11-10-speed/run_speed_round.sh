#!/usr/bin/env bash
# Stage 1b, the speed round (STAGE1B_PLAN.md sections 3-6; designlab/stage1/speedfit.py, speedreport.py): the exact
# commands per arm and body.  Run from hardware/trex_form_study in the main checkout, after this branch and the
# faster-knee branch are merged and arm K's prepare and G1-G4 have run (run_arm_k.sh; plan section 6, decisions 5 and
# 7).  Every step can be run again after a restart: a fit resumes from its last generation's checkpoint, speed G5
# reuses the validation evaluations it kept (speedfit.G5_RESUME_RULE) and eval skips the records it has.
#
#   bash stage1/runs/2026-11-10-speed/run_speed_round.sh check            # the merge, arm K's gates, the tests
#   bash stage1/runs/2026-11-10-speed/run_speed_round.sh fit  BODY        # the speed fit, one worker (or resume it)
#   bash stage1/runs/2026-11-10-speed/run_speed_round.sh g5   BODY [G]    # speed G5 (at generation G: plan section 5)
#   bash stage1/runs/2026-11-10-speed/run_speed_round.sh rows BODY        # the validation rows, attributed
#   bash stage1/runs/2026-11-10-speed/run_speed_round.sh all  BODY        # fit, then g5, then rows
#   bash stage1/runs/2026-11-10-speed/run_speed_round.sh report           # plan section 4's reading (speedreport)
#
# The four bodies at once, one core each (plan section 6, decision 7), with nothing else heavy on the machine:
#   for b in 60-N-T-10 60-N-H-10 60-N-T-10-K 60-N-H-10-K; do
#       nohup bash stage1/runs/2026-11-10-speed/run_speed_round.sh all $b > /dev/null 2>&1 &
#   done
# If the fits cannot reach generation 30 in time (plan section 5), stop them, read the generation each reached
# (<body>/speedfit/42/cma.json: len(history)), and judge every arm at the lowest:  g5 BODY G, then rows BODY.
#
# Arm S (the picks as they are) reads its references from Stage 1's run; arm K's are in this run; arm K starts from
# its pick's Stage 1 selection (its candidate id, speedfit.START_RULE) and also runs R2@0ms (decision 3):
#   speedfit --body 60-N-T-10   --inputs stage1/runs/2026-11-08 --workers 1 --out stage1/runs/2026-11-10-speed
#   speedfit --body 60-N-T-10-K --start-from 60-N-T-10           --workers 1 --out stage1/runs/2026-11-10-speed
#   speedg5  --body B [--inputs stage1/runs/2026-11-08] --workers 1 --out stage1/runs/2026-11-10-speed
#   STAGE1B_SPEED_RUN=stage1/runs/2026-11-10-speed eval --body B --row-module designlab.stage1.speedfit
#       --rows R2s/42,R2s/42@10ms,R2@10ms[,R2@0ms] --behaviors walk,turn --attribute --workers 1
#       [--inputs stage1/runs/2026-11-08] --out stage1/runs/2026-11-10-speed
#   speedreport --out stage1/runs/2026-11-10-speed --stage1 stage1/runs/2026-11-08
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1  # hardware/trex_form_study
PY=${PY:-python}
OUT=stage1/runs/2026-11-10-speed
STAGE1=stage1/runs/2026-11-08
BODIES=(60-N-T-10 60-N-H-10 60-N-T-10-K 60-N-H-10-K)
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 STAGE1B_SPEED_RUN=$OUT
LOG=${LOG:-$OUT/logs}
mkdir -p "$LOG"

arm() { case "$1" in *-K) echo K ;; *) echo S ;; esac; }
inputs() { if [ "$(arm "$1")" = S ]; then echo "--inputs $STAGE1"; fi; }
start_from() { if [ "$(arm "$1")" = K ]; then echo "--start-from ${1%-K}"; fi; }
rows() { if [ "$(arm "$1")" = K ]; then echo "R2s/42,R2s/42@10ms,R2@10ms,R2@0ms"; else echo "R2s/42,R2s/42@10ms,R2@10ms"; fi; }
say() { echo "$(date -u +%T) $*" | tee -a "$LOG/run_speed_round.log"; }

check() {
    "$PY" - <<'EOF' || return 1
import json
from pathlib import Path
from designlab.stage1 import BODIES, panels, speedfit, tune
from designlab.stage1.__main__ import round_two_models
missing = {"60-N-T-10-K", "60-N-H-10-K"} - set(BODIES)
assert not missing, f"not merged: {sorted(missing)} not registered (STAGE1B_PLAN.md section 6, decision 5)"
for b in ("60-N-T-10", "60-N-H-10", "60-N-T-10-K", "60-N-H-10-K"):  # the git-ignored files each body needs here
    assert round_two_models(b), f"{b}: no round2/refine/<case>/models here (git-ignored): copy them in first"
    run = speedfit.STAGE1_RUN if speedfit.pick_of(b) == b else speedfit.RUN
    files = sorted(set(speedfit.mix_files()) | set(panels.panel_files(panels.select(panels.conditions(),
                                                                                    names=speedfit.ROW_CONDITIONS))))
    tune.read_inputs(run / b, files)  # every reference and transition the round reads, by its recorded sha256
    print(b, "inputs", run / b, "checked:", len(files), "files")
for b in ("60-N-T-10-K", "60-N-H-10-K"):
    g = json.loads((speedfit.RUN / b / "gates.json").read_text())
    assert all(k in g for k in ("G1", "G2", "G3", "G4")), f"{b}: G1-G4 not all recorded"
    print(b, "pick", speedfit.pick_of(b), "push bound", speedfit.env.PUSH_BOUND_M_S[b], "m/s, gates",
          {k: g[k].get("pass", g[k].get("pass_")) for k in ("G1", "G2", "G3", "G4")}, "all:", g.get("pass"))
s = speedfit.speed_sources()
print("speed sources", s["sha256"], "over", len(s["modules"]), "modules: record it; it must not change (decision 5)")
EOF
    "$PY" -m pytest -q tests/test_stage1_speedfit.py
}

fit() {
    local b=$1 resume=""
    [ -f "$OUT/$b/speedfit/42/cma.json" ] && resume="--resume"
    say "$b speedfit $resume"
    # shellcheck disable=SC2046
    "$PY" -m designlab.stage1 speedfit --body "$b" $(inputs "$b") $(start_from "$b") --workers 1 $resume \
        --out "$OUT" >> "$LOG/${b}_speedfit.log" 2>&1
}

g5() {
    local b=$1 at=${2:+--at-generation $2}
    say "$b speedg5 $at"
    # shellcheck disable=SC2046,SC2086
    "$PY" -m designlab.stage1 speedg5 --body "$b" $(inputs "$b") --workers 1 $at --out "$OUT" >> "$LOG/${b}_speedg5.log" 2>&1
}

rows_() {
    local b=$1
    say "$b eval $(rows "$b")"
    # shellcheck disable=SC2046
    "$PY" -m designlab.stage1 eval --body "$b" --row-module designlab.stage1.speedfit --rows "$(rows "$b")" \
        --behaviors walk,turn --attribute --workers 1 $(inputs "$b") --out "$OUT" >> "$LOG/${b}_eval.log" 2>&1
}

body_ok() { case " ${BODIES[*]} " in *" $1 "*) return 0 ;; *) echo "body $1: one of ${BODIES[*]}"; return 1 ;; esac; }

case "${1:-}" in
    check) check ;;
    fit) body_ok "${2:?body}" && fit "$2" ;;
    g5) body_ok "${2:?body}" && g5 "$2" "${3:-}" ;;
    rows) body_ok "${2:?body}" && rows_ "$2" ;;
    all) body_ok "${2:?body}" && fit "$2" && g5 "$2" && rows_ "$2" && say "$2 done" ;;
    report) "$PY" -m designlab.stage1 speedreport --out "$OUT" --stage1 "$STAGE1" ;;
    *) sed -n '2,32p' "$0"; exit 2 ;;
esac
