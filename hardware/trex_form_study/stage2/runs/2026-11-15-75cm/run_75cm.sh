#!/usr/bin/env bash
# Stage 2, the 75 cm track, tasks 3 and 4 (STAGE2_75_PLAN.md, rule H6): the bar's runs on the three frozen sets of
# 75-M-H-10, as one chain of steps in the pre-registered order.  Run from anywhere; it works in hardware/trex_form_study.
#
#   bash stage2/runs/2026-11-15-75cm/run_75cm.sh list      # every step in order with its command; runs nothing
#   bash stage2/runs/2026-11-15-75cm/run_75cm.sh cost      # write COST.md (rates measured on Stage 1's own conditions)
#   bash stage2/runs/2026-11-15-75cm/run_75cm.sh all       # refcheck of each set, task 3, task 4, the report
#
# Before it (the lead; code state A): the legacy check, the regression, and for each set build, screen, verify, freeze:
#   python -m designlab.stage2.moves legacy --body 75-M-H-10 --procs 2 --record stage2/sets/legacy_check.json
#   python -m designlab.stage2.regress run --tier full --record stage2/regress/regress.json
#   for S in V0 F25F9 F25F9cl: sets build, setscreen run, setscreen md, sets verify, then sets freeze  (--set S --body 75-M-H-10)
# COST.md is committed before task 3's first unit; `all` refuses to start without it.
#
# Each step is `step <name> <command...>`: it is skipped when $RUN/done/<name> exists, runs through tools/lowprio.sh,
# appends the command's own exit code to $RUN/logs/<name>.log, and writes the marker only on exit 0.  Exit 3 is the
# budget ledger's "cut by the pre-registered order": the step gets $RUN/done/<name>.cut and the chain goes on, so
# that every later unit of that task is listed as cut in budget.json too (none is skipped to reach a cheaper one).
# Exit 4 is "episodes failed to run" (rule C8: the same seeds are run again on the next run of the same key): the
# step's command is run once more at once (the ledger counts it); if it is 4 again the step gets
# $RUN/done/<name>.incomplete and no done marker, and the chain goes on.  At the end done/ALL is written only if no
# step is incomplete; otherwise the chain writes done/STOPPED, exits 4 and does not start again by itself (a broken
# episode must not spend the task's budget on a watchdog's restarts): look at the step's log, then remove
# done/STOPPED and start `all` again, which runs the incomplete steps again.  Any other exit code stops the chain.
# Every command resumes by itself (finished records are skipped, finished episodes of a killed run are kept), and
# the bar's lock on the run directory refuses a second copy: the chain is safe to run twice and after a kill.  One
# step is one unit of the budget: one set at one delay within a tier; a tier listed "with attribution" has its
# attribution steps directly after its episode steps, set by set (each record's attribution is a unit of its own).
#
# Watchdog (tools/watchdog_example.sh's form; run it from tools/keepalive.sh every 10 min, from hardware/trex_form_study):
#   [ -e stage2/runs/2026-11-15-75cm/done/ALL ] || [ -e stage2/runs/2026-11-15-75cm/done/STOPPED ] || pgrep -f 'run_75cm.sh all' > /dev/null || PY=.venv/bin/python setsid nohup bash stage2/runs/2026-11-15-75cm/run_75cm.sh all > /dev/null 2>&1 < /dev/null &
# Check `uptime` and `df -h .` before starting.  The markers and logs are git-ignored: they show the order on the
# machine that ran it and are no evidence; budget.json and the records are.
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1  # hardware/trex_form_study
PY=${PY:-.venv/bin/python}
RUN=stage2/runs/2026-11-15-75cm
BODY=75-M-H-10
PROCS=${PROCS:-2}
STAGE1=stage1/runs/2026-11-08
export PYTHONPATH=. MUJOCO_GL=disable OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
MODE=${1:-}
LIST=""
[ "$MODE" = list ] && LIST=1

say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$RUN/logs/run_75cm.log"; }

step() {  # step <name> <command...>
    local name=$1 code try
    shift
    if [ -n "$LIST" ]; then
        local state="to run"
        [ -e "$RUN/done/$name.incomplete" ] && state="incomplete"
        [ -e "$RUN/done/$name" ] && state="done"
        [ -e "$RUN/done/$name.cut" ] && state="cut"
        echo "$name [$state]: $*"
        return 0
    fi
    [ -e "$RUN/done/$name" ] && return 0
    [ -e "$RUN/done/$name.cut" ] && return 0
    for try in 1 2; do  # exit 4 (episodes failed to run): the same command once more, rule C8's second run
        [ "$try" = 1 ] && say "start $name" || say "again $name (episodes failed to run)"
        echo "$(date -u +%FT%TZ) $*" >> "$RUN/logs/$name.log"
        bash tools/lowprio.sh "$@" >> "$RUN/logs/$name.log" 2>&1
        code=$?
        echo "$(date -u +%FT%TZ) exit $code" >> "$RUN/logs/$name.log"
        [ "$code" = 4 ] || break
    done
    case $code in
        0) rm -f "$RUN/done/$name.incomplete"; touch "$RUN/done/$name"; say "done $name" ;;
        3) rm -f "$RUN/done/$name.incomplete"; touch "$RUN/done/$name.cut"
           say "cut $name (the budget: cut by the pre-registered order)" ;;
        4) touch "$RUN/done/$name.incomplete"
           say "INCOMPLETE $name (episodes failed to run, twice): $RUN/logs/$name.log; the chain goes on" ;;
        *) say "FAILED $name (exit $code): $RUN/logs/$name.log"; exit "$code" ;;
    esac
}

finish() {  # done/ALL only when no step is incomplete; else done/STOPPED and exit 4 (no restart by itself)
    local left
    left=$(cd "$RUN/done" && ls | grep '\.incomplete$' | sed 's/\.incomplete$//' | tr '\n' ' ')
    if [ -n "$left" ]; then
        touch "$RUN/done/STOPPED"
        say "NOT all done: incomplete steps (episodes failed to run): ${left}; remove $RUN/done/STOPPED to run them again"
        exit 4
    fi
    rm -f "$RUN/done/STOPPED"
    touch "$RUN/done/ALL"
    say "all steps done"
}

r2() { echo "R2/42@${1}ms,R2/43@${1}ms"; }              # the two fitted rows at a delay
r20() { echo "R2/42@${1}ms,R2/43@${1}ms,R0@${1}ms"; }   # ... and the open-loop row
r0_traces() {  # R0 rows' traces are hashed and not kept when less than 1.5 GB is free (the records say so)
    [ "$(df -Pk . | awk 'NR==2 {print $4}')" -lt 1572864 ] && echo "--no-traces-for R0"
    return 0
}

unit() {  # unit <task> <step name> <unit name> <set> <rows> <conditions> [bar arguments...]
    local task=$1 name=$2 unit=$3 set=$4 rows=$5 conds=$6
    shift 6
    # shellcheck disable=SC2046
    step "$name" "$PY" -m designlab.stage2.bar run --set "$set" --body "$BODY" --rows "$rows" --conditions "$conds" \
        --procs "$PROCS" --task "$task" --unit "$unit" --cost "$RUN/COST.md" --stage1-run "$STAGE1" --out "$RUN" \
        $(r0_traces) "$@"
}
run() { unit "$1" "$2" "$2" "${@:3}"; }                       # a unit named after its step
attr() { unit "$1" "$2-attribution" "$2" "${@:3}" --attribute judged; }  # a tier "with attribution": its command again
again() { unit "$1" "T$1.10-${2#T"$1".}" "$2" "${@:3}"; }      # tier 10: a tier's command again, for its attribution
rep() { run "$1" "$2-seeds5042" "${@:3}" --block seeds5042 --passed-of seeds3042 --attribute none; }

task3() {  # the walk: walk/v0.4 and walk/v0.4_mu0.54 (@walk), 16 core-hours
    local s d f
    for s in F25F9cl F25F9 V0; do run 3 "T3.1-$s-10ms" "$s" "$(r20 10)" @walk --attribute none; done
    # (1) is "with attribution": its attribution units stand directly after its episode units, before tier (2)
    for s in F25F9cl F25F9 V0; do attr 3 "T3.1-$s-10ms" "$s" "$(r20 10)" @walk; done
    rep 3 "T3.2-F25F9cl-10ms" F25F9cl "$(r2 10)" @walk
    for s in F25F9cl F25F9 V0; do run 3 "T3.3-$s-0ms" "$s" "$(r20 0)" @walk --attribute none; done
    for d in 10 0; do  # the omega control: Stage 1's 0.03 H crouch; no verdict, not replicated, not attributed
        run 3 "T3.3-F25F9cl-omega_s1-${d}ms" F25F9cl "R2/42@${d}ms" walk/v0.4 --control omega_s1 --attribute none
    done
    for d in 4 6; do for s in F25F9cl F25F9 V0; do run 3 "T3.4-$s-${d}ms" "$s" "$(r20 "$d")" @walk --attribute none; done; done
    for d in 2 8; do for s in F25F9cl F25F9 V0; do run 3 "T3.5-$s-${d}ms" "$s" "$(r20 "$d")" @walk --attribute none; done; done
    for d in 10 0; do for s in F25F9cl V0; do run 3 "T3.6-$s-${d}ms" "$s" "$(r2 "$d")" @speed-ladder --attribute none; done; done
    run 3 "T3.7-F25F9-0ms" F25F9 "$(r2 0)" walk/v0.5 --attribute none
    run 3 "T3.7-F25F9cl-0ms" F25F9cl "$(r2 0)" walk/v0.45_mu0.54,walk/v0.5_mu0.54 --attribute none
    run 3 "T3.7-F25F9cl-2ms" F25F9cl "$(r2 2)" walk/v0.5 --attribute none
    # (8) replication of (3)-(7): the candidate only (the controls are not run on seeds5042)
    rep 3 "T3.8-F25F9cl-walk-0ms" F25F9cl "$(r2 0)" @walk
    for d in 4 6 2 8; do rep 3 "T3.8-F25F9cl-walk-${d}ms" F25F9cl "$(r2 "$d")" @walk; done
    for d in 10 0; do rep 3 "T3.8-F25F9cl-ladder-${d}ms" F25F9cl "$(r2 "$d")" @speed-ladder; done
    rep 3 "T3.8-F25F9cl-pr-0ms" F25F9cl "$(r2 0)" walk/v0.45_mu0.54,walk/v0.5_mu0.54
    rep 3 "T3.8-F25F9cl-pr-2ms" F25F9cl "$(r2 2)" walk/v0.5
    for d in 4 6; do for s in F25F9cl V0; do run 3 "T3.9-$s-${d}ms" "$s" "$(r2 "$d")" @speed-ladder --attribute none; done; done
    # (10) attribution of the remaining records, in the tiers' order: each record's attribution is a unit of its own
    again 3 "T3.2-F25F9cl-10ms-seeds5042" F25F9cl "$(r2 10)" @walk --block seeds5042 --passed-of seeds3042 --attribute all
    for s in F25F9cl F25F9 V0; do again 3 "T3.3-$s-0ms" "$s" "$(r20 0)" @walk --attribute all; done
    for d in 4 6; do for s in F25F9cl F25F9 V0; do again 3 "T3.4-$s-${d}ms" "$s" "$(r20 "$d")" @walk --attribute all; done; done
    for d in 2 8; do for s in F25F9cl F25F9 V0; do again 3 "T3.5-$s-${d}ms" "$s" "$(r20 "$d")" @walk --attribute all; done; done
    for d in 10 0; do for s in F25F9cl V0; do again 3 "T3.6-$s-${d}ms" "$s" "$(r2 "$d")" @speed-ladder --attribute all; done; done
    again 3 "T3.7-F25F9-0ms" F25F9 "$(r2 0)" walk/v0.5 --attribute all
    again 3 "T3.7-F25F9cl-0ms" F25F9cl "$(r2 0)" walk/v0.45_mu0.54,walk/v0.5_mu0.54 --attribute all
    again 3 "T3.7-F25F9cl-2ms" F25F9cl "$(r2 2)" walk/v0.5 --attribute all
    f="--block seeds5042 --passed-of seeds3042 --attribute all"
    # shellcheck disable=SC2086
    {
        again 3 "T3.8-F25F9cl-walk-0ms-seeds5042" F25F9cl "$(r2 0)" @walk $f
        for d in 4 6 2 8; do again 3 "T3.8-F25F9cl-walk-${d}ms-seeds5042" F25F9cl "$(r2 "$d")" @walk $f; done
        for d in 10 0; do again 3 "T3.8-F25F9cl-ladder-${d}ms-seeds5042" F25F9cl "$(r2 "$d")" @speed-ladder $f; done
        again 3 "T3.8-F25F9cl-pr-0ms-seeds5042" F25F9cl "$(r2 0)" walk/v0.45_mu0.54,walk/v0.5_mu0.54 $f
        again 3 "T3.8-F25F9cl-pr-2ms-seeds5042" F25F9cl "$(r2 2)" walk/v0.5 $f
    }
    for d in 4 6; do for s in F25F9cl V0; do again 3 "T3.9-$s-${d}ms" "$s" "$(r2 "$d")" @speed-ladder --attribute all; done; done
}

task4() {  # the rest of the bar: the 14 turn, nudge and stop conditions (@rest), 12 core-hours; F25F9cl, then V0
    local s d t
    for s in F25F9cl V0; do run 4 "T4.1-$s-10ms" "$s" "$(r20 10)" @rest --attribute none; done
    for s in F25F9cl V0; do attr 4 "T4.1-$s-10ms" "$s" "$(r20 10)" @rest; done  # (1) "with attribution", before (2)
    rep 4 "T4.2-F25F9cl-10ms" F25F9cl "$(r2 10)" @rest
    t=3
    for d in 0 4 6; do
        for s in F25F9cl V0; do run 4 "T4.$t-$s-${d}ms" "$s" "$(r20 "$d")" @rest --attribute none; done
        t=$((t + 1))
    done
    for s in F25F9cl V0; do run 4 "T4.6-$s-10ms" "$s" "$(r2 10)" @ladders --attribute none; done
    for d in 0 4 6; do rep 4 "T4.7-F25F9cl-${d}ms" F25F9cl "$(r2 "$d")" @rest; done
}

cost() {  # COST.md: the rates, measured on Stage 1's own committed condition, and the candidate's bar at 10 ms
    [ -n "$LIST" ] && { echo "cost: bar dry-run --set F25F9cl ... > $RUN/COST.md"; return 0; }
    [ -s "$RUN/COST.md" ] && { echo "$RUN/COST.md is there already"; return 0; }
    bash tools/lowprio.sh "$PY" -m designlab.stage2.bar dry-run --set F25F9cl --body "$BODY" --rows "$(r20 10)" \
        --conditions @gated --stage1-run "$STAGE1" --out "$RUN" > "$RUN/COST.md.tmp" || { rm -f "$RUN/COST.md.tmp"; return 1; }
    mv "$RUN/COST.md.tmp" "$RUN/COST.md"
    echo "wrote $RUN/COST.md: commit it before task 3's first unit"
}

all() {
    local s
    if [ -z "$LIST" ] && [ ! -s "$RUN/COST.md" ]; then
        echo "$RUN/COST.md is missing: run '$0 cost' and commit it before task 3's first unit" >&2
        exit 2
    fi
    if [ -z "$LIST" ] && [ -e "$RUN/done/STOPPED" ]; then
        echo "$RUN/done/STOPPED: the chain stopped with incomplete steps; read their logs, then remove it" >&2
        exit 4
    fi
    for s in V0 F25F9 F25F9cl; do
        # bar refcheck exits 1 when a reference fails a term of its own item: the plan states that before the runs
        # (C3's expectation) and it is a report, no gate.  The step passes when the check wrote its record and the
        # record's items and input hashes are those of the record committed beside the set (the one its freeze cites).
        step "refcheck-$s" bash -c '
            rm -f "$3/$1/$2/refcheck.json"
            "$0" -m designlab.stage2.bar refcheck --set "$1" --body "$2" --out "$3"; c=$?
            { [ "$c" -le 1 ] && [ -s "$3/$1/$2/refcheck.json" ]; } || exit $(( c > 1 ? c : 1 ))
            exec "$0" -c "import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
same = a[\"items\"] == b[\"items\"] and a[\"inputs\"][\"files\"] == b[\"inputs\"][\"files\"]
print(\"refcheck: items and inputs \" + (\"equal\" if same else \"DIFFER from\") + \" those of \" + sys.argv[2])
sys.exit(0 if same else 1)" "$3/$1/$2/refcheck.json" "stage2/sets/$1/$2/refcheck.json"' "$PY" "$s" "$BODY" "$RUN"
    done
    task3
    task4
    step report "$PY" -m designlab.stage2.barreport --out "$RUN"
    [ -n "$LIST" ] || finish
}

case "$MODE" in
    list) all ;;
    cost) mkdir -p "$RUN/logs"; cost ;;
    all) mkdir -p "$RUN/done" "$RUN/logs"; all ;;
    *) sed -n '2,8p' "$0"; exit 2 ;;
esac
