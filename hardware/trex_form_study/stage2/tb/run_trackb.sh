#!/bin/sh
# Track B's run chain: the validation records, S2-4, the 60 cm local run and R2L, steps V1-V10 of the run plan
# (round C design section 6) as one idempotent chain.  Simulation only.  Run from anywhere; it works in
# hardware/trex_form_study.  POSIX sh (dash): `sh -n` passes.
#
#   sh stage2/tb/run_trackb.sh            # every step not yet done, in order; stops at the first that fails
#   TB_DRY=1 sh stage2/tb/run_trackb.sh   # prints every step in order with its command; runs and writes nothing
#
# Environment: PY (default .venv/bin/python), TB_WORK (default stage2/tb/work: markers, logs, live outputs; git
# ignores it), TB_DEV (default stage2/tb/dev_episodes.json), TB_REREAD (round A's `bar reread` record of Stage 1b's
# 10 ms turn episodes of 60-N-T-10, for G9-run (e) of the 60 cm body and head), TB_BODIES (default the four bodies
# and heads), TB_DRY.
#
# Each step is `step <name> <command...>` (round A's pattern, stage2/runs/2026-11-15-75cm/run_75cm.sh): skipped when
# $TB_WORK/done/<name> exists; otherwise run through tools/lowprio.sh with the command and its exit code appended to
# $TB_WORK/logs/<name>.log, and the marker written on exit 0 only.  Any other exit code stops the chain (a gate that
# fails goes to the lead).  Every command resumes by itself: a gate writes its one record again; kill-resume, the
# probes, the trainer (--resume when its run.json exists), the bar runner and R2L keep what they finished; `nb
# collect` leaves a file that is already there with the same bytes.  Every long job writes under $TB_WORK and is
# collected with `nb collect` when it ends (design D22).  One process at a time (the chain is sequential).
#
# Rule X1: each closed-loop step on a look set that is no record (the rates, the probes, kill-resume) is listed in
# $TB_DEV before it runs and again when it is done (`gates2 dev-episode`); the records' own steps (G8a, R0L's records
# and G9-run, S2-4, the 60 cm run, the R2L fits) are not.  gl-start and r6-start run on the view S1 (rule X1 as
# settled on 15 Nov 2026): no look-set episode.  Every closed-loop gate refuses a look set whose g9_ref.json has not
# passed (gates2.require_g9_ref).
#
# Not in this chain (the lead's hands):
#   V0   the -X preconditions (D5): G1 of both -X bodies on an idle machine, the jaw cover decision, one kinematic
#        `sets build` of a walk and a start on an -X model; the look sets V6 and V6noF5 built, screened, verified and
#        frozen for every body and head (the lead's step after the look package's review).
#   V0b  the upload route (D20): the lead uploads and re-downloads a 20 MB tar, a 5 MB tar and a 170 kB file, each
#        approved by the owner; outcome in stage2/tb/validation/drive_route.json.
#   V11  the bundle: `bundle build --bundle-id <id> --bodies <trained> --sets V6 --out stage2/tb/bundle/<id>`; commit
#        C1 (D24); push; the uploads (the lead, one approval per part); UPLOADS.json.
#   V12  the acceptance run: `nb acceptance --ref <C1> --parts stage2/tb/bundle/<id>/parts --scratch
#        /home/user/mesozoic-wt/tb-accept --record stage2/tb/bundle/<id>/ACCEPTANCE.json` (the lead).
#   V13  the steps sheet: commit C2 (stage2/tb/colab_steps.txt with C1's SHA, ACCEPTANCE.json) (the lead).
# The records this chain writes under stage2/tb/gates/ and stage2/tb/validation/ are committed by the lead as they
# are made; those it collects to stage2/tb/runs/, stage2/tb/bar/ and stage2/tb/r2l/ likewise.
set -u
cd "$(dirname "$0")/../.." || exit 1  # hardware/trex_form_study
PY=${PY:-.venv/bin/python}
TB=stage2/tb
W=${TB_WORK:-$TB/work}
DONE=$W/done
LOGS=$W/logs
DEV=${TB_DEV:-$TB/dev_episodes.json}
REREAD=${TB_REREAD:-}
BODIES=${TB_BODIES:-"75-M-H-10 75-M-H-10-X 60-N-T-10 60-N-T-10-X"}
PLAIN="75-M-H-10 60-N-T-10"
DRY=${TB_DRY:-}
G="$PY -m designlab.stage2.gates2"
T2="$PY -m designlab.stage2.train2"
NB="$PY -m designlab.stage2.nb"
export PYTHONPATH=. MUJOCO_GL=disable OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0

say() {
    if [ -n "$DRY" ]; then
        echo "$*"
    else
        mkdir -p "$LOGS"
        echo "$(date -u +%FT%TZ) $*" | tee -a "$LOGS/run_trackb.log"
    fi
}

step() {  # step <name> <command...>
    _name=$1
    shift
    if [ -e "$DONE/$_name" ]; then
        say "skip $_name (done)"
        return 0
    fi
    if [ -n "$DRY" ]; then
        echo "[$_name] $*"
        return 0
    fi
    mkdir -p "$DONE" "$LOGS"
    say "start $_name"
    echo "$(date -u +%FT%TZ) $*" >> "$LOGS/$_name.log"
    bash tools/lowprio.sh "$@" >> "$LOGS/$_name.log" 2>&1
    _code=$?
    echo "$(date -u +%FT%TZ) exit $_code" >> "$LOGS/$_name.log"
    if [ "$_code" = 0 ]; then
        touch "$DONE/$_name"
        say "done $_name"
    else
        say "FAILED $_name (exit $_code): $LOGS/$_name.log; the chain stops (it goes to the lead)"
        exit "$_code"
    fi
}

dev() {  # dev <state> <step> <set> <body> <rows or mode> <count> <reason> <command text>: rule X1's list
    if [ -n "$DRY" ]; then
        echo "  (rule X1: $2 in $DEV as $1: set $3, body $4, $5, count $6)"
        return 0
    fi
    $G dev-episode --file "$DEV" --state "$1" --step "$2" --set "$3" --body "$4" --what "$5" --count "$6" \
        --reason "$7" --command "$8" >> "$LOGS/run_trackb.log" 2>&1 || { say "FAILED to list $2 in $DEV"; exit 2; }
}

devstep() {  # devstep <name> <set> <body> <rows or mode> <count> <reason> <command...>
    _dn=$1 _ds=$2 _db=$3 _dw=$4 _dc=$5 _dr=$6
    shift 6
    [ -e "$DONE/$_dn" ] && { say "skip $_dn (done)"; return 0; }
    [ -n "$DRY" ] || mkdir -p "$DONE" "$LOGS"
    dev listed "$_dn" "$_ds" "$_db" "$_dw" "$_dc" "$_dr" "$*"
    step "$_dn" "$@"
    dev done "$_dn" "$_ds" "$_db" "$_dw" "$_dc" "$_dr" "$*"
}

resume_flag() {  # --resume when the job directory holds a run.json (the trainer refuses a new run over one)
    [ -f "$1/run.json" ] && echo "--resume"
    return 0
}

# ---------------------------------------------------------------------------------------------------------------
# V1  G9-ref per body and head (kinematic; before any closed-loop episode on the set), and on V6noF5 for R2L's
#     bodies (rule Z2 in the form it gives that set; before R2L's first episode).  2-3 minutes each.
# ---------------------------------------------------------------------------------------------------------------
for B in $BODIES; do
    step "V1-g9ref-V6-$B" $G g9-ref --body "$B" --set V6 --record "$TB/gates/V6/$B/g9_ref.json"
done
for B in $PLAIN; do
    step "V1-g9ref-V6noF5-$B" $G g9-ref --body "$B" --set V6noF5 --record "$TB/gates/V6noF5/$B/g9_ref.json"
done

# ---------------------------------------------------------------------------------------------------------------
# V2  G8a per body and head (rule Z1: 10 conditions x seeds 3042, 3043 x 0 and 10 ms; Z1's own episodes, a record).
# ---------------------------------------------------------------------------------------------------------------
for B in $BODIES; do
    step "V2-g8a-$B" $G g8a --body "$B" --set V6 --record "$TB/gates/V6/$B/g8a.json"
done

# ---------------------------------------------------------------------------------------------------------------
# V3  R0L's records (the 16 gated conditions and walk/nominal, 20 seeds each), collected, then G9-run.
#     NOTE for the lead: as built, the bar's runner lets only rows of the controller R0 run on a set outside the 75 cm
#     round's three (bar._check_rule, RULES["checks"]), so `lookbar run` refuses R0L on V6 until that is ruled.
# ---------------------------------------------------------------------------------------------------------------
for B in $BODIES; do
    step "V3-r0l-$B" $PY -m designlab.stage2.lookbar run --set V6 --body "$B" --rows R0L@10ms \
        --conditions @gated,walk/nominal --out "$W/bar/r0l" --procs 2
    step "V3-collect-$B" $NB collect --src "$W/bar/r0l/V6/$B" --dst "$TB/bar/r0l/V6/$B" --kind bar
    RR=""  # (without TB_REREAD, G9-run's part (e) of the 60 cm body and head fails: no reread record given)
    case $B in
        60-N-T-10*) [ -n "$REREAD" ] && RR="--stage1b-reread $REREAD" ;;
    esac
    # shellcheck disable=SC2086
    step "V3-g9run-$B" $G g9-run --body "$B" --set V6 --bar-out "$TB/bar/r0l" --record "$TB/gates/V6/$B/g9_run.json" $RR
done

# ---------------------------------------------------------------------------------------------------------------
# V5  rule O2's sigma probe (reported) and rule O4's filter (decided on 75-M-H-10; 60-N-T-10 reported).
# ---------------------------------------------------------------------------------------------------------------
devstep V5-sigma-probe V6 75-M-H-10,60-N-T-10 "R5 noise episodes" 240 "rule O2's sigma probe (reported)" \
    $T2 probe-sigma --bodies 75-M-H-10,60-N-T-10 --set V6 --out "$TB/validation/sigma_probe.json"
devstep V5-filter V6 75-M-H-10,60-N-T-10 "R5 noise episodes" 80 "rule O4's filter measurement" \
    $T2 filter --bodies 75-M-H-10,60-N-T-10 --set V6 --out "$TB/validation/filter.json"

# ---------------------------------------------------------------------------------------------------------------
# V6  rule O3's learning-rate probe, every body and head (6 updates of 8,192 steps per cell, at most 3 rates x 2
#     sigmas per body and head), 2 workers.
# ---------------------------------------------------------------------------------------------------------------
devstep V6-lr-probe V6 "$(echo $BODIES | tr ' ' ',')" "R5 training env steps" 1179648 "rule O3's learning-rate probe" \
    $T2 probe-lr --bodies "$(echo $BODIES | tr ' ' ',')" --set V6 --out "$TB/validation/lr_probe.json" --workers 2

# ---------------------------------------------------------------------------------------------------------------
# V7  the training start with MUJOCO_GL set against it and R6's start (both on the view S1, rule X1), then rule
#     K3's kill and resume on the real env (V6; two runs of 131,072 steps).
# ---------------------------------------------------------------------------------------------------------------
step V7-gl-start $G gl-start --out "$W/gl_start" --record "$TB/validation/gl_start.json"
step V7-r6-start $G r6-start --body 75-M-H-10 --set S1 --out "$W/r6_start" --record "$TB/validation/r6_start.json"
devstep V7-kill-resume V6 75-M-H-10 "R5 training env steps (K3's two runs)" 278528 "rule K3's kill and resume" \
    $G kill-resume --body 75-M-H-10 --set V6 --out "$W/kill_resume" --record "$TB/validation/kill_resume.json"

# ---------------------------------------------------------------------------------------------------------------
# V4  rates and memory: last of the validation steps and alone (2 floors in one process; 8 envs at 1-4 workers x
#     16,384 steps); recorded provisional unless the load average before it is under 0.5 (lesson 10).
# ---------------------------------------------------------------------------------------------------------------
devstep V4-rates V6 75-M-H-10 "R5 env steps (rates)" 98304 "the rate and memory measurement" \
    $G rates --body 75-M-H-10 --set V6 --record "$TB/validation/rates.json"

# ---------------------------------------------------------------------------------------------------------------
# V8  S2-4 (rule M3): R0L's baseline made by the monitor's baseline function and committed as
#     stage2/tb/runs/s2-4/R0L_BASELINE.json before S2-4 starts; the run (2,000,000 steps asked, checks every
#     500,000, 3 workers); collected; S2-10's record against the committed baseline.  NOTE for the lead: the trainer
#     takes no committed baseline, so "else the run stops before its first rollout" is checked after the run by
#     s2-10 --committed-baseline, not before its first rollout; and the run plan's V8 line is to be rewritten to
#     the notebook's real lines (the lead's ruling 6 of 17 Nov 2026): until then the trainer runs here directly.
# ---------------------------------------------------------------------------------------------------------------
J4=$W/local/s2-4/jobs/R5-75-M-H-10-s42
step V8-baseline $PY -c "from designlab.stage2 import monitor2; monitor2.baseline('75-M-H-10', 'V6', '$W/s2-4-baseline')"
step V8-baseline-commit sh -c "mkdir -p '$TB/runs/s2-4' && { [ -e '$TB/runs/s2-4/R0L_BASELINE.json' ] || cp '$W/s2-4-baseline/r0l.json' '$TB/runs/s2-4/R0L_BASELINE.json'; } && cmp '$W/s2-4-baseline/r0l.json' '$TB/runs/s2-4/R0L_BASELINE.json'"
# shellcheck disable=SC2046
step V8-s2-4 $T2 run --mode R5 --body 75-M-H-10 --seed 42 --set V6 --out "$W/local/s2-4" --run-id s2-4 \
    --steps 2000000 --monitor-every 500000 --workers 3 $(resume_flag "$J4")
step V8-collect $NB collect --src "$W/local/s2-4" --dst "$TB/runs/s2-4" --kind run
step V8-s2-10 $PY -m designlab.stage2.monitor2 s2-10 --job "$TB/runs/s2-4/jobs/R5-75-M-H-10-s42" \
    --committed-baseline "$TB/runs/s2-4/R0L_BASELINE.json" --out "$TB/runs/s2-4/S2_10.json"

# ---------------------------------------------------------------------------------------------------------------
# V9  the 60 cm local run (D11, M6; recommended): 5,000,000 steps asked, its checks also at 0 ms, 3 workers.
# ---------------------------------------------------------------------------------------------------------------
J9=$W/local/local-60/jobs/R5-60-N-T-10-s41
# shellcheck disable=SC2046
step V9-local-60 $T2 run --mode R5 --body 60-N-T-10 --seed 41 --set V6 --out "$W/local/local-60" --run-id local-60 \
    --steps 5000000 --monitor-also-k 0 --workers 3 $(resume_flag "$J9")
step V9-collect $NB collect --src "$W/local/local-60" --dst "$TB/runs/local-60" --kind run

# ---------------------------------------------------------------------------------------------------------------
# V10 R2L (rules Q1-Q6): the rotation table, the fit, the selection and the row's publication for (V6noF5,
#     75-M-H-10) and (V6noF5, 60-N-T-10), 2 workers; the candidate (F25F9cl, 75-M-H-10) only if granted (not here).
#     NOTE: designlab.stage2.r2lfit (work package CF) is not on the working branch yet.
# ---------------------------------------------------------------------------------------------------------------
for B in $PLAIN; do
    step "V10-r2l-rotation-$B" $PY -m designlab.stage2.r2lfit rotation --body "$B" --set V6noF5 --seed 42 --out "$W/r2l"
    step "V10-r2l-fit-$B" $PY -m designlab.stage2.r2lfit fit --body "$B" --set V6noF5 --seed 42 --out "$W/r2l" \
        --workers 2 --resume
    step "V10-r2l-select-$B" $PY -m designlab.stage2.r2lfit select --body "$B" --set V6noF5 --seed 42 --out "$W/r2l"
    step "V10-r2l-publish-$B" $PY -m designlab.stage2.r2lfit publish --body "$B" --set V6noF5 --seed 42 --out "$W/r2l"
done
step V10-collect $NB collect --src "$W/r2l" --dst "$TB/r2l" --kind r2l

if [ -n "$DRY" ]; then
    say "dry run (TB_DRY): nothing was run or written"
else
    say "all steps of the chain are done"
fi
