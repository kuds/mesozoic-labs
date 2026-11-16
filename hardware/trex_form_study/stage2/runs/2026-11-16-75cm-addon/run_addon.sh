#!/bin/bash
# Section A of STAGE2_75_ADDON_PLAN.md: today's walk (set V0, body 75-M-H-10) at more servo delays.
# This script is in no record's key.  It runs the stock runner unit by unit, holds section A's own cap (rule A-3)
# and code state (rule A-4), and writes what it decided into the committed file A/ADDON_LEDGER.json.
#
#   bash stage2/runs/2026-11-16-75cm-addon/run_addon.sh list    # the units, their estimates, what would start
#   bash stage2/runs/2026-11-16-75cm-addon/run_addon.sh all     # run them in the registered order
#
# A unit that is done (A/done/<unit>) is not run again; a kill is resumed by running "all" again: a unit the
# runner's budget.json already holds has started and finishes without a new cap test (A-3).  A unit whose episodes
# failed to run twice is marked A/done/<unit>.incomplete, is not started a third time, and the chain goes on.
set -u
cd "$(dirname "$0")/../../.." || exit 1          # hardware/trex_form_study
PY=${PY:-.venv/bin/python}
PROCS=${PROCS:-3}
ADD=stage2/runs/2026-11-16-75cm-addon
RUN=$ADD/A
COST=stage2/runs/2026-11-15-75cm/COST.md
STAGE1=stage1/runs/2026-11-08
CAP_CORE_H=1.0                                   # rule A-3
LEDGER=$RUN/ADDON_LEDGER.json
MODE=${1:-}
export PYTHONPATH=. MUJOCO_GL=disable PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
[ "$MODE" = list ] || [ "$MODE" = all ] || { echo "usage: run_addon.sh list|all"; exit 2; }
mkdir -p "$RUN/logs" "$RUN/done"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$RUN/logs/run_addon.log"; }

# Rule A-4: the runner's own key sources (the hashes every record's key holds) must be those of the registered
# records of stage2/runs/2026-11-15-75cm.
WANT_KEYS="bar 45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d
panels c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1"
HAVE_KEYS=$("$PY" -c "
from designlab.stage2 import bar
from designlab.stage1 import panels
print('bar', bar.bar_sources()['sha256'])
print('panels', panels.panel_sources()['sha256'])")
if [ "$HAVE_KEYS" != "$WANT_KEYS" ]; then
    say "FAILED code state: the runner's key sources are not the registered ones"; echo "$HAVE_KEYS"; exit 1
fi
if [ "$MODE" = all ]; then
    DIRTY=$(git status --short -- . ":(exclude)$RUN") || { say "FAILED code state: git status did not answer"; exit 1; }
    [ -z "$DIRTY" ] || { say "FAILED code state: the checkout has uncommitted changes outside $RUN"; echo "$DIRTY" | head; exit 1; }
    exec 9> "$RUN/logs/chain.lock"               # one chain at a time: a second "all" would misread a running unit
    flock -n 9 || { say "FAILED: another chain holds $RUN/logs/chain.lock"; exit 1; }
fi

# The units of rule A-2, in their order: name | rows | conditions.
UNITS="A1-V0-14ms|R2/42@14ms,R2/43@14ms,R0@14ms|@walk,walk/nominal
A1-V0-16ms|R2/42@16ms,R2/43@16ms,R0@16ms|@walk,walk/nominal
A1-V0-20ms|R2/42@20ms,R2/43@20ms,R0@20ms|@walk,walk/nominal
A2-V0-12ms|R2/42@12ms,R2/43@12ms,R0@12ms|@walk,walk/nominal
A2-V0-18ms|R2/42@18ms,R2/43@18ms,R0@18ms|@walk,walk/nominal
A3-V0-nominal-8ms|R2/42@8ms,R2/43@8ms|walk/nominal"

ledger() {  # ledger <unit> <state> <estimate_core_h> <spent_before_core_h>: one line of A/ADDON_LEDGER.json
    "$PY" - "$LEDGER" "$RUN/budget.json" "$CAP_CORE_H" "$@" <<'EOF'
import json, os, sys, time
path, budget, cap, unit, state, est, spent = sys.argv[1:8]
doc = json.load(open(path)) if os.path.exists(path) else dict(
    schema="addon-ledger-1", rule="STAGE2_75_ADDON_PLAN.md A-3 and A-4", cap_core_h=float(cap),
    note="the runner's own budget.json names its task 3 with a cap of 16 core-hours; this section's cap is the one here",
    key_sha256={}, units=[])
doc["key_sha256"] = dict(line.split() for line in os.environ["HAVE_KEYS"].splitlines())
row = dict(unit=unit, state=state, estimate_core_h=float(est), spent_before_core_h=float(spent), used_core_h=None,
           started_at=None, at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
if os.path.exists(budget):  # a unit the runner started: its first estimate, spent-before and start, and what it used
    for u in json.load(open(budget))["tasks"].get("3", {}).get("units", []):
        if u["unit"] == unit and u.get("estimate_core_h") is not None:
            row.update(estimate_core_h=u["estimate_core_h"], spent_before_core_h=u["spent_before_core_h"],
                       started_at=u.get("started_at"), used_core_h=u.get("used_core_h"))
doc["units"] = [u for u in doc["units"] if u["unit"] != unit] + [row]
with open(path + ".tmp", "w") as f:
    json.dump(doc, f, indent=1, sort_keys=True)
    f.write("\n")
os.replace(path + ".tmp", path)
EOF
}
export HAVE_KEYS

spent() {  # the core-hours section A has spent: the sum of the runner's own units in A/budget.json
    "$PY" - "$RUN/budget.json" <<'EOF'
import json, os, sys
p = sys.argv[1]
t = json.load(open(p))["tasks"].get("3", {}) if os.path.exists(p) else {}
print(f"{sum((u.get('used_core_h') or 0.0) for u in t.get('units', [])):.6f}")
EOF
}

started() {  # started <unit>: "<estimate> <spent before>" of a unit the runner's budget.json holds; fails otherwise
    "$PY" - "$RUN/budget.json" "$1" <<'EOF'
import json, os, sys
p, unit = sys.argv[1:3]
t = json.load(open(p))["tasks"].get("3", {}) if os.path.exists(p) else {}
u = next((u for u in t.get("units", []) if u["unit"] == unit), None)
if u is None or u.get("estimate_core_h") is None:
    sys.exit(1)
print(f"{u['estimate_core_h']:.6f} {u['spent_before_core_h']:.6f}")
EOF
}
fail() { say "FAILED $*"; exit 1; }

CUT=""
echo "$UNITS" | while IFS='|' read -r NAME ROWS CONDS; do
    ARGS="--set V0 --body 75-M-H-10 --rows $ROWS --conditions $CONDS --cost $COST --stage1-run $STAGE1 --out $RUN"
    if [ -e "$RUN/done/$NAME" ]; then [ "$MODE" = list ] && echo "$NAME [done]"; continue; fi
    if [ -e "$RUN/done/$NAME.incomplete" ]; then [ "$MODE" = list ] && echo "$NAME [incomplete: not run a third time]"; continue; fi
    SPENT=$(spent) || fail "$NAME: $RUN/budget.json cannot be read"
    case "$SPENT" in ""|*[!0-9.]*) fail "$NAME: spent core-hours '$SPENT' is no number" ;; esac
    if FIRST=$(started "$NAME" 2>/dev/null); then   # a started unit finishes (A-3): no cap test, its first estimate stands
        EST=${FIRST% *}; BEFORE=${FIRST#* }; FITS=1; WORD="started before"
    else                                            # the dry run's Total at the committed rates, as A-3 says
        EST=$(bash tools/lowprio.sh "$PY" -m designlab.stage2.bar dry-run $ARGS 2>/dev/null < /dev/null | sed -n 's/^Total: .* \([0-9.]*\) core-hours (.*$/\1/p' | tail -1)
        [ -n "$EST" ] || fail "$NAME: no estimate from the dry run"
        BEFORE=$SPENT; WORD="to run"
        FITS=$("$PY" -c "print(1 if $SPENT + $EST <= $CAP_CORE_H + 1e-9 else 0)") || fail "$NAME: the cap test did not run"
        [ "$FITS" = 1 ] || [ "$FITS" = 0 ] || fail "$NAME: the cap test gave '$FITS'"
    fi
    if [ -e "$RUN/done/$NAME.cut" ] || [ -n "$CUT" ] || [ "$FITS" != 1 ]; then
        WHY="after a cut unit"; [ -n "$CUT" ] || [ -e "$RUN/done/$NAME.cut" ] || WHY="$SPENT + $EST over the cap $CAP_CORE_H"
        CUT=1
        if [ "$MODE" = list ]; then echo "$NAME [cut]: estimate $EST core-h, spent before $SPENT, cap $CAP_CORE_H ($WHY)"; continue; fi
        if [ ! -e "$RUN/done/$NAME.cut" ]; then
            ledger "$NAME" cut "$EST" "$SPENT" || fail "$NAME: the ledger was not written"
            touch "$RUN/done/$NAME.cut"; say "cut $NAME ($WHY)"
        fi
        continue
    fi
    if [ "$MODE" = list ]; then
        echo "$NAME [$WORD]: estimate $EST core-h, spent before $BEFORE, spent now $SPENT, cap $CAP_CORE_H, fits $FITS :: bar run $ARGS --procs $PROCS --attribute none --task 3 --unit $NAME"
        continue
    fi
    ledger "$NAME" started "$EST" "$BEFORE" || fail "$NAME: the ledger was not written"
    CODE=1
    for TRY in 1 2; do              # exit 4 (episodes failed to run): the same command once more (C8)
        [ "$TRY" = 1 ] && say "start $NAME" || say "again $NAME (episodes failed to run)"
        bash tools/lowprio.sh "$PY" -m designlab.stage2.bar run $ARGS --procs "$PROCS" --attribute none --task 3 --unit "$NAME" \
            >> "$RUN/logs/$NAME.log" 2>&1 < /dev/null
        CODE=$?
        [ "$CODE" = 4 ] || break
    done
    if [ "$CODE" = 4 ]; then   # episodes failed to run twice (A-3, C8): not started a third time; the chain goes on
        ledger "$NAME" incomplete "$EST" "$BEFORE" || fail "$NAME: the ledger was not written"
        touch "$RUN/done/$NAME.incomplete"; say "INCOMPLETE $NAME (episodes failed to run, twice): $RUN/logs/$NAME.log"; continue
    fi
    if [ "$CODE" != 0 ]; then say "FAILED $NAME (exit $CODE)"; ledger "$NAME" "failed-exit-$CODE" "$EST" "$BEFORE"; exit 1; fi
    ledger "$NAME" done "$EST" "$BEFORE" || fail "$NAME: the ledger was not written"
    touch "$RUN/done/$NAME"; say "done $NAME"
done || exit 1
[ "$MODE" = all ] && { touch "$RUN/done/ALL"; say "all units of section A handled"; }
exit 0
