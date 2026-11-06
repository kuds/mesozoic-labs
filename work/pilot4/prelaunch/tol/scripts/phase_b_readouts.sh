#!/bin/bash
# Tolerance lane (copied from the fix lane, from the e2e lane), phase B: on the judged stand-in cells, pilot4_straight.py seed mode on all 12 cells, pilot4_habit.py
# (ledger mode on each stand-in ledger, seed mode on each 2,000,000 cell), pilot4_mirror.py and pilot4_mech.py on each
# 2,000,000 cell, and pilot4_noise.py on each run; at most 4 jobs at a time. Every command goes through job.sh
# (CMDLOG.txt + e2e/status.txt). Nothing trains.
SP=$(ls -d /tmp/*/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad)
T=$SP/work/pilot4/tools
LANE=$SP/work/pilot4/prelaunch/tol
PY=$SP/work/pilot3/branch/venv_full/bin/python
TREE=$T/tree
RUNS=$LANE/runs
OUT=$LANE/e2e/OUT
LG=$LANE/e2e/logs
ST=$LANE/e2e/straight
HB=$LANE/e2e/habit
RD=$LANE/e2e/readouts
NZ=$LANE/e2e/noise
mkdir -p "$LG" "$ST" "$HB" "$RD" "$NZ"
ISO="cd $TREE && ISO_WT=$TREE $PY -I $T/iso.py"
JOBS=$LANE/e2e/phase_b_jobs.txt
: > "$JOBS"
add() { printf 'bash %s/scripts/job.sh %s "%s"\n' "$LANE" "$1" "$2" >> "$JOBS"; }
for s in 42 52 60; do
  add "straight_s${s}_2000000" "$ISO $T/pilot4_straight.py seed --seed $s --run $RUNS/pilot4_walk_s$s --cell $OUT/s$s/2000000 --json $ST/s${s}_2000000.json --plot $ST/s${s}_2000000.png --traces $ST/s${s}_2000000.npz > $LG/straight_s${s}_2000000.log 2>&1"
done
for s in 42 52 60; do
  add "habit_seed_s${s}" "$ISO $T/pilot4_habit.py seed --seed $s --ledger $RUNS/pilot4_walk_s$s/03_locomotion/eval_episodes.csv --cell $OUT/s$s/2000000 --json $HB/habit_seed_s$s.json > $LG/habit_seed_s$s.log 2>&1"
  add "mirror_s${s}_2000000" "$ISO $T/pilot4_mirror.py --cell $OUT/s$s/2000000 --json $RD/mirror_s${s}_2000000.json > $LG/mirror_s${s}_2000000.log 2>&1"
  add "mech_s${s}_2000000" "$ISO $T/pilot4_mech.py --cell $OUT/s$s/2000000 --json $RD/mech_s${s}_2000000.json > $LG/mech_s${s}_2000000.log 2>&1"
done
for k in 500000 1000000 1500000; do
  for s in 42 52 60; do
    add "straight_s${s}_${k}" "$ISO $T/pilot4_straight.py seed --seed $s --run $RUNS/pilot4_walk_s$s --cell $OUT/s$s/$k --json $ST/s${s}_${k}.json --plot $ST/s${s}_${k}.png > $LG/straight_s${s}_${k}.log 2>&1"
  done
done
for s in 42 52 60; do
  add "habit_ledger_s${s}" "$PY -I -B $T/pilot4_habit.py ledger $RUNS/pilot4_walk_s$s/03_locomotion/eval_episodes.csv --through 2000000 --json $HB/habit_ledger_s$s.json > $LG/habit_ledger_s$s.log 2>&1"
  add "noise_s${s}" "$PY -I -B $T/pilot4_noise.py $RUNS/pilot4_walk_s$s --json $NZ/noise_s$s.json > $LG/noise_s$s.log 2>&1"
done
xargs -d '\n' -P 4 -I{} bash -c '{}' < "$JOBS"
bash "$LANE/scripts/job.sh" "tree_clean_after_phase_b" "git -C $TREE status --porcelain --ignored > $LG/tree_status_after_phase_b.txt && test ! -s $LG/tree_status_after_phase_b.txt"
