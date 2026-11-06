#!/bin/bash
# e2e lane, phase A: pilot4_judge.py on the stand-in runs (--check-only first, then the 12 cells: 3 seeds in
# parallel, each seed's 4 checkpoints in order), a second judgement refused, and the tampered cases (each must be
# refused, exit 2, nothing written). Every command goes through job.sh (CMDLOG.txt + e2e/status.txt). Nothing trains.
SP=/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad
T=$SP/work/pilot4/tools
LANE=$T/out/fix
PY=$SP/work/pilot3/branch/venv_full/bin/python
TREE=$T/tree
HEAD=0e6a57bb4d6ba30aef36dd9aa30ff4d175786cc8
RUNS=$LANE/runs
OUT=$LANE/e2e/OUT
LG=$LANE/e2e/logs
J="bash $LANE/scripts/job.sh"
declare -A P=([42]=sha256:41fc0b0cd7167ab25a2567779b89f431b829f6a2bae4c72fa3c89192732e87c4
              [52]=sha256:2a87a6874f252de9462960dbfe5bfd6dc0986241679800111d68ef8c68f34c00
              [60]=sha256:2c066770802541e47fb453b8dbd04b030bc8053d4224af27a52eb0281e4645a2)
mkdir -p "$LG"
ISO="cd $TREE && ISO_WT=$TREE $PY -I $T/iso.py"
JUDGE="$ISO $T/pilot4_judge.py"
# A1. --check-only on every cell (nothing may be written)
for s in 42 52 60; do
  for k in 500000 1000000 1500000 2000000; do
    $J "judge_check_s${s}_${k}" "$JUDGE $RUNS/pilot4_walk_s$s --seed $s --steps $k --parent-sha256 ${P[$s]} --out $OUT --expect-head $HEAD --check-only > $LG/judge_check_s${s}_${k}.log 2>&1"
  done
done
$J "out_absent_after_check_only" "test ! -e $OUT"
# A2. the judge: seeds in parallel, checkpoints in order
for s in 42 52 60; do
  (
    for k in 500000 1000000 1500000 2000000; do
      $J "judge_s${s}_${k}" "$JUDGE $RUNS/pilot4_walk_s$s --seed $s --steps $k --parent-sha256 ${P[$s]} --out $OUT --expect-head $HEAD > $LG/judge_s${s}_${k}.log 2>&1"
    done
  ) &
done
wait
# A3. a second judgement of a cell is refused (exit 2)
$J "judge_again_s42_2000000_expect2" "$JUDGE $RUNS/pilot4_walk_s42 --seed 42 --steps 2000000 --parent-sha256 ${P[42]} --out $OUT --expect-head $HEAD > $LG/judge_again_s42_2000000.log 2>&1"
# A4. tampered cases (each with its own OUT, which must not exist afterwards)
TP=$LANE/tampered
$J "tamper_wrong_parent_cli_expect2" "$JUDGE $RUNS/pilot4_walk_s42 --seed 42 --steps 2000000 --parent-sha256 ${P[52]} --out $TP/OUT_wrong_parent_cli --expect-head $HEAD > $LG/tamper_wrong_parent_cli.log 2>&1"
$J "tamper_parent_record_expect2" "$JUDGE $TP/parent_record/pilot4_walk_s42 --seed 42 --steps 2000000 --parent-sha256 ${P[42]} --out $TP/OUT_parent_record --expect-head $HEAD > $LG/tamper_parent_record.log 2>&1"
$J "tamper_command_table_expect2" "$JUDGE $TP/command_table/pilot4_walk_s42 --seed 42 --steps 2000000 --parent-sha256 ${P[42]} --out $TP/OUT_command_table --expect-head $HEAD > $LG/tamper_command_table.log 2>&1"
$J "tamper_raw_parent_expect2" "$JUDGE $TP/raw_parent/pilot4_walk_s42 --seed 42 --steps 2000000 --parent-sha256 ${P[42]} --out $TP/OUT_raw_parent --expect-head $HEAD > $LG/tamper_raw_parent.log 2>&1"
$J "setup_check_record_verbatim_expect2" "$JUDGE $TP/setup_check_record/pilot4_walk_s42 --seed 42 --steps 2000000 --parent-sha256 ${P[42]} --out $TP/OUT_setup_check_record --expect-head $HEAD > $LG/setup_check_record_verbatim.log 2>&1"
$J "setup_check_record_allow_mismatch_check_only_expect0" "$JUDGE $TP/setup_check_record/pilot4_walk_s42 --seed 42 --steps 2000000 --parent-sha256 ${P[42]} --out $TP/OUT_setup_check_record --expect-head $HEAD --allow-commit-mismatch --check-only > $LG/setup_check_record_allow_mismatch.log 2>&1"
for c in wrong_parent_cli parent_record command_table raw_parent setup_check_record; do
  $J "tamper_out_absent_$c" "test ! -e $TP/OUT_$c"
done
# A5. a dirty tree: my own clone at the same head with one tracked file edited (restored afterwards)
C=$LANE/tmp/clone
$J "dirty_make" "printf '\n(e2e lane: a local edit that makes the tree dirty)\n' >> $C/README.md && git -C $C status --porcelain --untracked-files=no > $LG/dirty_status.txt"
$J "dirty_judge_expect2" "cd $C && ISO_WT=$C $PY -I $T/iso.py $T/pilot4_judge.py $RUNS/pilot4_walk_s42 --seed 42 --steps 2000000 --parent-sha256 ${P[42]} --out $TP/OUT_dirty --expect-head $HEAD --check-only > $LG/dirty_judge.log 2>&1"
$J "dirty_straight_expect2" "cd $C && ISO_WT=$C $PY -I $T/iso.py $T/pilot4_straight.py seed --seed 42 --run $RUNS/pilot4_walk_s42 --cell $OUT/s42/2000000 --json $TP/dirty_straight_s42.json > $LG/dirty_straight.log 2>&1"
$J "dirty_habit_expect2" "cd $C && ISO_WT=$C $PY -I $T/iso.py $T/pilot4_habit.py seed --seed 42 --ledger $RUNS/pilot4_walk_s42/03_locomotion/eval_episodes.csv --cell $OUT/s42/2000000 --json $TP/dirty_habit_s42.json > $LG/dirty_habit.log 2>&1"
$J "dirty_outputs_absent" "test ! -e $TP/OUT_dirty && test ! -e $TP/dirty_straight_s42.json && test ! -e $TP/dirty_habit_s42.json"
$J "dirty_restore" "git -C $C checkout -- README.md && test -z \"\$(git -C $C status --porcelain --untracked-files=no)\""
$J "tree_clean_after_phase_a" "git -C $TREE status --porcelain --ignored > $LG/tree_status_after_phase_a.txt && test ! -s $LG/tree_status_after_phase_a.txt"
# ---- fix stage additions
# A6. the default --expect-head (PILOT_HEAD): the shared tree passes without --expect-head (check-only, a fresh OUT)
$J "judge_default_head_check_only_expect0" "$JUDGE $RUNS/pilot4_walk_s42 --seed 42 --steps 2000000 --parent-sha256 ${P[42]} --out $TP/OUT_default_head --check-only > $LG/judge_default_head_check_only.log 2>&1"
# A7. my clone moved to HEAD~1 on the pilot branch: refused for its head without --expect-head; then restored
$J "clone_to_parent" "git -C $C checkout -q -B claude/pilot-walk-r10-p4 $HEAD~1 && git -C $C rev-parse HEAD > $LG/clone_parent_head.txt"
$J "judge_default_head_at_parent_expect2" "cd $C && ISO_WT=$C $PY -I $T/iso.py $T/pilot4_judge.py $RUNS/pilot4_walk_s42 --seed 42 --steps 2000000 --parent-sha256 ${P[42]} --out $TP/OUT_default_head_parent --check-only > $LG/judge_default_head_at_parent.log 2>&1"
$J "clone_restore_head" "git -C $C checkout -q -B claude/pilot-walk-r10-p4 $HEAD && test \"\$(git -C $C rev-parse HEAD)\" = $HEAD && test -z \"\$(git -C $C status --porcelain --untracked-files=no)\""
# A8. the validity review's drift case: reward_weights.forward_vel_weight 3.0 with the task digest as recorded; the
# judge (not --check-only) must refuse before the probe and write nothing
$J "tamper_env_drift_expect2" "$JUDGE $TP/env_drift/pilot4_walk_s42 --seed 42 --steps 2000000 --parent-sha256 ${P[42]} --out $TP/OUT_env_drift --expect-head $HEAD > $LG/tamper_env_drift.log 2>&1"
$J "fix_outputs_absent" "test ! -e $TP/OUT_default_head && test ! -e $TP/OUT_default_head_parent && test ! -e $TP/OUT_env_drift"
$J "tree_clean_after_phase_a_fix" "git -C $TREE status --porcelain --ignored > $LG/tree_status_after_phase_a_fix.txt && test ! -s $LG/tree_status_after_phase_a_fix.txt"
