#!/bin/bash
# Fix lane, phase 0 (the e2e lane's set-up commands, scripted): a clone of the shared tree at 0e6a57b, byte copies of
# the three 8M pairs, the smoke runner's params, the runner's new QUICK_TEST refusal on the real notebook, the REAL
# set-up path to learn() per seed (edge_pairs.py: the notebook's manual cell through the fixed nb_pilot_runner4.py
# --stop-at-learn, the edge-prepared model saved by SB3's CheckpointCallback), the stand-in runs and tampered copies,
# the final-model fallback copies, the sidecar and weight facts. Every command goes through job.sh. Nothing trains.
SP=/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad
T=$SP/work/pilot4/tools
LANE=$T/out/fix
PY=$SP/work/pilot3/branch/venv_full/bin/python
TREE=$T/tree
C=$LANE/tmp/clone
S=$LANE/scripts
J="bash $S/job.sh"
LG=$LANE/e2e/logs
P3=$SP/work/pilot3/runs/final
FAKE=$SP/work/pilot4/build/check/judge/fake_runs
mkdir -p "$LG" "$LANE/setup"
$J clone "git clone --shared --branch claude/pilot-walk-r10-p4 $TREE $C > $LG/clone.log 2>&1 && test \"\$(git -C $C rev-parse HEAD)\" = 0e6a57bb4d6ba30aef36dd9aa30ff4d175786cc8 && test -z \"\$(git -C $C status --porcelain --ignored)\""
for s in 42 52 60; do
  $J "pairs_copy_s$s" "mkdir -p $LANE/tmp/pairs/pilot3_walk_s$s/03_locomotion/models && cp -p $P3/pilot3_walk_s$s/03_locomotion/models/stage2_8000000_steps.zip $P3/pilot3_walk_s$s/03_locomotion/models/stage2_vecnormalize_8000000_steps.pkl $LANE/tmp/pairs/pilot3_walk_s$s/03_locomotion/models/ && cmp $P3/pilot3_walk_s$s/03_locomotion/models/stage2_8000000_steps.zip $LANE/tmp/pairs/pilot3_walk_s$s/03_locomotion/models/stage2_8000000_steps.zip && cmp $P3/pilot3_walk_s$s/03_locomotion/models/stage2_vecnormalize_8000000_steps.pkl $LANE/tmp/pairs/pilot3_walk_s$s/03_locomotion/models/stage2_vecnormalize_8000000_steps.pkl"
done
$J make_params "$PY -I -B $S/from_e2e/make_params.py $LANE > $LG/make_params.log 2>&1"
# the runner's new refusal on the real notebook: QUICK_TEST false, no --stop-at-learn, no --plan-only (must exit 2,
# before any cell, so nothing can train)
$J runner_not_quick_test_expect2 "cd $C && env -u PYTHONPATH OMP_NUM_THREADS=1 $PY -I $T/nb_pilot_runner4.py --params $LANE/setup/params_s42.json --out $LANE/e2e/runner_not_quick_test --clone $C > $LG/runner_not_quick_test.log 2>&1"
$J runner_not_quick_test_ran_no_cell "test ! -e $LANE/e2e/runner_not_quick_test/runner.stdout.log && test ! -e $LANE/e2e/runner_not_quick_test/cells_edited.json && grep -q 'QUICK_TEST is False, not true' $LANE/e2e/runner_not_quick_test/record.json"
for s in 42 52 60; do
  $J "edge_pairs_s$s" "cd $C && env -u PYTHONPATH ISO_WT=$C OMP_NUM_THREADS=1 GIT_OPTIONAL_LOCKS=0 $PY -I $T/iso.py $S/from_e2e/edge_pairs.py $s $LANE/setup/params_s$s.json $LANE/setup/s$s $C > $LG/edge_pairs_s$s.stdout.txt 2> $LG/edge_pairs_s$s.stderr.txt"
done
$J clone_clean_after_setup "git -C $C status --porcelain --untracked-files=no > $LG/clone_status_after_setup.txt && test ! -s $LG/clone_status_after_setup.txt"
$J cmp_records "$PY -I -B $S/from_e2e/cmp_records.py $LANE/setup $FAKE $LANE/setup/records_vs_setup_check.json > $LG/cmp_records.log 2>&1"
$J make_runs "$PY -I -B $S/from_e2e/make_runs.py $LANE $FAKE $P3 > $LG/make_runs.stdout.txt 2> $LG/make_runs.stderr.txt"
$J make_tampered_fix "$PY -I -B $S/make_tampered_fix.py $LANE > $LG/make_tampered_fix.log 2>&1"
$J make_final "cd $TREE && env -u PYTHONPATH ISO_WT=$TREE $PY -I $T/iso.py $S/from_e2e/make_final.py $LANE > $LG/make_final.log 2>&1"
$J sidecar_facts "cd $TREE && env -u PYTHONPATH ISO_WT=$TREE $PY -I $T/iso.py $S/from_e2e/sidecar_facts.py $LANE $LANE/setup/sidecar_facts.json > $LG/sidecar_facts.log 2>&1"
$J weights_facts "$PY -I -B $S/from_e2e/weights_facts.py $LANE $LANE/setup/weights_facts.json > $LG/weights_facts.log 2>&1"
$J dedupe_models "$PY -I -B $S/from_e2e/dedupe_models.py $LANE $LANE/setup/models_dedupe.json > $LG/dedupe_models.log 2>&1"
$J tree_clean_after_phase_0 "git -C $TREE status --porcelain --ignored > $LG/tree_status_after_phase_0.txt && test ! -s $LG/tree_status_after_phase_0.txt"
