#!/bin/bash
# Fix lane, phase D: the fix stage's end-to-end refusals on the judged stand-in cells. A drifted copy of the cells:
# the straightness tool refuses it (exit 2, nothing written) and the scorer leaves s42 unscored; a forged
# straightness record (PASS over FAIL values, another tool, looser bars): the scorer refuses it, so s42 waits for a
# record. Every command goes through job.sh. Nothing trains.
SP=/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad
T=$SP/work/pilot4/tools
LANE=$T/out/fix
PY=$SP/work/pilot3/branch/venv_full/bin/python
TREE=$T/tree
E=$LANE/e2e
LG=$E/logs
ST=$E/straight
SC=$E/score
IN=$E/inputs
J="bash $LANE/scripts/job.sh"
$J make_fix_inputs "$PY -I -B $LANE/scripts/make_fix_inputs.py $LANE > $LG/make_fix_inputs.log 2>&1"
$J straight_drift_cell_expect2 "cd $TREE && ISO_WT=$TREE $PY -I $T/iso.py $T/pilot4_straight.py seed --seed 42 --run $LANE/runs/pilot4_walk_s42 --cell $E/OUT_drift/s42/2000000 --json $ST/drift_s42_2000000.json > $LG/straight_drift_cell.log 2>&1"
$J straight_drift_output_absent "test ! -e $ST/drift_s42_2000000.json"
STR=""
for s in 42 52 60; do STR="$STR --straight $ST/s${s}_2000000.json"; done
$J score_drift_cells "$PY -I -B $T/pilot4_score.py $E/OUT_drift --seeds 42,52,60 $STR --video-review $IN/video_review_agree_standin.json --json $SC/score_drift_cells.json > $SC/score_drift_cells.txt 2>&1"
$J score_forged_record "$PY -I -B $T/pilot4_score.py $E/OUT --seeds 42,52,60 --straight $IN/forged_s42_2000000.json --straight $ST/s52_2000000.json --straight $ST/s60_2000000.json --video-review $IN/video_review_agree_standin.json --json $SC/score_forged_record.json > $SC/score_forged_record.txt 2>&1"
$J tree_clean_after_phase_d "git -C $TREE status --porcelain --ignored > $LG/tree_status_after_phase_d.txt && test ! -s $LG/tree_status_after_phase_d.txt"
