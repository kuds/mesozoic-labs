#!/bin/bash
# Smoke lane: the pinned readouts on the smoke's judged cells and run copies (nothing trains; every command through
# job.sh). TREE is this lane's clean clone at the head; the cells are phase_judge.sh's.
#   S: pilot4_straight.py seed on the launch's 65,536 cell (--validation; without it: refused);
#   C: pilot4_score.py on OUT (as is: validation-only cells are never scored; --validation --primary 65536);
#   V: pilot4_video.py on the 65,536 cell under MUJOCO_GL=egl, then osmesa;
#   H: pilot4_habit.py ledger on both run copies' ledgers, seed on the 65,536 cell (--validation; without: refused);
#   N: pilot4_noise.py on both run copies.
W=$(ls -d /tmp/*/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work)
K=$W/pilot4/prelaunch/smoke
T=$W/pilot4/tools
PY=$W/pilot3/branch/venv_full/bin/python
TREE=$K/judge_tree
RL=$K/runs/launch/pilot4_walk_s42
RR=$K/runs/resume/pilot4_walk_s42
OUT=$K/judge/OUT
OUTR=$K/judge/OUT_resume
RO=$K/readouts
LG=$RO/logs
export JOB_STATUS=$RO/status.txt
J="bash $K/scripts/job.sh"
mkdir -p "$LG" "$RO/straight" "$RO/score" "$RO/habit" "$RO/noise" "$RO/videos"
ISO="cd $TREE && ISO_WT=$TREE $PY -I $T/iso.py"
# S
$J "S1_straight_seed_validation_65536_expect0" "$ISO $T/pilot4_straight.py seed --seed 42 --run $RL --cell $OUT/s42/65536 --json $RO/straight/s42_65536.json --plot $RO/straight/s42_65536.png --traces $RO/straight/s42_65536.npz --validation > $LG/S1_straight_seed_validation.log 2>&1" &
$J "H3_habit_seed_validation_65536_expect0" "$ISO $T/pilot4_habit.py seed --seed 42 --ledger $RL/03_locomotion/eval_episodes.csv --cell $OUT/s42/65536 --json $RO/habit/s42_65536_seed.json --validation > $LG/H3_habit_seed_validation.log 2>&1" &
$J "S2_straight_seed_no_validation_expect2" "$ISO $T/pilot4_straight.py seed --seed 42 --run $RL --cell $OUT/s42/65536 --json $RO/straight/s42_65536_noval.json > $LG/S2_straight_seed_no_validation.log 2>&1"
$J "H4_habit_seed_no_validation_expect2" "$ISO $T/pilot4_habit.py seed --seed 42 --ledger $RL/03_locomotion/eval_episodes.csv --cell $OUT/s42/65536 --json $RO/habit/s42_65536_seed_noval.json > $LG/H4_habit_seed_no_validation.log 2>&1"
$J "H1_habit_ledger_launch" "$PY -I $T/pilot4_habit.py ledger $RL/03_locomotion/eval_episodes.csv --json $RO/habit/ledger_launch.json > $LG/H1_habit_ledger_launch.log 2>&1"
$J "H2_habit_ledger_resume" "$PY -I $T/pilot4_habit.py ledger $RR/03_locomotion/eval_episodes.csv --json $RO/habit/ledger_resume.json > $LG/H2_habit_ledger_resume.log 2>&1"
$J "N1_noise_launch" "$PY -I $T/pilot4_noise.py $RL --json $RO/noise/launch.json > $LG/N1_noise_launch.log 2>&1"
$J "N2_noise_resume" "$PY -I $T/pilot4_noise.py $RR --json $RO/noise/resume.json > $LG/N2_noise_resume.log 2>&1"
$J "V1_video_egl" "cd $TREE && MUJOCO_GL=egl PYOPENGL_PLATFORM=egl ISO_WT=$TREE $PY -I $T/iso.py $T/pilot4_video.py $OUT/s42/65536 --episodes 0,39 --out $RO/videos/egl > $LG/V1_video_egl.log 2>&1"
$J "V2_video_osmesa" "cd $TREE && MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa ISO_WT=$TREE $PY -I $T/iso.py $T/pilot4_video.py $OUT/s42/65536 --episodes 0,39 --out $RO/videos/osmesa > $LG/V2_video_osmesa.log 2>&1"
wait
# C (after S1)
$J "C1_score_as_is" "$PY -I $T/pilot4_score.py $OUT --seeds 42 --json $RO/score/score_as_is.json > $LG/C1_score_as_is.log 2>&1"
$J "C2_score_validation_primary_65536" "$PY -I $T/pilot4_score.py $OUT --seeds 42 --validation --primary 65536 --straight $RO/straight/s42_65536.json --habit $RO/habit/s42_65536_seed.json --json $RO/score/score_validation_65536.json > $LG/C2_score_validation_65536.log 2>&1"
$J "tree_clean_end_readouts" "git -C $TREE status --porcelain --ignored --untracked-files=all > $LG/tree_status_end.txt && test ! -s $LG/tree_status_end.txt"
