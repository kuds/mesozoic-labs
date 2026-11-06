#!/bin/bash
# e2e lane, phase C: pilot4_score.py over the three seeds' stand-in cells with the twelve straightness records, in
# four input states (no review and no look; a stand-in video review "agree" x3; that plus a stand-in look "straight"
# x3; that plus a look "not straight" x3) and once more with the habit records; then pilot4_video.py on s42's
# 2,000,000 cell from the scorer's listing (MUJOCO_GL=disable is binding here, so rendering is expected to fail).
# Every command goes through job.sh (CMDLOG.txt + e2e/status.txt). Nothing trains.
SP=/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad
T=$SP/work/pilot4/tools
LANE=$T/out/e2e
PY=$SP/work/pilot3/branch/venv_full/bin/python
TREE=$T/tree
OUT=$LANE/e2e/OUT
LG=$LANE/e2e/logs
ST=$LANE/e2e/straight
HB=$LANE/e2e/habit
SC=$LANE/e2e/score
IN=$LANE/e2e/inputs
J="bash $LANE/scripts/job.sh"
mkdir -p "$SC" "$IN"
$J "make_input_review" "printf '{\"42\": \"agree\", \"52\": \"agree\", \"60\": \"agree\"}\n' > $IN/video_review_agree_standin.json"
$J "make_input_look_straight" "printf '{\"42\": \"straight\", \"52\": \"straight\", \"60\": \"straight\"}\n' > $IN/look_straight_standin.json"
$J "make_input_look_not_straight" "printf '{\"42\": \"not straight\", \"52\": \"not straight\", \"60\": \"not straight\"}\n' > $IN/look_not_straight_standin.json"
STR=""
for s in 42 52 60; do for k in 500000 1000000 1500000 2000000; do STR="$STR --straight $ST/s${s}_${k}.json"; done; done
SCORE="$PY -I -B $T/pilot4_score.py $OUT --seeds 42,52,60 $STR"
$J "score_no_review_no_look" "$SCORE --json $SC/score_no_review_no_look.json > $SC/score_no_review_no_look.txt 2>&1"
$J "score_review" "$SCORE --video-review $IN/video_review_agree_standin.json --json $SC/score_review.json > $SC/score_review.txt 2>&1"
$J "score_review_look_straight" "$SCORE --video-review $IN/video_review_agree_standin.json --straight-review $IN/look_straight_standin.json --json $SC/score_review_look_straight.json > $SC/score_review_look_straight.txt 2>&1"
$J "score_review_look_not_straight" "$SCORE --video-review $IN/video_review_agree_standin.json --straight-review $IN/look_not_straight_standin.json --json $SC/score_review_look_not_straight.json > $SC/score_review_look_not_straight.txt 2>&1"
$J "score_review_look_straight_habit" "$SCORE --video-review $IN/video_review_agree_standin.json --straight-review $IN/look_straight_standin.json --habit $HB/habit_seed_s42.json --habit $HB/habit_seed_s52.json --habit $HB/habit_seed_s60.json --json $SC/score_review_look_straight_habit.json > $SC/score_review_look_straight_habit.txt 2>&1"
$J "score_without_straight_records" "$PY -I -B $T/pilot4_score.py $OUT --seeds 42,52,60 --video-review $IN/video_review_agree_standin.json --json $SC/score_review_no_straight.json > $SC/score_review_no_straight.txt 2>&1"
$J "video_s42_2000000_from_score" "cd $TREE && ISO_WT=$TREE $PY -I $T/iso.py $T/pilot4_video.py $OUT/s42/2000000 --from-score $SC/score_review.json --out $LANE/e2e/videos > $LG/video_s42_2000000.log 2>&1"
$J "tree_clean_after_phase_c" "git -C $TREE status --porcelain --ignored > $LG/tree_status_after_phase_c.txt && test ! -s $LG/tree_status_after_phase_c.txt"
