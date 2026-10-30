#!/bin/bash
# After queue 1: test 3 (static balance, open-loop replays), then the 20 s long-episode check of the s42 illusion.
# One simulation process at a time: waits for q1.finished.
set -u
S=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad
P=$S/com_revisit/notes/posture_policy
cd $P
until [ -f q1.finished ]; do sleep 15; done
export PYTHONPATH=$S/com_revisit/notes/policies/code_copy:$P OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="nice -n 15 $S/venv/bin/python -B"
S42=s42_A_1101,s42_A_1111,s42_A_1121,s42_A_1131,s42_A_1141,s42_A_1151,s42_B_1201,s42_B_1211,s42_B_1221,s42_B_1231,s42_B_1241,s42_B_1251
date -u +%FT%TZ > q2.started
$PY run_plant.py static run/base plant_static.json > q2_static.log 2>&1; echo "static exit $?" >> q2_static.log
$PY run_plant.py replay run/base plant_replay.jsonl > q2_replay.log 2>&1; echo "replay exit $?" >> q2_replay.log
for sd in 3042 3046 3050 3055 3061; do
  $PY run_long.py run/long_base --steps 2000 --seeds $sd-$sd --who start_s42,$S42 >> q2_long.log 2>&1
  $PY run_long.py run/long_toward --steps 2000 --pitch 1.10 --seeds $sd-$sd --who start_s42,$S42 >> q2_long.log 2>&1
  mkdir -p run/long_base/arrays_$sd run/long_toward/arrays_$sd
  mv run/long_base/arrays/*.npz run/long_base/arrays_$sd/ 2>/dev/null; mv run/long_toward/arrays/*.npz run/long_toward/arrays_$sd/ 2>/dev/null
done
echo "long exit $?" >> q2_long.log
date -u +%FT%TZ > q2.finished
