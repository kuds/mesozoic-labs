#!/bin/bash
# Resume after the 20:50 UTC restart, part 2 (queue 2 never started: no q2.started, no plant_* or run/long_* files).
# Waits for queue 3 (the away arms).  One simulation process at a time.
#  1. test 3 static balance and open-loop replays (run_plant.py), as queue 2 planned;
#  2. ADDED on resume: the s42 "toward" pitch illusion on the gravity and accelerometer channels only (prey direction
#     untouched), because the observation-only sweep put about half of the pitch pathway on the prey channel;
#  3. the 20 s long-episode check of the s42 illusion (run_long.py), as queue 2 planned.
set -u
S=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad
P=$S/com_revisit/notes/posture_policy
cd $P
until [ -f q3.finished ]; do sleep 20; done
export PYTHONPATH=$S/com_revisit/notes/policies/code_copy:$P OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="nice -n 15 $S/venv/bin/python -B"
S42=s42_A_1101,s42_A_1111,s42_A_1121,s42_A_1131,s42_A_1141,s42_A_1151,s42_B_1201,s42_B_1211,s42_B_1221,s42_B_1231,s42_B_1241,s42_B_1251
date -u +%FT%TZ > q4.started
rm -f plant_static.json plant_replay.jsonl
$PY run_plant.py static run/base plant_static.json > q4_static.log 2>&1; echo "static exit $?" >> q4_static.log
$PY run_plant.py replay run/base plant_replay.jsonl > q4_replay.log 2>&1; echo "replay exit $?" >> q4_replay.log
$PY run_base.py run/towardgrav_s42 --pitch 1.10 --channels grav,acc --who start_s42,$S42 > q4_towardgrav.log 2>&1; echo "towardgrav_s42 exit $?" >> q4_towardgrav.log
for sd in 3042 3046 3050 3055 3061; do
  $PY run_long.py run/long_base --steps 2000 --seeds $sd-$sd --who start_s42,$S42 >> q4_long.log 2>&1
  $PY run_long.py run/long_toward --steps 2000 --pitch 1.10 --seeds $sd-$sd --who start_s42,$S42 >> q4_long.log 2>&1
  mkdir -p run/long_base/arrays_$sd run/long_toward/arrays_$sd
  mv run/long_base/arrays/*.npz run/long_base/arrays_$sd/ 2>/dev/null; mv run/long_toward/arrays/*.npz run/long_toward/arrays_$sd/ 2>/dev/null
done
echo "long done" >> q4_long.log
date -u +%FT%TZ > q4.finished
