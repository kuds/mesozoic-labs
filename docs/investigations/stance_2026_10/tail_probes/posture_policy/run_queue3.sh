#!/bin/bash
# Resume after the 20:50 UTC container restart: queue 1's base, toward_s42 and toward_s80 finished complete (checked:
# 27/13/13 policies x 20 seeds, every npz loads).  The two "away" arms never started (q1_away.log empty); run them.
# One simulation process at a time.
set -u
S=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad
P=$S/com_revisit/notes/posture_policy
cd $P
export PYTHONPATH=$S/com_revisit/notes/policies/code_copy:$P OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="nice -n 15 $S/venv/bin/python -B"
S42=s42_A_1101,s42_A_1111,s42_A_1121,s42_A_1131,s42_A_1141,s42_A_1151,s42_B_1201,s42_B_1211,s42_B_1221,s42_B_1231,s42_B_1241,s42_B_1251
S80=s80_A_1301,s80_A_1311,s80_A_1321,s80_A_1331,s80_A_1341,s80_A_1351,s80_B_1401,s80_B_1411,s80_B_1421,s80_B_1431,s80_B_1441,s80_B_1451
date -u +%FT%TZ > q3.started
$PY run_base.py run/away_s42 --pitch -1.10 --who start_s42,$S42 > q1_away.log 2>&1; echo "away_s42 exit $?" >> q1_away.log
$PY run_base.py run/away_s80 --pitch 1.10 --who start_s80,$S80 >> q1_away.log 2>&1; echo "away_s80 exit $?" >> q1_away.log
date -u +%FT%TZ > q3.finished
