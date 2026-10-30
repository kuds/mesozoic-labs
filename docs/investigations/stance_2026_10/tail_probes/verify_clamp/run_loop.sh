#!/bin/bash
# Loop sweep (two processes), then the static probe (one).  Writes only under verify_clamp.
set -u
S=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad
cd $S/com_revisit/notes/verify_clamp
export PYTHONPATH=$S/com_revisit/notes/policies/code_copy:$S/com_revisit/notes/verify_clamp OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="nice -n 10 $S/venv/bin/python -B"
A=s42_A_1101,s42_A_1111,s42_A_1121,s42_A_1131,s42_A_1141,s42_A_1151,s42_B_1201,s42_B_1211,s42_B_1221,s42_B_1231,s42_B_1241,s42_B_1251,start_s42
B=s80_A_1301,s80_A_1311,s80_A_1321,s80_A_1331,s80_A_1341,s80_A_1351,s80_B_1401,s80_B_1411,s80_B_1421,s80_B_1431,s80_B_1441,s80_B_1451
( $PY verify_loop.py out/loop_s42.jsonl --who $A; echo "L1 exit $?" ) > out/loop_p1.log 2>&1 &
( $PY verify_loop.py out/loop_s80.jsonl --who $B; echo "L2 exit $?" ) > out/loop_p2.log 2>&1 &
wait
$PY static_tail.py > static_tail.txt 2> static_tail.err; echo "static exit $?" >> static_tail.err
date -u +%FT%TZ > out/loop_finished
