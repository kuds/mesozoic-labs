#!/bin/bash
# The clamp test's queue: at most two simulation processes at a time, nice 10, one thread each.
set -u
S=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad
cd $S/com_revisit/notes/clamp
export PYTHONPATH=$S/com_revisit/notes/policies/code_copy OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="nice -n 10 $S/venv/bin/python -B run_clamp.py"
LENS=$S/com_revisit/notes/policies/run
date -u +%FT%TZ > run.started
(
  $PY run/s42_unforced --group s42 --compare $LENS
  $PY run/s80_unforced --group s80 --compare $LENS
  $PY long/s42_unforced --group s42 --steps 2000 --only s42_A_1101,s42_A_1151,s42_B_1201
  $PY long/s42_held0 --group s42 --steps 2000 --force 0 --only s42_A_1101,s42_A_1151,s42_B_1201
  echo "P1 exit $?"
) > run_p1.log 2>&1 &
(
  $PY run/s42_held0 --group s42 --force 0
  $PY run/s80_forced1 --group s80 --force 1
  $PY long/s80_unforced --group s80 --steps 2000 --only s80_A_1301,s80_A_1351,s80_B_1401
  $PY long/s80_forced1 --group s80 --steps 2000 --force 1 --only s80_A_1301,s80_A_1351,s80_B_1401
  echo "P2 exit $?"
) > run_p2.log 2>&1 &
wait
date -u +%FT%TZ > run.finished
