#!/bin/bash
# The start-checkpoint arms, after the D-9 reproduction (waits for its exit line): two processes at most.
S=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad
cd $S/com_revisit/notes/clamp
until grep -q "^exit" d9_repro.err 2>/dev/null; do sleep 20; done
export PYTHONPATH=$S/com_revisit/notes/policies/code_copy OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="nice -n 10 $S/venv/bin/python -B run_start.py"
LENS=$S/com_revisit/notes/policies/run
(
  $PY start/s42_unforced --group s42 --only start_s42 --compare $LENS
  $PY start/s42_held0 --group s42 --only start_s42 --force 0
  echo "S1 exit $?"
) > start_p1.log 2>&1 &
(
  $PY start/s80_unforced --group s80 --only start_s80 --compare $LENS
  $PY start/s80_forced1 --group s80 --only start_s80 --force 1
  echo "S2 exit $?"
) > start_p2.log 2>&1 &
wait
date -u +%FT%TZ > start.finished
