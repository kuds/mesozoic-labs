#!/bin/bash
# The open-loop replay and statue arms, queued after run_all.sh (waits for its run.finished): two processes at most.
set -u
S=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad
cd $S/com_revisit/notes/clamp
until [ -e run.finished ]; do sleep 20; done
export PYTHONPATH=$S/com_revisit/notes/policies/code_copy OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="nice -n 10 $S/venv/bin/python -B run_replay.py"
LENS=$S/com_revisit/notes/policies/run
date -u +%FT%TZ > replay.started
(
  $PY replay/check --lens $LENS --only s42_A_1101,s80_B_1401 --seeds 3042-3043
  $PY replay/s42_held0 --group s42 --lens $LENS --force 0
  $PY replay/statue --statue
  echo "R1 exit $?"
) > replay_p1.log 2>&1 &
(
  $PY replay/s80_forced1 --group s80 --lens $LENS --force 1
  $PY replay/statue_forced1 --statue --force 1
  echo "R2 exit $?"
) > replay_p2.log 2>&1 &
wait
date -u +%FT%TZ > replay.finished
