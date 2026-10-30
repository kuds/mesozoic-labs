#!/bin/bash
# Verifier's queue: two simulation processes at most, nice 10, one thread each.  Writes only under verify_clamp/out.
set -u
S=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad
cd $S/com_revisit/notes/verify_clamp
export PYTHONPATH=$S/com_revisit/notes/policies/code_copy OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="nice -n 10 $S/venv/bin/python -B verify_sim.py"
D9M=/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6/fetch/F2/stance6_r10LTSB_s42/copies/01_stance/models
S42=s42_A_1121,s42_B_1221,s42_B_1251
S80=s80_A_1321,s80_B_1411,s80_B_1441
date -u +%FT%TZ > out/started
(
  $PY out/s42_unforced.jsonl --who $S42
  $PY out/s42_held0.jsonl --who $S42 --force 0
  $PY out/s42_hold05.jsonl --who $S42 --force 0.5
  $PY out/d9_held0.jsonl --who d9_onset --force 0 --seeds 3042,3050,3061 --zip $D9M/stage1_3670016_steps.zip.copy1 --pkl $D9M/stage1_vecnormalize_3670016_steps.pkl.copy1
  $PY out/start_s42_unforced.jsonl --who start_s42
  $PY out/start_s42_held0.jsonl --who start_s42 --force 0
  echo "P1 exit $?"
) > out/p1.log 2>&1 &
(
  $PY out/s80_unforced_long.jsonl --who $S80 --steps 2000
  $PY out/s80_forced1_long.jsonl --who $S80 --force 1 --steps 2000
  $PY out/start_s42_forced1.jsonl --who start_s42 --force 1 --seeds 3042,3043,3044,3045,3046,3047,3048,3049,3050,3051,3052,3053,3054,3055,3056,3057,3058,3059,3060,3061
  echo "P2 exit $?"
) > out/p2.log 2>&1 &
wait
date -u +%FT%TZ > out/finished
