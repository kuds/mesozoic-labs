#!/bin/bash
# Step 3 closed-loop swaps, one process at a time, in series.  Writes only under probe/.
set -u
S=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad
P=$S/com_revisit/notes/probe
cd $P
export PYTHONPATH=$S/com_revisit/notes/policies/code_copy:$P OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="nice -n 10 $S/venv/bin/python -B"
date -u +%FT%TZ > swaps.started
$PY run_swaps.py swaps --lin s42 --conds own,endW_startN,endW_startN_tail,startW_endN,startW_endN_tail > swaps_s42.log 2>&1; echo "exit $?" >> swaps_s42.log
$PY run_swaps.py swaps --lin s80 --conds own,endW_startN,endW_startN_tail,startW_endN,startW_endN_tail > swaps_s80.log 2>&1; echo "exit $?" >> swaps_s80.log
date -u +%FT%TZ > swaps.finished
