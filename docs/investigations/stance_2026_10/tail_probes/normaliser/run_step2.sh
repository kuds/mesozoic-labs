#!/bin/bash
# Step 2 rollouts (one process).  Writes only under probe/.
set -u
S=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad
P=$S/com_revisit/notes/probe
cd $P
export PYTHONPATH=$S/com_revisit/notes/policies/code_copy:$P OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
date -u +%FT%TZ > ckpts.started
nice -n 10 $S/venv/bin/python -B run_ckpts.py ckpts --lin s42,s80 --seeds 3042 > ckpts.log 2>&1
echo "exit $?" >> ckpts.log
date -u +%FT%TZ > ckpts.finished
