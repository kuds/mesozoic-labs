#!/bin/bash
# Runs eb_update.py over a job list (RUN STEPS SEED per line), at most 2 processes at a time.
S=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad
H=$S/com_revisit/notes/early_bias
cd $H
export PYTHONPATH=$S/r10round/ecs/code OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
xargs -P 2 -L 1 bash -c 'f=$H/eb_out2/$0_$1_$2.json; [ -s "$f" ] || nice -n 10 $S/venv/bin/python -B $H/eb_update.py $0 $1 $2 >> $H/eb_logs/$0_$1.out 2>> $H/eb_logs/$0_$1.err' < "$1"
