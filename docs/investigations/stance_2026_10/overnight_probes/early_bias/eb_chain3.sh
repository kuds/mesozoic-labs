#!/bin/bash
H=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/com_revisit/notes/early_bias
S=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad
export H S
until grep -q CHAIN2-DONE $H/eb_logs/chain2.log 2>/dev/null; do sleep 20; done
$H/eb_jobs.sh $H/eb_jobs_wave4.txt
cd $H && PYTHONPATH=$S/r10round/ecs/code OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 $S/venv/bin/python -B $H/eb_gradnorm.py > $H/eb_logs/gradnorm.out 2> $H/eb_logs/gradnorm.err
echo CHAIN3-DONE
