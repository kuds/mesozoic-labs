#!/bin/bash
H=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/com_revisit/notes/early_bias
S=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad
until grep -q CHAIN-DONE $H/eb_logs/chain.log; do sleep 20; done
cd $H && PYTHONPATH=$S/r10round/ecs/code OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 $S/venv/bin/python -B $H/eb_realized.py > $H/eb_logs/realized.out 2> $H/eb_logs/realized.err
cd $H && PYTHONPATH=$S/r10round/ecs/code OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 $S/venv/bin/python -B $H/eb_gradnorm.py > $H/eb_logs/gradnorm.out 2> $H/eb_logs/gradnorm.err
echo CHAIN2-DONE
