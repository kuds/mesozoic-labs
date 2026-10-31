#!/bin/bash
H=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/com_revisit/notes/early_bias
S=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad
until grep -q CHAIN3-DONE $H/eb_logs/chain3.log 2>/dev/null; do sleep 15; done
cd $H && PYTHONPATH=$S/r10round/ecs/code OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 $S/venv/bin/python -B $H/eb_proxy.py > $H/eb_logs/proxy.out 2> $H/eb_logs/proxy.err
cd $H && PYTHONPATH=$S/r10round/ecs/code OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 $S/venv/bin/python -B $H/eb_realized_s60early.py > $H/eb_logs/realized_s60early.out 2> $H/eb_logs/realized_s60early.err
echo CHAIN4-DONE
