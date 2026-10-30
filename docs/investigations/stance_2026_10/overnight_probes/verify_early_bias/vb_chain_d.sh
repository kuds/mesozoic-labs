#!/bin/bash
cd /tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/com_revisit/notes/verify_early_bias
while pgrep -f vb_chain_b.sh > /dev/null; do sleep 20; done
export PYTHONPATH=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/ecs/code OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
nice -n 10 /tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/venv/bin/python -B vb_realized.py > logs/d1.log 2>&1
nice -n 10 /tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/venv/bin/python -B vb_update.py s80 1310720 7041 16 /tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/com_revisit/notes/verify_early_bias/ckpt/s80_stage1_1310720_steps.zip /tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/com_revisit/notes/verify_early_bias/ckpt/s80_stage1_vecnormalize_1310720_steps.pkl > logs/d2.log 2>&1
