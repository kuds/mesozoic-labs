#!/bin/bash
cd /tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/com_revisit/notes/verify_early_bias
export PYTHONPATH=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/ecs/code OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
nice -n 10 /tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/venv/bin/python -B vb_update.py s42 1310720 7001 40 /tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/com_revisit/notes/verify_early_bias/ckpt/s42_stage1_1310720_steps.zip /tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/com_revisit/notes/verify_early_bias/ckpt/s42_stage1_vecnormalize_1310720_steps.pkl > logs/a1.log 2>&1
nice -n 10 /tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/venv/bin/python -B vb_update.py s42 2097152 7101 24 > logs/a2.log 2>&1
nice -n 10 /tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/venv/bin/python -B vb_update.py s42 3145728 7201 16 > logs/a3.log 2>&1
