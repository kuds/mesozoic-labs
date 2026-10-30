#!/bin/bash
# Verifier rollouts, one process at a time, in series.  Writes only under verify_probe/.
set -u
V=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/com_revisit/notes/verify_probe
cd $V
export PYTHONPATH=$V/code:$V OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="nice -n 10 /tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/venv/bin/python -B"
date -u +%FT%TZ > v_loop.started
$PY v_loop.py ckpts v_ckpts.jsonl > v_ckpts.log 2>&1; echo "exit $?" >> v_ckpts.log
$PY v_loop.py swaps v_swaps.jsonl > v_swaps.log 2>&1; echo "exit $?" >> v_swaps.log
date -u +%FT%TZ > v_loop.finished
