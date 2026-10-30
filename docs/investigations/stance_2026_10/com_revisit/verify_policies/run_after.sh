#!/bin/bash
# Verifier's runs after the panel, one simulation process at a time.
set -u
S=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad
V=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/com_revisit/notes/verify_policies
PY="nice -n 10 $S/venv/bin/python -B"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd $V
date -u +%FT%TZ
PYTHONPATH=$V/r9_tree $PY run_r9_plant.py $V/r9_on_r9_plant.jsonl > $V/r9_on_r9_plant.log 2>&1; echo "r9 plant exit $?"
date -u +%FT%TZ
PYTHONPATH=$V/main_tree $PY run_verify.py long $V/sim_long > $V/sim_long.log 2>&1; echo "long exit $?"
date -u +%FT%TZ
