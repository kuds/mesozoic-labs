#!/bin/bash
S=/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad
V=/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance5/k3diag/verify_3
ISO=$S/foot/stance3/tools/iso.py
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPYCACHEPREFIX=$S/caches/pyc MPLCONFIGDIR=$S/caches/mpl
r5() { ISO_WT=$S/fix/scratch_copband nice -n 19 python3 $ISO $V/v3roll.py --arm LTSB --run stance5_LTSB_s42 --npz-dir $V/npz --conds t3hold0nz "$@"; }
( r5 --steps final --out $V/q4.jsonl > $V/q4.log 2>&1 ) &
( r5 --steps 7340032 --out $V/q2.jsonl > $V/q2.log 2>&1 ) &
( r5 --steps 10485760 --out $V/q3.jsonl > $V/q3.log 2>&1 ) &
( r5 --steps 8388608 --out $V/q1.jsonl > $V/q1.log 2>&1 ) &
wait
echo ALLDONE > $V/all2.done
