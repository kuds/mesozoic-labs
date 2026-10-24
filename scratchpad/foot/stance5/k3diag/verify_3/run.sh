#!/bin/bash
S=/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad
V=/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance5/k3diag/verify_3
ISO=$S/foot/stance3/tools/iso.py
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPYCACHEPREFIX=$S/caches/pyc MPLCONFIGDIR=$S/caches/mpl
r5() { ISO_WT=$S/fix/scratch_copband nice -n 19 python3 $ISO $V/v3roll.py --arm LTSB --run stance5_LTSB_s42 --npz-dir $V/npz "$@"; }
( r5 --steps final --out $V/p4.jsonl > $V/p4.log 2>&1; echo DONE >> $V/p4.log ) &
( r5 --steps 7340032 6291456 --out $V/p2.jsonl > $V/p2.log 2>&1; echo DONE >> $V/p2.log ) &
( r5 --steps 10485760 8388608 --out $V/p3.jsonl > $V/p3.log 2>&1; echo DONE >> $V/p3.log ) &
( r5 --steps 4194304 5242880 --out $V/p1.jsonl > $V/p1.log 2>&1; echo DONE >> $V/p1.log ) &
wait
echo ALLDONE > $V/all.done
