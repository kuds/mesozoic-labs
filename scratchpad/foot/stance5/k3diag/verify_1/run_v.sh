#!/bin/bash
S=/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad
V=/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance5/k3diag/verify_1
ISO=$S/foot/stance3/tools/iso.py
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPYCACHEPREFIX=$S/caches/pyc MPLCONFIGDIR=$S/caches/mpl
r5() { ISO_WT=$S/fix/scratch_copband nice -n 19 python3 $ISO $V/vroll.py --arm LTSB "$@"; }
r4() { ISO_WT=$S/fix/screen_s4_LTS nice -n 19 python3 $ISO $V/vroll.py --arm LTS "$@"; }
( r5 --run stance5_LTSB_s42 --steps 8388608 --conds full hip_pitch hpA hpS hpD hpC fullD fullC tail neckhead legs_no_hippitch --out $V/p1.jsonl > $V/p1.log 2>&1; echo DONE >> $V/p1.log ) &
( r5 --run stance5_LTSB_s42 --steps 7340032 final --conds full hip_pitch hpD hpC fullD fullC --out $V/p2.jsonl > $V/p2.log 2>&1; echo DONE >> $V/p2.log ) &
( r5 --run stance5_LTSB_s42 --steps 10485760 --conds full hpD hpC fullD fullC --out $V/p3.jsonl > $V/p3.log 2>&1; r5 --run stance5_LTSB_s60 --steps final --conds full hpD hpC fullD fullC --out $V/p3.jsonl >> $V/p3.log 2>&1; echo DONE >> $V/p3.log ) &
( r4 --run stance4_LTS_s42 --steps final --conds full knee legs_no_knee hip_pitch hpD hpC fullD fullC --out $V/p4.jsonl > $V/p4.log 2>&1; echo DONE >> $V/p4.log ) &
wait
echo ALLDONE > $V/all.done
