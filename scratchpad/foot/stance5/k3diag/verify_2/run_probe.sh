#!/bin/bash
S=/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad
V=/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance5/k3diag/verify_2
ISO=$S/foot/stance3/tools/iso.py
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPYCACHEPREFIX=$S/caches/pyc MPLCONFIGDIR=$S/caches/mpl
r5() { ISO_WT=$S/fix/scratch_copband nice -n 19 python3 $ISO $V/v2_probe.py --arm LTSB "$@" --out-dir $V/probe; }
r4() { ISO_WT=$S/fix/screen_s4_LTS nice -n 19 python3 $ISO $V/v2_probe.py --arm LTS "$@" --out-dir $V/probe; }
C="full noise0 hpA hpD"
( r5 --run stance5_LTSB_s42 --steps 4194304 5242880 --conds $C; r4 --run stance4_LTS_s42 --steps final --conds full noise0 hpA; echo DONE ) > $V/logs/p1.log 2>&1 &
( r5 --run stance5_LTSB_s42 --steps 6291456 --conds $C; r5 --run stance5_LTSB_s42 --steps 7340032 --conds noise0 hpA hpD; r4 --run stance4_LTS_s52 --steps final --conds full noise0; echo DONE ) > $V/logs/p2.log 2>&1 &
( r5 --run stance5_LTSB_s42 --steps 8388608 10485760 --conds $C; r4 --run stance4_LTS_s60 --steps final --conds full noise0; echo DONE ) > $V/logs/p3.log 2>&1 &
( r5 --run stance5_LTSB_s42 --steps final --conds $C; r5 --run stance5_LTSB_s52 --steps final --conds full noise0 hpA; r5 --run stance5_LTSB_s60 --steps final --conds full noise0 hpA; echo DONE ) > $V/logs/p4.log 2>&1 &
wait
echo ALLDONE > $V/logs/all.done
