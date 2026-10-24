#!/bin/bash
S=/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad
V=/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance5/k3diag/verify_2
ISO=$S/foot/stance3/tools/iso.py
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPYCACHEPREFIX=$S/caches/pyc MPLCONFIGDIR=$S/caches/mpl
f5() { ISO_WT=$S/fix/scratch_copband nice -n 19 python3 $ISO $V/v2_fb.py --arm LTSB --run stance5_LTSB_s42 --conds full --out-dir $V/fb "$@"; }
( f5 --steps 7340032 --kfb 0.53; f5 --steps 7340032 --kfb 1.0; echo DONE ) > $V/logs/fb1.log 2>&1 &
( f5 --steps final --kfb 0.7; f5 --steps final --kfb 1.0; echo DONE ) > $V/logs/fb2.log 2>&1 &
( f5 --steps 10485760 --kfb 0.87; f5 --steps 8388608 --kfb 0.5; echo DONE ) > $V/logs/fb3.log 2>&1 &
( f5 --steps 6291456 --kfb -0.53; f5 --steps 6291456 --kfb -0.87; echo DONE ) > $V/logs/fb4.log 2>&1 &
wait
echo ALLDONE > $V/logs/fb.done
