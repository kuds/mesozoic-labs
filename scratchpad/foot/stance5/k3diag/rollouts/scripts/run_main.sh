#!/bin/bash
# Sequential K3 rollouts (one process at a time, nice 19, one thread).
S=/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad
O=/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance5/k3diag/rollouts
ISO=$S/foot/stance3/tools/iso.py
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPYCACHEPREFIX=$S/caches/pyc MPLCONFIGDIR=$S/caches/mpl
B="full noise0 legs tail neckhead notail"
r5() { ISO_WT=$S/fix/scratch_copband nice -n 19 python3 $ISO $O/scripts/k3roll.py --arm LTSB "$@" --out-dir $O/npz; }
r4() { ISO_WT=$S/fix/screen_s4_LTS nice -n 19 python3 $ISO $O/scripts/k3roll.py --arm LTS "$@" --out-dir $O/npz; }
r5 --run stance5_LTSB_s42 --steps final --conds legs tail neckhead notail
r5 --run stance5_LTSB_s42 --steps 10485760 6291456 --conds $B
r5 --run stance5_LTSB_s60 --steps final --conds $B
r5 --run stance5_LTSB_s52 --steps final --conds $B
r4 --run stance4_LTS_s52 --steps final --conds $B
r4 --run stance4_LTS_s60 --steps final --conds $B
r5 --run stance5_LTSB_s42 --steps 7340032 8388608 4194304 5242880 --conds $B
echo ALLDONE
