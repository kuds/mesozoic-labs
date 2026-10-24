#!/bin/bash
# Batch 6 (sequential, after batch 5): differential hip-pitch pulse amplitude sweep (0.1, 0.2).
S=/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad
O=/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance5/k3diag/rollouts
ISO=$S/foot/stance3/tools/iso.py
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPYCACHEPREFIX=$S/caches/pyc MPLCONFIGDIR=$S/caches/mpl
until grep -q ALLDONE $O/logs/run_b5.log; do sleep 30; done
r5() { ISO_WT=$S/fix/scratch_copband nice -n 19 python3 $ISO $O/scripts/k3roll.py --arm LTSB "$@" --out-dir $O/npz; }
r4() { ISO_WT=$S/fix/screen_s4_LTS nice -n 19 python3 $ISO $O/scripts/k3roll.py --arm LTS "$@" --out-dir $O/npz; }
r5 --run stance5_LTSB_s42 --steps 6291456 7340032 10485760 final --conds pulse_anti10 pulse_anti20
r5 --run stance5_LTSB_s52 --steps final --conds pulse_anti10 pulse_anti20
r5 --run stance5_LTSB_s60 --steps final --conds pulse_anti10 pulse_anti20
r4 --run stance4_LTS_s52 --steps final --conds pulse_anti10 pulse_anti20
r4 --run stance4_LTS_s60 --steps final --conds pulse_anti10 pulse_anti20
r4 --run stance4_LTS_s42 --steps final --conds pulse_anti10 pulse_anti20
echo ALLDONE
