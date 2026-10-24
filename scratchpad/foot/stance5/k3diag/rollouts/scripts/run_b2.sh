#!/bin/bash
# Batch 2 (sequential, after batch 1): leg sub-group ablations, the tail_3 counterfactual, stance4 s42 final.
S=/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad
O=/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance5/k3diag/rollouts
ISO=$S/foot/stance3/tools/iso.py
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPYCACHEPREFIX=$S/caches/pyc MPLCONFIGDIR=$S/caches/mpl
until grep -q ALLDONE $O/logs/run_main.log; do sleep 30; done
SUB="hip_pitch hip_roll knee ankle legs_no_ankle legs_no_hiproll legs_no_hippitch legs_no_knee"
B="full noise0 legs tail neckhead notail"
r5() { ISO_WT=$S/fix/scratch_copband nice -n 19 python3 $ISO $O/scripts/k3roll.py --arm LTSB "$@" --out-dir $O/npz; }
r4() { ISO_WT=$S/fix/screen_s4_LTS nice -n 19 python3 $ISO $O/scripts/k3roll.py --arm LTS "$@" --out-dir $O/npz; }
r5 --run stance5_LTSB_s42 --steps final 10485760 --conds $SUB
r5 --run stance5_LTSB_s60 --steps final --conds $SUB
r5 --run stance5_LTSB_s42 --steps final 10485760 6291456 --conds t3hold0_full t3hold0_noise0
r4 --run stance4_LTS_s42 --steps final --conds $B
r5 --run stance5_LTSB_s42 --steps 7340032 8388608 --conds $SUB
echo ALLDONE
