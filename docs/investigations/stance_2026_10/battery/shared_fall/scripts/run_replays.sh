#!/bin/bash
S=scratchpad
O=scratchpad/foot/battery/shared_fall
cd $O/work
export PYTHONPATH=$S/fix/main_ro:$S/tbpkgs MUJOCO_GL=egl PYTHONPYCACHEPREFIX=$S/caches/pyc MPLCONFIGDIR=$S/caches/mpl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p $O/logs $O/replays
run() { # seedlabel
  R=$S/foot/symmetry/screen/runs/sym_lean_settled_s$1/resume6m/models
  python3 -P $O/scripts/fall_replay.py --role s$1_6M --model $R/stage1_6291456_steps.zip --vecnorm $R/stage1_vecnormalize_6291456_steps.pkl --yaws=-90,-45,0,45,90,180 --full-yaws 0 --out $O/replays > $O/logs/replay_s$1_6M.log 2>&1
}
run 60 & run 52 &
wait
python3 -P $O/scripts/fall_replay.py --role statue --yaws=-90,-45,0,45,90,180 --full-yaws 0 --out $O/replays > $O/logs/replay_statue.log 2>&1
echo REPLAYSDONE
