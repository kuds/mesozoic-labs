#!/bin/bash
# Context only: the two 6M policies at heading 0 on the audit panel 7042-7081 (the seeds the audits rolled).
S=scratchpad
O=scratchpad/foot/battery/shared_fall
cd $O/work
export PYTHONPATH=$S/fix/main_ro:$S/tbpkgs MUJOCO_GL=egl PYTHONPYCACHEPREFIX=$S/caches/pyc MPLCONFIGDIR=$S/caches/mpl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
SEEDS=$(python3 -c "print(','.join(str(s) for s in range(7042,7082)))")
mkdir -p $O/logs $O/context
run() {
  R=$S/foot/symmetry/screen/runs/sym_lean_settled_s$1/resume6m/models
  python3 -P $O/scripts/fall_replay.py --role ctx_s$1_6M --model $R/stage1_6291456_steps.zip --vecnorm $R/stage1_vecnormalize_6291456_steps.pkl --seeds $SEEDS --yaws 0 --full-yaws "" --out $O/context > $O/logs/context_s$1_6M.log 2>&1
}
run 60 & run 52 &
wait
python3 -P $O/scripts/fall_replay.py --role ctx_statue --seeds $SEEDS --yaws 0 --full-yaws "" --out $O/context > $O/logs/context_statue.log 2>&1
echo CONTEXTDONE
