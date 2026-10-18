#!/bin/bash
S=scratchpad
O=scratchpad/foot/next_signals/howclose
cd $S/fix/main_ro
export PYTHONPYCACHEPREFIX=$S/caches/pyc MPLCONFIGDIR=$S/caches/mpl PYTHONPATH=$S/fix/main_ro:$S/tbpkgs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
run() { # seed options
  R=$S/foot/symmetry/screen/runs/sym_lean_settled_s$1/resume6m/models
  python3 $O/scripts/run_cert.py --model $R/stage1_6291456_steps.zip --vecnorm $R/stage1_vecnormalize_6291456_steps.pkl --out $O/cert_r9armL_s$1_6291456_opt$2 --options $2 > $O/logs/cert_s$1_opt$2.log 2>&1
}
mkdir -p $O/logs
run 42 off & run 52 off & run 60 off & run 60 armL &
wait
run 42 armL & run 52 armL &
wait
echo ALLDONE
