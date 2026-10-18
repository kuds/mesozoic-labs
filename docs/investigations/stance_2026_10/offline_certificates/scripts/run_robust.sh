#!/bin/bash
S=scratchpad
O=scratchpad/foot/next_signals/howclose
cd $S/fix/main_ro
export PYTHONPYCACHEPREFIX=$S/caches/pyc MPLCONFIGDIR=$S/caches/mpl PYTHONPATH=$S/fix/main_ro:$S/tbpkgs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
rb() {
  R=$S/foot/symmetry/screen/runs/sym_lean_settled_s$1/models
  python3 $O/scripts/run_cert.py --model $R/robust_best_model.zip --vecnorm $R/robust_best_model_vecnorm.pkl --out $O/cert_r9armL_s$1_robustbest_optoff --options off > $O/logs/cert_robustbest_s$1.log 2>&1
}
rb 42 & rb 52 & rb 60 &
wait
echo RBDONE
