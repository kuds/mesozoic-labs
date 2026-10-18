#!/bin/bash
S=scratchpad
O=scratchpad/foot/next_signals/howclose
until [ -f $O/cert_r9armL_s42_6291456_optoff/run_summary.json ] && [ -f $O/cert_r9armL_s52_6291456_optoff/run_summary.json ] && [ -f $O/cert_r9armL_s60_6291456_optoff/run_summary.json ] && [ -f $O/cert_r9armL_s60_6291456_optarmL/run_summary.json ]; do
  if grep -q Traceback $O/logs/cert_s*_opt*.log 2>/dev/null; then echo "round-1 error"; exit 1; fi
  sleep 20
done
cd $S/fix/main_ro
export PYTHONPYCACHEPREFIX=$S/caches/pyc MPLCONFIGDIR=$S/caches/mpl PYTHONPATH=$S/fix/main_ro:$S/tbpkgs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
audit() {
  R=$S/foot/symmetry/screen/runs/sym_lean_settled_s$1/resume6m/models
  python3 $O/scripts/run_cert.py --model $R/stage1_6291456_steps.zip --vecnorm $R/stage1_vecnormalize_6291456_steps.pkl --out $O/audit7042_r9armL_s$1_6291456_optoff --options off --seed 7042 > $O/logs/audit7042_s$1.log 2>&1
}
audit 60 & audit 52 &
wait
echo AUDITDONE
