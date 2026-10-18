#!/bin/bash
# Part B: push-and-release telemetry sweep (main's rollout replicated, read-only telemetry), queued roles.
S=scratchpad
P=$S/foot/battery/push
cd $P/work
export PYTHONPATH=$S/fix/main_ro:$S/tbpkgs MUJOCO_GL=egl PYTHONPYCACHEPREFIX=$S/caches/pyc MPLCONFIGDIR=$S/caches/mpl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p $P/logs $P/telemetry
run() { # role seed
  if [ "$1" = statue ]; then
    python3 -P $P/scripts/push_trace.py --role statue --out $P/telemetry > $P/logs/push_statue.log 2>&1
  else
    R=$S/foot/symmetry/screen/runs/sym_lean_settled_s$2/resume6m/models
    python3 -P $P/scripts/push_trace.py --role $1 --model $R/stage1_6291456_steps.zip --vecnorm $R/stage1_vecnormalize_6291456_steps.pkl --out $P/telemetry > $P/logs/push_$1.log 2>&1
  fi
}
for job in "$@"; do
  case $job in
    statue) run statue ;;
    s60) run s60_6M 60 ;;
    s52) run s52_6M 52 ;;
    s42) run s42_6M 42 ;;
  esac
done
echo PUSHDONE "$@"
