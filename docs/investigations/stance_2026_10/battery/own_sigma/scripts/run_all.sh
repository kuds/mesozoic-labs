#!/bin/bash
# Own-sigma battery: statue first, then two workers (at most 2 heavy processes). Report-only offline evaluation.
S=scratchpad
O=$S/foot/battery/own_sigma
export PYTHONPYCACHEPREFIX=$S/caches/pyc MPLCONFIGDIR=$S/caches/mpl PYTHONPATH=$S/fix/main_ro:$S/tbpkgs MUJOCO_GL=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd $O/work
RUNS=$S/foot/symmetry/screen/runs
one() { # label mode model vecnorm
  local OUT=$O/panels/$1_$2
  [ -f $OUT/panel.json ] && { echo "skip $1 $2" >> $O/logs/queue.log; return; }
  echo "start $(date -Is) $1 $2" >> $O/logs/queue.log
  if [ "$2" = statue ]; then
    python3 -P $O/scripts/sigma_panel.py --label $1 --mode statue --out $OUT > $O/logs/$1_$2.log 2>&1
  else
    python3 -P $O/scripts/sigma_panel.py --label $1 --mode $2 --model $3 --vecnorm $4 --statue-ref $O/panels/statue_statue/panel.json --out $OUT > $O/logs/$1_$2.log 2>&1
  fi
  echo "end $(date -Is) $1 $2 rc=$?" >> $O/logs/queue.log
}
six() { echo $RUNS/sym_lean_settled_s$1/resume6m/models/stage1_6291456_steps.zip $RUNS/sym_lean_settled_s$1/resume6m/models/stage1_vecnormalize_6291456_steps.pkl; }
rb() { echo $RUNS/sym_lean_settled_s$1/models/robust_best_model.zip $RUNS/sym_lean_settled_s$1/models/robust_best_model_vecnorm.pkl; }
one statue statue
[ -f $O/panels/statue_statue/panel.json ] || { echo "statue failed"; exit 1; }
workerA() { for s in 60 52 42; do one s${s}_6M stochastic $(six $s); one s${s}_6M deterministic $(six $s); done; }
workerB() { one s60_rb524288 stochastic $(rb 60); one s60_rb524288 deterministic $(rb 60)
            one s52_rb917504 stochastic $(rb 52); one s52_rb917504 deterministic $(rb 52)
            one s42_rb655360 stochastic $(rb 42); one s42_rb655360 deterministic $(rb 42); }
workerA & workerB &
wait
echo ALLDONE >> $O/logs/queue.log
