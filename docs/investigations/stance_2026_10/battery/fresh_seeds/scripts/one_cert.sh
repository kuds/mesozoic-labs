#!/bin/bash
# one offline six-heading certificate: $1 = run seed (52|60), $2 = ckpt (6291456|robustbest), $3 = first eval seed
S=scratchpad
F=$S/foot/battery/fresh_seeds
RUNNER=$S/foot/next_signals/howclose/scripts/run_cert.py   # the earlier runner, used read-only
export PYTHONPYCACHEPREFIX=$S/caches/pyc MPLCONFIGDIR=$S/caches/mpl PYTHONPATH=$S/fix/main_ro:$S/tbpkgs MUJOCO_GL=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd $F/work
D=$S/foot/symmetry/screen/runs/sym_lean_settled_s$1
if [ "$2" = 6291456 ]; then M=$D/resume6m/models/stage1_6291456_steps.zip; V=$D/resume6m/models/stage1_vecnormalize_6291456_steps.pkl
else M=$D/models/robust_best_model.zip; V=$D/models/robust_best_model_vecnorm.pkl; fi
OUT=$F/cert$3_r9armL_s$1_$2_optoff
[ -f $OUT/run_summary.json ] && { echo "skip $OUT"; exit 0; }
echo "start $(date -Is) s$1 $2 seed $3" >> $F/logs/queue.log
python3 -P $RUNNER --model $M --vecnorm $V --out $OUT --options off --seed $3 > $F/logs/cert$3_s$1_$2.log 2>&1
echo "end $(date -Is) s$1 $2 seed $3 rc=$?" >> $F/logs/queue.log
