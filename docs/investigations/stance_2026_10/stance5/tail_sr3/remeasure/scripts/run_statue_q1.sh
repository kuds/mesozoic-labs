#!/bin/bash
# queue 1: stance/locomotion/behavior statue on the branch (one process)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
P=scratchpad2/venv/bin/python; R=scratchpad2/prs/r10; B=scratchpad2/prs/r10_work/build
cd $R
for st in 1 2 3; do
  echo "== zab stage $st start $(date -u)"
  (date -u; git -C $R rev-parse HEAD; git -C $R status --short | wc -l; nice -n 10 $P -B environments/shared/scripts/zero_action_baseline.py --model-id tyrannosaurus_rex anatomical --stage $st --episodes 40 --seed 3042; echo exit $?; date -u) > $B/remeasure/zab_stage${st}_r10.log 2>&1
  echo "== zab stage $st done $(date -u)"
done
echo "== sqb start $(date -u)"
(date -u; nice -n 10 $P -B -m environments.shared.scripts.stance_quality_baseline 0.05 40 --model-id tyrannosaurus_rex anatomical; echo exit $?; date -u) > $B/remeasure/sqb_stance_r10.log 2>&1
echo "== q1 done $(date -u)"
