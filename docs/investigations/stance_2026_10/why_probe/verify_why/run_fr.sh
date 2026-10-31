#!/bin/bash
# verify_why fresh rollouts (PREREG Part F): two lanes, nice 10, at most 2 simulation processes.
V=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/com_revisit/notes/verify_why
PY=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/venv/bin/python
cd $V || exit 1
( nice -n 10 $PY -I -B vw_roll.py --label C11E --mode stoch --seed 8810 --rollouts 6 --out runs/FR1_C11E_stoch.npz > logs/FR1.log 2>&1; echo "FR1 exit $?" >> logs/lanes.log
  nice -n 10 $PY -I -B vw_roll.py --label C11E --mode det --seed 8820 --episodes-per-env 6 --out runs/FR2_C11E_det.npz > logs/FR2.log 2>&1; echo "FR2 exit $?" >> logs/lanes.log ) &
( nice -n 10 $PY -I -B vw_roll.py --label P2 --mode stoch --seed 8830 --rollouts 6 --out runs/FR3_P2_stoch.npz > logs/FR3.log 2>&1; echo "FR3 exit $?" >> logs/lanes.log ) &
wait
echo "all done $(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> logs/lanes.log
