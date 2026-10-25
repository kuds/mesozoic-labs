#!/bin/bash
# queue 2: recovery pushed statue on the branch, then the same command on the origin/main archive (control)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
P=scratchpad2/venv/bin/python; B=scratchpad2/prs/r10_work/build
cd scratchpad2/prs/r10
echo "== recovery r10 start $(date -u)"
(date -u; nice -n 10 $P -B -m environments.shared.harnesses.recovery_offdist_panel --controller statue --schedule on --safe-set provisional --out $B/remeasure/recovery_statue_r10.json; echo exit $?; date -u) > $B/remeasure/recovery_statue_r10.log 2>&1
cd $B/tree_main
echo "== recovery main start $(date -u)"
(date -u; nice -n 10 $P -B -m environments.shared.harnesses.recovery_offdist_panel --controller statue --schedule on --safe-set provisional --out $B/remeasure/recovery_statue_main_r9.json; echo exit $?; date -u) > $B/remeasure/recovery_statue_main_r9.log 2>&1
echo "== q2 done $(date -u)"
