#!/bin/bash
# Tolerance lane: the straightness tool's validation records with the edited tool (TWIN_HEADING_TOL_DEG 1e-4), in
# parallel, each one a logged job (job.sh; statuses in val/status.txt): the entry baseline on each 8M parent pair
# (pilot-4 tree 0e6a57b, the committed pilot-4 env, labels by pilot 3's pinned scorer; v3's pre-launch ENTRY BASELINE)
# and the --validation seed runs on pilot 3's judged 8M cells (pilot 3's tree 31d02e42, read only). The fix lane's
# run_val.sh with this lane's paths. Nothing trains (the tool blocks every training entry point).
W=$(ls -d /tmp/*/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work)
LANE=$W/pilot4/prelaunch/tol
T=$W/pilot4/tools
PY=$W/pilot3/branch/venv_full/bin/python
V=$LANE/val
export JOB_STATUS=$V/status.txt
J="bash $LANE/scripts/job.sh"
mkdir -p $V $LANE/tmp/mpl
for s in 42 52 60; do
  M=$W/pilot3/runs/final/pilot3_walk_s$s/03_locomotion/models
  $J "baseline_s$s" "cd $T/tree && env -u PYTHONPATH ISO_WT=$T/tree MPLCONFIGDIR=$LANE/tmp/mpl OMP_NUM_THREADS=1 nice -n 10 $PY -I $T/iso.py $T/pilot4_straight.py baseline --seed $s --checkpoint $M/stage2_8000000_steps.zip --vecnorm $M/stage2_vecnormalize_8000000_steps.pkl --json $V/final_baseline_s$s.json --plot $V/final_baseline_s$s.png --traces $V/final_baseline_s$s.npz --zero-nudge-check --expect-head 0e6a57bb4d6ba30aef36dd9aa30ff4d175786cc8 --labels-from $W/pilot3/tools/pilot3_score.py > $V/final_baseline_s$s.log 2> $V/final_baseline_s$s.err" &
  $J "p3val_s$s" "cd $W/pilot3/tools/tree && env -u PYTHONPATH ISO_WT=$W/pilot3/tools/tree MPLCONFIGDIR=$LANE/tmp/mpl OMP_NUM_THREADS=1 nice -n 10 $PY -I $T/iso.py $T/pilot4_straight.py seed --seed $s --run $W/pilot3/runs/final/pilot3_walk_s$s --cell $W/pilot3/judging/OUT/s$s/8000000 --validation --json $V/final_p3val_s$s.json --plot $V/final_p3val_s$s.png --traces $V/final_p3val_s$s.npz > $V/final_p3val_s$s.log 2> $V/final_p3val_s$s.err" &
done
wait
echo "run_val done"
cat $V/status.txt
