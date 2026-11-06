#!/bin/bash
# Fix lane: the straightness tool's validation records with the final tool, in parallel (each one logged command):
# the entry baseline on each 8M parent (pilot-4 tree 0e6a57b) and the --validation seed runs on pilot 3's 8M cells
# (pilot 3's tree 31d02e42, read only). Nothing trains (the tool blocks every training entry point).
F=/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/pilot4/tools/out/fix
S=/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad
T=$S/work/pilot4/tools
PY=$S/work/pilot3/branch/venv_full/bin/python
mkdir -p $F/val $F/tmp/mpl
for s in 42 52 60; do
  M=$S/work/pilot3/runs/final/pilot3_walk_s$s/03_locomotion/models
  $F/L.sh "cd $T/tree && env -u PYTHONPATH ISO_WT=$T/tree MPLCONFIGDIR=$F/tmp/mpl OMP_NUM_THREADS=1 $PY -I $T/iso.py $T/pilot4_straight.py baseline --seed $s --checkpoint $M/stage2_8000000_steps.zip --vecnorm $M/stage2_vecnormalize_8000000_steps.pkl --json $F/val/final_baseline_s$s.json --plot $F/val/final_baseline_s$s.png --traces $F/val/final_baseline_s$s.npz --zero-nudge-check --expect-head 0e6a57bb4d6ba30aef36dd9aa30ff4d175786cc8 --labels-from $S/work/pilot3/tools/pilot3_score.py > $F/val/final_baseline_s$s.log 2> $F/val/final_baseline_s$s.err" &
  $F/L.sh "cd $S/work/pilot3/tools/tree && env -u PYTHONPATH ISO_WT=$S/work/pilot3/tools/tree MPLCONFIGDIR=$F/tmp/mpl OMP_NUM_THREADS=1 $PY -I $T/iso.py $T/pilot4_straight.py seed --seed $s --run $S/work/pilot3/runs/final/pilot3_walk_s$s --cell $S/work/pilot3/judging/OUT/s$s/8000000 --validation --json $F/val/final_p3val_s$s.json --plot $F/val/final_p3val_s$s.png --traces $F/val/final_p3val_s$s.npz > $F/val/final_p3val_s$s.log 2> $F/val/final_p3val_s$s.err" &
done
wait
echo "run_val done"
