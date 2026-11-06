#!/bin/bash
# Tolerance lane: the fix lane's by-hand run of the frozen probe on the drifted copy (its CMDLOG line 800, [437]),
# scripted: 1 panel episode on tampered/env_drift's 2,000,000 pair with its stage_config.json, so e2e_fix_checks.py
# can show the probe only warns and records the drift the judge refuses on. Through job.sh. Nothing trains.
SP=$(ls -d /tmp/*/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad)
T=$SP/work/pilot4/tools
LANE=$SP/work/pilot4/prelaunch/tol
PY=$SP/work/pilot3/branch/venv_full/bin/python
TREE=$T/tree
D=$LANE/tampered/env_drift/pilot4_walk_s42/03_locomotion
J="bash $LANE/scripts/job.sh"
mkdir -p $LANE/data $LANE/logs
$J probe_on_drift "cd $TREE && env -u PYTHONPATH ISO_WT=$TREE OMP_NUM_THREADS=1 MPLCONFIGDIR=$LANE/tmp/mpl $PY -I $T/iso.py $T/gait_probe_pilot.py --embodiment anatomical -- --species tyrannosaurus_rex --stage locomotion --episodes 1 --seed-scheme panel --seed-base 3042 --settle-steps 100 --checkpoint $D/models/stage2_2000000_steps.zip --vecnorm $D/models/stage2_vecnormalize_2000000_steps.pkl --run-id drift_check --out $LANE/data/probe_on_drift --stage-config-json $D/stage_config.json > $LANE/logs/probe_on_drift.log 2>&1"
$J probe_on_drift_warned "grep -q 'WARNING: env kwargs drifted' $LANE/logs/probe_on_drift.log"
$J tree_clean_after_probe_on_drift "git -C $TREE status --porcelain --ignored > $LANE/e2e/logs/tree_status_after_probe_on_drift.txt && test ! -s $LANE/e2e/logs/tree_status_after_probe_on_drift.txt"
