#!/bin/bash
# Tolerance lane: after phase 0, the fix lane's by-hand job remove_setup_model_duplicates (its CMDLOG line 416),
# scripted: the set-up path's saved models are byte copies of the stand-in runs' pairs (dedupe_models.py says
# all_equal), so the duplicates under setup/s*/models are removed. Through job.sh. Nothing trains.
SP=$(ls -d /tmp/*/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad)
LANE=$SP/work/pilot4/prelaunch/tol
J="bash $LANE/scripts/job.sh"
$J remove_setup_model_duplicates "grep -q '\"all_equal\": true' $LANE/e2e/logs/dedupe_models.log && rm -rf $LANE/setup/s42/models $LANE/setup/s52/models $LANE/setup/s60/models && test ! -e $LANE/setup/s42/models"
