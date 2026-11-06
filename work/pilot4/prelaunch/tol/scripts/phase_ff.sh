#!/bin/bash
# Tolerance lane (copied from the fix lane): the final-model fallback cases the e2e lane ran by hand after make_final.py (its CMDLOG lines 738-744),
# scripted: --final-fallback --check-only accepted; the same copy without the flag refused; a final zip at 2,000,000
# refused; nothing written. Every command goes through job.sh. Nothing trains.
SP=$(ls -d /tmp/*/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad)
T=$SP/work/pilot4/tools
LANE=$SP/work/pilot4/prelaunch/tol
PY=$SP/work/pilot3/branch/venv_full/bin/python
TREE=$T/tree
HEAD=0e6a57bb4d6ba30aef36dd9aa30ff4d175786cc8
TP=$LANE/tampered
LG=$LANE/e2e/logs
P42=sha256:41fc0b0cd7167ab25a2567779b89f431b829f6a2bae4c72fa3c89192732e87c4
J="bash $LANE/scripts/job.sh"
JUDGE="cd $TREE && ISO_WT=$TREE $PY -I $T/iso.py $T/pilot4_judge.py"
$J final_fallback_check_only_expect0 "$JUDGE $TP/final_fallback/pilot4_walk_s42 --seed 42 --steps 2000000 --parent-sha256 $P42 --out $TP/OUT_final_fallback --expect-head $HEAD --final-fallback --check-only > $LG/final_fallback_check_only.log 2>&1"
$J final_fallback_without_flag_expect2 "$JUDGE $TP/final_fallback/pilot4_walk_s42 --seed 42 --steps 2000000 --parent-sha256 $P42 --out $TP/OUT_final_fallback --expect-head $HEAD --check-only > $LG/final_fallback_without_flag.log 2>&1"
$J final_fallback_short_expect2 "$JUDGE $TP/final_fallback_short/pilot4_walk_s42 --seed 42 --steps 2000000 --parent-sha256 $P42 --out $TP/OUT_final_fallback_short --expect-head $HEAD --final-fallback --check-only > $LG/final_fallback_short.log 2>&1"
$J final_fallback_out_absent "test ! -e $TP/OUT_final_fallback && test ! -e $TP/OUT_final_fallback_short"
