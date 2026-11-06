#!/bin/bash
# Tolerance lane: after run_val.sh, the records the tests read (val_final: the three baseline JSONs and traces and the
# three pilot-3 validation JSONs, byte copies, as the fix lane's val_final), their comparison with the fix lane's
# records, the scorer's own acceptance of them (tool sha256, constants, recomputed verdicts) and the facts for the
# text. Each step a logged job (val/status.txt). Reads and copies files only; nothing trains.
SP=$(ls -d /tmp/*/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad)
T=$SP/work/pilot4/tools
LANE=$SP/work/pilot4/prelaunch/tol
PY=$SP/work/pilot3/branch/venv_full/bin/python
V=$LANE/val
VF=$LANE/val_final
export JOB_STATUS=$V/status.txt
J="bash $LANE/scripts/job.sh"
mkdir -p $VF
for s in 42 52 60; do
  $J "val_final_copy_s$s" "cp -p $V/final_baseline_s$s.json $VF/baseline_s$s.json && cp -p $V/final_baseline_s$s.npz $VF/baseline_s$s.npz && cp -p $V/final_p3val_s$s.json $VF/p3val_s$s.json && cmp $V/final_baseline_s$s.json $VF/baseline_s$s.json && cmp $V/final_baseline_s$s.npz $VF/baseline_s$s.npz && cmp $V/final_p3val_s$s.json $VF/p3val_s$s.json"
done
$J cmp_tol_records "$PY -I -B $LANE/scripts/cmp_tol_records.py $T/out/fix/val_final $VF $LANE/logs/cmp_tol_records.json > $LANE/logs/cmp_tol_records.txt 2>&1"
$J scorer_accepts_records "$PY -I -B -c 'import glob, json, sys; sys.path.insert(0, sys.argv[1]); import pilot4_score as ps; rows = [(p.split(\"/\")[-1], json.load(open(p))[\"tool_sha256\"] == ps.STRAIGHT_TOOL_SHA256 == ps.file_sha256(sys.argv[1] + \"/pilot4_straight.py\"), json.load(open(p))[\"constants\"] == ps.STRAIGHT_CONSTANTS, ps.recomputed_verdicts(json.load(open(p))[\"parts\"]) == {k: json.load(open(p))[\"parts\"][k][\"verdict\"] for k in \"abc\"}) for p in sorted(glob.glob(sys.argv[2] + \"/*.json\"))]; [print(*r) for r in rows]; sys.exit(0 if all(all(r[1:]) for r in rows) and len(rows) == 6 else 1)' $T $VF > $LANE/logs/scorer_accepts_records.txt 2>&1"
$J facts_tol "$PY -I -B $LANE/scripts/facts_tol.py $VF $LANE/logs/facts_tol.json > $LANE/logs/facts_tol.log 2>&1"
cat $V/status.txt
