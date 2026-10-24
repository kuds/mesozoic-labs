#!/bin/bash
# panels5r_one.sh [--dry-run] [--wait] [--any-slot] STEPS RUN_ID : report-only diagnostic panels for one stance5
# replication checkpoint (SP/foot/stance5/tools/panels5_one.sh with the runs s70/s80 and R = SP/foot/stance5/replication:
# RUN_DIR=R/runs/RUN_ID, PANELS_OUT=R/panels/out; panels_one5.sh and panels5.py are stance5's, unchanged, read-only).
#   STEPS  a J checkpoint: 4194304 5242880 6291456 8388608 10485760 final (--any-slot also allows other placed slots)
#   RUN_ID stance5_LTSB_s70 | stance5_LTSB_s80
# Wraps S5/panels/panels_one5.sh STEPS RUN_ID scratch_copband --expect-head <the config commit> with RUN_DIR=R/runs/RUN_ID, nice -n 19,
# the statue, and the four variants the criteria name for the band runs (PRE-REGISTERED READOUTS: "--variants standard
# mirror_legs prey_ahead mirror_prey_ahead"; the standard variant carries the yaw and step-200 stagger watches).
# Never part of a verdict.
# Output: R/panels/out/RUN_ID/panels_STEPS.json; logs in R/panels/out/logs/ (queue.log, RUN_ID_STEPS.log) and one line
# per job in R/panels/out/logs/panels5_replication.log.
# One panels process at a time (the budget is one): if another panels5.py or panels4.py is running (for example
# a stance4 control panel), this exits 3, or with --wait it waits until that one ends.
# Resumable: a complete output is skipped (exit 0); an unreadable or incomplete one is moved aside to *.bad.<time>.
# --check exits 0 when the output is complete, 1 when it is not, 3 when the checkpoint is not placed.  Writes nothing.
# --dry-run prints the checks and the command, and runs nothing.  About 14 minutes per checkpoint on one CPU (measured
# on stance3 Run 3 8.39M: 814 s for the four variants with their statues).
S=/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad
S5=$S/foot/stance5; P=$S5/panels; R=$S5/replication
CL=$S/fix/scratch_copband
HEAD_EXPECT=cf80ecb14d1630ce3b4dcdea28a05835d143470b
J="4194304 5242880 6291456 8388608 10485760 final"
usage() { awk 'NR>1 && !/^#/{exit} NR>1{sub(/^# ?/,""); print}' "$0"; }
DRY=0; WAIT=0; ANY=0
CHECK=0
while [ $# -gt 0 ]; do case "$1" in -h|--help) usage; exit 0;; --dry-run) DRY=1; shift;; --check) CHECK=1; shift;; --wait) WAIT=1; shift;;
  --any-slot) ANY=1; shift;; *) break;; esac; done
[ $# -eq 2 ] || { usage; exit 2; }
STEPS=$1; RUN=$2
case "$RUN" in stance5_LTSB_s70|stance5_LTSB_s80) ;; *) echo "bad RUN_ID $RUN"; exit 2;; esac
if ! echo " $J " | grep -q " $STEPS "; then
  [ $ANY -eq 1 ] || { echo "STEPS $STEPS is not a J checkpoint ($J); --any-slot to run it anyway (report only)"; exit 2; }
fi
RD=$R/runs/$RUN; M=$RD/01_stance/models; OUT=$R/panels/out/$RUN/panels_$STEPS.json; LOGD=$R/panels/out/logs; LOG=$LOGD/panels5_replication.log
case "$STEPS" in final) Z=$M/stage1_final.zip; PK=$M/stage1_final_vecnorm.pkl;;
  best|robust_best) Z=$M/${STEPS}_model.zip; PK=$M/${STEPS}_model_vecnorm.pkl;;
  *) Z=$M/stage1_${STEPS}_steps.zip; PK=$M/stage1_vecnormalize_${STEPS}_steps.pkl;; esac

valid() {  # the same completeness test as queue_controls.sh, for this clone
  python3 -I - "$1" "$2" "$HEAD_EXPECT" <<'EOF'
import json, sys
path, steps, head = sys.argv[1:4]
try:
    with open(path) as fh:
        d = json.load(fh)
except Exception as exc:
    print(f"unreadable: {exc}"); sys.exit(1)
bad = []
if d.get("schema") != "stance5-panels/v1": bad.append("schema")
if d.get("report_only") is not True: bad.append("report_only")
if d.get("smoke") is not False: bad.append("smoke")
if d.get("statue_rolled") is not True: bad.append("statue_rolled")
if str(d.get("steps")) != steps: bad.append(f"steps {d.get('steps')}")
if (d.get("git") or {}).get("head") != head: bad.append("head")
if (d.get("git") or {}).get("status"): bad.append("dirty clone")
if (d.get("binding") or {}).get("checkpoint_task_match") is not True: bad.append("task")
v = d.get("variants") or {}
for name in ("standard", "mirror_legs", "prey_ahead", "mirror_prey_ahead"):
    if not isinstance(v.get(name), dict) or "clean" not in v[name]: bad.append(f"variant {name}")
    elif not (v[name].get("band") or {}).get("window_foot_steps") and (v[name].get("band") or {}).get("full_horizon_episodes"):
        bad.append(f"variant {name}: no band readouts")
if "seconds" not in d: bad.append("seconds")
if bad:
    print("incomplete: " + ", ".join(bad)); sys.exit(1)
EOF
}
other_panels() { pgrep -f '(^|[ /])panels[45]\.py( |$)'; }

if [ "$HEAD_EXPECT" = PLACEHOLDER_HEAD_LTSB ] && [ $DRY -eq 0 ] && [ $CHECK -eq 0 ]; then
  echo "HEAD_EXPECT is a PLACEHOLDER: fill it with the config commit first"; exit 4
fi
if [ $CHECK -eq 1 ]; then
  [ -e $OUT ] && valid $OUT $STEPS >/dev/null 2>&1 && exit 0
  for f in $RD/01_stance/task_fingerprint.json $Z $PK; do [ -f $f ] || exit 3; done
  exit 1
fi
VARIANTS="standard mirror_legs prey_ahead mirror_prey_ahead"
CMD="RUN_DIR=$RD PANELS_OUT=$R/panels/out nice -n 19 $P/panels_one5.sh $STEPS $RUN $CL --expect-head $HEAD_EXPECT --variants $VARIANTS"
if [ $DRY -eq 1 ]; then
  [ "$(git -C $CL rev-parse HEAD 2>/dev/null)" = "$HEAD_EXPECT" ] && echo "clone HEAD $HEAD_EXPECT: ok" || echo "PROBLEM: clone HEAD is not $HEAD_EXPECT"
  [ -z "$(git -C $CL status --porcelain 2>/dev/null)" ] && echo "clone clean: ok" || echo "PROBLEM: clone has uncommitted changes (panels5.py refuses)"
  for f in $RD/01_stance/task_fingerprint.json $Z $PK; do [ -f $f ] && echo "placed: $(basename $f)" || echo "MISSING: $f"; done
  if [ -e $OUT ]; then n=$(valid $OUT $STEPS 2>&1) && echo "output complete: would skip" || echo "output exists but $n: would move aside and rerun"
  else echo "output $OUT: not yet written"; fi
  [ -n "$(other_panels)" ] && echo "another panels5.py is running now: $(other_panels | tr '\n' ' ') (exit 3, or wait with --wait)"
  echo "WOULD RUN: $CMD"
  exit 0
fi

mkdir -p $LOGD
if [ -e $OUT ]; then
  if valid $OUT $STEPS >/dev/null 2>&1; then echo "$(date -Is) skip-done $RUN $STEPS" >> $LOG; exit 0; fi
  mv $OUT $OUT.bad.$(date +%Y%m%dT%H%M%S); echo "$(date -Is) moved aside an incomplete $OUT" >> $LOG
fi
for f in $RD/01_stance/task_fingerprint.json $Z $PK; do [ -f $f ] || { echo "$(date -Is) missing $RUN $STEPS ($f)" >> $LOG; exit 3; }; done
waited=0
while [ -n "$(other_panels)" ]; do
  [ $WAIT -eq 1 ] || { echo "$(date -Is) busy: another panels5.py is running; $RUN $STEPS not started" >> $LOG; exit 3; }
  [ $((waited % 600)) -eq 0 ] && echo "$(date -Is) waiting for another panels5.py ($(other_panels | tr '\n' ' '))" >> $LOG
  sleep 30; waited=$((waited + 30))
done
JL=$LOGD/${RUN}_$STEPS.log; [ -e $JL ] && mv $JL $JL.prev.$(date +%Y%m%dT%H%M%S)
echo "$(date -Is) run $RUN $STEPS" >> $LOG
t0=$(date +%s)
RUN_DIR=$RD PANELS_OUT=$R/panels/out nice -n 19 $P/panels_one5.sh $STEPS $RUN $CL --expect-head $HEAD_EXPECT --variants $VARIANTS
rc=$?
if [ $rc -eq 0 ] && ! valid $OUT $STEPS >/dev/null 2>&1; then rc=4; fi
echo "$(date -Is) end $RUN $STEPS rc=$rc $(( $(date +%s) - t0 ))s" >> $LOG
exit $rc
