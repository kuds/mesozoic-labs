#!/bin/bash
# k3readout_one5r.sh RUN STEPS [--arm LTSB|LTSBr] [--check-only] : the report-only K3 mechanism readouts of one
# checkpoint (R/readouts_addendum_k3.txt governs; R = SP/foot/stance5/replication).  REPORT ONLY: no verdict, no decision.
#   RUN    --arm LTSBr (default, the replication): stance5_LTSB_s70 | stance5_LTSB_s80
#          --arm LTSB  (validation / reference on stance5): stance5_LTSB_s42 | stance5_LTSB_s52 | stance5_LTSB_s60
#   STEPS  a multiple of 524288, or final.  The addendum's slots: LTSBr 6291456 and final ("readout"); LTSB s42 6291456,
#          7340032 and final, s52 and s60 final ("reference").  Any other slot is rolled but its output says
#          addendum_slot=no (the tool prints a NOTE): a measurement, not one of the addendum's readouts.
# Runs R/k3readouts/k3readout5r.py through SP/foot/stance3/tools/iso.py with ISO_WT=SP/fix/scratch_copband (clean at
# cf80ecb1; read-only), nice -n 19, OMP/MKL/OPENBLAS threads 1, PYTHONPYCACHEPREFIX/MPLCONFIGDIR under SP/caches, from
# R/k3readouts (never from the main checkout).
# At most two k3readout runs at a time (two slot locks under logs/); a third call exits 9 and runs nothing.  The slot is
# taken before the judge-file check, so the tool always runs under a slot.
# Refuses (exit 3) unless the run's judge file exists: R/judged/RUN/judge_STEPS.json (LTSBr) or
# SP/foot/stance5/judged/RUN/judge_STEPS.json (LTSB).  The refusal is the tool's: it writes out/RUN/refused_STEPS.json
# (never overwritten) and the runner adds a "refused" line to logs/runner.log.  The tool also refuses (exit 3, refused
# JSON) on the identity checks.  Otherwise it writes out/RUN/k3r_STEPS.json (or the next free k3r_STEPS.<n>.json; never
# overwrites) and exits 0 when the output is valid ("full" reproduces the judge file), 4 when it is written but not
# valid.  Exit 2: bad arguments (nothing run).  Exit 1 or other: a crash (the runner says so on stderr).
# Readers select outputs by their "valid" (and "addendum_slot") fields, never by file name.
# Logs: R/k3readouts/logs/RUN_k3r_STEPS_<utc>_<pid>[_check].log (one per call; noclobber) and logs/runner.log.
# --check-only: every identity check and the checkpoint load; writes no output (the log is still written).
S=scratchpad
K=$S/foot/stance5/replication/k3readouts
CL=$S/fix/scratch_copband
ISO=$S/foot/stance3/tools/iso.py
TOOL=$K/k3readout5r.py
usage() { awk 'NR>1 && !/^#/{exit} NR>1{sub(/^# ?/,""); print}' "$0"; }
case "$1" in -h|--help|"") usage; exit 2;; esac
[ $# -ge 2 ] || { usage; exit 2; }
RUN=$1; STEPS=$2; shift 2
ARM=LTSBr; EXTRA=""
while [ $# -gt 0 ]; do
  case "$1" in
    --arm) [ $# -ge 2 ] || { echo "--arm needs a value (LTSB or LTSBr)"; usage; exit 2; }; ARM=$2; shift 2;;
    --arm=*) ARM=${1#--arm=}; shift;;
    --check-only) EXTRA=--check-only; shift;;
    *) echo "unknown argument $1"; usage; exit 2;;
  esac
done
case "$ARM:$RUN" in
  LTSBr:stance5_LTSB_s70|LTSBr:stance5_LTSB_s80) ROOT=$S/foot/stance5/replication;;
  LTSB:stance5_LTSB_s42|LTSB:stance5_LTSB_s52|LTSB:stance5_LTSB_s60) ROOT=$S/foot/stance5;;
  *) echo "RUN $RUN is not a run of arm $ARM"; exit 2;;
esac
case "$STEPS" in final) ;; *[!0-9]*|"") echo "bad STEPS $STEPS"; exit 2;;
  *) [ $((STEPS % 524288)) -eq 0 ] && [ "$STEPS" -gt 0 ] || { echo "STEPS $STEPS is not a multiple of 524288"; exit 2; };; esac
mkdir -p $K/logs
JF=$ROOT/judged/$RUN/judge_$STEPS.json
# at most two k3readout rollouts at a time
exec 9>$K/logs/.slot1.lock
if ! flock -n 9; then
  exec 9>$K/logs/.slot2.lock
  if ! flock -n 9; then
    echo "busy $(date -Is) $ARM $RUN $STEPS: two k3readout runs are active" >> $K/logs/runner.log
    echo "refused: two k3readout runs are already active (at most two at a time)"; exit 9
  fi
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPYCACHEPREFIX=$S/caches/pyc MPLCONFIGDIR=$S/caches/mpl
export K3R_RUNNER=$(realpath "$0")
unset K3R_TEST_ROOT
cd $K || exit 2
LOG=$K/logs/${RUN}_k3r_${STEPS}_$(date -u +%Y%m%dT%H%M%SZ)_$$${EXTRA:+_check}.log
# a missing judge file is refused by the tool itself (it writes out/RUN/refused_STEPS.json); the slot is already held,
# so a judge file that appears meanwhile is rolled under it
[ -f "$JF" ] || echo "the judge file $JF does not exist: the tool will refuse and record it (judge the slot first)"
echo "start $(date -Is) $ARM $RUN $STEPS ${EXTRA} log $LOG" >> $K/logs/runner.log
set -o noclobber  # a per-call log is never overwritten
ISO_WT=$CL nice -n 19 python3 $ISO $TOOL --arm $ARM --run $RUN --steps $STEPS $EXTRA > $LOG 2>&1
rc=$?
set +o noclobber
echo "end $(date -Is) $ARM $RUN $STEPS rc=$rc" >> $K/logs/runner.log
[ $rc -eq 3 ] && echo "refused $(date -Is) $ARM $RUN $STEPS: $(grep -m1 '^K3R ' $LOG 2>/dev/null)" >> $K/logs/runner.log
grep '^K3R \|exists; wrote\|NOTE:' $LOG 2>/dev/null | tail -3
case $rc in
  0) ;;
  3) echo "k3readout refused (rc=3); the reasons are in the K3R line above and in $LOG" >&2;;
  4) echo "k3readout wrote an output that is NOT valid (rc=4); see the K3R line above and $LOG" >&2;;
  *) echo "k3readout FAILED rc=$rc, see $LOG" >&2;;
esac
exit $rc
