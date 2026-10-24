#!/bin/bash
# judge_one5r.sh [--dry-run] STEPS RUN_ID SEED [MODE] : judge one stance5 replication checkpoint placed under
# R/runs/RUN_ID/01_stance (R = SP/foot/stance5/replication; the variable S5 below names R).
#   STEPS  a multiple of 524288, or final, best, robust_best
#   RUN_ID stance5_LTSB_s70 | stance5_LTSB_s80      SEED 70 | 80 (must match RUN_ID)
#   MODE   full (default; the verdict input) | quick | common-noise | tail | fresh (report only)
# Runs judge_stance5r.py --arm LTSB --seed SEED --steps STEPS on the clone fix/scratch_copband (HEAD = the config commit, clean),
# through stance3/tools/iso.py with ISO_WT=<clone>, one CPU (OMP/MKL/OPENBLAS 1; iso.py sets torch threads 1).
# Output: R/judged/RUN_ID/judge_STEPS.json (full; what score_stance5r.py reads) or judge_<mode>_STEPS.json.
# Logs: R/judged/logs/queue.log (start/end lines) and R/judged/logs/RUN_ID_<name>.log (older ones kept as *.prev.<time>).
# Resumable: a complete output is skipped (exit 0); an unreadable or incomplete one (a restart during the write) is
# moved aside to *.bad.<time> and the job reruns.  The judge never overwrites an output itself.
# --dry-run prints the checks and the command, and runs nothing.  It writes nothing.
# --check exits 0 when the output is complete, 1 when it is not, 3 when the checkpoint is not placed, 5 (full mode)
# when the slot holds a refusal record, and 6 (full mode) when it holds a refusal held for review.  Writes nothing.
# REFUSALS (criteria RUN HANDLING, the maintainer's rule of 2026-10-22 16:32 UTC): a full-mode judge that refuses on
# the run's own records (seed, commit, n_envs, hyperparameters, task, env kwargs, gate bars, plant, a record absent on
# Drive: the judge's "refusing (nothing written): ..." line naming stage_config.json, plant_identity.json or
# task_fingerprint.json) leaves R/judged/RUN_ID/refused_STEPS.json with the refusal text and exits 5.
# score_stance5r.py counts that record as an INVALID J file.  While it stands, a later call exits 5 and judges
# nothing.  A record is absent on Drive only when place5r.py config --run-records-fail listed it so
# (01_stance/records_check_failed.json); any other missing record is a local file not placed.  An operator-side
# refusal (a wrong path or clone, a missing local file: no such line, a record not placed, or a symmetry check the
# judge could not run) writes no record: fix it and judge again.  A refusal that names only the checkpoint's own
# attributes (its seed, plant, task or symmetry record) while the run's records pass is most likely a zip fetched
# from another run's folder (the operator's side): it is held for review as refused_STEPS.review.json (exit 6; the
# scorer does not read it), and nothing is judged while it stands.  Confirmed as the run's (the fetched zip's parent
# is the run's models folder): rename it to refused_STEPS.json.  Otherwise move it and the placed pair aside by hand
# and fetch the slot again.  A record found to be the operator's side is moved aside by hand (never deleted), and
# the slot is judged again.
# JUDGE5R_TEST_ROOT (tests only): runs/ and judged/ under a scratch root; the judge stays R's.
# SP/foot/stance5/tools/judge_one5.sh with the replication identity (runs s70/s80, judge_stance5r.py, R); before it,
# SP/foot/stance4/tools/judge_one4.sh with the stance5 identity.  HEAD_EXPECT is the config commit (a PLACEHOLDER until
# it is filled; while it is, nothing is judged: exit 4).
S=/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad
S3=$S/foot/stance3; S5=$S/foot/stance5/replication   # R
CL=$S/fix/scratch_copband
HEAD_EXPECT=cf80ecb14d1630ce3b4dcdea28a05835d143470b
JUDGE=$S5/judge/judge_stance5r.py
S5=${JUDGE5R_TEST_ROOT:-$S5}   # tests only: runs/ and judged/ under a scratch root (the judge above stays R's)
usage() { awk 'NR>1 && !/^#/{exit} NR>1{sub(/^# ?/,""); print}' "$0"; }
DRY=0
CHECK=0
case "$1" in -h|--help) usage; exit 0;; --dry-run) DRY=1; shift;; --check) CHECK=1; shift;; esac
[ $# -ge 3 ] && [ $# -le 4 ] || { usage; exit 2; }
STEPS=$1; RUN=$2; SEED=$3; MODE=${4:-full}
case "$RUN:$SEED" in stance5_LTSB_s70:70|stance5_LTSB_s80:80) ;; *) echo "RUN_ID $RUN and SEED $SEED do not match"; exit 2;; esac
case "$STEPS" in final|best|robust_best) ;; *[!0-9]*|"") echo "bad STEPS $STEPS"; exit 2;;
  *) [ $((STEPS % 524288)) -eq 0 ] && [ "$STEPS" -gt 0 ] || { echo "STEPS $STEPS is not a multiple of 524288"; exit 2; };; esac
case "$MODE" in full) NAME=judge_$STEPS; MFLAG="";; quick|tail|fresh) NAME=judge_${MODE}_$STEPS; MFLAG=--$MODE;;
  common-noise) NAME=judge_common_noise_$STEPS; MFLAG=--common-noise;; *) echo "bad MODE $MODE"; exit 2;; esac
RD=$S5/runs/$RUN; OUTD=$S5/judged/$RUN; OUT=$OUTD/$NAME.json; LOGD=$S5/judged/logs; JL=$LOGD/${RUN}_$NAME.log
REFUSED=$OUTD/refused_$STEPS.json   # criteria RUN HANDLING (full mode)
REVIEW=$OUTD/refused_$STEPS.review.json; MARKER=$RD/01_stance/records_check_failed.json   # (full mode)
M=$RD/01_stance/models
case "$STEPS" in final) Z=$M/stage1_final.zip; P=$M/stage1_final_vecnorm.pkl;;
  best|robust_best) Z=$M/${STEPS}_model.zip; P=$M/${STEPS}_model_vecnorm.pkl;;
  *) Z=$M/stage1_${STEPS}_steps.zip; P=$M/stage1_vecnormalize_${STEPS}_steps.pkl;; esac
MODE_KEY=${MODE//-/_}

# valid OUT : 0 when OUT is a complete judge JSON for this run, slot, mode and clone HEAD.
valid() {
  python3 -I - "$1" "$STEPS" "$MODE_KEY" "$SEED" "$HEAD_EXPECT" <<'EOF'
import json, sys
path, steps, mode, seed, head = sys.argv[1:6]
try:
    with open(path) as fh:
        d = json.load(fh)
except Exception as exc:
    print(f"unreadable: {exc}"); sys.exit(1)
bad = [k for k, ok in (("schema", d.get("schema") == "stance5r-judge/v1"), ("arm", d.get("arm") == "LTSB"),
                       ("mode", d.get("mode") == mode), ("steps", str(d.get("steps")) == steps),
                       ("seed", str(d.get("seed")) == seed), ("smoke", d.get("smoke") is False),
                       ("head", (d.get("git") or {}).get("head") == head), ("clean", (d.get("git") or {}).get("status") == ""),
                       ("finished", "judge_seconds" in d)) if not ok]
if bad:
    print("incomplete or foreign: " + ", ".join(bad)); sys.exit(1)
EOF
}

# refusal RC : 0 after writing REFUSED from the judge log when the judge refused on the run's own records; 2 after
# writing REVIEW when it refused on the checkpoint's own attributes only (held for review); else 1 (nothing written).
refusal() {
  python3 -I - "$JL" "$REFUSED" "$RUN" "$SEED" "$STEPS" "$JUDGE" "$Z" "$P" "$HEAD_EXPECT" "$1" "$REVIEW" "$MARKER" <<'EOF'
import datetime, hashlib, json, os, re, sys
log, out, run, seed, steps, judge, z, p, head, rc, review, marker = sys.argv[1:13]
PREFIX = "[judge_stance5r] refusing (nothing written): "
RECS = r"(plant_identity|stage_config|task_fingerprint)\.json"
hits = []
if os.path.exists(log):
    with open(log, errors="replace") as fh:
        hits = [x.rstrip("\n")[len(PREFIX):] for x in fh if x.startswith(PREFIX)]
if not hits:
    print("no refusal on the run's own records in the log (the operator's side, or a crash): nothing recorded")
    sys.exit(1)
items = [x.strip() for x in hits[-1].split(" | ") if x.strip()]
# A record the judge found missing is the run's own only when place5r.py config --run-records-fail listed it as absent
# on Drive (records_check_failed.json); otherwise it is a local file not placed: the operator's side.
absent = []
if os.path.exists(marker):
    with open(marker) as fh:
        absent = json.load(fh).get("absent_on_drive") or []


def missing(x):
    m = re.search(r"/" + RECS + r" missing \(required\)$", x)
    return m.group(1) + ".json" if m else None


operator = [x for x in items if (missing(x) and missing(x) not in absent) or "(not checked; clone mirror map" in x]
if operator:
    print("the operator's side (fix it and judge again; nothing recorded): " + " | ".join(operator))
    sys.exit(1)
run_records = [x for x in items if re.search(r"(^|/)" + RECS, x)]


def sha(path):
    with open(path, "rb") as fh:
        return "sha256:" + hashlib.sha256(fh.read()).hexdigest()


rec = {"schema": "stance5r-refusal/v1", "run": run, "seed": int(seed), "steps": int(steps) if steps.isdigit() else steps,
       "mode": "full", "refusal": hits[-1], "items": items, "judge": judge, "judge_sha256": sha(judge),
       "clone_head": head, "model": z, "model_sha256": sha(z), "vecnorm": p, "vecnorm_sha256": sha(p), "log": log,
       "judge_rc": int(rc), "written_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
       "rule": "criteria RUN HANDLING: a judge refused on the run's own records counts as an INVALID J file"}
if not run_records:
    # Only the checkpoint's own attributes (its seed, plant, task or symmetry record) while the run's placed records
    # pass: most likely a zip fetched from another run's models folder, the operator's side.  Held for review; the
    # scorer does not read this file.
    rec["held_for_review"] = (
        "the refusal names only the checkpoint's own attributes while the run's placed records pass: confirm with "
        "get_file_metadata that the fetched zip's parent is this run's models folder.  If it is, rename this file to "
        f"{os.path.basename(out)} (an invalid J file); if not, move it and the placed pair aside by hand and fetch the "
        "slot again (tools/README.md, Run handling rule 2)")
    with open(review, "x") as fh:
        json.dump(rec, fh, indent=1)
    print("held for review " + review + " (the checkpoint's own attributes only: " + " | ".join(items) + ")")
    sys.exit(2)
with open(out, "x") as fh:
    json.dump(rec, fh, indent=1)
print("recorded " + out)
EOF
}

if [ "$HEAD_EXPECT" = PLACEHOLDER_HEAD_LTSB ] && [ $DRY -eq 0 ] && [ $CHECK -eq 0 ]; then
  echo "HEAD_EXPECT is a PLACEHOLDER: fill it with the config commit first"; exit 4
fi
if [ $CHECK -eq 1 ]; then
  [ -e $OUT ] && valid $OUT >/dev/null 2>&1 && exit 0
  [ "$MODE" = full ] && [ -e $REFUSED ] && exit 5
  [ "$MODE" = full ] && [ -e $REVIEW ] && exit 6
  for f in $Z $P; do [ -f $f ] || exit 3; done
  exit 1
fi
CMD="ISO_WT=$CL nice -n 19 python3 $S3/tools/iso.py $JUDGE $RD --arm LTSB --seed $SEED --steps $STEPS --out $OUT $MFLAG"
if [ $DRY -eq 1 ]; then
  h=$(git -C $CL rev-parse HEAD 2>/dev/null); st=$(git -C $CL status --porcelain 2>/dev/null)
  [ "$h" = "$HEAD_EXPECT" ] && echo "clone $CL HEAD $h: ok" || echo "PROBLEM: clone HEAD $h is not $HEAD_EXPECT"
  [ -z "$st" ] && echo "clone clean: ok" || echo "PROBLEM: clone has uncommitted changes (the judge refuses)"
  grep -q "\"LTSB\".*\"head\": \"$HEAD_EXPECT\"" $JUDGE && echo "judge ARMS[LTSB] head $HEAD_EXPECT: ok" || echo "PROBLEM: judge ARMS[LTSB] head is not $HEAD_EXPECT"
  for f in plant_identity.json stage_config.json task_fingerprint.json; do
    [ -f $RD/01_stance/$f ] && echo "record $f: placed" || echo "MISSING record $RD/01_stance/$f (place5r.py config)"; done
  for f in $Z $P; do [ -f $f ] && echo "checkpoint $(basename $f): placed" || echo "MISSING checkpoint $f (fetch, then place5r.py)"; done
  if [ -e $OUT ]; then n=$(valid $OUT 2>&1) && echo "output exists and is complete: would skip" || echo "output exists but $n: would move it aside and rerun"
  else echo "output $OUT: not yet written"; fi
  [ "$MODE" = full ] && [ -e $REFUSED ] && echo "REFUSAL RECORD $REFUSED: the judge refused on the run's own records (an invalid J file); nothing is judged while it stands"
  [ "$MODE" = full ] && [ -e $REVIEW ] && echo "REFUSAL HELD FOR REVIEW $REVIEW: the checkpoint's own attributes only (confirm the zip's Drive parent; tools/README.md rule 2); nothing is judged while it stands"
  echo "free KB on scratch: $(df -Pk $S | awk 'NR==2{print $4}')"
  echo "WOULD RUN (cd $S5): $CMD"
  echo "           log $JL"
  exit 0
fi

mkdir -p $OUTD $LOGD
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPYCACHEPREFIX=$S/caches/pyc MPLCONFIGDIR=$S/caches/mpl
cd $S5 || exit 2
if [ -e $OUT ]; then
  if valid $OUT >/dev/null 2>&1; then echo "exists $(date -Is) $RUN $NAME" >> $LOGD/queue.log; exit 0; fi
  mv $OUT $OUT.bad.$(date +%Y%m%dT%H%M%S); echo "moved-aside $(date -Is) $RUN $NAME (incomplete)" >> $LOGD/queue.log
fi
if [ "$MODE" = full ] && [ -e $REFUSED ]; then echo "refused-recorded $(date -Is) $RUN $NAME ($REFUSED)" >> $LOGD/queue.log; exit 5; fi
if [ "$MODE" = full ] && [ -e $REVIEW ]; then echo "held-for-review $(date -Is) $RUN $NAME ($REVIEW)" >> $LOGD/queue.log; exit 6; fi
for f in $Z $P; do [ -f $f ] || { echo "missing $(date -Is) $RUN $NAME ($f)" >> $LOGD/queue.log; exit 3; }; done
[ -e $JL ] && mv $JL $JL.prev.$(date +%Y%m%dT%H%M%S)
echo "start $(date -Is) $RUN $NAME" >> $LOGD/queue.log
ISO_WT=$CL nice -n 19 python3 $S3/tools/iso.py $JUDGE $RD --arm LTSB --seed $SEED --steps $STEPS --out $OUT $MFLAG > $JL 2>&1
rc=$?
if [ $rc -eq 0 ] && ! valid $OUT >/dev/null 2>&1; then rc=4; fi
if [ $rc -ne 0 ] && [ "$MODE" = full ] && [ ! -e $OUT ]; then  # criteria RUN HANDLING: a refusal on the run's records
  msg=$(refusal $rc); case $? in 0) rc=5;; 2) rc=6;; esac
  echo "$msg"; echo "refusal $(date -Is) $RUN $NAME: $msg" >> $LOGD/queue.log
fi
echo "end $(date -Is) $RUN $NAME rc=$rc" >> $LOGD/queue.log
exit $rc
