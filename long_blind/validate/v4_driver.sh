#!/usr/bin/env bash
# LB validation driver B: V4, the resume test (no optimizer step anywhere; one process at a time).
#   attempt 1: run_bs.py --arm critic_blind_tail3 --seed 9311 --no-update --until 2,654,208 --out <LB>/validate/v4,
#              killed (SIGKILL, as a machine loss would) once its heartbeat passes 2,629,632, i.e. after its second saved
#              pair (2,621,440) and that pair's checkpoint-call writes; its folder is then renamed <name>_crashed1, as the
#              runner retires a crashed attempt.
#   attempt 2: the same command with --resume-from <LB>/validate/v4/critic_blind_tail3_9311_crashed1: resumes from the
#              newest complete pair (expected 2,359,296: the 2,621,440 pair's probe row and training-ledger episodes
#              after it never reached the disk), runs to 2,654,208; then CHECK (read_bs.py --check) on it.
# nice 10, OMP/MKL/OPENBLAS 1, cwd /tmp, PY -I -B.  Writes only under <LB>/validate/ (v4/, logs/) and <LB>/tmp/.
set -u
S2=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad
LB=$S2/r10round/long_blind
PY=$S2/venv/bin/python
L=$LB/validate/logs
OUT=$LB/validate/v4
ARM=critic_blind_tail3
SEED=9311
UNTIL=2654208
KILL_AT=2629632
NAME=${ARM}_$SEED
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$L/driverB.log"; }
hb() { "$PY" -I -B -c "import json,sys
try: m = json.load(open(sys.argv[1]))
except Exception: m = {}
print((m.get('heartbeat') or {}).get('rollout_end') or 0)" "$OUT/$NAME/run_meta.json"; }
log "start; until $UNTIL; kill at heartbeat >= $KILL_AT; load $(cat /proc/loadavg)"
(cd /tmp && exec nice -n 10 "$PY" -I -B "$LB/tools/run_bs.py" --arm "$ARM" --seed "$SEED" --no-update \
   --until "$UNTIL" --out "$OUT") > "$L/v4_attempt1.log" 2>&1 &
pid=$!
log "attempt 1 started, pid $pid"
while kill -0 "$pid" 2> "$L/v4_kill0.err"; do
  n=$(hb)
  if (( n >= KILL_AT )); then
    kill -9 "$pid"; log "attempt 1 killed (SIGKILL) at heartbeat $n"; break
  fi
  sleep 5
done
wait "$pid"; log "attempt 1 exit $?"
mv "$OUT/$NAME" "$OUT/${NAME}_crashed1"; log "renamed to ${NAME}_crashed1 (as the runner retires a crash)"
(cd /tmp && exec nice -n 10 "$PY" -I -B "$LB/tools/run_bs.py" --arm "$ARM" --seed "$SEED" --no-update \
   --until "$UNTIL" --out "$OUT" --resume-from "$OUT/${NAME}_crashed1") > "$L/v4_attempt2.log" 2>&1
log "attempt 2 exit $?"
(cd /tmp && exec "$PY" -I -B "$LB/tools/read_bs.py" --check "$OUT/$NAME") > "$L/v4_check.log" 2>&1
log "CHECK attempt 2 exit $?"
log "done; load $(cat /proc/loadavg)"
