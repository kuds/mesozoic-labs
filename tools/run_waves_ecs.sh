#!/usr/bin/env bash
# ECS wave runner (build list v2 item 8; criteria_ecs_v3.txt RUN HANDLING and WHEN THE READING IS MADE).  Ported from
# D-D43's tools/run_waves.sh.  Runs ONLY after <ECS>/APPROVED and <ECS>/criteria.txt exist; every continuation still
# passes run_ecs.py's own preflight (which refuses on any unpinned or changed input).
#
#   run_waves_ecs.sh            run (or resume) the queue
#   run_waves_ecs.sh --plan     print the queue and the commands; check nothing, launch nothing, write nothing
#
# How it is run (D-D43's form, RUNLOG l565): the runner is started detached (setsid nohup) by the coordinator, and a
# harness-tracked watcher that reads only logs/run_waves_ecs.log is what is relaunched when it exits.  Nothing
# relaunches the runner automatically: the coordinator starts it again after a container restart (the relaunch
# handling below) or after a halt once the cause is fixed.  While logs/HALTED exists the runner refuses to start
# (exit 2); it writes logs/HALTED whenever a halt (exit 2: a refusal, an early crash, a CHECK error) or a VOID (exit
# 4) stops the queue, and only the coordinator removes it (review H8).
#
# The queue is ecs_common.QUEUE, in its fixed order (for j = 1..6: s42 A j, s42 B j, s80 A j, s80 B j), run in
# ecs_common.WAVE_SLOTS (3) slots at nice ecs_common.WAVE_NICE (10): a slot that frees starts the next entry.
#   - a fresh queue (no decision, no crash, no run folder yet) starts only with free disk (df -B1) >=
#     ecs_common.LAUNCH_DISK_BYTES (6 GiB, T4), else it refuses (exit 2; nothing started).
#   - before each start: free disk >= ecs_common.DISK_FLOOR_BYTES (3 GiB), else PAUSED: no new start, the running
#     ones go on, logs/PAUSED_disk names the shortfall (the coordinator tells the maintainer); it resumes by itself once
#     space is free.  The process table and the load are appended to logs/proctable.log at each start.
#   - each attempt's output goes to its own logs/<name>.attempt<k>.log.  An attempt that reached learn() prints every
#     evaluation's return there: those logs are not opened until all 24 are decided (criteria RUN HANDLING,
#     outcome-blind).  A crash's cause is read from run_meta.json's error and traceback; a refused attempt (exit 2)
#     never reaches learn(), so its own log holds no evaluation and may be read for the reason.
#   - as each exits, by its exit code:
#       0: `read_ecs.py --check <run>` (CHECK only; nothing else of the run is read).  Exit 0: COUNTED (stop reason
#          "end") or STOPPED ("collapse").  Exit 4 (fail): VOID: decided VOID, then runs/<name> ->
#          runs/<name>_void<k>; no new continuation is started (those running finish and are checked), and the runner
#          exits 4 at the end.  Any other CHECK exit: CHECK ERROR, the folder is left as it is, no new start, exit 2.
#       5: the run reached its end but a post-run check failed (run_meta status "failed": the code seal or the
#          modules' origin): checked like exit 0; CHECK fails on it, so it is VOID (never a crash, never rerun).
#       2: REFUSED before any optimizer step (preflight or a pre-learn() check): runs/<name> (if made) ->
#          runs/<name>_refused<k>, no new start, exit 2 at the end; the cause is fixed and the runner started again
#          (the continuation starts again under the same name).
#       other (an exception, killed, out of memory): CRASH: runs/<name> -> runs/<name>_crashed<k>, recorded in
#          logs/crash_ledger.tsv.  An EARLY crash (run_meta's last recorded step, heartbeat rollout_end or last_step,
#          below S + 131,072, or none: no evaluation of the window exists) is not counted and halts new starts (the
#          coordinator finds and fixes the cause; the continuation then starts again under its name).  Any other crash
#          is one of its own crashes and is rerun at once from its start pair with the same arguments (the rerun is the
#          continuation); a third own crash writes it to <ECS>/lost.json (LOST).
# Relaunch: it takes logs/run_waves_ecs.lock (waiting up to 30 s; another live runner: exit 3, nothing done; no
# long-lived child holds the lock), waits while any of this ECS's run_ecs.py processes is still alive (an orphan of
# the previous runner finishes on its own), then reads its own ledger logs/decided.tsv (COUNTED / STOPPED / VOID /
# LOST, one line per decision) and <ECS>/lost.json.  A runs/<name> found done (run_meta status "done", exit 0) is
# checked as if it had just exited; one found "failed" is checked as exit 5; one found refused is handled as exit 2;
# any other (half-way: starting, running, crashed, unreadable) is a CRASH, renamed <name>_crashed<k>, counted as its
# own crash only when its run_meta says "crashed" (and then under the early-crash rule above; a container restart
# leaves "starting"/"running" and is not counted) and otherwise rerun.  A VOID in the ledger refuses the relaunch
# (exit 4): the coordinator fixes the cause from the CHECK record (criteria VOID) and removes that ledger line and
# logs/HALTED before the rerun.
# When all 24 are COUNTED, STOPPED or LOST and none is VOID: R3 (M1 end traces, 3 at a time at nice 10, --out
# <ECS>/traces/<name>: the stage1_<E>_steps pair, or a STOPPED one's stage1_final pair), then R1/R2 (`read_ecs.py
# <run>`, the full read, which also takes the trace's m1_summary.json) for each COUNTED or STOPPED continuation,
# then `score_ecs.py`.  A trace that fails is logged and does not hold up the score (R3 is report only).  No push, no
# Drive (D9).
# Exit codes: 0 read and scored (score_ecs's 0 or 3); 2 refused / halted (see the log and logs/HALTED); 3 another
# runner holds the lock; 4 VOID.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LC_ALL=C

S2=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad
S1=/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad
ECS_REAL=$S2/r10round/ecs
PY=$S2/venv/bin/python
TOOLS=$ECS_REAL/tools
M1=$S2/r10round/lean/build/laneA/tools/measure_r10round.py
ROUND_APPROVED=$S1/foot/stance6/APPROVED

die() { echo "run_waves_ecs: refused: $*" >&2; exit 2; }

ECS=$ECS_REAL
RUN_CMD=(nice -n 10 "$PY" -I -B "$TOOLS/run_ecs.py")
CHECK_CMD=("$PY" -I -B "$TOOLS/read_ecs.py" --check)
READ_CMD=("$PY" -I -B "$TOOLS/read_ecs.py")
TRACE_CMD=(nice -n 10 "$PY" -I -B "$M1" m1)
SCORE_CMD=("$PY" -I -B "$TOOLS/score_ecs.py")
PS_CMD=(ps -eo pid,ppid,ni,pcpu,pmem,etime,args --sort=-pcpu)
# The orphan wait (pgrep -f, an extended regex on the whole command line): this ECS's venv python running this ECS's
# run_ecs.py, anchored at the start.  The bracketed letters keep the pattern's own text from matching a process that
# quotes it (a watcher's or a shell's command line; RUNLOG l88-l90) (review I5).
ORPHAN_PATTERN="^$PY -I -B $TOOLS/[r]un_ecs[.]py --start "
POLL=10
LOCK_WAIT=30      # seconds a start waits for the lock (a just-killed runner's last short-lived command may hold it)
QUEUE_OVERRIDE=""
SLOTS_OVERRIDE=""
FLOOR_OVERRIDE=""
LAUNCH_OVERRIDE=""
# Test hook (tests/test_harness_waves.py): honoured only with ECS_WAVES_TEST=1 and a root that is not <ECS>.
if [[ "${ECS_WAVES_TEST:-}" == 1 ]]; then
  [[ -n "${ECS_ROOT:-}" ]] || die "ECS_WAVES_TEST needs ECS_ROOT"
  [[ "$(realpath -m "$ECS_ROOT")" != "$(realpath -m "$ECS_REAL")" ]] || die "ECS_WAVES_TEST may not use the real <ECS>"
  ECS=$(realpath -m "$ECS_ROOT")
  read -r -a RUN_CMD <<< "$ECS_RUN_CMD"
  read -r -a CHECK_CMD <<< "$ECS_CHECK_CMD"
  read -r -a READ_CMD <<< "$ECS_READ_CMD"
  read -r -a TRACE_CMD <<< "$ECS_TRACE_CMD"
  read -r -a SCORE_CMD <<< "$ECS_SCORE_CMD"
  [[ -n "${ECS_PS_CMD:-}" ]] && read -r -a PS_CMD <<< "$ECS_PS_CMD"
  ORPHAN_PATTERN=${ECS_ORPHAN_PATTERN:-$ORPHAN_PATTERN}
  POLL=${ECS_POLL_SECONDS:-1}
  LOCK_WAIT=${ECS_LOCK_WAIT_SECONDS:-2}
  QUEUE_OVERRIDE=${ECS_QUEUE:-}
  SLOTS_OVERRIDE=${ECS_SLOTS:-}
  FLOOR_OVERRIDE=${ECS_DISK_FLOOR_BYTES:-}
  LAUNCH_OVERRIDE=${ECS_LAUNCH_DISK_BYTES:-}
fi
RUNS=$ECS/runs
LOGS=$ECS/logs
TRACES=$ECS/traces
LOST_JSON=$ECS/lost.json
DECIDED=$LOGS/decided.tsv
CRASHES=$LOGS/crash_ledger.tsv
HALTED_FILE=$LOGS/HALTED

# The queue, the slots, the floors, the crash limit, the evaluation grid and the post-run-check exit come from
# ecs_common (one source); each queue line: start arm seed name S E.
COMMON_OUT=$("$PY" -I -B - "$TOOLS/ecs_common.py" <<'PYEOF'
import importlib.util, sys
s = importlib.util.spec_from_file_location("c", sys.argv[1]); C = importlib.util.module_from_spec(s)
s.loader.exec_module(C)
print(C.WAVE_SLOTS, C.WAVE_NICE, C.DISK_FLOOR_BYTES, C.MAX_CRASHES, C.LOST_SCHEMA, C.LAUNCH_DISK_BYTES, C.EVAL_FREQ,
      C.EXIT_POST_CHECK)
for start, arm, seed in C.QUEUE:
    S, E = C.window(start)
    print(start, arm, seed, C.run_name(start, arm, seed), S, E)
PYEOF
) || die "ecs_common.py could not be read"
read -r SLOTS NICE FLOOR MAX_CRASHES LOST_SCHEMA LAUNCH EVAL_FREQ POST_CHECK <<< "$(head -n 1 <<< "$COMMON_OUT")"
mapfile -t QUEUE < <(tail -n +2 <<< "$COMMON_OUT")
[[ "$NICE" == 10 ]] || die "ecs_common.WAVE_NICE is $NICE; RUN_CMD runs at nice 10"
if [[ -n "$QUEUE_OVERRIDE" ]]; then IFS=';' read -r -a QUEUE <<< "$QUEUE_OVERRIDE"; fi
[[ -n "$SLOTS_OVERRIDE" ]] && SLOTS=$SLOTS_OVERRIDE
[[ -n "$FLOOR_OVERRIDE" ]] && FLOOR=$FLOOR_OVERRIDE
[[ -n "$LAUNCH_OVERRIDE" ]] && LAUNCH=$LAUNCH_OVERRIDE
declare -A S_OF=() E_OF=() ARGS_OF=() LOG_OF=()
NAMES=()
for line in "${QUEUE[@]}"; do
  read -r st arm seed name S E <<< "$line"
  S_OF[$name]=$S; E_OF[$name]=$E; ARGS_OF[$name]="$st $arm $seed"; NAMES+=("$name")
done

if [[ "${1:-}" == "--plan" ]]; then
  [[ $# -eq 1 ]] || die "--plan takes no other argument"
  echo "slots $SLOTS at nice $NICE; disk floor $FLOOR bytes; launch floor $LAUNCH bytes; crash limit $MAX_CRASHES;" \
       "early crash: last step below S + $EVAL_FREQ"
  echo "orphan pattern: $ORPHAN_PATTERN"
  for name in "${NAMES[@]}"; do
    read -r st arm seed <<< "${ARGS_OF[$name]}"
    echo "  ${RUN_CMD[*]} --start $st --arm $arm --seed $seed > $LOGS/$name.attempt<k>.log  (S ${S_OF[$name]}," \
         "E ${E_OF[$name]})"
  done
  echo "then: ${CHECK_CMD[*]} <run>; after all are decided: ${TRACE_CMD[*]} ...; ${READ_CMD[*]} <run>; ${SCORE_CMD[*]}"
  exit 0
fi
[[ $# -eq 0 ]] || die "usage: run_waves_ecs.sh [--plan]"
[[ -f "$ECS/APPROVED" ]] || die "no <ECS>/APPROVED"
[[ -f "$ECS/criteria.txt" ]] || die "no <ECS>/criteria.txt"

# ------------------------------------------------------------------------------------------------------------ helpers
# Every long-lived command the runner starts runs with fd 9 (the lock) closed, so no child, and above all no
# continuation left running when the runner dies, holds the lock (review H1); a short-lived one is covered by the
# start's lock wait.
sleep() { command sleep "$@" 9>&-; }

note() {  # the runner's log line; the log file first, then stdout, which may be closed (review H2)
  local line
  line="[run_waves_ecs $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
  printf '%s\n' "$line" >> "$LOGS/run_waves_ecs.log"
  ( printf '%s\n' "$line" ) 2> /dev/null || true
}

halt_marker() {  # halt_marker <reason>: logs/HALTED, which the next start refuses until the coordinator removes it
  printf '%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" > "$HALTED_FILE"
  note "halted: $* (logs/HALTED; the coordinator fixes the cause and removes it before starting the runner again)"
}

meta_field() {  # meta_field <run dir> <key>: the value, or "none" (no folder) / "unreadable"
  local m=$1/run_meta.json
  [[ -d "$1" ]] || { echo none; return; }
  [[ -f "$m" ]] || { echo unreadable; return; }
  "$PY" -I -B -c "import json,sys
try: v = json.load(open(sys.argv[1])).get(sys.argv[2])
except Exception: v = 'unreadable'
print('null' if v is None else v)" "$m" "$2" 9>&-
}

reached_step() {  # reached_step <run dir>: the last step run_meta records (heartbeat rollout_end, last_step) or none
  [[ -f "$1/run_meta.json" ]] || { echo none; return; }
  "$PY" -I -B -c "import json,sys
try: m = json.load(open(sys.argv[1]))
except Exception: m = {}
hb = m.get('heartbeat') if isinstance(m.get('heartbeat'), dict) else {}
v = [x for x in (hb.get('rollout_end'), m.get('last_step')) if isinstance(x, int) and not isinstance(x, bool)]
print(max(v) if v else 'none')" "$1/run_meta.json" 9>&-
}

decided() {  # decided <name>: its last decision (COUNTED / STOPPED / VOID / LOST) or empty
  awk -F'\t' -v n="$1" '$2 == n {d = $3} END {print d}' "$DECIDED"
}

decide() {  # decide <name> <decision> <detail>
  printf '%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" "$3" >> "$DECIDED"
  note "$1: $2 ($3)"
}

own_crashes() { awk -F'\t' -v n="$1" '$2 == n && $5 == 1 {c++} END {print c + 0}' "$CRASHES"; }

next_free() {  # next_free <name> <kind>: the first runs/<name>_<kind><k> not taken (printed, not made)
  local k=1
  while [[ -e "$RUNS/$1_$2$k" ]]; do k=$((k + 1)); done
  echo "$1_$2$k"
}

retire() {  # retire <name> <kind>: runs/<name> -> runs/<name>_<kind><k>; prints the new name
  local new
  new=$(next_free "$1" "$2")
  mv "$RUNS/$1" "$RUNS/$new"
  echo "$new"
}

retire_void() {  # retire_void <name> <detail>: decided VOID first, then the folder renamed (review I4)
  local new
  new=$(next_free "$1" void)
  VOID_SEEN=1
  decide "$1" VOID "$2; folder $new"
  mv "$RUNS/$1" "$RUNS/$new"
}

attempt_log() {  # attempt_log <name>: the next free logs/<name>.attempt<k>.log
  local k=1
  while [[ -e "$LOGS/$1.attempt$k.log" ]]; do k=$((k + 1)); done
  echo "$LOGS/$1.attempt$k.log"
}

in_lost() {  # in_lost <name>: 0 if <ECS>/lost.json names it
  [[ -f "$LOST_JSON" ]] || return 1
  "$PY" -I -B -c "import json,sys
d = json.load(open(sys.argv[1]))
sys.exit(0 if sys.argv[2] in (d.get('lost') or {}) else 1)" "$LOST_JSON" "$1" 9>&-
}

write_lost() {  # write_lost <name> <detail>: adds the entry (atomic rewrite; other entries kept)
  "$PY" -I -B - "$LOST_JSON" "$1" "$2" "$LOST_SCHEMA" 9>&- <<'PYEOF'
import datetime, json, os, sys
path, name, detail, schema = sys.argv[1:5]
d = json.load(open(path)) if os.path.exists(path) else {"schema": schema, "lost": {}}
if d.get("schema") != schema:
    sys.exit(f"{path}: schema {d.get('schema')!r} != {schema!r}")
d.setdefault("lost", {})[name] = {"cause": "crashes", "detail": detail, "by": "run_waves_ecs.sh",
                                  "utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
tmp = path + ".tmp"
with open(tmp, "w") as fh:
    json.dump(d, fh, indent=1)
    fh.write("\n")
os.replace(tmp, path)
PYEOF
}

free_bytes() {
  if [[ "${ECS_WAVES_TEST:-}" == 1 && -n "${ECS_FREE_BYTES_FILE:-}" ]]; then cat "$ECS_FREE_BYTES_FILE"; return; fi
  df -B1 --output=avail "$ECS" | tail -n 1 | tr -d ' '
}

proctable() {  # a pipe that closes early (head) must never end the runner under pipefail (review H2)
  { echo "== $(date -u +%Y-%m-%dT%H:%M:%SZ) start $1"; echo "load: $(cat /proc/loadavg)"
    "${PS_CMD[@]}" 2> /dev/null 9>&- | head -n 12 | cut -c1-200 || true; } >> "$LOGS/proctable.log"
}

HALT=0
HALT_WHY=()
VOID_SEEN=0
REFUSED=()
PENDING=()
declare -A PID=()

record_crash() {  # record_crash <name> <cause> <counts 0|1>
  local name=$1 cause=$2 counts=$3 new=none reached=none early=0 n
  [[ -d "$RUNS/$name" ]] && reached=$(reached_step "$RUNS/$name")
  if (( counts == 1 )); then
    if [[ "$reached" == none ]] || (( reached < ${S_OF[$name]} + EVAL_FREQ )); then early=1; counts=0; fi
  fi
  [[ -d "$RUNS/$name" ]] && new=$(retire "$name" crashed)
  (( early )) && cause="EARLY CRASH, no window evaluation (last step $reached): $cause"
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$name" "$new" "$cause" "$counts" >> "$CRASHES"
  n=$(own_crashes "$name")
  if (( early )); then
    HALT=1
    HALT_WHY+=("early crash of $name (runs/$new)")
    note "$name: $cause; folder $new; not counted (own crashes $n of $MAX_CRASHES); no new start: the coordinator" \
         "finds the cause from runs/$new/run_meta.json (error, traceback; never a process log) and fixes it"
    return
  fi
  note "$name: CRASH ($cause; folder $new; own crash $n of $MAX_CRASHES$( ((counts == 0)) && echo ', not counted'))"
  if (( n >= MAX_CRASHES )); then
    if ! write_lost "$name" "$n own crashes, last: $cause"; then
      note "lost.json could not be written for $name"
      HALT=1
      HALT_WHY+=("lost.json not written for $name")
      return
    fi
    decide "$name" LOST "$n own crashes"
  else
    PENDING=("$name" "${PENDING[@]+"${PENDING[@]}"}")  # rerun at once, same arguments
  fi
}

after_exit() {  # after_exit <name> <rc>
  local name=$1 rc=$2 run=$RUNS/$1 crc reason new
  if (( rc == 0 || rc == POST_CHECK )); then
    set +e
    "${CHECK_CMD[@]}" "$run" >> "$LOGS/$name.check.log" 2>&1 9>&-
    crc=$?
    set -e
    if (( crc == 0 && rc == 0 )); then
      reason=$(meta_field "$run" stop_reason)
      case "$reason" in
        end) decide "$name" COUNTED "CHECK passed, stop reason end" ;;
        collapse) decide "$name" STOPPED "CHECK passed, stop reason collapse" ;;
        *) retire_void "$name" "CHECK passed but stop reason '$reason'" ;;
      esac
    elif (( crc == 4 )); then
      retire_void "$name" "CHECK failed (logs/$name.check.log; run exit $rc)"
    elif (( crc == 0 )); then
      retire_void "$name" "run exit $rc (a post-run check failed) although CHECK passed (logs/$name.check.log)"
    else
      HALT=1
      HALT_WHY+=("CHECK ERROR on $name")
      note "$name: CHECK ERROR (read_ecs exit $crc, logs/$name.check.log); folder left as it is; no new start"
    fi
  elif (( rc == 2 )); then
    new=none
    [[ -d "$run" ]] && new=$(retire "$name" refused)
    HALT=1
    REFUSED+=("$name")
    HALT_WHY+=("refused $name")
    note "$name: REFUSED before any optimizer step (the reason: ${LOG_OF[$name]:-runs/$new/run_meta.json error}," \
         "an attempt that exits 2 never reaches learn(); folder $new); no new start"
  else
    record_crash "$name" "exit $rc" 1
  fi
}

# ------------------------------------------------------------------------------------------------------------ start
mkdir -p "$LOGS" "$RUNS"
exec 9> "$LOGS/run_waves_ecs.lock"
if ! flock -w "$LOCK_WAIT" 9; then
  echo "run_waves_ecs: another run_waves_ecs.sh holds $LOGS/run_waves_ecs.lock (waited ${LOCK_WAIT} s; exit 3;" \
       "nothing done)" >&2
  exit 3
fi
touch "$DECIDED" "$CRASHES"
trap 'echo "$? $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOGS/run_waves_ecs.exit"' EXIT
if [[ -e "$HALTED_FILE" ]]; then
  die "halted ($(head -n 1 "$HALTED_FILE" | cut -c1-300)); the coordinator fixes the cause and removes $HALTED_FILE"
fi
if [[ ! -s "$DECIDED" && ! -s "$CRASHES" && -z "$(ls -A "$RUNS")" ]]; then
  free=$(free_bytes)
  (( free >= LAUNCH )) || die "launch: $free bytes free < the launch floor $LAUNCH bytes (T4): housekeeping first"
  note "launch: $free bytes free >= the launch floor $LAUNCH bytes"
fi

# Relaunch: wait for any continuation process left by a previous runner, then read the state.
while pgrep -f "$ORPHAN_PATTERN" > /dev/null; do
  note "a run_ecs.py process is still alive (pgrep -f '$ORPHAN_PATTERN'); waiting for it to finish"
  sleep "$POLL"
done
for name in "${NAMES[@]}"; do
  d=$(decided "$name")
  if [[ "$d" == VOID ]]; then
    VOID_SEEN=1
    continue
  fi
  [[ -n "$d" ]] && continue
  if in_lost "$name"; then decide "$name" LOST "named in lost.json"; continue; fi
  if [[ -d "$RUNS/$name" ]]; then
    status=$(meta_field "$RUNS/$name" status)
    code=$(meta_field "$RUNS/$name" exit)
    if [[ "$status" == done && "$code" == 0 ]]; then
      note "$name: found done at relaunch; checking"
      after_exit "$name" 0
    elif [[ "$status" == failed ]]; then
      note "$name: found failed (a post-run check) at relaunch; checking"
      after_exit "$name" "$POST_CHECK"
    elif [[ "$status" == refused ]]; then
      after_exit "$name" 2
    else
      counts=0
      [[ "$status" == crashed ]] && counts=1
      record_crash "$name" "half-way at relaunch (status $status)" "$counts"
    fi
    continue
  fi
  PENDING+=("$name")
done
# record_crash put reruns at the front; keep the fixed order of the queue for everything pending.
ORDERED=()
for name in "${NAMES[@]}"; do
  for p in "${PENDING[@]+"${PENDING[@]}"}"; do [[ "$p" == "$name" ]] && { ORDERED+=("$name"); break; }; done
done
PENDING=("${ORDERED[@]+"${ORDERED[@]}"}")
if (( VOID_SEEN )); then
  halt_marker "VOID in logs/decided.tsv: no continuation is started; the coordinator fixes the cause (criteria VOID)"
  exit 4
fi

PAUSED=0
note "queue: ${#PENDING[@]} pending of ${#NAMES[@]}; $SLOTS slots"
while true; do
  while (( ${#PID[@]} < SLOTS && ${#PENDING[@]} > 0 && HALT == 0 && VOID_SEEN == 0 )); do
    free=$(free_bytes)
    if (( free < FLOOR )); then
      if (( ! PAUSED )); then
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) free $free < floor $FLOOR bytes under $ECS" > "$LOGS/PAUSED_disk"
        note "PAUSED: $free bytes free < $FLOOR; no new start until space is freed (tell the maintainer)"
        PAUSED=1
      fi
      break
    fi
    if (( PAUSED )); then rm -f "$LOGS/PAUSED_disk"; note "resumed: $free bytes free"; PAUSED=0; fi
    name=${PENDING[0]}
    PENDING=("${PENDING[@]:1}")
    read -r st arm seed <<< "${ARGS_OF[$name]}"
    log=$(attempt_log "$name")
    LOG_OF[$name]=$log
    proctable "$name"
    note "start $name (${log#"$LOGS"/})"
    (exec 9>&-; cd /tmp && exec "${RUN_CMD[@]}" --start "$st" --arm "$arm" --seed "$seed") >> "$log" 2>&1 &
    PID[$name]=$!
  done
  if (( ${#PID[@]} == 0 )); then
    if (( ${#PENDING[@]} > 0 && HALT == 0 && VOID_SEEN == 0 )); then sleep "$POLL"; continue; fi
    break
  fi
  sleep "$POLL"
  for name in "${!PID[@]}"; do
    if ! kill -0 "${PID[$name]}" 2> /dev/null; then
      set +e
      wait "${PID[$name]}"
      rc=$?
      set -e
      unset "PID[$name]"
      note "$name exited $rc"
      after_exit "$name" "$rc"
    fi
  done
done

if (( VOID_SEEN )); then
  halt_marker "VOID (logs/decided.tsv); the coordinator fixes the cause (criteria VOID)"
  exit 4
fi
if (( HALT )); then
  halt_marker "${HALT_WHY[*]:-halted}"
  exit 2
fi
for name in "${NAMES[@]}"; do
  d=$(decided "$name")
  if [[ "$d" != COUNTED && "$d" != STOPPED && "$d" != LOST ]]; then
    halt_marker "not all decided ($name: '${d}')"
    exit 2
  fi
done
note "all ${#NAMES[@]} decided, none VOID: the reading (R3 end traces, then R1/R2, then the score)"

READ_OK=()
for name in "${NAMES[@]}"; do
  d=$(decided "$name")
  [[ "$d" == COUNTED || "$d" == STOPPED ]] && READ_OK+=("$name")
done

# R3 first: read_ecs.py's full read takes the end trace's m1_summary.json from <ECS>/traces/<name> (a missing one is
# printed as missing, report only).
mkdir -p "$TRACES"
declare -A TPID=()
trace_one() {  # trace_one <name>
  local name=$1 d stage zip pkl
  d=$(decided "$name")
  stage=$RUNS/$name/01_stance
  if [[ "$d" == STOPPED ]]; then
    zip=$stage/models/stage1_final.zip; pkl=$stage/models/stage1_final_vecnorm.pkl
  else
    zip=$stage/models/stage1_${E_OF[$name]}_steps.zip; pkl=$stage/models/stage1_vecnormalize_${E_OF[$name]}_steps.pkl
  fi
  (exec 9>&-; cd /tmp && exec "${TRACE_CMD[@]}" --copy "$ECS/code" --zip "$zip" --pkl "$pkl" --stage-config \
     "$stage/stage_config.json" --out "$TRACES/$name" --approved "$ROUND_APPROVED") >> "$LOGS/$name.trace.log" 2>&1 &
  TPID[$name]=$!
}
TODO=()
for name in "${READ_OK[@]+"${READ_OK[@]}"}"; do
  [[ -f "$TRACES/$name/m1_summary.json" ]] && continue
  TODO+=("$name")
done
while (( ${#TODO[@]} > 0 || ${#TPID[@]} > 0 )); do
  while (( ${#TPID[@]} < SLOTS && ${#TODO[@]} > 0 )); do
    trace_one "${TODO[0]}"
    TODO=("${TODO[@]:1}")
  done
  sleep "$POLL"
  for name in "${!TPID[@]}"; do
    if ! kill -0 "${TPID[$name]}" 2> /dev/null; then
      set +e
      wait "${TPID[$name]}"
      rc=$?
      set -e
      unset "TPID[$name]"
      note "$name: end trace (R3) exit $rc$( ((rc != 0)) && echo ' (report only; rerun if disk)')"
    fi
  done
done

for name in "${READ_OK[@]+"${READ_OK[@]}"}"; do
  set +e
  "${READ_CMD[@]}" "$RUNS/$name" >> "$LOGS/$name.read.log" 2>&1 9>&-
  rrc=$?
  set -e
  note "$name: read_ecs (R1/R2/R3) exit $rrc"
done

set +e
"${SCORE_CMD[@]}" >> "$LOGS/score.log" 2>&1 9>&-
src=$?
set -e
note "score_ecs.py exit $src (0 final, 3 provisional or incomplete, 2 input error)"
(( src == 0 || src == 3 )) && exit 0
exit 2
