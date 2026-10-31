#!/usr/bin/env bash
# BS2: BS's run_waves_bs.sh (sha256 13fb6182...) with the root changed (BS_REAL = S2/r10round/bias_screen2); the
# queue mechanics, slots, nice, floors, crash rules and the outcome-blind reads are BS's, unchanged.  "<BS>" below = BS2.
# BS2 review fixes (the fixer, 2026-10-30; <BS2>/review/FIXES.txt), additions only:
#   PR-M4  the runner refuses (exit 2, nothing started) unless APPROVED's tool_sha256["score_bs.py"] equals the current
#          tools/score_bs.py (run_bs.py's preflight pins the five bs_common.TOOL_FILES only);
#   PR-M2  lesson P4 (relatime): right after a continuation is decided COUNTED or STOPPED, every regular file under
#          runs/<name>/ and tmp/<name>/ and its logs/<name>.attempt<k>.log is re-armed (atime = mtime - 1 ns; content,
#          mtime and sha256 unchanged), so any later read of one moves its atime; at ALL_DECIDED, before
#          logs/ALL_DECIDED is written and before the full reads, logs/atimes_at_ALL_DECIDED.tsv records path, atime,
#          mtime and ctime (lstat, which moves no atime) for runs/**, logs/** and tmp/**, with rearmed_unread = 1 where
#          the atime is still mtime - 1 ns;
#   PR-M3  a REFUSED note names the read-only command that shows the cause (run_bs.py --preflight-only for that
#          entry), never the attempt log.
# BS wave runner, derived from the ECS runner (S2/r10round/ecs/tools/run_waves_ecs.sh, sha256 c007762c...): the same
# crash handling, relaunch, lock, HALTED and VOID rules; the bias screen's queue, slots, nice and floors.  Runs ONLY
# after <BS>/APPROVED, <BS>/criteria.txt and <BS>/SCREEN.json exist; every continuation still passes run_bs.py's own
# preflight (which refuses on any unpinned or changed input: APPROVED pins criteria.txt, SCREEN.json and the tools).
#
#   run_waves_bs.sh            run (or resume) the queue
#   run_waves_bs.sh --plan     print the queue and the commands; check nothing, launch nothing, write nothing
#
# OUTCOME-BLIND: while the queue runs, the runner reads only each attempt's exit code, run_meta.json (status, exit,
# stop_reason, heartbeat, last_step) and the CHECK's exit code (read_bs.py --check).  It never opens an attempt's log,
# a ledger, a probe or a checkpoint (the atime re-arm and snapshot use lstat and utime only: no file is opened).  Started detached (setsid nohup) by the coordinator; a harness-tracked watcher
# reads only logs/run_waves_bs.log.  Nothing relaunches it automatically.  While logs/HALTED exists it refuses to start
# (exit 2); it writes logs/HALTED whenever a halt (exit 2) or a VOID (exit 4) stops the queue; only the coordinator
# removes it.
#
# The queue is SCREEN.json's (bs_common.load_screen: control and ONE intervention, at most 9 continuations), in its
# order, run in bs_common.WAVE_SLOTS (4) slots at nice bs_common.WAVE_NICE (10): a slot that frees starts the next.
#   - a fresh queue (no decision, no crash, no run folder) starts only with free disk >= bs_common.LAUNCH_DISK_BYTES
#     (the 1.5 GiB floor plus 9 x 50 MiB), else it refuses (exit 2; nothing started).
#   - before each start: free disk >= bs_common.DISK_FLOOR_BYTES (1.5 GiB), else PAUSED: no new start, the running
#     ones go on, logs/PAUSED_disk names the shortfall; it resumes by itself once space is free.  The process table and
#     the load are appended to logs/proctable.log at each start.
#   - each attempt's output goes to logs/<name>.attempt<k>.log (never opened by the runner; it may print evaluation
#     returns).  A crash's cause is read from run_meta.json's error and traceback.
#   - as each exits, by its exit code:
#       0: `read_bs.py --check <run>` (CHECK only).  Exit 0: COUNTED (stop reason "end") or STOPPED ("collapse").
#          Exit 4: VOID: decided VOID, runs/<name> -> runs/<name>_void<k>; no new start; the runner exits 4 at the
#          end.  Any other CHECK exit: CHECK ERROR, the folder left as it is, no new start, exit 2.
#       5: the run reached its end but a post-run check failed (status "failed"): checked like 0; CHECK fails: VOID.
#       2: REFUSED before any optimizer step: runs/<name> (if made) -> runs/<name>_refused<k>, no new start, exit 2.
#       other: CRASH: runs/<name> -> runs/<name>_crashed<k>, logs/crash_ledger.tsv.  An EARLY crash (last recorded step
#          below S + 131,072, or none) is not counted and halts new starts.  Any other crash is one of its own and is
#          rerun at once from the start pair with the same arguments; a third own crash writes it to <BS>/lost.json.
# Relaunch: logs/run_waves_bs.lock (waiting up to 30 s; another live runner: exit 3), waits while any of this BS's
# run_bs.py processes is alive, then reads logs/decided.tsv and <BS>/lost.json; a runs/<name> found done is checked; one
# found failed is checked as exit 5; one found refused is handled as exit 2; any other is a CRASH (counted only if its
# run_meta says "crashed", under the early-crash rule).  A VOID in the ledger refuses the relaunch (exit 4).
# When all are COUNTED, STOPPED or LOST and none is VOID: logs/ALL_DECIDED, then the full read (`read_bs.py <run>`,
# the markers; refused by read_bs.py itself until all are decided) of each COUNTED or STOPPED continuation, WAVE_SLOTS
# at a time at nice 10.  No scorer is run here (the criteria's, written and pinned before launch).  No push.
# Exit codes: 0 all decided and read; 2 refused / halted; 3 another runner holds the lock; 4 VOID.
# No redirection onto any path outside <BS> (stderr of probes goes to logs/runner_stderr.log).
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LC_ALL=C

S2=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad
BS_REAL=$S2/r10round/bias_screen2
PY=$S2/venv/bin/python
TOOLS=$BS_REAL/tools

die() { echo "run_waves_bs: refused: $*" >&2; exit 2; }

BS=$BS_REAL
RUN_CMD=(nice -n 10 "$PY" -I -B "$TOOLS/run_bs.py")
CHECK_CMD=("$PY" -I -B "$TOOLS/read_bs.py" --check)
READ_CMD=(nice -n 10 "$PY" -I -B "$TOOLS/read_bs.py")
PS_CMD=(ps -eo pid,ppid,ni,pcpu,pmem,etime,args --sort=-pcpu)
ORPHAN_PATTERN="^$PY -I -B $TOOLS/[r]un_bs[.]py --arm "
POLL=10
LOCK_WAIT=30
QUEUE_OVERRIDE=""
SLOTS_OVERRIDE=""
FLOOR_OVERRIDE=""
LAUNCH_OVERRIDE=""
# Test hook (tests/test_waves_bs.py): honoured only with BS_WAVES_TEST=1 and a root that is not <BS>.
if [[ "${BS_WAVES_TEST:-}" == 1 ]]; then
  [[ -n "${BS_ROOT:-}" ]] || die "BS_WAVES_TEST needs BS_ROOT"
  [[ "$(realpath -m "$BS_ROOT")" != "$(realpath -m "$BS_REAL")" ]] || die "BS_WAVES_TEST may not use the real <BS>"
  BS=$(realpath -m "$BS_ROOT")
  read -r -a RUN_CMD <<< "$BS_RUN_CMD"
  read -r -a CHECK_CMD <<< "$BS_CHECK_CMD"
  read -r -a READ_CMD <<< "$BS_READ_CMD"
  [[ -n "${BS_PS_CMD:-}" ]] && read -r -a PS_CMD <<< "$BS_PS_CMD"
  ORPHAN_PATTERN=${BS_ORPHAN_PATTERN:-$ORPHAN_PATTERN}
  POLL=${BS_POLL_SECONDS:-1}
  LOCK_WAIT=${BS_LOCK_WAIT_SECONDS:-2}
  QUEUE_OVERRIDE=${BS_QUEUE:-}
  SLOTS_OVERRIDE=${BS_SLOTS:-}
  FLOOR_OVERRIDE=${BS_DISK_FLOOR_BYTES:-}
  LAUNCH_OVERRIDE=${BS_LAUNCH_DISK_BYTES:-}
fi
RUNS=$BS/runs
LOGS=$BS/logs
LOST_JSON=$BS/lost.json
DECIDED=$LOGS/decided.tsv
CRASHES=$LOGS/crash_ledger.tsv
HALTED_FILE=$LOGS/HALTED
ERRLOG=$LOGS/runner_stderr.log

# One source for the slots, the floors, the crash limit, the grid and the queue: bs_common (the queue from SCREEN.json
# under the root; each queue line: arm seed name S E).  A missing or invalid SCREEN.json refuses (except in tests with
# BS_QUEUE).
COMMON_OUT=$("$PY" -I -B - "$TOOLS/bs_common.py" "$BS/SCREEN.json" "${QUEUE_OVERRIDE:+override}" <<'PYEOF'
import importlib.util, sys
s = importlib.util.spec_from_file_location("c", sys.argv[1]); C = importlib.util.module_from_spec(s)
s.loader.exec_module(C)
print(C.WAVE_SLOTS, C.WAVE_NICE, C.DISK_FLOOR_BYTES, C.MAX_CRASHES, C.LOST_SCHEMA, C.LAUNCH_DISK_BYTES, C.EVAL_FREQ,
      C.EXIT_POST_CHECK, C.S, C.E, C.MAX_CONTINUATIONS)
if len(sys.argv) > 3 and sys.argv[3] == "override":
    sys.exit(0)
d, problems = C.load_screen(sys.argv[2])
if problems:
    print("SCREEN-PROBLEMS " + " | ".join(problems))
    sys.exit(0)
for arm, seed in C.queue_of(d):
    print(arm, seed, C.run_name(arm, seed), C.S, C.E)
PYEOF
) || die "bs_common.py could not be read"
read -r SLOTS NICE FLOOR MAX_CRASHES LOST_SCHEMA LAUNCH EVAL_FREQ POST_CHECK S_ALL E_ALL MAXN <<< "$(head -n 1 <<< "$COMMON_OUT")"
[[ "$NICE" == 10 ]] || die "bs_common.WAVE_NICE is $NICE; RUN_CMD runs at nice 10"
if [[ -n "$QUEUE_OVERRIDE" ]]; then
  IFS=';' read -r -a QUEUE <<< "$QUEUE_OVERRIDE"
else
  if grep -q '^SCREEN-PROBLEMS' <<< "$COMMON_OUT"; then die "$(grep '^SCREEN-PROBLEMS' <<< "$COMMON_OUT")"; fi
  mapfile -t QUEUE < <(tail -n +2 <<< "$COMMON_OUT")
fi
(( ${#QUEUE[@]} >= 1 && ${#QUEUE[@]} <= MAXN )) || die "queue of ${#QUEUE[@]} (1 to $MAXN allowed)"
[[ -n "$SLOTS_OVERRIDE" ]] && SLOTS=$SLOTS_OVERRIDE
[[ -n "$FLOOR_OVERRIDE" ]] && FLOOR=$FLOOR_OVERRIDE
[[ -n "$LAUNCH_OVERRIDE" ]] && LAUNCH=$LAUNCH_OVERRIDE
declare -A S_OF=() E_OF=() ARGS_OF=() LOG_OF=()
NAMES=()
for line in "${QUEUE[@]}"; do
  read -r arm seed name S E <<< "$line"
  S_OF[$name]=$S; E_OF[$name]=$E; ARGS_OF[$name]="$arm $seed"; NAMES+=("$name")
done

if [[ "${1:-}" == "--plan" ]]; then
  [[ $# -eq 1 ]] || die "--plan takes no other argument"
  echo "slots $SLOTS at nice $NICE; disk floor $FLOOR bytes; launch floor $LAUNCH bytes; crash limit $MAX_CRASHES;" \
       "early crash: last step below S + $EVAL_FREQ"
  echo "orphan pattern: $ORPHAN_PATTERN"
  for name in "${NAMES[@]}"; do
    read -r arm seed <<< "${ARGS_OF[$name]}"
    echo "  ${RUN_CMD[*]} --arm $arm --seed $seed > $LOGS/$name.attempt<k>.log  (S ${S_OF[$name]}, E ${E_OF[$name]})"
  done
  echo "then: ${CHECK_CMD[*]} <run>; after all are decided: ${READ_CMD[*]} <run> (the markers; no scorer)"
  exit 0
fi
[[ $# -eq 0 ]] || die "usage: run_waves_bs.sh [--plan]"
[[ -f "$BS/APPROVED" ]] || die "no <BS>/APPROVED"
[[ -f "$BS/criteria.txt" ]] || die "no <BS>/criteria.txt"
[[ -n "$QUEUE_OVERRIDE" || -f "$BS/SCREEN.json" ]] || die "no <BS>/SCREEN.json"
# PR-M4: APPROVED must pin the scorer as well (the criteria's PROVISIONAL rule needs tool_sha256["score_bs.py"], and
# APPROVED is never edited after the first start); run_bs.py's preflight checks the other pins for each continuation.
SCORER_PIN=$("$PY" -I -B - "$BS/APPROVED" "$TOOLS/score_bs.py" <<'PYEOF'
import hashlib, json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as exc:  # noqa: BLE001 - any unreadable APPROVED refuses
    sys.exit(print(f"APPROVED does not parse ({exc.__class__.__name__})"))
tp = d.get("tool_sha256") if isinstance(d, dict) and isinstance(d.get("tool_sha256"), dict) else {}
try:
    got = hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest()
except OSError as exc:
    sys.exit(print(f"tools/score_bs.py unreadable ({exc.__class__.__name__})"))
print("ok" if tp.get("score_bs.py") == got else f"APPROVED tool_sha256['score_bs.py'] {tp.get('score_bs.py')!r} "
      f"!= the file's {got}")
PYEOF
) || die "the scorer pin in APPROVED could not be checked"
[[ "$SCORER_PIN" == ok ]] || die "$SCORER_PIN (APPROVED must pin tools/score_bs.py: criteria, HOW THE COORDINATOR PINS)"

# ------------------------------------------------------------------------------------------------------------ helpers
sleep() { command sleep "$@" 9>&-; }

note() {  # the runner's log line; the log file first, then stdout, which may be closed
  local line
  line="[run_waves_bs $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
  printf '%s\n' "$line" >> "$LOGS/run_waves_bs.log"
  ( printf '%s\n' "$line" ) 2>> "$ERRLOG" || true
}

halt_marker() {
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

reached_step() {  # the last step run_meta records (heartbeat rollout_end, last_step) or none
  [[ -f "$1/run_meta.json" ]] || { echo none; return; }
  "$PY" -I -B -c "import json,sys
try: m = json.load(open(sys.argv[1]))
except Exception: m = {}
hb = m.get('heartbeat') if isinstance(m.get('heartbeat'), dict) else {}
v = [x for x in (hb.get('rollout_end'), m.get('last_step')) if isinstance(x, int) and not isinstance(x, bool)]
print(max(v) if v else 'none')" "$1/run_meta.json" 9>&-
}

decided() { awk -F'\t' -v n="$1" '$2 == n {d = $3} END {print d}' "$DECIDED"; }

decide() {
  printf '%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" "$3" >> "$DECIDED"
  note "$1: $2 ($3)"
}

own_crashes() { awk -F'\t' -v n="$1" '$2 == n && $5 == 1 {c++} END {print c + 0}' "$CRASHES"; }

rearm_atimes() {  # PR-M2 / P4: atime = mtime - 1 ns on every regular file of a decided continuation (prints the count)
  "$PY" -I -B - "$RUNS/$1" "$BS/tmp/$1" "$LOGS" "$1" 9>&- <<'PYEOF'
import glob, os, stat, sys
run, tmp, logs, name = sys.argv[1:5]
paths = [os.path.join(d, f) for top in (run, tmp) for d, _dirs, files in os.walk(top) for f in files]
paths += glob.glob(os.path.join(glob.escape(logs), glob.escape(name) + ".attempt*.log"))
n = 0
for p in paths:
    try:
        st = os.lstat(p)
        if stat.S_ISREG(st.st_mode):
            os.utime(p, ns=(st.st_mtime_ns - 1, st.st_mtime_ns), follow_symlinks=False)
            n += 1
    except OSError:
        pass
print(n)
PYEOF
}

rearm() {
  local n
  if n=$(rearm_atimes "$1"); then
    note "$1: atimes re-armed on $n files (P4: a later read of any shows in logs/atimes_at_ALL_DECIDED.tsv)"
  else
    note "$1: the atime re-arm failed (P4: re-reads of its files cannot be excluded)"
  fi
}

atime_snapshot() {  # PR-M2 / P4: logs/atimes_at_ALL_DECIDED.tsv (lstat only: taking it moves no atime)
  "$PY" -I -B - "$BS" "$LOGS/atimes_at_ALL_DECIDED.tsv" 9>&- <<'PYEOF'
import os, sys
root, out = sys.argv[1:3]
rows, unread = [], 0
for sub in ("runs", "logs", "tmp"):
    for d, _dirs, files in os.walk(os.path.join(root, sub)):
        for f in files:
            p = os.path.join(d, f)
            if p in (out, out + ".tmp"):
                continue
            try:
                st = os.lstat(p)
            except OSError:
                continue
            flag = int(st.st_atime_ns == st.st_mtime_ns - 1)
            unread += flag
            rows.append(f"{os.path.relpath(p, root)}\t{st.st_atime_ns}\t{st.st_mtime_ns}\t{st.st_ctime_ns}\t{flag}")
with open(out + ".tmp", "w") as fh:
    fh.write("path\tatime_ns\tmtime_ns\tctime_ns\trearmed_unread\n")
    fh.writelines(r + "\n" for r in sorted(rows))
os.replace(out + ".tmp", out)
print(f"{len(rows)} files, {unread} still re-armed (atime = mtime - 1 ns)")
PYEOF
}

next_free() {
  local k=1
  while [[ -e "$RUNS/$1_$2$k" ]]; do k=$((k + 1)); done
  echo "$1_$2$k"
}

retire() {  # runs/<name> -> runs/<name>_<kind><k> (inside <BS>/runs); prints the new name
  local new
  new=$(next_free "$1" "$2")
  mv "$RUNS/$1" "$RUNS/$new"
  echo "$new"
}

retire_void() {
  local new
  new=$(next_free "$1" void)
  VOID_SEEN=1
  decide "$1" VOID "$2; folder $new"
  mv "$RUNS/$1" "$RUNS/$new"
}

attempt_log() {
  local k=1
  while [[ -e "$LOGS/$1.attempt$k.log" ]]; do k=$((k + 1)); done
  echo "$LOGS/$1.attempt$k.log"
}

in_lost() {
  [[ -f "$LOST_JSON" ]] || return 1
  "$PY" -I -B -c "import json,sys
d = json.load(open(sys.argv[1]))
sys.exit(0 if sys.argv[2] in (d.get('lost') or {}) else 1)" "$LOST_JSON" "$1" 9>&-
}

write_lost() {
  "$PY" -I -B - "$LOST_JSON" "$1" "$2" "$LOST_SCHEMA" 9>&- <<'PYEOF'
import datetime, json, os, sys
path, name, detail, schema = sys.argv[1:5]
d = json.load(open(path)) if os.path.exists(path) else {"schema": schema, "lost": {}}
if d.get("schema") != schema:
    sys.exit(f"{path}: schema {d.get('schema')!r} != {schema!r}")
d.setdefault("lost", {})[name] = {"cause": "crashes", "detail": detail, "by": "run_waves_bs.sh",
                                  "utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
tmp = path + ".tmp"
with open(tmp, "w") as fh:
    json.dump(d, fh, indent=1)
    fh.write("\n")
os.replace(tmp, path)
PYEOF
}

free_bytes() {
  if [[ "${BS_WAVES_TEST:-}" == 1 && -n "${BS_FREE_BYTES_FILE:-}" ]]; then cat "$BS_FREE_BYTES_FILE"; return; fi
  df -B1 --output=avail "$BS" | tail -n 1 | tr -d ' '
}

proctable() {
  { echo "== $(date -u +%Y-%m-%dT%H:%M:%SZ) start $1"; echo "load: $(cat /proc/loadavg)"
    "${PS_CMD[@]}" 2>> "$ERRLOG" 9>&- | head -n 12 | cut -c1-200 || true; } >> "$LOGS/proctable.log"
}

alive() { kill -0 "$1" 2>> "$ERRLOG"; }

orphans() { [[ -n "$(pgrep -f "$ORPHAN_PATTERN" 9>&- || true)" ]]; }

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
    PENDING=("$name" "${PENDING[@]+"${PENDING[@]}"}")
  fi
}

after_exit() {  # after_exit <name> <rc>
  local name=$1 rc=$2 run=$RUNS/$1 crc reason new arm_ seed_
  if (( rc == 0 || rc == POST_CHECK )); then
    set +e
    "${CHECK_CMD[@]}" "$run" >> "$LOGS/$name.check.log" 2>&1 9>&-
    crc=$?
    set -e
    if (( crc == 0 && rc == 0 )); then
      reason=$(meta_field "$run" stop_reason)
      case "$reason" in
        end) decide "$name" COUNTED "CHECK passed, stop reason end"; rearm "$name" ;;
        collapse) decide "$name" STOPPED "CHECK passed, stop reason collapse"; rearm "$name" ;;
        *) retire_void "$name" "CHECK passed but stop reason '$reason'" ;;
      esac
    elif (( crc == 4 )); then
      retire_void "$name" "CHECK failed (logs/$name.check.log; run exit $rc)"
    elif (( crc == 0 )); then
      retire_void "$name" "run exit $rc (a post-run check failed) although CHECK passed (logs/$name.check.log)"
    else
      HALT=1
      HALT_WHY+=("CHECK ERROR on $name")
      note "$name: CHECK ERROR (read_bs exit $crc, logs/$name.check.log); folder left as it is; no new start"
    fi
  elif (( rc == 2 )); then
    new=none
    [[ -d "$run" ]] && new=$(retire "$name" refused)
    HALT=1
    REFUSED+=("$name")
    HALT_WHY+=("refused $name")
    read -r arm_ seed_ <<< "${ARGS_OF[$name]}"
    note "$name: REFUSED before any optimizer step (an attempt that exits 2 never reaches learn(); folder $new); the" \
         "cause: re-run, read only, cd /tmp && ${RUN_CMD[*]} --arm $arm_ --seed $seed_ --preflight-only" \
         "$([[ "$new" != none ]] && echo "(or runs/$new/run_meta.json error and traceback)")" \
         "- never the attempt log; no new start"
  else
    record_crash "$name" "exit $rc" 1
  fi
}

# ------------------------------------------------------------------------------------------------------------ start
mkdir -p "$LOGS" "$RUNS"
exec 9> "$LOGS/run_waves_bs.lock"
if ! flock -w "$LOCK_WAIT" 9; then
  echo "run_waves_bs: another run_waves_bs.sh holds $LOGS/run_waves_bs.lock (waited ${LOCK_WAIT} s; exit 3)" >&2
  exit 3
fi
touch "$DECIDED" "$CRASHES"
trap 'echo "$? $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOGS/run_waves_bs.exit"' EXIT
if [[ -e "$HALTED_FILE" ]]; then
  die "halted ($(head -n 1 "$HALTED_FILE" | cut -c1-300)); the coordinator fixes the cause and removes $HALTED_FILE"
fi
if [[ ! -s "$DECIDED" && ! -s "$CRASHES" && -z "$(ls -A "$RUNS")" ]]; then
  free=$(free_bytes)
  (( free >= LAUNCH )) || die "launch: $free bytes free < the launch floor $LAUNCH bytes: housekeeping first"
  note "launch: $free bytes free >= the launch floor $LAUNCH bytes; queue ${NAMES[*]}"
fi

while orphans; do
  note "a run_bs.py process is still alive (pgrep -f '$ORPHAN_PATTERN'); waiting for it to finish"
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
ORDERED=()
for name in "${NAMES[@]}"; do
  for p in "${PENDING[@]+"${PENDING[@]}"}"; do [[ "$p" == "$name" ]] && { ORDERED+=("$name"); break; }; done
done
PENDING=("${ORDERED[@]+"${ORDERED[@]}"}")
if (( VOID_SEEN )); then
  halt_marker "VOID in logs/decided.tsv: no continuation is started; the coordinator fixes the cause"
  exit 4
fi

PAUSED=0
note "queue: ${#PENDING[@]} pending of ${#NAMES[@]}; $SLOTS slots"
while true; do
  while (( ${#PID[@]} < SLOTS && ${#PENDING[@]} > 0 && HALT == 0 && VOID_SEEN == 0 )); do
    free=$(free_bytes)
    if (( free < FLOOR )); then
      if (( ! PAUSED )); then
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) free $free < floor $FLOOR bytes under $BS" > "$LOGS/PAUSED_disk"
        note "PAUSED: $free bytes free < $FLOOR; no new start until space is freed (tell the maintainer)"
        PAUSED=1
      fi
      break
    fi
    if (( PAUSED )); then rm -f "$LOGS/PAUSED_disk"; note "resumed: $free bytes free"; PAUSED=0; fi
    name=${PENDING[0]}
    PENDING=("${PENDING[@]:1}")
    read -r arm seed <<< "${ARGS_OF[$name]}"
    log=$(attempt_log "$name")
    LOG_OF[$name]=$log
    proctable "$name"
    note "start $name (${log#"$LOGS"/})"
    (exec 9>&-; cd /tmp && exec "${RUN_CMD[@]}" --arm "$arm" --seed "$seed") >> "$log" 2>&1 &
    PID[$name]=$!
  done
  if (( ${#PID[@]} == 0 )); then
    if (( ${#PENDING[@]} > 0 && HALT == 0 && VOID_SEEN == 0 )); then sleep "$POLL"; continue; fi
    break
  fi
  sleep "$POLL"
  for name in "${!PID[@]}"; do
    if ! alive "${PID[$name]}"; then
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
  halt_marker "VOID (logs/decided.tsv); the coordinator fixes the cause"
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
if snap=$(atime_snapshot); then
  note "atimes at ALL_DECIDED (P4), before the full reads: logs/atimes_at_ALL_DECIDED.tsv, $snap"
else
  note "the atime snapshot at ALL_DECIDED failed (P4: re-reads of CHECK's inputs cannot be excluded)"
fi
date -u +%Y-%m-%dT%H:%M:%SZ > "$LOGS/ALL_DECIDED"
note "all ${#NAMES[@]} decided, none VOID: the full reads (read_bs.py, the markers), $SLOTS at a time"

READ_OK=()
for name in "${NAMES[@]}"; do
  d=$(decided "$name")
  [[ "$d" == COUNTED || "$d" == STOPPED ]] && READ_OK+=("$name")
done
declare -A RPID=()
TODO=("${READ_OK[@]+"${READ_OK[@]}"}")
while (( ${#TODO[@]} > 0 || ${#RPID[@]} > 0 )); do
  while (( ${#RPID[@]} < SLOTS && ${#TODO[@]} > 0 )); do
    name=${TODO[0]}
    TODO=("${TODO[@]:1}")
    (exec 9>&-; cd /tmp && exec "${READ_CMD[@]}" "$RUNS/$name") >> "$LOGS/$name.read.log" 2>&1 &
    RPID[$name]=$!
  done
  sleep "$POLL"
  for name in "${!RPID[@]}"; do
    if ! alive "${RPID[$name]}"; then
      set +e
      wait "${RPID[$name]}"
      rrc=$?
      set -e
      unset "RPID[$name]"
      note "$name: read_bs (the markers) exit $rrc"
    fi
  done
done
exit 0
