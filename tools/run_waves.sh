#!/usr/bin/env bash
# Tail_3 bound screen: the 18 replicates in 6 waves of 3 (criteria RUNS and RUN HANDLING; BUILD_SPEC section 8).
# Runs ONLY after SCREEN/APPROVED and tests_out/gate_pass.json exist; every replicate still passes run_tailscreen.py's
# own preflight (which refuses on any unpinned or changed input).
#
#   run_waves.sh            run (or resume) the waves
#   run_waves.sh --plan     print the waves and the commands; check nothing, launch nothing, write nothing
#
# Per wave (screen_common.WAVES order: (s42,101) (s80,101) (s42,202) (s80,202) (s42,303) (s80,303)):
#   - free disk >= 5 GB, else PAUSED (exit 5; nothing discarded; tell the maintainer);
#   - the three arms (control, b90, b60) of that (state, seed) start together as background
#     `python -I -B tools/run_tailscreen.py ...` processes, logs in logs/<name>.log; the wave waits for all three;
#   - exit 0: `copy_durable.sh replicate <name>` (the durable copy; a push failure is reported, never fatal);
#   - exit 2 (refused on the replicate's own records; nothing created): the screen stops after the wave; the cause is
#     fixed and run_waves.sh is started again (the replicate starts again under the same name);
#   - any other exit (its own crash: exception, killed, out of memory): the crash rule.  `measure_tailscreen.py
#     --retire-crashed <name>` moves runs/<name> -> runs/<name>_crashed<k>, panels/<name> -> panels/<name>_crashed<k>
#     and that replicate's panels.csv rows to panels_crashed.csv; the replicate reruns from the start checkpoint with
#     the same seed and arguments (the rerun IS the replicate).  After a successful rerun the checkpoints both
#     attempts wrote are compared (policy, optimizer and VecNormalize bytes) into logs/<name>_rerun_compare.json,
#     report only (the rerun governs).  A third own crash stops that replicate: LOST is declared by the maintainer
#     in RUNLOG and SCREEN/LOST.txt (never by this script).
# Resume: a replicate whose run_meta.json says "done" is skipped.  A runs/<name> left "starting"/"running" with no
# live run_waves.sh (this script holds logs/run_waves.lock) means the SCREEN was INTERRUPTED, not a crash: the script
# refuses (exit 3) and the maintainer decides (criteria SCREEN INTERRUPTED).
# The durable branch is first created once training has started: after wave 1's first run_meta.json appears,
# `copy_durable.sh init` runs once (logs/durable_init.done).
# The panels: `measure_tailscreen.py --all-pending --follow` (nice 19) is (re)started beside the waves whenever it is
# not running.  Nothing here reads a panel, and nothing is stopped, changed or rerun because of a readout.
# Exit codes: 0 all waves done (LOST candidates reported); 2 refused; 3 screen interrupted state found; 5 paused (disk).
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LC_ALL=C

S2=/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad
SCREEN=$S2/tailround/screen
PY=$S2/venv/bin/python
TOOLS=$SCREEN/tools
LOGS=$SCREEN/logs
RUNS=$SCREEN/runs
STEPS=1048576
MIN_FREE_KB=$((5 * 1024 * 1024))
MAX_OWN_CRASHES=3
WAVES=("s42 101" "s80 101" "s42 202" "s80 202" "s42 303" "s80 303")
ARMS=(control b90 b60)

die() { echo "run_waves: refused: $*" >&2; exit 2; }
note() { echo "[run_waves $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

train_cmd() {  # train_cmd <state> <arm> <seed>
  echo "$PY -I -B $TOOLS/run_tailscreen.py --state $1 --arm $2 --rep-seed $3 --steps $STEPS"
}

if [[ "${1:-}" == "--plan" ]]; then
  [[ $# -eq 1 ]] || die "--plan takes no other argument"
  for w in "${WAVES[@]}"; do
    read -r st seed <<< "$w"
    echo "wave ($st, $seed):"
    for arm in "${ARMS[@]}"; do echo "  $(train_cmd "$st" "$arm" "$seed") > $LOGS/${st}_${arm}_r${seed}.log"; done
  done
  echo "beside: nice -n 19 $PY -I -B $TOOLS/measure_tailscreen.py --all-pending --follow"
  exit 0
fi
[[ $# -eq 0 ]] || die "usage: run_waves.sh [--plan]"
[[ -f "$SCREEN/APPROVED" ]] || die "no SCREEN/APPROVED"
[[ -f "$SCREEN/tests_out/gate_pass.json" ]] || die "no tests_out/gate_pass.json (run tools/gate.sh first)"
# The pinned constants must be the ones the waves above restate.
"$PY" -I -B - "$TOOLS/screen_common.py" "$(printf '%s,' "${WAVES[@]}")" "${ARMS[*]}" "$STEPS" <<'PYEOF' || die "the waves/arms/steps here differ from screen_common"
import importlib.util, sys
s = importlib.util.spec_from_file_location("c", sys.argv[1]); C = importlib.util.module_from_spec(s); s.loader.exec_module(C)
waves = [(w.split()[0], int(w.split()[1])) for w in sys.argv[2].split(",") if w]
ok = waves == [tuple(w) for w in C.WAVES] and sys.argv[3].split() == list(C.ARMS) and int(sys.argv[4]) == C.STEPS
sys.exit(0 if ok else 1)
PYEOF

mkdir -p "$LOGS" "$RUNS"
exec 9> "$LOGS/run_waves.lock"
flock -n 9 || die "another run_waves.sh holds $LOGS/run_waves.lock"

meta_status() {  # meta_status <name>: done | crashed | running | starting | none | unreadable
  local m=$RUNS/$1/run_meta.json
  if [[ ! -d "$RUNS/$1" ]]; then echo none; return; fi
  if [[ ! -f "$m" ]]; then echo unreadable; return; fi
  "$PY" -I -B -c "import json,sys
try: print(json.load(open(sys.argv[1])).get('status') or 'unreadable')
except Exception: print('unreadable')" "$m"
}

# SCREEN INTERRUPTED: anything left half-way by a previous session is the maintainer's call, never a crash.
for w in "${WAVES[@]}"; do
  read -r st seed <<< "$w"
  for arm in "${ARMS[@]}"; do
    name=${st}_${arm}_r${seed}
    s=$(meta_status "$name")
    case "$s" in
      none|done|crashed) ;;
      *) echo "run_waves: SCREEN INTERRUPTED: runs/$name is '$s' with no live run_waves.sh; the maintainer decides" \
           "(criteria SCREEN INTERRUPTED). Nothing was changed." >&2; exit 3 ;;
    esac
  done
done

ensure_measure() {
  if ! pgrep -f "measure_tailscreen.py --all-pending" > /dev/null; then
    note "starting the panel follower (nice 19)"
    nohup nice -n 19 "$PY" -I -B "$TOOLS/measure_tailscreen.py" --all-pending --follow >> "$LOGS/measure_follow.log" 2>&1 &
  fi
}

durable() {  # durable <copy_durable args...>; a failure is reported, never fatal to training
  if ! "$TOOLS/copy_durable.sh" "$@" >> "$LOGS/durable.log" 2>&1; then
    note "WARNING: copy_durable.sh $* failed (see logs/durable.log); the stage is cumulative, the next push retries"
    return 0
  fi
  note "durable copy: $* pushed"
}

durable_init_once() {
  [[ -f "$LOGS/durable_init.done" ]] && return 0
  compgen -G "$RUNS/*/run_meta.json" > /dev/null || return 0
  if "$TOOLS/copy_durable.sh" init >> "$LOGS/durable.log" 2>&1; then
    date -u +%Y-%m-%dT%H:%M:%SZ > "$LOGS/durable_init.done"
    note "durable branch created/updated (init)"
  else
    note "WARNING: copy_durable.sh init failed (see logs/durable.log); retried after the next replicate"
  fi
}

own_crashes() { compgen -G "$RUNS/$1_crashed*" | wc -l; }

compare_rerun() {  # report only: the checkpoints both attempts wrote
  "$PY" -I -B - "$RUNS" "$1" > "$LOGS/$1_rerun_compare.json" <<'PYEOF' || true
import glob, hashlib, json, os, sys, zipfile
runs, name = sys.argv[1], sys.argv[2]
def members(path):
    with zipfile.ZipFile(path) as z:
        return {m: hashlib.sha256(z.read(m)).hexdigest() for m in sorted(z.namelist()) if m.endswith(".pth")}
def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()
new = os.path.join(runs, name)
out = {"replicate": name, "rerun_governs": True, "attempts": {}}
for old in sorted(glob.glob(os.path.join(runs, name + "_crashed*"))):
    rows = {}
    for z in sorted(glob.glob(os.path.join(old, "stage1_*_steps.zip"))):
        label = os.path.basename(z)[len("stage1_"):-len("_steps.zip")]
        z2 = os.path.join(new, os.path.basename(z))
        p1 = os.path.join(old, f"stage1_vecnormalize_{label}_steps.pkl")
        p2 = os.path.join(new, f"stage1_vecnormalize_{label}_steps.pkl")
        if not (os.path.isfile(z2) and os.path.isfile(p1) and os.path.isfile(p2)):
            continue
        a, b = members(z), members(z2)
        rows[label] = {"pth_equal": a == b, "vecnormalize_equal": sha(p1) == sha(p2),
                       "differs": sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))}
    out["attempts"][os.path.basename(old)] = rows
out["any_difference"] = any(not (r["pth_equal"] and r["vecnormalize_equal"])
                            for rows in out["attempts"].values() for r in rows.values())
print(json.dumps(out, indent=1))
PYEOF
  note "rerun of $1 compared with its crashed attempt(s): logs/$1_rerun_compare.json (report only)"
}

free_ok() {
  local free
  free=$(df -Pk "$SCREEN" | awk 'NR==2 {print $4}')
  (( free >= MIN_FREE_KB ))
}

REFUSED=()
LOST_CANDIDATES=()
ensure_measure
for w in "${WAVES[@]}"; do
  read -r st seed <<< "$w"
  todo=()
  for arm in "${ARMS[@]}"; do
    name=${st}_${arm}_r${seed}
    [[ "$(meta_status "$name")" == done ]] && { note "$name done: skipped"; continue; }
    todo+=("$arm")
  done
  while (( ${#todo[@]} )); do
    if ! free_ok; then
      echo "run_waves: PAUSED: below 5 GB free under $SCREEN before wave ($st, $seed); nothing discarded." \
        "Tell the maintainer, free space, and start run_waves.sh again." >&2
      exit 5
    fi
    ensure_measure
    declare -A PID=()
    for arm in "${todo[@]}"; do
      name=${st}_${arm}_r${seed}
      if [[ "$(meta_status "$name")" == crashed ]]; then  # left by a previous session after a crash
        "$PY" -I -B "$TOOLS/measure_tailscreen.py" --retire-crashed "$name" >> "$LOGS/crash_rule.log" 2>&1
      fi
      note "start $name"
      # shellcheck disable=SC2046
      $(train_cmd "$st" "$arm" "$seed") >> "$LOGS/$name.log" 2>&1 &
      PID[$arm]=$!
    done
    # The durable branch is first created once training has started (a run_meta.json exists): wait for the first one
    # while any of the wave's processes is alive, then create it once.
    while [[ ! -f "$LOGS/durable_init.done" ]] && ! compgen -G "$RUNS/*/run_meta.json" > /dev/null; do
      alive=0
      for arm in "${todo[@]}"; do kill -0 "${PID[$arm]}" 2> /dev/null && alive=1; done
      (( alive )) || break
      sleep 5
    done
    durable_init_once
    next=()
    for arm in "${todo[@]}"; do
      name=${st}_${arm}_r${seed}
      set +e
      wait "${PID[$arm]}"
      rc=$?
      set -e
      note "$name exited $rc"
      durable_init_once
      if (( rc == 0 )); then
        durable replicate "$name"
        (( $(own_crashes "$name") > 0 )) && compare_rerun "$name"
      elif (( rc == 2 )); then
        REFUSED+=("$name")
      else
        "$PY" -I -B "$TOOLS/measure_tailscreen.py" --retire-crashed "$name" >> "$LOGS/crash_rule.log" 2>&1 \
          || die "the crash rule could not retire $name (see logs/crash_rule.log); the maintainer decides"
        n=$(own_crashes "$name")
        if (( n >= MAX_OWN_CRASHES )); then
          note "$name: $n own crashes: stopped. LOST candidate: the maintainer declares it in RUNLOG and LOST.txt"
          LOST_CANDIDATES+=("$name")
        else
          note "$name: own crash $n of $MAX_OWN_CRASHES: rerun from the start checkpoint, same seed and arguments"
          next+=("$arm")
        fi
      fi
    done
    unset PID
    todo=("${next[@]+"${next[@]}"}")
  done
  if (( ${#REFUSED[@]} )); then
    echo "run_waves: refused replicate(s) ${REFUSED[*]} (see their logs): fix the cause, then start run_waves.sh" \
      "again (they start again under the same names). Stopped after wave ($st, $seed)." >&2
    exit 2
  fi
done
ensure_measure
note "all waves done.${LOST_CANDIDATES[*]:+ LOST candidates for the maintainer: ${LOST_CANDIDATES[*]}}"
note "panels: the follower finishes the remaining checkpoints (logs/measure_follow.log); then tools/score_tailscreen.py"
exit 0
