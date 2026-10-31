"""The fixer's edit of tools/run_waves_bs.sh (findings PR-M2, PR-M3, PR-M4): exact old -> new replacements, each old
text present exactly once.  Run: PY -I -B edit_runner.py <BS2>/tools/run_waves_bs.sh"""
import sys

p = sys.argv[1]
src = open(p, encoding="utf-8").read()
E = []

E.append(('''# BS2: BS's run_waves_bs.sh (sha256 13fb6182...) with only the root changed (BS_REAL = S2/r10round/bias_screen2); the
# queue mechanics, slots, nice, floors, crash rules and the outcome-blind reads are BS's, unchanged.  "<BS>" below = BS2.
''', '''# BS2: BS's run_waves_bs.sh (sha256 13fb6182...) with the root changed (BS_REAL = S2/r10round/bias_screen2); the
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
'''))

E.append(('''# stop_reason, heartbeat, last_step) and the CHECK's exit code (read_bs.py --check).  It never opens an attempt's log,
# a ledger, a probe or a checkpoint.''', '''# stop_reason, heartbeat, last_step) and the CHECK's exit code (read_bs.py --check).  It never opens an attempt's log,
# a ledger, a probe or a checkpoint (the atime re-arm and snapshot use lstat and utime only: no file is opened).'''))

E.append(('''[[ -n "$QUEUE_OVERRIDE" || -f "$BS/SCREEN.json" ]] || die "no <BS>/SCREEN.json"
''', '''[[ -n "$QUEUE_OVERRIDE" || -f "$BS/SCREEN.json" ]] || die "no <BS>/SCREEN.json"
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
'''))

E.append(('''own_crashes() { awk -F'\\t' -v n="$1" '$2 == n && $5 == 1 {c++} END {print c + 0}' "$CRASHES"; }
''', '''own_crashes() { awk -F'\\t' -v n="$1" '$2 == n && $5 == 1 {c++} END {print c + 0}' "$CRASHES"; }

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
            rows.append(f"{os.path.relpath(p, root)}\\t{st.st_atime_ns}\\t{st.st_mtime_ns}\\t{st.st_ctime_ns}\\t{flag}")
with open(out + ".tmp", "w") as fh:
    fh.write("path\\tatime_ns\\tmtime_ns\\tctime_ns\\trearmed_unread\\n")
    fh.writelines(r + "\\n" for r in sorted(rows))
os.replace(out + ".tmp", out)
print(f"{len(rows)} files, {unread} still re-armed (atime = mtime - 1 ns)")
PYEOF
}
'''))

E.append(('''        end) decide "$name" COUNTED "CHECK passed, stop reason end" ;;
        collapse) decide "$name" STOPPED "CHECK passed, stop reason collapse" ;;''',
          '''        end) decide "$name" COUNTED "CHECK passed, stop reason end"; rearm "$name" ;;
        collapse) decide "$name" STOPPED "CHECK passed, stop reason collapse"; rearm "$name" ;;'''))

E.append(('''  local name=$1 rc=$2 run=$RUNS/$1 crc reason new
''', '''  local name=$1 rc=$2 run=$RUNS/$1 crc reason new arm_ seed_
'''))

E.append(('''    note "$name: REFUSED before any optimizer step (the reason: ${LOG_OF[$name]:-runs/$new/run_meta.json error}," \\
         "an attempt that exits 2 never reaches learn(); folder $new); no new start"''',
          '''    read -r arm_ seed_ <<< "${ARGS_OF[$name]}"
    note "$name: REFUSED before any optimizer step (an attempt that exits 2 never reaches learn(); folder $new); the" \\
         "cause: re-run, read only, cd /tmp && ${RUN_CMD[*]} --arm $arm_ --seed $seed_ --preflight-only" \\
         "$([[ "$new" != none ]] && echo "(or runs/$new/run_meta.json error and traceback)")" \\
         "- never the attempt log; no new start"'''))

E.append(('''date -u +%Y-%m-%dT%H:%M:%SZ > "$LOGS/ALL_DECIDED"
''', '''if snap=$(atime_snapshot); then
  note "atimes at ALL_DECIDED (P4), before the full reads: logs/atimes_at_ALL_DECIDED.tsv, $snap"
else
  note "the atime snapshot at ALL_DECIDED failed (P4: re-reads of CHECK's inputs cannot be excluded)"
fi
date -u +%Y-%m-%dT%H:%M:%SZ > "$LOGS/ALL_DECIDED"
'''))

for old, new in E:
    assert src.count(old) == 1, old[:100]
    src = src.replace(old, new)
open(p, "w", encoding="utf-8").write(src)
print("ok", len(E))
