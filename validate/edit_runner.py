"""PS build: the edits of run_waves_bs.sh (each old string asserted unique), run once.  RR-1 and I1 are LB's code
(S2/r10round/long_blind/tools/run_waves_bs.sh, its hunks #58-#61, #63, #67), ported verbatim."""
import sys
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_text()
END = "PY" + "EOF"          # the inner heredoc delimiter, never written literally in this file


def rep(old, new):
    global s
    old, new = old.replace("@END@", END), new.replace("@END@", END)
    assert s.count(old) == 1, (s.count(old), old[:90])
    s = s.replace(old, new)


rep('''#!/usr/bin/env bash
# BS2: BS's run_waves_bs.sh (sha256 13fb6182...) with the root changed (BS_REAL = S2/r10round/bias_screen2); the
# queue mechanics, slots, nice, floors, crash rules and the outcome-blind reads are BS's, unchanged.  "<BS>" below = BS2.''',
    '''#!/usr/bin/env bash
# PS: BS2's run_waves_bs.sh (S2/r10round/bias_screen2/tools/run_waves_bs.sh, sha256 6e0484f6...) with the root changed
# (BS_REAL = /home/user/work/phase_screen) and LB's two process fixes ported (S2/r10round/long_blind/tools, its
# hunks #58-#61, #63, #67); "<BS>" below = PS:
#   RR-1   a relaunch after ALL_DECIDED keeps the first logs/atimes_at_ALL_DECIDED.tsv; a later snapshot goes to
#          logs/atimes_at_ALL_DECIDED.relaunch<k>.tsv (and lists no snapshot file); logs/ALL_DECIDED keeps its first time.
#   I1     right after each attempt exits, before CHECK, the re-arm and any rename, logs/atimes_at_exit/<name>.attempt<k>
#          .tsv records path, atime, mtime and ctime (lstat only) of runs/<name>/**, tmp/<name>/** and the attempt logs.
# The queue mechanics, slots, nice, floors and the outcome-blind reads are BS2's, unchanged.
# BS2: BS's run_waves_bs.sh (sha256 13fb6182...) with the root changed (BS_REAL = S2/r10round/bias_screen2); the
# queue mechanics, slots, nice, floors, crash rules and the outcome-blind reads are BS's, unchanged.''')
rep('''BS_REAL=$S2/r10round/bias_screen2''', '''BS_REAL=/home/user/work/phase_screen''')
rep('''decided() { awk -F'\\t' -v n="$1" '$2 == n {d = $3} END {print d}' "$DECIDED"; }''',
    '''current_attempt() {  # LB / PS: the number of the newest logs/<name>.attempt<k>.log (0 when none)
  local k=0
  while [[ -e "$LOGS/$1.attempt$((k + 1)).log" ]]; do k=$((k + 1)); done
  echo "$k"
}

exit_snapshot() {  # LB / PS, process I1: lstat (no file opened) of an attempt's files right after it exits
  local k
  k=$(current_attempt "$1")
  mkdir -p "$LOGS/atimes_at_exit"
  "$PY" -I -B - "$RUNS/$1" "$BS/tmp/$1" "$LOGS" "$1" "$LOGS/atimes_at_exit/$1.attempt$k.tsv" 9>&- <<'@END@'
import glob, os, stat, sys
run, tmp, logs, name, out = sys.argv[1:6]
paths = [os.path.join(d, f) for top in (run, tmp) for d, _dirs, files in os.walk(top) for f in files]
paths += glob.glob(os.path.join(glob.escape(logs), glob.escape(name) + ".attempt*.log"))
rows = []
for p in sorted(paths):
    try:
        st = os.lstat(p)
    except OSError:
        continue
    if stat.S_ISREG(st.st_mode):
        rows.append(f"{p}\\t{st.st_atime_ns}\\t{st.st_mtime_ns}\\t{st.st_ctime_ns}\\t{int(st.st_atime_ns > st.st_mtime_ns)}")
with open(out + ".tmp", "w") as fh:
    fh.write("path\\tatime_ns\\tmtime_ns\\tctime_ns\\tatime_after_mtime\\n")
    fh.writelines(r + "\\n" for r in rows)
os.replace(out + ".tmp", out)
print(f"{len(rows)} files")
@END@
}

decided() { awk -F'\\t' -v n="$1" '$2 == n {d = $3} END {print d}' "$DECIDED"; }''')
rep('''atime_snapshot() {  # PR-M2 / P4: logs/atimes_at_ALL_DECIDED.tsv (lstat only: taking it moves no atime)
  "$PY" -I -B - "$BS" "$LOGS/atimes_at_ALL_DECIDED.tsv" 9>&- <<'@END@'
import os, sys
root, out = sys.argv[1:3]
rows, unread = [], 0
for sub in ("runs", "logs", "tmp"):
    for d, _dirs, files in os.walk(os.path.join(root, sub)):
        for f in files:
            p = os.path.join(d, f)
            if p in (out, out + ".tmp"):
                continue''', '''atime_snapshot() {  # PR-M2 / P4: logs/atimes_at_ALL_DECIDED.tsv (lstat only: taking it moves no atime)
  # RR-1 (LB / PS): a relaunch after ALL_DECIDED keeps the first snapshot; this one goes to .relaunch<k>.tsv
  local out=$LOGS/atimes_at_ALL_DECIDED.tsv k=1
  if [[ -e "$out" ]]; then
    while [[ -e "$LOGS/atimes_at_ALL_DECIDED.relaunch$k.tsv" ]]; do k=$((k + 1)); done
    out=$LOGS/atimes_at_ALL_DECIDED.relaunch$k.tsv
  fi
  "$PY" -I -B - "$BS" "$out" 9>&- <<'@END@'
import os, sys
root, out = sys.argv[1:3]
rows, unread = [], 0
for sub in ("runs", "logs", "tmp"):
    for d, _dirs, files in os.walk(os.path.join(root, sub)):
        for f in files:
            p = os.path.join(d, f)
            if p in (out, out + ".tmp") or os.path.basename(p).startswith("atimes_at_ALL_DECIDED"):
                continue''')
rep('''os.replace(out + ".tmp", out)
print(f"{len(rows)} files, {unread} still re-armed (atime = mtime - 1 ns)")
@END@
}''', '''os.replace(out + ".tmp", out)
print(f"{os.path.basename(out)}: {len(rows)} files, {unread} still re-armed (atime = mtime - 1 ns)")
@END@
}''')
rep('''after_exit() {  # after_exit <name> <rc>
  local name=$1 rc=$2 run=$RUNS/$1 crc reason new arm_ seed_
  if (( rc == 0 || rc == POST_CHECK )); then''', '''after_exit() {  # after_exit <name> <rc>
  local name=$1 rc=$2 run=$RUNS/$1 crc reason new arm_ seed_ snap
  if snap=$(exit_snapshot "$name"); then   # I1 (LB / PS): before CHECK, the re-arm and any rename
    note "$name: atimes at exit (I1, lstat): logs/atimes_at_exit/$name.attempt$(current_attempt "$name").tsv, $snap"
  else
    note "$name: the atime snapshot at exit failed (I1)"
  fi
  if (( rc == 0 || rc == POST_CHECK )); then''')
rep('''date -u +%Y-%m-%dT%H:%M:%SZ > "$LOGS/ALL_DECIDED"''', '''if [[ -e "$LOGS/ALL_DECIDED" ]]; then   # RR-1 (LB / PS): a relaunch keeps the first ALL_DECIDED
  note "ALL_DECIDED kept as first written ($(head -n 1 "$LOGS/ALL_DECIDED")); relaunch at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
else
  date -u +%Y-%m-%dT%H:%M:%SZ > "$LOGS/ALL_DECIDED"
fi''')
p.write_text(s)
print("ok")
