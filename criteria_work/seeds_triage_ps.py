"""PS design input: strict triage of the candidate seeds' text hits (seeds_check_ps.py's broad search printed only the
first 8 seed-like lines per number).  Read only; prints to stdout.

A hit is SEED-CONTEXT when the number appears (as a whole number, not inside a longer number, a decimal or a hex
string) on a line that also names a seed or a queue entry within 60 characters: 'seed', 'SEED', '--seed', 'queue',
'derived', 'env_seeds', or an arm/run name joined to the number (control_N, critic_..._N, blind_N, phase_N, _sN, sN_).
Every SEED-CONTEXT hit is printed in full (path:line: text, cut at 220 characters).  Files and folders as
seeds_check_ps.py (S1, S2, /home/user/work; this folder, .git, venv, site-packages, caches excluded; text-like files
under 2 MiB).
Usage: cd /tmp && nice -n 10 PY -I -B seeds_triage_ps.py > seeds_triage_ps.out 2> seeds_triage_ps.err
"""
import os
import re
from pathlib import Path

CW = Path(__file__).resolve().parent
WORK = CW.parent.parent
import sys as _sys
# S2 is the venv owner: PY = S2/venv/bin/python (run with PY -I -B); S1 is its sibling session scratchpad.
S2 = Path(_sys.executable).parents[2]
S1 = S2.parent.parent / "0fdf74a4-2989-56c9-a773-92101004d7de" / "scratchpad"
assert (S2 / "r10round/bias_screen2").is_dir() and (S1 / "foot/stance6").is_dir(), "run with S2/venv/bin/python"


def clean(s):
    """Printed text without the scratchpads' temp-root prefix (short names instead)."""
    return s.replace(str(S2), "S2").replace(str(S1), "S1").replace(str(S2.parent.parent) + "/", "<TMPROOT>/")
SETS = {"2601 / 2611 / 2621 / 2631": (2601, 2611, 2621, 2631), "2901 / 2911 / 2921 / 2931": (2901, 2911, 2921, 2931)}


def derived(s):
    return (s, s + 1, s + 2, s + 3, s + 1000)


def short(p):
    s = str(p)
    for k, v in ((str(S1), "S1"), (str(S2), "S2"), (str(WORK), "W")):
        s = s.replace(k, v)
    return s


nums = sorted({x for seeds in SETS.values() for s in seeds for x in derived(s)})
NUM = r"(?<![\w.])(" + "|".join(map(str, nums)) + r")(?![\w]|\.\d)"
pat = re.compile(NUM)
CTX = re.compile(r"seed|SEED|queue|derived|env_seeds|control_|critic_|blind_|phase_|\bs\d{2,4}_|_s\d{2,4}\b")
EXT = {".py", ".txt", ".json", ".md", ".sh", ".toml", ".log", ".out", ".jsonl", ".tsv", ".csv", ".js", ".yaml", ".yml",
       ".cfg", ".ini", ".err"}
hits = {n: [] for n in nums}
scanned = 0
for root in (S1, S2, WORK):
    for dp, dn, fn in os.walk(root):
        if str(dp).startswith(str(CW)):
            dn[:] = []
            continue
        dn[:] = [d for d in dn if d not in (".git", "venv", "__pycache__", "node_modules", "caches", "site-packages")]
        for f in fn:
            p = Path(dp) / f
            if p.suffix not in EXT:
                continue
            try:
                if p.is_symlink() or p.stat().st_size > 2 * 1024 * 1024:
                    continue
                txt = p.read_text(errors="replace")
            except OSError:
                continue
            scanned += 1
            for ln, line in enumerate(txt.splitlines(), 1):
                for m in pat.finditer(line):
                    lo, hi = max(0, m.start() - 60), m.end() + 60
                    if CTX.search(line[lo:hi]):
                        hits[int(m.group(1))].append(f"{short(p)}:{ln}: {clean(line.strip())[:220]}")
print(f"files scanned {scanned}")
for label, seeds in SETS.items():
    total = sum(len(hits[x]) for s in seeds for x in derived(s))
    print(f"\nSET {label}: {total} seed-context hit(s)")
    for s in seeds:
        for x in derived(s):
            for h in hits[x]:
                print(f"  {x}: {h}")
print("\nEND")
