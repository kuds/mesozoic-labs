"""Read-only: peak number of simulation processes (manip_check / M1 / M3) running at once across the six measuring
agents, from each process's redirect files: start = birth time of <name>.err (created by the shell redirect when the
process starts), end = latest mtime of <name>.out / <name>.exit. Also cross-checks against the panel records' own
utc (end) and child seconds where present.

argv: measure_dir
"""
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

base = Path(sys.argv[1])
iv = []
pat = re.compile(r"^(c8|manip|m1|m3)_(\d+|final)\.err$")
for run in sorted(p for p in base.iterdir() if p.is_dir()):
    for f in sorted(run.iterdir()):
        m = pat.match(f.name)
        if not m:
            continue
        stem = f.name[:-4]
        bw = subprocess.run(["stat", "-c", "%w", str(f)], capture_output=True, text=True).stdout.strip()
        if bw and bw != "-":
            d, t, z = bw.split()
            whole, frac = t.split(".")
            start = datetime.strptime(f"{d} {whole}", "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=timezone.utc).timestamp() + float("0." + frac)
        else:
            start = os.stat(f).st_mtime
        ends = [os.stat(run / f"{stem}{ext}").st_mtime for ext in (".out", ".exit") if (run / f"{stem}{ext}").exists()]
        end = max(ends) if ends else None
        kind = "c8" if m.group(1) in ("c8", "manip") else m.group(1)
        iv.append((start, end, run.name, kind, m.group(2)))
print("simulation processes found:", len(iv))
by = {}
for s, e, r, k, lab in iv:
    by.setdefault(r, []).append(k)
for r, ks in by.items():
    print(f"  {r}: c8 {ks.count('c8')}, m1 {ks.count('m1')}, m3 {ks.count('m3')}")
ev = []
for s, e, r, k, lab in iv:
    if e is None:
        print("NO END:", r, k, lab)
        continue
    ev.append((s, 1, r, k, lab))
    ev.append((e, -1, r, k, lab))
ev.sort(key=lambda x: (x[0], x[1]))
cur, peak, peak_at, running = 0, 0, None, set()
per_agent_over = []
for t, d, r, k, lab in ev:
    if d == 1:
        running.add((r, k, lab))
    else:
        running.discard((r, k, lab))
    cur += d
    agents = {}
    for rr, kk, ll in running:
        agents[rr] = agents.get(rr, 0) + 1
    if any(v > 1 for v in agents.values()):
        per_agent_over.append((t, sorted(running)))
    if cur > peak:
        peak, peak_at = cur, (t, sorted(running))
from datetime import datetime, timezone
print("peak concurrent simulation processes:", peak, "at",
      datetime.fromtimestamp(peak_at[0], timezone.utc).isoformat() if peak_at else None, peak_at[1] if peak_at else None)
print("moments with two processes of one agent at once:", len(per_agent_over))
for t, rs in per_agent_over[:5]:
    print("  ", datetime.fromtimestamp(t, timezone.utc).isoformat(), rs)
over2 = []
cur = 0
running = set()
for t, d, r, k, lab in ev:
    if d == 1:
        running.add((r, k, lab))
    else:
        running.discard((r, k, lab))
    if len(running) > 2:
        over2.append((datetime.fromtimestamp(t, timezone.utc).isoformat(), sorted(running)))
print("moments with more than 2 at once:", len(over2))
for x in over2[:8]:
    print("  ", x)
