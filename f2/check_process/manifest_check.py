"""Read-only checks on round/fetch/manifest.csv and RUNLOG F2-START lines.

argv: manifest.csv RUNLOG
Prints: per-fetch row counts, F2 row time range, F2 rows before each run's F2-START, F2 rows per run by status,
and the list of F2 COPY rows (run, path, sha256, stored_as) as JSON on the last line.
"""
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

man, runlog = sys.argv[1], sys.argv[2]
starts = {}
with open(runlog, encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        m = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}(?::\d{2})?) UTC\s+STANCE7 F2-START (\S+)", line)
        if m:
            ts = m.group(1)
            fmt = "%Y-%m-%d %H:%M:%S" if ts.count(":") == 2 else "%Y-%m-%d %H:%M"
            starts.setdefault(m.group(2), []).append((i, datetime.strptime(ts, fmt).replace(tzinfo=timezone.utc)))
print("F2-START lines:", {k: [(i, t.isoformat()) for i, t in v] for k, v in starts.items()})
rows = list(csv.DictReader(open(man, newline="", encoding="utf-8")))
print("rows:", len(rows), "by fetch:", dict(Counter(r["fetch"] for r in rows)))
f2 = [r for r in rows if r["fetch"] == "F2"]
times = sorted(r["utc"] for r in f2)
print("F2 rows:", len(f2), "first:", times[0] if times else None, "last:", times[-1] if times else None)
latest_start = max(t for v in starts.values() for _, t in v)
early = []
for r in f2:
    t = datetime.strptime(r["utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    st = starts.get(r["run_id"])
    if t < latest_start or not st or t < st[0][1] or t < datetime(2026, 11, 2, 16, 39, tzinfo=timezone.utc):
        early.append((r["utc"], r["run_id"], r["path"], r["status"]))
print("F2 rows before any F2-START / 16:39:", early)
other_after = [(r["utc"], r["fetch"], r["run_id"], r["path"], r["status"]) for r in rows
               if r["fetch"] != "F2" and r["utc"] >= "2026-11-02T16:39:00Z"]
print("non-F2 rows at or after 16:39:", other_after)
by = defaultdict(Counter)
for r in f2:
    by[r["run_id"]][r["status"]] += 1
print("F2 statuses by run:", {k: dict(v) for k, v in by.items()})
methods = Counter((r["status"], r["method"]) for r in f2)
print("F2 (status, method):", dict(methods))
copies = [{k: r[k] for k in ("utc", "run_id", "path", "drive_id", "listing_size", "transfer", "decoded_size",
                             "status", "copy", "sha256", "stored_as", "method", "note")}
          for r in f2 if r["status"] not in ("LISTING",)]
print("F2_NONLISTING_JSON " + json.dumps(copies))
