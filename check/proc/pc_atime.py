"""Process lens: under relatime, a file's atime moves to the read time on the first read after its last write.  List every
file of a continuation (runs/<name>/**, logs/<name>.*) whose atime is after its mtime (read after last write) and before
logs/ALL_DECIDED's mtime, i.e. read before all were decided.  Reads only; prints."""
import datetime as dt
import os
import sys

BS = sys.argv[1]
ad = os.stat(os.path.join(BS, "logs", "ALL_DECIDED")).st_mtime_ns


def t(ns):
    return dt.datetime.fromtimestamp(ns / 1e9, dt.timezone.utc).strftime("%H:%M:%S.%f")


print("ALL_DECIDED mtime", t(ad))
early, late, never = [], [], []
paths = []
for name in sorted(os.listdir(os.path.join(BS, "runs"))):
    for dp, dn, fn in os.walk(os.path.join(BS, "runs", name)):
        paths += [os.path.join(dp, f) for f in fn]
for f in sorted(os.listdir(os.path.join(BS, "logs"))):
    paths.append(os.path.join(BS, "logs", f))
for p in paths:
    st = os.stat(p)
    rel = os.path.relpath(p, BS)
    if st.st_atime_ns > st.st_mtime_ns:
        (early if st.st_atime_ns < ad else late).append((rel, t(st.st_mtime_ns), t(st.st_atime_ns)))
    else:
        never.append(rel)
print("READ AFTER LAST WRITE, BEFORE ALL_DECIDED:")
for r in early:
    print("  ", r)
print("READ AFTER LAST WRITE, AT/AFTER ALL_DECIDED:", len(late))
for r in late:
    print("  ", r)
print("NOT READ SINCE LAST WRITE:", len(never))
for r in never:
    print("  ", r)
