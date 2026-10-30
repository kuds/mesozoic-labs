"""P4 audit: the runner's atime snapshot at ALL_DECIDED against the re-arm rule, plus my own scan taken before reading."""
import os, sys, json
from decimal import Decimal

BS2 = sys.argv[1]
MYSCAN = sys.argv[2]
QUEUE = ["control_2501", "critic_blind_tail3_2501", "control_2511", "critic_blind_tail3_2511",
         "control_2521", "critic_blind_tail3_2521", "control_2531", "critic_blind_tail3_2531"]

snap = {}
with open(os.path.join(BS2, "logs", "atimes_at_ALL_DECIDED.tsv")) as fh:
    hdr = fh.readline().rstrip("\n").split("\t")
    assert hdr == ["path", "atime_ns", "mtime_ns", "ctime_ns", "rearmed_unread"], hdr
    for line in fh:
        p, a, m, c, f = line.rstrip("\n").split("\t")
        snap[p] = (int(a), int(m), int(c), int(f))
print("snapshot rows", len(snap), "flag=1:", sum(v[3] for v in snap.values()))
# my own recomputation of the flag
bad_flag = [p for p, (a, m, c, f) in snap.items() if int(a == m - 1) != f]
print("rows whose flag != (atime == mtime - 1):", bad_flag)

def owned(p, name):
    return (p.startswith(f"runs/{name}/") or p.startswith(f"tmp/{name}/")
            or (p.startswith(f"logs/{name}.attempt") and p.endswith(".log")))

per = {}
for name in QUEUE:
    rows = {p: v for p, v in snap.items() if owned(p, name)}
    not_unread = {p: v for p, v in rows.items() if v[3] != 1}
    per[name] = (len(rows), len(not_unread))
    print(f"{name}: {len(rows)} files in snapshot, {len(rows)-len(not_unread)} rearmed_unread=1, not: {sorted(not_unread)}")
# other rows with flag 0
print("\nrows with rearmed_unread=0 (not owned by a queue continuation):")
for p, (a, m, c, f) in sorted(snap.items()):
    if f == 0 and not any(owned(p, n) for n in QUEUE):
        rel = "atime>max(m,c)" if a > max(m, c) else "atime<=max(m,c)"
        print(f"  {p}\t{rel}")
# any snapshot row of a crashed/void/refused folder?
print("\nrows under runs/ not in a queue folder:", [p for p in snap if p.startswith("runs/") and not any(owned(p, n) for n in QUEUE)])

# current state of each snapshot file vs snapshot: mtime changes, and files now present but absent from snapshot
changed, missing_now = [], []
for p, (a, m, c, f) in snap.items():
    fp = os.path.join(BS2, p)
    try:
        st = os.lstat(fp)
    except FileNotFoundError:
        missing_now.append(p); continue
    if st.st_mtime_ns != m:
        changed.append((p, m, st.st_mtime_ns))
print("\nsnapshot files whose mtime changed since the snapshot:", [x[0] for x in changed])
print("snapshot files missing now:", missing_now)
new_files = []
for sub in ("runs", "tmp"):
    for d, _ds, fs in os.walk(os.path.join(BS2, sub)):
        for f in fs:
            rel = os.path.relpath(os.path.join(d, f), BS2)
            if rel not in snap:
                new_files.append(rel)
print("files under runs/ tmp/ now that are not in the snapshot:", len(new_files))
for x in sorted(new_files)[:50]:
    print("   ", x)

# my scan (taken before I opened anything): which continuation files were read after the snapshot and when
myscan = {}
with open(MYSCAN) as fh:
    for line in fh:
        A, T, C, B, s, mode, typ, p = line.rstrip("\n").split("\t")
        if typ != "f":
            continue
        myscan[p[2:]] = (Decimal(A), Decimal(T), Decimal(C))
print("\nper continuation, files whose atime moved after the snapshot (first read after re-arm), from my scan:")
allfirst = []
for name in QUEUE:
    moved = []
    for p, (a, m, c, f) in snap.items():
        if not owned(p, name):
            continue
        if p not in myscan:
            moved.append((p, "absent in my scan")); continue
        A, T, C = myscan[p]
        a_snap = Decimal(a) / Decimal(10**9)
        if A != a_snap:
            moved.append((p, str(A)))
            allfirst.append((A, p))
    unread_after = sum(1 for p in snap if owned(p, name)) - len(moved)
    print(f"  {name}: {len(moved)} read after ALL_DECIDED; {unread_after} still unread at my scan")
    for p, A in sorted(moved, key=lambda x: x[1]):
        print(f"      {A}  {p}")
