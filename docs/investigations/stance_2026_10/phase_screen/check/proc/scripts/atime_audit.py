import csv, datetime as dt, glob, os, re, sys
PS = "/home/user/work/phase_screen"
L = PS + "/logs"
def t(ns): return dt.datetime.fromtimestamp(int(ns) / 1e9, dt.timezone.utc).strftime("%H:%M:%S.%f")[:-3]
def iso(s): return int(dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).timestamp() * 1e9)
NAMES = ["control_2901", "critic_phase_2901", "control_2911", "critic_phase_2911",
         "control_2921", "critic_phase_2921", "control_2931", "critic_phase_2931"]
log = open(L + "/run_waves_bs.log").read().splitlines()
ev = {}
for line in log:
    m = re.match(r"\[run_waves_bs (\S+)\] (\S+?):? (.*)", line)
    if not m: continue
    ts, name, rest = m.groups()
    if rest.startswith("exited"): ev.setdefault(name, {})["exit"] = iso(ts)
    if rest.startswith("COUNTED"): ev.setdefault(name, {})["decided"] = iso(ts)
    if name == "start": pass
starts = {}
for line in log:
    m = re.match(r"\[run_waves_bs (\S+)\] start (\S+) \((\S+)\)", line)
    if m: starts[m.group(3)] = iso(m.group(1))
alld = iso(open(L + "/ALL_DECIDED").read().strip())
print("ALL_DECIDED", t(alld))
bad = 0
print("\n== (1) I1 exit snapshots: files read after their last write (atime > mtime) and when")
for f in sorted(glob.glob(L + "/atimes_at_exit/*.tsv")):
    base = os.path.basename(f)[:-4]; name, att = base.rsplit(".", 1)
    rows = list(csv.DictReader(open(f), delimiter="\t"))
    st = starts.get(f"{name}.{att}.log"); ex = ev[name]["exit"]
    late = [r for r in rows if int(r["atime_ns"]) > int(r["mtime_ns"])]
    outside = [r for r in late if not (st - 2e9 <= int(r["atime_ns"]) <= ex + 2e9)]
    print(f"{base}: {len(rows)} files; start {t(st)} exit {t(ex)}; atime>mtime {len(late)}; of those outside [start, exit] {len(outside)}")
    for r in late:
        d_ms = (int(r["atime_ns"]) - int(r["mtime_ns"])) / 1e6
        print(f"    {r['path'].replace(PS + '/', '')}  atime {t(r['atime_ns'])}  mtime {t(r['mtime_ns'])}  (+{d_ms:.1f} ms)")
    bad += len(outside)
    # files of earlier attempts (logs, tmp) in this snapshot: atime vs mtime
print("\n== (2) ALL_DECIDED snapshot")
rows = list(csv.DictReader(open(L + "/atimes_at_ALL_DECIDED.tsv"), delimiter="\t"))
print("rows", len(rows), "rearmed_unread=1:", sum(int(r["rearmed_unread"]) for r in rows))
def owner(p):
    for n in NAMES:
        if p.startswith(f"runs/{n}/") or p.startswith(f"tmp/{n}/") or re.match(rf"logs/{n}\.attempt\d+\.log$", p): return n, "counted"
        if re.match(rf"runs/{n}_crashed\d+/", p): return n, "crashed"
    return None, "other"
by = {}
for r in rows:
    n, kind = owner(r["path"]); by.setdefault((kind, n), []).append(r)
for n in NAMES:
    rs = by.get(("counted", n), [])
    nr = [r for r in rs if r["rearmed_unread"] != "1"]
    after = [r for r in rs if int(r["atime_ns"]) > ev[n]["decided"] + 5e9]
    print(f"{n}: {len(rs)} files, re-armed and unread at ALL_DECIDED {len(rs) - len(nr)}, NOT re-armed {len(nr)}; atime later than decision+5s: {len(after)}")
    for r in nr: print("    NOT re-armed:", r["path"], t(r["atime_ns"]), t(r["mtime_ns"]))
    bad += len(nr)
print("\n== (3) crashed folders in the ALL_DECIDED snapshot (retired at 18:08:57-58 and 18:55:53)")
relaunch = {1: (iso("2026-10-31T18:08:57Z"), iso("2026-10-31T18:08:59Z")), 2: (iso("2026-10-31T18:55:53Z"), iso("2026-10-31T18:55:54Z"))}
cr = [r for r in rows if owner(r["path"])[1] == "crashed"]
print("crashed files:", len(cr))
sens = ins = 0; lat = []
for r in cr:
    k = int(re.search(r"_crashed(\d+)/", r["path"]).group(1))
    a, m = int(r["atime_ns"]), int(r["mtime_ns"])
    if a <= m: sens += 1
    else: ins += 1
    if a > relaunch[k][0] - 1e9:
        lat.append((r["path"], t(a), t(m)))
print(f"atime <= mtime (a later read would have shown): {sens}; atime > mtime (relatime hides a later read): {ins}")
print("files whose atime is at or after their retirement relaunch:")
for x in lat: print("   ", x)
ok_meta = all(p.endswith("/run_meta.json") for p, _, _ in lat)
print("only run_meta.json (the runner's allowed status/last_step read at relaunch):", ok_meta)
if not ok_meta: bad += 1
print("\n== (4) other files in the ALL_DECIDED snapshot with atime inside the screen window (15:56:58 .. 20:06:54) and atime > mtime")
w0 = iso("2026-10-31T15:56:58Z")
for r in rows:
    if owner(r["path"])[1] != "other": continue
    a, m = int(r["atime_ns"]), int(r["mtime_ns"])
    if w0 <= a <= alld and a > m:
        print("   ", r["path"], "atime", t(a), "mtime", t(m))
print("\n== (5) ALL_DECIDED vs my snapshot at 20:21 for crashed-folder files (any atime change after ALL_DECIDED?)")
mine = {}
for line in open(sys.argv[1]):
    p, A, M, C, s, mode, ty = line.rstrip("\n").split("\t")
    mine[p] = A
chg = 0
for r in cr:
    p = PS + "/" + r["path"]
    A = mine.get(p)
    ns = int(r["atime_ns"])
    a = dt.datetime.fromtimestamp(ns // 10**9, dt.timezone.utc).strftime("%Y-%m-%d+%H:%M:%S.") + f"{ns % 10**9:09d}"
    if A is None or A[:29] != a[:29]:
        chg += 1; print("   changed/missing:", r["path"], A, a)
print("crashed files whose atime changed between ALL_DECIDED and 20:21:", chg)
bad += chg
print("\nTOTAL anomalies:", bad)
