"""(b) Outcome-blind order: birth/atime/mtime/ctime (os.stat, st_birthtime via statx where available) of every file
written by the reading, compared with the runner's 'all 24 decided' line.  Reads no file content except the runner's
log line (run_waves_ecs.log) and decided.tsv.  Never opens an attempt log (so its atime is not disturbed by me)."""
import os, sys, re, subprocess, datetime as dt, glob
ECS = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/ecs"
def ts(s): return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).timestamp()
log = open(ECS + "/logs/run_waves_ecs.log").read().splitlines()
dec_line = [l for l in log if "all 24 decided" in l]
assert len(dec_line) == 1, dec_line
T_DEC = ts(re.search(r"\[run_waves_ecs (\S+)\]", dec_line[0]).group(1))
def st(p):
    out = subprocess.run(["stat", "-c", "%W %X %Y %Z", p], capture_output=True, text=True).stdout.split()
    return tuple(int(x) for x in out)
def iso(t): return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%H:%M:%S") if t > 0 else "-"
print("all-24-decided line at", dec_line[0][:36], T_DEC)
groups = {
  "score": sorted(glob.glob(ECS + "/score/*")) + sorted(glob.glob(ECS + "/score/read/*/*")),
  "traces": sorted(glob.glob(ECS + "/traces/*/*")),
  "read/trace/score logs": sorted(glob.glob(ECS + "/logs/*.read.log") + glob.glob(ECS + "/logs/*.trace.log") + [ECS + "/logs/score.log"]),
}
bad = []
for g, files in groups.items():
    earliest = min((st(f)[0] if st(f)[0] > 0 else st(f)[2]) for f in files if os.path.isfile(f))
    print(f"{g}: {len(files)} entries; earliest birth {iso(earliest)} ({'after' if earliest > T_DEC else 'BEFORE'} the decided line)")
    for f in files:
        if not os.path.isfile(f): continue
        b, a, m, c = st(f)
        if (b and b < T_DEC) or m < T_DEC:
            bad.append((f, iso(b), iso(m)))
print("files of the reading born or modified before the decided line:", bad or "none")
# per-continuation order: trace written, then read.json/m5.json, then score
logt = {}
for l in log:
    m = re.match(r"\[run_waves_ecs (\S+)\] (\S+): (end trace|read_ecs)", l)
    if m: logt[(m.group(2), m.group(3))] = ts(m.group(1))
order_bad = []
score_b = st(ECS + "/score/score.txt")[0]
for name in sorted({k[0] for k in logt}):
    tb = st(f"{ECS}/traces/{name}/m1_summary.json")
    rb = st(f"{ECS}/score/read/{name}/read.json"); mb = st(f"{ECS}/score/read/{name}/m5.json")
    if not (tb[2] <= mb[0] <= rb[0] <= score_b):
        order_bad.append((name, iso(tb[2]), iso(mb[0]), iso(rb[0]), iso(score_b)))
print("per continuation trace-mtime <= m5.json birth <= read.json birth <= score birth violated:", order_bad or "none")
last_trace = max(st(f)[2] for f in glob.glob(ECS + "/traces/*/m1_summary.json"))
first_read = min(st(f)[0] for f in glob.glob(ECS + "/score/read/*/m5.json"))
print(f"last trace m1_summary mtime {iso(last_trace)}; first m5.json birth {iso(first_read)}; score.txt birth {iso(score_b)}")
# check.json: each written at its decision, by --check, before the decided line
dec = {l.split("\t")[1]: ts(l.split("\t")[0]) for l in open(ECS + "/logs/decided.tsv").read().splitlines()}
for name, t in sorted(dec.items(), key=lambda kv: kv[1]):
    cb = st(f"{ECS}/runs/{name}/check.json"); rm = st(f"{ECS}/runs/{name}/run_meta.json")
    ok = rm[2] <= cb[0] <= t + 1
    if not ok: print("check.json timing odd:", name, iso(rm[2]), iso(cb[0]), iso(t))
print("check.json born after its run_meta's last write and at its decision: checked (odd ones printed above)")
# attempt logs: atime vs birth vs mtime (relatime ext4)
print("\nattempt logs (birth / atime / mtime / ctime), relatime ext4:")
for f in sorted(glob.glob(ECS + "/logs/*.attempt*.log")):
    b, a, m, c = st(f)
    flag = "atime==birth" if abs(a - b) <= 1 else ("atime<mtime" if a < m else ("atime>=mtime, after decided" if a > T_DEC else "atime>=mtime, BEFORE decided"))
    print(f"  {os.path.basename(f):28s} {iso(b)} {iso(a)} {iso(m)} {iso(c)}  {flag}")
# per-run eval ledgers and probes: atime
print("\nrun ledgers / probes / end pairs: atime before the decided line but after their mtime (a read after the last write, during the run)?")
for name in sorted(dec):
    for rel in ("01_stance/eval_episodes.csv", "01_stance/policy_probe.npz", "01_stance/evaluations.npz", "01_stance/diagnostics.npz"):
        p = f"{ECS}/runs/{name}/{rel}"
        b, a, m, c = st(p)
        if m < a < T_DEC:
            print("  ", name, rel, "atime", iso(a), "mtime", iso(m), "decided", iso(dec[name]))
print("done")
