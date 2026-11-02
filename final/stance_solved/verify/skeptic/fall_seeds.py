"""Read-only: which eval episode index / env fell in PHASE s60's training evaluations; late-window fall rate."""
import csv
import sys

P = "/home/user/work/pc_study/round/fetch/F2/stance7_r10LTSBpc_s60/copies/01_stance/eval_episodes.csv.copy1"
rows = list(csv.DictReader(open(P, newline="")))
falls = [r for r in rows if int(float(r["length"])) < 1000]
for r in falls:
    print("fall", r["evaluation"], r["timesteps"], "episode", r["episode"], "env", r["env"], "len", r["length"], "cause", r["cause"])
ts = sorted({int(float(r["timesteps"])) for r in rows})
print("n evals", len(ts), "first", ts[0], "last", ts[-1], "spacing", ts[1] - ts[0])
for lo in (0, 4194304, 6291456, 8388608):
    sub = [r for r in rows if int(float(r["timesteps"])) >= lo]
    f = sum(1 for r in sub if int(float(r["length"])) < 1000)
    print(f"from {lo}: falls {f} of {len(sub)} episodes")
eps = sorted({r["episode"] for r in rows})
print("distinct episode indices", eps[:20])
sys.exit(0)
