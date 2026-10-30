"""Analyse vp_roll.jsonl, vp_illus.jsonl, vp_plant.jsonl (no simulation).  -> vp_analyze.txt"""
import json
from collections import defaultdict

import numpy as np

OUT = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/com_revisit/notes/verify_posture"
base = {(r["who"], r["seed"]): r for r in map(json.loads, open(f"{OUT}/vp_roll.jsonl"))}
L = []
P = L.append
P("CLOSED-LOOP PITCH ILLUSION, paired with each policy's own baseline on the same seed (seeds 5001-5010)")
P("  arm: obs gravity, accelerometer and prey direction rotated as if the pelvis were pitched D more nose-up")
ill = defaultdict(list)
for r in map(json.loads, open(f"{OUT}/vp_illus.jsonl")):
    ill[(r["who"], r["arm"])].append(r)
for (who, arm), rs in ill.items():
    d_cmd = [r["cmd14"] - base[(who, r["seed"])]["cmd14"] for r in rs]
    d_q3 = [r["q3"] - base[(who, r["seed"])]["q3"] for r in rs]
    d_p = [r["pitch"] - base[(who, r["seed"])]["pitch"] for r in rs]
    rb = sum(base[(who, r["seed"])]["railed"] for r in rs)
    ri = sum(r["railed"] for r in rs)
    full = sum(r["steps"] == 1000 for r in rs)
    P(f"  {who} {arm}: n {len(rs)} (full {full}); own cmd14 window mean change median {np.median(d_cmd):+.3f} "
      f"(+{sum(x > 0 for x in d_cmd)}/-{sum(x < 0 for x in d_cmd)}); railed {rb}/{len(rs)} -> {ri}/{len(rs)}; "
      f"tail_3 {np.median(d_q3):+.2f} deg; actual pelvis pitch {np.median(d_p):+.3f} deg "
      f"(perceived change {float(arm[8:]) + np.median(d_p):+.3f})")
P("")
P("PLANT: tail_3 angle (window mean steps 501-1000) against the tail_3 command a; least squares over 5 values x 5 seeds")
pl = defaultdict(list)
for r in map(json.loads, open(f"{OUT}/vp_plant.jsonl")):
    pl[(r["who"], r["arm"])].append(r)
for (who, arm), rs in pl.items():
    if arm == "replay_unchanged":
        for r in rs:
            b = base[(who, r["seed"])]
            P(f"  {who} unchanged replay seed {r['seed']}: return {r['ret']:.6f} vs base {b['ret']:.6f} (diff {r['ret']-b['ret']:+.2e})")
        continue
    a = np.array([r["a"] for r in rs])
    q = np.array([r["q3"] for r in rs])
    p = np.array([r["pitch"] for r in rs])
    ret = np.array([r["ret"] for r in rs])
    X = np.column_stack([np.ones(len(a)), a])
    c, *_ = np.linalg.lstsq(X, q, rcond=None)
    resid = q - X @ c
    full = sum(r["steps"] == 1000 for r in rs)
    pa = {av: np.mean(p[a == av]) for av in sorted(set(a))}
    ra = {av: np.mean(ret[a == av]) for av in sorted(set(a))}
    P(f"  {who:11s} {arm:12s}: q3 = {c[0]:+.3f} + {c[1]:.3f} a deg (max |resid| {np.abs(resid).max():.3f}; n {len(a)}, full {full});"
      f" pelvis pitch at a=-1/0/+1: {pa[-1.0]:+.3f} / {pa[0.0]:+.3f} / {pa[1.0]:+.3f};"
      f" return/step a=+1 minus a=0: {(ra[1.0]-ra[0.0])/1000:+.4f}")
txt = "\n".join(L)
open(f"{OUT}/vp_analyze.txt", "w").write(txt + "\n")
print(txt)
