"""Verifier: tail_3 normaliser statistics in every checkpoint pickle (r10 F2 copies + ECS), read with plain pickle.
For an observation-only sweep of obs[6], a swept point is clipped iff |(theta - mean) / sqrt(var + eps)| >= clip_obs,
independent of the state.  Reports z at 1 deg and the angle at which +clip is reached.  No simulation.

    python -B vew_clip.py > vew_clip.txt
"""
from __future__ import annotations

import glob
import hashlib
import math
import pickle
import re
from pathlib import Path

import numpy as np

SCR = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad")
F2 = Path("/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6/fetch/F2")
ECS = SCR / "r10round/ecs/runs"

rows = []
for run in ("s42", "s60", "s80"):
    m = F2 / f"stance6_r10LTSB_{run}/copies/01_stance/models"
    for p in glob.glob(str(m / "stage1_vecnormalize_*_steps.pkl.copy1")):
        rows.append((run, int(re.search(r"_(\d+)_steps", p).group(1)), Path(p)))
    rows.append((run, 12001280, m / "stage1_final_vecnorm.pkl.copy1"))
for d in sorted(ECS.iterdir()):
    for p in glob.glob(str(d / "01_stance/models/stage1_vecnormalize_*_steps.pkl")):
        rows.append((d.name, int(re.search(r"_(\d+)_steps", p).group(1)), Path(p)))
rows.sort()
seen = {}
out = []
for run, steps, p in rows:
    h = hashlib.sha1(p.read_bytes()).hexdigest()
    with open(p, "rb") as fh:
        vn = pickle.load(fh)
    mu = float(vn.obs_rms.mean[6])
    sd = math.sqrt(float(vn.obs_rms.var[6]) + float(vn.epsilon))
    clip = float(vn.clip_obs)
    z1 = (math.radians(1.0) - mu) / sd
    z6 = (math.radians(6.0) - mu) / sd
    deg_clip = math.degrees(mu + clip * sd)
    out.append((run, steps, h, math.degrees(mu), math.degrees(sd), z1, z6, deg_clip, h in seen))
    seen.setdefault(h, (run, steps))
uniq = [r for r in out if not r[8]]
print(f"pickles {len(out)}, unique by sha1 {len(uniq)}")
print(f"z at 1 deg: max {max(r[5] for r in uniq):+.3f} ({max(uniq, key=lambda r: r[5])[:2]}); z at 0 deg min "
      f"{min(-r[3] / r[4] for r in uniq):+.3f}")
n6 = sum(r[6] >= 10 for r in uniq)
print(f"0-6 deg sweep reaches the clip (z >= 10 at 6 deg) at {n6} of {len(uniq)} unique pickles")
print("run          steps      t3_mean_deg  t3_sd_deg  z@1deg  z@6deg  deg_at_clip")
for r in out:
    if r[0] in ("s42", "s60", "s80") or r[1] in (5767168, 4718592):
        print(f"{r[0]:12s} {r[1]:9d}  {r[3]:+9.3f}  {r[4]:8.3f}  {r[5]:+6.2f}  {r[6]:+6.2f}  {r[7]:7.2f}{'  dup' if r[8] else ''}")
