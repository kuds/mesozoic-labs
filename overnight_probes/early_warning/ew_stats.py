"""Tail_3 observation-normaliser statistics at every checkpoint (read only): mean, sd (deg), and the z of the
tail_3 angle at 0, 2, 3, 6 deg; the angle where z reaches +10 (clip_obs).  Network not loaded."""
import glob, math, pickle, sys, re
import numpy as np
from pathlib import Path
S1 = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6/fetch/F2"
E = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/ecs/runs"
rows = []
for run in ("s42", "s60", "s80"):
    for p in sorted(glob.glob(f"{S1}/stance6_r10LTSB_{run}/copies/01_stance/models/stage1_vecnormalize_*_steps.pkl.copy1")):
        rows.append((run, int(re.search(r"_(\d+)_steps", p).group(1)), p))
for p in sorted(glob.glob(f"{E}/*/01_stance/models/stage1_vecnormalize_*_steps.pkl")):
    rows.append((p.split("/")[-4], int(re.search(r"_(\d+)_steps", p).group(1)), p))
rows.sort(key=lambda r: (r[0], r[1]))
print("run         steps      mean_deg  sd_deg  z@0    z@1    z@2    z@3    z@6   deg@z=+10  deg@z=+4  clip eps")
for run, st, p in rows:
    with open(p, "rb") as fh:
        v = pickle.load(fh)
    m = float(v.obs_rms.mean[6]); sd = math.sqrt(float(v.obs_rms.var[6]) + v.epsilon)
    z = lambda d: (math.radians(d) - m) / sd
    print(f"{run:11s} {st:9d}  {math.degrees(m):+7.3f}  {math.degrees(sd):6.3f} {z(0):+6.2f} {z(1):+6.2f} {z(2):+6.2f} {z(3):+6.2f} {z(6):+6.2f}  {math.degrees(m+10*sd):7.2f}  {math.degrees(m+4*sd):7.2f}  {v.clip_obs} {v.epsilon}")
