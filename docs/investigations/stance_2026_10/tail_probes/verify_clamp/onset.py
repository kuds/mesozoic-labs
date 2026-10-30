"""Recompute the runner's rail_onset.txt from its tracks (own_tail3_<i>, tailq_<i> column 3 = tail_3), 0.25 s bins.
Read only.  python -B onset.py > onset.txt"""
import math
from pathlib import Path
import numpy as np
C = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/com_revisit/notes/clamp/run")
for arm in ("s42_unforced", "s42_held0", "s80_unforced", "s80_forced1"):
    own, deg, per_policy_own_end = [], [], {}
    for f in sorted((C / arm / "tracks").glob("*.npz")):
        z = np.load(f)
        n = len(z["seeds"])
        po = []
        for i in range(n):
            o = z[f"own_tail3_{i}"].astype(np.float64)
            q = z[f"tailq_{i}"][:, 3].astype(np.float64)
            own.append([o[k:k + 25].mean() for k in range(0, 400, 25)])
            deg.append([math.degrees(q[k:k + 25].mean()) for k in range(0, 400, 25)])
            po.append(o[500:].mean())
        per_policy_own_end[f.stem] = float(np.median(po))
    print(f"{arm} (n {len(own)})")
    print("  own tail_3 cmd " + " ".join(f"{v:+.2f}" for v in np.mean(own, axis=0)))
    print("  tail_3 deg     " + " ".join(f"{v:+.2f}" for v in np.mean(deg, axis=0)))
    v = np.array(list(per_policy_own_end.values()))
    print(f"  per-policy median own cmd 5-10 s: min {v.min():+.3f} median {np.median(v):+.3f} max {v.max():+.3f}; >0 {int((v > 0).sum())} <0 {int((v < 0).sum())}")
    print("   " + " ".join(f"{k[4:]}:{x:+.2f}" for k, x in per_policy_own_end.items()))
