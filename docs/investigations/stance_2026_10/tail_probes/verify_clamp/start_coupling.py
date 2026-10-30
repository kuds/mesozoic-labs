"""Rail-minus-no-rail posture response of the s42 START checkpoint (my forced +1 run minus the runner's held-0 run, per
seed) against the same response of the 12 s42 ENDS (the runner's unforced minus held-0, per seed).  Read only.
python -B start_coupling.py > start_coupling.txt"""
import json, math
from pathlib import Path
import numpy as np
C = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/com_revisit/notes/clamp")
V = Path(__file__).resolve().parent / "out"
ld = lambda p: {(r["who"], r["seed"]): r for r in map(json.loads, open(p))}
sf, sh, su = ld(V / "start_s42_forced1.jsonl"), ld(C / "start/s42_held0/episodes.jsonl"), ld(C / "start/s42_unforced/episodes.jsonl")
eu, eh = ld(C / "run/s42_unforced/episodes.jsonl"), ld(C / "run/s42_held0/episodes.jsonl")
names = sorted({w for w, s in eu})
for k, sc in (("com_fore_aft_mean_m", 1000), ("root_pitch_mean_deg", 1), ("com_height_mean_m", 1000)):
    ds = [(sf[key]["metrics"]["5.0"][k] - sh[key]["metrics"]["5.0"][k]) * sc for key in sh]
    de = {n: float(np.median([(eu[key]["metrics"]["5.0"][k] - eh[key]["metrics"]["5.0"][k]) * sc for key in eu if key[0] == n])) for n in names}
    v = np.array(list(de.values()))
    smed = float(np.median(ds))
    print(f"{k:20s} rail - no rail: START {smed:+.3f} (seeds +{sum(x > 0 for x in ds)}/-{sum(x < 0 for x in ds)}); "
          f"ENDS median {np.median(v):+.3f} [{v.min():+.3f}, {v.max():+.3f}] (+{int((v > 0).sum())}/-{int((v < 0).sum())}); "
          f"ends below the start's response {int((v < smed).sum())}/12")
    # partial: start unforced (3.1 deg) - held0
    dp = [(su[key]["metrics"]["5.0"][k] - sh[key]["metrics"]["5.0"][k]) * sc for key in sh]
    print(f"{'':20s} START own partial rail (3.1 deg) - no rail {np.median(dp):+.3f}")
