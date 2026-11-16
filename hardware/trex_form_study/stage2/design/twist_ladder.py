"""Kinematic screen (zmp_iters=0): hip-yaw range use of steady turns from the 0.40 m/s walk against yaw rate and
stride scale (same speed, shorter stride and period).  Left turns only (right turns mirror).  Scratch, not a record."""
import math, sys, json
from dataclasses import replace
import numpy as np
from designlab import gait, turning, statics
from designlab.stage1 import refs
from designlab.stage2 import look
gait.set_accuracy("r2")
out = {}
for body in sys.argv[1:]:
    src = refs.source(body); M, q0, H = src["M"], src["q_home"], src["H"]
    base = look.speed_spec(H)
    rows = []
    for f in (1.0, 0.9, 0.8, 0.7, 0.6):
        sp = replace(base, stride_rel=base.stride_rel * f, zmp_iters=0)
        T = float(sp.describe(H)["period"])
        for w in (0.35, 0.5, math.pi / 4, 1.0):
            tr = turning.generate(M, q0, H, sp, yaw_rate=w)
            m = M.m
            use = []
            for side in statics.SIDES:
                lo, hi = m.jnt_range[statics.jid(m, f"{side}_hip_yaw")]
                qv = tr["q"][:, M.leg_q[side][list(statics.LEG_JOINTS).index("hip_yaw")]]
                use.append(float(max(np.max(qv) / hi, np.min(qv) / lo)))
            ok = bool(np.all(tr["ik_ok"]))
            rows.append(dict(stride_scale=f, stride_H=round(sp.stride_rel, 4), period_s=round(T, 4), yaw_rate=round(w, 4), use=round(max(use), 3), ik_ok=ok))
            print(f"{body} x{f:.1f} stride {sp.stride_rel:.3f} H T {T:.3f} s  w {w:.3f}: use {max(use):.3f} ik_ok {ok}", flush=True)
    out[body] = rows
json.dump(out, open("/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/design/twist_ladder.json", "w"), indent=1)
