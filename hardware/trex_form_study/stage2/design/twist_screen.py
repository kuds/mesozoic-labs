"""Kinematic screen (no ZMP passes, no dynamics): thigh-twist (hip yaw) range use of a steady pi/4 rad/s turn at
0.40 m/s on 75-M-H-10 and 60-N-T-10, as a function of the stride period ("quicker steps").  Scratch, not a record."""
import math, sys
from dataclasses import replace
import numpy as np
from designlab import gait, turning
from designlab.stage1 import refs
from designlab.stage2 import look

gait.set_accuracy("r2")
def use(M, q0, H, spec, w):
    tr = turning.generate(M, q0, H, replace(spec, zmp_iters=0), yaw_rate=w)
    m = M.m
    out = {}
    from designlab import statics
    for side in statics.SIDES:
        j = statics.jid(m, f"{side}_hip_yaw")
        lo, hi = m.jnt_range[j]
        qi = M.leg_q[side][list(statics.LEG_JOINTS).index("hip_yaw")]
        qv = tr["q"][:, qi]
        out[side] = float(max(np.max(qv) / hi, np.min(qv) / lo))
    ok = bool(np.all(tr["ik_ok"])) if "ik_ok" in tr else None
    nfail = int(np.sum(~np.asarray(tr["ik_ok"]))) if "ik_ok" in tr and np.ndim(tr["ik_ok"]) else (0 if ok else -1)
    return out, ok, nfail

for body in sys.argv[1:]:
    src = refs.source(body); M, q0, H = src["M"], src["q_home"], src["H"]
    base = look.speed_spec(H)  # V0's 0.40 m/s walk
    info = base.describe(H)
    print(f"{body}: H {H:.4f} V0@0.40: T {info['period']:.4f} duty {base.duty:.4f} stride_rel {base.stride_rel:.4f}")
    v = info["speed"]
    for label, sr, duty in [("V0", base.stride_rel, base.duty)] + [(f"sr{f:.2f}", base.stride_rel * f, base.duty) for f in (0.9, 0.8, 0.7, 0.6)] + [("F25F9cl", 0.7918 if body.startswith("75") else None, 0.59)]:
        if sr is None:
            continue
        sp = replace(base, stride_rel=sr, duty=duty)
        T = sp.describe(H)["period"]
        for w in (math.pi / 4, -math.pi / 4):
            try:
                u, ok, nf = use(M, q0, H, sp, w)
                print(f"  {label:8s} stride {sr:.3f} H  T {T:.3f} s  duty {duty:.3f}  w {w:+.3f}: L {u['left']:.3f} R {u['right']:.3f}  ik_ok {ok} fail {nf}")
            except Exception as e:
                print(f"  {label:8s} T {T:.3f} w {w:+.3f}: ERROR {type(e).__name__}: {e}")
