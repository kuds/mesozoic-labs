"""E1: the generalised plan with the legacy schedule equals transitions.plan on all 18 transitions (plan arrays)."""
import sys, time
import numpy as np
from designlab import gait
from designlab.stage1 import refs, transitions as tr
import proto_moves as pm

body = sys.argv[1]
gait.set_accuracy("r2")
t0 = time.perf_counter()
src = refs.source(body)
print(f"source {body}: {time.perf_counter() - t0:.1f} s, H {src['H']:.6f}")
M, q0, H = src["M"], src["q_home"], src["H"]
bad = 0
for name, (an, bn) in tr.TRANSITIONS.items():
    a, b = tr.gait_of(body, an, H), tr.gait_of(body, bn, H)
    walk = b if a.stand else a
    geo = tr.Geometry.of(M, q0, walk.crouch)
    t1 = time.perf_counter()
    p0 = tr.plan(name, a, b, geo)
    t2 = time.perf_counter()
    p1 = pm.legacy(name, a, b, geo)
    t3 = time.perf_counter()
    same = True
    for key in ("phase", "t", "foot_pos", "foot_yaw", "swing", "swing_u", "zmp", "heading", "path", "root_z", "neck", "tail"):
        x, y = getattr(p0, key), getattr(p1, key)
        if not np.array_equal(x, y, equal_nan=True):
            same = False
            print("   DIFF", name, key, float(np.nanmax(np.abs(x - y))))
    same &= p0.out == p1.out and p0.ramps == p1.ramps and p0.first == p1.first and p0.last == p1.last
    same &= np.array_equal(p0.stand_point["initial"], p1.stand_point["initial"]) and np.array_equal(p0.stand_point["final"], p1.stand_point["final"])
    same &= len(p0.swings) == len(p1.swings) and all(s0[:4] == s1[:4] and np.array_equal(s0[4][0], s1[4][0]) and np.array_equal(s0[5][0], s1[5][0]) for s0, s1 in zip(p0.swings, p1.swings))
    bad += not same
    print(f"{name:28s} equal {same}  plan {t2 - t1:.2f} s / {t3 - t2:.2f} s  frames {len(p0.phase)}")
print("ALL EQUAL" if not bad else f"{bad} DIFFER")
