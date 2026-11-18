"""E8: a direct change from a set's nominal-speed walk to its 0.40 m/s walk with Stage 1's own construction (one
stride of ramp, window (-2, 4)): does it pass G3's joins?"""
import sys
import numpy as np
from designlab import gait, sizing
from designlab.stage1 import refs, transitions as tr
from designlab.stage2 import look
import proto_moves as pm

body, which = sys.argv[1], sys.argv[2]
gait.set_accuracy("r2")
src = refs.source(body)
M, q0, H = src["M"], src["q_home"], src["H"]


def f25(v, duty=None, lift=None):
    fr = v * v / (gait.G * H)
    sr = round(min(0.94, max(0.62, 0.863 * ((max(fr, 0.025) / 0.05) ** 0.20))), 4)
    return gait.GaitSpec(froude=fr, duty=(0.58 if v < 0.36 else 0.55) if duty is None else duty, lift_frac=0.081 if lift is None else lift,
                         head_amp=0.05, tail_amp=-0.125, sway_gain=1.0, stride_rel=sr, crouch=0.05, zmp_iters=6, zmp_gain=0.5, n=160)


if which == "V0":
    sa, sb = sizing.NOMINAL, look.speed_spec(H)
else:
    v_nom = float(sizing.NOMINAL.describe(H)["speed"])
    sa, sb = f25(v_nom, 0.59, 0.095), f25(0.40, 0.59, 0.095)
a, b = pm.gait_from_spec("nominal", sa, H), pm.gait_from_spec("v0.4", sb, H)
print(f"{body} {which}: a T {a.period:.4f} v {a.speed:.4f} duty {a.duty:.3f}; b T {b.period:.4f} v {b.speed:.4f} duty {b.duty:.3f}")
ra, rb = pm.periodic_ref(src, f"{which}_nominal", sa), pm.periodic_ref(src, f"{which}_v0.4", sb)
geo = tr.Geometry.of(M, q0, a.crouch)
pl = pm.legacy("nominal_to_v0.4", a, b, geo)
g = pm.solve(src, pl, geo, ra, b.zmp_iters, b.zmp_gain, verbose=False)
g.update(geo=geo, tau=np.zeros((pl.n_out, len(M.act_dofs))))
ja, jb = tr.compare(g, ra, "a", q0), tr.compare(g, rb, "b", q0)
t = pl.t[pl.out]
tu, lp, raw = pm.lp_forward_acc(t, g["com"][:, :2], pl.heading[pl.out])
print(f"  {g['seconds']:.0f} s, ik_ok {g['ik_ok']}, ZMP history mm {[round(1e3 * x, 3) for x in g['zmp_error_history_m']]}")
print(f"  join a: CoM {ja['com_max_mm']:.4f} mm ZMP {ja['zmp_max_mm']:.4f} mm; join b (G3): CoM {jb['com_max_mm']:.4f} mm ZMP {jb['zmp_max_mm']:.4f} mm, joints {jb['joint_max_rad']:.2e} rad -> pass {jb['com_ok'] and jb['zmp_ok'] and g['ik_ok']}")
print(f"  forward acc 1 Hz low-passed max {lp.max():.3f} m/s^2; duration {t[-1] - t[0]:.2f} s")
