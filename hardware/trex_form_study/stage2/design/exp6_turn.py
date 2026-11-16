"""E6: the causal 90 degree quick turn from a 0.40 m/s walk (proto_causal): kinematic passes (no ZMP correction) over
stride factors and command phases for the thigh twist, or one full solve.
usage: exp6_turn.py BODY SET left|right f:r:p0[,f:r:p0...] [full] [rate]"""
import math, sys
import numpy as np
from designlab import gait, statics
from designlab.stage1 import refs, transitions as tr
from designlab.stage2 import look
import proto_moves as pm
import proto_causal as pc
from proto_moves import const, _ramp

body, which, direction = sys.argv[1], sys.argv[2], sys.argv[3]
cases = [tuple(float(x) for x in c.split(":")) for c in sys.argv[4].split(",")]
full = "full" in sys.argv[5:]
rate = next((float(x) for x in sys.argv[5:] if x.replace(".", "").isdigit()), math.pi / 4)
gait.set_accuracy("r2")
src = refs.source(body)
M, q0, H = src["M"], src["q_home"], src["H"]
if which == "V0":
    spec = look.speed_spec(H)
else:
    v = 0.40
    fr = v * v / (gait.G * H)
    sr = round(min(0.94, max(0.62, 0.863 * ((max(fr, 0.025) / 0.05) ** 0.20))), 4)
    spec = gait.GaitSpec(froude=fr, duty=0.59, lift_frac=0.095, head_amp=0.05, tail_amp=-0.125, sway_gain=1.0,
                         stride_rel=sr, crouch=0.05, zmp_iters=6, zmp_gain=0.5, n=160)
a = pm.gait_from_spec("v0.4", spec, H)
ref_a = pm.periodic_ref(src, f"{which}_v0.4", spec)
geo = tr.Geometry.of(M, q0, a.crouch)
d, n = a.duty, 160
w = rate * (1 if direction == "left" else -1)
fn_pre = dict(T=const(a.period), v=const(a.speed), w=const(0.0))
fn_walk = dict(T=const(a.period), duty=const(d), lift=const(a.lift), v=const(a.speed), w=const(0.0), amp=const(1.0),
               head_amp=const(a.head_amp), tail_amp=const(a.tail_amp), crouch=const(a.crouch))
delta = pm.walk_correction(ref_a, geo.omega)
x = np.linspace(0, 1, 200001)
A2 = float(np.trapezoid(_ramp(x) ** 2, x))
print(f"{body} {which} {direction} rate {rate:.4f}: walk T {a.period:.4f} duty {d:.4f}; A2 {A2:.6f}")
for f, r, p0 in cases:
    Tt = f * a.period
    L = (math.pi / (2 * abs(w)) + (Tt - a.period) * r * (1 - 2 * A2)) / Tt
    p1 = p0 + L
    sig = lambda p, p0=p0, p1=p1, r=r: _ramp((np.asarray(p, float) - p0) / r) - _ramp((np.asarray(p, float) - p1) / r)
    fn = dict(T=lambda p, sig=sig, Tt=Tt: a.period + (Tt - a.period) * sig(p), duty=const(d), lift=const(a.lift),
              v=const(a.speed), w=lambda p, sig=sig: w * sig(p), amp=const(1.0), head_amp=const(a.head_amp),
              tail_amp=const(a.tail_amp), crouch=const(a.crouch))
    last = math.ceil(p1 + r) + 2
    pl = pm.gplan("turn", "walk", a, a, geo, fn, {}, (p0, last), cmd=p0, fn_pre=fn_pre, pre=1.0)
    wl = pm.gplan("walk", "walk", a, a, geo, fn_walk, {}, (p0, last + 6), pre=1.0)
    o = pl.out
    g = pc.solve_causal(src, pl, geo, ref_a, wl, delta, const(1.0), a.zmp_iters if full else 0, a.zmp_gain, p0, verbose=full)
    t = pl.t[o] - pl.t[o][0]
    h = np.degrees(pl.heading[o] - pl.heading[o][0])
    use = pm.hip_yaw_use(M, g["q"])
    # where the peak twist comes: phase of the largest |hip yaw| per side
    pk = {}
    for side in statics.SIDES:
        qi = M.leg_q[side][list(statics.LEG_JOINTS).index("hip_yaw")]
        k = int(np.argmax(np.abs(g["q"][:, qi])))
        pk[side] = (round(float(pl.phase[o][k]), 2), round(float(g["q"][k, qi]), 3))
    roll = {}
    for side in statics.SIDES:
        j = statics.jid(M.m, f"{side}_hip_roll"); lo, hi = M.m.jnt_range[j]
        qi = M.leg_q[side][list(statics.LEG_JOINTS).index("hip_roll")]
        roll[side] = round(float(min(hi - g["q"][:, qi].max(), g["q"][:, qi].min() - lo)), 3)
    steps = [(s[0][0], round(s[1], 2), round(float(np.linalg.norm(s[5][0] - s[4][0])), 3), round(math.degrees(s[5][1] - s[4][1]), 1)) for s in pl.swings if p0 - 1e-9 <= s[1] < p1 + r + 0.6]
    print(f" f {f:g} ramp {r:g} cmd phase {p0:+.1f}: frames {pl.n_out} ({g['seconds']:.0f} s) ik_ok {g['ik_ok']} ({g['ik_fail']} frames) "
          f"hip-yaw use L {use['left']:.3f} R {use['right']:.3f} (peaks {pk}); hip-roll margin rad {roll}; "
          f"heading at 2.1 s ref {float(np.interp(2.1, t, np.abs(h))):.2f} deg, total {h[-1]:+.4f}; turn ends phase {p1 + r:.3f}; "
          f"ZMP hist mm {[round(1e3 * z, 2) for z in g['zmp_error_history_m']]}")
    print(f"    steps (side, lift-off, length m, yaw deg): {steps}")
    if full:
        k_cmd = o.start
        cw = pc.walk_frames(ref_a, pl.phase[:k_cmd + 1], pl.first, "com")
        kq = np.round(pl.phase[:k_cmd + 1] * n).astype(int) % n
        dq = np.abs(g["q_all"][:k_cmd + 1, 7:] - ref_a.q[kq, 7:]).max(1)
        dc = 1e3 * np.linalg.norm(g["com_all"][:k_cmd + 1, :2] - cw, axis=1)
        # join b: the last output stride against the walk carried to the final path pose
        P, th = pl.pose_fn(float(pl.last - 1))
        mot = (th, geo.origin, P)
        cb = tr._move(ref_a.com[:, :2], mot); zb = tr._move(ref_a.zmp_mb, mot)
        jc = 1e3 * np.linalg.norm(g["com"][-n:, :2] - cb, axis=1)
        jz = 1e3 * np.linalg.norm(g["zmp_mb"][-n:] - zb, axis=1)
        jq = np.abs(g["q"][-n:, 7:] - ref_a.q[:, 7:]).max()
        e = 1e3 * np.linalg.norm(np.nan_to_num(g["zmp_mb"] - pl.zmp[o]), axis=1)
        print(f"    FULL: join at the command: CoM {dc[-1]:.4f} mm joints {dq[-1]:.2e} rad; last stride against the walk on the new heading: CoM {jc.max():.4f} mm, "
              f"ZMP {np.nanmax(jz):.4f} mm, joints {jq:.2e} rad; ZMP off the target: first 0.5 stride {e[:80].max():.2f}, rest {e[80:].max():.2f} mm")
        np.savez(pm.SCR / "cache" / f"{body}_{which}_turn_{direction}_proto.npz", t=t, q=g["q"], qd=g["qd"], qdd=g["qdd"], com=g["com"],
                 zmp_mb=g["zmp_mb"], phase=pl.phase[o], swing=pl.swing[o], heading=pl.heading[o], zmp=pl.zmp[o])
