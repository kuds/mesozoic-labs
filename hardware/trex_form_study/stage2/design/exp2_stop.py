"""E2: one-step (legacy) stop against the two-step stop, from a set's 0.40 m/s walk.  First the preview CoM only (no
IK: com_path on the uncorrected ZMP target), then optionally the full solve of the two-step stop."""
import sys, time
import numpy as np
from dataclasses import replace
from designlab import gait
from designlab.stage1 import refs, transitions as tr
from designlab.stage2 import look
import proto_moves as pm
from proto_moves import const, _ramp

body, which, full = sys.argv[1], sys.argv[2], len(sys.argv) > 3 and sys.argv[3] == "full"
gait.set_accuracy("r2")
src = refs.source(body)
M, q0, H = src["M"], src["q_home"], src["H"]
if which == "V0":
    spec = look.speed_spec(H)
else:  # F25F9cl at 0.40 m/s on 75 cm (the PR's _tuned_f9_spec with duty 0.59, lift 0.095)
    v = 0.40
    fr = v * v / (gait.G * H)
    sr = round(min(0.94, max(0.62, 0.863 * ((max(fr, 0.025) / 0.05) ** 0.20))), 4)
    spec = gait.GaitSpec(froude=fr, duty=0.59, lift_frac=0.095, head_amp=0.05, tail_amp=-0.125, sway_gain=1.0,
                         stride_rel=sr, crouch=0.05, zmp_iters=6, zmp_gain=0.5, n=160)
a = pm.gait_from_spec("v0.4", spec, H)
stand = tr.Gait("stand")
print(f"{body} {which}: T {a.period:.4f} duty {a.duty:.4f} speed {a.speed:.4f} lift {a.lift:.4f} crouch {a.crouch:.4f}")
ref_a = pm.periodic_ref(src, f"{which}_v0.4", spec)
geo = tr.Geometry.of(M, q0, a.crouch)
d = a.duty


def stop2_fn(p_cmd=-0.5, p_end=None):
    p_end = 1.0 - d if p_end is None else p_end
    end = 1.0 - d + tr.SETTLE_STRIDES
    down = lambda p: 1.0 - _ramp((np.asarray(p, float) - p_cmd) / (p_end - p_cmd))
    amp = lambda p: 1.0 - _ramp((np.asarray(p, float) - p_cmd) / (end - p_cmd))
    fn = dict(T=const(a.period), duty=const(d), lift=const(a.lift), v=lambda p: a.speed * down(p), w=const(0.0), amp=amp,
              head_amp=const(a.head_amp), tail_amp=const(a.tail_amp), crouch=const(a.crouch))
    return fn, dict(path=[p_cmd, p_end], amplitude=[p_cmd, end], settle=[1.0 - d, end])


fn_pre = dict(T=const(a.period), v=const(a.speed), w=const(0.0))
fn_walk = dict(T=const(a.period), duty=const(d), lift=const(a.lift), v=const(a.speed), w=const(0.0), amp=const(1.0),
               head_amp=const(a.head_amp), tail_amp=const(a.tail_amp), crouch=const(a.crouch))
plans = {}
plans["one-step (legacy)"] = (pm.legacy("stop", a, stand, geo), tr.stop_command_phase(d))
fn, ramps = stop2_fn()
plans["two-step, ramp [-0.5, 1-d], rule F"] = (pm.gplan("stop2", "stop", a, stand, geo, fn, ramps, (-2, 3), lift_max=0.0, cmd=-0.5, fn_pre=fn_pre), -0.5)
plans["two-step, ramp [-0.5, 1-d], no rule F"] = (pm.gplan("stop2", "stop", a, stand, geo, fn, ramps, (-2, 3), lift_max=0.0), -0.5)
fn, ramps = stop2_fn(p_end=0.2)
plans["two-step, ramp [-0.5, 0.2], rule F"] = (pm.gplan("stop2", "stop", a, stand, geo, fn, ramps, (-2, 3), lift_max=0.0, cmd=-0.5, fn_pre=fn_pre), -0.5)
walk = pm.gplan("walk", "walk", a, a, geo, fn_walk, {}, (-2, 3))
x0 = tr.initial_state(walk, ref_a, geo)
com_walk = tr.com_path(walk, walk.zmp, x0, geo.omega)
for label, (pl, p_cmd) in plans.items():
    com = tr.com_path(pl, pl.zmp, tr.initial_state(pl, ref_a, geo), geo.omega)
    o = pl.out
    t, ph = pl.t, pl.phase
    t_cmd = float(np.interp(p_cmd, ph, t))
    tu, lp, raw = pm.lp_forward_acc(t, com, pl.heading)
    after = tu >= t_cmd - 0.3
    dev = np.linalg.norm(com - com_walk, axis=1)
    k_dep = int(np.argmax(dev > 1e-3))
    vpath = np.gradient(np.linalg.norm(pl.path - pl.path[0], axis=1), t)
    apath = np.gradient(vpath, t)
    st = tr.stop_time_arrays(ph[o], t[o], com[o], pl.heading[-1], pl.stand_point["final"], a.speed, a.period, d)
    v_f = np.gradient(com[:, 0], t)
    k_stop = next(k for k in range(len(t)) if t[k] >= t_cmd and np.all(np.abs(v_f[k:]) < 0.05 * a.speed))
    tds = sorted((s[2], s[0]) for s in pl.swings if s[2] > p_cmd + 1e-9)
    steps = [(s[0], round(s[1], 3), round(s[2], 3), round(float(np.linalg.norm(s[5][0] - s[4][0])), 4)) for s in pl.swings if s[1] >= -1.0]
    print(f"\n{label}: command phase {p_cmd:+.4f} (t {t_cmd:.3f} s)")
    print(f"  touchdowns after the command: {[(round(p, 3), s) for p, s in tds]}")
    print(f"  last steps (side, lift-off, touchdown, length m): {steps}")
    print(f"  path decel peak {np.min(apath):.3f} m/s^2; preview-CoM forward acc: raw min {raw[after].min():.3f}, 1 Hz low-passed min {lp[after].min():.3f} max {lp[after].max():.3f} m/s^2")
    print(f"  forward speed < 5 % from {(t[k_stop] - t_cmd) / a.period:.3f} strides ({t[k_stop] - t_cmd:.3f} s) after the command; stop_time_arrays(legacy command): {st['forward_stopped_strides']:.3f}, at rest {st['at_rest_strides']}")
    print(f"  CoM departs from the continuing walk (> 1 mm) at phase {ph[k_dep]:+.3f}, {t[k_dep] - t_cmd:+.3f} s from the command")
if full:
    pl, p_cmd = plans["two-step, ramp [-0.5, 1-d], rule F"]
    g = pm.solve(src, pl, geo, ref_a, a.zmp_iters, a.zmp_gain)
    o = pl.out
    n = 160
    stand_f = pl.stand_point["final"]
    dc = 1e3 * np.linalg.norm(g["com"][-n:, :2] - stand_f, axis=1)
    dz = 1e3 * np.linalg.norm(g["zmp_mb"][-n:] - stand_f, axis=1)
    tu, lp, raw = pm.lp_forward_acc(pl.t[o], g["com"][:, :2], pl.heading[o])
    print(f"\nFULL two-step stop: {g['seconds']:.0f} s, ik_ok {g['ik_ok']} ({g['ik_fail']} frames), ZMP history mm {[round(1e3 * x, 3) for x in g['zmp_error_history_m']]}")
    print(f"  join to the stand (last stride): CoM max {dc.max():.4f} mm, ZMP max {np.nanmax(dz):.4f} mm; model-CoM forward acc 1 Hz: min {lp.min():.3f} max {lp.max():.3f} m/s^2")
    print(f"  hip-yaw use {pm.hip_yaw_use(M, g['q'])}")
