"""E5: causal transients on the cart-table model alone (no IK, the uncorrected ZMP target world): the preview
controller warms up on the continuing walk and sees the commanded plan only from the command frame; how far must the
cart-table ZMP leave the plan's ZMP target?  Stops and 90 degree turns from a 0.40 m/s walk."""
import math, sys
import numpy as np
from designlab import gait
from designlab.stage1 import refs, transitions as tr
from designlab.stage2 import look
import proto_moves as pm
import proto_causal as pc
from proto_moves import const, _ramp

body = sys.argv[1]
what = sys.argv[2]
gait.set_accuracy("r2")
src = refs.source(body)
M, q0, H = src["M"], src["q_home"], src["H"]
spec = look.speed_spec(H)
a = pm.gait_from_spec("v0.4", spec, H)
stand = tr.Gait("stand")
geo = tr.Geometry.of(M, q0, a.crouch)
d, n = a.duty, 160
pad = M.pad_polygon("left") - geo.feet0["left"][:2]
print(f"{body}: T {a.period:.4f} duty {d:.4f}; pad polygon x [{pad[:,0].min():.3f}, {pad[:,0].max():.3f}] y [{pad[:,1].min():.3f}, {pad[:,1].max():.3f}] about the foot origin, centroid {geo.pad_c['left']}")
fn_pre = dict(T=const(a.period), v=const(a.speed), w=const(0.0))
fn_walk = dict(T=const(a.period), duty=const(d), lift=const(a.lift), v=const(a.speed), w=const(0.0), amp=const(1.0),
               head_amp=const(a.head_amp), tail_amp=const(a.tail_amp), crouch=const(a.crouch))


def cart(pl, wl, p_cmd):
    """Causal preview CoM of plan pl (computed from p_cmd - 2), warm-up on the walk wl; cart-table ZMP error."""
    k_cmd = pl.out.start
    # the walk's own periodic preview state: run the walk two strides first and take its state as x0
    w0 = pm.gplan("walk", "walk", a, a, geo, fn_walk, {}, (float(pl.phase[0]), float(pl.phase[0]) + 1), pre=3.0)
    xw = np.zeros((3, 2)); xw[0] = geo.com0[:2] + (w0.path[0] - geo.origin)
    xw[1] = [a.speed, 0.0]
    cw = tr.com_path(w0, w0.zmp, xw, geo.omega)
    k = w0.out.start
    x0 = np.zeros((3, 2))
    x0[0] = cw[k]
    x0[1] = (cw[k + 1] - cw[k - 1]) / (w0.t[k + 1] - w0.t[k - 1])
    x0[2] = (cw[k + 1] - 2 * cw[k] + cw[k - 1]) / ((w0.t[k + 1] - w0.t[k]) ** 2)
    x0[0] += pl.path[0] - w0.path[k]
    com = pc.com_path_causal(pl, pl.zmp, wl.t, wl.zmp, x0, geo.omega, k_cmd)
    t = pl.t
    acc = np.gradient(np.gradient(com, t, axis=0), t, axis=0)
    zc = com - acc / geo.omega**2
    return com, zc


def report(label, pl, wl, p_cmd, extra=""):
    com, zc = cart(pl, wl, p_cmd)
    o = pl.out
    k_cmd = o.start
    t = pl.t - pl.t[k_cmd]
    e = zc - pl.zmp
    c, s = np.cos(pl.heading), np.sin(pl.heading)
    ef, el = c * e[:, 0] + s * e[:, 1], -s * e[:, 0] + c * e[:, 1]
    pre = slice(k_cmd - 150, k_cmd - 5)
    post = slice(k_cmd, len(t) - 5)
    kmax = k_cmd + int(np.argmax(np.linalg.norm(e[post], axis=1)))
    tu, lp, raw = pm.lp_forward_acc(pl.t[k_cmd:], com[k_cmd:], pl.heading[k_cmd:])
    print(f"  {label}: cart-table ZMP off the target: warm-up max {1e3 * np.linalg.norm(e[pre], axis=1).max():.2f} mm; after the command max "
          f"{1e3 * np.linalg.norm(e[post], axis=1).max():.1f} mm (forward {1e3 * ef[kmax]:+.1f}, lateral {1e3 * el[kmax]:+.1f}) at {t[kmax]:.3f} s, "
          f"single-support frames max {1e3 * np.linalg.norm(e[post], axis=1)[np.sum(pl.swing[post], axis=1) == 1].max():.1f} mm; "
          f"1 Hz forward acc min {lp.min():.3f} max {lp.max():.3f}{extra}")
    return com


if what == "stop":
    for label, p_cmd, p0, p1 in (("S1 command -0.5, ramp [-0.5, 1-d] (two-step, immediate)", -0.5, -0.5, 1 - d),
                                 ("S2 command -0.5, ramp [-0.25, 1-d]", -0.5, -0.25, 1 - d),
                                 ("S3 command -0.5, ramp [0, 0.4] (the one-step plan, half a stride's notice)", -0.5, 0.0, 0.4),
                                 ("S3b command -0.5, ramp [phi_c-0.2, phi_c+0.2] (the legacy plan exactly)", -0.5, tr.stop_command_phase(d), tr.stop_command_phase(d) + 0.4),
                                 ("S4 command -1.0, ramp [-0.5, 1-d] (a full stride's notice: three touchdowns)", -1.0, -0.5, 1 - d),
                                 ("S5 command 0.0, ramp [0, 0.4] (one-step, immediate)", 0.0, 0.0, 0.4),
                                 ("S6 command -0.573 (left touchdown), ramp [-0.5, 1-d]", -0.575, -0.5, 1 - d)):
        end = 1.0 - d + tr.SETTLE_STRIDES
        down = (lambda p0, p1: (lambda p: 1.0 - _ramp((np.asarray(p, float) - p0) / (p1 - p0))))(p0, p1)
        amp = (lambda p0: (lambda p: 1.0 - _ramp((np.asarray(p, float) - p0) / (end - p0))))(p0)
        fn = dict(T=const(a.period), duty=const(d), lift=const(a.lift), v=lambda p, down=down: a.speed * down(p), w=const(0.0),
                  amp=amp, head_amp=const(a.head_amp), tail_amp=const(a.tail_amp), crouch=const(a.crouch))
        cmd_rule = p0 if p0 <= -0.5 + 1e-9 else None   # rule F only when the ramp starts at the right foot's lift-off
        pl = pm.gplan("stop", "stop", a, stand, geo, fn, {}, (p_cmd, 3), lift_max=0.0, cmd=cmd_rule, fn_pre=fn_pre, pre=2.0)
        wl = pm.gplan("walk", "walk", a, a, geo, fn_walk, {}, (p_cmd, 9), pre=2.0)
        com = report(label, pl, wl, p_cmd)
        k_cmd = pl.out.start
        t = pl.t[k_cmd:] - pl.t[k_cmd]
        v_f = np.gradient(com[k_cmd:, 0], t)
        k_stop = next(k for k in range(len(t)) if np.all(np.abs(v_f[k:]) < 0.05 * a.speed))
        tds = sorted((round(s[2], 3), s[0]) for s in pl.swings if s[2] > p_cmd + 1e-9)
        print(f"      touchdowns after the command: {len(tds)} {tds}; forward speed < 5 % from {t[k_stop]:.3f} s ({t[k_stop] / a.period:.3f} strides); "
              f"last touchdown at {(tds[-1][0] - p_cmd) * a.period:.3f} s")
else:
    w = math.pi / 4 * (1 if what == "left" else -1)
    for f, r, p0 in ((0.8, 0.5, 0.0), (0.8, 0.75, 0.0), (0.8, 0.5, 0.5), (0.7, 0.5, 0.0), (0.9, 0.5, 0.0), (0.8, 0.5, -0.5)):
        Tt = f * a.period
        A2 = 0.3912  # integral of the min-jerk step squared over [0, 1] (numeric below)
        x = np.linspace(0, 1, 200001); A2 = float(np.trapezoid(_ramp(x) ** 2, x))
        L = (math.pi / (2 * abs(w)) + (Tt - a.period) * r * (1 - 2 * A2)) / Tt
        p1 = p0 + L
        sig = lambda p, p0=p0, p1=p1, r=r: _ramp((np.asarray(p, float) - p0) / r) - _ramp((np.asarray(p, float) - p1) / r)
        fn = dict(T=lambda p, sig=sig, Tt=Tt: a.period + (Tt - a.period) * sig(p), duty=const(d), lift=const(a.lift), v=const(a.speed),
                  w=lambda p, sig=sig: w * sig(p), amp=const(1.0), head_amp=const(a.head_amp), tail_amp=const(a.tail_amp), crouch=const(a.crouch))
        last = math.ceil(p1 + r) + 2
        p_cmd = p0 if p0 >= 0 else p0
        pl = pm.gplan("turn", "walk", a, a, geo, fn, {}, (p0, last), cmd=p0, fn_pre=fn_pre, pre=2.0)
        wl = pm.gplan("walk", "walk", a, a, geo, fn_walk, {}, (p0, last + 6), pre=2.0)
        k_cmd = pl.out.start
        t = pl.t[k_cmd:] - pl.t[k_cmd]
        h = np.degrees(pl.heading[k_cmd:] - pl.heading[k_cmd])
        h21, h20 = float(np.interp(2.1, t, h)), float(np.interp(2.0, t, h))
        t80 = float(np.interp(80.0, np.abs(h), t))
        R = float(np.interp(p0 + r, pl.phase, pl.t) - pl.t[k_cmd])
        report(f"turn {what} f {f:g} ramp {r:g} strides ({R:.3f} s) command phase {p0:+.1f}", pl, wl, p0,
               extra=f"; heading total {h[-1]:+.3f} deg, at 2.0 s ref {h20:+.2f}, at 2.1 s ref (bar 2.0 s) {h21:+.2f}, reaches 80 deg at {t80:.3f} s ref = {t80 / 1.05:.3f} s bar; ends phase {p1 + r:.3f}, window to {last}")
