"""Reference walking cycle and inverse dynamics for sizing (not a controller).

A periodic, kinematically prescribed walk is generated for each candidate: constant forward
speed, sinusoidal lateral sway sized with the linear-inverted-pendulum relation so the zero-moment
point reaches the stance foot, minimum-jerk swing feet with a sine lift, and small head-pitch and
tail-yaw oscillations.  Joint angles come from inverse kinematics on the compiled model; torques
come from the full free-base dynamics with contact wrenches solved from the base equations
(``statics.contact_torques``).  The cycle is a *reference trajectory for screening loads, speeds
and power*; it is not evidence that any controller can walk the robot.

Speeds are set by Froude number Fr = v^2 / (g H) so sizes are compared at dynamically similar
gaits; stride length follows Alexander's (1976) relative-stride relation lambda/H = 2.3 Fr^0.3.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from . import statics

G = 9.81

#: Accuracy of the reference walk (DECISION_LOG F14).  The defaults are round one's, so round one regenerates
#: unchanged: the per-frame centre-of-mass correction stops at 2e-4 m or after 4 passes, and the stride boundary
#: is differenced with an approximate unwrap of the root x.  Differenced twice, the 2e-4 m residual moves the
#: multibody ZMP by centimetres, and the approximate unwrap distorts the velocities and accelerations of frames
#: 0, 1, n-2 and n-1.  ``set_accuracy("r2")`` (the round-two runner's workers) sets 1e-6 m, 12 passes and the
#: exact unwrap, with which the multibody-ZMP correction converges to about a millimetre (neither change is
#: enough alone; see turning.py).
ACCURACY = "r1"
EXACT_UNWRAP = False
COM_TOL = 2e-4
COM_ITERS = 4
_ACCURACY_LEVELS = {"r1": (False, 2e-4, 4), "r2": (True, 1e-6, 12)}


def set_accuracy(level: str = "r1") -> None:
    """Select the reference-walk accuracy for this process: "r1" (round one, the default) or "r2"."""
    global ACCURACY, EXACT_UNWRAP, COM_TOL, COM_ITERS
    EXACT_UNWRAP, COM_TOL, COM_ITERS = _ACCURACY_LEVELS[level]
    ACCURACY = level


@dataclass
class GaitSpec:
    froude: float = 0.012
    duty: float = 0.60  # fraction of the stride each foot is on the ground
    lift_frac: float = 0.10  # swing foot lift / hip height
    head_amp: float = 0.05  # rad, neck pitch oscillation at step frequency
    tail_amp: float = 0.12  # rad, tail yaw oscillation at stride frequency
    sway_gain: float = 1.0  # multiplies the LIPM sway amplitude
    stride_rel: float = 0.55  # stride length / hip height (animals at this Froude number: ~0.6-0.7)
    crouch: float = 0.03  # walking hip height reduction, fraction of H (keeps the knee off its singularity)
    zmp_iters: int = 6  # multibody ZMP correction passes on the CoM reference
    zmp_gain: float = 0.5
    n: int = 160  # samples per stride

    def describe(self, H: float) -> dict:
        v = np.sqrt(self.froude * G * H)
        stride = self.stride_rel * H
        return dict(speed=v, stride=stride, period=stride / v, froude=self.froude, duty=self.duty,
                    stride_rel=self.stride_rel, cadence_hz=2 * v / stride)


def _min_jerk(u):
    return u**3 * (10 - 15 * u + 6 * u * u)


def _foot_x(base_x, v, T, duty, stride, tk, u):
    sw = 1.0 - duty
    if u < sw:
        a = _min_jerk(u / sw)
        return base_x + v * (tk - u * T) - duty * stride / 2 + a * stride, True
    return base_x + v * (tk - (u - sw) * T) + duty * stride / 2, False


def generate(M: statics.Model, q_home: np.ndarray, H: float, spec: GaitSpec):
    """One periodic stride.

    Footprints follow a symmetric walk; the centre-of-mass path is the periodic solution of the
    linear inverted pendulum (omega^2 = g / z_com) driven by a zero-moment-point reference that sits at
    the stance foot's pad centroid in single support and moves linearly between feet in double
    support (solved harmonic by harmonic).  The pelvis is corrected each frame so the *model's*
    centre of mass follows that path; legs are solved by IK.
    """
    m = M.m
    info = spec.describe(H)
    v, stride, T = info["speed"], info["stride"], info["period"]
    n = spec.n
    dt = T / n
    t = np.arange(n) * dt
    M.forward(q_home)
    feet0 = {s: M.foot_pose(s)[0].copy() for s in statics.SIDES}
    pad_c = {}
    for s in statics.SIDES:
        poly = M.pad_polygon(s)
        pad_c[s] = statics._polygon_centroid(poly) - feet0[s][:2]  # pad centroid relative to foot origin
    com0 = M.com()
    root0 = q_home[:3].copy()
    sw = 1.0 - spec.duty
    lift = spec.lift_frac * H
    # --- ZMP reference (world), periodic part after removing v*t
    zmp = np.zeros((n, 2))
    phases = []
    feet_x = {s: np.zeros(n) for s in statics.SIDES}
    swing = {s: np.zeros(n, bool) for s in statics.SIDES}
    for k, tk in enumerate(t):
        ph = (tk / T) % 1.0
        sup = []
        for side, off in (("left", 0.0), ("right", 0.5)):
            u = (ph - off) % 1.0
            x, is_sw = _foot_x(feet0[side][0], v, T, spec.duty, stride, tk, u)
            feet_x[side][k], swing[side][k] = x, is_sw
            if not is_sw:
                sup.append(side)
        phases.append(tuple(sup))
        if len(sup) == 1:
            sd = sup[0]
            zmp[k] = np.array([feet_x[sd][k], feet0[sd][1]]) + pad_c[sd]
        else:
            # double support: from the foot about to lift to the foot that just landed
            if ph < 0.5:  # left landed at ph = sw, right lifts at 0.5
                frm, to, s0, s1 = "right", "left", sw, 0.5
            else:
                frm, to, s0, s1 = "left", "right", 0.5 + sw, 1.0
            a = (ph - s0) / (s1 - s0)
            pf = np.array([feet_x[frm][k], feet0[frm][1]]) + pad_c[frm]
            pt = np.array([feet_x[to][k], feet0[to][1]]) + pad_c[to]
            zmp[k] = (1 - a) * pf + a * pt
    z_c = com0[2] - spec.crouch * H
    omega = np.sqrt(G / z_c)
    trend = v * t
    kk = np.fft.rfftfreq(n, d=dt) * 2 * np.pi
    gain = 1.0 / (1.0 + (kk / omega) ** 2)
    neck = statics.jid(m, "neck_pitch")
    tail = statics.jid(m, "tail_yaw")
    w = 2 * np.pi / T
    off = com0[:2] - root0[:2]

    def solve_frames(zref):
        cx = np.fft.irfft(np.fft.rfft(zref[:, 0] - trend) * gain, n) + trend
        cy = np.fft.irfft(np.fft.rfft(zref[:, 1]) * gain * spec.sway_gain, n)
        qs = np.zeros((n, m.nq))
        q = q_home.copy()
        ok_all, fails = True, []
        for k, tk in enumerate(t):
            ph = (tk / T) % 1.0
            targets = {}
            for side, offp in (("left", 0.0), ("right", 0.5)):
                u = (ph - offp) % 1.0
                z = feet0[side][2] + (lift * np.sin(np.pi * u / sw) ** 2 if swing[side][k] else 0.0)
                targets[side] = (np.array([feet_x[side][k], feet0[side][1], z]), np.eye(3))
            q[2] = root0[2] - spec.crouch * H
            q[3:7] = q_home[3:7]
            q[m.jnt_qposadr[neck]] = spec.head_amp * np.sin(2 * w * tk)
            q[m.jnt_qposadr[tail]] = spec.tail_amp * np.sin(w * tk + np.pi / 2)
            q[0], q[1] = cx[k] - off[0], cy[k] - off[1]
            ok = False
            for _ in range(COM_ITERS):
                q, ok = statics.leg_ik(M, q, targets, iters=120)
                c = M.com()
                err = np.array([cx[k], cy[k]]) - c[:2]
                if np.linalg.norm(err) < COM_TOL:
                    break
                q[0] += err[0]
                q[1] += err[1]
            ok_all &= ok
            if not ok:
                fails.append(k)
            qs[k] = q
        return qs, cx, cy, ok_all, fails

    zref = zmp.copy()
    history = []
    for it in range(spec.zmp_iters + 1):
        qs, cx, cy, ok_all, fails = solve_frames(zref)
        qd, qdd = derivatives(M, qs, dt, stride=stride)
        zmb = np.array([statics.total_zmp(M, qs[k], qd[k], qdd[k]) for k in range(n)])
        e = zmb - zmp
        history.append(float(np.nanmax(np.linalg.norm(e, axis=1))))
        if it == spec.zmp_iters:
            break
        zref = zref - spec.zmp_gain * np.nan_to_num(e)
    amp = float(0.5 * (cy.max() - cy.min()))
    return dict(t=t, q=qs, phases=phases, dt=dt, zmp_ref=zmp, com_ref=np.column_stack([cx, cy]), zmp_mb=zmb,
                info=info | dict(sway_amp=amp, lift=lift, omega=float(omega), zmp_error_history_m=history,
                                 accuracy=ACCURACY),
                ik_ok=ok_all, ik_fail_frames=fails)


def derivatives(M: statics.Model, qs: np.ndarray, dt: float, stride: float | None = None):
    """Central-difference qvel and qacc of one periodic stride of ``generate``'s straight walk.

    The trajectory advances one stride in root x per period.  With ``EXACT_UNWRAP`` and ``stride`` given, the
    pose after the last frame is the first frame moved forward one stride and the pose before the first frame
    is the last moved back one stride, which is exact for this walk.  Otherwise (round one) the next stride's
    root x is extrapolated from the first frame-to-frame step."""
    m = M.m
    n = len(qs)
    exact = EXACT_UNWRAP and stride is not None
    qd = np.zeros((n, m.nv))
    for k in range(n):
        a, b = qs[(k - 1) % n], qs[(k + 1) % n]
        # the trajectory advances in x each stride; unwrap the periodic jump in root x
        b = b.copy()
        a = a.copy()
        if k == n - 1:
            if exact:
                b[0] = qs[0][0] + stride
            else:
                b[0] += qs[-1][0] - qs[0][0] + (qs[1][0] - qs[0][0])
        if k == 0:
            if exact:
                a[0] = qs[-1][0] - stride
            else:
                a[0] -= qs[-1][0] - qs[0][0] + (qs[1][0] - qs[0][0])
        v = np.zeros(m.nv)
        mujoco.mj_differentiatePos(m, v, 2 * dt, a, b)
        qd[k] = v
    qdd = np.zeros_like(qd)
    for k in range(n):
        qdd[k] = (qd[(k + 1) % n] - qd[(k - 1) % n]) / (2 * dt)
    return qd, qdd


def inverse_dynamics(M: statics.Model, traj: dict):
    qs, dt, phases = traj["q"], traj["dt"], traj["phases"]
    info = traj.get("info", {})
    # the exact unwrap is a translation along x: straight walks only (a turn's trajectory carries its yaw rate)
    stride = info.get("stride") if not info.get("yaw_rate") else None
    qd, qdd = derivatives(M, qs, dt, stride=stride)
    n = len(qs)
    na = len(M.act_dofs)
    tau = np.zeros((n, na))
    cop_margin = np.full((n, 2), np.nan)
    friction = np.zeros(n)
    normal = np.zeros((n, 2))
    for k in range(n):
        sup = phases[k] if phases[k] else ("left", "right")
        tk, w = statics.contact_torques(M, qs[k], qd[k], qdd[k], support=sup)
        tau[k] = tk
        for j, side in enumerate(statics.SIDES):
            if side in w:
                f = w[side][:3]
                normal[k, j] = f[2]
                friction[k] = max(friction[k], np.linalg.norm(f[:2]) / max(f[2], 1e-6))
                c = statics.cop(M, side, w[side])
                M.forward(qs[k])
                cop_margin[k, j] = statics.support_margin(c, M.pad_polygon(side))
    omega = qd[:, M.act_dofs]
    return dict(tau=tau, omega=omega, cop_margin=cop_margin, friction=friction, normal=normal, qd=qd, qdd=qdd)


def summarize(M: statics.Model, traj: dict, idr: dict) -> dict:
    names = M.names
    tau, om = idr["tau"], idr["omega"]
    out = {}
    for j, nm in enumerate(names):
        out[nm] = dict(rms=float(np.sqrt(np.mean(tau[:, j] ** 2))), peak=float(np.max(np.abs(tau[:, j]))),
                       speed_peak=float(np.max(np.abs(om[:, j]))),
                       pos_power_mean=float(np.mean(np.clip(tau[:, j] * om[:, j], 0, None))),
                       neg_power_mean=float(np.mean(np.clip(tau[:, j] * om[:, j], None, 0))))
    margins = idr["cop_margin"]
    single = [k for k, p in enumerate(traj["phases"]) if len(p) == 1]
    min_margin = float(np.nanmin(margins)) if np.isfinite(margins).any() else float("nan")
    min_single = float(np.nanmin(margins[single])) if single else float("nan")
    return dict(joints=out, min_cop_margin=min_margin, min_cop_margin_single=min_single,
                max_friction_ratio=float(np.max(idr["friction"])), gait=traj["info"], ik_ok=bool(traj["ik_ok"]),
                ik_fail_frames=len(traj.get("ik_fail_frames", [])))
