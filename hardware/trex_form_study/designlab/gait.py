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

Stage 2 look features (``STAGE2_PLAN.md`` section 4 A; ``stage2/research/research_look.md`` section 3.1)
-----------------------------------------------------------------------------------------------------
``GaitSpec`` carries the look features as fields whose defaults are the walk above (``LOOK_DEFAULTS``).  With every
one at its default ``generate`` runs exactly the code it always ran, so Stage 1's references rebuild bit for bit
(gate G2); with any of them set it runs ``_generate_look``, the same construction (footprints, LIPM centre of mass,
per-frame CoM correction, ``statics.leg_ik``, multibody-ZMP correction passes) with the features added:

* F1/F2 ``bob``, ``bob_phase``, ``bob_skew``, ``bob_drop``: the pelvis dips by ``bob`` H (peak to peak) twice per
  stride, at today's height at mid single support (mid-stance) and lowest ``bob_skew`` of the stance after each
  touchdown (None: midway between the mid-stances).  Without ``bob_drop`` the profile is (1 - cos 2 pi g) / 2 of a
  smoothly warped step phase g (C-infinity; most of a skewed dip then comes before the touchdown).  With it (F2's
  weight acceptance) the hips hold the crest through late single support, drop (min-jerk) over ``bob_drop`` seconds to
  the trough and rise (min-jerk) to the next mid-stance: a quick dip around the touchdown, then a slow rise (C2).
* F3 ``swing_time``: a swing of fixed duration; duty = clip(1 - swing_time / T, ``DUTY_RANGE``) (``describe``).
* F4 ``lift_peak``: the swing lift lift sin^2(pi g(s)) with g = s + A s^2 (1 - s)^4 a warp of the swing fraction s
  that puts the peak at s = lift_peak (0.5 is today's profile; mirrored for a later peak): the foot rises quickly and
  descends more slowly, its warp sits in the first half, so the middle half of the swing keeps at least 0.45 of the
  lift for a peak at 35-40 % (today's 0.5; the clearance screen's 0.04 H with a 0.10 H lift); lift-off and touchdown
  keep zero vertical speed and today's acceleration.
* F5 ``toe_off``, ``toe_off_start``, ``toe_off_margin``, ``toe_off_rate``, ``toe_off_accel``, ``toe_reach``: the foot
  pitches heel-up and rolls without slipping on its real convex sole (``roll_offset``): about the front bottom edge of
  its frontmost toe pad, then over a short transition arc on to the middle claw's tip, which reaches the floor first
  as the foot pitches (``foot_geometry``'s ``claw``: the toes and claws reach ahead of the pads and carry no contact in
  the plant).  The arc keeps the roll smooth (a rigid foot rolling from the edge on to the claw at a finite rate would
  strike it) and the real foot within a fraction of a millimetre above the floor; no part goes below it.  The roll
  starts at the later of ``toe_off_start`` of the stance and the moment the ZMP reference enters the rolling support
  (the toe edge and the other foot's pads) by ``toe_off_margin``, every rolling frame's actual contact checked; the
  pitch rises (min-jerk) to its peak at the lift-off and falls back to level by mid-swing (``toe_off_bump``).  The
  peak is ``toe_off``, or less where the roll's time on the floor is short: at most ``toe_off_rate`` (rad/s, mean) and
  ``toe_off_accel`` (rad/s^2, peak) allow (``roll_amplitude``).  After mid-swing the toes rise by ``toe_reach`` about
  the heel pad's rear edge and the foot lands flat.  The foot targets carry the pitch and ``statics.leg_ik`` solves
  them as 6-D poses.
* F6 ``roll_amp``, ``roll_phase``: the trunk rolls about its own long axis at the stride frequency, by ``roll_amp``
  toward the stance leg at its mid-stance (negative: the swing side drops, as human pelvic list).
* F7 ``zmp_inset_y``: the single-support ZMP reference moves this far toward the body's midline.
* F9 ``tail_phase``: tail = tail_amp sin(2 pi phase + tail_phase); today's is pi / 2.
* F10 ``head_mode`` "stabilise", ``head_gain``, ``head_settle``, ``settle_lag``: no nod; the neck pitch cancels
  ``head_gain`` of the trunk's pitch away from home and dips by ``head_settle`` ``settle_lag`` after each touchdown.
* F14 ``strides``, ``phase_jitter``, ``asym``, ``gaze_var``, ``tail_var``, ``var_band``, ``noise_seed``: a window of
  ``strides`` strides, periodic as a whole, whose steps last T / 2 (1 + e_j) with e_j seeded, |e_j| <= phase_jitter +
  asym / 2 and sum zero (``_Warp``), plus slow seeded variations of the head's pitch (``gaze_var`` rad) and of the
  tail's amplitude (``tail_var`` of it) made of the window's harmonics inside ``var_band`` (Hz).

F8 (trunk pitch) is not a field: none is imposed.  F11-F13 are not fields either: F13 is ``stride_rel``; F11 and F12
(head-first turns, anticipation) belong to the turn and transition generators.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

from . import statics

G = 9.81
#: F3: the duty range a fixed swing time is clipped to (research_look F3: tinamous 0.55-0.70; spec band to 0.72).
DUTY_RANGE = (0.55, 0.72)

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
    # --- Stage 2 look features (module docstring); the defaults are the walk above (LOOK_DEFAULTS)
    bob: float = 0.0  # F1: pelvis dip, fraction of H, peak to peak
    bob_phase: float = 0.0  # F1: shift of the bob pattern (stride fraction); 0 = highest at mid single support
    bob_skew: float | None = None  # F2: trough time after each touchdown, fraction of stance (None: midway)
    swing_time: float | None = None  # F3: swing duration (s); None keeps the fixed duty
    lift_peak: float = 0.5  # F4: fraction of the swing at peak lift
    toe_off: float = 0.0  # F5: peak heel-up foot pitch about the toe pads' front edge (rad)
    toe_off_start: float = 0.825  # F5: earliest start of the roll, fraction of stance
    toe_off_margin: float = 0.005  # F5: the ZMP reference's margin inside the rolling support before a roll (m)
    toe_reach: float = 0.0  # F5: toes-up pitch before landing (rad)
    roll_amp: float = 0.0  # F6: trunk roll toward the stance leg at its mid-stance (rad)
    roll_phase: float = 0.0  # F6: shift of the roll (stride fraction)
    zmp_inset_y: float = 0.0  # F7: single-support ZMP moved toward the midline (m)
    tail_phase: float = math.pi / 2  # F9: tail = tail_amp sin(2 pi phase + tail_phase)
    head_mode: str = "nod"  # F10: "nod" (head_amp sin 2wt) or "stabilise"
    head_gain: float = 0.9  # F10: fraction of the trunk's pitch the neck cancels ("stabilise")
    head_settle: float = 0.0  # F10: neck dip after each touchdown (rad)
    settle_lag: float = 0.075  # F10: time of the dip's deepest point after the touchdown (s)
    strides: int = 1  # F14: strides in the (periodic) window
    phase_jitter: float = 0.0  # F14: largest step-time deviation (fraction)
    asym: float = 0.0  # F14: left-right step-time asymmetry (fraction)
    gaze_var: float = 0.0  # F14: slow head-pitch variation (rad, largest)
    tail_var: float = 0.0  # F14: slow tail-amplitude variation (fraction of tail_amp, largest)
    var_band: tuple = (0.05, 0.2)  # F14: band of the slow variations (Hz)
    noise_seed: int = 0  # F14
    bob_drop: float | None = None  # F2: the dip into the trough lasts this long (s): hold, drop, rise; None: smooth
    toe_off_rate: float | None = None  # F5: largest mean pitch rate of the roll on the floor (rad/s); None: no cap
    toe_off_accel: float | None = None  # F5: largest pitch acceleration of the roll (rad/s^2); None: no cap

    def describe(self, H: float) -> dict:
        v = np.sqrt(self.froude * G * H)
        stride = self.stride_rel * H
        duty = self.duty
        if self.swing_time is not None:  # F3: the duty follows from the swing time
            duty = float(np.clip(1.0 - self.swing_time / (stride / v), *DUTY_RANGE))
        return dict(speed=v, stride=stride, period=stride / v, froude=self.froude, duty=duty,
                    stride_rel=self.stride_rel, cadence_hz=2 * v / stride)


#: The look fields and their defaults (today's walk): with all of them at these values ``generate`` is today's.
LOOK_FIELDS = ("bob", "bob_phase", "bob_skew", "swing_time", "lift_peak", "toe_off", "toe_off_start", "toe_off_margin",
               "toe_reach", "roll_amp", "roll_phase", "zmp_inset_y", "tail_phase", "head_mode", "head_gain",
               "head_settle", "settle_lag", "strides", "phase_jitter", "asym", "gaze_var", "tail_var", "var_band",
               "noise_seed", "bob_drop", "toe_off_rate", "toe_off_accel")
LOOK_DEFAULTS = {f.name: f.default for f in fields(GaitSpec) if f.name in LOOK_FIELDS}


def _same_value(a, default) -> bool:
    """A field's value equals its default; a sequence default (``var_band``) also equals a list or array with the same
    items, so a default spec read back from JSON (where tuples become lists) stays today's walk."""
    if isinstance(default, tuple):
        return isinstance(a, (tuple, list, np.ndarray)) and len(a) == len(default) and tuple(a) == default
    return a == default


def look_active(spec: GaitSpec) -> bool:
    """Whether any look field differs from today's walk (``LOOK_DEFAULTS``)."""
    return not all(_same_value(getattr(spec, k), v) for k, v in LOOK_DEFAULTS.items())


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

    A spec with any look field set (``look_active``) is generated by ``_generate_look`` (module docstring).
    """
    if look_active(spec):
        return _generate_look(M, q_home, H, spec)
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
    if stride is not None and info.get("strides", 1) != 1:  # a look window of several strides advances by all of them
        stride = stride * info["strides"]
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


# ================================================================================================
# Stage 2 look features (module docstring; research_look.md section 3.1)
# ================================================================================================
SIDES = statics.SIDES
#: F14: the slow variations are normalised on this many samples of the window (then bounded on the frames too).
VAR_GRID = 4096


def _skew_kappa(peak: float) -> float:
    """kappa of the warp g(s) = s + kappa (1 - cos 2 pi s) / (2 pi) of [0, 1] that maps ``peak`` to 0.5.  g(0) = 0,
    g(1) = 1 and g' = 1 at both ends; g is monotone only for |kappa| < 1 (a peak in about 0.297-0.703)."""
    if not 0.0 < peak < 1.0:
        raise ValueError(f"a peak at {peak} is outside (0, 1)")
    k = 2 * math.pi * (0.5 - peak) / (1.0 - math.cos(2 * math.pi * peak))
    if abs(k) >= 1.0:
        raise ValueError(f"a peak at {peak} needs a non-monotone warp (|kappa| = {abs(k):.3f} >= 1)")
    return k


def _skew(s, kappa: float):
    return s + kappa * (1.0 - np.cos(2 * np.pi * s)) / (2 * np.pi)


#: max over [1/3, 1] of s (1 - s)^3 (6 s - 2): F4's warp s + A s^2 (1 - s)^4 has g' >= 1 - A x this (``_lift_warp``).
_LIFT_HMAX = float(np.max((lambda s: s * (1 - s) ** 3 * (6 * s - 2))(np.linspace(1 / 3, 1.0, 200001))))


def _lift_warp(s, peak: float):
    """F4's warp of the swing fraction: g(s) = s + A s^2 (1 - s)^4 with A = (0.5 - peak) / (peak^2 (1 - peak)^4), so
    g(peak) = 0.5; a peak after mid-swing mirrors it, 1 - g(1 - s; 1 - peak).  g' = 1 at both ends and g'' = 0 at the
    far end, so lift-off and touchdown keep today's zero speed and acceleration; the warp is monotone only for a peak in
    about 0.223-0.777 (``ValueError`` outside)."""
    s = np.asarray(s, float)
    if not 0.0 < peak < 1.0:
        raise ValueError(f"a peak at {peak} is outside (0, 1)")
    if peak > 0.5:
        return 1.0 - _lift_warp(1.0 - s, 1.0 - peak)
    A = (0.5 - peak) / (peak**2 * (1.0 - peak) ** 4)
    if A * _LIFT_HMAX >= 1.0:
        raise ValueError(f"a peak at {peak} needs a non-monotone warp (A max |h| = {A * _LIFT_HMAX:.3f} >= 1)")
    return s + A * s**2 * (1.0 - s) ** 4


def lift_profile(s, lift_peak: float = 0.5):
    """F4: the swing lift as a fraction of the lift at swing fraction s: sin^2(pi s) (today's) or, with the peak moved
    to ``lift_peak``, sin^2(pi g(s)) with ``_lift_warp``'s g: a quick rise and a slower descent that keeps the middle
    half of the swing high (at s = 0.75, 0.45-0.47 of the lift for a peak at 35-40 %)."""
    if lift_peak == 0.5:
        return np.sin(np.pi * np.asarray(s, float)) ** 2
    return np.sin(np.pi * _lift_warp(s, lift_peak)) ** 2


def bob_profile(tau, trough: float = 0.5, drop: float | None = None):
    """F1/F2: the dip as a fraction of ``bob`` over one step, tau in [0, 1) from a mid-stance (0, no dip) to the next,
    deepest (1) at ``trough``.  Without ``drop``: (1 - cos 2 pi g(tau)) / 2 with ``_skew``'s warp.  With ``drop`` (the
    dip's duration as a fraction of the step; F2's weight acceptance): at the crest until ``drop`` before the trough, a
    min-jerk drop to it, then a min-jerk rise to the next crest (C2: zero rate and curvature at the crest, at the drop's
    start and at the trough)."""
    tau = np.asarray(tau, float)
    if drop is None:
        g = tau if trough == 0.5 else _skew(tau, _skew_kappa(trough))
        return 0.5 * (1.0 - np.cos(2 * np.pi * g))
    if not (0.0 < trough < 1.0 and drop > 0.0):
        raise ValueError(f"a trough at {trough} and a drop of {drop} (fractions of the step)")
    start = max(trough - drop, 0.0)
    out = np.zeros_like(tau)
    down = (tau > start) & (tau <= trough)
    out[down] = _min_jerk((tau[down] - start) / (trough - start))
    up = tau > trough
    out[up] = 1.0 - _min_jerk((tau[up] - trough) / (1.0 - trough))
    return out


def bob_trough(spec: GaitSpec, duty: float) -> float:
    """F2: the trough's place in the step (``bob_profile``'s tau): ``bob_skew`` of the stance after the touchdown, the
    step running from the trailing foot's mid-stance to the leading foot's; 0.5 (midway) without ``bob_skew``."""
    if spec.bob_skew is None:
        return 0.5
    return (1.0 - duty) + 2.0 * spec.bob_skew * duty


def _bump3(u):
    """64 u^3 (1 - u)^3 on [0, 1] (0 outside): peak 1 at 0.5, zero rate and curvature at both ends."""
    u = np.asarray(u, float)
    inside = (u > 0.0) & (u < 1.0)
    return np.where(inside, 64.0 * np.clip(u, 0, 1) ** 3 * (1.0 - np.clip(u, 0, 1)) ** 3, 0.0)


def toe_off_bump(t, t_r: float, t_lo: float, t_e: float, amp: float):
    """F5's foot pitch over one roll (rad): a min-jerk rise from 0 at the roll's start t_r to ``amp`` at the lift-off
    t_lo, then a min-jerk fall back to 0 at t_e (mid-swing): the foot leaves the floor at its peak pitch, and the pitch
    only falls in the air.  Returns (pitch, peak time = t_lo)."""
    if not t_r < t_lo < t_e:
        raise ValueError(f"a roll needs t_r < t_lo < t_e ({t_r}, {t_lo}, {t_e})")
    t = np.asarray(t, float)
    out = np.zeros_like(t)
    rise = (t > t_r) & (t <= t_lo)
    out[rise] = amp * _min_jerk((t[rise] - t_r) / (t_lo - t_r))
    fall = (t > t_lo) & (t < t_e)
    out[fall] = amp * (1.0 - _min_jerk((t[fall] - t_lo) / (t_e - t_lo)))
    return out, t_lo


#: A min-jerk rise of A over D peaks at 5.77 A / D^2 (10 sqrt(3) / 3): F5's ``toe_off_accel`` bounds it.
MIN_JERK_ACC = 10.0 * math.sqrt(3.0) / 3.0


def roll_amplitude(spec: GaitSpec, t_r: float, t_lo: float) -> float:
    """F5: one roll's peak pitch: ``toe_off``, or less where the roll's time on the floor D = t_lo - t_r is short: at
    most ``toe_off_rate`` D (its mean rate) and ``toe_off_accel`` D^2 / 5.77 (its min-jerk peak acceleration)."""
    D = t_lo - t_r
    amp = float(spec.toe_off)
    if spec.toe_off_rate is not None:
        amp = min(amp, spec.toe_off_rate * D)
    if spec.toe_off_accel is not None:
        amp = min(amp, spec.toe_off_accel * D * D / MIN_JERK_ACC)
    return float(amp)


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _contact_spheres(m, d, g, p, R) -> list[tuple[np.ndarray, float]]:
    """A geom's candidate floor-contact points in the foot frame (foot origin p, orientation R) as (centre, radius):
    a capsule's two end spheres, a sphere, a box's corners and a mesh's vertices (radius 0)."""
    Rg = R.T @ d.geom_xmat[g].reshape(3, 3)
    c = R.T @ (d.geom_xpos[g] - p)
    s = m.geom_size[g]
    t = m.geom_type[g]
    if t in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):  # a cylinder's end discs as spheres
        return [(c + e * s[1] * Rg[:, 2], float(s[0])) for e in (1.0, -1.0)]
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        return [(c, float(s[0]))]
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        return [(c + Rg @ (np.array([a, b, e]) * s[:3]), 0.0) for a in (-1, 1) for b in (-1, 1) for e in (-1, 1)]
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        return [(c + Rg @ x, 0.0) for x in v]
    raise ValueError(f"geom {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)}: type {t} not handled")


def _touch_pitch(centre: np.ndarray, r: float, pivot: np.ndarray) -> float:
    """The heel-up pitch about ``pivot`` (on the floor; rotation about the foot's y axis) at which a sphere of centre
    ``centre`` and radius r (foot frame) first touches the floor: its height above the pivot is
    -dx sin(th) + dz cos(th) - r.  Infinite if it never does (behind the pivot, or already touching)."""
    dx, dz = centre[0] - pivot[0], centre[2] - pivot[2]
    rho = math.hypot(dx, dz)
    if dx <= 0.0 or dz <= r or rho <= r:
        return math.inf
    return math.atan2(dz, dx) - math.asin(r / rho)


def foot_geometry(M: statics.Model) -> dict:
    """Per side, in the foot body's frame at the model's current pose (call it at home, feet flat): the floor pads'
    bottom corners and hull, the toe pivot (the front bottom edge of the frontmost toe pad: its two end points) and the
    heel pivot (the heel pad's rear bottom edge: its two end points).  A pivot point carries the edge's x and z (the
    pitch is about the foot's y axis, so its y does not matter).

    ``claw``: the foot's real front contact past a small pitch.  The toes, claws, hallux and plate carry no contact in
    the plant (``contype`` 0), but the middle toe's claw reaches ahead of the pads, so a rigid foot pitched heel-up about
    the toe edge touches the floor with it first (``touch_pitch``, rad) and then rolls on its tip (``roll_offset``).
    ``claw`` holds that geom, its tip sphere (centre and radius, foot frame), the floor contact at the touch pitch
    (relative to the flat foot's origin), the next-nearest geom's touch pitch about the toe edge (``next``) and the
    transition arc from the edge to the claw (``_transition_arc``, from ``ARC_START`` of the touch pitch) with how far
    the real foot's lowest point stays above the floor over it (``hover_max_m``, ``hover_min_m``); None where no geom
    reaches ahead of the edge."""
    m, d = M.m, M.d
    out = {}
    for side in SIDES:
        p, R = M.foot_pose(side)
        pads = {}
        for g in M.pads[side]:
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or str(g)
            Rg = R.T @ d.geom_xmat[g].reshape(3, 3)
            c = R.T @ (d.geom_xpos[g] - p)
            h = m.geom_size[g][:3]
            pads[nm] = np.array([c + Rg @ (np.array([sx, sy, -1.0]) * h) for sx in (-1, 1) for sy in (-1, 1)])
        corners = np.vstack(list(pads.values()))
        toes = {k: v for k, v in pads.items() if "toe" in k} or pads
        front = max(toes, key=lambda k: toes[k][:, 0].max())
        heels = {k: v for k, v in pads.items() if "heel" in k} or pads
        rear = min(heels, key=lambda k: heels[k][:, 0].min())

        def edge(c, x):
            e = c[np.abs(c[:, 0] - x) < 1e-9]
            return e[np.argsort(e[:, 1])][[0, -1]]

        toe_edge = edge(toes[front], toes[front][:, 0].max())
        heel_edge = edge(heels[rear], heels[rear][:, 0].min())
        toe_pivot = np.array([toe_edge[0, 0], 0.0, toe_edge[:, 2].mean()])
        # the first non-pad geom to reach the floor as the foot pitches heel-up about the toe edge
        touches = []
        foot_b = M.foot[side]
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != foot_b or m.geom_contype[g] == 1 or not np.any(m.geom_size[g] > 0):
                continue
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or str(g)
            for c, r in _contact_spheres(m, d, g, p, R):
                touches.append((_touch_pitch(c, r, toe_pivot), nm, c, r))
        touches.sort(key=lambda x: x[0])
        claw = None
        if touches and math.isfinite(touches[0][0]):
            th, nm, c, r = touches[0]
            Rt = _rot_y(th)
            centre_at_touch = toe_pivot + Rt @ (c - toe_pivot)  # relative to the flat foot origin
            nxt = next((x for x in touches[1:] if x[1] != nm), None)
            claw = dict(geom=nm, touch_pitch=float(th), centre=np.asarray(c, float), radius=float(r),
                        centre_at_touch=centre_at_touch, contact_at_touch=centre_at_touch - np.array([0.0, 0.0, r]),
                        next=None if nxt is None else dict(geom=nxt[1], touch_pitch=float(nxt[0])))
            claw["arc"] = _transition_arc(toe_pivot, claw["centre"], r, ARC_START * th)
            # the real foot over the arc: its lowest candidate point above the floor (never below it)
            cands = [(cc, rr) for _, _, cc, rr in touches] + [(toe_pivot, 0.0)]
            gaps = []
            for ph in np.linspace(claw["arc"]["start_pitch"], claw["arc"]["end_pitch"], 41):
                P = _contact_in_foot(dict(toe_pivot=toe_pivot, toe_edge=toe_edge, claw=claw), float(ph))[0][0]
                R_ = _rot_y(float(ph))
                gaps.append(min((R_ @ (cc - P))[2] - rr for cc, rr in cands))
            claw["arc"]["hover_max_m"] = float(max(gaps))
            claw["arc"]["hover_min_m"] = float(min(gaps))
        xy = corners[:, :2]
        out[side] = dict(pads=pads, corners=corners, hull_xy=xy[ConvexHull(xy).vertices], toe_pad=front,
                         toe_edge=toe_edge, toe_pivot=toe_pivot, heel_pad=rear, heel_edge=heel_edge,
                         heel_pivot=np.array([heel_edge[0, 0], 0.0, heel_edge[:, 2].mean()]), claw=claw)
    return out


#: F5: the claw transition arc starts at this fraction of the claw's touch pitch (``_transition_arc``).
ARC_START = 0.5


def _n(phi: float) -> np.ndarray:
    """The floor-contact normal, in the foot frame's x-z plane, of a profile point that is lowest at pitch phi."""
    return np.array([math.sin(phi), -math.cos(phi)])


def _transition_arc(E: np.ndarray, C: np.ndarray, r: float, th_e: float) -> dict:
    """F5's roll from the toe edge E (a vertex) to the claw tip sphere (centre C, radius r), all in the foot frame: a
    circular arc (radius rho, centre O) through E, lowest there at pitch th_e, and internally tangent to the sphere at
    its point lowest at pitch th_t (E - rho n(th_e) = O = C + (r - rho) n(th_t)).  Rolling on it carries the contact
    point continuously from the edge to the claw, so the foot never pivots through a corner at a finite rate (an
    impact: the foot's velocity would jump); the real foot, whose sole between the edge and the claw is the straight
    face of its hull, hovers above the floor by at most the arc's sagitta meanwhile (``foot_geometry``'s ``hover``)."""
    E2, C2 = np.array([E[0], E[2]]), np.array([C[0], C[2]])

    def f(tt):
        D = E2 - C2 - r * _n(tt)
        u = _n(th_e) - _n(tt)
        return D[0] * u[1] - D[1] * u[0]

    grid = np.linspace(th_e + 1e-6, 0.5 * math.pi, 4001)
    vals = np.array([f(x) for x in grid])
    i = int(np.flatnonzero(np.sign(vals[:-1]) != np.sign(vals[1:]))[0])
    lo, hi = grid[i], grid[i + 1]
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if np.sign(f(mid)) == np.sign(f(lo)) else (lo, mid)
    th_t = 0.5 * (lo + hi)
    D = E2 - C2 - r * _n(th_t)
    u = _n(th_e) - _n(th_t)
    rho = float(np.dot(D, u) / np.dot(u, u))
    if not rho > r:
        raise ValueError(f"F5 transition arc: radius {rho} not above the claw's {r}")
    centre = E2 - rho * _n(th_e)
    return dict(start_pitch=float(th_e), end_pitch=float(th_t), radius=rho, centre=np.array([centre[0], 0.0, centre[1]]),
                length=float(rho * (th_t - th_e)))


def _contact_in_foot(gs: dict, th: float) -> tuple[np.ndarray, str]:
    """F5 on the rigid foot at pitch th: its floor contact in the foot frame (two end points; one point twice off the
    edge) and which part carries it: "edge" (the toe pad's front bottom edge), "arc" (``_transition_arc``), "claw"
    (the claw tip sphere's lowest point) or "heel" (toes-up: the heel pad's rear bottom edge)."""
    if th < 0.0:
        return gs["heel_edge"].copy(), "heel"
    cl = gs["claw"]
    if cl is None or th <= cl["arc"]["start_pitch"]:
        return gs["toe_edge"].copy(), "edge"
    a = cl["arc"]
    if th <= a["end_pitch"]:
        P = a["centre"] + a["radius"] * np.array([math.sin(th), 0.0, -math.cos(th)])
        return np.vstack([P, P]), "arc"
    P = cl["centre"] + cl["radius"] * np.array([math.sin(th), 0.0, -math.cos(th)])
    return np.vstack([P, P]), "claw"


def roll_offset(gs: dict, th: float) -> tuple[np.ndarray, np.ndarray, str]:
    """F5 on the rigid foot (one side's ``foot_geometry``): the foot origin's offset from its flat pose at pitch th
    (rad; positive heel-up), its floor contact's two end points (relative to the flat foot origin) and the part that
    carries it (``_contact_in_foot``).  Heel-up it rolls without slipping on the convex sole: about the toe pad's front
    bottom edge (the edge stays where it lay), then on the transition arc and the claw tip sphere, the contact moving
    forward by the length rolled; toes-up (negative) about the heel pad's rear bottom edge."""
    R = _rot_y(th)
    P, part = _contact_in_foot(gs, th)
    if part == "heel":
        return (np.eye(3) - R) @ gs["heel_pivot"], P, part
    v = gs["toe_pivot"]
    if part == "edge":
        return (np.eye(3) - R) @ v, P, part
    cl = gs["claw"]
    a = cl["arc"]
    rolled = (a["radius"] * (min(th, a["end_pitch"]) - a["start_pitch"])
              + (cl["radius"] * (th - a["end_pitch"]) if part == "claw" else 0.0))
    W = v + np.array([rolled, 0.0, 0.0])  # the contact on the floor, relative to the flat foot origin (y irrelevant)
    W[1] = P[0][1]
    return W - R @ P[0], np.vstack([W, W]), part


def rolling_entry(geo: dict, pad_c: dict, inset: dict, feet0: dict, stride: float, margin: float) -> dict:
    """F5: per trailing side, the fraction a* of the double support's ZMP hand-over (``generate``'s linear hand-over
    from the trailing to the leading pad centroid) from which the ZMP reference lies inside the rolling support by at
    least ``margin`` (m): the trailing foot's toe pivot edge and the leading foot's pads (``a``, where the roll may
    start) and, where the foot rolls on to its claw (``foot_geometry``), the claw's floor contact and the leading pads
    (``by_support["claw"]``: the roll's min-jerk rise reaches the claw's touch pitch later than that, which
    ``_generate_look`` checks frame by frame).  With the edge support's hull and the smaller margin at the hand-over's
    end.  None where the leading pad centroid itself is not that far inside either support."""
    out = {}
    for trail, lead in (("right", "left"), ("left", "right")):
        Pt = np.array([feet0[trail][0], feet0[trail][1]])
        Pl = np.array([feet0[lead][0] + 0.5 * stride, feet0[lead][1]])
        supports = {"toe_edge": geo[trail]["toe_edge"][:, :2]}
        if geo[trail]["claw"] is not None:
            supports["claw"] = geo[trail]["claw"]["contact_at_touch"][None, :2]
        ct, cl = Pt + pad_c[trail] + inset[trail], Pl + pad_c[lead] + inset[lead]
        res = {}
        for key, sup in supports.items():
            pts = np.vstack([sup + Pt, geo[lead]["hull_xy"] + Pl])
            hull = pts[ConvexHull(pts).vertices]

            def marg(a, hull=hull):
                return statics.support_margin((1 - a) * ct + a * cl, hull)

            if marg(1.0) < margin:
                res = None
                break
            lo, hi = 0.0, 1.0
            if marg(0.0) >= margin:
                hi = 0.0
            else:  # the margin along the hand-over is concave (a minimum of affine functions): one crossing
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    lo, hi = (lo, mid) if marg(mid) >= margin else (mid, hi)
            res[key] = dict(a=float(hi), margin_at_end_m=float(marg(1.0)), hull=hull)
        if res is None:
            out[trail] = None
            continue
        out[trail] = dict(a=res["toe_edge"]["a"], margin_at_end_m=min(r["margin_at_end_m"] for r in res.values()),
                          hull=res["toe_edge"]["hull"], by_support={k: r["a"] for k, r in res.items()})
    return out


class _Warp:
    """F14's time warp of a window of K strides: phase phi (strides) to time t.  Step j (phases [j/2, (j+1)/2), the
    left foot lifting at integer phases) lasts T / 2 (1 + e_j) with dt/dphi = T (1 + e_j 2 sin^2(2 pi (phi - j/2))),
    so the time is C2 in phase; with sum(e) = 0 the window lasts K T and repeats.  No deviations: t = phi T."""

    def __init__(self, T: float, K: int, e=None):
        self.T, self.K = float(T), int(K)
        self.e = np.zeros(2 * self.K) if e is None else np.asarray(e, float)
        if len(self.e) != 2 * self.K or abs(self.e.sum()) > 1e-12 or np.any(np.abs(self.e) >= 0.5):
            raise ValueError("step deviations: one per step of the window, summing to zero, each under 0.5")
        self.cum = np.concatenate([[0.0], np.cumsum(0.5 * self.e)])
        self.identity = not np.any(self.e)

    def _parts(self, phi):
        phi = np.asarray(phi, float)
        w = np.floor(phi / self.K)
        p = phi - w * self.K
        j = np.clip(np.floor(2.0 * p).astype(int), 0, 2 * self.K - 1)
        return w, p, j, p - 0.5 * j

    def time(self, phi):
        if self.identity:
            return np.asarray(phi, float) * self.T
        w, p, j, u = self._parts(phi)
        return self.T * (p + self.cum[j] + self.e[j] * (u - np.sin(4 * np.pi * u) / (4 * np.pi))) + w * self.K * self.T

    def rate(self, phi):
        """dt / dphi."""
        if self.identity:
            return np.full(np.shape(phi), self.T)
        _, _, j, u = self._parts(phi)
        return self.T * (1.0 + 2.0 * self.e[j] * np.sin(2 * np.pi * u) ** 2)

    def phase(self, t):
        t = np.asarray(t, float)
        if self.identity:
            return t / self.T
        phi = t / self.T
        for _ in range(50):
            f = self.time(phi) - t
            phi = phi - f / self.rate(phi)
            if np.max(np.abs(f)) < 1e-13 * self.T:
                break
        return phi


def step_deviations(spec: GaitSpec, rng: np.random.Generator) -> np.ndarray:
    """F14: e_j per step of the window: seeded jitter of largest size ``phase_jitter`` (uniform draws, mean removed,
    scaled so the largest is exactly that) plus +-asym/2 on the left and right steps; the sum is zero."""
    steps = 2 * int(spec.strides)
    side = np.where(np.arange(steps) % 2 == 0, 1.0, -1.0)
    e = 0.5 * spec.asym * side
    if spec.phase_jitter:
        z = rng.uniform(-1.0, 1.0, steps)
        z = z - z.mean()
        e = e + spec.phase_jitter * z / np.max(np.abs(z))
    return e - e.mean()


def slow_variation(rng: np.random.Generator, window_s: float, band: tuple, t) -> tuple[np.ndarray, list]:
    """F14: a seeded smooth periodic signal of the window, |g| <= 1 with 1 reached: the window's harmonics k / W that
    lie inside ``band`` (Hz), with amplitudes in [0.5, 1] and phases drawn at random.  Zero if none lies inside."""
    t = np.asarray(t, float)
    lo, hi = band
    ks = [k for k in range(1, int(math.floor(hi * window_s + 1e-9)) + 1) if lo - 1e-12 <= k / window_s <= hi + 1e-12]
    if not ks:
        return np.zeros_like(t), []
    amp = rng.uniform(0.5, 1.0, len(ks))
    ph = rng.uniform(0.0, 2 * np.pi, len(ks))

    def g(x):
        return sum(a * np.sin(2 * np.pi * k * x / window_s + p) for a, k, p in zip(amp, ks, ph, strict=True))

    scale = max(np.max(np.abs(g(np.arange(VAR_GRID) * window_s / VAR_GRID))), np.max(np.abs(g(t))))
    return g(t) / scale, [k / window_s for k in ks]


def _quat_mul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


def _nose_up(quat) -> float:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(quat, float))
    x = R.reshape(3, 3)[:, 0]
    return float(math.atan2(x[2], math.hypot(x[0], x[1])))


def support_margins(M: statics.Model, qs, phases, zmb, zmp, rolling: dict | None = None, contact: dict | None = None):
    """Per frame, the support (the convex hull of the stance feet's contact: a flat foot's pad bottoms, a rolling
    foot's floor contact ``contact[side][k]``, its toe edge's or claw tip's two end points (world), where
    ``rolling[side][k]``) and the signed margins inside it (m, positive inside) of the multibody ZMP ``zmb`` and the ZMP
    reference ``zmp``.  A support that is a single edge or point has no inside: its margin is minus the distance to its
    nearer end.  Returns (margin, margin_ref, hulls)."""
    n = len(qs)
    margin, margin_ref, hulls = np.zeros(n), np.zeros(n), []
    for k in range(n):
        M.forward(qs[k])
        pts = []
        for s in phases[k] or SIDES:
            if rolling is not None and rolling[s][k]:
                pts.append(np.asarray(contact[s][k])[:, :2])
            else:
                pts.append(M.pad_polygon(s))
        P = np.vstack(pts)
        try:
            hull = P[ConvexHull(P).vertices]
        except Exception:  # a single rolling edge: degenerate
            hull = P
        hulls.append(hull)
        if len(hull) >= 3:
            margin[k] = statics.support_margin(zmb[k], hull)
            margin_ref[k] = statics.support_margin(zmp[k], hull)
        else:
            margin[k] = -float(np.min(np.linalg.norm(hull - zmb[k], axis=1)))
            margin_ref[k] = -float(np.min(np.linalg.norm(hull - zmp[k], axis=1)))
    return margin, margin_ref, hulls


def _generate_look(M: statics.Model, q_home: np.ndarray, H: float, spec: GaitSpec):
    """``generate`` with the look features (module docstring).  The trajectory dict has ``generate``'s keys, with the
    window's frames (``spec.strides`` x ``spec.n``), plus ``look``: the per-frame feature values and support."""
    m = M.m
    info = spec.describe(H)
    v, stride, T, duty = info["speed"], info["stride"], info["period"], info["duty"]
    if not 0.5 < duty < 1.0:
        raise ValueError(f"duty {duty} must lie in (0.5, 1): the walk alternates single and double support")
    if spec.head_mode not in ("nod", "stabilise"):
        raise ValueError(f"head_mode {spec.head_mode!r}: 'nod' or 'stabilise'")
    K = int(spec.strides)
    if K < 1:
        raise ValueError("strides must be at least 1")
    n1 = spec.n
    n = n1 * K
    dt = T / n1
    t = np.arange(n) * dt
    W = K * T
    M.forward(q_home)
    feet0 = {s: M.foot_pose(s)[0].copy() for s in SIDES}
    pad_c = {s: statics._polygon_centroid(M.pad_polygon(s)) - feet0[s][:2] for s in SIDES}
    geo = foot_geometry(M)
    com0 = M.com()
    root0 = q_home[:3].copy()
    sw = 1.0 - duty
    lift = spec.lift_frac * H
    rng_steps, rng_gaze, rng_tail = (np.random.default_rng(s) for s in np.random.SeedSequence(spec.noise_seed).spawn(3))
    warp = _Warp(T, K, step_deviations(spec, rng_steps))
    phi = warp.phase(t)
    ph = np.mod(phi, 1.0)
    offs = {"left": 0.0, "right": 0.5}
    inset = {s: np.array([0.0, -np.sign(feet0[s][1]) * spec.zmp_inset_y]) for s in SIDES}

    # --- feet: footprints, swing path (F4 lift shape), swing flags
    u, swing, base = {}, {}, {}
    for s, off in offs.items():
        uu = np.mod(phi - off, 1.0)
        sws = uu < sw
        a = _min_jerk(np.clip(uu / sw, 0.0, 1.0))
        x = np.where(sws, feet0[s][0] + stride * (phi - uu) - duty * stride / 2 + a * stride,
                     feet0[s][0] + stride * (phi - (uu - sw)) + duty * stride / 2)
        z = feet0[s][2] + np.where(sws, lift * lift_profile(np.clip(uu / sw, 0.0, 1.0), spec.lift_peak), 0.0)
        u[s], swing[s] = uu, sws
        base[s] = np.column_stack([x, np.full(n, feet0[s][1]), z])
    phases = [tuple(s for s in SIDES if not swing[s][k]) for k in range(n)]

    # --- ZMP reference: generate's (stance pad centroid; linear hand-over in double support), F7's inset
    zmp = np.zeros((n, 2))
    for k in range(n):
        sup = phases[k]
        if len(sup) == 1:
            sd = sup[0]
            zmp[k] = base[sd][k, :2] + pad_c[sd] + inset[sd]
        else:
            if ph[k] < 0.5:
                frm, to, s0, s1 = "right", "left", sw, 0.5
            else:
                frm, to, s0, s1 = "left", "right", 0.5 + sw, 1.0
            a = (ph[k] - s0) / (s1 - s0)
            zmp[k] = (1 - a) * (base[frm][k, :2] + pad_c[frm] + inset[frm]) + a * (base[to][k, :2] + pad_c[to] + inset[to])

    # --- F5: foot pitch about the toe edge, then the claw tip (rise to each lift-off, fall by mid-swing), and the
    # toes-up reach before landing
    pitch = {s: np.zeros(n) for s in SIDES}
    rolls = {s: [] for s in SIDES}
    entry = None
    if spec.toe_off:
        entry = rolling_entry(geo, pad_c, inset, feet0, stride, spec.toe_off_margin)
        for s, off in offs.items():
            if entry[s] is None:
                raise ValueError(f"F5: the ZMP reference never lies {spec.toe_off_margin} m inside the rolling support "
                                 f"when the {s} foot trails")
            for i in range(-1, K + 1):
                f_lo = i + off
                f_r = max(f_lo - (1.0 - spec.toe_off_start) * duty, f_lo - (1.0 - entry[s]["a"]) * (duty - 0.5))
                t_r, t_lo, t_e = (float(warp.time(f)) for f in (f_r, f_lo, f_lo + 0.5 * sw))
                amp = roll_amplitude(spec, t_r, t_lo)
                cl = geo[s]["claw"]
                th, t_pk = toe_off_bump(t, t_r, t_lo, t_e, amp)
                pitch[s] += th
                rolls[s].append(dict(lift_off_phase=f_lo, start_phase=f_r, t_start=t_r, t_lift_off=t_lo, t_peak=t_pk,
                                     t_level=t_e, start_frac_stance=1.0 - (f_lo - f_r) / duty, amplitude=amp,
                                     ground_roll_s=t_lo - t_r, at_lift_off=amp,
                                     past_toe_edge_at_lift_off=bool(cl is not None and amp > cl["arc"]["start_pitch"])))
    if spec.toe_reach:
        for s in SIDES:
            sfrac = np.clip(u[s] / sw, 0.0, 1.0)
            pitch[s] -= np.where(swing[s] & (sfrac > 0.5), spec.toe_reach * _bump3(2.0 * sfrac - 1.0), 0.0)
    target_pos, target_R, contact, on_claw = {}, {}, {}, {}
    for s in SIDES:
        P = base[s].copy()
        Rs = np.tile(np.eye(3), (n, 1, 1))
        C = np.full((n, 2, 3), np.nan)
        cw = np.zeros(n, bool)
        for k in np.flatnonzero(pitch[s]):
            offset, sup, part = roll_offset(geo[s], float(pitch[s][k]))
            cw[k] = part in ("arc", "claw")
            P[k] = base[s][k] + offset
            Rs[k] = _rot_y(float(pitch[s][k]))
            C[k] = base[s][k] + sup
        target_pos[s], target_R[s], contact[s], on_claw[s] = P, Rs, C, cw

    # --- F1/F2 bob, F6 trunk roll, F9 tail, F10 head, F14 slow variations
    trough = bob_trough(spec, duty)
    dz = np.zeros(n)
    if spec.bob:
        tau = np.mod(phi - (0.5 - duty / 2) - spec.bob_phase, 0.5) / 0.5
        drop = None if spec.bob_drop is None else spec.bob_drop / (0.5 * T)
        dz = -spec.bob * H * bob_profile(tau, trough, drop)
    roll = -spec.roll_amp * np.cos(2 * np.pi * (phi - (1.0 - duty / 2) - spec.roll_phase)) if spec.roll_amp else np.zeros(n)
    trunk_q = np.array([_quat_mul(q_home[3:7], [math.cos(r / 2), math.sin(r / 2), 0.0, 0.0]) for r in roll])
    g_gaze, f_gaze = slow_variation(rng_gaze, W, spec.var_band, t) if spec.gaze_var else (np.zeros(n), [])
    g_tail, f_tail = slow_variation(rng_tail, W, spec.var_band, t) if spec.tail_var else (np.zeros(n), [])
    tail = spec.tail_amp * (1.0 + spec.tail_var * g_tail) * np.sin(2 * np.pi * phi + spec.tail_phase)
    touchdowns = sorted(float(warp.time(i + off + sw)) for off in offs.values() for i in range(-1, K + 1))
    settle = np.zeros(n)
    if spec.head_mode == "nod":
        neck = spec.head_amp * np.sin(4 * np.pi * phi)
    else:
        pitch0 = _nose_up(q_home[3:7])
        neck = -spec.head_gain * np.array([_nose_up(qt) - pitch0 for qt in trunk_q])
        if spec.head_settle:
            for t_td in touchdowns:
                settle += _bump3((t - t_td) / (2.0 * spec.settle_lag))
            neck = neck - spec.head_settle * settle
    neck = neck + spec.gaze_var * g_gaze

    # --- centre of mass and the multibody-ZMP correction passes, as generate
    z_c = com0[2] - spec.crouch * H
    omega = np.sqrt(G / z_c)
    trend = v * t
    kk = np.fft.rfftfreq(n, d=dt) * 2 * np.pi
    gain = 1.0 / (1.0 + (kk / omega) ** 2)
    neck_adr = m.jnt_qposadr[statics.jid(m, "neck_pitch")]
    tail_adr = m.jnt_qposadr[statics.jid(m, "tail_yaw")]
    off_c = com0[:2] - root0[:2]

    def solve_frames(zref):
        cx = np.fft.irfft(np.fft.rfft(zref[:, 0] - trend) * gain, n) + trend
        cy = np.fft.irfft(np.fft.rfft(zref[:, 1]) * gain * spec.sway_gain, n)
        qs = np.zeros((n, m.nq))
        q = q_home.copy()
        ok_all, fails = True, []
        for k in range(n):
            targets = {s: (target_pos[s][k], target_R[s][k]) for s in SIDES}
            q[2] = root0[2] - spec.crouch * H + dz[k]
            q[3:7] = trunk_q[k]
            q[neck_adr] = neck[k]
            q[tail_adr] = tail[k]
            q[0], q[1] = cx[k] - off_c[0], cy[k] - off_c[1]
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
        qd, qdd = derivatives(M, qs, dt, stride=K * stride)
        zmb = np.array([statics.total_zmp(M, qs[k], qd[k], qdd[k]) for k in range(n)])
        e = zmb - zmp
        history.append(float(np.nanmax(np.linalg.norm(e, axis=1))))
        if it == spec.zmp_iters:
            break
        zref = zref - spec.zmp_gain * np.nan_to_num(e)

    # --- the support each frame (flat pads, or a rolling foot's toe edge or claw tip) and the ZMP's margin inside it
    rolling = {s: ~swing[s] & (pitch[s] > 1e-9) for s in SIDES}
    if any(len(phases[k]) == 1 and rolling[phases[k][0]][k] for k in range(n)):
        raise ValueError("F5: a foot rolls in single support (the roll must start in double support)")
    margin, margin_ref, hulls = support_margins(M, qs, phases, zmb, zmp, rolling, contact)
    rolling_any = np.any([rolling[s] for s in SIDES], axis=0)
    if rolling_any.any() and np.min(margin_ref[rolling_any]) < spec.toe_off_margin - 1e-9:
        raise ValueError(f"F5: the ZMP reference comes within {1e3 * np.min(margin_ref[rolling_any]):.2f} mm of a rolling "
                         f"foot's support edge (toe_off_margin {1e3 * spec.toe_off_margin:g} mm)")
    amp = float(0.5 * (cy.max() - cy.min()))
    info = info | dict(sway_amp=amp, lift=lift, omega=float(omega), zmp_error_history_m=history, accuracy=ACCURACY,
                       strides=K, window_s=float(W), swing_time=float(sw * T),
                       look={k: getattr(spec, k) for k in LOOK_FIELDS},
                       bob_m=float(spec.bob * H), bob_trough_step=float(trough),
                       bob_drop_step=None if spec.bob_drop is None else float(spec.bob_drop / (0.5 * T)),
                       step_deviations=warp.e.tolist(), gaze_freqs_hz=f_gaze, tail_freqs_hz=f_tail,
                       rolling_entry=None if entry is None else {s: dict(a=e_["a"], margin_at_end_m=e_["margin_at_end_m"],
                                                                         by_support=e_["by_support"])
                                                                for s, e_ in entry.items()},
                       zmp_margin_min_m=float(np.min(margin)), zmp_margin_ref_min_m=float(np.min(margin_ref)))
    look = dict(phase=phi, swing=np.column_stack([swing[s] for s in SIDES]),
                swing_u=np.column_stack([np.where(swing[s], u[s] / sw, np.nan) for s in SIDES]),
                foot_pitch=np.column_stack([pitch[s] for s in SIDES]),
                rolling=np.column_stack([rolling[s] for s in SIDES]),
                rolling_contact=np.stack([np.where(rolling[s][:, None, None], contact[s], np.nan) for s in SIDES], axis=1),
                on_claw=np.column_stack([on_claw[s] for s in SIDES]),
                foot_base=np.stack([base[s] for s in SIDES], axis=1),
                foot_target=np.stack([target_pos[s] for s in SIDES], axis=1),
                bob=dz, trunk_roll=roll, trunk_quat=trunk_q, neck=neck, tail=tail, head_settle=settle,
                gaze_var=g_gaze, tail_var=g_tail, zmp_margin=margin, zmp_ref_margin=margin_ref, support_hull=hulls,
                touchdowns=touchdowns, rolls=rolls, geometry=geo, warp=warp)
    return dict(t=t, q=qs, phases=phases, dt=dt, zmp_ref=zmp, com_ref=np.column_stack([cx, cy]), zmp_mb=zmb,
                info=info, ik_ok=ok_all, ik_fail_frames=fails, look=look)
