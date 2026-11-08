"""Turning loads on the reference robot: prescribed turning walks and turns in place (not a controller).

Round one sized every joint from straight reference walks, so thigh-twist (``hip_yaw``) and hip-roll
loads in turns were never computed (SCREENING.md section 7).  This module prescribes turning reference
trajectories, solves them with the machinery of ``gait.generate`` and ``gait.inverse_dynamics``, checks
them for self-collision with ``clearance.screen`` and compares the loads with each candidate's selected
drives on the planning limits and margins of ``sizing.py``.  It screens loads and clearances along
prescribed motion.  It does not show that a controller can turn the robot, and nothing is simulated
forward in time.

Run from hardware/trex_form_study:
    python -m designlab.turning 60-M-T-10 75-M-H-10 --out <dir>         # Stage A bodies (round one)
    python -m designlab.turning 60-M-T-10 --round2 round2 --out <dir>   # round-two sized records
writes <dir>/<id>_turning.json and .png and <dir>/turning_index.json; the exit status is 1 if a candidate
failed.  Round two calls ``screen_record`` from ``round2.screen_one``: it writes <scr>/<id>_turning.json, whose
per-run ``groups`` feed ``envelope.turning_rms_from``, and returns ``record_summary`` for the review index
(``shortlist`` reads targets_within_limits, worst_ratio_targets, hip_yaw_range_used_max and mu_needed_max).

Method (computed from the prescribed trajectory)
------------------------------------------------
* Path frame.  A frame moves at speed v along a circle of radius R = v / omega with heading omega t
  (omega: yaw rate, positive to the left).  It is a straight line for omega = 0 and a fixed point for a
  turn in place (v = 0).  Its origin starts at the midpoint of the two hip-pitch joints in the home pose.
  Each stride is the previous one carried by one rigid planar motion D (rotation omega T about the
  turning centre), so the gait is periodic in the path frame.
* Footprints.  A footprint is the home foot pose carried by the path frame to that step's mid-stance.
  The footprints of each foot therefore lie on a circle about the turning centre (radius R minus or
  plus the foot's lateral offset) and are yawed with the body.  Stance feet stay fixed.  A swing foot
  moves along the chord between consecutive footprints and yaws between their headings, both with
  the minimum-jerk profile, and lifts with the sine-squared profile of ``gait.generate``.  A turn in
  place keeps the nominal cadence and duty with v = 0, so the footprints rotate about the vertical
  axis through the hip midpoint.
* Events.  A frame that falls on a lift-off counts as swing and one on a touchdown as stance, within
  ``PHASE_TOL``.  ``gait.generate``'s rule (u < 1 - duty) leaves those frames to the rounding of t / T, so
  depending on the stride period a touchdown frame becomes swing for one foot or both: at 160 frames per
  stride its nominal walk has 32 double-support frames at 60 cm, 31 on 75-M-T-10 and 30 on 75-M-H-10.
* Centre of mass.  ``gait.generate``'s linear inverted pendulum c'' = w0^2 (c - p), w0^2 = g / z_com, is
  solved in the path frame.  With c = P + R(theta) u and the zero-moment-point (ZMP) reference
  p = P + R(theta) q (complex x + i y), the periodic solution is, harmonic by harmonic,
      u_k = (w0^2 q_k + i omega v delta_k0) / ((lambda_k + omega)^2 + w0^2),
  lambda_k being the harmonic's angular frequency.  At omega = 0 this is ``gait.generate``'s gain
  1 / (1 + (lambda / w0)^2).  The constant term is the inward lean of the turn.  The ZMP reference is
  ``gait.generate``'s (pad centroid in single support, linear hand-over in double support).  It is
  corrected against the multibody ZMP with the same passes and gain.
* Joint angles.  The trunk keeps its home roll and pitch and yaws at omega.  Neck and tail move as in
  ``gait.generate``.  The legs are solved by ``statics.leg_ik``, whose joint ranges are hard limits:
  a turn that needs more thigh twist than the joint allows shows up as an IK failure.  IK success is
  judged on the final pose of each frame (foot pose error <= 1e-4).
* Convergence.  The pelvis is shifted until the model's CoM is within ``COM_TOL`` = 1e-6 m of the pendulum
  path (``gait.generate`` at round one's accuracy stops at 2e-4 m), and the stride boundary is differenced
  with the exact unwrap below (``gait.derivatives`` at round one's accuracy approximates the next stride's root
  x).  The multibody-ZMP correction converges to about a millimetre only with both: with either one alone
  9-71 mm of ZMP error remain on the round-one bodies (output ``validation.omega0_vs_gait_generate``).  Round
  two runs ``gait.generate`` with both (``gait.set_accuracy("r2")``), so the screen's straight walk and the
  sizing's walk agree there.
* Periodicity: exact unwrapping, not extra strides.  One stride is generated.  The pose after the
  last frame is D applied to the first: root position rotated about the turning centre, root
  quaternion pre-multiplied by the yaw omega T, root linear velocity (world frame) rotated, body-frame
  angular velocity and joint coordinates unchanged.  Derivatives are central differences, as in
  ``gait.derivatives``.
* Loads.  For each frame the joint torques and foot wrenches come from the base rows of the equations of
  motion, as in ``statics.contact_torques``: unique in single support, the minimum-joint-effort split in
  double support with normal force >= 0, a friction pyramid (mu 0.8) and the CoP inside each pad's box.
  Here the pyramid and the box are taken in each foot's own axes and the QP is solved exactly (``qp_exact``,
  Lawson and Hanson's least-distance method by NNLS).  ``statics`` takes them in world axes: in a turn the
  world-axis box of a yawed pad is larger than the pad's own and turns with the heading, so the split
  depended on the absolute heading, and mirrored left and right turns (whose headings differ by half a
  stride's yaw) got different ankle loads.  In a straight walk the two rules give the same split.
* Friction, single support (the wrench is unique).  For the stance foot: the tangential-to-normal ratio
  F_t / F_n, the free (vertical) moment M_z about the centre of pressure (CoP) and the friction utilisation
  |M_z| / (mu F_n r_eff).  r_eff is the mean distance from the pad-area centroid over the four pads, the
  torsional friction radius of a flat contact under uniform pressure (2R/3 for a disc).  A second radius,
  r_cop, comes from a pressure field linear in position and zero in tension (a rigid foot on uniform
  elastic pads) with its resultant at the computed CoP.  The combined slip index is the ellipsoidal limit
  surface (Howe and Cutkosky 1996): (F_t / mu F_n)^2 + (M_z / mu F_n r)^2 <= 1.  The friction coefficient at
  which it reaches 1 is ``mu_req``; it does not depend on the assumed mu.  ``mu_req_centroid`` takes M_z
  about the pad-area centroid instead, the form the double-support requirement uses.
* Friction, double support.  The split there is a choice, so its per-foot values are not requirements
  (reported as ``double_support_qp_allocation``, a diagnostic).  The requirement is ``ds_min_mu``: the
  smallest mu for which some split keeps both CoPs inside their pad hulls with
  ||(F_x, F_y, M_z / r_eff)|| <= mu F_n per foot (M_z about the pad-area centroid), found by linear programs
  on a 256-plane polyhedral cone; the exact cone ratio at the split found is reported, with the polyhedral
  optimum as a lower bound.
* Self-collision.  ``clearance.screen`` (round-two gate set) runs on every turning run, both directions,
  and on the screen's straight walk.  A case passes if no pair of screened parts on two bodies
  interferes (signed distance < 0) in either direction, the gate of the round-one and round-two walks.
  Extra left turns (``CLEARANCE_PROBES``, kinematics only) locate the yaw rate at which interference
  begins, by linear interpolation of the smallest gap between computed rates.
* Requirements.  These follow ``sizing.requirements``: sustained planning torque >= max(RMS,
  single-support hold); planning peak >= 1.25 x max(peak, hold); speed on the bus >= 1.15 x peak speed.
  RMS is taken per motor (the worse leg in either turning direction), because a turn loads the two
  legs differently.  ``sizing`` stacks both legs, which gives the same value for a straight walk.  The
  hold and the requirement as sized come from ``sizing.requirements`` on the same model.  The
  "more than 10 % above the straight walk" flag compares with straight nominal and brisk walks made
  by this generator with the same settings, so only the turn differs; the ratio to the requirement as
  sized is reported next to it.  Ratios within ``NEAR_FLAG_BAND`` below the flag or below a planning limit
  are listed as near misses: halving the frame interval moves these ratios by up to 0.010.
* Gates per case: drives within their planning limits (``targets_within_limits``, the shortlist's gate),
  IK on every frame, and no self-interference; ``targets_pass`` requires all three.

Assumptions (numbers not computed here; sources in data/research/turning_2026-11-07.json)
----------------------------------------------------------------------------------------
* Yaw rates: 0.35 rad/s walking at the nominal speed and 0.5 rad/s in place (BEHAVIOR_SPEC.md section
  1), plus a 0.5 rad/s walking turn as a stretch case.  Speed, cadence, duty, lift, crouch, head and
  tail motion are those of ``sizing.NOMINAL``.
* mu = 0.8 at the pads: assumption, the friction pyramid of ``statics._double_support_qp``.  The
  compiled MJCF uses 0.9.  Context only: Crossville classes its tile products for interior dry use at a
  dry DCOF of 0.42 or more and for interior wet use at a wet DCOF of 0.42 or more (ANSI A326.3 test), cites
  0.42 wet as the industry benchmark for level interior spaces walked on when wet, and says the classes
  should not be relied on alone.  These describe tile measured with the standard's device, not the robot's
  pads, and are not a floor minimum.
* Uniform pressure (r_eff), or a linear no-tension pressure (r_cop), over the pad boxes (assumption).

Limits
------
* A prescribed reference.  No controller, no balance feedback, no slip, no impacts, no carpet.  After
  the correction passes the multibody ZMP is within about 1 mm of the reference (``zmp_error_history_m``).
* Pads are rigid, flat and Coulomb.  Elastomer pads, pad wear and floor contamination change mu and r.
* Trunk roll and pitch are held at home.  A real turn may bank or pitch the trunk.
* The double-support split is the minimum-effort choice, not a measurement; the double-support friction
  requirement does not depend on it.
* Clearance uses the mid-surface proxy hulls of ``clearance``: the outer skins sit closer than the gaps
  reported.
* No transitions: steady turning only (no entry into or exit from the turn, no change of direction).
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

from . import gait, sizing, statics

G = gait.G
SIDES = statics.SIDES

#: Turning cases.  Targets: BEHAVIOR_SPEC.md section 1 ("Turning": >= 0.35 rad/s while walking at nominal
#: speed, >= 0.5 rad/s turning in place).  The 0.5 rad/s walking turn is the round-two stretch case.
CASES = {
    "walk_0.35": dict(yaw_rate=0.35, in_place=False, role="target"),
    "in_place_0.50": dict(yaw_rate=0.50, in_place=True, role="target"),
    "walk_0.50": dict(yaw_rate=0.50, in_place=False, role="stretch"),
}
#: Friction coefficient assumed at the pads (assumption): the friction pyramid of statics._double_support_qp.
MU = 0.8
#: Context only (sourced): the DCOF of the ANSI A326.3 product classes for level interior dry and interior wet use,
#: via Crossville's summary (accessed 2026-11-07, re-read 2026-11-08).  A classification of tile products, not a floor minimum.
DCOF_A326_CLASS = 0.42
#: Double support, QP-allocation diagnostic only: a foot enters those statistics while carrying this fraction of
#: the weight (assumption).  The double-support requirement is ``ds_min_mu``.
FN_MIN_FRAC = 0.10
#: Flag a turning requirement above the straight-walk requirement by more than this fraction (round-two brief).
INCREASE_FLAG = 0.10
#: Ratios within this band below the 10 % flag, or below a planning limit, are listed as "near": inside the
#: method's measured sensitivity (validation in METHOD.md: frame resolution, mirror, double-support split).
NEAR_FLAG_BAND = 0.02
#: Clearance gate set used on the turning runs (``clearance.screen``).
CLEARANCE_GATES = "r2"
#: Extra left turns (kinematics only) screened for self-interference to locate the yaw rate at which it begins.
CLEARANCE_PROBES = {"walk": (0.40, 0.45), "in_place": (0.35, 0.40, 0.45)}
#: Straight-walk requirements below this (N·m or rad/s) count as zero when forming ratios.
ZERO_REQ = 1e-3
#: Samples per pad edge for the pad-area integrals (PAD_GRID^2 per pad).
PAD_GRID = 8
#: Foot pose error (m + rad, as statics.leg_ik) above which a frame counts as an IK failure.
IK_TOL = 1e-4
#: CoM-tracking tolerance (m) and pass limit per frame.  gait.generate at round one's accuracy uses 2e-4 m and 4
#: passes (round two's, gait.set_accuracy("r2"), uses these values); differenced
#: twice, that residual moves the multibody ZMP by centimetres, and its ZMP correction then does not converge.
#: At 1e-6 m, together with the exact stride-boundary unwrap, it converges to about a millimetre (neither change
#: is enough alone; validation block of the output).
COM_TOL = 1e-6
COM_ITERS = 12
#: Frames within this stride fraction of a lift-off or touchdown are classified by the event (see foot_schedule).
PHASE_TOL = 1e-9
#: Tangent planes of the polyhedral friction cone in the double-support friction requirement (``ds_min_mu``);
#: the exact cone ratio at the split found is reported, with the polyhedral optimum as a lower bound.
DS_DIRECTIONS = 256

LEG_GROUPS = ("hip_roll", "hip_yaw", "hip_pitch", "knee", "ankle_a", "ankle_b")
PLOT_JOINTS = (("hip_roll", "Hip roll"), ("hip_yaw", "Thigh twist (hip_yaw)"), ("hip_pitch", "Hip pitch"),
               ("knee", "Knee"))


# ================================================================================================
# Planar helpers
# ================================================================================================
def _rot2(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def _rotz(a):
    R = np.eye(3)
    R[:2, :2] = _rot2(a)
    return R


def _quat_yaw(a, quat):
    """``quat`` pre-multiplied by a rotation of ``a`` about the world z axis."""
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.array([math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)]), np.asarray(quat, float))
    return out


def path_frame(t, speed: float, yaw_rate: float, origin):
    """Origin (n, 2) and heading (n,) of the path frame at times ``t``.

    The origin moves at ``speed`` along a circle of curvature yaw_rate / speed with heading yaw_rate * t,
    starting at ``origin`` with heading 0.  sin(w t)/w and (1 - cos w t)/w are written with sinc, so the
    straight walk (yaw_rate = 0) is the exact limit, and speed = 0 gives a fixed point (turn in place)."""
    t = np.atleast_1d(np.asarray(t, float))
    th = yaw_rate * t
    s = t * np.sinc(th / np.pi)
    c = 0.5 * yaw_rate * t * t * np.sinc(th / (2 * np.pi)) ** 2
    P = np.asarray(origin, float)[None, :] + speed * np.column_stack([s, c])
    return P, th


def turning_centre(info: dict):
    """World xy of the vertical axis the robot turns about (None for a straight walk)."""
    w, v = info["yaw_rate"], info["speed"]
    if w == 0:
        return None
    return np.asarray(info["origin"], float) + np.array([0.0, v / w])


def move_pose(q, phi: float, a, b):
    """qpos with its root moved by the planar motion x -> b + R(phi) (x - a); joint coordinates unchanged."""
    q = np.array(q, float)
    q[:2] = np.asarray(b, float) + _rot2(phi) @ (q[:2] - np.asarray(a, float))
    q[3:7] = _quat_yaw(phi, q[3:7])
    return q


def move_velocity(qd, phi: float):
    """qvel under the same motion.  MuJoCo's free-joint linear velocity is in the world frame and rotates;
    its angular velocity is in the body frame and, like the joint rates, does not change."""
    qd = np.array(qd, float)
    qd[:2] = _rot2(phi) @ qd[:2]
    return qd


def derivatives(M: statics.Model, qs: np.ndarray, dt: float, transform):
    """Central-difference qvel and qacc of one periodic stride, unwrapped exactly with the stride motion
    ``transform = (phi, P0, PT)``: the pose after the last frame is D(first frame) and the pose before the
    first frame is D^-1(last frame)."""
    phi, P0, PT = transform
    m = M.m
    n = len(qs)
    ext = np.vstack([move_pose(qs[-1], -phi, PT, P0)[None], qs, move_pose(qs[0], phi, P0, PT)[None]])
    qd = np.zeros((n, m.nv))
    v = np.zeros(m.nv)
    for k in range(n):
        mujoco.mj_differentiatePos(m, v, 2 * dt, ext[k], ext[k + 2])
        qd[k] = v
    vext = np.vstack([move_velocity(qd[-1], -phi)[None], qd, move_velocity(qd[0], phi)[None]])
    qdd = (vext[2:] - vext[:-2]) / (2 * dt)
    return qd, qdd


# ================================================================================================
# Reference trajectory
# ================================================================================================
def foot_schedule(t, T: float, duty: float, lift: float, feet0: dict, origin, speed: float, yaw_rate: float,
                  phase_tol: float = PHASE_TOL) -> dict:
    """Foot targets per frame: position (n, 3), yaw (n,), swing flags and support sets.

    Phase convention and swing profiles as ``gait.generate`` (left foot lifts at phase 0, right at 0.5).
    The footprint a foot stands on is the home foot pose carried by the path frame to mid-stance.
    A frame within ``phase_tol`` (in stride fractions) of a lift-off counts as swing and one within
    ``phase_tol`` of a touchdown as stance, so frames that fall on an event are classified the same way for
    both feet and every stride period (``phase_tol=0`` is ``gait.generate``'s rule, ``u < 1 - duty``, under
    which rounding of t / T decides those frames)."""
    sw = 1.0 - duty
    t = np.asarray(t, float)

    def footprint(side, tm):
        P, th = path_frame(tm, speed, yaw_rate, origin)
        rel = feet0[side][:2] - np.asarray(origin, float)
        c, s = np.cos(th), np.sin(th)
        return P + np.column_stack([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]]), th

    ph = np.mod(t / T, 1.0)
    pos, yaw, swing = {}, {}, {}
    for side, off in (("left", 0.0), ("right", 0.5)):
        u = np.mod(ph - off, 1.0)
        if phase_tol > 0:
            u = np.where(u > 1.0 - phase_tol, u - 1.0, u)  # just before a lift-off by rounding: the lift-off
        is_sw = u < sw - phase_tol
        xy_st, th_st = footprint(side, t - (u - sw) * T + 0.5 * duty * T)
        t0 = t - u * T - 0.5 * duty * T  # mid-stance time of the footprint being left
        xy0, th0 = footprint(side, t0)
        xy1, th1 = footprint(side, t0 + T)
        a = gait._min_jerk(np.clip(u / sw, 0.0, 1.0))
        xy = np.where(is_sw[:, None], (1 - a)[:, None] * xy0 + a[:, None] * xy1, xy_st)
        th = np.where(is_sw, (1 - a) * th0 + a * th1, th_st)
        z = feet0[side][2] + np.where(is_sw, lift * np.sin(np.pi * u / sw) ** 2, 0.0)
        pos[side], yaw[side], swing[side] = np.column_stack([xy, z]), th, is_sw
    phases = [tuple(s for s in SIDES if not swing[s][k]) for k in range(len(t))]
    return dict(pos=pos, yaw=yaw, swing=swing, phases=phases)


def zmp_reference(t, T: float, duty: float, sched: dict, pad_c: dict) -> np.ndarray:
    """``gait.generate``'s ZMP reference with yawed feet: the stance pad centroid in single support, a linear
    hand-over from the foot about to lift to the foot that just landed in double support."""
    sw = 1.0 - duty
    zmp = np.zeros((len(t), 2))
    for k, tk in enumerate(t):
        ph = (tk / T) % 1.0
        sup = sched["phases"][k]

        def centroid(side):
            return sched["pos"][side][k, :2] + _rot2(sched["yaw"][side][k]) @ pad_c[side]

        if len(sup) == 1:
            zmp[k] = centroid(sup[0])
        else:
            if ph < 0.5:
                frm, to, s0, s1 = "right", "left", sw, 0.5
            else:
                frm, to, s0, s1 = "left", "right", 0.5 + sw, 1.0
            a = (ph - s0) / (s1 - s0)
            zmp[k] = (1 - a) * centroid(frm) + a * centroid(to)
    return zmp


def lipm_com(zref, P, theta, dt: float, w0: float, yaw_rate: float, speed: float, sway_gain: float = 1.0):
    """Periodic linear-inverted-pendulum CoM path for a ZMP reference that is periodic in the path frame.

    Returns the world CoM path (n, 2) and its path-frame coordinates u (n, 2); see the module docstring.
    ``sway_gain`` scales the lateral oscillation about its mean, as in ``gait.generate`` (1.0 nominal)."""
    n = len(zref)
    d = np.asarray(zref, float) - P
    c, s = np.cos(theta), np.sin(theta)
    q = (c * d[:, 0] + s * d[:, 1]) + 1j * (-s * d[:, 0] + c * d[:, 1])
    lam = 2 * np.pi * np.fft.fftfreq(n, d=dt)
    U = w0**2 * np.fft.fft(q)
    U[0] += 1j * yaw_rate * speed * n
    u = np.fft.ifft(U / ((lam + yaw_rate) ** 2 + w0**2))
    ux, uy = u.real, u.imag
    uy = uy.mean() + sway_gain * (uy - uy.mean())
    com = np.column_stack([P[:, 0] + c * ux - s * uy, P[:, 1] + s * ux + c * uy])
    return com, np.column_stack([ux, uy])


def _foot_errors(M: statics.Model, targets: dict) -> np.ndarray:
    err = []
    for side in SIDES:
        p, R = M.foot_pose(side)
        p_t, R_t = targets[side]
        err.append(np.linalg.norm(np.concatenate([p_t - p, statics._rot_err(R, R_t)])))
    return np.array(err)


def generate(M: statics.Model, q_home: np.ndarray, H: float, spec: gait.GaitSpec = sizing.NOMINAL,
             yaw_rate: float = 0.0, in_place: bool = False, com_tol: float = COM_TOL, com_iters: int = COM_ITERS,
             phase_tol: float = PHASE_TOL, unwrap: str = "exact") -> dict:
    """One periodic stride of a steady turn (positive yaw_rate turns left).  ``in_place`` steps at the spec's
    cadence with zero forward speed.  Returns a trajectory dict with the keys of ``gait.generate`` plus the
    stride motion ``transform``, the foot ``schedule``, per-frame foot pose errors and CoM tracking errors.
    ``com_tol`` and ``com_iters`` are the CoM-tracking tolerance (m) and pass limit per frame;
    ``com_tol=2e-4, com_iters=4, phase_tol=0, unwrap="gait"`` reproduces ``gait.generate`` at yaw_rate 0
    (``unwrap="gait"`` differences the stride boundary with ``gait.derivatives``' approximate root-x unwrap; it
    is meaningful only for a straight walk and is kept for that validation)."""
    m = M.m
    base = spec.describe(H)
    T = base["period"]
    v = 0.0 if in_place else base["speed"]
    n = spec.n
    dt = T / n
    t = np.arange(n) * dt
    M.forward(q_home)
    feet0 = {s: M.foot_pose(s)[0].copy() for s in SIDES}
    pad_c = {s: statics._polygon_centroid(M.pad_polygon(s)) - feet0[s][:2] for s in SIDES}
    origin = 0.5 * sum(M.d.xanchor[statics.jid(m, f"{s}_hip_pitch")][:2] for s in SIDES)
    com0 = M.com()
    root0 = q_home[:3].copy()
    sw = 1.0 - spec.duty
    lift = spec.lift_frac * H
    P, theta = path_frame(t, v, yaw_rate, origin)
    sched = foot_schedule(t, T, spec.duty, lift, feet0, origin, v, yaw_rate, phase_tol=phase_tol)
    zmp = zmp_reference(t, T, spec.duty, sched, pad_c)
    z_c = com0[2] - spec.crouch * H
    w0 = math.sqrt(G / z_c)
    neck = m.jnt_qposadr[statics.jid(m, "neck_pitch")]
    tail = m.jnt_qposadr[statics.jid(m, "tail_yaw")]
    w = 2 * np.pi / T
    off = com0[:2] - root0[:2]
    PT = path_frame([T], v, yaw_rate, origin)[0][0]
    transform = (yaw_rate * T, origin.copy(), PT)
    R_t = {s: [_rotz(a) for a in sched["yaw"][s]] for s in SIDES}

    def solve_frames(zref):
        com_t, u = lipm_com(zref, P, theta, dt, w0, yaw_rate, v, spec.sway_gain)
        qs = np.zeros((n, m.nq))
        q = q_home.copy()
        foot_err = np.zeros((n, 2))
        com_err = np.zeros(n)
        for k in range(n):
            targets = {s: (sched["pos"][s][k], R_t[s][k]) for s in SIDES}
            q[2] = root0[2] - spec.crouch * H
            q[3:7] = _quat_yaw(theta[k], q_home[3:7])
            q[neck] = spec.head_amp * np.sin(2 * w * t[k])
            q[tail] = spec.tail_amp * np.sin(w * t[k] + np.pi / 2)
            q[:2] = com_t[k] - _rot2(theta[k]) @ off
            for _ in range(com_iters):  # the CoM correction of gait.generate
                q, _ok = statics.leg_ik(M, q, targets, iters=120)
                err = com_t[k] - M.com()[:2]
                if np.linalg.norm(err) < com_tol:
                    break
                q[0] += err[0]
                q[1] += err[1]
            M.forward(q)  # judge the pose actually stored
            foot_err[k] = _foot_errors(M, targets)
            com_err[k] = np.linalg.norm(com_t[k] - M.com()[:2])
            qs[k] = q
        return qs, com_t, u, foot_err, com_err

    if unwrap not in ("exact", "gait") or (unwrap == "gait" and yaw_rate != 0.0):
        raise ValueError("unwrap='gait' (gait.derivatives' approximate unwrap) applies to a straight walk only")
    zref = zmp.copy()
    history = []
    for it in range(spec.zmp_iters + 1):
        qs, com_t, u, foot_err, com_err = solve_frames(zref)
        qd, qdd = derivatives(M, qs, dt, transform) if unwrap == "exact" else gait.derivatives(M, qs, dt)
        zmb = np.array([statics.total_zmp(M, qs[k], qd[k], qdd[k]) for k in range(n)])
        e = zmb - zmp
        history.append(float(np.nanmax(np.linalg.norm(e, axis=1))))
        if it == spec.zmp_iters:
            break
        zref = zref - spec.zmp_gain * np.nan_to_num(e)
    if unwrap != "exact":  # the same poses measured with the exact unwrap (what inverse_dynamics uses)
        qd, qdd = derivatives(M, qs, dt, transform)
        zmb = np.array([statics.total_zmp(M, qs[k], qd[k], qdd[k]) for k in range(n)])
    e_exact = np.linalg.norm(zmb - zmp, axis=1)
    fails = [int(k) for k in np.flatnonzero(foot_err.max(axis=1) > IK_TOL)]
    info = dict(base)
    info.update(speed=v, stride=v * T, period=T, cadence_hz=2.0 / T, yaw_rate=float(yaw_rate), in_place=bool(in_place),
                radius=(v / abs(yaw_rate) if (yaw_rate and v) else (0.0 if yaw_rate else None)),
                heading_per_stride=float(yaw_rate * T), origin=origin.tolist(),
                foot_yaw_vs_trunk_amp=float(abs(yaw_rate) * spec.duty * T / 2),
                swing_yaw_rate_peak=float(1.875 * abs(yaw_rate) / sw),
                sway_amp=float(0.5 * (u[:, 1].max() - u[:, 1].min())), lean_inward=float(np.sign(yaw_rate) * u[:, 1].mean()),
                lift=lift, omega=w0, zmp_error_history_m=history, zmp_error_exact_final_m=float(np.nanmax(e_exact)),
                zmp_error_exact_final_frame=int(np.nanargmax(e_exact)), com_tol=com_tol, com_iters=com_iters,
                phase_tol=phase_tol, unwrap=unwrap)
    return dict(t=t, q=qs, phases=sched["phases"], dt=dt, zmp_ref=zmp, com_ref=com_t, zmp_mb=zmb, path=P, heading=theta,
                transform=transform, schedule=sched, info=info, ik_ok=not fails, ik_fail_frames=fails,
                foot_error=foot_err, com_error=com_err, qd=qd, qdd=qdd)


# ================================================================================================
# Pads and friction
# ================================================================================================
def pad_samples(M: statics.Model, side: str, k: int = PAD_GRID):
    """Area-weighted sample points (world xy) of the bottom faces of one foot's floor pads at the current pose."""
    g_ = ((np.arange(k) + 0.5) / k) * 2 - 1
    pts, wts = [], []
    for g in M.pads[side]:
        R = M.d.geom_xmat[g].reshape(3, 3)
        h = M.m.geom_size[g]
        gx, gy = np.meshgrid(g_ * h[0], g_ * h[1], indexing="ij")
        local = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, -h[2])])
        pts.append((M.d.geom_xpos[g] + local @ R.T)[:, :2])
        wts.append(np.full(gx.size, 4 * h[0] * h[1] / gx.size))
    return np.vstack(pts), np.concatenate(wts)


def uniform_radius(pts, w) -> float:
    """Torsional friction radius of a flat contact under uniform pressure: mean distance from the area
    centroid, so the friction moment in pure spin about the centroid is mu F_n r (2R/3 for a disc)."""
    pts, w = np.asarray(pts, float), np.asarray(w, float)
    c = (w[:, None] * pts).sum(0) / w.sum()
    return float((w * np.linalg.norm(pts - c, axis=1)).sum() / w.sum())


def pressure_radius(pts, w, cop, iters: int = 50) -> float:
    """Torsional friction radius about the CoP for a rigid foot on uniform elastic pads: pressure linear in
    position, zero in tension (no-tension active set), resultant at ``cop``.  0 if no such field exists."""
    pts, w = np.asarray(pts, float), np.asarray(w, float)
    if not np.all(np.isfinite(cop)):
        return float("nan")
    hull = pts[ConvexHull(pts).vertices]
    if statics.support_margin(np.asarray(cop, float), hull) < 0:
        return 0.0  # no compressive pressure field has its resultant outside the contact
    d = pts - np.asarray(cop, float)
    active = np.ones(len(d), bool)
    p = np.ones(len(d))
    for _ in range(iters):
        wa = w * active
        s = (wa[:, None] * d).sum(0)
        J = np.einsum("i,ij,ik->jk", wa, d, d)
        if active.sum() < 3 or np.linalg.cond(J) > 1e12:
            return 0.0
        b = -np.linalg.solve(J, s)  # p = 1 + b.d puts the resultant of the active set at the CoP
        p = 1.0 + d @ b
        new = p > 0
        if np.array_equal(new, active):
            break
        active = new
    p = np.where(active, np.clip(p, 0.0, None), 0.0)
    W = float((w * p).sum())
    if W <= 0:
        return 0.0
    return float((w * p * np.linalg.norm(d, axis=1)).sum() / W)


# ================================================================================================
# Contact wrenches: exact double-support split and the double-support friction requirement
# ================================================================================================
def _base_problem(M: statics.Model, q, qd, qdd, support):
    """The base rows of the equations of motion as ``statics.contact_torques`` forms them: generalised force
    ``rhs`` = S^T tau + J^T w, the stacked foot Jacobians ``J`` and the base equations A w = b."""
    m, d = M.m, M.d
    d.qpos[:] = q
    d.qvel[:] = qd
    mujoco.mj_forward(m, d)
    Mfull = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, Mfull)
    rhs = Mfull @ qdd + d.qfrc_bias
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    Js = []
    for side in support:
        mujoco.mj_jacBody(m, d, jacp, jacr, M.foot[side])
        Js.append(np.vstack([jacp, jacr]).copy())
    J = np.vstack(Js)
    return rhs, J, J[:, :6].T, rhs[:6]


def _qp_matrices(M: statics.Model, J, rhs, support, mu: float = 0.8):
    """``statics._double_support_qp``'s problem as matrices, min 1/2 w'Hw + g'w subject to A w = b and G w >= h:
    normal force >= 0, friction pyramid mu, CoP inside the pad polygon's bounding box, at the current pose.

    The pyramid and the box are taken in each foot's own axes (its yaw psi), where ``statics`` takes them in
    world axes.  For a foot pointing along x (every straight-walk frame) the two are identical; in a turn
    the world-axis box of a yawed pad is larger than the pad's own and turns with the heading, which made
    the split depend on the absolute heading (so mirrored turns and successive strides differed)."""
    act = M.act_dofs
    Ja = J[:, act].T
    nw = J.shape[0]
    Hq = Ja.T @ Ja + 1e-6 * np.eye(nw)
    g = -Ja.T @ rhs[act]
    rows = []
    for k, side in enumerate(support):
        o = 6 * k
        p, R = M.foot_pose(side)
        psi = math.atan2(R[1, 0], R[0, 0])
        c, s_ = math.cos(psi), math.sin(psi)
        hgt = p[2]
        poly = (M.pad_polygon(side) - p[:2]) @ _rot2(psi)  # pad polygon in the foot's axes
        (xmin, ymin), (xmax, ymax) = poly.min(0), poly.max(0)
        fx, fy, fz = np.zeros(nw), np.zeros(nw), np.zeros(nw)
        fx[o + 0], fy[o + 1], fz[o + 2] = 1.0, 1.0, 1.0
        # fz (cop - p) in world axes: X = -(m_y + h f_x), Y = m_x - h f_y; then in the foot's axes
        X, Y = np.zeros(nw), np.zeros(nw)
        X[o + 4], X[o + 0] = -1.0, -hgt
        Y[o + 3], Y[o + 1] = 1.0, -hgt
        fxl, fyl = c * fx + s_ * fy, -s_ * fx + c * fy
        Xl, Yl = c * X + s_ * Y, -s_ * X + c * Y
        rows += [fz, mu * fz - fxl, mu * fz + fxl, mu * fz - fyl, mu * fz + fyl,
                 Xl - xmin * fz, xmax * fz - Xl, Yl - ymin * fz, ymax * fz - Yl]
    G = np.array(rows)
    return Hq, g, G, np.zeros(len(G))


def qp_exact(Hq, g, A, b, G, h):
    """Exact solution of the strictly convex QP min 1/2 w'Hw + g'w s.t. A w = b, G w >= h.

    The equalities are removed with a null-space basis, the objective is completed to a least-distance problem
    and that is solved by non-negative least squares (Lawson and Hanson's LDP algorithm, a finite active-set
    method).  Returns (w, ok)."""
    from scipy.linalg import cho_factor, cho_solve, solve_triangular
    from scipy.optimize import nnls

    U, S, Vt = np.linalg.svd(A)
    r = int(np.sum(S > S[0] * 1e-12))
    w_p = Vt[:r].T @ ((U[:, :r].T @ b) / S[:r])
    N = Vt[r:].T
    Q = N.T @ Hq @ N
    c = N.T @ (Hq @ w_p + g)
    D, e = G @ N, h - G @ w_p
    L = np.linalg.cholesky(Q)
    z0 = cho_solve(cho_factor(Q), c)  # Q^-1 c
    Gt = solve_triangular(L, D.T, lower=True).T  # D L^-T
    ht = e + D @ z0
    E = np.vstack([Gt.T, ht[None, :]])
    f = np.zeros(E.shape[0])
    f[-1] = 1.0
    try:
        y, _ = nnls(E, f, maxiter=50 * E.shape[1])
    except RuntimeError:  # no convergence: fall back to SLSQP from the equality-only optimum
        return _qp_slsqp(Hq, g, A, b, G, h, w_p + N @ (-z0))
    res = E @ y - f
    if np.linalg.norm(res) < 1e-12 or res[-1] >= 0:
        return w_p + N @ (-z0), False  # incompatible constraints: the unconstrained optimum
    u = -res[:-1] / res[-1]
    z = solve_triangular(L.T, u, lower=False) - z0
    return w_p + N @ z, True


def _qp_slsqp(Hq, g, A, b, G, h, w0):
    """SLSQP on the same QP (the fallback of ``qp_exact``); returns (w, ok)."""
    from scipy.optimize import minimize

    res = minimize(lambda x: 0.5 * x @ Hq @ x + g @ x, w0, jac=lambda x: Hq @ x + g, method="SLSQP",
                   constraints=[dict(type="eq", fun=lambda x: A @ x - b, jac=lambda x: A),
                                dict(type="ineq", fun=lambda x: G @ x - h, jac=lambda x: G)],
                   options=dict(maxiter=300, ftol=1e-12))
    return (res.x, True) if res.success else (w0, False)


def contact_torques(M: statics.Model, q, qd, qdd, support, split: str = "exact"):
    """Joint torques and foot wrenches as ``statics.contact_torques``: unique in single support; in double
    support the minimum-joint-effort split with non-negative normal forces, a friction pyramid and the CoP
    inside each pad's box.  ``split="exact"`` takes the pyramid and the box in each foot's own axes
    (``_qp_matrices``) and solves the QP exactly (``qp_exact``) wherever the equality-only optimum violates
    an inequality (where ``statics`` runs SLSQP, and keeps the infeasible equality optimum if SLSQP fails).
    In a straight walk both give the same split.  ``split="statics"`` calls ``statics.contact_torques``.
    Returns (tau, wrenches, how), ``how`` naming how the split was found."""
    if split == "statics" or len(support) == 1:
        tau, wr = statics.contact_torques(M, q, qd, qdd, support=support)
        return tau, wr, ("unique" if len(support) == 1 else "statics")
    rhs, J, A, b = _base_problem(M, q, qd, qdd, support)
    Hq, g, G, h = _qp_matrices(M, J, rhs, support)
    nw = J.shape[0]
    K = np.block([[Hq, A.T], [A, np.zeros((6, 6))]])
    w = np.linalg.solve(K, np.concatenate([-g, b]))[:nw]  # statics' warm start
    how = "equality_optimum"
    if not np.all(G @ w >= h - 1e-9):
        w, ok = qp_exact(Hq, g, A, b, G, h)
        how = "exact_qp" if ok else "infeasible"
    tau = rhs[M.act_dofs] - J[:, M.act_dofs].T @ w
    return tau, {side: w[6 * k : 6 * k + 6] for k, side in enumerate(support)}, how


def _fibonacci_directions(n: int) -> np.ndarray:
    """n quasi-uniform unit vectors on the sphere (golden-angle spiral)."""
    k = np.arange(n) + 0.5
    z = 1.0 - 2.0 * k / n
    phi = np.pi * (3.0 - np.sqrt(5.0)) * k
    s = np.sqrt(1.0 - z * z)
    return np.column_stack([s * np.cos(phi), s * np.sin(phi), z])


FRICTION_DIRS = _fibonacci_directions(DS_DIRECTIONS)


def ds_min_mu(M: statics.Model, q, qd, qdd, r_eff: dict, centroid: dict, max_iter: int = 30) -> dict:
    """Double support: the smallest friction coefficient for which SOME split of the (unique) total contact
    wrench is feasible.  Per foot: F_n >= 0, CoP inside the pad hull, and the limit surface of ``mu_req``,
    ||(F_x, F_y, M_z / r_eff)|| <= mu F_n, with M_z about the pad-area centroid (which keeps the problem
    linear).  The cone is replaced by ``DS_DIRECTIONS`` tangent planes (an outer polyhedron) and the
    quasi-convex min-max ratio is solved by the Dinkelbach-type iteration of Crouzeix, Ferland and Schaible
    (one LP per step).  Returned: ``mu`` = the exact cone ratio at the final split (an achievable value, so
    an upper bound), ``mu_lower`` = the polyhedral optimum (a lower bound, None if the iteration did not
    converge) and the per-foot normal forces of that split.  The single-support counterpart is
    ``mu_req_centroid`` (``inverse_dynamics``)."""
    from scipy.optimize import linprog

    sup = statics.SIDES
    rhs, J, A, b = _base_problem(M, q, qd, qdd, sup)
    nw = 12
    feet = []
    for k, side in enumerate(sup):
        o = 6 * k
        p, _ = M.foot_pose(side)
        cen = np.asarray(centroid[side], float)
        fz = np.zeros(nw)
        fz[o + 2] = 1.0
        V = np.zeros((3, nw))  # v = (fx, fy, mz_centroid / r)
        V[0, o + 0] = V[1, o + 1] = 1.0
        V[2, o + 5] = 1.0
        V[2, o + 1] += p[0] - cen[0]
        V[2, o + 0] -= p[1] - cen[1]
        V[2] /= r_eff[side]
        # CoP inside the pad hull: fz cop = fz pf + (-(my + pz fx), mx - pz fy)
        cx, cy = np.zeros(nw), np.zeros(nw)
        cx[o + 2], cx[o + 4], cx[o + 0] = p[0], -1.0, -p[2]
        cy[o + 2], cy[o + 3], cy[o + 1] = p[1], 1.0, -p[2]
        poly = M.pad_polygon(side)
        hull = []
        for i in range(len(poly)):
            a0, a1 = poly[i], poly[(i + 1) % len(poly)]
            ed = a1 - a0
            nin = np.array([-ed[1], ed[0]]) / np.linalg.norm(ed)
            row = -(nin[0] * cx + nin[1] * cy)
            row = row + (nin @ a0) * fz
            hull.append(row)  # row . w <= 0
        feet.append(dict(fz=fz, P=FRICTION_DIRS @ V, V=V, hull=np.array(hull)))
    fixed_ub = np.vstack([f["hull"] for f in feet] + [-f["fz"][None, :] for f in feet])

    tiny = 1e-6 * (1.0 + abs(b[2]))  # an unloaded foot (LP tolerance), N

    def ratios(w):
        out_p, out_e = [], []
        for f in feet:
            n_, v = float(f["fz"] @ w), f["V"] @ w
            if n_ <= tiny:
                out_p.append(0.0 if np.linalg.norm(v) <= 10 * tiny else np.inf)
                out_e.append(out_p[-1])
            else:
                out_p.append(float(np.max(f["P"] @ w)) / n_)
                out_e.append(float(np.linalg.norm(v)) / n_)
        return out_p, out_e

    # a feasible split to start from (CoP inside both hulls, F_n >= 0): minimise the larger polyhedral ratio
    # against a large mu; infeasible means no split puts both CoPs inside the pads
    mu_k, scale = 10.0, np.ones(2)
    w, converged = None, False
    for it in range(max_iter):
        # variables (w, t): minimise t subject to (P_i w - mu_k fz_i) / scale_i <= t for both feet
        rows = [np.hstack([f["P"] - mu_k * f["fz"][None, :], -scale[i] * np.ones((len(f["P"]), 1))])
                for i, f in enumerate(feet)]
        A_ub = np.vstack(rows + [np.hstack([fixed_ub, np.zeros((len(fixed_ub), 1))])])
        b_ub = np.zeros(len(A_ub))
        c = np.zeros(nw + 1)
        c[-1] = 1.0
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=np.hstack([A, np.zeros((6, 1))]), b_eq=b,
                      bounds=[(None, None)] * (nw + 1), method="highs")
        if res.status != 0:
            return dict(mu=None if w is None else float(max(ratios(w)[1])), mu_lower=None, feasible=w is not None,
                        iterations=it)
        w = res.x[:nw]
        rp, _ = ratios(w)
        new = max(rp)
        scale = np.array([max(float(f["fz"] @ w), 1e-3 * abs(b[2])) for f in feet])
        if mu_k - new <= 1e-7 * max(1.0, new) or res.x[-1] >= -1e-12:
            mu_k, converged = min(mu_k, new), True
            break
        mu_k = new
    rp, re = ratios(w)
    return dict(mu=float(max(re)), mu_lower=float(mu_k) if converged else None, feasible=True, iterations=it + 1,
                fn={s: float(f["fz"] @ w) for s, f in zip(sup, feet, strict=True)},
                mu_by_foot={s: float(v) for s, v in zip(sup, re, strict=True)})


def inverse_dynamics(M: statics.Model, traj: dict, mu: float = MU, split: str = "exact", ds_friction: bool = True) -> dict:
    """Joint torques and per-foot contact quantities for every frame (see module docstring).

    Keys shared with ``gait.inverse_dynamics``: tau, omega, cop_margin, friction, normal, qd, qdd.  Added:
    ``feet[side]`` per-frame arrays (fn, ft, mz, r_cop, util, util_cop, mu_req, mu_req_cop, mu_req_centroid,
    support size), ``r_eff`` per side, the total contact force (world), the total vertical moment about the
    turning centre, the path-frame CoM, how each double-support split was solved (``split_how``) and, with
    ``ds_friction``, the double-support friction requirement per frame (``ds``, see ``ds_min_mu``).
    ``split`` selects the double-support split (``contact_torques``)."""
    qs, dt, phases = traj["q"], traj["dt"], traj["phases"]
    if traj["info"].get("unwrap", "exact") == "gait":
        qd, qdd = gait.derivatives(M, qs, dt)
    else:
        qd, qdd = derivatives(M, qs, dt, traj["transform"])
    n = len(qs)
    tau = np.zeros((n, len(M.act_dofs)))
    cop_margin = np.full((n, 2), np.nan)
    friction = np.zeros(n)
    normal = np.zeros((n, 2))
    keys = ("fn", "ft", "mz", "r_cop", "util", "util_cop", "mu_req", "mu_req_cop", "mu_req_centroid", "margin")
    feet = {s: {k: np.full(n, np.nan) for k in keys} | dict(nsup=np.zeros(n, int)) for s in SIDES}
    M.forward(qs[0])
    r_eff = {s: uniform_radius(*pad_samples(M, s)) for s in SIDES}
    centre = turning_centre(traj["info"])
    O3 = None if centre is None else np.array([centre[0], centre[1], 0.0])
    total_f = np.zeros((n, 3))
    total_mO = np.full(n, np.nan)
    u_com = np.zeros((n, 2))
    how = []
    ds = dict(min_mu=np.full(n, np.nan), min_mu_lower=np.full(n, np.nan), fn_left=np.full(n, np.nan),
              fn_right=np.full(n, np.nan), feasible=np.ones(n, bool))
    for k in range(n):
        sup = phases[k] if phases[k] else SIDES
        tk, wr, hw = contact_torques(M, qs[k], qd[k], qdd[k], sup, split)
        how.append(hw)
        tau[k] = tk
        c_ = M.com()[:2] - traj["path"][k]
        u_com[k] = _rot2(-traj["heading"][k]) @ c_
        mO = 0.0
        cen = {}
        for j, side in enumerate(SIDES):
            pts, wts = pad_samples(M, side)
            cen[side] = (wts[:, None] * pts).sum(0) / wts.sum()  # pad-area centroid (world xy)
            if side not in wr:
                continue
            f, mom = wr[side][:3], wr[side][3:]
            p, _ = M.foot_pose(side)
            total_f[k] += f
            if O3 is not None:
                mO += mom[2] + np.cross(p - O3, f)[2]
            F = feet[side]
            F["nsup"][k] = len(sup)
            fn, ft = float(f[2]), float(np.linalg.norm(f[:2]))
            F["fn"][k], F["ft"][k] = fn, ft
            normal[k, j] = fn
            friction[k] = max(friction[k], ft / max(fn, 1e-6))
            c = statics.cop(M, side, wr[side])
            if not np.all(np.isfinite(c)):
                continue
            mz = float(mom[2] + np.cross(p - np.array([c[0], c[1], 0.0]), f)[2])
            mz_cen = float(mom[2] + (p[0] - cen[side][0]) * f[1] - (p[1] - cen[side][1]) * f[0])
            margin = statics.support_margin(c, M.pad_polygon(side))
            r_cop = pressure_radius(pts, wts, c) if margin >= 0 else 0.0
            F["mz"][k], F["margin"][k], F["r_cop"][k] = mz, margin, r_cop
            cop_margin[k, j] = margin
            F["util"][k] = abs(mz) / (mu * fn * r_eff[side])
            F["util_cop"][k] = abs(mz) / (mu * fn * r_cop) if r_cop > 0 else np.inf
            F["mu_req"][k] = math.hypot(ft / fn, mz / (fn * r_eff[side]))
            F["mu_req_cop"][k] = math.hypot(ft / fn, mz / (fn * r_cop)) if r_cop > 0 else np.inf
            F["mu_req_centroid"][k] = math.hypot(ft / fn, mz_cen / (fn * r_eff[side]))
        if O3 is not None:
            total_mO[k] = mO
        if ds_friction and len(sup) == 2:
            r = ds_min_mu(M, qs[k], qd[k], qdd[k], r_eff, cen)
            ds["feasible"][k] = bool(r["feasible"])
            if r["mu"] is not None:
                ds["min_mu"][k] = r["mu"]
            if r.get("mu_lower") is not None:
                ds["min_mu_lower"][k] = r["mu_lower"]
            if r.get("fn") is not None:
                ds["fn_left"][k], ds["fn_right"][k] = r["fn"]["left"], r["fn"]["right"]
    return dict(tau=tau, omega=qd[:, M.act_dofs], qd=qd, qdd=qdd, cop_margin=cop_margin, friction=friction,
                normal=normal, feet=feet, r_eff=r_eff, mu=mu, total_force=total_f, moment_about_centre=total_mO,
                u_com=u_com, split=split, split_how=how, ds=ds if ds_friction else None)


def _amax(v):
    v = np.abs(np.asarray(v, float))
    v = v[np.isfinite(v)]
    return float(v.max()) if len(v) else None


def _amin(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(v.min()) if len(v) else None


def _foot_stats(F: dict, idx: np.ndarray, phases) -> dict:
    """Friction statistics of one foot over the frames ``idx``.  Frames whose CoP lies outside the pad
    polygon have no pressure field (r_cop = 0); they are counted and left out of the r_cop maxima."""
    if len(idx) == 0:
        return dict(frames=0)
    util = np.where(np.isfinite(F["util"][idx]), F["util"][idx], -1.0)
    kmax = int(idx[int(np.argmax(util))])
    return dict(frames=int(len(idx)), fn_min=_amin(F["fn"][idx]), ft_over_fn_max=_amax(F["ft"][idx] / F["fn"][idx]),
                mz_abs_max=_amax(F["mz"][idx]), util_max=_amax(F["util"][idx]), util_cop_max=_amax(F["util_cop"][idx]),
                mu_req_max=_amax(F["mu_req"][idx]), mu_req_cop_max=_amax(F["mu_req_cop"][idx]),
                mu_req_centroid_max=_amax(F["mu_req_centroid"][idx]),
                r_cop_min=_amin(np.where(F["r_cop"][idx] > 0, F["r_cop"][idx], np.nan)),
                frames_cop_outside=int(np.sum(F["margin"][idx] < 0)), cop_margin_min=_amin(F["margin"][idx]),
                util_max_frame=kmax, util_max_support=list(phases[kmax]))


def summarize(M: statics.Model, traj: dict, idr: dict) -> dict:
    """``gait.summarize`` plus friction, joint-range use and physical checks for one trajectory."""
    out = gait.summarize(M, traj, idr)
    out["ik_ok"] = bool(traj["ik_ok"])
    out["ik_fail_frames"] = len(traj["ik_fail_frames"])
    out["ik_fail_frame_list"] = list(traj["ik_fail_frames"])
    out["foot_error_max"] = float(traj["foot_error"].max())
    out["com_error_max"] = float(traj["com_error"].max())
    weight = M.mass() * G
    phases = traj["phases"]
    feet = {}
    for side in SIDES:
        F = idr["feet"][side]
        single = np.array([k for k, p in enumerate(phases) if p == (side,)], int)
        double = np.array([k for k, p in enumerate(phases) if len(p) == 2 and F["fn"][k] >= FN_MIN_FRAC * weight], int)
        # double support: the minimum-effort split's per-foot values are an allocation, not a requirement
        # (diagnostic only; the requirement is out["double_support"])
        feet[side] = dict(r_eff=idr["r_eff"][side], single_support=_foot_stats(F, single, phases),
                          double_support_qp_allocation=_foot_stats(F, double, phases))
    out["feet"] = feet
    out["mu_assumed"] = idr["mu"]
    dsk = [k for k, p in enumerate(phases) if len(p) == 2]
    how = idr.get("split_how", [])
    out["split"] = dict(method=idr.get("split"), double_support_frames=len(dsk),
                        solved=dict((h, sum(1 for x in how if x == h)) for h in sorted(set(how)) if h != "unique"))
    if idr.get("ds") is not None and dsk:
        D = idr["ds"]
        mm = np.where(np.isfinite(D["min_mu"][dsk]), D["min_mu"][dsk], -1.0)
        kmax = int(dsk[int(np.argmax(mm))])
        out["double_support"] = dict(
            frames=len(dsk), min_mu_max=_amax(D["min_mu"][dsk]), min_mu_lower_max=_amax(D["min_mu_lower"][dsk]),
            min_mu_max_frame=kmax, infeasible_frames=int(np.sum(~D["feasible"][dsk])),
            note="smallest mu for which some split of the total contact wrench keeps both CoPs on their pads with "
                 "||(F_t, M_z/r_eff)|| <= mu F_n per foot (M_z about the pad-area centroid); the counterpart of "
                 "the single-support mu_req")
    m = M.m
    rng = {}
    for side in SIDES:
        for jn, qi in zip(statics.LEG_JOINTS, M.leg_q[side], strict=True):
            lo, hi = m.jnt_range[statics.jid(m, f"{side}_{jn}")]
            qv = traj["q"][:, qi]
            rng[f"{side}_{jn}"] = float(max(np.max(qv) / hi, np.min(qv) / lo))
    out["range_use"] = rng  # largest fraction of the joint limit reached (1.0 = at the limit)
    info = traj["info"]
    th = traj["heading"]
    Fp = np.array([_rot2(-a) @ f[:2] for a, f in zip(th, idr["total_force"], strict=True)]).mean(0)
    ub = idr["u_com"].mean(0)
    wz, v = info["yaw_rate"], info["speed"]
    mass = M.mass()
    expect = mass * np.array([-wz * wz * ub[0], wz * v - wz * wz * ub[1]])
    qd = idr["qd"]
    yaw_rate = []
    for k in range(len(qd)):
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, traj["q"][k, 3:7])
        yaw_rate.append((R.reshape(3, 3) @ qd[k, 3:6])[2])
    mO = idr["moment_about_centre"]
    out["checks"] = dict(
        vertical_force_mean_over_weight=float(idr["total_force"][:, 2].mean() / weight),
        path_frame_force_mean_n=Fp.tolist(), path_frame_force_expected_n=expect.tolist(),
        trunk_yaw_rate_mean=float(np.mean(yaw_rate)),
        moment_about_turn_axis_mean_nm=(float(np.nanmean(mO)) if np.isfinite(mO).any() else None),
        moment_about_turn_axis_rms_nm=(float(np.sqrt(np.nanmean(mO**2))) if np.isfinite(mO).any() else None))
    return out


# ================================================================================================
# Requirements and comparison with the selected drives
# ================================================================================================
def motor_space_by_side(names, tau, omega, layout: str) -> dict:
    """Per group and side, the motor torque and speed series; the same mapping as ``sizing._motor_space``
    (ankle parallel linkage with unit crank ratios for tandem and side layouts), not stacked."""
    idx = {nm: i for i, nm in enumerate(names)}
    out = {g: {} for g in sizing.GROUPS}
    for g, j in sizing.JOINT_OF.items():
        for s in SIDES:
            out[g][s] = (tau[:, idx[f"{s}_{j}"]], omega[:, idx[f"{s}_{j}"]])
    for s in SIDES:
        tp, tr = tau[:, idx[f"{s}_ankle_pitch"]], tau[:, idx[f"{s}_ankle_roll"]]
        wp, wr = omega[:, idx[f"{s}_ankle_pitch"]], omega[:, idx[f"{s}_ankle_roll"]]
        if layout in ("tandem", "side"):
            out["ankle_a"][s], out["ankle_b"][s] = (0.5 * (tp + tr), wp + wr), (0.5 * (tp - tr), wp - wr)
        else:
            out["ankle_a"][s], out["ankle_b"][s] = (tp, wp), (tr, wr)
    out["neck"]["-"] = (tau[:, idx["neck_pitch"]], omega[:, idx["neck_pitch"]])
    out["tail"]["-"] = (tau[:, idx["tail_yaw"]], omega[:, idx["tail_yaw"]])
    return out


def group_loads(names, runs: dict, layout: str) -> dict:
    """Per group over several runs (e.g. left and right turns): RMS of each motor in each run, the worst of
    those, the stacked RMS (sizing's convention), and the peak torque and speed over all motors and runs."""
    out = {}
    ms = {r: motor_space_by_side(names, idr["tau"], idr["omega"], layout) for r, idr in runs.items()}
    for g in sizing.GROUPS:
        rms, sq, peak, speed = {}, [], 0.0, 0.0
        for r in runs:
            for s, (t_, w_) in ms[r][g].items():
                rms[f"{r}:{s}"] = float(np.sqrt(np.mean(t_**2)))
                sq.append(t_**2)
                peak = max(peak, float(np.max(np.abs(t_))))
                speed = max(speed, float(np.max(np.abs(w_))))
        worst = max(rms, key=rms.get)
        out[g] = dict(rms=rms[worst], rms_motor=worst, rms_stacked=float(np.sqrt(np.mean(np.concatenate(sq)))),
                      rms_per_motor=rms, peak=peak, speed=speed)
    return out


def requirement(loads: dict, hold: float) -> dict:
    """``sizing.requirements``' rule applied to one group's loads and its single-support hold."""
    peak = max(loads["peak"], hold)
    return dict(rms=loads["rms"], hold=hold, peak=peak, speed=loads["speed"], cont=max(loads["rms"], hold) * sizing.CONT_MARGIN,
                peak_req=peak * sizing.PEAK_MARGIN, speed_req=loads["speed"] * sizing.SPEED_MARGIN)


def _ratio(a: float, b: float):
    return None if b < ZERO_REQ else a / b


def compare(req: dict, straight: dict, act, bus) -> dict:
    """One group's turning requirement against its drive's planning limits and its straight-walk requirement.

    ``near_increase_flag`` lists metrics whose ratio to the straight walk lies within ``NEAR_FLAG_BAND`` below
    the 10 % flag, and ``near_limit`` those within the band below their planning limit: closer than the
    method's measured sensitivity, so the verdict there should not be read as a margin."""
    lim = dict(cont=act.sustained, peak_req=act.peak, speed_req=act.speed_at(bus))
    over = [k for k in lim if req[k] > lim[k]]
    vs = {k: _ratio(req[k], straight[k]) for k in lim}
    up = [k for k, r in vs.items() if (r is None and req[k] >= ZERO_REQ) or (r is not None and r > 1 + INCREASE_FLAG)]
    near = [k for k, r in vs.items() if r is not None and k not in up and r > 1 + INCREASE_FLAG - NEAR_FLAG_BAND]
    to_lim = {k: req[k] / max(lim[k], 1e-9) for k in lim}
    near_lim = [k for k, r in to_lim.items() if k not in over and r > 1.0 - NEAR_FLAG_BAND]
    return dict(actuator=act.id, bus=bus, limits=lim, required=dict((k, req[k]) for k in lim), ratio_to_limit=to_lim,
                meets=sizing.meets(act, req, bus), over_limit=over, near_limit=near_lim,
                straight_required=dict((k, straight[k]) for k in lim), ratio_to_straight=vs, increase_over_10pct=up,
                near_increase_flag=near)


def run_groups(names, idr: dict, layout: str) -> dict:
    """Per joint group in one run: the RMS torque of the more loaded motor (``rms``), the stacked RMS of both
    legs (``sizing``'s convention), peak torque and peak speed.  ``envelope.turning_rms_from`` reads ``rms``."""
    gl = group_loads(names, {"run": idr}, layout)
    return {g: dict(rms=v["rms"], rms_stacked=v["rms_stacked"], peak=v["peak"], speed=v["speed"]) for g, v in gl.items()}


# ================================================================================================
# Self-collision along the turns (clearance.screen)
# ================================================================================================
def _range_margins(cl_summary: dict, M: statics.Model, trajs: dict, joints=("hip_yaw", "hip_roll")) -> dict:
    """Per leg joint: the excursion over ``trajs`` against the usable range of ``clearance``'s single-joint
    sweeps (first new interference with the trunk or the same leg, else the joint limit), the rule of
    ``clearance._range_margins``."""
    usable = cl_summary.get("joint_range_usable") or {}
    out = {}
    for side in SIDES:
        for j in joints:
            jn = f"{side}_{j}"
            rec = usable.get(jn)
            if rec is None:
                continue
            adr = M.m.jnt_qposadr[statics.jid(M.m, jn)]
            vals = np.concatenate([np.asarray(tr["q"], float)[:, adr] for tr in trajs.values()])
            lim = []
            for i in (0, 1):
                fni, by = rec["first_new_interference"][i], rec["limited_by"][i]
                inter_leg = bool(by) and any("left_" in g for g in by) and any("right_" in g for g in by)
                lim.append(rec["limits"][i] if (fni is None or inter_leg) else fni)
            out[jn] = dict(min=float(vals.min()), max=float(vals.max()), usable=lim,
                           margin_min=None if lim[0] is None else float(vals.min() - lim[0]),
                           margin_max=None if lim[1] is None else float(lim[1] - vals.max()))
    return out


def _onset(points):
    """First yaw rate at which the smallest gap crosses zero, by linear interpolation between computed rates."""
    pts = sorted((r, 25.0 if g is None else g) for r, g in points)
    if not pts:
        return dict(onset_rad_s=None, note="no runs")
    if pts[0][1] < 0:
        return dict(onset_rad_s=None, below_rad_s=pts[0][0], points=pts, note="interference at the smallest rate run")
    for (r0, g0), (r1, g1) in zip(pts, pts[1:], strict=False):
        if g0 >= 0 > g1:
            return dict(onset_rad_s=r0 + (r1 - r0) * g0 / (g0 - g1), bracket=[r0, r1], points=pts,
                        note="linear interpolation of the smallest gap (left turns)")
    return dict(onset_rad_s=None, above_rad_s=pts[-1][0], points=pts, note="no interference up to the largest rate run")


def clearance_screen(xml: str, M: statics.Model, q_home, H: float, trajs: dict, cases: dict,
                     rates: dict | None = None) -> dict:
    """``clearance.screen`` (round-two gate set) on every turning run and on the screen's straight walk.

    Per run: interference pairs (signed distance d < 0 between screened shell proxies or hardware of two
    bodies), tight pairs (0 <= d < 3 mm), the smallest gap, the deepest pair, the frames in contact, the
    mid-swing pad clearance and the swing foot's lateral gap.  Per case (left and right turn): the gate
    ``interference_free``, the smallest gap, the deepest pair and that body pair's gap in the straight walk,
    and the thigh-twist and hip-roll excursions against the usable ranges of the joint sweeps.  ``rates`` maps
    run names to (mode, yaw rate), mode "walk" or "in_place"; with it, the yaw rate at which interference
    begins is estimated per mode from the left turns (``_onset``)."""
    from . import clearance

    cl = clearance.screen(xml, q_home, trajs, H=H, gates=CLEARANCE_GATES)
    sm = cl["summary"]
    runs = {}
    for fam, rep in cl["interference"].items():
        if not fam.startswith("gait_"):
            continue
        name = fam[len("gait_"):]
        w = rep.get("worst") or {}
        by_frame = rep.get("min_mm_by_frame", [])
        neg = [i for i, v in enumerate(by_frame) if v is not None and v < 0]
        sw = (sm.get("swing_clearance") or {}).get(name, {})
        runs[name] = dict(
            interference_pairs=rep["counts"]["interference_pairs"], tight_pairs=rep["counts"]["tight_pairs"],
            interference_by_class={k: v["interference"] for k, v in rep["counts"]["by_class"].items()},
            min_mm=w.get("min_mm"),
            worst=(dict(geoms=w["geoms"], bodies=w["bodies"], **{"class": w["class"]}, at=w["at"])
                   if w.get("min_mm") is not None else None),
            frames_in_contact=len(neg), contact_frames=neg,
            pairs_in_contact=[dict(geoms=r["geoms"], min_mm=r["min_mm"], poses=r.get("n_poses_interference"), at=r["at"])
                              for r in rep.get("pairs", []) if r["min_mm"] < 0],
            body_pairs={"/".join(b["bodies"]): b["min_mm"] for b in rep.get("body_pairs", [])},
            mid_swing_min_frac_H=sw.get("mid_swing_min_frac_H"), lateral_gap_mm=sw.get("lateral_gap_mm"),
            max_step_mm=rep.get("max_step_mm"))
    base = runs.get("straight", {})
    out = {}
    for c in cases:
        dirs = [f"{c}_{d}" for d in ("left", "right") if f"{c}_{d}" in runs]
        rr = [runs[d] for d in dirs]
        gaps = [r["min_mm"] for r in rr if r["min_mm"] is not None]
        deep = min((r for r in rr if r["min_mm"] is not None), key=lambda r: r["min_mm"], default=None)
        same = None
        if deep is not None and deep["worst"] is not None:
            same = base.get("body_pairs", {}).get("/".join(deep["worst"]["bodies"]))
        swing = [r["mid_swing_min_frac_H"] for r in rr if r["mid_swing_min_frac_H"] is not None]
        out[c] = dict(interference_free=all(r["interference_pairs"] == 0 for r in rr),
                      interference_pairs=[r["interference_pairs"] for r in rr], min_gap_mm=min(gaps) if gaps else None,
                      deepest=None if deep is None else deep["worst"],
                      frames_in_contact=[r["frames_in_contact"] for r in rr],
                      same_body_pair_straight_walk_mm=same,
                      swing_clearance_ok=bool(min(swing) >= clearance.SWING_MIN_FRAC) if swing else None,
                      lateral_gap_mm=min((r["lateral_gap_mm"] for r in rr if r["lateral_gap_mm"] is not None), default=None),
                      range_margins=_range_margins(sm, M, {d: trajs[d] for d in dirs}))
    onset = None
    if rates:
        onset = {mode: _onset([(r, runs[k]["min_mm"]) for k, (md, r) in rates.items() if md == mode and k in runs])
                 for mode in sorted({md for md, _ in rates.values()})}
    return dict(cases=out, runs=runs, home_interference_free=sm["pass"].get("interference_free_home"),
                straight_min_gap_mm=base.get("min_mm"), interference_onset=onset,
                straight_range_margins=(_range_margins(sm, M, {"straight": trajs["straight"]}) if "straight" in trajs
                                        else None),
                joint_range_usable=sm.get("joint_range_usable"),
                verification=cl.get("verification"), gate_set=CLEARANCE_GATES, timing_s=cl.get("timing_s"),
                note="clearance.screen on each turning run (left and right) and on the screen's straight nominal walk; "
                     "the proxies are mid-surface hulls, so the outer skins sit closer than these gaps")


# ================================================================================================
# Screening one model
# ================================================================================================
def mirror_signs(M: statics.Model) -> dict:
    """Sign that maps each joint's torque onto its mirror partner's convention (y -> -y): -1 for roll and yaw
    axes, +1 for pitch axes; computed from the world joint axes at the current pose (call it at home)."""
    m, d = M.m, M.d
    out = {}
    for nm in M.names:
        partner = _partner(nm)
        a = d.xaxis[statics.jid(m, nm)]
        b = d.xaxis[statics.jid(m, partner)]
        a_m = -np.array([a[0], -a[1], a[2]])
        out[nm] = float(np.sign(a_m @ b))
    return out


def _partner(nm: str) -> str:
    if nm.startswith("left_"):
        return "right_" + nm[5:]
    if nm.startswith("right_"):
        return "left_" + nm[6:]
    return nm


def summary_difference(names, a: dict, b: dict, partner: bool = False) -> dict:
    """Largest relative difference of each joint's RMS, peak and peak speed between summaries ``a`` and ``b``
    (``b``'s mirror partner joint when ``partner``), relative to the larger value with a 0.05 N·m or rad/s floor;
    also without the ankle joints, whose loads follow the CoP and the double-support split most closely."""
    out = {}
    for key in ("rms", "peak", "speed_peak"):
        rel = {}
        for nm in names:
            x, y = a["joints"][nm][key], b["joints"][_partner(nm) if partner else nm][key]
            rel[nm] = abs(x - y) / max(abs(x), abs(y), 0.05)
        worst = max(rel, key=rel.get)
        rest = {nm: r for nm, r in rel.items() if "ankle" not in nm}
        worst_rest = max(rest, key=rest.get)
        out[key] = dict(max_rel_diff=rel[worst], at=worst, max_rel_diff_without_ankles=rest[worst_rest],
                        at_without_ankles=worst_rest)
    return out


def _single_support_cop(s: dict) -> float:
    return s["min_cop_margin_single"]


def _ds_frames(phases) -> int:
    return sum(1 for p in phases if len(p) == 2)


def screen_model(M: statics.Model, q_home: np.ndarray, design, spec: gait.GaitSpec = sizing.NOMINAL,
                 cases: dict | None = None, straight: dict | None = None, mu: float = MU, com_tol: float = COM_TOL,
                 com_iters: int = COM_ITERS, xml: str | None = None, validate: bool = True,
                 brisk: gait.GaitSpec | None = None, probes: dict | None = None) -> dict:
    """Turning screen of one built candidate (``design`` from ``pipeline.build_candidate``).

    ``straight`` is the ``sizing.requirements`` result for the same model (the requirement as sized); it is
    computed here if not given.  The turns, and the straight nominal and brisk walks they are compared with
    like for like, all come from ``generate`` with the same settings.  ``xml`` (the model's MJCF) enables the
    self-collision screen of every turning run (``clearance_screen``); without it the interference gates are
    None (not checked).  ``brisk`` is the brisk straight walk compared with (default ``sizing.BRISK``).
    ``probes`` ({mode: yaw rates}, default ``CLEARANCE_PROBES``) adds left-turn trajectories, kinematics only,
    to the clearance screen to locate the yaw rate at which self-interference begins."""
    cases = CASES if cases is None else cases
    brisk = sizing.BRISK if brisk is None else brisk
    H = design.sk.H
    layout = design.cand.leg_layout
    names = M.names
    t0 = time.perf_counter()
    if straight is None:
        straight = sizing.requirements(M, q_home, H, layout)
    nominal = straight["runs"]["nominal"]  # gait.generate, the reference walk of the sizing
    M.forward(q_home)
    signs = mirror_signs(M)
    timing = dict(sized_requirements_s=time.perf_counter() - t0)

    def run(sp, yaw_rate=0.0, in_place=False, split="exact", ds=True, **gen):
        kw = dict(com_tol=com_tol, com_iters=com_iters) | gen
        tr = generate(M, q_home, H, sp, yaw_rate=yaw_rate, in_place=in_place, **kw)
        idr = inverse_dynamics(M, tr, mu, split=split, ds_friction=ds)
        s = summarize(M, tr, idr)
        s["groups"] = run_groups(names, idr, layout)
        return tr, idr, s

    runs, summaries = {}, {}
    for key, sp in (("straight", spec), ("straight_brisk", brisk)):
        tr, idr, s = run(sp)
        runs[key], summaries[key] = (tr, idr), s
    for cname, c in cases.items():
        for direction, label in ((1.0, "left"), (-1.0, "right")):
            tr, idr, s = run(spec, direction * c["yaw_rate"], c["in_place"])
            runs[f"{cname}_{label}"], summaries[f"{cname}_{label}"] = (tr, idr), s
    timing["screen_runs_s"] = time.perf_counter() - t0 - timing["sized_requirements_s"]

    # --- self-collision along the turns ------------------------------------------------------------
    clear = None
    if xml is not None:
        t1 = time.perf_counter()
        trajs = {"straight": runs["straight"][0]} | {k: v[0] for k, v in runs.items()
                                                     if k not in ("straight", "straight_brisk")}
        rates = {"straight": ("walk", 0.0)} | {f"{c}_left": ("in_place" if cases[c]["in_place"] else "walk",
                                                             cases[c]["yaw_rate"]) for c in cases}
        for mode, rr in (probes if probes is not None else CLEARANCE_PROBES).items():
            for r in rr:  # kinematics only: where between the computed rates does interference begin?
                k = f"probe_{mode}_{r:.2f}_left"
                trajs[k] = generate(M, q_home, H, spec, yaw_rate=r, in_place=mode == "in_place", com_tol=com_tol,
                                    com_iters=com_iters)
                rates[k] = (mode, r)
        clear = clearance_screen(xml, M, q_home, H, trajs, cases, rates)
        timing["clearance_s"] = time.perf_counter() - t1

    # --- validation -------------------------------------------------------------------------------
    validation = {}
    ref = nominal["summary"]
    if validate:
        t1 = time.perf_counter()
        # (1) gait.generate's settings and rules throughout (its CoM tolerance, passes and stride-boundary unwrap
        # as set by gait.set_accuracy, its phase rule and statics' double-support split); this must reproduce
        # gait.generate + inverse_dynamics
        gen_rules = dict(com_tol=gait.COM_TOL, com_iters=gait.COM_ITERS, phase_tol=0.0,
                         unwrap="exact" if gait.EXACT_UNWRAP else "gait")
        trr, _, srr = run(spec, split="statics", ds=False, **gen_rules)
        # (2) round one's CoM settings (2e-4 m, 4 passes) with the exact stride-boundary unwrap (the screen's)
        rules = dict(com_tol=2e-4, com_iters=4, phase_tol=0.0)
        trs_, _, sss = run(spec, split="statics", ds=False, **rules)
        # (3) the screen's CoM tolerance with gait.generate's approximate unwrap: does the ZMP loop converge?
        tru, _, ssu = run(spec, split="statics", ds=False, unwrap="gait", phase_tol=0.0)
        validation["omega0_vs_gait_generate"] = dict(
            gait_accuracy=gait.ACCURACY,
            reproduction=summary_difference(names, srr, ref) | dict(
                max_abs_qpos_diff=float(np.max(np.abs(trr["q"] - nominal["traj"]["q"]))),
                zmp_error_final_m=[srr["gait"]["zmp_error_history_m"][-1], ref["gait"]["zmp_error_history_m"][-1]],
                note="gait.generate's settings (gait_accuracy), phase rule, unwrap and split: must equal "
                     "gait.generate"),
            same_settings_exact_unwrap=summary_difference(names, sss, ref) | dict(
                max_abs_qpos_diff=float(np.max(np.abs(trs_["q"] - nominal["traj"]["q"]))),
                zmp_error_final_m=sss["gait"]["zmp_error_history_m"][-1],
                min_cop_margin_single_m=sss["min_cop_margin_single"],
                note="round one's CoM settings (2e-4 m, 4 passes) with the exact stride-boundary unwrap only"),
            tight_tolerance_gait_unwrap=dict(
                zmp_error_final_m=ssu["gait"]["zmp_error_history_m"][-1],
                zmp_error_exact_derivatives_m=tru["info"]["zmp_error_exact_final_m"],
                zmp_error_exact_frame=tru["info"]["zmp_error_exact_final_frame"],
                min_cop_margin_single_m=ssu["min_cop_margin_single"],
                note="the screen's CoM tolerance (1e-6 m) with gait.generate's approximate unwrap; ZMP error by the "
                     "loop's own measure and with exact derivatives of the same poses"),
            screen_settings=summary_difference(names, summaries["straight"], ref),
            zmp_error_final_m=dict(gait_generate=ref["gait"]["zmp_error_history_m"][-1],
                                   screen=summaries["straight"]["gait"]["zmp_error_history_m"][-1]),
            min_cop_margin_single_m=dict(gait_generate=ref["min_cop_margin_single"],
                                         screen=summaries["straight"]["min_cop_margin_single"]),
            double_support_frames=dict(gait_generate=_ds_frames(nominal["traj"]["phases"]),
                                       screen=_ds_frames(runs["straight"][0]["phases"]),
                                       expected=int(round(2 * (spec.duty - 0.5) * spec.n))),
            note="generate() at yaw rate 0 against gait.generate(sizing.NOMINAL) + gait.inverse_dynamics on the same "
                 "model; 'screen_settings' is the screen's straight walk (CoM 1e-6 m, exact unwrap, event phase "
                 "rule, exact foot-axis split)")
        validation["straight_left_right_asymmetry"] = dict(
            gait_generate=summary_difference(names, ref, ref, partner=True),
            screen=summary_difference(names, summaries["straight"], summaries["straight"], partner=True))
        if "walk_0.35" in cases:
            a, b = (run(spec, d * cases["walk_0.35"]["yaw_rate"], split="statics", ds=False, **rules)[2]
                    for d in (1.0, -1.0))
            validation["mirror_at_round_one_rules"] = dict(walk_0_35=summary_difference(names, a, b, partner=True) | dict(
                min_cop_margin_single_m=[_single_support_cop(a), _single_support_cop(b)],
                zmp_error_final_m=[a["gait"]["zmp_error_history_m"][-1], b["gait"]["zmp_error_history_m"][-1]],
                note="left vs right 0.35 rad/s turn with gait.generate's CoM tolerance and phase rule and statics' "
                     "world-axis double-support split"))
        timing["validation_s"] = time.perf_counter() - t1
    mirror = {}
    for cname in cases:
        a, b = summaries[f"{cname}_left"], summaries[f"{cname}_right"]
        fa, fb = a["feet"]["left"]["single_support"], b["feet"]["right"]["single_support"]
        mirror[cname] = summary_difference(names, a, b, partner=True) | dict(
            single_support_mz_max_nm=[fa.get("mz_abs_max"), fb.get("mz_abs_max")],
            min_cop_margin_single_m=[_single_support_cop(a), _single_support_cop(b)],
            double_support_min_mu_max=[(a.get("double_support") or {}).get("min_mu_max"),
                                       (b.get("double_support") or {}).get("min_mu_max")])
    validation["mirror_left_vs_right_turn"] = mirror
    ik = {k: s["ik_fail_frames"] for k, s in summaries.items()}
    validation.update(ik_fail_frames=ik, ik_ok_all=all(v == 0 for v in ik.values()), signs=signs)

    # --- requirements and comparison ---------------------------------------------------------------
    holds = {g: straight["groups"][g]["hold"] for g in sizing.GROUPS}
    l_nom = group_loads(names, {"straight": runs["straight"][1]}, layout)
    l_brk = group_loads(names, {"straight_brisk": runs["straight_brisk"][1]}, layout)
    straight_same = {g: requirement(dict(rms=l_nom[g]["rms"], peak=max(l_nom[g]["peak"], l_brk[g]["peak"]),
                                         speed=max(l_nom[g]["speed"], l_brk[g]["speed"])), holds[g])
                     for g in sizing.GROUPS}
    loads_turn, req_turn, verdict = {}, {}, {}
    for cname in cases:
        loads_turn[cname] = group_loads(names, {f"{cname}_{d}": runs[f"{cname}_{d}"][1] for d in ("left", "right")},
                                        layout)
        req_turn[cname] = {g: requirement(lt, holds[g]) | dict(rms_motor=lt["rms_motor"], rms_stacked=lt["rms_stacked"])
                           for g, lt in loads_turn[cname].items()}
    sel_bus = design.sel["bus"]
    for cname in cases:
        verdict[cname] = {}
        for g in sizing.GROUPS:
            act = design.act(g)
            bus = sel_bus if act.box is None else None
            v = compare(req_turn[cname][g], straight_same[g], act, bus)
            v["ratio_to_as_sized"] = {k: _ratio(req_turn[cname][g][k], straight["groups"][g][k])
                                      for k in ("cont", "peak_req", "speed_req")}
            lt, sl = loads_turn[cname][g], l_nom[g]
            v["like_for_like"] = dict(rms=_ratio(lt["rms"], sl["rms"]), peak=_ratio(lt["peak"], sl["peak"]),
                                      speed=_ratio(lt["speed"], sl["speed"]), turn_rms=lt["rms"], turn_peak=lt["peak"],
                                      turn_speed=lt["speed"], straight_rms=sl["rms"], straight_peak=sl["peak"],
                                      straight_speed=sl["speed"])
            verdict[cname][g] = v
    # gates per case: drives within their planning limits, IK on every frame, no self-interference
    gates = {}
    for cname in cases:
        within = all(not v["over_limit"] for v in verdict[cname].values())
        ik_ok = all(summaries[f"{cname}_{d}"]["ik_fail_frames"] == 0 for d in ("left", "right"))
        free = None if clear is None else bool(clear["cases"][cname]["interference_free"])
        gates[cname] = dict(role=cases[cname]["role"], drives_within_limits=within, ik_ok=ik_ok, interference_free=free,
                            **{"pass": bool(within and ik_ok and free is not False)})
    # envelope: what the drives would have to meet if the turns joined the straight-walk sizing cases
    envelope = {}
    for label, names_c in (("targets", [c for c in cases if cases[c]["role"] == "target"]),
                           ("targets_and_stretch", list(cases))):
        envelope[label] = {}
        for g in sizing.GROUPS:
            act = design.act(g)
            bus = sel_bus if act.box is None else None
            comb = {k: max([straight["groups"][g][k], straight_same[g][k]] + [req_turn[c][g][k] for c in names_c])
                    for k in ("cont", "peak_req", "speed_req")}
            envelope[label][g] = dict(actuator=act.id, required=comb, meets=sizing.meets(act, comb, bus),
                                      ratio_to_limit=dict(cont=comb["cont"] / act.sustained, peak_req=comb["peak_req"] / act.peak,
                                                          speed_req=comb["speed_req"] / max(act.speed_at(bus), 1e-9)))
    flags = []
    for cname, rows in verdict.items():
        role = cases[cname]["role"]
        for g, v in rows.items():
            for k in v["over_limit"]:
                flags.append(f"{cname} ({role}): {g} {k} {v['required'][k]:.3g} exceeds the {v['actuator']} planning "
                             f"limit {v['limits'][k]:.3g}")
            for k in v["increase_over_10pct"]:
                r, r1 = v["ratio_to_straight"][k], v["ratio_to_as_sized"][k]
                txt = ("new: the straight walk needs none" if r is None else
                       f"{r:.2f}x the straight-walk requirement" + ("" if r1 is None else f" ({r1:.2f}x the requirement as sized)"))
                flags.append(f"{cname} ({role}): {g} {k} {v['required'][k]:.3g}, {txt}")
            for k in v["near_increase_flag"]:
                flags.append(f"{cname} ({role}): {g} {k} {v['ratio_to_straight'][k]:.3f}x the straight-walk requirement, "
                             f"within {NEAR_FLAG_BAND:.2f} of the 10 % flag (inside the method's sensitivity)")
            for k in v["near_limit"]:
                flags.append(f"{cname} ({role}): {g} {k} at {v['ratio_to_limit'][k]:.3f} of the {v['actuator']} planning "
                             f"limit, within {NEAR_FLAG_BAND:.2f} of it")
        if clear is not None and not clear["cases"][cname]["interference_free"]:
            cc = clear["cases"][cname]
            dp = cc["deepest"] or {}
            at = dp.get("at", {})
            flags.append(f"{cname} ({role}): self-interference, {sum(cc['interference_pairs'])} pair(s) over the left and "
                         f"right turns, deepest {cc['min_gap_mm']:.2f} mm ({' vs '.join(dp.get('geoms', []))}, frame "
                         f"{at.get('frame')}, support {at.get('support')}); that body pair is "
                         f"{cc['same_body_pair_straight_walk_mm']} mm apart at its closest in the straight walk")
        if clear is not None and clear["cases"][cname]["swing_clearance_ok"] is False:
            flags.append(f"{cname} ({role}): mid-swing pad clearance below 0.04 H")
    for k, s in summaries.items():
        if s["ik_fail_frames"]:
            flags.append(f"{k}: IK failed on {s['ik_fail_frames']} of {spec.n} frames")
        if s["min_cop_margin_single"] < -0.010:
            flags.append(f"{k}: reference CoP leaves the pad by {-1000 * s['min_cop_margin_single']:.1f} mm in single support")
        for side in SIDES:
            st = s["feet"][side]["single_support"]
            if st.get("mu_req_max") is not None and st["mu_req_max"] > mu:
                flags.append(f"{k}: {side} foot single support needs mu {st['mu_req_max']:.2f} (yaw moment "
                             f"{st['mz_abs_max']:.2f} N·m, r_eff {1000 * s['feet'][side]['r_eff']:.1f} mm), above the "
                             f"assumed {mu}")
        if s["split"]["solved"].get("infeasible"):
            flags.append(f"{k}: the double-support QP has no feasible split on {s['split']['solved']['infeasible']} "
                         "frames (equality-only optimum kept)")
        dsb = s.get("double_support") or {}
        if dsb.get("infeasible_frames"):
            flags.append(f"{k}: no double-support split keeps both CoPs on the pads on {dsb['infeasible_frames']} frames")
        elif dsb.get("min_mu_max") is not None and dsb["min_mu_max"] > mu:
            flags.append(f"{k}: double support needs mu {dsb['min_mu_max']:.2f}, above the assumed {mu}")
    timing["total_s"] = time.perf_counter() - t0
    return dict(spec=dict(spec.__dict__), cases=cases, mu=mu, dcof_a326_product_class_context=DCOF_A326_CLASS,
                fn_min_frac_qp_allocation=FN_MIN_FRAC, com_tol=com_tol, com_iters=com_iters, phase_tol=PHASE_TOL,
                near_flag_band=NEAR_FLAG_BAND, split="exact QP; CoP box and friction pyramid in each foot's axes",
                straight=dict(as_sized=straight["groups"], same_method=straight_same, holds=straight["holds"],
                              as_sized_nominal=ref),
                runs=summaries, requirements=req_turn, verdict=verdict, gates=gates, clearance=clear,
                envelope=envelope, flags=flags, validation=validation, timing=timing, _runs=runs)


# ================================================================================================
# Plot
# ================================================================================================
def own_cycle(series: np.ndarray, side: str) -> np.ndarray:
    """Series re-indexed so that index 0 is that leg's lift-off (the right leg lifts at half the stride)."""
    return series if side == "left" else np.roll(series, -len(series) // 2)


def plot(cid: str, M: statics.Model, q_home, nominal_idr: dict, turns: dict, out_png, duty: float):
    """Four small multiples (hip roll, thigh twist, hip pitch, knee): torque over each leg's own cycle for the
    straight nominal walk (left leg) and the left turns in ``turns`` = {label: (traj, idr)} (inner = left leg,
    outer = right leg shifted half a stride and mirrored to the left leg's sign convention at home)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surf, ink, ink2, muted, grid, axis_c, band = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#f0efec"
    inner_c, outer_c = "#2a78d6", "#eb6834"  # categorical slots 1-2 (reference palette, validated)
    names = M.names
    M.forward(q_home)  # the sign convention of the home pose, not of whatever frame was evaluated last
    signs = mirror_signs(M)
    fig, axs = plt.subplots(2, 2, figsize=(8.6, 5.6), dpi=110, sharex=True)
    fig.patch.set_facecolor(surf)
    styles = [("-", 2.0), ("--", 1.6), (":", 1.8)]
    for ax, (j, ttl) in zip(axs.ravel(), PLOT_JOINTS, strict=True):
        ax.set_facecolor(surf)
        ax.axvspan(0, 1 - duty, color=band, lw=0, zorder=0)
        nm_l, nm_r = f"left_{j}", f"right_{j}"
        n = len(nominal_idr["tau"])
        ph = np.arange(n) / n
        ax.plot(ph, nominal_idr["tau"][:, names.index(nm_l)], color=muted, lw=2.0, label="straight nominal walk",
                solid_capstyle="round", zorder=2)
        for (label, (tr, idr)), (ls, lw) in zip(turns.items(), styles, strict=False):
            n_t = len(idr["tau"])
            ph_t = np.arange(n_t) / n_t
            ax.plot(ph_t, own_cycle(idr["tau"][:, names.index(nm_l)], "left"), color=inner_c, ls=ls, lw=lw,
                    label=f"{label}, inner leg", zorder=3)
            ax.plot(ph_t, signs[nm_r] * own_cycle(idr["tau"][:, names.index(nm_r)], "right"), color=outer_c, ls=ls, lw=lw,
                    label=f"{label}, outer leg", zorder=3)
        ax.set_title(ttl, loc="left", fontsize=9, color=ink)
        ax.grid(True, axis="y", color=grid, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(axis_c)
        ax.tick_params(colors=axis_c, labelcolor=ink2, labelsize=7.5)
        ax.set_xlim(0, 1)
    axs[0, 0].annotate("swing", xy=(0.5 * (1 - duty), 1.0), xycoords=("data", "axes fraction"), xytext=(0, -10),
                       textcoords="offset points", ha="center", fontsize=7.5, color=ink2)
    for ax in axs[:, 0]:
        ax.set_ylabel("Torque, N·m", fontsize=8, color=ink2)
    for ax in axs[1, :]:
        ax.set_xlabel("Phase of each leg's own cycle (0 = lift-off)", fontsize=8, color=ink2)
    h, lab = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper left", bbox_to_anchor=(0.01, 0.955), ncol=3, frameon=False, fontsize=7.5, labelcolor=ink2)
    fig.suptitle(f"{cid}: leg torques, straight walk vs prescribed left turns (reference trajectories, not a controller)",
                 x=0.012, ha="left", fontsize=9.5, color=ink, fontweight="bold")
    fig.text(0.012, 0.005, "Inner = left leg (the side turned toward), outer = right leg, shifted half a stride and "
             "mirrored to the left leg's sign convention. Shaded: swing.", fontsize=7, color=ink2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.9))
    fig.savefig(out_png, facecolor=surf)
    plt.close(fig)
    from .round1 import compact_png

    compact_png(out_png)
    return str(out_png)


# ================================================================================================
# Candidates, the round-two hook and the CLI
# ================================================================================================
def _clean(x):
    """JSON-safe copy: numpy to Python, non-finite floats to None, private keys dropped."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items() if not str(k).startswith("_")}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, (np.floating, float)):
        return float(x) if math.isfinite(float(x)) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _sha256(path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def screen_built(cid: str, M: statics.Model, q_home, design, xml, rec: dict, out: Path, record: str,
                 spec: gait.GaitSpec = sizing.NOMINAL) -> dict:
    """Screen one built candidate against its sized record ``rec``; write ``<out>/<id>_turning.json`` and
    ``<out>/<id>_turning.png``; return the document."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    r = screen_model(M, q_home, design, spec, xml=str(xml))
    # the recomputed straight-walk requirement must equal the sized record (same model, same code)
    dev = 0.0
    for g, mg in rec["margins"].items():
        for k_rec, k in (("cont_req", "cont"), ("peak_req", "peak_req"), ("speed_req", "speed_req")):
            a, b = mg[k_rec], r["straight"]["as_sized"][g][k]
            dev = max(dev, abs(a - b) / max(abs(a), 1e-6))
    sha = _sha256(xml)
    r["validation"]["sized_requirements_max_rel_diff"] = dev
    r["validation"]["xml_sha256_matches_record"] = sha == rec.get("xml_sha256")
    head = dict(id=cid, record=record, rev=rec.get("rev", 1), overrides=rec.get("overrides", []), H=design.sk.H,
                mass=M.mass(), layout=design.cand.leg_layout, bus=design.sel["bus"],
                selection=dict(joints=design.sel["joints"], neck=design.sel["neck"], tail=design.sel["tail"]),
                xml_sha256=sha, record_xml_sha256=rec.get("xml_sha256"))
    runs = r["_runs"]
    png = plot(cid, M, q_home, runs["straight"][1],
               {"walk 0.35 rad/s": runs["walk_0.35_left"], "in place 0.50 rad/s": runs["in_place_0.50_left"]},
               out / f"{cid}_turning.png", spec.duty)
    doc = _clean(head | r | dict(png=Path(png).name))
    (out / f"{cid}_turning.json").write_text(json.dumps(doc, indent=1))
    return doc


def screen_record(cid: str, rec: dict, xml, q_home, design, scr, spec: gait.GaitSpec = sizing.NOMINAL) -> dict:
    """Round-two hook (``round2.screen_one``): screen a sized, rebuilt candidate from its own MJCF.

    A fresh ``statics.Model`` is loaded from ``xml`` (the other screens may have changed a shared one), the
    MJCF's SHA-256 is checked against ``rec``, ``<scr>/<id>_turning.json`` and ``.png`` are written, and the
    compact ``record_summary`` is returned for the review index (``shortlist`` reads its keys)."""
    M = statics.Model.load(str(xml))
    doc = screen_built(cid, M, np.asarray(q_home, float), design, xml, rec, Path(scr), record="round2", spec=spec)
    return record_summary(doc)


def _build(cid: str, out: Path, stage_a: Path | None, round2: Path | None):
    """Rebuild a sized candidate exactly as its record: Stage A (as ``round1.rebuild``) or round two (as
    ``round2.screen_one``: generator revision 2 and the record's packaging overrides)."""
    from . import pipeline

    if round2 is not None:
        from . import round2 as r2

        rec = json.loads((Path(round2) / "candidates" / f"{cid}.json").read_text())
        cand = r2.candidate(cid, rec.get("overrides", ()))
        sel = dict(rec["selection"])
        record = "round2"
    else:
        from . import params, round1

        rec = json.loads((Path(stage_a) / "candidates" / f"{cid}.json").read_text())
        cand = params.parse(cid)
        sel = dict(rec["selection"])
        sel["mass_guess"] = round1._final_mass_guess(rec)
        record = "stage_a"
    res = pipeline.build_candidate(cand, out / "mjcf", selection=sel, B=rec["B"], write=False)
    return rec, res, out / "mjcf" / f"{cid}.xml", record


def screen_candidate(cid: str, out: Path, stage_a: Path = Path("round1/stage_a"), spec: gait.GaitSpec = sizing.NOMINAL,
                     round2: Path | None = None) -> dict:
    """Rebuild a sized candidate (Stage A, or a round-two record when ``round2`` names the round-two directory),
    screen it and write ``<out>/<id>_turning.json`` and ``.png``; models go to ``<out>/mjcf``."""
    out = Path(out)
    if round2 is not None:  # a round-two record: the runner's walk accuracy (round2.GAIT_ACCURACY)
        from . import round2 as r2

        gait.set_accuracy(r2.GAIT_ACCURACY)
    rec, res, xml, record = _build(cid, out, stage_a, round2)
    return screen_built(cid, res["model"], res["q_home"], res["design"], xml, rec, out, record, spec)


def _case_stats(doc: dict, c: str) -> dict:
    runs = [doc["runs"][f"{c}_{d}"] for d in ("left", "right")]
    ss = [s["feet"][side]["single_support"] for s in runs for side in SIDES]
    ds = [s.get("double_support") or {} for s in runs]
    mu_ss = max((v["mu_req_max"] for v in ss if v.get("mu_req_max") is not None), default=None)
    mu_ds = max((v["min_mu_max"] for v in ds if v.get("min_mu_max") is not None), default=None)
    return dict(runs=runs, mu_single=mu_ss, mu_double=mu_ds,
                mu_needed=max((x for x in (mu_ss, mu_ds) if x is not None), default=None),
                hip_yaw_range_use=max(s["range_use"][f"{side}_hip_yaw"] for s in runs for side in SIDES))


def record_summary(doc: dict) -> dict:
    """Compact per-candidate result for the round-two review index (``round2`` stores it; ``shortlist`` reads
    ``targets_within_limits``, ``worst_ratio_targets``, ``hip_yaw_range_used_max`` and ``mu_needed_max``).
    ``targets_within_limits`` is the drive gate only; ``targets_pass`` adds IK and self-interference."""
    cases, gates = doc["cases"], doc["gates"]
    tg = [c for c in cases if cases[c]["role"] == "target"]
    st = [c for c in cases if cases[c]["role"] != "target"]
    worst = max(((v["ratio_to_limit"][k], g, k, c) for c in tg for g, v in doc["verdict"][c].items()
                 for k in v["ratio_to_limit"]), default=(None, None, None, None))
    cs = {c: _case_stats(doc, c) for c in cases}
    cl = (doc.get("clearance") or {}).get("cases")

    def _all(keys, field):
        vals = [gates[c][field] for c in keys]
        return None if (not vals or any(v is None for v in vals)) else all(vals)

    return dict(
        targets_pass=all(gates[c]["pass"] for c in tg) if tg else None,
        targets_within_limits=_all(tg, "drives_within_limits"),
        targets_interference_free=_all(tg, "interference_free"),
        targets_ik_ok=_all(tg, "ik_ok"),
        worst_ratio_targets=worst[0], worst_targets=dict(group=worst[1], metric=worst[2], case=worst[3]),
        hip_yaw_range_used_max=max(cs[c]["hip_yaw_range_use"] for c in tg) if tg else None,
        mu_needed_max=max((cs[c]["mu_needed"] for c in tg if cs[c]["mu_needed"] is not None), default=None),
        min_gap_mm=None if cl is None else {c: cl[c]["min_gap_mm"] for c in cases},
        interference_pairs=None if cl is None else {c: cl[c]["interference_pairs"] for c in cases},
        interference_onset_rad_s=(None if not (doc.get("clearance") or {}).get("interference_onset") else
                                  {m: v.get("onset_rad_s") for m, v in doc["clearance"]["interference_onset"].items()}),
        stretch_pass=all(gates[c]["pass"] for c in st) if st else None,
        stretch_within_limits=_all(st, "drives_within_limits"),
        stretch_interference_free=_all(st, "interference_free"),
        hip_yaw_speed_required={c: doc["verdict"][c]["hip_yaw"]["required"]["speed_req"] for c in cases},
        hip_yaw_speed_limit=doc["verdict"][tg[0]]["hip_yaw"]["limits"]["speed_req"] if tg else None,
        mu_single_support_max={c: cs[c]["mu_single"] for c in cases},
        mu_double_support_max={c: cs[c]["mu_double"] for c in cases},
        increases_over_10pct={c: {g: v["increase_over_10pct"] for g, v in doc["verdict"][c].items() if v["increase_over_10pct"]}
                              for c in cases},
        n_flags=len(doc["flags"]), xml_sha256_matches_record=doc["validation"].get("xml_sha256_matches_record"),
        sized_requirements_max_rel_diff=doc["validation"].get("sized_requirements_max_rel_diff"),
        ik_ok_all=doc["validation"].get("ik_ok_all"), file=f"{doc['id']}_turning.json")


def headline(doc: dict) -> dict:
    """Per-candidate summary for the CLI index (``turning_index.json``): ``record_summary`` plus, per case, the
    leg groups against their drives, friction (the single-support requirement, the double-support requirement
    and the QP allocation as a diagnostic), clearance, the envelope and the validation block."""
    out = dict(id=doc["id"], record=doc["record"], H=doc["H"], mass=doc["mass"], bus=doc["bus"]) | record_summary(doc)
    cases = {}
    cl = (doc.get("clearance") or {}).get("cases", {})
    keys = ("ft_over_fn_max", "mz_abs_max", "util_max", "util_cop_max", "mu_req_max", "mu_req_cop_max", "mu_req_centroid_max")
    for c, rows in doc["verdict"].items():
        cs = _case_stats(doc, c)
        runs = cs["runs"]
        fr = {}
        for kind in ("single_support", "double_support_qp_allocation"):
            vals = [s["feet"][side][kind] for s in runs for side in SIDES]
            fr[kind] = {k: max([v[k] for v in vals if v.get(k) is not None], default=None) for k in keys}
        fr["double_support_requirement"] = dict(
            min_mu_max=cs["mu_double"],
            min_mu_lower_max=max(((s.get("double_support") or {}).get("min_mu_lower_max") or 0.0 for s in runs), default=None),
            infeasible_frames=sum((s.get("double_support") or {}).get("infeasible_frames", 0) for s in runs))
        cases[c] = dict(role=doc["cases"][c]["role"], yaw_rate=doc["cases"][c]["yaw_rate"], in_place=doc["cases"][c]["in_place"],
                        gate=doc["gates"][c],
                        over_limit={g: v["over_limit"] for g, v in rows.items() if v["over_limit"]},
                        near_limit={g: v["near_limit"] for g, v in rows.items() if v["near_limit"]},
                        increase_over_10pct={g: v["increase_over_10pct"] for g, v in rows.items() if v["increase_over_10pct"]},
                        near_increase_flag={g: v["near_increase_flag"] for g, v in rows.items() if v["near_increase_flag"]},
                        groups={g: dict(actuator=v["actuator"], required=v["required"], limits=v["limits"],
                                        ratio_to_limit=v["ratio_to_limit"], ratio_to_straight=v["ratio_to_straight"],
                                        ratio_to_as_sized=v["ratio_to_as_sized"], like_for_like=v["like_for_like"])
                                for g, v in rows.items() if g in LEG_GROUPS},
                        hip_yaw_range_use=cs["hip_yaw_range_use"], friction=fr,
                        clearance=cl.get(c), ik_fail_frames=sum(s["ik_fail_frames"] for s in runs),
                        min_cop_margin_single=min(s["min_cop_margin_single"] for s in runs),
                        zmp_error_final_m=max(s["gait"]["zmp_error_history_m"][-1] for s in runs))
    out["cases"] = cases
    s0 = doc["runs"]["straight"]
    out["straight_friction"] = {kind: {k: max([s0["feet"][side][kind][k] for side in SIDES
                                               if s0["feet"][side][kind].get(k) is not None], default=None) for k in keys}
                                for kind in ("single_support", "double_support_qp_allocation")}
    out["straight_friction"]["double_support_requirement"] = dict(min_mu_max=(s0.get("double_support") or {}).get("min_mu_max"))
    sb = doc["runs"]["straight_brisk"]
    out["straight_brisk_friction"] = dict(
        single_support_mu_req_max=max((sb["feet"][side]["single_support"].get("mu_req_max") or 0.0) for side in SIDES),
        double_support_min_mu_max=(sb.get("double_support") or {}).get("min_mu_max"))
    out["straight_min_cop_margin_single"] = s0["min_cop_margin_single"]
    out["r_eff"] = s0["feet"]["left"]["r_eff"]
    if doc.get("clearance"):
        out["straight_min_gap_mm"] = doc["clearance"].get("straight_min_gap_mm")
    out["envelope"] = {k: {g: dict(meets=v["meets"], ratio_to_limit=v["ratio_to_limit"]) for g, v in e.items()}
                       for k, e in doc["envelope"].items()}
    out["validation"] = {k: v for k, v in doc["validation"].items() if k != "signs"}
    out["flags"] = doc["flags"]
    return out


def _screen_star(a):
    try:
        return headline(screen_candidate(*a))
    except Exception as e:  # report and continue with the other candidates
        return dict(id=a[0], error=repr(e), trace=traceback.format_exc())


def main(argv=None):
    ap = argparse.ArgumentParser(description="Turning-load screen on prescribed turning trajectories (see module docstring).")
    ap.add_argument("ids", nargs="+", help="candidate ids, e.g. 60-M-T-10")
    ap.add_argument("--out", required=True, help="output directory (never round1/)")
    ap.add_argument("--stage-a", default="round1/stage_a", help="Stage A record directory (round-one bodies)")
    ap.add_argument("--round2", default=None, help="round-two directory: screen its sized records (revision 2, overrides)")
    ap.add_argument("--workers", type=int, default=1, help="parallel processes (at most 2)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.resolve().is_relative_to(Path("round1").resolve()):
        raise SystemExit("write outputs outside round1/")
    out.mkdir(parents=True, exist_ok=True)
    r2 = Path(a.round2) if a.round2 else None
    jobs = [(cid, out, Path(a.stage_a), sizing.NOMINAL, r2) for cid in a.ids]
    workers = max(1, min(2, a.workers))
    if workers == 1 or len(jobs) == 1:
        results = [_screen_star(j) for j in jobs]
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            results = list(pool.imap(_screen_star, jobs))
    (out / "turning_index.json").write_text(json.dumps(_clean(results), indent=1))
    failed = [r["id"] for r in results if "error" in r]
    for r in results:
        if "error" in r:
            print(r["id"], "ERROR", r["error"])
            continue
        print(r["id"], "targets pass:", r["targets_pass"], "flags:", len(r["flags"]))
        for f in r["flags"]:
            print("  ", f)
    if failed:
        print("failed:", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
