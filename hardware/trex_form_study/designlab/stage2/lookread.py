"""The look reading (reading 1): the four look measures on an episode's 50 Hz poses, their comparison with a
reference's own values against the bands, and the reference's values by kinematic playback of a track.

Rules: ``STAGE2_TRACKB_PLAN.md`` U1 (the measures), U2 (the window), U3 (the bands and what is not read), U5 (the
reference's values; too few steps); gate Z2 (G9-ref) parts (b) and (c) are read with the helpers at the end.  The
readings of rule U4 (what "success with the look" is on which set and item, the stop item's
``on_command_at_stop``) and rule U6's companion record are not here: they are the row wrapper's (``lookbar``).  No
module of the bar is edited, and ``bar`` itself is not imported at module level (``readings``, whose contact rule
and constants are read, and ``s2io`` are): ``lookbar.LookRow`` (another package) lends an episode a recorder with
poses and calls this module; ``playback`` imports ``bar`` inside the function.

Clocks.  Every time here is simulated seconds on the bar's own convention (``readings``' module docstring): pose j
is the update row at ``j * every * dt``, physics step ``a`` is the instant ``a * dt``, and a recorder's pose and
load with the same step index are one instant.  A lift-off (touchdown) is the first physics step out of (in)
contact after a step in (out of) it, by the 2 % load rule with the 20 ms debounce (``readings.contact_flags``).

``series``    by ``mujoco.mj_kinematics`` on a private ``MjData`` of the plant's model, joints and bodies found by
              name: the hip midpoint's height (the midpoint of the two hip-roll joints' anchors, which are fixed on
              the trunk: ``look.arrays``' ``hip_mid``), the head body's pitch in the world (the elevation of its x
              axis, nose up positive: ``look.arrays``' ``head_pitch_world``, an absolute angle; the reading is
              about its mean, so home's value does not enter), each foot's pitch (the elevation of the foot body's
              x axis below the horizontal, toe down and heel up positive: the look files' ``foot_pitch_solved`` on
              a foot that points along world x, and the same angle at any heading, which a turn item needs), the
              tail-yaw joint's angle; contact, lift-offs and touchdowns per foot.
``reading``   the four numbers over a window, from what lies inside the window alone (samples, lift-offs and
              touchdowns with ``ta <= t <= tb``; a value at a lift-off or touchdown is interpolated between the
              two poses around it), so that an episode's trace and a playback of just the window are read alike.
``compare``   the four terms against a reference's reading (``BANDS``, ``NOT_READ_RAD``, ``MIN_STEPS``).
``playback``  the reference's reading: the track's own files placed in one world frame by ``bar._Played``, played
              at the nominal commanded clock over the window's reference times.
``window``    rule U2 from a built bar condition and its recorder (simulated time; a walk episode that fell has
              none, as the bar judges none); ``window_ref`` the reference times that window covers at the
              nominal commanded clock, from the built condition alone (what ``playback`` takes: one definition
              for the gate and for an episode's wrapper).

For gate G9-ref: ``played`` (a playback's arrays), ``degrade`` and ``controls`` (part (c)'s negative controls),
``ladder_numbers`` and ``against_ladder`` (part (b): the four numbers of a ``stage2/look/metrics.json`` rung and
Z2's limits).

This file is hashed into the sources of every Track B row (``sources``).
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from . import motion, readings, s2io

#: Research 4.2's bands: bob within +-``bob_rel`` of the reference's, head pitch at most
#: ``head_world_pitch_rms_deg`` RMS, tail phase within +-``tail_phase_cycles``, toe-off at least ``toe_off_rel`` of
#: the reference's (rule U3).
BANDS = motion.BANDS
#: Own steps and own lift-offs a window needs (rule U5).
MIN_STEPS = 4
#: The tail term is not read when the reference's tail amplitude in the window is under this, the toe-off term when
#: the reference's toe-off is (rad; rule U3).  Compared without slack: ``reference < NOT_READ_RAD``.
NOT_READ_RAD = 0.02
TERMS = ("bob", "head", "tail", "toe_off")
#: The terms that are segmented by the robot's own steps, which too few steps fail (rule U5).
STEP_TERMS = ("bob", "tail", "toe_off")
SIDES = ("left", "right")
#: The nominal commanded clock (``panels.RATE``) and the poses' interval on the simulated clock (50 Hz).
RATE = readings.RATE
POSE_S = 0.02
EPS_S = readings.EPS_S
#: ``control.G``: a recorder's weight is the trunk's subtree mass times this (``metrics.Recorder.begin``).
G = 9.81
#: Rule U2's windows (``tune.WALK_WINDOW_S``, ``readings.HEADING_HOLD_S``).
WALK_WINDOW_S = readings.WALK_WINDOW_S
TURN_WINDOW_S = readings.HEADING_HOLD_S

RULES = dict(
    measures=(
        "STAGE2_TRACKB_PLAN rule U1.  From 50 Hz poses, by kinematics on the plant's model: bob = the mean, over "
        "the robot's own steps wholly inside the window (a step runs from one touchdown of either foot to the "
        "next), of max less min of the hip midpoint's height (the midpoint of the two hip-roll joints' anchors; "
        "the poses between the two touchdowns and the height interpolated at each of them); head = the RMS about "
        "its mean of the head body's pitch in the world (the elevation of its x axis) over the poses in the "
        "window, degrees; tail phase = psi / (2 pi) of the least-squares fit tail = A sin(2 pi phi + psi) + c of "
        "the tail-yaw joint's angle on the robot's own phase phi (0 at its first left lift-off in the window, one "
        "more at each later one, linear in time between them; the poses between the window's first and last left "
        "lift-off), and A; toe-off = the mean, over the lift-offs of either foot in the window, of the lifting "
        "foot's pitch (the elevation of the foot body's x axis below the horizontal, toe down positive) at the "
        "lift-off instant, interpolated between poses.  Contact is the 2 % load rule with the 20 ms debounce "
        "(readings.contact_flags); a lift-off or touchdown is the first physics step after the change."),
    window=(
        f"STAGE2_TRACKB_PLAN rule U2.  A walk item: the last {WALK_WINDOW_S:g} s of an episode that ran to its "
        f"planned end (the bar's judged window).  A turn item: the {TURN_WINDOW_S:g} s after t_c + at_s (t_c the "
        "simulated time at which the reference clock reached the command, at_s the item's judged time).  Nudge and "
        "stop items are not read."),
    bands=(
        f"STAGE2_TRACKB_PLAN rule U3.  Against the reference's own values: bob |own / reference - 1| <= "
        f"{BANDS['bob_rel']:g}; head pitch own <= {BANDS['head_world_pitch_rms_deg']:g} deg RMS (not relative); "
        f"tail phase |own - reference| wrapped to half a cycle <= {BANDS['tail_phase_cycles']:g} cycle; toe-off own "
        f">= {BANDS['toe_off_rel']:g} x reference.  The tail term is not read when the reference's tail amplitude "
        f"in the window is under {NOT_READ_RAD:g} rad, the toe-off term when the reference's toe-off is under "
        f"{NOT_READ_RAD:g} rad; both then pass, labelled, and the reference's value and its side of the threshold "
        "are recorded.  A value of the robot's that cannot be computed fails its term.  The bob has no such "
        "threshold: against a reference without a bob the term fails.  The tail term reads the phase alone: the "
        "ratio of the two amplitudes is recorded and no band reads it."),
    reference=(
        f"STAGE2_TRACKB_PLAN rule U5.  The reference's values are the same functions on a kinematic playback of "
        f"the episode's own track over the window's reference times at the nominal commanded clock ({RATE:g}): "
        "the files' poses placed in one world frame by bar._Played, contact from the files' support flags.  When "
        f"that playback has at least {MIN_STEPS} steps and {MIN_STEPS} lift-offs in the window and the robot has "
        "fewer, the bob, tail and toe-off terms fail (too_few_steps); it is no error.  When the playback itself "
        "has fewer, or lacks one of its own values, it is an error of the definition (LookError)."),
)


class LookError(RuntimeError):
    """The look reading cannot be defined: a model without the named joints and bodies, arrays that do not fit, a
    reference whose own window holds too few steps (rule U5), or a track that cannot be played (``not played``)."""


# ------------------------------------------------------------------------------------------------
# Series: kinematics of the poses, contact events from the loads
# ------------------------------------------------------------------------------------------------
def _ids(m) -> dict:
    """The joints and bodies the measures read, by name (an ``-X`` body has 16 hinges: never by index)."""

    def jid(name):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j < 0:
            raise LookError(f"the model has no joint {name!r}")
        return int(j)

    def bid(name):
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        if b < 0:
            raise LookError(f"the model has no body {name!r}")
        return int(b)

    return dict(hips=[jid(f"{s}_hip_roll") for s in SIDES], head=bid("head"), feet=[bid(f"{s}_foot") for s in SIDES],
                trunk=bid("trunk"), tail=int(m.jnt_qposadr[jid("tail_yaw")]), neck=int(m.jnt_qposadr[jid("neck_pitch")]),
                ankles=[int(m.jnt_qposadr[jid(f"{s}_ankle_pitch")]) for s in SIDES])


def weight(p) -> float:
    """The weight a recorder divides the feet's loads by (N): the trunk's subtree mass times ``control.G``."""
    return float(p.m.body_subtreemass[_ids(p.m)["trunk"]]) * G


def _nose_up(d, body: int) -> float:
    """The elevation of a body's x axis (rad, up positive): ``gait._nose_up`` of its world quaternion."""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, d.xquat[body])
    x = R.reshape(3, 3)[:, 0]
    return float(math.atan2(x[2], math.hypot(x[0], x[1])))


def _kinematics(m, d, ids: dict, q) -> tuple[float, float, float, float]:
    """(hip midpoint height, head pitch in the world, left foot pitch, right foot pitch) of one pose."""
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    hip = 0.5 * (d.xanchor[ids["hips"][0]][2] + d.xanchor[ids["hips"][1]][2])
    return (float(hip), _nose_up(d, ids["head"]), -_nose_up(d, ids["feet"][0]), -_nose_up(d, ids["feet"][1]))


def _events(contact, dt: float, t0: float) -> tuple[list, list]:
    """Per foot, the lift-off and the touchdown times (s) of debounced contact flags ``[steps, 2]``."""
    lo = readings.touchdowns(~np.asarray(contact, bool), dt)  # a "touchdown" of the air flags is a lift-off
    td = readings.touchdowns(contact, dt)
    return ([np.array([t0 + t for t, i in lo if i == j], float) for j in (0, 1)],
            [np.array([t0 + t for t, i in td if i == j], float) for j in (0, 1)])


def series(p, poses, loads, dt, every, t0: float = 0.0) -> dict:
    """The look series of an episode (module docstring).  ``p`` is a ``plant.Plant`` (only ``p.m`` is read, on a
    private ``MjData``: the plant's own state is not touched); ``poses [n, nq]`` the full ``qpos`` every ``every``
    physics steps (50 Hz: ``every * dt = 0.02``); ``loads [steps, 2]`` the feet's vertical loads (N) at every
    physics step; pose j and load ``j * every`` are one instant, ``t0 + j * every * dt``, and every pose needs its
    load (``LookError`` for a pose past the last load).

    Returns ``t50 [n]``, ``hip_mid_z [n]`` (m, world), ``head_pitch_world [n]`` (rad), ``foot_pitch [n, 2]`` (rad,
    toe down positive), ``tail [n]`` (rad), ``contact [steps, 2]`` (bool, debounced), ``lift_offs`` and
    ``touchdowns`` (per foot, left then right, an array of times), and ``dt``, ``every``, ``steps``, ``weight``."""
    m = p.m
    ids = _ids(m)
    poses = np.asarray(poses, float)
    if poses.ndim != 2 or poses.shape[1] != m.nq:
        raise LookError(f"poses {poses.shape}: [n, {m.nq}] full qpos rows are needed")
    loads = np.asarray(loads, float)
    if loads.ndim != 2 or loads.shape[1] != 2:
        raise LookError(f"loads {loads.shape}: [steps, 2] are needed")
    dt, every = float(dt), int(every)
    if not (dt > 0 and every >= 1):
        raise LookError(f"physics step {dt!r}, poses every {every!r} steps")
    n, steps = len(poses), len(loads)
    if n and (n - 1) * every >= steps:  # pose j is physics step j * every, and the last load is step steps - 1
        raise LookError(f"{n} poses every {every} steps reach past the {steps} physics steps of the loads")
    d = mujoco.MjData(m)
    K = np.zeros((n, 4))
    for k in range(n):
        K[k] = _kinematics(m, d, ids, poses[k])
    w = float(m.body_subtreemass[ids["trunk"]]) * G
    contact = readings.contact_flags(loads, w, dt) if steps else np.zeros((0, 2), bool)
    lo, td = _events(contact, dt, float(t0))
    return dict(t50=float(t0) + np.arange(n) * every * dt, hip_mid_z=K[:, 0].copy(), head_pitch_world=K[:, 1].copy(),
                foot_pitch=K[:, 2:4].copy(), tail=poses[:, ids["tail"]].copy(), contact=contact, lift_offs=lo,
                touchdowns=td, dt=dt, every=every, steps=steps, weight=w)


# ------------------------------------------------------------------------------------------------
# The four numbers over a window
# ------------------------------------------------------------------------------------------------
def _fit_phase(phi, y) -> tuple[float, float]:
    """Least-squares ``y = A sin(2 pi phi + psi) + c``: (A, psi) (``look._fit_phase``'s form)."""
    X = np.column_stack([np.sin(2 * np.pi * phi), np.cos(2 * np.pi * phi), np.ones(len(y))])
    a, b, _ = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(math.hypot(a, b)), float(math.atan2(b, a))


def own_phase(t, left_lift_offs) -> np.ndarray:
    """The own phase at times ``t`` (strides): 0 at the first of ``left_lift_offs``, one more at each later one,
    linear in time between two of them; NaN before the first and after the last."""
    t, L = np.asarray(t, float), np.asarray(left_lift_offs, float)
    out = np.full(len(t), np.nan)
    if len(L) < 2:
        return out
    ok = (t >= L[0] - EPS_S) & (t <= L[-1] + EPS_S)
    i = np.clip(np.searchsorted(L, t[ok], side="right") - 1, 0, len(L) - 2)
    out[ok] = i + (t[ok] - L[i]) / (L[i + 1] - L[i])
    return out


def reading(ser: dict, ta: float, tb: float) -> dict:
    """The four numbers of rule U1 over the simulated times ``[ta, tb]`` of a series (``series``' value, or any
    dict with ``t50``, ``hip_mid_z``, ``head_pitch_world``, ``foot_pitch``, ``tail``, ``lift_offs``,
    ``touchdowns``).  Only what lies inside the window is read.

    ``bob_pp_m`` (None without a step), ``head_world_pitch_rms_deg`` (None with fewer than two poses), ``tail``
    ``{phase_cycles, amp_rad, strides}`` (``phase_cycles`` in (-0.5, 0.5]; None with fewer than two left lift-offs
    in the window), ``toe_off_rad`` (None without a lift-off), ``steps`` (the steps wholly inside the window),
    ``lift_offs`` (those of either foot inside it), ``samples`` and ``window``."""
    ta, tb = float(ta), float(tb)
    t = np.asarray(ser["t50"], float)
    z, hp = np.asarray(ser["hip_mid_z"], float), np.asarray(ser["head_pitch_world"], float)
    tail, pitch = np.asarray(ser["tail"], float), np.asarray(ser["foot_pitch"], float)

    def inside(x):
        x = np.asarray(x, float)
        return x[(x >= ta - EPS_S) & (x <= tb + EPS_S)]

    lo = [inside(x) for x in ser["lift_offs"]]
    td = np.sort(np.concatenate([inside(x) for x in ser["touchdowns"]]))
    sel = (t >= ta - EPS_S) & (t <= tb + EPS_S)
    out = dict(window=[ta, tb], samples=int(sel.sum()), steps=max(0, len(td) - 1), lift_offs=int(len(lo[0]) + len(lo[1])))
    # bob: per own step, the range of the hip midpoint's height
    ranges = []
    for a, b in zip(td[:-1], td[1:], strict=True):
        v = np.concatenate([[np.interp(a, t, z)], z[(t > a) & (t < b)], [np.interp(b, t, z)]]) if len(t) else []
        if len(v):
            ranges.append(float(np.max(v) - np.min(v)))
    out["bob_pp_m"] = float(np.mean(ranges)) if ranges else None
    # head: RMS about the mean
    out["head_world_pitch_rms_deg"] = (float(math.degrees(math.sqrt(np.mean((hp[sel] - hp[sel].mean()) ** 2))))
                                       if out["samples"] >= 2 else None)
    # tail: the fitted sinusoid on the own phase
    phi = own_phase(t, lo[0])
    fit = np.isfinite(phi) & sel
    out["tail"] = dict(phase_cycles=None, amp_rad=None, strides=max(0, len(lo[0]) - 1))
    if fit.sum() >= 3:
        amp, psi = _fit_phase(phi[fit], tail[fit])
        out["tail"].update(phase_cycles=psi / (2 * math.pi), amp_rad=amp)
    # toe-off: the lifting foot's pitch at each lift-off
    at = [float(np.interp(x, t, pitch[:, j])) for j in (0, 1) for x in lo[j]] if len(t) else []
    out["toe_off_rad"] = float(np.mean(at)) if at else None
    return out


# ------------------------------------------------------------------------------------------------
# The terms
# ------------------------------------------------------------------------------------------------
def _num(x) -> float | None:
    return float(x) if x is not None and math.isfinite(float(x)) else None


def wrap_cycles(x: float) -> float:
    """A phase difference (cycles) wrapped to [-0.5, 0.5)."""
    return float((x + 0.5) % 1.0 - 0.5)


def compare(own: dict, ref: dict) -> dict:
    """The four terms of rule U3 for a reading ``own`` against the reference's reading ``ref`` (both ``reading``'s
    values), with rule U5's too few steps.

    ``terms {bob, head, tail, toe_off}``; ``ok`` (all four); ``not_read`` (the terms the reference's own values
    leave unread: they pass); ``too_few_steps``: the reference has at least ``MIN_STEPS`` steps and lift-offs and
    the robot has fewer of either: ``bob``, ``tail`` and ``toe_off`` are then False (also one that ``not_read``
    names) and ``head`` is still read.  ``LookError`` (a definition fault, which gate Z2 (b) must have caught)
    when the reference itself has fewer, or when one of its own values is missing or not finite: its bob, its tail
    amplitude, its tail phase where the tail is read, its toe-off.  A term is never left unread for want of the
    reference's value: only for a value under ``NOT_READ_RAD``.
    Beside them ``bob_ratio``, ``tail_phase_diff_cycles``, ``toe_off_ratio`` (None where a value of ``own`` is
    missing, or the reference's is not above zero), ``tail_amp_ratio`` (recorded: no band reads the tail's
    amplitude) and ``reference`` (its tail amplitude and toe-off, and the side of ``NOT_READ_RAD`` each lies on).
    A value of ``own`` that is missing fails its term.  Limits are compared without slack."""
    if int(ref["steps"]) < MIN_STEPS or int(ref["lift_offs"]) < MIN_STEPS:
        raise LookError(f"the reference's own window holds {ref['steps']} steps and {ref['lift_offs']} lift-offs: at "
                        f"least {MIN_STEPS} of each are needed (STAGE2_TRACKB_PLAN rule U5; gate Z2 (b))")
    too_few = int(own["steps"]) < MIN_STEPS or int(own["lift_offs"]) < MIN_STEPS
    terms, not_read = {}, []
    ob, rb = _num(own.get("bob_pp_m")), _num(ref.get("bob_pp_m"))
    oh = _num(own.get("head_world_pitch_rms_deg"))
    ot, rt = own.get("tail") or {}, ref.get("tail") or {}
    r_amp, r_ph, o_ph = _num(rt.get("amp_rad")), _num(rt.get("phase_cycles")), _num(ot.get("phase_cycles"))
    oo, ro = _num(own.get("toe_off_rad")), _num(ref.get("toe_off_rad"))
    tail_read = r_amp is not None and not r_amp < NOT_READ_RAD
    toe_read = ro is not None and not ro < NOT_READ_RAD
    missing = [name for name, v in (("bob_pp_m", rb), ("tail.amp_rad", r_amp), ("toe_off_rad", ro)) if v is None]
    if tail_read and r_ph is None:
        missing.append("tail.phase_cycles")
    if missing:
        raise LookError(f"the reference's own reading holds no finite {', '.join(missing)} (a definition fault: "
                        "STAGE2_TRACKB_PLAN rule U5; gate Z2 (b) asks for the four numbers of every playback)")
    # bob
    ratio = ob / rb if (ob is not None and rb > 0.0) else None
    terms["bob"] = bool(ratio is not None and abs(ratio - 1.0) <= BANDS["bob_rel"])
    # head
    terms["head"] = bool(oh is not None and oh <= BANDS["head_world_pitch_rms_deg"])
    # tail
    diff = wrap_cycles(o_ph - r_ph) if (o_ph is not None and r_ph is not None) else None
    if tail_read:
        terms["tail"] = bool(diff is not None and abs(diff) <= BANDS["tail_phase_cycles"])
    else:
        terms["tail"] = True
        not_read.append("tail")
    # toe-off
    if toe_read:
        terms["toe_off"] = bool(oo is not None and oo >= BANDS["toe_off_rel"] * ro)
    else:
        terms["toe_off"] = True
        not_read.append("toe_off")
    if too_few:
        for k in STEP_TERMS:
            terms[k] = False
    o_amp = _num(ot.get("amp_rad"))

    def side(v):
        return f"under {NOT_READ_RAD:g} rad" if v < NOT_READ_RAD else f"at or over {NOT_READ_RAD:g} rad"

    return dict(terms={k: terms[k] for k in TERMS}, ok=all(terms[k] for k in TERMS), not_read=not_read,
                too_few_steps=bool(too_few), bob_ratio=ratio, head_world_pitch_rms_deg=oh, tail_phase_diff_cycles=diff,
                tail_amp_ratio=(o_amp / r_amp if (o_amp is not None and r_amp > 0.0) else None),
                toe_off_ratio=(oo / ro if (oo is not None and ro > 0.0) else None),
                steps=dict(own=int(own["steps"]), reference=int(ref["steps"]), needed=MIN_STEPS),
                lift_offs=dict(own=int(own["lift_offs"]), reference=int(ref["lift_offs"]), needed=MIN_STEPS),
                reference=dict(tail_amp_rad=r_amp, tail_amp_side=side(r_amp), toe_off_rad=ro, toe_off_side=side(ro),
                               not_read_under_rad=NOT_READ_RAD),
                bands=dict(BANDS))


# ------------------------------------------------------------------------------------------------
# The reference's values: kinematic playback of a track
# ------------------------------------------------------------------------------------------------
def played(p, track, run_dir, ta_ref: float, tb_ref: float, rate: float = RATE) -> dict:
    """A track's own poses and planned loads over the reference times ``[ta_ref, tb_ref]`` at the clock ``rate``:
    ``poses`` at ``ta_ref, ta_ref + POSE_S * rate, ...`` by ``bar._Played(track, run_dir).q`` (the files' frames
    placed in one world frame), ``loads`` at every physics step by ``_Played.loads`` (the files' support flags), and
    ``dt``, ``every``, ``t0`` (= ``ta_ref / rate``: the simulated time of the first pose on that clock), ``rate``,
    ``window_ref``.  ``LookError("not played: ...")`` for a track ``_Played`` refuses (a periodic turning gait)."""
    from . import bar  # not at module level: the bar is no part of this module's import

    ta_ref, tb_ref, rate = float(ta_ref), float(tb_ref), float(rate)
    if not (tb_ref > ta_ref and rate > 0):
        raise LookError(f"reference times [{ta_ref!r}, {tb_ref!r}] at the clock {rate!r}")
    try:
        play = bar._Played(track, run_dir)
    except ValueError as e:
        raise LookError(f"not played: {e}") from e
    dt = float(p.m.opt.timestep)
    every = int(round(POSE_S / dt))
    if abs(every * dt - POSE_S) > 1e-12:
        raise LookError(f"the physics step {dt!r} s does not divide the poses' {POSE_S:g} s")
    w = weight(p)
    steps = int(math.floor((tb_ref - ta_ref) / (rate * dt) + 1e-9)) + 1
    n = (steps - 1) // every + 1
    t_last = float(track.t_end) - 1e-9 if getattr(track, "t_end", None) is not None else math.inf
    poses = np.array([play.q(min(ta_ref + rate * j * every * dt, t_last)) for j in range(n)], float)
    loads = np.array([play.loads(min(ta_ref + rate * k * dt, t_last), w) for k in range(steps)], float).reshape(steps, 2)
    return dict(poses=poses, loads=loads, dt=dt, every=every, t0=ta_ref / rate, rate=rate, window_ref=[ta_ref, tb_ref])


def playback(p, track, run_dir, ta_ref: float, tb_ref: float) -> dict:
    """The reference's reading (rule U5): ``reading`` of ``series`` of the track's own poses and planned loads over
    the reference times ``[ta_ref, tb_ref]`` at the nominal commanded clock (``played`` at ``RATE``), so that the
    result is on simulated time: its ``window`` is ``[ta_ref / RATE, tb_ref / RATE]``.  With ``window_ref`` and
    ``clock_rate`` beside the reading's keys.  ``LookError("not played: ...")`` as ``played``."""
    a = played(p, track, run_dir, ta_ref, tb_ref)
    ser = series(p, a["poses"], a["loads"], a["dt"], a["every"], t0=a["t0"])
    return dict(reading(ser, a["t0"], float(tb_ref) / RATE), window_ref=a["window_ref"], clock_rate=RATE)


# ------------------------------------------------------------------------------------------------
# The window of a bar item
# ------------------------------------------------------------------------------------------------
def window(b, rec) -> tuple[float, float] | None:
    """Rule U2's window (simulated seconds) of a built bar condition ``b`` and its episode's recorder ``rec``, or
    None: the item is not read (a nudge, a stop, a stock condition) or its window was not reached.

    A walk item (``b.cond.judge == "bar_walk"``): the last ``WALK_WINDOW_S`` of an episode that ran to its planned
    end without a fall, ``(t_end - 10, t_end)`` with ``t_end`` the trace's last update (the bar's own window and
    its own test of the end, ``bar._ran_to_end``).  An episode that fell has none, also when it fell inside the
    last two updates, which the test of the end lets through: the bar judges no window after a fall
    (``bar._judge``).  The fall is the recorder's own ``rec.out["fell"]`` (``metrics.Recorder.end``); a recorder
    without ``out`` (a kinematic playback's) did not fall.  A turn item (``"bar_turn"``): ``(t_c + at_s, t_c + at_s
    + TURN_WINDOW_S)`` with ``t_c = rec.sim_time(b.command["ref_s"])`` and ``at_s = b.resolved["at_s"]``, when the
    trace spans it (also in an episode that fell later: the bar reads that window too).  Only attributes of ``b``
    and ``rec`` are read."""
    judge = getattr(getattr(b, "cond", None), "judge", None)
    if judge not in ("bar_walk", "bar_turn") or not getattr(rec, "rows", 0):
        return None
    t_end = float(rec.t[-1])
    if judge == "bar_walk":
        from ..stage1 import control  # (the update interval: the bar's slack at the planned end)

        if (getattr(rec, "out", None) or {}).get("fell"):
            return None
        planned = float(b.resolved.get("cut_from_s", b.seconds))
        if t_end < planned - 2 * control.FEEDBACK_EVERY * float(rec.dt):
            return None
        return (t_end - WALK_WINDOW_S, t_end)
    if getattr(b, "command", None) is None:
        return None
    t_c = rec.sim_time(float(b.command["ref_s"]))
    if t_c is None:
        return None
    ta = float(t_c) + float(b.resolved["at_s"])
    tb = ta + TURN_WINDOW_S
    return None if t_end < tb - EPS_S else (ta, tb)


def window_ref(b) -> tuple[float, float] | None:
    """The reference times (the track's own clock, s) that rule U2's window of a built bar condition ``b`` covers
    at the nominal commanded clock, from the condition alone, or None (an item that is not read, or a cut walk
    episode, which has no window): the ``ta_ref, tb_ref`` that ``playback`` takes for that item (rule U5), the same
    for every episode of the condition whatever its own clock did.

    A walk item: ``RATE`` times the window of an episode that runs to its planned end, ``[t_end - WALK_WINDOW_S,
    t_end]`` with ``t_end`` the last update of ``round(b.seconds / dt)`` physics steps (``control._episode``; an
    episode starts with the track's clock at 0).  A turn item: ``[ref_s + RATE * at_s, ref_s + RATE * (at_s +
    TURN_WINDOW_S)]`` with ``ref_s`` the command's reference time: on the nominal clock the simulated window
    ``[t_c + at_s, t_c + at_s + TURN_WINDOW_S]`` covers these times.  Only attributes of ``b`` are read."""
    judge = getattr(getattr(b, "cond", None), "judge", None)
    if judge == "bar_walk":
        from ..stage1 import control  # (the update interval)

        if "cut_from_s" in b.resolved:
            return None
        dt = float(b.p.m.opt.timestep)
        n = int(round(float(b.seconds) / dt))
        if n < 1:
            return None
        t_end = ((n - 1) // control.FEEDBACK_EVERY) * control.FEEDBACK_EVERY * dt
        return (RATE * (t_end - WALK_WINDOW_S), RATE * t_end)
    if judge == "bar_turn" and getattr(b, "command", None) is not None:
        ta = float(b.command["ref_s"]) + RATE * float(b.resolved["at_s"])
        return (ta, ta + RATE * TURN_WINDOW_S)
    return None


def sources() -> dict:
    """``{path: sha256}`` of this file (its path relative to the study directory)."""
    path = Path(__file__).resolve()
    return {s2io.portable(str(path)): s2io.sha256_file(path)}


# ------------------------------------------------------------------------------------------------
# Gate G9-ref (b): the look ladder's own numbers
# ------------------------------------------------------------------------------------------------
#: Gate Z2 (b)'s limits between a playback's numbers and the look ladder's own: bob within 5 %, head pitch RMS
#: within 0.1 deg, tail phase within 0.01 cycle, toe-off within 0.02 rad and within 10 % of the ladder's number.
LADDER_LIMITS = dict(bob_rel=0.05, head_deg=0.1, tail_cycles=0.01, toe_off_rad=0.02, toe_off_rel=0.10)


def ladder_numbers(rung: dict) -> dict:
    """The four numbers of one rung of ``stage2/look/metrics.json`` (``bodies[body][speed][variant]``) in
    ``reading``'s units: the bob per step (``bob.per_step_pp_mm``, there per half stride of the gait's phase), the
    head's world pitch RMS, the tail's phase against the stride clock in cycles (``tail.phase_rad / 2 pi``; phase 0
    at the left foot's planned lift-off) with its amplitude, and the toe-off at lift-off."""
    return dict(bob_pp_m=1e-3 * float(rung["bob"]["per_step_pp_mm"]),
                head_world_pitch_rms_deg=float(rung["head"]["world_pitch_rms_deg"]),
                tail=dict(phase_cycles=float(rung["tail"]["phase_rad"]) / (2 * math.pi), amp_rad=float(rung["tail"]["amp_rad"])),
                toe_off_rad=float(rung["toe_off"]["at_lift_off_rad"]))


def against_ladder(read: dict, ladder: dict, toe_off: bool = True) -> dict:
    """Gate Z2 (b): a playback's reading against the ladder's numbers (``ladder_numbers``) within
    ``LADDER_LIMITS``.  ``ladder`` may also be another playback's reading, which has the same four keys: the plain
    body's for an ``-X`` body, the V6 set's for ``V6noF5``.  ``toe_off=False`` leaves the toe-off out (a set
    without F5: recorded "not read").  Returns the differences, ``terms {bob, head, tail, toe_off}`` and ``ok``."""
    L = LADDER_LIMITS
    ob, oh, oo = _num(read.get("bob_pp_m")), _num(read.get("head_world_pitch_rms_deg")), _num(read.get("toe_off_rad"))
    o_ph = _num((read.get("tail") or {}).get("phase_cycles"))
    lb, lh, lo = _num(ladder.get("bob_pp_m")), _num(ladder.get("head_world_pitch_rms_deg")), _num(ladder.get("toe_off_rad"))
    l_ph = _num((ladder.get("tail") or {}).get("phase_cycles"))
    d = dict(bob_rel=None if ob is None or lb is None or not lb > 0 else ob / lb - 1.0,
             head_deg=None if oh is None or lh is None else oh - lh,
             tail_cycles=None if o_ph is None or l_ph is None else wrap_cycles(o_ph - l_ph),
             toe_off_rad=None if oo is None or lo is None else oo - lo)
    d["toe_off_rel"] = None if (d["toe_off_rad"] is None or not lo > 0) else d["toe_off_rad"] / lo
    terms = dict(bob=d["bob_rel"] is not None and abs(d["bob_rel"]) <= L["bob_rel"],
                 head=d["head_deg"] is not None and abs(d["head_deg"]) <= L["head_deg"],
                 tail=d["tail_cycles"] is not None and abs(d["tail_cycles"]) <= L["tail_cycles"])
    if toe_off:
        terms["toe_off"] = bool(d["toe_off_rad"] is not None and abs(d["toe_off_rad"]) <= L["toe_off_rad"]
                                and d["toe_off_rel"] is not None and abs(d["toe_off_rel"]) <= L["toe_off_rel"])
    terms = {k: bool(v) for k, v in terms.items()}
    return dict(differences=d, terms=terms, ok=all(terms.values()), toe_off="read" if toe_off else "not read",
                limits=dict(L), ladder=ladder)


# ------------------------------------------------------------------------------------------------
# Gate G9-ref (c): negative controls on a playback
# ------------------------------------------------------------------------------------------------
#: The degraded copies of a playback and the one term each must fail.
DEGRADED = dict(bob_half="bob", neck_3deg="head", tail_late="tail", flat_feet="toe_off")
NECK_DEG, NECK_HZ = 3.0, 1.0
TAIL_LATE_CYCLES = 0.2


def degrade(p, poses, kind: str, pose_s: float = POSE_S, period_s: float | None = None) -> np.ndarray:
    """A degraded copy of ``poses [n, nq]`` (``DEGRADED``): ``bob_half`` scales the root height's deviation from
    its mean by 0.5; ``neck_3deg`` adds a ``NECK_DEG`` degree sinusoid at ``NECK_HZ`` to the neck-pitch joint;
    ``tail_late`` delays the tail-yaw joint's angle by ``TAIL_LATE_CYCLES`` of ``period_s`` (the stride on the
    poses' clock; the first value is held before it); ``flat_feet`` turns each ankle-pitch joint until its foot's
    pitch is zero.  Contact is not touched: the loads stay the playback's."""
    m = p.m
    ids = _ids(m)
    q = np.array(poses, float)
    t = np.arange(len(q)) * float(pose_s)
    if kind == "bob_half":
        q[:, 2] = q[:, 2].mean() + 0.5 * (q[:, 2] - q[:, 2].mean())
    elif kind == "neck_3deg":
        q[:, ids["neck"]] += math.radians(NECK_DEG) * np.sin(2 * math.pi * NECK_HZ * t)
    elif kind == "tail_late":
        if period_s is None or not period_s > 0:
            raise LookError("tail_late needs the stride period on the poses' clock (period_s)")
        q[:, ids["tail"]] = np.interp(t - TAIL_LATE_CYCLES * float(period_s), t, q[:, ids["tail"]])
    elif kind == "flat_feet":
        d = mujoco.MjData(m)
        for k in range(len(q)):
            for j in (0, 1):
                a = ids["ankles"][j]
                th = _kinematics(m, d, ids, q[k])[2 + j]
                q[k, a] += 1e-3
                slope = (_kinematics(m, d, ids, q[k])[2 + j] - th) / 1e-3
                q[k, a] -= 1e-3
                for _ in range(8):
                    if abs(th) < 1e-12:
                        break
                    q[k, a] -= th / slope
                    th = _kinematics(m, d, ids, q[k])[2 + j]
    else:
        raise LookError(f"degraded copy {kind!r}: one of {tuple(DEGRADED)}")
    return q


def controls(p, poses, loads, dt, every, t0: float = 0.0, ta: float | None = None, tb: float | None = None,
             kinds=tuple(DEGRADED)) -> dict:
    """Gate Z2 (c) on a playback's arrays (``played``): the unchanged playback against itself passes all four terms
    with ratio 1 and difference 0; each degraded copy of ``kinds`` (``DEGRADED``) fails exactly its term; the
    window cut to 3 steps reads ``too_few_steps``.  ``ok`` holds when all of that does.  The window is ``[ta, tb]``
    (default: all of the series).  A term the reference leaves unread (``not_read``) cannot fail, so its control is
    not ok: on a set without toe-off the gate leaves ``flat_feet`` out of ``kinds``."""
    unknown = [k for k in kinds if k not in DEGRADED]
    if unknown:
        raise LookError(f"degraded copies {unknown}: of {tuple(DEGRADED)}")
    ser = series(p, poses, loads, dt, every, t0=t0)
    ta = float(ser["t50"][0]) if ta is None else float(ta)
    tb = float(t0) + (ser["steps"] - 1) * ser["dt"] if tb is None else float(tb)
    ref = reading(ser, ta, tb)
    same = compare(ref, ref)
    exact = dict(bob=same["bob_ratio"] == 1.0,
                 tail="tail" in same["not_read"] or same["tail_phase_diff_cycles"] == 0.0,
                 toe_off="toe_off" in same["not_read"] or same["toe_off_ratio"] == 1.0)
    out = dict(window=[ta, tb], reference=ref, unchanged=dict(compare=same, exact=exact,
                                                               ok=bool(same["ok"] and all(exact.values()))))
    left = [x for x in ser["lift_offs"][0] if ta - EPS_S <= x <= tb + EPS_S]
    period = float(np.mean(np.diff(left))) if len(left) >= 2 else None
    out["stride_s"] = period
    out["degraded"] = {}
    for kind in kinds:
        term = DEGRADED[kind]
        own = reading(series(p, degrade(p, poses, kind, every * float(dt), period), loads, dt, every, t0=t0), ta, tb)
        c = compare(own, ref)
        failed = [k for k in TERMS if not c["terms"][k]]
        out["degraded"][kind] = dict(own=own, compare=c, failed=failed, expected=[term], ok=failed == [term])
    td = np.sort(np.concatenate([x[(x >= ta - EPS_S) & (x <= tb + EPS_S)] for x in ser["touchdowns"]]))
    cut = reading(ser, ta, float(td[MIN_STEPS - 1])) if len(td) >= MIN_STEPS else None
    c = None if cut is None else compare(cut, ref)
    out["cut"] = dict(own=cut, compare=c, ok=bool(c is not None and cut["steps"] == MIN_STEPS - 1 and c["too_few_steps"]
                                                   and not any(c["terms"][k] for k in STEP_TERMS)))
    out["ok"] = bool(out["unchanged"]["ok"] and out["cut"]["ok"] and all(v["ok"] for v in out["degraded"].values()))
    return out
