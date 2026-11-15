"""The bar's readings as pure functions of arrays (numpy and ``scipy.signal`` only: no MuJoCo, no Stage 1 import).

``bar.py`` calls these on a live recorder (``metrics.Recorder``); tests call them on synthetic arrays; a trace that
holds the contact bits and the feet's 100 Hz low points can be read again with them.  This file is hashed into every
bar record's key.  It measures; except where a function is given its tolerance as an argument, the caller applies
the thresholds (the constants below name them).

Arrays and clocks.  ``t_u`` are the times of the 100 Hz update rows (simulated seconds); ``contact`` is bool
``[steps, 2]`` per physics step (left, right), already debounced (``contact_flags``); ``dt`` is the physics step.
By the recorder's convention a row's time is the start of the physics step just taken and a step's loads are read
after it, so physics step ``a`` and a row with ``t = a dt`` are one instant: a step is put on the rows' clock as
``(a + STEP_OFFSET) dt``.  Times are compared with ``EPS_S`` of slack, far below a physics step.

(a) ``toe_clearance_own_swing``: per foot, an *air run* is a maximal run of steps out of contact that starts after
a contact step (lift-off step a, ``t_lo``) and ends at a contact step (touchdown step b, ``t_td``); ``D = t_td -
t_lo``.  It is a **swing** if any update row with ``t_lo <= t < t_td`` has ``ref_swing`` equal to that foot,
whatever its length, or if ``D >= air_gap_max_s``; otherwise it is an *air gap* (stance chatter): counted, not
read.  A *reference swing* is a maximal run of rows with ``ref_swing`` equal to the foot; a swing belongs to every
reference swing it shares such a row with, and a reference swing with two or more swings is **interrupted**.  A
swing wholly inside the window (``t_lo >= ta``, ``t_td <= tb``) is read as the minimum of the foot's low point over
its value interpolated linearly between rows at ``t_lo + 0.25 D`` and at ``t_lo + 0.75 D`` and every row strictly
between those times.  Choices this file makes where the rule's text leaves room, each stated in the function's
docstring: which reference swings are "in the window" (those wholly inside it and complete in the trace), and that
air gaps are counted when wholly inside the window.

(b) ``line_reading``, (c) ``turn_reading``, (e) ``stop_reading`` and ``touchdowns``: the walk's, the turn's and the
stop's readings.  ``drive_ratios``: each drive's RMS over its sustained rating over a span of steps, with the worst
leg drive and the worst neck or tail drive apart.  ``reference_heading_at`` and ``reference_readings``: what a
reference's own file can say (no toe clearance: a file holds no toe points).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import butter, filtfilt

#: Slack of every time comparison (s).
EPS_S = 1e-9
#: Physics step ``a`` is the instant ``(a + STEP_OFFSET) * dt`` on the update rows' clock (module docstring).
STEP_OFFSET = 0
#: Contact, as Stage 1 reads it: a foot carrying more than this share of the weight, runs shorter than
#: ``DEBOUNCE_S`` merged into their neighbours (``control.CONTACT_LOAD_FRAC``, ``metrics.DEBOUNCE_S``).
CONTACT_LOAD_FRAC = 0.02
DEBOUNCE_S = 0.02
#: The panels' commanded clock rate and the tracker's phase-rate band (``panels.RATE``,
#: ``control.PHASE_RATE_LIMIT``): the bar's clock tau is the reference at ``RATE * tau`` on the nominal clock and at
#: ``(1 - PHASE_RATE_LIMIT) * RATE * tau`` on the slowest.
RATE = 1.05
PHASE_RATE_LIMIT = 0.05
#: Decision 10's filter (``metrics.ACCEL_LP_HZ``, ``ACCEL_LP_ORDER``) and the rate a reference is resampled at for it.
ACCEL_LP_HZ = 1.0
ACCEL_LP_ORDER = 2
RESAMPLE_HZ = 500.0
#: Thresholds of the bar (the callers apply them unless a function takes one as an argument).
TOE_CLEARANCE_FRAC_H = 0.04  # own-swing toe clearance, of the hip height
MID_SWING = (0.25, 0.75)  # the middle half of a swing
WALK_WINDOW_S = 10.0  # the walk item's judged window
LINE_MAX_M = 0.25  # the CoM's largest distance from the line along the commanded heading
HEADING_TOL_DEG = 10.0
HEADING_HOLD_S = 3.0
STOP_SPEED_FRAC = 0.05  # stopped: forward speed below this share of the commanded speed from then on
STOP_TOUCHDOWNS_MAX = 2
ACCEL_MAX_M_S2 = 1.5  # decision 10
REFERENCE_HEADING_MIN_DEG = 80.0  # a turn reference's heading at the bar's instant (G9)
#: Classes of the per-swing table: read; counted and not read; and the same two for an air run that the window cuts.
SWING_CLASSES = ("swing", "air_gap", "swing_cut", "air_gap_cut")


# ------------------------------------------------------------------------------------------------
# Filters and contact
# ------------------------------------------------------------------------------------------------
def lowpass(x, fs: float, hz: float = ACCEL_LP_HZ, order: int = ACCEL_LP_ORDER) -> np.ndarray:
    """Zero-phase Butterworth low-pass (forward and backward); a series too short to filter is returned as its
    mean.  Identical to ``designlab.stage1.metrics.lowpass``."""
    b, a = butter(order, hz, fs=fs)
    if len(x) <= 3 * max(len(a), len(b)):
        return np.full(len(x), float(np.mean(x)) if len(x) else 0.0)
    return filtfilt(b, a, x)


def _runs(flag: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs of equal values: starts, ends (exclusive) and values."""
    flag = np.asarray(flag, bool)
    if not len(flag):
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0, bool)
    change = np.flatnonzero(flag[1:] != flag[:-1]) + 1
    s = np.r_[0, change]
    e = np.r_[change, len(flag)]
    return s, e, flag[s]


def debounce(flag, min_len: int) -> np.ndarray:
    """``flag`` with every inner run shorter than ``min_len`` merged into its neighbours, shortest first (the
    first of equally short ones).  Identical to ``designlab.stage1.metrics.debounce``."""
    f = np.asarray(flag, bool).copy()
    while True:
        s, e, v = _runs(f)
        inner = np.flatnonzero((e - s < min_len) & (s > 0) & (e < len(f)))
        if not len(inner):
            return f
        i = inner[np.argmin((e - s)[inner])]
        f[s[i] : e[i]] = not v[i]


def contact_flags(loads, weight: float, dt: float, load_frac: float = CONTACT_LOAD_FRAC,
                  debounce_s: float = DEBOUNCE_S) -> np.ndarray:
    """Per physics step and foot: in contact, from the feet's vertical loads ``[steps, 2]`` (N).  Identical to
    ``metrics.Recorder.contact``."""
    raw = np.asarray(loads, float) > load_frac * weight
    k = max(1, int(round(debounce_s / dt)))
    return np.column_stack([debounce(raw[:, i], k) for i in range(raw.shape[1])])


def air_runs(contact_i, dt: float) -> list[tuple[int, int]]:
    """The complete air runs of one foot as (lift-off step a, touchdown step b): steps ``a .. b - 1`` are out of
    contact, step ``a - 1`` and step ``b`` are in contact.  A run the trace cuts at either end is not complete and
    is left out.  (``dt`` only has to be a positive step; the steps do not depend on it.)"""
    if not dt > 0:
        raise ValueError(f"physics step {dt!r}")
    c = np.asarray(contact_i, bool)
    if c.ndim != 1:
        raise ValueError("air_runs takes one foot's contact flags")
    s, e, v = _runs(c)
    return [(int(a), int(b)) for a, b, x in zip(s, e, v, strict=True) if not x and a > 0 and b < len(c)]


def touchdowns(contact, dt: float) -> list[tuple[float, int]]:
    """(time s, foot) of every touchdown, in time order: a physics step a with ``contact[a, i]`` and not
    ``contact[a - 1, i]``, at ``(a + STEP_OFFSET) dt``.  Step 0 is never one."""
    c = np.asarray(contact, bool)
    out = []
    for i in range(c.shape[1]):
        for a in np.flatnonzero(c[1:, i] & ~c[:-1, i]) + 1:
            out.append(((int(a) + STEP_OFFSET) * dt, i))
    return sorted(out)


# ------------------------------------------------------------------------------------------------
# (a) Toe clearance over each foot's own swing
# ------------------------------------------------------------------------------------------------
def toe_clearance_own_swing(t_u, low, contact, dt: float, ref_swing, ta: float, tb: float, air_gap_max_s: float,
                            mid=MID_SWING) -> dict:
    """The own-swing toe clearance over the window ``[ta, tb]`` (module docstring, (a)).

    ``low`` is ``[rows, 2]`` (each foot's lowest point above the floor at the update rows), ``ref_swing`` is
    ``[rows]`` (the reference's swing foot at each row: 0, 1 or -1), ``air_gap_max_s`` half the shortest planned
    swing of the track on the simulated clock.  Returns

    * ``toe_clearance_own_swing_m``: the smallest reading of the swings wholly inside the window (None: none);
    * ``own_swings`` [2]: per foot, the swings wholly inside the window (the ones read);
    * ``ref_swings`` [2]: per foot, the reference swings wholly inside the window: every row of the run within
      ``[ta, tb]``, and the run complete in the trace (it neither starts at the first row nor ends at the last);
    * ``interrupted_swings``: those reference swings, over both feet, that hold two or more swings (a swing counts
      for a reference swing whether or not the window cuts the swing itself);
    * ``air_gaps``, ``air_gap_longest_s``: the air gaps wholly inside the window and the longest of them (0.0);
    * ``air_gap_max_s``: as given;
    * ``swings``: ``[foot, t_lo, t_td, min_m, class]`` for every complete air run that overlaps the window, in time
      order; ``class`` is one of ``SWING_CLASSES`` and ``min_m`` is None unless the class is ``"swing"``;
    * ``count_ok``: per foot ``own_swings >= ref_swings - 1``.

    The term (``toe_clearance_own_swing_ok``) needs the reading at or above 0.04 H, no interrupted swing and
    ``count_ok``; this function does not apply the 0.04 H threshold."""
    t = np.asarray(t_u, float)
    low = np.asarray(low, float)
    contact = np.asarray(contact, bool)
    rs = np.asarray(ref_swing, float)
    if low.shape != (len(t), 2) or rs.shape != (len(t),) or contact.ndim != 2 or contact.shape[1] != 2:
        raise ValueError(f"shapes: t_u {t.shape}, low {low.shape}, ref_swing {rs.shape}, contact {contact.shape}")
    if not np.all(np.isfinite(rs)):
        raise ValueError("ref_swing holds a value that is not finite")
    if not 0.0 <= mid[0] < mid[1] <= 1.0:
        raise ValueError(f"mid {mid!r}")
    rs = np.rint(rs).astype(int)
    table, readings = [], []
    own, refs, gaps, interrupted = [0, 0], [0, 0], [], 0
    for i in (0, 1):
        s, e, v = _runs(rs == i)
        ref_runs = [(int(a), int(b)) for a, b, x in zip(s, e, v, strict=True) if x]
        label = np.full(len(t), -1)
        for j, (a, b) in enumerate(ref_runs):
            label[a:b] = j
        held = [0] * len(ref_runs)
        for a, b in air_runs(contact[:, i], dt):
            t_lo, t_td = (a + STEP_OFFSET) * dt, (b + STEP_OFFSET) * dt
            D = t_td - t_lo
            rows = np.flatnonzero((t >= t_lo - EPS_S) & (t < t_td - EPS_S))
            mine = np.unique(label[rows][label[rows] >= 0])
            swing = bool(len(mine)) or D >= air_gap_max_s - EPS_S
            inside = t_lo >= ta - EPS_S and t_td <= tb + EPS_S
            if swing:
                for j in mine:
                    held[j] += 1
            if not (t_td > ta + EPS_S and t_lo < tb - EPS_S):
                continue
            value = None
            if swing and inside:
                t1, t2 = t_lo + mid[0] * D, t_lo + mid[1] * D
                between = low[(t > t1) & (t < t2), i]
                value = float(min(np.interp(t1, t, low[:, i]), np.interp(t2, t, low[:, i]),
                                  np.min(between) if len(between) else math.inf))
                readings.append(value)
                own[i] += 1
            elif inside:
                gaps.append(D)
            cls = ("swing" if swing else "air_gap") + ("" if inside else "_cut")
            table.append([i, float(t_lo), float(t_td), value, cls])
        for j, (a, b) in enumerate(ref_runs):
            if a > 0 and b < len(t) and t[a] >= ta - EPS_S and t[b - 1] <= tb + EPS_S:
                refs[i] += 1
                interrupted += held[j] >= 2
    table.sort(key=lambda r: (r[1], r[0]))
    return dict(toe_clearance_own_swing_m=min(readings) if readings else None, own_swings=own, ref_swings=refs,
                interrupted_swings=int(interrupted), air_gaps=len(gaps),
                air_gap_longest_s=float(max(gaps)) if gaps else 0.0, air_gap_max_s=float(air_gap_max_s), swings=table,
                count_ok=bool(all(own[i] >= refs[i] - 1 for i in (0, 1))))


def toe_clearance_own_swing_ok(reading: dict, H: float, frac: float = TOE_CLEARANCE_FRAC_H) -> bool:
    """The term ``toe_clearance_own_swing`` on ``toe_clearance_own_swing``'s result: a swing was read, the smallest
    reading is at least ``frac`` of the hip height ``H``, no reference swing in the window is interrupted, and each
    foot has at least its number of reference swings less one."""
    v = reading["toe_clearance_own_swing_m"]
    return bool(v is not None and v >= frac * H and reading["interrupted_swings"] == 0 and reading["count_ok"])


# ------------------------------------------------------------------------------------------------
# (b) The walk item's line
# ------------------------------------------------------------------------------------------------
def line_reading(t_u, com_xy, heading: float, ta: float, tb: float, c_episode_start=None) -> dict:
    """The CoM against the line along the commanded heading (``heading``, rad) over the window ``[ta, tb]``.

    ``c0`` is the CoM at the first update at or after ``ta`` (``anchor_s``, ``anchor_xy``);
    ``along_commanded_heading_m = (c(tb) - c0) . h`` with ``c(tb)`` interpolated at ``tb`` (the last update of a
    complete episode); ``lateral_from_line_max_m`` is the largest ``|(c(t) - c0) x h|`` over the window's updates;
    ``lateral_end_m`` is the offset at ``tb``, positive to the left of the heading;
    ``lateral_from_episode_start_line_max_m`` is the same largest distance from the line through
    ``c_episode_start`` (reported; None without it).  ``complete`` is false, and the values None, when the window
    holds fewer than two updates."""
    t = np.asarray(t_u, float)
    c = np.asarray(com_xy, float)
    h = np.array([math.cos(heading), math.sin(heading)])
    left = np.array([-h[1], h[0]])
    sel = (t >= ta - EPS_S) & (t <= tb + EPS_S)
    out = dict(t_start_s=float(ta), t_end_s=float(tb), updates=int(sel.sum()))
    if out["updates"] < 2:
        return dict(out, complete=False, anchor_s=None, anchor_xy=None, along_commanded_heading_m=None,
                    lateral_from_line_max_m=None, lateral_end_m=None, lateral_from_episode_start_line_max_m=None)
    k0 = int(np.flatnonzero(sel)[0])
    c0 = c[k0]
    c_end = np.array([np.interp(tb, t, c[:, 0]), np.interp(tb, t, c[:, 1])])
    out.update(complete=True, anchor_s=float(t[k0]), anchor_xy=[float(c0[0]), float(c0[1])],
               along_commanded_heading_m=float((c_end - c0) @ h),
               lateral_from_line_max_m=float(np.max(np.abs((c[sel] - c0) @ left))),
               lateral_end_m=float((c_end - c0) @ left), lateral_from_episode_start_line_max_m=None)
    if c_episode_start is not None:
        cs = np.asarray(c_episode_start, float)[:2]
        out["lateral_from_episode_start_line_max_m"] = float(np.max(np.abs((c[sel] - cs) @ left)))
    return out


# ------------------------------------------------------------------------------------------------
# (c) The turn item's heading
# ------------------------------------------------------------------------------------------------
def turn_reading(t_u, yaw_unwrapped, t_c: float, psi_cmd: float, sign: float, angle: float, at_s: float,
                 hold_s: float = HEADING_HOLD_S, tol_deg: float = HEADING_TOL_DEG) -> dict:
    """The trunk's heading against the commanded turn: ``sign`` +1 left, -1 right; ``angle`` rad (pi/2); the turn
    is read ``at_s`` after its command ``t_c`` (``(pi/2) / |rate|``) and held ``hold_s`` after.

    ``psi`` is ``yaw_unwrapped`` moved by whole turns so that its first row is within half a turn of the commanded
    heading ``psi_cmd`` (every plan starts from the home stance); ``err = psi - psi_cmd - sign * angle``.
    ``heading_error_at_deg`` is ``err`` interpolated at ``t_c + at_s``; ``heading_error_hold_max_deg`` the largest
    ``|err|`` over the updates in ``[t_c + at_s, t_c + at_s + hold_s]``; ``heading`` (the term): ``|err|`` within
    ``tol_deg`` at ``t_c + at_s`` and at every update of that interval; ``time_to_within_10deg_s``: the time after
    the command of the first update in ``[t_c, t_c + at_s + hold_s]`` from which every update up to the interval's
    end is within ``tol_deg`` (None: the last one is not).  Reported: ``turned_own_deg = psi(t_c + at_s) -
    psi(t_c)`` (positive to the left) and ``trunk_heading_at_command_deg = psi(t_c) - psi_cmd``.
    ``window_reached`` is false when the trace does not span ``[t_c, t_c + at_s + hold_s]``: the term is then
    false and what cannot be read is None."""
    t = np.asarray(t_u, float)
    psi = np.asarray(yaw_unwrapped, float).copy()
    t_at, t_end = t_c + at_s, t_c + at_s + hold_s
    out = dict(command_s=float(t_c), at_s=float(at_s), hold_s=float(hold_s), tol_deg=float(tol_deg),
               angle_deg=float(math.degrees(angle)), sign=float(sign), window_reached=False, heading=False,
               heading_error_at_deg=None, heading_error_hold_max_deg=None, time_to_within_10deg_s=None,
               turned_own_deg=None, trunk_heading_at_command_deg=None)
    if not len(t):
        return out
    psi -= 2.0 * math.pi * round((psi[0] - psi_cmd) / (2.0 * math.pi))
    err = np.degrees(psi - psi_cmd - sign * angle)
    if t[0] > t_c + EPS_S or t[-1] < t_c - EPS_S:
        return out
    psi_c = float(np.interp(t_c, t, psi))
    out["trunk_heading_at_command_deg"] = math.degrees(psi_c - psi_cmd)
    if t[-1] < t_at - EPS_S:
        return out
    err_at = float(np.interp(t_at, t, err))
    out["heading_error_at_deg"] = err_at
    out["turned_own_deg"] = math.degrees(float(np.interp(t_at, t, psi)) - psi_c)
    if t[-1] < t_end - EPS_S:
        return out
    hold = (t >= t_at - EPS_S) & (t <= t_end + EPS_S)
    worst = float(np.max(np.abs(err[hold]))) if hold.any() else abs(err_at)
    out.update(window_reached=True, heading_error_hold_max_deg=worst,
               heading=bool(abs(err_at) <= tol_deg and worst <= tol_deg))
    span = np.flatnonzero((t >= t_c - EPS_S) & (t <= t_end + EPS_S))
    within = np.abs(err[span]) <= tol_deg
    if len(span) and within[-1]:
        off = np.flatnonzero(~within)
        first = span[off[-1] + 1] if len(off) else span[0]
        out["time_to_within_10deg_s"] = max(0.0, float(t[first] - t_c))
    return out


# ------------------------------------------------------------------------------------------------
# (e) The stop item
# ------------------------------------------------------------------------------------------------
def stop_reading(t_u, com_xy, v_xy, contact, dt: float, t_c: float, heading: float, command_speed: float,
                 stride_s: float) -> dict:
    """The stop after the command at ``t_c``: ``heading`` the commanded heading (rad), ``command_speed`` the
    commanded speed on the simulated clock (0.40 m/s x 1.05) and ``stride_s`` the commanded stride there.

    ``v_f = v_xy . h`` at the updates.  ``t_stop`` is the first update time at or after ``t_c`` from which ``|v_f|
    < 0.05 command_speed`` holds to the end of the trace (``stopped_by_end`` false, ``forward_stopped_s`` None:
    never).  ``touchdowns_after_command`` counts the touchdowns of both feet with ``t_c < t <= t_stop``; while the
    body never stops the count runs to the end of the trace (it is then ``touchdowns_to_end``).  Reported:
    ``touchdown_table`` (``[foot, t, t - t_c]`` of each counted touchdown), ``last_touchdown_before_command``
    (``{foot, t_s, before_command_s}`` of the last touchdown at or before ``t_c``: a late landing of the earlier
    step shows as a counted touchdown and a long ``before_command_s``), ``touchdowns_to_end``,
    ``forward_stopped_strides``, ``distance_after_command_m`` (along the heading, to ``t_stop`` or the end), and
    over the updates in ``[t_c, t_c + 2 stride_s]``: ``deceleration_raw_peak_m_s2``, the largest ``-d v_f / dt``
    (central differences of the CoM's forward speed, unfiltered), and ``com_accel_forward_lp1hz_peak_m_s2``, the
    largest ``|d v_f / dt|`` after ``lowpass`` (the CoM's twin of the judged IMU reading, which the caller takes
    from ``metrics.acceleration_peak``).  ``reached_command`` is false, and the readings None, when the trace
    ends before ``t_c``."""
    t = np.asarray(t_u, float)
    c = np.asarray(com_xy, float)
    h = np.array([math.cos(heading), math.sin(heading)])
    vf = np.asarray(v_xy, float) @ h
    thr = STOP_SPEED_FRAC * command_speed
    out = dict(command_s=float(t_c), command_speed_m_s=float(command_speed), stride_s=float(stride_s),
               speed_frac=STOP_SPEED_FRAC, reached_command=False, stopped_by_end=False, forward_stopped_s=None,
               forward_stopped_strides=None, touchdowns_after_command=None, touchdown_table=[],
               touchdowns_to_end=None, last_touchdown_before_command=None, distance_after_command_m=None,
               deceleration_raw_peak_m_s2=None, com_accel_forward_lp1hz_peak_m_s2=None)
    after = np.flatnonzero(t >= t_c - EPS_S)
    if not len(after) or t[0] > t_c + EPS_S:
        return out
    k0 = int(after[0])
    fast = np.flatnonzero(np.abs(vf[k0:]) >= thr)
    if not len(fast):
        t_stop = float(t[k0])
    elif fast[-1] == len(t) - 1 - k0:
        t_stop = None
    else:
        t_stop = float(t[k0 + fast[-1] + 1])
    tds = touchdowns(contact, dt)
    later = [(x, i) for x, i in tds if x > t_c + EPS_S]
    counted = [(x, i) for x, i in later if t_stop is None or x <= t_stop + EPS_S]
    before = [(x, i) for x, i in tds if x <= t_c + EPS_S]
    t_far = t_stop if t_stop is not None else float(t[-1])
    c_cmd = np.array([np.interp(t_c, t, c[:, 0]), np.interp(t_c, t, c[:, 1])])
    c_far = np.array([np.interp(t_far, t, c[:, 0]), np.interp(t_far, t, c[:, 1])])
    out.update(reached_command=True, stopped_by_end=t_stop is not None,
               forward_stopped_s=None if t_stop is None else t_stop - t_c,
               forward_stopped_strides=None if t_stop is None else (t_stop - t_c) / stride_s,
               touchdowns_after_command=len(counted),
               touchdown_table=[[int(i), float(x), float(x - t_c)] for x, i in counted],
               touchdowns_to_end=len(later),
               last_touchdown_before_command=(dict(foot=int(before[-1][1]), t_s=float(before[-1][0]),
                                                   before_command_s=float(t_c - before[-1][0])) if before else None),
               distance_after_command_m=float((c_far - c_cmd) @ h))
    span = (t >= t_c - EPS_S) & (t <= t_c + 2.0 * stride_s + EPS_S)
    if len(t) >= 3 and span.any():
        af = np.gradient(vf, t)
        out["deceleration_raw_peak_m_s2"] = float(np.max(-af[span]))
        lp = lowpass(af, 1.0 / float(np.median(np.diff(t))))
        out["com_accel_forward_lp1hz_peak_m_s2"] = float(np.max(np.abs(lp[span])))
    return out


# ------------------------------------------------------------------------------------------------
# The three readings' drive term
# ------------------------------------------------------------------------------------------------
def drive_ratios(force_sq, sustained, actuators, leg_mask, k0: int, k1: int) -> dict:
    """Each drive's RMS force over the physics steps ``k0 .. k1 - 1`` over its sustained rating:
    ``sqrt(mean(force_sq[k0:k1], 0)) / sustained`` (``metrics.window``'s form).  ``leg_mask`` is true for a leg
    drive and false for a neck or tail drive (A1's).  Returns ``ratios {actuator: r}``, ``worst [actuator, r]``,
    ``worst_leg``, ``worst_a1`` (None when the mask leaves none), the terms ``drive_sustained`` (every drive at or
    under 1) and ``drive_sustained_legs`` (every leg drive), and ``steps [k0, k1]``."""
    fs = np.asarray(force_sq, float)[k0:k1]
    if not len(fs):
        raise ValueError(f"no physics step in [{k0}, {k1})")
    r = np.sqrt(np.mean(fs, axis=0)) / np.asarray(sustained, float)
    names = [str(a) for a in actuators]
    leg = np.asarray(leg_mask, bool)
    if not (len(names) == len(r) == len(leg)):
        raise ValueError(f"{len(r)} drives, {len(names)} actuators, {len(leg)} mask entries")

    def worst(mask):
        idx = np.flatnonzero(mask)
        if not len(idx):
            return None
        j = int(idx[np.argmax(r[idx])])
        return [names[j], float(r[j])]

    return dict(ratios={n: float(x) for n, x in zip(names, r, strict=True)}, worst=worst(np.ones(len(r), bool)),
                worst_leg=worst(leg), worst_a1=worst(~leg), drive_sustained=bool(np.all(r <= 1.0)),
                drive_sustained_legs=bool(np.all(r[leg] <= 1.0)), steps=[int(k0), int(k1)])


# ------------------------------------------------------------------------------------------------
# What a reference's own file can say
# ------------------------------------------------------------------------------------------------
def reference_heading_at(arrays: dict, tau_s: float, rate: float) -> float:
    """The heading of a transition-format file (``t``, ``heading``) at its own time ``rate * tau_s``, less its
    first heading, in degrees (positive to the left).  ``tau_s`` is bar time since the command, which is the file's
    first frame; ``rate`` is the clock's (``RATE`` nominal).  Past the file's last frame the last heading holds."""
    t = np.asarray(arrays["t"], float)
    h = np.asarray(arrays["heading"], float)
    return math.degrees(float(np.interp(rate * tau_s, t, h)) - float(h[0]))


def _forward_acceleration(t, com_xy, heading) -> tuple[np.ndarray, np.ndarray]:
    """The model CoM's acceleration along the per-frame heading: low-passed at 1 Hz on a uniform ``RESAMPLE_HZ``
    resample of the file's own frames (linear, no wrap), and unfiltered on the frames themselves."""
    tu = np.arange(t[0], t[-1], 1.0 / RESAMPLE_HZ)
    x, y, hd = (np.interp(tu, t, s) for s in (com_xy[:, 0], com_xy[:, 1], heading))
    ax, ay = (np.gradient(np.gradient(s, tu), tu) for s in (x, y))
    lp = lowpass(ax * np.cos(hd) + ay * np.sin(hd), RESAMPLE_HZ)
    a = np.gradient(np.gradient(com_xy, t, axis=0), t, axis=0)
    return lp, a[:, 0] * np.cos(heading) + a[:, 1] * np.sin(heading)


def reference_readings(arrays: dict, kind: str, walk: dict | None = None) -> dict:
    """What a move's own file (transition format: ``t``, ``com``, ``heading``, ``support``) says of the bar's
    readings, on the reference's clock unless a key says otherwise.  No toe clearance: a file holds no toe points
    (the set's screen has it).  ``walk`` gives the numbers of the move's own walk and command that the file does
    not hold: ``speed_m_s`` and ``period_s`` (or ``speed``, ``period``, as an entry's ``gaits.a``) for a stop,
    ``duty`` optionally; ``at_s`` or ``rate_rad_s`` for a turn.

    Every kind: ``accel_forward_lp1hz_peak_m_s2``, the largest ``|a|`` of the model CoM's acceleration along the
    heading after the 1 Hz filter (on a uniform 500 Hz resample).

    ``"stop"``: ``touchdowns_after_command`` (touchdowns after frame 0 in the planned ``support``, with
    ``touchdown_s`` and ``touchdown_feet``; frame 0 is the command of a causal move), ``forward_stopped_s`` (the
    first frame time from which the model CoM's speed along the heading stays under 5 % of the walk's; None: the
    last frame is not under it) and ``forward_stopped_strides``, ``accel_forward_raw_peak_m_s2`` (the largest
    ``|a|`` unfiltered, on the file's own frames less two at each end), ``margin_before_command_s`` (``(duty -
    0.5) period``: how long before the command the earlier step is planned to land; None without ``duty``).

    ``"turn"``: ``heading_total_rad`` (last less first), ``at_s``, and in the turn's own direction
    ``heading_at_bar_deg`` (at bar time ``at_s`` on the nominal clock: the file's time ``RATE at_s``),
    ``heading_at_slowest_clock_deg`` (``(1 - PHASE_RATE_LIMIT) RATE at_s``), ``heading_at_ref_clock_deg``
    (``at_s`` of the file's own time), ``time_to_80deg_bar_s`` (bar time, nominal clock, at which the heading
    first reaches 80 deg; None: never) and ``heading_ok`` (both the nominal and the slowest clock's reading at
    least 80 deg, G9's rule)."""
    t = np.asarray(arrays["t"], float)
    com = np.asarray(arrays["com"], float)[:, :2]
    heading = np.asarray(arrays["heading"], float)
    walk = dict(walk or {})
    lp, raw = _forward_acceleration(t, com, heading)
    out = dict(kind=kind, accel_forward_lp1hz_peak_m_s2=float(np.max(np.abs(lp))))
    if kind == "stop":
        speed = walk.get("speed_m_s", walk.get("speed"))
        period = walk.get("period_s", walk.get("period"))
        if speed is None or period is None:
            raise ValueError("a stop's readings need the walk's speed_m_s and period_s")
        sup = np.asarray(arrays["support"], bool)
        tds = sorted((float(t[k]), i) for i in range(sup.shape[1]) for k in np.flatnonzero(sup[1:, i] & ~sup[:-1, i]) + 1)
        v = np.gradient(com, t, axis=0)
        vf = v[:, 0] * np.cos(heading) + v[:, 1] * np.sin(heading)
        fast = np.flatnonzero(np.abs(vf) >= STOP_SPEED_FRAC * float(speed))
        if not len(fast):
            t_stop = float(t[0])
        elif fast[-1] == len(t) - 1:
            t_stop = None
        else:
            t_stop = float(t[fast[-1] + 1])
        duty = walk.get("duty")
        out.update(touchdowns_after_command=len(tds), touchdown_s=[x for x, _ in tds],
                   touchdown_feet=[int(i) for _, i in tds], forward_stopped_s=t_stop,
                   forward_stopped_strides=None if t_stop is None else t_stop / float(period),
                   accel_forward_raw_peak_m_s2=float(np.max(np.abs(raw[2:-2]))),
                   margin_before_command_s=None if duty is None else (float(duty) - 0.5) * float(period))
    elif kind == "turn":
        if walk.get("at_s") is not None:
            at_s = float(walk["at_s"])
        elif walk.get("rate_rad_s") is not None:
            at_s = (math.pi / 2.0) / abs(float(walk["rate_rad_s"]))
        else:
            raise ValueError("a turn's readings need at_s or rate_rad_s (the file does not hold its commanded rate)")
        total = float(heading[-1] - heading[0])
        sgn = 1.0 if total >= 0 else -1.0
        turned = sgn * np.degrees(heading - heading[0])
        k = np.flatnonzero(turned >= REFERENCE_HEADING_MIN_DEG)
        if not len(k):
            t80 = None
        elif k[0] == 0:
            t80 = 0.0
        else:
            j = int(k[0])
            t80 = float(np.interp(REFERENCE_HEADING_MIN_DEG, turned[j - 1 : j + 1], t[j - 1 : j + 1])) / RATE
        bar = sgn * reference_heading_at(arrays, at_s, RATE)
        slow = sgn * reference_heading_at(arrays, at_s, (1.0 - PHASE_RATE_LIMIT) * RATE)
        out.update(heading_total_rad=total, at_s=at_s, heading_at_bar_deg=bar, heading_at_slowest_clock_deg=slow,
                   heading_at_ref_clock_deg=sgn * reference_heading_at(arrays, at_s, 1.0), time_to_80deg_bar_s=t80,
                   heading_ok=bool(bar >= REFERENCE_HEADING_MIN_DEG and slow >= REFERENCE_HEADING_MIN_DEG))
    return out
