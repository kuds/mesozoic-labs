"""The quality of a standing episode's stance, from its centre-of-mass trajectory: pure numpy, report-only.

A stance can survive its horizon and still be a poor one: leaning to one side, drifting toward a support
edge, turning or twisting on its feet, sagging, or rocking.  The survival and keyframe tests
(``static_balance_helpers``) cannot see any of that.  :func:`stance_track_quality` reduces one episode's
per-sample arrays (the whole-body centre of mass, the root orientation, one reference point per foot and,
where known, the loaded support points, the per-foot normal loads and the foot headings) to named numbers over
a window, and :func:`stance_track_series` gives the per-sample series behind them, so a failing check can name
the step.  The arrays are plain and the module imports nothing but numpy, so the same reduction reads a
simulated statue (``tests.com_stance_helpers.EnvStanceSampler`` builds its :class:`StanceTrack`), an evaluation
episode the gait recorder already captured (:func:`track_from_episode_trace`), a trained policy's rollout or a
robot's state-estimate log.  Nothing here is a gate criterion, a reward or an observation, and no digest reads
it.  The key names are this module's own: ``gait.stance_metrics`` has gate keys that read similar quantities
over other windows (its ``min_foot_load_share`` is the lighter foot's share of the window's summed load; here
:data:`LOAD_KEYS`' ``lightest_foot_share_min`` is the least per-sample share of any foot).

Frames.  World z is up and the floor is the plane ``z = floor_z``.  Each sample's heading frame has its forward
axis along the root's x axis projected on the floor and its left axis that one turned +90 deg about z, and the
CoM's offsets are taken from the centroid of the foot reference points: the rule of
``balance_telemetry.com_fore_aft_mean_m`` and of the T. rex lateral-lean reward's ``_com_lateral_offset``.  So
the numbers do not change when the whole track is turned about z or moved, which is what lets a robot log,
whose world frame is arbitrary, be read with the same function.  The extrapolated centre of mass (XcoM) is
``xi = c_xy + v_xy / omega0`` with ``omega0 = sqrt(g / (c_z - floor_z))`` (Hof 2008, as in
``balance_telemetry``); without a measured CoM velocity it is differentiated from the positions.

Reading a robot's log.  Express every sample in one gravity-aligned frame (z up, the floor a horizontal plane
at ``floor_z``; a state estimator's odometry frame, or the stance feet's frame from forward kinematics), give
the root's attitude with its x axis forward and its y axis left (rotate an IMU mounted otherwise into that
convention first: nothing here can tell a wrongly mounted IMU from a lean), and drop or interpolate missing
samples first: a non-finite CoM, attitude or foot point is refused.  Without a world pose estimate (feet and CoM
from forward kinematics in a frame fixed to the feet), the world-yaw keys read the yaw of that frame, and the
foot travel keys read zero.  Without foot force sensing, leave ``foot_load`` and ``support`` out: the load keys
and the support keys are then absent, not wrong.

Signs: lateral + left, fore-aft + ahead, roll + left side up, pitch + nose up, yaw + counter-clockwise seen
from above; a drift is the least-squares line's change across the window.  Angles are reported in degrees and
everything else in SI units.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

NAN = float("nan")
#: The fewest window samples :func:`stance_track_quality` reduces.
MIN_WINDOW_SAMPLES = 10
#: The length of the window's last stretch that the ``_end_`` keys read: still turning, still moving.
END_S = 2.0
#: Foot labels on the animal's left (bipeds ``l``; quadrupeds ``fl`` and ``rl``); front feet start with ``f``.
LEFT_LABELS: tuple[str, ...] = ("l", "fl", "rl")

#: The keys every track gives: the CoM, the root orientation and the foot points alone.
KINEMATIC_KEYS: tuple[str, ...] = (
    "com_lateral_mean_m",
    "com_lateral_drift_m",
    "root_roll_mean_deg",
    "root_roll_drift_deg",
    "com_fore_aft_mean_m",
    "com_fore_aft_drift_m",
    "root_pitch_mean_deg",
    "com_height_mean_m",
    "com_height_drift_m",
    "root_turn_deg",
    "root_turn_end_deg",
    "root_turn_from_spawn_deg",
    "foot_travel_max_m",
    "foot_travel_from_spawn_max_m",
    "com_speed_rms_mps",
    "com_speed_end_rms_mps",
    "com_lateral_sd_m",
    "com_fore_aft_sd_m",
    "com_height_sd_m",
    "root_roll_sd_deg",
    "root_pitch_sd_deg",
    "root_yaw_sd_deg",
)
#: The keys of a track that carries its loaded support points: the least margin of the CoM and the XcoM inside
#: the support polygon to its nearest edge, and along the heading frame's lateral and fore-aft axes.
SUPPORT_KEYS: tuple[str, ...] = (
    "support_defined_fraction",
    "com_margin_min_m",
    "xcom_margin_min_m",
    "com_margin_lateral_min_m",
    "xcom_margin_lateral_min_m",
    "com_margin_fore_aft_min_m",
    "xcom_margin_fore_aft_min_m",
)
#: The keys of a track that carries its foot headings.
HEADING_KEYS: tuple[str, ...] = (
    "foot_turn_max_deg",
    "foot_turn_from_spawn_max_deg",
    "foot_splay_change_max_deg",
    "root_minus_feet_yaw_from_spawn_deg",
)
#: The keys of a track that carries its per-foot loads (``front_load_share_mean`` on four feet only).
LOAD_KEYS: tuple[str, ...] = ("left_load_share_mean", "lightest_foot_share_min")

#: Which stance fault each key reads: off its support, leaning or drifting sideways, off its fore-aft posture,
#: sagging, turning or twisting on its feet, or rocking.
FAMILIES: dict[str, tuple[str, ...]] = {
    "support": (
        "com_margin_min_m",
        "xcom_margin_min_m",
        "com_margin_lateral_min_m",
        "xcom_margin_lateral_min_m",
        "com_margin_fore_aft_min_m",
        "xcom_margin_fore_aft_min_m",
    ),
    "lean": (
        "com_lateral_mean_m",
        "root_roll_mean_deg",
        "left_load_share_mean",
        "lightest_foot_share_min",
        "com_lateral_drift_m",
        "root_roll_drift_deg",
    ),
    "sagittal": ("com_fore_aft_mean_m", "com_fore_aft_drift_m", "root_pitch_mean_deg", "front_load_share_mean"),
    "height": ("com_height_mean_m", "com_height_drift_m"),
    "pivot": (
        "root_turn_deg",
        "root_turn_end_deg",
        "root_turn_from_spawn_deg",
        "root_minus_feet_yaw_from_spawn_deg",
        "foot_turn_max_deg",
        "foot_turn_from_spawn_max_deg",
        "foot_splay_change_max_deg",
        "foot_travel_max_m",
        "foot_travel_from_spawn_max_m",
    ),
    "sway": (
        "com_speed_rms_mps",
        "com_speed_end_rms_mps",
        "com_lateral_sd_m",
        "com_fore_aft_sd_m",
        "com_height_sd_m",
        "root_roll_sd_deg",
        "root_pitch_sd_deg",
        "root_yaw_sd_deg",
    ),
}
#: Each support key's per-sample series in :func:`stance_track_series`.
SUPPORT_SERIES: dict[str, str] = {
    "com_margin_min_m": "com_margin",
    "xcom_margin_min_m": "xcom_margin",
    "com_margin_lateral_min_m": "com_margin_lateral",
    "xcom_margin_lateral_min_m": "xcom_margin_lateral",
    "com_margin_fore_aft_min_m": "com_margin_fore_aft",
    "xcom_margin_fore_aft_min_m": "xcom_margin_fore_aft",
}


@dataclass(frozen=True)
class StanceTrack:
    """One standing episode as per-sample arrays: T samples, F feet, SI units, world frame with z up.

    Sample 0 is the spawn (after the reset) and sample k the state after control step k.  ``support``,
    ``foot_load`` and ``foot_heading`` are optional: a key that reads one exists only when it is given.  A
    non-finite ``foot_load`` row is a sample with no load reading (it is skipped).
    """

    #: (T,) s.
    t: np.ndarray
    #: (T, 3) m: the whole-body centre of mass.
    com: np.ndarray
    #: (T, 4) w, x, y, z: the root body's orientation (MuJoCo's ``xquat`` order; an IMU's attitude quaternion).
    root_quat: np.ndarray
    #: (T, F, 3) m: one reference point per foot.
    foot_pos: np.ndarray
    #: The F foot labels (``l``, ``r`` or ``fl``, ``fr``, ``rl``, ``rr``).
    labels: tuple[str, ...]
    #: Per sample, the (N, 2) loaded support points, in any order.
    support: Sequence[np.ndarray] | None = None
    #: (T, F) N: each foot's normal load from the floor.
    foot_load: np.ndarray | None = None
    #: (T, F) rad: each foot's heading about world z.
    foot_heading: np.ndarray | None = None
    #: (T, 3) m/s; None differentiates ``com`` over ``t`` (central differences).
    com_vel: np.ndarray | None = None
    floor_z: float = 0.0
    gravity: float = 9.81


def track_from_episode_trace(trace: Any, *, gravity: float = 9.81) -> StanceTrack:
    """A :class:`StanceTrack` from a ``gait.recorder.EpisodeTrace`` recorded with ``balance=True``.

    The trace's spawn sample (``spawn_com_pos``, ``spawn_root_quat``, ``spawn_foot_pos``) becomes sample 0 and
    its per-step boundary rows samples 1..T, at ``k * dt``.  The per-foot load is ``leg_floor_mean``, the limb's
    floor force averaged over the step's substeps (the spawn sample has none).  The trace has no contact points
    and no foot headings, so the support and heading keys are absent; the floor is the plane z = 0, as the
    recorder's.  Reads only attributes, so this module still imports nothing but numpy.
    """
    if getattr(trace, "com_pos", None) is None or getattr(trace, "spawn_com_pos", None) is None:
        raise ValueError("the trace has no centre of mass: record it with SubstepContactRecorder(..., balance=True)")
    steps = int(np.asarray(trace.com_pos).shape[0])
    feet = np.asarray(trace.foot_pos, dtype=np.float64)
    load = np.asarray(trace.leg_floor_mean, dtype=np.float64)
    return StanceTrack(
        t=np.arange(steps + 1, dtype=np.float64) * float(trace.dt),
        com=np.vstack([np.asarray(trace.spawn_com_pos, dtype=np.float64)[None], np.asarray(trace.com_pos)]),
        root_quat=np.vstack([np.asarray(trace.spawn_root_quat, dtype=np.float64)[None], np.asarray(trace.root_quat)]),
        foot_pos=np.concatenate([np.asarray(trace.spawn_foot_pos, dtype=np.float64)[None], feet], axis=0),
        labels=tuple(trace.labels),
        foot_load=np.vstack([np.full((1, load.shape[1]), np.nan), load]),
        gravity=gravity,
    )


# ── the support polygon ──────────────────────────────────────────────────────


def convex_hull(points: np.ndarray) -> np.ndarray:
    """(N, 2) points -> the counter-clockwise hull vertices (monotone chain; duplicates and collinear points dropped).

    Fewer than three distinct points come back as they are, sorted: a point or a segment has no area.
    """
    pts = sorted({(float(x), float(y)) for x, y in np.asarray(points, dtype=np.float64).reshape(-1, 2)})
    if len(pts) < 3:
        return np.asarray(pts, dtype=np.float64).reshape(-1, 2)

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64).reshape(-1, 2)


def _edge_distances(hull: np.ndarray, point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each edge's signed distance from *point* (+ inside) and its outward unit normal, for a CCW hull."""
    edges = np.roll(hull, -1, axis=0) - hull
    length = np.linalg.norm(edges, axis=1)
    rel = np.asarray(point, dtype=np.float64)[:2] - hull
    signed = (edges[:, 0] * rel[:, 1] - edges[:, 1] * rel[:, 0]) / length
    outward = np.stack([edges[:, 1], -edges[:, 0]], axis=1) / length[:, None]
    return signed, outward


def signed_margin(hull: np.ndarray, point: np.ndarray) -> float:
    """How far *point* lies inside a counter-clockwise hull (+ inside, - outside): the least signed edge distance.

    NaN when the hull has fewer than three vertices (a point or a line of contacts has no inside).  Outside a
    corner the value is the nearer edge line's distance, not the distance to the corner.
    """
    hull = np.asarray(hull, dtype=np.float64).reshape(-1, 2)
    if hull.shape[0] < 3:
        return NAN
    signed, _ = _edge_distances(hull, point)
    return float(signed.min())


def ray_margin(hull: np.ndarray, point: np.ndarray, direction: np.ndarray) -> float:
    """How far *point* can move along the unit *direction* before it crosses the hull's boundary (+ inside).

    The least ``s_i / (n_i . d)`` over the edges facing the direction (``n_i . d > 0``), ``s_i`` the signed
    distance to edge i and ``n_i`` its outward normal: the ray's exit distance from an inside point, and minus the
    distance back to the boundary from a point outside past a facing edge.  NaN without an area.
    """
    hull = np.asarray(hull, dtype=np.float64).reshape(-1, 2)
    if hull.shape[0] < 3:
        return NAN
    signed, outward = _edge_distances(hull, point)
    facing = outward @ np.asarray(direction, dtype=np.float64)[:2]
    ok = facing > 1e-12
    return float((signed[ok] / facing[ok]).min()) if ok.any() else NAN


def nearest_edge_side(hull: np.ndarray, point: np.ndarray, forward: np.ndarray) -> str:
    """Which side of the stance *point*'s nearest hull edge faces: ``ahead``, ``behind``, ``left`` or ``right``."""
    hull = np.asarray(hull, dtype=np.float64).reshape(-1, 2)
    if hull.shape[0] < 3:
        return "none (no support area)"
    signed, outward = _edge_distances(hull, point)
    normal = outward[int(np.argmin(signed))]
    f = np.asarray(forward, dtype=np.float64)[:2]
    along, across = float(normal @ f), float(normal @ np.array([-f[1], f[0]]))
    if abs(along) >= abs(across):
        return "ahead" if along > 0.0 else "behind"
    return "left" if across > 0.0 else "right"


# ── per-sample series ────────────────────────────────────────────────────────


def yaw_pitch_roll(root_quat: np.ndarray) -> np.ndarray:
    """(T, 4) w, x, y, z -> (T, 3) rad: yaw (the x axis's heading), pitch (+ x axis above horizontal), roll (+ left up)."""
    q = np.asarray(root_quat, dtype=np.float64).reshape(-1, 4)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r10 = 2.0 * (x * y + w * z)
    r20 = 2.0 * (x * z - w * y)
    r21 = 2.0 * (y * z + w * x)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    # The x axis in the world is R's first column, so R20 is the sine of its elevation (+ nose up).
    return np.stack([np.arctan2(r10, r00), np.arcsin(np.clip(r20, -1.0, 1.0)), np.arctan2(r21, r22)], axis=1)


def _circular_mean(angles: np.ndarray) -> np.ndarray:
    return np.asarray(np.arctan2(np.sin(angles).mean(axis=1), np.cos(angles).mean(axis=1)))


def stance_track_series(track: StanceTrack) -> dict[str, np.ndarray]:
    """The per-sample quantities the window reductions read, SI units, angles in radians.

    Always: ``t``, ``com_fore_aft``, ``com_lateral``, ``com_height``, ``com_speed`` (horizontal), ``root_yaw``
    (unwrapped), ``root_pitch``, ``root_roll``, ``forward`` (T, 2), ``foot_xy`` (T, F, 2) and ``xcom`` (T, 2).
    With support points: the :data:`SUPPORT_SERIES` margins (NaN where the support has no area).  With foot
    headings: ``foot_heading`` (T, F, unwrapped), ``foot_splay`` (each foot's heading less the feet's circular
    mean, unwrapped) and ``root_minus_feet_yaw`` (the root's yaw less the feet's circular-mean heading).  With
    loads: ``foot_load_share`` (T, F; NaN on an unloaded sample), ``left_load_share`` and, on feet with both
    front and rear labels, ``front_load_share``.  Raises ``ValueError`` on a non-finite CoM, attitude, foot
    point or time.
    """
    t = np.asarray(track.t, dtype=np.float64)
    com = np.asarray(track.com, dtype=np.float64)
    feet = np.asarray(track.foot_pos, dtype=np.float64)
    for name, array in (("t", t), ("com", com), ("root_quat", track.root_quat), ("foot_pos", feet)):
        if not np.isfinite(np.asarray(array, dtype=np.float64)).all():
            raise ValueError(f"the track's {name} has non-finite samples: drop or interpolate them first")
    if track.com_vel is not None:
        vel = np.asarray(track.com_vel, dtype=np.float64)
    elif t.size > 1:
        vel = np.gradient(com, t, axis=0)
    else:
        vel = np.zeros_like(com)
    ypr = yaw_pitch_roll(track.root_quat)
    forward = np.stack([np.cos(ypr[:, 0]), np.sin(ypr[:, 0])], axis=1)
    left = np.stack([-forward[:, 1], forward[:, 0]], axis=1)
    offset = com[:, :2] - feet[:, :, :2].mean(axis=1)
    height = com[:, 2] - track.floor_z
    with np.errstate(invalid="ignore", divide="ignore"):
        omega0 = np.where(height > 0.0, np.sqrt(track.gravity / np.where(height > 0.0, height, 1.0)), np.nan)
    xcom = com[:, :2] + vel[:, :2] / omega0[:, None]
    out: dict[str, np.ndarray] = {
        "t": t,
        "com_fore_aft": np.einsum("tj,tj->t", forward, offset),
        "com_lateral": np.einsum("tj,tj->t", left, offset),
        "com_height": height,
        "com_speed": np.linalg.norm(vel[:, :2], axis=1),
        "root_yaw": np.unwrap(ypr[:, 0]),
        "root_pitch": ypr[:, 1],
        "root_roll": ypr[:, 2],
        "forward": forward,
        "foot_xy": feet[:, :, :2],
        "xcom": xcom,
    }
    if track.support is not None:
        series = {name: np.full(t.size, np.nan) for name in SUPPORT_SERIES.values()}
        for i, points in enumerate(track.support):
            hull = convex_hull(points) if len(points) else np.zeros((0, 2))
            if hull.shape[0] < 3:
                continue
            for prefix, point in (("com", com[i, :2]), ("xcom", xcom[i])):
                signed, outward = _edge_distances(hull, point)
                series[f"{prefix}_margin"][i] = signed.min()
                for axis, direction in (("lateral", left[i]), ("fore_aft", forward[i])):
                    facing = outward @ direction
                    # The exit distance along +direction and along -direction, the nearer of the two.
                    ahead, behind = facing > 1e-12, facing < -1e-12
                    series[f"{prefix}_margin_{axis}"][i] = min(
                        (signed[ahead] / facing[ahead]).min() if ahead.any() else np.inf,
                        (signed[behind] / -facing[behind]).min() if behind.any() else np.inf,
                    )
        out.update(series)
    if track.foot_heading is not None:
        raw = np.asarray(track.foot_heading, dtype=np.float64)
        mean_heading = _circular_mean(raw)
        out["foot_heading"] = np.unwrap(raw, axis=0)
        out["foot_splay"] = np.unwrap(np.angle(np.exp(1j * (raw - mean_heading[:, None]))), axis=0)
        out["root_minus_feet_yaw"] = out["root_yaw"] - np.unwrap(mean_heading)
    if track.foot_load is not None:
        load = np.asarray(track.foot_load, dtype=np.float64)
        total = load.sum(axis=1)
        good = np.isfinite(total) & (total > 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            share = np.where(good[:, None], load / np.where(good, total, 1.0)[:, None], np.nan)
        out["foot_load_share"] = share
        left_mask = np.array([label in LEFT_LABELS for label in track.labels])
        front_mask = np.array([label.startswith("f") for label in track.labels])
        out["left_load_share"] = share[:, left_mask].sum(axis=1) if left_mask.any() else np.full(t.size, np.nan)
        if front_mask.any() and not front_mask.all():
            out["front_load_share"] = share[:, front_mask].sum(axis=1)
    return out


# ── window reductions ────────────────────────────────────────────────────────


def _slope(t: np.ndarray, y: np.ndarray) -> float:
    """The least-squares slope of *y* over *t* (finite samples only); NaN under three."""
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return NAN
    tt = t[ok] - t[ok].mean()
    den = float(tt @ tt)
    return float(tt @ (y[ok] - y[ok].mean()) / den) if den > 0.0 else NAN


def _detrended_sd(t: np.ndarray, y: np.ndarray) -> float:
    """The standard deviation of *y* about its least-squares line over *t*: the sway left once the drift is out."""
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return NAN
    coef = np.polyfit(t[ok] - t[ok][0], y[ok], 1)
    return float(np.std(y[ok] - np.polyval(coef, t[ok] - t[ok][0])))


def window_mask(t: np.ndarray, window_start_s: float) -> np.ndarray:
    """The samples at or after ``t[0] + window_start_s`` (1e-9 s tolerance)."""
    t = np.asarray(t, dtype=np.float64)
    return np.asarray(t >= t[0] + window_start_s - 1e-9)


def stance_track_quality(
    track: StanceTrack, *, window_start_s: float, series: Mapping[str, np.ndarray] | None = None
) -> dict[str, float]:
    """Named stance-quality numbers over the samples at or after ``t[0] + window_start_s`` (the window).

    :data:`KINEMATIC_KEYS` always; :data:`SUPPORT_KEYS`, :data:`HEADING_KEYS` and :data:`LOAD_KEYS` (and
    ``front_load_share_mean`` on four feet) when the track carries what they read.  Also ``window_samples``,
    ``window_s`` and ``end_s``.  NaN means "not measured this time" (no loaded sample, no support area).
    Raises ``ValueError`` on a window of fewer than :data:`MIN_WINDOW_SAMPLES` samples.  *series*, when given,
    is this track's :func:`stance_track_series` (computed once by a caller that also needs it).

    Lean and drift: the mean lateral CoM offset and root roll, and how far their least-squares lines move
    across the window.  Fore-aft posture: the mean fore-aft offset, its drift and the mean root pitch.  Height:
    the mean CoM height and its drift (- sagging).  Pivot: the root's turn across the window (signed, the yaw's
    last less first window sample), its fitted turn over the window's last :data:`END_S` s (still turning), its
    turn from the spawn, and the root's yaw relative to its feet, last sample less the spawn (the body twisted
    over its own feet); the largest turn of any foot over the window and from the spawn, the largest change from
    the spawn of any foot's heading relative to the feet's mean (a splay: toes turned out or in, invisible in
    the mean); and the largest horizontal travel of any foot point over the window and from the spawn.  Sway:
    the RMS horizontal CoM speed over the window and over its last :data:`END_S` s, and the spread about the
    trend of the lateral, fore-aft and vertical CoM and of the root's roll, pitch and yaw.  Support: the least
    CoM and XcoM margins inside the support polygon, to the nearest edge and along the lateral and fore-aft
    axes (so a lean shows even when the nearest edge is the toe), and the window fraction where it has an
    area.  Loads: the mean left share of the floor load, the least share any foot carries on a loaded sample,
    and the mean front share.
    """
    s = stance_track_series(track) if series is None else series
    t = s["t"]
    w = window_mask(t, window_start_s)
    if int(w.sum()) < MIN_WINDOW_SAMPLES:
        raise ValueError(f"the window holds {int(w.sum())} samples; at least {MIN_WINDOW_SAMPLES} are needed")
    tw = t[w]
    span = float(tw[-1] - tw[0])
    end = w & (t >= t[-1] - END_S - 1e-9)
    te = t[end]
    deg = math.degrees(1.0)
    out: dict[str, float] = {"window_samples": float(w.sum()), "window_s": span, "end_s": float(te[-1] - te[0])}

    def drift(y: np.ndarray) -> float:
        return _slope(tw, y[w]) * span

    out["com_lateral_mean_m"] = float(np.mean(s["com_lateral"][w]))
    out["com_lateral_drift_m"] = drift(s["com_lateral"])
    out["root_roll_mean_deg"] = float(np.mean(s["root_roll"][w]) * deg)
    out["root_roll_drift_deg"] = drift(s["root_roll"]) * deg
    out["com_fore_aft_mean_m"] = float(np.mean(s["com_fore_aft"][w]))
    out["com_fore_aft_drift_m"] = drift(s["com_fore_aft"])
    out["root_pitch_mean_deg"] = float(np.mean(s["root_pitch"][w]) * deg)
    out["com_height_mean_m"] = float(np.mean(s["com_height"][w]))
    out["com_height_drift_m"] = drift(s["com_height"])
    yaw = s["root_yaw"]
    out["root_turn_deg"] = float((yaw[w][-1] - yaw[w][0]) * deg)
    out["root_turn_end_deg"] = _slope(te, yaw[end]) * float(te[-1] - te[0]) * deg
    out["root_turn_from_spawn_deg"] = float((yaw[-1] - yaw[0]) * deg)
    foot_xy = s["foot_xy"]
    out["foot_travel_max_m"] = float(np.linalg.norm(foot_xy[w] - foot_xy[w][0], axis=2).max())
    out["foot_travel_from_spawn_max_m"] = float(np.linalg.norm(foot_xy[w] - foot_xy[0], axis=2).max())
    out["com_speed_rms_mps"] = float(np.sqrt(np.mean(s["com_speed"][w] ** 2)))
    out["com_speed_end_rms_mps"] = float(np.sqrt(np.mean(s["com_speed"][end] ** 2)))
    out["com_lateral_sd_m"] = _detrended_sd(tw, s["com_lateral"][w])
    out["com_fore_aft_sd_m"] = _detrended_sd(tw, s["com_fore_aft"][w])
    out["com_height_sd_m"] = _detrended_sd(tw, s["com_height"][w])
    out["root_roll_sd_deg"] = _detrended_sd(tw, s["root_roll"][w]) * deg
    out["root_pitch_sd_deg"] = _detrended_sd(tw, s["root_pitch"][w]) * deg
    out["root_yaw_sd_deg"] = _detrended_sd(tw, yaw[w]) * deg
    if "com_margin" in s:
        out["support_defined_fraction"] = float(np.mean(np.isfinite(s["com_margin"][w])))
        for key, name in SUPPORT_SERIES.items():
            margin = s[name][w]
            out[key] = float(np.nanmin(margin)) if np.isfinite(margin).any() else NAN
    if "foot_heading" in s:
        heading, splay = s["foot_heading"], s["foot_splay"]
        out["foot_turn_max_deg"] = float(np.abs(heading[w] - heading[w][0]).max() * deg)
        out["foot_turn_from_spawn_max_deg"] = float(np.abs(heading[w] - heading[0]).max() * deg)
        out["foot_splay_change_max_deg"] = float(np.abs(splay[w] - splay[0]).max() * deg)
        twist = s["root_minus_feet_yaw"]
        out["root_minus_feet_yaw_from_spawn_deg"] = float((twist[-1] - twist[0]) * deg)
    if "foot_load_share" in s:
        share = s["foot_load_share"][w]
        loaded = np.isfinite(share).all(axis=1)
        out["left_load_share_mean"] = float(np.mean(s["left_load_share"][w][loaded])) if loaded.any() else NAN
        out["lightest_foot_share_min"] = float(np.min(share[loaded])) if loaded.any() else NAN
        if "front_load_share" in s:
            out["front_load_share_mean"] = float(np.mean(s["front_load_share"][w][loaded])) if loaded.any() else NAN
    return out


def first_violation(t: np.ndarray, series: np.ndarray, *, below: float, window_start_s: float) -> int | None:
    """The index of the first window sample under *below* (or not finite), else None."""
    w = window_mask(t, window_start_s)
    bad = w & ~(np.asarray(series, dtype=np.float64) >= below)
    hits = np.flatnonzero(bad)
    return int(hits[0]) if hits.size else None


# ── across seeds ─────────────────────────────────────────────────────────────


def sign_test_p(positive: int, negative: int) -> float:
    """Two-sided sign-test p-value: the chance of a split at least this uneven if each sign had probability 1/2."""
    n = positive + negative
    if n == 0:
        return 1.0
    k = min(positive, negative)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0**n)


def lean_sign_summary(
    per_seed: Sequence[Mapping[str, float]],
    keys: Sequence[str] = ("com_lateral_mean_m", "root_roll_mean_deg"),
) -> dict[str, dict[str, float]]:
    """Whether a lean keeps one sign across seeds: the lean per-episode bars cannot see.

    A policy is not mirror-exact, so each of its episodes leans a little; a lean of the same sign on (nearly)
    every seed is a bias, however small each one is.  Per key: ``n``, ``positive``, ``negative``, ``mean``,
    ``sd`` (over seeds), ``same_sign_fraction`` (the larger sign's share) and ``sign_test_p`` (two-sided; 10 of
    10 one way is 0.002).  Zero and non-finite values count as neither sign.
    """
    out: dict[str, dict[str, float]] = {}
    for key in keys:
        values = np.array([float(metrics.get(key, NAN)) for metrics in per_seed], dtype=np.float64)
        values = values[np.isfinite(values)]
        positive, negative = int((values > 0.0).sum()), int((values < 0.0).sum())
        signed = positive + negative
        out[key] = {
            "n": float(values.size),
            "positive": float(positive),
            "negative": float(negative),
            "mean": float(values.mean()) if values.size else NAN,
            "sd": float(values.std(ddof=1)) if values.size > 1 else NAN,
            "same_sign_fraction": max(positive, negative) / signed if signed else NAN,
            "sign_test_p": sign_test_p(positive, negative),
        }
    return out
