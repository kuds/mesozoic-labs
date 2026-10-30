"""Tests for ``gait.stance_track`` (the centre-of-mass stance reduction), the bars of ``com_stance_helpers``
and the pure rules of ``com_stance_calibration``.

Synthetic tracks only, so every expected number is known in closed form: the support hull, its signed and
directional margins and the side of its nearest edge, the root's yaw, pitch and roll with their signs, the
means, drifts and detrended spreads of a ramped offset, the oscillation keys a wobble or a rock moves, the turns
and twists from the spawn, a mirror splay of the feet, the load shares, the XcoM from differentiated positions,
the lean-sign summary over seeds, the reading of a recorder trace, and the reduction's invariance to turning or
moving the whole track (what lets a robot log, in its own world frame, be read with it).  The real-env tests are
each species' ``test_com_stance_quality.py``.
"""

from __future__ import annotations

import ast
import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from environments.shared.gait.stance_track import (
    FAMILIES,
    HEADING_KEYS,
    KINEMATIC_KEYS,
    LOAD_KEYS,
    MIN_WINDOW_SAMPLES,
    SUPPORT_KEYS,
    SUPPORT_SERIES,
    StanceTrack,
    convex_hull,
    first_violation,
    lean_sign_summary,
    nearest_edge_side,
    ray_margin,
    sign_test_p,
    signed_margin,
    stance_track_quality,
    stance_track_series,
    track_from_episode_trace,
    yaw_pitch_roll,
)
from environments.shared.tests import com_stance_calibration as calibration
from environments.shared.tests.com_stance_helpers import (
    BAR_KINDS,
    SETTLE_READING,
    SYMMETRIC_VALUES,
    TARGETS,
    WORST_SECOND_READING,
    Bar,
    ComStanceQualityBase,
    TargetMissed,
    TrackedEpisode,
    _Statue,
    check,
    feet_to_centre_m,
    format_violations,
    last_whole_second,
    resolution_floor,
    speed_rms_by_second,
    worst_second_speed_rms,
)

DT = 0.01
#: Two feet 0.3 m apart across the heading, at x = 0, and their four sole corners (0.2 m long, 0.1 m wide).
FEET = np.array([[0.0, 0.15, 0.0], [0.0, -0.15, 0.0]])
SOLE_CORNERS = np.array(
    [[0.1, 0.2], [-0.1, 0.2], [-0.1, 0.1], [0.1, 0.1], [0.1, -0.1], [-0.1, -0.1], [-0.1, -0.2], [0.1, -0.2]]
)


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _axis_quat(axis: int, angle: float) -> np.ndarray:
    q = np.zeros(4)
    q[0] = math.cos(angle / 2.0)
    q[1 + axis] = math.sin(angle / 2.0)
    return q


def _quat(yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """The root orientation with that heading, nose-up pitch and left-up roll (z, then y, then x)."""
    return _qmul(_qmul(_axis_quat(2, yaw), _axis_quat(1, -pitch)), _axis_quat(0, roll))


def _track(
    samples: int = 201,
    *,
    com_xy=None,
    height=None,
    yaw=None,
    pitch=None,
    roll=None,
    support: bool = True,
    loads=None,
    headings=None,
    labels=("r", "l"),
    feet: np.ndarray = FEET,
    foot_xy=None,
    com_vel=None,
) -> StanceTrack:
    t = np.arange(samples) * DT
    com = np.zeros((samples, 3))
    com[:, :2] = 0.0 if com_xy is None else com_xy(t)
    com[:, 2] = 0.8 if height is None else height(t)
    yaw_t = np.zeros(samples) if yaw is None else yaw(t)
    pitch_t = np.zeros(samples) if pitch is None else pitch(t)
    roll_t = np.zeros(samples) if roll is None else roll(t)
    quat = np.array([_quat(yaw=y, pitch=p, roll=r) for y, p, r in zip(yaw_t, pitch_t, roll_t)])
    foot_pos = np.repeat(feet[None], samples, axis=0)
    if foot_xy is not None:
        foot_pos[:, :, :2] += foot_xy(t)
    return StanceTrack(
        t=t,
        com=com,
        root_quat=quat,
        foot_pos=foot_pos,
        labels=tuple(labels),
        support=tuple(SOLE_CORNERS.copy() for _ in range(samples)) if support else None,
        foot_load=None if loads is None else np.asarray(loads, dtype=np.float64),
        foot_heading=None if headings is None else np.asarray(headings, dtype=np.float64),
        com_vel=com_vel,
    )


def _turned_and_moved(track: StanceTrack, angle: float, shift: np.ndarray) -> StanceTrack:
    """The same episode in another world frame: turned by *angle* about z, then moved by *shift* (x, y)."""
    c, s = math.cos(angle), math.sin(angle)
    rot = np.array([[c, -s], [s, c]])

    def move(points: np.ndarray) -> np.ndarray:
        out = np.array(points, dtype=np.float64)
        out[..., :2] = out[..., :2] @ rot.T + shift
        return out

    quat = np.array([_qmul(_axis_quat(2, angle), q) for q in track.root_quat])
    support = None if track.support is None else tuple(move(points) for points in track.support)
    heading = None if track.foot_heading is None else track.foot_heading + angle
    return StanceTrack(
        t=track.t,
        com=move(track.com),
        root_quat=quat,
        foot_pos=move(track.foot_pos),
        labels=track.labels,
        support=support,
        foot_load=track.foot_load,
        foot_heading=heading,
    )


# ── the support polygon ──────────────────────────────────────────────────────


def test_the_hull_drops_duplicates_and_points_on_its_edges_and_turns_counter_clockwise():
    square = [[0, 0], [1, 0], [1, 1], [0, 1]]
    noisy = np.array(square + [[0.5, 0.0], [1.0, 0.5], [0.5, 0.5], [0, 0], [1, 1]], dtype=float)
    hull = convex_hull(noisy)
    assert sorted(map(tuple, hull)) == sorted(map(tuple, np.array(square, dtype=float)))
    edges = np.roll(hull, -1, axis=0) - hull
    assert np.all(edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0] > 0)


def test_the_margin_is_the_least_signed_edge_distance_and_nan_without_an_area():
    hull = convex_hull(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float))
    assert signed_margin(hull, np.array([0.5, 0.5])) == pytest.approx(0.5)
    assert signed_margin(hull, np.array([0.9, 0.5])) == pytest.approx(0.1)
    assert signed_margin(hull, np.array([1.1, 0.5])) == pytest.approx(-0.1)
    assert signed_margin(hull, np.array([0.5, -0.2])) == pytest.approx(-0.2)
    assert math.isnan(signed_margin(convex_hull(np.array([[0, 0], [1, 0], [2, 0]], dtype=float)), np.zeros(2)))
    assert math.isnan(signed_margin(convex_hull(np.array([[0.3, 0.3]])), np.zeros(2)))
    assert math.isnan(signed_margin(np.zeros((0, 2)), np.zeros(2)))


def test_the_ray_margin_is_the_exit_distance_along_a_direction_and_names_the_nearest_edge():
    # A 2 m long, 1 m wide stance (x forward): the point 0.2 m from the right edge, 0.7 m from the toe.
    hull = convex_hull(np.array([[-1.0, -0.5], [1.0, -0.5], [1.0, 0.5], [-1.0, 0.5]]))
    point = np.array([0.3, -0.3])
    assert ray_margin(hull, point, np.array([1.0, 0.0])) == pytest.approx(0.7)
    assert ray_margin(hull, point, np.array([-1.0, 0.0])) == pytest.approx(1.3)
    assert ray_margin(hull, point, np.array([0.0, -1.0])) == pytest.approx(0.2)
    assert ray_margin(hull, point, np.array([0.0, 1.0])) == pytest.approx(0.8)
    diagonal = np.array([1.0, 1.0]) / math.sqrt(2.0)
    assert ray_margin(hull, np.zeros(2), diagonal) == pytest.approx(0.5 * math.sqrt(2.0))
    # Outside past the right edge: minus the way back.
    assert ray_margin(hull, np.array([0.0, -0.6]), np.array([0.0, -1.0])) == pytest.approx(-0.1)
    assert math.isnan(ray_margin(np.zeros((0, 2)), point, np.array([1.0, 0.0])))
    assert nearest_edge_side(hull, point, np.array([1.0, 0.0])) == "right"
    assert nearest_edge_side(hull, np.array([0.9, 0.0]), np.array([1.0, 0.0])) == "ahead"
    assert nearest_edge_side(hull, np.array([0.9, 0.0]), np.array([-1.0, 0.0])) == "behind"
    assert nearest_edge_side(hull, np.array([0.0, 0.45]), np.array([1.0, 0.0])) == "left"


def test_a_sideways_lean_shrinks_the_lateral_margin_even_when_the_nearest_edge_is_the_toe():
    # The soles are 0.2 m long and the stance 0.4 m wide: the toe/heel edges (0.1 m) are the nearest ones, so a
    # 30 mm lean leaves the nearest-edge margin at 0.1 m while the lateral margin falls from 0.2 to 0.17 m.
    still = stance_track_quality(_track(100), window_start_s=0.0)
    leaning = stance_track_quality(
        _track(100, com_xy=lambda t: np.stack([np.zeros_like(t), np.full_like(t, -0.03)], axis=1)),
        window_start_s=0.0,
    )
    assert still["com_margin_min_m"] == pytest.approx(0.1) == leaning["com_margin_min_m"]
    assert still["com_margin_lateral_min_m"] == pytest.approx(0.2)
    assert leaning["com_margin_lateral_min_m"] == pytest.approx(0.17)
    assert leaning["com_margin_fore_aft_min_m"] == pytest.approx(0.1)


# ── the root's angles ────────────────────────────────────────────────────────


def test_yaw_pitch_and_roll_read_back_with_their_signs():
    angles = yaw_pitch_roll(np.array([_quat(yaw=0.5, pitch=0.2, roll=-0.1), _quat(pitch=-0.3), _quat(roll=0.25)]))
    np.testing.assert_allclose(angles, [[0.5, 0.2, -0.1], [0.0, -0.3, 0.0], [0.0, 0.0, 0.25]], atol=1e-12)
    # + nose up: the pitched root's x axis points above horizontal.
    w, x, y, z = _quat(pitch=0.2)
    assert 2.0 * (x * z - w * y) > 0.0
    # + left up: the rolled root's y (left) axis points above horizontal.
    w, x, y, z = _quat(roll=0.25)
    assert 2.0 * (y * z + w * x) > 0.0


# ── window reductions ────────────────────────────────────────────────────────


def test_a_ramped_lateral_offset_gives_its_mean_drift_and_detrended_spread():
    # CoM 2 mm left of the feet at the window's start, drifting left at 1 mm/s, with a 1 Hz, 0.5 mm wobble
    # (a cosine about the window's middle, so the wobble does not lean the fitted line).
    def com_xy(t):
        lateral = 0.002 + 0.001 * (t - 1.0) + 0.0005 * np.cos(2.0 * np.pi * t)
        return np.stack([np.full_like(t, 0.01), lateral], axis=1)

    q = stance_track_quality(_track(301, com_xy=com_xy), window_start_s=1.0)
    window = np.arange(100, 301) * DT
    lateral = 0.002 + 0.001 * (window - 1.0) + 0.0005 * np.cos(2.0 * np.pi * window)
    assert q["window_samples"] == 201 and q["window_s"] == pytest.approx(2.0)
    assert q["com_lateral_mean_m"] == pytest.approx(lateral.mean(), abs=1e-12)
    assert q["com_lateral_drift_m"] == pytest.approx(0.002, abs=1e-8)  # 1 mm/s across the 2 s window
    assert q["com_lateral_sd_m"] == pytest.approx(0.0005 / math.sqrt(2.0), rel=0.01)
    assert q["com_fore_aft_mean_m"] == pytest.approx(0.01)
    assert q["com_fore_aft_drift_m"] == pytest.approx(0.0, abs=1e-12)
    assert q["com_height_mean_m"] == pytest.approx(0.8) and q["com_height_drift_m"] == pytest.approx(0.0)
    assert q["root_roll_mean_deg"] == pytest.approx(0.0) and q["root_turn_deg"] == pytest.approx(0.0)
    assert q["foot_travel_max_m"] == 0.0 and q["foot_travel_from_spawn_max_m"] == 0.0


def test_a_sag_is_a_negative_height_drift_and_a_bounce_a_height_spread():
    # Sinking 2 mm/s with a 2 Hz, 1 mm bounce (a cosine about the window's middle, so it does not tilt the line).
    def height(t):
        return 0.8 - 0.002 * t + 0.001 * np.cos(2.0 * np.pi * 2.0 * (t - 2.5))

    q = stance_track_quality(_track(401, height=height), window_start_s=1.0)
    assert q["com_height_drift_m"] == pytest.approx(-0.006, abs=1e-6)  # 2 mm/s across the 3 s window
    assert q["com_height_sd_m"] == pytest.approx(0.001 / math.sqrt(2.0), rel=0.02)


def test_a_yaw_ramp_is_the_roots_turn_and_without_the_feet_a_twist_over_them():
    # The root turns 2 deg over 2 s; the feet hold their heading, so the body twists 2 deg over them.
    def yaw(t):
        return np.radians(1.0) * t

    q = stance_track_quality(_track(201, yaw=yaw, headings=np.zeros((201, 2))), window_start_s=0.5)
    assert q["root_turn_deg"] == pytest.approx(1.5, abs=1e-9)  # over the window, 0.5-2.0 s
    assert q["root_turn_end_deg"] == pytest.approx(1.5, abs=1e-9)  # the window is shorter than END_S: all of it
    assert q["root_turn_from_spawn_deg"] == pytest.approx(2.0, abs=1e-9)
    assert q["root_minus_feet_yaw_from_spawn_deg"] == pytest.approx(2.0, abs=1e-9)  # from the spawn
    assert q["foot_turn_max_deg"] == 0.0 and q["foot_splay_change_max_deg"] == pytest.approx(0.0, abs=1e-12)
    assert q["root_yaw_sd_deg"] == pytest.approx(0.0, abs=1e-9)  # a steady turn is not a wobble
    # Feet that turn with the root: no twist, and the feet's own turn over the window and from the spawn.
    turning = np.repeat(yaw(np.arange(201) * DT)[:, None], 2, axis=1)
    q = stance_track_quality(_track(201, yaw=yaw, headings=turning), window_start_s=0.5)
    assert q["root_minus_feet_yaw_from_spawn_deg"] == pytest.approx(0.0, abs=1e-9)
    assert q["foot_turn_max_deg"] == pytest.approx(1.5, abs=1e-9)
    assert q["foot_turn_from_spawn_max_deg"] == pytest.approx(2.0, abs=1e-9)
    assert q["foot_splay_change_max_deg"] == pytest.approx(0.0, abs=1e-9)


def test_the_end_keys_see_a_turn_or_a_motion_that_has_not_stopped():
    # A turn that stops at 1 s, and one that keeps going: only the second turns over the last 2 s.
    stopped = stance_track_quality(_track(401, yaw=lambda t: np.radians(np.minimum(t, 1.0))), window_start_s=0.5)
    going = stance_track_quality(_track(401, yaw=lambda t: np.radians(0.5 * t)), window_start_s=0.5)
    assert stopped["end_s"] == pytest.approx(2.0)
    assert stopped["root_turn_end_deg"] == pytest.approx(0.0, abs=1e-9)
    assert going["root_turn_end_deg"] == pytest.approx(1.0, abs=1e-9)
    # The CoM sliding at 10 mm/s until 1 s, then still: over the window 0.5-4 s it moved; at its end it did not.
    settled = stance_track_quality(
        _track(401, com_xy=lambda t: np.stack([0.01 * np.minimum(t, 1.0), np.zeros_like(t)], axis=1)),
        window_start_s=0.5,
    )
    assert settled["com_speed_rms_mps"] > 0.003 and settled["com_speed_end_rms_mps"] == pytest.approx(0.0, abs=1e-12)


def test_a_wobble_or_a_rock_scores_on_the_sway_keys_and_not_on_the_turn_and_means():
    # The root yawing +-3 deg at 0.5 Hz over planted feet, and the CoM rocking +-5 mm fore-aft at 1 Hz with the
    # pitch (cosines about the 8 s window's middle, so none of them tilts the fitted line).
    def wave(amplitude, hz):
        return lambda t: amplitude * np.cos(2.0 * np.pi * hz * (t - 4.0))

    wobble = stance_track_quality(
        _track(801, yaw=wave(np.radians(3.0), 0.5), headings=np.zeros((801, 2))), window_start_s=0.0
    )
    assert wobble["root_yaw_sd_deg"] == pytest.approx(3.0 / math.sqrt(2.0), rel=1e-3)
    assert abs(wobble["root_turn_deg"]) < 1e-9 and abs(wobble["root_turn_from_spawn_deg"]) < 1e-9
    rock = stance_track_quality(
        _track(
            801,
            com_xy=lambda t: np.stack([wave(0.005, 1.0)(t), np.zeros_like(t)], axis=1),
            pitch=wave(np.radians(0.5), 1.0),
        ),
        window_start_s=0.0,
    )
    assert rock["com_fore_aft_sd_m"] == pytest.approx(0.005 / math.sqrt(2.0), rel=1e-3)
    assert rock["root_pitch_sd_deg"] == pytest.approx(0.5 / math.sqrt(2.0), rel=1e-3)
    # Central differences under-read a 1 Hz speed by (2 pi f dt)^2 / 6, 7e-4, and more at the two ends.
    assert rock["com_speed_rms_mps"] == pytest.approx(0.005 * 2.0 * math.pi / math.sqrt(2.0), rel=3e-3)
    assert abs(rock["com_fore_aft_mean_m"]) < 1e-5 and abs(rock["root_pitch_mean_deg"]) < 1e-3


def test_a_mirror_splay_shows_as_a_splay_and_not_as_a_twist():
    # Both feet turn 5 deg outward (the right one clockwise) over the first second: the mean heading stays.
    turn = np.radians(5.0) * np.minimum(np.arange(201) * DT, 1.0)
    headings = np.stack([-turn, turn], axis=1)  # labels ("r", "l")
    q = stance_track_quality(_track(201, headings=headings), window_start_s=1.0)
    assert q["foot_splay_change_max_deg"] == pytest.approx(5.0, abs=1e-9)
    assert q["foot_turn_from_spawn_max_deg"] == pytest.approx(5.0, abs=1e-9)
    assert q["foot_turn_max_deg"] == pytest.approx(0.0, abs=1e-9)  # done before the window
    assert q["root_minus_feet_yaw_from_spawn_deg"] == pytest.approx(0.0, abs=1e-9)


def test_a_foot_slide_before_the_window_shows_only_from_the_spawn():
    def foot_xy(t):
        slide = np.zeros((t.size, 2, 2))
        slide[:, 0, 0] = 0.004 * np.minimum(t, 1.0)  # the right foot slides 4 mm forward in the first second
        return slide

    q = stance_track_quality(_track(201, foot_xy=foot_xy), window_start_s=1.0)
    assert q["foot_travel_max_m"] == pytest.approx(0.0, abs=1e-12)
    assert q["foot_travel_from_spawn_max_m"] == pytest.approx(0.004)


def test_the_yaw_is_unwrapped_through_the_heading_seam():
    # A turn from 179 deg through 180 deg to -179 deg is a 2 deg turn, not a 358 deg one.
    def yaw(t):
        return np.radians(179.0 + t)

    q = stance_track_quality(_track(201, yaw=yaw, headings=np.tile(np.radians(179.0), (201, 2))), window_start_s=0.0)
    assert q["root_turn_deg"] == pytest.approx(2.0, abs=1e-9)
    assert q["root_minus_feet_yaw_from_spawn_deg"] == pytest.approx(2.0, abs=1e-9)


def test_a_steady_roll_is_a_mean_and_a_roll_drift_a_drift():
    def roll(t):
        return np.radians(0.3 + 0.05 * t)

    q = stance_track_quality(_track(201, roll=roll), window_start_s=1.0)
    assert q["root_roll_mean_deg"] == pytest.approx(0.3 + 0.05 * 1.5, abs=1e-9)
    assert q["root_roll_drift_deg"] == pytest.approx(0.05, abs=1e-9)  # 0.05 deg/s across the 1 s window
    assert q["root_pitch_mean_deg"] == pytest.approx(0.0, abs=1e-12)


def test_the_margins_read_the_support_and_the_xcom_leads_the_com_by_its_velocity():
    # The CoM 5 mm ahead of the centre, moving forward at 20 mm/s: its margin to the toe edge (x = 0.1) is
    # 0.095 m less the travel, and the XcoM's is less again by v / omega0.
    def com_xy(t):
        return np.stack([0.005 + 0.02 * t, np.zeros_like(t)], axis=1)

    track = _track(101, com_xy=com_xy)
    series = stance_track_series(track)
    omega0 = math.sqrt(9.81 / 0.8)
    np.testing.assert_allclose(series["com_margin"], 0.095 - 0.02 * track.t, atol=1e-12)
    np.testing.assert_allclose(series["xcom_margin"], 0.095 - 0.02 * track.t - 0.02 / omega0, atol=1e-12)
    np.testing.assert_allclose(series["com_margin_fore_aft"], series["com_margin"], atol=1e-12)
    q = stance_track_quality(track, window_start_s=0.5)
    assert q["support_defined_fraction"] == 1.0
    assert q["com_margin_min_m"] == pytest.approx(0.095 - 0.02)
    assert q["xcom_margin_min_m"] == pytest.approx(0.095 - 0.02 - 0.02 / omega0)
    assert q["com_speed_rms_mps"] == pytest.approx(0.02)
    # A supplied velocity is used as given.
    supplied = _track(101, com_xy=com_xy, com_vel=np.tile([0.0, 0.0, 0.0], (101, 1)))
    assert stance_track_quality(supplied, window_start_s=0.5)["xcom_margin_min_m"] == pytest.approx(0.095 - 0.02)


def test_differentiated_positions_track_an_analytic_velocity():
    # A 0.5 Hz, 10 mm fore-aft sway: central differences against the true velocity.
    def com_xy(t):
        return np.stack([0.01 * np.sin(np.pi * t), np.zeros_like(t)], axis=1)

    t = np.arange(401) * DT
    velocity = np.zeros((401, 3))
    velocity[:, 0] = 0.01 * np.pi * np.cos(np.pi * t)
    differenced = stance_track_quality(_track(401, com_xy=com_xy), window_start_s=1.0)
    exact = stance_track_quality(_track(401, com_xy=com_xy, com_vel=velocity), window_start_s=1.0)
    # The central difference's relative error is (pi dt)^2 / 6, 1.6e-4: about 1.5 um of XcoM here.
    assert differenced["xcom_margin_min_m"] == pytest.approx(exact["xcom_margin_min_m"], abs=1e-5)
    assert differenced["com_speed_rms_mps"] == pytest.approx(exact["com_speed_rms_mps"], rel=1e-3)


def test_a_support_with_no_area_reads_nan_and_lowers_the_defined_fraction():
    track = _track(100)
    support = list(track.support or ())
    for i in range(50, 60):
        support[i] = np.array([[0.0, 0.0], [0.1, 0.0]])  # a line of contacts
    for i in range(60, 65):
        support[i] = np.zeros((0, 2))  # no loaded contact
    track = replace(track, support=tuple(support))
    series = stance_track_series(track)
    for name in SUPPORT_SERIES.values():
        assert np.isnan(series[name][50:65]).all() and np.isfinite(series[name][:50]).all(), name
    q = stance_track_quality(track, window_start_s=0.0)
    assert q["support_defined_fraction"] == pytest.approx(0.85)
    assert q["com_margin_min_m"] == pytest.approx(0.1)


def test_the_load_shares_on_four_feet_skip_an_unloaded_or_unread_sample():
    loads = np.tile([36.0, 24.0, 24.0, 16.0], (100, 1))  # fr, fl, rr, rl: 60 % right, 60 % front
    loads[40] = 0.0  # airborne for a sample
    loads[41] = np.nan  # no reading
    feet = np.array([[0.5, -0.2, 0.0], [0.5, 0.2, 0.0], [-0.5, -0.2, 0.0], [-0.5, 0.2, 0.0]])
    track = _track(100, loads=loads, labels=("fr", "fl", "rr", "rl"), feet=feet, support=False)
    q = stance_track_quality(track, window_start_s=0.0)
    assert q["left_load_share_mean"] == pytest.approx(0.4)
    assert q["front_load_share_mean"] == pytest.approx(0.6)
    assert q["lightest_foot_share_min"] == pytest.approx(0.16)
    assert np.isnan(stance_track_series(track)["foot_load_share"][40:42]).all()
    # Two feet have no front share.
    two = stance_track_quality(_track(100, loads=np.tile([60.0, 40.0], (100, 1))), window_start_s=0.0)
    assert two["left_load_share_mean"] == pytest.approx(0.4) and two["lightest_foot_share_min"] == pytest.approx(0.4)
    assert "front_load_share_mean" not in two


def test_each_key_exists_only_where_the_track_carries_what_it_reads():
    extra = {"window_samples", "window_s", "end_s"}
    bare = stance_track_quality(_track(100, support=False), window_start_s=0.0)
    assert set(bare) == set(KINEMATIC_KEYS) | extra
    full = stance_track_quality(
        _track(100, loads=np.ones((100, 2)), headings=np.zeros((100, 2))),
        window_start_s=0.0,
    )
    assert set(full) == set(KINEMATIC_KEYS + SUPPORT_KEYS + HEADING_KEYS + LOAD_KEYS) | extra
    # Every key a family names is one the reduction gives (front_load_share_mean on four feet).
    assert {key for keys in FAMILIES.values() for key in keys} - set(full) == {"front_load_share_mean"}
    assert set(BAR_KINDS) == {key for keys in FAMILIES.values() for key in keys}
    assert set(SUPPORT_SERIES) == set(FAMILIES["support"])
    assert set(SYMMETRIC_VALUES) <= set(BAR_KINDS)


def test_a_window_of_fewer_than_ten_samples_and_a_non_finite_track_are_refused():
    track = _track(100)
    stance_track_quality(track, window_start_s=(100 - MIN_WINDOW_SAMPLES) * DT)
    with pytest.raises(ValueError, match="at least 10"):
        stance_track_quality(track, window_start_s=(100 - MIN_WINDOW_SAMPLES + 1) * DT)
    com = track.com.copy()
    com[30, 1] = np.nan  # a dropped sample of a robot log
    with pytest.raises(ValueError, match="com has non-finite samples"):
        stance_track_quality(replace(track, com=com), window_start_s=0.0)


def test_turning_or_moving_the_whole_track_changes_no_number():
    """The robot-log guarantee: the reduction reads the stance in its own frame, not the world's."""

    def com_xy(t):
        return np.stack([0.004 + 0.003 * np.sin(3.0 * t), -0.002 + 0.001 * t], axis=1)

    def yaw(t):
        return 0.1 + np.radians(0.5) * t + np.radians(0.2) * np.sin(2.0 * t)

    def roll(t):
        return np.radians(0.2) * np.cos(t)

    headings = np.stack([0.1 + 0.01 * np.arange(301) * DT, np.full(301, 0.12)], axis=1)
    track = _track(
        301,
        com_xy=com_xy,
        yaw=yaw,
        roll=roll,
        loads=np.tile([45.0, 55.0], (301, 1)),
        headings=headings,
    )
    base = stance_track_quality(track, window_start_s=1.0)
    for angle in (math.radians(30.0), math.radians(-135.0), math.pi):
        moved = stance_track_quality(_turned_and_moved(track, angle, np.array([12.0, -7.5])), window_start_s=1.0)
        assert set(moved) == set(base)
        for key, value in base.items():
            assert moved[key] == pytest.approx(value, abs=1e-9), key


def test_first_violation_names_the_first_window_sample_under_the_bar():
    t = np.arange(10) * DT
    series = np.array([0.0, 0.0, 5.0, 5.0, 1.0, 5.0, np.nan, 5.0, 0.5, 5.0])
    assert first_violation(t, series, below=2.0, window_start_s=0.0) == 0
    assert first_violation(t, series, below=2.0, window_start_s=0.02) == 4  # samples before the window ignored
    assert first_violation(t, series, below=0.8, window_start_s=0.05) == 6  # a sample with no value counts
    assert first_violation(t, np.full(10, 5.0), below=2.0, window_start_s=0.0) is None


# ── across seeds, and a recorder's trace ─────────────────────────────────────


def test_the_lean_sign_summary_flags_a_lean_of_one_sign_on_every_seed():
    assert sign_test_p(10, 0) == pytest.approx(2.0 / 1024.0)
    assert sign_test_p(9, 1) == pytest.approx(22.0 / 1024.0)
    assert sign_test_p(5, 5) == 1.0 and sign_test_p(0, 0) == 1.0
    seeds = [{"com_lateral_mean_m": -0.001 * (k + 1), "root_roll_mean_deg": 0.01 if k else -0.01} for k in range(10)]
    seeds.append({"com_lateral_mean_m": math.nan, "root_roll_mean_deg": 0.0})
    summary = lean_sign_summary(seeds)
    lateral, roll = summary["com_lateral_mean_m"], summary["root_roll_mean_deg"]
    assert (lateral["n"], lateral["positive"], lateral["negative"]) == (10, 0, 10)
    assert lateral["mean"] == pytest.approx(-0.0055) and lateral["same_sign_fraction"] == 1.0
    assert lateral["sign_test_p"] == pytest.approx(2.0 / 1024.0)
    assert (roll["n"], roll["positive"], roll["negative"]) == (11, 9, 1)
    assert roll["same_sign_fraction"] == pytest.approx(0.9)


def test_a_recorder_trace_reads_as_a_track_with_its_spawn_first():
    steps, dt = 50, 0.02
    # The CoM creeping forward, so the spawn sample differs from the first step's.
    track = _track(steps + 1, com_xy=lambda t: np.stack([0.01 + 0.001 * t, -0.002 + 0.0 * t], axis=1))
    trace = SimpleNamespace(
        dt=dt,
        labels=("r", "l"),
        com_pos=track.com[1:],
        spawn_com_pos=track.com[0],
        root_quat=track.root_quat[1:],
        spawn_root_quat=track.root_quat[0],
        foot_pos=track.foot_pos[1:],
        spawn_foot_pos=track.foot_pos[0],
        leg_floor_mean=np.tile([60.0, 40.0], (steps, 1)),
    )
    read = track_from_episode_trace(trace)
    np.testing.assert_allclose(read.t, np.arange(steps + 1) * dt)
    np.testing.assert_array_equal(read.com, track.com)
    assert np.isnan(np.asarray(read.foot_load)[0]).all() and read.support is None and read.foot_heading is None
    q = stance_track_quality(read, window_start_s=0.0)
    assert q["com_lateral_mean_m"] == pytest.approx(-0.002) and q["left_load_share_mean"] == pytest.approx(0.4)
    with pytest.raises(ValueError, match="balance=True"):
        track_from_episode_trace(SimpleNamespace(com_pos=None, spawn_com_pos=None))


# ── the bars ─────────────────────────────────────────────────────────────────


def test_each_bar_kind_holds_its_band_and_fails_a_missing_value():
    assert Bar("com_lateral_mean_m", "abs", 0.001).holds(-0.001) and not Bar("x", "abs", 0.001).holds(0.0011)
    upper = Bar("com_speed_rms_mps", "upper", 0.001, reference=0.0003)
    assert upper.holds(-1.0) and upper.holds(0.0013) and not upper.holds(0.00131)
    assert Bar("left_load_share_mean", "half", 0.01).holds(0.509) and not Bar("x", "half", 0.01).holds(0.489)
    dev = Bar("com_height_mean_m", "dev", 0.001, reference=0.8669)
    assert dev.holds(0.8678) and not dev.holds(0.8658)
    lower = Bar("com_margin_min_m", "lower", 0.002, reference=0.0579)
    assert lower.holds(0.0560) and lower.holds(1.0) and not lower.holds(0.0558)
    assert lower.ratio(0.0539) == pytest.approx(2.0)
    assert not lower.holds(math.nan) and math.isnan(lower.ratio(math.nan))
    with pytest.raises(ValueError, match="unknown bar kind"):
        Bar("x", "between", 1.0).badness(0.0)


def test_check_lists_every_broken_bar_with_its_number_and_its_bar():
    bars = [
        Bar("com_lateral_mean_m", "abs", 0.001, basis="mirror symmetry"),
        Bar("left_load_share_mean", "half", 0.01),
        Bar("lightest_foot_share_min", "lower", 0.01, reference=0.5),
        Bar("root_roll_mean_deg", "abs", 0.1),
    ]
    violations = check(
        {"com_lateral_mean_m": -0.00249, "left_load_share_mean": 0.553, "lightest_foot_share_min": 0.447}, bars
    )
    assert [violation.bar.key for violation in violations] == [
        "com_lateral_mean_m",
        "left_load_share_mean",
        "lightest_foot_share_min",
        "root_roll_mean_deg",  # not measured
    ]
    text = format_violations("statue leans", violations)
    assert "com_lateral_mean_m = -2.490 mm (bar |x| <= 1.000 mm; mirror symmetry; 2.49x its width)" in text
    assert "left_load_share_mean = 0.5530 (bar |x - 0.5| <= 0.0100; 5.30x its width)" in text
    assert "lightest_foot_share_min = 0.4470 (bar >= 0.4900 (0.5000 - 0.0100); 5.30x its width)" in text
    assert "root_roll_mean_deg = nan (bar |x| <= 0.1000 deg, not measured)" in text
    assert check({"com_lateral_mean_m": 0.0002}, bars[:1]) == []


def test_a_value_just_over_its_bar_never_prints_equal_to_it():
    # 1.0002 mm against a 1 mm bar: the default 3 decimals would show "1.000 mm" on both sides.
    (violation,) = check({"com_lateral_mean_m": 0.0010002}, [Bar("com_lateral_mean_m", "abs", 0.001)])
    text = violation.describe()
    assert "1.0002 mm" in text and "<= 1.0000 mm" in text and "1.00x its width" in text
    (violation,) = check({"root_roll_mean_deg": -0.10001}, [Bar("root_roll_mean_deg", "abs", 0.1)])
    assert "-0.10001 deg" in violation.describe()


def test_the_floors_and_the_two_tiers_of_bars():
    assert resolution_floor("com_lateral_mean_m") == resolution_floor("com_speed_rms_mps") == 0.0005
    assert resolution_floor("root_turn_deg") == 0.05 and resolution_floor("left_load_share_mean") == 0.005

    class Biped(ComStanceQualityBase):
        reference = {"com_height_mean_m": 0.8, "com_speed_rms_mps": 0.0003}
        spread_width = {"com_height_mean_m": 0.004, "com_lateral_mean_m": 0.002}

    class Quadruped(Biped):
        quadruped = True

    biped, quadruped = Biped(), Quadruped()
    assert biped.statue_bar("com_lateral_mean_m") == Bar(
        "com_lateral_mean_m", "abs", 0.001, 0.0, "mirror symmetry fixes it on the noise-free statue"
    )
    height = biped.statue_bar("com_height_mean_m")
    assert (height.kind, height.width, height.reference) == ("dev", 0.001, 0.8)
    speed = biped.statue_bar("com_speed_rms_mps")
    assert speed.bounds() == (-math.inf, pytest.approx(0.0013))
    assert biped.spread_bar("com_height_mean_m").width == 0.004
    assert biped.symmetric("lightest_foot_share_min") and not quadruped.symmetric("lightest_foot_share_min")
    assert "front_load_share_mean" in quadruped.keys("sagittal") and "front_load_share_mean" not in biped.keys()


def test_a_target_reads_its_own_window_and_names_each_reading_over_its_limit():
    # The body turns 2 deg in its first second; the CoM slides sideways at 5 mm/s until 4 s, then stands still.
    def track_of(samples):
        return _track(
            samples,
            yaw=lambda t: math.radians(2.0) * np.minimum(t, 1.0),
            com_xy=lambda t: np.stack([np.zeros_like(t), 0.005 * np.minimum(t, 4.0)], axis=1),
            support=False,
        )

    track = track_of(1001)
    turn, still, settled = TARGETS["turn_from_spawn"], TARGETS["still_from_3s"], TARGETS["settled_by_2s"]
    readings = turn.readings(track, 6.0)
    assert readings == {"root_turn_from_spawn_deg": pytest.approx(2.0)}
    assert turn.misses(readings) == ["root_turn_from_spawn_deg = 2.0000 deg, target |x| <= 1.0000 deg"]
    assert turn.misses({"root_turn_from_spawn_deg": -0.5}) == []  # the size of a turn, either way
    # Read at the end: a body that turns 2 deg and comes back meets it.
    back = _track(1001, yaw=lambda t: math.radians(2.0) * np.sin(math.pi * t / 10.0), support=False)
    assert turn.misses(turn.readings(back, 6.0)) == []
    # From 3 s the CoM moves for one more second, at 5 mm/s: the worst second reads it whatever the horizon,
    # where the RMS over the window from 3 s thins out (1.9 mm/s over 10 s, 1.2 mm/s over 20 s).
    for samples, thinned in ((1001, 7.0), (2001, 17.0)):
        sliding = track_of(samples)
        reading = still.readings(sliding, 6.0)
        assert reading[WORST_SECOND_READING] == pytest.approx(0.005, rel=0.02)
        assert [miss.split(" = ")[0] for miss in still.misses(reading)] == [WORST_SECOND_READING]
        window_rms = stance_track_quality(sliding, window_start_s=3.0)["com_speed_rms_mps"]
        assert window_rms == pytest.approx(0.005 * math.sqrt(1.0 / thinned), rel=0.02)
    assert still.readings(track, 4.5) == still.readings(track, 6.0)  # its own window, not the settled one
    assert settled.readings(track, 6.0) == {SETTLE_READING: 4.0}
    assert settled.misses({SETTLE_READING: 4.0}) == [f"{SETTLE_READING} = 4 s, target |x| <= 2 s"]
    assert settled.misses({SETTLE_READING: math.inf}) == [f"{SETTLE_READING} = never, target |x| <= 2 s"]
    assert TARGETS["lean"].misses(TARGETS["lean"].readings(_track(1001, support=False), 6.0)) == []


def test_no_rocking_after_3s_is_the_settle_rule_read_from_3s():
    def sliding_until(stop):
        return _track(
            1001, com_xy=lambda t: np.stack([np.zeros_like(t), 0.01 * np.minimum(t, stop)], axis=1), support=False
        )

    for stop in (0.5, 2.5, 2.95, 3.05, 3.5, 6.5, 20.0):
        track = sliding_until(stop)
        worst = worst_second_speed_rms(stance_track_series(track), 3.0)
        assert (worst <= 0.001) == (calibration.settle_time([track]) <= 3.0), stop
    with pytest.raises(ValueError, match="no whole second"):
        worst_second_speed_rms(stance_track_series(sliding_until(0.5)), 10.0)


def test_the_last_whole_second_is_read_when_the_summed_time_ends_a_hair_under_it():
    # MuJoCo sums its timestep: 10,000 steps of 0.002 s end a 20 s episode at 19.999999999999794 s.
    end = 0.0
    for _ in range(10_000):
        end += 0.002
    assert end < 20.0 and last_whole_second(np.array([0.0, end])) == 20
    # A track that stands still and then slides at 5 mm/s in its last second only, [19, 20) s.
    track = _track(
        2001, com_xy=lambda t: np.stack([0.005 * np.clip(t - 19.0, 0.0, 1.0), np.zeros_like(t)], axis=1), support=False
    )
    track = replace(track, t=np.r_[track.t[:-1], end])
    series = stance_track_series(track)
    assert max(speed_rms_by_second(series, 3.0)) == 19
    assert worst_second_speed_rms(series, 3.0) == pytest.approx(0.005, rel=0.02)
    assert calibration.settle_time([track]) == math.inf  # it still moves in its last second
    still, settled = TARGETS["still_from_3s"], TARGETS["settled_by_2s"]
    assert [miss.split(" = ")[0] for miss in still.misses(still.readings(track, 2.0))] == [WORST_SECOND_READING]
    assert settled.misses(settled.readings(track, 2.0)) == [f"{SETTLE_READING} = never, target |x| <= 2 s"]


def test_a_nan_in_a_series_a_target_reads_fails_every_target_outright_and_never_reads_as_still():
    # The CoM moves at 3 mm/s in [2, 3) s only: it settles at 3 s.
    track = _track(
        1001, com_xy=lambda t: np.stack([np.zeros_like(t), 0.003 * np.clip(t - 2.0, 0.0, 1.0)], axis=1), support=False
    )
    series = stance_track_series(track)
    assert TARGETS["settled_by_2s"].readings(track, 6.0, series=series) == {SETTLE_READING: 3.0}
    # A CoM velocity with a dropped sample is refused, as a dropped CoM sample is.
    velocity = np.zeros((1001, 3))
    velocity[250] = np.nan
    with pytest.raises(ValueError, match="com_vel has non-finite samples"):
        stance_track_series(replace(track, com_vel=velocity))
    # A NaN over the only second that moves, or in the fore-aft offset, fails every target with ValueError: never
    # read as still or settled (a met target), nor as a missed one a strict expected failure would absorb.
    for name, where in (("com_speed", slice(200, 300)), ("com_fore_aft", 350)):
        broken = dict(series)
        broken[name] = series[name].copy()
        broken[name][where] = np.nan
        for target in TARGETS.values():
            with pytest.raises(ValueError, match="a broken test, not a missed target"):
                target.for_body(0.15).readings(track, 6.0, series=broken)
    speed = series["com_speed"].copy()
    speed[250] = np.nan
    with pytest.raises(ValueError, match=r"not finite in \[2, 3\) s"):
        speed_rms_by_second({"t": series["t"], "com_speed": speed})


def test_a_reading_the_reduction_cannot_give_fails_outright_and_is_never_a_missed_target(monkeypatch):
    still, turned = _track(1001, support=False), _track(1001, yaw=lambda t: math.radians(3.0) * np.minimum(t, 1.0))
    renamed = replace(TARGETS["turn_from_spawn"], limits={"root_turn_from_spawn_deg_renamed": 1.0})
    with pytest.raises(ValueError, match="a broken test, not a missed target"):
        renamed.readings(still, 6.0)
    for broken in ({}, {"root_turn_from_spawn_deg": math.nan}, {"root_turn_from_spawn_deg": math.inf}):
        with pytest.raises(ValueError, match="a broken test, not a missed target"):
            TARGETS["turn_from_spawn"].misses(broken)
    with pytest.raises(ValueError, match="per body"):
        TARGETS["foot_slip_after_landing"].misses({"foot_travel_max_m": 0.0})
    assert not issubclass(ValueError, TargetMissed) and issubclass(TargetMissed, AssertionError)

    class Body(ComStanceQualityBase):
        species_label = "a synthetic biped"

    def run(track):
        steps = len(track.t) - 1
        episode = TrackedEpisode(track, steps, steps, False, True, None, DT)
        return _Statue(episode, stance_track_series(track), {})

    Body().check_target(run(still), "turn_from_spawn")  # met: nothing raised
    with pytest.raises(TargetMissed, match="noise-free statue: root_turn_from_spawn_deg = 3.0000 deg"):
        Body().check_target(run(turned), "turn_from_spawn")
    # A renamed reading fails with ValueError, which a strict xfail(raises=TargetMissed) does not absorb.
    monkeypatch.setitem(TARGETS, "turn_from_spawn", renamed)
    with pytest.raises(ValueError, match="a broken test, not a missed target"):
        Body().check_target(run(turned), "turn_from_spawn")


def test_the_slip_limit_is_a_one_degree_turn_at_the_bodys_own_feet_to_centre_distance():
    slip = TARGETS["foot_slip_after_landing"]
    still = _track(1001, support=False)
    assert feet_to_centre_m(still, 2.0) == pytest.approx(0.15)  # two feet 0.3 m apart
    body = slip.for_body(0.15)
    assert body.limits == {"foot_travel_max_m": pytest.approx(0.15 * math.radians(1.0))}
    assert "|foot_travel_max_m| <= 2.618 mm" in body.describe() and body.pivot_deg is None
    assert slip.for_body(0.02).limits == {"foot_travel_max_m": 0.001}  # never under twice the 0.5 mm floor
    assert TARGETS["lean"].for_body(0.15) is TARGETS["lean"]
    # Four feet 1.4 m long and 0.6 m wide: the farthest foot's distance takes in its fore-aft offset.
    quad = np.array([[0.7, 0.3, 0.0], [0.7, -0.3, 0.0], [-0.7, 0.3, 0.0], [-0.7, -0.3, 0.0]])
    four = _track(1001, support=False, feet=quad, labels=("fl", "fr", "rl", "rr"))
    assert feet_to_centre_m(four, 2.0) == pytest.approx(math.hypot(0.7, 0.3))

    # The feet turned rigidly about their centre by 1 deg over 1-2 s read just under the limit (the chord of the
    # arc): the slip target and the turn target agree.
    def turning(t):
        angle = math.radians(1.0) * np.clip(t - 1.0, 0.0, 1.0)
        cos, sin = np.cos(angle)[:, None], np.sin(angle)[:, None]
        x, y = FEET[:, 0][None], FEET[:, 1][None]
        return np.stack([cos * x - sin * y - x, sin * x + cos * y - y], axis=2)

    reading = body.readings(_track(1001, support=False, foot_xy=turning), 6.0)["foot_travel_max_m"]
    assert (
        reading == pytest.approx(0.15 * math.radians(1.0), rel=1e-4)
        and body.misses({"foot_travel_max_m": reading}) == []
    )
    # A foot that slides 3 mm in the first 0.5 s (a landing) and then stands: not a slip after landing.
    landing = _track(1001, support=False, foot_xy=lambda t: np.stack([np.minimum(t, 0.5) * 0.006, 0 * t], 1)[:, None])
    assert body.readings(landing, 6.0) == {"foot_travel_max_m": pytest.approx(0.0)}
    sliding = _track(1001, support=False, foot_xy=lambda t: np.stack([t * 0.0005, 0 * t], 1)[:, None])
    assert body.readings(sliding, 6.0)["foot_travel_max_m"] == pytest.approx(0.0045)  # 0.5 mm/s over 1-10 s
    assert len(body.misses(body.readings(sliding, 6.0))) == 1


# ── the calibration rules ────────────────────────────────────────────────────


def test_the_settle_time_is_the_first_second_after_which_every_track_stays_still():
    def sliding_until(stop):
        return _track(
            1001, com_xy=lambda t: np.stack([0.01 * np.minimum(t, stop), np.zeros_like(t)], axis=1), support=False
        )

    assert calibration.settle_time([sliding_until(0.5)]) == 2.0  # never before EARLIEST_WINDOW_S
    assert calibration.settle_time([sliding_until(0.5), sliding_until(4.5)]) == 5.0
    assert calibration.settle_time([sliding_until(20.0)]) == math.inf


def test_the_target_report_names_each_target_met_or_missed_with_its_worst_spawn():
    still = _track(1001, support=False)
    turned = _track(1001, yaw=lambda t: math.radians(3.0) * np.minimum(t, 1.0), support=False)
    lines = calibration.target_report(still, {7: still, 8: turned}, window_start_s=2.0)
    assert len(lines) == 1 + len(TARGETS)
    assert lines[0] == "# targets: the statue's farthest foot stands 150.0 mm from the feet's centre"
    turn = next(line for line in lines if line.startswith("# target turn_from_spawn:"))
    assert "MISSED (the statue meets; 1 of 2 spawns miss: [8])" in turn and "worst 3.0000 deg (seed 8)" in turn
    lean = next(line for line in lines if line.startswith("# target lean:"))
    assert "met by the statue and all 2 spawns" in lean
    slip = next(line for line in lines if line.startswith("# target foot_slip_after_landing:"))
    assert "foot_travel_max_m <= 2.618 mm: statue 0.000 mm" in slip


def test_the_rounding_rules():
    assert calibration.round_reference("com_height_mean_m", 0.8668964) == pytest.approx(0.86690)
    assert calibration.round_reference("root_pitch_mean_deg", -0.960559) == pytest.approx(-0.9606)
    assert calibration.round_up_2sf(0.0022001) == pytest.approx(0.0023)
    assert calibration.round_up_2sf(9.71) == pytest.approx(9.8)
    assert calibration.round_up_2sf(0.001) == pytest.approx(0.001)


def test_the_spread_width_is_twice_the_worst_bad_side_deviation_never_under_twice_the_floor():
    class Biped(ComStanceQualityBase):
        pass

    base = Biped()
    reference = {"com_margin_min_m": 0.06, "com_speed_rms_mps": 0.0002}
    spawns = [
        {key: 0.0 for key in base.keys()} | {"left_load_share_mean": 0.5, "lightest_foot_share_min": 0.5}
        for _ in range(3)
    ]
    for spawn, margin, lateral in zip(spawns, (0.061, 0.059, 0.0585), (0.0004, -0.0031, 0.001)):
        spawn |= {"com_margin_min_m": margin, "com_speed_rms_mps": 0.0001, "com_lateral_mean_m": lateral}
    reference |= {key: 0.0 for key in base.keys() if not base.symmetric(key) and key not in reference}
    widths = calibration.spread_widths(base, reference, spawns)
    assert widths["com_margin_min_m"] == pytest.approx(0.003)  # 2 x (0.06 - 0.0585); a wider margin is fine
    assert widths["com_lateral_mean_m"] == pytest.approx(0.0062)  # 2 x 3.1 mm
    assert widths["com_speed_rms_mps"] == pytest.approx(0.001)  # every spawn under the statue: the floor
    ratio, key = calibration.usage(base, reference, widths, spawns[1])
    assert (ratio, key) == (pytest.approx(0.5), "com_lateral_mean_m")


# ── the ratchet ──────────────────────────────────────────────────────────────

_SPECIES_FILES = sorted(calibration.REPO_ROOT.glob("environments/*/tests/test_com_stance_quality.py"))
_CLASS_SOURCE = """class TestThing(ComStanceQualityBase):
    model_id = ModelId("x", "y")
    physics_revision = 3
    window_start_s = 3.0
    # The noise-free statue's window values, physics r3, mujoco 3.10.0, read on 2026-01-01 with
    # com_stance_calibration and rounded to 0.01 mm and 1e-4 deg and of the load (under 1 % of a width).
    reference = {
        "com_margin_min_m": 0.06,
        "com_height_mean_m": 0.8,
    }
    # Twice the worst of calibration seeds 0-29 at reset noise 0.02, never under twice the floor (SI, deg).
    # Held-out seeds 100-149: none breaks a width; the largest use is 0.50 (root_turn_deg, seed 101).
    spread_width = {
        "root_turn_deg": 0.3,
        "com_lateral_mean_m": 0.002,
    }
    # A hand-written note about the seeds.
    noisy_seeds = (101, 102)
    rock_force_bw = 0.04
"""


def test_the_ratchet_classes_each_pin_change_as_its_bar_reads_it():
    old = {
        "physics_revision": 3,
        "window_start_s": 3.0,
        "reference": {"com_speed_rms_mps": 0.0002, "com_margin_min_m": 0.06, "com_height_mean_m": 0.8},
        "spread_width": {"root_turn_deg": 0.3, "com_lateral_mean_m": 0.002, "foot_travel_max_m": 0.001},
        "noisy_seeds": (101, 102),
    }
    assert calibration.pin_changes(old, old) == []
    new = {
        "physics_revision": 4,
        "window_start_s": 4.0,
        # upper: more motion accepted; lower: a wider margin demanded; dev: two-sided.
        "reference": {"com_speed_rms_mps": 0.0003, "com_margin_min_m": 0.061, "com_height_mean_m": 0.79},
        # narrower, wider, dropped, and a new bar.
        "spread_width": {"root_turn_deg": 0.2, "com_lateral_mean_m": 0.003, "root_yaw_sd_deg": 0.1},
        "noisy_seeds": (101, 103),
    }
    directions = {(change.pin, change.key): change.direction for change in calibration.pin_changes(old, new)}
    assert directions == {
        ("physics_revision", None): "moved",
        ("window_start_s", None): "looser",
        ("reference", "com_speed_rms_mps"): "looser",
        ("reference", "com_margin_min_m"): "tighter",
        ("reference", "com_height_mean_m"): "moved",
        ("spread_width", "root_turn_deg"): "tighter",
        ("spread_width", "com_lateral_mean_m"): "looser",
        ("spread_width", "foot_travel_max_m"): "looser",
        ("spread_width", "root_yaw_sd_deg"): "tighter",
        ("noisy_seeds", None): "moved",
    }
    back = {(change.pin, change.key): change.direction for change in calibration.pin_changes(new, old)}
    assert back[("window_start_s", None)] == back[("reference", "com_speed_rms_mps")] == "tighter"
    assert back[("spread_width", "com_lateral_mean_m")] == back[("spread_width", "foot_travel_max_m")] == "tighter"
    assert back[("reference", "com_margin_min_m")] == back[("spread_width", "root_yaw_sd_deg")] == "looser"


@pytest.mark.parametrize("path", _SPECIES_FILES, ids=lambda path: path.parts[-3])
def test_the_ratchet_leaves_every_species_file_as_it_is_when_its_pins_are_unchanged(path):
    source = path.read_text()
    lines = source.splitlines()
    classes = [node for node in ast.parse(source).body if isinstance(node, ast.ClassDef)]
    pinned = [node for node in classes if calibration.read_pins(source, node.name)]
    assert pinned
    for node in pinned:
        pins = calibration.read_pins(source, node.name)
        assert set(pins) == set(calibration.PIN_NAMES), (node.name, sorted(pins))
        provenance = "physics r99, mujoco 9.9.9, read on 2099-01-01"
        assert calibration.rewrite_pins(source, node.name, pins, provenance=provenance) == source
        # Each pin's committed lines are what the ratchet renders for its value, so a re-pin's diff shows the
        # values that moved and nothing else.
        for statement in node.body:
            if isinstance(statement, ast.Assign) and getattr(statement.targets[0], "id", None) in pins:
                name = statement.targets[0].id
                committed = lines[statement.lineno - 1 : statement.end_lineno]
                assert calibration.render_pin(name, pins[name], " " * statement.col_offset) == committed, name


def test_the_ratchet_rewrites_only_the_changed_pins_and_their_provenance():
    new = calibration.read_pins(_CLASS_SOURCE, "TestThing")
    new["spread_width"] = {"root_turn_deg": 0.2, "com_lateral_mean_m": 0.002}
    new["noisy_seeds"] = (101, 107)
    provenance = "physics r3, mujoco 3.10.0, read on 2026-10-29"
    # The same words wrapped otherwise: the committed block stays as it is.
    same = {
        "spread_width": [
            "Twice the worst of calibration seeds 0-29 at reset noise 0.02, never under twice the floor (SI, deg).  "
            "Held-out seeds 100-149: none breaks a width; the largest use is 0.50 (root_turn_deg, seed 101)."
        ],
        "noisy_seeds": ["Held-out seeds: the largest turn, re-read."],
    }
    out = calibration.rewrite_pins(_CLASS_SOURCE, "TestThing", new, provenance=provenance, comments=same)
    assert calibration.read_pins(out, "TestThing") == new
    removed = sorted(set(_CLASS_SOURCE.splitlines()) - set(out.splitlines()))
    added = sorted(set(out.splitlines()) - set(_CLASS_SOURCE.splitlines()))
    assert removed == [
        '        "root_turn_deg": 0.3,',
        "    # The noise-free statue's window values, physics r3, mujoco 3.10.0, read on 2026-01-01 with",
        "    noisy_seeds = (101, 102)",
    ]
    assert added == [
        '        "root_turn_deg": 0.2,',
        "    # The noise-free statue's window values, physics r3, mujoco 3.10.0, read on 2026-10-29 with",
        "    noisy_seeds = (101, 107)",
    ]
    # The hand-written note above noisy_seeds is someone's own: it stays whatever the new text.
    assert "    # A hand-written note about the seeds.\n    noisy_seeds = (101, 107)\n" in out
    # New words replace the block they re-read.
    other = {
        "spread_width": ["Twice the worst of calibration seeds 0-29, re-read.", "Held-out seeds 100-149: re-read."]
    }
    out = calibration.rewrite_pins(_CLASS_SOURCE, "TestThing", new, provenance=provenance, comments=other)
    block = "    # Twice the worst of calibration seeds 0-29, re-read.\n    # Held-out seeds 100-149: re-read.\n"
    assert block + "    spread_width = {\n" in out and "none breaks a width" not in out
    # Nothing changes, nothing is rewritten, the provenance and the comments included.
    unchanged = calibration.read_pins(_CLASS_SOURCE, "TestThing")
    assert calibration.rewrite_pins(_CLASS_SOURCE, "TestThing", unchanged, provenance=provenance, comments=other) == (
        _CLASS_SOURCE
    )
    with pytest.raises(ValueError, match="does not assign"):
        calibration.rewrite_pins(
            _CLASS_SOURCE.replace("    window_start_s = 3.0\n", ""), "TestThing", new | {"window_start_s": 4.0}
        )


def test_the_ratchet_refuses_to_write_a_looser_bar_without_the_flag(tmp_path):
    path = tmp_path / "test_thing.py"
    path.write_text(_CLASS_SOURCE)
    pins = calibration.read_pins(_CLASS_SOURCE, "TestThing")
    assert calibration.ratchet(path, "TestThing", pins, write=True, allow_loosen=False) == (0, "", [])

    tighter = pins | {"spread_width": {"root_turn_deg": 0.2, "com_lateral_mean_m": 0.002}}
    status, diff, changes = calibration.ratchet(path, "TestThing", tighter, write=False, allow_loosen=False)
    assert status == 1 and [change.direction for change in changes] == ["tighter"]
    assert '-        "root_turn_deg": 0.3,\n+        "root_turn_deg": 0.2,\n' in diff
    assert diff.startswith(f"--- a/{path}\n+++ b/{path}\n") and path.read_text() == _CLASS_SOURCE

    looser = tighter | {"reference": {"com_margin_min_m": 0.059, "com_height_mean_m": 0.8}}
    status, diff, changes = calibration.ratchet(path, "TestThing", looser, write=True, allow_loosen=False)
    assert status == 2 and path.read_text() == _CLASS_SOURCE
    assert sorted(change.direction for change in changes) == ["looser", "tighter"]
    status, _, _ = calibration.ratchet(path, "TestThing", looser, write=True, allow_loosen=True)
    assert status == 1 and calibration.read_pins(path.read_text(), "TestThing") == looser


def test_the_ratchet_states_the_calibration_in_the_committed_words_and_finds_the_known_poor_seeds():
    class Biped(ComStanceQualityBase):
        def test_known_poor_spawn_seed111_stays_within_the_calibrated_spread(self):
            pass

    usage = {101: (0.5, "root_turn_deg"), 102: (0.64, "foot_travel_max_m"), 111: (61.1, "com_margin_min_m")}
    heldout = {101: {"root_turn_from_spawn_deg": 1.92}, 102: {"com_lateral_mean_m": -1e-6}}
    result = calibration.Calibration(2.0, {}, {}, usage, (101, 102), {}, heldout)
    comments = calibration.pin_comments(Biped(), result)
    assert set(comments) == set(calibration.PIN_COMMENT_STARTS)
    assert " ".join(comments["spread_width"]) == (
        "Twice the worst of calibration seeds 0-29 at reset noise 0.02, never under twice the floor (SI, deg). "
        "Held-out seeds 100-149: 1 break a width (111: see known_poor_spawns); the largest use among the rest is "
        "0.64 (foot_travel_max_m, seed 102)."
    )
    assert " ".join(comments["noisy_seeds"]) == (
        "Held-out seeds: the largest turn from the spawn (1.92 deg) and the largest lateral offset (-0.001 mm) "
        "among those holding every width; they use 0.50 and 0.64 of a width."
    )
    assert all(len(line) <= 112 for lines in comments.values() for line in lines)
    assert calibration.known_poor_seeds(Biped) == [111]
