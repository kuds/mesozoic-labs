"""Tests for the floor-truth stance metrics and events, on synthetic traces (pure numpy, no env).

Every fixture starts from :func:`statue_trace` -- a trace of a perfect statue:
every leg loaded with its share of body weight on every substep, flat soles
with four loaded contacts, no motion, zero action -- and edits only the
arrays the behaviour under test would change, so each test reads as "this
defect, and nothing else".  The real-env checks (decode exactness, the six
statues) are in ``test_gait_recorder.py``; the measured separations they
were calibrated on are in the floor-truth map of the 2026-10 stance-hack
audit, not here.

Also pinned here: the layering rules of ``environments/shared/gait/`` (pure
modules import no mujoco; the package imports neither ``curriculum`` nor
``reporting``; no species env module imports it).
"""

from __future__ import annotations

import ast
import csv
import io
import math
import sys
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from environments.shared.gait import constants, events
from environments.shared.gait.recorder import EpisodeTrace
from environments.shared.gait.stance_metrics import (
    _WINDOW_FIELDS,
    STANCE_METRIC_FIELDS,
    STANCE_METRIC_FOOT_FIELDS,
    StanceEpisodeMetrics,
    episode_stance_metrics,
    spawn_grace_steps,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
GAIT = REPO_ROOT / "environments" / "shared" / "gait"
BODY_WEIGHT = 100.0
NU = 4

#: Foot reference sites (x, y, z) of the synthetic bipeds (right, left) and quadrupeds (fr, fl, rr, rl).
BIPED_FEET = np.array([[0.0, -0.15, 0.0], [0.0, 0.15, 0.0]])
QUADRUPED_FEET = np.array([[0.3, -0.1, 0.0], [0.3, 0.1, 0.0], [-0.3, -0.1, 0.0], [-0.3, 0.1, 0.0]])


def statue_trace(steps: int = 400, *, n_feet: int = 2, dt: float = 0.01, frame_skip: int = 5) -> EpisodeTrace:
    """A perfect statue: one support geom per foot, every quantity at its ideal value."""
    feet = BIPED_FEET if n_feet == 2 else QUADRUPED_FEET
    load = np.full((steps, n_feet), BODY_WEIGHT / n_feet)
    ones = np.ones((steps, n_feet))
    zeros = np.zeros((steps, n_feet))
    labels = ("r", "l") if n_feet == 2 else ("fr", "fl", "rr", "rl")
    return EpisodeTrace(
        species="synthetic",
        dt=dt,
        frame_skip=frame_skip,
        body_weight_n=BODY_WEIGHT,
        labels=labels,
        support_geom_foot=tuple(range(n_feet)),
        support_geom_names=tuple(f"{label}_sole" for label in labels),
        leg_floor_mean=load.copy(),
        leg_floor_min=load.copy(),
        leg_floor_max=load.copy(),
        leg_down_frac=ones.copy(),
        support_floor_mean=load.copy(),
        support_loaded_frac=ones.copy(),
        offsupport_floor_mean=zeros.copy(),
        nonleg_floor_mean=np.zeros(steps),
        nonleg_floor_max=np.zeros(steps),
        total_floor_mean=np.full(steps, BODY_WEIGHT),
        total_floor_max=np.full(steps, BODY_WEIGHT),
        feet_airborne_substeps=np.zeros(steps, dtype=np.int64),
        interleg_force_mean=zeros.copy(),
        interleg_substep_frac=np.zeros(steps),
        leg_body_force_mean=zeros.copy(),
        leg_body_substep_frac=np.zeros(steps),
        sole_contacts_mean=np.full((steps, n_feet), 4.0),
        sole_cop=np.zeros((steps, n_feet, 2)),
        touch=load.copy(),
        foot_pos=np.repeat(feet[None], steps, axis=0),
        root_pos=np.tile([0.0, 0.0, 0.5], (steps, 1)),
        root_quat=np.tile([1.0, 0.0, 0.0, 0.0], (steps, 1)),
        joint_qpos=np.zeros((steps, 3)),
        sole_tilt_deg=zeros.copy(),
        sole_roll_deg=zeros.copy(),
        sole_pitch_deg=zeros.copy(),
        sole_corner_lift=zeros.copy(),
        sole_min_height=zeros.copy(),
        action=np.zeros((steps, NU)),
        ctrl=np.zeros((steps, NU)),
        reward=np.ones(steps),
        spawn_foot_pos=feet.copy(),
        spawn_root_pos=np.array([0.0, 0.0, 0.5]),
        spawn_root_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        spawn_sole_tilt_deg=np.zeros(n_feet),
        home_sole_tilt_deg=np.zeros(n_feet),
        truncated=True,
    )


def edited(trace: EpisodeTrace, **arrays: Any) -> EpisodeTrace:
    """``trace`` with the named fields replaced (callers pass edited copies)."""
    return replace(trace, **arrays)


def metrics(trace: EpisodeTrace, settle_steps: int = 100) -> StanceEpisodeMetrics:
    return episode_stance_metrics(trace, settle_steps=settle_steps)


def same(a: Any, b: Any) -> bool:
    """Equality that counts NaN as equal to NaN, elementwise through dicts, tuples and lists."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[key], b[key]) for key in a)
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return bool(a == b) and type(a) is type(b)


# ── the statue ───────────────────────────────────────────────────────────────


def test_the_statue_scores_the_ideal_on_every_metric():
    result = metrics(statue_trace())
    expected = {
        "length": 400,
        "terminated": False,
        "reward": 400.0,
        "settle_steps": 100,
        "all_feet_support": 1.0,
        "flight_fraction": 0.0,
        "touchdown_rate": 0.0,
        "min_foot_load_share": 0.5,
        "min_foot_load_share_windowed": 0.5,
        "window_displacement_m": 0.0,
        "window_yaw_change_deg": 0.0,
        "episode_yaw_change_deg": 0.0,
        "max_foot_slide_m": 0.0,
        "foot_on_foot_fraction": 0.0,
        "phantom_support_fraction": 0.0,
        "touch_floor_agreement": 1.0,
        "nonfoot_load_fraction": 0.0,
        "offsupport_load_fraction": 0.0,
        "leg_body_contact_fraction": 0.0,
        "min_support_geom_duty": 1.0,
        "min_support_geom_coverage": 1.0,
        "max_sole_tilt_deg": 0.0,
        "max_sole_tilt_excess_deg": 0.0,
        "max_sole_abs_roll_deg": 0.0,
        "sole_rolled_fraction": 0.0,
        "max_sole_corner_lift_m": 0.0,
        "min_sole_contacts": 4.0,
        "max_sole_cop_outer": 0.0,
        "spawn_peak_floor_force_bw": 1.0,
        "spawn_airborne_substeps": 0.0,
        "settle_unsupported_steps": 0.0,
        "settle_unsupported_events": 0.0,
        "settle_airborne_substeps": 0.0,
        "settle_peak_floor_force_bw": 1.0,
        "settle_touchdowns": 0.0,
        "settle_stance_width_change_m": 0.0,
        "settle_max_foot_shift_m": 0.0,
        "max_stance_width_change_m": 0.0,
        "max_actuator_saturation_fraction": 0.0,
        "mean_actuator_saturation_fraction": 0.0,
        "foot_load_share": (0.5, 0.5),
        "support_geom_duty": (1.0, 1.0),
        "support_geom_share": (1.0, 1.0),
        "mean_sole_tilt_deg": (0.0, 0.0),
        "mean_sole_roll_deg": (0.0, 0.0),
        "mean_sole_pitch_deg": (0.0, 0.0),
        "mean_sole_corner_lift_m": (0.0, 0.0),
    }
    assert set(expected) == {spec.name for spec in fields(StanceEpisodeMetrics)}
    assert {name: getattr(result, name) for name in expected} == expected


# ── settle window: hops after the spawn grace, the reset pop inside it ───────


def test_a_hop_after_the_spawn_grace_fires_the_settle_hop_and_impact_checks():
    trace = statue_trace()
    down, airborne, peak = trace.leg_down_frac.copy(), trace.feet_airborne_substeps.copy(), trace.total_floor_max.copy()
    down[30:34] = 0.0  # 40 ms in the air at t = 0.3 s, well after the 0.1 s grace
    airborne[30:34] = 5
    peak[34] = 2.0 * BODY_WEIGHT  # the landing
    result = metrics(edited(trace, leg_down_frac=down, feet_airborne_substeps=airborne, total_floor_max=peak))
    assert result.settle_airborne_substeps == 20.0
    assert result.settle_unsupported_steps == 4.0
    assert result.settle_unsupported_events == 1.0
    assert result.settle_peak_floor_force_bw == pytest.approx(2.0)
    assert result.settle_touchdowns == 2.0  # each foot lands once
    assert result.spawn_peak_floor_force_bw == 1.0 and result.spawn_airborne_substeps == 0.0
    assert result.all_feet_support == 1.0  # the window itself is clean


def test_a_reset_pop_inside_the_spawn_grace_fires_only_the_spawn_metrics():
    trace = statue_trace()
    airborne, peak = trace.feet_airborne_substeps.copy(), trace.total_floor_max.copy()
    airborne[0] = 3  # every statue but trex pops off its reset pose; dibothrosuchus flies up to 9 substeps
    peak[2] = 2.4 * BODY_WEIGHT  # the velociraptor statue's pop reaches 2.42 body weights
    result = metrics(edited(trace, feet_airborne_substeps=airborne, total_floor_max=peak))
    assert result.spawn_airborne_substeps == 3.0
    assert result.spawn_peak_floor_force_bw == pytest.approx(2.4)
    assert result.settle_airborne_substeps == 0.0
    assert result.settle_peak_floor_force_bw == 1.0


def test_the_spawn_grace_is_defined_in_seconds():
    assert constants.SPAWN_GRACE_S == 0.10
    assert spawn_grace_steps(0.01) == 10
    assert spawn_grace_steps(0.02) == 5
    # At dt 0.02 the same 0.1 s grace is 5 steps: a hop at step 6 is a settle hop there.
    trace = statue_trace(dt=0.02, frame_skip=10)
    airborne = trace.feet_airborne_substeps.copy()
    airborne[6] = 2
    assert metrics(edited(trace, feet_airborne_substeps=airborne), settle_steps=50).settle_airborne_substeps == 2.0
    assert metrics(edited(statue_trace(), feet_airborne_substeps=airborne)).settle_airborne_substeps == 0.0


# ── load shares ──────────────────────────────────────────────────────────────


def test_a_one_leg_stance_has_a_zero_load_share():
    trace = statue_trace()
    load, down = trace.leg_floor_mean.copy(), trace.leg_down_frac.copy()
    load[100:, 0], load[100:, 1] = BODY_WEIGHT, 0.0
    down[100:, 1] = 0.0
    result = metrics(edited(trace, leg_floor_mean=load, leg_down_frac=down))
    assert result.min_foot_load_share == 0.0
    assert result.foot_load_share == (1.0, 0.0)
    assert result.all_feet_support == 0.0
    assert result.flight_fraction == 0.0


def test_the_windowed_share_catches_a_temporary_one_leg_stance_the_window_mean_averages_away():
    trace = statue_trace()  # window [100, 400): three whole 1 s blocks at dt 0.01
    load = trace.leg_floor_mean.copy()
    load[200:300, 0], load[200:300, 1] = BODY_WEIGHT, 0.0
    result = metrics(edited(trace, leg_floor_mean=load))
    assert result.min_foot_load_share == pytest.approx(1 / 3)
    assert result.min_foot_load_share_windowed == 0.0


def test_the_windowed_share_is_nan_without_one_whole_block():
    assert constants.LOAD_WINDOW_S == 1.0
    assert math.isnan(metrics(statue_trace(150)).min_foot_load_share_windowed)  # 0.5 s of window
    assert metrics(statue_trace(200)).min_foot_load_share_windowed == 0.5


# ── touchdowns and the debounce, in seconds ──────────────────────────────────


def test_ten_hertz_chatter_is_ten_touchdowns_per_foot_per_second():
    trace = statue_trace()
    down = trace.leg_down_frac.copy()
    phase = np.arange(400) % 10 >= 5  # 50 ms down, 50 ms up
    down[100:][phase[100:]] = 0.0
    result = metrics(edited(trace, leg_down_frac=down))
    assert result.touchdown_rate == pytest.approx(10.0, abs=0.5)
    assert result.all_feet_support == pytest.approx(0.5)
    assert result.flight_fraction == pytest.approx(0.5)


def test_the_debounce_drops_a_10_ms_flicker_at_dt_001_and_keeps_a_20_ms_one_at_dt_002():
    assert constants.DEBOUNCE_S == 0.020
    for dt, frame_skip, rate in ((0.01, 5, 0.0), (0.02, 10, 1 / (10 * 0.02))):
        trace = statue_trace(dt=dt, frame_skip=frame_skip)
        down = trace.leg_down_frac.copy()
        down[100:][np.arange(300) % 10 == 5] = 0.0  # one step up in every ten
        result = metrics(edited(trace, leg_down_frac=down))
        assert result.touchdown_rate == pytest.approx(rate, abs=0.2), dt
        assert result.all_feet_support == pytest.approx(0.9)  # the down state itself is not debounced


def test_a_foot_that_spawns_in_the_air_counts_its_first_landing():
    trace = statue_trace()
    down = trace.leg_down_frac.copy()
    down[0:3, 1] = 0.0  # the trex statue spawns one foot in the air on some seeds
    assert metrics(edited(trace, leg_down_frac=down)).settle_touchdowns == 1.0


# ── flatness ─────────────────────────────────────────────────────────────────


def test_a_rolled_box_sole_fails_every_flatness_metric():
    trace = statue_trace()
    tilt, roll, lift, contacts = (
        trace.sole_tilt_deg.copy(),
        trace.sole_roll_deg.copy(),
        trace.sole_corner_lift.copy(),
        trace.sole_contacts_mean.copy(),
    )
    tilt[100:, 0] = roll[100:, 0] = 3.0  # outer edge up 3 degrees on a 2 * 0.045 m wide pad
    lift[100:, 0] = 2 * 0.045 * math.sin(math.radians(3.0))
    contacts[100:, 0] = 1.0  # on a corner
    cop = trace.sole_cop.copy()
    cop[100:, 0, 1] = -1.0  # all the load on the inner edge
    result = metrics(
        edited(
            trace,
            sole_tilt_deg=tilt,
            sole_roll_deg=roll,
            sole_corner_lift=lift,
            sole_contacts_mean=contacts,
            sole_cop=cop,
        )
    )
    assert result.max_sole_tilt_deg == pytest.approx(3.0)
    assert result.max_sole_tilt_excess_deg == pytest.approx(3.0)
    assert result.max_sole_abs_roll_deg == pytest.approx(3.0)
    assert result.sole_rolled_fraction == 1.0  # 3 > SOLE_ROLLED_DEG
    assert result.max_sole_corner_lift_m == pytest.approx(0.0047, abs=1e-4)
    assert result.min_sole_contacts == 1.0
    assert result.max_sole_cop_outer == 1.0
    assert result.mean_sole_tilt_deg == pytest.approx((3.0, 0.0))


def test_an_authored_tilt_counts_only_beyond_the_keyframe():
    # Brachiosaurus authors its ellipsoid forefeet at 19.2 degrees; its statue reads 26-27 absolute.
    trace = statue_trace()
    tilt = np.full_like(trace.sole_tilt_deg, 20.0)
    lift = np.full_like(trace.sole_corner_lift, np.nan)  # an ellipsoid sole has no box corners
    home = np.array([19.2, 19.2])
    result = metrics(edited(trace, sole_tilt_deg=tilt, sole_corner_lift=lift, home_sole_tilt_deg=home))
    assert result.max_sole_tilt_deg == pytest.approx(20.0)
    assert result.max_sole_tilt_excess_deg == pytest.approx(0.8)
    assert result.sole_rolled_fraction == 0.0  # 0.8 < SOLE_ROLLED_DEG
    assert math.isnan(result.max_sole_corner_lift_m)


def test_feet_without_a_sole_leave_every_flatness_metric_unmeasured():
    trace = statue_trace()  # velociraptor: capsule toes, no sole frame
    nan = np.full_like(trace.sole_tilt_deg, np.nan)
    result = metrics(
        edited(
            trace,
            sole_tilt_deg=nan,
            sole_roll_deg=nan.copy(),
            sole_pitch_deg=nan.copy(),
            sole_corner_lift=nan.copy(),
            sole_contacts_mean=np.zeros_like(nan),
            sole_cop=np.full_like(trace.sole_cop, np.nan),
            home_sole_tilt_deg=np.full(2, np.nan),
        )
    )
    for name in (
        "max_sole_tilt_deg",
        "max_sole_tilt_excess_deg",
        "max_sole_abs_roll_deg",
        "sole_rolled_fraction",
        "max_sole_corner_lift_m",
        "min_sole_contacts",
        "max_sole_cop_outer",
    ):
        assert math.isnan(getattr(result, name)), name
    assert all(math.isnan(value) for value in result.mean_sole_tilt_deg)
    assert result.all_feet_support == 1.0


# ── support geoms ────────────────────────────────────────────────────────────


def test_a_support_geom_held_off_the_floor_has_zero_duty():
    # The velociraptor 0922 hack never loads its right d4 toe; its statue loads d3/d4/metatarsus 0.56/0.18/0.26.
    trace = statue_trace()
    loads = np.column_stack([trace.support_floor_mean, np.zeros(400)])  # a second geom on the left foot
    loaded = np.column_stack([trace.support_loaded_frac, np.full(400, 0.2)])  # loaded on 1 substep in 5
    result = metrics(
        edited(
            trace,
            support_geom_foot=(0, 1, 1),
            support_geom_names=("r_sole", "l_sole", "l_toe"),
            support_floor_mean=loads,
            support_loaded_frac=loaded,
        )
    )
    assert result.min_support_geom_duty == 0.0
    assert result.support_geom_duty == (1.0, 1.0, 0.0)
    assert result.min_support_geom_coverage == 0.5
    assert result.support_geom_share == (1.0, 1.0, 0.0)


def test_floor_load_off_the_support_registry_and_off_the_legs_is_measured():
    trace = statue_trace()
    nonleg, offsupport, total = (
        trace.nonleg_floor_mean.copy(),
        trace.offsupport_floor_mean.copy(),
        trace.total_floor_mean.copy(),
    )
    nonleg[100:] = 20.0  # a tail on the floor
    offsupport[100:, 0] = 5.0  # a shin on the floor
    total[100:] = BODY_WEIGHT + 25.0
    result = metrics(edited(trace, nonleg_floor_mean=nonleg, offsupport_floor_mean=offsupport, total_floor_mean=total))
    assert result.nonfoot_load_fraction == pytest.approx(20.0 / 125.0)
    assert result.offsupport_load_fraction == pytest.approx(5.0 / 125.0)


# ── foot-on-foot, phantom support, body contact ──────────────────────────────


def test_foot_on_foot_and_phantom_support_are_measured_against_the_floor():
    # The compsognathus_robot hack stands its right sole on its left foot: touch reads support, the floor does not.
    trace = statue_trace()
    interleg, touch, down, load = (
        trace.interleg_substep_frac.copy(),
        trace.touch.copy(),
        trace.leg_down_frac.copy(),
        trace.leg_floor_mean.copy(),
    )
    interleg[100:] = 0.95
    down[100:, 0] = 0.0
    load[100:, 0], load[100:, 1] = 0.0, BODY_WEIGHT
    touch[100:, 0] = 40.0  # the right foot's touch sensor is pressed by the other foot
    result = metrics(
        edited(trace, interleg_substep_frac=interleg, touch=touch, leg_down_frac=down, leg_floor_mean=load)
    )
    assert result.foot_on_foot_fraction == pytest.approx(0.95)
    assert result.phantom_support_fraction == 1.0
    assert result.touch_floor_agreement == 0.0
    assert result.min_foot_load_share == 0.0


def test_leg_against_body_contact_is_reported():
    trace = statue_trace()
    contact = trace.leg_body_substep_frac.copy()
    contact[100:250] = 1.0
    assert metrics(edited(trace, leg_body_substep_frac=contact)).leg_body_contact_fraction == pytest.approx(0.5)


# ── actions ──────────────────────────────────────────────────────────────────


def test_a_saturated_command_is_measured_at_the_saturation_threshold():
    assert constants.SATURATION_ABS == 0.99
    trace = statue_trace()
    action = trace.action.copy()
    action[100:, 0] = 1.0  # the velociraptor hack holds hip pitch at +1 on 98% of its steps
    action[100:, 1] = -0.99  # at the threshold counts
    action[100:, 2] = 0.989  # below it does not
    result = metrics(edited(trace, action=action))
    assert result.max_actuator_saturation_fraction == 1.0
    assert result.mean_actuator_saturation_fraction == pytest.approx(0.5)


def test_saturation_is_unmeasured_when_a_window_step_has_no_action():
    trace = statue_trace()
    action = trace.action.copy()
    action[250] = np.nan  # end_step() was given no action on that step
    result = metrics(edited(trace, action=action))
    assert math.isnan(result.max_actuator_saturation_fraction)
    assert math.isnan(result.mean_actuator_saturation_fraction)
    # A settle-window step without an action does not matter.
    action = trace.action.copy()
    action[50] = np.nan
    assert metrics(edited(trace, action=action)).max_actuator_saturation_fraction == 0.0


# ── motion: displacement, slides, yaw, stance width ──────────────────────────


def test_root_travel_foot_slides_and_yaw_are_measured_over_the_window():
    trace = statue_trace()
    root, feet, quat = trace.root_pos.copy(), trace.foot_pos.copy(), trace.root_quat.copy()
    root[99:, 0] = np.linspace(0.0, 0.2, 301)  # 0.2 m of travel from the last settle step
    feet[399, 1, 1] += 0.03
    half = math.radians(30.0) / 2
    quat[200:] = [math.cos(half), 0.0, 0.0, math.sin(half)]  # a 30 degree turn inside the window
    result = metrics(edited(trace, root_pos=root, foot_pos=feet, root_quat=quat))
    assert result.window_displacement_m == pytest.approx(0.2)
    assert result.max_foot_slide_m == pytest.approx(0.03)
    assert result.window_yaw_change_deg == pytest.approx(30.0)
    assert result.episode_yaw_change_deg == pytest.approx(30.0)


def test_a_quadruped_measures_all_six_foot_pairs():
    trace = statue_trace(n_feet=4)
    feet = trace.foot_pos.copy()
    feet[99:, 3, 1] += 0.05  # the left hind foot steps out sideways during the settle
    result = metrics(edited(trace, foot_pos=feet))
    assert result.foot_load_share == (0.25, 0.25, 0.25, 0.25)
    assert result.min_foot_load_share == 0.25
    assert result.settle_stance_width_change_m == pytest.approx(0.05)  # the rr-rl pair
    assert result.settle_max_foot_shift_m == pytest.approx(0.05)
    assert result.max_stance_width_change_m == pytest.approx(0.05)
    # A move along a pair's own line changes only the pairs that see it: the diagonal ones less.
    feet = trace.foot_pos.copy()
    feet[99:, 0, 0] += 0.02  # the right forefoot steps forward
    result = metrics(edited(trace, foot_pos=feet))
    assert result.settle_stance_width_change_m == pytest.approx(0.02)  # the fr-rr pair


# ── NaN: unmeasured, never passing ───────────────────────────────────────────


def test_an_episode_that_ends_before_the_window_leaves_every_window_metric_unmeasured():
    result = metrics(statue_trace(80))
    for name in _WINDOW_FIELDS:
        assert math.isnan(getattr(result, name)), name
    assert all(math.isnan(value) for value in result.foot_load_share)
    # The settle checks still see [grace, 80).
    assert result.settle_airborne_substeps == 0.0
    assert result.settle_peak_floor_force_bw == 1.0
    assert math.isnan(result.settle_stance_width_change_m)  # the settle window did not end


def test_a_settle_window_inside_the_spawn_grace_leaves_the_settle_checks_unmeasured():
    result = metrics(statue_trace(), settle_steps=10)  # = the grace at dt 0.01
    for name in (
        "settle_unsupported_steps",
        "settle_unsupported_events",
        "settle_airborne_substeps",
        "settle_peak_floor_force_bw",
    ):
        assert math.isnan(getattr(result, name)), name
    assert result.spawn_peak_floor_force_bw == 1.0
    assert result.all_feet_support == 1.0
    result = metrics(statue_trace(), settle_steps=0)
    assert math.isnan(result.settle_airborne_substeps) and math.isnan(result.spawn_peak_floor_force_bw)
    assert result.window_displacement_m == 0.0  # measured from the spawn


def test_a_non_finite_reward_makes_the_episode_reward_unmeasured():
    trace = statue_trace()
    reward = trace.reward.copy()
    reward[5] = np.nan
    assert math.isnan(metrics(edited(trace, reward=reward)).reward)


def test_a_floor_with_no_foot_load_leaves_the_shares_unmeasured():
    trace = statue_trace()
    load = trace.leg_floor_mean.copy()
    load[100:] = 0.0
    total = trace.total_floor_mean.copy()
    total[100:] = 0.0
    result = metrics(edited(trace, leg_floor_mean=load, total_floor_mean=total))
    assert math.isnan(result.min_foot_load_share)
    assert math.isnan(result.nonfoot_load_fraction)
    assert result.min_foot_load_share_windowed == 0.0  # a block with no load is the worst share


@pytest.mark.parametrize("settle_steps", [-1, 2.5, True])
def test_settle_steps_must_be_a_whole_non_negative_count(settle_steps):
    with pytest.raises(ValueError, match="settle_steps"):
        episode_stance_metrics(statue_trace(), settle_steps=settle_steps)


def test_a_trace_with_no_steps_is_refused():
    with pytest.raises(ValueError, match="no steps"):
        metrics(statue_trace(0))


# ── the row: as_row / from_row and the CSV columns ───────────────────────────


def _defective_metrics() -> StanceEpisodeMetrics:
    trace = statue_trace(n_feet=4)
    down = trace.leg_down_frac.copy()
    down[100:][np.arange(300) % 7 == 3] = 0.0
    nan = np.full_like(trace.sole_corner_lift, np.nan)
    return metrics(edited(trace, leg_down_frac=down, sole_corner_lift=nan), settle_steps=150)


def test_a_row_round_trips_through_json_values_and_through_csv_strings():
    original = _defective_metrics()
    assert any(math.isnan(value) for value in original.mean_sole_corner_lift_m)  # NaN survives the trip
    row = original.as_row()
    assert all(isinstance(row[name], list) for name in STANCE_METRIC_FOOT_FIELDS)
    assert same(StanceEpisodeMetrics.from_row(row).as_row(), row)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=["seed", *row])
    writer.writeheader()
    writer.writerow({"seed": 3042, **row})
    (read,) = csv.DictReader(io.StringIO(buffer.getvalue(), newline=""))
    restored = StanceEpisodeMetrics.from_row(read)
    assert same(restored.as_row(), row)


def test_from_row_refuses_a_missing_or_unreadable_field():
    row = _defective_metrics().as_row()
    with pytest.raises(ValueError, match="lacks"):
        StanceEpisodeMetrics.from_row({key: value for key, value in row.items() if key != "touchdown_rate"})
    for name, value in (
        ("length", "1000.5"),
        ("length", ""),
        ("terminated", "maybe"),
        ("reward", "high"),
        ("reward", True),
    ):
        with pytest.raises(ValueError, match=name):
            StanceEpisodeMetrics.from_row({**row, name: value})
    assert math.isnan(StanceEpisodeMetrics.from_row({**row, "reward": ""}).reward)  # empty is unmeasured
    assert StanceEpisodeMetrics.from_row({**row, "terminated": "True"}).terminated is True


def test_the_csv_columns_are_the_scalar_fields_in_order_and_hold_every_gated_metric():
    names = [spec.name for spec in fields(StanceEpisodeMetrics)]
    assert list(STANCE_METRIC_FIELDS) == [name for name in names if name not in STANCE_METRIC_FOOT_FIELDS]
    assert list(STANCE_METRIC_FOOT_FIELDS) == [name for name in names if name not in STANCE_METRIC_FIELDS]
    # The episode metric every stance_quality/v2 threshold key reads (DESIGN: required and optional keys).
    gated = {
        "length",
        "reward",
        "all_feet_support",
        "touchdown_rate",
        "window_displacement_m",
        "min_foot_load_share",
        "max_actuator_saturation_fraction",
        "settle_airborne_substeps",
        "settle_peak_floor_force_bw",
        "min_foot_load_share_windowed",
        "foot_on_foot_fraction",
        "phantom_support_fraction",
        "nonfoot_load_fraction",
        "settle_stance_width_change_m",
        "min_support_geom_duty",
        "min_support_geom_coverage",
        "max_sole_tilt_deg",
        "max_sole_tilt_excess_deg",
        "max_sole_corner_lift_m",
        "min_sole_contacts",
    }
    assert gated <= set(STANCE_METRIC_FIELDS)
    assert "foot_load_share" in STANCE_METRIC_FOOT_FIELDS  # the statue-relative share ratio


# ── events ───────────────────────────────────────────────────────────────────


def test_steps_for_rounds_seconds_to_at_least_one_step():
    assert events.steps_for(0.020, 0.01) == 2
    assert events.steps_for(0.020, 0.02) == 1
    assert events.steps_for(0.001, 0.02) == 1
    assert events.steps_for(1.0, 0.01) == 100


def test_down_is_at_least_half_the_substeps():
    assert constants.DOWN_SUBSTEP_FRACTION == 0.5
    assert events.down_mask(np.array([0.0, 0.4, 0.5, 1.0])).tolist() == [False, False, True, True]


def test_debounce_fills_short_gaps_then_drops_short_blips_and_keeps_the_ends():
    series = np.array([0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1], dtype=bool)
    assert events.debounce(series, 1).tolist() == series.tolist()
    debounced = events.debounce(series, 2)
    # The 1-step gap at 3 is filled; the 1-step blip at 10 is dropped; the 1-step runs at
    # either end are kept.
    assert debounced.astype(int).tolist() == [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1]
    assert events.debounce(np.zeros(0, dtype=bool), 2).size == 0


def test_edges_and_runs():
    series = np.array([1, 0, 0, 1, 1, 0, 1], dtype=bool)
    assert events.rising_edges(series).tolist() == [3, 6]
    assert events.falling_edges(series).tolist() == [1, 5]
    assert events.count_runs(series) == 3
    assert events.count_runs(np.zeros(4, dtype=bool)) == 0


# ── layering ─────────────────────────────────────────────────────────────────

PURE_MODULES = ("__init__.py", "constants.py", "events.py", "stance_metrics.py")


def _import_time_imports(path: Path) -> list[str]:
    """Modules imported when ``path`` is imported (``TYPE_CHECKING`` blocks and function bodies excluded)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: list[str] = []
    pending: list[ast.AST] = list(tree.body)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(node, ast.If) and ast.unparse(node.test).split(".")[-1] == "TYPE_CHECKING":
            pending += node.orelse
            continue
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module is None:
            imported += ["." * node.level + alias.name for alias in node.names]  # from . import events
        elif isinstance(node, ast.ImportFrom):
            imported.append("." * node.level + str(node.module))
        pending += ast.iter_child_nodes(node)
    return imported


@pytest.mark.parametrize("module", PURE_MODULES)
def test_the_pure_modules_import_no_mujoco(module):
    """The gate (pure) imports constants, events and stance_metrics: they import only the standard library,
    numpy and each other."""
    allowed_relative = {"." + name.removesuffix(".py") for name in PURE_MODULES if name != "__init__.py"}
    imported = _import_time_imports(GAIT / module)
    foreign = [
        name
        for name in imported
        if name.split(".")[0] not in sys.stdlib_module_names and name != "numpy" and name not in allowed_relative
    ]
    assert foreign == []


def test_the_gait_package_imports_neither_curriculum_nor_reporting():
    """Both build on gait/ (the v2 gate, the stance report); an import the other way would be a cycle."""
    offenders = []
    for path in sorted(GAIT.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [alias.name for alias in node.names]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            if any(part in ("curriculum", "reporting") for name in names for part in name.split(".")):
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == []


def test_no_species_env_module_imports_the_gait_package():
    """An import line would change an env module's bytes, and with them its identity (and digests)."""
    modules = sorted((REPO_ROOT / "environments").glob("*/envs/*.py"))
    modules += sorted((REPO_ROOT / "environments").glob("*/mjx_config.py"))
    modules += [
        REPO_ROOT / "environments" / "shared" / name for name in ("base_env.py", "behavior_env.py", "mjx_env.py")
    ]
    assert len(modules) > 10
    offenders = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in modules
        if any("gait" in name.split(".") for name in _all_imports(path))
    ]
    assert offenders == []


def _all_imports(path: Path) -> list[str]:
    names = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names += [module] + [f"{module}.{alias.name}" for alias in node.names]
        elif isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
    return names
