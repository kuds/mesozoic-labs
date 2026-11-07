"""Crouch rise must move under finite effort and sole-only floor support."""

from __future__ import annotations

import json

import mujoco
import numpy as np
import pytest

from examples.tyrannosaurus_rex_50cm.demo import load_model
from examples.tyrannosaurus_rex_50cm.recovery_checks import (
    LEG_JOINTS,
    _settling_window,
    prepare_crouch,
    run_crouch_trial,
)


def test_initial_crouch_has_level_grounded_soles_and_preserves_model():
    model, _ = load_model()
    fields = ("key_qpos", "key_ctrl", "body_mass", "body_inertia", "actuator_forcerange", "jnt_range", "eq_active0")
    before = {name: getattr(model, name).copy() for name in fields}
    data = prepare_crouch(model, knee_angle=0.6)
    assert data.qpos[2] < model.key("home").qpos[2] - 0.02
    for side in ("left", "right"):
        sole = model.geom(side + "_sole_collision").id
        assert data.geom_xpos[sole, 2] - model.geom_size[sole, 2] == pytest.approx(0, abs=1e-10)
        np.testing.assert_allclose(data.geom_xmat[sole].reshape(3, 3)[:, 2], [0, 0, 1], atol=1e-10)
        assert data.qpos[model.joint(side + "_knee").qposadr[0]] == pytest.approx(0.6)
    assert not np.any(data.eq_active)
    assert not np.any(data.xfrc_applied) and not np.any(data.qfrc_applied)
    for name, expected in before.items():
        np.testing.assert_array_equal(getattr(model, name), expected)


def test_current_model_rises_from_a_held_crouch_without_external_support():
    model, _ = load_model()
    original = model.actuator_forcerange.copy()
    report = run_crouch_trial(model, knee_angle=0.6, duration=6)
    assert report["passed"]
    assert all(report["checks"].values())
    assert report["requested_crouch_depth_m"] > 0.02
    assert report["measured_rise_m"] > 0.02
    assert report["final_root_height_mean_m"] > 0.95 * report["home_root_height_m"]
    assert min(report["held_foot_normal_mean_N"]) > 30
    assert min(report["settled_foot_normal_mean_N"]) > 30
    assert report["held_window"]["settled"] and report["settled_window"]["settled"]
    assert report["settled_window"]["simultaneous_foot_support_fraction"] >= 0.95
    assert report["settled_window"]["maximum_root_linear_speed_m_s"] < 0.01
    assert report["maximum_unexpected_support_N"] < 0.05
    assert max(report["peak_torque_nm"]) < 4
    samples = report["pose_samples"]
    assert 24 <= len(samples) <= 27
    assert samples[0]["time_s"] == 0
    assert samples[-1]["time_s"] == pytest.approx(report["duration_s"])
    assert all(len(sample["qpos"]) == model.nq for sample in samples)
    assert samples[-1]["qpos"][2] > samples[0]["qpos"][2] + 0.02
    assert all(a["time_s"] < b["time_s"] for a, b in zip(samples, samples[1:], strict=False))
    np.testing.assert_array_equal(model.actuator_forcerange, original)
    json.dumps(report, allow_nan=False)


def test_remaining_upright_without_a_rise_command_cannot_pass():
    model, _ = load_model()
    report = run_crouch_trial(model, knee_angle=0.6, duration=6, rise=False)
    assert report["checks"]["completed"]
    assert report["checks"]["initial_crouch_held"]
    assert report["checks"]["bounded_tilt"]
    assert report["measured_rise_m"] < 0.002
    assert not report["checks"]["measured_rise"]
    assert not report["passed"]


def test_underpowered_legs_fail_without_relaxing_or_overwriting_the_caps():
    model, _ = load_model()
    for name in LEG_JOINTS:
        model.actuator(name + "_servo").forcerange[:] = [-0.1, 0.1]
    limits = model.actuator_forcerange.copy()
    report = run_crouch_trial(model, knee_angle=0.6, duration=6)
    assert not report["passed"]
    assert not report["checks"]["initial_crouch_held"]
    assert report["maximum_unexpected_support_N"] > 0.05
    np.testing.assert_array_equal(model.actuator_forcerange, limits)


def test_loaded_display_fixture_is_rejected_before_simulation():
    model, _ = load_model()
    model.eq_active0[model.equality("display_support").id] = 1
    with pytest.raises(ValueError, match="fixture"):
        run_crouch_trial(model, duration=6)
    assert model.eq_active0[model.equality("display_support").id]


@pytest.mark.parametrize("angle", [0, -0.3, float("nan"), float("inf"), 1.2])
def test_invalid_or_unreachable_crouch_angles_are_rejected(angle):
    model, _ = load_model()
    with pytest.raises(ValueError, match="knee_angle|range"):
        prepare_crouch(model, knee_angle=angle)


def test_trial_needs_time_to_hold_rise_and_settle():
    model, _ = load_model()
    with pytest.raises(ValueError, match="six seconds"):
        run_crouch_trial(model, duration=3)


@pytest.mark.parametrize("actuator", ["left_knee_servo", "head_nod_servo"])
@pytest.mark.parametrize("gear", [1.1, 1.2])
def test_geared_actuators_cannot_report_actuator_force_as_joint_torque(actuator, gear):
    model, _ = load_model()
    model.actuator(actuator).gear[0] = gear
    with pytest.raises(ValueError, match="unit gear"):
        run_crouch_trial(model, duration=6)
    assert model.actuator(actuator).gear[0] == gear


def test_torque_control_cannot_be_driven_with_joint_angle_targets():
    model, _ = load_model()
    actuator = model.actuator("left_knee_servo").id
    model.actuator_biastype[actuator] = mujoco.mjtBias.mjBIAS_NONE
    with pytest.raises(ValueError, match="position drive"):
        prepare_crouch(model)


def test_additional_joint_force_limit_cannot_hide_behind_actuator_cap():
    model, _ = load_model()
    model.jnt_actfrclimited[model.joint("left_knee").id] = True
    with pytest.raises(ValueError, match="additional force limit"):
        prepare_crouch(model)


def test_actuators_cannot_share_one_joint_and_leave_another_uncontrolled():
    model, _ = load_model()
    model.actuator_trnid[1, 0] = model.actuator_trnid[0, 0]
    with pytest.raises(ValueError, match="exactly one"):
        prepare_crouch(model)


def test_settling_requires_simultaneous_support_and_quiet_motion_not_only_good_means():
    heights = [0.23] * 100
    root_velocities = [np.zeros(6) for _ in heights]
    leg_velocities = [np.zeros(12) for _ in heights]
    loads = [np.array([50.0, 50.0]) for _ in heights]
    stable = _settling_window(heights, root_velocities, leg_velocities, loads, 100)
    assert stable["settled"]

    # Both feet average half the body weight, but they never support it together.
    alternate = [np.array([100.0, 0.0]) if index % 2 else np.array([0.0, 100.0]) for index in range(100)]
    np.testing.assert_allclose(np.mean(alternate, axis=0), np.mean(loads, axis=0))
    changing_feet = _settling_window(heights, root_velocities, leg_velocities, alternate, 100)
    assert changing_feet["simultaneous_foot_support_fraction"] == 0
    assert not changing_feet["settled"]

    # A bouncing root can have the exact desired mean height and zero mean speed.
    phase = np.arange(100) * 2 * np.pi / 100
    bouncing_heights = (0.23 + 0.003 * np.sin(phase)).tolist()
    bouncing_velocities = [np.array([0, 0, 0.003 * 2 * np.pi * np.cos(value), 0, 0, 0]) for value in phase]
    assert np.mean(bouncing_heights) == pytest.approx(np.mean(heights))
    np.testing.assert_allclose(np.mean(bouncing_velocities, axis=0), np.zeros(6), atol=1e-12)
    bouncing = _settling_window(bouncing_heights, bouncing_velocities, leg_velocities, loads, 100)
    assert not bouncing["settled"]
    assert bouncing["height_span_m"] > stable["height_span_m"]
