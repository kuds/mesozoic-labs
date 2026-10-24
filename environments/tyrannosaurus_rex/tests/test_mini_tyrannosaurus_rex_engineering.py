"""Concept revision gates for actual motor space, linked arms and head clearance.

These intentionally do not certify fabrication tolerances, motor thermal duty,
or walking. The unsupported-standing regression lives in the example tests.
"""

import json
import math

import mujoco
import numpy as np
import pytest

from examples.mini_tyrannosaurus_rex import demo, engineering


def test_all_actuators_have_full_size_motor_cases_with_sampled_clearance():
    model, _ = demo.load_model()
    cases = engineering.motor_cases(model)
    assert len(cases) == model.nu == 14
    for geom in cases:
        assert np.sort(2 * model.geom_size[geom]) == pytest.approx([0.020, 0.026, 0.034])

    report = engineering.motor_clearance_sweep(model, samples=64)
    assert report["overlap_poses"] == 0, report["witness"]
    assert report["minimum_separating_axis_gap_m"] > 0.001, report["witness"]


def test_linked_arms_move_together_with_one_actuator_under_gravity():
    model, data = demo.load_model(supported=True)
    actuator = model.actuator("left_shoulder_pitch_servo").id
    left, right = [model.joint(f"{side}_shoulder_pitch").id for side in ("left", "right")]
    addresses = [model.jnt_qposadr[joint] for joint in (left, right)]
    direct_arm_actuators = [
        index
        for index in range(model.nu)
        if model.actuator_trntype[index] == mujoco.mjtTrn.mjTRN_JOINT
        and model.actuator_trnid[index, 0] in (left, right)
    ]
    assert direct_arm_actuators == [actuator]
    minimum, maximum, maximum_link_error = 0.0, 0.0, 0.0
    for _ in range(round(4.0 / model.opt.timestep)):
        ramp = min(float(data.time), 1.0)
        data.ctrl[actuator] = ramp * (-0.225 + 0.325 * math.sin(math.pi * data.time))
        mujoco.mj_step(model, data)
        angles = data.qpos[addresses]
        minimum, maximum = min(minimum, float(angles[0])), max(maximum, float(angles[0]))
        maximum_link_error = max(maximum_link_error, float(abs(angles[0] - angles[1])))
        assert np.isfinite(data.qpos).all()
    assert maximum - minimum > 0.40
    assert maximum_link_error < 0.003
    assert not data.warning.number.any()


def test_head_jaw_and_linked_arm_command_grid_has_no_proxy_intersections():
    model, _ = demo.load_model()
    report = engineering.upper_body_collision_sweep(model, points=5)
    assert set(report["joint_ranges_rad"]) == {"head_yaw", "head_pitch", "jaw", "left_shoulder_pitch"}
    assert report["poses"] == 625
    assert report["maximum_penetration_m"] < 0.0001, report["worst_poses"]


def test_home_hip_roll_load_has_design_reserve_and_balanced_support():
    model, _ = demo.load_model()
    report = engineering.static_equilibrium(model)
    assert report["nonnegative_normal_forces"]
    assert sum(report["normal_force_fractions"].values()) == pytest.approx(1.0)
    assert np.max(np.abs(report["base_wrench_residual"])) < 1e-8
    for side in ("left", "right"):
        assert abs(report["joint_loads_Nm"][f"{side}_hip_roll"]) < 0.05
        assert abs(report["joint_loads_Nm"][f"{side}_knee"]) < 0.05
    # This is a geometry target relative to an estimated .10 Nm duty figure,
    # not proof that the real XL330 can sustain any particular duty cycle.


def test_component_mass_and_motor_budget_matches_model_allocation():
    model, _ = demo.load_model()
    ledger = json.loads(demo.MODEL_PATH.with_name("components.json").read_text())
    assert sum(row["mass_budget_g"] for row in ledger["components"]) == pytest.approx(model.body_mass.sum() * 1000)
    assert sum(ledger["actuator_allocation"].values()) == ledger["active_servo_count"] == model.nu
    assert not ledger["budget_reconciliation"]["all_parts_weighed"]


def test_sampled_combined_leg_and_expression_commands_are_clear():
    model, _ = demo.load_model()
    report = engineering.combined_collision_sweep(model, samples=128)
    assert report["maximum_penetration_m"] < 0.0001, report["witness"]


def test_single_foot_probe_matches_gravity_line_after_a_flat_foot_lean():
    model, data = demo.load_model()
    lean = 0.10
    data.qpos[3:7] = [math.cos(lean / 2), math.sin(lean / 2), 0, 0]
    engineering.set_pose(model, data, {"left_hip_roll": -lean, "right_hip_roll": -lean})
    com = data.subtree_com[model.body("torso").id]
    for side in ("left", "right"):
        site = model.site(f"{side}_sole_center").id
        rotation = data.site_xmat[site].reshape(3, 3)
        assert rotation[:, 2] == pytest.approx([0, 0, 1], abs=1e-10)
        expected = (rotation.T @ (com - data.site_xpos[site]))[:2]
        report = engineering.single_foot_cop(model, data, side)
        assert report["cop_from_sole_center_m"] == pytest.approx(expected)
        assert report["normal_force_N"] == pytest.approx(model.body_mass.sum() * 9.81)
