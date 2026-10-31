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


@pytest.mark.parametrize(
    "hold", [engineering.free_hold, engineering.leg_cap_sensitivity], ids=["configured", "sensitivity"]
)
@pytest.mark.parametrize("assistance", ["fixture", "gravity-compensation", "disabled-gravity", "disabled-contact"])
def test_engineering_holds_refuse_assisted_or_disabled_physics(hold, assistance):
    model, _ = demo.load_model()
    if assistance == "fixture":
        model.eq_active0[model.equality("display_support").id] = True
        message = "world weld"
    elif assistance == "gravity-compensation":
        model.body_gravcomp[model.body("torso").id] = 0.5
        message = "gravity compensation"
    else:
        model.opt.disableflags |= (
            mujoco.mjtDisableBit.mjDSBL_GRAVITY
            if assistance == "disabled-gravity"
            else mujoco.mjtDisableBit.mjDSBL_CONTACT
        )
        message = "physics must be enabled"
    original_limits = model.actuator_forcerange.copy()
    with pytest.raises(ValueError, match=message):
        hold(model, duration=3)
    assert model.actuator_forcerange == pytest.approx(original_limits)


@pytest.mark.parametrize(
    "hold", [engineering.free_hold, engineering.leg_cap_sensitivity], ids=["configured", "sensitivity"]
)
def test_engineering_holds_preserve_weaker_motor_limits(hold):
    model, _ = demo.load_model()
    legs = [i for i in range(model.nu) if any(part in model.actuator(i).name for part in ("hip_", "knee", "ankle"))]
    model.actuator_forcerange[legs] = [-0.025, 0.025]
    original_limits = model.actuator_forcerange.copy()
    report = hold(model, duration=3)
    assert model.actuator_forcerange == pytest.approx(original_limits)
    for i in legs:
        assert report["actuators"][model.actuator(i).name]["peak_Nm"] <= 0.025 + 1e-12
    assert report["minimum_height_ratio"] < 0.9 or report["maximum_tilt_deg"] > 10


@pytest.mark.parametrize(
    "hold", [engineering.free_hold, engineering.leg_cap_sensitivity], ids=["configured", "sensitivity"]
)
def test_engineering_holds_report_nonfoot_ground_support(hold):
    # Put the torso collision proxy on the floor while leaving both declared
    # soles unchanged. This contact must not be counted as legitimate support.
    spec = mujoco.MjSpec.from_file(str(demo.MODEL_PATH))
    spec.geom("torso_collision").pos = [0, 0, -0.105]
    model = spec.compile()
    report = hold(model, duration=3)
    assert report["unexpected_ground_contact"]
    assert report["maximum_unexpected_support_N"] > 0.05


def test_free_hold_simulates_the_configured_asymmetric_limits(monkeypatch):
    model, _ = demo.load_model()
    model.actuator_forcerange[model.actuator("left_knee_servo").id] = [-0.025, 0.20]
    original_limits = model.actuator_forcerange.copy()
    step = mujoco.mj_step
    samples = 0

    def observe_limits(current_model, data):
        nonlocal samples
        # Check the limits during real dynamics, not only after the helper has
        # restored them. A symmetric minimum would silently weaken this motor.
        assert current_model.actuator_forcerange == pytest.approx(original_limits)
        samples += 1
        step(current_model, data)

    monkeypatch.setattr(mujoco, "mj_step", observe_limits)
    report = engineering.free_hold(model, duration=3)
    assert samples > 0
    assert report["duration_s"] == pytest.approx(3, abs=model.opt.timestep)
    assert model.actuator_forcerange == pytest.approx(original_limits)


def test_sensitivity_restores_limits_if_simulation_raises(monkeypatch):
    model, _ = demo.load_model()
    original_limits = model.actuator_forcerange.copy()
    knee = model.actuator("left_knee_servo").id
    step = mujoco.mj_step

    def fail_after_step(current_model, data):
        assert current_model.actuator_forcerange[knee] == pytest.approx([-0.10, 0.10])
        step(current_model, data)
        raise RuntimeError("injected simulation failure")

    monkeypatch.setattr(mujoco, "mj_step", fail_after_step)
    with pytest.raises(RuntimeError, match="injected simulation failure"):
        engineering.leg_cap_sensitivity(model, duration=3)
    assert model.actuator_forcerange == pytest.approx(original_limits)
