"""Static ground-load bookkeeping and limits for the medium motor-location study."""

import copy
import json

import mujoco
import numpy as np
import pytest

from examples.tyrannosaurus_rex_50cm.demo import MODEL_PATH
from examples.tyrannosaurus_rex_50cm.leg_load_screen import MOVED_MOTORS, compare_motor_locations
from examples.tyrannosaurus_rex_50cm.recovery_checks import prepare_crouch


def inputs():
    return mujoco.MjModel.from_xml_path(str(MODEL_PATH)), json.loads(
        MODEL_PATH.with_name("components.json").read_text()
    )["components"]


def test_remote_motor_mass_can_raise_crouch_torque_despite_reducing_leg_mass():
    model, components = inputs()
    report = compare_motor_locations(model, components)
    home, crouch = report["poses"]
    assert report["relocated_motor_mass_kg"] == pytest.approx(1.98)
    assert home["root_height_m"] == pytest.approx(0.253)
    assert crouch["root_height_m"] == pytest.approx(0.1963121224)
    assert abs(crouch["actual_layout"]["required_leg_torque_nm"]["left_knee"]) == pytest.approx(2.9329829812)
    assert abs(crouch["ideal_torso_relocation"]["required_leg_torque_nm"]["left_knee"]) == pytest.approx(3.7946113212)
    assert all(crouch["actual_layout"]["checks"].values())
    assert not crouch["ideal_torso_relocation"]["checks"]["within_gravity_torque_budget"]
    assert home["ideal_torso_relocation"]["whole_body_com_m"][2] > home["actual_layout"]["whole_body_com_m"][2]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("pose_index", [0, 1])
@pytest.mark.parametrize("scenario", ["actual_layout", "ideal_torso_relocation"])
def test_required_knee_torque_matches_independent_component_and_ground_moments(pose_index, scenario):
    model, components = inputs()
    report = compare_motor_locations(model, components)
    if pose_index:
        data = prepare_crouch(model, knee_angle=1.1)
    else:
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        mujoco.mj_forward(model, data)
    row = report["poses"][pose_index][scenario]
    joint = model.joint("left_knee").id
    subtree = int(model.jnt_bodyid[joint])
    moment = np.zeros(3)
    for part in components:
        if scenario == "ideal_torso_relocation" and part["name"] in MOVED_MOTORS:
            continue  # Relocated motors are above this knee in the torso.
        body = ancestor = model.body(part["body"]).id
        while ancestor > 0 and ancestor != subtree:
            ancestor = int(model.body_parentid[ancestor])
        if ancestor != subtree:
            continue
        point = data.xpos[body] + data.xmat[body].reshape(3, 3) @ np.asarray(part["pos_m"])
        moment += np.cross(point - data.xanchor[joint], part["mass_kg"] * model.opt.gravity)
    foot = row["feet"]["left"]
    moment += np.cross(
        np.array(foot["assumed_center_of_pressure_m"]) - data.xanchor[joint],
        [0, 0, foot["assumed_vertical_force_n"]],
    )
    assert row["required_leg_torque_nm"]["left_knee"] == pytest.approx(-np.dot(moment, data.xaxis[joint]), abs=1e-9)
    assert row["free_root_equilibrium_residual_norm"] < 1e-8


def test_unsupported_center_of_pressure_is_reported_even_with_adequate_torque():
    model, components = inputs()
    for side in ("left", "right"):
        model.geom_size[model.geom(side + "_sole_collision").id, 0] = 0.001
    report = compare_motor_locations(model, components)
    home = report["poses"][0]["actual_layout"]
    assert not home["checks"]["cop_inside_both_soles"]
    assert home["checks"]["within_gravity_torque_budget"]
    assert home["feet"]["left"]["cop_margin_xy_m"][0] < 0


def test_study_rejects_a_ledger_that_does_not_describe_the_compiled_model():
    model, components = inputs()
    components[0]["mass_kg"] += 0.1
    with pytest.raises(ValueError, match="body masses do not match"):
        compare_motor_locations(model, components)


def test_study_rejects_a_crouch_outside_joint_ranges():
    model, components = inputs()
    with pytest.raises(ValueError, match="exceeds the range"):
        compare_motor_locations(model, components, knee_angle=2)


def test_study_preserves_model_and_ledger():
    model, components = inputs()
    fields = (
        "body_mass",
        "body_ipos",
        "body_inertia",
        "geom_pos",
        "geom_size",
        "jnt_range",
        "actuator_forcerange",
        "key_qpos",
    )
    original = {name: getattr(model, name).copy() for name in fields}
    ledger = copy.deepcopy(components)
    first = compare_motor_locations(model, components)
    assert first == compare_motor_locations(model, components)
    assert components == ledger
    for name, before in original.items():
        np.testing.assert_array_equal(getattr(model, name), before)
