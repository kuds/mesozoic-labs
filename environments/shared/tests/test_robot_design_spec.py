"""Metadata and solver-scope checks for reusable mechanical design inputs."""

import json
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from environments.shared.robot_design_spec import (
    ClearancePair,
    JointLoadSpec,
    PoseSweep,
    RobotDesignSpec,
    pose_count,
    validate_spec,
)


def _scene(*, extra_actuator=""):
    model = mujoco.MjModel.from_xml_string(f"""
    <mujoco>
      <compiler angle="radian"/>
      <option gravity="0 0 -9.81"/>
      <worldbody>
        <geom name="floor" type="plane" size="10 10 .1"/>
        <body name="torso" pos="0 0 .5">
          <freejoint name="root"/>
          <geom name="torso_case" type="box" size=".1 .1 .1" mass="1"/>
          <body name="attachment"><site name="cable_route"/></body>
          <body name="leg" pos="0 0 -.2">
            <joint name="leg_pitch" type="hinge" axis="0 1 0" range="-1 1"/>
            <geom name="sole" type="box" size=".1 .1 .05" mass=".2"/>
          </body>
          <body name="head" pos=".2 0 0">
            <joint name="head_pitch" type="hinge" axis="0 1 0" range="-1 1"/>
            <geom name="head_case" type="box" size=".1 .05 .05" mass=".3"/>
          </body>
        </body>
      </worldbody>
      <actuator>
        <motor name="head_motor" joint="head_pitch" gear="1" forcelimited="true" forcerange="-.5 .8"/>
        <motor name="leg_motor" joint="leg_pitch" gear="1" forcelimited="true" forcerange="-2 2"/>
        {extra_actuator}
      </actuator>
      <equality><weld name="fixture" body1="torso" active="false"/></equality>
      <keyframe><key name="home" qpos="0 0 .5 1 0 0 0 0 0"/></keyframe>
    </mujoco>
    """)
    spec = RobotDesignSpec(
        robot_id="synthetic_robot",
        root_joint="root",
        home_keyframe="home",
        feet={"support": ("sole",)},
        loads=(JointLoadSpec("head_pitch", "head_motor"),),
        sweeps=(PoseSweep("head", {"head_pitch": (-1, 0, 1)}),),
        clearances=(ClearancePair("torso_case", "head_case", 0.001),),
        assumptions=("Synthetic example; capacities are estimated.",),
    )
    return model, spec


def test_valid_spec_preserves_model_and_serializes_all_inputs():
    model, spec = _scene()
    before = [field.copy() for field in (model.key_qpos, model.body_mass, model.actuator_forcerange)]
    validate_spec(model, spec)
    assert model.body("attachment").mass == 0  # Fixed massless frames are permitted.
    for field, expected in zip((model.key_qpos, model.body_mass, model.actuator_forcerange), before):
        np.testing.assert_array_equal(field, expected)
    encoded = json.loads(json.dumps(spec.as_dict(), allow_nan=False))
    assert encoded["mass_evidence"] == "estimated"
    assert encoded["loads"][0]["bearing_limit_nm"] is None
    assert encoded["assumptions"] == list(spec.assumptions)
    assert pose_count(spec) == 4


def test_home_only_and_clearance_only_are_explicitly_supported():
    model, spec = _scene()
    spec = replace(spec, loads=(), sweeps=())
    validate_spec(model, spec)
    assert pose_count(spec) == 1
    with pytest.raises(ValueError, match="at least one load axis or clearance"):
        validate_spec(model, replace(spec, clearances=()))


def test_pose_budget_counts_cartesian_groups_and_explicit_home():
    model, spec = _scene()
    spec = replace(spec, sweeps=(PoseSweep("combined", {"head_pitch": (-1, 0, 1), "leg_pitch": (-1, 1)}),))
    assert pose_count(spec) == 7
    validate_spec(model, replace(spec, max_poses=7))
    with pytest.raises(ValueError, match="exceed max_poses"):
        validate_spec(model, replace(spec, max_poses=6))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"root_joint": "missing"}, "Unknown"),
        ({"root_joint": "head_pitch"}, "must be a free joint"),
        ({"home_keyframe": "missing"}, "Unknown"),
        ({"robot_id": " "}, "robot_id"),
        ({"mass_evidence": "guessed"}, "mass_evidence"),
        ({"max_poses": True}, "positive integer"),
        ({"assumptions": ("",)}, "nonempty text"),
        ({"feet": {}}, "at least one foot"),
        ({"feet": {"left": ()}}, "at least one support geom"),
        ({"feet": {"left": ("sole",), "right": ("sole",)}}, "exactly one foot"),
        ({"feet": {"left": ("floor",)}}, "outside the robot root"),
        ({"clearances": (ClearancePair("floor", "head_case", 0.001),)}, "outside the robot root"),
    ],
)
def test_metadata_resolves_names_evidence_and_support_geometry(changes, message):
    model, spec = _scene()
    with pytest.raises(ValueError, match=message):
        validate_spec(model, replace(spec, **changes))


@pytest.mark.parametrize(
    ("load", "message"),
    [
        (JointLoadSpec("leg_pitch", "leg_motor"), "supports a foot"),
        (JointLoadSpec("head_pitch", "leg_motor"), "directly mapped"),
        (JointLoadSpec("head_pitch", "missing"), "Unknown"),
        (JointLoadSpec("head_pitch", "head_motor", bearing_limit_nm=-1), "bearing_limit_nm"),
        (JointLoadSpec("head_pitch", "head_motor", max_torque_fraction=1.1), "must not exceed one"),
        (JointLoadSpec("head_pitch", "head_motor", max_torque_fraction=float("nan")), "max_torque_fraction"),
        (JointLoadSpec("head_pitch", "head_motor", torque_evidence="guessed"), "torque_evidence"),  # type: ignore[arg-type]
    ],
)
def test_load_axes_require_uncoupled_expression_joints_and_supported_capacities(load, message):
    model, spec = _scene()
    with pytest.raises(ValueError, match=message):
        validate_spec(model, replace(spec, loads=(load,)))


def test_duplicate_axis_or_multiple_actuators_cannot_silently_inflate_capacity():
    model, spec = _scene()
    with pytest.raises(ValueError, match="one-to-one"):
        validate_spec(model, replace(spec, loads=spec.loads * 2))
    model, spec = _scene(extra_actuator='<motor name="second" joint="head_pitch"/>')
    with pytest.raises(ValueError, match="exactly one directly mapped"):
        validate_spec(model, spec)


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("actuator_gear", [2, 0, 0, 0, 0, 0], "unit gear"),
        ("actuator_forcelimited", 0, "force capacity"),
        ("actuator_forcerange", [0, 1], "force capacity"),
        ("actuator_forcerange", [-1, float("inf")], "force capacity"),
    ],
)
def test_actuator_torque_mapping_rejects_unsupported_model_settings(attribute, value, message):
    model, spec = _scene()
    getattr(model, attribute)[model.actuator("head_motor").id] = value
    with pytest.raises(ValueError, match=message):
        validate_spec(model, spec)


@pytest.mark.parametrize(
    ("sweeps", "message"),
    [
        ((PoseSweep("head", {"head_pitch": (-1.01,)}),), "exceeds the limits"),
        ((PoseSweep("head", {"head_pitch": (float("nan"),)}),), "must be finite"),
        ((PoseSweep("head", {"head_pitch": ()}),), "at least once"),
        ((PoseSweep("head", {}),), "at least one joint"),
        ((PoseSweep("home", {"head_pitch": (0,)}),), "distinct from home"),
        ((PoseSweep("head", {"root": (0,)}),), "must be a hinge"),
        ((PoseSweep("head", {"head_pitch": (0,)}),) * 2, "unique"),
    ],
)
def test_sweeps_are_named_finite_and_within_joint_limits(sweeps, message):
    model, spec = _scene()
    with pytest.raises(ValueError, match=message):
        validate_spec(model, replace(spec, sweeps=sweeps))


@pytest.mark.parametrize(
    "pairs",
    [
        (ClearancePair("head_case", "head_case", 0.001),),
        (ClearancePair("torso_case", "head_case", 0.001), ClearancePair("head_case", "torso_case", 0.001)),
        (ClearancePair("torso_case", "head_case", 0),),
        (ClearancePair("torso_case", "head_case", float("nan")),),
    ],
)
def test_clearance_pairs_need_unique_geometry_and_positive_finite_thresholds(pairs):
    model, spec = _scene()
    with pytest.raises(ValueError, match="must not repeat|positive and finite"):
        validate_spec(model, replace(spec, clearances=pairs))


@pytest.mark.parametrize("kind", [mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_HFIELD, mujoco.mjtGeom.mjGEOM_SDF])
@pytest.mark.parametrize("name", ["torso_case", "head_case"])
def test_clearance_screen_rejects_unsupported_distance_geometry(kind, name):
    model, spec = _scene()
    # These shapes need different narrow-phase handling; an unsupported
    # distance query must never turn into an apparently clear pair.
    model.geom_type[model.geom(name).id] = kind
    with pytest.raises(ValueError, match="convex primitives and mesh convex hulls"):
        validate_spec(model, spec)


@pytest.mark.parametrize("gravity", [[0, 0, 0], [0, -9.81, 0], [0, 0, float("nan")]])
def test_load_screen_rejects_zero_rotated_and_nonfinite_gravity(gravity):
    model, spec = _scene()
    model.opt.gravity[:] = gravity
    with pytest.raises(ValueError, match="Earth gravity"):
        validate_spec(model, spec)


def test_load_screen_rejects_disabled_gravity_compensation_and_active_constraints():
    model, spec = _scene()
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_GRAVITY
    with pytest.raises(ValueError, match="Gravity physics"):
        validate_spec(model, spec)
    model.opt.disableflags = 0
    model.body_gravcomp[model.body("head").id] = 0.5
    with pytest.raises(ValueError, match="Gravity compensation"):
        validate_spec(model, spec)
    model.body_gravcomp[:] = 0
    model.eq_active0[:] = True
    with pytest.raises(ValueError, match="Active equality constraints"):
        validate_spec(model, spec)


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("body_mass", float("nan"), "finite and nonnegative"),
        ("body_mass", 0, "positive mass"),
        ("body_inertia", [0, 0.1, 0.1], "positive mass and principal inertia"),
    ],
)
def test_moving_body_mass_and_inertia_must_be_physically_valid(attribute, value, message):
    model, spec = _scene()
    getattr(model, attribute)[model.body("head").id] = value
    with pytest.raises(ValueError, match=message):
        validate_spec(model, spec)


def test_home_pose_and_additional_joint_force_limits_are_checked():
    model, spec = _scene()
    joint = model.joint("head_pitch").id
    model.key_qpos[0, model.jnt_qposadr[joint]] = 1.01
    with pytest.raises(ValueError, match="Home pose exceeds"):
        validate_spec(model, spec)
    model.key_qpos[0, model.jnt_qposadr[joint]] = 0
    model.jnt_actfrclimited[joint] = True
    with pytest.raises(ValueError, match="additional joint force limit"):
        validate_spec(model, spec)
