"""Analytical loads and deliberate failures for the sampled mechanical screen."""

import copy
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np
import pytest
from PIL import Image

from environments.shared.robot_design_spec import ClearancePair, JointLoadSpec, PoseSweep, RobotDesignSpec
from environments.shared.robot_design_validation import render_witnesses, screen_design, write_report


def _scene(*, mass=1.0, limits=(-5, 5), bearing_limit=2.0, samples=(0.0,), interference=False):
    # A 1 kg payload at (.2, .1, 0) relative to a Y hinge needs 1.962 Nm
    # holding torque and carries .981 Nm of transverse bending at home.
    case_x = -0.25 if interference else 0.25
    model = mujoco.MjModel.from_xml_string(f"""
    <mujoco>
      <compiler angle="radian" inertiafromgeom="false"/>
      <option gravity="0 0 -9.81"/>
      <worldbody>
        <body name="torso" pos="0 0 1">
          <freejoint name="root"/>
          <inertial pos="0 0 0" mass="2" diaginertia=".1 .1 .1"/>
          <geom name="left_sole" type="box" pos="0 .2 -.9" size=".1 .05 .02"/>
          <geom name="right_sole" type="box" pos="0 -.2 -.9" size=".1 .05 .02"/>
          <geom name="torso_case" type="sphere" pos="-.25 0 0" size=".05"/>
          <body name="arm">
            <joint name="arm_pitch" type="hinge" axis="0 1 0" limited="true" range="{-math.pi} {math.pi}"/>
            <inertial pos=".2 .1 0" mass="{mass}" diaginertia=".01 .01 .01"/>
            <geom name="arm_case" type="sphere" pos="{case_x} 0 0" size=".05"/>
          </body>
        </body>
      </worldbody>
      <actuator>
        <motor name="arm_servo" joint="arm_pitch" gear="1" forcelimited="true" forcerange="{limits[0]} {limits[1]}"/>
      </actuator>
      <keyframe><key name="home" qpos="0 0 1 1 0 0 0 0"/></keyframe>
    </mujoco>
    """)
    spec = RobotDesignSpec(
        robot_id="analytical_arm",
        root_joint="root",
        home_keyframe="home",
        feet={"left": ("left_sole",), "right": ("right_sole",)},
        loads=(JointLoadSpec("arm_pitch", "arm_servo", bearing_limit_nm=bearing_limit),),
        sweeps=(PoseSweep("arm_motion", {"arm_pitch": samples}),),
        clearances=(ClearancePair("torso_case", "arm_case", 0.002),),
    )
    return model, spec


def test_static_loads_match_analytical_gravity_and_transverse_moments():
    model, spec = _scene()
    result = screen_design(model, spec)
    load = result["load_checks"][0]
    assert load["max_gravity_torque_nm"] == pytest.approx(1.962)
    assert load["maximum_torque_fraction"] == pytest.approx(1.962 / 5)
    assert load["max_bearing_moment_nm"] == pytest.approx(0.981)
    assert load["torque_status"] == "passed"
    assert load["bearing_status"] == "passed"
    assert result["clearance_checks"][0]["min_distance_m"] == pytest.approx(0.4)
    assert result["status"] == "passed"
    assert result["known_limits_passed"]
    assert result["hardware_assessment"]["status"] == "incomplete"
    assert not result["coverage"]["continuous_path_checked"]
    assert not result["coverage"]["ground_support_solved"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("mass,expected_torque", [(2.2, 4.3164), (3, 5.886)])
def test_heavy_payload_exceeds_torque_budget_without_changing_actuator_capacity(mass, expected_torque):
    model, spec = _scene(mass=mass)
    original_limits = model.actuator_forcerange.copy()
    result = screen_design(model, spec)
    load = result["load_checks"][0]
    assert load["max_gravity_torque_nm"] == pytest.approx(expected_torque)
    assert load["torque_status"] == "failed"
    assert result["status"] == "failed"
    assert not result["known_limits_passed"]
    assert np.array_equal(model.actuator_forcerange, original_limits)


def test_adjacent_body_interference_is_found_without_generated_contacts():
    model, spec = _scene(interference=True)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    # MuJoCo suppresses collisions between directly connected bodies. The
    # independent distance screen must still reject their overlapping cases.
    assert data.ncon == 0
    result = screen_design(model, spec)
    clearance = result["clearance_checks"][0]
    assert clearance["min_distance_m"] < 0.002
    assert clearance["status"] == "failed"
    assert clearance["worst_pose"] == {"arm_pitch": 0.0}
    assert result["status"] == "failed"


def test_asymmetric_motor_capacity_uses_the_required_holding_torque_direction():
    model, spec = _scene(limits=(-5, 1), samples=(0.0, math.pi))
    result = screen_design(model, spec)
    load = result["load_checks"][0]
    # Gravity requires negative holding torque at zero, but positive at pi.
    # The latter reaches 196.2% of the smaller positive capacity.
    assert load["max_gravity_torque_nm"] == pytest.approx(1.962)
    assert load["maximum_torque_fraction"] == pytest.approx(1.962)
    assert load["torque_worst_pose"]["arm_pitch"] == pytest.approx(math.pi)
    assert load["torque_status"] == "failed"


def test_missing_bearing_capacity_is_incomplete_even_when_known_limits_pass():
    model, spec = _scene(bearing_limit=None)
    result = screen_design(model, spec)
    assert result["known_limits_passed"]
    assert result["load_checks"][0]["torque_status"] == "passed"
    assert result["load_checks"][0]["bearing_status"] == "incomplete"
    assert result["status"] == "incomplete"


def test_transverse_bearing_overload_fails_even_when_holding_torque_passes():
    model, spec = _scene(bearing_limit=0.5)
    result = screen_design(model, spec)
    assert result["load_checks"][0]["torque_status"] == "passed"
    assert result["load_checks"][0]["bearing_status"] == "failed"
    assert not result["known_limits_passed"]
    assert result["status"] == "failed"


def test_screen_rejects_unsupported_actuator_transmission_before_assessment():
    model, spec = _scene()
    model.actuator_gear[0, 0] = 2
    with pytest.raises(ValueError, match="unit gear"):
        screen_design(model, spec)


@pytest.mark.parametrize("combine", [False, True])
def test_joint_samples_respect_cartesian_groups_and_record_complete_witness(combine):
    model = mujoco.MjModel.from_xml_string(f"""
    <mujoco>
      <compiler angle="radian" inertiafromgeom="false"/>
      <option gravity="0 0 -9.81"/>
      <worldbody>
        <body name="torso" pos="0 0 1">
          <freejoint name="root"/>
          <inertial pos="0 0 0" mass="2" diaginertia=".1 .1 .1"/>
          <geom name="sole" type="box" pos="0 0 -.9" size=".1 .1 .02"/>
          <body name="arm">
            <joint name="arm_pitch" type="hinge" axis="0 1 0" range="0 {math.pi / 2}"/>
            <inertial pos="0 0 .1" mass=".1" diaginertia=".01 .01 .01"/>
            <body name="tip">
              <joint name="tip_pitch" type="hinge" axis="0 1 0" range="0 {math.pi / 2}"/>
              <inertial pos="0 0 .3" mass="1" diaginertia=".01 .01 .01"/>
            </body>
          </body>
        </body>
      </worldbody>
      <actuator>
        <motor name="arm_servo" joint="arm_pitch" gear="1" forcelimited="true" forcerange="-10 10"/>
        <motor name="tip_servo" joint="tip_pitch" gear="1" forcelimited="true" forcerange="-10 10"/>
      </actuator>
      <keyframe><key name="home" qpos="0 0 1 1 0 0 0 0 0"/></keyframe>
    </mujoco>
    """)
    sweeps = (
        (PoseSweep("combined", {"arm_pitch": (0.0, math.pi / 4), "tip_pitch": (0.0, math.pi / 4)}),)
        if combine
        else (
            PoseSweep("arm", {"arm_pitch": (math.pi / 4,)}),
            PoseSweep("tip", {"tip_pitch": (math.pi / 6,)}),
        )
    )
    spec = RobotDesignSpec(
        robot_id="two_joint_arm",
        root_joint="root",
        home_keyframe="home",
        feet={"foot": ("sole",)},
        loads=(JointLoadSpec("tip_pitch", "tip_servo", bearing_limit_nm=10),),
        sweeps=sweeps,
        clearances=(),
    )
    result = screen_design(model, spec)
    load = result["load_checks"][0]
    assert load["max_gravity_torque_nm"] == pytest.approx(2.943 if combine else 2.943 / math.sqrt(2))
    assert load["torque_worst_pose"] == pytest.approx(
        {"arm_pitch": math.pi / 4, "tip_pitch": math.pi / 4 if combine else 0.0}
    )
    assert result["coverage"]["sampled_poses"] == (5 if combine else 3)


def test_screen_is_repeatable_and_does_not_mutate_the_input_model():
    model, spec = _scene(samples=(-0.4, 0.0, 0.4))
    fields = ("body_mass", "body_inertia", "geom_pos", "jnt_range", "actuator_forcerange", "eq_active0", "key_qpos")
    original = {name: getattr(model, name).copy() for name in fields}
    first = screen_design(model, spec)
    second = screen_design(model, spec)
    assert first == second
    for name, expected in original.items():
        assert np.array_equal(getattr(model, name), expected), name


def test_written_report_binds_results_to_sources_and_detects_source_changes(tmp_path):
    model, spec = _scene()
    report = screen_design(model, spec)
    original = copy.deepcopy(report)
    root = Path(__file__).resolve().parents[3]
    command = ["python", "mechanical_checks.py", "--output", "report.json"]
    # The real validator sources and changing model must share a repository
    # root, while all generated report artifacts remain in pytest's directory.
    with TemporaryDirectory(prefix=".mechanical-test-", dir=root) as temporary:
        model_path = Path(temporary) / "model.xml"
        mujoco.mj_saveLastXML(str(model_path), model)
        output = tmp_path / "reports" / "report.json"
        first = write_report(report, output=output, source_paths=[model_path], source_root=root, command=command)
        assert json.loads(output.read_text()) == json.loads(json.dumps(first, allow_nan=False))
        relative_model = model_path.relative_to(root).as_posix()
        hashes = first["source_sha256"]
        assert hashes[relative_model] == hashlib.sha256(model_path.read_bytes()).hexdigest()
        for name in ("robot_design_validation.py", "robot_design_spec.py"):
            relative = f"environments/shared/{name}"
            assert hashes[relative] == hashlib.sha256((root / relative).read_bytes()).hexdigest()
        assert first["provenance"]["command"] == command
        assert first["provenance"]["mujoco_version"] == mujoco.__version__
        model_path.write_text(model_path.read_text() + "\n<!-- changed input -->\n")
        second = write_report(report, output=output, source_paths=[model_path], source_root=root, command=command)
        assert second["source_sha256"][relative_model] != hashes[relative_model]
    assert report == original


@pytest.mark.parametrize("nonfinite", [math.nan, math.inf])
def test_nonfinite_report_values_are_rejected_before_replacing_existing_evidence(tmp_path, nonfinite):
    root = Path(__file__).resolve().parents[3]
    output = tmp_path / "report.json"
    existing = '{"status": "previous result"}\n'
    output.write_text(existing)
    with pytest.raises(ValueError, match="JSON compliant"):
        write_report(
            {"load_checks": [{"max_gravity_torque_nm": nonfinite}]},
            output=output,
            source_paths=[Path(__file__)],
            source_root=root,
            command=["python", "mechanical_checks.py"],
        )
    assert output.read_text() == existing


def test_witness_images_show_highlighted_geometry_without_mutating_inputs(tmp_path):
    model, spec = _scene(samples=(-0.4, 0.0, 0.4))
    report = screen_design(model, spec)
    original_report = copy.deepcopy(report)
    fields = ("geom_group", "geom_rgba", "geom_matid", "geom_pos", "geom_quat", "key_qpos", "key_ctrl")
    original_model = {name: getattr(model, name).copy() for name in fields}
    output = render_witnesses(model, spec, report, tmp_path / "witnesses")
    assert set(output) == {"clearance", "torque", "bearing"}
    for path in output.values():
        with Image.open(path) as rendered:
            assert rendered.format == "PNG"
            pixels = np.asarray(rendered, dtype=int)
            red = (pixels[:, :, 0] > 1.5 * pixels[:, :, 1]) & (pixels[:, :, 0] > 1.5 * pixels[:, :, 2])
            # Highlighted robot geometry must be visible below the captions.
            assert np.count_nonzero(red[rendered.height // 3 :]) > 10
    assert report == original_report
    for name, expected in original_model.items():
        assert np.array_equal(getattr(model, name), expected), name


def test_load_witnesses_highlight_failed_limits_over_larger_raw_loads(tmp_path, monkeypatch):
    model = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <compiler angle="radian" inertiafromgeom="false"/>
      <option gravity="0 0 -9.81"/>
      <worldbody>
        <body name="torso" pos="0 0 1">
          <freejoint name="root"/>
          <inertial pos="0 0 0" mass="2" diaginertia=".1 .1 .1"/>
          <geom name="sole" type="box" pos="0 0 -.9" size=".1 .1 .02"/>
          <body name="large_arm" pos="0 .2 0">
            <joint name="large_pitch" type="hinge" axis="0 1 0" range="-1 1"/>
            <inertial pos=".3 .3 0" mass="1" diaginertia=".01 .01 .01"/>
            <geom name="large_case" type="sphere" size=".05"/>
          </body>
          <body name="weak_arm" pos="0 -.2 0">
            <joint name="weak_pitch" type="hinge" axis="0 1 0" range="-1 1"/>
            <inertial pos=".1 .1 0" mass="1" diaginertia=".01 .01 .01"/>
            <geom name="weak_case" type="sphere" size=".05"/>
          </body>
        </body>
      </worldbody>
      <actuator>
        <motor name="large_servo" joint="large_pitch" gear="1" forcelimited="true" forcerange="-5 5"/>
        <motor name="weak_servo" joint="weak_pitch" gear="1" forcelimited="true" forcerange="-2 2"/>
      </actuator>
      <keyframe><key name="home" qpos="0 0 1 1 0 0 0 0 0"/></keyframe>
    </mujoco>
    """)
    spec = RobotDesignSpec(
        robot_id="unequal_arms",
        root_joint="root",
        home_keyframe="home",
        feet={"foot": ("sole",)},
        loads=(
            JointLoadSpec("large_pitch", "large_servo", bearing_limit_nm=5),
            JointLoadSpec("weak_pitch", "weak_servo", bearing_limit_nm=0.5, max_torque_fraction=0.4),
        ),
        sweeps=(),
        clearances=(),
    )
    report = screen_design(model, spec)
    large, weak = report["load_checks"]
    assert large["maximum_torque_fraction"] > weak["maximum_torque_fraction"]
    assert large["max_bearing_moment_nm"] > weak["max_bearing_moment_nm"]
    assert large["torque_status"] == large["bearing_status"] == "passed"
    assert weak["torque_status"] == weak["bearing_status"] == "failed"
    highlighted = []

    def capture_highlights(diagnostic, data, **options):
        highlighted.append(
            {
                diagnostic.geom(geom).name
                for geom in range(diagnostic.ngeom)
                if diagnostic.geom_group[geom] == 3
                and diagnostic.geom_rgba[geom, 0] > 1.5 * diagnostic.geom_rgba[geom, 1]
            }
        )
        return Image.new("RGB", (options["width"], options["height"]))

    monkeypatch.setattr("environments.shared.robot_rendering.render", capture_highlights)
    output = render_witnesses(model, spec, report, tmp_path)
    assert set(output) == {"torque", "bearing"}
    assert highlighted and all(geoms == {"weak_case"} for geoms in highlighted)


@pytest.mark.parametrize("load_geom_group", [0, 3])
def test_load_witness_keeps_loaded_geometry_visible_outside_clearance_selection(tmp_path, monkeypatch, load_geom_group):
    model, spec = _scene()
    model.geom_group[model.geom("arm_case").id] = load_geom_group
    spec = replace(spec, clearances=(ClearancePair("torso_case", "left_sole", 0.002),))
    report = screen_design(model, spec)
    highlighted = []

    def capture_highlights(diagnostic, data, **options):
        highlighted.append(
            frozenset(
                diagnostic.geom(geom).name
                for geom in range(diagnostic.ngeom)
                if diagnostic.geom_group[geom] in options["groups"]
                and diagnostic.geom_rgba[geom, 0] > 1.5 * diagnostic.geom_rgba[geom, 1]
            )
        )
        return Image.new("RGB", (options["width"], options["height"]))

    monkeypatch.setattr("environments.shared.robot_rendering.render", capture_highlights)
    output = render_witnesses(model, spec, report, tmp_path)
    assert set(output) == {"clearance", "torque", "bearing"}
    # Neither load diagnostic may lose the arm merely because another pair
    # defines clearance coverage, including models without group-3 proxies.
    assert set(highlighted) == {frozenset({"torso_case", "left_sole"}), frozenset({"arm_case"})}
