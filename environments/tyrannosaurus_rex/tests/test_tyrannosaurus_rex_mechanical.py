"""Exercise the shared mechanical screen against the unchanged medium concept."""

from __future__ import annotations

import json
from dataclasses import replace
from itertools import combinations

import mujoco
import numpy as np
import pytest

from environments.shared.robot_design_spec import pose_count, validate_spec
from environments.shared.robot_design_validation import screen_design
from examples.tyrannosaurus_rex_50cm import demo, mechanical_checks


def clearance_row(report, first, second):
    return next(row for row in report["clearance_checks"] if {row["first"], row["second"]} == {first, second})


def test_medium_spec_covers_adjacent_cases_expression_pairs_and_declared_ranges():
    model, _ = demo.load_model()
    spec = mechanical_checks.design_spec(model)
    validate_spec(model, spec)
    assert pose_count(spec) == 790
    assert len(spec.clearances) == 245
    pairs = {frozenset((pair.first, pair.second)) for pair in spec.clearances}
    assert all(frozenset(pair) in pairs for pair in combinations(mechanical_checks.MOTOR_CASES, 2))
    assert frozenset(("head_collision", "jaw_collision")) in pairs
    assert frozenset(("jaw_collision", "left_hip_yaw_case_collision")) in pairs
    assert all(pair.min_distance_m == 0.002 for pair in spec.clearances)
    assert {load.joint for load in spec.loads} == set(mechanical_checks.EXPRESSION_JOINTS)
    assert spec.mass_evidence == "estimated"
    assert all(load.torque_evidence == "estimated" and load.bearing_limit_nm is None for load in spec.loads)
    for sweep in spec.sweeps:
        for name, values in sweep.samples.items():
            assert (values[0], values[-1]) == pytest.approx(model.joint(name).range)
    for first, second in mechanical_checks.ATTACHMENT_EXCLUSIONS:
        assert frozenset((first, second)) not in pairs
        assert any(first in note and second in note for note in spec.assumptions)


def test_baseline_reports_proxy_conflicts_without_changing_geometry_or_capacity():
    model, _ = demo.load_model()
    fields = ("body_mass", "body_inertia", "body_pos", "geom_pos", "geom_size", "actuator_forcerange", "key_qpos")
    original = {field: getattr(model, field).copy() for field in fields}
    report = screen_design(model, mechanical_checks.design_spec(model))
    assert report["status"] == "failed"
    assert not report["known_limits_passed"]
    # The unchanged concept has these documented proxy conflicts. Retaining
    # them is not design approval: this list prevents new conflicts from hiding
    # behind an already-failed aggregate result, while allowing fixes to reduce it.
    known_conflicts = {
        frozenset(pair)
        for pair in (
            ("neck_support_collision", "head_collision"),
            ("head_collision", "jaw_collision"),
            ("head_collision", "left_jaw_rail_collision"),
            ("head_collision", "right_jaw_rail_collision"),
            ("jaw_collision", "left_arm_collision"),
            ("jaw_collision", "right_arm_collision"),
            ("left_jaw_rail_collision", "left_arm_collision"),
            ("right_jaw_rail_collision", "right_arm_collision"),
            ("torso_collision", "left_hip_yaw_case_collision"),
            ("torso_collision", "left_hip_roll_case_collision"),
            ("torso_collision", "left_hip_pitch_case_collision"),
            ("torso_collision", "right_hip_yaw_case_collision"),
            ("torso_collision", "right_hip_roll_case_collision"),
            ("torso_collision", "right_hip_pitch_case_collision"),
            ("jaw_collision", "left_hip_yaw_case_collision"),
            ("jaw_collision", "right_hip_yaw_case_collision"),
        )
    }
    failed_pairs = {
        frozenset((row["first"], row["second"])) for row in report["clearance_checks"] if row["status"] == "failed"
    }
    assert failed_pairs <= known_conflicts, failed_pairs - known_conflicts
    assert clearance_row(report, "jaw_collision", "left_arm_collision")["min_distance_m"] < -0.03
    assert clearance_row(report, "torso_collision", "left_hip_roll_case_collision")["min_distance_m"] < -0.02
    for first, second in combinations(mechanical_checks.MOTOR_CASES, 2):
        row = clearance_row(report, first, second)
        assert row["min_distance_m"] >= 0.002
    assert len(report["load_checks"]) == 6
    assert all(row["torque_status"] == "passed" for row in report["load_checks"])
    assert all(row["bearing_status"] != "passed" for row in report["load_checks"])
    assert all(row["bearing_limit_nm"] is None for row in report["load_checks"])
    for field, before in original.items():
        np.testing.assert_array_equal(getattr(model, field), before)


def test_overweight_head_changes_a_passing_gravity_margin_to_a_failure():
    model, _ = demo.load_model()
    full = mechanical_checks.design_spec(model)
    spec = replace(full, loads=tuple(load for load in full.loads if load.joint == "head_nod"), sweeps=(), clearances=())
    baseline = screen_design(model, spec)["load_checks"][0]
    assert baseline["torque_status"] == "passed"
    head = model.body("head").id
    model.body_mass[head] *= 10
    model.body_inertia[head] *= 10
    mujoco.mj_setConst(model, mujoco.MjData(model))
    report = screen_design(model, spec)
    overloaded = report["load_checks"][0]
    assert overloaded["max_gravity_torque_nm"] > baseline["max_gravity_torque_nm"] * 5
    assert overloaded["torque_status"] == "failed"
    assert report["status"] == "failed"


def test_adjacent_case_interference_is_detected_despite_contact_filtering():
    model, data = demo.load_model()
    full = mechanical_checks.design_spec(model)
    first, second = "left_ankle_pitch_case_collision", "left_ankle_roll_case_collision"
    pair = next(pair for pair in full.clearances if {pair.first, pair.second} == {first, second})
    spec = replace(full, loads=(), sweeps=(), clearances=(pair,))
    baseline = screen_design(model, spec)
    assert clearance_row(baseline, first, second)["status"] == "passed"
    first_id, second_id = model.geom(first).id, model.geom(second).id
    assert model.body_parentid[model.geom_bodyid[second_id]] == model.geom_bodyid[first_id]
    model.geom_size[first_id, 0] *= 2
    mujoco.mj_forward(model, data)
    assert not any({contact.geom1, contact.geom2} == {first_id, second_id} for contact in data.contact[: data.ncon])
    report = screen_design(model, spec)
    row = clearance_row(report, first, second)
    assert row["min_distance_m"] < 0
    assert row["status"] == "failed"
    assert report["status"] == "failed"


def test_medium_cli_writes_a_failed_screen_with_reproducible_inputs(tmp_path):
    output = tmp_path / "mechanical_validation.json"
    assert mechanical_checks.main(["--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert report["schema_version"]
    assert report["robot_id"] == "tyrannosaurus_rex_50cm_concept"
    assert report["status"] == "failed"
    assert report["provenance"]["command"][-2:] == ["--output", str(output)]
    assert all(path.is_file() for path in mechanical_checks.source_paths())
