"""Silhouette-C regeneration, hardware invariants and head motion checks."""

import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from examples.mini_tyrannosaurus_rex import character_checks, demo, engineering, head_design, tail_option


def test_character_generator_reproduces_model_and_explicit_head_inertias(tmp_path):
    generated = head_design.build(tmp_path)

    def canonical(path):
        root = ET.parse(path).getroot()
        for include in root.iter("include"):
            include.set("file", str((path.parent / include.get("file")).resolve()))
        return ET.tostring(root)

    assert canonical(generated) == canonical(demo.MODEL_PATH)
    model = mujoco.MjModel.from_xml_path(str(generated))
    allocation = json.loads((tmp_path / "head_allocation.json").read_text())
    expected = json.loads((head_design.HERE / "head_allocation.json").read_text())
    for key in expected.keys() - {"head_pitch_body", "jaw_body"}:
        assert allocation[key] == expected[key]
    for name in ("head_pitch_body", "jaw_body"):
        for actual, reference in zip(allocation[name], expected[name], strict=True):
            assert actual["name"] == reference["name"]
            assert actual["basis"] == reference["basis"]
            assert actual["mass_kg"] == reference["mass_kg"]
            np.testing.assert_allclose(actual["pos_m"], reference["pos_m"], rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(actual["inertia_kg_m2"], reference["inertia_kg_m2"], rtol=1e-12, atol=1e-16)
        body = model.body(name)
        rows = allocation[name]
        mass = sum(row["mass_kg"] for row in rows)
        center = sum(row["mass_kg"] * np.array(row["pos_m"]) for row in rows) / mass
        assert body.mass[0] == pytest.approx(mass)
        assert body.ipos == pytest.approx(center)
    assert model.body_subtreemass[model.body("head_yaw_body").id] == pytest.approx(0.091)


def test_character_preserves_baseline_pivots_motor_cases_and_actuator_limits():
    old = mujoco.MjModel.from_xml_path(str(head_design.BASELINE))
    new, _ = demo.load_model()
    for attribute in (
        "body_pos",
        "body_quat",
        "body_mass",
        "jnt_pos",
        "jnt_axis",
        "jnt_range",
        "actuator_ctrlrange",
        "actuator_forcerange",
        "actuator_gainprm",
        "actuator_biasprm",
    ):
        assert getattr(new, attribute) == pytest.approx(getattr(old, attribute))
    assert new.nu == old.nu == 14
    for index in engineering.motor_cases(old):
        name = old.geom(index).name
        previous, current = old.geom(name), new.geom(name)
        for attribute in ("pos", "quat", "size"):
            assert getattr(current, attribute) == pytest.approx(getattr(previous, attribute))


@pytest.mark.parametrize("tail", [False, True], ids=["fixed-tail", "wag-tail"])
def test_jaw_sweep_has_positive_shell_separation_even_between_parent_and_child(tail):
    model = tail_option.load_model() if tail else demo.load_model()[0]
    report = character_checks.jaw_separation(model)
    assert report["poses"] == 101
    assert min(report["minimum_separating_axis_gaps_m"].values()) > 0.002
    # The contact engine must also include these normally filtered adjacent bodies.
    for name in ("jaw_collision", "left_jaw_rail_collision", "right_jaw_rail_collision"):
        pair = {model.geom("head_collision").id, model.geom(name).id}
        assert any({first, second} == pair for first, second in zip(model.pair_geom1, model.pair_geom2, strict=True))


def test_shell_clearance_guard_rejects_a_jaw_moved_into_the_skull():
    model, _ = demo.load_model()
    model.geom_pos[model.geom("jaw_collision").id, 2] += 0.012
    report = character_checks.jaw_separation(model)
    assert report["minimum_separating_axis_gaps_m"]["jaw_collision"] < 0


@pytest.mark.parametrize("tail", [False, True], ids=["fixed-tail", "wag-tail"])
def test_free_character_gesture_moves_head_jaw_and_arms_without_internal_contacts(tail):
    model = tail_option.load_model() if tail else demo.load_model()[0]
    report = character_checks.gesture_trial(model, duration=5)
    assert not report["externally_supported"]
    assert report["finite"] and not report["warnings"]
    assert report["duration_s"] == pytest.approx(5)
    assert report["minimum_height_ratio"] > 0.95
    assert report["maximum_tilt_deg"] < 5
    assert report["maximum_internal_penetration_m"] < 0.0001
    for name, minimum in {
        "head_yaw": 0.4,
        "head_pitch": 0.1,
        "jaw": 0.15,
        "left_shoulder_pitch": 0.4,
        "right_shoulder_pitch": 0.4,
    }.items():
        assert report["joint_excursions_rad"][name] > minimum
    for row in report["actuators"].values():
        assert row["peak_Nm"] <= max(abs(x) for x in row["cap_Nm"]) + 1e-10
