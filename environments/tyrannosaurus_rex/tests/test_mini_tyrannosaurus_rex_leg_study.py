"""Keep the leg-direction comparison fair and its negative results visible."""

import hashlib
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from examples.mini_tyrannosaurus_rex.leg_study import validate
from examples.mini_tyrannosaurus_rex.leg_study.build import BASE, HERE, PITCH, VARIANTS, build


@pytest.mark.parametrize("name", VARIANTS)
def test_same_torso_full_hardware_and_forward_facing_feet(name, tmp_path):
    build(name, tmp_path)
    original = mujoco.MjModel.from_xml_path(str(BASE))
    model = mujoco.MjModel.from_xml_path(str(tmp_path / "model.xml"))
    assert model.nu == 10 and model.neq == model.nmocap == 0
    assert model.body_mass.sum() == pytest.approx(0.7541)
    np.testing.assert_array_equal(model.body_mass, original.body_mass)
    # Reflected capsule endpoints are serialized to ten significant digits;
    # their compiled half-lengths can differ by less than a picometre.
    np.testing.assert_allclose(model.geom_size, original.geom_size, atol=2e-12, rtol=0)
    cases = [g for g in range(model.ngeom) if model.geom(g).name.endswith("_servo_case")]
    assert len(cases) == 10
    np.testing.assert_array_equal(model.geom_size[cases], original.geom_size[cases])
    np.testing.assert_array_equal(model.actuator_gainprm, original.actuator_gainprm)
    np.testing.assert_array_equal(model.actuator_biasprm, original.actuator_biasprm)
    np.testing.assert_array_equal(model.actuator_forcerange, original.actuator_forcerange)
    # The torso, fixed payloads and upper hip assemblies are physically exact
    # copies; neither study is helped by relocating the battery or shrinking it.
    base_xml, candidate = ET.parse(BASE).getroot(), ET.parse(tmp_path / "model.xml").getroot()
    for tree in (base_xml, candidate):
        for side in ("left", "right"):
            roll = tree.find(f".//body[@name='{side}_hip_roll_body']")
            thigh = tree.find(f".//body[@name='{side}_thigh']")
            assert roll is not None and thigh is not None
            roll.remove(thigh)
        ET.indent(tree, space="  ")
    assert ET.tostring(base_xml.find("worldbody")) == ET.tostring(candidate.find("worldbody"))
    for side in ("left", "right"):
        foot = model.body(f"{side}_foot").id
        geoms = np.where(model.geom_bodyid == foot)[0]
        np.testing.assert_array_equal(model.geom_pos[geoms], original.geom_pos[geoms])
        np.testing.assert_array_equal(model.body_ipos[foot], original.body_ipos[foot])
        for g in geoms:
            if model.geom(g).name.endswith("flat_pad_collision"):
                assert model.geom_pos[g, 0] > 0


def test_reflection_preserves_link_length_and_maps_pitch_commands():
    forward, fd = validate.load("forward")
    rear, rd = validate.load("rearward")
    for alpha in (0.0, 0.17, 0.35, validate.alpha_for_drop(30)):
        validate.set_fold(forward, fd, 1, alpha)
        validate.set_fold(rear, rd, -1, alpha)
        for side in ("left", "right"):
            for part in ("shin", "foot"):
                a, b = forward.body(f"{side}_{part}").id, rear.body(f"{side}_{part}").id
                np.testing.assert_allclose(fd.xpos[a] * [-1, 1, 1], rd.xpos[b], atol=1e-10)
                assert np.linalg.norm(forward.body_pos[a]) == pytest.approx(np.linalg.norm(rear.body_pos[b]))
    for a in range(rear.nu):
        j = rear.actuator_trnid[a, 0]
        suffix = rear.joint(j).name.split("_", 1)[1]
        expected = -forward.jnt_range[j, ::-1] if suffix in PITCH else forward.jnt_range[j]
        np.testing.assert_array_equal(rear.jnt_range[j], expected)
        np.testing.assert_array_equal(rear.actuator_ctrlrange[a], expected)


def test_uncertain_zero_distance_is_not_a_clearance_pass():
    assert validate.distance_status(dict(min_distance_m=0.0, distance_is_lower_bound=False)) == "incomplete"
    assert validate.distance_status(dict(min_distance_m=0.001, distance_is_lower_bound=True)) == "incomplete"
    assert validate.distance_status(dict(min_distance_m=-0.001, distance_is_lower_bound=False)) == "fail"


def test_overlapping_board_reserves_fail_even_when_contacts_are_disabled():
    model, data = validate.load("rearward")
    a, b = model.geom("part_computer_reserve").id, model.geom("part_battery_reserve").id
    model.geom_pos[a] = model.geom_pos[b]
    mujoco.mj_forward(model, data)
    assert model.geom_contype[a] == model.geom_contype[b] == 0
    assert validate.evidence_for_pair(model, data, a, b)["status"] != "pass"


@pytest.mark.parametrize("name", VARIANTS)
def test_expanded_fold_keeps_real_motor_interference_visible(name):
    model, data = validate.load(name)
    clear = validate.fold_snapshot(model, data, VARIANTS[name], 0.73)
    blocked = validate.fold_snapshot(model, data, VARIANTS[name], 0.735)
    assert clear["status"] == "pass"
    assert not clear["within_delivered_joint_limits"]
    assert blocked["status"] == "fail"
    assert any("left_hip_pitch_servo_case" in row["pair"] for row in blocked["blockers"])
    assert any("left_ankle_servo_case" in row["pair"] for row in blocked["blockers"])


def test_assisted_motion_cannot_pass_as_unsupported(monkeypatch):
    real_load = validate.load

    def assisted(name):
        model, data = real_load(name)
        model.body_gravcomp[model.body("torso").id] = 1
        return model, data

    monkeypatch.setattr(validate, "load", assisted)
    result = validate.motion("forward", 12.0112755570818)
    assert not result["checks"]["no_support_fixtures"]
    assert result["status"] == "fail"


def test_evidence_matches_sources_and_retains_loaded_rear_failure():
    report = json.loads((HERE / "comparison.json").read_text())
    for path, expected in report["sources"].items():
        assert hashlib.sha256((validate.REPO / path).read_bytes()).hexdigest() == expected, path
    assert hashlib.sha256((HERE / "samples.json").read_bytes()).hexdigest() == report["samples_sha256"]
    assert report["status"] == "fail" and not report["hardware_ready"]
    fwd, rear = report["variants"]
    for candidate in (fwd, rear):
        assert candidate["packing"]["status"] == "pass"
        assert candidate["packing"]["pose_count"] == 1050
        assert candidate["packing"]["evaluations"] == 162805
        assert all(t["status"] == "pass" for t in candidate["standing"])
        assert candidate["crouching"][0]["status"] == candidate["crouching"][2]["status"] == "pass"
        assert not candidate["crouching"][2]["expanded_limits_for_experiment"]
        assert candidate["crouching"][3]["expanded_limits_for_experiment"]
    assert fwd["crouching"][3]["status"] == rear["crouching"][1]["status"] == "pass"
    assert rear["crouching"][3]["status"] == "fail"
    assert not rear["crouching"][3]["checks"]["completed_14_seconds"]
    assert rear["crouching"][3]["max_tilt_deg"] >= 60
