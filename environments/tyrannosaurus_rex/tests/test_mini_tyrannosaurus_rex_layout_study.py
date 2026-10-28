"""Guard the comparison's hardware fairness and failure detection."""

import hashlib
import json

import mujoco
import numpy as np
import pytest

from examples.mini_tyrannosaurus_rex.layout_study import validate
from examples.mini_tyrannosaurus_rex.layout_study.build import HERE, LAYOUTS, PARTS, build, corners


@pytest.mark.parametrize("name", LAYOUTS)
def test_generated_candidates_keep_real_parts_and_identical_loads(name, tmp_path):
    layout = build(name, tmp_path)
    model = mujoco.MjModel.from_xml_path(str(tmp_path / "model.xml"))
    data = mujoco.MjData(model)
    metrics = validate.home_metrics(model, data)
    assert model.nu == 10 and model.neq == 0
    assert metrics["all_cases_full_size"] and metrics["motor_count"] == 10
    assert metrics["loaded_mass_g"] == pytest.approx(754.1)
    assert metrics["core_mass_g"] == pytest.approx(639.1)
    for original, installed in zip(PARTS, layout["parts"], strict=True):
        assert sorted(original.raw) == sorted(installed["raw_mm"])
        assert sorted(original.installed) == sorted(installed["installed_mm"])
        assert np.all(np.array(installed["raw_mm"]) <= installed["installed_mm"])
        vertices = corners(installed["pos_mm"], installed["installed_mm"], 3)
        norm = np.sum(((vertices - layout["silhouette_center_mm"]) / layout["silhouette_radii_mm"]) ** 2, axis=1)
        assert norm.max() <= 1 + 1e-7


def test_shrunken_motor_is_not_accepted_as_full_size():
    model, data, _ = validate.load("a_low_spine")
    model.geom_size[model.geom("left_knee_servo_case").id, 0] *= 0.8
    assert not validate.home_metrics(model, data)["all_cases_full_size"]


def test_installed_component_overlap_is_detected_even_without_contact_masks(monkeypatch):
    model, data, _ = validate.load("a_low_spine")
    computer = model.geom("part_computer_reserve").id
    battery = model.geom("part_battery_reserve").id
    assert model.geom_contype[computer] == model.geom_contype[battery] == 0
    model.geom_pos[computer] = model.geom_pos[battery]
    monkeypatch.setattr(validate, "pose_set", lambda model, random_count: [("interference", model.key_qpos[0].copy())])
    result = validate.packing(model, data, 0)
    # MuJoCo can return zero for overlapping convex boxes. An unresolved zero
    # must remain incomplete rather than becoming a false clearance pass.
    assert result["status"] in {"fail", "incomplete"}
    assert result["failed_evaluations"] + result["inconclusive_evaluations"] > 0
    assert result["closest_pairs"][0]["min_distance_m"] <= 0


def test_gravity_assistance_cannot_pass_unsupported_standing():
    model, _, _ = validate.load("a_low_spine")
    model.body_gravcomp[model.body("torso").id] = 1
    result = validate.stand(model, duration=0.1)
    assert not result["no_support_fixtures"]
    assert result["status"] == "fail"


def test_evidence_is_current_and_keeps_failed_candidate_visible():
    report = json.loads((HERE / "comparison.json").read_text())
    for path, expected in report["sources"].items():
        assert hashlib.sha256((validate.REPO / path).read_bytes()).hexdigest() == expected, path
    for candidate in report["layouts"]:
        assert candidate["packing"]["status"] == "pass"
        assert candidate["packing"]["pose_count"] == 1050
        assert candidate["packing"]["evaluations"] == 162805
        assert candidate["standing"][1]["status"] == "pass"
        assert not candidate["hardware_ready"]
    assert [r["overall_status"] for r in report["layouts"]] == ["incomplete", "incomplete", "fail"]
    assert report["status"] == "fail"
    assert report["layouts"][2]["standing"][0]["status"] == "fail"
    assert report["layouts"][2]["standing"][0]["max_horizontal_drift_mm"] > 10
