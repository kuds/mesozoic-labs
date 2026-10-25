"""Guard the isolated medium redesign and its retained mechanical assumptions."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest

from environments.shared.robot_design_validation import screen_design
from environments.shared.robot_geometry import visual_bounds
from examples.tyrannosaurus_rex_50cm import demo, redesign, redesign_review

HERE = Path(redesign.__file__).parent


def test_candidate_regenerates_without_changing_the_current_model(tmp_path):
    current_files = (HERE / "model.xml", HERE / "components.json", *sorted((HERE / "assets").glob("*.obj")))
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in current_files}
    model_path = redesign.build(tmp_path)
    assert model_path == tmp_path / "model.xml"
    committed = HERE / "long_body"
    for path in (tmp_path / "model.xml", tmp_path / "components.json", *sorted((tmp_path / "assets").glob("*.obj"))):
        assert path.read_bytes() == (committed / path.relative_to(tmp_path)).read_bytes(), path.name
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == expected for path, expected in before.items())


def test_longer_lower_layout_keeps_hardware_caps_leg_geometry_and_reserve():
    baseline, _ = demo.load_model()
    model, data = demo.load_model(path=HERE / "long_body/model.xml")
    low, high = visual_bounds(model, data)
    length, _, height = high - low
    assert 0.90 <= length <= 1.10
    assert 0.50 <= height <= 0.55
    assert 7.957 < model.body_mass.sum() < 8.5
    assert [model.actuator(i).name for i in range(model.nu)] == [baseline.actuator(i).name for i in range(baseline.nu)]
    np.testing.assert_array_equal(model.actuator_forcerange, baseline.actuator_forcerange)
    np.testing.assert_array_equal(model.jnt_range, baseline.jnt_range)
    for side in ("left", "right"):
        for suffix in ("hip_yaw_body", "hip_roll_body", "thigh", "shin", "ankle_pitch_body", "foot"):
            np.testing.assert_array_equal(model.body(side + "_" + suffix).pos, baseline.body(side + "_" + suffix).pos)
        for suffix in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll"):
            name = f"{side}_{suffix}_case_collision"
            np.testing.assert_array_equal(model.geom(name).size, baseline.geom(name).size)
            np.testing.assert_array_equal(model.geom(name).pos, baseline.geom(name).pos)
    ledger = json.loads((HERE / "long_body/components.json").read_text())
    assert ledger["categories_kg"]["unallocated_integration_reserve"] == 0.6
    assert ledger["categories_kg"]["leg_motors_12_x_330g"] == 3.96
    assert ledger["mass_kg"] == pytest.approx(model.body_mass.sum())
    assert model.body_subtreemass[model.body("head_yaw_body").id] == pytest.approx(0.530)


def test_candidate_clears_selected_pairs_without_relaxing_planning_limits():
    model, _ = demo.load_model(path=HERE / "long_body/model.xml")
    spec = redesign_review.candidate_spec(model)
    report = screen_design(model, spec)
    assert all(pair.min_distance_m == 0.002 for pair in spec.clearances)
    assert all(load.max_torque_fraction == 0.8 for load in spec.loads)
    assert report["known_limits_passed"]
    assert report["status"] == "incomplete"
    assert all(row["bearing_status"] == "incomplete" for row in report["load_checks"])
    assert report["coverage"]["sampled_poses"] == 790
    assert len(spec.clearances) >= 245


def test_forward_camera_centers_are_not_hidden_inside_the_skull():
    model, data = demo.load_model(path=HERE / "long_body/model.xml")
    groups = np.array([0, 1, 1, 0, 0, 0], dtype=np.uint8)
    hit = np.array([-1], dtype=np.int32)
    for label in ("left", "rgb", "right"):
        camera = model.camera("snout_" + label).id
        direction = -data.cam_xmat[camera].reshape(3, 3)[:, 2]
        distance = mujoco.mj_ray(model, data, data.cam_xpos[camera], direction, groups, 1, -1, hit)
        assert distance < 0, (label, model.geom(int(hit[0])).name, distance)


def test_parameterized_build_records_actual_layout_and_updates_mass(tmp_path):
    layout = replace(redesign.Layout(), battery_x_m=-0.130, neck_support_mass_g=100)
    path = redesign.build(tmp_path, layout)
    ledger = json.loads(path.with_name("components.json").read_text())
    assert ledger["layout"] == layout.as_dict()
    assert next(part for part in ledger["components"] if part["name"] == "battery")["pos_m"][0] == -0.130
    model = mujoco.MjModel.from_xml_path(str(path))
    current = mujoco.MjModel.from_xml_path(str(HERE / "long_body/model.xml"))
    assert model.body_mass.sum() - current.body_mass.sum() == pytest.approx(0.020)
