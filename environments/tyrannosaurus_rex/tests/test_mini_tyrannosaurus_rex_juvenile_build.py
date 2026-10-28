"""Prevent appearance changes from silently weakening the retained mechanism."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

from examples.mini_tyrannosaurus_rex.juvenile.build import SOURCE, build
from examples.mini_tyrannosaurus_rex.juvenile.torso import WALL_M


@pytest.fixture(scope="module")
def candidate(tmp_path_factory):
    path = build(tmp_path_factory.mktemp("juvenile"))
    return path, mujoco.MjModel.from_xml_path(str(path))


def test_retains_joint_limits_motor_dimensions_and_actuator_limits(candidate):
    _, model = candidate
    baseline = mujoco.MjModel.from_xml_path(str(SOURCE))
    assert model.nu == baseline.nu == 14
    for index in range(baseline.njnt):
        joint = baseline.joint(index)
        result = model.joint(joint.name)
        np.testing.assert_allclose(result.range, joint.range)
        np.testing.assert_allclose(result.axis, joint.axis)
        np.testing.assert_allclose(result.pos, joint.pos)
    for index in range(baseline.nu):
        actuator = baseline.actuator(index)
        result = model.actuator(actuator.name)
        np.testing.assert_allclose(result.forcerange, actuator.forcerange)
        np.testing.assert_allclose(result.ctrlrange, actuator.ctrlrange)
        np.testing.assert_allclose(result.gainprm, actuator.gainprm)
    motors = [baseline.geom(i) for i in range(baseline.ngeom) if baseline.geom(i).name.endswith("_servo_case")]
    assert len(motors) == 14
    for geom in motors:
        np.testing.assert_allclose(model.geom(geom.name).size, geom.size)
        np.testing.assert_allclose(model.geom(geom.name).pos, geom.pos)
        np.testing.assert_allclose(model.geom(geom.name).quat, geom.quat)


def test_torso_wall_prisms_have_declared_normal_thickness(candidate):
    path, _ = candidate
    root = ET.parse(path).getroot()
    walls = [mesh for mesh in root.findall("asset/mesh") if mesh.get("name", "").startswith("juv_torso_wall_")]
    assert len(walls) > 100
    for mesh in walls:
        points = np.fromstring(mesh.attrib["vertex"], sep=" ").reshape(-1, 3)
        assert points.shape == (6, 3)
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        normal /= np.linalg.norm(normal)
        np.testing.assert_allclose((points[:3] - points[3:]) @ normal, WALL_M, atol=1e-9)
        np.testing.assert_allclose(np.linalg.norm(points[:3] - points[3:], axis=1), WALL_M, atol=1e-9)


def test_mass_ledger_reconciles_compiled_model(candidate):
    path, model = candidate
    ledger = json.loads(Path(path).with_name("mass_ledger.json").read_text())
    total = 0.0
    for name, rows in ledger["body_components"].items():
        allocated = sum(row["mass_kg"] for row in rows)
        assert allocated == pytest.approx(model.body(name).mass[0], abs=1e-9)
        total += allocated
    total += sum(model.body(name).mass[0] for name in ledger["retained_body_inertias"])
    assert total == pytest.approx(model.body_mass.sum(), abs=1e-9)
    camera = [
        row for row in ledger["body_components"]["head_pitch_body"] if row["name"] == "camera_installed_allowance"
    ]
    assert len(camera) == 1 and camera[0]["mass_kg"] == 0.008
