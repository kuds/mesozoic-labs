"""Functional sensor checks and physical regressions for the styled geometry."""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from environments.compsognathus import MODEL_PATHS
from environments.compsognathus.model import PARAMETERS, floor_contacts, load_model, robot_body_ids, set_motors_enabled
from environments.compsognathus.sensors import onboard_readings


@pytest.mark.parametrize("variant", ["biological", "robot"])
def test_onboard_encoders_exclude_privileged_state(variant):
    model, data = load_model(variant)
    data.qvel[:] = np.linspace(-0.2, 0.3, model.nv)
    mujoco.mj_forward(model, data)
    readings = onboard_readings(model, data)
    assert set(readings) == {
        "joint_names",
        "joint_position_rad",
        "joint_velocity_rad_s",
        "gyro_rad_s",
        "specific_force_m_s2",
    }
    joints = model.actuator_trnid[:, 0]
    np.testing.assert_allclose(readings["joint_position_rad"], data.qpos[model.jnt_qposadr[joints]])
    np.testing.assert_allclose(readings["joint_velocity_rad_s"], data.qvel[model.jnt_dofadr[joints]])
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, model.site("imu").id, velocity, 1)
    np.testing.assert_allclose(readings["gyro_rad_s"], velocity[:3], atol=1e-12)
    assert readings["joint_position_rad"].shape == (model.nu,)


@pytest.mark.parametrize("variant", ["biological", "robot"])
def test_accelerometer_reports_specific_force_and_airborne_feet_are_zero(variant):
    model, data = load_model(variant)
    for _ in range(5000):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    rotation = data.site("imu").xmat.reshape(3, 3)
    np.testing.assert_allclose(data.sensor("pelvis_accel").data, rotation.T @ -model.opt.gravity, atol=0.04)
    np.testing.assert_allclose(data.sensor("pelvis_gyro").data, 0, atol=0.002)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    data.qpos[2] += 1
    set_motors_enabled(model, False)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.sensor("pelvis_accel").data, 0, atol=1e-8)
    np.testing.assert_array_equal(
        onboard_readings(model, data, include_foot_contacts=True)["foot_normal_force_N_right_left"], 0
    )


@pytest.mark.parametrize("variant", ["biological", "robot"])
@pytest.mark.parametrize("roll_deg", [-2, 2])
def test_each_touch_sensor_captures_its_own_contacts_during_landing(variant, roll_deg):
    model, data = load_model(variant)
    angle = np.deg2rad(roll_deg) / 2
    data.qpos[3:7] = [np.cos(angle), np.sin(angle), 0, 0]
    data.qpos[2] += 0.005
    observed = {"r": False, "l": False}
    for _ in range(300):
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        contacts = floor_contacts(model, data)
        for short, long in (("r", "right"), ("l", "left")):
            prefix = long if variant == "robot" else short
            force = sum(c["normal_force"] for c in contacts if c["geom"].startswith(prefix + "_"))
            assert float(data.sensor(f"{short}_foot_touch").data[0]) == pytest.approx(force, abs=1e-7)
            observed[short] |= force > 0.1
    assert all(observed.values())


@pytest.mark.parametrize("variant", ["biological", "robot"])
def test_camera_orientation_target_and_frustum_are_unoccluded(variant):
    model, data = load_model(variant)
    cam = model.camera("head_camera").id
    rotation = data.cam_xmat[cam].reshape(3, 3)
    forward = -rotation[:, 2]
    np.testing.assert_allclose(forward, [1, 0, 0], atol=1e-10)
    np.testing.assert_allclose(rotation[:, 0], [0, -1, 0], atol=1e-10)
    np.testing.assert_allclose(rotation[:, 1], [0, 0, 1], atol=1e-10)
    np.testing.assert_array_equal(model.cam_resolution[cam], [640, 480])
    # Put a target on the optical axis to prove that the camera can see forward.
    mocap = model.body_mocapid[model.body("prey").id]
    data.mocap_pos[mocap] = data.cam_xpos[cam] + forward * 0.5
    mujoco.mj_forward(model, data)
    geom_id = np.full(1, -1, dtype=np.int32)
    distance = mujoco.mj_ray(model, data, data.cam_xpos[cam], forward, None, True, -1, geom_id)
    assert 0 < distance < 0.5
    assert model.geom_bodyid[geom_id[0]] == model.body("prey").id
    tan_v = np.tan(np.deg2rad(model.cam_fovy[cam]) / 2)
    bodies = set(robot_body_ids(model))
    # Includes visual-only geoms and the camera's own body; no exclusion cheat.
    for x in np.linspace(-1, 1, 9):
        for y in np.linspace(-1, 1, 7):
            direction = rotation @ np.array([x * tan_v * 4 / 3, y * tan_v, -1])
            direction /= np.linalg.norm(direction)
            geom_id[:] = -1
            distance = mujoco.mj_ray(model, data, data.cam_xpos[cam], direction, None, True, -1, geom_id)
            assert distance < 0 or int(model.geom_bodyid[geom_id[0]]) not in bodies, (
                x,
                y,
                model.geom(int(geom_id[0])).name,
            )


def test_robot_imu_is_inside_core_and_head_mass_is_accounted_for():
    model, data = load_model("robot")
    core = model.geom("core_envelope").id
    local = data.geom_xmat[core].reshape(3, 3).T @ (data.site("imu").xpos - data.geom_xpos[core])
    assert np.sum((local / model.geom_size[core]) ** 2) < 1
    parts = PARAMETERS["robot"]["head_parts"]
    total = sum(p["mass_kg"] for p in parts)
    head = model.body("fixed_head").id
    assert total == pytest.approx(0.045)
    assert model.body_mass[head] == pytest.approx(total)
    expected_com = sum(p["mass_kg"] * np.array(p["pos_m"]) for p in parts) / total
    np.testing.assert_allclose(model.body_ipos[head], expected_com, atol=1e-12)
    # Independent cross-check: let MuJoCo integrate the component primitives.
    root = ET.Element("mujoco")
    body = ET.SubElement(ET.SubElement(root, "worldbody"), "body", name="head")
    for part in parts:
        ET.SubElement(
            body,
            "geom",
            type=part["type"],
            mass=str(part["mass_kg"]),
            pos=" ".join(map(str, part["pos_m"])),
            size=" ".join(map(str, part["size_m"])),
        )
    reference = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    np.testing.assert_allclose(model.body_inertia[head], reference.body_inertia[1], rtol=1e-8)


def test_anatomical_palette_and_major_proportions_follow_species_style():
    root = ET.parse(MODEL_PATHS["biological"])
    trex = ET.parse(MODEL_PATHS["biological"].parents[2] / "trex/assets/trex.xml")
    for material in ("body_mat", "belly_mat", "head_mat", "claw_mat", "tooth_mat"):
        actual = root.find(f"asset/material[@name='{material}']").get("rgba")
        expected = trex.find(f"asset/material[@name='{material}']").get("rgba")
        np.testing.assert_allclose(np.fromstring(actual, sep=" "), np.fromstring(expected, sep=" "))
    p = PARAMETERS["biological"]
    assert sum(p["tail_segment_lengths_m"]) > 0.4
    # These are proxy proportions, not a specimen-level accuracy claim.
    model, _ = load_model("biological")
    assert np.linalg.norm(model.body_pos[model.body("r_metatarsus").id]) > np.linalg.norm(
        model.body_pos[model.body("r_tibia").id]
    )
