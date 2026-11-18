"""Headless regression checks for the two standalone Triceratops scenes."""

from pathlib import Path

import mujoco
import numpy as np


MODEL_DIR = Path(__file__).resolve().parent
CORNERS = ("left_fore", "right_fore", "left_hind", "right_hind")


def check_stand(model, data):
    """Load the default pose and test support under normal gravity for three seconds."""
    assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE, "Expected a free root"
    np.testing.assert_allclose(model.opt.gravity, [0, 0, -9.81], atol=0.05)
    root = model.jnt_bodyid[0]
    mujoco.mj_forward(model, data)
    initial_height = data.xpos[root, 2]
    for _ in range(30):
        mujoco.mj_step(model, data, nstep=round(0.1 / model.opt.timestep))
        mujoco.mj_forward(model, data)
        assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
        assert data.xpos[root, 2] > 0.8 * initial_height, "Root lost standing height"
        assert data.xmat[root, 8] > 0.9, "Root tipped over"
    assert not np.any(data.warning.number), "Numerical warning during stance"
    assert np.isfinite(data.sensordata).all()

    touch = np.flatnonzero(model.sensor_type == mujoco.mjtSensor.mjSENS_TOUCH)
    assert len(touch) == 4, "Expected one load sensor on each foot"
    sites = model.sensor_objid[touch]
    assert len(set(model.site_bodyid[sites])) == 4, "Foot sensors must be on distinct bodies"
    loads = data.sensordata[model.sensor_adr[touch]]
    weight = -model.opt.gravity[2] * model.body_mass.sum()
    assert np.all(loads > 0.02 * weight), "A foot is unloaded"
    np.testing.assert_allclose(loads.sum(), weight, rtol=0.10)
    return loads


def check_robot(model, data):
    """Exercise each servo at a fixed state, then move the tail in a free stance."""
    axes = {f"{corner}_{axis}" for corner in CORNERS for axis in ("roll", "pitch", "flexion")}
    axes.update(("neck_pitch", "tail_yaw"))
    joint_names = [model.joint(j).name for j in model.actuator_trnid[:, 0]]
    assert model.nu == 14 and set(joint_names) == axes, "Incorrect motor-to-joint mapping"
    assert np.all(model.actuator_forcelimited), "Expected bounded servo effort"
    assert model.camera("oak_d_lite_rgb").id >= 0
    for name in ("imu_orientation", "imu_angular_velocity", "imu_specific_force"):
        assert model.sensor(name).id >= 0
    for side in ("left", "right"):
        assert data.sensor(f"{side}_tof_distance").data[0] > 0, "ToF lost the floor"

    # At the same qpos/qvel, a small target change must alter that drive's joint effort.
    home = data.ctrl.copy()
    baseline = data.qfrc_actuator.copy()
    for i, joint in enumerate(model.actuator_trnid[:, 0]):
        data.ctrl[i] += 0.015
        mujoco.mj_forward(model, data)
        dof = model.jnt_dofadr[joint]
        assert data.qfrc_actuator[dof] > baseline[dof] + 0.001, joint_names[i]
        data.ctrl[i] = home[i]

    before = data.joint("tail_yaw").qpos[0]
    data.ctrl[joint_names.index("tail_yaw")] = before + 0.05
    mujoco.mj_step(model, data, nstep=round(0.5 / model.opt.timestep))
    assert data.joint("tail_yaw").qpos[0] > before + 0.02, "Tail failed to move"
    assert not np.any(data.warning.number)


def check_anatomical(model, data):
    before = data.joint("jaw_gape").qpos[0]
    data.ctrl[model.actuator("jaw_gape_torque").id] = 0.4
    mujoco.mj_step(model, data, nstep=round(0.5 / model.opt.timestep))
    assert data.joint("jaw_gape").qpos[0] > before + 0.05, "Jaw failed to open"
    assert not np.any(data.warning.number)


def main():
    for name in ("anatomical", "robot"):
        model = mujoco.MjModel.from_xml_path(str(MODEL_DIR / f"{name}.xml"))
        data = mujoco.MjData(model)
        loads = check_stand(model, data)
        if name == "robot":
            check_robot(model, data)
        else:
            check_anatomical(model, data)
        print(
            f"{name}: {model.nu} actuators, {model.body_mass.sum():.2f} kg, "
            f"four settled foot loads {loads.round(1).tolist()} N"
        )


if __name__ == "__main__":
    main()
