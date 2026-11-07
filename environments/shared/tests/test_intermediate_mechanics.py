"""Release regressions for the isolated 60 cm candidate and installed drives."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from environments.shared.intermediate_drive import InstalledDrive
from examples.tyrannosaurus_rex_60cm.platform.build import HERE, build


@pytest.mark.parametrize("name", ["compact", "nominal", "wide"])
def test_candidate_is_reproducible_and_balances_ledger(name, tmp_path):
    import json

    out = build(name, tmp_path / name)
    assert (out / "model.xml").read_bytes() == (HERE / name / "model.xml").read_bytes()
    # Component ledgers carry unrounded structure factors; compare numeric
    # data with the precision at which inertias enter the XML.
    ledger = json.loads((out / "components.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(out / "model.xml"))
    assert model.body_mass.sum() == pytest.approx(ledger["mass_kg"], abs=1e-8)
    assert model.nu == 14
    if name == "nominal":
        registered = Path(__file__).resolve().parents[2] / "tyrannosaurus_rex/intermediate/assets"
        assert (registered / "tyrannosaurus_rex_intermediate.xml").read_bytes() == (out / "model.xml").read_bytes()


def test_bus_speed_limit_and_reversal_delay():
    """Low bus headroom narrows positive motoring torque; braking remains available."""
    model = mujoco.MjModel.from_xml_path(str(HERE / "nominal/model.xml"))
    data = mujoco.MjData(model)
    drive = InstalledDrive(model, voltage=20)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    drive.reset(data)
    target = np.full(model.nu, 0.12)
    joint = model.jnt_dofadr[model.joint("left_hip_roll").id]
    actuator = model.actuator("left_hip_roll_servo").id
    data.qvel[joint] = 6.0
    drive.step(data, target)
    assert model.actuator_forcerange[actuator, 1] == 0
    assert model.actuator_forcerange[actuator, 0] == -12
    assert not np.any(data.ctrl)
    for _ in range(drive.delay):
        drive.step(data, target)
    np.testing.assert_allclose(data.ctrl, target - drive.lash)
    np.testing.assert_allclose(target, 0.12)


def test_camera_central_ray_is_clear_at_home():
    model = mujoco.MjModel.from_xml_path(str(HERE / "nominal/model.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cam = data.cam("front_rgbd")
    geom = np.zeros(1, dtype=np.int32)
    distance = mujoco.mj_ray(
        model,
        data,
        cam.xpos,
        -cam.xmat.reshape(3, 3)[:, 2],
        np.array([1, 1, 1, 0, 1, 0], dtype=np.uint8),
        True,
        -1,
        geom,
    )
    assert distance < 0  # empty forward indoor scene, camera aimed above the floor
