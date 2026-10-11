"""Every plant stands on the elliptic friction cone, so every world heading is physically identical.

MuJoCo's default cone is pyramidal, and a pyramidal cone is weaker between the axes of a contact's tangent frame. A
box or ellipsoid on the floor plane keeps that frame on the world axes whichever way the animal faces, so under the
default a turned animal stood on slightly different friction (the T. rex's box pads until its physics r9, D-D29; the
brachiosaurus's ellipsoid feet and the dibothrosuchus's box feet until D-D30). A capsule's frame turns with the foot,
so the velociraptor's statue was heading-invariant under either cone, though a fall onto its ellipsoid head was not.
Since D-D30 every species declares ``cone="elliptic"`` (``configs/plant_versions.toml`` notes 16 and 17), the cone
the two compsognathus plants were built with, so the whole zoo has one friction model.

The guard fails a new species, or an MJCF edit that drops the attribute, by name, instead of letting it fall back to
the default unnoticed. The turn tests are the T. rex's (``test_trex_env.py``) on the three species D-D30 moved: raw
MuJoCo on the env's model, so nothing but the physics is compared.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from environments.shared.plant_contract.versions import _species_entries
from environments.shared.species_registry import get_species_config

REGISTERED_SPECIES = sorted(_species_entries())


def test_the_guard_reads_every_registered_species():
    """The registry the guard walks is the species manifest's, so a new species is guarded from its first commit."""
    assert {"trex", "velociraptor", "brachiosaurus", "dibothrosuchus", "compsognathus", "compsognathus_robot"} <= set(
        REGISTERED_SPECIES
    )


@pytest.mark.parametrize("species", REGISTERED_SPECIES)
def test_every_registered_plant_declares_the_elliptic_cone(species):
    env = get_species_config(species).env_class()
    try:
        assert env.model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC, (
            f"{species}'s MJCF falls back to MuJoCo's default pyramidal friction cone: declare "
            '<option cone="elliptic"> (D-D30), or a turned animal stands on different friction'
        )
    finally:
        env.close()


STEPS = 2000
HOLD_STEPS = 10  # physics steps each random command is held
AMPLITUDE = 0.15  # of each actuator's half range, about the home ctrl


def _joint_trajectory(model: mujoco.MjModel, yaw_deg: float) -> np.ndarray:
    """Joint qpos per step from the home keyframe, the root and every mocap body turned *yaw_deg* about world z."""
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    half = np.radians(yaw_deg) / 2
    turn = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
    root = np.zeros(4)
    mujoco.mju_mulQuat(root, turn, data.qpos[3:7].copy())
    data.qpos[3:7] = root
    pivot = data.qpos[:2].copy()
    cos, sin = np.cos(2 * half), np.sin(2 * half)
    for body in range(model.nmocap):
        offset = data.mocap_pos[body, :2] - pivot
        data.mocap_pos[body, :2] = pivot + [cos * offset[0] - sin * offset[1], sin * offset[0] + cos * offset[1]]
        mocap = np.zeros(4)
        mujoco.mju_mulQuat(mocap, turn, data.mocap_quat[body].copy())
        data.mocap_quat[body] = mocap
    mujoco.mj_forward(model, data)
    rng = np.random.default_rng(0)
    low, high = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    home = model.key("home").ctrl.copy()
    rows = []
    for step in range(STEPS):
        if step % HOLD_STEPS == 0:
            noise = rng.uniform(-AMPLITUDE, AMPLITUDE, model.nu) * (high - low) / 2
            data.ctrl[:] = np.clip(home + noise, low, high)
        mujoco.mj_step(model, data)
        rows.append(data.qpos[7:].copy())
    return np.array(rows)


@pytest.mark.parametrize("species", ["velociraptor", "brachiosaurus", "dibothrosuchus"])
def test_a_45_degree_whole_scene_turn_reproduces_the_joints(species):
    """The turned run repeats the unturned one to rounding error (at most 4e-14 rad measured).

    Under the pyramidal cone the same turn separated the joints by 5.4e-3 rad on velociraptor physics r3 (once it
    fell onto its ellipsoid head), 7.9e-3 on brachiosaurus r4 and 8.7e-2 on dibothrosuchus r1.
    """
    env = get_species_config(species).env_class()
    try:
        straight = _joint_trajectory(env.model, 0.0)
        turned = _joint_trajectory(env.model, 45.0)
    finally:
        env.close()
    np.testing.assert_allclose(turned, straight, rtol=0, atol=1e-10)
