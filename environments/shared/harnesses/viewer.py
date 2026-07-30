"""Shared passive viewer for MJCF dinosaur models.

Each species' ``scripts/view_model.py`` calls :func:`view_model` with its
species-specific parameters, eliminating duplicated viewer boilerplate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer


@dataclass
class ViewerConfig:
    """Species-specific viewer parameters."""

    model_path: Path
    species_name: str
    height_label: str = "Pelvis height"
    camera_distance: float = 3.0
    camera_lookat_z: float = 0.5


def view_model(cfg: ViewerConfig) -> None:
    """Load a MuJoCo model, print diagnostics, and launch the passive viewer."""
    print(f"Loading model from: {cfg.model_path}")

    model = mujoco.MjModel.from_xml_path(str(cfg.model_path))
    data = mujoco.MjData(model)

    print(f"\n{'=' * 50}")
    print(f"Model: {cfg.species_name}")
    print(f"{'=' * 50}")
    print(f"Bodies: {model.nbody}")
    print(f"Joints: {model.njnt}")
    print(f"Actuators: {model.nu}")
    print(f"Sensors: {model.nsensor}")
    print(f"Total DOF: {model.nv}")
    print(f"Total mass: {sum(model.body_mass):.2f} kg")
    print(f"Timestep: {model.opt.timestep * 1000:.1f} ms")
    print(f"{'=' * 50}\n")

    print("Joints:")
    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        joint_type = ["free", "ball", "slide", "hinge"][model.jnt_type[i]]
        print(f"  [{i:2d}] {joint_name}: {joint_type}")

    print("\nActuators:")
    for i in range(model.nu):
        act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"  [{i:2d}] {act_name}")

    mujoco.mj_forward(model, data)
    com = data.subtree_com[1].copy()
    print(f"\nInitial CoM: x={com[0]:.3f}, y={com[1]:.3f}, z={com[2]:.3f}")
    print(f"{cfg.height_label}: {data.qpos[2]:.3f} m")

    print("\n" + "=" * 50)
    print("Launching viewer... (close window to exit)")
    print("=" * 50 + "\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = cfg.camera_distance
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.cam.lookat[:] = [0, 0, cfg.camera_lookat_z]

        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
