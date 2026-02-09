#!/usr/bin/env python3
"""
Passive viewer for T-Rex MJCF model.

Usage:
    python view_model.py

Controls:
    - Mouse drag: rotate view
    - Scroll: zoom
    - Double-click: track body
    - Ctrl+drag: pan
    - Space: pause/unpause
    - Backspace: reset
    - Tab: toggle UI panels
"""

from pathlib import Path

import mujoco
import mujoco.viewer


def main():
    model_path = Path(__file__).parent.parent / "assets" / "trex.xml"
    print(f"Loading model from: {model_path}")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    print(f"\n{'=' * 50}")
    print("Model: tyrannosaurus_rex")
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
    print(f"Pelvis height: {data.qpos[2]:.3f} m")

    print("\n" + "=" * 50)
    print("Launching viewer... (close window to exit)")
    print("=" * 50 + "\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 5.0
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.cam.lookat[:] = [0, 0, 0.5]

        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
