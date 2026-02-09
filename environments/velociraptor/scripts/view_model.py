#!/usr/bin/env python3
"""
Passive viewer for velociraptor MJCF iteration.

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
    # Load model
    model_path = Path(__file__).parent.parent / "assets" / "raptor.xml"
    print(f"Loading model from: {model_path}")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    # Print model info
    print(f"\n{'=' * 50}")
    print(f"Model: {model.name if hasattr(model, 'name') else 'velociraptor'}")
    print(f"{'=' * 50}")
    print(f"Bodies: {model.nbody}")
    print(f"Joints: {model.njnt}")
    print(f"Actuators: {model.nu}")
    print(f"Sensors: {model.nsensor}")
    print(f"Total DOF: {model.nv}")
    print(f"Total mass: {sum(model.body_mass):.2f} kg")
    print(f"Timestep: {model.opt.timestep * 1000:.1f} ms")
    print(f"{'=' * 50}\n")

    # Print joint info
    print("Joints:")
    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        joint_type = ["free", "ball", "slide", "hinge"][model.jnt_type[i]]
        print(f"  [{i:2d}] {joint_name}: {joint_type}")

    # Print actuator info
    print("\nActuators:")
    for i in range(model.nu):
        act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"  [{i:2d}] {act_name}")

    # Compute initial CoM
    mujoco.mj_forward(model, data)
    com = data.subtree_com[1].copy()  # subtree_com[1] is the raptor's CoM (body 0 is world)
    print(f"\nInitial CoM position: x={com[0]:.3f}, y={com[1]:.3f}, z={com[2]:.3f}")

    # Check if CoM is roughly over the feet
    # Feet are at approximately x=0.1 (forward of pelvis origin due to digitigrade stance)
    print(f"Pelvis height: {data.qpos[2]:.3f} m")

    print("\n" + "=" * 50)
    print("Launching viewer... (close window to exit)")
    print("=" * 50 + "\n")

    # Launch passive viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set initial camera position for good view
        viewer.cam.distance = 2.0
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.cam.lookat[:] = [0, 0, 0.3]

        while viewer.is_running():
            # Step simulation
            mujoco.mj_step(model, data)

            # Sync viewer
            viewer.sync()


if __name__ == "__main__":
    main()
