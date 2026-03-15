#!/usr/bin/env python3
"""
Test actuators by applying sinusoidal control signals.
Useful for verifying joint ranges and actuator gains.

Usage:
    python test_actuators.py
"""

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


def main():
    # Load model
    model_path = Path(__file__).parent.parent / "assets" / "trex.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    print(f"Testing {model.nu} actuators...")
    print("Each actuator will move through its range.\n")

    # Map actuator index to name
    act_names = []
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        act_names.append(name)
        print(f"  [{i:2d}] {name}: range [{model.actuator_ctrlrange[i, 0]:.1f}, {model.actuator_ctrlrange[i, 1]:.1f}]")

    print("\nStarting viewer with sinusoidal actuation...")
    print("Watch the joints move through their ranges.")

    start_time = time.time()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -15

        while viewer.is_running():
            t = time.time() - start_time

            # Apply sinusoidal control to each actuator
            for i in range(model.nu):
                # Different frequency for each actuator to see them move independently
                freq = 0.5 + 0.1 * i
                phase = i * 0.5

                # Get control range
                ctrl_min = model.actuator_ctrlrange[i, 0]
                ctrl_max = model.actuator_ctrlrange[i, 1]

                # Position: oscillate through joint range
                mid = (ctrl_min + ctrl_max) / 2
                amp = (ctrl_max - ctrl_min) / 2 * 0.8  # 80% of range
                data.ctrl[i] = mid + amp * np.sin(2 * np.pi * freq * t + phase)

            # Step simulation
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
