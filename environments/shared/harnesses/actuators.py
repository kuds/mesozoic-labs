"""Shared actuator test viewer for MJCF dinosaur models.

Each species' ``scripts/test_actuators.py`` calls :func:`run_actuator_test`
with its species-specific parameters, eliminating duplicated test logic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


@dataclass
class ActuatorTestConfig:
    """Species-specific actuator test parameters."""

    model_path: Path
    camera_distance: float = 3.0
    motor_keywords: list[str] = field(default_factory=list)


def run_actuator_test(cfg: ActuatorTestConfig) -> None:
    """Load a model and apply sinusoidal control signals to each actuator."""
    model = mujoco.MjModel.from_xml_path(str(cfg.model_path))
    data = mujoco.MjData(model)

    print(f"Testing {model.nu} actuators...")
    print("Each actuator will move through its range.\n")

    act_names: list[str] = []
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        act_names.append(name)
        print(f"  [{i:2d}] {name}: range [{model.actuator_ctrlrange[i, 0]:.1f}, {model.actuator_ctrlrange[i, 1]:.1f}]")

    print("\nStarting viewer with sinusoidal actuation...")
    print("Watch the joints move through their ranges.")

    start_time = time.time()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = cfg.camera_distance
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -15

        while viewer.is_running():
            t = time.time() - start_time

            for i in range(model.nu):
                freq = 0.5 + 0.1 * i
                phase = i * 0.5

                ctrl_min = model.actuator_ctrlrange[i, 0]
                ctrl_max = model.actuator_ctrlrange[i, 1]

                # Motor actuators (e.g. claws) oscillate between -1 and 1;
                # position actuators oscillate through their joint range.
                is_motor = any(kw in act_names[i] for kw in cfg.motor_keywords)
                if is_motor:
                    data.ctrl[i] = np.sin(2 * np.pi * freq * t + phase)
                else:
                    mid = (ctrl_min + ctrl_max) / 2
                    amp = (ctrl_max - ctrl_min) / 2 * 0.8
                    data.ctrl[i] = mid + amp * np.sin(2 * np.pi * freq * t + phase)

            mujoco.mj_step(model, data)
            viewer.sync()
