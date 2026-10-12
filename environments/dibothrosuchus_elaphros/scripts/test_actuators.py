#!/usr/bin/env python3
"""
Test actuators by applying sinusoidal control signals.
Useful for verifying joint ranges and actuator gains.

Usage (from the repository root):
    python -m environments.dibothrosuchus.scripts.test_actuators
"""

from pathlib import Path

from environments.shared.harnesses.actuators import ActuatorTestConfig, run_actuator_test

if __name__ == "__main__":
    run_actuator_test(
        ActuatorTestConfig(
            model_path=Path(__file__).parent.parent / "assets" / "dibothrosuchus.xml",
            camera_distance=1.8,
        )
    )
