#!/usr/bin/env python3
"""
Test actuators by applying sinusoidal control signals.
Useful for verifying joint ranges and actuator gains.

Usage:
    python test_actuators.py
"""

from pathlib import Path

from environments.shared.test_actuators_base import ActuatorTestConfig, test_actuators

if __name__ == "__main__":
    test_actuators(
        ActuatorTestConfig(
            model_path=Path(__file__).parent.parent / "assets" / "raptor.xml",
            camera_distance=2.5,
            motor_keywords=["claw"],
        )
    )
