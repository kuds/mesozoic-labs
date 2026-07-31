#!/usr/bin/env python3
"""
Test actuators by applying sinusoidal control signals.
Useful for verifying joint ranges and actuator gains.

Usage:
    python test_actuators.py
"""

from pathlib import Path

from environments.shared.harnesses.actuators import ActuatorTestConfig, run_actuator_test

if __name__ == "__main__":
    run_actuator_test(
        ActuatorTestConfig(
            model_path=Path(__file__).parent.parent / "assets" / "trex.xml",
            camera_distance=3.0,
        )
    )
