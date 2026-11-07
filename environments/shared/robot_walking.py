"""Sensor and command interface for a phase-conditioned residual walking policy."""

from __future__ import annotations

import math

import numpy as np


def update_gravity(up: np.ndarray, gyro: np.ndarray, accel: np.ndarray, dt: float) -> np.ndarray:
    """Propagate gravity with gyro; reduce accelerometer trust under impacts."""
    estimate = up - np.cross(gyro, up) * dt
    magnitude = float(np.linalg.norm(accel))
    if 7.5 < magnitude < 12.0:
        estimate += min(1.0, 2.0 * dt) * (accel / magnitude - estimate)
    return np.asarray(estimate / max(1e-6, np.linalg.norm(estimate)))


def pack_sensors(angles, rates, up, gyro, loads, phase, expression, last_action, command) -> np.ndarray:
    """50 Hz actor packet. All arguments have physical sensor or controller sources."""
    return np.concatenate(
        (
            angles,
            rates * 0.1,
            up,
            gyro * 0.2,
            loads / 100.0,
            (math.sin(phase), math.cos(phase)),
            expression,
            last_action,
            command,
        )
    ).astype(np.float32)


def reference_angles(phase: float, command: np.ndarray, expression: np.ndarray, amplitude: float) -> np.ndarray:
    """Alternating step reference. PPO supplies bounded joint corrections."""
    speed, _, yaw = command
    target = np.zeros(14)
    # Left swing around pi/2. Positive hip roll places feet to the left,
    # shifting the torso right. Stance pitch moves the planted foot backward.
    wave = math.sin(phase + 1.20)
    for side in range(2):
        angle = phase + side * math.pi
        swing = max(0.0, math.sin(angle))
        stride = -math.cos(angle)
        offset = side * 6
        hip_roll = 0.126 * wave
        hip_pitch = -0.087 * (speed / 0.12) * stride - 0.095 * swing - 0.024
        knee = 0.315 * swing
        target[offset : offset + 6] = (
            0.14 * yaw * stride / 0.5,
            hip_roll,
            hip_pitch,
            knee,
            0.026 - hip_pitch - knee,
            -hip_roll,
        )
    target[:12] *= amplitude
    target[12:] = expression
    return target
