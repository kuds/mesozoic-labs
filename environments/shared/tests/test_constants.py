"""Tests for environments.shared.constants — regression guards."""

from environments.shared.constants import (
    DEFAULT_CLIP_OBS,
    DEFAULT_CLIP_REWARD,
    DEFAULT_FRAME_SKIP,
    DEFAULT_NORM_OBS,
    DEFAULT_NORM_REWARD,
    SENSOR_ACCEL_START,
    SENSOR_GYRO_START,
    SENSOR_QUAT_START,
    TAIL_ANGULAR_VEL_MAX,
)


def test_sensor_layout_values():
    """Sensor indices must match the MJCF sensor definition order."""
    assert SENSOR_GYRO_START == 0
    assert SENSOR_ACCEL_START == 3
    assert SENSOR_QUAT_START == 6


def test_vecnormalize_defaults():
    """VecNormalize defaults must match expected training configuration."""
    assert DEFAULT_NORM_OBS is True
    assert DEFAULT_NORM_REWARD is True
    assert DEFAULT_CLIP_OBS == 10.0
    assert DEFAULT_CLIP_REWARD == 50.0


def test_physics_defaults():
    """Physics constants must match expected simulation configuration."""
    assert DEFAULT_FRAME_SKIP == 5
    assert TAIL_ANGULAR_VEL_MAX == 10.0


def test_sensor_indices_are_non_overlapping():
    """Sensor ranges must not overlap (gyro=3, accel=3, quat=4)."""
    assert SENSOR_GYRO_START + 3 <= SENSOR_ACCEL_START
    assert SENSOR_ACCEL_START + 3 <= SENSOR_QUAT_START
