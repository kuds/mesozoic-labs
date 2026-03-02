"""Tests for shared constants — regression guards."""

from environments.shared.constants import (
    DEFAULT_CLIP_OBS,
    DEFAULT_CLIP_REWARD,
    DEFAULT_FRAME_SKIP,
    SENSOR_ACCEL_START,
    SENSOR_GYRO_START,
    SENSOR_QUAT_START,
    TAIL_ANGULAR_VEL_MAX,
)


def test_sensor_layout_gyro():
    assert SENSOR_GYRO_START == 0


def test_sensor_layout_accel():
    assert SENSOR_ACCEL_START == 3


def test_sensor_layout_quat():
    assert SENSOR_QUAT_START == 6


def test_vecnormalize_defaults():
    assert DEFAULT_CLIP_OBS == 10.0
    assert DEFAULT_CLIP_REWARD == 50.0


def test_frame_skip():
    assert DEFAULT_FRAME_SKIP == 5


def test_tail_max():
    assert TAIL_ANGULAR_VEL_MAX == 10.0
