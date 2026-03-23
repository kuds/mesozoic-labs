"""Tests for the pure observation-building functions."""

import numpy as np
import pytest

from environments.shared.obs_functions import SensorLayout, build_bipedal_obs, build_quadruped_obs


class TestSensorLayout:
    def test_defaults(self):
        layout = SensorLayout()
        assert layout.gyro_start == 0
        assert layout.accel_start == 3
        assert layout.quat_start == 6
        assert layout.foot_indices == ()

    def test_frozen(self):
        layout = SensorLayout()
        with pytest.raises(AttributeError):
            layout.gyro_start = 10  # type: ignore[misc]


class TestBuildBipedalObs:
    """Test the bipedal observation builder with NumPy arrays."""

    @pytest.fixture
    def mock_state(self):
        """Create minimal mock state arrays for a bipedal env."""
        # 7 root qpos + 24 joint qpos = 31
        qpos = np.zeros(31)
        # 6 root qvel + 24 joint qvel = 30
        qvel = np.zeros(30)
        # Sensor data: gyro(3) + accel(3) + quat(4) + r_foot(1) + l_foot(1) = 12
        sensordata = np.zeros(12)
        sensordata[6:10] = [1.0, 0.0, 0.0, 0.0]  # identity quaternion
        pelvis_xpos = np.array([0.0, 0.0, 0.5])
        target_pos = np.array([5.0, 0.0, 0.3])
        layout = SensorLayout(gyro_start=0, accel_start=3, quat_start=6, foot_indices=(10, 11))
        return qpos, qvel, sensordata, pelvis_xpos, target_pos, layout

    def test_output_shape(self, mock_state):
        qpos, qvel, sensordata, pelvis_xpos, target_pos, layout = mock_state
        obs = build_bipedal_obs(qpos, qvel, sensordata, pelvis_xpos, target_pos, layout)
        # 24 joint_pos + 24 joint_vel + 4 quat + 3 gyro + 3 linvel + 3 accel + 2 feet + 3 dir + 1 dist = 67
        assert obs.shape == (67,)

    def test_dtype_is_float32(self, mock_state):
        qpos, qvel, sensordata, pelvis_xpos, target_pos, layout = mock_state
        obs = build_bipedal_obs(qpos, qvel, sensordata, pelvis_xpos, target_pos, layout)
        assert obs.dtype == np.float32

    def test_target_distance_positive(self, mock_state):
        qpos, qvel, sensordata, pelvis_xpos, target_pos, layout = mock_state
        obs = build_bipedal_obs(qpos, qvel, sensordata, pelvis_xpos, target_pos, layout)
        # Last element is target distance
        assert obs[-1] > 0.0


class TestBuildQuadrupedObs:
    """Test that the quadruped builder delegates correctly."""

    def test_output_shape(self):
        # 7 root qpos + 26 joint qpos = 33
        qpos = np.zeros(33)
        # 6 root qvel + 26 joint qvel = 32
        qvel = np.zeros(32)
        # Sensor data with 4 foot sensors
        sensordata = np.zeros(14)
        sensordata[6:10] = [1.0, 0.0, 0.0, 0.0]
        torso_xpos = np.array([0.0, 0.0, 1.5])
        target_pos = np.array([5.0, 0.0, 3.0])
        layout = SensorLayout(gyro_start=0, accel_start=3, quat_start=6, foot_indices=(10, 11, 12, 13))

        obs = build_quadruped_obs(qpos, qvel, sensordata, torso_xpos, target_pos, layout)
        # 26 joint_pos + 26 joint_vel + 4 quat + 3 gyro + 3 linvel + 3 accel + 4 feet + 3 dir + 1 dist = 73
        assert obs.shape == (73,)
        assert obs.dtype == np.float32
