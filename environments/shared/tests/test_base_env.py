"""Tests for BaseDinoEnv via RaptorEnv (concrete subclass)."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from environments.shared.base_env import BaseDinoEnv
from environments.velociraptor.envs.raptor_env import RaptorEnv


@pytest.fixture(scope="module")
def env():
    """Create a RaptorEnv once per module for performance."""
    e = RaptorEnv()
    yield e
    e.close()


class TestQuatToTilt:
    """Test the static _quat_to_tilt method."""

    def test_upright_quaternion(self):
        # Identity quaternion (w=1, x=0, y=0, z=0) => no tilt
        tilt = BaseDinoEnv._quat_to_tilt(np.array([1.0, 0.0, 0.0, 0.0]))
        assert tilt == pytest.approx(0.0, abs=1e-6)

    def test_sideways_quaternion(self):
        # 90-degree rotation about x-axis: w=cos(pi/4), x=sin(pi/4), y=0, z=0
        angle = np.pi / 2
        quat = np.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0])
        tilt = BaseDinoEnv._quat_to_tilt(quat)
        assert tilt == pytest.approx(np.pi / 2, abs=0.01)

    def test_inverted_quaternion(self):
        # 180-degree rotation about x-axis
        quat = np.array([0.0, 1.0, 0.0, 0.0])
        tilt = BaseDinoEnv._quat_to_tilt(quat)
        assert tilt == pytest.approx(np.pi, abs=0.01)


class TestInitialization:
    def test_action_space(self, env):
        assert env.action_space.shape[0] == env.model.nu
        assert np.all(env.action_space.low == -1.0)
        assert np.all(env.action_space.high == 1.0)

    def test_observation_space(self, env):
        assert env.observation_space.shape[0] > 0
        assert env.observation_space.dtype == np.float32

    def test_frame_skip(self, env):
        assert env.frame_skip == 5
        assert env.dt == env.model.opt.timestep * env.frame_skip

    def test_default_params(self, env):
        assert env.max_episode_steps == 1000
        assert env.alive_bonus > 0
        assert env.energy_penalty_weight > 0


class TestScaleAction:
    def test_zero_action_maps_to_midpoint(self, env):
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        scaled = env._scale_action(action)
        ctrl_range = env.model.actuator_ctrlrange
        expected = (ctrl_range[:, 0] + ctrl_range[:, 1]) / 2.0
        np.testing.assert_allclose(scaled, expected, atol=1e-5)

    def test_positive_one_maps_to_max(self, env):
        action = np.ones(env.action_space.shape, dtype=np.float32)
        scaled = env._scale_action(action)
        ctrl_max = env.model.actuator_ctrlrange[:, 1]
        np.testing.assert_allclose(scaled, ctrl_max, atol=1e-5)

    def test_negative_one_maps_to_min(self, env):
        action = -np.ones(env.action_space.shape, dtype=np.float32)
        scaled = env._scale_action(action)
        ctrl_min = env.model.actuator_ctrlrange[:, 0]
        np.testing.assert_allclose(scaled, ctrl_min, atol=1e-5)


class TestStep:
    def test_step_returns_correct_tuple(self, env):
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        result = env.step(action)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, (float, np.floating))
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert isinstance(info, dict)
        assert "step" in info

    def test_step_increments_count(self, env):
        env.reset(seed=42)
        env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert env._step_count == 1
        env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert env._step_count == 2

    def test_fall_penalty_on_termination(self, env):
        """When terminated, fall_penalty should be added to reward."""
        env.reset(seed=42)
        # Force pelvis to ground to trigger termination
        env.data.qpos[2] = 0.01
        import mujoco

        mujoco.mj_forward(env.model, env.data)

        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, reward, terminated, _, info = env.step(action)
        if terminated:
            # reward_total in info should include fall_penalty
            assert reward < 0


class TestIsTruncated:
    def test_not_truncated_initially(self, env):
        env.reset(seed=42)
        assert not env._is_truncated()

    def test_truncated_at_max_steps(self, env):
        env.reset(seed=42)
        env._step_count = env.max_episode_steps
        assert env._is_truncated()


class TestReset:
    def test_reset_returns_obs_and_info(self, env):
        obs, info = env.reset(seed=42)
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == np.float32
        assert isinstance(info, dict)
        assert info["step"] == 0

    def test_reset_clears_step_count(self, env):
        env.reset(seed=42)
        env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert env._step_count > 0
        env.reset(seed=43)
        assert env._step_count == 0

    def test_reset_with_different_seeds(self, env):
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=99)
        # Different seeds should give different observations (due to noise)
        assert not np.allclose(obs1, obs2)


class TestMakeCamera:
    def test_returns_camera(self, env):
        import mujoco

        cam = env._make_camera()
        assert isinstance(cam, mujoco.MjvCamera)
        assert cam.distance == env._camera_distance
        assert cam.azimuth == env._camera_azimuth
        assert cam.elevation == env._camera_elevation


class TestRender:
    def test_no_render_mode_returns_none(self, env):
        # env has render_mode=None by default
        env.reset(seed=42)
        result = env.render()
        assert result is None

    def test_human_render_mode_sets_attribute(self, env):
        # Just verify the render_mode attribute is accepted
        e = RaptorEnv(render_mode="human")
        assert e.render_mode == "human"
        e.close()

    def test_rgb_array_mode_sets_attribute(self):
        e = RaptorEnv(render_mode="rgb_array")
        assert e.render_mode == "rgb_array"
        assert e._renderer is None  # Not yet created
        e.close()


class TestClose:
    def test_close_idempotent(self):
        e = RaptorEnv()
        e.close()
        e.close()  # should not raise

    def test_close_clears_viewer_attributes(self):
        e = RaptorEnv()
        # Simulate having a renderer by setting mocks
        mock_renderer = MagicMock()
        e._renderer = mock_renderer
        e.close()
        assert e._renderer is None
        mock_renderer.close.assert_called_once()

    def test_close_clears_viewer(self):
        e = RaptorEnv()
        mock_viewer = MagicMock()
        e._viewer = mock_viewer
        e.close()
        assert e._viewer is None
        mock_viewer.close.assert_called_once()
