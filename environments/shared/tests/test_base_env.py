import numpy as np
import pytest

from environments.shared.base_env import BaseDinoEnv
from environments.velociraptor.envs.raptor_env import RaptorEnv


def test_base_env_lifecycle():
    env = RaptorEnv(render_mode="rgb_array")

    # Test reset
    obs, info = env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)

    # Test step
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)

    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)
    assert isinstance(info, dict)

    # Test render (may fail in headless environments without GPU/OpenGL)
    try:
        frame = env.render()
        assert frame is not None
        assert isinstance(frame, np.ndarray)
    except Exception:
        # Rendering requires a valid OpenGL context which may not be
        # available in CI or headless environments.
        pass

    # Test close
    env.close()


def test_base_env_scale_action():
    env = RaptorEnv()

    # Test action scaling
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    scaled = env._scale_action(action)
    assert scaled.shape == action.shape

    env.close()


# ── _quat_to_tilt ────────────────────────────────────────────────────────


class TestQuatToTilt:
    """Test the quaternion-to-tilt-angle static method on BaseDinoEnv."""

    def test_upright_quaternion_is_zero_tilt(self):
        """Identity quaternion (w=1, x=0, y=0, z=0) means perfectly upright."""
        tilt = BaseDinoEnv._quat_to_tilt(np.array([1.0, 0.0, 0.0, 0.0]))
        assert tilt == pytest.approx(0.0, abs=1e-6)

    def test_90_degree_pitch(self):
        """Quaternion representing 90-degree pitch forward around Y axis."""
        # q = [cos(45°), 0, sin(45°), 0] = [0.7071, 0, 0.7071, 0]
        angle = np.pi / 2
        quat = np.array([np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0])
        tilt = BaseDinoEnv._quat_to_tilt(quat)
        assert tilt == pytest.approx(np.pi / 2, abs=0.01)

    def test_90_degree_roll(self):
        """Quaternion representing 90-degree roll around X axis."""
        angle = np.pi / 2
        quat = np.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0])
        tilt = BaseDinoEnv._quat_to_tilt(quat)
        assert tilt == pytest.approx(np.pi / 2, abs=0.01)

    def test_upside_down(self):
        """180-degree flip: body Z-axis points downward."""
        # Rotation of pi around X axis: q = [0, 1, 0, 0]
        quat = np.array([0.0, 1.0, 0.0, 0.0])
        tilt = BaseDinoEnv._quat_to_tilt(quat)
        assert tilt == pytest.approx(np.pi, abs=0.01)

    def test_small_tilt(self):
        """A small tilt angle should be close to the rotation angle."""
        angle = 0.1  # ~5.7 degrees
        quat = np.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0])
        tilt = BaseDinoEnv._quat_to_tilt(quat)
        assert tilt == pytest.approx(angle, abs=0.01)

    def test_yaw_only_is_zero_tilt(self):
        """Pure yaw rotation (around Z) should produce zero tilt."""
        angle = np.pi / 4
        quat = np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])
        tilt = BaseDinoEnv._quat_to_tilt(quat)
        assert tilt == pytest.approx(0.0, abs=1e-6)

    def test_return_type_is_float(self):
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        assert isinstance(BaseDinoEnv._quat_to_tilt(quat), float)


# ── set_reward_weight ─────────────────────────────────────────────────────


class TestSetRewardWeight:
    """Test dynamic reward weight mutation via set_reward_weight()."""

    @pytest.fixture
    def env(self):
        e = RaptorEnv()
        yield e
        e.close()

    def test_set_existing_weight(self, env):
        env.set_reward_weight("forward_vel_weight", 2.5)
        assert env.forward_vel_weight == 2.5

    def test_set_alive_bonus(self, env):
        env.set_reward_weight("alive_bonus", 0.0)
        assert env.alive_bonus == 0.0

    def test_nonexistent_attribute_raises(self, env):
        with pytest.raises(AttributeError, match="has no attribute"):
            env.set_reward_weight("nonexistent_weight", 1.0)

    def test_set_to_zero(self, env):
        env.set_reward_weight("energy_penalty_weight", 0.0)
        assert env.energy_penalty_weight == 0.0

    def test_set_negative_value(self, env):
        env.set_reward_weight("forward_vel_weight", -1.0)
        assert env.forward_vel_weight == -1.0


# ── truncation ────────────────────────────────────────────────────────────


class TestTruncation:
    """Test episode truncation at max steps."""

    def test_truncates_at_max_steps(self):
        env = RaptorEnv(max_episode_steps=5)
        env.reset(seed=42)
        for i in range(5):
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            _, _, terminated, truncated, _ = env.step(action)
            if terminated:
                break
        # If not already terminated, the 5th step should truncate
        if not terminated:
            assert truncated
        env.close()

    def test_not_truncated_before_max_steps(self):
        env = RaptorEnv(max_episode_steps=1000)
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, truncated, _ = env.step(action)
        assert not truncated
        env.close()


# ── action scaling ────────────────────────────────────────────────────────


class TestActionScaling:
    """Test action normalization from [-1, 1] to actuator ranges."""

    @pytest.fixture
    def env(self):
        e = RaptorEnv()
        yield e
        e.close()

    def test_zero_action_maps_to_midpoint(self, env):
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        scaled = env._scale_action(action)
        ctrl_range = env.model.actuator_ctrlrange
        midpoint = (ctrl_range[:, 0] + ctrl_range[:, 1]) / 2
        np.testing.assert_allclose(scaled, midpoint, atol=1e-6)

    def test_plus_one_maps_to_max(self, env):
        action = np.ones(env.action_space.shape, dtype=np.float32)
        scaled = env._scale_action(action)
        ctrl_max = env.model.actuator_ctrlrange[:, 1]
        np.testing.assert_allclose(scaled, ctrl_max, atol=1e-6)

    def test_minus_one_maps_to_min(self, env):
        action = -np.ones(env.action_space.shape, dtype=np.float32)
        scaled = env._scale_action(action)
        ctrl_min = env.model.actuator_ctrlrange[:, 0]
        np.testing.assert_allclose(scaled, ctrl_min, atol=1e-6)
