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


# ── distance tracking ────────────────────────────────────────────────────


class TestDistanceTracking:
    """Test cumulative XY distance traveled tracking."""

    @pytest.fixture
    def env(self):
        e = RaptorEnv(max_episode_steps=100)
        yield e
        e.close()

    def test_distance_starts_at_zero(self, env):
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        # First step distance should be very small (near zero with zero action)
        assert "distance_traveled" in info
        assert info["distance_traveled"] >= 0.0

    def test_distance_is_non_negative(self, env):
        env.reset(seed=42)
        action = env.action_space.sample()
        for _ in range(10):
            _, _, terminated, _, info = env.step(action)
            assert info["distance_traveled"] >= 0.0
            if terminated:
                break

    def test_distance_is_monotonically_increasing(self, env):
        env.reset(seed=42)
        action = env.action_space.sample()
        prev_dist = 0.0
        for _ in range(10):
            _, _, terminated, _, info = env.step(action)
            assert info["distance_traveled"] >= prev_dist
            prev_dist = info["distance_traveled"]
            if terminated:
                break

    def test_distance_resets_on_reset(self, env):
        env.reset(seed=42)
        action = env.action_space.sample()
        for _ in range(5):
            _, _, terminated, _, _ = env.step(action)
            if terminated:
                break
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        # After reset, distance should be near zero again
        assert info["distance_traveled"] < 0.1


# ── gait symmetry ────────────────────────────────────────────────────────


class TestGaitSymmetry:
    """Test the shared gait symmetry helper on BaseDinoEnv."""

    @pytest.fixture
    def env(self):
        e = RaptorEnv(gait_symmetry_weight=1.0)
        e.reset(seed=42)
        yield e
        e.close()

    def test_no_touchdowns_gives_zero(self, env):
        """With no foot contacts, alternation ratio should be 0."""
        reward, ratio = env._compute_gait_symmetry(0.0, 0.0, 1.0)
        assert ratio == 0.0
        assert reward == 0.0

    def test_single_touchdown_gives_zero(self, env):
        """A single touchdown cannot produce alternation."""
        # First call: no contact
        env._compute_gait_symmetry(0.0, 0.0, 1.0)
        # Second call: right foot lands (off→on)
        reward, ratio = env._compute_gait_symmetry(1.0, 0.0, 1.0)
        assert ratio == 0.0

    def test_perfect_alternation(self, env):
        """L→R→L should produce alternation_ratio = 1.0."""
        # Start: no contact
        env._compute_gait_symmetry(0.0, 0.0, 1.0)
        # Left touchdown
        env._compute_gait_symmetry(0.0, 1.0, 1.0)
        # Left lifts, right lands
        env._compute_gait_symmetry(1.0, 0.0, 1.0)
        # Right lifts, left lands
        reward, ratio = env._compute_gait_symmetry(0.0, 1.0, 1.0)
        assert ratio == pytest.approx(1.0)
        assert reward == pytest.approx(1.0)

    def test_same_foot_repeated(self, env):
        """R→R→R (same foot) should produce alternation_ratio = 0.0."""
        # Start: no contact
        env._compute_gait_symmetry(0.0, 0.0, 1.0)
        # Right touchdown
        env._compute_gait_symmetry(1.0, 0.0, 1.0)
        # Right lifts
        env._compute_gait_symmetry(0.0, 0.0, 1.0)
        # Right lands again
        env._compute_gait_symmetry(1.0, 0.0, 1.0)
        # Right lifts
        env._compute_gait_symmetry(0.0, 0.0, 1.0)
        # Right lands again
        reward, ratio = env._compute_gait_symmetry(1.0, 0.0, 1.0)
        assert ratio == pytest.approx(0.0)
        assert reward == pytest.approx(0.0)

    def test_weight_scales_reward(self, env):
        """Reward should scale linearly with weight."""
        # Perfect alternation
        env._compute_gait_symmetry(0.0, 0.0, 1.0)
        env._compute_gait_symmetry(0.0, 1.0, 1.0)
        env._compute_gait_symmetry(1.0, 0.0, 1.0)
        _, ratio = env._compute_gait_symmetry(0.0, 1.0, 1.0)
        # Reset and redo with weight=0.5
        env._reset_gait_state()
        env._compute_gait_symmetry(0.0, 0.0, 0.5)
        env._compute_gait_symmetry(0.0, 1.0, 0.5)
        env._compute_gait_symmetry(1.0, 0.0, 0.5)
        reward, _ = env._compute_gait_symmetry(0.0, 1.0, 0.5)
        assert reward == pytest.approx(0.5 * ratio)

    def test_reset_clears_state(self, env):
        """After _reset_gait_state, history should be empty."""
        env._compute_gait_symmetry(0.0, 1.0, 1.0)
        env._compute_gait_symmetry(1.0, 0.0, 1.0)
        env._reset_gait_state()
        reward, ratio = env._compute_gait_symmetry(0.0, 0.0, 1.0)
        assert ratio == 0.0

    def test_trex_uses_shared_helper(self):
        """TRexEnv should use the shared gait symmetry (not the old buggy version)."""
        from environments.trex.envs.trex_env import TRexEnv

        env = TRexEnv(gait_symmetry_weight=1.0)
        env.reset(seed=42)
        # Verify gait state was initialised
        assert hasattr(env, "_touchdown_sequence")
        assert env._touchdown_sequence == []
        env.close()


class TestQuadrupedGaitSymmetry:
    """Test the quadrupedal diagonal pair gait symmetry helper."""

    @pytest.fixture
    def env(self):
        from environments.brachiosaurus.envs.brachio_env import BrachioEnv

        e = BrachioEnv(gait_symmetry_weight=1.0)
        e.reset(seed=42)
        yield e
        e.close()

    def test_no_touchdowns_gives_zero(self, env):
        """With no foot contacts, alternation ratio should be 0."""
        reward, ratio = env._compute_quadruped_gait_symmetry(0.0, 0.0, 0.0, 0.0, 1.0)
        assert ratio == 0.0
        assert reward == 0.0

    def test_single_diagonal_touchdown_gives_zero(self, env):
        """A single diagonal touchdown cannot produce alternation."""
        env._compute_quadruped_gait_symmetry(0.0, 0.0, 0.0, 0.0, 1.0)
        # Diagonal A (FR + RL) lands
        reward, ratio = env._compute_quadruped_gait_symmetry(1.0, 0.0, 0.0, 1.0, 1.0)
        assert ratio == 0.0

    def test_perfect_diagonal_alternation(self, env):
        """A→B→A should produce alternation_ratio = 1.0."""
        # Start: no contact
        env._compute_quadruped_gait_symmetry(0.0, 0.0, 0.0, 0.0, 1.0)
        # Diagonal A: FR + RL land
        env._compute_quadruped_gait_symmetry(1.0, 0.0, 0.0, 1.0, 1.0)
        # All off
        env._compute_quadruped_gait_symmetry(0.0, 0.0, 0.0, 0.0, 1.0)
        # Diagonal B: FL + RR land
        env._compute_quadruped_gait_symmetry(0.0, 1.0, 1.0, 0.0, 1.0)
        # All off
        env._compute_quadruped_gait_symmetry(0.0, 0.0, 0.0, 0.0, 1.0)
        # Diagonal A again
        reward, ratio = env._compute_quadruped_gait_symmetry(1.0, 0.0, 0.0, 1.0, 1.0)
        assert ratio == pytest.approx(1.0)
        assert reward == pytest.approx(1.0)

    def test_same_diagonal_repeated(self, env):
        """A→A→A (same diagonal) should produce alternation_ratio = 0.0."""
        env._compute_quadruped_gait_symmetry(0.0, 0.0, 0.0, 0.0, 1.0)
        # Diagonal A lands
        env._compute_quadruped_gait_symmetry(1.0, 0.0, 0.0, 1.0, 1.0)
        env._compute_quadruped_gait_symmetry(0.0, 0.0, 0.0, 0.0, 1.0)
        # Diagonal A again
        env._compute_quadruped_gait_symmetry(1.0, 0.0, 0.0, 1.0, 1.0)
        env._compute_quadruped_gait_symmetry(0.0, 0.0, 0.0, 0.0, 1.0)
        # Diagonal A again
        reward, ratio = env._compute_quadruped_gait_symmetry(1.0, 0.0, 0.0, 1.0, 1.0)
        assert ratio == pytest.approx(0.0)
        assert reward == pytest.approx(0.0)

    def test_partial_diagonal_triggers_pair(self, env):
        """A single foot from a diagonal pair should trigger that pair."""
        env._compute_quadruped_gait_symmetry(0.0, 0.0, 0.0, 0.0, 1.0)
        # Only FR lands (partial diagonal A)
        env._compute_quadruped_gait_symmetry(1.0, 0.0, 0.0, 0.0, 1.0)
        env._compute_quadruped_gait_symmetry(0.0, 0.0, 0.0, 0.0, 1.0)
        # Only FL lands (partial diagonal B)
        reward, ratio = env._compute_quadruped_gait_symmetry(0.0, 1.0, 0.0, 0.0, 1.0)
        assert ratio == pytest.approx(1.0)

    def test_weight_scales_reward(self, env):
        """Reward should scale linearly with weight."""
        env._compute_quadruped_gait_symmetry(0.0, 0.0, 0.0, 0.0, 1.0)
        env._compute_quadruped_gait_symmetry(1.0, 0.0, 0.0, 1.0, 1.0)
        env._compute_quadruped_gait_symmetry(0.0, 0.0, 0.0, 0.0, 1.0)
        _, ratio = env._compute_quadruped_gait_symmetry(0.0, 1.0, 1.0, 0.0, 1.0)
        # Reset and redo with weight=0.5
        env._reset_quadruped_gait_state()
        env._compute_quadruped_gait_symmetry(0.0, 0.0, 0.0, 0.0, 0.5)
        env._compute_quadruped_gait_symmetry(1.0, 0.0, 0.0, 1.0, 0.5)
        env._compute_quadruped_gait_symmetry(0.0, 0.0, 0.0, 0.0, 0.5)
        reward, _ = env._compute_quadruped_gait_symmetry(0.0, 1.0, 1.0, 0.0, 0.5)
        assert reward == pytest.approx(0.5 * ratio)

    def test_reset_clears_state(self, env):
        """After _reset_quadruped_gait_state, history should be empty."""
        env._compute_quadruped_gait_symmetry(1.0, 0.0, 0.0, 1.0, 1.0)
        env._compute_quadruped_gait_symmetry(0.0, 1.0, 1.0, 0.0, 1.0)
        env._reset_quadruped_gait_state()
        reward, ratio = env._compute_quadruped_gait_symmetry(0.0, 0.0, 0.0, 0.0, 1.0)
        assert ratio == 0.0

    def test_brachio_uses_quadruped_helper(self, env):
        """BrachioEnv should use quadrupedal gait state."""
        assert hasattr(env, "_quad_touchdown_sequence")
        assert env._quad_touchdown_sequence == []

    def test_brachio_info_has_all_foot_contacts(self, env):
        """BrachioEnv step info should include all 4 foot contacts."""
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert "r_foot_contact" in info
        assert "l_foot_contact" in info
        assert "rr_foot_contact" in info
        assert "rl_foot_contact" in info
