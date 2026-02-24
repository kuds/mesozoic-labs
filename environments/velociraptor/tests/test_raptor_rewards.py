"""Tests for Velociraptor reward function behavior."""

import numpy as np
import pytest

from environments.velociraptor.envs.raptor_env import RaptorEnv


@pytest.fixture
def env():
    e = RaptorEnv()
    yield e
    e.close()


class TestRewardComponents:
    """Verify individual reward components produce expected values."""

    def test_alive_bonus_is_positive(self, env):
        """Alive bonus should always be positive when not terminated."""
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert info["reward_alive"] > 0

    def test_energy_penalty_zero_for_zero_action(self, env):
        """Energy penalty should be zero when no action is applied."""
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert info["reward_energy"] == 0.0

    def test_energy_penalty_negative_for_large_action(self, env):
        """Energy penalty should be negative for non-zero actions."""
        env.reset(seed=42)
        action = np.ones(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert info["reward_energy"] < 0

    def test_approach_reward_zero_on_first_step(self, env):
        """Approach reward should be zero on the first step (no prior distance)."""
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        # First step has no prior distance, so approach delta is zero
        assert info["reward_approach"] == 0.0
        assert info["approach_delta"] == 0.0

    def test_strike_success_is_zero_initially(self, env):
        """No strike success on the first step (prey is far away)."""
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert info["strike_success"] == 0.0
        assert info["reward_strike"] == 0.0

    def test_total_reward_is_sum_of_components(self, env):
        """Total reward should equal sum of all components."""
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, terminated, _, info = env.step(action)
        expected = (
            info["reward_forward"]
            + info["reward_backward"]
            + info["reward_alive"]
            + info["reward_energy"]
            + info["reward_tail"]
            + info["reward_strike"]
            + info["reward_approach"]
            + info["reward_posture"]
            + info["reward_nosedive"]
            + info["reward_gait"]
            + info["reward_smoothness"]
            + info["reward_heading"]
            + info["reward_lateral"]
        )
        if terminated:
            expected += env.fall_penalty
        assert abs(info["reward_total"] - expected) < 1e-6

    def test_posture_reward_negative_or_zero(self, env):
        """Posture reward should be non-positive (penalty for tilt)."""
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert info["reward_posture"] <= 0.0
        assert info["tilt_angle"] >= 0.0

    def test_gait_reward_non_negative(self, env):
        """Gait symmetry reward should be non-negative."""
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_gait"] >= 0.0
        assert 0.0 <= info["contact_asymmetry"] <= 1.0

    def test_smoothness_zero_on_first_step(self, env):
        """Smoothness penalty should be zero on first step (no prior action)."""
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_smoothness"] == 0.0
        assert info["action_delta"] == 0.0

    def test_smoothness_penalty_for_action_change(self, env):
        """Smoothness penalty should be negative when action changes between steps."""
        env.reset(seed=42)
        action1 = np.ones(env.action_space.shape, dtype=np.float32)
        env.step(action1)
        action2 = -np.ones(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action2)
        assert info["reward_smoothness"] < 0.0
        assert info["action_delta"] > 0.0

    def test_backward_vel_penalty_non_positive(self, env):
        """Backward velocity penalty should be non-positive."""
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_backward"] <= 0.0
        assert info["backward_vel"] >= 0.0


class TestRewardWeightEffects:
    """Verify that changing reward weights affects the output."""

    def test_zero_forward_weight_zeroes_forward_reward(self):
        env = RaptorEnv(forward_vel_weight=0.0)
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_forward"] == 0.0
        env.close()

    def test_high_alive_bonus_dominates(self):
        env = RaptorEnv(alive_bonus=100.0, forward_vel_weight=0.0, strike_approach_weight=0.0, strike_bonus=0.0)
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, terminated, _, info = env.step(action)
        if not terminated:
            # With all other weights zero/tiny, alive bonus should dominate
            assert info["reward_alive"] == 100.0
        env.close()

    def test_zero_strike_bonus_gives_no_strike_reward(self):
        env = RaptorEnv(strike_bonus=0.0)
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_strike"] == 0.0
        env.close()

    def test_zero_posture_weight_zeroes_posture_reward(self):
        env = RaptorEnv(posture_weight=0.0)
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_posture"] == 0.0
        env.close()

    def test_zero_gait_weight_zeroes_gait_reward(self):
        env = RaptorEnv(gait_symmetry_weight=0.0)
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_gait"] == 0.0
        env.close()

    def test_zero_smoothness_weight_zeroes_smoothness_reward(self):
        env = RaptorEnv(smoothness_weight=0.0)
        env.reset(seed=42)
        action = env.action_space.sample()
        env.step(action)
        action2 = env.action_space.sample()
        _, _, _, _, info = env.step(action2)
        assert info["reward_smoothness"] == 0.0
        env.close()

    def test_zero_backward_vel_weight_zeroes_backward_reward(self):
        env = RaptorEnv(backward_vel_penalty_weight=0.0)
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_backward"] == 0.0
        env.close()

    def test_nonzero_backward_vel_weight_penalizes_backward_motion(self):
        """Backward penalty should be negative when raptor moves backward."""
        env = RaptorEnv(backward_vel_penalty_weight=1.0, forward_vel_weight=0.0)
        env.reset(seed=42)
        # Run several steps with random actions to induce some velocity
        for _ in range(10):
            action = env.action_space.sample()
            _, _, terminated, _, info = env.step(action)
            if terminated:
                break
            if info["backward_vel"] > 0:
                assert info["reward_backward"] < 0.0
                break
        env.close()

    def test_actuator_count(self):
        """All actuators should be enabled (22 total: 14 legs + 4 tail + 4 arms)."""
        env = RaptorEnv()
        assert env.model.nu == 22, f"Expected 22 actuators, got {env.model.nu}"
        assert env.action_space.shape == (22,)
        env.close()


class TestCurriculumStageRewards:
    """Test that reward configs from TOML produce expected behavior."""

    def test_stage1_balance_no_forward_reward(self):
        """Stage 1 config disables forward velocity reward but penalizes backward drift."""
        env = RaptorEnv(
            forward_vel_weight=0.0,
            backward_vel_penalty_weight=0.5,
            strike_bonus=0.0,
            strike_approach_weight=0.0,
        )
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_forward"] == 0.0
        assert info["reward_strike"] == 0.0
        # Backward penalty should be active (non-positive)
        assert info["reward_backward"] <= 0.0
        env.close()

    def test_stage3_strike_has_approach_shaping(self):
        """Stage 3 config enables approach shaping (delta-based)."""
        env = RaptorEnv(strike_approach_weight=10.0)
        env.reset(seed=42)
        # First step initialises the previous distance (delta is zero)
        action = env.action_space.sample()
        env.step(action)
        # Second step should produce a non-zero approach delta from movement
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert "approach_delta" in info
        assert "reward_approach" in info
        # With a random action the raptor moves, so delta should be non-zero
        assert info["approach_delta"] != 0.0
        env.close()
