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
        _, _, _, _, info = env.step(action)
        expected = (
            info["reward_forward"]
            + info["reward_alive"]
            + info["reward_energy"]
            + info["reward_tail"]
            + info["reward_strike"]
            + info["reward_approach"]
        )
        assert abs(info["reward_total"] - expected) < 1e-6


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


class TestCurriculumStageRewards:
    """Test that reward configs from TOML produce expected behavior."""

    def test_stage1_balance_no_forward_reward(self):
        """Stage 1 config disables forward velocity reward."""
        env = RaptorEnv(forward_vel_weight=0.0, strike_bonus=0.0, strike_approach_weight=0.0)
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_forward"] == 0.0
        assert info["reward_strike"] == 0.0
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
