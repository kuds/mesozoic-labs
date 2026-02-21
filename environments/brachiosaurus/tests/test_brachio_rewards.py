"""Tests for Brachiosaurus reward function behavior."""

import numpy as np
import pytest

from environments.brachiosaurus.envs.brachio_env import BrachioEnv


@pytest.fixture
def env():
    e = BrachioEnv()
    yield e
    e.close()


class TestRewardComponents:
    """Verify individual reward components produce expected values."""

    def test_alive_bonus_is_positive(self, env):
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert info["reward_alive"] > 0

    def test_energy_penalty_zero_for_zero_action(self, env):
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert info["reward_energy"] == 0.0

    def test_energy_penalty_negative_for_large_action(self, env):
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

    def test_food_not_reached_initially(self, env):
        """Food is far away on first step."""
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert info["food_reached"] == 0.0
        assert info["reward_food"] == 0.0

    def test_total_reward_is_sum_of_components(self, env):
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, terminated, _, info = env.step(action)
        expected = (
            info["reward_forward"]
            + info["reward_alive"]
            + info["reward_energy"]
            + info["reward_gait"]
            + info["reward_food"]
            + info["reward_approach"]
        )
        if terminated:
            expected += env.fall_penalty
        assert abs(info["reward_total"] - expected) < 1e-6


class TestRewardWeightEffects:
    def test_zero_forward_weight_zeroes_forward_reward(self):
        env = BrachioEnv(forward_vel_weight=0.0)
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_forward"] == 0.0
        env.close()

    def test_zero_food_bonus_gives_no_food_reward(self):
        env = BrachioEnv(food_reach_bonus=0.0)
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_food"] == 0.0
        env.close()
