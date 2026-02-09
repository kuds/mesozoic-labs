"""Tests for T-Rex reward function behavior."""

import numpy as np
import pytest

from environments.trex.envs.trex_env import TRexEnv


@pytest.fixture
def env():
    e = TRexEnv()
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

    def test_approach_reward_negative(self, env):
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert info["reward_approach"] <= 0

    def test_bite_not_triggered_initially(self, env):
        """Prey is far away on first step."""
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert info["bite_success"] == 0.0
        assert info["reward_bite"] == 0.0

    def test_total_reward_is_sum_of_components(self, env):
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        expected = (
            info["reward_forward"]
            + info["reward_alive"]
            + info["reward_energy"]
            + info["reward_tail"]
            + info["reward_bite"]
            + info["reward_approach"]
        )
        assert abs(info["reward_total"] - expected) < 1e-6


class TestRewardWeightEffects:

    def test_zero_forward_weight_zeroes_forward_reward(self):
        env = TRexEnv(forward_vel_weight=0.0)
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_forward"] == 0.0
        env.close()

    def test_zero_bite_bonus_gives_no_bite_reward(self):
        env = TRexEnv(bite_bonus=0.0)
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_bite"] == 0.0
        env.close()
