"""Species-specific reward tests for Brachiosaurus.

Common reward invariants (alive bonus, energy penalty, approach zero on first
step, zero forward weight) are tested in
environments/shared/tests/test_species_integration.py::TestRewardConsistency.
"""

import numpy as np
import pytest

from environments.brachiosaurus.envs.brachio_env import BrachioEnv


@pytest.fixture
def env():
    e = BrachioEnv()
    yield e
    e.close()


class TestBrachioRewardComponents:
    """Brachiosaurus-specific reward component tests."""

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


class TestBrachioRewardWeightEffects:
    """Verify that changing reward weights affects the output."""

    def test_zero_food_bonus_gives_no_food_reward(self):
        env = BrachioEnv(food_reach_bonus=0.0)
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_food"] == 0.0
        env.close()

    def test_high_alive_bonus_dominates(self):
        env = BrachioEnv(
            alive_bonus=100.0,
            forward_vel_weight=0.0,
            food_approach_weight=0.0,
            food_reach_bonus=0.0,
        )
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, terminated, _, info = env.step(action)
        if not terminated:
            assert info["reward_alive"] == 100.0
        env.close()

    def test_zero_gait_weight_zeroes_gait_reward(self):
        env = BrachioEnv(gait_stability_weight=0.0)
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_gait"] == 0.0
        env.close()

    def test_actuator_count(self):
        """26 actuators: 6 neck + 20 legs (5 per leg: hip pitch/roll, knee, ankle, toe)."""
        env = BrachioEnv()
        assert env.model.nu == 26, f"Expected 26 actuators, got {env.model.nu}"
        assert env.action_space.shape == (26,)
        env.close()


class TestCurriculumStageRewards:
    """Test that reward configs from TOML produce expected behavior."""

    def test_stage1_balance_no_forward_reward(self):
        env = BrachioEnv(
            forward_vel_weight=0.0,
            food_reach_bonus=0.0,
            food_approach_weight=0.0,
        )
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_forward"] == 0.0
        assert info["reward_food"] == 0.0
        env.close()

    def test_stage3_food_reach_has_approach_shaping(self):
        """Stage 3 config enables approach shaping (delta-based)."""
        env = BrachioEnv(food_approach_weight=10.0)
        env.reset(seed=42)
        action = env.action_space.sample()
        env.step(action)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert "approach_delta" in info
        assert "reward_approach" in info
        assert info["approach_delta"] != 0.0
        env.close()
