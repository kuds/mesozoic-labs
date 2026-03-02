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

    def test_gait_reward_non_positive(self, env):
        """Gait stability reward is a penalty (non-positive) for angular velocity."""
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_gait"] <= 0.0
        assert info["gait_instability"] >= 0.0

    def test_pelvis_height_tracked(self, env):
        """pelvis_height (aliased from torso height) should be present."""
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert "pelvis_height" in info
        assert info["pelvis_height"] > 0

    def test_foot_contacts_tracked(self, env):
        """Foot contact sensors should be reported in info."""
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert "r_foot_contact" in info
        assert "l_foot_contact" in info

    def test_head_food_distance_reported(self, env):
        """Head-food distance should be tracked in info."""
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert "head_food_distance" in info
        assert info["head_food_distance"] > 0  # Food starts far away


class TestRewardWeightEffects:
    """Verify that changing reward weights affects the output."""

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
        """22 actuators: 6 neck + 16 legs (4 per leg)."""
        env = BrachioEnv()
        assert env.model.nu == 22, f"Expected 22 actuators, got {env.model.nu}"
        assert env.action_space.shape == (22,)
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
