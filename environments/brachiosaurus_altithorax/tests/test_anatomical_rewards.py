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
            + info["reward_backward"]
            + info["reward_drift"]
            + info["reward_alive"]
            + info["reward_energy"]
            + info["reward_gait"]
            + info["reward_tail"]
            + info["reward_posture"]
            + info["reward_nosedive"]
            + info["reward_height"]
            + info["reward_gait_symmetry"]
            + info["reward_smoothness"]
            + info["reward_heading"]
            + info["reward_lateral"]
            + info["reward_spin"]
            + info["reward_speed"]
            + info["reward_idle"]
            + info["reward_food"]
            + info["reward_approach"]
            + info["reward_head_proximity"]
        )
        if terminated:
            expected += env.fall_penalty
        assert abs(info["reward_total"] - expected) < 1e-6

    def test_gait_reward_non_positive(self, env):
        """Brachio gait reward is an angular velocity penalty (always <= 0)."""
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_gait"] <= 0.0

    def test_food_reached_is_zero_initially(self, env):
        """Food should not be reached on the first step (food is far away)."""
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert info["food_reached"] == 0.0
        assert info["reward_food"] == 0.0

    def test_gait_instability_non_negative(self, env):
        """Gait instability metric should be non-negative."""
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["gait_instability"] >= 0.0

    def test_speed_penalty_non_positive(self, env):
        """Speed penalty should be non-positive."""
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_speed"] <= 0.0


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

    def test_nonzero_gait_weight_gives_nonpositive_reward(self):
        """Gait stability penalty should be non-positive when weight > 0."""
        env = BrachioEnv(gait_stability_weight=0.5)
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_gait"] <= 0.0
        env.close()

    def test_zero_speed_weight_zeroes_speed_reward(self):
        env = BrachioEnv(speed_penalty_weight=0.0)
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_speed"] == 0.0
        env.close()

    def test_food_reach_threshold_respected(self):
        """Food reach should only trigger within threshold distance."""
        env = BrachioEnv(food_reach_threshold=0.001)
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        # With a tiny threshold, food should not be reached
        assert info["food_reached"] == 0.0
        env.close()

    def test_actuator_count(self):
        """30 actuators: 6 neck + 20 legs (5 per leg) + 4 tail (pitch/yaw on tail_1, pitch on tail_2/3)."""
        env = BrachioEnv()
        assert env.model.nu == 30, f"Expected 30 actuators, got {env.model.nu}"
        assert env.action_space.shape == (30,)
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
