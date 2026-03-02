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

    def test_approach_reward_zero_on_first_step(self, env):
        """Approach reward should be zero on the first step (no prior distance)."""
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert info["reward_approach"] == 0.0
        assert info["approach_delta"] == 0.0

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
        _, _, terminated, _, info = env.step(action)
        expected = (
            info["reward_forward"]
            + info["reward_alive"]
            + info["reward_energy"]
            + info["reward_tail"]
            + info["reward_bite"]
            + info["reward_approach"]
            + info["reward_posture"]
            + info["reward_nosedive"]
            + info["reward_height"]
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

    def test_height_reward_non_negative(self, env):
        """Height reward should be non-negative (bonus for staying upright)."""
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert info["reward_height"] >= 0.0

    def test_nosedive_penalty_non_positive(self, env):
        """Nosedive penalty should be non-positive."""
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert info["reward_nosedive"] <= 0.0


class TestRewardWeightEffects:
    """Verify that changing reward weights affects the output."""

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

    def test_high_alive_bonus_dominates(self):
        env = TRexEnv(
            alive_bonus=100.0,
            forward_vel_weight=0.0,
            bite_approach_weight=0.0,
            bite_bonus=0.0,
        )
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, terminated, _, info = env.step(action)
        if not terminated:
            assert info["reward_alive"] == 100.0
        env.close()

    def test_zero_posture_weight_zeroes_posture_reward(self):
        env = TRexEnv(posture_weight=0.0)
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_posture"] == 0.0
        env.close()

    def test_zero_gait_weight_zeroes_gait_reward(self):
        env = TRexEnv(gait_symmetry_weight=0.0)
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_gait"] == 0.0
        env.close()

    def test_zero_smoothness_weight_zeroes_smoothness_reward(self):
        env = TRexEnv(smoothness_weight=0.0)
        env.reset(seed=42)
        action = env.action_space.sample()
        env.step(action)
        action2 = env.action_space.sample()
        _, _, _, _, info = env.step(action2)
        assert info["reward_smoothness"] == 0.0
        env.close()

    def test_zero_height_weight_zeroes_height_reward(self):
        env = TRexEnv(height_weight=0.0)
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_height"] == 0.0
        env.close()

    def test_actuator_count(self):
        """21 actuators: 3 neck/head + 14 legs + 4 tail (no arms)."""
        env = TRexEnv()
        assert env.model.nu == 21, f"Expected 21 actuators, got {env.model.nu}"
        assert env.action_space.shape == (21,)
        env.close()


class TestCurriculumStageRewards:
    """Test that reward configs from TOML produce expected behavior."""

    def test_stage1_balance_no_forward_reward(self):
        """Stage 1 config disables forward velocity reward."""
        env = TRexEnv(
            forward_vel_weight=0.0,
            bite_bonus=0.0,
            bite_approach_weight=0.0,
        )
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_forward"] == 0.0
        assert info["reward_bite"] == 0.0
        env.close()

    def test_stage3_bite_has_approach_shaping(self):
        """Stage 3 config enables approach shaping (delta-based)."""
        env = TRexEnv(bite_approach_weight=10.0)
        env.reset(seed=42)
        action = env.action_space.sample()
        env.step(action)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert "approach_delta" in info
        assert "reward_approach" in info
        assert info["approach_delta"] != 0.0
        env.close()
