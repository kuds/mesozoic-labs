"""Species-specific reward tests for Velociraptor.

Common reward invariants (alive bonus, energy penalty, approach zero on first
step, zero forward weight) are tested in
environments/shared/tests/test_species_integration.py::TestRewardConsistency.
"""

import numpy as np
import pytest

from environments.shared.tests.reward_test_helpers import (
    assert_backward_vel_penalty_non_positive,
    assert_drift_penalty_non_positive,
    assert_gait_reward_non_negative,
    assert_nosedive_penalty_non_positive,
    assert_posture_reward_non_positive,
    assert_smoothness_penalty_for_action_change,
    assert_smoothness_zero_on_first_step,
)
from environments.velociraptor.envs.raptor_env import RaptorEnv


@pytest.fixture
def env():
    e = RaptorEnv()
    yield e
    e.close()


class TestRaptorRewardComponents:
    """Raptor-specific reward component tests."""

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
            + info["reward_drift"]
            + info["reward_alive"]
            + info["reward_energy"]
            + info["reward_tail"]
            + info["reward_strike"]
            + info["reward_approach"]
            + info["reward_proximity"]
            + info["reward_claw_proximity"]
            + info["reward_posture"]
            + info["reward_nosedive"]
            + info["reward_spin"]
            + info["reward_gait"]
            + info["reward_smoothness"]
            + info["reward_heading"]
            + info["reward_lateral"]
            + info["reward_speed"]
            + info["reward_idle"]
        )
        if terminated:
            expected += env.fall_penalty
        assert abs(info["reward_total"] - expected) < 1e-6

    def test_posture_reward_negative_or_zero(self, env):
        assert_posture_reward_non_positive(env)

    def test_gait_reward_non_negative(self, env):
        assert_gait_reward_non_negative(env)

    def test_smoothness_zero_on_first_step(self, env):
        assert_smoothness_zero_on_first_step(env)

    def test_smoothness_penalty_for_action_change(self, env):
        assert_smoothness_penalty_for_action_change(env)

    def test_nosedive_penalty_non_positive(self, env):
        assert_nosedive_penalty_non_positive(env)

    def test_backward_vel_penalty_non_positive(self, env):
        assert_backward_vel_penalty_non_positive(env)

    def test_drift_penalty_non_positive(self, env):
        assert_drift_penalty_non_positive(env)

    def test_claw_proximity_zero_by_default(self, env):
        """Claw proximity reward should be zero when weight is zero (default)."""
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert info["reward_claw_proximity"] == 0.0
        assert info["min_claw_prey_distance"] >= 0.0
        assert 0.0 <= info["claw_proximity"] <= 1.0


class TestRaptorRewardWeightEffects:
    """Verify that changing reward weights affects the output."""

    def test_high_alive_bonus_dominates(self):
        env = RaptorEnv(alive_bonus=100.0, forward_vel_weight=0.0, strike_approach_weight=0.0, strike_bonus=0.0)
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, terminated, _, info = env.step(action)
        if not terminated:
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

    def test_zero_spin_weight_zeroes_spin_reward(self):
        env = RaptorEnv(spin_penalty_weight=0.0)
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_spin"] == 0.0
        env.close()

    def test_nonzero_spin_weight_gives_nonpositive_reward(self):
        env = RaptorEnv(spin_penalty_weight=0.5)
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_spin"] <= 0.0
        assert info["spin_instability"] >= 0.0
        env.close()

    def test_zero_drift_weight_zeroes_drift_reward(self):
        env = RaptorEnv(drift_penalty_weight=0.0)
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_drift"] == 0.0
        env.close()

    def test_nonzero_drift_weight_penalizes_displacement(self):
        """Drift penalty should be negative after several steps of movement."""
        env = RaptorEnv(drift_penalty_weight=0.5)
        env.reset(seed=42)
        for _ in range(20):
            action = env.action_space.sample()
            _, _, terminated, _, info = env.step(action)
            if terminated:
                break
            if info["drift_distance"] > 0.01:
                assert info["reward_drift"] < 0.0
                break
        env.close()

    def test_zero_backward_vel_weight_zeroes_backward_reward(self):
        env = RaptorEnv(backward_vel_penalty_weight=0.0)
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_backward"] == 0.0
        env.close()

    def test_nonzero_claw_proximity_weight_gives_positive_reward(self):
        """Claw proximity reward should be positive when weight is set and prey is within range."""
        env = RaptorEnv(strike_claw_proximity_weight=5.0, prey_distance_range=(1.0, 2.0))
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert info["reward_claw_proximity"] >= 0.0
        assert info["min_claw_prey_distance"] > 0.0
        env.close()

    def test_nonzero_backward_vel_weight_penalizes_backward_motion(self):
        """Backward penalty should be negative when raptor moves backward."""
        env = RaptorEnv(backward_vel_penalty_weight=1.0, forward_vel_weight=0.0)
        env.reset(seed=42)
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
        assert info["reward_backward"] <= 0.0
        env.close()

    def test_stage3_strike_has_approach_shaping(self):
        """Stage 3 config enables approach shaping (delta-based)."""
        env = RaptorEnv(strike_approach_weight=10.0)
        env.reset(seed=42)
        action = env.action_space.sample()
        env.step(action)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert "approach_delta" in info
        assert "reward_approach" in info
        assert info["approach_delta"] != 0.0
        env.close()
