"""Species-specific reward tests for T-Rex.

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
    assert_heading_reward_bounded,
    assert_nosedive_penalty_non_positive,
    assert_posture_reward_non_positive,
    assert_smoothness_penalty_for_action_change,
    assert_smoothness_zero_on_first_step,
)
from environments.trex.envs.trex_env import TRexEnv


@pytest.fixture
def env():
    e = TRexEnv()
    yield e
    e.close()


class TestTRexRewardComponents:
    """T-Rex-specific reward component tests."""

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
            + info["reward_tail"]
            + info["reward_bite"]
            + info["reward_approach"]
            + info["reward_head_proximity"]
            + info["reward_head_clearance"]
            + info["reward_neck_posture"]
            + info["reward_posture"]
            + info["reward_nosedive"]
            + info["reward_height"]
            + info["reward_bilateral_support"]
            + info["reward_foot_load_balance"]
            + info["reward_leg_home_pose"]
            + info["reward_gait"]
            + info["reward_smoothness"]
            + info["reward_heading"]
            + info["reward_lateral"]
            + info["reward_spin"]
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

    def test_heading_alignment_bounded(self, env):
        assert_heading_reward_bounded(env)

    def test_bite_success_is_zero_initially(self, env):
        """No bite success on the first step (prey is far away)."""
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert info["bite_success"] == 0.0
        assert info["reward_bite"] == 0.0

    def test_height_reward_non_negative(self, env):
        """Height maintenance reward should be non-negative."""
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert info["reward_height"] >= 0.0


class TestTRexRewardWeightEffects:
    """Verify that changing reward weights affects the output."""

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

    def test_zero_head_proximity_weight_zeroes_head_proximity_reward(self):
        env = TRexEnv(bite_head_proximity_weight=0.0)
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_head_proximity"] == 0.0
        env.close()

    def test_positive_head_proximity_weight_gives_reward(self):
        env = TRexEnv(bite_head_proximity_weight=1.0)
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert info["reward_head_proximity"] >= 0.0
        assert info["head_proximity"] >= 0.0
        assert info["head_prey_distance"] >= 0.0
        env.close()

    def test_zero_height_weight_zeroes_height_reward(self):
        env = TRexEnv(height_weight=0.0)
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_height"] == 0.0
        env.close()

    def test_nonzero_height_weight_gives_positive_reward(self):
        """Height reward should be positive when the T-Rex is standing."""
        env = TRexEnv(height_weight=1.0)
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, terminated, _, info = env.step(action)
        if not terminated:
            assert info["reward_height"] > 0.0
        env.close()

    def test_new_stance_rewards_default_to_zero(self):
        env = TRexEnv(reset_noise_scale=0.0)
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert info["reward_bilateral_support"] == 0.0
        assert info["reward_foot_load_balance"] == 0.0
        assert info["reward_leg_home_pose"] == 0.0
        assert info["reward_head_clearance"] == 0.0
        assert info["reward_neck_posture"] == 0.0
        assert info["alive_gate"] == 1.0
        assert info["reward_alive"] == info["raw_alive"]
        env.close()

    def test_zero_spin_weight_zeroes_spin_reward(self):
        env = TRexEnv(spin_penalty_weight=0.0)
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_spin"] == 0.0
        env.close()

    def test_nonzero_spin_weight_gives_nonpositive_reward(self):
        env = TRexEnv(spin_penalty_weight=0.5)
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_spin"] <= 0.0
        assert info["spin_instability"] >= 0.0
        env.close()

    def test_zero_drift_weight_zeroes_drift_reward(self):
        env = TRexEnv(drift_penalty_weight=0.0)
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_drift"] == 0.0
        env.close()

    def test_zero_backward_vel_weight_zeroes_backward_reward(self):
        env = TRexEnv(backward_vel_penalty_weight=0.0)
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_backward"] == 0.0
        env.close()

    def test_actuator_count(self):
        """15 actuators: 3 neck/head + 8 legs + 4 tail (no arms; toes passive)."""
        env = TRexEnv()
        assert env.model.nu == 15, f"Expected 15 actuators, got {env.model.nu}"
        assert env.action_space.shape == (15,)
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

    def test_stage2_locomotion_has_heading_and_gait(self):
        """Stage 2 enables heading alignment and gait symmetry."""
        env = TRexEnv(heading_weight=0.5, gait_symmetry_weight=0.3)
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert "reward_heading" in info
        assert "reward_gait" in info
        env.close()


class TestStageOneStanceRewardPrimitives:
    """Focused coverage for the opt-in Stage-1 stance signals."""

    @staticmethod
    def _reward_info(env):
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, info = env._get_reward_info(action)
        return info

    def test_bilateral_support_is_bounded_and_uses_weaker_foot(self):
        env = TRexEnv(
            bilateral_support_weight=2.0,
            foot_contact_saturation_force=100.0,
            reset_noise_scale=0.0,
        )
        try:
            env.reset(seed=0)
            env._foot_contact_forces = lambda: (50.0, 250.0)
            info = self._reward_info(env)
            assert info["bilateral_support_quality"] == pytest.approx(0.5)
            assert info["reward_bilateral_support"] == pytest.approx(1.0)

            env._foot_contact_forces = lambda: (1_000.0, 1_000.0)
            info = self._reward_info(env)
            assert info["bilateral_support_quality"] == pytest.approx(1.0)
            assert info["reward_bilateral_support"] == pytest.approx(2.0)
        finally:
            env.close()

    @pytest.mark.parametrize(
        ("forces", "expected_imbalance"),
        [
            ((100.0, 100.0), 0.0),
            ((100.0, 300.0), 0.5),
            ((0.0, 100.0), 1.0),
            # Airborne is MAXIMALLY imbalanced, not perfectly balanced. This
            # case asserted 0.0 until the arithmetic hole was closed, which
            # made flight (0.000) cheaper than honest single support (-0.300)
            # on the stage whose entire job is to stand still.
            # See docs/STAGE1_SPLIT_PLAN.md section 7.1.
            ((0.0, 0.0), 1.0),
        ],
    )
    def test_foot_load_balance_is_normalized(self, forces, expected_imbalance):
        env = TRexEnv(foot_load_balance_weight=2.0, reset_noise_scale=0.0)
        try:
            env.reset(seed=0)
            env._foot_contact_forces = lambda: forces
            info = self._reward_info(env)
            assert info["foot_load_imbalance"] == pytest.approx(expected_imbalance)
            assert info["reward_foot_load_balance"] == pytest.approx(-2.0 * expected_imbalance)
        finally:
            env.close()

    def test_alive_bonus_can_be_partially_conditioned_on_support(self):
        env = TRexEnv(
            alive_bonus=2.0,
            support_conditioned_alive_fraction=0.25,
            foot_contact_saturation_force=100.0,
            reset_noise_scale=0.0,
        )
        try:
            env.reset(seed=0)
            env._foot_contact_forces = lambda: (50.0, 200.0)
            info = self._reward_info(env)
            assert info["raw_alive"] == pytest.approx(2.0)
            assert info["bilateral_support_quality"] == pytest.approx(0.5)
            assert info["alive_gate"] == pytest.approx(0.875)
            assert info["reward_alive"] == pytest.approx(1.75)
        finally:
            env.close()

    def test_leg_home_pose_is_soft_and_uses_authored_keyframe(self):
        tolerance = 0.35
        env = TRexEnv(
            leg_home_pose_weight=3.0,
            leg_home_pose_tolerance=tolerance,
            reset_noise_scale=0.0,
        )
        try:
            env.reset(seed=0)
            home_info = self._reward_info(env)
            assert home_info["leg_home_pose_error"] == pytest.approx(0.0)
            assert home_info["leg_home_pose_quality"] == pytest.approx(1.0)
            assert home_info["reward_leg_home_pose"] == pytest.approx(3.0)

            env.data.qpos[env._leg_home_qpos_indices[0]] += tolerance
            displaced_info = self._reward_info(env)
            expected_quality = (7.0 + np.exp(-1.0)) / 8.0
            assert displaced_info["leg_home_pose_error"] == pytest.approx(tolerance / np.sqrt(8.0))
            assert displaced_info["leg_home_pose_quality"] == pytest.approx(expected_quality)
            assert 0.0 < displaced_info["reward_leg_home_pose"] < 3.0
        finally:
            env.close()

    def test_neck_posture_is_soft_and_uses_authored_keyframe(self):
        tolerance = 0.25
        env = TRexEnv(
            neck_posture_weight=2.0,
            neck_posture_tolerance=tolerance,
            reset_noise_scale=0.0,
        )
        try:
            env.reset(seed=0)
            home_info = self._reward_info(env)
            assert home_info["neck_posture_error"] == pytest.approx(0.0)
            assert home_info["neck_posture_quality"] == pytest.approx(1.0)

            env.data.qpos[env._neck_home_qpos_indices[0]] += tolerance
            displaced_info = self._reward_info(env)
            expected_quality = (2.0 + np.exp(-1.0)) / 3.0
            assert displaced_info["neck_posture_error"] == pytest.approx(tolerance / np.sqrt(3.0))
            assert displaced_info["neck_posture_quality"] == pytest.approx(expected_quality)
            assert 0.0 < displaced_info["reward_neck_posture"] < 2.0
        finally:
            env.close()

    def test_head_clearance_is_smooth_and_saturates_at_target(self):
        env = TRexEnv(
            head_clearance_target=0.60,
            head_clearance_tolerance=0.48,
        )
        try:
            lower = env.head_clearance_target - env.head_clearance_tolerance
            midpoint = lower + env.head_clearance_tolerance / 2.0
            assert env._head_clearance_quality(lower) == pytest.approx(0.0)
            assert env._head_clearance_quality(midpoint) == pytest.approx(0.5)
            assert env._head_clearance_quality(env.head_clearance_target) == pytest.approx(1.0)
            assert env._head_clearance_quality(10.0) == pytest.approx(1.0)
        finally:
            env.close()

    def test_target_centered_height_is_opt_in(self):
        tolerance = 0.10
        target_z = 0.9260
        legacy = TRexEnv(height_weight=2.0, reset_noise_scale=0.0)
        centered = TRexEnv(
            height_weight=2.0,
            height_target_tolerance=tolerance,
            reset_noise_scale=0.0,
        )
        try:
            legacy.reset(seed=0)
            centered.reset(seed=0)
            legacy.data.xpos[legacy.pelvis_id, 2] = target_z + tolerance
            centered.data.xpos[centered.pelvis_id, 2] = target_z + tolerance

            legacy_info = self._reward_info(legacy)
            centered_info = self._reward_info(centered)
            assert legacy_info["height_error"] == pytest.approx(tolerance)
            assert legacy_info["height_quality"] == pytest.approx(1.0)
            assert centered_info["height_error"] == pytest.approx(tolerance)
            assert centered_info["height_quality"] == pytest.approx(np.exp(-1.0))
            assert centered_info["reward_height"] == pytest.approx(2.0 * np.exp(-1.0))
        finally:
            legacy.close()
            centered.close()

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"height_target_tolerance": -0.1}, "height_target_tolerance"),
            ({"foot_contact_saturation_force": 0.0}, "foot_contact_saturation_force"),
            ({"support_conditioned_alive_fraction": -0.1}, "support_conditioned_alive_fraction"),
            ({"support_conditioned_alive_fraction": 1.1}, "support_conditioned_alive_fraction"),
            ({"leg_home_pose_tolerance": 0.0}, "leg_home_pose_tolerance"),
            ({"head_clearance_tolerance": 0.0}, "head_clearance_tolerance"),
            ({"neck_posture_tolerance": 0.0}, "neck_posture_tolerance"),
        ],
    )
    def test_invalid_stance_settings_fail_fast(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            TRexEnv(**kwargs)
