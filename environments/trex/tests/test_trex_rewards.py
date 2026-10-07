"""Species-specific reward tests for T-Rex.

Common reward invariants (alive bonus, energy penalty, approach zero on first
step, zero forward weight) are tested in
environments/shared/tests/test_species_integration.py::TestRewardConsistency.
"""

import mujoco
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

#: The stance follow-up's TRexEnv kwargs (decision D-D27), all inert at their defaults.
_FOLLOWUP_KEYS = frozenset(
    {
        "stance_width_reference",
        "stance_width_settle_steps",
        "foot_terms_min_support_force",
        "floor_impact_weight",
        "floor_impact_threshold_bw",
        "airborne_substep_weight",
        "action_penalty_source",
    }
)


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
            + info["reward_foot_flatness"]
            + info["reward_stance_width"]
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
        assert info["reward_foot_flatness"] == 0.0
        assert info["reward_stance_width"] == 0.0
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

    def test_settled_neck_reference_targets_the_statue_not_the_keyframe(self):
        tolerance = 0.25
        env = TRexEnv(
            neck_posture_weight=2.0,
            neck_posture_tolerance=tolerance,
            neck_posture_reference="settled",
            reset_noise_scale=0.0,
        )
        try:
            env.reset(seed=0)
            settled = np.asarray(TRexEnv._NECK_SETTLED_QPOS)
            np.testing.assert_array_equal(env._neck_home_qpos, settled)
            # The noise-free reset holds the keyframe's zeros, a full servo
            # sag away from the settled target.
            keyframe_info = self._reward_info(env)
            assert keyframe_info["neck_posture_error"] == pytest.approx(np.sqrt(np.mean(settled**2)))
            assert keyframe_info["neck_posture_quality"] == pytest.approx(
                np.mean(np.exp(-((settled / tolerance) ** 2)))
            )

            env.data.qpos[env._neck_home_qpos_indices] = settled
            settled_info = self._reward_info(env)
            assert settled_info["neck_posture_error"] == pytest.approx(0.0)
            assert settled_info["neck_posture_quality"] == pytest.approx(1.0)
            assert settled_info["reward_neck_posture"] == pytest.approx(2.0)
        finally:
            env.close()

    def test_foot_flatness_averages_a_gaussian_of_each_pad_tilt(self):
        tolerance = 3.0
        env = TRexEnv(foot_flatness_weight=2.0, foot_flatness_tolerance_deg=tolerance, reset_noise_scale=0.0)
        try:
            env.reset(seed=0)
            level_info = self._reward_info(env)
            assert level_info["r_sole_tilt_deg"] == pytest.approx(0.0, abs=1e-3)
            assert level_info["l_sole_tilt_deg"] == pytest.approx(0.0, abs=1e-3)
            assert level_info["foot_flatness_quality"] == pytest.approx(1.0)
            assert level_info["reward_foot_flatness"] == pytest.approx(2.0)

            # One pad rolled by a tolerance forfeits that foot's share only.
            env._sole_tilts_deg = lambda: np.array([0.0, tolerance])
            rolled_info = self._reward_info(env)
            assert rolled_info["l_sole_tilt_deg"] == pytest.approx(tolerance)
            assert rolled_info["foot_flatness_quality"] == pytest.approx((1.0 + np.exp(-1.0)) / 2.0)
            assert rolled_info["reward_foot_flatness"] == pytest.approx(1.0 + np.exp(-1.0))
        finally:
            env.close()

    @pytest.mark.parametrize("axis", [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], ids=["roll", "pitch"])
    def test_sole_tilt_reads_the_plantar_pad_frames(self, axis):
        """Tipping the whole animal by a known angle tips both level pads by exactly that angle."""
        env = TRexEnv(reset_noise_scale=0.0)
        try:
            env.reset(seed=0)
            angle = np.radians(4.0)
            env.data.qpos[3:7] = [np.cos(angle / 2.0), *(np.sin(angle / 2.0) * np.asarray(axis))]
            mujoco.mj_forward(env.model, env.data)
            np.testing.assert_allclose(env._sole_tilts_deg(), [4.0, 4.0], atol=1e-3)
        finally:
            env.close()

    def test_stance_width_is_a_gaussian_around_the_home_keyframe_width(self):
        tolerance = 0.05
        env = TRexEnv(stance_width_weight=2.0, stance_width_tolerance_m=tolerance, reset_noise_scale=0.0)
        try:
            env.reset(seed=0)
            # Computed plant-side from the home keyframe's foot sites.
            assert env._home_stance_width == pytest.approx(0.280, abs=1e-6)
            home_info = self._reward_info(env)
            assert home_info["stance_width"] == pytest.approx(env._home_stance_width)
            assert home_info["stance_width_error"] == pytest.approx(0.0, abs=1e-9)
            assert home_info["reward_stance_width"] == pytest.approx(2.0)

            # Wider and narrower by one tolerance are priced alike.
            for offset in (tolerance, -tolerance):
                env._stance_width = lambda data, width=env._home_stance_width + offset: width
                info = self._reward_info(env)
                assert info["stance_width_error"] == pytest.approx(tolerance)
                assert info["stance_width_quality"] == pytest.approx(np.exp(-1.0))
                assert info["reward_stance_width"] == pytest.approx(2.0 * np.exp(-1.0))
        finally:
            env.close()

    def test_stance_width_is_free_of_heading(self):
        env = TRexEnv(reset_noise_scale=0.0)
        try:
            env.reset(seed=0)
            home_width = env._stance_width(env.data)
            yaw = np.radians(30.0)
            env.data.qpos[3:7] = [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]
            mujoco.mj_forward(env.model, env.data)
            assert env._stance_width(env.data) == pytest.approx(home_width, abs=1e-12)
        finally:
            env.close()

    def test_legacy_defaults_leave_the_total_reward_bit_identical(self):
        """The physics-r8 terms are opt-in: at their defaults the total is the pre-r8 sum, bit for bit.

        Summed here in the order ``_get_reward_info`` summed before the terms
        existed, with exact equality, on the shipped stance shaping minus the
        new keys, so every legacy term is live.
        """
        from environments.shared.config import load_stage_config

        new_keys = {
            "neck_posture_reference",
            "foot_flatness_weight",
            "foot_flatness_tolerance_deg",
            "stance_width_weight",
            "stance_width_tolerance_m",
            *_FOLLOWUP_KEYS,
        }
        kwargs = {k: v for k, v in load_stage_config("trex", "stance")["env_kwargs"].items() if k not in new_keys}
        legacy_terms = (
            "reward_forward",
            "reward_backward",
            "reward_drift",
            "reward_alive",
            "reward_energy",
            "reward_tail",
            "reward_bite",
            "reward_approach",
            "reward_head_proximity",
            "reward_head_clearance",
            "reward_neck_posture",
            "reward_tail_home_pose",
            "reward_posture",
            "reward_nosedive",
            "reward_height",
            "reward_bilateral_support",
            "reward_foot_load_balance",
            "reward_leg_home_pose",
            "reward_gait",
            "reward_smoothness",
            "reward_action_jerk",
            "reward_action_saturation",
            "reward_heading",
            "reward_lateral",
            "reward_spin",
            "reward_speed",
            "reward_idle",
        )
        env = TRexEnv(**kwargs)
        try:
            assert env.neck_posture_reference == "keyframe"
            np.testing.assert_array_equal(
                env._neck_home_qpos, env.model.key_qpos[env.home_keyframe_id][env._neck_home_qpos_indices]
            )
            env.reset(seed=3042)
            rng = np.random.default_rng(0)
            for _ in range(60):
                _, reward, terminated, _, info = env.step(rng.uniform(-0.3, 0.3, env.action_space.shape))
                assert not terminated
                legacy_total = info[legacy_terms[0]]
                for name in legacy_terms[1:]:
                    legacy_total = legacy_total + info[name]
                assert reward == legacy_total
                assert info["reward_foot_flatness"] == 0.0 and info["reward_stance_width"] == 0.0
                # The diagnostics are live even while the terms are off.
                assert 0.0 < info["foot_flatness_quality"] <= 1.0
                assert 0.0 < info["stance_width_quality"] <= 1.0
        finally:
            env.close()

    def test_followup_defaults_leave_the_r8_total_bit_identical(self):
        """The D-D27 terms are opt-in: at their defaults the total is the physics-r8 (D-D24) sum, bit for bit.

        On the shipped stance shaping minus the seven D-D27 keys, through the
        real step (the 10 Hz filter included), summed in the order
        ``_get_reward_info`` summed before the terms existed.
        """
        from environments.shared.config import load_stage_config

        kwargs = {k: v for k, v in load_stage_config("trex", "stance")["env_kwargs"].items() if k not in _FOLLOWUP_KEYS}
        r8_terms = (
            "reward_forward",
            "reward_backward",
            "reward_drift",
            "reward_alive",
            "reward_energy",
            "reward_tail",
            "reward_bite",
            "reward_approach",
            "reward_head_proximity",
            "reward_head_clearance",
            "reward_neck_posture",
            "reward_tail_home_pose",
            "reward_posture",
            "reward_nosedive",
            "reward_height",
            "reward_bilateral_support",
            "reward_foot_load_balance",
            "reward_leg_home_pose",
            "reward_foot_flatness",
            "reward_stance_width",
            "reward_gait",
            "reward_smoothness",
            "reward_action_jerk",
            "reward_action_saturation",
            "reward_heading",
            "reward_lateral",
            "reward_spin",
            "reward_speed",
            "reward_idle",
        )
        env = TRexEnv(**kwargs)
        try:
            assert (env.stance_width_reference, env.action_penalty_source) == ("keyframe", "filtered")
            assert env.foot_terms_min_support_force == env.floor_impact_weight == env.airborne_substep_weight == 0.0
            env.reset(seed=3042)
            rng = np.random.default_rng(0)
            for _ in range(240):
                _, reward, terminated, _, info = env.step(rng.uniform(-0.3, 0.3, env.action_space.shape))
                if terminated:
                    break
                total = info[r8_terms[0]]
                for name in r8_terms[1:]:
                    total = total + info[name]
                assert reward == total
                assert info["reward_floor_impact"] == 0.0 and info["reward_airborne_substeps"] == 0.0
                assert info["stance_width_target"] == env._home_stance_width
            # The diagnostics are live even while the terms are off.
            assert info["peak_foot_force_bw"] > 0.5
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
            ({"neck_posture_reference": "home"}, "neck_posture_reference"),
            ({"foot_flatness_tolerance_deg": 0.0}, "foot_flatness_tolerance_deg"),
            ({"stance_width_tolerance_m": -0.01}, "stance_width_tolerance_m"),
            ({"stance_width_reference": "spawn"}, "stance_width_reference"),
            ({"stance_width_settle_steps": 0}, "stance_width_settle_steps"),
            ({"stance_width_settle_steps": 2.5}, "stance_width_settle_steps"),
            ({"stance_width_settle_steps": True}, "stance_width_settle_steps"),
            ({"foot_terms_min_support_force": -1.0}, "foot_terms_min_support_force"),
            ({"foot_terms_min_support_force": float("inf")}, "foot_terms_min_support_force"),
            ({"floor_impact_weight": -2.0}, "floor_impact_weight"),
            ({"floor_impact_threshold_bw": 0.0}, "floor_impact_threshold_bw"),
            ({"floor_impact_threshold_bw": float("nan")}, "floor_impact_threshold_bw"),
            ({"airborne_substep_weight": -1.0}, "airborne_substep_weight"),
            ({"action_penalty_source": "applied"}, "action_penalty_source"),
        ],
    )
    def test_invalid_stance_settings_fail_fast(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            TRexEnv(**kwargs)


class TestStanceFollowupTerms:
    """The D-D27 terms: settled width reference, load gate, floor impact and airborne substeps, raw pricing."""

    @staticmethod
    def _reward_info(env):
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, info = env._get_reward_info(action)
        return info

    def test_settled_width_reference_captures_the_animals_own_width_and_pays_only_after(self):
        env = TRexEnv(
            stance_width_weight=2.0,
            stance_width_reference="settled",
            stance_width_settle_steps=5,
            reset_noise_scale=0.0,
        )
        try:
            env.reset(seed=0)
            settled = env._home_stance_width + 0.03
            env._stance_width = lambda data: settled
            for step in range(1, 6):
                env._step_count = step
                info = self._reward_info(env)
                # Unpaid through the capture step, centred on the keyframe until it.
                assert info["reward_stance_width"] == 0.0
                expected_target = env._home_stance_width if step < 5 else settled
                assert info["stance_width_target"] == pytest.approx(expected_target)
            env._step_count = 6
            info = self._reward_info(env)
            assert info["stance_width_error"] == pytest.approx(0.0, abs=1e-12)
            assert info["reward_stance_width"] == pytest.approx(2.0)
            # Moving away from the CAPTURED width is priced, the keyframe no longer is.
            env._stance_width = lambda data: settled + env.stance_width_tolerance_m
            env._step_count = 7
            assert self._reward_info(env)["reward_stance_width"] == pytest.approx(2.0 * np.exp(-1.0))
            # Each episode captures its own.
            env.reset(seed=1)
            assert env._stance_width_target == env._home_stance_width
        finally:
            env.close()

    def test_a_horizon_shorter_than_the_settle_never_pays_the_settled_width(self):
        env = TRexEnv(
            stance_width_weight=1.0, stance_width_reference="settled", max_episode_steps=40, reset_noise_scale=0.0
        )
        try:
            env.reset(seed=0)
            infos = [env.step(np.zeros(env.model.nu))[4] for _ in range(40)]
            assert all(info["reward_stance_width"] == 0.0 for info in infos)
            assert infos[-1]["stance_width_target"] == env._home_stance_width
        finally:
            env.close()

    def test_keyframe_reference_pays_from_the_first_step(self):
        env = TRexEnv(stance_width_weight=2.0, reset_noise_scale=0.0)
        try:
            env.reset(seed=0)
            env._step_count = 1
            assert self._reward_info(env)["reward_stance_width"] == pytest.approx(2.0, abs=1e-9)
            env._step_count = 200
            assert self._reward_info(env)["stance_width_target"] == env._home_stance_width
        finally:
            env.close()

    def test_target_is_defined_before_the_first_reset(self):
        env = TRexEnv(stance_width_reference="settled")
        try:
            assert env._stance_width_target == env._home_stance_width
        finally:
            env.close()

    def test_load_gate_drops_an_unloaded_foots_flatness_share_and_the_width_term(self):
        env = TRexEnv(
            foot_flatness_weight=2.0,
            stance_width_weight=1.0,
            foot_terms_min_support_force=168.0,
            reset_noise_scale=0.0,
        )
        try:
            env.reset(seed=0)
            env._sole_tilts_deg = lambda: np.array([0.0, 0.0])
            env._aggregated_foot_contact_forces = lambda: (420.0, 420.0)
            info = self._reward_info(env)
            assert info["reward_foot_flatness"] == pytest.approx(2.0)
            assert info["reward_stance_width"] == pytest.approx(1.0, abs=1e-6)
            assert (info["r_foot_terms_supported"], info["l_foot_terms_supported"]) == (1.0, 1.0)
            # The left foot carries less than the bar: it forfeits exactly its half, and the pair the width term.
            env._aggregated_foot_contact_forces = lambda: (700.0, 167.9)
            info = self._reward_info(env)
            assert info["reward_foot_flatness"] == pytest.approx(1.0)
            assert info["reward_stance_width"] == 0.0
            assert (info["r_foot_terms_supported"], info["l_foot_terms_supported"]) == (1.0, 0.0)
            # Exactly at the bar is supported.
            env._aggregated_foot_contact_forces = lambda: (700.0, 168.0)
            info = self._reward_info(env)
            assert info["reward_foot_flatness"] == pytest.approx(2.0)
            assert info["reward_stance_width"] == pytest.approx(1.0, abs=1e-6)
            assert (info["r_foot_terms_supported"], info["l_foot_terms_supported"]) == (1.0, 1.0)
            # Airborne: nothing.
            env._aggregated_foot_contact_forces = lambda: (0.0, 0.0)
            info = self._reward_info(env)
            assert info["reward_foot_flatness"] == 0.0 and info["reward_stance_width"] == 0.0
        finally:
            env.close()

    def test_ungated_terms_ignore_the_load(self):
        env = TRexEnv(foot_flatness_weight=2.0, stance_width_weight=1.0, reset_noise_scale=0.0)
        try:
            env.reset(seed=0)
            env._sole_tilts_deg = lambda: np.array([0.0, 0.0])
            env._aggregated_foot_contact_forces = lambda: (0.0, 0.0)
            info = self._reward_info(env)
            assert info["reward_foot_flatness"] == pytest.approx(2.0)
            assert info["reward_stance_width"] == pytest.approx(1.0, abs=1e-6)
        finally:
            env.close()

    def test_body_weight_is_the_whole_animal(self):
        env = TRexEnv()
        try:
            # 85.72 kg x 9.81: stance.toml's foot_load_balance_min_support_force divides the same weight.
            assert env._body_weight_n == pytest.approx(840.9, abs=0.1)
        finally:
            env.close()

    def test_floor_impact_prices_the_peak_above_the_threshold_linearly(self):
        env = TRexEnv(floor_impact_weight=2.0, floor_impact_threshold_bw=1.4, reset_noise_scale=0.0)
        try:
            env.reset(seed=0)
            weight = env._body_weight_n
            for peak_bw, expected in ((1.0, 0.0), (1.4, 0.0), (1.9, -1.0), (2.4, -2.0)):
                block = np.full((5, 2), 0.5 * weight)
                block[3] = (0.5 * peak_bw * weight, 0.5 * peak_bw * weight)
                env._substep_foot_force_block = lambda block=block: block
                info = self._reward_info(env)
                assert info["peak_foot_force_bw"] == pytest.approx(peak_bw)
                assert info["reward_floor_impact"] == pytest.approx(expected)
        finally:
            env.close()

    def test_floor_impact_peak_is_the_summed_force_not_each_foots_own_peak(self):
        env = TRexEnv(floor_impact_weight=2.0, floor_impact_threshold_bw=1.2, reset_noise_scale=0.0)
        try:
            env.reset(seed=0)
            weight = env._body_weight_n
            # Each foot peaks on its own substep (a single-foot re-landing): the floor never carries
            # 0.9 + 0.9 = 1.8 BW at once, so the step's peak is 1.2 BW, at this threshold and unpriced.
            block = np.array([[0.5, 0.5], [0.9, 0.3], [0.3, 0.9], [0.5, 0.5], [0.5, 0.5]]) * weight
            env._substep_foot_force_block = lambda: block
            info = self._reward_info(env)
            assert info["peak_foot_force_bw"] == pytest.approx(1.2)
            assert info["reward_floor_impact"] == pytest.approx(0.0)
            # The threshold is the configured one, not the stance's 1.4: 0.1 BW over 1.2 costs 0.2.
            block = np.array([[0.5, 0.5], [1.0, 0.3], [0.3, 1.0], [0.5, 0.5], [0.5, 0.5]]) * weight
            env._substep_foot_force_block = lambda: block
            info = self._reward_info(env)
            assert info["peak_foot_force_bw"] == pytest.approx(1.3)
            assert info["reward_floor_impact"] == pytest.approx(-0.2)
        finally:
            env.close()

    def test_airborne_substeps_count_substeps_with_every_foot_unloaded(self):
        env = TRexEnv(airborne_substep_weight=1.0, reset_noise_scale=0.0)
        try:
            env.reset(seed=0)
            # Substeps 0 and 1 airborne (every foot <= 0.1 N); 2 is single support, not airborne.
            block = np.array([[0.0, 0.1], [0.05, 0.0], [0.0, 300.0], [400.0, 400.0], [420.0, 410.0]])
            env._substep_foot_force_block = lambda: block
            info = self._reward_info(env)
            assert info["airborne_substeps"] == 2.0
            assert info["reward_airborne_substeps"] == pytest.approx(-2.0 / 5.0)
        finally:
            env.close()

    def test_new_terms_enter_the_total_only_when_weighted(self):
        env = TRexEnv(floor_impact_weight=2.0, airborne_substep_weight=1.0, reset_noise_scale=0.0)
        try:
            env.reset(seed=0)
            block = np.array([[0.0, 0.0]] * 4 + [[2000.0, 2000.0]])
            env._substep_foot_force_block = lambda: block
            total, info = env._get_reward_info(np.zeros(env.action_space.shape))
            others = sum(
                value
                for key, value in info.items()
                if key.startswith("reward_")
                and key not in ("reward_total", "reward_floor_impact", "reward_airborne_substeps")
            )
            assert info["reward_floor_impact"] < 0.0 and info["reward_airborne_substeps"] == pytest.approx(-0.8)
            assert total == pytest.approx(others + info["reward_floor_impact"] + info["reward_airborne_substeps"])
        finally:
            env.close()

    def test_substep_block_matches_the_min_aggregate_and_is_step_tagged(self):
        env = TRexEnv()
        try:
            env.reset(seed=0)
            # Before any step: the instantaneous read, as one row.
            assert env._substep_foot_force_block().shape == (1, 2)
            env.step(np.zeros(env.action_space.shape))
            block = env._substep_foot_force_block()
            assert block.shape == (env.frame_skip, 2)
            np.testing.assert_array_equal(block.min(axis=0), env._aggregated_foot_contact_forces())
            np.testing.assert_array_equal(block[-1], env._foot_contact_forces())
            block[:] = -1.0  # a copy: the step's record cannot be edited through it
            assert env._substep_foot_force_block().min() >= 0.0
            env._invalidate_substep_aggregates()
            assert env._substep_foot_force_block().shape == (1, 2)
            env.step(np.zeros(env.action_space.shape))
            env.reset(seed=1)
            assert env._substep_foot_force_block().shape == (1, 2)
        finally:
            env.close()

    @staticmethod
    def _square_wave_run(source: str, steps: int = 6, amplitude: float = 1.0, **kwargs):
        env = TRexEnv(
            smoothness_weight=2.0,
            action_jerk_weight=3.0,
            action_saturation_weight=0.5,
            energy_penalty_weight=0.075,
            action_penalty_source=source,
            reset_noise_scale=0.0,
            **kwargs,
        )
        try:
            env.reset(seed=0)
            infos = []
            for index in range(steps):
                infos.append(env.step(np.full(env.model.nu, amplitude if index % 2 else -amplitude))[4])
            return infos
        finally:
            env.close()

    def test_raw_source_prices_the_pre_filter_command(self):
        from environments.shared.reward_functions import (
            reward_action_jerk,
            reward_action_saturation,
            reward_action_smoothness,
        )

        raw = self._square_wave_run("raw")
        filtered = self._square_wave_run("filtered")
        n = 15
        commands = [np.full(n, 1.0 if index % 2 else -1.0) for index in range(len(raw))]
        for index in range(2, len(raw)):
            # Exactly the pure terms on the clipped +-1 commands the policy sent.
            smooth, delta = reward_action_smoothness(commands[index], commands[index - 1], n, 2.0)
            jerk_reward, jerk = reward_action_jerk(commands[index], commands[index - 1], commands[index - 2], n, 3.0)
            saturation, _ = reward_action_saturation(commands[index], 0.5, 0.9)
            assert raw[index]["reward_smoothness"] == pytest.approx(float(smooth))
            assert raw[index]["action_delta"] == pytest.approx(float(delta))
            assert raw[index]["reward_action_jerk"] == pytest.approx(float(jerk_reward))
            assert raw[index]["action_jerk"] == pytest.approx(float(jerk))
            assert raw[index]["reward_action_saturation"] == pytest.approx(float(saturation))
            # The filter cuts most of the square wave, so the filtered command is priced far less.
            assert raw[index]["reward_smoothness"] < 5.0 * filtered[index]["reward_smoothness"]
            assert raw[index]["reward_action_jerk"] < 5.0 * filtered[index]["reward_action_jerk"]
            # Energy is the APPLIED command's under either source.
            assert raw[index]["reward_energy"] == filtered[index]["reward_energy"]

    def test_an_out_of_range_raw_command_is_priced_as_its_clip(self):
        # SB3 clips to the action space; a direct caller may not, and the saturation ramp would over-price it.
        clipped = self._square_wave_run("raw")
        beyond = self._square_wave_run("raw", amplitude=1.5)
        for index in range(len(clipped)):
            for key in ("reward_smoothness", "action_delta", "reward_action_jerk", "reward_action_saturation"):
                assert beyond[index][key] == clipped[index][key], (index, key)

    def test_raw_equals_filtered_without_the_filter(self):
        class _UnfilteredTRex(TRexEnv):
            action_filter_cutoff_hz = 0.0

        def run(source):
            env = _UnfilteredTRex(smoothness_weight=2.0, action_jerk_weight=3.0, action_penalty_source=source)
            try:
                env.reset(seed=0)
                rng = np.random.default_rng(1)
                return [env.step(rng.uniform(-1, 1, env.action_space.shape))[1] for _ in range(5)]
            finally:
                env.close()

        assert run("raw") == run("filtered")

    def test_a_stale_policy_command_is_never_priced(self):
        env = TRexEnv(smoothness_weight=2.0, action_penalty_source="raw", reset_noise_scale=0.0)
        try:
            env.reset(seed=0)
            env.step(np.ones(env.action_space.shape))
            # Scored out of band after the step: the kept command belongs to the step that ran, and a
            # direct call at another step count reads the action it is given.
            env._step_count += 1
            env._prev_action = np.zeros(env.action_space.shape)
            info = self._reward_info(env)
            assert info["action_delta"] == 0.0
        finally:
            env.close()
