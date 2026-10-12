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
            + info["reward_bilateral_support"]
            + info["reward_energy"]
            + info["reward_tail"]
            + info["reward_strike"]
            + info["reward_approach"]
            + info["reward_proximity"]
            + info["reward_claw_proximity"]
            + info["reward_posture"]
            + info["reward_nosedive"]
            + info["reward_leg_home_pose"]
            + info["reward_spin"]
            + info["reward_gait"]
            + info["reward_smoothness"]
            + info["reward_action_jerk"]
            + info["reward_action_saturation"]
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

    def test_posture_reward_prefers_xml_home_lean_to_vertical(self, env):
        home_quat = env.model.key_qpos[env.home_keyframe_id, 3:7]
        home_reward, home_tilt = env._compute_lean_aware_posture_reward(
            home_quat,
            1.0,
            env._natural_forward_z,
        )
        upright_reward, upright_tilt = env._compute_lean_aware_posture_reward(
            np.array([1.0, 0.0, 0.0, 0.0]),
            1.0,
            env._natural_forward_z,
        )

        assert home_reward == pytest.approx(0.0, abs=1e-5)
        assert home_reward > upright_reward
        assert home_tilt == pytest.approx(np.deg2rad(20.0), abs=1e-5)
        assert upright_tilt == pytest.approx(0.0, abs=1e-6)

    def test_step_reports_lean_aware_posture_reward(self, env):
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        pelvis_quat = env.data.sensordata[env._sensor_quat_start : env._sensor_quat_start + 4]
        expected, absolute_tilt = env._compute_lean_aware_posture_reward(
            pelvis_quat,
            env.posture_weight,
            env._natural_forward_z,
        )

        assert info["reward_posture"] == pytest.approx(expected, abs=1e-8)
        assert info["tilt_angle"] == pytest.approx(absolute_tilt, abs=1e-8)

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


class TestStanceQualityRewards:
    """The stance-quality terms added with the physics-r3 plant (TRexEnv's names).

    Every default is inert, so an env that does not set them computes exactly
    the reward that predates them; stage1_balance.toml sets them.
    """

    #: The reward components that predate the stance-quality terms, in the
    #: order the legacy total summed them.
    LEGACY_TERMS = (
        "reward_forward",
        "reward_backward",
        "reward_drift",
        "reward_alive",
        "reward_energy",
        "reward_tail",
        "reward_strike",
        "reward_approach",
        "reward_proximity",
        "reward_claw_proximity",
        "reward_posture",
        "reward_nosedive",
        "reward_spin",
        "reward_gait",
        "reward_smoothness",
        "reward_heading",
        "reward_lateral",
        "reward_speed",
        "reward_idle",
    )

    @staticmethod
    def _reward_info(env, action=None):
        if action is None:
            action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, info = env._get_reward_info(action)
        return info

    def test_new_stance_rewards_default_to_zero(self):
        env = RaptorEnv(reset_noise_scale=0.0)
        try:
            env.reset(seed=42)
            _, _, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
            assert info["reward_bilateral_support"] == 0.0
            assert info["reward_leg_home_pose"] == 0.0
            assert info["reward_action_jerk"] == 0.0
            assert info["reward_action_saturation"] == 0.0
            assert info["alive_gate"] == 1.0
            assert info["reward_alive"] == info["raw_alive"]
        finally:
            env.close()

    def test_legacy_defaults_reproduce_the_reward_bit_for_bit(self):
        """At the defaults the total is the legacy left-to-right sum, exactly.

        The new terms are exact zeros (and the alive gate exactly 1.0), and a
        zero anywhere in a float sum leaves every bit of it unchanged, so the
        defaulted env computes the reward that predates them -- compared with
        ``==``, not a tolerance, over a perturbed episode with every branch
        live (random actions, stage-1 reset noise).
        """
        env = RaptorEnv(reset_noise_scale=0.05)
        try:
            env.reset(seed=3042)
            rng = np.random.default_rng(0)
            for _ in range(200):
                action = rng.uniform(-1.0, 1.0, env.action_space.shape).astype(np.float32)
                _, reward, terminated, truncated, info = env.step(action)
                legacy = info[self.LEGACY_TERMS[0]]
                for key in self.LEGACY_TERMS[1:]:
                    legacy = legacy + info[key]
                if terminated and not info.get("success"):
                    legacy += env.fall_penalty
                assert reward == legacy
                assert info["reward_alive"] == info["raw_alive"]
                if terminated or truncated:
                    env.reset(seed=3043)
        finally:
            env.close()

    def test_bilateral_support_is_bounded_and_uses_weaker_foot(self):
        env = RaptorEnv(bilateral_support_weight=2.0, foot_contact_saturation_force=50.0, reset_noise_scale=0.0)
        try:
            env.reset(seed=0)
            env._foot_contact_forces = lambda: (25.0, 60.0)
            info = self._reward_info(env)
            assert info["bilateral_support_quality"] == pytest.approx(0.5)
            assert info["reward_bilateral_support"] == pytest.approx(1.0)

            env._foot_contact_forces = lambda: (500.0, 500.0)
            info = self._reward_info(env)
            assert info["bilateral_support_quality"] == pytest.approx(1.0)
            assert info["reward_bilateral_support"] == pytest.approx(2.0)
        finally:
            env.close()

    def test_alive_bonus_can_be_partially_conditioned_on_support(self):
        env = RaptorEnv(
            alive_bonus=2.0,
            support_conditioned_alive_fraction=0.25,
            foot_contact_saturation_force=50.0,
            reset_noise_scale=0.0,
        )
        try:
            env.reset(seed=0)
            env._foot_contact_forces = lambda: (25.0, 100.0)
            info = self._reward_info(env)
            assert info["raw_alive"] == pytest.approx(2.0)
            assert info["bilateral_support_quality"] == pytest.approx(0.5)
            assert info["alive_gate"] == pytest.approx(0.875)
            assert info["reward_alive"] == pytest.approx(1.75)
        finally:
            env.close()

    def test_quiet_statue_earns_full_support_credit_at_the_stage_saturation(self):
        """The settled statue's weaker foot clears stage 1's 50 N saturation.

        Each foot carries half of the 132.4 N animal (66.2 N) once the summed
        touch reads the whole foot, so the stage's saturation force pays the
        statue full credit; the old toe-d3-only reading (36 N) could not.
        """
        env = RaptorEnv(bilateral_support_weight=1.0, foot_contact_saturation_force=50.0, reset_noise_scale=0.0)
        try:
            env.reset(seed=0)
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            for _ in range(200):
                _, _, terminated, _, info = env.step(action)
                assert not terminated
            assert min(info["r_foot_contact"], info["l_foot_contact"]) > 60.0
            assert info["bilateral_support_quality"] == pytest.approx(1.0)
        finally:
            env.close()

    def test_leg_home_pose_is_soft_and_uses_the_home_keyframe(self):
        tolerance = 0.20
        env = RaptorEnv(leg_home_pose_weight=3.0, leg_home_pose_tolerance=tolerance, reset_noise_scale=0.0)
        try:
            env.reset(seed=0)
            home = env.model.key_qpos[env.home_keyframe_id]
            np.testing.assert_array_equal(env._leg_home_qpos, home[env._leg_home_qpos_indices])
            assert len(env._leg_home_qpos_indices) == 12
            home_info = self._reward_info(env)
            assert home_info["leg_home_pose_error"] == pytest.approx(0.0)
            assert home_info["leg_home_pose_quality"] == pytest.approx(1.0)
            assert home_info["reward_leg_home_pose"] == pytest.approx(3.0)

            env.data.qpos[env._leg_home_qpos_indices[0]] += tolerance
            displaced_info = self._reward_info(env)
            assert displaced_info["leg_home_pose_error"] == pytest.approx(tolerance / np.sqrt(12.0))
            assert displaced_info["leg_home_pose_quality"] == pytest.approx((11.0 + np.exp(-1.0)) / 12.0)
            assert 0.0 < displaced_info["reward_leg_home_pose"] < 3.0
        finally:
            env.close()

    def test_action_saturation_prices_parked_commands_only(self):
        env = RaptorEnv(action_saturation_weight=0.5, action_saturation_threshold=0.9, reset_noise_scale=0.0)
        try:
            env.reset(seed=0)
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            action[:2] = 0.85
            assert self._reward_info(env, action)["reward_action_saturation"] == 0.0
            action[:2] = (1.0, -1.0)
            info = self._reward_info(env, action)
            assert info["action_saturation"] == pytest.approx(2.0 / env.model.nu)
            assert info["reward_action_saturation"] == pytest.approx(-0.5 * 2.0 / env.model.nu)
        finally:
            env.close()

    def test_action_jerk_reads_the_two_previous_actions(self):
        """Jerk runs BEFORE smoothness rotates the action history.

        A step-and-hold sequence 0, 1, 1 has second differences 0 then -1:
        the jerk term must see (1 - 2*1 + 0) on the third step, which it only
        does if it reads the history before _reward_action_smoothness moves it.
        """
        env = RaptorEnv(action_jerk_weight=2.0, reset_noise_scale=0.0)
        try:
            env.reset(seed=0)
            zero = np.zeros(env.action_space.shape, dtype=np.float32)
            step = zero.copy()
            step[0] = 1.0
            jerks = [env.step(action)[4]["reward_action_jerk"] for action in (zero, step, step)]
            assert jerks[0] == 0.0 and jerks[1] == 0.0
            assert jerks[2] == pytest.approx(-2.0 * 1.0 / (env.model.nu * 16.0))
            env.reset(seed=1)
            assert env._prev_prev_action is None
            assert env.step(step)[4]["reward_action_jerk"] == 0.0
        finally:
            env.close()

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"foot_contact_saturation_force": 0.0}, "foot_contact_saturation_force"),
            ({"support_conditioned_alive_fraction": -0.1}, "support_conditioned_alive_fraction"),
            ({"support_conditioned_alive_fraction": 1.1}, "support_conditioned_alive_fraction"),
            ({"action_saturation_threshold": 1.0}, "action_saturation_threshold"),
            ({"leg_home_pose_tolerance": 0.0}, "leg_home_pose_tolerance"),
        ],
    )
    def test_invalid_stance_settings_fail_fast(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            RaptorEnv(**kwargs)


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
