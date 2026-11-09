"""Species-specific reward tests for Velociraptor.

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
    assert_nosedive_penalty_non_positive,
    assert_posture_reward_non_positive,
    assert_smoothness_penalty_for_action_change,
    assert_smoothness_zero_on_first_step,
)
from environments.velociraptor_mongoliensis.anatomical.env import VelociraptorMongoliensisAnatomicalEnv


@pytest.fixture
def env():
    e = VelociraptorMongoliensisAnatomicalEnv()
    yield e
    e.close()


class TestVelociraptorMongoliensisRewardComponents:
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

    def test_posture_reward_centres_on_the_settled_pitch_beside_the_level_home(self, env):
        """The posture target is the statue's settled pitch, 0.002 rad nose-down of the level home keyframe.

        Since physics r5 the keyframe authors the trunk level (the r4 home leaned 20 deg nose-down
        and its statue settled there), so the home pose is the upright one, and the target sits
        0.002 rad from it: the level home pays 4.0e-6 of the unit weight, and r4's 20 deg lean
        pays 0.119 of it (measured).
        """

        def posture(pitch):
            quat = np.array([np.cos(pitch / 2.0), 0.0, np.sin(pitch / 2.0), 0.0])
            return env._compute_lean_aware_posture_reward(quat, 1.0, env._natural_forward_z)

        home_quat = env.model.key_qpos[env.home_keyframe_id, 3:7]
        np.testing.assert_array_equal(home_quat, [1.0, 0.0, 0.0, 0.0])
        home_reward, home_tilt = env._compute_lean_aware_posture_reward(home_quat, 1.0, env._natural_forward_z)
        settled_reward, settled_tilt = posture(0.002)
        r4_reward, _ = posture(np.deg2rad(20.0))

        assert settled_reward == pytest.approx(0.0, abs=1e-12)
        assert settled_tilt == pytest.approx(0.002, abs=1e-9)
        assert home_tilt == pytest.approx(0.0, abs=1e-6)
        assert -1e-5 < home_reward < settled_reward
        assert r4_reward < -0.1 and r4_reward < home_reward
        assert posture(-0.05)[0] < home_reward and posture(0.05)[0] < settled_reward

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


class TestVelociraptorMongoliensisRewardWeightEffects:
    """Verify that changing reward weights affects the output."""

    def test_high_alive_bonus_dominates(self):
        env = VelociraptorMongoliensisAnatomicalEnv(
            alive_bonus=100.0, forward_vel_weight=0.0, strike_approach_weight=0.0, strike_bonus=0.0
        )
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, terminated, _, info = env.step(action)
        if not terminated:
            assert info["reward_alive"] == 100.0
        env.close()

    def test_zero_strike_bonus_gives_no_strike_reward(self):
        env = VelociraptorMongoliensisAnatomicalEnv(strike_bonus=0.0)
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_strike"] == 0.0
        env.close()

    def test_zero_posture_weight_zeroes_posture_reward(self):
        env = VelociraptorMongoliensisAnatomicalEnv(posture_weight=0.0)
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_posture"] == 0.0
        env.close()

    def test_zero_gait_weight_zeroes_gait_reward(self):
        env = VelociraptorMongoliensisAnatomicalEnv(gait_symmetry_weight=0.0)
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_gait"] == 0.0
        env.close()

    def test_zero_smoothness_weight_zeroes_smoothness_reward(self):
        env = VelociraptorMongoliensisAnatomicalEnv(smoothness_weight=0.0)
        env.reset(seed=42)
        action = env.action_space.sample()
        env.step(action)
        action2 = env.action_space.sample()
        _, _, _, _, info = env.step(action2)
        assert info["reward_smoothness"] == 0.0
        env.close()

    def test_zero_spin_weight_zeroes_spin_reward(self):
        env = VelociraptorMongoliensisAnatomicalEnv(spin_penalty_weight=0.0)
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_spin"] == 0.0
        env.close()

    def test_nonzero_spin_weight_gives_nonpositive_reward(self):
        env = VelociraptorMongoliensisAnatomicalEnv(spin_penalty_weight=0.5)
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_spin"] <= 0.0
        assert info["spin_instability"] >= 0.0
        env.close()

    def test_zero_drift_weight_zeroes_drift_reward(self):
        env = VelociraptorMongoliensisAnatomicalEnv(drift_penalty_weight=0.0)
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_drift"] == 0.0
        env.close()

    def test_nonzero_drift_weight_penalizes_displacement(self):
        """Drift penalty should be negative after several steps of movement."""
        env = VelociraptorMongoliensisAnatomicalEnv(drift_penalty_weight=0.5)
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
        env = VelociraptorMongoliensisAnatomicalEnv(backward_vel_penalty_weight=0.0)
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_backward"] == 0.0
        env.close()

    def test_nonzero_claw_proximity_weight_gives_positive_reward(self):
        """Claw proximity reward should be positive when weight is set and prey is within range."""
        env = VelociraptorMongoliensisAnatomicalEnv(strike_claw_proximity_weight=5.0, prey_distance_range=(1.0, 2.0))
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert info["reward_claw_proximity"] >= 0.0
        assert info["min_claw_prey_distance"] > 0.0
        env.close()

    def test_nonzero_backward_vel_weight_penalizes_backward_motion(self):
        """Backward penalty should be negative when raptor moves backward."""
        env = VelociraptorMongoliensisAnatomicalEnv(backward_vel_penalty_weight=1.0, forward_vel_weight=0.0)
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
        """17 position servos since physics r5: neck and head 3, per leg 4 + the claw, tail 4.

        The arms are welded passive bodies and digits II-IV hinge passively on springs, as on
        the T-Rex (r4: 22, with servoed digits III / IV and four arm actuators).
        """
        env = VelociraptorMongoliensisAnatomicalEnv()
        assert env.model.nu == 17, f"Expected 17 actuators, got {env.model.nu}"
        assert env.action_space.shape == (17,)
        names = [mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(env.model.nu)]
        assert names == [
            "neck_pitch_act",
            "neck_yaw_act",
            "head_pitch_act",
            *(f"{side}_{joint}_act" for side in "rl" for joint in ("hip_pitch", "hip_roll", "knee", "ankle", "claw")),
            "tail_1_pitch_act",
            "tail_1_yaw_act",
            "tail_2_pitch_act",
            "tail_3_pitch_act",
        ]
        assert set(env.model.actuator_trntype) == {mujoco.mjtTrn.mjTRN_JOINT}
        assert set(env.model.actuator_gaintype) == {mujoco.mjtGain.mjGAIN_FIXED}
        assert set(env.model.actuator_biastype) == {mujoco.mjtBias.mjBIAS_AFFINE}
        env.close()


class TestStanceQualityRewards:
    """The stance-quality terms added with the physics-r3 plant (TyrannosaurusRexAnatomicalEnv's names).

    Every default is inert, so an env that does not set them computes exactly
    the reward that predates them; stance.toml sets them.
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
        env = VelociraptorMongoliensisAnatomicalEnv(reset_noise_scale=0.0)
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
        env = VelociraptorMongoliensisAnatomicalEnv(reset_noise_scale=0.05)
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
        env = VelociraptorMongoliensisAnatomicalEnv(
            bilateral_support_weight=2.0, foot_contact_saturation_force=50.0, reset_noise_scale=0.0
        )
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
        env = VelociraptorMongoliensisAnatomicalEnv(
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
        """The settled statue's weaker foot clears the stance stage's 55 N saturation.

        Each foot carries half of the 147.15 N animal of physics r5 (73.6 N measured at step
        200 noise-free; r4 66.2 N of 132.4) once the summed touch reads the whole foot, so the
        stage's saturation force pays the statue full credit; the old toe-d3-only reading (36 N
        on r2) could not.
        """
        env = VelociraptorMongoliensisAnatomicalEnv(
            bilateral_support_weight=1.0, foot_contact_saturation_force=55.0, reset_noise_scale=0.0
        )
        try:
            env.reset(seed=0)
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            for _ in range(200):
                _, _, terminated, _, info = env.step(action)
                assert not terminated
            assert min(info["r_foot_contact"], info["l_foot_contact"]) > 70.0
            assert info["bilateral_support_quality"] == pytest.approx(1.0)
        finally:
            env.close()

    def test_leg_home_pose_is_soft_and_uses_the_home_keyframe(self):
        tolerance = 0.20
        env = VelociraptorMongoliensisAnatomicalEnv(
            leg_home_pose_weight=3.0, leg_home_pose_tolerance=tolerance, reset_noise_scale=0.0
        )
        try:
            env.reset(seed=0)
            home = env.model.key_qpos[env.home_keyframe_id]
            np.testing.assert_array_equal(env._leg_home_qpos, home[env._leg_home_qpos_indices])
            # The eight servoed leg joints, hips to ankles (physics r5, as the T-Rex): the passive digits' angle
            # is the ground's, and the claw is the strike's (r4 priced 12, its servoed digits III / IV too).
            names = [
                mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_JOINT, env.model.jnt_qposadr.tolist().index(int(i)))
                for i in env._leg_home_qpos_indices
            ]
            assert names == [f"{side}_{joint}" for side in "rl" for joint in ("hip_pitch", "hip_roll", "knee", "ankle")]
            home_info = self._reward_info(env)
            assert home_info["leg_home_pose_error"] == pytest.approx(0.0)
            assert home_info["leg_home_pose_quality"] == pytest.approx(1.0)
            assert home_info["reward_leg_home_pose"] == pytest.approx(3.0)

            env.data.qpos[env._leg_home_qpos_indices[0]] += tolerance
            displaced_info = self._reward_info(env)
            assert displaced_info["leg_home_pose_error"] == pytest.approx(tolerance / np.sqrt(8.0))
            assert displaced_info["leg_home_pose_quality"] == pytest.approx((7.0 + np.exp(-1.0)) / 8.0)
            assert 0.0 < displaced_info["reward_leg_home_pose"] < 3.0
        finally:
            env.close()

    def test_action_saturation_prices_parked_commands_only(self):
        env = VelociraptorMongoliensisAnatomicalEnv(
            action_saturation_weight=0.5, action_saturation_threshold=0.9, reset_noise_scale=0.0
        )
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
        env = VelociraptorMongoliensisAnatomicalEnv(action_jerk_weight=2.0, reset_noise_scale=0.0)
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
            VelociraptorMongoliensisAnatomicalEnv(**kwargs)


class TestCurriculumStageRewards:
    """Test that reward configs from TOML produce expected behavior."""

    def test_stage1_balance_no_forward_reward(self):
        """Stage 1 config disables forward velocity reward but penalizes backward drift."""
        env = VelociraptorMongoliensisAnatomicalEnv(
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
        env = VelociraptorMongoliensisAnatomicalEnv(strike_approach_weight=10.0)
        env.reset(seed=42)
        action = env.action_space.sample()
        env.step(action)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert "approach_delta" in info
        assert "reward_approach" in info
        assert info["approach_delta"] != 0.0
        env.close()


class TestPortedStanceTerms:
    """The T-Rex's tail-pose, floor-impact and airborne-substep terms on the raptor (the maintainer's decision of 2026-11-06).

    Each is off by default and declared in ``OPT_IN_TASK_KWARGS``: off, it emits
    no info key and leaves the reward and the task fingerprint as they were;
    the stance stage turns all three on.  The neck home pose of physics r5 (the
    maintainer's decision of 2026-11-08) joined them; TestNeckHomePose covers it.
    """

    #: Every info key the four terms emit while on.
    ON_KEYS = (
        "tail_home_pose_error",
        "tail_home_pose_quality",
        "reward_tail_home_pose",
        "neck_home_pose_error",
        "neck_home_pose_quality",
        "reward_neck_home_pose",
        "peak_foot_force_bw",
        "reward_floor_impact",
        "airborne_substeps",
        "reward_airborne_substeps",
    )
    #: The stance stage's weights (configs/velociraptor_mongoliensis/anatomical/stance.toml).
    ON = {
        "tail_home_pose_weight": 0.25,
        "neck_home_pose_weight": 0.25,
        "floor_impact_weight": 0.5,
        "airborne_substep_weight": 1.0,
    }

    def test_off_by_default_they_emit_no_info_key(self):
        env = VelociraptorMongoliensisAnatomicalEnv(reset_noise_scale=0.0)
        try:
            env.reset(seed=42)
            _, _, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
            assert not set(self.ON_KEYS) & set(info)
        finally:
            env.close()

    def test_on_they_enter_the_total_after_the_legacy_sum(self):
        """The total is the legacy left-to-right sum, then the enabled terms in declaration order, exactly."""
        env = VelociraptorMongoliensisAnatomicalEnv(reset_noise_scale=0.05, **self.ON)
        try:
            env.reset(seed=3042)
            rng = np.random.default_rng(0)
            for _ in range(200):
                action = rng.uniform(-1.0, 1.0, env.action_space.shape).astype(np.float32)
                _, reward, terminated, truncated, info = env.step(action)
                assert set(self.ON_KEYS) <= set(info)
                expected = info[TestStanceQualityRewards.LEGACY_TERMS[0]]
                for key in TestStanceQualityRewards.LEGACY_TERMS[1:]:
                    expected = expected + info[key]
                # The opt-in terms, in the order _get_reward_info sums them.
                expected = (
                    expected
                    + info["reward_tail_home_pose"]
                    + info["reward_neck_home_pose"]
                    + info["reward_floor_impact"]
                    + info["reward_airborne_substeps"]
                )
                if terminated and not info.get("success"):
                    expected += env.fall_penalty
                # Exact, as the legacy test compares: summing the new terms anywhere else changes the last bits
                # on most steps (89 of these 200 with them first).
                assert reward == expected
                if terminated or truncated:
                    env.reset(seed=3043)
        finally:
            env.close()

    def test_the_statue_earns_the_pose_terms_and_pays_neither_contact_term(self):
        env = VelociraptorMongoliensisAnatomicalEnv(reset_noise_scale=0.0, **self.ON)
        try:
            env.reset(seed=0)
            infos = [env.step(np.zeros(env.action_space.shape, dtype=np.float32))[4] for _ in range(300)]
            window = infos[200:]
            assert min(info["tail_home_pose_quality"] for info in window) > 0.999
            assert min(info["neck_home_pose_quality"] for info in window) > 0.999
            assert all(info["reward_airborne_substeps"] == 0.0 for info in infos)
            assert all(info["reward_floor_impact"] == 0.0 for info in infos)
        finally:
            env.close()

    def test_a_tail_parked_on_its_stop_forfeits_its_joint(self):
        env = VelociraptorMongoliensisAnatomicalEnv(reset_noise_scale=0.0, tail_home_pose_weight=1.0)
        try:
            env.reset(seed=0)
            env.data.qpos[env._tail_home_qpos_indices] = env._tail_home_qpos
            settled = env._get_reward_info(np.zeros(env.model.nu))[1]["tail_home_pose_quality"]
            tail_1_pitch = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "tail_1_pitch")
            ventral_stop = float(env.model.jnt_range[tail_1_pitch, 0])
            assert ventral_stop == pytest.approx(-np.deg2rad(12.0))  # physics r5 (r4: -15 deg)
            env.data.qpos[env._tail_home_qpos_indices[0]] = ventral_stop  # tail_1_pitch on its ventral stop
            parked = env._get_reward_info(np.zeros(env.model.nu))[1]
            assert settled == pytest.approx(1.0)
            # One joint of four 0.21 rad off a 0.05 rad tolerance: that joint's Gaussian is ~0, so a quarter goes.
            assert parked["tail_home_pose_quality"] == pytest.approx(0.75, abs=1e-4)
            assert parked["reward_tail_home_pose"] == parked["tail_home_pose_quality"]
        finally:
            env.close()

    def test_impact_prices_the_peak_above_the_threshold_with_the_excess_capped(self, monkeypatch):
        env = VelociraptorMongoliensisAnatomicalEnv(reset_noise_scale=0.0, floor_impact_weight=0.5)
        try:
            env.reset(seed=0)
            # The 15.0 kg life-size animal of physics r5 (r4 13.5 kg); the default threshold is the stance's
            # 3.0 BW, above the 2.56 BW the reset pop reaches on 3000 seeds.
            weight = env._body_weight_n
            assert weight == pytest.approx(15.0 * 9.81, rel=1e-12)
            assert env.floor_impact_threshold_bw == 3.0
            for peak_bw, excess in ((2.56, 0.0), (3.5, 0.5), (12.0, 1.0)):
                block = np.array([[0.5, 0.5], [peak_bw / 2, peak_bw / 2], [0.4, 0.6]]) * weight
                monkeypatch.setattr(env, "_substep_foot_force_block", lambda block=block: block.copy())
                info = env._get_reward_info(np.zeros(env.model.nu))[1]
                assert info["peak_foot_force_bw"] == pytest.approx(peak_bw)
                assert info["reward_floor_impact"] == pytest.approx(-0.5 * excess)
        finally:
            env.close()

    def test_airborne_prices_the_share_of_substeps_with_every_foot_unloaded(self, monkeypatch):
        env = VelociraptorMongoliensisAnatomicalEnv(reset_noise_scale=0.0, airborne_substep_weight=1.0)
        try:
            env.reset(seed=0)
            # Five substeps: two with both feet at or under 0.1 N, one with one foot down, two standing.
            block = np.array([[0.0, 0.1], [0.05, 0.0], [0.0, 40.0], [60.0, 60.0], [66.0, 66.0]])
            monkeypatch.setattr(env, "_substep_foot_force_block", lambda: block.copy())
            info = env._get_reward_info(np.zeros(env.model.nu))[1]
            assert info["airborne_substeps"] == 2.0
            assert info["reward_airborne_substeps"] == pytest.approx(-2.0 / 5.0)
        finally:
            env.close()

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"tail_home_pose_weight": -0.1}, "tail_home_pose_weight"),
            ({"tail_home_pose_tolerance": 0.0}, "tail_home_pose_tolerance"),
            ({"floor_impact_weight": float("inf")}, "floor_impact_weight"),
            ({"floor_impact_threshold_bw": 0.0}, "floor_impact_threshold_bw"),
            ({"airborne_substep_weight": -1.0}, "airborne_substep_weight"),
            ({"neck_home_pose_weight": -0.1}, "neck_home_pose_weight"),
            ({"neck_home_pose_weight": float("nan")}, "neck_home_pose_weight"),
            ({"neck_home_pose_tolerance": 0.0}, "neck_home_pose_tolerance"),
            ({"neck_home_pose_tolerance": float("inf")}, "neck_home_pose_tolerance"),
        ],
    )
    def test_invalid_settings_fail_fast(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            VelociraptorMongoliensisAnatomicalEnv(**kwargs)

    def test_off_they_stay_out_of_the_task_fingerprint_and_on_they_are_hashed(self):
        from environments.shared.model_id import ModelId
        from environments.shared.task_fingerprint import stage_task_fingerprint

        model_id = ModelId("velociraptor_mongoliensis", "anatomical")
        names = {name for _, _, kwargs in VelociraptorMongoliensisAnatomicalEnv.OPT_IN_TASK_KWARGS for name in kwargs}
        assert names == {
            "tail_home_pose_weight",
            "tail_home_pose_tolerance",
            "neck_home_pose_weight",
            "neck_home_pose_tolerance",
            "floor_impact_weight",
            "floor_impact_threshold_bw",
            "airborne_substep_weight",
        }
        assert names <= set(stage_task_fingerprint(model_id, "stance")["env"])
        for stage in ("locomotion", "behavior"):
            assert not names & set(stage_task_fingerprint(model_id, stage)["env"]), stage


class TestNeckHomePose:
    """The neck and head carriage term of the life-size plant (physics r5, the maintainer's decision of 2026-11-08).

    Shaped as the tail term: the mean per-joint Gaussian of (neck_pitch, neck_yaw,
    head_pitch) around the statue's settled carriage (``_NECK_SETTLED_QPOS``;
    TestSettledPoseTargetsTrackTheStatue re-measures it).  The posture term reads only the
    pelvis frame, so without it a policy carries the head anywhere for free: stage 2 measured
    a 0.10 rad head raise earning +0.85 per episode with the term off, and with the stance's
    0.25 / 0.05 rad on a drooped or raised head losing 85-176 per episode against the statue,
    which keeps 0.998 of the term's maximum.  Off by default and in ``OPT_IN_TASK_KWARGS``:
    off it emits no info key, changes no reward bit and stays out of the task fingerprint
    (TestPortedStanceTerms).
    """

    @staticmethod
    def _neck_indices(env):
        return env._neck_home_qpos_indices

    def test_its_joints_and_target_are_the_neck_and_head(self, env):
        adr = env.model.jnt_qposadr.tolist()
        names = [
            mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_JOINT, adr.index(int(i)))
            for i in env._neck_home_qpos_indices
        ]
        assert names == ["neck_pitch", "neck_yaw", "head_pitch"]
        np.testing.assert_array_equal(env._neck_home_qpos, VelociraptorMongoliensisAnatomicalEnv._NECK_SETTLED_QPOS)
        assert (env.neck_home_pose_weight, env.neck_home_pose_tolerance) == (0.0, 0.05)

    def test_off_its_tolerance_is_inert_bit_for_bit(self):
        """Off, the tolerance changes nothing: observations, rewards and every info value equal, exactly."""
        envs = [
            VelociraptorMongoliensisAnatomicalEnv(reset_noise_scale=0.05),
            VelociraptorMongoliensisAnatomicalEnv(reset_noise_scale=0.05, neck_home_pose_tolerance=0.2),
        ]
        try:
            for env in envs:
                env.reset(seed=3042)
            rng = np.random.default_rng(0)
            for _ in range(200):
                action = rng.uniform(-1.0, 1.0, envs[0].action_space.shape).astype(np.float32)
                (obs_a, reward_a, term_a, trunc_a, info_a), (obs_b, reward_b, term_b, trunc_b, info_b) = (
                    env.step(action) for env in envs
                )
                np.testing.assert_array_equal(obs_a, obs_b)
                assert reward_a == reward_b and (term_a, trunc_a) == (term_b, trunc_b)
                assert info_a.keys() == info_b.keys() and all(
                    np.array_equal(info_a[key], info_b[key]) for key in info_a
                )
                assert not {key for key in info_a if key.startswith(("neck_home_pose", "reward_neck_home_pose"))}
                if term_a or trunc_a:
                    for env in envs:
                        env.reset(seed=3043)
        finally:
            for env in envs:
                env.close()

    def test_on_it_is_the_soft_gaussian_around_the_settled_carriage(self):
        weight, tolerance = 0.25, 0.05
        env = VelociraptorMongoliensisAnatomicalEnv(
            reset_noise_scale=0.0, neck_home_pose_weight=weight, neck_home_pose_tolerance=tolerance
        )
        try:
            env.reset(seed=0)
            indices = env._neck_home_qpos_indices
            env.data.qpos[indices] = env._neck_home_qpos
            info = env._get_reward_info(np.zeros(env.model.nu))[1]
            assert info["neck_home_pose_error"] == pytest.approx(0.0, abs=1e-15)
            assert info["neck_home_pose_quality"] == pytest.approx(1.0)
            assert info["reward_neck_home_pose"] == pytest.approx(weight)

            env.data.qpos[indices[2]] += 0.10  # head_pitch 0.10 rad nose-down: a drooped head
            info = env._get_reward_info(np.zeros(env.model.nu))[1]
            assert info["neck_home_pose_error"] == pytest.approx(0.10 / np.sqrt(3.0))
            assert info["neck_home_pose_quality"] == pytest.approx((2.0 + np.exp(-((0.10 / tolerance) ** 2))) / 3.0)
            assert info["reward_neck_home_pose"] == pytest.approx(weight * info["neck_home_pose_quality"])
        finally:
            env.close()

    def test_the_statue_keeps_the_term_and_a_drooped_head_forfeits_it(self):
        """Zero action holds the carriage; a held head-pitch residual droops the head and pays for it.

        The head_pitch residual +0.3 commands 0.101 rad (the home ctrl is -0.081, the gravity
        preload); the head settles 0.178 rad nose-down and the window's mean quality falls to
        0.667, where the statue's is 1.000 (noise-free, seed 0, steps 200-299, measured).
        """
        qualities = {}
        for label, residual in (("statue", 0.0), ("drooped", 0.3)):
            env = VelociraptorMongoliensisAnatomicalEnv(reset_noise_scale=0.0, neck_home_pose_weight=0.25)
            try:
                env.reset(seed=0)
                action = np.zeros(env.action_space.shape, dtype=np.float32)
                action[2] = residual  # head_pitch_act
                window = []
                for step in range(300):
                    _, _, terminated, _, info = env.step(action)
                    assert not terminated
                    if step >= 200:
                        window.append(info["neck_home_pose_quality"])
                qualities[label] = float(np.mean(window))
            finally:
                env.close()
        assert qualities["statue"] > 0.999
        assert qualities["drooped"] < 0.8
