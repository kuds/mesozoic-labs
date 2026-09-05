"""Tests for the pure reward functions in environments.shared.reward_functions.

These tests verify that the pure functions produce correct results with
NumPy arrays.  When JAX is available, they also verify NumPy/JAX parity.
"""

import numpy as np
import pytest

from environments.shared.reward_functions import (
    check_height_tilt_termination,
    check_nosedive_termination,
    quat_to_forward_2d,
    quat_to_forward_z,
    quat_to_tilt,
    reward_action_jerk,
    reward_action_saturation,
    reward_action_smoothness,
    reward_alive,
    reward_approach_shaping,
    reward_backward_penalty,
    reward_bilateral_support,
    reward_energy,
    reward_foot_load_balance,
    reward_forward_velocity,
    reward_head_clearance,
    reward_height_maintenance,
    reward_idle_penalty,
    reward_lean_aware_posture,
    reward_posture,
    reward_proximity,
    reward_soft_home_pose,
    reward_speed_penalty,
    reward_target_centered_height,
)

# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------


class TestQuatToTilt:
    def test_upright_is_zero(self):
        tilt = quat_to_tilt(np.array([1.0, 0.0, 0.0, 0.0]))
        assert tilt == pytest.approx(0.0, abs=1e-6)

    def test_90_degree_pitch(self):
        angle = np.pi / 2
        quat = np.array([np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0])
        assert quat_to_tilt(quat) == pytest.approx(np.pi / 2, abs=0.01)

    def test_yaw_only_is_zero(self):
        angle = np.pi / 4
        quat = np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])
        assert quat_to_tilt(quat) == pytest.approx(0.0, abs=1e-6)


class TestQuatToForward2d:
    def test_identity_gives_x_forward(self):
        fwd = quat_to_forward_2d(np.array([1.0, 0.0, 0.0, 0.0]))
        np.testing.assert_allclose(fwd, [1.0, 0.0], atol=1e-6)

    def test_90_yaw_gives_y_forward(self):
        angle = np.pi / 2
        quat = np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])
        fwd = quat_to_forward_2d(quat)
        np.testing.assert_allclose(fwd, [0.0, 1.0], atol=0.01)


class TestQuatToForwardZ:
    def test_upright_gives_zero(self):
        assert quat_to_forward_z(np.array([1.0, 0.0, 0.0, 0.0])) == pytest.approx(0.0, abs=1e-6)

    def test_pitch_forward_gives_negative(self):
        # 45° pitch forward around Y axis
        angle = np.pi / 4
        quat = np.array([np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0])
        fz = quat_to_forward_z(quat)
        # Forward vector tilts downward → negative Z
        assert fz < -0.1


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------


class TestRewardForwardVelocity:
    def test_positive_forward(self):
        vel = np.array([1.0, 0.0])
        fwd = np.array([1.0, 0.0])
        reward, raw_vel = reward_forward_velocity(vel, fwd, 10.0, 1.0)
        assert reward > 0.0
        assert raw_vel == pytest.approx(1.0)

    def test_backward_gives_negative(self):
        vel = np.array([-1.0, 0.0])
        fwd = np.array([1.0, 0.0])
        reward, _ = reward_forward_velocity(vel, fwd, 10.0, 1.0)
        assert reward < 0.0

    def test_clamped_to_weight(self):
        vel = np.array([100.0, 0.0])
        fwd = np.array([1.0, 0.0])
        reward, _ = reward_forward_velocity(vel, fwd, 10.0, 2.0)
        assert reward == pytest.approx(2.0)


class TestRewardEnergy:
    def test_zero_action_no_penalty(self):
        action = np.zeros(10)
        assert reward_energy(action, 10, 1.0) == pytest.approx(0.0)

    def test_full_action_penalty(self):
        action = np.ones(10)
        penalty = reward_energy(action, 10, 1.0)
        assert penalty == pytest.approx(-1.0)


class TestRewardAlive:
    def test_returns_bonus(self):
        assert reward_alive(0.1) == pytest.approx(0.1)
        assert reward_alive(0.0) == pytest.approx(0.0)


class TestRewardApproachShaping:
    def test_closing_distance(self):
        reward, delta = reward_approach_shaping(5.0, 6.0, 1.0, 10.0, 0.1)
        assert reward > 0.0
        assert delta == pytest.approx(1.0)

    def test_retreating(self):
        reward, delta = reward_approach_shaping(6.0, 5.0, 1.0, 10.0, 0.1)
        assert reward < 0.0

    def test_first_step_zero(self):
        reward, delta = reward_approach_shaping(5.0, None, 1.0, 10.0, 0.1)
        assert reward == 0.0
        assert delta == 0.0


class TestRewardPosture:
    def test_upright_no_penalty(self):
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        reward, tilt = reward_posture(quat, 1.047, 1.0)
        assert reward == pytest.approx(0.0, abs=1e-6)
        assert tilt == pytest.approx(0.0, abs=1e-6)


class TestRewardLeanAwarePosture:
    natural_pitch = 0.35
    natural_forward_z = -np.sin(natural_pitch)
    max_tilt = 1.047

    @staticmethod
    def _pitch_quat(angle: float) -> np.ndarray:
        return np.array([np.cos(angle / 2.0), 0.0, np.sin(angle / 2.0), 0.0])

    @staticmethod
    def _roll_quat(angle: float) -> np.ndarray:
        return np.array([np.cos(angle / 2.0), np.sin(angle / 2.0), 0.0, 0.0])

    @staticmethod
    def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = left
        w2, x2, y2, z2 = right
        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ]
        )

    def _reward(self, quat: np.ndarray, weight: float = 1.0) -> tuple[float, float]:
        return reward_lean_aware_posture(
            quat,
            self.max_tilt,
            weight,
            self.natural_forward_z,
        )

    def test_natural_forward_pitch_minimizes_penalty(self):
        reward, tilt = self._reward(self._pitch_quat(self.natural_pitch))
        assert reward == pytest.approx(0.0, abs=1e-12)
        assert tilt == pytest.approx(self.natural_pitch, abs=1e-6)

    def test_upright_backward_pitch_and_roll_are_penalized(self):
        upright_reward, upright_tilt = self._reward(np.array([1.0, 0.0, 0.0, 0.0]))
        backward_reward, _ = self._reward(self._pitch_quat(-self.natural_pitch))
        natural_pitch = self._pitch_quat(self.natural_pitch)
        rolled_natural_pitch = self._quat_multiply(natural_pitch, self._roll_quat(self.natural_pitch))
        roll_reward, _ = self._reward(rolled_natural_pitch)

        assert upright_reward < 0.0
        assert backward_reward < upright_reward
        assert roll_reward < 0.0
        assert upright_tilt == pytest.approx(0.0, abs=1e-6)

    def test_world_yaw_does_not_change_reward(self):
        pitch = self._pitch_quat(self.natural_pitch)
        yaw_angle = 1.2
        yaw = np.array([np.cos(yaw_angle / 2.0), 0.0, 0.0, np.sin(yaw_angle / 2.0)])
        yawed_pitch = self._quat_multiply(yaw, pitch)

        reward, _ = self._reward(pitch)
        yawed_reward, _ = self._reward(yawed_pitch)
        assert yawed_reward == pytest.approx(reward, abs=1e-12)

    def test_quaternion_sign_does_not_change_reward(self):
        quat = self._pitch_quat(0.1)
        reward, tilt = self._reward(quat)
        negated_reward, negated_tilt = self._reward(-quat)
        assert negated_reward == pytest.approx(reward, abs=1e-12)
        assert negated_tilt == pytest.approx(tilt, abs=1e-12)

    def test_reward_is_bounded_and_zero_weight_is_exactly_zero(self):
        reward, _ = self._reward(self._pitch_quat(np.pi / 2.0), weight=2.0)
        zero_weight_reward, _ = self._reward(self._pitch_quat(-0.5), weight=0.0)
        assert -2.0 <= reward <= 0.0
        assert zero_weight_reward == 0.0


class TestRewardProximity:
    def test_at_target(self):
        reward, prox = reward_proximity(0.0, 10.0, 1.0)
        assert reward == pytest.approx(1.0)
        assert prox == pytest.approx(1.0)

    def test_at_max_distance(self):
        reward, prox = reward_proximity(10.0, 10.0, 1.0)
        assert reward == pytest.approx(0.0)

    def test_beyond_max(self):
        reward, prox = reward_proximity(15.0, 10.0, 1.0)
        assert reward == pytest.approx(0.0)


class TestRewardHeightMaintenance:
    def test_at_target(self):
        r = reward_height_maintenance(0.9, 0.5, 0.9, 1.0)
        assert r == pytest.approx(1.0)

    def test_at_min(self):
        r = reward_height_maintenance(0.5, 0.5, 0.9, 1.0)
        assert r == pytest.approx(0.0)


class TestStageOneStancePrimitives:
    def test_bilateral_support_uses_weaker_saturated_foot(self):
        reward, quality = reward_bilateral_support(np.array([125.0, 40.0]), 100.0, 2.0)

        assert quality == pytest.approx(0.4)
        assert reward == pytest.approx(0.8)

    def test_load_balance_is_zero_when_equal_and_one_when_unilateral(self):
        equal_reward, equal_imbalance = reward_foot_load_balance(np.array([80.0, 80.0]), 1.5)
        unilateral_reward, unilateral_imbalance = reward_foot_load_balance(np.array([80.0, 0.0]), 1.5)

        assert equal_imbalance == pytest.approx(0.0)
        assert equal_reward == pytest.approx(0.0)
        assert unilateral_imbalance == pytest.approx(1.0)
        assert unilateral_reward == pytest.approx(-1.5)

    def test_airborne_is_maximally_imbalanced_not_perfectly_balanced(self):
        """Regression: ``[0, 0]`` used to return imbalance 0.0.

        ``|R - L| / (R + L + 1e-8)`` evaluates to zero when both feet read
        zero, so being airborne scored the same as standing evenly and
        strictly better than honest single support -- which pays the full
        penalty. On a balance stage that makes flight the cheapest way to
        avoid the term, and a policy off the ground cannot reject a
        disturbance at all. See docs/STAGE1_SPLIT_PLAN.md section 7.1.
        """
        reward, imbalance = reward_foot_load_balance(np.array([0.0, 0.0]), 1.5)
        assert imbalance == pytest.approx(1.0)
        assert reward == pytest.approx(-1.5)

    def test_airborne_is_no_cheaper_than_single_support(self):
        """The ordering the stage-1 reward depends on."""
        _, airborne = reward_foot_load_balance(np.array([0.0, 0.0]), 1.5)
        _, single = reward_foot_load_balance(np.array([80.0, 0.0]), 1.5)
        _, even = reward_foot_load_balance(np.array([80.0, 80.0]), 1.5)
        assert even < single <= airborne

    def test_min_support_force_denies_credit_for_a_grazing_contact(self):
        forces = np.array([0.25, 0.25])
        _, unguarded = reward_foot_load_balance(forces, 1.5)
        _, guarded = reward_foot_load_balance(forces, 1.5, 10.0)
        assert unguarded == pytest.approx(0.0)
        assert guarded == pytest.approx(1.0)

    def test_default_leaves_every_loaded_state_unchanged(self):
        """The default 0.0 threshold must close only the exact [0, 0] case."""
        for forces in ([80.0, 80.0], [80.0, 0.0], [120.0, 40.0], [1e-6, 1e-6]):
            _, imbalance = reward_foot_load_balance(np.array(forces), 1.0)
            expected = abs(forces[0] - forces[1]) / (forces[0] + forces[1] + 1e-8)
            assert imbalance == pytest.approx(expected)

    def test_soft_home_pose_reports_rms_and_mean_joint_quality(self):
        tolerance = 0.5
        joint_positions = np.array([0.0, tolerance])
        home_positions = np.zeros(2)
        reward, error, quality = reward_soft_home_pose(
            joint_positions,
            home_positions,
            tolerance,
            2.0,
        )

        assert error == pytest.approx(tolerance / np.sqrt(2.0))
        assert quality == pytest.approx((1.0 + np.exp(-1.0)) / 2.0)
        assert reward == pytest.approx(2.0 * quality)

    @pytest.mark.parametrize(
        ("head_height", "expected_quality"),
        (
            (0.12, 0.0),
            (0.36, 0.5),
            (0.60, 1.0),
            (0.80, 1.0),
        ),
    )
    def test_head_clearance_smoothsteps_from_floor_to_target(self, head_height, expected_quality):
        reward, quality = reward_head_clearance(
            head_height,
            target_height=0.60,
            tolerance=0.48,
            weight=1.25,
        )

        assert quality == pytest.approx(expected_quality)
        assert reward == pytest.approx(1.25 * expected_quality)

    def test_target_centered_height_is_bounded_and_symmetric(self):
        at_target, target_error, target_quality = reward_target_centered_height(0.926, 0.926, 0.08, 1.0)
        below, below_error, below_quality = reward_target_centered_height(0.846, 0.926, 0.08, 1.0)
        above, above_error, above_quality = reward_target_centered_height(1.006, 0.926, 0.08, 1.0)

        assert at_target == pytest.approx(1.0)
        assert target_error == pytest.approx(0.0)
        assert target_quality == pytest.approx(1.0)
        assert below_error == pytest.approx(0.08)
        assert above_error == pytest.approx(0.08)
        assert below_quality == pytest.approx(np.exp(-1.0))
        assert above_quality == pytest.approx(below_quality)
        assert below == pytest.approx(below_quality)
        assert above == pytest.approx(above_quality)


class TestRewardBackwardPenalty:
    def test_no_backward(self):
        reward, bw = reward_backward_penalty(1.0, 10.0, 1.0)
        assert reward == 0.0
        assert bw == 0.0

    def test_backward(self):
        reward, bw = reward_backward_penalty(-5.0, 10.0, 1.0)
        assert reward < 0.0
        assert bw == pytest.approx(5.0)


class TestRewardSpeedPenalty:
    def test_below_threshold(self):
        vel = np.array([0.05, 0.0])
        reward, speed = reward_speed_penalty(vel, 1.0, 0.1)
        assert reward == 0.0

    def test_above_threshold(self):
        vel = np.array([1.0, 0.0])
        reward, speed = reward_speed_penalty(vel, 1.0, 0.1)
        assert reward < 0.0


class TestRewardIdlePenalty:
    def test_moving_no_penalty(self):
        vel = np.array([1.0, 0.0])
        reward, speed = reward_idle_penalty(vel, 1.0, 0.05)
        assert reward == 0.0

    def test_stationary_penalised(self):
        vel = np.array([0.0, 0.0])
        reward, speed = reward_idle_penalty(vel, 1.0, 0.05)
        assert reward == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Termination checks
# ---------------------------------------------------------------------------


class TestHeightTiltTermination:
    def test_healthy(self):
        terminated, reason = check_height_tilt_termination(0.5, 0.1, (0.3, 1.0), 1.047)
        assert not terminated
        assert reason is None

    def test_fallen(self):
        terminated, reason = check_height_tilt_termination(0.1, 0.1, (0.3, 1.0), 1.047)
        assert terminated
        assert reason == "fallen"

    def test_too_high(self):
        terminated, reason = check_height_tilt_termination(2.0, 0.1, (0.3, 1.0), 1.047)
        assert terminated
        assert reason == "too_high"

    def test_excessive_tilt(self):
        terminated, reason = check_height_tilt_termination(0.5, 2.0, (0.3, 1.0), 1.047)
        assert terminated
        assert reason == "excessive_tilt"


class TestNosediveTermination:
    def test_normal(self):
        terminated, _ = check_nosedive_termination(-0.1, -0.17)
        assert not terminated

    def test_nosedive(self):
        terminated, reason = check_nosedive_termination(-0.8, -0.17)
        assert terminated
        assert reason == "nosedive"


# ---------------------------------------------------------------------------
# NumPy / JAX parity (skipped if JAX not installed)
# ---------------------------------------------------------------------------

_has_jax = False
try:
    import jax
    import jax.numpy as jnp

    _has_jax = True
except ImportError:
    pass


@pytest.mark.skipif(not _has_jax, reason="JAX not installed")
class TestNumpyJaxParity:
    """Verify that pure functions produce identical results with NumPy and JAX arrays."""

    def test_reward_forward_velocity_parity(self):
        vel_np = np.array([1.5, 0.3])
        fwd_np = np.array([0.8, 0.6])
        vel_jax = jnp.array(vel_np)
        fwd_jax = jnp.array(fwd_np)

        r_np, v_np = reward_forward_velocity(vel_np, fwd_np, 10.0, 1.0)
        r_jax, v_jax = reward_forward_velocity(vel_jax, fwd_jax, 10.0, 1.0)
        assert r_np == pytest.approx(r_jax, abs=1e-5)
        assert v_np == pytest.approx(v_jax, abs=1e-5)

    def test_quat_to_tilt_parity(self):
        quat = np.array([0.9, 0.1, 0.2, 0.3])
        quat = quat / np.linalg.norm(quat)
        tilt_np = quat_to_tilt(quat)
        tilt_jax = quat_to_tilt(jnp.array(quat))
        assert tilt_np == pytest.approx(tilt_jax, abs=1e-5)

    def test_reward_energy_parity(self):
        action = np.array([0.5, -0.3, 0.8, 0.1])
        r_np = reward_energy(action, 4, 0.01)
        r_jax = reward_energy(jnp.array(action), 4, 0.01)
        assert r_np == pytest.approx(r_jax, abs=1e-6)

    def test_lean_aware_posture_parity(self):
        natural_pitch = 0.35
        target = -np.sin(natural_pitch)
        quat = np.array([np.cos(0.2), 0.0, np.sin(0.2), 0.0])
        r_np, tilt_np = reward_lean_aware_posture(quat, 1.047, 1.5, target)
        r_jax, tilt_jax = reward_lean_aware_posture(jnp.array(quat), 1.047, 1.5, target)
        assert r_np == pytest.approx(r_jax, abs=1e-6)
        assert tilt_np == pytest.approx(tilt_jax, abs=1e-6)

    def test_lean_aware_posture_gradient_is_finite_at_target(self):
        natural_pitch = 0.35
        target = -np.sin(natural_pitch)

        def posture_reward_for_pitch(pitch):
            quat = jnp.array([jnp.cos(pitch / 2.0), 0.0, jnp.sin(pitch / 2.0), 0.0])
            reward, _ = reward_lean_aware_posture(quat, 1.047, 1.5, target)
            return reward

        gradient = jax.grad(posture_reward_for_pitch)(jnp.float32(natural_pitch))
        assert np.isfinite(float(gradient))


class TestFootLoadBalanceMonotoneOrdering:
    """Airborne must be strictly worse than honest single support.

    Before ``airborne_penalty_weight`` the two states both scored ``-weight``,
    a flat region with no gradient out of the air on the stage whose entire
    job is to stay on the ground.  Run ``20260801_021545`` ended at 28.4%
    unsupported duty against the statue's 0.000, so the tie was not academic.
    """

    WEIGHT = 0.3
    BILATERAL = 0.6
    MIN_SUPPORT = 42.0
    AIRBORNE = 0.3

    def _score(self, right, left, bilateral_credit):
        reward, imbalance = reward_foot_load_balance(
            np.array([right, left]), self.WEIGHT, self.MIN_SUPPORT, self.AIRBORNE
        )
        return float(reward) + bilateral_credit, float(imbalance)

    def test_ordering_is_strictly_monotone(self):
        even, _ = self._score(420.0, 420.0, self.BILATERAL)
        single, _ = self._score(840.0, 0.0, 0.0)
        airborne, _ = self._score(0.0, 0.0, 0.0)
        assert even > single > airborne, f"expected even > single > airborne, got {even} {single} {airborne}"

    def test_grazing_contact_earns_no_credit(self):
        """A token contact below the support threshold counts as airborne."""
        grazing, imbalance = self._score(0.05, 0.0, 0.0)
        airborne, _ = self._score(0.0, 0.0, 0.0)
        assert grazing == pytest.approx(airborne)
        assert imbalance == pytest.approx(1.0)

    def test_diagnostic_stays_in_unit_range(self):
        """The ordering lives in the reward; the diagnostic keeps [0, 1]."""
        for right, left in ((420.0, 420.0), (840.0, 0.0), (0.0, 0.0), (0.05, 0.0)):
            _, imbalance = self._score(right, left, 0.0)
            assert 0.0 <= imbalance <= 1.0

    def test_defaults_preserve_the_previous_behaviour(self):
        """Both new parameters default to the pre-existing semantics."""
        legacy, _ = reward_foot_load_balance(np.array([840.0, 0.0]), self.WEIGHT)
        assert float(legacy) == pytest.approx(-self.WEIGHT)


class TestActionJerkIsFrequencyAware:
    """The second difference charges buzz that the first difference cannot see.

    Measured motivation (PLANT_VALIDATION section 11.2): from the best to the
    final checkpoint of run ``20260731_132102``, ``action_delta`` FELL
    12.0 -> 10.5 and the smoothness penalty IMPROVED while toe-motion power
    above 4 Hz DOUBLED.
    """

    N = 21

    def _pair(self, a2, a1, a0):
        _, jerk = reward_action_jerk(a2, a1, a0, self.N, 1.0)
        _, delta = reward_action_smoothness(a2, a1, self.N, 1.0)
        return float(jerk), float(delta)

    def test_constant_ramp_has_zero_jerk_but_nonzero_delta(self):
        """The exact blindness inversion: a ramp moves more than a slow wave."""
        ramp = [np.full(self.N, 0.5 * t) for t in (0, 1, 2)]
        jerk, delta = self._pair(ramp[2], ramp[1], ramp[0])
        assert jerk == pytest.approx(0.0)
        assert delta > 0.0

    def test_alternating_buzz_dominates(self):
        """A Nyquist-rate limit cycle is maximally penalised."""
        buzz = [np.full(self.N, (-1.0) ** t) for t in (0, 1, 2)]
        buzz_jerk, _ = self._pair(buzz[2], buzz[1], buzz[0])
        ramp = [np.full(self.N, 0.5 * t) for t in (0, 1, 2)]
        ramp_jerk, _ = self._pair(ramp[2], ramp[1], ramp[0])
        assert buzz_jerk > 100.0 * max(ramp_jerk, 1e-9)

    def test_first_two_steps_are_never_charged(self):
        a = np.zeros(self.N)
        assert reward_action_jerk(a, None, None, self.N, 1.0) == (0.0, 0.0)
        assert reward_action_jerk(a, a, None, self.N, 1.0) == (0.0, 0.0)

    def test_zero_weight_is_a_no_op(self):
        buzz = [np.full(self.N, (-1.0) ** t) for t in (0, 1, 2)]
        reward, _ = reward_action_jerk(buzz[2], buzz[1], buzz[0], self.N, 0.0)
        assert float(reward) == pytest.approx(0.0)


class TestActionSaturation:
    def test_zero_below_threshold(self):
        reward, fraction = reward_action_saturation(np.array([0.0, 0.5, -0.89, 0.9]), 1.0, 0.9)
        assert float(reward) == pytest.approx(0.0)
        assert float(fraction) == pytest.approx(0.0)

    def test_full_pin_pays_full_rate(self):
        reward, fraction = reward_action_saturation(np.array([1.0, -1.0]), 0.5, 0.9)
        assert float(fraction) == pytest.approx(1.0)
        assert float(reward) == pytest.approx(-0.5)

    def test_linear_ramp_and_mean_across_actuators(self):
        # One actuator halfway up the ramp (|a|=0.95 with threshold 0.9),
        # three quiet: fraction = 0.5/4.
        action = np.array([0.95, 0.0, 0.0, 0.0])
        _, fraction = reward_action_saturation(action, 1.0, 0.9)
        assert float(fraction) == pytest.approx(0.125)

    def test_clip_beyond_range(self):
        # Raw policy outputs can exceed [-1, 1]; the ramp saturates at 1.
        _, fraction = reward_action_saturation(np.array([3.0]), 1.0, 0.9)
        assert float(fraction) == pytest.approx(1.0)


class TestSoftHomePoseBroadTail:
    def test_default_preserves_single_gaussian(self):
        positions = np.array([0.3, -0.1])
        home = np.zeros(2)
        legacy = np.mean(np.exp(-np.square(positions / 0.1)))
        reward, _, quality = reward_soft_home_pose(positions, home, 0.1, 0.5)
        assert float(quality) == pytest.approx(legacy)
        assert float(reward) == pytest.approx(0.5 * legacy)

    def test_maximum_unchanged_at_home(self):
        home = np.array([0.2, -0.4])
        _, _, quality = reward_soft_home_pose(home, home, 0.1, 1.0, 0.25, 6.0)
        assert float(quality) == pytest.approx(1.0)

    def test_broad_tail_keeps_gradient_at_long_range(self):
        # At 5 tolerance widths the narrow Gaussian is numerically dead;
        # the mixture must still strictly prefer the closer pose.
        home = np.zeros(3)
        far = np.full(3, 0.50)
        nearer = np.full(3, 0.45)
        _, _, q_far = reward_soft_home_pose(far, home, 0.1, 1.0, 0.25, 6.0)
        _, _, q_nearer = reward_soft_home_pose(nearer, home, 0.1, 1.0, 0.25, 6.0)
        assert float(q_nearer) > float(q_far) + 1e-4
        # And without the tail, the same comparison is flat to 1e-6.
        _, _, q_far_n = reward_soft_home_pose(far, home, 0.1, 1.0)
        _, _, q_nearer_n = reward_soft_home_pose(nearer, home, 0.1, 1.0)
        assert abs(float(q_nearer_n) - float(q_far_n)) < 1e-6


@pytest.mark.skipif(not _has_jax, reason="JAX not installed")
class TestNewTermsNumpyJaxParity:
    def test_action_saturation_parity(self):
        action = np.array([0.97, -1.0, 0.2, -0.85])
        r_np, f_np = reward_action_saturation(action, 0.5, 0.9)
        r_jax, f_jax = reward_action_saturation(jnp.array(action), 0.5, 0.9)
        assert float(r_np) == pytest.approx(float(r_jax), abs=1e-6)
        assert float(f_np) == pytest.approx(float(f_jax), abs=1e-6)

    def test_soft_home_pose_broad_tail_parity(self):
        positions = np.array([0.31, -0.52, 0.07, 0.44])
        home = np.array([-0.045, 0.0, -0.433, 1.342])
        r_np, e_np, q_np = reward_soft_home_pose(positions, home, 0.1, 0.5, 0.25, 6.0)
        r_jax, e_jax, q_jax = reward_soft_home_pose(jnp.array(positions), jnp.array(home), 0.1, 0.5, 0.25, 6.0)
        assert float(r_np) == pytest.approx(float(r_jax), abs=1e-6)
        assert float(e_np) == pytest.approx(float(e_jax), abs=1e-6)
        assert float(q_np) == pytest.approx(float(q_jax), abs=1e-6)
