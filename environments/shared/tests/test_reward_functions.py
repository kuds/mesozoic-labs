"""Tests for the pure reward functions in environments.shared.reward_functions.

These tests verify that the pure functions produce correct results with
NumPy arrays.  When JAX is available, they also verify NumPy/JAX parity.
"""

import numpy as np
import pytest

from environments.shared.reward_functions import (
    check_distance_contact,
    check_height_tilt_termination,
    check_nosedive_termination,
    quat_to_forward_2d,
    quat_to_forward_z,
    quat_to_tilt,
    reward_alive,
    reward_approach_shaping,
    reward_backward_penalty,
    reward_energy,
    reward_forward_velocity,
    reward_height_maintenance,
    reward_idle_penalty,
    reward_posture,
    reward_proximity,
    reward_speed_penalty,
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


class TestDistanceContact:
    def test_close(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([0.1, 0.0, 0.0])
        assert check_distance_contact(a, b, 0.15)

    def test_far(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        assert not check_distance_contact(a, b, 0.15)


# ---------------------------------------------------------------------------
# NumPy / JAX parity (skipped if JAX not installed)
# ---------------------------------------------------------------------------

_has_jax = False
try:
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
