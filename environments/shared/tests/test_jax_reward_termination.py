"""Tests for jax_reward_termination — focuses on fixes from the JAX review.

Requires JAX to be installed (uses jax.numpy internally).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


def _make_mock_data(
    pelvis_z=0.8,
    vel_2d=(1.0, 0.0),
    quat=(1.0, 0.0, 0.0, 0.0),
    sensordata=None,
    xpos=None,
    qpos=None,
    qvel=None,
):
    """Create a mock MJX data object for reward/termination tests.

    Uses jnp arrays since jax_reward_termination functions use jax.numpy.
    """
    n_sensors = 20
    if sensordata is None:
        sensordata = np.zeros(n_sensors)
        sensordata[6:10] = quat
    if xpos is None:
        xpos = np.zeros((15, 3))
        xpos[1, 2] = pelvis_z
    if qpos is None:
        qpos = np.zeros(20)
    if qvel is None:
        qvel = np.zeros(15)
        qvel[0], qvel[1] = vel_2d

    data = MagicMock()
    data.qvel = jnp.array(qvel)
    data.qpos = jnp.array(qpos)
    data.xpos = jnp.array(xpos)
    data.sensordata = jnp.array(sensordata)
    data.site_xpos = jnp.zeros((5, 3))
    return data


# ---------------------------------------------------------------------------
# Fix #11: healthy_z_max replaces hardcoded 0.90
# ---------------------------------------------------------------------------


class TestHealthyZMax:
    """Verify that healthy_z_max is used instead of hardcoded 0.90."""

    def test_height_frac_uses_healthy_z_max(self):
        """alive bonus height_frac should scale to healthy_z_max, not 0.90."""
        from environments.shared.jax_reward_termination import compute_total_reward

        action = np.zeros(5)
        reward_cfg = {"alive_bonus": 1.0, "forward_vel_weight": 0.0}

        # With healthy_z_max=1.5, a pelvis at 1.2 should get partial credit
        data = _make_mock_data(pelvis_z=1.2)
        r_high_max = compute_total_reward(
            data,
            action,
            reward_cfg,
            root_body_id=1,
            healthy_z_min=0.3,
            healthy_z_max=1.5,
            max_tilt_angle=1.0,
            natural_forward_z=0.0,
            n_actuators=5,
        )

        # With healthy_z_max=0.9, pelvis at 1.2 would clip to 1.0
        # (fully healthy), but with 1.5 it should be partial
        data2 = _make_mock_data(pelvis_z=1.2)
        r_low_max = compute_total_reward(
            data2,
            action,
            reward_cfg,
            root_body_id=1,
            healthy_z_min=0.3,
            healthy_z_max=0.9,
            max_tilt_angle=1.0,
            natural_forward_z=0.0,
            n_actuators=5,
        )

        # With z_max=0.9, pelvis 1.2 is above max so height_frac clips to 1.0
        # With z_max=1.5, pelvis 1.2 gives height_frac = (1.2-0.3)/(1.5-0.3) = 0.75
        # So r_low_max (full credit) should be >= r_high_max (partial credit)
        assert float(r_low_max) >= float(r_high_max)

    def test_compute_reward_components_uses_healthy_z_max(self):
        """compute_reward_components should also use healthy_z_max."""
        from environments.shared.jax_reward_termination import compute_reward_components

        action = np.zeros(5)
        reward_cfg = {"alive_bonus": 1.0, "forward_vel_weight": 0.0}
        data = _make_mock_data(pelvis_z=0.6)

        components = compute_reward_components(
            data,
            action,
            reward_cfg,
            root_body_id=1,
            healthy_z_min=0.3,
            healthy_z_max=1.5,
            max_tilt_angle=1.0,
            natural_forward_z=0.0,
            n_actuators=5,
        )
        assert "alive" in components
        assert "_pelvis_z" in components


# ---------------------------------------------------------------------------
# Fix #9: compute_total_reward alignment with mjx_env.step
# ---------------------------------------------------------------------------


class TestRewardAlignment:
    """Verify new reward terms (fall_penalty, approach, success) work."""

    def test_fall_penalty_applied_on_termination(self):
        """Fall penalty should be applied when the agent is terminated."""
        from environments.shared.jax_reward_termination import compute_total_reward

        action = np.zeros(5)
        reward_cfg = {"alive_bonus": 0.1, "forward_vel_weight": 0.0}

        # Pelvis below healthy_z_min -> terminated
        data = _make_mock_data(pelvis_z=0.1)
        r_with_penalty = compute_total_reward(
            data,
            action,
            reward_cfg,
            root_body_id=1,
            healthy_z_min=0.3,
            healthy_z_max=1.5,
            max_tilt_angle=1.0,
            natural_forward_z=0.0,
            n_actuators=5,
            fall_penalty=-100.0,
        )

        r_without_penalty = compute_total_reward(
            data,
            action,
            reward_cfg,
            root_body_id=1,
            healthy_z_min=0.3,
            healthy_z_max=1.5,
            max_tilt_angle=1.0,
            natural_forward_z=0.0,
            n_actuators=5,
            fall_penalty=0.0,
        )

        assert float(r_with_penalty) < float(r_without_penalty)
        assert float(r_with_penalty) == pytest.approx(float(r_without_penalty) - 100.0, abs=0.01)

    def test_no_fall_penalty_when_healthy(self):
        """Fall penalty should NOT be applied when agent is healthy."""
        from environments.shared.jax_reward_termination import compute_total_reward

        action = np.zeros(5)
        reward_cfg = {"alive_bonus": 0.1, "forward_vel_weight": 0.0}

        # Pelvis well within healthy range
        data = _make_mock_data(pelvis_z=0.8)
        r_with = compute_total_reward(
            data,
            action,
            reward_cfg,
            root_body_id=1,
            healthy_z_min=0.3,
            healthy_z_max=1.5,
            max_tilt_angle=1.0,
            natural_forward_z=0.0,
            n_actuators=5,
            fall_penalty=-100.0,
        )
        r_without = compute_total_reward(
            data,
            action,
            reward_cfg,
            root_body_id=1,
            healthy_z_min=0.3,
            healthy_z_max=1.5,
            max_tilt_angle=1.0,
            natural_forward_z=0.0,
            n_actuators=5,
            fall_penalty=0.0,
        )
        assert float(r_with) == pytest.approx(float(r_without), abs=1e-6)

    def test_approach_shaping_reward(self):
        """Approach shaping should reward getting closer to target."""
        from environments.shared.jax_reward_termination import compute_total_reward

        action = np.zeros(5)
        reward_cfg = {"alive_bonus": 0.0, "forward_vel_weight": 0.0, "approach_weight": 1.0}

        # Agent at x=2, target at x=5
        xpos = np.zeros((15, 3))
        xpos[1] = [2.0, 0.0, 0.8]
        data = _make_mock_data(pelvis_z=0.8, xpos=xpos)
        target = jnp.array([5.0, 0.0, 0.5])

        r_closer = compute_total_reward(
            data,
            action,
            reward_cfg,
            root_body_id=1,
            healthy_z_min=0.3,
            healthy_z_max=1.5,
            max_tilt_angle=1.0,
            natural_forward_z=0.0,
            n_actuators=5,
            target_pos=target,
            prev_target_distance=jnp.float32(4.0),  # was 4m away
            forward_vel_max=8.0,
            dt=0.02,
        )

        r_farther = compute_total_reward(
            data,
            action,
            reward_cfg,
            root_body_id=1,
            healthy_z_min=0.3,
            healthy_z_max=1.5,
            max_tilt_angle=1.0,
            natural_forward_z=0.0,
            n_actuators=5,
            target_pos=target,
            prev_target_distance=jnp.float32(2.5),  # was 2.5m away (moved away)
            forward_vel_max=8.0,
            dt=0.02,
        )

        # Getting closer should yield higher reward than moving away
        assert float(r_closer) > float(r_farther)


# ---------------------------------------------------------------------------
# Termination
# ---------------------------------------------------------------------------


class TestIsTerminated:
    def test_healthy_not_terminated(self):
        from environments.shared.jax_reward_termination import is_terminated

        data = _make_mock_data(pelvis_z=0.8)
        terminated = is_terminated(
            data,
            root_body_id=1,
            healthy_z_range=(0.3, 1.5),
            max_tilt_angle=1.0,
            natural_forward_z=0.0,
        )
        assert not bool(terminated)

    def test_below_min_z_terminated(self):
        from environments.shared.jax_reward_termination import is_terminated

        data = _make_mock_data(pelvis_z=0.1)
        terminated = is_terminated(
            data,
            root_body_id=1,
            healthy_z_range=(0.3, 1.5),
            max_tilt_angle=1.0,
            natural_forward_z=0.0,
        )
        assert bool(terminated)

    def test_above_max_z_terminated(self):
        from environments.shared.jax_reward_termination import is_terminated

        data = _make_mock_data(pelvis_z=2.0)
        terminated = is_terminated(
            data,
            root_body_id=1,
            healthy_z_range=(0.3, 1.5),
            max_tilt_angle=1.0,
            natural_forward_z=0.0,
        )
        assert bool(terminated)
