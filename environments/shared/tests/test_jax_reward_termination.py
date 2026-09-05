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
# Velociraptor natural-pitch posture parity
# ---------------------------------------------------------------------------


class TestLeanAwarePosture:
    def test_total_and_detailed_rewards_use_natural_posture_target(self):
        from environments.shared.jax_reward_termination import compute_reward_components, compute_total_reward

        natural_pitch = 0.35
        target_forward_z = -np.sin(natural_pitch)
        target_quat = (np.cos(natural_pitch / 2.0), 0.0, np.sin(natural_pitch / 2.0), 0.0)
        reward_cfg = {
            "alive_bonus": 0.0,
            "energy_penalty_weight": 0.0,
            "forward_vel_weight": 0.0,
            "posture_weight": 1.5,
        }
        action = np.zeros(5)

        target_data = _make_mock_data(quat=target_quat)
        target_total = compute_total_reward(
            target_data,
            action,
            reward_cfg,
            root_body_id=1,
            healthy_z_min=0.3,
            healthy_z_max=1.5,
            max_tilt_angle=1.047,
            natural_forward_z=-0.342,
            posture_target_forward_z=target_forward_z,
            n_actuators=5,
        )
        target_components = compute_reward_components(
            target_data,
            action,
            reward_cfg,
            root_body_id=1,
            healthy_z_min=0.3,
            healthy_z_max=1.5,
            max_tilt_angle=1.047,
            natural_forward_z=-0.342,
            posture_target_forward_z=target_forward_z,
            n_actuators=5,
        )

        upright_data = _make_mock_data(quat=(1.0, 0.0, 0.0, 0.0))
        upright_components = compute_reward_components(
            upright_data,
            action,
            reward_cfg,
            root_body_id=1,
            healthy_z_min=0.3,
            healthy_z_max=1.5,
            max_tilt_angle=1.047,
            natural_forward_z=-0.342,
            posture_target_forward_z=target_forward_z,
            n_actuators=5,
        )

        assert float(target_total) == pytest.approx(float(target_components["posture"]), abs=1e-6)
        assert float(target_components["posture"]) == pytest.approx(0.0, abs=1e-6)
        assert float(upright_components["posture"]) < float(target_components["posture"])

    def test_none_target_preserves_upright_posture_reward(self):
        from environments.shared.jax_reward_termination import compute_reward_components

        components = compute_reward_components(
            _make_mock_data(quat=(1.0, 0.0, 0.0, 0.0)),
            np.zeros(5),
            {
                "alive_bonus": 0.0,
                "energy_penalty_weight": 0.0,
                "forward_vel_weight": 0.0,
                "posture_weight": 1.5,
            },
            root_body_id=1,
            healthy_z_min=0.3,
            healthy_z_max=1.5,
            max_tilt_angle=1.047,
            natural_forward_z=-0.169,
            posture_target_forward_z=None,
            n_actuators=5,
        )

        assert float(components["posture"]) == pytest.approx(0.0, abs=1e-7)


# ---------------------------------------------------------------------------
# T. rex Stage-1 stance reward parity
# ---------------------------------------------------------------------------


class TestStageOneStanceRewards:
    @staticmethod
    def _isolated_reward_cfg(**overrides):
        return {
            "forward_vel_weight": 0.0,
            "alive_bonus": 0.0,
            "energy_penalty_weight": 0.0,
            "posture_weight": 0.0,
            **overrides,
        }

    def test_support_conditioned_alive_is_opt_in_and_preserves_legacy_default(self):
        from environments.shared.jax_reward_termination import compute_reward_components, compute_total_reward

        sensordata = np.zeros(20)
        sensordata[6:10] = (1.0, 0.0, 0.0, 0.0)
        sensordata[10:12] = (100.0, 0.0)
        data = _make_mock_data(pelvis_z=0.9, sensordata=sensordata)
        base_cfg = self._isolated_reward_cfg(
            alive_bonus=2.0,
            foot_contact_gate=1.0,
        )
        common = dict(
            root_body_id=1,
            healthy_z_min=0.5,
            healthy_z_max=1.5,
            max_tilt_angle=1.0,
            natural_forward_z=0.0,
            n_actuators=5,
            foot_sensor_groups=((10,), (11,)),
        )

        legacy = compute_total_reward(data, np.zeros(5), base_cfg, **common)
        explicit_zero = compute_total_reward(
            data,
            np.zeros(5),
            {**base_cfg, "support_conditioned_alive_fraction": 0.0},
            **common,
        )
        conditioned_cfg = {
            **base_cfg,
            "support_conditioned_alive_fraction": 0.5,
            "foot_contact_saturation_force": 100.0,
        }
        conditioned = compute_total_reward(data, np.zeros(5), conditioned_cfg, **common)
        conditioned_components = compute_reward_components(
            data,
            np.zeros(5),
            conditioned_cfg,
            **common,
        )

        # Legacy MJX gate: height fraction 0.4 times any-foot contact.
        assert float(legacy) == pytest.approx(0.8)
        assert float(explicit_zero) == pytest.approx(float(legacy))
        # Opt-in parity gate: (1-f) + f*bilateral_quality = 0.5.
        assert float(conditioned) == pytest.approx(1.0)
        assert float(conditioned_components["_raw_alive"]) == pytest.approx(2.0)
        assert float(conditioned_components["_alive_gate"]) == pytest.approx(0.5)
        assert float(conditioned_components["alive"]) == pytest.approx(1.0)

    def test_composer_exposes_and_sums_all_new_components(self):
        from environments.shared.jax_reward_termination import compute_reward_components, compute_total_reward

        sensordata = np.zeros(20)
        sensordata[6:10] = (1.0, 0.0, 0.0, 0.0)
        sensordata[10:12] = (100.0, 25.0)
        qpos = np.zeros(20)
        qpos[2:4] = (0.0, 0.35)
        data = _make_mock_data(pelvis_z=0.926, sensordata=sensordata, qpos=qpos)
        site_xpos = np.zeros((5, 3))
        site_xpos[0, 2] = 0.36
        data.site_xpos = jnp.asarray(site_xpos)

        reward_cfg = self._isolated_reward_cfg(
            alive_bonus=2.0,
            bilateral_support_weight=1.2,
            foot_contact_saturation_force=100.0,
            foot_load_balance_weight=0.5,
            support_conditioned_alive_fraction=0.5,
            leg_home_pose_weight=0.8,
            leg_home_pose_tolerance=0.35,
            head_clearance_weight=0.6,
            head_clearance_target=0.60,
            head_clearance_tolerance=0.48,
            neck_posture_weight=0.4,
            neck_posture_tolerance=0.35,
            height_weight=1.0,
            height_target_tolerance=0.08,
        )
        common = dict(
            root_body_id=1,
            healthy_z_min=0.5,
            healthy_z_max=1.5,
            target_standing_z=0.926,
            max_tilt_angle=1.0,
            natural_forward_z=0.0,
            n_actuators=5,
            foot_sensor_groups=((10,), (11,)),
            leg_home_pose_qpos_indices=(2, 3),
            leg_home_pose_targets=jnp.zeros(2),
            neck_posture_qpos_indices=(4, 5, 6),
            neck_posture_targets=jnp.zeros(3),
            head_clearance_site_id=0,
        )

        total = compute_total_reward(data, np.zeros(5), reward_cfg, **common)
        components = compute_reward_components(data, np.zeros(5), reward_cfg, **common)
        component_names = (
            "forward",
            "alive",
            "energy",
            "posture",
            "foot_contact",
            "bilateral_support",
            "foot_load_balance",
            "height",
            "leg_home_pose",
            "head_clearance",
            "neck_posture",
        )

        assert float(components["_bilateral_support_quality"]) == pytest.approx(0.25)
        assert float(components["_foot_load_imbalance"]) == pytest.approx(0.6)
        assert float(components["_alive_gate"]) == pytest.approx(0.625)
        assert float(components["_head_clearance_quality"]) == pytest.approx(0.5)
        assert float(components["_height_error"]) == pytest.approx(0.0)
        assert float(components["_height_quality"]) == pytest.approx(1.0)
        assert float(components["_neck_posture_error"]) == pytest.approx(0.0)
        assert float(components["_neck_posture_quality"]) == pytest.approx(1.0)
        assert float(total) == pytest.approx(sum(float(components[name]) for name in component_names), abs=1e-6)

    def test_zero_height_tolerance_retains_legacy_one_sided_quality(self):
        from environments.shared.jax_reward_termination import compute_reward_components

        data = _make_mock_data(pelvis_z=0.7)
        base_cfg = self._isolated_reward_cfg(height_weight=2.0)
        common = dict(
            root_body_id=1,
            healthy_z_min=0.5,
            healthy_z_max=1.5,
            target_standing_z=0.9,
            max_tilt_angle=1.0,
            natural_forward_z=0.0,
            n_actuators=5,
        )

        legacy = compute_reward_components(data, np.zeros(5), base_cfg, **common)
        explicit_zero = compute_reward_components(
            data,
            np.zeros(5),
            {**base_cfg, "height_target_tolerance": 0.0},
            **common,
        )
        bounded = compute_reward_components(
            data,
            np.zeros(5),
            {**base_cfg, "height_target_tolerance": 0.1},
            **common,
        )

        assert float(legacy["height"]) == pytest.approx(1.0)
        assert float(explicit_zero["height"]) == pytest.approx(float(legacy["height"]))
        assert float(legacy["_height_quality"]) == pytest.approx(0.5)
        assert float(bounded["_height_error"]) == pytest.approx(0.2)
        assert float(bounded["_height_quality"]) == pytest.approx(np.exp(-4.0))


# ---------------------------------------------------------------------------
# Fix #9: compute_total_reward alignment with mjx_env.step
# ---------------------------------------------------------------------------


class TestRewardAlignment:
    """Verify new reward terms (fall_penalty, approach, success) work."""

    def test_drift_is_measured_from_nonzero_spawn_for_total_and_components(self):
        from environments.shared.jax_reward_termination import (
            compute_reward_components,
            compute_total_reward,
        )

        xpos = np.zeros((15, 3))
        xpos[1] = [1.5, -0.5, 0.8]
        data = _make_mock_data(pelvis_z=0.8, vel_2d=(0.0, 0.0), xpos=xpos)
        reward_cfg = {
            "alive_bonus": 0.0,
            "forward_vel_weight": 0.0,
            "energy_penalty_weight": 0.0,
            "posture_weight": 0.0,
            "drift_penalty_weight": 1.0,
        }
        common = dict(
            root_body_id=1,
            healthy_z_min=0.3,
            healthy_z_max=1.5,
            max_tilt_angle=1.0,
            natural_forward_z=0.0,
            n_actuators=5,
            initial_pos_2d=jnp.array([1.0, -0.5]),
        )

        total = compute_total_reward(data, np.zeros(5), reward_cfg, **common)
        components = compute_reward_components(data, np.zeros(5), reward_cfg, **common)

        # 0.5 m from spawn -> -(0.5 / 2)^2.
        assert float(total) == pytest.approx(-0.0625)
        assert float(components["drift"]) == pytest.approx(-0.0625)

    def test_additional_body_or_site_termination_applies_fall_penalty_once(self):
        from environments.shared.jax_reward_termination import compute_total_reward

        data = _make_mock_data(pelvis_z=0.8, vel_2d=(0.0, 0.0))
        reward_cfg = {
            "alive_bonus": 0.0,
            "forward_vel_weight": 0.0,
            "energy_penalty_weight": 0.0,
            "posture_weight": 0.0,
        }
        common = dict(
            root_body_id=1,
            healthy_z_min=0.3,
            healthy_z_max=1.5,
            max_tilt_angle=1.0,
            natural_forward_z=0.0,
            n_actuators=5,
            fall_penalty=-10.0,
        )

        healthy = compute_total_reward(
            data,
            np.zeros(5),
            reward_cfg,
            additional_terminated=False,
            **common,
        )
        extra_terminated = compute_total_reward(
            data,
            np.zeros(5),
            reward_cfg,
            additional_terminated=True,
            **common,
        )

        assert float(extra_terminated) == pytest.approx(float(healthy) - 10.0)

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


class TestAggregatedFootForcesOverride:
    """The composers must price the substep-MIN forces when a stepped caller
    supplies them, and keep single-state semantics when it does not.

    The training backends aggregate touch force across the frame_skip physics
    substeps; ``evaluate_policy_cpu`` passes the aggregate through this
    override so eval scores the same quantity training optimizes.  The
    single-state default is what the SB3-vs-JAX parity test scores.
    """

    _COMMON = dict(
        root_body_id=1,
        healthy_z_min=0.5,
        healthy_z_max=1.5,
        max_tilt_angle=1.0,
        natural_forward_z=0.0,
        n_actuators=5,
        foot_sensor_groups=((10,), (11,)),
    )

    @staticmethod
    def _stance_cfg():
        return {
            "forward_vel_weight": 0.0,
            "alive_bonus": 0.0,
            "energy_penalty_weight": 0.0,
            "posture_weight": 0.0,
            "bilateral_support_weight": 1.0,
            "foot_contact_saturation_force": 100.0,
        }

    def _snapshot_data(self):
        sensordata = np.zeros(20)
        sensordata[6:10] = (1.0, 0.0, 0.0, 0.0)
        # The boundary snapshot reads both feet fully loaded...
        sensordata[10:12] = (100.0, 100.0)
        return _make_mock_data(pelvis_z=0.9, sensordata=sensordata)

    def test_the_override_replaces_the_snapshot_forces(self):
        import jax.numpy as jnp

        from environments.shared.jax_reward_termination import compute_reward_components

        data = self._snapshot_data()
        snapshot = compute_reward_components(data, np.zeros(5), self._stance_cfg(), **self._COMMON)
        # ...but the substep MIN saw the right foot unload mid-control-step.
        aggregated = compute_reward_components(
            data,
            np.zeros(5),
            self._stance_cfg(),
            aggregated_foot_forces=jnp.asarray([0.0, 100.0]),
            **self._COMMON,
        )
        assert float(snapshot["bilateral_support"]) == pytest.approx(1.0)
        assert float(aggregated["bilateral_support"]) == pytest.approx(0.0)

    def test_default_none_keeps_single_state_semantics(self):
        from environments.shared.jax_reward_termination import compute_total_reward

        data = self._snapshot_data()
        without = compute_total_reward(data, np.zeros(5), self._stance_cfg(), **self._COMMON)
        explicit = compute_total_reward(
            data, np.zeros(5), self._stance_cfg(), aggregated_foot_forces=None, **self._COMMON
        )
        assert float(without) == pytest.approx(float(explicit))

    def test_the_override_moves_the_total_reward_and_survives_the_setup_filter(self):
        """Two pins the components test cannot give: the SCALAR eval reward
        consumes the override, and jax_setup's kwarg whitelist forwards it --
        dropping either silently reverts eval to boundary-snapshot forces."""
        import inspect

        import jax.numpy as jnp

        from environments.shared import jax_setup
        from environments.shared.jax_reward_termination import compute_total_reward

        data = self._snapshot_data()
        snapshot_total = compute_total_reward(data, np.zeros(5), self._stance_cfg(), **self._COMMON)
        aggregated_total = compute_total_reward(
            data,
            np.zeros(5),
            self._stance_cfg(),
            aggregated_foot_forces=jnp.asarray([0.0, 100.0]),
            **self._COMMON,
        )
        # bilateral_support_weight 1.0 drops from quality 1.0 to 0.0.
        assert float(snapshot_total) - float(aggregated_total) == pytest.approx(1.0)

        source = inspect.getsource(jax_setup)
        assert '"aggregated_foot_forces"' in source, (
            "jax_setup's compute_reward_detailed kwarg whitelist no longer forwards "
            "aggregated_foot_forces -- evaluate_policy_cpu's reward decomposition would "
            "silently revert to boundary-snapshot forces"
        )


# ---------------------------------------------------------------------------
# JX1: the CPU-eval composer must pay the action-jerk term the MJX step pays
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestStepComposerJerkParity:
    """The composer reproduces the MJX step's reward once jerk is enabled.

    Before this pin the composer omitted ``action_jerk_weight`` entirely, so
    ``run_stage_evaluation``'s episode rewards, the min_avg_reward rail and
    the reward-component panel overscored high-frequency-chatter policies
    relative to training (stance/recovery weight 3.0) and to the SB3
    backend.  The state-shaped weights the composer would need species
    geometry for are zeroed; alive/energy/posture/smoothness stay on so the
    parity is over a non-trivial sum, and the plant's 10 Hz command filter is
    live, so the lags compared are the FILTERED commands the kernel carries.
    """

    _ISOLATED = {
        "forward_vel_weight": 0.0,
        "alive_bonus": 1.0,
        "energy_penalty_weight": 0.05,
        "posture_weight": 0.5,
        "smoothness_weight": 1.0,
        "action_jerk_weight": 3.0,
        "approach_weight": 0.0,
        "foot_contact_gate": 0.0,
        "foot_contact_weight": 0.0,
        "bilateral_support_weight": 0.0,
        "foot_load_balance_weight": 0.0,
        "support_conditioned_alive_fraction": 0.0,
        "height_weight": 0.0,
        "nosedive_weight": 0.0,
        "drift_penalty_weight": 0.0,
        "speed_penalty_weight": 0.0,
        "spin_penalty_weight": 0.0,
        "tail_stability_weight": 0.0,
        "heading_weight": 0.0,
        "lateral_penalty_weight": 0.0,
        "action_saturation_weight": 0.0,
        "bite_head_proximity_weight": 0.0,
        "bite_bonus": 0.0,
    }

    def test_composer_reproduces_the_step_reward_with_jerk_enabled(self):
        pytest.importorskip("mujoco.mjx")
        import environments.trex.mjx_config  # noqa: F401  (registers trex)
        from environments.shared.jax_reward_termination import compute_total_reward
        from environments.shared.mjx_env import MJXDinoEnv

        env = MJXDinoEnv("trex", stage=1, num_envs=1, env_kwargs=dict(self._ISOLATED))
        cfg = env.config
        assert cfg.reward_weights["action_jerk_weight"] == 3.0
        composer_kw = dict(
            root_body_id=cfg.body_ids["pelvis"],
            healthy_z_min=cfg.healthy_z_range[0],
            healthy_z_max=cfg.healthy_z_range[1],
            max_tilt_angle=cfg.max_tilt_angle,
            natural_forward_z=cfg.natural_forward_z,
            n_actuators=env.mj_model.nu,
            posture_target_forward_z=cfg.posture_target_forward_z,
            sensor_quat_start=cfg.sensor_quat_start,
            sensor_gyro_start=cfg.sensor_gyro_start,
            foot_indices=cfg.sensor_foot_indices,
            fall_penalty=cfg.fall_penalty,
        )
        no_jerk_cfg = {**cfg.reward_weights, "action_jerk_weight": 0.0}

        rng = jax.random.PRNGKey(3)
        state = env._reset_single(rng)
        actions = jax.random.uniform(jax.random.PRNGKey(4), (3, env.action_dim), minval=-0.6, maxval=0.6)
        for k in range(3):
            # The kernel's second lag BEFORE this step (zeros while k < 2).
            prev_prev_before = state.prev_prev_action
            state, step_reward, terminated, _ = env._step_single(state, actions[k], rng, jnp.float32(1.0))
            assert not bool(terminated)
            # After the step the carries are: prev_action == this step's
            # (filtered) command, prev_prev_action == the previous one.
            lags = {"prev_action": state.prev_prev_action, "prev_prev_action": prev_prev_before}
            composer = compute_total_reward(state.data, state.prev_action, cfg.reward_weights, **lags, **composer_kw)
            np.testing.assert_allclose(float(composer), float(step_reward), rtol=1e-5, atol=1e-6)
            # Teeth: without the jerk term the composer overscores every one
            # of these steps -- including the first two, which the kernel
            # charges against its zero carries.
            without = compute_total_reward(state.data, state.prev_action, no_jerk_cfg, **lags, **composer_kw)
            assert float(without) > float(step_reward) + 1e-4

    def test_components_expose_the_jerk_term_under_the_same_zero_lag_convention(self):
        from environments.shared.jax_reward_termination import compute_reward_components, compute_total_reward
        from environments.shared.reward_functions import reward_action_jerk

        data = _make_mock_data(pelvis_z=0.8)
        common = dict(root_body_id=1, healthy_z_min=0.3, max_tilt_angle=1.0, natural_forward_z=0.0, n_actuators=4)
        cfg = {"forward_vel_weight": 0.0, "alive_bonus": 0.0, "energy_penalty_weight": 0.0, "posture_weight": 0.0}
        action = jnp.array([0.5, -0.5, 0.25, 0.0])
        prev = jnp.array([0.1, 0.1, 0.1, 0.1])
        prev_prev = jnp.array([-0.2, 0.3, 0.0, 0.4])
        expected, raw = reward_action_jerk(action, prev, prev_prev, 4, 3.0)

        with_jerk = compute_reward_components(
            data, action, {**cfg, "action_jerk_weight": 3.0}, prev_action=prev, prev_prev_action=prev_prev, **common
        )
        assert float(with_jerk["action_jerk"]) == pytest.approx(float(expected))
        assert float(with_jerk["_action_jerk"]) == pytest.approx(float(raw))
        assert "action_jerk" not in compute_reward_components(data, action, cfg, **common)

        # Missing lags mean zeros (the MJX reset carries), for both entry points.
        first_step_expected, _ = reward_action_jerk(action, jnp.zeros(4), jnp.zeros(4), 4, 3.0)
        total_first = compute_total_reward(data, action, {**cfg, "action_jerk_weight": 3.0}, **common)
        assert float(total_first) == pytest.approx(float(first_step_expected))
        assert float(
            compute_reward_components(data, action, {**cfg, "action_jerk_weight": 3.0}, **common)["action_jerk"]
        ) == pytest.approx(float(first_step_expected))
