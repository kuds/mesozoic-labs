"""Species-specific tests for the Dibothrosuchus Gymnasium environment.

Common env tests (spaces, reset, step, determinism, observation bounds) are in
environments/shared/tests/test_species_integration.py.
"""

import numpy as np
import pytest

from environments.dibothrosuchus.envs.dibothrosuchus_env import DibothrosuchusEnv


@pytest.fixture
def env():
    e = DibothrosuchusEnv()
    yield e
    e.close()


class TestDibothrosuchusSpecific:
    """Observation and info keys unique to this species."""

    def test_trunk_height_tracked_under_both_names(self, env):
        """torso_height is the species term; pelvis_height is the metrics alias."""
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert info["torso_height"] > 0
        assert info["pelvis_height"] == info["torso_height"]

    def test_four_foot_contacts_tracked(self, env):
        """All four pads report through the touch sensors."""
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        for key in ("r_foot_contact", "l_foot_contact", "rr_foot_contact", "rl_foot_contact"):
            assert key in info

    def test_snout_prey_distance_reported(self, env):
        """Approach shaping is measured from the snout, not the trunk."""
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert info["snout_prey_distance"] > 0  # prey spawns 1.5-4 m away
        assert info["prey_distance"] == info["snout_prey_distance"]

    def test_snap_not_successful_initially(self, env):
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert info["snap_success"] == 0.0
        assert info["reward_snap"] == 0.0

    def test_gait_reward_non_positive(self, env):
        """The trunk angular-velocity term is a penalty."""
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_gait"] <= 0.0
        assert info["gait_instability"] >= 0.0

    def test_observation_layout_dimensions(self, env):
        """28 qpos + 28 qvel + quat + gyro + linvel + accel + 4 feet + dir + dist."""
        obs, _ = env.reset(seed=0)
        assert obs.shape == (28 + 28 + 4 + 3 + 3 + 3 + 4 + 3 + 1,)

    def test_prey_direction_is_unit_length(self, env):
        obs, _ = env.reset(seed=7)
        direction = obs[-4:-1]
        assert np.linalg.norm(direction) == pytest.approx(1.0, abs=1e-5)


class TestHomeResidualActionMapping:
    """Action zero must be exactly the XML home control vector."""

    def test_action_mapping_is_declared(self, env):
        assert env.action_mapping == "home-keyframe-residual/v1"

    def test_zero_action_maps_to_home_ctrl(self, env):
        scaled = env._scale_action(np.zeros(env.action_space.shape, dtype=np.float32))
        np.testing.assert_allclose(scaled, env.model.key_ctrl[env.home_keyframe_id], atol=1e-12)

    def test_action_endpoints_map_to_ctrlrange(self, env):
        low = env._scale_action(-np.ones(env.action_space.shape, dtype=np.float32))
        high = env._scale_action(np.ones(env.action_space.shape, dtype=np.float32))
        np.testing.assert_allclose(low, env.model.actuator_ctrlrange[:, 0], atol=1e-12)
        np.testing.assert_allclose(high, env.model.actuator_ctrlrange[:, 1], atol=1e-12)

    def test_out_of_range_action_is_clipped(self, env):
        far = env._scale_action(np.full(env.action_space.shape, 5.0, dtype=np.float32))
        np.testing.assert_allclose(far, env.model.actuator_ctrlrange[:, 1], atol=1e-12)

    def test_reset_uses_the_home_keyframe(self, env):
        assert env._reset_keyframe_id == env.home_keyframe_id


class TestSnapGating:
    """Success termination must be gated on snap_bonus > 0 (matching the
    raptor/T-Rex strike/bite gating) so stages 1-2 don't end episodes on
    incidental snout-prey contact."""

    @staticmethod
    def _teleport_prey_onto_snout(env):
        env.data.mocap_pos[0] = env.data.site_xpos[env.snout_tip_site_id].copy()
        # Recompute contacts so the snout snap geom actually overlaps the prey.
        import mujoco

        mujoco.mj_forward(env.model, env.data)

    def test_no_success_termination_when_bonus_zero(self):
        env = DibothrosuchusEnv(snap_bonus=0.0)
        try:
            env.reset(seed=42)
            self._teleport_prey_onto_snout(env)
            _, info = env._is_terminated()
            assert info.get("termination_reason") != "snap_success"
            assert not info.get("success", False)
        finally:
            env.close()

    def test_success_termination_when_bonus_positive(self):
        env = DibothrosuchusEnv(snap_bonus=10.0)
        try:
            env.reset(seed=42)
            self._teleport_prey_onto_snout(env)
            terminated, info = env._is_terminated()
            assert terminated
            assert info["termination_reason"] == "snap_success"
            assert info["success"] is True
        finally:
            env.close()

    def test_snap_reward_paid_on_contact(self):
        env = DibothrosuchusEnv(snap_bonus=25.0)
        try:
            env.reset(seed=42)
            self._teleport_prey_onto_snout(env)
            _, info = env._get_reward_info(np.zeros(env.action_space.shape, dtype=np.float32))
            assert info["snap_success"] == 1.0
            assert info["reward_snap"] == 25.0
        finally:
            env.close()


class TestSnoutTerminations:
    """The snout is long and carried low; it must not be usable as a prop."""

    def test_snout_ground_contact_terminates(self, env):
        """Pitch the trunk nose-down and flex the neck so ONLY the snout goes low.

        Dropping the whole animal instead just trips the healthy-height floor,
        which returns before the snout check ever runs -- so a test written
        that way passes without exercising this gate at all.  At 20 deg the
        trunk stays at 0.366 m, tilt is 0.35 rad against a 0.9 limit, and
        forward_z is -0.34 against a -0.55 nosedive threshold, so every check
        that could preempt the snout gate has clear margin.
        """
        import mujoco

        model, data = env.model, env.data
        env.reset(seed=0)
        pitch = np.radians(20.0)
        data.qpos[3:7] = [np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0]
        for joint in ("neck_pitch", "head_pitch"):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            data.qpos[model.jnt_qposadr[jid]] = model.jnt_range[jid][1]
        mujoco.mj_forward(model, data)

        terminated, info = env._is_terminated()
        assert terminated
        assert info["termination_reason"] == "head_contact", (
            f"expected the snout gate, got {info['termination_reason']} -- another check preempted it"
        )
        assert info["snout_tip_z"] < 0.04

    def test_snout_tip_height_reported_when_upright(self, env):
        env.reset(seed=0)
        terminated, info = env._is_terminated()
        assert not terminated
        assert info["snout_tip_z"] > 0.04

    def test_between_sample_snout_dip_terminates_while_boundary_reads_healthy(self, env):
        """The snout gate consumes the substep MIN recorded by the base step
        loop, so a dip that recovers before the control boundary still
        terminates; the info key keeps the boundary sample."""
        env.reset(seed=0)
        env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        boundary_snout = float(env.data.site_xpos[env.snout_tip_site_id, 2])
        assert boundary_snout > 0.04

        env._substep_min_heights = np.array([0.03])
        env._substep_contact_step = env._step_count
        terminated, info = env._is_terminated()
        assert terminated
        assert info["termination_reason"] == "head_contact"
        assert info["snout_tip_z"] == pytest.approx(boundary_snout)

        env.reset(seed=1)
        terminated, info = env._is_terminated()
        assert not terminated, f"stale height aggregate survived reset: {info}"

    def test_snap_target_clears_the_snout_prop_gate(self, env):
        """A successful snap must never be reported as snout-propping.

        ``_is_terminated`` checks the 0.04 m snout-height gate before the snap
        contact, following the same ordering as the other species, so the prey
        sphere's underside has to stay above that gate with margin.
        """
        import mujoco

        env.reset(seed=0)
        prey_geom = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "prey_geom")
        prey_radius = float(env.model.geom_size[prey_geom, 0])
        prey_underside = float(env.data.mocap_pos[0][2]) - prey_radius
        assert prey_underside > 0.04 + 0.03, (
            f"Prey underside at {prey_underside:.3f} m leaves too little margin over the "
            "0.04 m snout-prop gate; a snap at the bottom of the sphere would terminate as head_contact."
        )


class TestResetStaysInsideTheHealthyEnvelope:
    """Reset noise must perturb the pose, not spawn already-terminated episodes.

    ``reset_noise_scale`` is a joint-angle scale in radians, but the shared base
    also used it as a root-height jitter in metres.  On this 0.313 m stance that
    put a quarter of stage-1 episodes outside ``healthy_z_range`` at step 1,
    which inflated the zero-action baseline's apparent difficulty with noise no
    policy can act on.  ``reset_height_noise_scale`` decoupled the two.

    Since the ground settle the whole height channel is STATE-INERT — the root
    height is overwritten from the sampled joint pose (see
    TestHeightJitterIsInertSinceGroundSettling in test_base_env.py) — but the
    knob still exists for RNG-stream compatibility, so these tests keep pinning
    that it stays set and stays decoupled until the channel is removed.
    """

    def test_height_noise_is_decoupled_from_joint_noise(self):
        env = DibothrosuchusEnv()
        try:
            assert env.reset_height_noise_scale is not None, (
                "this species must set its own metre-scale height jitter; inheriting the "
                "radian-scale joint noise spawns episodes outside healthy_z_range"
            )
            assert env.reset_height_noise_scale < 0.25 * (env.healthy_z_range[1] - env.healthy_z_range[0])
        finally:
            env.close()

    def test_stage1_reset_never_spawns_out_of_bounds(self):
        """At the committed stage-1 noise, no seed starts already terminated."""
        from environments.shared.config import load_stage_config

        kwargs = load_stage_config("dibothrosuchus", 1)["env_kwargs"]
        kwargs = {k: v for k, v in kwargs.items() if k not in {"foot_contact_weight", "foot_contact_gate"}}
        env = DibothrosuchusEnv(**kwargs)
        try:
            low, high = env.healthy_z_range
            for seed in range(40):
                env.reset(seed=seed)
                trunk_z = float(env.data.xpos[env.torso_id, 2])
                assert low < trunk_z < high, f"seed {seed} spawned at {trunk_z:.3f} m, outside [{low}, {high}]"
                terminated, info = env._is_terminated()
                assert not terminated, f"seed {seed} starts terminated: {info.get('termination_reason')}"
        finally:
            env.close()
