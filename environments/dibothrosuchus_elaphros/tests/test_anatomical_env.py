"""Species-specific tests for the Dibothrosuchus Gymnasium environment.

Common env tests (spaces, reset, step, determinism, observation bounds) are in
environments/shared/tests/test_species_integration.py.
"""

import mujoco
import numpy as np
import pytest

from environments.dibothrosuchus_elaphros.anatomical.env import DibothrosuchusElaphrosAnatomicalEnv
from environments.shared.model_id import ModelId


@pytest.fixture
def env():
    e = DibothrosuchusElaphrosAnatomicalEnv()
    yield e
    e.close()


class TestDibothrosuchusElaphrosSpecific:
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

    def test_gait_reward_non_positive(self, env):
        """The trunk angular-velocity term is a penalty."""
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_gait"] <= 0.0
        assert info["gait_instability"] >= 0.0

    def test_observation_layout_dimensions(self, env):
        """28 qpos + 28 qvel + gravity + gyro + heading-frame linvel + accel + 4 feet + command: 75.

        quadrupedal-heading-free/v1 since policy interface r11 (decision
        D-D59; r10: 80, with the trunk quaternion and the world-frame
        linear velocity, and the prey direction and distance before the
        command).  The trailing 3 dims are the Phase C body-relative command
        segment (BEHAVIOR_RECIPES_PLAN §4.6), zero under
        ``command_mode = "none"``.
        """
        obs, _ = env.reset(seed=0)
        assert obs.shape == (28 + 28 + 3 + 3 + 3 + 3 + 4 + 3,) == env.observation_space.shape
        np.testing.assert_array_equal(obs[-3:], np.zeros(3, dtype=np.float32))

    def test_projected_gravity_is_the_unit_down_vector(self, env):
        obs, _ = env.reset(seed=7)
        # The root block starts right after the 28 + 28 joint channels.
        gravity = obs[56:59]
        assert np.linalg.norm(gravity) == pytest.approx(1.0, abs=1e-5)
        assert gravity[2] < -0.99  # the trunk frame is level at home

    def test_foot_channels_are_the_four_touch_sensors(self, env):
        env.reset(seed=7)
        obs, _, terminated, _, _ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert not terminated
        # The four foot channels sit just before the command (3), the last segment.
        expected = [env.data.sensordata[i] for i in (env._sensor_fr_foot, env._sensor_fl_foot)]
        expected += [env.data.sensordata[i] for i in (env._sensor_rr_foot, env._sensor_rl_foot)]
        np.testing.assert_array_equal(obs[-7:-3], np.asarray(expected, dtype=np.float32))

    def test_turning_the_whole_scene_about_vertical_leaves_the_observation_unchanged(self, env):
        """No channel carries the world heading: a 45 deg yaw of the root's pose and velocity changes nothing."""
        env.reset(seed=7)
        for _ in range(5):
            env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        # After a step the sensors still hold the state before its last integration: recompute them first.
        mujoco.mj_forward(env.model, env.data)
        before = env._get_obs()
        yaw = np.radians(45.0)
        turn = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        quat = np.empty(4)
        mujoco.mju_mulQuat(quat, turn, env.data.qpos[3:7].copy())
        rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
        env.data.qpos[0:3] = rotation @ env.data.qpos[0:3]
        env.data.qpos[3:7] = quat
        env.data.qvel[0:3] = rotation @ env.data.qvel[0:3]
        mujoco.mj_forward(env.model, env.data)
        after = env._get_obs()
        np.testing.assert_allclose(after, before, rtol=0.0, atol=1e-6)


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


class TestNoPreyNoSnap:
    """The snap was removed with the hunt (decision D-D59), and the prey with it in physics r4.

    The scene holds the animal and the floor alone: no mocap body, no prey
    geom, and no ``snout_snap`` box (contype/conaffinity 2/2, which met only
    the prey).  The ``snout_tip`` site stays for the snout-prop termination.
    The forward reference is the spawn's own heading.
    """

    def test_no_success_key_and_no_snap_or_prey_kwarg(self):
        assert DibothrosuchusElaphrosAnatomicalEnv.SUCCESS_KEYS == ()
        with pytest.raises(TypeError, match="snap_bonus"):
            DibothrosuchusElaphrosAnatomicalEnv(snap_bonus=10.0)
        for kwarg in ("prey_distance_range", "prey_lateral_range"):
            with pytest.raises(TypeError, match=kwarg):
                DibothrosuchusElaphrosAnatomicalEnv(**{kwarg: (1.0, 2.0)})

    def test_the_scene_has_no_prey_no_mocap_body_and_no_snap_geom(self, env):
        model = env.model
        assert model.nmocap == 0
        for kind, name in (
            (mujoco.mjtObj.mjOBJ_BODY, "prey"),
            (mujoco.mjtObj.mjOBJ_GEOM, "prey_geom"),
            (mujoco.mjtObj.mjOBJ_GEOM, "snout_snap"),
        ):
            assert mujoco.mj_name2id(model, kind, name) == -1
        for attr in ("prey_geom_id", "prey_mocap_id", "snout_snap_geom_id", "_initial_prey_dir_2d"):
            assert not hasattr(env, attr)
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "snout_tip") == env.snout_tip_site_id >= 0

    def test_every_colliding_geom_is_on_the_floor_channel(self, env):
        """Only channel 1/1 remains: the floor's, which every colliding animal geom shares."""
        model = env.model
        colliding = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        assert np.all(model.geom_contype[colliding] == 1) and np.all(model.geom_conaffinity[colliding] == 1)

    def test_no_hunt_info_key_and_no_success(self, env):
        env.reset(seed=42)
        _, info = env._get_reward_info(np.zeros(env.action_space.shape, dtype=np.float32))
        hunt = {
            "snap_success",
            "reward_snap",
            "snout_prey_distance",
            "prey_distance",
            "approach_delta",
            "reward_approach",
            "snout_proximity",
            "reward_snout_proximity",
        }
        assert not hunt & set(info)
        _, info = env._is_terminated()
        assert info.get("termination_reason") != "snap_success"
        assert not info.get("success", False)

    def test_the_forward_reference_is_the_spawn_heading(self, env):
        """The progress term reads _reference_heading_2d: velocity along it pays, across it does not."""
        env.reset(seed=3042)
        np.testing.assert_array_equal(env._reference_heading_2d, env._quat_to_forward_2d(env.data.qpos[3:7]))
        env.forward_vel_weight = 1.0
        env._reference_heading_2d = np.array([0.0, 1.0])
        env.data.qvel[0:2] = (0.0, 0.5)
        _, along = env._get_reward_info(np.zeros(env.action_space.shape, dtype=np.float32))
        env.data.qvel[0:2] = (0.5, 0.0)
        _, across = env._get_reward_info(np.zeros(env.action_space.shape, dtype=np.float32))
        assert along["reward_forward"] > 0.0
        assert across["reward_forward"] == pytest.approx(0.0, abs=1e-12)

    def test_stance_pays_no_heading_term(self):
        """stance.toml sets heading_weight 0 since policy interface r11: the observation cannot see heading."""
        from environments.shared.config import load_stage_config

        kwargs = load_stage_config(ModelId("dibothrosuchus_elaphros", "anatomical"), 1)["env_kwargs"]
        assert kwargs["heading_weight"] == 0.0


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


class TestResetStaysInsideTheHealthyEnvelope:
    """Reset noise must perturb the pose, not spawn already-terminated episodes.

    ``reset_noise_scale`` is a joint-angle scale in radians.  The shared base
    once also used it as a root-height jitter in metres, which on this 0.313 m
    stance put a quarter of stage-1 episodes outside ``healthy_z_range`` at
    step 1; a separate ``reset_height_noise_scale`` (0.03 m here) decoupled
    the two.  Since the ground settle sets the root height from the sampled
    joint pose, the height channel no longer reached the state, and policy
    interface r11 removed its draw and its knob (decision D-D59).
    """

    def test_the_height_jitter_knob_is_gone(self):
        with pytest.raises(TypeError, match="reset_height_noise_scale"):
            DibothrosuchusElaphrosAnatomicalEnv(reset_height_noise_scale=0.03)

    def test_stage1_reset_never_spawns_out_of_bounds(self):
        """At the committed stage-1 noise, no seed starts already terminated."""
        from environments.shared.config import load_stage_config

        kwargs = load_stage_config(ModelId("dibothrosuchus_elaphros", "anatomical"), 1)["env_kwargs"]
        kwargs = {k: v for k, v in kwargs.items() if k not in {"foot_contact_weight", "foot_contact_gate"}}
        env = DibothrosuchusElaphrosAnatomicalEnv(**kwargs)
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
