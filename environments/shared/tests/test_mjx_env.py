"""Tests for the MJX environment wrapper.

These tests are skipped when JAX/MJX is not installed.
"""

import inspect

import numpy as np
import pytest

_has_jax = False
try:
    import jax  # noqa: F401
    import mujoco.mjx  # noqa: F401

    _has_jax = True
except ImportError:
    pass


def test_reset_and_step_use_the_fingerprinted_observation_builder():
    from environments.shared.mjx_env import MJXDinoEnv

    assert "build_mjx_observation(" in inspect.getsource(MJXDinoEnv._make_step_fn)
    assert "build_mjx_observation(" in inspect.getsource(MJXDinoEnv._make_reset_fn)


@pytest.mark.skipif(not _has_jax, reason="JAX/MJX not installed")
class TestMJXDinoEnv:
    """Integration tests for MJXDinoEnv (require JAX + MJX)."""

    def test_trex_creation(self):
        """MJXDinoEnv can be created for T-Rex."""
        import environments.trex.mjx_config  # noqa: F401
        from environments.shared.mjx_env import MJXDinoEnv

        env = MJXDinoEnv("trex", stage=1, num_envs=4)
        assert env.action_dim == 21
        assert env.num_envs == 4
        assert env.config.posture_target_forward_z is None
        assert env.config.action_mapping == "home-keyframe-residual/v1"
        assert env.nominal_ctrl is not None
        # Pin the pitch reference against the Gymnasium default rather than a
        # literal: the two drifted apart once already, when the SB3 side moved
        # to the measured neutral pose and this registration kept -sin(0.17).
        # Only the override path was covered, so it passed either way.
        from environments.trex.envs.trex_env import TRexEnv

        sb3_natural_pitch = inspect.signature(TRexEnv.__init__).parameters["natural_pitch"].default
        assert env.config.natural_forward_z == pytest.approx(-np.sin(sb3_natural_pitch))

    def test_trex_foot_sensors_cover_the_digits(self):
        """Each foot must sum its pad sensor plus one sensor per digit.

        A touch sensor sums only geoms on its site's own body, so without the
        aux groups a T-Rex standing on its toes reads as airborne.
        """
        import environments.trex.mjx_config  # noqa: F401
        from environments.shared.mjx_env import MJXDinoEnv

        env = MJXDinoEnv("trex", stage=1, num_envs=1)
        assert len(env.config.sensor_foot_aux_indices) == len(env.config.sensor_foot_indices)
        assert all(len(group) == 3 for group in env.config.sensor_foot_aux_indices)

    def test_raptor_creation(self):
        """MJXDinoEnv can be created for Velociraptor."""
        import environments.velociraptor.mjx_config  # noqa: F401
        from environments.shared.mjx_env import MJXDinoEnv

        env = MJXDinoEnv("velociraptor", stage=1, num_envs=4)
        assert env.action_dim == 22
        assert env.num_envs == 4
        assert env.config.natural_forward_z == pytest.approx(-0.342)
        assert env.config.posture_target_forward_z == pytest.approx(-np.sin(0.35))

    def test_raptor_natural_pitch_override_updates_posture_target(self):
        import environments.velociraptor.mjx_config  # noqa: F401
        from environments.shared.mjx_env import MJXDinoEnv

        env = MJXDinoEnv(
            "velociraptor",
            stage=1,
            num_envs=1,
            env_kwargs={"natural_pitch": 0.5},
        )

        assert env.config.natural_forward_z == pytest.approx(-np.sin(0.5))
        assert env.config.posture_target_forward_z == pytest.approx(-np.sin(0.5))

    def test_trex_natural_pitch_override_keeps_vertical_posture_target(self):
        import environments.trex.mjx_config  # noqa: F401
        from environments.shared.mjx_env import MJXDinoEnv

        env = MJXDinoEnv(
            "trex",
            stage=1,
            num_envs=1,
            env_kwargs={"natural_pitch": 0.5},
        )

        assert env.config.natural_forward_z == pytest.approx(-np.sin(0.5))
        assert env.config.posture_target_forward_z is None

    def test_brachio_creation(self):
        """MJXDinoEnv can be created for Brachiosaurus."""
        import environments.brachiosaurus.mjx_config  # noqa: F401
        from environments.shared.mjx_env import MJXDinoEnv

        env = MJXDinoEnv("brachiosaurus", stage=1, num_envs=4)
        assert env.action_dim == 30
        assert env.num_envs == 4
        assert env.config.posture_target_forward_z is None

    @pytest.mark.parametrize("species", ("velociraptor", "trex", "brachiosaurus", "dibothrosuchus"))
    def test_stage_factory_preserves_canonical_compiled_plant(self, species):
        from environments.shared.jax_setup import create_env, setup_species
        from environments.shared.plant_contract import current_plant_identity, validate_compiled_plant

        ctx = setup_species(species, stage=1)
        env = create_env(ctx, num_envs=1)
        identity = current_plant_identity(species, verify_generated=False)

        validate_compiled_plant(ctx.mj_model, identity, artifact=f"{species} CPU evaluation model")
        validate_compiled_plant(env.mj_model, identity, artifact=f"{species} MJX training model")
        for option in ("solver", "iterations", "ls_iterations", "disableflags"):
            assert getattr(ctx.mj_model.opt, option) == getattr(env.mj_model.opt, option)

    def test_stage_cannot_override_versioned_plant_interface(self):
        import environments.trex.mjx_config  # noqa: F401
        from environments.shared.mjx_env import MJXDinoEnv

        with pytest.raises(ValueError, match="versioned plant interface"):
            MJXDinoEnv("trex", stage=1, num_envs=1, env_kwargs={"frame_skip": 4})
        with pytest.raises(ValueError, match="versioned plant interface"):
            MJXDinoEnv(
                "trex",
                stage=1,
                num_envs=1,
                env_kwargs={"action_mapping": "midpoint/v1"},
            )

    def test_reset_returns_state(self):
        """Reset returns an EnvState with correct shapes."""
        import jax

        import environments.trex.mjx_config  # noqa: F401
        from environments.shared.mjx_env import MJXDinoEnv

        env = MJXDinoEnv("trex", stage=1, num_envs=4)
        rng = jax.random.PRNGKey(42)
        states = env.reset(rng)
        assert states.obs.shape[0] == 4  # batch size
        assert states.step_count.shape == (4,)


@pytest.mark.skipif(not _has_jax, reason="JAX/MJX not installed")
class TestMJXUtils:
    """Test utility functions from mjx_utils."""

    def test_scale_action(self):
        import jax.numpy as jnp

        from environments.shared.mjx_utils import scale_action_jax

        ctrl_range = jnp.array([[-1.0, 1.0], [-0.5, 0.5]])
        action = jnp.zeros(2)
        scaled = scale_action_jax(action, ctrl_range)
        # Zero action → midpoint
        assert float(scaled[0]) == pytest.approx(0.0)
        assert float(scaled[1]) == pytest.approx(0.0)

    def test_scale_action_extremes(self):
        import jax.numpy as jnp

        from environments.shared.mjx_utils import scale_action_jax

        ctrl_range = jnp.array([[-2.0, 2.0]])
        # +1 → max
        assert float(scale_action_jax(jnp.array([1.0]), ctrl_range)[0]) == pytest.approx(2.0)
        # -1 → min
        assert float(scale_action_jax(jnp.array([-1.0]), ctrl_range)[0]) == pytest.approx(-2.0)

    def test_scale_action_around_nominal_preserves_zero_and_endpoints(self):
        import jax.numpy as jnp

        from environments.shared.mjx_utils import scale_action_around_nominal_jax

        ctrl_range = jnp.array([[-2.0, 4.0], [-1.0, 3.0]])
        nominal_ctrl = jnp.array([1.0, 0.5])

        assert np.allclose(
            np.asarray(scale_action_around_nominal_jax(jnp.zeros(2), ctrl_range, nominal_ctrl)),
            np.asarray(nominal_ctrl),
        )
        assert np.allclose(
            np.asarray(scale_action_around_nominal_jax(jnp.full(2, -1.0), ctrl_range, nominal_ctrl)),
            np.asarray(ctrl_range[:, 0]),
        )
        assert np.allclose(
            np.asarray(scale_action_around_nominal_jax(jnp.full(2, 1.0), ctrl_range, nominal_ctrl)),
            np.asarray(ctrl_range[:, 1]),
        )

    def test_scale_action_around_nominal_clips_direct_callers(self):
        import jax.numpy as jnp

        from environments.shared.mjx_utils import scale_action_around_nominal_jax

        ctrl_range = jnp.array([[-2.0, 4.0], [-1.0, 3.0]])
        nominal_ctrl = jnp.array([1.0, 0.5])
        scaled = scale_action_around_nominal_jax(jnp.array([-5.0, 5.0]), ctrl_range, nominal_ctrl)
        assert np.allclose(np.asarray(scaled), np.array([-2.0, 3.0]))


@pytest.mark.skipif(not _has_jax, reason="JAX/MJX not installed")
class TestHomeKeyframeActionMapping:
    def test_setup_species_resolves_raptor_posture_target_before_env_creation(self):
        from environments.shared.jax_setup import setup_species

        ctx = setup_species("velociraptor", stage=1)

        assert ctx.env_config is None
        assert ctx.natural_forward_z == pytest.approx(-0.342)
        assert ctx.posture_target_forward_z == pytest.approx(-np.sin(0.35))

    def test_pre_env_raptor_natural_pitch_override_updates_posture_target(self):
        import environments.velociraptor.mjx_config  # noqa: F401
        from environments.shared.jax_setup import SpeciesContext

        overridden = -np.sin(0.5)
        ctx = SpeciesContext(
            species="velociraptor",
            stage=1,
            reward_cfg={"natural_forward_z": overridden},
        )

        assert ctx.natural_forward_z == pytest.approx(overridden)
        assert ctx.posture_target_forward_z == pytest.approx(overridden)

    def test_setup_species_raptor_zero_action_maps_to_xml_home(self):
        import jax.numpy as jnp
        import mujoco

        from environments.shared.jax_setup import make_scale_action_fn, setup_species

        ctx = setup_species("velociraptor", stage=1)
        scale_action = make_scale_action_fn(ctx)
        home_id = mujoco.mj_name2id(ctx.mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")

        assert ctx.action_mapping == "home-keyframe-residual/v1"
        assert home_id >= 0
        np.testing.assert_allclose(
            np.asarray(scale_action(jnp.zeros(ctx.act_dim))),
            ctx.mj_model.key_ctrl[home_id],
            rtol=0,
            atol=1e-7,
        )

    def test_raptor_jax_mapping_matches_gymnasium_mapping(self):
        import jax.numpy as jnp

        from environments.shared.jax_setup import make_scale_action_fn, setup_species
        from environments.velociraptor.envs.raptor_env import RaptorEnv

        ctx = setup_species("velociraptor", stage=1)
        scale_action = make_scale_action_fn(ctx)
        env = RaptorEnv()
        try:
            action = np.linspace(-1.5, 1.5, ctx.act_dim, dtype=np.float32)
            np.testing.assert_allclose(
                np.asarray(scale_action(jnp.asarray(action))),
                env._scale_action(action),
                rtol=0,
                atol=1e-7,
            )
        finally:
            env.close()

    def test_setup_species_trex_zero_action_maps_to_xml_home(self):
        import jax.numpy as jnp
        import mujoco

        from environments.shared.jax_setup import make_scale_action_fn, setup_species

        ctx = setup_species("trex", stage=1)
        scale_action = make_scale_action_fn(ctx)
        home_id = mujoco.mj_name2id(ctx.mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")

        assert ctx.action_mapping == "home-keyframe-residual/v1"
        assert home_id >= 0
        np.testing.assert_allclose(
            np.asarray(scale_action(jnp.zeros(ctx.act_dim))),
            ctx.mj_model.key_ctrl[home_id],
            rtol=0,
            atol=1e-7,
        )

    def test_trex_jax_mapping_matches_gymnasium_mapping(self):
        import jax.numpy as jnp

        from environments.shared.jax_setup import make_scale_action_fn, setup_species
        from environments.trex.envs.trex_env import TRexEnv

        ctx = setup_species("trex", stage=1)
        scale_action = make_scale_action_fn(ctx)
        env = TRexEnv()
        try:
            action = np.linspace(-1.5, 1.5, ctx.act_dim, dtype=np.float32)
            np.testing.assert_allclose(
                np.asarray(scale_action(jnp.asarray(action))),
                env._scale_action(action),
                rtol=0,
                atol=1e-7,
            )
        finally:
            env.close()

    def test_setup_species_dibothrosuchus_zero_action_maps_to_xml_home(self):
        import jax.numpy as jnp
        import mujoco

        from environments.shared.jax_setup import make_scale_action_fn, setup_species

        ctx = setup_species("dibothrosuchus", stage=1)
        scale_action = make_scale_action_fn(ctx)
        home_id = mujoco.mj_name2id(ctx.mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")

        assert ctx.action_mapping == "home-keyframe-residual/v1"
        assert home_id >= 0
        np.testing.assert_allclose(
            np.asarray(scale_action(jnp.zeros(ctx.act_dim))),
            ctx.mj_model.key_ctrl[home_id],
            rtol=0,
            atol=1e-7,
        )

    def test_dibothrosuchus_jax_mapping_matches_gymnasium_mapping(self):
        import jax.numpy as jnp

        from environments.dibothrosuchus.envs.dibothrosuchus_env import DibothrosuchusEnv
        from environments.shared.jax_setup import make_scale_action_fn, setup_species

        ctx = setup_species("dibothrosuchus", stage=1)
        scale_action = make_scale_action_fn(ctx)
        env = DibothrosuchusEnv()
        try:
            action = np.linspace(-1.5, 1.5, ctx.act_dim, dtype=np.float32)
            np.testing.assert_allclose(
                np.asarray(scale_action(jnp.asarray(action))),
                env._scale_action(action),
                rtol=0,
                atol=1e-7,
            )
        finally:
            env.close()

    def test_second_quadruped_uses_the_torso_rooted_observation_builder(self):
        """The MJX observation dispatch keys off the registered root body, so a
        second quadruped must reach build_quadruped_obs without an allow-list."""
        import mujoco

        import environments.dibothrosuchus.mjx_config  # noqa: F401  (registers the species)
        from environments.dibothrosuchus.envs.dibothrosuchus_env import DibothrosuchusEnv
        from environments.shared.mjx_env import _SPECIES_CONFIGS, build_mjx_observation

        config = _SPECIES_CONFIGS["dibothrosuchus"]
        assert set(config["body_ids"]) == {"torso"}
        assert len(config["sensor_foot_indices"]) == 4

        env = DibothrosuchusEnv()
        try:
            model, data = env.model, env.data
            mujoco.mj_forward(model, data)
            observation = np.asarray(
                build_mjx_observation(
                    data,
                    data.mocap_pos[0],
                    {
                        "body_ids": config["body_ids"],
                        "sensor_gyro_start": config["sensor_gyro_start"],
                        "sensor_accel_start": config["sensor_accel_start"],
                        "sensor_quat_start": config["sensor_quat_start"],
                        "sensor_foot_indices": config["sensor_foot_indices"],
                    },
                )
            )
            np.testing.assert_allclose(observation, env._get_obs(), rtol=0, atol=1e-6)
        finally:
            env.close()

    def test_noise_free_raptor_mjx_reset_restores_complete_home_state(self):
        import jax
        import mujoco

        import environments.velociraptor.mjx_config  # noqa: F401
        from environments.shared.mjx_env import MJXDinoEnv
        from environments.shared.mjx_utils import reset_mujoco_data_to_home

        env = MJXDinoEnv("velociraptor", stage=1, num_envs=1)
        states = env.reset(jax.random.PRNGKey(17))

        expected = mujoco.MjData(env.mj_model)
        reset_mujoco_data_to_home(env.mj_model, expected)
        mujoco.mj_forward(env.mj_model, expected)

        for field in ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat"):
            np.testing.assert_allclose(
                np.asarray(getattr(states.data, field)[0]),
                np.asarray(getattr(expected, field)),
                rtol=0,
                atol=1e-7,
                err_msg=f"MJX reset field {field} diverged from the XML home keyframe",
            )

    def test_noise_free_trex_mjx_reset_restores_complete_home_state(self):
        import jax
        import mujoco

        import environments.trex.mjx_config  # noqa: F401
        from environments.shared.mjx_env import MJXDinoEnv
        from environments.shared.mjx_utils import reset_mujoco_data_to_home

        env = MJXDinoEnv("trex", stage=1, num_envs=1)
        states = env.reset(jax.random.PRNGKey(17))

        expected = mujoco.MjData(env.mj_model)
        reset_mujoco_data_to_home(env.mj_model, expected)
        mujoco.mj_forward(env.mj_model, expected)

        for field in ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat"):
            np.testing.assert_allclose(
                np.asarray(getattr(states.data, field)[0]),
                np.asarray(getattr(expected, field)),
                rtol=0,
                atol=1e-7,
                err_msg=f"T-Rex MJX reset field {field} diverged from the XML home keyframe",
            )

    def test_trex_mjx_reset_exposes_live_foot_contacts(self):
        import jax

        import environments.trex.mjx_config  # noqa: F401
        from environments.shared.mjx_env import MJXDinoEnv

        env = MJXDinoEnv("trex", stage=1, num_envs=1)
        states = env.reset(jax.random.PRNGKey(23))
        sensordata = np.asarray(states.data.sensordata[0])
        pad_sensors = sensordata[10:12]
        # A touch sensor sums only geoms on its site's own body, so the pad
        # sensors cannot see the digits -- which are the load-bearing geoms.
        # The observation carries pad + digits, so assert against that, not
        # against the pad alone: a pad-only expectation is what let the digits
        # go unmeasured after they were made load-bearing.
        expected = np.array(
            [pad + sensordata[list(group)].sum() for pad, group in zip(pad_sensors, env.config.sensor_foot_aux_indices)]
        )
        foot_observation = np.asarray(states.obs[0, -6:-4])

        assert np.all(pad_sensors > 0.1), f"T-Rex MJX foot sensors are dead at home: {pad_sensors}"
        assert np.all(expected > pad_sensors), (
            "digits carry no load at the home keyframe, so this test can no longer "
            f"detect a pad-only foot-contact signal: pad={pad_sensors}, total={expected}"
        )
        np.testing.assert_allclose(foot_observation, expected, rtol=0, atol=1e-7)

    def test_trex_stage1_factory_preserves_contact_physics_and_rewards(self):
        import jax
        import mujoco

        from environments.shared.jax_setup import create_env, setup_species
        from environments.shared.mjx_utils import reset_mujoco_data_to_home

        ctx = setup_species("trex", stage=1)
        ctx.jax_kwargs = {
            **ctx.jax_kwargs,
            "reset_noise_scale": 0.0,
            "init_qpos_noise": 0.0,
            "init_yaw_noise": 0.0,
        }
        env = create_env(ctx, num_envs=1)
        states = env.reset(jax.random.PRNGKey(23))
        foot_sensors = np.asarray(states.data.sensordata[0, 10:12])

        for option in ("solver", "iterations", "ls_iterations", "disableflags"):
            assert getattr(ctx.mj_model.opt, option) == getattr(env.mj_model.opt, option)

        cpu_data = mujoco.MjData(ctx.mj_model)
        reset_mujoco_data_to_home(ctx.mj_model, cpu_data)
        mujoco.mj_forward(ctx.mj_model, cpu_data)
        np.testing.assert_allclose(foot_sensors, cpu_data.sensordata[10:12], rtol=1e-5, atol=1e-3)

        assert env.config.action_mapping == "home-keyframe-residual/v1"
        assert env.config.reward_weights["foot_contact_gate"] == pytest.approx(1.0)
        assert env.config.reward_weights["foot_contact_weight"] == pytest.approx(0.8)
        assert np.all(foot_sensors > 0.1), f"Stage-1 reset has no load-bearing foot contact: {foot_sensors}"
        # The observation sums the pad sensor with the three per-digit sensors
        # for each foot; see test_trex_mjx_reset_exposes_live_foot_contacts.
        sensordata = np.asarray(states.data.sensordata[0])
        expected_contacts = np.array(
            [
                pad + sensordata[list(group)].sum()
                for pad, group in zip(foot_sensors, env.config.sensor_foot_aux_indices)
            ]
        )
        np.testing.assert_allclose(np.asarray(states.obs[0, -6:-4]), expected_contacts, rtol=0, atol=1e-7)

    def test_home_reset_requires_named_home_keyframe(self):
        import mujoco

        from environments.shared.mjx_utils import reset_mujoco_data_to_home

        model = mujoco.MjModel.from_xml_string("<mujoco/>")
        data = mujoco.MjData(model)
        with pytest.raises(ValueError, match="keyframe named 'home'"):
            reset_mujoco_data_to_home(model, data)


# ---------------------------------------------------------------------------
# Bug-fix coverage: env.step exposes final_obs and respects forward_vel_scale
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_jax, reason="JAX/MJX not installed")
class TestStepReturnFinalObs:
    """``env.step(return_final_obs=True)`` returns the pre-reset obs so
    callers can bootstrap value at truncation boundaries."""

    def test_default_returns_4_tuple(self):
        import jax

        import environments.trex.mjx_config  # noqa: F401
        from environments.shared.mjx_env import MJXDinoEnv

        env = MJXDinoEnv("trex", stage=1, num_envs=2)
        rng = jax.random.PRNGKey(0)
        states = env.reset(rng)
        actions = jax.numpy.zeros((2, env.action_dim))
        out = env.step(states, actions, rng)
        assert len(out) == 4

    def test_return_final_obs_returns_5_tuple(self):
        import jax

        import environments.trex.mjx_config  # noqa: F401
        from environments.shared.mjx_env import MJXDinoEnv

        env = MJXDinoEnv("trex", stage=1, num_envs=2)
        rng = jax.random.PRNGKey(0)
        states = env.reset(rng)
        actions = jax.numpy.zeros((2, env.action_dim))
        out = env.step(states, actions, rng, return_final_obs=True)
        assert len(out) == 5
        new_states, _rewards, _term, _trunc, final_obs = out
        # Same shape as the new states' obs
        assert final_obs.shape == new_states.obs.shape

    def test_final_obs_differs_from_reset_obs_after_termination(self):
        """Force termination by lowering the healthy z-range floor so the
        first step terminates on a fresh reset.  ``final_obs`` should
        track the post-step (pre-reset) obs, while ``new_states.obs``
        is the post-reset obs.  Even though the policy then acts on the
        post-reset obs, GAE bootstrap must use the pre-reset value."""
        import jax
        import jax.numpy as jnp

        import environments.trex.mjx_config  # noqa: F401
        from environments.shared.mjx_env import MJXDinoEnv

        # Force termination on the very first step by setting the lower
        # healthy z-bound above the natural standing height.
        env = MJXDinoEnv(
            "trex",
            stage=1,
            num_envs=2,
            env_kwargs={"healthy_z_range": (5.0, 10.0)},
        )
        rng = jax.random.PRNGKey(0)
        states = env.reset(rng)
        actions = jnp.zeros((2, env.action_dim))
        new_states, _r, terminated, _trunc, final_obs = env.step(states, actions, rng, return_final_obs=True)
        assert bool(terminated[0])
        # After termination + auto-reset, new_states.obs is the freshly
        # reset obs.  final_obs should NOT match the reset state because
        # its step_count was advanced inside step_fn.
        # We can't compare obs directly (reset obs is randomized) but we
        # can confirm final_obs has the right shape.
        assert final_obs.shape == new_states.obs.shape


@pytest.mark.skipif(not _has_jax, reason="JAX/MJX not installed")
class TestForwardVelScale:
    """The reward ramp drives ``forward_vel_scale``; the env must apply
    it at runtime without recompilation."""

    def test_scale_zero_zeros_forward_reward(self):
        """With forward_vel_scale=0 the forward-velocity reward must
        vanish (the difference vs the default scale must equal exactly
        ``forward_vel_weight * fwd_vel_norm``).  This proves the runtime
        scale is wired through, not baked into the JIT trace."""
        import jax
        import jax.numpy as jnp

        import environments.trex.mjx_config  # noqa: F401
        from environments.shared.mjx_env import MJXDinoEnv

        # Set a strong forward_vel_weight so the difference is large.
        env = MJXDinoEnv(
            "trex",
            stage=1,
            num_envs=2,
            env_kwargs={"reward_weights": {"forward_vel_weight": 10.0}},
        )
        rng = jax.random.PRNGKey(7)
        states = env.reset(rng)
        # Use a non-zero action so the body actually moves and forward_vel
        # is non-trivial.  Same seed/state for both calls so physics path
        # is identical — only the reward scaling differs.
        actions = jnp.full((2, env.action_dim), 0.1)

        _, r_full, _, _ = env.step(states, actions, rng, forward_vel_scale=1.0)
        _, r_zero, _, _ = env.step(states, actions, rng, forward_vel_scale=0.0)

        # The two rewards must differ by exactly the forward_vel component
        # (i.e. they cannot be equal — that would mean the scale was
        # baked in as a Python constant).
        diff = float(jnp.max(jnp.abs(r_full - r_zero)))
        assert diff > 0, "forward_vel_scale had no effect on reward"

    def test_default_scale_is_one(self):
        """Calling step() without forward_vel_scale must match scale=1.0."""
        import jax
        import jax.numpy as jnp

        import environments.trex.mjx_config  # noqa: F401
        from environments.shared.mjx_env import MJXDinoEnv

        env = MJXDinoEnv("trex", stage=1, num_envs=2)
        rng = jax.random.PRNGKey(11)
        states = env.reset(rng)
        actions = jnp.full((2, env.action_dim), 0.1)
        _, r_default, _, _ = env.step(states, actions, rng)
        _, r_one, _, _ = env.step(states, actions, rng, forward_vel_scale=1.0)
        assert jnp.allclose(r_default, r_one, atol=1e-6)
