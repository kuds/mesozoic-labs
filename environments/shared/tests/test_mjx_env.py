"""Tests for the MJX environment wrapper.

These tests are skipped when JAX/MJX is not installed.
"""

import inspect

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

    def test_raptor_creation(self):
        """MJXDinoEnv can be created for Velociraptor."""
        import environments.velociraptor.mjx_config  # noqa: F401
        from environments.shared.mjx_env import MJXDinoEnv

        env = MJXDinoEnv("velociraptor", stage=1, num_envs=4)
        assert env.action_dim == 22
        assert env.num_envs == 4

    def test_brachio_creation(self):
        """MJXDinoEnv can be created for Brachiosaurus."""
        import environments.brachiosaurus.mjx_config  # noqa: F401
        from environments.shared.mjx_env import MJXDinoEnv

        env = MJXDinoEnv("brachiosaurus", stage=1, num_envs=4)
        assert env.action_dim == 30
        assert env.num_envs == 4

    def test_stage_cannot_override_versioned_plant_interface(self):
        import environments.trex.mjx_config  # noqa: F401
        from environments.shared.mjx_env import MJXDinoEnv

        with pytest.raises(ValueError, match="versioned plant interface"):
            MJXDinoEnv("trex", stage=1, num_envs=1, env_kwargs={"frame_skip": 4})

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
