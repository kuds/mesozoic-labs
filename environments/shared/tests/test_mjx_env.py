"""Tests for the MJX environment wrapper.

These tests are skipped when JAX/MJX is not installed.
"""

import pytest

_has_jax = False
try:
    import jax  # noqa: F401
    import mujoco.mjx  # noqa: F401

    _has_jax = True
except ImportError:
    pass


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
        assert env.action_dim == 26
        assert env.num_envs == 4

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
