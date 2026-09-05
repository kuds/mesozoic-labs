"""Tests for the training-time command low-pass filter.

The filter is a plant-interface feature (see action_filter.py): trex enables
it at 10 Hz in both backends, every other species leaves it off and must keep
the exact legacy step arithmetic.  The SB3 and MJX implementations share the
alpha formula with the stance-gate probe filter, so probe cutoffs and plant
cutoffs are directly comparable.
"""

from __future__ import annotations

import numpy as np
import pytest

from environments.shared.action_filter import apply_low_pass, low_pass_alpha
from environments.trex.envs.trex_env import TRexEnv


class TestLowPassPrimitives:
    def test_alpha_matches_probe_discretization(self) -> None:
        # alpha = dt / (RC + dt), RC = 1 / (2*pi*fc): the formula
        # _low_pass_predict in reporting/stance_report.py has always used.
        dt = 0.01
        rc = 1.0 / (2.0 * np.pi * 10.0)
        assert low_pass_alpha(10.0, dt) == pytest.approx(dt / (rc + dt))

    def test_alpha_rejects_disabled_cutoff(self) -> None:
        # A zero cutoff means "no filter"; callers gate on that themselves.
        with pytest.raises(ValueError):
            low_pass_alpha(0.0, 0.01)
        with pytest.raises(ValueError):
            low_pass_alpha(10.0, 0.0)

    def test_dc_passthrough(self) -> None:
        # A constant input is a fixed point of the update.
        state = np.full(3, 0.25)
        assert apply_low_pass(state, np.full(3, 0.25), 0.4) == pytest.approx(state)

    def test_nyquist_alternation_is_attenuated(self) -> None:
        # +-1 alternation at the control Nyquist rate must decay towards a
        # small residual around zero once the transient settles.
        alpha = low_pass_alpha(5.0, 0.01)
        y = np.array([1.0])
        for i in range(200):
            x = np.array([1.0 if i % 2 == 0 else -1.0])
            y = apply_low_pass(y, x, alpha)
        assert abs(float(y[0])) < 0.4


class TestSB3ActionFilter:
    def test_trex_declares_ten_hz(self) -> None:
        env = TRexEnv()
        assert env.action_filter_cutoff_hz == 10.0
        assert env._action_filter_alpha == pytest.approx(low_pass_alpha(10.0, env.dt))

    def test_cutoff_is_not_a_constructor_kwarg(self) -> None:
        # Plant-level by construction: stage TOML [env] tables become
        # constructor kwargs, so the cutoff must not be accepted there.
        with pytest.raises(TypeError):
            TRexEnv(action_filter_cutoff_hz=5.0)  # type: ignore[call-arg]

    def test_first_step_seeds_with_the_action(self) -> None:
        # No transient toward zero at the episode boundary: the first
        # filtered command equals the first (clipped) action exactly.
        env = TRexEnv()
        env.reset(seed=0)
        action = np.full(env.model.nu, 0.5, dtype=np.float32)
        env.step(action)
        expected = env._scale_action(np.full(env.model.nu, 0.5))
        assert env.data.ctrl == pytest.approx(expected)

    def test_blend_and_reseed_across_reset(self) -> None:
        env = TRexEnv()
        env.reset(seed=0)
        up = np.ones(env.model.nu, dtype=np.float32)
        down = -up
        env.step(up)
        env.step(down)
        alpha = env._action_filter_alpha
        assert env._action_filter_state == pytest.approx(1.0 + alpha * (-1.0 - 1.0))
        # Reset invalidates the carry through _step_count, like the substep
        # aggregates; the next episode seeds fresh with its own first action.
        env.reset(seed=1)
        env.step(down)
        assert env._action_filter_state == pytest.approx(-1.0)

    def test_disabled_cutoff_preserves_legacy_arithmetic(self) -> None:
        class UnfilteredTRexEnv(TRexEnv):
            action_filter_cutoff_hz = 0.0

        env = UnfilteredTRexEnv()
        env.reset(seed=0)
        rng = np.random.default_rng(3)
        for _ in range(3):
            action = rng.uniform(-1.0, 1.0, env.model.nu).astype(np.float32)
            env.step(action)
            # ctrl reflects the raw action every step -- no carried state.
            assert env.data.ctrl == pytest.approx(env._scale_action(action))
        assert env._action_filter_state is None

    def test_filtered_command_feeds_the_reward_terms(self) -> None:
        # The smoothness/jerk history must hold the filtered command, not
        # the raw policy output: raw content above the cutoff never reaches
        # the plant, and pricing it would penalise a phantom.
        env = TRexEnv()
        env.reset(seed=0)
        up = np.ones(env.model.nu, dtype=np.float32)
        env.step(up)
        env.step(-up)
        assert env._prev_action == pytest.approx(env._action_filter_state)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestMJXActionFilter:
    @pytest.fixture(scope="class")
    def mjx_env(self):
        pytest.importorskip("jax")
        pytest.importorskip("mujoco.mjx")
        import environments.trex.mjx_config  # noqa: F401  (registers trex)
        from environments.shared.mjx_env import MJXDinoEnv

        return MJXDinoEnv("trex", stage=1, num_envs=1)

    def test_registered_cutoff_matches_sb3(self, mjx_env) -> None:
        assert mjx_env.config.action_filter_cutoff_hz == TRexEnv.action_filter_cutoff_hz

    def test_stage_toml_cannot_override_the_cutoff(self) -> None:
        pytest.importorskip("jax")
        pytest.importorskip("mujoco.mjx")
        import environments.trex.mjx_config  # noqa: F401
        from environments.shared.mjx_env import MJXDinoEnv

        with pytest.raises(ValueError, match="versioned plant interface"):
            MJXDinoEnv("trex", stage=1, num_envs=1, env_kwargs={"action_filter_cutoff_hz": 5.0})

    def test_step_seeds_then_blends_in_lockstep_with_sb3(self, mjx_env) -> None:
        import jax
        import jax.numpy as jnp

        rng = jax.random.PRNGKey(0)
        state = mjx_env._reset_single(rng)
        assert state.filtered_action.shape == (mjx_env.action_dim,)

        up = jnp.ones(mjx_env.action_dim)
        state1, _, _, _ = mjx_env._step_single(state, up, rng, jnp.float32(1.0))
        # Seeded: the first filtered command is the clipped action itself,
        # and prev_action carries the filtered signal for the reward lags.
        np.testing.assert_allclose(np.asarray(state1.filtered_action), 1.0, atol=1e-6)
        np.testing.assert_allclose(np.asarray(state1.prev_action), 1.0, atol=1e-6)

        alpha = low_pass_alpha(10.0, float(mjx_env.mj_model.opt.timestep) * mjx_env.config.frame_skip)
        state2, _, _, _ = mjx_env._step_single(state1, -up, rng, jnp.float32(1.0))
        np.testing.assert_allclose(
            np.asarray(state2.filtered_action),
            1.0 + alpha * (-1.0 - 1.0),
            atol=1e-6,
        )
