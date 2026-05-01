"""Tests for jax_trainer and jax_hooks modules."""

from __future__ import annotations

import pytest

from environments.shared.jax_hooks import BestModelHook, CSVLoggingHook, LoggingHook, StabilityHook
from environments.shared.jax_trainer import JaxTrainer, StopTraining, TrainerState

_has_jax = False
try:
    import jax  # noqa: F401
    import mujoco.mjx  # noqa: F401

    _has_jax = True
except ImportError:
    pass


class TestTrainerState:
    def test_defaults(self):
        state = TrainerState(
            params=None,
            opt_state=None,
            obs_stats=None,
            env_states=None,
            rng=None,
        )
        assert state.update == 0
        assert state.total_steps == 0
        assert state.history == []
        assert state.reward_history == []
        assert state.loss_history == []
        assert state.episode_return_history == []
        assert state.t_rollout_cumulative == 0.0
        assert state.t_ppo_cumulative == 0.0

    def test_history_is_independent(self):
        s1 = TrainerState(
            params=None,
            opt_state=None,
            obs_stats=None,
            env_states=None,
            rng=None,
        )
        s2 = TrainerState(
            params=None,
            opt_state=None,
            obs_stats=None,
            env_states=None,
            rng=None,
        )
        s1.history.append({"x": 1})
        s1.reward_history.append(1.0)
        assert len(s2.history) == 0
        assert len(s2.reward_history) == 0


class TestStopTraining:
    def test_with_reason(self):
        exc = StopTraining("KL exploded")
        assert exc.reason == "KL exploded"
        assert str(exc) == "KL exploded"

    def test_empty_reason(self):
        exc = StopTraining()
        assert exc.reason == ""


class TestLoggingHook:
    def test_interval(self):
        hook = LoggingHook(interval=5, num_updates=100)
        assert hook.interval == 5
        assert hook.num_updates == 100

    def test_on_update_end_no_crash(self):
        hook = LoggingHook(interval=1, num_updates=10)
        state = TrainerState(
            params=None,
            opt_state=None,
            obs_stats=None,
            env_states=None,
            rng=None,
            update=0,
            total_steps=100,
        )
        # Should not raise
        hook.on_update_end(state, {"update": 0, "mean_reward": 1.5, "fps": 1000.0})

    def test_skips_non_interval(self, caplog):
        hook = LoggingHook(interval=10, num_updates=100)
        state = TrainerState(
            params=None,
            opt_state=None,
            obs_stats=None,
            env_states=None,
            rng=None,
            update=3,
            total_steps=100,
        )
        import logging

        with caplog.at_level(logging.INFO):
            hook.on_update_end(state, {"update": 3, "mean_reward": 1.0, "fps": 500.0})
        # Should NOT log for update 3 with interval 10
        assert "Update" not in caplog.text


class TestStabilityHook:
    def test_stable_no_halt(self):
        hook = StabilityHook()
        state = TrainerState(
            params=None,
            opt_state=None,
            obs_stats=None,
            env_states=None,
            rng=None,
            update=0,
        )
        # Normal metrics — should not raise
        hook.on_update_end(
            state,
            {
                "approx_kl": 0.01,
                "grad_norm": 5.0,
                "total_loss": 1.0,
            },
        )

    def test_kl_halt_raises(self):
        hook = StabilityHook(kl_halt=1e6)
        state = TrainerState(
            params=None,
            opt_state=None,
            obs_stats=None,
            env_states=None,
            rng=None,
            update=5,
        )
        with pytest.raises(StopTraining, match="HALTING"):
            hook.on_update_end(
                state,
                {
                    "approx_kl": 1e7,
                    "grad_norm": 5.0,
                    "total_loss": 1.0,
                },
            )

    def test_consecutive_warnings_halt(self):
        hook = StabilityHook(kl_warn=0.5, max_warnings=3)
        state = TrainerState(
            params=None,
            opt_state=None,
            obs_stats=None,
            env_states=None,
            rng=None,
            update=0,
        )
        metrics = {"approx_kl": 1.0, "grad_norm": 0.0, "total_loss": 0.0}
        # First two warnings — no halt
        hook.on_update_end(state, metrics)
        hook.on_update_end(state, metrics)
        # Third consecutive warning — should halt
        with pytest.raises(StopTraining, match="HALTING"):
            hook.on_update_end(state, metrics)

    def test_monitor_accessible(self):
        hook = StabilityHook(kl_warn=10.0)
        assert hook.monitor.kl_warn == 10.0


class TestTrainingHookProtocol:
    """Verify that arbitrary objects implementing hook methods are accepted."""

    def test_custom_hook_class(self):
        class MyHook:
            def __init__(self):
                self.updates = []

            def on_update_end(self, state, metrics):
                self.updates.append(metrics)

        hook = MyHook()
        # Ensure JaxTrainer accepts it (structural typing check)
        trainer = JaxTrainer.__new__(JaxTrainer)
        trainer.hooks = [hook]
        trainer._dispatch("on_update_end", None, {"reward": 1.0})
        assert len(hook.updates) == 1
        assert hook.updates[0] == {"reward": 1.0}

    def test_partial_hook(self):
        """Hook that only implements on_train_end still works."""

        class EndOnlyHook:
            def __init__(self):
                self.called = False

            def on_train_end(self, state):
                self.called = True

        hook = EndOnlyHook()
        trainer = JaxTrainer.__new__(JaxTrainer)
        trainer.hooks = [hook]
        # Dispatching a method the hook doesn't have should not crash
        trainer._dispatch("on_update_end", None, {})
        assert not hook.called
        trainer._dispatch("on_train_end", None)
        assert hook.called


class TestBestModelHook:
    def test_tracks_improvement(self):
        hook = BestModelHook(metric_key="mean_reward")
        state = TrainerState(
            params={"w": 1.0},
            opt_state=None,
            obs_stats=None,
            env_states=None,
            rng=None,
            update=0,
        )
        hook.on_update_end(state, {"update": 0, "mean_reward": 1.0})
        assert hook.best_reward == 1.0
        assert hook.best_update == 0

        state.params = {"w": 2.0}
        hook.on_update_end(state, {"update": 5, "mean_reward": 2.0})
        assert hook.best_reward == 2.0
        assert hook.best_update == 5

    def test_ignores_regression(self):
        hook = BestModelHook()
        state = TrainerState(
            params={"w": 1.0},
            opt_state=None,
            obs_stats=None,
            env_states=None,
            rng=None,
        )
        hook.on_update_end(state, {"update": 0, "mean_reward": 5.0})
        hook.on_update_end(state, {"update": 1, "mean_reward": 3.0})
        assert hook.best_reward == 5.0
        assert hook.best_update == 0


class TestCSVLoggingHook:
    def test_creates_and_writes(self, tmp_path):
        csv_path = tmp_path / "log.csv"
        hook = CSVLoggingHook(path=csv_path)
        state = TrainerState(
            params=None,
            opt_state=None,
            obs_stats=None,
            env_states=None,
            rng=None,
            update=0,
            total_steps=1000,
        )
        hook.on_update_end(state, {"update": 0, "mean_reward": 1.5, "fps": 500.0})
        hook.on_train_end(state)
        assert csv_path.exists()
        content = csv_path.read_text()
        assert "update" in content
        assert "1.5" in content

    def test_path_property(self, tmp_path):
        csv_path = tmp_path / "log2.csv"
        hook = CSVLoggingHook(path=csv_path)
        assert hook.path == csv_path


# ---------------------------------------------------------------------------
# End-to-end JaxTrainer tests — guard the bugs fixed in this review.
# Skipped when JAX/MJX is not installed.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_jax, reason="JAX/MJX not installed")
class TestJaxTrainerSmoke:
    """One-update smoke run that catches obs-dim and rollout-tuple bugs."""

    def _build(self, num_envs=4, rollout_len=4, n_minibatches=2):
        import environments.trex.mjx_config  # noqa: F401
        from environments.shared.jax_ppo import PPOConfig, make_actor_critic, make_optimizer
        from environments.shared.mjx_env import MJXDinoEnv

        env = MJXDinoEnv("trex", stage=1, num_envs=num_envs)
        ppo_cfg = PPOConfig(
            n_epochs=1,
            n_minibatches=n_minibatches,
            target_kl=None,  # disable KL early-stop for determinism
            total_updates=2,
        )
        network = make_actor_critic(env.action_dim, hidden_dims=(8,))
        optimizer = make_optimizer(ppo_cfg)
        trainer = JaxTrainer(
            env=env,
            network=network,
            optimizer=optimizer,
            ppo_config=ppo_cfg,
            num_envs=num_envs,
            rollout_len=rollout_len,
        )
        return trainer, env

    def test_one_update_runs(self):
        """Bug #1 regression: network init obs_dim must match the env's
        actual obs (which includes foot contacts).  A wrong dummy_obs
        size causes a shape error on the first ``apply``."""
        trainer, _env = self._build()
        params, metrics, state = trainer.train(num_updates=1, seed=0)
        assert state.update == 0
        assert state.total_steps == trainer.num_envs * trainer.rollout_len
        assert "mean_reward" in metrics

    def test_rollout_unpacks_8_tuple(self):
        """Bug #2 regression: ``_build_collect_rollout`` returns the
        new (raw_obs, action, log_prob, value, reward, gae_done,
        full_done, final_obs) layout — not the old 6-tuple."""
        import jax

        trainer, env = self._build()
        # Monkeypatch in init so we exercise the build but stop early.
        rng = jax.random.PRNGKey(0)
        states = env.reset(rng)
        collect = trainer._build_collect_rollout()
        from environments.shared.jax_normalization import RunningMeanStd

        stats = RunningMeanStd.create(int(states.obs.shape[-1]))
        # We need params to actually run the network — re-use trainer.train
        # to do init then call collect with returned params.
        params, _metrics, tstate = trainer.train(num_updates=1, seed=0)
        (_states, _), rollout = collect(tstate.env_states, rng, params, tstate.obs_stats)
        assert len(rollout) == 8
        raw_obs, raw_action, log_prob, value, reward, gae_done, full_done, final_obs = rollout
        assert raw_obs.shape[-1] == int(states.obs.shape[-1])
        assert final_obs.shape == raw_obs.shape
        # gae_done <= full_done elementwise (truncation cannot fire without
        # also flipping full_done)
        import jax.numpy as jnp

        assert bool(jnp.all(gae_done <= full_done))


@pytest.mark.skipif(not _has_jax, reason="JAX/MJX not installed")
class TestNotebookTrainRewardRamp:
    """Bug #5 regression: the notebook ``train()`` reward ramp must
    actually affect the env's reward, and must reject unsupported
    ramp targets rather than silently doing nothing."""

    def _make_train_inputs(self, *, ramp_attr="forward_vel_weight", ramp_updates=2):
        import environments.trex.mjx_config  # noqa: F401
        import jax
        import jax.numpy as jnp  # noqa: F401

        from environments.shared.jax_ppo import PPOConfig, make_actor_critic, make_optimizer
        from environments.shared.jax_normalization import RunningMeanStd
        from environments.shared.jax_trainer import TrainConfig
        from environments.shared.mjx_env import MJXDinoEnv

        num_envs, rollout_len = 4, 4
        env = MJXDinoEnv(
            "trex",
            stage=1,
            num_envs=num_envs,
            env_kwargs={"reward_weights": {"forward_vel_weight": 1.0}},
        )
        ppo_cfg = PPOConfig(n_epochs=1, n_minibatches=2, target_kl=None, total_updates=2)
        network = make_actor_critic(env.action_dim, hidden_dims=(8,))
        optimizer = make_optimizer(ppo_cfg)

        rng = jax.random.PRNGKey(0)
        rng, init_rng, reset_rng = jax.random.split(rng, 3)
        states = env.reset(reset_rng)
        obs_dim = int(states.obs.shape[-1])
        params = network.init(init_rng, states.obs[0])
        opt_state = optimizer.init(params)
        obs_rms = RunningMeanStd.create(obs_dim)

        cfg = TrainConfig(
            num_envs=num_envs,
            rollout_len=rollout_len,
            num_updates=2,
            ppo_epochs=1,
            minibatch_size=8,
            obs_dim=obs_dim,
            act_dim=env.action_dim,
            ramp_updates=ramp_updates,
            ramp_attr=ramp_attr,
            ramp_start_fraction=0.1,
            verbose=0,
            output_dir=".",
            model_dir=".",
            checkpoint_freq=10,
        )
        return cfg, env, network, params, opt_state, obs_rms, optimizer, rng

    def test_unsupported_ramp_attr_raises(self, tmp_path):
        from environments.shared.jax_trainer import train

        cfg, env, network, params, opt_state, obs_rms, optimizer, rng = self._make_train_inputs(
            ramp_attr="energy_penalty_weight"
        )
        cfg.output_dir = tmp_path
        cfg.model_dir = tmp_path
        with pytest.raises(ValueError, match="forward_vel_weight"):
            train(
                cfg,
                env,
                network,
                params,
                opt_state,
                obs_rms,
                reward_cfg={"energy_penalty_weight": 0.5},
                rng=rng,
                optimizer=optimizer,
            )


@pytest.mark.skipif(not _has_jax, reason="JAX/MJX not installed")
class TestJaxTrainerObsStatsRawOnly:
    """Bug #4 regression: obs_rms must be updated from RAW obs only."""

    def test_running_stats_track_raw_distribution(self):
        """After several updates, mean and std should reflect the raw
        obs distribution — NOT collapse toward (0, 1) which would
        indicate stats were updated using already-normalized obs."""
        import environments.trex.mjx_config  # noqa: F401
        from environments.shared.jax_ppo import PPOConfig, make_actor_critic, make_optimizer
        from environments.shared.mjx_env import MJXDinoEnv

        num_envs, rollout_len = 4, 4
        env = MJXDinoEnv("trex", stage=1, num_envs=num_envs)
        ppo_cfg = PPOConfig(n_epochs=1, n_minibatches=2, target_kl=None, total_updates=3)
        network = make_actor_critic(env.action_dim, hidden_dims=(8,))
        optimizer = make_optimizer(ppo_cfg)
        trainer = JaxTrainer(
            env=env,
            network=network,
            optimizer=optimizer,
            ppo_config=ppo_cfg,
            num_envs=num_envs,
            rollout_len=rollout_len,
        )
        _params, _metrics, state = trainer.train(num_updates=3, seed=0)
        # Raw obs include qpos with z-position around standing height
        # (positive, ~0.6-1.0).  If obs_rms were being fed normalized
        # obs, the mean would converge toward 0 across updates.  We
        # check that at least one channel of running mean is clearly
        # away from zero, indicating raw-obs accumulation is working.
        import jax.numpy as jnp

        obs_mean = state.obs_stats.mean
        assert float(jnp.max(jnp.abs(obs_mean))) > 0.05, (
            f"obs_rms.mean is suspiciously close to zero (max abs = "
            f"{float(jnp.max(jnp.abs(obs_mean))):.4f}); stats may be "
            "updated from normalized obs"
        )
