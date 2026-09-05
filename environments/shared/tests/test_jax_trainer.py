"""Tests for jax_trainer and jax_hooks modules."""

from __future__ import annotations

import inspect

import pytest

from environments.shared.jax_hooks import BestModelHook, CSVLoggingHook, LoggingHook, StabilityHook
from environments.shared.jax_trainer import JaxTrainer, StopTraining, TrainerState


def test_headless_training_binds_runtime_plant_before_writing_artifacts():
    from environments.shared.jax_training import train_jax

    source = inspect.getsource(train_jax)
    env_created = source.index("env = MJXDinoEnv(")
    env_validated = source.index("validate_mjx_environment_plant(")
    identity_written = source.index("write_plant_identity(")
    assert env_created < env_validated < identity_written


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

    def _build(self, num_envs=4, rollout_len=4, n_minibatches=2, **trainer_kwargs):
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
            **trainer_kwargs,
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
        rng = jax.random.PRNGKey(0)
        states = env.reset(rng)
        collect = trainer._build_collect_rollout()

        # We need params to actually run the network — re-use trainer.train
        # to do init then call collect with the returned params.
        import jax.numpy as _jnp

        params, _metrics, tstate = trainer.train(num_updates=1, seed=0)
        (_states, _), rollout = collect(tstate.env_states, rng, params, tstate.obs_stats, _jnp.float32(1.0))
        assert len(rollout) == 8
        raw_obs, _raw_action, _log_prob, _value, _reward, gae_done, full_done, final_obs = rollout
        assert raw_obs.shape[-1] == int(states.obs.shape[-1])
        assert final_obs.shape == raw_obs.shape
        # gae_done <= full_done elementwise (truncation cannot fire without
        # also flipping full_done)
        import jax.numpy as jnp

        assert bool(jnp.all(gae_done <= full_done))

    def test_warmup_and_ramp_smoke(self):
        """warmup_updates / ramp_updates are wired through the class
        trainer (previously TOML warmup_*/ramp_* were silently inert on
        the CLI path)."""
        trainer, _env = self._build(
            warmup_updates=1,
            warmup_clip_range=0.05,
            warmup_ent_coef=0.03,
            ramp_updates=2,
            ramp_start_fraction=0.5,
        )
        params, metrics, state = trainer.train(num_updates=2, seed=0)
        assert trainer._scan_ppo_update_warmup is not None
        assert state.update == 1
        assert "mean_reward" in metrics

    def test_resume_accepts_opt_state_and_obs_stats(self):
        """init_opt_state / init_obs_stats round-trip through train()."""
        trainer, _env = self._build()
        params, _metrics, state = trainer.train(num_updates=1, seed=0)
        trainer2, _ = self._build()
        params2, _m2, state2 = trainer2.train(
            num_updates=1,
            seed=1,
            init_params=state.params,
            init_opt_state=state.opt_state,
            init_obs_stats=state.obs_stats,
        )
        assert state2.total_steps == trainer2.num_envs * trainer2.rollout_len


@pytest.mark.skipif(not _has_jax, reason="JAX/MJX not installed")
class TestNotebookTrainRewardRamp:
    """Bug #5 regression: the notebook ``train()`` reward ramp must
    actually affect the env's reward, and must reject unsupported
    ramp targets rather than silently doing nothing."""

    def _make_train_inputs(self, *, ramp_attr="forward_vel_weight", ramp_updates=2):
        import jax

        import environments.trex.mjx_config  # noqa: F401
        from environments.shared.jax_normalization import RunningMeanStd
        from environments.shared.jax_ppo import PPOConfig, make_actor_critic, make_optimizer
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


class TestRampAttrGuard:
    """The TOML ramp_attr key is honored with the same contract everywhere:
    'forward_vel_weight' works, anything else fails at construction — the MJX
    env bakes other weights into the jitted reward at trace time, so a
    different attr cannot ramp and must not pretend to."""

    def test_an_unsupported_ramp_attr_is_refused_at_construction(self):
        with pytest.raises(ValueError, match="forward_vel_weight"):
            JaxTrainer(
                env=None,
                network=None,
                optimizer=None,
                ppo_config=None,
                ramp_updates=5,
                ramp_attr="tail_stability_weight",
            )

    def test_the_supported_attr_and_a_disabled_ramp_construct_fine(self):
        trainer = JaxTrainer(env=None, network=None, optimizer=None, ppo_config=None, ramp_updates=5)
        assert trainer.ramp_attr == "forward_vel_weight"
        # With the ramp disabled the attr is inert, so any value is tolerated.
        JaxTrainer(env=None, network=None, optimizer=None, ppo_config=None, ramp_updates=0, ramp_attr="whatever")


# ---------------------------------------------------------------------------
# JX5 — per-step time-limit bootstrap
# ---------------------------------------------------------------------------


def _gae_segment(rewards, values, bootstrap, gamma, lam):
    """Plain-Python GAE over ONE episode segment with an explicit tail bootstrap."""
    adv = [0.0] * len(rewards)
    carry = 0.0
    next_value = bootstrap
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value - values[t]
        carry = delta + gamma * lam * carry
        adv[t] = carry
        next_value = values[t]
    return adv


@pytest.mark.skipif(not _has_jax, reason="JAX not installed")
class TestTruncationBootstrap:
    """A time-limit truncation at a NON-final rollout position must bootstrap
    ``gamma * V(final_obs_t)`` and stop the lambda carry at the boundary.

    Both trainers used the termination-only mask for GAE and ``final_obs``
    only for the rollout's last step, so a mid-rollout truncation bootstrapped
    the post-auto-reset obs and leaked the next episode's advantages into the
    ended one.  With synchronized resets, ``max_episode_steps=1000`` and
    ``rollout_len=64`` (1000 mod 64 = 40) the whole fleet does this at once as
    soon as a policy reaches the horizon — the regime the stance gate selects
    for.
    """

    GAMMA, LAM = 0.9, 0.8
    REWARDS = [1.0, 2.0, 3.0, 4.0]
    VALUES = [0.5, 1.0, 1.5, 2.0]
    TAIL = 2.5  # V(s_T) — value of the last step's final_obs

    def _arrays(self, jnp, *, gae_dones, full_dones, final_values):
        column = lambda xs: jnp.array([[x] for x in xs], dtype=jnp.float32)  # noqa: E731
        values_arr = jnp.concatenate([column(self.VALUES), column([self.TAIL])], axis=0)
        return column(self.REWARDS), values_arr, column(gae_dones), column(full_dones), column(final_values)

    def test_mid_rollout_truncation_matches_an_explicit_final_obs_bootstrap(self):
        import jax.numpy as jnp

        from environments.shared.jax_ppo import compute_gae
        from environments.shared.jax_trainer import _bootstrap_truncations

        v_final_1 = 3.0  # V(final_obs) of the truncated step 1
        rewards, values, gae_dones, full_dones, final_values = self._arrays(
            jnp,
            gae_dones=[0, 0, 0, 0],
            full_dones=[0, 1, 0, 0],
            # Entries at non-truncated steps must be ignored: make them wild.
            final_values=[99.0, v_final_1, -99.0, self.TAIL],
        )
        rewards_boot, mask = _bootstrap_truncations(rewards, gae_dones, full_dones, final_values, self.GAMMA)
        adv, ret = compute_gae(rewards_boot, values, mask, self.GAMMA, self.LAM)

        expected = _gae_segment(self.REWARDS[:2], self.VALUES[:2], v_final_1, self.GAMMA, self.LAM) + _gae_segment(
            self.REWARDS[2:], self.VALUES[2:], self.TAIL, self.GAMMA, self.LAM
        )
        assert [float(a) for a in adv[:, 0]] == pytest.approx(expected, abs=1e-5)
        # The truncated step's return target is its reward plus the discounted
        # value of the obs it was truncated at — not the post-reset obs.
        assert float(ret[1, 0]) == pytest.approx(self.REWARDS[1] + self.GAMMA * v_final_1, abs=1e-5)

        # The old path (termination-only mask, raw rewards) bootstrapped
        # V(s_2) — the NEXT episode's first state — and carried its
        # advantage back into step 0: pin that it was wrong, not merely
        # different by construction.
        old_adv, _ = compute_gae(rewards, values, gae_dones, self.GAMMA, self.LAM)
        assert float(old_adv[1, 0]) != pytest.approx(expected[1], abs=1e-3)
        assert float(old_adv[0, 0]) != pytest.approx(expected[0], abs=1e-3)

    def test_a_termination_is_unchanged(self):
        import jax.numpy as jnp

        from environments.shared.jax_ppo import compute_gae
        from environments.shared.jax_trainer import _bootstrap_truncations

        rewards, values, gae_dones, full_dones, final_values = self._arrays(
            jnp, gae_dones=[0, 1, 0, 0], full_dones=[0, 1, 0, 0], final_values=[99.0, 99.0, 99.0, self.TAIL]
        )
        rewards_boot, mask = _bootstrap_truncations(rewards, gae_dones, full_dones, final_values, self.GAMMA)
        assert jnp.array_equal(rewards_boot, rewards)
        assert jnp.array_equal(mask, gae_dones)
        new_adv, _ = compute_gae(rewards_boot, values, mask, self.GAMMA, self.LAM)
        old_adv, _ = compute_gae(rewards, values, gae_dones, self.GAMMA, self.LAM)
        assert jnp.allclose(new_adv, old_adv)

    def test_a_truncation_at_the_last_step_is_unchanged(self):
        """The one case the old code already handled — via the tail bootstrap."""
        import jax.numpy as jnp

        from environments.shared.jax_ppo import compute_gae
        from environments.shared.jax_trainer import _bootstrap_truncations

        rewards, values, gae_dones, full_dones, final_values = self._arrays(
            jnp, gae_dones=[0, 0, 0, 0], full_dones=[0, 0, 0, 1], final_values=[99.0, 99.0, 99.0, self.TAIL]
        )
        rewards_boot, mask = _bootstrap_truncations(rewards, gae_dones, full_dones, final_values, self.GAMMA)
        new_adv, _ = compute_gae(rewards_boot, values, mask, self.GAMMA, self.LAM)
        old_adv, _ = compute_gae(rewards, values, gae_dones, self.GAMMA, self.LAM)
        assert jnp.allclose(new_adv, old_adv, atol=1e-5)


def _install_gae_spies(monkeypatch):
    """Record what each trainer feeds ``_bootstrap_truncations`` and ``compute_gae``."""
    from environments.shared import jax_ppo, jax_trainer

    calls: dict = {}
    real_boot = jax_trainer._bootstrap_truncations
    real_gae = jax_ppo.compute_gae

    def spy_boot(rewards, gae_dones, full_dones, final_values, gamma):
        out = real_boot(rewards, gae_dones, full_dones, final_values, gamma)
        calls["boot"] = {
            "rewards": rewards,
            "gae_dones": gae_dones,
            "full_dones": full_dones,
            "final_values": final_values,
            "gamma": gamma,
        }
        return out

    def spy_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
        calls["gae"] = {"rewards": rewards, "values": values, "dones": dones}
        return real_gae(rewards, values, dones, gamma, gae_lambda)

    monkeypatch.setattr(jax_trainer, "_bootstrap_truncations", spy_boot)
    monkeypatch.setattr(jax_ppo, "compute_gae", spy_gae)
    return calls


def _assert_mid_rollout_truncation_was_bootstrapped(calls):
    import jax.numpy as jnp

    boot, gae = calls["boot"], calls["gae"]
    truncated_only = boot["full_dones"] - boot["gae_dones"]
    # max_episode_steps=3 inside a 4-step rollout: every env that did not
    # fall truncates at index 2 — a NON-final position.
    assert float(jnp.sum(truncated_only[:-1])) > 0
    assert jnp.array_equal(gae["dones"], boot["full_dones"]), "GAE carry mask must be the full done"
    expected = boot["rewards"] + boot["gamma"] * boot["final_values"] * truncated_only
    assert jnp.allclose(gae["rewards"], expected, atol=1e-6)
    assert gae["values"].shape[0] == boot["rewards"].shape[0] + 1
    assert jnp.allclose(gae["values"][-1], boot["final_values"][-1])


@pytest.mark.skipif(not _has_jax, reason="JAX/MJX not installed")
class TestTrainersWireTheTruncationBootstrap:
    """Both trainers apply the per-step bootstrap identically (JX5)."""

    def test_jax_trainer_class(self, monkeypatch):
        import environments.trex.mjx_config  # noqa: F401
        from environments.shared.jax_ppo import PPOConfig, make_actor_critic, make_optimizer
        from environments.shared.mjx_env import MJXDinoEnv

        calls = _install_gae_spies(monkeypatch)
        num_envs, rollout_len = 4, 4
        env = MJXDinoEnv("trex", stage=1, num_envs=num_envs, env_kwargs={"max_episode_steps": 3})
        ppo_cfg = PPOConfig(n_epochs=1, n_minibatches=2, target_kl=None, total_updates=1)
        trainer = JaxTrainer(
            env=env,
            network=make_actor_critic(env.action_dim, hidden_dims=(8,)),
            optimizer=make_optimizer(ppo_cfg),
            ppo_config=ppo_cfg,
            num_envs=num_envs,
            rollout_len=rollout_len,
        )
        trainer.train(num_updates=1, seed=0)
        _assert_mid_rollout_truncation_was_bootstrapped(calls)

    def test_notebook_train_function(self, monkeypatch, tmp_path):
        import jax

        import environments.trex.mjx_config  # noqa: F401
        from environments.shared.jax_normalization import RunningMeanStd
        from environments.shared.jax_ppo import PPOConfig, make_actor_critic, make_optimizer
        from environments.shared.jax_trainer import TrainConfig, train
        from environments.shared.mjx_env import MJXDinoEnv

        calls = _install_gae_spies(monkeypatch)
        num_envs, rollout_len = 4, 4
        env = MJXDinoEnv("trex", stage=1, num_envs=num_envs, env_kwargs={"max_episode_steps": 3})
        network = make_actor_critic(env.action_dim, hidden_dims=(8,))
        optimizer = make_optimizer(PPOConfig(n_epochs=1, n_minibatches=2, target_kl=None, total_updates=1))
        rng = jax.random.PRNGKey(0)
        rng, init_rng, reset_rng = jax.random.split(rng, 3)
        states = env.reset(reset_rng)
        obs_dim = int(states.obs.shape[-1])
        params = network.init(init_rng, states.obs[0])
        cfg = TrainConfig(
            num_envs=num_envs,
            rollout_len=rollout_len,
            num_updates=1,
            ppo_epochs=1,
            minibatch_size=8,
            obs_dim=obs_dim,
            act_dim=env.action_dim,
            verbose=0,
            output_dir=tmp_path,
            model_dir=tmp_path,
        )
        train(
            cfg,
            env,
            network,
            params,
            optimizer.init(params),
            RunningMeanStd.create(obs_dim),
            reward_cfg={},
            rng=rng,
            optimizer=optimizer,
        )
        _assert_mid_rollout_truncation_was_bootstrapped(calls)


# ---------------------------------------------------------------------------
# JX8 — the headline episode return must not be poisoned by a NaN window entry
# ---------------------------------------------------------------------------


class TestFiniteEpisodeReturn:
    def test_finite_mean_skips_nan_entries_and_collapses_to_zero_when_none(self):
        from environments.shared.jax_trainer import _finite_mean

        nan = float("nan")
        assert _finite_mean([nan, 5.0, 7.0]) == pytest.approx(6.0)
        assert _finite_mean([nan, nan]) == 0.0
        assert _finite_mean([]) == 0.0

    @pytest.mark.skipif(not _has_jax, reason="JAX/MJX not installed")
    def test_eval_metrics_return_is_finite_when_no_episode_completed(self):
        """One update of a 4-step rollout completes no episode: the window is
        all NaN, which used to make ``mean_episode_return`` NaN (a spurious
        gate failure) while ``mean_episode_length`` beside it was already 0.0."""
        import math

        trainer, _env = TestJaxTrainerSmoke()._build()
        _params, metrics, _state = trainer.train(num_updates=1, seed=0)
        assert math.isfinite(metrics["mean_episode_return"])
        assert metrics["mean_episode_return"] == 0.0
        assert metrics["mean_episode_length"] == 0.0


# ---------------------------------------------------------------------------
# JX2 — same-stage resume through train_jax
# ---------------------------------------------------------------------------


def _adam_count(opt_state) -> int:
    import jax

    leaves, _ = jax.tree_util.tree_flatten_with_path(opt_state)
    counts = [leaf for path, leaf in leaves if any(getattr(key, "name", None) == "count" for key in path)]
    assert counts, "no Adam count leaf in the optimizer state"
    return int(counts[0])


@pytest.mark.skipif(not _has_jax, reason="JAX/MJX not installed")
class TestTrainJaxResume:
    """``train_jax(resume_from=...)`` continues the optimizer and obs stats.

    ``restore_train_state`` could restore params/opt_state/obs_rms but had no
    non-test caller, and ``train_jax`` exposed no ``init_opt_state`` although
    ``JaxTrainer.train`` supports it — so a same-stage resume re-initialised
    Adam and (via the curriculum key) decayed the obs stats.
    """

    @staticmethod
    def _run(checkpoint_dir, **extra):
        from environments.shared.jax_training import train_jax

        return train_jax(
            "trex",
            stage=1,
            num_envs=2,
            rollout_len=2,
            num_updates=1,
            seed=0,
            checkpoint_dir=str(checkpoint_dir),
            policy_kwargs={"net_arch": [8]},
            n_epochs=1,
            minibatch_size=2,
            target_kl=None,
            **extra,
        )

    def test_resume_continues_the_optimizer_and_leaves_the_obs_count_undecayed(self, tmp_path):
        from environments.shared.jax_checkpoint import restore_train_state
        from environments.shared.plant_contract import current_plant_identity

        plant = current_plant_identity("trex")
        self._run(tmp_path / "first")
        ckpt = tmp_path / "first" / "trex_s1_0.pkl"
        assert ckpt.exists()
        _, opt_before, rms_before, _ = restore_train_state(ckpt, current_plant=plant)
        # 1 epoch x 2 minibatches = 2 Adam steps; 2 envs x 2 steps = 4 obs.
        assert _adam_count(opt_before) == 2
        assert float(rms_before.count) == pytest.approx(4.0001)

        self._run(tmp_path / "second", resume_from=str(ckpt))
        _, opt_after, rms_after, _ = restore_train_state(tmp_path / "second" / "trex_s1_0.pkl", current_plant=plant)
        # Continued, not re-initialised: 2 more Adam steps on top of the 2.
        assert _adam_count(opt_after) == 4
        # Undecayed (effective obs_rms_decay_on_resume = 1.0): 4 more obs on
        # top of the restored count.  A 0.01 decay would read ~4.04 here.
        assert float(rms_after.count) == pytest.approx(float(rms_before.count) + 4.0)

    def test_decay_is_opt_in_and_resume_excludes_the_cross_stage_init_args(self, tmp_path, monkeypatch):
        from types import SimpleNamespace

        from environments.shared import jax_trainer

        self._run(tmp_path / "first")
        ckpt = tmp_path / "first" / "trex_s1_0.pkl"
        seen: dict = {}

        def fake_train(self, **kwargs):
            seen.update(kwargs)
            return ({}, {}, SimpleNamespace(obs_stats=kwargs["init_obs_stats"]))

        monkeypatch.setattr(jax_trainer.JaxTrainer, "train", fake_train)
        self._run(tmp_path / "second", resume_from=str(ckpt), obs_rms_decay_on_resume=0.01)
        assert seen["init_params"] is not None
        assert _adam_count(seen["init_opt_state"]) == 2
        assert float(seen["init_obs_stats"].count) == pytest.approx(4.0001 * 0.01)

        with pytest.raises(ValueError, match="resume_from"):
            self._run(tmp_path / "third", resume_from=str(ckpt), init_params={"w": 1.0})


# ---------------------------------------------------------------------------
# JX3d — semantic stage ids on the JAX path
# ---------------------------------------------------------------------------


class _FakeEvalSummary:
    rewards = [1.0, 2.0]
    mean_reward = std_reward = mean_length = std_length = 0.0
    mean_forward_vel = mean_distance = mean_tilt = mean_height = 0.0
    mean_success_rate = 0.5


class TestSemanticStageRefs:
    def test_print_eval_summary_accepts_semantic_ids(self, capsys):
        from environments.shared.jax_setup import print_eval_summary

        print_eval_summary(_FakeEvalSummary(), True, [], "recovery")
        assert "Success rate" not in capsys.readouterr().out
        for behavior in (3, "behavior"):
            print_eval_summary(_FakeEvalSummary(), True, [], behavior)
            assert "Success rate" in capsys.readouterr().out
        print_eval_summary(_FakeEvalSummary(), True, [], 2)
        assert "Success rate" not in capsys.readouterr().out


@pytest.mark.skipif(not _has_jax, reason="JAX/MJX not installed")
class TestSemanticStageOnJaxPath:
    """``stage="recovery"`` flows setup_species -> make_reward_fns; ints keep meaning."""

    def test_recovery_builds_reward_fns_without_error(self):
        from environments.shared.jax_setup import make_reward_fns, setup_species

        ctx = setup_species("trex", stage="recovery")
        assert ctx.stage == "recovery"
        assert ctx.stage_entry.id == "recovery" and ctx.stage_entry.legacy_number is None
        assert ctx.stage_name == "Recovery"
        assert ctx.is_behavior_stage is False
        assert ctx.success_sites == ()
        compute_reward, compute_reward_detailed, is_terminated = make_reward_fns(ctx)
        assert all(callable(fn) for fn in (compute_reward, compute_reward_detailed, is_terminated))

    def test_legacy_ints_resolve_to_the_same_stages_as_before(self):
        from environments.shared.jax_setup import setup_species

        stance, locomotion, behavior = (setup_species("trex", stage=n) for n in (1, 2, 3))
        assert (stance.stage_name, locomotion.stage_name, behavior.stage_name) == ("Balance", "Locomotion", "Bite")
        assert (stance.is_behavior_stage, locomotion.is_behavior_stage, behavior.is_behavior_stage) == (
            False,
            False,
            True,
        )
        # trex locomotion is manifest position 3 but legacy number 2 — an
        # integer keeps its legacy meaning, never the position.
        assert locomotion.stage_entry.legacy_number == 2 and locomotion.stage_entry.position == 3

    def test_success_bonus_engages_only_on_the_behavior_stage(self, monkeypatch):
        from environments.shared import jax_reward_termination
        from environments.shared.jax_setup import make_reward_fns, setup_species

        seen: dict = {}

        def spy(data, action, reward_cfg, **kwargs):
            seen["success_bonus"] = kwargs["success_bonus"]
            return 0.0

        monkeypatch.setattr(jax_reward_termination, "compute_total_reward", spy)
        for stage, engaged in ((3, True), ("behavior", True), ("recovery", False), (2, False)):
            compute_reward, _, _ = make_reward_fns(setup_species("trex", stage=stage))
            compute_reward(None, None, {})
            assert (seen["success_bonus"] > 0.0) is engaged, stage

    def test_train_jax_rejects_an_unknown_stage_before_building_the_env(self):
        from environments.shared.jax_training import train_jax
        from environments.shared.stage_manifest import StageManifestError

        with pytest.raises(StageManifestError, match="nonsense"):
            train_jax("trex", stage="nonsense", num_envs=2, rollout_len=2, num_updates=1)


# ---------------------------------------------------------------------------
# JX9 — [jax.policy_kwargs] net_arch sizes the network on both paths
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_jax, reason="JAX/MJX not installed")
class TestNetArchReachesTheNetwork:
    def test_training_and_load_paths_build_the_toml_widths(self, monkeypatch):
        from types import SimpleNamespace

        import jax
        import jax.numpy as jnp

        from environments.shared import jax_trainer
        from environments.shared.jax_setup import make_network, setup_species
        from environments.shared.jax_training import train_jax

        seen: dict = {}

        def fake_train(self, **kwargs):
            seen["network"] = self.network
            return ({}, {}, SimpleNamespace(obs_stats=None))

        monkeypatch.setattr(jax_trainer.JaxTrainer, "train", fake_train)
        train_jax("trex", stage=1, num_envs=2, rollout_len=2, num_updates=1, policy_kwargs={"net_arch": [256, 128]})
        assert tuple(seen["network"].hidden_dims) == (256, 128)

        # The load/eval path rebuilds the module from the same stage table.
        ctx = setup_species("trex", stage=1)
        ctx.jax_kwargs = {**ctx.jax_kwargs, "policy_kwargs": {"net_arch": [256, 128]}}
        loaded = make_network(ctx)
        assert tuple(loaded.hidden_dims) == (256, 128)
        obs = jnp.zeros((ctx.obs_dim,))
        shapes = lambda net: jax.tree.map(lambda a: a.shape, net.init(jax.random.PRNGKey(0), obs))  # noqa: E731
        assert shapes(loaded) == shapes(seen["network"]), "params from training must fit the eval network"

    def test_default_widths_when_the_table_is_absent(self):
        from environments.shared.jax_setup import make_network, setup_species

        ctx = setup_species("trex", stage=1)
        ctx.jax_kwargs = {k: v for k, v in ctx.jax_kwargs.items() if k != "policy_kwargs"}
        assert tuple(make_network(ctx).hidden_dims) == (512, 256)


# ---------------------------------------------------------------------------
# jax_setup eval wiring pins (eval agent's contract)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_jax, reason="JAX/MJX not installed")
class TestStageEvaluationWiring:
    _PERTURBATION_FIELDS = (
        "perturbation_capture_velocity_multiple",
        "perturbation_interval",
        "perturbation_jitter",
        "perturbation_duration",
        "perturbation_direction",
    )

    def test_eval_config_receives_the_env_perturbation_fields(self, monkeypatch):
        from types import SimpleNamespace

        from environments.shared import jax_eval
        from environments.shared.jax_setup import create_env, run_stage_evaluation, setup_species

        ctx = setup_species("trex", stage="recovery")
        env = create_env(ctx, num_envs=2)
        assert env.config.perturbation_capture_velocity_multiple > 0.0, "recovery must be a pushed stage"

        seen: dict = {}

        def fake_eval(mj_model, params, network, obs_rms, *, config, **kwargs):
            seen["config"] = config
            return SimpleNamespace(
                mean_reward=0.0,
                std_reward=0.0,
                mean_length=0.0,
                std_length=0.0,
                mean_forward_vel=0.0,
                forward_vels=[],
                mean_distance=0.0,
                mean_success_rate=0.0,
                rewards=[],
            )

        monkeypatch.setattr(jax_eval, "evaluate_policy_cpu", fake_eval)
        monkeypatch.setattr(jax_eval, "check_stage_gate_for_config", lambda results, cfg: (True, []))
        run_stage_evaluation(ctx, env, params=None, network=None, obs_rms=None, n_episodes=1)

        for name in self._PERTURBATION_FIELDS:
            assert getattr(seen["config"], name) == getattr(env.config, name), name

    def test_detailed_reward_forwards_both_action_lags(self, monkeypatch):
        from environments.shared import jax_reward_termination
        from environments.shared.jax_setup import make_reward_fns, setup_species

        seen: dict = {}

        def spy(data, action, reward_cfg, **kwargs):
            seen.update(kwargs)
            return {}

        monkeypatch.setattr(jax_reward_termination, "compute_reward_components", spy)
        _, compute_reward_detailed, _ = make_reward_fns(setup_species("trex", stage=1))
        compute_reward_detailed(
            None, None, {}, prev_action="lag1", prev_prev_action="lag2", target_pos="not-a-component"
        )
        assert seen["prev_action"] == "lag1"
        assert seen["prev_prev_action"] == "lag2"
        assert "target_pos" not in seen
