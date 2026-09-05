"""Tests for jax_trainer, jax_train_fn and jax_hooks modules."""

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

    def test_append_keeps_the_prior_session_rows(self, tmp_path):
        """A same-stage resume into the interrupted session's directory must
        not truncate its training log to the resumed rows."""
        csv_path = tmp_path / "log.csv"
        state = TrainerState(params=None, opt_state=None, obs_stats=None, env_states=None, rng=None)
        first = CSVLoggingHook(path=csv_path)
        for update in (0, 1):
            first.on_update_end(state, {"update": update, "mean_reward": 1.0})
        first.on_train_end(state)

        resumed = CSVLoggingHook(path=csv_path, append=True)
        resumed.on_update_end(state, {"update": 2, "mean_reward": 1.0})
        resumed.on_train_end(state)

        lines = csv_path.read_text().splitlines()
        assert sum(line.startswith("update,") for line in lines) == 1
        assert [line.split(",")[0] for line in lines[1:]] == ["0", "1", "2"]


class TestCheckpointHook:
    """The final save must not double-count the last periodic save, and a run
    that completed no update must not checkpoint its untouched params."""

    @staticmethod
    def _replay(directory, num_updates, interval, max_keep, start_update=0):
        from environments.shared.jax_hooks import CheckpointHook

        hook = CheckpointHook(directory=str(directory), prefix="ck", interval=interval, max_keep=max_keep)
        state = TrainerState(params={"w": 1.0}, opt_state=None, obs_stats=None, env_states=None, rng=None)
        # Exactly JaxTrainer.train's dispatch: state.update = index per update,
        # then on_train_end with the last index still set.
        for update in range(start_update, start_update + num_updates):
            state.update = update
            hook.on_update_end(state, {"update": update})
        hook.on_train_end(state)
        return hook, sorted(int(p.stem[len("ck_") :]) for p in directory.glob("ck_*.pkl"))

    def test_a_budget_aligned_with_the_interval_keeps_max_keep_distinct_files(self, tmp_path):
        """train_jax's defaults (500 updates, interval 50, max_keep 5) ended
        with ck_500 saved twice and ck_300 evicted for the duplicate."""
        hook, files = self._replay(tmp_path, 500, 50, 5)
        assert files == [300, 350, 400, 450, 500]
        assert [p.name for p in hook.manager._recent] == [f"ck_{u}.pkl" for u in files]
        assert hook.manager.latest == tmp_path / "ck_500.pkl"

    def test_max_keep_one_never_unlinks_the_final_checkpoint(self, tmp_path):
        hook, files = self._replay(tmp_path, 100, 50, 1)
        assert files == [100]
        assert hook.manager.latest is not None and hook.manager.latest.exists()

    def test_an_unaligned_budget_and_a_resume_still_get_their_final_save(self, tmp_path):
        _, files = self._replay(tmp_path / "fresh", 120, 50, 5)
        assert files == [50, 100, 120]
        _, files = self._replay(tmp_path / "resumed", 200, 50, 5, start_update=300)
        assert files == [350, 400, 450, 500]

    def test_no_completed_update_means_no_checkpoint(self, tmp_path):
        """A zero-update run saved the restored params as ck_1.pkl (the
        TrainerState default update 0 + 1), evicting a genuine checkpoint."""
        import pickle

        for u in (300, 350, 400, 450, 500):
            with open(tmp_path / f"ck_{u}.pkl", "wb") as f:
                pickle.dump({"params": {}, "update": u}, f)
        _, files = self._replay(tmp_path, 0, 50, 5)
        assert files == [300, 350, 400, 450, 500]


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
    from environments.shared import jax_ppo, jax_train_fn, jax_trainer

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

    # Each loop resolves ``_bootstrap_truncations`` in its own module's globals:
    # JaxTrainer in jax_trainer, the notebook train() in jax_train_fn.
    monkeypatch.setattr(jax_trainer, "_bootstrap_truncations", spy_boot)
    monkeypatch.setattr(jax_train_fn, "_bootstrap_truncations", spy_boot)
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
    def test_eval_metrics_return_skips_nan_window_entries_and_collapses_to_zero_when_none(self):
        """``mean_episode_return`` over the trainer's 10-update window: a NaN
        entry (an update with no completed episode) must not poison it, and
        an all-NaN window reads 0.0 like the length beside it."""
        from environments.shared.jax_trainer import _build_eval_metrics

        nan = float("nan")
        state = TrainerState(params=None, opt_state=None, obs_stats=None, env_states=None, rng=None, total_steps=8)
        state.history = [{"mean_reward": 1.0}, {"mean_reward": 3.0}]
        state.episode_return_history = [nan, 5.0, 7.0]
        state.episode_length_history = [nan, nan, nan]

        metrics = _build_eval_metrics(state, elapsed=2.0)
        assert metrics["mean_episode_return"] == pytest.approx(6.0)
        assert metrics["mean_episode_length"] == 0.0
        assert metrics["mean_reward"] == pytest.approx(2.0)
        assert metrics["total_steps"] == 8 and metrics["elapsed"] == 2.0

        state.episode_return_history = [nan, nan]
        assert _build_eval_metrics(state, elapsed=0.0)["mean_episode_return"] == 0.0

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
            num_updates=extra.pop("num_updates", 1),
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
        # Checkpoints are numbered by updates COMPLETED (1 after a 1-update run).
        ckpt = tmp_path / "first" / "trex_s1_1.pkl"
        assert ckpt.exists()
        assert not (tmp_path / "first" / "trex_s1_0.pkl").exists()
        _, opt_before, rms_before, resumed = restore_train_state(ckpt, current_plant=plant)
        assert resumed == 1
        # 1 epoch x 2 minibatches = 2 Adam steps; 2 envs x 2 steps = 4 obs.
        assert _adam_count(opt_before) == 2
        assert float(rms_before.count) == pytest.approx(4.0001)

        # A 2-update budget with 1 done: exactly one more update runs, and its
        # checkpoint continues the numbering instead of overwriting _1.
        self._run(tmp_path / "second", resume_from=str(ckpt), num_updates=2)
        second = tmp_path / "second" / "trex_s1_2.pkl"
        assert second.exists()
        assert not (tmp_path / "second" / "trex_s1_1.pkl").exists()
        _, opt_after, rms_after, resumed_after = restore_train_state(second, current_plant=plant)
        assert resumed_after == 2
        # Continued, not re-initialised: 2 more Adam steps on top of the 2.
        assert _adam_count(opt_after) == 4
        # Undecayed (effective obs_rms_decay_on_resume = 1.0): 4 more obs on
        # top of the restored count.  A 0.01 decay would read ~4.04 here.
        assert float(rms_after.count) == pytest.approx(float(rms_before.count) + 4.0)

    def test_decay_is_opt_in_and_resume_excludes_the_cross_stage_init_args(self, tmp_path, monkeypatch):
        from types import SimpleNamespace

        from environments.shared import jax_trainer

        self._run(tmp_path / "first")
        ckpt = tmp_path / "first" / "trex_s1_1.pkl"
        seen: dict = {}

        def fake_train(self, **kwargs):
            seen.update(kwargs)
            return ({}, {}, SimpleNamespace(obs_stats=kwargs["init_obs_stats"]))

        monkeypatch.setattr(jax_trainer.JaxTrainer, "train", fake_train)
        self._run(tmp_path / "second", resume_from=str(ckpt), obs_rms_decay_on_resume=0.01, num_updates=3)
        # The remaining budget, from where the checkpoint left off.
        assert seen["start_update"] == 1
        assert seen["num_updates"] == 2
        assert seen["init_params"] is not None
        assert _adam_count(seen["init_opt_state"]) == 2
        assert float(seen["init_obs_stats"].count) == pytest.approx(4.0001 * 0.01)

        with pytest.raises(ValueError, match="resume_from"):
            self._run(tmp_path / "third", resume_from=str(ckpt), init_params={"w": 1.0})

    def test_same_dir_resume_appends_the_csv_and_a_resume_at_the_budget_writes_nothing(self, tmp_path):
        import csv
        import math

        run_dir = tmp_path / "run"
        self._run(run_dir, num_updates=2)
        csv_path = run_dir / "trex_s1_training_log.csv"
        assert [r["update"] for r in csv.DictReader(csv_path.open())] == ["0", "1"]

        # Into the interrupted session's own directory: its rows survive and
        # the resumed rows continue the update numbering (the hook used to
        # open the log in mode "w").
        self._run(run_dir, resume_from=str(run_dir / "trex_s1_2.pkl"), num_updates=3)
        lines = csv_path.read_text().splitlines()
        assert sum(line.startswith("update,") for line in lines) == 1
        assert [r["update"] for r in csv.DictReader(csv_path.open())] == ["0", "1", "2"]
        assert sorted(p.name for p in run_dir.glob("*.pkl")) == ["trex_s1_2.pkl", "trex_s1_3.pkl"]

        # Already at the budget: nothing to train, so nothing is written --
        # a zero-update trainer run saved the restored params as trex_s1_1.pkl
        # (evicting a genuine checkpoint) and truncated the CSV to its header.
        before = {p.name: p.read_bytes() for p in run_dir.iterdir()}
        params, metrics, obs_stats = self._run(run_dir, resume_from=str(run_dir / "trex_s1_3.pkl"), num_updates=3)
        assert metrics["already_complete"] is True
        assert all(math.isfinite(v) for k, v in metrics.items() if k != "already_complete")
        assert {p.name: p.read_bytes() for p in run_dir.iterdir()} == before
        assert params is not None and obs_stats is not None

    def test_a_checkpoint_without_optimizer_state_is_refused_before_anything_is_written(self, tmp_path):
        """<prefix>_best.pkl carries no opt_state; resuming from it restarted
        Adam and left the LR schedule short of learning_rate_end."""
        from environments.shared.jax_checkpoint import save_checkpoint
        from environments.shared.plant_contract import current_plant_identity

        best = tmp_path / "trex_s1_best.pkl"
        save_checkpoint(best, params={"w": 1.0}, update=2, plant_identity=current_plant_identity("trex"))
        with pytest.raises(ValueError, match="no optimizer state"):
            self._run(tmp_path / "out", resume_from=str(best), num_updates=3)
        assert not (tmp_path / "out").exists()

    def test_the_best_file_records_a_completed_count_and_only_a_better_policy_replaces_it(self, tmp_path, monkeypatch):
        """Drives the real hooks with scripted episode returns (JaxTrainer.train
        faked, as above): the best file's ``update`` is a completed COUNT like
        the periodic checkpoints, and a same-directory resume seeds the
        tracker from it instead of overwriting it with the continuation's best."""
        from types import SimpleNamespace

        from environments.shared import jax_trainer
        from environments.shared.jax_checkpoint import load_checkpoint
        from environments.shared.plant_contract import current_plant_identity

        returns: dict[int, float] = {}

        def fake_train(self, *, num_updates, seed, init_params, init_opt_state, init_obs_stats, start_update):
            params, state = init_params, None
            for update in range(start_update, start_update + num_updates):
                params = {"w": float(update)}
                state = SimpleNamespace(params=params, opt_state="opt", obs_stats=None, update=update, total_steps=0)
                metrics = {"update": update, "episode_return": returns.get(update, -1.0), "mean_reward": 0.0}
                for hook in self.hooks:
                    if hasattr(hook, "on_update_end"):
                        hook.on_update_end(state, metrics)
            for hook in self.hooks:
                if hasattr(hook, "on_train_end"):
                    hook.on_train_end(state)
            return params, {"mean_episode_return": 0.0}, state

        monkeypatch.setattr(jax_trainer.JaxTrainer, "train", fake_train)
        plant = current_plant_identity("trex")
        run_dir = tmp_path / "run"
        best = run_dir / "trex_s1_best.pkl"

        returns.update({1: 2500.0})
        _, metrics, _ = self._run(run_dir, num_updates=3)
        ckpt = load_checkpoint(best, current_plant=plant)
        # Best at index 1 -> 2 updates completed, as trex_s1_3.pkl records 3.
        assert (ckpt["update"], ckpt["params"], ckpt["best_episode_return"]) == (2, {"w": 1.0}, 2500.0)
        assert metrics["best_episode_return"] == 2500.0

        # A weaker continuation leaves the file alone and still reports the best.
        returns.clear()
        returns.update({4: 1900.0})
        stamp = best.read_bytes()
        _, metrics, _ = self._run(run_dir, resume_from=str(run_dir / "trex_s1_3.pkl"), num_updates=5)
        assert best.read_bytes() == stamp
        assert metrics["best_episode_return"] == 2500.0

        # A genuinely better policy replaces it.
        returns.clear()
        returns.update({6: 9000.0})
        _, metrics, _ = self._run(run_dir, resume_from=str(run_dir / "trex_s1_5.pkl"), num_updates=7)
        ckpt = load_checkpoint(best, current_plant=plant)
        assert (ckpt["update"], ckpt["params"], ckpt["best_episode_return"]) == (7, {"w": 6.0}, 9000.0)
        assert metrics["best_episode_return"] == 9000.0


class TestResumeFromCli:
    def test_curriculum_and_resume_from_are_refused_together(self, capsys):
        """The curriculum branch never forwarded --resume-from, so the run
        silently restarted stage 1 from scratch in the same directory."""
        from environments.shared.jax_training import _parse_args

        assert _parse_args(["--species", "trex", "--stage", "1", "--resume-from", "ckpt.pkl"]).resume_from == "ckpt.pkl"
        with pytest.raises(SystemExit) as exc:
            _parse_args(["--species", "trex", "--curriculum", "--resume-from", "ckpt.pkl"])
        assert exc.value.code == 2
        assert "--resume-from" in capsys.readouterr().err


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


# ---------------------------------------------------------------------------
# The training-time reward-component panel scores the step as the kernel did
# ---------------------------------------------------------------------------


class TestRewardComponentPanelLags:
    def test_lag_keywords_are_detected(self):
        from environments.shared.jax_train_fn import _detail_fn_accepts_action_lags

        assert not _detail_fn_accepts_action_lags(lambda data, action: {})
        assert _detail_fn_accepts_action_lags(lambda data, action, **step_kwargs: {})
        assert _detail_fn_accepts_action_lags(lambda data, action, prev_action=None, prev_prev_action=None: {})

    @pytest.mark.skipif(not _has_jax, reason="JAX/MJX not installed")
    def test_the_panel_receives_the_command_and_the_lags_the_kernel_scored(self, caplog):
        """The panel used to vmap ``reward_detail_fn(states.data, act_t[-1])``
        only, so the notebook's action_jerk row was charged against zero
        lags.  The rollout now keeps the last step's pre-step carries and the
        panel scores the command the kernel scored against them."""
        import logging

        import jax
        import jax.numpy as jnp
        import optax

        import environments.trex.mjx_config  # noqa: F401
        from environments.shared.jax_normalization import RunningMeanStd
        from environments.shared.jax_ppo import make_actor_critic
        from environments.shared.jax_train_fn import TrainConfig, _build_jit_fns
        from environments.shared.mjx_env import MJXDinoEnv

        num_envs, rollout_len = 2, 2
        env = MJXDinoEnv("trex", stage=1, num_envs=num_envs)
        rng = jax.random.PRNGKey(0)
        rng, init_rng, reset_rng = jax.random.split(rng, 3)
        states = env.reset(reset_rng)
        obs_dim = int(states.obs.shape[-1])
        network = make_actor_critic(env.action_dim, hidden_dims=(8,))
        params = network.init(init_rng, states.obs[0])
        cfg = TrainConfig(num_envs=num_envs, rollout_len=rollout_len, obs_dim=obs_dim, act_dim=env.action_dim)

        def spy(data, action, **step_kwargs):
            return {"action": action, **step_kwargs}

        fns = _build_jit_fns(cfg, network, optax.adam(1e-3), spy, env=env)
        states, _rollout, last_lags = fns["collect_rollout"](
            states, rng, params, RunningMeanStd.create(obs_dim), jnp.float32(1.0)
        )
        # Two steps from reset: the kernel scored the last step against
        # (a_0, 0) and its post-step carries are (a_1, a_0).
        assert bool(jnp.any(states.prev_action != 0.0))
        assert jnp.array_equal(last_lags[0], states.prev_prev_action)
        assert jnp.array_equal(last_lags[1], jnp.zeros_like(states.prev_action))

        seen = fns["batched_reward_components"](states, states.prev_action, *last_lags)
        assert set(seen) == {"action", "prev_action", "prev_prev_action"}
        assert jnp.array_equal(seen["action"], states.prev_action)
        assert jnp.array_equal(seen["prev_action"], last_lags[0])
        assert jnp.array_equal(seen["prev_prev_action"], last_lags[1])

        # A detail fn without the keywords still works, on the zero-lag path, loudly.
        with caplog.at_level(logging.WARNING, logger="environments.shared.jax_train_fn"):
            legacy = _build_jit_fns(cfg, network, optax.adam(1e-3), lambda data, action: {"a": action}, env=env)
        assert "zero lags" in caplog.text
        out = legacy["batched_reward_components"](states, states.prev_action, *last_lags)
        assert jnp.array_equal(out["a"], states.prev_action)


# ---------------------------------------------------------------------------
# The notebook's functional train(): a same-stage resume is a continuation
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_jax, reason="JAX/MJX not installed")
class TestNotebookTrainResumeContinues:
    def test_a_resume_past_warmup_and_ramp_keeps_the_stage_schedule(self, monkeypatch, tmp_path):
        """start_update=400 of a 402-update stage with warmup 40 / ramp 200:
        the first resumed update must use the post-warmup clip/ent and the
        full ramp (both restarted, keyed off the relative update), log the LR
        at its absolute position on the stage schedule (it snapped back to
        learning_rate), and record whole-stage steps (it recorded the
        continuation only)."""
        import csv

        import jax

        import environments.trex.mjx_config  # noqa: F401
        from environments.shared import jax_train_fn
        from environments.shared.jax_normalization import RunningMeanStd
        from environments.shared.jax_ppo import PPOConfig, make_actor_critic, make_optimizer
        from environments.shared.jax_trainer import TrainConfig, train
        from environments.shared.mjx_env import MJXDinoEnv

        num_envs, rollout_len, start, budget = 4, 4, 400, 402
        lr, lr_end = 3e-4, 1e-5
        env = MJXDinoEnv("trex", stage=1, num_envs=num_envs)
        network = make_actor_critic(env.action_dim, hidden_dims=(8,))
        optimizer = make_optimizer(
            PPOConfig(learning_rate=lr, learning_rate_end=lr_end, n_epochs=1, n_minibatches=2, target_kl=None)
        )
        rng = jax.random.PRNGKey(0)
        rng, init_rng, reset_rng = jax.random.split(rng, 3)
        states = env.reset(reset_rng)
        obs_dim = int(states.obs.shape[-1])
        params = network.init(init_rng, states.obs[0])

        seen: dict = {"ppo": [], "scale": [], "weight": []}
        real_build = jax_train_fn._build_jit_fns

        def spy_build(*args, **kwargs):
            fns = real_build(*args, **kwargs)
            real_ppo, real_collect = fns["scan_ppo_epochs"], fns["collect_rollout"]

            def ppo(params, opt_state, *rest):
                seen["ppo"].extend((float(rest[-2]), float(rest[-1])))
                return real_ppo(params, opt_state, *rest)

            def collect(states, rng, params, obs_rms, scale):
                seen["scale"].append(float(scale))
                return real_collect(states, rng, params, obs_rms, scale)

            fns["scan_ppo_epochs"], fns["collect_rollout"] = ppo, collect
            return fns

        monkeypatch.setattr(jax_train_fn, "_build_jit_fns", spy_build)

        reward_cfg = {"forward_vel_weight": 1.0}
        cfg = TrainConfig(
            num_envs=num_envs,
            rollout_len=rollout_len,
            num_updates=budget - start,
            ppo_epochs=1,
            minibatch_size=8,
            learning_rate=lr,
            learning_rate_end=lr_end,
            clip_range=0.2,
            ent_coef=0.01,
            obs_dim=obs_dim,
            act_dim=env.action_dim,
            warmup_updates=40,
            warmup_clip_range=0.02,
            warmup_ent_coef=0.02,
            ramp_updates=200,
            ramp_start_fraction=0.2,
            verbose=0,
            output_dir=tmp_path,
            model_dir=tmp_path,
            checkpoint_freq=1000,
            start_update=start,
        )
        result = train(
            cfg,
            env,
            network,
            params,
            optimizer.init(params),
            RunningMeanStd.create(obs_dim),
            reward_cfg=reward_cfg,
            rng=rng,
            optimizer=optimizer,
            callback=lambda update, metrics: seen["weight"].append(reward_cfg["forward_vel_weight"]),
        )

        assert seen["ppo"] == pytest.approx([0.2, 0.01, 0.2, 0.01])
        assert seen["scale"] == [1.0, 1.0]
        assert seen["weight"] == [1.0, 1.0]

        expected_lr = [lr + (lr_end - lr) * update / budget for update in (400, 401)]
        assert [d["learning_rate"] for d in result.diagnostics_history] == pytest.approx(expected_lr)

        assert result.actual_updates == 2
        assert result.session_steps == 2 * rollout_len * num_envs
        assert result.total_steps == budget * rollout_len * num_envs
        rows = list(csv.DictReader((tmp_path / "training_log.csv").open()))
        assert [int(r["steps"]) for r in rows] == [401 * rollout_len * num_envs, 402 * rollout_len * num_envs]
        assert [float(r["learning_rate"]) for r in rows] == pytest.approx(expected_lr, rel=1e-2)
