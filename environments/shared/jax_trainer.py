"""PPO training loop for JAX/MJX environments.

Two APIs are provided:

1. **Notebook API** (``train()`` function): direct, config-driven training
   with built-in logging, checkpointing, warmup/ramp, and minibatch
   shuffling.  Used by ``jax_training.ipynb``.  Defined in
   :mod:`environments.shared.jax_train_fn` and re-exported here
   (``TrainConfig``, ``TrainResult``, ``_build_jit_fns``, ``train``).

2. **Library API** (``JaxTrainer`` class): hook-based training loop used by
   ``jax_training.py`` (CLI) and ``jax_hooks.py``.  Extensible via
   :class:`TrainingHook` protocol.  Defined in this module.

The helpers both loops share (:func:`_bootstrap_truncations`,
:func:`_build_batched_value`) live in
:mod:`environments.shared.jax_trainer_types`.

Usage (notebook)::

    from environments.shared.jax_trainer import TrainConfig, TrainResult, train
    result = train(config, env, network, params, obs_rms, reward_cfg)

Usage (library)::

    from environments.shared.jax_trainer import JaxTrainer, TrainerState
    trainer = JaxTrainer(env=env, network=network, optimizer=opt, ppo_config=cfg)
    params, metrics, state = trainer.train(num_updates=500)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from .jax_train_fn import TrainConfig, TrainResult, _build_jit_fns, train
from .jax_trainer_types import (
    StopTraining,
    TrainerState,
    TrainingHook,
    _bootstrap_truncations,
    _build_batched_value,
)
from .mjx_utils import check_jax

_logger = logging.getLogger(__name__)

# Re-export surface: the notebook-path names from ``jax_train_fn`` and the
# hook-API types from ``jax_trainer_types`` stay importable from here, so
# ``from environments.shared.jax_trainer import TrainConfig, train`` (the
# notebook) and ``... import TrainerState`` (hooks) keep working.
__all__ = [
    "JaxTrainer",
    "StopTraining",
    "TrainConfig",
    "TrainResult",
    "TrainerState",
    "TrainingHook",
    "train",
    # Private, but part of the notebook path's surface (imported by its tests).
    "_build_jit_fns",
]


# ---------------------------------------------------------------------------
# Eval metrics
# ---------------------------------------------------------------------------


def _finite_mean(values: "list[float]") -> float:
    """Mean of the finite entries, or ``0.0`` when there are none.

    Episode-length windows carry ``nan`` for updates where no episode
    completed.  A gate threshold must not be handed a ``nan`` -- every
    comparison against it is ``False``, which reads as "gate failed" for the
    length check but would be indistinguishable from a real short-episode
    failure.  Collapsing to ``0.0`` keeps the fail-closed reading explicit.
    """
    finite = [v for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else 0.0


def _build_eval_metrics(state: TrainerState, elapsed: float) -> dict[str, float]:
    """The ``eval_metrics`` :meth:`JaxTrainer.train` returns for *state*.

    Windows are the last 10 updates.  Episode-level entries are NaN for
    updates with no completed episode, and one NaN poisoned the whole
    ``mean_episode_return`` (a spurious gate failure and a NaN headline
    metric); both episode reductions use the same finite mean, collapsing
    an all-NaN window to ``0.0`` -- the fail-closed reading for a gate.
    """
    return {
        "mean_reward": float(np.mean([h["mean_reward"] for h in state.history[-10:]])) if state.history else 0.0,
        # Episode-level return (comparable to SB3's mean episode reward
        # and the TOML min_avg_reward gates); mean_reward above is the
        # mean PER-STEP rollout reward, orders of magnitude smaller.
        "mean_episode_return": _finite_mean(state.episode_return_history[-10:]),
        # Episode length over the same window, so check_stage_gate can
        # enforce min_avg_episode_length.
        "mean_episode_length": _finite_mean(state.episode_length_history[-10:]),
        "total_steps": state.total_steps,
        "elapsed": elapsed,
        "t_rollout_cumulative": state.t_rollout_cumulative,
        "t_ppo_cumulative": state.t_ppo_cumulative,
    }


# ===========================================================================
# Library API — JaxTrainer class with hook-based architecture
# ===========================================================================
# Used by jax_training.py (CLI), jax_hooks.py, and tests.
#
# ``TrainerState``, ``TrainingHook``, and ``StopTraining`` are defined in
# :mod:`environments.shared.jax_trainer_types` and re-exported at the top
# of this module for backward compatibility.  The notebook-path ``train()``
# (with ``TrainConfig``, ``TrainResult`` and ``_build_jit_fns``) is defined
# in :mod:`environments.shared.jax_train_fn` and re-exported the same way.


# ---------------------------------------------------------------------------
# Trainer class
# ---------------------------------------------------------------------------


class JaxTrainer:
    """Reusable JIT-compiled JAX/PPO training loop.

    Owns the rollout collection and PPO update as JIT-compiled functions.
    Dispatches lifecycle events to pluggable :class:`TrainingHook`
    instances for logging, checkpointing, monitoring, etc.

    Args:
        env: ``MJXDinoEnv`` instance (functional batched environment).
        network: Flax ``ActorCritic`` module.
        optimizer: Optax ``GradientTransformation``.
        ppo_config: PPO hyperparameters.
        num_envs: Number of parallel environments.
        rollout_len: Steps per rollout.
        hooks: Optional list of ``TrainingHook`` instances.
        warmup_updates: Constrain policy updates for the first N updates of
            a curriculum stage (smaller clip range, higher entropy) while
            the critic adapts to the new reward landscape.  0 disables.
        warmup_clip_range: Clip range during warm-up.
        warmup_ent_coef: Entropy coefficient during warm-up.
        ramp_updates: Linearly ramp the forward-velocity reward scale from
            ``ramp_start_fraction`` to 1.0 over the first N updates
            (TOML ``ramp_updates``).  0 disables.
        ramp_start_fraction: Starting fraction of forward_vel_weight.
        ramp_attr: Reward attribute the ramp applies to (TOML ``ramp_attr``).
            Only ``"forward_vel_weight"`` is supported on the MJX path — it is
            the one weight wired through ``env.step`` as a runtime scale;
            everything else is baked into the jitted reward at trace time.
            Accepted (and refused loudly for other values) so the TOML key
            behaves identically here and in the notebook's :func:`train` path,
            instead of being silently dropped.
    """

    def __init__(
        self,
        env: Any,
        network: Any,
        optimizer: Any,
        ppo_config: Any,
        *,
        num_envs: int = 2048,
        rollout_len: int = 64,
        hooks: list[TrainingHook] | None = None,
        warmup_updates: int = 0,
        warmup_clip_range: float = 0.02,
        warmup_ent_coef: float = 0.02,
        ramp_updates: int = 0,
        ramp_start_fraction: float = 0.1,
        ramp_attr: str = "forward_vel_weight",
    ):
        if ramp_updates > 0 and ramp_attr != "forward_vel_weight":
            # Same contract as the module-level train(): the MJX env captures
            # most weights as Python constants at trace time, so they cannot
            # be ramped without recompilation.  Fail loud rather than
            # silently ramping the wrong attribute (or nothing).
            raise ValueError(
                f"Reward ramp on attr={ramp_attr!r} is not supported by the MJX path; "
                "only 'forward_vel_weight' is dynamic at runtime."
            )
        self.env = env
        self.network = network
        self.optimizer = optimizer
        self.ppo_config = ppo_config
        self.num_envs = num_envs
        self.rollout_len = rollout_len
        self.hooks = list(hooks) if hooks else []
        self.warmup_updates = warmup_updates
        self.warmup_clip_range = warmup_clip_range
        self.warmup_ent_coef = warmup_ent_coef
        self.ramp_updates = ramp_updates
        self.ramp_start_fraction = ramp_start_fraction
        self.ramp_attr = ramp_attr

        self._collect_rollout: Callable | None = None
        self._scan_ppo_update: Callable | None = None
        self._scan_ppo_update_warmup: Callable | None = None

    def _dispatch(self, method: str, *args: Any) -> None:
        for hook in self.hooks:
            fn = getattr(hook, method, None)
            if fn is not None:
                fn(*args)

    def _build_collect_rollout(self):
        """Build the JIT-compiled rollout collector.

        Returned tuple ordering (per timestep):
        ``(raw_obs, raw_action, log_prob, value, reward, gae_done, full_done, final_obs)``.

        - ``raw_obs`` is the unnormalized obs at the input to the policy.
          Callers normalize once on the host with the *same* stats used
          inside this rollout to keep PPO importance sampling consistent.
        - ``gae_done`` masks ONLY natural termination — with ``full_done``
          it identifies time-limit truncations, which bootstrap their value
          per step (see :func:`_bootstrap_truncations`).
        - ``full_done`` marks any episode boundary (terminated or
          truncated) — the GAE carry mask, and used for episode tracking
          and fall-rate stats.
        - ``final_obs`` is the post-step pre-auto-reset obs, valued at
          every step to bootstrap truncation boundaries correctly.
        """
        import jax
        import jax.numpy as jnp

        from .jax_normalization import normalize_obs
        from .jax_ppo import sample_action

        env = self.env
        network = self.network
        num_envs = self.num_envs
        rollout_len = self.rollout_len

        @jax.jit
        def collect_rollout(states, rng, params, obs_stats_arg, forward_vel_scale):
            def step_fn(carry, _):
                states, rng = carry
                rng, action_rng = jax.random.split(rng)

                # Normalize for the policy but store the RAW obs so the
                # caller can re-normalize with the exact same stats later.
                raw_obs = states.obs
                obs_normed = normalize_obs(raw_obs, obs_stats_arg)
                raw_action, log_prob, value = jax.vmap(
                    sample_action,
                    in_axes=(None, None, 0, 0),
                )(params, network, obs_normed, jax.random.split(action_rng, num_envs))

                # Clip for env; store raw action for PPO ratio consistency
                action = jnp.clip(raw_action, -1.0, 1.0)
                rng, step_rng = jax.random.split(rng)
                new_states, rewards, terminated, truncated, final_obs = env.step(
                    states, action, step_rng, return_final_obs=True, forward_vel_scale=forward_vel_scale
                )
                gae_done = terminated.astype(jnp.float32)
                full_done = (terminated | truncated).astype(jnp.float32)

                return (
                    (new_states, rng),
                    (raw_obs, raw_action, log_prob, value, rewards, gae_done, full_done, final_obs),
                )

            return jax.lax.scan(step_fn, (states, rng), None, length=rollout_len)

        return collect_rollout

    def _build_scan_ppo_update(self, ppo_config: Any | None = None):
        """Build the JIT-compiled PPO updater with minibatch shuffling and KL early stopping.

        Args:
            ppo_config: PPO hyperparameters baked into the jitted update
                (defaults to ``self.ppo_config``).  A separate warm-up
                variant is built with reduced clip range / raised entropy.
        """
        import jax
        import jax.numpy as jnp

        from .jax_ppo import ppo_update

        optimizer = self.optimizer
        network = self.network
        ppo_config = self.ppo_config if ppo_config is None else ppo_config
        n_minibatches = ppo_config.n_minibatches

        @jax.jit
        def scan_ppo_update(params, opt_state, batch, rng):
            total_samples = batch["obs"].shape[0]
            minibatch_size = total_samples // n_minibatches

            def epoch_fn(carry, _):
                params, opt_state, rng, kl_exceeded = carry
                rng, rng_perm = jax.random.split(rng)
                perm = jax.random.permutation(rng_perm, total_samples)

                def to_mbs(arr):
                    return arr[perm[: n_minibatches * minibatch_size]].reshape(
                        n_minibatches, minibatch_size, *arr.shape[1:]
                    )

                mb_batch = jax.tree.map(to_mbs, batch)

                def mb_step(carry, mb):
                    params, opt_state, kl_exceeded = carry
                    new_params, new_opt_state, loss_info = ppo_update(
                        params,
                        opt_state,
                        optimizer,
                        network,
                        mb,
                        ppo_config,
                    )
                    approx_kl = loss_info["approx_kl"]

                    # Compare in Python so we never evaluate ``traced > None``
                    # when target_kl is disabled.
                    if ppo_config.target_kl is not None:
                        should_skip = kl_exceeded | (approx_kl > ppo_config.target_kl)
                    else:
                        should_skip = kl_exceeded

                    # See note in scan_ppo_epochs: skipped updates revert
                    # the entire opt_state (including LR-schedule step), so
                    # the schedule effectively pauses on KL early-stop.
                    out_params = jax.tree.map(
                        lambda new, old: jnp.where(should_skip, old, new),
                        new_params,
                        params,
                    )
                    out_opt_state = jax.tree.map(
                        lambda new, old: jnp.where(should_skip, old, new) if hasattr(new, "shape") else new,
                        new_opt_state,
                        opt_state,
                    )

                    return (out_params, out_opt_state, should_skip), loss_info

                (params, opt_state, kl_exceeded), all_mb_info = jax.lax.scan(
                    mb_step, (params, opt_state, kl_exceeded), mb_batch
                )
                return (params, opt_state, rng, kl_exceeded), all_mb_info

            init_carry = (params, opt_state, rng, jnp.bool_(False))
            (params, opt_state, _, _), all_info = jax.lax.scan(
                epoch_fn,
                init_carry,
                None,
                length=ppo_config.n_epochs,
            )
            mean_info = jax.tree.map(jnp.mean, all_info)
            return params, opt_state, mean_info

        return scan_ppo_update

    def train(
        self,
        num_updates: int = 500,
        seed: int = 42,
        init_params: Any | None = None,
        init_opt_state: Any | None = None,
        init_obs_stats: Any | None = None,
        start_update: int = 0,
    ) -> tuple[Any, dict[str, float], TrainerState]:
        """Run the training loop.

        Args:
            num_updates: Number of PPO update iterations to run in this call.
            seed: Random seed.
            init_params: Optional initial network parameters.
            init_opt_state: Optional optimizer state to resume from (see
                :func:`environments.shared.jax_checkpoint.restore_train_state`).
                Ignored unless *init_params* is also provided.
            init_obs_stats: Optional observation RunningMeanStd to resume
                from (must match *init_params*' normalization).
            start_update: Absolute index of the first update this call runs.
                A same-stage resume passes the checkpoint's update so the
                curriculum ramp, the progress log and — through
                ``state.update`` — the checkpoint numbering continue from
                where the interrupted session stopped instead of restarting
                at 0 and overwriting its files.

        Returns:
            ``(params, eval_metrics, state)`` tuple.
        """
        check_jax()
        if start_update < 0:
            raise ValueError(f"start_update must be non-negative, got {start_update}")

        import jax
        import jax.numpy as jnp

        from .jax_normalization import RunningMeanStd, normalize_obs, update_running_stats
        from .jax_ppo import compute_gae

        _logger.info("num_envs=%d rollout_len=%d num_updates=%d", self.num_envs, self.rollout_len, num_updates)

        rng = jax.random.PRNGKey(seed)
        rng, init_rng, reset_rng = jax.random.split(rng, 3)

        # Derive obs_dim from a real reset rather than reconstructing it from
        # nq/nv arithmetic — the latter forgets foot-contact channels.
        states = self.env.reset(reset_rng)
        obs_dim = int(states.obs.shape[-1])
        dummy_obs = jnp.zeros((obs_dim,))
        params = self.network.init(init_rng, dummy_obs) if init_params is None else init_params
        if init_params is not None and init_opt_state is not None:
            opt_state = init_opt_state
        else:
            opt_state = self.optimizer.init(params)

        obs_stats = init_obs_stats if init_obs_stats is not None else RunningMeanStd.create(obs_dim)

        self._collect_rollout = self._build_collect_rollout()
        self._scan_ppo_update = self._build_scan_ppo_update()
        if self.warmup_updates > 0:
            warmup_cfg = self.ppo_config._replace(
                clip_range=self.warmup_clip_range,
                ent_coef=self.warmup_ent_coef,
            )
            self._scan_ppo_update_warmup = self._build_scan_ppo_update(warmup_cfg)
            _logger.info(
                "Warmup: updates 0..%d (clip_range=%s, ent_coef=%s)",
                self.warmup_updates - 1,
                self.warmup_clip_range,
                self.warmup_ent_coef,
            )
        if self.ramp_updates > 0:
            _logger.info(
                "Ramp: %s scale %s -> 1.0 over %d updates",
                self.ramp_attr,
                self.ramp_start_fraction,
                self.ramp_updates,
            )

        # Jitted value head over the per-step final_obs — built once so it is
        # not re-traced each update.
        _batched_value = _build_batched_value(self.network)

        state = TrainerState(
            params=params,
            opt_state=opt_state,
            obs_stats=obs_stats,
            env_states=states,
            rng=rng,
        )

        self._dispatch("on_train_start", state)

        t0 = time.time()

        from .jax_training_utils import EpisodeStatsAccumulator, compute_episode_stats

        ep_stats_acc = EpisodeStatsAccumulator()

        try:
            for update in range(start_update, start_update + num_updates):
                state.update = update

                t_rollout_start = time.time()
                rng, collect_rng = jax.random.split(state.rng)
                state.rng = rng
                # Snapshot the obs stats used during this rollout — we'll
                # normalize all PPO inputs with the SAME stats so the
                # importance-sampling ratio stays consistent.  Stats are
                # updated AFTER PPO consumes the rollout.
                rollout_obs_stats = state.obs_stats
                # Reward ramp: scale forward_vel_weight from
                # ramp_start_fraction up to 1.0 over the first ramp_updates.
                if self.ramp_updates > 0 and update < self.ramp_updates:
                    ramp_progress = update / self.ramp_updates
                    forward_vel_scale = self.ramp_start_fraction + (1.0 - self.ramp_start_fraction) * ramp_progress
                else:
                    forward_vel_scale = 1.0
                (states, _), rollout_data = self._collect_rollout(
                    state.env_states,
                    collect_rng,
                    state.params,
                    rollout_obs_stats,
                    jnp.float32(forward_vel_scale),
                )
                state.env_states = states
                (
                    rollout_obs,
                    rollout_actions,
                    rollout_log_probs,
                    rollout_values,
                    rollout_rewards,
                    rollout_gae_dones,
                    rollout_full_dones,
                    rollout_final_obs,
                ) = rollout_data

                jax.block_until_ready(rollout_full_dones)
                t_rollout = time.time() - t_rollout_start
                state.t_rollout_cumulative += t_rollout
                state.total_steps += self.num_envs * self.rollout_len

                mean_reward = float(jnp.mean(rollout_rewards))
                elapsed = time.time() - t0
                fps = state.total_steps / elapsed if elapsed > 0 else 0

                # Compute fall rate on-device; only transfer scalars.
                # gae_dones marks natural termination only -- counting
                # full_dones would book time-limit truncations as "falls".
                fall_rate = float(jnp.sum(rollout_gae_dones)) / (self.rollout_len * self.num_envs)
                # Defer full GPU→CPU transfer for episode tracking
                rew_np = np.array(rollout_rewards)
                done_np = np.array(rollout_full_dones)
                completed_returns, completed_lengths = compute_episode_stats(rew_np, done_np, ep_stats_acc)
                mean_ep_return = float(np.mean(completed_returns)) if completed_returns else float("nan")
                mean_ep_length = float(np.mean(completed_lengths)) if completed_lengths else float("nan")

                rollout_metrics = {
                    "mean_reward": mean_reward,
                    "episode_return": mean_ep_return,
                    "episode_length": mean_ep_length,
                    "fall_rate": fall_rate,
                    "fps": fps,
                    "elapsed": elapsed,
                    "t_rollout": t_rollout,
                }
                self._dispatch("on_rollout_end", state, rollout_metrics)

                # Normalize PPO inputs with the SAME stats used during sampling
                rollout_obs_norm = normalize_obs(rollout_obs.reshape(-1, obs_dim), rollout_obs_stats)

                # Value every step's pre-auto-reset ``final_obs`` under the
                # rollout policy and stats.  The last entry is the tail
                # bootstrap (``state.env_states.obs`` would be the post-reset
                # obs if the last step truncated); every other entry
                # bootstraps a MID-rollout truncation via
                # _bootstrap_truncations.  Terminations stay masked out.
                final_obs_norm = normalize_obs(rollout_final_obs.reshape(-1, obs_dim), rollout_obs_stats)
                final_values = _batched_value(state.params, final_obs_norm).reshape(self.rollout_len, self.num_envs)

                rollout_values_arr = jnp.concatenate([rollout_values, final_values[-1][None]], axis=0)
                rewards_boot, gae_mask = _bootstrap_truncations(
                    rollout_rewards,
                    rollout_gae_dones,
                    rollout_full_dones,
                    final_values,
                    self.ppo_config.gamma,
                )
                advantages, returns = compute_gae(
                    rewards_boot,
                    rollout_values_arr,
                    gae_mask,
                    self.ppo_config.gamma,
                    self.ppo_config.gae_lambda,
                )

                # Update running obs stats AFTER the rollout has been fully
                # consumed for normalization, using RAW obs (never normalized).
                state.obs_stats = update_running_stats(
                    state.obs_stats,
                    rollout_obs.reshape(-1, obs_dim),
                )

                batch = {
                    "obs": rollout_obs_norm,
                    "action": rollout_actions.reshape(-1, self.env.action_dim),
                    "old_log_prob": rollout_log_probs.reshape(-1),
                    "old_value": rollout_values.reshape(-1),
                    "advantage": advantages.reshape(-1),
                    "return_": returns.reshape(-1),
                }

                t_ppo_start = time.time()
                rng, ppo_rng = jax.random.split(state.rng)
                state.rng = rng
                in_warmup = self._scan_ppo_update_warmup is not None and update < self.warmup_updates
                _updater = self._scan_ppo_update_warmup if in_warmup else self._scan_ppo_update
                assert _updater is not None  # built above
                state.params, state.opt_state, ppo_info = _updater(
                    state.params,
                    state.opt_state,
                    batch,
                    ppo_rng,
                )
                ppo_info = {k: float(v) for k, v in ppo_info.items()}
                t_ppo = time.time() - t_ppo_start
                state.t_ppo_cumulative += t_ppo

                state.reward_history.append(mean_reward)
                state.episode_return_history.append(mean_ep_return)
                state.episode_length_history.append(mean_ep_length)

                update_metrics = {
                    "update": update,
                    "total_steps": state.total_steps,
                    "mean_reward": mean_reward,
                    "episode_return": mean_ep_return,
                    "episode_length": mean_ep_length,
                    "fall_rate": fall_rate,
                    "fps": fps,
                    "elapsed": elapsed,
                    "t_rollout": t_rollout,
                    "t_ppo": t_ppo,
                    **ppo_info,
                }
                state.history.append(update_metrics)
                self._dispatch("on_update_end", state, update_metrics)

        except StopTraining as e:
            _logger.info("Training stopped early: %s", e.reason)

        elapsed = time.time() - t0
        if state.total_steps > 0:
            _logger.info(
                "Done. %s steps in %.1fs (%.0f fps)",
                f"{state.total_steps:,}",
                elapsed,
                state.total_steps / elapsed,
            )

        self._dispatch("on_train_end", state)

        return state.params, _build_eval_metrics(state, elapsed), state
