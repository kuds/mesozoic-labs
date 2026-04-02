"""Reusable JAX training loop library.

Separates the generic training loop (rollout collection, advantage
estimation, policy update, hook dispatch) from algorithm-specific and
application-specific concerns.

The main entry point is :class:`JaxTrainer`, which runs a JIT-compiled
training loop and dispatches lifecycle events to pluggable
:class:`TrainingHook` instances.

Usage::

    from environments.shared.jax_trainer import JaxTrainer, TrainerState
    from environments.shared.jax_ppo import PPOConfig, make_actor_critic, make_optimizer

    env = MJXDinoEnv("trex", stage=1, num_envs=2048)
    network = make_actor_critic(env.action_dim)
    config = PPOConfig(learning_rate=3e-4)
    optimizer = make_optimizer(config)

    trainer = JaxTrainer(
        env=env,
        network=network,
        optimizer=optimizer,
        ppo_config=config,
        hooks=[LoggingHook(interval=10)],
    )
    params, metrics = trainer.train(num_updates=500, seed=42)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .mjx_utils import check_jax

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------


@dataclass
class TrainerState:
    """All mutable state for a training run.

    Passed to hooks so they can inspect (but should not mutate) training
    progress.  The trainer itself updates state between steps.
    """

    params: Any
    opt_state: Any
    obs_stats: Any  # RunningMeanStd
    env_states: Any  # EnvState (batched)
    rng: Any  # jax PRNGKey
    update: int = 0
    total_steps: int = 0
    history: list[dict[str, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Hook protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class TrainingHook(Protocol):
    """Lifecycle hook for the JAX training loop.

    All methods are optional — implement only the ones you need.
    The default implementations are no-ops.
    """

    def on_train_start(self, state: TrainerState) -> None:
        """Called once before the first update."""
        ...

    def on_rollout_end(
        self,
        state: TrainerState,
        rollout_metrics: dict[str, float],
    ) -> None:
        """Called after each rollout collection, before the PPO update."""
        ...

    def on_update_end(
        self,
        state: TrainerState,
        update_metrics: dict[str, float],
    ) -> None:
        """Called after each PPO update with per-update metrics.

        Return is ignored.  To halt training early, raise
        ``StopTraining``.
        """
        ...

    def on_train_end(self, state: TrainerState) -> None:
        """Called once after the last update (or after early stop)."""
        ...


class StopTraining(Exception):
    """Raise from a hook to halt training early."""

    def __init__(self, reason: str = ""):
        self.reason = reason
        super().__init__(reason)


# ---------------------------------------------------------------------------
# Trainer
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
    ):
        self.env = env
        self.network = network
        self.optimizer = optimizer
        self.ppo_config = ppo_config
        self.num_envs = num_envs
        self.rollout_len = rollout_len
        self.hooks = list(hooks) if hooks else []

        # JIT-compiled functions (created lazily in train())
        self._collect_rollout: Callable | None = None
        self._scan_ppo_update: Callable | None = None

    # -- Hook dispatch helpers -----------------------------------------------

    def _dispatch(self, method: str, *args: Any) -> None:
        for hook in self.hooks:
            fn = getattr(hook, method, None)
            if fn is not None:
                fn(*args)

    # -- JIT function builders -----------------------------------------------

    def _build_collect_rollout(self):
        """Build the JIT-compiled rollout collector."""
        import jax
        import jax.numpy as jnp

        from .jax_normalization import normalize_obs
        from .jax_ppo import sample_action

        env = self.env
        network = self.network
        num_envs = self.num_envs
        rollout_len = self.rollout_len

        @jax.jit
        def collect_rollout(states, rng, params, obs_stats_arg):
            def step_fn(carry, _):
                states, rng = carry
                rng, action_rng = jax.random.split(rng)

                obs = normalize_obs(states.obs, obs_stats_arg)
                action, log_prob, value = jax.vmap(
                    sample_action,
                    in_axes=(None, None, 0, 0),
                )(params, network, obs, jax.random.split(action_rng, num_envs))

                rng, step_rng = jax.random.split(rng)
                new_states, rewards, terminated, truncated = env.step(states, action, step_rng)
                dones = (terminated | truncated).astype(jnp.float32)

                return (new_states, rng), (states.obs, action, log_prob, value, rewards, dones)

            return jax.lax.scan(step_fn, (states, rng), None, length=rollout_len)

        return collect_rollout

    def _build_scan_ppo_update(self):
        """Build the JIT-compiled PPO updater with KL early stopping."""
        import jax
        import jax.numpy as jnp

        from .jax_ppo import ppo_update

        optimizer = self.optimizer
        network = self.network
        ppo_config = self.ppo_config

        @jax.jit
        def scan_ppo_update(params, opt_state, batch, rng):
            def epoch_fn(carry, _):
                params, opt_state, rng, kl_exceeded = carry
                new_params, new_opt_state, loss_info = ppo_update(
                    params,
                    opt_state,
                    optimizer,
                    network,
                    batch,
                    ppo_config,
                )
                approx_kl = loss_info["approx_kl"]

                use_target_kl = ppo_config.target_kl is not None
                kl_over = use_target_kl & (approx_kl > ppo_config.target_kl)
                should_skip = kl_exceeded | kl_over

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

                return (out_params, out_opt_state, rng, should_skip), loss_info

            init_carry = (params, opt_state, rng, jnp.bool_(False))
            (params, opt_state, _, _), all_info = jax.lax.scan(
                epoch_fn,
                init_carry,
                None,
                length=ppo_config.n_epochs,
            )
            return params, opt_state

        return scan_ppo_update

    # -- Main training loop --------------------------------------------------

    def train(
        self,
        num_updates: int = 500,
        seed: int = 42,
        init_params: Any | None = None,
    ) -> tuple[Any, dict[str, float]]:
        """Run the training loop.

        Args:
            num_updates: Number of PPO update iterations.
            seed: Random seed.
            init_params: Optional initial network parameters (e.g. from
                a previous curriculum stage).

        Returns:
            ``(params, eval_metrics)`` tuple.
        """
        check_jax()

        import jax
        import jax.numpy as jnp

        from .jax_normalization import RunningMeanStd, normalize_obs, update_running_stats
        from .jax_ppo import compute_gae, sample_action

        _logger.info("num_envs=%d rollout_len=%d num_updates=%d", self.num_envs, self.rollout_len, num_updates)
        _logger.info("device: %s", jax.devices()[0])

        # Initialise network
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        dummy_obs = jnp.zeros(
            self.env.mj_model.nq - 7 + self.env.mj_model.nv - 6 + 17,
        )
        params = self.network.init(init_rng, dummy_obs) if init_params is None else init_params
        opt_state = self.optimizer.init(params)

        obs_dim = dummy_obs.shape[0]
        obs_stats = RunningMeanStd.create(obs_dim)

        # Reset environments
        rng, reset_rng = jax.random.split(rng)
        states = self.env.reset(reset_rng)

        # Build JIT-compiled functions
        self._collect_rollout = self._build_collect_rollout()
        self._scan_ppo_update = self._build_scan_ppo_update()

        # Create state container
        state = TrainerState(
            params=params,
            opt_state=opt_state,
            obs_stats=obs_stats,
            env_states=states,
            rng=rng,
        )

        # Dispatch on_train_start
        self._dispatch("on_train_start", state)

        t0 = time.time()
        _logger.info("Compiling scan functions (first update only)...")

        try:
            for update in range(num_updates):
                state.update = update

                # -- Collect rollout --
                rng, collect_rng = jax.random.split(state.rng)
                state.rng = rng
                (states, _), rollout_data = self._collect_rollout(
                    state.env_states,
                    collect_rng,
                    state.params,
                    state.obs_stats,
                )
                state.env_states = states
                (
                    rollout_obs,
                    rollout_actions,
                    rollout_log_probs,
                    rollout_values,
                    rollout_rewards,
                    rollout_dones,
                ) = rollout_data

                state.total_steps += self.num_envs * self.rollout_len

                # Update obs stats
                state.obs_stats = update_running_stats(
                    state.obs_stats,
                    rollout_obs.reshape(-1, obs_dim),
                )

                mean_reward = float(jnp.mean(rollout_rewards))
                elapsed = time.time() - t0
                fps = state.total_steps / elapsed if elapsed > 0 else 0

                rollout_metrics = {
                    "mean_reward": mean_reward,
                    "fps": fps,
                    "elapsed": elapsed,
                }

                self._dispatch("on_rollout_end", state, rollout_metrics)

                # -- Normalise observations for PPO --
                rollout_obs_norm = normalize_obs(
                    rollout_obs.reshape(-1, obs_dim),
                    state.obs_stats,
                )

                # Bootstrap value
                rng, bootstrap_rng = jax.random.split(state.rng)
                state.rng = rng
                final_obs = normalize_obs(state.env_states.obs, state.obs_stats)
                _, _, bootstrap_value = jax.vmap(
                    lambda o, r: sample_action(state.params, self.network, o, r),
                )(final_obs, jax.random.split(bootstrap_rng, self.num_envs))

                # GAE
                rollout_values_arr = jnp.concatenate(
                    [rollout_values, bootstrap_value[None]],
                    axis=0,
                )
                advantages, returns = compute_gae(
                    rollout_rewards,
                    rollout_values_arr,
                    rollout_dones,
                    self.ppo_config.gamma,
                    self.ppo_config.gae_lambda,
                )

                batch = {
                    "obs": rollout_obs_norm,
                    "action": rollout_actions.reshape(-1, self.env.action_dim),
                    "old_log_prob": rollout_log_probs.reshape(-1),
                    "old_value": rollout_values.reshape(-1),
                    "advantage": advantages.reshape(-1),
                    "return_": returns.reshape(-1),
                }

                # -- PPO update --
                rng, ppo_rng = jax.random.split(state.rng)
                state.rng = rng
                state.params, state.opt_state = self._scan_ppo_update(
                    state.params,
                    state.opt_state,
                    batch,
                    ppo_rng,
                )

                # Metrics
                update_metrics = {
                    "update": update,
                    "total_steps": state.total_steps,
                    "mean_reward": mean_reward,
                    "fps": fps,
                    "elapsed": elapsed,
                }
                state.history.append(update_metrics)

                self._dispatch("on_update_end", state, update_metrics)

        except StopTraining as e:
            _logger.info("Training stopped early: %s", e.reason)

        # Final metrics
        elapsed = time.time() - t0
        if state.total_steps > 0:
            _logger.info(
                "Done. %s steps in %.1fs (%.0f fps)",
                f"{state.total_steps:,}",
                elapsed,
                state.total_steps / elapsed,
            )

        self._dispatch("on_train_end", state)

        eval_metrics = {
            "mean_reward": float(jnp.mean(jnp.array([h["mean_reward"] for h in state.history[-10:]])))
            if state.history
            else 0.0,
            "total_steps": state.total_steps,
        }

        return state.params, eval_metrics
