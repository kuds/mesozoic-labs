"""PPO training loop for JAX/MJX environments.

Two APIs are provided:

1. **Notebook API** (``train()`` function): direct, config-driven training
   with built-in logging, checkpointing, warmup/ramp, and minibatch
   shuffling.  Used by ``jax_training.ipynb``.

2. **Library API** (``JaxTrainer`` class): hook-based training loop used by
   ``jax_training.py`` (CLI) and ``jax_hooks.py``.  Extensible via
   :class:`TrainingHook` protocol.

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .jax_trainer_types import StopTraining, TrainerState, TrainingHook  # noqa: F401  (re-exported)
from .mjx_utils import check_jax

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """All hyperparameters and settings for a PPO training run."""

    # Core PPO hyperparameters
    num_envs: int = 2048
    rollout_len: int = 64
    num_updates: int = 500
    ppo_epochs: int = 4
    minibatch_size: int = 512
    learning_rate: float = 3e-4
    learning_rate_end: float | None = None
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    vf_clip_range: float | None = None
    target_kl: float | None = 0.05

    # Dimensions (set from env/species context)
    obs_dim: int = 0
    act_dim: int = 0

    # Curriculum warmup
    warmup_updates: int = 0
    warmup_clip_range: float = 0.02
    warmup_ent_coef: float = 0.02

    # Reward ramp
    ramp_updates: int = 0
    ramp_attr: str = "forward_vel_weight"
    ramp_start_fraction: float = 0.1

    # Checkpointing
    checkpoint_freq: int = 25
    max_checkpoints: int = 5

    # Logging
    verbose: int = 1  # 0=summary only, 1=periodic, 2=every update
    reward_component_interval: int = 10

    # Output paths (converted to Path in __post_init__)
    output_dir: Any = "."
    model_dir: Any = "."

    # Resume
    start_update: int = 0

    # Env metadata (for summary printing)
    species: str = ""
    stage: int = 1
    frame_skip: int = 5

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.model_dir = Path(self.model_dir)

    @property
    def log_interval(self) -> int | None:
        return {0: None, 1: 20, 2: 1}.get(self.verbose, 20)

    @property
    def batch_size(self) -> int:
        return self.rollout_len * self.num_envs

    @property
    def total_env_steps(self) -> int:
        return self.num_updates * self.rollout_len * self.num_envs


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class TrainResult:
    """Everything returned from a training run."""

    params: Any
    opt_state: Any
    obs_rms: Any
    best_params: Any
    best_reward: float
    best_update: int

    reward_history: list[float] = field(default_factory=list)
    loss_history: list[float] = field(default_factory=list)
    episode_return_history: list[float] = field(default_factory=list)
    diagnostics_history: list[dict[str, Any]] = field(default_factory=list)
    reward_component_history: list[dict[str, Any]] = field(default_factory=list)

    total_steps: int = 0
    elapsed: float = 0.0
    actual_updates: int = 0
    csv_path: Path | None = None


# ---------------------------------------------------------------------------
# JIT function builders
# ---------------------------------------------------------------------------


def _build_jit_fns(config: TrainConfig, network, optimizer, reward_detail_fn, env=None):
    """Build all JIT-compiled functions for the training loop.

    Returns a dict of JIT functions.  These close over the network, optimizer,
    and config values that are fixed for the entire run.
    """
    import jax
    import jax.numpy as jnp
    import optax

    from .jax_ppo import ppo_loss, sample_action

    MINIBATCH_SIZE = config.minibatch_size
    PPO_EPOCHS = config.ppo_epochs
    TARGET_KL = config.target_kl

    # --- Thin wrappers for the network ---

    def _sample_action(params, obs, rng):
        return sample_action(params, network, obs, rng)

    VF_COEF = config.vf_coef

    def _ppo_loss(
        params,
        obs,
        actions,
        old_log_probs,
        advantages,
        returns,
        old_values=None,
        clip_range=0.2,
        vf_coef=VF_COEF,
        ent_coef=0.01,
        vf_clip_range=None,
    ):
        batch = {
            "obs": obs,
            "action": actions,
            "old_log_prob": old_log_probs,
            "advantage": advantages,
            "return_": returns,
        }
        if old_values is not None:
            batch["old_value"] = old_values
        from .jax_ppo import PPOConfig as _PPOConfig

        cfg = _PPOConfig(clip_range=clip_range, vf_coef=vf_coef, ent_coef=ent_coef, vf_clip_range=vf_clip_range)
        return ppo_loss(params, network, batch, cfg)

    # --- Batched action sampling ---

    @jax.jit
    def batched_sample(params, obs_batch, rng):
        rngs = jax.random.split(rng, obs_batch.shape[0])
        return jax.vmap(_sample_action, in_axes=(None, 0, 0))(params, obs_batch, rngs)

    # --- Single PPO gradient step ---

    @jax.jit
    def ppo_update(
        params,
        opt_state,
        obs,
        actions,
        log_probs,
        advantages,
        returns,
        old_values=None,
        clip_range=0.2,
        ent_coef=0.01,
        vf_clip_range=None,
    ):
        (loss, aux), grads = jax.value_and_grad(_ppo_loss, has_aux=True)(
            params,
            obs,
            actions,
            log_probs,
            advantages,
            returns,
            old_values=old_values,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_clip_range=vf_clip_range,
        )
        grad_norm = optax.global_norm(grads)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux, grad_norm

    # --- Fused scan PPO epochs with KL early stopping ---

    @jax.jit
    def scan_ppo_epochs(
        params, opt_state, flat_obs, flat_act, flat_lp, flat_adv, flat_ret, flat_val, rng, clip_range, ent_coef
    ):
        total_samples = flat_obs.shape[0]
        n_minibatches = total_samples // MINIBATCH_SIZE

        def epoch_fn(carry, _):
            params, opt_state, rng, kl_exceeded = carry
            rng, rng_perm = jax.random.split(rng)
            perm = jax.random.permutation(rng_perm, total_samples)

            def to_mbs(arr):
                return arr[perm[: n_minibatches * MINIBATCH_SIZE]].reshape(
                    n_minibatches, MINIBATCH_SIZE, *arr.shape[1:]
                )

            mb_data = (
                to_mbs(flat_obs),
                to_mbs(flat_act),
                to_mbs(flat_lp),
                to_mbs(flat_adv),
                to_mbs(flat_ret),
                to_mbs(flat_val),
            )

            def mb_step(carry, mb):
                params, opt_state, kl_exceeded = carry
                obs, act, lp, adv, ret, val = mb
                new_params, new_opt_state, loss, aux, gn = ppo_update(
                    params, opt_state, obs, act, lp, adv, ret, old_values=val, clip_range=clip_range, ent_coef=ent_coef
                )

                approx_kl = aux["approx_kl"]
                use_target_kl = TARGET_KL is not None
                kl_over = use_target_kl & (approx_kl > TARGET_KL)
                should_skip = kl_exceeded | kl_over

                out_params = jax.tree.map(lambda new, old: jnp.where(should_skip, old, new), new_params, params)
                out_opt_state = jax.tree.map(
                    lambda new, old: jnp.where(should_skip, old, new) if hasattr(new, "shape") else new,
                    new_opt_state,
                    opt_state,
                )

                return (out_params, out_opt_state, should_skip), (loss, aux, gn)

            (params, opt_state, kl_exceeded), (losses, auxs, gns) = jax.lax.scan(
                mb_step, (params, opt_state, kl_exceeded), mb_data
            )
            return (params, opt_state, rng, kl_exceeded), (losses, auxs, gns)

        init_carry = (params, opt_state, rng, jnp.bool_(False))
        (params, opt_state, _, _), (all_losses, all_auxs, all_gns) = jax.lax.scan(
            epoch_fn, init_carry, None, length=PPO_EPOCHS
        )

        mean_loss = jnp.mean(all_losses)
        mean_gn = jnp.mean(all_gns)
        mean_aux = jax.tree.map(jnp.mean, all_auxs)
        return params, opt_state, mean_loss, mean_aux, mean_gn

    # --- Scan-based rollout collection (replaces Python for-loop) ---

    collect_rollout = None
    if env is not None:
        from .jax_normalization import normalize_obs

        ROLLOUT_LEN = config.rollout_len
        NUM_ENVS = config.num_envs

        @jax.jit
        def collect_rollout(states, rng, params, obs_rms):
            def step_fn(carry, _):
                states, rng = carry
                rng, action_rng, step_rng = jax.random.split(rng, 3)

                obs = normalize_obs(states.obs, obs_rms)
                rngs = jax.random.split(action_rng, NUM_ENVS)
                raw_actions, log_probs, values = jax.vmap(_sample_action, in_axes=(None, 0, 0))(params, obs, rngs)

                actions = jnp.clip(raw_actions, -1.0, 1.0)
                new_states, rewards, terminated, truncated = env.step(states, actions, step_rng)
                dones = terminated | truncated
                gae_dones = terminated

                return (new_states, rng), (
                    obs,
                    raw_actions,
                    log_probs,
                    values,
                    rewards,
                    gae_dones.astype(jnp.float32),
                    dones.astype(jnp.float32),
                )

            (states, _), rollout = jax.lax.scan(step_fn, (states, rng), None, length=ROLLOUT_LEN)
            return states, rollout

    # --- Reward component diagnostics ---

    batched_reward_components = None
    if reward_detail_fn is not None:

        @jax.jit
        def batched_reward_components(states, action_batch):
            return jax.vmap(reward_detail_fn, in_axes=(0, 0))(states.data, action_batch)

    return {
        "batched_sample": batched_sample,
        "ppo_update": ppo_update,
        "scan_ppo_epochs": scan_ppo_epochs,
        "collect_rollout": collect_rollout,
        "batched_reward_components": batched_reward_components,
    }


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train(
    config: TrainConfig,
    env,
    network,
    params,
    opt_state,
    obs_rms,
    reward_cfg: dict[str, Any],
    rng,
    reward_detail_fn: Callable | None = None,
    optimizer=None,
    callback: Callable[[int, dict[str, Any]], None] | None = None,
) -> TrainResult:
    """Run the PPO training loop.

    Args:
        config: Training hyperparameters and settings.
        env: ``MJXDinoEnv`` instance (already created with ``num_envs``).
        network: Flax ActorCritic module.
        params: Initial network parameters.
        opt_state: Initial optimizer state.
        obs_rms: Initial observation normalisation statistics.
        reward_cfg: Reward configuration dict (may be mutated by reward ramp).
        rng: JAX PRNGKey.
        reward_detail_fn: Optional per-component reward function for diagnostics.
            Signature: ``(mjx_data, action) -> dict[str, scalar]``.
        optimizer: Optax optimizer (needed for gradient updates).
        callback: Optional ``(update, metrics_dict) -> None`` called each update.

    Returns:
        ``TrainResult`` with final params, histories, and metadata.
    """
    import jax
    import jax.numpy as jnp

    from .jax_checkpoint import CheckpointManager, save_checkpoint
    from .jax_normalization import normalize_obs, update_running_stats
    from .jax_ppo import compute_gae
    from .jax_training_utils import (
        EpisodeStatsAccumulator,
        StabilityMonitor,
        TrainingCSVLogger,
        compute_episode_stats,
    )
    from .reporting import format_duration

    assert optimizer is not None, "optimizer is required"

    # Build JIT functions
    jit_fns = _build_jit_fns(config, network, optimizer, reward_detail_fn, env=env)
    batched_sample = jit_fns["batched_sample"]
    scan_ppo_epochs = jit_fns["scan_ppo_epochs"]
    collect_rollout = jit_fns["collect_rollout"]
    batched_reward_components = jit_fns["batched_reward_components"]

    # Aliases
    NUM_ENVS = config.num_envs
    ROLLOUT_LEN = config.rollout_len
    OBS_DIM = config.obs_dim
    ACT_DIM = config.act_dim
    GAMMA = config.gamma
    GAE_LAMBDA = config.gae_lambda
    CHECKPOINT_FREQ = config.checkpoint_freq

    _start_update = config.start_update
    _log_interval = config.log_interval

    # Histories
    reward_history: list[float] = []
    loss_history: list[float] = []
    diagnostics_history: list[dict[str, Any]] = []
    reward_component_history: list[dict[str, Any]] = []
    episode_return_history: list[float] = []

    # Best model tracking
    best_reward = -float("inf")
    best_params = None
    best_update = -1

    # Episode tracking
    _ep_stats_acc = EpisodeStatsAccumulator()

    # Library utilities
    _ckpt_mgr = CheckpointManager(config.model_dir, prefix="checkpoint", max_keep=config.max_checkpoints)
    _stability = StabilityMonitor()
    _csv_logger = TrainingCSVLogger(config.output_dir / "training_log.csv")
    csv_path = _csv_logger.path

    # Warmup/ramp state
    _warmup_active = config.warmup_updates > 0
    _ramp_active = config.ramp_updates > 0
    _ramp_target_value = reward_cfg.get(config.ramp_attr, 0.0) if _ramp_active else 0.0

    # Reset environments
    rng, reset_rng = jax.random.split(rng)
    states = env.reset(reset_rng)

    # Print banner
    _batch_size_total = ROLLOUT_LEN * NUM_ENVS
    print(f"Batch size: {_batch_size_total:,} ({ROLLOUT_LEN} steps x {NUM_ENVS} envs)")
    print(
        f"PPO:        {config.ppo_epochs} epochs x "
        f"{_batch_size_total // config.minibatch_size} minibatches of {config.minibatch_size}"
    )
    print(f"Total:      {config.total_env_steps:,} env steps over {config.num_updates} updates")
    print(
        f"\nStarting training: updates {_start_update}..{_start_update + config.num_updates - 1} "
        f"({ROLLOUT_LEN} steps x {NUM_ENVS} envs)"
    )
    print(f"Checkpoint frequency: every {CHECKPOINT_FREQ} updates (keep last {config.max_checkpoints})")
    if _warmup_active:
        print(
            f"Warmup: updates 0..{config.warmup_updates - 1} "
            f"(clip_range={config.warmup_clip_range}, ent_coef={config.warmup_ent_coef})"
        )
    if _ramp_active:
        print(
            f"Reward ramp: {config.ramp_attr} from "
            f"{_ramp_target_value * config.ramp_start_fraction:.4f} to "
            f"{_ramp_target_value:.4f} over updates 0..{config.ramp_updates - 1}"
        )
    print(f"CSV log: {csv_path}")
    print("=" * 70)

    t_start = time.time()
    _cum_t_rollout = 0.0
    _cum_t_ppo = 0.0
    update = _start_update  # ensure defined for finally block

    try:
        for update in range(_start_update, _start_update + config.num_updates):
            relative_update = update - _start_update

            # ---------- Warmup ----------
            if _warmup_active and relative_update < config.warmup_updates:
                _active_clip_range = config.warmup_clip_range
                _active_ent_coef = config.warmup_ent_coef
                if relative_update == 0 and _log_interval is not None:
                    pass  # printed in banner
            else:
                _active_clip_range = config.clip_range
                _active_ent_coef = config.ent_coef
                if _warmup_active and relative_update == config.warmup_updates and _log_interval is not None:
                    print(
                        f"  >>> Warmup complete at update {update}: "
                        f"restoring clip_range={config.clip_range}, ent_coef={config.ent_coef}"
                    )

            # ---------- Reward ramp ----------
            if _ramp_active:
                if relative_update < config.ramp_updates:
                    ramp_progress = relative_update / config.ramp_updates
                    ramp_value = _ramp_target_value * (
                        config.ramp_start_fraction + (1.0 - config.ramp_start_fraction) * ramp_progress
                    )
                else:
                    ramp_value = _ramp_target_value
                reward_cfg[config.ramp_attr] = ramp_value

            # ---------- Collect rollout ----------
            _t_phase = time.time()
            rng, rng_collect = jax.random.split(rng)
            states, rollout = collect_rollout(states, rng_collect, params, obs_rms)
            obs_t, act_t, lp_t, val_t, rew_t, done_t, full_done_t = rollout

            jax.block_until_ready(done_t)
            _t_rollout = time.time() - _t_phase
            _cum_t_rollout += _t_rollout

            # ---------- Episode stats ----------
            # Compute fall rate on-device; only transfer scalars
            fall_rate = float(jnp.sum(full_done_t)) / (ROLLOUT_LEN * NUM_ENVS)
            # Defer full GPU→CPU transfer for episode tracking
            rew_np = np.array(rew_t)
            full_done_np = np.array(full_done_t)
            _completed_returns, _completed_lengths = compute_episode_stats(rew_np, full_done_np, _ep_stats_acc)

            # ---------- Update obs normalisation ----------
            obs_batch_flat = obs_t.reshape(-1, OBS_DIM)
            obs_rms = update_running_stats(obs_rms, obs_batch_flat)

            # ---------- Bootstrap value for GAE ----------
            rng, rng_bootstrap = jax.random.split(rng)
            obs_final = normalize_obs(states.obs, obs_rms)
            _, _, bootstrap_values = batched_sample(params, obs_final, rng_bootstrap)
            val_t_plus1 = jnp.concatenate([val_t, bootstrap_values[None]], axis=0)

            # ---------- Compute advantages ----------
            advantages, returns = compute_gae(rew_t, val_t_plus1, done_t, GAMMA, GAE_LAMBDA)

            flat_obs = obs_t.reshape(-1, OBS_DIM)
            flat_act = act_t.reshape(-1, ACT_DIM)
            flat_lp = lp_t.reshape(-1)
            flat_adv = advantages.reshape(-1)
            flat_ret = returns.reshape(-1)
            flat_val = val_t.reshape(-1)

            # ---------- PPO update ----------
            _t_phase = time.time()
            rng, rng_ppo = jax.random.split(rng)
            params, opt_state, avg_loss, avg_aux, avg_grad_norm = scan_ppo_epochs(
                params,
                opt_state,
                flat_obs,
                flat_act,
                flat_lp,
                flat_adv,
                flat_ret,
                flat_val,
                rng_ppo,
                jnp.float32(_active_clip_range),
                jnp.float32(_active_ent_coef),
            )

            avg_loss = float(avg_loss)
            avg_grad_norm = float(avg_grad_norm)
            avg_aux = {k: float(v) for k, v in avg_aux.items()}
            _t_ppo = time.time() - _t_phase
            _cum_t_ppo += _t_ppo

            # Compute current learning rate for logging
            if config.learning_rate_end is not None and config.learning_rate_end != config.learning_rate:
                total_lr_steps = config.num_updates * config.ppo_epochs
                lr_step = min(relative_update * config.ppo_epochs, total_lr_steps)
                current_lr = (
                    config.learning_rate + (config.learning_rate_end - config.learning_rate) * lr_step / total_lr_steps
                )
            else:
                current_lr = config.learning_rate

            avg_reward = float(rew_t.mean())

            # Episode return stats
            if _completed_returns:
                mean_ep_return = float(np.mean(_completed_returns))
                mean_ep_length = float(np.mean(_completed_lengths))
            else:
                mean_ep_return = float("nan")
                mean_ep_length = float("nan")
            episode_return_history.append(mean_ep_return)

            reward_history.append(avg_reward)
            loss_history.append(avg_loss)
            diagnostics_history.append(
                {
                    "reward": avg_reward,
                    "episode_return": mean_ep_return,
                    "episode_length": mean_ep_length,
                    "loss": avg_loss,
                    "grad_norm": avg_grad_norm,
                    "fall_rate": fall_rate,
                    "learning_rate": current_lr,
                    "t_rollout": _t_rollout,
                    "t_ppo": _t_ppo,
                    **avg_aux,
                }
            )

            # Per-component reward diagnostics
            if relative_update % config.reward_component_interval == 0 and batched_reward_components is not None:
                try:
                    _comp = batched_reward_components(states, act_t[-1])
                    _comp_means = {k: float(jnp.mean(v)) for k, v in _comp.items()}
                    _comp_means["update"] = update
                    reward_component_history.append(_comp_means)
                except Exception:
                    _logger.debug("Reward component diagnostics failed at update %d", update, exc_info=True)

            # ---------- Stability watchdog ----------
            _kl = avg_aux.get("approx_kl", 0.0)
            should_halt, _is_unstable, _stab_msg = _stability.check(_kl, avg_grad_norm, avg_loss, update)
            if _stab_msg:
                print(f"  {_stab_msg}")
            if should_halt:
                break

            # Track best model
            _track_metric: float = mean_ep_return if not np.isnan(mean_ep_return) else avg_reward
            if _track_metric > best_reward and not _is_unstable:
                best_reward = _track_metric
                best_params = jax.device_get(params)
                best_update = update

            # CSV log
            elapsed = time.time() - t_start
            steps_done = (update - _start_update + 1) * ROLLOUT_LEN * NUM_ENVS
            sps = steps_done / elapsed
            _csv_logger.log(
                {
                    "update": update,
                    "reward_per_step": f"{avg_reward:.4f}",
                    "episode_return": f"{mean_ep_return:.2f}" if not np.isnan(mean_ep_return) else "",
                    "episode_length": f"{mean_ep_length:.1f}" if not np.isnan(mean_ep_length) else "",
                    "total_loss": f"{avg_loss:.4f}",
                    "policy_loss": f"{avg_aux['policy_loss']:.4f}",
                    "value_loss": f"{avg_aux['value_loss']:.4f}",
                    "entropy": f"{avg_aux['entropy']:.4f}",
                    "approx_kl": f"{avg_aux['approx_kl']:.6f}",
                    "clip_fraction": f"{avg_aux['clip_fraction']:.4f}",
                    "grad_norm": f"{avg_grad_norm:.4f}",
                    "mean_std": f"{avg_aux['mean_std']:.4f}",
                    "steps": steps_done,
                    "sps": f"{sps:.0f}",
                    "fall_rate": f"{fall_rate:.4f}",
                    "learning_rate": f"{current_lr:.2e}",
                    "elapsed": f"{elapsed:.1f}",
                    "t_rollout": f"{_t_rollout:.3f}",
                    "t_ppo": f"{_t_ppo:.3f}",
                }
            )

            # Console logging
            if _log_interval is not None and (
                relative_update % _log_interval == 0 or update == _start_update + config.num_updates - 1
            ):
                updates_done = update - _start_update + 1
                updates_left = config.num_updates - updates_done
                eta = (elapsed / updates_done) * updates_left if updates_done > 0 else 0
                eta_str = f"{eta / 60:.0f}m" if eta > 60 else f"{eta:.0f}s"
                ep_ret_str = f"ep_ret={mean_ep_return:+.1f}" if not np.isnan(mean_ep_return) else "ep_ret=n/a"
                print(
                    f"[{update:4d}/{_start_update + config.num_updates}]  "
                    f"r/step={avg_reward:+.3f}  {ep_ret_str}  "
                    f"loss={avg_loss:.4f}  "
                    f"pi={avg_aux['policy_loss']:.3f}  v={avg_aux['value_loss']:.3f}  "
                    f"ent={avg_aux['entropy']:.3f}  kl={avg_aux['approx_kl']:.4f}  "
                    f"grad={avg_grad_norm:.3f}  falls={fall_rate:.1%}  "
                    f"SPS={sps:,.0f}  ETA={eta_str}  "
                    f"[{_t_rollout:.1f}s+{_t_ppo:.2f}s]"
                )

            # Callback
            if callback is not None:
                callback(update, diagnostics_history[-1])

            # Periodic checkpointing
            if (relative_update + 1) % CHECKPOINT_FREQ == 0:
                _ckpt_mgr.save(
                    params,
                    update + 1,
                    obs_rms=obs_rms,
                    history={
                        "reward": reward_history,
                        "loss": loss_history,
                        "episode_return": episode_return_history,
                    },
                )
                if _log_interval is not None:
                    print(f"  >>> Checkpoint saved: {_ckpt_mgr.latest}")

    finally:
        _csv_logger.close()

    # ==================== Training Summary ====================
    elapsed = time.time() - t_start
    actual_updates = update - _start_update + 1
    total_steps = actual_updates * ROLLOUT_LEN * NUM_ENVS
    print("=" * 70)
    print(f"Done! {total_steps:,} steps in {format_duration(elapsed)} ({total_steps / elapsed:,.0f} SPS)")
    print(f"Best metric: {best_reward:+.4f} at update {best_update}")

    _pct_rollout = 100 * _cum_t_rollout / elapsed if elapsed > 0 else 0
    _pct_ppo = 100 * _cum_t_ppo / elapsed if elapsed > 0 else 0
    _pct_other = 100 - _pct_rollout - _pct_ppo
    print("\nTiming breakdown:")
    print(f"  Rollout (MJX physics): {_cum_t_rollout:7.1f}s  ({_pct_rollout:4.1f}%)")
    print(f"  PPO updates:           {_cum_t_ppo:7.1f}s  ({_pct_ppo:4.1f}%)")
    print(f"  Other (GAE/IO/log):    {elapsed - _cum_t_rollout - _cum_t_ppo:7.1f}s  ({_pct_other:4.1f}%)")

    _physics_steps = actual_updates * ROLLOUT_LEN * NUM_ENVS * config.frame_skip
    if _cum_t_rollout > 0:
        print(
            f"\nMJX physics: {_physics_steps:,} steps in {_cum_t_rollout:.1f}s "
            f"({_physics_steps / _cum_t_rollout:,.0f} physics steps/sec)"
        )

    if len(reward_history) >= 10:
        _first10 = np.mean(reward_history[:10])
        _last10 = np.mean(reward_history[-10:])
        _trend = "improved" if _last10 > _first10 else "declined"
        print(f"\nReward trend: {_first10:.3f} (first 10) -> {_last10:.3f} (last 10)  [{_trend}]")

    if _stability.total_warnings > 0:
        print(f"\nTotal stability warnings: {_stability.total_warnings}")

    # Save final parameters
    params_path = config.model_dir / "params.pkl"
    save_checkpoint(
        params_path,
        params,
        obs_rms=obs_rms,
        extra={
            "best_params": best_params,
            "best_reward": best_reward,
            "best_update": best_update,
        },
        history={
            "reward": reward_history,
            "loss": loss_history,
            "episode_return": episode_return_history,
            "diagnostics": diagnostics_history,
        },
    )
    print(f"\nParameters saved to: {params_path}")
    print(f"Training log CSV: {csv_path}")

    return TrainResult(
        params=params,
        opt_state=opt_state,
        obs_rms=obs_rms,
        best_params=best_params,
        best_reward=best_reward,
        best_update=best_update,
        reward_history=reward_history,
        loss_history=loss_history,
        episode_return_history=episode_return_history,
        diagnostics_history=diagnostics_history,
        reward_component_history=reward_component_history,
        total_steps=total_steps,
        elapsed=elapsed,
        actual_updates=actual_updates,
        csv_path=csv_path,
    )


# ===========================================================================
# Library API — JaxTrainer class with hook-based architecture
# ===========================================================================
# Used by jax_training.py (CLI), jax_hooks.py, and tests.
#
# ``TrainerState``, ``TrainingHook``, and ``StopTraining`` are defined in
# :mod:`environments.shared.jax_trainer_types` and re-exported at the top
# of this module for backward compatibility.


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

        self._collect_rollout: Callable | None = None
        self._scan_ppo_update: Callable | None = None

    def _dispatch(self, method: str, *args: Any) -> None:
        for hook in self.hooks:
            fn = getattr(hook, method, None)
            if fn is not None:
                fn(*args)

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
                raw_action, log_prob, value = jax.vmap(
                    sample_action,
                    in_axes=(None, None, 0, 0),
                )(params, network, obs, jax.random.split(action_rng, num_envs))

                # Clip for env; store raw action for PPO ratio consistency
                action = jnp.clip(raw_action, -1.0, 1.0)
                rng, step_rng = jax.random.split(rng)
                new_states, rewards, terminated, truncated = env.step(states, action, step_rng)
                dones = (terminated | truncated).astype(jnp.float32)

                return (new_states, rng), (states.obs, raw_action, log_prob, value, rewards, dones)

            return jax.lax.scan(step_fn, (states, rng), None, length=rollout_len)

        return collect_rollout

    def _build_scan_ppo_update(self):
        """Build the JIT-compiled PPO updater with minibatch shuffling and KL early stopping."""
        import jax
        import jax.numpy as jnp

        from .jax_ppo import ppo_update

        optimizer = self.optimizer
        network = self.network
        ppo_config = self.ppo_config
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
    ) -> tuple[Any, dict[str, float], TrainerState]:
        """Run the training loop.

        Args:
            num_updates: Number of PPO update iterations.
            seed: Random seed.
            init_params: Optional initial network parameters.

        Returns:
            ``(params, eval_metrics, state)`` tuple.
        """
        check_jax()

        import jax
        import jax.numpy as jnp

        from .jax_normalization import RunningMeanStd, normalize_obs, update_running_stats
        from .jax_ppo import compute_gae, sample_action

        _logger.info("num_envs=%d rollout_len=%d num_updates=%d", self.num_envs, self.rollout_len, num_updates)

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        dummy_obs = jnp.zeros(
            self.env.mj_model.nq - 7 + self.env.mj_model.nv - 6 + 17,
        )
        params = self.network.init(init_rng, dummy_obs) if init_params is None else init_params
        opt_state = self.optimizer.init(params)

        obs_dim = dummy_obs.shape[0]
        obs_stats = RunningMeanStd.create(obs_dim)

        rng, reset_rng = jax.random.split(rng)
        states = self.env.reset(reset_rng)

        self._collect_rollout = self._build_collect_rollout()
        self._scan_ppo_update = self._build_scan_ppo_update()

        # Jitted bootstrap sampler — avoids re-tracing a lambda each update
        _network = self.network
        _num_envs = self.num_envs

        @jax.jit
        def _batched_bootstrap(params, obs_batch, rng):
            rngs = jax.random.split(rng, _num_envs)
            _, _, values = jax.vmap(
                lambda o, r: sample_action(params, _network, o, r),
                in_axes=(0, 0),
            )(obs_batch, rngs)
            return values

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
            for update in range(num_updates):
                state.update = update

                t_rollout_start = time.time()
                rng, collect_rng = jax.random.split(state.rng)
                state.rng = rng
                (states, _), rollout_data = self._collect_rollout(
                    state.env_states,
                    collect_rng,
                    state.params,
                    state.obs_stats,
                )
                state.env_states = states
                rollout_obs, rollout_actions, rollout_log_probs, rollout_values, rollout_rewards, rollout_dones = (
                    rollout_data
                )

                jax.block_until_ready(rollout_dones)
                t_rollout = time.time() - t_rollout_start
                state.t_rollout_cumulative += t_rollout
                state.total_steps += self.num_envs * self.rollout_len

                state.obs_stats = update_running_stats(
                    state.obs_stats,
                    rollout_obs.reshape(-1, obs_dim),
                )

                mean_reward = float(jnp.mean(rollout_rewards))
                elapsed = time.time() - t0
                fps = state.total_steps / elapsed if elapsed > 0 else 0

                # Compute fall rate on-device; only transfer scalars
                fall_rate = float(jnp.sum(rollout_dones)) / (self.rollout_len * self.num_envs)
                # Defer full GPU→CPU transfer for episode tracking
                rew_np = np.array(rollout_rewards)
                done_np = np.array(rollout_dones)
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

                rollout_obs_norm = normalize_obs(rollout_obs.reshape(-1, obs_dim), state.obs_stats)

                rng, bootstrap_rng = jax.random.split(state.rng)
                state.rng = rng
                final_obs = normalize_obs(state.env_states.obs, state.obs_stats)
                bootstrap_value = _batched_bootstrap(state.params, final_obs, bootstrap_rng)

                rollout_values_arr = jnp.concatenate([rollout_values, bootstrap_value[None]], axis=0)
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

                t_ppo_start = time.time()
                rng, ppo_rng = jax.random.split(state.rng)
                state.rng = rng
                state.params, state.opt_state, ppo_info = self._scan_ppo_update(
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

        eval_metrics = {
            "mean_reward": float(jnp.mean(jnp.array([h["mean_reward"] for h in state.history[-10:]])))
            if state.history
            else 0.0,
            "total_steps": state.total_steps,
            "elapsed": elapsed,
            "t_rollout_cumulative": state.t_rollout_cumulative,
            "t_ppo_cumulative": state.t_ppo_cumulative,
        }

        return state.params, eval_metrics, state
