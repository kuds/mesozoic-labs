"""Config-driven PPO training function for JAX/MJX environments (notebook path).

The module-level :func:`train` -- with :class:`TrainConfig`,
:class:`TrainResult` and the :func:`_build_jit_fns` JIT builder it runs on
-- is the direct, config-driven training loop with built-in logging,
checkpointing, warmup/ramp, and minibatch shuffling used by
``notebooks/jax_training.ipynb``.

Extracted from :mod:`environments.shared.jax_trainer`, which keeps the
hook-based library loop (:class:`~environments.shared.jax_trainer.JaxTrainer`,
the CLI path) and re-exports the four names defined here, so
``from environments.shared.jax_trainer import TrainConfig, train`` keeps
working.  The helpers both loops share -- the per-step truncation bootstrap
and the batched value head -- live in
:mod:`environments.shared.jax_trainer_types`, so neither loop module has to
import the other.

Usage::

    from environments.shared.jax_train_fn import TrainConfig, TrainResult, train
    result = train(config, env, network, params, opt_state, obs_rms, reward_cfg, rng, optimizer=optimizer)
"""

from __future__ import annotations

import inspect
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .jax_trainer_types import _bootstrap_truncations, _build_batched_value

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
    stage: "int | str" = 1
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

    # Env steps for the WHOLE stage (a same-stage resume counts the
    # checkpoint's prior updates); ``session_steps`` is this call's segment.
    total_steps: int = 0
    session_steps: int = 0
    elapsed: float = 0.0
    actual_updates: int = 0
    csv_path: Path | None = None


# ---------------------------------------------------------------------------
# JIT function builders
# ---------------------------------------------------------------------------


def _detail_fn_accepts_action_lags(fn: Callable) -> bool:
    """Whether *fn* takes ``prev_action`` / ``prev_prev_action`` keywords."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return True
    return "prev_action" in params and "prev_prev_action" in params


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
    VF_CLIP_RANGE = config.vf_clip_range

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
                    params,
                    opt_state,
                    obs,
                    act,
                    lp,
                    adv,
                    ret,
                    old_values=val,
                    clip_range=clip_range,
                    ent_coef=ent_coef,
                    # Without this the scan path silently ran with value
                    # clipping off while the TOML (and the printed config)
                    # said it was on.
                    vf_clip_range=VF_CLIP_RANGE,
                )

                approx_kl = aux["approx_kl"]
                # Compare in Python (TARGET_KL is closure-captured) so we
                # never evaluate ``traced > None`` when target_kl is disabled.
                if TARGET_KL is not None:
                    should_skip = kl_exceeded | (approx_kl > TARGET_KL)
                else:
                    should_skip = kl_exceeded

                # On KL early-stop we revert BOTH params and the full
                # opt_state — that includes Optax counters/moments and any
                # LR-schedule step counter, so the schedule effectively
                # rewinds to its pre-update value.  This is intentional
                # (a skipped update consumes no schedule step), but it is
                # subtle.  Under jit every leaf is a traced array, so the
                # ``hasattr(new, "shape")`` check is always True and the
                # ``else`` branch is dead — kept defensively for any
                # future non-array opt-state field.
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
        def collect_rollout(states, rng, params, obs_rms, forward_vel_scale):
            """Collect a rollout under the current obs stats.

            Stores RAW (un-normalized) observations and the pre-reset
            ``final_obs`` for each step.  The caller normalizes once
            with the same stats — that keeps the PPO importance ratio
            consistent and lets running stats track raw obs only.

            ``forward_vel_scale`` is a scalar JAX array routed through to
            the env so the reward ramp can take effect without retracing.
            """

            def step_fn(carry, _):
                states, rng, _ = carry
                rng, action_rng, step_rng = jax.random.split(rng, 3)

                raw_obs = states.obs
                obs_normed = normalize_obs(raw_obs, obs_rms)
                rngs = jax.random.split(action_rng, NUM_ENVS)
                raw_actions, log_probs, values = jax.vmap(_sample_action, in_axes=(None, 0, 0))(
                    params, obs_normed, rngs
                )

                # The action lags the kernel scores THIS step against; its
                # post-step state overwrites them, so keep the last step's
                # for the reward-component panel.
                pre_step_lags = (states.prev_action, states.prev_prev_action)
                actions = jnp.clip(raw_actions, -1.0, 1.0)
                new_states, rewards, terminated, truncated, final_obs = env.step(
                    states, actions, step_rng, return_final_obs=True, forward_vel_scale=forward_vel_scale
                )
                full_done = terminated | truncated
                gae_done = terminated

                return (new_states, rng, pre_step_lags), (
                    raw_obs,
                    raw_actions,
                    log_probs,
                    values,
                    rewards,
                    gae_done.astype(jnp.float32),
                    full_done.astype(jnp.float32),
                    final_obs,
                )

            init_lags = (states.prev_action, states.prev_prev_action)
            (states, _, last_step_lags), rollout = jax.lax.scan(
                step_fn, (states, rng, init_lags), None, length=ROLLOUT_LEN
            )
            return states, rollout, last_step_lags

    # --- Reward component diagnostics ---

    # Scores ``(states.data, action)`` per env with the action lags the
    # kernel held for that step, so the panel's smoothness / action_jerk
    # rows equal the terms the kernel charged instead of a zero-lag
    # stand-in.  A detail fn without the keywords still gets the zero-lag
    # rows, loudly.
    batched_reward_components = None
    if reward_detail_fn is not None:
        if _detail_fn_accepts_action_lags(reward_detail_fn):

            def _detail_with_lags(data, action, prev_action, prev_prev_action):
                return reward_detail_fn(data, action, prev_action=prev_action, prev_prev_action=prev_prev_action)

            @jax.jit
            def batched_reward_components(states, action_batch, prev_action, prev_prev_action):
                return jax.vmap(_detail_with_lags)(states.data, action_batch, prev_action, prev_prev_action)

        else:
            _logger.warning(
                "reward_detail_fn %r does not accept prev_action / prev_prev_action; the component panel's "
                "smoothness and action_jerk rows are computed against zero lags (accept **step_kwargs to fix)",
                getattr(reward_detail_fn, "__name__", reward_detail_fn),
            )

            @jax.jit
            def batched_reward_components(states, action_batch, prev_action, prev_prev_action):
                return jax.vmap(reward_detail_fn, in_axes=(0, 0))(states.data, action_batch)

    return {
        "batched_sample": batched_sample,
        "batched_value": _build_batched_value(network),
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
            Signature: ``(mjx_data, action, *, prev_action, prev_prev_action)
            -> dict[str, scalar]`` -- the two action lags the kernel scored
            the step against are forwarded when the function accepts them
            (``**step_kwargs`` will do), so the panel's smoothness and jerk
            rows equal the charged terms.
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
    from .plant_contract import current_plant_identity, validate_mjx_environment_plant, write_plant_identity
    from .reporting import format_duration

    assert optimizer is not None, "optimizer is required"

    # Build JIT functions
    jit_fns = _build_jit_fns(config, network, optimizer, reward_detail_fn, env=env)
    batched_value = jit_fns["batched_value"]
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
    plant_species = config.species or getattr(getattr(env, "config", None), "species", "")
    plant_identity = current_plant_identity(plant_species) if plant_species else None
    if plant_identity is not None:
        runtime_model = getattr(env, "mj_model", None)
        if runtime_model is None:
            raise ValueError("JAX training environment has no mj_model to bind to the plant identity")
        validate_mjx_environment_plant(env, plant_identity, artifact="JAX training environment")
    _ckpt_mgr = CheckpointManager(
        config.model_dir,
        prefix="checkpoint",
        max_keep=config.max_checkpoints,
        plant_identity=plant_identity,
    )
    if plant_identity is not None:
        write_plant_identity(config.model_dir / "plant_identity.json", plant_identity)
    _stability = StabilityMonitor()
    # Append on resume (start_update > 0) so the prior stage-run's rows
    # aren't truncated away.
    _csv_logger = TrainingCSVLogger(config.output_dir / "training_log.csv", append=_start_update > 0)
    csv_path = _csv_logger.path

    # Warmup/ramp state
    _warmup_active = config.warmup_updates > 0
    _ramp_active = config.ramp_updates > 0
    _ramp_target_value = reward_cfg.get(config.ramp_attr, 0.0) if _ramp_active else 0.0
    if _ramp_active and config.ramp_attr != "forward_vel_weight":
        # The MJX env captures most weights as Python constants at trace
        # time, so they cannot be ramped without recompilation.  Only
        # ``forward_vel_weight`` is wired through ``env.step`` as a
        # runtime scale.  Fail loud rather than silently doing nothing.
        raise ValueError(
            f"Reward ramp on attr={config.ramp_attr!r} is not supported by the MJX path; "
            "only 'forward_vel_weight' is dynamic at runtime."
        )

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
    # Warmup and ramp are stage-absolute (updates 0..N-1 of the stage): a
    # same-stage resume past them must not re-warm or restart the ramp.
    if _warmup_active:
        if _start_update < config.warmup_updates:
            print(
                f"Warmup: updates {_start_update}..{config.warmup_updates - 1} "
                f"(clip_range={config.warmup_clip_range}, ent_coef={config.warmup_ent_coef})"
            )
        else:
            print(f"Warmup: updates 0..{config.warmup_updates - 1} already complete at update {_start_update}")
    if _ramp_active:
        if _start_update < config.ramp_updates:
            print(
                f"Reward ramp: {config.ramp_attr} from "
                f"{_ramp_target_value * config.ramp_start_fraction:.4f} to "
                f"{_ramp_target_value:.4f} over updates 0..{config.ramp_updates - 1}"
                + (f" (resuming at {_start_update})" if _start_update else "")
            )
        else:
            print(f"Reward ramp: updates 0..{config.ramp_updates - 1} already complete at update {_start_update}")
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
            # Keyed off the ABSOLUTE update, like JaxTrainer: a same-stage
            # resume continues past the warmup instead of re-running it (a
            # cross-stage init starts at 0, so its warmup is unchanged).
            if _warmup_active and update < config.warmup_updates:
                _active_clip_range = config.warmup_clip_range
                _active_ent_coef = config.warmup_ent_coef
            else:
                _active_clip_range = config.clip_range
                _active_ent_coef = config.ent_coef
                if _warmup_active and update == config.warmup_updates and _log_interval is not None:
                    print(
                        f"  >>> Warmup complete at update {update}: "
                        f"restoring clip_range={config.clip_range}, ent_coef={config.ent_coef}"
                    )

            # ---------- Reward ramp ----------
            # The ramp scales forward_vel_weight in [start_fraction, 1.0]
            # and is passed through to env.step at runtime so the JIT
            # trace doesn't need to be invalidated.  Absolute, as above.
            if _ramp_active and update < config.ramp_updates:
                ramp_progress = update / config.ramp_updates
                forward_vel_scale = config.ramp_start_fraction + (1.0 - config.ramp_start_fraction) * ramp_progress
                # Mirror the effective weight into reward_cfg so callers
                # inspecting it (e.g. diagnostics) see the current value.
                reward_cfg[config.ramp_attr] = _ramp_target_value * forward_vel_scale
            else:
                forward_vel_scale = 1.0
                if _ramp_active:
                    reward_cfg[config.ramp_attr] = _ramp_target_value
            forward_vel_scale_arr = jnp.float32(forward_vel_scale)

            # ---------- Collect rollout ----------
            _t_phase = time.time()
            rng, rng_collect = jax.random.split(rng)
            # Snapshot the obs stats used during this rollout — PPO must
            # see the same normalization, otherwise the importance ratio
            # is biased.  We update obs_rms only after consuming the batch.
            rollout_obs_rms = obs_rms
            states, rollout, last_step_lags = collect_rollout(
                states, rng_collect, params, rollout_obs_rms, forward_vel_scale_arr
            )
            obs_t, act_t, lp_t, val_t, rew_t, gae_done_t, full_done_t, final_obs_t = rollout

            jax.block_until_ready(full_done_t)
            _t_rollout = time.time() - _t_phase
            _cum_t_rollout += _t_rollout

            # ---------- Episode stats ----------
            # Compute fall rate on-device; only transfer scalars.
            # gae_done marks natural termination only -- counting full_done
            # would book every time-limit truncation as a "fall".
            fall_rate = float(jnp.sum(gae_done_t)) / (ROLLOUT_LEN * NUM_ENVS)
            # Defer full GPU→CPU transfer for episode tracking
            rew_np = np.array(rew_t)
            full_done_np = np.array(full_done_t)
            _completed_returns, _completed_lengths = compute_episode_stats(rew_np, full_done_np, _ep_stats_acc)

            # ---------- Bootstrap values for GAE ----------
            # Value every step's pre-auto-reset ``final_obs`` under the
            # rollout policy and rollout-time stats.  The last entry is the
            # rollout's tail bootstrap (V(s_T), or V(final_obs) if the last
            # step truncated); every other entry bootstraps a MID-rollout
            # truncation via _bootstrap_truncations, which folds it into
            # that step's reward and masks the step as done.  Natural
            # terminations are masked out by the GAE done flag as before.
            final_obs_normed = normalize_obs(final_obs_t.reshape(-1, OBS_DIM), rollout_obs_rms)
            final_values = batched_value(params, final_obs_normed).reshape(ROLLOUT_LEN, NUM_ENVS)
            val_t_plus1 = jnp.concatenate([val_t, final_values[-1][None]], axis=0)

            # ---------- Compute advantages ----------
            rew_boot_t, gae_mask_t = _bootstrap_truncations(rew_t, gae_done_t, full_done_t, final_values, GAMMA)
            advantages, returns = compute_gae(rew_boot_t, val_t_plus1, gae_mask_t, GAMMA, GAE_LAMBDA)

            # Normalize PPO inputs with the rollout-time stats so the new
            # policy sees the same scaled inputs as the behaviour policy.
            flat_obs_raw = obs_t.reshape(-1, OBS_DIM)
            flat_obs = normalize_obs(flat_obs_raw, rollout_obs_rms)
            flat_act = act_t.reshape(-1, ACT_DIM)
            flat_lp = lp_t.reshape(-1)
            flat_adv = advantages.reshape(-1)
            flat_ret = returns.reshape(-1)
            flat_val = val_t.reshape(-1)

            # ---------- Update obs normalisation (from RAW obs) ----------
            # Done after the rollout has been consumed so PPO sees stats
            # that match what the behaviour policy used.
            obs_rms = update_running_stats(obs_rms, flat_obs_raw)

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

            # Compute current learning rate for logging, from the ABSOLUTE
            # position on the stage schedule: on a same-stage resume
            # config.num_updates is the remaining budget while the restored
            # optax count continues, so relative progress would log the LR
            # snapping back to learning_rate.
            if config.learning_rate_end is not None and config.learning_rate_end != config.learning_rate:
                total_lr_steps = (_start_update + config.num_updates) * config.ppo_epochs
                lr_step = min(update * config.ppo_epochs, total_lr_steps)
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
                    # Re-score the rollout's last step as the kernel did: the
                    # command it scored is the post-step prev_action carry
                    # (clipped, low-passed), against the lags it held BEFORE
                    # that step.
                    _comp = batched_reward_components(states, states.prev_action, *last_step_lags)
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
            session_steps_done = (update - _start_update + 1) * ROLLOUT_LEN * NUM_ENVS
            sps = session_steps_done / elapsed
            # Stage-cumulative, so a resumed session's rows continue the
            # interrupted session's count.
            steps_done = (update + 1) * ROLLOUT_LEN * NUM_ENVS
            _csv_logger.log(
                {
                    "update": update,
                    "reward_per_step": f"{avg_reward:.4f}",
                    "episode_return": f"{mean_ep_return:.2f}" if not np.isnan(mean_ep_return) else "",
                    "episode_length": f"{mean_ep_length:.1f}" if not np.isnan(mean_ep_length) else "",
                    "total_loss": f"{avg_loss:.4f}",
                    "policy_loss": f"{avg_aux['policy_loss']:.4f}",
                    "value_loss": f"{avg_aux['value_loss']:.4f}",
                    "explained_variance": f"{avg_aux['explained_variance']:.4f}",
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
                    f"ev={avg_aux['explained_variance']:+.2f}  "
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
                    opt_state=jax.device_get(opt_state),
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
    session_steps = actual_updates * ROLLOUT_LEN * NUM_ENVS
    # The stage record counts the WHOLE stage: on a same-stage resume the
    # checkpoint's prior updates are part of it.
    total_steps = (update + 1) * ROLLOUT_LEN * NUM_ENVS
    print("=" * 70)
    print(
        f"Done! {session_steps:,} steps in {format_duration(elapsed)} ({session_steps / elapsed:,.0f} SPS)"
        + (f"; {total_steps:,} for the stage" if _start_update else "")
    )
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
        opt_state=jax.device_get(opt_state),
        update=update + 1,
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
        plant_identity=plant_identity,
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
        session_steps=session_steps,
        elapsed=elapsed,
        actual_updates=actual_updates,
        csv_path=csv_path,
    )
