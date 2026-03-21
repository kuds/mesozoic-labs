"""
Shared training infrastructure for all dinosaur species.

Provides the common training, curriculum, and evaluation logic used by every
species.  Species-specific entry points (``environments/<species>/scripts/train_sb3.py``)
define a :class:`SpeciesConfig` and delegate to :func:`main`.

The original monolithic module has been split into focused submodules for
maintainability:

- :mod:`~environments.shared.diagnostics` -- ``DiagnosticsCallback``
- :mod:`~environments.shared.evaluation` -- ``eval_policy``, ``evaluate``,
  ``record_stage_video``
- :mod:`~environments.shared.cli` -- ``main``, ``_apply_overrides``,
  ``_cast_value``

All public names are re-exported here so existing ``from
environments.shared.train_base import ...`` statements continue to work.
"""

from __future__ import annotations

import dataclasses
import logging
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .constants import DEFAULT_CLIP_OBS, DEFAULT_CLIP_REWARD, DEFAULT_NORM_OBS, DEFAULT_NORM_REWARD

logger = logging.getLogger(__name__)

# Suppress noisy tensorboardX NaN/Inf warnings (handled by _sanitize in diagnostics.py)
logging.getLogger("tensorboardX").setLevel(logging.ERROR)

try:
    import numpy as _np
except ImportError:
    _np = None  # type: ignore[assignment]


def _ensure_sb3():
    """Import SB3 or exit with a helpful error."""
    try:
        from stable_baselines3 import PPO, SAC
        from stable_baselines3.common.callbacks import (
            CallbackList,
            CheckpointCallback,
            EvalCallback,
        )
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.utils import set_random_seed
        from stable_baselines3.common.vec_env import (
            DummyVecEnv,
            SubprocVecEnv,
            VecNormalize,
        )

        return {
            "PPO": PPO,
            "SAC": SAC,
            "CallbackList": CallbackList,
            "CheckpointCallback": CheckpointCallback,
            "EvalCallback": EvalCallback,
            "Monitor": Monitor,
            "set_random_seed": set_random_seed,
            "DummyVecEnv": DummyVecEnv,
            "SubprocVecEnv": SubprocVecEnv,
            "VecNormalize": VecNormalize,
        }
    except ImportError:
        logger.error("stable-baselines3 not installed. Install with: pip install stable-baselines3[extra]")
        sys.exit(1)


@dataclasses.dataclass
class SpeciesConfig:
    """Species-specific parameters that differ across training scripts.

    Each species defines one of these in its thin ``train_sb3.py`` wrapper.
    """

    species: str
    """Species identifier used for directory names and config lookup
    (e.g. ``"velociraptor"``, ``"trex"``, ``"brachiosaurus"``)."""

    env_class: type
    """The Gymnasium environment class (e.g. ``RaptorEnv``)."""

    stage_descriptions: str
    """Short stage legend for ``--stage`` argparse help
    (e.g. ``"1=balance, 2=locomotion, 3=strike"``)."""

    height_label: str
    """Label used in evaluation log output (``"Pelvis height"`` or ``"Torso height"``)."""

    stage3_section_label: str
    """Section header for stage-3 eval results (``"Hunting"`` or ``"Food Reaching"``)."""

    success_keys: list
    """Info-dict keys that signal a successful episode
    (e.g. ``["strike_success", "bite_success"]``)."""


# ── Utility helpers ──────────────────────────────────────────────────────


def linear_schedule(initial_lr: float, final_lr: float):
    """Return a callable that linearly decays learning rate."""

    def schedule(progress_remaining: float) -> float:
        return final_lr + progress_remaining * (initial_lr - final_lr)

    return schedule


def cosine_schedule(initial_lr: float, final_lr: float):
    """Return a callable that decays learning rate on a cosine curve.

    Decays faster in mid-training than linear, then flattens near the end.
    This better protects converged policies from late-training destabilisation.
    """
    import math

    def schedule(progress_remaining: float) -> float:
        # progress_remaining goes from 1.0 → 0.0
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * (1.0 - progress_remaining)))
        return final_lr + cosine_decay * (initial_lr - final_lr)

    return schedule


# ── TensorBoard local buffering ───────────────────────────────────────────


def _is_gcs_path(path: str | Path) -> bool:
    """Return True if *path* is on a GCS FUSE mount (``/gcs/...``)."""
    return str(path).startswith("/gcs/")


def _make_local_tb_dir(gcs_tb_path: str | Path) -> Path:
    """Create a local temp directory for TensorBoard event buffering.

    Returns a ``Path`` under ``/tmp`` that mirrors the GCS structure so
    concurrent trials don't collide.
    """
    # Use a stable suffix derived from the GCS path so restarts reuse the dir.
    suffix = str(gcs_tb_path).replace("/", "_")
    local_dir = Path(tempfile.gettempdir()) / "tb_buffer" / suffix
    local_dir.mkdir(parents=True, exist_ok=True)
    return local_dir


def _sync_tb_to_gcs(local_tb_dir: Path, gcs_tb_path: str | Path) -> None:
    """Copy locally-buffered TensorBoard events to the GCS FUSE mount.

    Uses :func:`shutil.copy` (not ``copy2``) because GCS FUSE does not
    support the ``os.utime`` / ``os.chmod`` calls that ``copy2`` makes
    to preserve file metadata, which causes ``OSError`` on FUSE mounts.
    """
    gcs_dest = Path(gcs_tb_path)
    if not local_tb_dir.exists():
        return
    gcs_dest.mkdir(parents=True, exist_ok=True)
    n_copied = 0
    for src_file in local_tb_dir.rglob("*"):
        if src_file.is_file():
            rel = src_file.relative_to(local_tb_dir)
            dest = gcs_dest / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_file, dest)
            n_copied += 1
    logger.info("Synced %d TensorBoard files to %s", n_copied, gcs_dest)
    # Clean up temp dir
    shutil.rmtree(local_tb_dir, ignore_errors=True)


# ── Environment creation ─────────────────────────────────────────────────


def make_env(
    species_cfg: SpeciesConfig,
    stage_configs: dict[int, dict[str, Any]],
    stage: int,
    rank: int,
    seed: int = 0,
):
    """Create a single environment instance."""
    sb3 = _ensure_sb3()

    def _init():
        sb3["set_random_seed"](seed + rank)
        env_kwargs = stage_configs[stage]["env_kwargs"].copy()
        env = species_cfg.env_class(**env_kwargs)
        env = sb3["Monitor"](env)
        env.reset(seed=seed + rank)
        return env

    return _init


def create_vec_env(
    species_cfg: SpeciesConfig,
    stage_configs: dict[int, dict[str, Any]],
    stage: int,
    n_envs: int,
    seed: int = 0,
    use_subproc: bool = False,
):
    """Create vectorized environment with observation/reward normalization."""
    sb3 = _ensure_sb3()

    env_fns = [make_env(species_cfg, stage_configs, stage, i, seed) for i in range(n_envs)]

    if use_subproc and n_envs > 1:
        env = sb3["SubprocVecEnv"](env_fns)
    else:
        env = sb3["DummyVecEnv"](env_fns)

    env = sb3["VecNormalize"](
        env,
        norm_obs=DEFAULT_NORM_OBS,
        norm_reward=DEFAULT_NORM_REWARD,
        clip_obs=DEFAULT_CLIP_OBS,
        clip_reward=DEFAULT_CLIP_REWARD,
    )
    return env


# ── Shared setup helpers (used by both train() and train_curriculum()) ──


def _prepare_alg_kwargs(
    config: dict[str, Any],
    algorithm: str,
    verbose: int,
    log_path: Path,
    use_tensorboard: bool,
) -> tuple[dict[str, Any], Path | None, Path]:
    """Build algorithm kwargs with LR schedule, clip annealing, and TB setup.

    Returns ``(alg_kwargs, local_tb_dir, gcs_tb_path)`` where *local_tb_dir*
    is ``None`` when the output is not on a GCS FUSE mount.
    """
    alg_kwargs = config["sac_kwargs"].copy() if algorithm == "sac" else config["ppo_kwargs"].copy()
    alg_kwargs["verbose"] = verbose

    # TensorBoard buffering
    local_tb_dir = None
    gcs_tb_path = log_path / "tensorboard"
    if use_tensorboard:
        if _is_gcs_path(log_path):
            local_tb_dir = _make_local_tb_dir(gcs_tb_path)
            alg_kwargs["tensorboard_log"] = str(local_tb_dir)
            logger.info(
                "TensorBoard buffering locally at %s (will sync to GCS after training)",
                local_tb_dir,
            )
        else:
            alg_kwargs["tensorboard_log"] = str(gcs_tb_path)
    else:
        logger.info("TensorBoard logging disabled")

    # PPO-specific schedule setup
    if algorithm == "ppo":
        lr_end = alg_kwargs.pop("learning_rate_end", None)
        lr_schedule_type = alg_kwargs.pop("lr_schedule", "linear")
        if lr_end is not None:
            lr_start = alg_kwargs["learning_rate"]
            if lr_schedule_type == "cosine":
                alg_kwargs["learning_rate"] = cosine_schedule(lr_start, lr_end)
            else:
                alg_kwargs["learning_rate"] = linear_schedule(lr_start, lr_end)
            logger.info("Using %s LR schedule: %s -> %s", lr_schedule_type, lr_start, lr_end)

        clip_range_end = alg_kwargs.pop("clip_range_end", None)
        if clip_range_end is not None:
            clip_start = alg_kwargs["clip_range"]
            alg_kwargs["clip_range"] = linear_schedule(clip_start, clip_range_end)
            logger.info("Using clip_range schedule: %s -> %s", clip_start, clip_range_end)

    return alg_kwargs, local_tb_dir, gcs_tb_path


def _load_vecnorm_into_envs(
    load_path: str | None,
    train_env,
    eval_env,
) -> None:
    """Carry forward VecNormalize stats from a prior stage or reset eval env."""
    from .curriculum import load_vecnorm_stats

    if load_path:
        _base = load_path[:-4] if load_path.endswith(".zip") else load_path
        _vecnorm_path = _base + "_vecnorm.pkl"
        if not load_vecnorm_stats(_vecnorm_path, train_env, eval_env):
            logger.warning("VecNormalize file not found: %s — eval env will use defaults", _vecnorm_path)
            eval_env.training = False
            eval_env.norm_reward = False
    else:
        eval_env.training = False
        eval_env.norm_reward = False


def _create_or_load_model(
    sb3: dict,
    algorithm: str,
    alg_kwargs: dict[str, Any],
    train_env,
    load_path: str | None = None,
) -> Any:
    """Create a new model or load from checkpoint.

    Pops ``policy_kwargs`` from *alg_kwargs* (mutating it) so that network
    architecture is only applied to new models, not loaded ones.
    """
    alg_cls = sb3["SAC"] if algorithm == "sac" else sb3["PPO"]
    policy_kwargs = alg_kwargs.pop("policy_kwargs", None)

    if load_path:
        logger.info("Loading model from: %s", load_path)
        model = alg_cls.load(load_path, env=train_env, **alg_kwargs)
    else:
        logger.info("Creating new %s model...", algorithm.upper())
        model = alg_cls("MlpPolicy", train_env, policy_kwargs=policy_kwargs, **alg_kwargs)

    return model


def _build_core_callbacks(
    sb3: dict,
    eval_env,
    model_dir: Path,
    log_path: Path,
    stage: int,
    n_envs: int,
    eval_freq: int,
    save_freq: int,
    verbose: int,
    use_wandb: bool = False,
) -> tuple[list, Any, Any]:
    """Build the standard callback set shared by train() and train_curriculum().

    Returns ``(callbacks, eval_callback, save_vecnorm_cb)`` so callers can
    append additional stage-specific callbacks.
    """
    from .curriculum import (
        EvalCollapseEarlyStopCallback,
        SaveVecNormalizeCallback,
    )
    from .diagnostics import DiagnosticsCallback as _DiagCB
    from .wandb_integration import WandbCallback

    callbacks = []

    save_vecnorm_cb = SaveVecNormalizeCallback(
        save_path=str(model_dir / "best_model_vecnorm.pkl"),
    )

    eval_callback = sb3["EvalCallback"](
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(log_path),
        eval_freq=eval_freq // n_envs,
        n_eval_episodes=30,
        deterministic=True,
        render=False,
        verbose=max(verbose, 1),
        callback_on_new_best=save_vecnorm_cb,
    )
    callbacks.append(eval_callback)

    checkpoint_callback = sb3["CheckpointCallback"](
        save_freq=save_freq // n_envs,
        save_path=str(model_dir),
        name_prefix=f"stage{stage}",
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)

    callbacks.append(_DiagCB(log_dir=str(log_path), verbose=verbose))

    callbacks.append(
        EvalCollapseEarlyStopCallback(eval_callback=eval_callback, min_evals=8, patience=5, verbose=verbose)
    )

    if use_wandb:
        callbacks.append(WandbCallback())

    return callbacks, eval_callback, save_vecnorm_cb


def _save_final_and_sync_tb(
    model,
    train_env,
    model_dir: Path,
    stage: int,
    local_tb_dir: Path | None,
    gcs_tb_path: Path,
) -> Path:
    """Save the final model checkpoint and sync TensorBoard events to GCS.

    Returns the final model path (without ``.zip`` extension).
    """
    final_path = model_dir / f"stage{stage}_final"
    model.save(str(final_path))
    train_env.save(str(final_path) + "_vecnorm.pkl")

    if local_tb_dir is not None:
        try:
            _sync_tb_to_gcs(local_tb_dir, gcs_tb_path)
        except Exception:
            logger.warning("TensorBoard sync to GCS failed.", exc_info=True)

    return final_path


# ── Single-stage training ────────────────────────────────────────────────


def train(
    species_cfg: SpeciesConfig,
    stage_configs: dict[int, dict[str, Any]],
    stage: int,
    total_timesteps: int,
    n_envs: int = 4,
    seed: int = 42,
    load_path: str | None = None,
    eval_freq: int = 50000,
    save_freq: int = 50000,
    log_dir: str | None = None,
    use_subproc: bool = False,
    verbose: int = 1,
    algorithm: str = "ppo",
    use_wandb: bool = False,
    output_dir: str | None = None,
    use_tensorboard: bool = True,
):
    """Train a single stage of the curriculum."""
    from .config import save_stage_config
    from .curriculum import (
        RewardRampCallback,
        StageWarmupCallback,
    )
    from .wandb_integration import init_wandb

    sb3 = _ensure_sb3()

    config = stage_configs[stage]
    species = species_cfg.species

    logger.info("=" * 60)
    logger.info("Training Stage %d: %s", stage, config["name"])
    logger.info("Description: %s", config["description"])
    logger.info("=" * 60)

    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path: Path
    if output_dir is not None:
        log_path = Path(output_dir)
    elif log_dir is None:
        log_path = Path(__file__).parent.parent / species / "logs" / species / f"stage{stage}_{timestamp}"
    else:
        log_path = Path(log_dir)

    log_path.mkdir(parents=True, exist_ok=True)
    model_dir = log_path / "models"
    model_dir.mkdir(exist_ok=True)

    logger.info("Log directory: %s", log_path)
    logger.info("Model directory: %s", model_dir)

    save_stage_config(
        log_path,
        stage,
        config,
        algorithm.upper(),
        extra={"seed": seed, "n_envs": n_envs, "timesteps": total_timesteps},
        env_class=species_cfg.env_class,
        species=species_cfg.species,
    )

    # Create environments
    # SAC benefits from SubprocVecEnv: MuJoCo is CPU-bound and SAC's off-policy
    # nature means env collection and gradient updates can overlap better when
    # envs run in separate processes.
    effective_subproc = use_subproc or (algorithm == "sac" and n_envs > 1)
    if effective_subproc and not use_subproc:
        logger.info("Auto-enabling SubprocVecEnv for SAC (use --subproc to make explicit)")
    logger.info("Creating %d training environments...", n_envs)
    train_env = create_vec_env(species_cfg, stage_configs, stage, n_envs, seed, effective_subproc)

    logger.info("Creating evaluation environment...")
    eval_env = create_vec_env(species_cfg, stage_configs, stage, 1, seed + 1000, use_subproc=False)

    _load_vecnorm_into_envs(load_path, train_env, eval_env)

    alg_kwargs, local_tb_dir, gcs_tb_path = _prepare_alg_kwargs(
        config,
        algorithm,
        verbose,
        log_path,
        use_tensorboard,
    )

    wandb_run = None
    if use_wandb:
        wandb_run = init_wandb(species=species, stage=stage, config=config)
        logger.info("W&B run initialized.")

    model = _create_or_load_model(sb3, algorithm, alg_kwargs, train_env, load_path)

    logger.info("Model architecture:")
    logger.info("  Policy: %s", model.policy)
    logger.info("  Learning rate: %s", model.learning_rate)
    logger.info("  Batch size: %s", alg_kwargs.get("batch_size", "N/A"))

    callbacks, eval_callback, _ = _build_core_callbacks(
        sb3,
        eval_env,
        model_dir,
        log_path,
        stage,
        n_envs,
        eval_freq,
        save_freq,
        verbose,
        use_wandb,
    )

    if stage > 1 and load_path:
        cur_kwargs = config.get("curriculum_kwargs", {})
        callbacks.append(
            StageWarmupCallback(
                warmup_timesteps=cur_kwargs.get("warmup_timesteps", 100_000),
                warmup_clip_range=cur_kwargs.get("warmup_clip_range", 0.02),
                warmup_ent_coef=cur_kwargs.get("warmup_ent_coef", 0.02),
                warmup_lr_scale=cur_kwargs.get("warmup_lr_scale", 0.1),
            )
        )
        target_fwd_weight = config["env_kwargs"].get("forward_vel_weight", 1.0)
        callbacks.append(
            RewardRampCallback(
                attr_name="forward_vel_weight",
                start_value=cur_kwargs.get("ramp_start_value", 0.1),
                end_value=target_fwd_weight,
                ramp_timesteps=cur_kwargs.get("ramp_timesteps", 500_000),
            )
        )

    callback_list = sb3["CallbackList"](callbacks)

    # Train
    logger.info("Starting training for %s timesteps...", f"{total_timesteps:,}")
    logger.info("-" * 60)

    train_start = time.monotonic()
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            progress_bar=verbose >= 1,
        )
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
    training_duration = time.monotonic() - train_start

    if wandb_run is not None:
        wandb_run.finish()

    # Report metrics to Vertex AI HPT (no-op when cloudml-hypertune not installed)
    _report_hpt_metrics(
        species_cfg,
        model,
        eval_env,
        eval_callback,
        log_path,
        model_dir,
        stage,
        total_timesteps,
        algorithm,
        training_duration_seconds=training_duration,
        stage_config=config,
    )

    final_path = _save_final_and_sync_tb(
        model,
        train_env,
        model_dir,
        stage,
        local_tb_dir,
        gcs_tb_path,
    )

    train_env.close()
    eval_env.close()

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("Final model: %s.zip", final_path)
    logger.info("VecNormalize stats: %s_vecnorm.pkl", final_path)
    logger.info("=" * 60)

    return model


# ── HPT metric reporting ─────────────────────────────────────────────────


def _report_hpt_metrics(
    species_cfg: SpeciesConfig,
    model,
    eval_env,
    eval_callback,
    log_path: Path,
    model_dir: Path,
    stage: int,
    total_timesteps: int,
    algorithm: str,
    training_duration_seconds: float = 0.0,
    stage_config: dict[str, Any] | None = None,
):
    """Report metrics to Vertex AI Hypertune and write a local JSON sidecar.

    Only ``best_mean_reward`` is declared in the HPT ``metric_spec`` (it
    is the sole optimisation target).  All auxiliary metrics are written
    to ``<log_path>/metrics.json`` so they can be collected from GCS
    after the sweep completes — without polluting the HPT objective.

    For forward velocity and success rate (stages 2+), the **best model**
    checkpoint is loaded with its matched VecNormalize stats so the
    reported metrics reflect the checkpoint that will be handed off to
    the next stage — not the final model which may have regressed.
    """
    import json as _json

    import numpy as _np

    sb3 = _ensure_sb3()

    # Accumulate all metrics for the JSON sidecar.
    aux_metrics: dict[str, float | str] = {
        "best_mean_reward": float(eval_callback.best_mean_reward),
        "training_duration_seconds": round(training_duration_seconds, 1),
    }

    # Report the primary optimisation metric to HPT (if available).
    try:
        import hypertune as _hypertune

        _hpt = _hypertune.HyperTune()
        _hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag="best_mean_reward",
            metric_value=eval_callback.best_mean_reward,
            global_step=total_timesteps,
        )
    except ImportError:
        _hpt = None
    logger.info(
        "HPT metric reported: best_mean_reward=%.4f",
        eval_callback.best_mean_reward,
    )

    eval_npz_path = Path(log_path) / "evaluations.npz"
    if eval_npz_path.exists():
        eval_data = _np.load(str(eval_npz_path))
        eval_rewards = eval_data["results"]
        eval_lengths = eval_data["ep_lengths"]
        mean_rewards_per_eval = eval_rewards.mean(axis=1)

        best_eval_idx = int(mean_rewards_per_eval.argmax())
        best_mean_ep_length = float(eval_lengths[best_eval_idx].mean())
        aux_metrics["best_mean_episode_length"] = best_mean_ep_length
        logger.info(
            "HPT metric reported: best_mean_episode_length=%.1f",
            best_mean_ep_length,
        )

        last_mean_reward = float(mean_rewards_per_eval[-1])
        last_mean_ep_length = float(eval_lengths[-1].mean())
        aux_metrics["last_mean_reward"] = last_mean_reward
        aux_metrics["last_mean_episode_length"] = last_mean_ep_length
        logger.info(
            "HPT metric reported: last_mean_reward=%.4f, last_mean_episode_length=%.1f",
            last_mean_reward,
            last_mean_ep_length,
        )

    # ── Post-training quality evaluation (all stages) ──────────────────
    # Run evaluation rollouts with LocomotionMetrics to collect spinning
    # detection signals, heading stability, and reward component breakdown.
    # These metrics enable model selection beyond raw reward.
    from .curriculum import load_vecnorm_stats

    best_model_zip = model_dir / "best_model.zip"
    best_vecnorm_path = model_dir / "best_model_vecnorm.pkl"
    alg_cls = sb3["SAC"] if algorithm == "sac" else sb3["PPO"]

    if best_model_zip.exists():
        eval_model = alg_cls.load(str(model_dir / "best_model"), env=eval_env)
        if best_vecnorm_path.exists():
            load_vecnorm_stats(str(best_vecnorm_path), eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
        logger.info("HPT eval: using best model + matched VecNormalize")
    else:
        eval_model = model
        eval_env.training = False
        eval_env.norm_reward = False
        logger.warning("HPT eval: best_model.zip not found, falling back to final model")

    # Quality evaluation with full LocomotionMetrics (spinning detection,
    # heading alignment, reward breakdown, etc.)
    from .evaluation import eval_policy_quality

    try:
        quality_metrics = eval_policy_quality(eval_model, eval_env, species_cfg.success_keys, n_episodes=50)
        aux_metrics.update(quality_metrics)
        logger.info(
            "Quality eval complete: %d metrics collected (angular_vel=%.3f, heading_align=%.3f)",
            len(quality_metrics),
            quality_metrics.get("eval_mean_pelvis_angular_velocity", float("nan")),
            quality_metrics.get("eval_mean_heading_alignment", float("nan")),
        )
    except Exception:
        logger.warning("Quality evaluation failed — skipping quality metrics.", exc_info=True)

    # Forward velocity, distance, and success rate evaluation.
    # Run for all stages so mean_distance_traveled is always captured.
    _, _, fwd_vels, success_flags, distances = eval_policy(
        eval_model,
        eval_env,
        species_cfg.success_keys,
        n_episodes=30,
    )
    if fwd_vels:
        mean_fwd = float(_np.mean(fwd_vels))
        std_fwd = float(_np.std(fwd_vels))
        aux_metrics["mean_forward_vel"] = mean_fwd
        aux_metrics["std_forward_vel"] = std_fwd
        # Keep backward-compat alias used by existing sweep analysis.
        aux_metrics["best_mean_forward_vel"] = mean_fwd
        logger.info("HPT metric reported: mean_forward_vel=%.4f (std=%.4f)", mean_fwd, std_fwd)
    if distances:
        mean_dist = float(_np.mean(distances))
        aux_metrics["mean_distance_traveled"] = mean_dist
        logger.info("HPT metric reported: mean_distance_traveled=%.4f", mean_dist)
    if success_flags:
        mean_success = float(_np.mean(success_flags))
        aux_metrics["mean_success_rate"] = mean_success
        # Keep backward-compat alias used by existing sweep analysis.
        aux_metrics["best_mean_success_rate"] = mean_success
        logger.info("HPT metric reported: mean_success_rate=%.4f", mean_success)

    # Include key hyperparameters in the sidecar so offline result
    # collection works even when stage_config.json is missing.
    if stage_config is not None:
        algo_key = "sac_kwargs" if algorithm == "sac" else "ppo_kwargs"
        algo_kwargs = stage_config.get(algo_key, {})
        _metric_keys = (
            ("learning_rate", "batch_size", "gamma", "n_steps", "ent_coef")
            if algorithm == "ppo"
            else ("learning_rate", "batch_size", "gamma", "tau", "buffer_size", "ent_coef")
        )
        for k in _metric_keys:
            if k in algo_kwargs:
                val = algo_kwargs[k]
                # Skip callable schedules — store the initial value description
                if not callable(val):
                    aux_metrics[f"{algorithm}_{k}"] = val
        net_arch = algo_kwargs.get("policy_kwargs", {}).get("net_arch")
        if net_arch is not None:
            aux_metrics[f"{algorithm}_net_arch"] = str(net_arch)

    # Write all metrics to a JSON sidecar so they can be collected from
    # GCS without relying on the HPT metric_spec.
    metrics_path = Path(log_path) / "metrics.json"
    with open(metrics_path, "w") as f:
        _json.dump(aux_metrics, f, indent=2)
    logger.info("Metrics sidecar written to: %s", metrics_path)


# ── Curriculum training ──────────────────────────────────────────────────


def train_curriculum(
    species_cfg: SpeciesConfig,
    stage_configs: dict[int, dict[str, Any]],
    n_envs: int = 4,
    seed: int = 42,
    eval_freq: int = 50000,
    save_freq: int = 50000,
    log_dir: str | None = None,
    use_subproc: bool = False,
    verbose: int = 1,
    algorithm: str = "ppo",
    use_wandb: bool = False,
    output_dir: str | None = None,
    gcs_bucket: str | None = None,
    gcs_project: str | None = None,
    use_tensorboard: bool = True,
):
    """Run the full 3-stage curriculum with automatic advancement."""
    from .config import (
        save_stage_config,
        upload_curriculum_artifacts,
    )
    from .curriculum import (
        CurriculumCallback,
        CurriculumManager,
        RewardRampCallback,
        StageWarmupCallback,
        thresholds_from_configs,
    )
    from .wandb_integration import init_wandb

    sb3 = _ensure_sb3()
    species = species_cfg.species

    thresholds = thresholds_from_configs(stage_configs)
    manager = CurriculumManager(species=species, stage_thresholds=thresholds)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is not None:
        base_dir = Path(output_dir)
    elif log_dir is None:
        base_dir = Path(__file__).parent.parent / species / "logs" / species / f"curriculum_{timestamp}"
    else:
        base_dir = Path(log_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Starting automated curriculum training (stages 1-3)")
    logger.info("Base directory: %s", base_dir)
    logger.info("=" * 60)

    model = None
    load_path = None
    prev_vecnorm_path = None

    for stage in range(1, 4):
        config = stage_configs[stage]
        cur_kwargs = config.get("curriculum_kwargs", {})
        total_timesteps = cur_kwargs.get("timesteps", 500000)

        stage_dir = base_dir / f"stage{stage}"
        stage_dir.mkdir(exist_ok=True)
        model_dir = stage_dir / "models"
        model_dir.mkdir(exist_ok=True)

        logger.info("=" * 60)
        logger.info("Curriculum Stage %d/%d: %s", stage, 3, config["name"])
        logger.info("Description: %s", config["description"])
        logger.info("Timesteps: %s", f"{total_timesteps:,}")
        logger.info("=" * 60)

        save_stage_config(
            stage_dir,
            stage,
            config,
            algorithm.upper(),
            extra={"seed": seed, "n_envs": n_envs, "timesteps": total_timesteps},
            env_class=species_cfg.env_class,
            species=species_cfg.species,
        )

        effective_subproc = use_subproc or (algorithm == "sac" and n_envs > 1)
        train_env = create_vec_env(species_cfg, stage_configs, stage, n_envs, seed, effective_subproc)
        eval_env = create_vec_env(species_cfg, stage_configs, stage, 1, seed + 1000, use_subproc=False)

        _load_vecnorm_into_envs(prev_vecnorm_path, train_env, eval_env)

        alg_kwargs, local_tb_dir, gcs_tb_path = _prepare_alg_kwargs(
            config,
            algorithm,
            verbose,
            stage_dir,
            use_tensorboard,
        )

        wandb_run = None
        if use_wandb:
            wandb_run = init_wandb(species=species, stage=stage, config=config)

        model = _create_or_load_model(sb3, algorithm, alg_kwargs, train_env, load_path)

        callbacks, eval_callback, _ = _build_core_callbacks(
            sb3,
            eval_env,
            model_dir,
            stage_dir,
            stage,
            n_envs,
            eval_freq,
            save_freq,
            verbose,
            use_wandb,
        )

        curriculum_cb = CurriculumCallback(
            curriculum_manager=manager,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=30,
            eval_callback=eval_callback,
        )
        callbacks.append(curriculum_cb)

        if stage > 1:
            callbacks.append(
                StageWarmupCallback(
                    warmup_timesteps=cur_kwargs.get("warmup_timesteps", 100_000),
                    warmup_clip_range=cur_kwargs.get("warmup_clip_range", 0.02),
                    warmup_ent_coef=cur_kwargs.get("warmup_ent_coef", 0.02),
                    warmup_lr_scale=cur_kwargs.get("warmup_lr_scale", 0.1),
                )
            )
        if stage > 1:
            target_fwd_weight = config["env_kwargs"].get("forward_vel_weight", 1.0)
            callbacks.append(
                RewardRampCallback(
                    attr_name="forward_vel_weight",
                    start_value=cur_kwargs.get("ramp_start_value", 0.1),
                    end_value=target_fwd_weight,
                    ramp_timesteps=cur_kwargs.get("ramp_timesteps", 500_000),
                )
            )

        interrupted = False
        try:
            model.learn(
                total_timesteps=total_timesteps,
                callback=sb3["CallbackList"](callbacks),
                progress_bar=verbose >= 1,
            )
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user.")
            interrupted = True

        if wandb_run is not None:
            wandb_run.finish()

        final_path = _save_final_and_sync_tb(
            model,
            train_env,
            model_dir,
            stage,
            local_tb_dir,
            gcs_tb_path,
        )

        # Prefer loading the best model + its matched VecNormalize for the
        # next stage.  SaveVecNormalizeCallback (wired to EvalCallback's
        # callback_on_new_best) saves best_model_vecnorm.pkl alongside
        # best_model.zip so the obs normalization matches the policy weights.
        best_model_zip = model_dir / "best_model.zip"
        best_vecnorm_path = str(model_dir / "best_model_vecnorm.pkl")
        if best_model_zip.exists() and Path(best_vecnorm_path).exists():
            load_path = str(model_dir / "best_model")
            prev_vecnorm_path = best_vecnorm_path
            logger.info(
                "Next stage will load best model (%s) with VecNormalize: %s",
                load_path,
                prev_vecnorm_path,
            )
        else:
            load_path = str(final_path)
            prev_vecnorm_path = str(final_path) + "_vecnorm.pkl"

        train_env.close()
        eval_env.close()

        # Record stage result to CSV
        _record_stage_result(
            species,
            algorithm,
            timestamp,
            base_dir,
            stage,
            config,
            cur_kwargs,
            eval_callback,
            stage_dir,
            seed,
            n_envs,
            total_timesteps,
            curriculum_cb,
        )

        if interrupted:
            break

        if curriculum_cb and curriculum_cb.ready_to_advance and not manager.is_final_stage:
            manager.advance()
            logger.info("Auto-advanced to stage %d", manager.current_stage)
        elif stage < 3:
            logger.warning(
                "Stage %d timestep budget exhausted without meeting advancement "
                "thresholds. Stopping curriculum — advancing with a weak policy "
                "causes catastrophic forgetting.",
                stage,
            )
            break

    upload_curriculum_artifacts(
        base_dir,
        species=species,
        algorithm=algorithm,
        bucket=gcs_bucket,
        project=gcs_project,
    )

    logger.info("=" * 60)
    logger.info("Curriculum training complete!")
    logger.info("Results directory: %s", base_dir)
    logger.info("=" * 60)


def _record_stage_result(
    species,
    algorithm,
    timestamp,
    base_dir,
    stage,
    config,
    cur_kwargs,
    eval_callback,
    stage_dir,
    seed,
    n_envs,
    total_timesteps,
    curriculum_cb,
):
    """Record stage hyperparameters and outcome to CSV."""
    import numpy as _np

    from .config import append_stage_result_csv

    algo_prefix = "sac" if algorithm == "sac" else "ppo"
    algo_key = f"{algo_prefix}_kwargs"
    algo_kwargs = config[algo_key]
    env_kwargs = config["env_kwargs"]

    best_mean_reward: float | str = eval_callback.best_mean_reward
    best_mean_episode_length: float | str = ""
    last_mean_reward: float | str = ""
    last_mean_episode_length: float | str = ""
    eval_npz = stage_dir / "evaluations.npz"
    if eval_npz.exists():
        eval_data = _np.load(str(eval_npz))
        eval_rewards = eval_data["results"]
        eval_lengths = eval_data["ep_lengths"]
        mean_rewards_per_eval = eval_rewards.mean(axis=1)
        best_idx = int(mean_rewards_per_eval.argmax())
        best_mean_reward = round(float(mean_rewards_per_eval[best_idx]), 2)
        best_mean_episode_length = round(float(eval_lengths[best_idx].mean()), 1)
        # Last eval as "final" metrics
        last_mean_reward = round(float(mean_rewards_per_eval[-1]), 2)
        last_mean_episode_length = round(float(eval_lengths[-1].mean()), 1)

    net_arch_val = algo_kwargs.get("policy_kwargs", {}).get("net_arch", "")
    if isinstance(net_arch_val, (list, tuple)):
        net_arch_str = str(list(net_arch_val))
    else:
        net_arch_str = str(net_arch_val) if net_arch_val else ""

    # Use canonical column names matching CSV_METRIC_COLUMNS and prefixed
    # hyperparameter conventions from the sweep CSV format.
    result_row: dict = {
        "species": species,
        "algorithm": algorithm.upper(),
        "run_date": timestamp,
        "run_dir": base_dir.name,
        "stage": stage,
        "stage_name": config["name"],
        "seed": seed,
        "n_envs": n_envs,
        # Prefixed hyperparameters (matching sweep CSV conventions)
        f"{algo_prefix}_learning_rate": algo_kwargs.get("learning_rate", ""),
        f"{algo_prefix}_batch_size": algo_kwargs.get("batch_size", ""),
        f"{algo_prefix}_gamma": algo_kwargs.get("gamma", ""),
        f"{algo_prefix}_net_arch": net_arch_str,
        "env_alive_bonus": env_kwargs.get("alive_bonus", ""),
        "env_energy_penalty_weight": env_kwargs.get("energy_penalty_weight", ""),
        "env_forward_vel_weight": env_kwargs.get("forward_vel_weight", ""),
        "env_posture_weight": env_kwargs.get("posture_weight", ""),
        # Canonical metric columns
        "best_mean_reward": best_mean_reward,
        "best_mean_episode_length": best_mean_episode_length,
        "last_mean_reward": last_mean_reward,
        "last_mean_episode_length": last_mean_episode_length,
        "training_duration_seconds": "",
        "reward_threshold": cur_kwargs.get("min_avg_reward", ""),
        "ep_length_threshold": cur_kwargs.get("min_avg_episode_length", ""),
        "forward_vel_threshold": cur_kwargs.get("min_avg_forward_vel", ""),
        "success_rate_threshold": cur_kwargs.get("min_success_rate", ""),
        "stage_passed": bool(curriculum_cb is not None and curriculum_cb.ready_to_advance),
    }
    append_stage_result_csv(base_dir / "curriculum_results.csv", result_row)
    logger.info(
        "Stage %d result appended to: %s",
        stage,
        base_dir / "curriculum_results.csv",
    )


# ── Backward-compatible re-exports ──────────────────────────────────────
# These were extracted to dedicated modules but are re-exported here so
# existing ``from environments.shared.train_base import ...`` continues
# to work without changes.

from .cli import _apply_overrides, _cast_value, main  # noqa: E402, F401
from .diagnostics import DiagnosticsCallback  # noqa: E402, F401
from .evaluation import eval_policy, evaluate, record_stage_video  # noqa: E402, F401
