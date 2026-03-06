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

import dataclasses
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

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


# ── Environment creation ─────────────────────────────────────────────────


def make_env(
    species_cfg: SpeciesConfig,
    stage_configs: Dict[int, Dict[str, Any]],
    stage: int,
    rank: int,
    seed: int = 0,
):
    """Create a single environment instance."""
    sb3 = _ensure_sb3()

    def _init():
        env_kwargs = stage_configs[stage]["env_kwargs"].copy()
        env = species_cfg.env_class(**env_kwargs)
        env = sb3["Monitor"](env)
        env.reset(seed=seed + rank)
        return env

    sb3["set_random_seed"](seed)
    return _init


def create_vec_env(
    species_cfg: SpeciesConfig,
    stage_configs: Dict[int, Dict[str, Any]],
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
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=50.0,
    )
    return env


# ── Single-stage training ────────────────────────────────────────────────


def train(
    species_cfg: SpeciesConfig,
    stage_configs: Dict[int, Dict[str, Any]],
    stage: int,
    total_timesteps: int,
    n_envs: int = 4,
    seed: int = 42,
    load_path: str | None = None,
    eval_freq: int = 10000,
    save_freq: int = 50000,
    log_dir: str | None = None,
    use_subproc: bool = False,
    verbose: int = 1,
    algorithm: str = "ppo",
    use_wandb: bool = False,
    output_dir: str | None = None,
):
    """Train a single stage of the curriculum."""
    from .config import save_stage_config
    from .curriculum import (
        EvalCollapseEarlyStopCallback,
        RewardRampCallback,
        SaveVecNormalizeCallback,
        StageWarmupCallback,
        load_vecnorm_stats,
    )
    from .diagnostics import DiagnosticsCallback  # noqa: F811
    from .wandb_integration import WandbCallback, init_wandb

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
    )

    # Create environments
    logger.info("Creating %d training environments...", n_envs)
    train_env = create_vec_env(species_cfg, stage_configs, stage, n_envs, seed, use_subproc)

    logger.info("Creating evaluation environment...")
    eval_env = create_vec_env(species_cfg, stage_configs, stage, 1, seed + 1000, use_subproc=False)

    # Carry forward normalization statistics from a prior stage
    if load_path:
        _vecnorm_path = load_path.replace(".zip", "") + "_vecnorm.pkl"
        if not _vecnorm_path.endswith("_vecnorm.pkl"):
            _vecnorm_path = load_path + "_vecnorm.pkl"
        if not load_vecnorm_stats(_vecnorm_path, train_env, eval_env):
            # VecNormalize file not found (e.g. missing from GCS mount).
            # Ensure eval env doesn't pollute running stats.
            logger.warning("VecNormalize file not found: %s — eval env will use defaults", _vecnorm_path)
            eval_env.training = False
            eval_env.norm_reward = False
    else:
        # Even without prior stats, the eval env should never update
        # running statistics or normalise rewards during evaluation.
        eval_env.training = False
        eval_env.norm_reward = False

    # Create or load model
    alg_cls = sb3["SAC"] if algorithm == "sac" else sb3["PPO"]
    alg_kwargs = config["sac_kwargs"].copy() if algorithm == "sac" else config["ppo_kwargs"].copy()
    alg_kwargs["verbose"] = verbose
    alg_kwargs["tensorboard_log"] = str(log_path / "tensorboard")

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

        # Clip range annealing
        clip_range_end = alg_kwargs.pop("clip_range_end", None)
        if clip_range_end is not None:
            clip_start = alg_kwargs["clip_range"]
            alg_kwargs["clip_range"] = linear_schedule(clip_start, clip_range_end)
            logger.info("Using clip_range schedule: %s -> %s", clip_start, clip_range_end)

    wandb_run = None
    if use_wandb:
        wandb_run = init_wandb(species=species, stage=stage, config=config)
        logger.info("W&B run initialized.")

    # policy_kwargs defines the network architecture and must only be used
    # when creating a *new* model.  When loading a saved model the
    # architecture is already baked into the weights; passing a (possibly
    # different) policy_kwargs to .load() would create a metadata mismatch.
    policy_kwargs = alg_kwargs.pop("policy_kwargs", None)

    if load_path:
        logger.info("Loading model from: %s", load_path)
        model = alg_cls.load(load_path, env=train_env, **alg_kwargs)
    else:
        logger.info("Creating new %s model...", algorithm.upper())
        model = alg_cls("MlpPolicy", train_env, policy_kwargs=policy_kwargs, **alg_kwargs)

    logger.info("Model architecture:")
    logger.info("  Policy: %s", model.policy)
    logger.info("  Learning rate: %s", model.learning_rate)
    logger.info("  Batch size: %s", alg_kwargs["batch_size"])

    # Setup callbacks
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

    callbacks.append(DiagnosticsCallback(log_dir=str(log_path), verbose=verbose))

    # Early stopping on eval reward collapse
    callbacks.append(EvalCollapseEarlyStopCallback(eval_callback=eval_callback, verbose=verbose))

    if use_wandb:
        callbacks.append(WandbCallback())

    if stage > 1 and load_path:
        cur_kwargs = config.get("curriculum_kwargs", {})
        if algorithm == "ppo":
            callbacks.append(
                StageWarmupCallback(
                    warmup_timesteps=cur_kwargs.get("warmup_timesteps", 100_000),
                    warmup_clip_range=cur_kwargs.get("warmup_clip_range", 0.02),
                    warmup_ent_coef=cur_kwargs.get("warmup_ent_coef", 0.02),
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

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            progress_bar=verbose >= 1,
        )
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")

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
    )

    # Save final model
    final_path = model_dir / f"stage{stage}_final"
    model.save(str(final_path))
    train_env.save(str(final_path) + "_vecnorm.pkl")

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
    aux_metrics: dict[str, float] = {
        "best_mean_reward": float(eval_callback.best_mean_reward),
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

    if stage >= 2:
        # Use the best model + its matched VecNormalize for forward_vel
        # and success_rate evaluation so the metrics reflect the checkpoint
        # that will actually be handed off to the next stage.
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

        _, _, fwd_vels, success_flags = eval_policy(eval_model, eval_env, species_cfg.success_keys, n_episodes=30)
        if fwd_vels:
            best_fwd = float(_np.mean(fwd_vels))
            aux_metrics["best_mean_forward_vel"] = best_fwd
            logger.info("HPT metric reported: best_mean_forward_vel=%.4f", best_fwd)
        if stage >= 3 and success_flags:
            best_success = float(_np.mean(success_flags))
            aux_metrics["best_mean_success_rate"] = best_success
            logger.info(
                "HPT metric reported: best_mean_success_rate=%.4f",
                best_success,
            )

    # Write all metrics to a JSON sidecar so they can be collected from
    # GCS without relying on the HPT metric_spec.
    metrics_path = Path(log_path) / "metrics.json"
    with open(metrics_path, "w") as f:
        _json.dump(aux_metrics, f, indent=2)
    logger.info("Metrics sidecar written to: %s", metrics_path)


# ── Curriculum training ──────────────────────────────────────────────────


def train_curriculum(
    species_cfg: SpeciesConfig,
    stage_configs: Dict[int, Dict[str, Any]],
    n_envs: int = 4,
    seed: int = 42,
    eval_freq: int = 10000,
    save_freq: int = 50000,
    log_dir: str | None = None,
    use_subproc: bool = False,
    verbose: int = 1,
    algorithm: str = "ppo",
    use_wandb: bool = False,
    output_dir: str | None = None,
    gcs_bucket: str | None = None,
    gcs_project: str | None = None,
):
    """Run the full 3-stage curriculum with automatic advancement."""
    from .config import (
        save_stage_config,
        upload_curriculum_artifacts,
    )
    from .curriculum import (
        CurriculumCallback,
        CurriculumManager,
        EvalCollapseEarlyStopCallback,
        RewardRampCallback,
        SaveVecNormalizeCallback,
        StageWarmupCallback,
        load_vecnorm_stats,
        thresholds_from_configs,
    )
    from .diagnostics import DiagnosticsCallback  # noqa: F811
    from .wandb_integration import WandbCallback, init_wandb

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
        )

        train_env = create_vec_env(species_cfg, stage_configs, stage, n_envs, seed, use_subproc)
        eval_env = create_vec_env(species_cfg, stage_configs, stage, 1, seed + 1000, use_subproc=False)

        if prev_vecnorm_path:
            if not load_vecnorm_stats(prev_vecnorm_path, train_env, eval_env):
                logger.warning("VecNormalize file not found: %s — eval env will use defaults", prev_vecnorm_path)
                eval_env.training = False
                eval_env.norm_reward = False
        else:
            eval_env.training = False
            eval_env.norm_reward = False

        alg_cls = sb3["SAC"] if algorithm == "sac" else sb3["PPO"]
        alg_kwargs = config["sac_kwargs"].copy() if algorithm == "sac" else config["ppo_kwargs"].copy()
        alg_kwargs["verbose"] = verbose
        alg_kwargs["tensorboard_log"] = str(stage_dir / "tensorboard")

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

            # Clip range annealing
            clip_range_end = alg_kwargs.pop("clip_range_end", None)
            if clip_range_end is not None:
                clip_start = alg_kwargs["clip_range"]
                alg_kwargs["clip_range"] = linear_schedule(clip_start, clip_range_end)
                logger.info("Using clip_range schedule: %s -> %s", clip_start, clip_range_end)

        wandb_run = None
        if use_wandb:
            wandb_run = init_wandb(species=species, stage=stage, config=config)

        policy_kwargs = alg_kwargs.pop("policy_kwargs", None)

        if load_path:
            logger.info("Loading model from previous stage: %s", load_path)
            model = alg_cls.load(load_path, env=train_env, **alg_kwargs)
        else:
            model = alg_cls("MlpPolicy", train_env, policy_kwargs=policy_kwargs, **alg_kwargs)

        # Build callbacks
        callbacks = []

        best_vecnorm_path = str(model_dir / "best_model_vecnorm.pkl")
        save_vecnorm_cb = SaveVecNormalizeCallback(save_path=best_vecnorm_path)

        eval_callback = sb3["EvalCallback"](
            eval_env,
            best_model_save_path=str(model_dir),
            log_path=str(stage_dir),
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

        callbacks.append(DiagnosticsCallback(log_dir=str(stage_dir), verbose=verbose))

        # Early stopping on eval reward collapse
        callbacks.append(EvalCollapseEarlyStopCallback(eval_callback=eval_callback, verbose=verbose))

        if use_wandb:
            callbacks.append(WandbCallback())

        curriculum_cb = CurriculumCallback(
            curriculum_manager=manager,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=30,
            eval_callback=eval_callback,
        )
        callbacks.append(curriculum_cb)

        if stage > 1 and algorithm == "ppo":
            callbacks.append(
                StageWarmupCallback(
                    warmup_timesteps=cur_kwargs.get("warmup_timesteps", 100_000),
                    warmup_clip_range=cur_kwargs.get("warmup_clip_range", 0.02),
                    warmup_ent_coef=cur_kwargs.get("warmup_ent_coef", 0.02),
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

        # Save stage checkpoint
        final_path = model_dir / f"stage{stage}_final"
        model.save(str(final_path))
        train_env.save(str(final_path) + "_vecnorm.pkl")

        # Prefer loading the best model + its matched VecNormalize for the
        # next stage.  SaveVecNormalizeCallback (wired to EvalCallback's
        # callback_on_new_best) saves best_model_vecnorm.pkl alongside
        # best_model.zip so the obs normalization matches the policy weights.
        best_model_zip = model_dir / "best_model.zip"
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

    algo_key = "sac_kwargs" if algorithm == "sac" else "ppo_kwargs"
    algo_kwargs = config[algo_key]
    env_kwargs = config["env_kwargs"]

    avg_reward: float | str = eval_callback.best_mean_reward
    std_reward: float | str = ""
    avg_ep_length: float | str = ""
    std_ep_length: float | str = ""
    eval_npz = stage_dir / "evaluations.npz"
    if eval_npz.exists():
        eval_data = _np.load(str(eval_npz))
        eval_rewards = eval_data["results"]
        eval_lengths = eval_data["ep_lengths"]
        mean_rewards_per_eval = eval_rewards.mean(axis=1)
        best_idx = int(mean_rewards_per_eval.argmax())
        avg_reward = round(float(mean_rewards_per_eval[best_idx]), 2)
        std_reward = round(float(eval_rewards[best_idx].std()), 2)
        avg_ep_length = round(float(eval_lengths[best_idx].mean()), 1)
        std_ep_length = round(float(eval_lengths[best_idx].std()), 1)

    net_arch_val = algo_kwargs.get("policy_kwargs", {}).get("net_arch", "")
    if isinstance(net_arch_val, (list, tuple)):
        net_arch_str = str(list(net_arch_val))
    else:
        net_arch_str = str(net_arch_val) if net_arch_val else ""

    result_row: dict = {
        "species": species,
        "algorithm": algorithm.upper(),
        "run_date": timestamp,
        "run_dir": base_dir.name,
        "stage": stage,
        "stage_name": config["name"],
        "passed": bool(curriculum_cb is not None and curriculum_cb.ready_to_advance),
        "avg_reward": avg_reward,
        "std_reward": std_reward,
        "avg_ep_length": avg_ep_length,
        "std_ep_length": std_ep_length,
        "timesteps": total_timesteps,
        "threshold_reward": cur_kwargs.get("min_avg_reward", ""),
        "threshold_ep_length": cur_kwargs.get("min_avg_episode_length", ""),
        "learning_rate": algo_kwargs.get("learning_rate", ""),
        "batch_size": algo_kwargs.get("batch_size", ""),
        "gamma": algo_kwargs.get("gamma", ""),
        "net_arch": net_arch_str,
        "seed": seed,
        "n_envs": n_envs,
        "alive_bonus": env_kwargs.get("alive_bonus", ""),
        "energy_penalty": env_kwargs.get("energy_penalty_weight", ""),
        "forward_vel_weight": env_kwargs.get("forward_vel_weight", ""),
        "posture_weight": env_kwargs.get("posture_weight", ""),
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
