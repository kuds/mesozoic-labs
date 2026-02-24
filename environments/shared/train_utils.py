"""
Shared training utilities for all species.

Consolidates common training logic (environment creation, algorithm setup,
evaluation, video recording, summary generation) so that each species'
``train_sb3.py`` is a thin wrapper supplying species-specific constants.

Usage::

    from environments.shared.train_utils import (
        create_vec_env,
        evaluate_with_forward_vel,
        linear_schedule,
        plot_training_curves,
        prepare_algo_kwargs,
        record_stage_video,
        save_results_json,
        write_stage_summary,
        write_training_summary,
    )
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import numpy as np

try:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        CallbackList,
        CheckpointCallback,
        EvalCallback,
    )
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.utils import get_schedule_fn, set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
except ImportError as _exc:
    raise ImportError(
        "stable-baselines3 is required for training. Install with: pip install stable-baselines3[extra]"
    ) from _exc

from environments.shared.config import load_all_stages, save_stage_config
from environments.shared.curriculum import (
    CurriculumCallback,
    CurriculumManager,
    thresholds_from_configs,
)
from environments.shared.diagnostics import DiagnosticsCallback

logger = logging.getLogger(__name__)

try:
    import mediapy

    _HAS_MEDIAPY = True
except ImportError:
    _HAS_MEDIAPY = False

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

_ALGO_CLASS: Dict[str, type] = {"PPO": PPO, "SAC": SAC}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def linear_schedule(initial_lr: float, final_lr: float):
    """Return a callable that linearly decays learning rate from *initial_lr* to *final_lr*."""

    def schedule(progress_remaining: float) -> float:
        return final_lr + progress_remaining * (initial_lr - final_lr)

    return schedule


def _format_duration(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def _format_duration_hms(seconds: float) -> str:
    """Format seconds as H:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"


def resolve_log_dir(
    explicit: Optional[str],
    species: str,
    prefix: str,
    fallback_root: Path,
) -> Path:
    """Determine the log directory, respecting ``LOG_DIR`` env var.

    Priority: *explicit* argument > ``LOG_DIR`` environment variable > local
    ``<fallback_root>/logs/<species>/<prefix>_<timestamp>``.
    """
    if explicit is not None:
        return Path(explicit)
    env_dir = os.environ.get("LOG_DIR")
    if env_dir:
        return Path(env_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return fallback_root / "logs" / species / f"{prefix}_{timestamp}"


# ---------------------------------------------------------------------------
# Environment creation
# ---------------------------------------------------------------------------


def make_env(
    env_class: type,
    stage_configs: Dict[int, Dict[str, Any]],
    stage: int,
    rank: int,
    seed: int = 0,
    log_dir: Optional[str] = None,
):
    """Create a single Gymnasium environment wrapped with Monitor."""

    def _init():
        env_kwargs = stage_configs[stage]["env_kwargs"].copy()
        env = env_class(**env_kwargs)
        monitor_kwargs: Dict[str, Any] = {}
        if log_dir is not None:
            monitor_dir = Path(log_dir) / "monitor"
            monitor_dir.mkdir(parents=True, exist_ok=True)
            monitor_kwargs["filename"] = str(monitor_dir / f"env_{rank}")
        env = Monitor(env, **monitor_kwargs)
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


def create_vec_env(
    env_class: type,
    stage_configs: Dict[int, Dict[str, Any]],
    stage: int,
    n_envs: int = 4,
    seed: int = 42,
    use_subproc: bool = False,
    vecnorm_path: Optional[str] = None,
    clip_reward: float = 10.0,
    log_dir: Optional[str] = None,
):
    """Create a vectorized environment with observation/reward normalization.

    If *vecnorm_path* is provided and the file exists, normalization
    statistics are loaded from a prior stage so the policy sees
    consistently-scaled observations across curriculum stages.
    """
    env_fns = [make_env(env_class, stage_configs, stage, i, seed, log_dir=log_dir) for i in range(n_envs)]
    if use_subproc and n_envs > 1:
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)

    if vecnorm_path and Path(vecnorm_path).exists():
        env = VecNormalize.load(vecnorm_path, env)
        env.training = True
        logger.info("Loaded VecNormalize stats from prior stage: %s", vecnorm_path)
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=clip_reward)

    return env


# ---------------------------------------------------------------------------
# Algorithm setup
# ---------------------------------------------------------------------------


def prepare_algo_kwargs(
    config: Dict[str, Any],
    algorithm: str,
    verbose: int,
    tensorboard_log: str,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Extract and prepare algorithm hyperparameters from a stage config.

    Handles ``policy_kwargs`` extraction, ``learning_rate_end`` schedule
    creation, and non-algorithm key removal.

    Returns:
        ``(algo_kwargs, policy_kwargs)``
    """
    key = "ppo_kwargs" if algorithm.upper() == "PPO" else "sac_kwargs"
    algo_kwargs = config[key].copy()
    algo_kwargs["verbose"] = verbose
    algo_kwargs["tensorboard_log"] = tensorboard_log

    # Extract policy_kwargs (e.g. net_arch) from algo_kwargs
    policy_kwargs = algo_kwargs.pop("policy_kwargs", {"net_arch": [256, 256]})

    # Apply linear LR schedule if learning_rate_end is specified
    lr_end = algo_kwargs.pop("learning_rate_end", None)
    if lr_end is not None:
        lr_start = algo_kwargs["learning_rate"]
        algo_kwargs["learning_rate"] = linear_schedule(lr_start, lr_end)
        logger.info("Using linear LR schedule: %s -> %s", lr_start, lr_end)

    return algo_kwargs, policy_kwargs


def create_model(
    algorithm: str,
    train_env,
    algo_kwargs: Dict[str, Any],
    policy_kwargs: Dict[str, Any],
    load_path: Optional[str] = None,
):
    """Create a new SB3 model or load one from a checkpoint.

    When loading, stage hyperparameters (learning rate, entropy coeff,
    clip range) are updated on the loaded model.
    """
    AlgoClass = _ALGO_CLASS[algorithm.upper()]
    if load_path:
        logger.info("Loading model from: %s", load_path)
        model = AlgoClass.load(load_path, env=train_env)
        # Update hyperparameters for the new stage
        lr = algo_kwargs.get("learning_rate")
        if lr is not None:
            model.learning_rate = lr if callable(lr) else get_schedule_fn(lr)
        if algorithm.upper() == "PPO":
            if "ent_coef" in algo_kwargs:
                model.ent_coef = algo_kwargs["ent_coef"]
            if "clip_range" in algo_kwargs:
                model.clip_range = get_schedule_fn(algo_kwargs["clip_range"])
    else:
        logger.info("Creating new %s model...", algorithm)
        model = AlgoClass(
            "MlpPolicy",
            train_env,
            policy_kwargs=policy_kwargs,
            **algo_kwargs,
        )
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_with_forward_vel(
    model,
    env_class: type,
    stage_configs: Dict[int, Dict[str, Any]],
    stage: int,
    n_episodes: int = 25,
    seed: int = 42,
    vecnorm_path: Optional[str] = None,
) -> tuple[List[float], List[int], List[float]]:
    """Evaluate a trained policy and collect per-episode forward velocities.

    Returns ``(episode_rewards, episode_lengths, episode_forward_vels)``.
    """
    env_kwargs = stage_configs[stage]["env_kwargs"].copy()
    env = env_class(**env_kwargs)

    vec_normalize = None
    if vecnorm_path and Path(vecnorm_path).exists():
        dummy_env = DummyVecEnv([lambda: env_class(**env_kwargs)])
        vec_normalize = VecNormalize.load(vecnorm_path, dummy_env)
        vec_normalize.training = False
        vec_normalize.norm_reward = False

    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    episode_forward_vels: List[float] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        step = 0
        ep_vels: List[float] = []
        while True:
            obs_input = vec_normalize.normalize_obs(obs) if vec_normalize is not None else obs
            action, _ = model.predict(obs_input, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            if "forward_vel" in info:
                ep_vels.append(float(info["forward_vel"]))
            if terminated or truncated:
                break
        episode_rewards.append(total_reward)
        episode_lengths.append(step)
        episode_forward_vels.append(float(np.mean(ep_vels)) if ep_vels else 0.0)

    env.close()
    if vec_normalize is not None:
        vec_normalize.close()
    return episode_rewards, episode_lengths, episode_forward_vels


# ---------------------------------------------------------------------------
# Single-stage training
# ---------------------------------------------------------------------------


def train_stage(
    env_class: type,
    species: str,
    stage_configs: Dict[int, Dict[str, Any]],
    stage: int,
    total_timesteps: int,
    algorithm: str = "PPO",
    n_envs: int = 4,
    seed: int = 42,
    load_path: Optional[str] = None,
    eval_freq: int = 10000,
    save_freq: int = 50000,
    log_dir: Optional[str] = None,
    use_subproc: bool = False,
    use_wandb: bool = False,
    clip_reward: float = 10.0,
    reward_keys: Optional[List[str]] = None,
    info_keys: Optional[List[str]] = None,
    verbose: int = 1,
    vecnorm_path: Optional[str] = None,
    fallback_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train a single curriculum stage and return results.

    This is the core training loop shared by all species. Returns a
    results dict containing evaluation metrics, timing, and file paths.
    """
    stage_start = time.time()
    config = stage_configs[stage]

    logger.info("=" * 60)
    logger.info("Training Stage %d: %s (%s)", stage, config["name"], algorithm)
    logger.info("Description: %s", config["description"])
    logger.info("Timesteps: %s", f"{total_timesteps:,}")
    logger.info("=" * 60)

    # Resolve log directory
    if fallback_root is None:
        fallback_root = Path.cwd()
    log_path = resolve_log_dir(log_dir, species, f"stage{stage}", fallback_root)
    log_path.mkdir(parents=True, exist_ok=True)
    model_dir = log_path / "models"
    model_dir.mkdir(exist_ok=True)

    logger.info("Log directory: %s", log_path)

    # Save config for reproducibility
    save_stage_config(
        log_path, stage, config, algorithm,
        extra={"seed": seed, "n_envs": n_envs, "timesteps": total_timesteps},
    )

    # Create environments
    train_env = create_vec_env(
        env_class, stage_configs, stage, n_envs, seed,
        use_subproc=use_subproc, vecnorm_path=vecnorm_path,
        clip_reward=clip_reward, log_dir=str(log_path),
    )
    eval_env = create_vec_env(
        env_class, stage_configs, stage, 1, seed + 1000,
        vecnorm_path=vecnorm_path, clip_reward=clip_reward,
    )

    # Prepare algorithm kwargs
    algo_kwargs, policy_kwargs = prepare_algo_kwargs(
        config, algorithm, verbose, str(log_path / "tensorboard"),
    )

    # Create or load model
    model = create_model(algorithm, train_env, algo_kwargs, policy_kwargs, load_path)

    # Build callbacks
    callbacks: List[BaseCallback] = []

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(log_path),
        eval_freq=max(eval_freq // n_envs, 1),
        n_eval_episodes=25,
        deterministic=True,
        verbose=max(verbose, 1),
    )
    callbacks.append(eval_callback)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(save_freq // n_envs, 1),
        save_path=str(model_dir),
        name_prefix=f"stage{stage}",
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)

    if reward_keys and info_keys:
        diagnostics_callback = DiagnosticsCallback(
            reward_keys=reward_keys,
            info_keys=info_keys,
            plateau_window=10,
            plateau_threshold=1.0,
            log_dir=str(log_path),
        )
        callbacks.append(diagnostics_callback)

    # W&B integration
    wandb_run = None
    if use_wandb:
        from environments.shared.wandb_integration import WandbCallback, init_wandb

        wandb_run = init_wandb(species, stage, config)
        callbacks.append(WandbCallback(log_freq=1000))

    # Train
    logger.info("Starting training for %s timesteps...", f"{total_timesteps:,}")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList(callbacks),
            progress_bar=verbose >= 1,
        )
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")

    # Save final model and VecNormalize stats
    final_path = model_dir / f"stage{stage}_final"
    model.save(str(final_path))
    vecnorm_save_path = str(final_path) + "_vecnorm.pkl"
    train_env.save(vecnorm_save_path)
    logger.info("Final model: %s.zip", final_path)
    logger.info("VecNormalize stats: %s", vecnorm_save_path)

    # Use best model if available
    best_model_zip = model_dir / "best_model.zip"
    AlgoClass = _ALGO_CLASS[algorithm.upper()]
    if best_model_zip.exists():
        best_path = model_dir / "best_model"
        model = AlgoClass.load(str(best_path), env=train_env)
        output_path = str(best_path)
        logger.info("Loaded best model for evaluation: %s.zip", best_path)
    else:
        output_path = str(final_path)

    # Evaluate with forward velocity
    episode_rewards, episode_lengths, episode_fwd_vels = evaluate_with_forward_vel(
        model, env_class, stage_configs, stage,
        n_episodes=25, seed=seed + 3000, vecnorm_path=vecnorm_save_path,
    )
    mean_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))
    mean_length = float(np.mean(episode_lengths))
    std_length = float(np.std(episode_lengths))
    mean_fwd_vel = float(np.mean(episode_fwd_vels))
    std_fwd_vel = float(np.std(episode_fwd_vels))

    # Try to get sim_dt from environment
    try:
        sim_dt = float(eval_env.get_attr("dt")[0])
    except Exception:
        sim_dt = 0.01

    logger.info("Eval: reward=%.2f +/- %.2f", mean_reward, std_reward)
    logger.info(
        "Eval: length=%.1f +/- %.1f steps (%.2fs sim time)",
        mean_length, std_length, mean_length * sim_dt,
    )
    logger.info("Eval: forward_vel=%.2f +/- %.2f m/s", mean_fwd_vel, std_fwd_vel)

    train_env.close()
    eval_env.close()

    if wandb_run is not None:
        wandb_run.finish()

    stage_duration = time.time() - stage_start

    results = {
        "stage": stage,
        "name": config["name"],
        "description": config["description"],
        "timesteps": total_timesteps,
        "duration_seconds": stage_duration,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_episode_length": mean_length,
        "std_episode_length": std_length,
        "mean_forward_vel": mean_fwd_vel,
        "std_forward_vel": std_fwd_vel,
        "sim_dt": sim_dt,
        "model_path": output_path,
        "vecnorm_path": vecnorm_save_path,
        "stage_dir": str(log_path),
    }

    # Write per-stage text summary
    write_stage_summary(log_path, results, species, algorithm)

    logger.info("Stage %d complete in %s", stage, _format_duration(stage_duration))

    return results


# ---------------------------------------------------------------------------
# Curriculum training
# ---------------------------------------------------------------------------


def train_curriculum(
    env_class: type,
    species: str,
    stage_configs: Dict[int, Dict[str, Any]],
    algorithm: str = "PPO",
    n_envs: int = 4,
    seed: int = 42,
    eval_freq: int = 10000,
    save_freq: int = 50000,
    log_dir: Optional[str] = None,
    use_subproc: bool = False,
    use_wandb: bool = False,
    clip_reward: float = 10.0,
    reward_keys: Optional[List[str]] = None,
    info_keys: Optional[List[str]] = None,
    verbose: int = 1,
    record_video: bool = True,
    fallback_root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Run the full 3-stage curriculum with automatic advancement.

    Returns a list of per-stage result dicts.
    """
    if fallback_root is None:
        fallback_root = Path.cwd()
    base_dir = resolve_log_dir(log_dir, species, "curriculum", fallback_root)
    base_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Starting automated curriculum training (stages 1-3)")
    logger.info("Species: %s | Algorithm: %s", species, algorithm)
    logger.info("Base directory: %s", base_dir)
    logger.info("=" * 60)

    stage_results_list: List[Dict[str, Any]] = []
    load_path: Optional[str] = None
    vecnorm_path: Optional[str] = None

    for stage in range(1, 4):
        config = stage_configs[stage]
        cur_kwargs = config.get("curriculum_kwargs", {})
        total_timesteps = cur_kwargs.get("timesteps", 500000)

        stage_dir = base_dir / f"stage{stage}"

        results = train_stage(
            env_class=env_class,
            species=species,
            stage_configs=stage_configs,
            stage=stage,
            total_timesteps=total_timesteps,
            algorithm=algorithm,
            n_envs=n_envs,
            seed=seed,
            load_path=load_path,
            eval_freq=eval_freq,
            save_freq=save_freq,
            log_dir=str(stage_dir),
            use_subproc=use_subproc,
            use_wandb=use_wandb,
            clip_reward=clip_reward,
            reward_keys=reward_keys,
            info_keys=info_keys,
            verbose=verbose,
            vecnorm_path=vecnorm_path,
            fallback_root=fallback_root,
        )

        stage_results_list.append(results)

        # Carry forward model and VecNormalize stats
        load_path = results["model_path"]
        vecnorm_path = results["vecnorm_path"]

        # Record video
        if record_video:
            AlgoClass = _ALGO_CLASS[algorithm.upper()]
            model = AlgoClass.load(load_path)
            record_stage_video(
                model, env_class, stage_configs, stage, species,
                results["stage_dir"], algorithm, seed,
                vecnorm_path=vecnorm_path,
            )

    # Post-curriculum outputs
    write_training_summary(base_dir, stage_results_list, species, algorithm, seed, n_envs)
    save_results_json(base_dir, stage_results_list, species, algorithm, seed)

    # Plot training curves
    stage_dirs = [(r["stage"], r["stage_dir"]) for r in stage_results_list]
    plot_training_curves(
        stage_dirs, stage_configs, algorithm, species,
        save_path=str(base_dir / "training_curves.png"),
    )

    logger.info("=" * 60)
    logger.info("Curriculum training complete!")
    logger.info("Results directory: %s", base_dir)
    logger.info("=" * 60)

    return stage_results_list


# ---------------------------------------------------------------------------
# Video recording
# ---------------------------------------------------------------------------


def record_stage_video(
    model,
    env_class: type,
    stage_configs: Dict[int, Dict[str, Any]],
    stage: int,
    species: str,
    stage_dir: str,
    algorithm: str,
    seed: int,
    vecnorm_path: Optional[str] = None,
    max_steps: int = 1000,
) -> None:
    """Record and save a video of the trained policy for a given stage."""
    if not _HAS_MEDIAPY:
        logger.info("Skipping video for stage %d (mediapy not installed).", stage)
        return

    env_kwargs = stage_configs[stage]["env_kwargs"].copy()
    render_env = env_class(render_mode="rgb_array", **env_kwargs)

    vec_normalize = None
    if vecnorm_path and Path(vecnorm_path).exists():
        dummy_env = DummyVecEnv([lambda: env_class(**env_kwargs)])
        vec_normalize = VecNormalize.load(vecnorm_path, dummy_env)
        vec_normalize.training = False
        vec_normalize.norm_reward = False

    obs, _ = render_env.reset(seed=seed + 2000 + stage)
    frames: List[Any] = []
    episode_reward = 0.0

    for _ in range(max_steps):
        obs_input = vec_normalize.normalize_obs(obs) if vec_normalize is not None else obs
        action, _ = model.predict(obs_input, deterministic=True)
        obs, reward, terminated, truncated, info = render_env.step(action)
        frames.append(render_env.render())
        episode_reward += reward
        if terminated or truncated:
            break

    render_env.close()
    if vec_normalize is not None:
        vec_normalize.close()

    if frames:
        video_path = str(Path(stage_dir) / f"{species}_{algorithm.lower()}_stage{stage}.mp4")
        mediapy.write_video(video_path, frames, fps=50)
        logger.info(
            "Stage %d video: reward=%.2f | %d frames -> %s",
            stage, episode_reward, len(frames), video_path,
        )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_training_curves(
    stage_dirs: List[tuple],
    stage_configs: Dict[int, Dict[str, Any]],
    algorithm: str,
    species: str,
    save_path: Optional[str] = None,
) -> None:
    """Plot evaluation reward, episode length, tilt angle, and forward velocity.

    Produces a 2x2 grid saved to *save_path* (if provided).
    """
    if not _HAS_MATPLOTLIB:
        logger.info("Skipping training curves (matplotlib not installed).")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    species_title = species.replace("_", " ").title()

    for stage_num, stage_dir in stage_dirs:
        eval_log = Path(stage_dir) / "evaluations.npz"
        if not eval_log.exists():
            logger.info("No evaluation log found for stage %d.", stage_num)
            continue

        data = np.load(eval_log)
        timesteps = data["timesteps"]
        results = data["results"]
        label = f"Stage {stage_num}: {stage_configs[stage_num]['name']}"

        # Reward curve
        mean_rewards = np.mean(results, axis=1)
        std_rewards = np.std(results, axis=1)
        axes[0, 0].plot(timesteps, mean_rewards, label=label)
        axes[0, 0].fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)

        # Episode length
        if "ep_lengths" in data:
            ep_lengths = data["ep_lengths"]
            mean_lengths = np.mean(ep_lengths, axis=1)
            std_lengths = np.std(ep_lengths, axis=1)
            axes[0, 1].plot(timesteps, mean_lengths, label=label)
            axes[0, 1].fill_between(timesteps, mean_lengths - std_lengths, mean_lengths + std_lengths, alpha=0.2)

        # Curriculum threshold lines
        cur = stage_configs[stage_num].get("curriculum_kwargs", {})
        min_reward = cur.get("min_avg_reward")
        min_length = cur.get("min_avg_episode_length")
        color = axes[0, 0].get_lines()[-1].get_color()
        if min_reward is not None:
            axes[0, 0].axhline(y=min_reward, color=color, linestyle="--", alpha=0.5)
        if min_length is not None:
            axes[0, 1].axhline(y=min_length, color=color, linestyle="--", alpha=0.5)

        # Diagnostics (tilt angle and forward velocity)
        diag_log = Path(stage_dir) / "diagnostics.npz"
        if diag_log.exists():
            diag = np.load(diag_log)
            if "tilt_angle" in diag and "timesteps" in diag:
                diag_ts = diag["timesteps"]
                tilt = np.degrees(diag["tilt_angle"])
                axes[1, 0].plot(diag_ts, tilt, label=label, color=color)
            if "forward_vel" in diag and "timesteps" in diag:
                diag_ts = diag["timesteps"]
                fwd_vel = diag["forward_vel"]
                axes[1, 1].plot(diag_ts, fwd_vel, label=label, color=color)

    axes[0, 0].set(xlabel="Timesteps", ylabel="Mean Reward", title=f"{species_title} {algorithm} - Eval Reward")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set(xlabel="Timesteps", ylabel="Mean Episode Length", title=f"{species_title} {algorithm} - Episode Length")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set(xlabel="Timesteps", ylabel="Mean Tilt Angle (deg)", title=f"{species_title} {algorithm} - Tilt Angle")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set(xlabel="Timesteps", ylabel="Mean Forward Vel (m/s)", title=f"{species_title} {algorithm} - Speed")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Training curves saved to: %s", save_path)

    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary / reporting
# ---------------------------------------------------------------------------


def write_stage_summary(
    stage_dir: Path,
    results: Dict[str, Any],
    species: str,
    algorithm: str,
) -> None:
    """Write a text summary for a single completed stage."""
    summary_path = Path(stage_dir) / "stage_summary.txt"
    mean_len = results.get("mean_episode_length", 0)
    std_len = results.get("std_episode_length", 0)
    sim_dt = results.get("sim_dt", 0.01)
    mean_vel = results.get("mean_forward_vel", 0.0)
    std_vel = results.get("std_forward_vel", 0.0)
    lines = [
        f"Mesozoic Labs: Stage {results['stage']} Summary",
        "=" * 50,
        "",
        f"Species:        {species}",
        f"Stage:          {results['stage']} ({results['name']})",
        f"Description:    {results['description']}",
        f"Algorithm:      {algorithm}",
        f"Date:           {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Timesteps:      {results['timesteps']:,}",
        f"Duration:       {_format_duration(results['duration_seconds'])}",
        f"Eval reward:    {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}",
        f"Avg ep length:  {mean_len:.1f} +/- {std_len:.1f} steps ({mean_len * sim_dt:.2f}s sim time)",
        f"Avg fwd vel:    {mean_vel:.2f} +/- {std_vel:.2f} m/s",
        f"Best model:     {results['model_path']}.zip",
        f"VecNormalize:   {results['vecnorm_path']}",
        "",
    ]
    summary_path.write_text("\n".join(lines) + "\n")
    logger.info("Stage %d summary saved to: %s", results["stage"], summary_path)


def write_training_summary(
    run_dir: Path,
    stage_results_list: List[Dict[str, Any]],
    species: str,
    algorithm: str,
    seed: int,
    n_envs: int,
) -> None:
    """Write a training summary text file to the run directory."""
    summary_path = Path(run_dir) / "training_summary.txt"
    total_duration = sum(r["duration_seconds"] for r in stage_results_list)

    lines = [
        "Mesozoic Labs Training Summary",
        "=" * 50,
        "",
        f"Species:        {species}",
        f"Algorithm:      {algorithm}",
        f"Date:           {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Seed:           {seed}",
        f"Parallel envs:  {n_envs}",
        f"Run directory:  {run_dir}",
        "",
    ]

    for r in stage_results_list:
        mean_len = r.get("mean_episode_length", 0)
        std_len = r.get("std_episode_length", 0)
        sim_dt = r.get("sim_dt", 0.01)
        mean_vel = r.get("mean_forward_vel", 0.0)
        std_vel = r.get("std_forward_vel", 0.0)
        lines.extend([
            f"Stage {r['stage']}: {r['name']}",
            f"  Description:    {r['description']}",
            f"  Timesteps:      {r['timesteps']:,}",
            f"  Duration:       {_format_duration(r['duration_seconds'])}",
            f"  Eval reward:    {r['mean_reward']:.2f} +/- {r['std_reward']:.2f}",
            f"  Avg ep length:  {mean_len:.1f} +/- {std_len:.1f} steps ({mean_len * sim_dt:.2f}s sim time)",
            f"  Avg fwd vel:    {mean_vel:.2f} +/- {std_vel:.2f} m/s",
            f"  Best model:     {r['model_path']}.zip",
            "",
        ])

    lines.extend([
        "-" * 50,
        f"Total training time: {_format_duration(total_duration)}",
    ])

    summary_path.write_text("\n".join(lines) + "\n")
    logger.info("Training summary saved to: %s", summary_path)


def save_results_json(
    run_dir: Path,
    stage_results_list: List[Dict[str, Any]],
    species: str,
    algorithm: str,
    seed: int,
) -> Path:
    """Save a machine-readable summary.json to the run directory."""
    total_duration = sum(r["duration_seconds"] for r in stage_results_list)
    total_timesteps = sum(r["timesteps"] for r in stage_results_list)
    final_result = stage_results_list[-1]

    stages: Dict[str, Any] = {}
    for r in stage_results_list:
        stage_data: Dict[str, Any] = {
            "name": r["name"],
            "timesteps": r["timesteps"],
            "avg_reward": round(r["mean_reward"], 2),
            "std_reward": round(r["std_reward"], 2),
            "training_time_seconds": round(r["duration_seconds"], 1),
            "training_time": _format_duration_hms(r["duration_seconds"]),
        }
        if "mean_forward_vel" in r:
            stage_data["avg_forward_vel"] = round(r["mean_forward_vel"], 2)
        stages[str(r["stage"])] = stage_data

    summary = {
        "species": species,
        "algorithm": algorithm,
        "seed": seed,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "stages": stages,
        "total_timesteps": total_timesteps,
        "total_training_time_seconds": round(total_duration, 1),
        "total_training_time": _format_duration_hms(total_duration),
        "final_avg_reward": round(final_result["mean_reward"], 2),
    }

    summary_path = Path(run_dir) / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("Results JSON saved to: %s", summary_path)
    return summary_path


# ---------------------------------------------------------------------------
# CLI argument builder
# ---------------------------------------------------------------------------


def add_common_args(parser) -> None:
    """Add common CLI arguments shared by all species training scripts."""
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a single stage")
    train_parser.add_argument("--stage", type=int, choices=[1, 2, 3], default=1, help="Curriculum stage")
    train_parser.add_argument("--timesteps", type=int, default=500000, help="Total training timesteps")
    train_parser.add_argument("--algo", type=str, choices=["PPO", "SAC"], default="PPO", help="Algorithm")
    train_parser.add_argument("--n-envs", type=int, default=4, help="Number of parallel environments")
    train_parser.add_argument("--load", type=str, default=None, help="Path to model checkpoint")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    train_parser.add_argument("--eval-freq", type=int, default=10000, help="Evaluation frequency (timesteps)")
    train_parser.add_argument("--save-freq", type=int, default=50000, help="Checkpoint save frequency")
    train_parser.add_argument("--log-dir", type=str, default=None, help="Custom log directory")
    train_parser.add_argument("--subproc", action="store_true", help="Use subprocess vectorization")
    train_parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    train_parser.add_argument(
        "--verbose", type=int, choices=[0, 1, 2], default=1,
        help="0=eval only, 1=training stats (default), 2=debug",
    )

    # Curriculum command
    cur_parser = subparsers.add_parser("curriculum", help="Run automated 3-stage curriculum")
    cur_parser.add_argument("--algo", type=str, choices=["PPO", "SAC"], default="PPO", help="Algorithm")
    cur_parser.add_argument("--n-envs", type=int, default=4, help="Number of parallel environments")
    cur_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    cur_parser.add_argument("--eval-freq", type=int, default=10000, help="Evaluation frequency")
    cur_parser.add_argument("--save-freq", type=int, default=50000, help="Checkpoint save frequency")
    cur_parser.add_argument("--log-dir", type=str, default=None, help="Custom log directory")
    cur_parser.add_argument("--subproc", action="store_true", help="Use subprocess vectorization")
    cur_parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    cur_parser.add_argument("--no-video", action="store_true", help="Skip video recording")
    cur_parser.add_argument(
        "--verbose", type=int, choices=[0, 1, 2], default=1,
        help="0=eval only, 1=training stats (default), 2=debug",
    )

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained policy")
    eval_parser.add_argument("model_path", type=str, help="Path to trained model")
    eval_parser.add_argument("--algo", type=str, choices=["PPO", "SAC"], default="PPO", help="Algorithm used")
    eval_parser.add_argument("--stage", type=int, choices=[1, 2, 3], default=None, help="Curriculum stage")
    eval_parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    eval_parser.add_argument("--no-render", action="store_true", help="Disable rendering")


def evaluate_cli(
    env_class: type,
    stage_configs: Dict[int, Dict[str, Any]],
    model_path: str,
    algorithm: str = "PPO",
    stage: Optional[int] = None,
    n_episodes: int = 10,
    render: bool = True,
) -> None:
    """Evaluate a trained model (CLI entry point)."""
    if stage is None:
        stage = 1
        for s in [1, 2, 3]:
            if f"stage{s}" in model_path:
                stage = s
                break
        logger.info("Auto-detected stage %d from filename", stage)

    env_kwargs = stage_configs[stage]["env_kwargs"].copy()

    vecnorm_path = model_path.replace(".zip", "_vecnorm.pkl")
    if not vecnorm_path.endswith("_vecnorm.pkl"):
        vecnorm_path = model_path + "_vecnorm.pkl"

    render_mode = "human" if render else None

    def _make_eval_env():
        env = env_class(render_mode=render_mode, **env_kwargs)
        return Monitor(env)

    vec_env = DummyVecEnv([_make_eval_env])

    if Path(vecnorm_path).exists():
        logger.info("Loading normalization stats from: %s", vecnorm_path)
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        logger.warning("No VecNormalize stats found. Results may differ from training.")

    AlgoClass = _ALGO_CLASS[algorithm.upper()]
    model = AlgoClass.load(model_path, env=vec_env)

    logger.info(
        "Evaluating for %d episodes (stage %d: %s)...",
        n_episodes, stage, stage_configs[stage]["name"],
    )

    episode_rewards: List[float] = []
    episode_lengths: List[int] = []

    for ep in range(n_episodes):
        obs = vec_env.reset()
        total_reward = 0.0
        step = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(action)
            total_reward += rewards[0]
            step += 1
            if dones[0]:
                break
        episode_rewards.append(total_reward)
        episode_lengths.append(step)
        logger.info("  Episode %d: reward=%.2f, length=%d", ep + 1, total_reward, step)

    vec_env.close()

    logger.info("Results:")
    logger.info("  Mean reward: %.2f +/- %.2f", np.mean(episode_rewards), np.std(episode_rewards))
    logger.info("  Mean length: %.1f +/- %.1f", np.mean(episode_lengths), np.std(episode_lengths))
