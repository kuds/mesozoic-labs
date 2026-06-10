"""Ray Tune sweep orchestration: Tuner setup, trial discovery, and result export.

Provides the high-level orchestration functions that the notebook (or a future
GCP Ray job script) calls to run and analyze a Ray Tune sweep:

- ``create_ray_tuner``: Build a configured ``tune.Tuner`` with scheduler, callbacks,
  and merged search space + fixed config.
- ``run_ray_sweep``: End-to-end sweep execution (create or restore Tuner, fit,
  sync final state to Drive).
- ``discover_and_rank_trials``: Find completed trial directories and rank them
  by reward (from evaluations.npz or fallback quick-eval).
- ``export_best_trial``: Save the best trial's config JSON and copy its model
  files to a convenient location.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tuner creation
# ---------------------------------------------------------------------------


def create_ray_tuner(
    *,
    train_fn: Any,
    search_space: dict[str, Any],
    species: str,
    algorithm: str,
    stage: int,
    timesteps: int,
    n_envs: int,
    seed: int,
    eval_freq: int,
    num_trials: int,
    max_concurrent: int,
    local_ray_dir: str | Path,
    local_trials_dir: str | Path,
    sweep_dir: str | Path,
    load_path: str = "",
    collapse_min_evals: int = 8,
    collapse_patience: int = 5,
    n_eval_episodes: int = 30,
    use_asha: bool = True,
    grace_period: int = 30,
    reduction_factor: int = 2,
    report_interval_s: int = 300,
    sync_interval_s: int = 300,
) -> Any:
    """Build a configured ``ray.tune.Tuner`` ready to call ``.fit()``.

    Parameters
    ----------
    train_fn:
        The Ray Tune trainable (typically ``train_trial``).
    search_space:
        Dict of Ray Tune search space params (from ``to_ray_tune()``).
    species, algorithm, stage:
        Sweep identity (used for experiment naming and fixed config).
    timesteps, n_envs, seed, eval_freq:
        Training budget and evaluation settings.
    num_trials, max_concurrent:
        Ray Tune trial budget and parallelism.
    local_ray_dir:
        Local path for Ray Tune internal storage (fast, non-FUSE).
    local_trials_dir:
        Local path for per-trial output directories.
    sweep_dir:
        Drive/persistent path for results, progress logs, and syncs.
    load_path:
        Optional warm-start checkpoint path.
    collapse_min_evals, collapse_patience:
        Reward-collapse early stopping settings.
    n_eval_episodes:
        Number of episodes per evaluation round (default 30; use 15 on T4).
    use_asha:
        If True, use ASHA scheduler; otherwise FIFO (no pruning).
    grace_period, reduction_factor:
        ASHA scheduler settings (ignored when ``use_asha=False``).
    report_interval_s:
        How often TrialTerminationCallback prints status summaries.
    sync_interval_s:
        How often ExperimentStateSyncCallback syncs state to Drive.

    Returns
    -------
    ray.tune.Tuner
        A configured Tuner. Call ``.fit()`` to start the sweep.
    """
    from ray import tune
    from ray.tune import CheckpointConfig as TuneCheckpointConfig
    from ray.tune import RunConfig as TuneRunConfig
    from ray.tune.schedulers import ASHAScheduler, FIFOScheduler

    from .ray_tune import (
        DriveProgressLogCallback,
        ExperimentStateSyncCallback,
        TrialTerminationCallback,
    )

    local_ray_dir = Path(local_ray_dir)
    sweep_dir = Path(sweep_dir)

    # Scheduler
    max_reports = timesteps // eval_freq
    if use_asha:
        scheduler = ASHAScheduler(
            metric="best_mean_reward",
            mode="max",
            max_t=max_reports,
            grace_period=grace_period,
            reduction_factor=reduction_factor,
        )
    else:
        scheduler = FIFOScheduler()

    # Callbacks
    experiment_name = f"{species}_stage{stage}_{algorithm}"
    local_experiment_dir = local_ray_dir / experiment_name

    trial_callback = TrialTerminationCallback(report_interval_s=report_interval_s)
    state_sync_callback = ExperimentStateSyncCallback(
        local_experiment_dir=local_experiment_dir,
        drive_ray_results_dir=sweep_dir / "ray_results",
        sync_interval_s=sync_interval_s,
    )
    progress_log_callback = DriveProgressLogCallback(drive_sweep_dir=sweep_dir)

    # Resource allocation: fractional GPU per trial
    gpu_fraction = 1.0 / max(max_concurrent, 1)
    trainable = tune.with_resources(train_fn, {"cpu": 2, "gpu": gpu_fraction})

    # Fixed parameters passed to every trial
    fixed_config = {
        "_species": species,
        "_algorithm": algorithm,
        "_stage": stage,
        "_timesteps": timesteps,
        "_n_envs": n_envs,
        "_seed": seed,
        "_eval_freq": eval_freq,
        "_load_path": load_path,
        "_local_trials_dir": str(local_trials_dir),
        "_drive_sweep_dir": str(sweep_dir),
        "_collapse_min_evals": collapse_min_evals,
        "_collapse_patience": collapse_patience,
        "_n_eval_episodes": n_eval_episodes,
    }
    full_config = {**fixed_config, **search_space}

    tuner = tune.Tuner(
        trainable,
        param_space=full_config,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=num_trials,
            max_concurrent_trials=max_concurrent,
        ),
        run_config=TuneRunConfig(
            name=experiment_name,
            storage_path=str(local_ray_dir),
            checkpoint_config=TuneCheckpointConfig(num_to_keep=2),
            verbose=0,
            callbacks=[trial_callback, state_sync_callback, progress_log_callback],
        ),
    )

    return tuner


# ---------------------------------------------------------------------------
# End-to-end sweep execution
# ---------------------------------------------------------------------------


def run_ray_sweep(
    *,
    train_fn: Any,
    search_space: dict[str, Any],
    species: str,
    algorithm: str,
    stage: int,
    timesteps: int,
    n_envs: int,
    seed: int,
    eval_freq: int,
    num_trials: int,
    max_concurrent: int,
    local_ray_dir: str | Path,
    local_trials_dir: str | Path,
    sweep_dir: str | Path,
    load_path: str = "",
    collapse_min_evals: int = 8,
    collapse_patience: int = 5,
    n_eval_episodes: int = 30,
    use_asha: bool = True,
    grace_period: int = 30,
    reduction_factor: int = 2,
    resume: bool = False,
    **kwargs: Any,
) -> Any:
    """Run a complete Ray Tune sweep, optionally resuming a prior run.

    Creates (or restores) a Tuner, calls ``.fit()``, and syncs final
    experiment state to Drive.

    Parameters
    ----------
    resume:
        If True and a local experiment directory exists, restore the Tuner
        from it (completed trials kept, partial trials restart).
    **kwargs:
        Forwarded to ``create_ray_tuner`` (e.g. ``report_interval_s``).

    Returns
    -------
    ray.tune.ResultGrid
        The results from ``tuner.fit()``.
    """
    from ray import tune

    local_ray_dir = Path(local_ray_dir)
    sweep_dir = Path(sweep_dir)

    experiment_name = f"{species}_stage{stage}_{algorithm}"
    local_experiment_dir = local_ray_dir / experiment_name
    drive_ray_results_dir = sweep_dir / "ray_results"

    # Resource allocation for restore path
    gpu_fraction = 1.0 / max(max_concurrent, 1)
    trainable = tune.with_resources(train_fn, {"cpu": 2, "gpu": gpu_fraction})

    # Fixed config (needed for both fresh and restore paths)
    fixed_config = {
        "_species": species,
        "_algorithm": algorithm,
        "_stage": stage,
        "_timesteps": timesteps,
        "_n_envs": n_envs,
        "_seed": seed,
        "_eval_freq": eval_freq,
        "_load_path": load_path,
        "_local_trials_dir": str(local_trials_dir),
        "_drive_sweep_dir": str(sweep_dir),
        "_collapse_min_evals": collapse_min_evals,
        "_collapse_patience": collapse_patience,
        "_n_eval_episodes": n_eval_episodes,
    }
    full_config = {**fixed_config, **search_space}

    restored = False
    if resume and local_experiment_dir.exists():
        logger.info("Restoring Tuner from: %s", local_experiment_dir)
        tuner = tune.Tuner.restore(
            path=str(local_experiment_dir),
            trainable=trainable,
            resume_unfinished=True,
            resume_errored=True,
            param_space=full_config,
        )
        restored = True
    elif resume:
        logger.warning(
            "RESUME=True but no experiment state found at %s. Starting a fresh sweep instead.",
            local_experiment_dir,
        )

    if not restored:
        tuner = create_ray_tuner(
            train_fn=train_fn,
            search_space=search_space,
            species=species,
            algorithm=algorithm,
            stage=stage,
            timesteps=timesteps,
            n_envs=n_envs,
            seed=seed,
            eval_freq=eval_freq,
            num_trials=num_trials,
            max_concurrent=max_concurrent,
            local_ray_dir=local_ray_dir,
            local_trials_dir=local_trials_dir,
            sweep_dir=sweep_dir,
            load_path=load_path,
            collapse_min_evals=collapse_min_evals,
            collapse_patience=collapse_patience,
            n_eval_episodes=n_eval_episodes,
            use_asha=use_asha,
            grace_period=grace_period,
            reduction_factor=reduction_factor,
            **kwargs,
        )

    results = tuner.fit()

    # Final incremental sync of experiment state to Drive.
    # ExperimentStateSyncCallback already synced periodically during the
    # sweep, so most files are already on Drive.  Only copy files that are
    # newer (by mtime) to avoid redundant I/O on the FUSE mount.
    # Uses retry logic for transient FUSE errors.
    if local_experiment_dir.exists():
        from .ray_tune import _copy_to_drive

        drive_dest = drive_ray_results_dir / local_experiment_dir.name
        logger.info("Final incremental sync to Drive: %s", drive_dest)
        try:
            copied = 0
            for src_file in local_experiment_dir.rglob("*"):
                if not src_file.is_file():
                    continue
                rel = src_file.relative_to(local_experiment_dir)
                dst = drive_dest / rel
                if dst.exists() and dst.stat().st_mtime >= src_file.stat().st_mtime:
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                if _copy_to_drive(src_file, dst):
                    copied += 1
            logger.info("Final sync complete: %d files copied", copied)
        except OSError as e:
            logger.warning("Drive sync of Ray results failed: %s", e)

    return results


# ---------------------------------------------------------------------------
# Trial discovery & ranking
# ---------------------------------------------------------------------------


def discover_and_rank_trials(
    sweep_dir: str | Path,
    *,
    species_cfg: Any = None,
    algorithm: str = "ppo",
    stage_configs: dict[int, dict[str, Any]] | None = None,
    stage: int = 1,
    top_k: int = 5,
) -> list[tuple[Path, float]]:
    """Find completed trial directories and rank them by best mean reward.

    Scans ``sweep_dir/trials/`` for directories containing a
    ``models/best_model.zip``. Ranks by peak mean reward from
    ``evaluations.npz`` when available, falling back to a quick 1-episode
    evaluation when most trials are missing the npz file (common after
    Colab session crashes where Drive sync was incomplete).

    Parameters
    ----------
    sweep_dir:
        Root sweep directory (contains ``trials/`` subdirectory).
    species_cfg:
        Species config (needed only for fallback quick-eval ranking).
    algorithm:
        Algorithm name (needed only for fallback quick-eval ranking).
    stage_configs:
        Stage configs dict (needed only for fallback quick-eval ranking).
    stage:
        Curriculum stage (needed only for fallback quick-eval ranking).
    top_k:
        Maximum number of top trials to return.

    Returns
    -------
    list[tuple[Path, float]]
        List of ``(trial_dir, best_reward)`` tuples, sorted best-first.
        Length is ``min(top_k, num_valid_trials)``.
    """
    trials_root = Path(sweep_dir) / "trials"
    if not trials_root.exists():
        raise FileNotFoundError(f"No trials directory found at {trials_root}")

    trial_dirs = sorted(trials_root.iterdir())
    valid_trials: list[tuple[Path, float]] = []
    missing_npz_count = 0

    for td in trial_dirs:
        if td.is_dir() and (td / "models" / "best_model.zip").exists():
            eval_npz = td / "evaluations.npz"
            reward = float("-inf")
            if eval_npz.exists():
                eval_data = np.load(str(eval_npz))
                mean_per_eval = eval_data["results"].mean(axis=1)
                reward = float(mean_per_eval.max())
            else:
                missing_npz_count += 1
            valid_trials.append((td, reward))

    if not valid_trials:
        logger.warning("No trials with saved models found in %s", trials_root)
        return []

    # If most trials are missing evaluations.npz, run quick 1-episode evals
    have_npz = sum(1 for _, r in valid_trials if r != float("-inf"))
    if have_npz < len(valid_trials) // 2 and species_cfg is not None and stage_configs is not None:
        logger.info(
            "%d/%d trials missing evaluations.npz. Running quick 1-episode ranking...",
            missing_npz_count,
            len(valid_trials),
        )
        valid_trials = _quick_rank_trials(
            valid_trials,
            species_cfg=species_cfg,
            algorithm=algorithm,
            stage_configs=stage_configs,
            stage=stage,
        )
    elif missing_npz_count > 0:
        logger.info("%d trial(s) missing evaluations.npz — ranked last.", missing_npz_count)

    valid_trials.sort(key=lambda x: x[1], reverse=True)
    return valid_trials[:top_k]


def _quick_rank_trials(
    trials: list[tuple[Path, float]],
    *,
    species_cfg: Any,
    algorithm: str,
    stage_configs: dict[int, dict[str, Any]],
    stage: int,
) -> list[tuple[Path, float]]:
    """Run quick 1-episode evals in parallel using Ray tasks.

    Trials that already have a reward from evaluations.npz are kept as-is.
    Only trials with reward == -inf are evaluated. Uses Ray remote tasks
    to parallelize across available CPUs.
    """
    import ray

    need_eval = [(td, reward) for td, reward in trials if reward == float("-inf")]
    have_reward = [(td, reward) for td, reward in trials if reward != float("-inf")]

    if not need_eval:
        return trials

    env_kwargs = stage_configs[stage]["env_kwargs"].copy()

    @ray.remote(num_cpus=1, num_gpus=0)
    def _eval_one_trial(model_dir: str, vecnorm_path: str, alg: str, env_kw: dict, species: str) -> float:
        """Evaluate a single trial for 1 episode inside a Ray task."""
        import os

        os.environ["MUJOCO_GL"] = "egl"
        from stable_baselines3 import PPO, SAC

        from environments.shared.species_registry import get_species_config
        from environments.shared.train_base import _ensure_sb3

        sb3 = _ensure_sb3()
        sp_cfg = get_species_config(species)
        alg_cls = SAC if alg == "sac" else PPO

        def _mk():
            return sb3["Monitor"](sp_cfg.env_class(**env_kw))

        ev = sb3["DummyVecEnv"]([_mk])
        vn_path = Path(vecnorm_path)
        if vn_path.exists():
            ev = sb3["VecNormalize"].load(str(vn_path), ev)
            ev.training = False
            ev.norm_reward = False
        m = alg_cls.load(model_dir, env=ev)
        obs = ev.reset()
        total = 0.0
        while True:
            action, _ = m.predict(obs, deterministic=True)
            obs, rews, dones, _ = ev.step(action)
            total += float(rews[0])
            if dones[0]:
                break
        ev.close()
        return total

    # Launch all eval tasks in parallel
    futures = {}
    for td, _ in need_eval:
        model_path = str(td / "models" / "best_model")
        vecnorm_path = str(td / "models" / "best_model_vecnorm.pkl")
        ref = _eval_one_trial.remote(
            model_path,
            vecnorm_path,
            algorithm,
            env_kwargs,
            species_cfg.species_name,
        )
        futures[ref] = td

    # Collect results
    reranked = list(have_reward)
    for ref, td in futures.items():
        try:
            reward = ray.get(ref)
            reranked.append((td, reward))
        except Exception as e:
            logger.warning("Could not eval %s: %s", td.name, e)
            reranked.append((td, float("-inf")))

    return reranked


# ---------------------------------------------------------------------------
# Parallel post-sweep evaluation
# ---------------------------------------------------------------------------


def evaluate_trials_parallel(
    top_trials: list[tuple[Path, float]],
    *,
    species: str,
    algorithm: str,
    stage: int,
    stage_config: dict[str, Any],
    n_eval_episodes: int = 30,
) -> list[dict[str, Any]]:
    """Evaluate multiple trials in parallel using Ray tasks.

    Each trial runs ``n_eval_episodes`` episodes with full LocomotionMetrics
    collection. Uses Ray remote tasks to parallelize across CPUs, which is
    significantly faster than sequential evaluation for multiple trials.

    Parameters
    ----------
    top_trials:
        List of ``(trial_dir, sweep_reward)`` tuples to evaluate.
    species:
        Species name (for loading species config inside Ray tasks).
    algorithm:
        Algorithm name (``"ppo"`` or ``"sac"``).
    stage:
        Curriculum stage number.
    stage_config:
        Stage configuration dict (for env_kwargs).
    n_eval_episodes:
        Number of evaluation episodes per trial.

    Returns
    -------
    list[dict[str, Any]]
        One row dict per trial with evaluation metrics, in the same order
        as ``top_trials``.
    """
    import ray

    env_kwargs = stage_config["env_kwargs"].copy()

    @ray.remote(num_cpus=1, num_gpus=0)
    def _eval_trial(
        trial_dir: str,
        model_path: str,
        vecnorm_path: str,
        alg: str,
        species_name: str,
        env_kw: dict,
        n_episodes: int,
        rank: int,
        sweep_reward: float,
    ) -> dict[str, Any]:
        """Evaluate one trial with full LocomotionMetrics inside a Ray task."""
        import os

        os.environ["MUJOCO_GL"] = "egl"

        from pathlib import Path as _Path

        from stable_baselines3 import PPO, SAC

        from environments.shared.metrics import LocomotionMetrics, env_dt
        from environments.shared.species_registry import get_species_config
        from environments.shared.train_base import _ensure_sb3

        sb3 = _ensure_sb3()
        sp_cfg = get_species_config(species_name)
        alg_cls = SAC if alg == "sac" else PPO

        def _mk():
            return sb3["Monitor"](sp_cfg.env_class(**env_kw))

        eval_env = sb3["DummyVecEnv"]([_mk])
        vn = _Path(vecnorm_path)
        if vn.exists():
            eval_env = sb3["VecNormalize"].load(str(vn), eval_env)
            eval_env.training = False
            eval_env.norm_reward = False

        model = alg_cls.load(model_path, env=eval_env)

        episode_reports = []
        eval_dt = env_dt(eval_env)
        for _ in range(n_episodes):
            obs = eval_env.reset()
            metrics = LocomotionMetrics(dt=eval_dt)
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = eval_env.step(action)
                metrics.record_step(infos[0], float(rewards[0]))
                if dones[0]:
                    break
            episode_reports.append(metrics.compute())

        eval_env.close()
        agg = LocomotionMetrics.aggregate_episodes(episode_reports)

        return {
            "rank": rank,
            "trial": _Path(trial_dir).name,
            "sweep_reward": round(sweep_reward, 2),
            "eval_reward": round(agg.get("mean_total_reward", 0), 2),
            "eval_reward_std": round(agg.get("std_total_reward", 0), 2),
            "ep_length": round(agg.get("mean_episode_length", 0), 1),
            "std_episode_length": round(agg.get("std_episode_length", 0), 1),
            "fwd_vel_m/s": round(agg.get("mean_mean_forward_velocity", 0), 3),
            "fwd_vel_std": round(agg.get("std_mean_forward_velocity", 0), 3),
            "max_fwd_vel": round(agg.get("mean_max_forward_velocity", 0), 3),
            "distance_m": round(agg.get("mean_total_distance", 0), 3),
            "vel_consistency": round(agg.get("mean_velocity_consistency", 0), 3),
            "gait_symmetry": round(agg.get("mean_gait_symmetry", 0), 3),
            "stride_freq_Hz": round(agg.get("mean_stride_frequency", 0), 3),
            "cost_of_transport": round(agg.get("mean_cost_of_transport", 0), 4),
            "tilt_rad": round(agg.get("mean_mean_tilt_angle", 0), 3),
            "pelvis_height_m": round(agg.get("mean_mean_pelvis_height", 0), 3),
            "mean_success_rate": round(agg.get("mean_success_rate", 0), 4),
        }

    # Launch all evaluations in parallel
    refs = []
    for rank, (trial_dir, sweep_reward) in enumerate(top_trials, 1):
        model_dir = trial_dir / "models"
        ref = _eval_trial.remote(
            str(trial_dir),
            str(model_dir / "best_model"),
            str(model_dir / "best_model_vecnorm.pkl"),
            algorithm,
            species,
            env_kwargs,
            n_eval_episodes,
            rank,
            sweep_reward,
        )
        refs.append(ref)

    # Collect results in submission order (preserves ranking)
    results: list[dict[str, Any]] = ray.get(refs)
    for row in results:
        logger.info(
            "Trial %d/%d %s: reward=%.2f  fwd_vel=%.3f m/s  distance=%.2f m",
            row["rank"],
            len(top_trials),
            row["trial"],
            row["eval_reward"],
            row["fwd_vel_m/s"],
            row["distance_m"],
        )

    return results


# ---------------------------------------------------------------------------
# Best trial export
# ---------------------------------------------------------------------------


def export_best_trial(
    best_result: Any,
    sweep_dir: str | Path,
    *,
    species: str,
    algorithm: str,
    stage: int,
    timesteps: int,
    num_trials: int,
    use_asha: bool = True,
    local_trials_dir: str | Path | None = None,
) -> Path:
    """Save the best trial's config and model files to the sweep directory.

    Writes ``best_trial_config.json`` and copies model files to
    ``sweep_dir/best_model/``.

    Parameters
    ----------
    best_result:
        The ``ray.tune.Result`` from ``results.get_best_result()``.
    sweep_dir:
        Root sweep directory.
    species, algorithm, stage, timesteps, num_trials, use_asha:
        Metadata included in the config JSON.
    local_trials_dir:
        Local trials directory (preferred source for model files).

    Returns
    -------
    Path
        Path to the ``best_model/`` directory.
    """
    sweep_dir = Path(sweep_dir)

    # Save config JSON
    best_config = {k: v for k, v in best_result.config.items() if not k.startswith("_")}
    best_config_with_meta = {
        "species": species,
        "algorithm": algorithm,
        "stage": stage,
        "best_mean_reward": best_result.metrics.get("best_mean_reward"),
        "timesteps_per_trial": timesteps,
        "num_trials": num_trials,
        "scheduler": "ASHA" if use_asha else "FIFO",
        "hyperparameters": best_config,
    }

    best_config_path = sweep_dir / "best_trial_config.json"
    with open(str(best_config_path), "w") as f:
        json.dump(best_config_with_meta, f, indent=2, default=str)
    logger.info("Best trial config saved to: %s", best_config_path)

    # Copy model files
    best_trial_id = best_result.metrics.get("trial_id", "")
    local_model_dir = Path(local_trials_dir) / str(best_trial_id) / "models" if local_trials_dir else None
    drive_model_dir = sweep_dir / "trials" / str(best_trial_id) / "models"

    # Prefer local (always complete), fall back to Drive (synced copy)
    if local_model_dir and local_model_dir.exists():
        source_model_dir = local_model_dir
    else:
        source_model_dir = drive_model_dir

    best_model_dest = sweep_dir / "best_model"
    best_model_dest.mkdir(parents=True, exist_ok=True)

    found_any = False
    for pattern in [
        f"stage{stage}_final.zip",
        f"stage{stage}_final_vecnorm.pkl",
        "best_model.zip",
        "best_model_vecnorm.pkl",
    ]:
        src_files = list(source_model_dir.glob(pattern)) if source_model_dir.exists() else []
        for src in src_files:
            dest = best_model_dest / src.name
            shutil.copy2(str(src), str(dest))
            logger.info("Copied: %s -> %s", src.name, dest)
            found_any = True

    if not found_any:
        logger.warning(
            "No model files found in %s. Check if the best trial completed.",
            source_model_dir,
        )

    return best_model_dest
