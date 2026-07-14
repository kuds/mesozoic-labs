"""Ray Tune sweep orchestration: Tuner setup, trial discovery, and result export.

Provides the high-level orchestration functions that the notebook (or a future
GCP Ray job script) calls to run and analyze a Ray Tune sweep:

- ``create_ray_tuner``: Build a configured ``tune.Tuner`` with scheduler, callbacks,
  and merged search space + fixed config.
- ``run_ray_sweep``: End-to-end sweep execution (create or restore Tuner, fit,
  sync final state to Drive).
- ``discover_and_rank_trials``: Find completed trial directories and rank them
  by reward (from evaluations.npz or fallback quick-eval).
- ``export_best_trial``: Validate and promote the best trial's config and model
  files to a convenient location.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from environments.shared.plant_contract import (
    PlantCompatibilityError,
    PlantIdentity,
    current_plant_identity,
    validate_recorded_identity,
    write_plant_identity,
)

logger = logging.getLogger(__name__)

PLANT_IDENTITY_FILENAME = "plant_identity.json"


def _validate_plant_identity_sidecar(
    path: str | Path,
    current_plant: PlantIdentity,
    *,
    artifact: str,
    allow_legacy_plant: bool = False,
) -> None:
    """Validate a persisted Ray sweep/trial identity sidecar."""
    identity_path = Path(path)
    recorded: dict[str, Any] | None = None
    if identity_path.exists():
        try:
            raw = json.loads(identity_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PlantCompatibilityError(f"{artifact} has an unreadable plant identity: {exc}") from exc
        if not isinstance(raw, dict):
            raise PlantCompatibilityError(f"{artifact} has invalid plant identity metadata: expected a JSON object")
        recorded = raw
    validate_recorded_identity(
        recorded,
        current_plant,
        artifact=artifact,
        allow_legacy=allow_legacy_plant,
    )


def _load_and_validate_promotion_artifacts(
    source_files: list[Path],
    *,
    algorithm: str,
    current_plant: PlantIdentity,
    allow_legacy_plant: bool,
) -> dict[Path, Any]:
    """Deserialize, validate, and retag model/VecNormalize promotion inputs."""
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.save_util import load_from_pkl

    from environments.shared.plant_contract import attach_plant_identity, validate_model_plant

    alg_cls = SAC if algorithm == "sac" else PPO
    loaded: dict[Path, Any] = {}
    for source in source_files:
        if source.suffix == ".zip":
            artifact = alg_cls.load(str(source), device="cpu")
        elif source.suffix == ".pkl":
            # VecNormalize.load() requires a live VecEnv. Its implementation
            # first unpickles the same object and then attaches that env, so
            # the SB3 pickle loader is the minimal promotion-time equivalent.
            artifact = load_from_pkl(source)
            # VecNormalize.__setstate__ deliberately leaves these transient
            # fields absent until set_venv(). They are excluded again by
            # __getstate__, but must exist for a safe promotion-time re-save.
            if "class_attributes" not in artifact.__dict__:
                artifact.class_attributes = {}
            if "returns" not in artifact.__dict__:
                artifact.returns = np.zeros(int(getattr(artifact, "num_envs", 1)))
        else:  # defensive: export patterns currently admit only ZIP and PKL
            raise ValueError(f"unsupported promotion artifact: {source}")
        validate_model_plant(
            artifact,
            current_plant,
            artifact=str(source),
            allow_legacy=allow_legacy_plant,
        )
        # Promotion creates a new artifact. Retag after successful validation
        # so explicit legacy migration and compatible visual-only revisions
        # both produce self-describing outputs.
        attach_plant_identity(artifact, current_plant)
        loaded[source] = artifact
    return loaded


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
    allow_legacy_plant: bool = False,
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
    allow_legacy_plant:
        Explicitly permit untagged warm-start artifacts. Incompatible tagged
        artifacts are always rejected.

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
    current_plant = current_plant_identity(species)
    write_plant_identity(sweep_dir / PLANT_IDENTITY_FILENAME, current_plant)

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
    write_plant_identity(
        sweep_dir / "ray_results" / experiment_name / PLANT_IDENTITY_FILENAME,
        current_plant,
    )

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
        "_allow_legacy_plant": allow_legacy_plant,
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
    allow_legacy_plant: bool = False,
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
    allow_legacy_plant:
        Explicitly permit a pre-contract sweep state or warm-start artifact
        with no plant identity. Incompatible tagged state is always rejected.
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
        "_allow_legacy_plant": allow_legacy_plant,
    }
    full_config = {**fixed_config, **search_space}

    restored = False
    if resume and local_experiment_dir.exists():
        current_plant = current_plant_identity(species)
        sweep_identity_path = sweep_dir / PLANT_IDENTITY_FILENAME
        experiment_identity_path = drive_ray_results_dir / experiment_name / PLANT_IDENTITY_FILENAME
        for identity_path, artifact in (
            (sweep_identity_path, f"Ray sweep state at {sweep_dir}"),
            (experiment_identity_path, f"Ray experiment state {experiment_name}"),
        ):
            _validate_plant_identity_sidecar(
                identity_path,
                current_plant,
                artifact=artifact,
                allow_legacy_plant=allow_legacy_plant,
            )
        # An explicitly accepted legacy state becomes tagged before any
        # resumed trial can produce another artifact.
        write_plant_identity(sweep_identity_path, current_plant)
        write_plant_identity(experiment_identity_path, current_plant)
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
            allow_legacy_plant=allow_legacy_plant,
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
    allow_legacy_plant: bool = False,
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
    allow_legacy_plant:
        Explicitly permit untagged legacy artifacts during fallback
        evaluation. Incompatible tagged artifacts are always rejected.

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
            allow_legacy_plant=allow_legacy_plant,
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
    allow_legacy_plant: bool,
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
    def _eval_one_trial(
        model_dir: str,
        vecnorm_path: str,
        alg: str,
        env_kw: dict,
        species: str,
        allow_legacy: bool,
    ) -> float:
        """Evaluate a single trial for 1 episode inside a Ray task."""
        import os

        os.environ["MUJOCO_GL"] = "egl"
        from stable_baselines3 import PPO, SAC

        from environments.shared.plant_contract import (
            current_plant_identity,
            validate_environment_plant,
            validate_model_plant,
        )
        from environments.shared.species_registry import get_species_config
        from environments.shared.train_base import _ensure_sb3

        sb3 = _ensure_sb3()
        sp_cfg = get_species_config(species)
        current_plant = current_plant_identity(species)
        alg_cls = SAC if alg == "sac" else PPO

        def _mk():
            raw_env = sp_cfg.env_class(**env_kw)
            try:
                validate_environment_plant(raw_env, current_plant, artifact=f"{species} Ray evaluation environment")
            except Exception:
                raw_env.close()
                raise
            return sb3["Monitor"](raw_env)

        ev = sb3["DummyVecEnv"]([_mk])
        vn_path = Path(vecnorm_path)
        if vn_path.exists():
            ev = sb3["VecNormalize"].load(str(vn_path), ev)
            validate_model_plant(
                ev,
                current_plant,
                artifact=str(vn_path),
                allow_legacy=allow_legacy,
            )
            ev.training = False
            ev.norm_reward = False
        m = alg_cls.load(model_dir, env=ev)
        validate_model_plant(
            m,
            current_plant,
            artifact=model_dir,
            allow_legacy=allow_legacy,
        )
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
            species_cfg.species,
            allow_legacy_plant,
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
    allow_legacy_plant: bool = False,
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
    allow_legacy_plant:
        Explicitly permit untagged legacy model and VecNormalize artifacts.
        Incompatible tagged artifacts are always rejected.

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
        allow_legacy: bool,
    ) -> dict[str, Any]:
        """Evaluate one trial with full LocomotionMetrics inside a Ray task."""
        import os

        os.environ["MUJOCO_GL"] = "egl"

        from pathlib import Path as _Path

        from stable_baselines3 import PPO, SAC

        from environments.shared.metrics import LocomotionMetrics, env_dt
        from environments.shared.plant_contract import (
            current_plant_identity,
            validate_environment_plant,
            validate_model_plant,
        )
        from environments.shared.species_registry import get_species_config
        from environments.shared.train_base import _ensure_sb3

        sb3 = _ensure_sb3()
        sp_cfg = get_species_config(species_name)
        current_plant = current_plant_identity(species_name)
        alg_cls = SAC if alg == "sac" else PPO

        def _mk():
            raw_env = sp_cfg.env_class(**env_kw)
            try:
                validate_environment_plant(
                    raw_env,
                    current_plant,
                    artifact=f"{species_name} Ray post-analysis environment",
                )
            except Exception:
                raw_env.close()
                raise
            return sb3["Monitor"](raw_env)

        eval_env = sb3["DummyVecEnv"]([_mk])
        vn = _Path(vecnorm_path)
        if vn.exists():
            eval_env = sb3["VecNormalize"].load(str(vn), eval_env)
            validate_model_plant(
                eval_env,
                current_plant,
                artifact=str(vn),
                allow_legacy=allow_legacy,
            )
            eval_env.training = False
            eval_env.norm_reward = False

        model = alg_cls.load(model_path, env=eval_env)
        validate_model_plant(
            model,
            current_plant,
            artifact=model_path,
            allow_legacy=allow_legacy,
        )

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
            allow_legacy_plant,
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
    allow_legacy_plant: bool = False,
) -> Path:
    """Validate and promote the best trial's config and model files.

    Writes ``best_trial_config.json`` and re-saves validated model files to
    ``sweep_dir/best_model/`` with the current plant identity embedded.

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
    allow_legacy_plant:
        Explicitly permit export of an untagged legacy trial. Incompatible
        tagged trials are always rejected.

    Returns
    -------
    Path
        Path to the ``best_model/`` directory.
    """
    sweep_dir = Path(sweep_dir)
    sweep_dir.mkdir(parents=True, exist_ok=True)
    current_plant = current_plant_identity(species)

    # Resolve and validate the source trial before producing any promoted
    # artifacts. The directory sidecar rejects a state-level mismatch first;
    # each embedded model/VecNormalize identity is validated below as well.
    best_trial_id = best_result.metrics.get("trial_id", "")
    local_model_dir = Path(local_trials_dir) / str(best_trial_id) / "models" if local_trials_dir else None
    drive_model_dir = sweep_dir / "trials" / str(best_trial_id) / "models"
    if local_model_dir and local_model_dir.exists():
        source_model_dir = local_model_dir
    else:
        source_model_dir = drive_model_dir

    patterns = (
        f"stage{stage}_final.zip",
        f"stage{stage}_final_vecnorm.pkl",
        "best_model.zip",
        "best_model_vecnorm.pkl",
    )
    source_files = [source for pattern in patterns for source in source_model_dir.glob(pattern)]
    loaded_artifacts: dict[Path, Any] = {}
    if source_files:
        _validate_plant_identity_sidecar(
            source_model_dir.parent / PLANT_IDENTITY_FILENAME,
            current_plant,
            artifact=f"Ray trial {best_trial_id}",
            allow_legacy_plant=allow_legacy_plant,
        )
        loaded_artifacts = _load_and_validate_promotion_artifacts(
            source_files,
            algorithm=algorithm,
            current_plant=current_plant,
            allow_legacy_plant=allow_legacy_plant,
        )

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
        "plant_identity": current_plant.to_dict(),
    }

    best_config_path = sweep_dir / "best_trial_config.json"
    with open(str(best_config_path), "w") as f:
        json.dump(best_config_with_meta, f, indent=2, default=str)
    logger.info("Best trial config saved to: %s", best_config_path)

    best_model_dest = sweep_dir / "best_model"
    best_model_dest.mkdir(parents=True, exist_ok=True)

    found_any = False
    for src in source_files:
        dest = best_model_dest / src.name
        loaded_artifacts[src].save(str(dest))
        logger.info("Promoted: %s -> %s", src.name, dest)
        found_any = True

    if not found_any:
        logger.warning(
            "No model files found in %s. Check if the best trial completed.",
            source_model_dir,
        )
    else:
        write_plant_identity(best_model_dest / PLANT_IDENTITY_FILENAME, current_plant)

    return best_model_dest
