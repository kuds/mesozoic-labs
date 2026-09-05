"""Ray Tune sweep infrastructure: callbacks, trial function, and result helpers.

Provides the core components needed to run a Ray Tune hyperparameter sweep:

- ``RayTuneReportCallback``: SB3 callback that reports eval metrics + checkpoints
  to Ray Tune's ASHA scheduler.
- ``TrialTerminationCallback``: Ray Tune callback that prints status summaries.
- ``apply_sampled_config``: Apply Ray Tune sampled hyperparameters to a stage config.
- ``train_trial``: Ray Tune trainable function for a single trial.
- ``collect_ray_results``: Convert Ray Tune results to the shared CSV row format.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .constants import NET_ARCH_PRESETS

if TYPE_CHECKING:
    from environments.shared.plant_contract import PlantIdentity

logger = logging.getLogger(__name__)

PLANT_IDENTITY_FILENAME = "plant_identity.json"


# ---------------------------------------------------------------------------
# Drive sync helpers
# ---------------------------------------------------------------------------

# Retry settings for Google Drive FUSE writes.  Transient FUSE errors
# (EIO, ETIMEDOUT, spurious ENOSPC from buffer pressure) typically
# resolve within a few seconds.
_DRIVE_RETRY_DELAYS = (1, 3, 5)  # seconds between retries


def _copy_to_drive(src: str | Path, dst: str | Path) -> bool:
    """Copy a single file to Drive with retry on transient FUSE errors.

    Returns True if the copy succeeded, False if all retries failed.
    """
    for attempt in range(len(_DRIVE_RETRY_DELAYS) + 1):
        try:
            shutil.copy2(str(src), str(dst))
            return True
        except OSError as e:
            if attempt < len(_DRIVE_RETRY_DELAYS):
                delay = _DRIVE_RETRY_DELAYS[attempt]
                logger.warning(
                    "Drive write failed for %s (attempt %d/%d): %s — retrying in %ds",
                    Path(src).name,
                    attempt + 1,
                    len(_DRIVE_RETRY_DELAYS) + 1,
                    e,
                    delay,
                )
                time.sleep(delay)
            else:
                logger.warning(
                    "Drive write failed for %s after %d attempts: %s",
                    Path(src).name,
                    len(_DRIVE_RETRY_DELAYS) + 1,
                    e,
                )
                return False
    return False  # unreachable, but keeps type checkers happy


def _sync_best_model(src_dir: str | Path, drive_dir: str | Path) -> None:
    """Copy only best_model* and final* files to Drive (skip periodic checkpoints)."""
    src_dir = Path(src_dir)
    drive_dir = Path(drive_dir)
    if not src_dir.exists():
        return
    drive_dir.mkdir(parents=True, exist_ok=True)
    for src_file in src_dir.iterdir():
        if src_file.is_file() and (
            src_file.name.startswith("best_model") or src_file.name.startswith("stage") and "final" in src_file.name
        ):
            _copy_to_drive(src_file, drive_dir / src_file.name)


def _sync_trial_metadata(
    source_dir: str | Path,
    drive_trial_dir: str | Path,
    filenames: tuple[str, ...] = (
        "evaluations.npz",
        "diagnostics.npz",
        "stage_config.json",
        "metrics.json",
        PLANT_IDENTITY_FILENAME,
    ),
    *,
    skip_unchanged: bool = True,
) -> None:
    """Copy trial metadata files (evaluations, diagnostics, config) to Drive.

    When *skip_unchanged* is True (the default), files whose size has not
    changed since the last copy are skipped.  This avoids re-copying
    ``stage_config.json`` (which never changes) on every best-reward
    improvement, and reduces redundant writes of large ``evaluations.npz``
    files that haven't grown since the last sync.
    """
    source_dir = Path(source_dir)
    drive_trial_dir = Path(drive_trial_dir)
    drive_trial_dir.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        src = source_dir / name
        if src.exists():
            dst = drive_trial_dir / name
            if skip_unchanged and dst.exists():
                try:
                    if src.stat().st_size == dst.stat().st_size:
                        continue
                except OSError:
                    pass  # stat failed — fall through to copy
            _copy_to_drive(src, dst)


# ---------------------------------------------------------------------------
# SB3 callback: report metrics + checkpoints to Ray Tune
# ---------------------------------------------------------------------------


def _make_ray_tune_report_callback_class():
    """Build RayTuneReportCallback as a proper BaseCallback subclass at runtime.

    Defers importing ``stable_baselines3`` until first call so the module can
    be loaded outside Ray workers where SB3 may not be installed.
    """
    from stable_baselines3.common.callbacks import BaseCallback

    class _RayTuneReportCallback(BaseCallback):
        """SB3 callback that reports eval metrics + checkpoints to Ray Tune.

        After each evaluation, reports metrics to ASHA and saves a Ray-native
        checkpoint containing the SB3 model and VecNormalize stats.  Also syncs
        the best model to Google Drive for crash resilience.
        """

        def __init__(
            self,
            eval_callback: Any,
            train_env: Any,
            model_ref: list[Any],
            algorithm: str,
            stage: int,
            plant_identity: PlantIdentity,
            drive_best_model_dir: str | Path | None = None,
            verbose: int = 0,
        ) -> None:
            super().__init__(verbose)
            self.eval_callback = eval_callback
            self.train_env = train_env
            self._model_ref = model_ref
            self.algorithm = algorithm
            self.stage = stage
            self.plant_identity = plant_identity
            self._last_eval_count = 0
            self._drive_best_model_dir = Path(drive_best_model_dir) if drive_best_model_dir else None
            self._best_mean_reward = float("-inf")

        def _on_step(self) -> bool:
            from ray import tune
            from ray.train import Checkpoint

            from environments.shared.plant_contract import write_plant_identity

            current_eval_count = len(getattr(self.eval_callback, "evaluations_timesteps", []))
            if current_eval_count <= self._last_eval_count:
                return True
            if (
                not hasattr(self.eval_callback, "last_mean_reward")
                or self.eval_callback.last_mean_reward is None
                or self.eval_callback.last_mean_reward == float("-inf")
            ):
                return True

            self._last_eval_count = current_eval_count

            with tempfile.TemporaryDirectory() as tmpdir:
                model_base = Path(tmpdir) / "model"
                self._model_ref[0].save(str(model_base))
                vecnorm_path = Path(tmpdir) / "vecnorm.pkl"
                self.train_env.save(str(vecnorm_path))
                write_plant_identity(Path(tmpdir) / PLANT_IDENTITY_FILENAME, self.plant_identity)

                checkpoint = Checkpoint.from_directory(tmpdir)
                tune.report(
                    {
                        "best_mean_reward": float(self.eval_callback.best_mean_reward),
                        "last_mean_reward": float(self.eval_callback.last_mean_reward),
                        "timesteps": self.num_timesteps,
                    },
                    checkpoint=checkpoint,
                )

            if self._drive_best_model_dir and self.eval_callback.best_mean_reward > self._best_mean_reward:
                self._best_mean_reward = self.eval_callback.best_mean_reward
                best_src = Path(self.eval_callback.best_model_save_path)
                _sync_best_model(best_src, self._drive_best_model_dir)

                # Sync evaluations.npz, diagnostics.npz, and stage_config.json to
                # the Drive trial dir (one level up from models/) so post-analysis
                # can rank trials and generate training curve / diagnostics graphs
                # without local /tmp/.
                _sync_trial_metadata(
                    Path(self.eval_callback.log_path),
                    self._drive_best_model_dir.parent,
                )

            return True

    return _RayTuneReportCallback


def RayTuneReportCallback(*args: Any, **kwargs: Any):
    """Create a RayTuneReportCallback instance.

    Defers importing ``stable_baselines3.common.callbacks.BaseCallback`` until
    first call to avoid importing SB3 at module level.
    """
    cls = _make_ray_tune_report_callback_class()
    return cls(*args, **kwargs)


# ---------------------------------------------------------------------------
# Ray Tune callback: print trial status summaries
# ---------------------------------------------------------------------------


def _make_trial_termination_callback_class():
    """Build TrialTerminationCallback as a proper Callback subclass at runtime."""
    from ray.tune import Callback

    class _TrialTerminationCallback(Callback):
        """Ray Tune callback that prints status on trial completion and periodically."""

        METRIC_COLS = ("best_mean_reward", "last_mean_reward", "timesteps")

        def __init__(self, report_interval_s: int = 300) -> None:
            self._report_interval_s = report_interval_s
            self._last_report_time = 0.0

        def on_trial_complete(self, iteration: int, trials: list[Any], trial: Any, **info: Any) -> None:
            metrics = {k: trial.last_result.get(k) for k in self.METRIC_COLS}
            metrics_str = "  ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())
            n_done = sum(1 for t in trials if t.status == "TERMINATED")
            print(f"[Trial {trial.trial_id} DONE] ({n_done}/{len(trials)} complete)  {metrics_str}")

        def on_trial_result(
            self,
            iteration: int,
            trials: list[Any],
            trial: Any,
            result: dict[str, Any],
            **info: Any,
        ) -> None:
            now = time.time()
            if now - self._last_report_time < self._report_interval_s:
                return
            self._last_report_time = now

            header = f"\n{'trial_id':<16}" + "".join(f"{c:>20}" for c in self.METRIC_COLS) + f"{'status':>14}"
            print(header)
            print("-" * len(header))
            for t in trials:
                cols = "".join(
                    f"{t.last_result.get(c, ''):>20}"
                    if not isinstance(t.last_result.get(c), float)
                    else f"{t.last_result[c]:>20.2f}"
                    for c in self.METRIC_COLS
                )
                print(f"{t.trial_id:<16}{cols}{t.status:>14}")
            print()

    return _TrialTerminationCallback


def TrialTerminationCallback(*args: Any, **kwargs: Any):
    """Create a TrialTerminationCallback instance.

    Defers importing ``ray.tune.Callback`` until first call to avoid importing
    Ray at module level.
    """
    cls = _make_trial_termination_callback_class()
    return cls(*args, **kwargs)


# ---------------------------------------------------------------------------
# Ray Tune callback: sync experiment state to Drive for cross-session resume
# ---------------------------------------------------------------------------


def _make_experiment_state_sync_callback_class():
    """Build ExperimentStateSyncCallback as a proper Callback subclass at runtime."""
    from ray.tune import Callback

    class _ExperimentStateSyncCallback(Callback):
        """Periodically syncs Ray Tune experiment state to Google Drive.

        This enables cross-session resume: if a Colab session terminates,
        the experiment state can be restored from Drive on the next session
        so that ``Tuner.restore()`` can pick up where it left off.

        Uses incremental sync: only copies files that are newer than the
        Drive copy (by mtime), avoiding redundant I/O as the experiment
        directory grows with completed trials.

        Syncs happen:
        - After every trial completes (captures newly finished results)
        - Periodically based on ``sync_interval_s`` (captures in-progress state)
        """

        def __init__(
            self,
            local_experiment_dir: str | Path,
            drive_ray_results_dir: str | Path,
            sync_interval_s: int = 300,
        ) -> None:
            self._local_dir = Path(local_experiment_dir)
            self._drive_dir = Path(drive_ray_results_dir) / self._local_dir.name
            self._sync_interval_s = sync_interval_s
            self._last_sync_time = 0.0
            # Cache of {relative_path: (size, mtime)} for files already
            # synced to Drive.  Avoids expensive stat() calls on the FUSE
            # mount for files that haven't changed since the last sync.
            self._synced_file_cache: dict[str, tuple[int, float]] = {}

        def _sync(self, reason: str = "") -> None:
            """Incrementally copy local experiment state to Drive."""
            if not self._local_dir.exists():
                return
            try:
                self._drive_dir.mkdir(parents=True, exist_ok=True)
                copied = 0
                for src_file in self._local_dir.rglob("*"):
                    if not src_file.is_file():
                        continue
                    rel = str(src_file.relative_to(self._local_dir))
                    src_stat = src_file.stat()
                    src_key = (src_stat.st_size, src_stat.st_mtime)
                    # Skip if we already synced this exact version
                    if self._synced_file_cache.get(rel) == src_key:
                        continue
                    dst = self._drive_dir / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if _copy_to_drive(src_file, dst):
                        self._synced_file_cache[rel] = src_key
                        copied += 1
                self._last_sync_time = time.time()
                if copied:
                    logger.info("Experiment state synced to Drive (%s, %d files copied)", reason, copied)
            except OSError as e:
                logger.warning("Experiment state sync failed: %s", e)

        def on_trial_complete(self, iteration: int, trials: list[Any], trial: Any, **info: Any) -> None:
            self._sync(reason=f"trial {trial.trial_id} complete")

        def on_trial_result(
            self,
            iteration: int,
            trials: list[Any],
            trial: Any,
            result: dict[str, Any],
            **info: Any,
        ) -> None:
            now = time.time()
            if now - self._last_sync_time >= self._sync_interval_s:
                self._sync(reason="periodic")

    return _ExperimentStateSyncCallback


def ExperimentStateSyncCallback(*args: Any, **kwargs: Any):
    """Create an ExperimentStateSyncCallback instance.

    Defers importing ``ray.tune.Callback`` until first call to avoid importing
    Ray at module level.
    """
    cls = _make_experiment_state_sync_callback_class()
    return cls(*args, **kwargs)


# ---------------------------------------------------------------------------
# Ray Tune callback: write trial progress CSV to Drive
# ---------------------------------------------------------------------------


def _make_drive_progress_log_callback_class():
    """Build DriveProgressLogCallback as a proper Callback subclass at runtime."""
    import csv

    from ray.tune import Callback

    class _DriveProgressLogCallback(Callback):
        """Writes a ``trial_progress.csv`` to Google Drive on each trial completion.

        This provides a simple, human-readable log of completed trials and their
        reward metrics that can be checked directly on Drive even when the
        notebook is disconnected.  Unlike ``collected_results.csv`` (which is
        written post-sweep with full evaluation metrics), this file is updated
        incrementally during the sweep with the metrics reported to ASHA.

        Columns: ``trial_id``, ``status``, ``best_mean_reward``,
        ``last_mean_reward``, ``timesteps``, ``timestamp``, plus any
        hyperparameters from the trial config.
        """

        METRIC_COLS = ("best_mean_reward", "last_mean_reward", "timesteps")

        def __init__(self, drive_sweep_dir: str | Path) -> None:
            self._csv_path = Path(drive_sweep_dir) / "trial_progress.csv"
            self._written_header = self._csv_path.exists()

        def _write_row(self, trial: Any, status: str) -> None:
            """Append a single row to the progress CSV."""
            from datetime import datetime

            row: dict[str, Any] = {
                "trial_id": trial.trial_id,
                "status": status,
            }
            for col in self.METRIC_COLS:
                val = trial.last_result.get(col)
                if isinstance(val, float):
                    row[col] = round(val, 4)
                else:
                    row[col] = val

            row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Include hyperparameters (skip internal _ prefixed keys)
            for key, value in trial.config.items():
                if not key.startswith("_"):
                    row[key] = value

            self._csv_path.parent.mkdir(parents=True, exist_ok=True)
            for attempt in range(len(_DRIVE_RETRY_DELAYS) + 1):
                try:
                    with open(self._csv_path, "a", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                        if not self._written_header:
                            writer.writeheader()
                            self._written_header = True
                        writer.writerow(row)
                    break  # success
                except OSError as e:
                    if attempt < len(_DRIVE_RETRY_DELAYS):
                        time.sleep(_DRIVE_RETRY_DELAYS[attempt])
                    else:
                        logger.warning("Failed to write trial progress to %s: %s", self._csv_path, e)

        def on_trial_complete(self, iteration: int, trials: list[Any], trial: Any, **info: Any) -> None:
            # Distinguish ASHA-pruned trials from those that ran to completion.
            # train_trial() reports done=True only in its final metrics; trials
            # stopped early by ASHA will not have this flag.
            if trial.last_result.get("done"):
                status = "COMPLETED"
            else:
                status = "PRUNED"
            self._write_row(trial, status=status)

        def on_trial_error(self, iteration: int, trials: list[Any], trial: Any, **info: Any) -> None:
            self._write_row(trial, status="ERROR")

    return _DriveProgressLogCallback


def DriveProgressLogCallback(*args: Any, **kwargs: Any):
    """Create a DriveProgressLogCallback instance.

    Defers importing ``ray.tune.Callback`` until first call to avoid importing
    Ray at module level.
    """
    cls = _make_drive_progress_log_callback_class()
    return cls(*args, **kwargs)


# ---------------------------------------------------------------------------
# Config application
# ---------------------------------------------------------------------------


def apply_sampled_config(
    stage_configs: "dict[int | str, dict[str, Any]]",
    stage: int,
    hpt_config: dict[str, Any],
    algorithm: str,
) -> None:
    """Apply Ray Tune sampled hyperparameters to the stage config dict.

    Uses the naming convention: ``ppo_*`` / ``sac_*`` → algo kwargs,
    ``env_*`` → env_kwargs, ``curriculum_*`` → curriculum_kwargs,
    ``*_net_arch`` → policy_kwargs.net_arch (resolved via NET_ARCH_PRESETS).
    """
    config = stage_configs[stage]
    algo_key = f"{algorithm}_kwargs"

    for key, value in hpt_config.items():
        for prefix in ("ppo", "sac", "env", "curriculum"):
            if key.startswith(prefix + "_"):
                param = key[len(prefix) + 1 :]
                if prefix in ("ppo", "sac"):
                    if param == "net_arch":
                        config[algo_key].setdefault("policy_kwargs", {})["net_arch"] = NET_ARCH_PRESETS[value]
                    else:
                        if param in ("batch_size", "n_steps", "n_epochs"):
                            value = int(value)
                        config[algo_key][param] = value
                elif prefix == "env":
                    config["env_kwargs"][param] = value
                elif prefix == "curriculum":
                    if param in ("warmup_timesteps", "ramp_timesteps"):
                        value = int(value)
                    config["curriculum_kwargs"][param] = value
                break


def apply_collapse_overrides(
    stage_config: dict[str, Any],
    *,
    min_evals: int | None = None,
    patience: int | None = None,
) -> dict[str, Any]:
    """Return a stage-config copy with positive Ray collapse overrides applied.

    Ray-specific values are optional overrides, not an independent set of
    collapse defaults. Copying the curriculum mapping preserves every other
    stage setting and keeps this transformation side-effect free.
    """
    resolved = dict(stage_config)
    curriculum_kwargs = dict(stage_config.get("curriculum_kwargs") or {})
    if min_evals is not None and min_evals > 0:
        curriculum_kwargs["collapse_min_evals"] = int(min_evals)
    if patience is not None and patience > 0:
        curriculum_kwargs["collapse_patience"] = int(patience)
    resolved["curriculum_kwargs"] = curriculum_kwargs
    return resolved


# ---------------------------------------------------------------------------
# Ray Tune trainable
# ---------------------------------------------------------------------------


def train_trial(config: dict[str, Any]) -> None:
    """Ray Tune trainable function for a single hyperparameter trial.

    Runs inside a Ray worker.  Reuses the project's existing training
    infrastructure rather than calling ``train()`` directly, because we need
    to inject the ``RayTuneReportCallback`` to report intermediate metrics
    to ASHA.

    Each trial trains from scratch (or from a warm-start checkpoint for
    stages 2+).  Mid-training resume is intentionally not supported because
    callback state cannot be reliably restored.
    """
    import os

    os.environ["MUJOCO_GL"] = "egl"
    os.environ["TUNE_WARN_EXCESSIVE_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S"] = "0"
    import logging as _logging

    _logging.getLogger("tensorboardX").setLevel(_logging.ERROR)
    _logging.getLogger("ray.tune.experiment_state").setLevel(_logging.ERROR)

    from ray import tune
    from ray.train import Checkpoint
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.callbacks import (
        CallbackList,
        CheckpointCallback,
    )

    from environments.shared.config import load_all_stages, save_stage_config
    from environments.shared.curriculum import (
        RewardRampCallback,
        SaveVecNormalizeCallback,
        StageWarmupCallback,
        build_eval_collapse_early_stop_callback,
        load_vecnorm_stats,
    )
    from environments.shared.eval_diagnostics import build_stage_evaluation_callbacks
    from environments.shared.plant_contract import (
        attach_plant_identity,
        current_plant_identity,
        validate_model_plant,
        write_plant_identity,
    )
    from environments.shared.species_registry import get_species_config
    from environments.shared.train_base import (
        cosine_schedule,
        create_vec_env,
        linear_schedule,
    )

    # Unpack fixed params
    species = config["_species"]
    algorithm = config["_algorithm"]
    stage = config["_stage"]
    timesteps = config["_timesteps"]
    n_envs = config["_n_envs"]
    seed = config["_seed"]
    eval_freq = config["_eval_freq"]
    load_path = config.get("_load_path") or None
    collapse_min_evals = config.get("_collapse_min_evals")
    collapse_patience = config.get("_collapse_patience")
    n_eval_episodes = config.get("_n_eval_episodes", 30)
    local_trials_dir = config.get("_local_trials_dir")
    drive_sweep_dir = config.get("_drive_sweep_dir")
    allow_legacy_plant = bool(config.get("_allow_legacy_plant", False))

    species_cfg = get_species_config(species)
    plant_identity = current_plant_identity(species)
    stage_configs = load_all_stages(species)

    # Apply sampled hyperparameters (skip _ prefixed fixed params)
    hpt_params = {k: v for k, v in config.items() if not k.startswith("_")}
    apply_sampled_config(stage_configs, stage, hpt_params, algorithm)

    stage_config = apply_collapse_overrides(
        stage_configs[stage],
        min_evals=collapse_min_evals,
        patience=collapse_patience,
    )
    stage_configs[stage] = stage_config

    # Setup output directory
    trial_id = tune.get_context().get_trial_id() or "local"

    # Give every trial its own seed (base_seed + stable hash of trial id).
    # With a shared seed, all trials start from identical network init and
    # env randomness, so the "best" config partly wins on seed luck and the
    # ranking does not generalise.  The effective seed is recorded in
    # stage_config.json / the results CSV.
    if config.get("_vary_seed_per_trial", True) and trial_id != "local":
        import zlib

        seed = seed + (zlib.crc32(trial_id.encode()) % 10_000)
        logger.info("Per-trial seed: %d (trial %s)", seed, trial_id)
    trial_dir = Path(local_trials_dir) / trial_id if local_trials_dir else Path(f"/tmp/ray_tune_trial_{trial_id}")
    trial_dir.mkdir(parents=True, exist_ok=True)
    model_dir = trial_dir / "models"
    model_dir.mkdir(exist_ok=True)

    save_stage_config(
        trial_dir,
        stage,
        stage_config,
        algorithm.upper(),
        extra={"seed": seed, "n_envs": n_envs, "trial_id": trial_id},
        env_class=species_cfg.env_class,
        species=species,
        plant_identity=plant_identity,
    )

    drive_best_model_dir = None
    if drive_sweep_dir:
        drive_best_model_dir = Path(drive_sweep_dir) / "trials" / trial_id / "models"

        # Sync stage_config.json to Drive immediately so it survives session
        # crashes.  Later syncs (in RayTuneReportCallback and post-training)
        # will overwrite with the same content, which is harmless.
        drive_trial_dir = Path(drive_sweep_dir) / "trials" / trial_id
        drive_trial_dir.mkdir(parents=True, exist_ok=True)
        _sync_trial_metadata(
            trial_dir,
            drive_trial_dir,
            filenames=("stage_config.json", PLANT_IDENTITY_FILENAME),
            skip_unchanged=False,
        )

    _train_start_time = time.time()

    # Create environments.
    # SAC benefits from SubprocVecEnv (parallel env stepping) since it
    # interleaves gradient updates with env steps in a tight loop.
    use_subproc = algorithm == "sac" and n_envs > 1
    alg_gamma = stage_config.get(f"{algorithm}_kwargs", {}).get("gamma")
    train_env = create_vec_env(
        species_cfg,
        stage_configs,
        stage,
        n_envs,
        seed,
        use_subproc=use_subproc,
        algorithm=algorithm,
        gamma=alg_gamma,
        plant_identity=plant_identity,
    )
    eval_env = create_vec_env(
        species_cfg,
        stage_configs,
        stage,
        1,
        seed + 1000,
        use_subproc=False,
        algorithm=algorithm,
        gamma=alg_gamma,
        plant_identity=plant_identity,
    )

    try:
        # Create or load model
        alg_cls = SAC if algorithm == "sac" else PPO
        algo_key = "sac_kwargs" if algorithm == "sac" else "ppo_kwargs"
        alg_kwargs = stage_config[algo_key].copy()
        alg_kwargs["verbose"] = 0
        alg_kwargs["tensorboard_log"] = str(trial_dir / "tensorboard")

        if algorithm == "ppo":
            lr_end = alg_kwargs.pop("learning_rate_end", None)
            lr_schedule_type = alg_kwargs.pop("lr_schedule", "linear")
            if lr_end is not None:
                lr_start = alg_kwargs["learning_rate"]
                if lr_schedule_type == "cosine":
                    alg_kwargs["learning_rate"] = cosine_schedule(lr_start, lr_end)
                else:
                    alg_kwargs["learning_rate"] = linear_schedule(lr_start, lr_end)

            clip_range_end = alg_kwargs.pop("clip_range_end", None)
            if clip_range_end is not None:
                clip_start = alg_kwargs["clip_range"]
                alg_kwargs["clip_range"] = linear_schedule(clip_start, clip_range_end)

        policy_kwargs = alg_kwargs.pop("policy_kwargs", None)

        if load_path:
            vecnorm_path = load_path.replace(".zip", "") + "_vecnorm.pkl"
            if not load_vecnorm_stats(
                vecnorm_path,
                train_env,
                eval_env,
                current_plant=plant_identity,
                allow_legacy_plant=allow_legacy_plant,
            ):
                eval_env.training = False
                eval_env.norm_reward = False
            model = alg_cls.load(load_path, env=train_env, **alg_kwargs)
            validate_model_plant(
                model,
                plant_identity,
                artifact=str(load_path),
                allow_legacy=allow_legacy_plant,
            )
        else:
            eval_env.training = False
            eval_env.norm_reward = False
            model = alg_cls("MlpPolicy", train_env, policy_kwargs=policy_kwargs, **alg_kwargs)
        # All Ray/SB3 save paths persist attributes on these objects.  Attach
        # only after validating a load so an incompatible artifact can never
        # be relabelled as current.
        attach_plant_identity(model, plant_identity)

        # Callbacks
        callbacks: list[Any] = []
        cur_kwargs = stage_config.get("curriculum_kwargs", {})

        save_vecnorm_cb = SaveVecNormalizeCallback(
            save_path=str(model_dir / "best_model_vecnorm.pkl"),
        )
        eval_callback, plateau_callback = build_stage_evaluation_callbacks(
            eval_env,
            stage=stage,
            stage_config=stage_config,
            best_model_save_path=str(model_dir),
            log_path=str(trial_dir),
            eval_freq=eval_freq // n_envs,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False,
            verbose=0,
            callback_on_new_best=save_vecnorm_cb,
        )
        callbacks.append(eval_callback)
        callbacks.append(plateau_callback)

        model_ref = [model]
        callbacks.append(
            RayTuneReportCallback(
                eval_callback,
                train_env,
                model_ref,
                algorithm,
                stage,
                plant_identity,
                drive_best_model_dir=drive_best_model_dir,
            )
        )

        callbacks.append(
            CheckpointCallback(
                save_freq=max(10 * eval_freq // n_envs, 1),
                save_path=str(model_dir),
                name_prefix=f"stage{stage}",
                save_vecnormalize=True,
            )
        )

        callbacks.append(
            build_eval_collapse_early_stop_callback(
                eval_callback,
                cur_kwargs,
                verbose=0,
            )
        )

        from ...diagnostics import DiagnosticsCallback

        callbacks.append(DiagnosticsCallback(log_dir=str(trial_dir), verbose=0))

        # Stage transition callbacks (stages 2+)
        if stage > 1 and load_path:
            callbacks.append(
                StageWarmupCallback(
                    warmup_timesteps=cur_kwargs.get("warmup_timesteps", 100_000),
                    warmup_clip_range=cur_kwargs.get("warmup_clip_range", 0.02),
                    warmup_ent_coef=cur_kwargs.get("warmup_ent_coef", 0.02),
                    warmup_lr_scale=cur_kwargs.get("warmup_lr_scale", 0.1),
                )
            )
            target_fwd_weight = stage_config["env_kwargs"].get("forward_vel_weight", 1.0)
            # Ramping forward_vel_weight only makes sense when the stage USES
            # it: recovery mirrors stance and sets it to 0.0, and ramping
            # 0.1 -> 0.0 would inject a walk incentive the task fingerprint
            # says is absent (same guard as train_base.py's launch paths).
            if target_fwd_weight > 0.0:
                callbacks.append(
                    RewardRampCallback(
                        attr_name="forward_vel_weight",
                        start_value=cur_kwargs.get("ramp_start_value", 0.1),
                        end_value=target_fwd_weight,
                        ramp_timesteps=cur_kwargs.get("ramp_timesteps", 500_000),
                    )
                )

        # Train
        model.learn(
            total_timesteps=timesteps,
            callback=CallbackList(callbacks),
            progress_bar=False,
        )

        # Save final model
        final_path = model_dir / f"stage{stage}_final"
        model.save(str(final_path))
        train_env.save(str(final_path) + "_vecnorm.pkl")

        if drive_best_model_dir:
            _sync_best_model(model_dir, drive_best_model_dir)
            _sync_trial_metadata(trial_dir, drive_best_model_dir.parent)

        # Post-training evaluation for distance + forward velocity metrics.
        # Load the best model for evaluation (matches what gets handed off).
        from ...evaluation import eval_policy

        best_model_zip = model_dir / "best_model.zip"
        best_vecnorm = model_dir / "best_model_vecnorm.pkl"
        eval_model = model
        if best_model_zip.exists():
            eval_model = alg_cls.load(str(model_dir / "best_model"), env=eval_env)
            validate_model_plant(
                eval_model,
                plant_identity,
                artifact=str(best_model_zip),
                allow_legacy=allow_legacy_plant,
            )
            if best_vecnorm.exists():
                load_vecnorm_stats(
                    str(best_vecnorm),
                    eval_env,
                    current_plant=plant_identity,
                    allow_legacy_plant=allow_legacy_plant,
                )
        eval_env.training = False
        eval_env.norm_reward = False

        _, eval_lengths, eval_fwd_vels, eval_successes, eval_distances = eval_policy(
            eval_model,
            eval_env,
            species_cfg.success_keys,
            n_episodes=n_eval_episodes,
        )
        import json as _json

        import numpy as _np

        _training_duration_s = time.time() - _train_start_time

        final_metrics = {
            "best_mean_reward": float(eval_callback.best_mean_reward),
            "best_mean_episode_length": float(_np.mean(eval_lengths)) if eval_lengths else 0.0,
            "mean_forward_vel": float(_np.mean(eval_fwd_vels)) if eval_fwd_vels else 0.0,
            "std_forward_vel": float(_np.std(eval_fwd_vels)) if eval_fwd_vels else 0.0,
            "mean_distance_traveled": float(_np.mean(eval_distances)) if eval_distances else 0.0,
            "mean_success_rate": float(_np.mean(eval_successes)) if eval_successes else 0.0,
            "training_duration_seconds": round(_training_duration_s, 1),
            "timesteps": timesteps,
            "done": True,
        }

        # ── Quality evaluation (matches Vertex AI's _report_hpt_metrics) ──
        # Collect spinning detection, heading alignment, and reward
        # component breakdown so Ray Tune trials produce the same eval_*
        # metrics that Vertex AI trials write to metrics.json.
        try:
            from ...evaluation import eval_policy_quality

            quality_metrics = eval_policy_quality(
                eval_model,
                eval_env,
                species_cfg.success_keys,
                n_episodes=n_eval_episodes,
            )
            final_metrics.update(quality_metrics)
            logger.info(
                "Quality eval complete: %d metrics (angular_vel=%.3f, heading_align=%.3f)",
                len(quality_metrics),
                quality_metrics.get("eval_mean_pelvis_angular_velocity", float("nan")),
                quality_metrics.get("eval_mean_heading_alignment", float("nan")),
            )
        except Exception:
            logger.warning("Quality evaluation failed — skipping quality metrics.", exc_info=True)

        # ── Write metrics.json sidecar (matches Vertex AI trial output) ───
        # Enables offline result collection via collect_results_from_disk()
        # and provides a consistent artifact format across both backends.
        aux_metrics: dict[str, Any] = {
            # Run identity + effective per-trial seed, so trials remain
            # reproducible from the collected CSV alone.
            "species": species,
            "algorithm": algorithm,
            "seed": seed,
            "best_mean_reward": final_metrics["best_mean_reward"],
            "best_mean_episode_length": final_metrics["best_mean_episode_length"],
            "last_mean_reward": float(eval_callback.last_mean_reward)
            if hasattr(eval_callback, "last_mean_reward") and eval_callback.last_mean_reward is not None
            else final_metrics["best_mean_reward"],
            "last_mean_episode_length": float(_np.mean(eval_lengths)) if eval_lengths else 0.0,
            "mean_forward_vel": final_metrics["mean_forward_vel"],
            "std_forward_vel": final_metrics["std_forward_vel"],
            "best_mean_forward_vel": final_metrics["mean_forward_vel"],
            "mean_distance_traveled": final_metrics["mean_distance_traveled"],
            "mean_success_rate": final_metrics["mean_success_rate"],
            "best_mean_success_rate": final_metrics["mean_success_rate"],
            "training_duration_seconds": final_metrics["training_duration_seconds"],
            "plant_identity": plant_identity.to_dict(),
        }
        # Include eval_* quality metrics
        for key, value in final_metrics.items():
            if key.startswith("eval_"):
                aux_metrics[key] = value
        # Include key hyperparameters so offline collection works even
        # without stage_config.json
        algo_key_cfg = f"{algorithm}_kwargs"
        algo_kwargs_cfg = stage_config.get(algo_key_cfg, {})
        _hparam_keys = (
            ("learning_rate", "batch_size", "gamma", "n_steps", "ent_coef")
            if algorithm == "ppo"
            else ("learning_rate", "batch_size", "gamma", "tau", "buffer_size", "ent_coef")
        )
        for k in _hparam_keys:
            if k in algo_kwargs_cfg:
                val = algo_kwargs_cfg[k]
                if not callable(val):
                    aux_metrics[f"{algorithm}_{k}"] = val
        net_arch = algo_kwargs_cfg.get("policy_kwargs", {}).get("net_arch")
        if net_arch is not None:
            aux_metrics[f"{algorithm}_net_arch"] = str(net_arch)

        metrics_path = trial_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            _json.dump(aux_metrics, f, indent=2)
        logger.info("Metrics sidecar written to: %s", metrics_path)

        # ── Generate stage artifacts (summary, graphs) ────────────────────
        # Produces the same training_curves.png, diagnostics graphs, and
        # stage_summary.txt that Vertex AI trials generate, making Ray Tune
        # trial directories self-contained for post-analysis.
        try:
            from ...reporting import generate_stage_artifacts

            generate_stage_artifacts(
                species_cfg=species_cfg,
                stage_config=stage_config,
                stage=stage,
                algorithm=algorithm,
                stage_dir=str(trial_dir),
                seed=seed,
                timesteps=timesteps,
                record_videos=False,
                generate_graphs=True,
            )
        except Exception:
            logger.warning("Stage artifact generation failed.", exc_info=True)

        # Sync metrics.json to Drive alongside other trial metadata
        if drive_best_model_dir:
            _sync_trial_metadata(
                trial_dir,
                drive_best_model_dir.parent,
                filenames=(
                    "evaluations.npz",
                    "diagnostics.npz",
                    "stage_config.json",
                    "metrics.json",
                    PLANT_IDENTITY_FILENAME,
                ),
            )

        # Final report
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(str(Path(tmpdir) / "model"))
            train_env.save(str(Path(tmpdir) / "vecnorm.pkl"))
            write_plant_identity(Path(tmpdir) / PLANT_IDENTITY_FILENAME, plant_identity)
            checkpoint = Checkpoint.from_directory(tmpdir)
            tune.report(final_metrics, checkpoint=checkpoint)
    finally:
        train_env.close()
        eval_env.close()


# ---------------------------------------------------------------------------
# Result collection helper
# ---------------------------------------------------------------------------


def collect_ray_results(
    results_df: Any,
    stage: int,
    stage_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Convert a Ray Tune results DataFrame to the shared sweep row-dict format.

    Returns a list of dicts compatible with ``write_results_csv()``.
    """
    from .results import _gate_row_fields

    rows: list[dict[str, Any]] = []
    cur = stage_config.get("curriculum_kwargs", {})

    for _, rt_row in results_df.iterrows():
        row: dict[str, Any] = {"trial_id": str(rt_row.get("trial_id", "")), "stage": stage}

        # Hyperparameters
        for col in results_df.columns:
            if col.startswith(("ppo_", "sac_", "env_", "curriculum_")):
                row[col] = rt_row[col]

        # Metrics
        for metric in (
            "best_mean_reward",
            "best_mean_episode_length",
            "last_mean_reward",
            "last_mean_episode_length",
            "mean_forward_vel",
            "std_forward_vel",
            "mean_distance_traveled",
            "mean_success_rate",
            "training_duration_seconds",
        ):
            row[metric] = rt_row.get(metric)

        # Quality evaluation metrics (eval_* keys from eval_policy_quality)
        for col in results_df.columns:
            if col.startswith("eval_"):
                row[col] = rt_row[col]

        # Curriculum thresholds
        row["reward_threshold"] = cur.get("min_avg_reward")
        row["ep_length_threshold"] = cur.get("min_avg_episode_length")
        row["forward_vel_threshold"] = cur.get("min_avg_forward_vel")
        row["success_rate_threshold"] = cur.get("min_success_rate")

        # Gate evaluation, routed through the stage's declared gate_kind
        row.update(_gate_row_fields(cur, row["best_mean_reward"], row))
        rows.append(row)

    return rows
