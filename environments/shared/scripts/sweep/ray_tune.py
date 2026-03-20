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
from typing import Any

from .constants import NET_ARCH_PRESETS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Drive sync helper
# ---------------------------------------------------------------------------


def _sync_to_drive(src_dir: str | Path, drive_dir: str | Path, label: str = "") -> None:
    """Copy files from a local directory to Drive, tolerating FUSE flakiness."""
    src_dir = Path(src_dir)
    drive_dir = Path(drive_dir)
    if not src_dir.exists():
        return
    drive_dir.mkdir(parents=True, exist_ok=True)
    for src_file in src_dir.iterdir():
        if src_file.is_file():
            try:
                shutil.copy2(str(src_file), str(drive_dir / src_file.name))
            except OSError as e:
                logger.warning("Drive sync failed for %s: %s", src_file.name, e)


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
            drive_best_model_dir: str | Path | None = None,
            verbose: int = 0,
        ) -> None:
            super().__init__(verbose)
            self.eval_callback = eval_callback
            self.train_env = train_env
            self._model_ref = model_ref
            self.algorithm = algorithm
            self.stage = stage
            self._last_eval_count = 0
            self._drive_best_model_dir = Path(drive_best_model_dir) if drive_best_model_dir else None
            self._best_mean_reward = float("-inf")

        def _on_step(self) -> bool:
            from ray import tune
            from ray.train import Checkpoint

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
                _sync_to_drive(best_src, self._drive_best_model_dir, label=f"best@{self.num_timesteps}")

                # Sync evaluations.npz and diagnostics.npz to the Drive trial dir
                # (one level up from models/) so post-analysis can rank trials and
                # generate training curve / diagnostics graphs without local /tmp/.
                drive_trial_dir = self._drive_best_model_dir.parent
                drive_trial_dir.mkdir(parents=True, exist_ok=True)
                _log_dir = Path(self.eval_callback.log_path)
                for _npz_name in ("evaluations.npz", "diagnostics.npz"):
                    _npz = _log_dir / _npz_name
                    if _npz.exists():
                        try:
                            shutil.copy2(str(_npz), str(drive_trial_dir / _npz_name))
                        except OSError as e:
                            logger.warning("Drive sync failed for %s: %s", _npz_name, e)

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

        def _sync(self, reason: str = "") -> None:
            """Copy local experiment state to Drive."""
            if not self._local_dir.exists():
                return
            try:
                self._drive_dir.mkdir(parents=True, exist_ok=True)
                shutil.copytree(
                    str(self._local_dir),
                    str(self._drive_dir),
                    dirs_exist_ok=True,
                )
                self._last_sync_time = time.time()
                logger.info("Experiment state synced to Drive (%s)", reason)
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
            self._written_header = False

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

            try:
                file_exists = self._csv_path.exists()
                self._csv_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._csv_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                    if not file_exists and not self._written_header:
                        writer.writeheader()
                        self._written_header = True
                    writer.writerow(row)
            except OSError as e:
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
    stage_configs: dict[int, dict[str, Any]],
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
        EvalCallback,
    )

    from environments.shared.config import load_all_stages
    from environments.shared.curriculum import (
        EvalCollapseEarlyStopCallback,
        RewardRampCallback,
        SaveVecNormalizeCallback,
        StageWarmupCallback,
        load_vecnorm_stats,
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
    collapse_min_evals = config.get("_collapse_min_evals", 8)
    collapse_patience = config.get("_collapse_patience", 5)
    local_trials_dir = config.get("_local_trials_dir")
    drive_sweep_dir = config.get("_drive_sweep_dir")

    species_cfg = get_species_config(species)
    stage_configs = load_all_stages(species)

    # Apply sampled hyperparameters (skip _ prefixed fixed params)
    hpt_params = {k: v for k, v in config.items() if not k.startswith("_")}
    apply_sampled_config(stage_configs, stage, hpt_params, algorithm)

    stage_config = stage_configs[stage]

    # Setup output directory
    trial_id = tune.get_context().get_trial_id() or "local"
    trial_dir = Path(local_trials_dir) / trial_id if local_trials_dir else Path(f"/tmp/ray_tune_trial_{trial_id}")
    trial_dir.mkdir(parents=True, exist_ok=True)
    model_dir = trial_dir / "models"
    model_dir.mkdir(exist_ok=True)

    drive_best_model_dir = None
    if drive_sweep_dir:
        drive_best_model_dir = Path(drive_sweep_dir) / "trials" / trial_id / "models"

    _train_start_time = time.time()

    # Create environments.
    # SAC benefits from SubprocVecEnv (parallel env stepping) since it
    # interleaves gradient updates with env steps in a tight loop.
    use_subproc = algorithm == "sac" and n_envs > 1
    train_env = create_vec_env(species_cfg, stage_configs, stage, n_envs, seed, use_subproc=use_subproc)
    eval_env = create_vec_env(species_cfg, stage_configs, stage, 1, seed + 1000, use_subproc=False)

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
            if not vecnorm_path.endswith("_vecnorm.pkl"):
                vecnorm_path = load_path + "_vecnorm.pkl"
            if not load_vecnorm_stats(vecnorm_path, train_env, eval_env):
                eval_env.training = False
                eval_env.norm_reward = False
            model = alg_cls.load(load_path, env=train_env, **alg_kwargs)
        else:
            eval_env.training = False
            eval_env.norm_reward = False
            model = alg_cls("MlpPolicy", train_env, policy_kwargs=policy_kwargs, **alg_kwargs)

        # Callbacks
        callbacks = []

        save_vecnorm_cb = SaveVecNormalizeCallback(
            save_path=str(model_dir / "best_model_vecnorm.pkl"),
        )
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(model_dir),
            log_path=str(trial_dir),
            eval_freq=eval_freq // n_envs,
            n_eval_episodes=30,
            deterministic=True,
            render=False,
            verbose=0,
            callback_on_new_best=save_vecnorm_cb,
        )
        callbacks.append(eval_callback)

        model_ref = [model]
        callbacks.append(
            RayTuneReportCallback(
                eval_callback,
                train_env,
                model_ref,
                algorithm,
                stage,
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
            EvalCollapseEarlyStopCallback(
                eval_callback=eval_callback, min_evals=collapse_min_evals, patience=collapse_patience, verbose=0
            )
        )

        from ...diagnostics import DiagnosticsCallback

        callbacks.append(DiagnosticsCallback(log_dir=str(trial_dir), verbose=0))

        # Stage transition callbacks (stages 2+)
        cur_kwargs = stage_config.get("curriculum_kwargs", {})
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
            _sync_to_drive(model_dir, drive_best_model_dir, label="final")
            # Sync evaluations.npz and diagnostics.npz to the Drive trial dir
            drive_trial_dir = drive_best_model_dir.parent
            drive_trial_dir.mkdir(parents=True, exist_ok=True)
            for _npz_name in ("evaluations.npz", "diagnostics.npz"):
                _npz = trial_dir / _npz_name
                if _npz.exists():
                    try:
                        shutil.copy2(str(_npz), str(drive_trial_dir / _npz_name))
                    except OSError as e:
                        logger.warning("Drive sync failed for %s: %s", _npz_name, e)

        # Post-training evaluation for distance + forward velocity metrics.
        # Load the best model for evaluation (matches what gets handed off).
        from ...evaluation import eval_policy

        best_model_zip = model_dir / "best_model.zip"
        best_vecnorm = model_dir / "best_model_vecnorm.pkl"
        eval_model = model
        if best_model_zip.exists():
            eval_model = alg_cls.load(str(model_dir / "best_model"), env=eval_env)
            if best_vecnorm.exists():
                load_vecnorm_stats(str(best_vecnorm), eval_env)
        eval_env.training = False
        eval_env.norm_reward = False

        _, eval_lengths, eval_fwd_vels, eval_successes, eval_distances = eval_policy(
            eval_model,
            eval_env,
            species_cfg.success_keys,
            n_episodes=30,
        )
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

        # Final report
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(str(Path(tmpdir) / "model"))
            train_env.save(str(Path(tmpdir) / "vecnorm.pkl"))
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
    from .results import _evaluate_curriculum_gate

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

        # Curriculum thresholds
        row["reward_threshold"] = cur.get("min_avg_reward")
        row["ep_length_threshold"] = cur.get("min_avg_episode_length")
        row["forward_vel_threshold"] = cur.get("min_avg_forward_vel")
        row["success_rate_threshold"] = cur.get("min_success_rate")

        # Gate evaluation
        passed, _ = _evaluate_curriculum_gate(
            row["best_mean_reward"],
            row,
            row["reward_threshold"],
            row["ep_length_threshold"],
            row["forward_vel_threshold"],
            row["success_rate_threshold"],
        )
        row["stage_passed"] = passed
        rows.append(row)

    return rows
