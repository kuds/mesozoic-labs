"""
Shared training infrastructure for all dinosaur species.

Provides the common training, curriculum, and evaluation logic used by every
species.  Species-specific entry points (``environments/<species>/scripts/train_sb3.py``)
define a :class:`SpeciesConfig` and delegate to :func:`main`.

This module consolidates ~2,800 lines of duplicated code across three species
into a single ~850 line module.
"""

import argparse
import dataclasses
import logging
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

try:
    import numpy as _np
except ImportError:
    _np = None  # type: ignore[assignment]

try:
    from stable_baselines3.common.callbacks import BaseCallback as _BaseCallback
except ImportError:
    _BaseCallback = object  # type: ignore[misc,assignment]


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


def _cast_value(v: str):
    """Auto-cast a string value to int, float, or keep as string.

    Handles float-encoded integers (e.g. ``"128.0"`` → ``128``) which
    Vertex AI HPT sends for ``DiscreteParameterSpec`` values.
    """
    try:
        return int(v)
    except ValueError:
        try:
            f = float(v)
            if f.is_integer():
                return int(f)
            return f
        except ValueError:
            return v


def _apply_overrides(configs: dict, overrides: list | None) -> None:
    """Apply dot-notation ``key=value`` overrides to stage configs.

    Two formats are supported:

    - ``section.key=value``   — applies to **all** stages
    - ``N.section.key=value`` — applies to stage *N* only
    """
    if not overrides:
        return
    for item in overrides:
        key, _, raw_value = item.partition("=")
        value = _cast_value(raw_value)
        parts = key.split(".")
        if len(parts) == 3 and parts[0].isdigit():
            stage_num, section, param = int(parts[0]), parts[1], parts[2]
            kwargs_key = "env_kwargs" if section == "env" else f"{section}_kwargs"
            if stage_num in configs:
                configs[stage_num][kwargs_key][param] = value
                logger.info(
                    "Stage %d override: %s.%s = %r",
                    stage_num,
                    section,
                    param,
                    value,
                )
        else:
            section, _, param = key.partition(".")
            kwargs_key = "env_kwargs" if section == "env" else f"{section}_kwargs"
            for stage_config in configs.values():
                stage_config[kwargs_key][param] = value
            logger.info("Override applied: %s.%s = %r", section, param, value)


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


# ── Policy evaluation ────────────────────────────────────────────────────


def eval_policy(
    model,
    eval_env,
    success_keys: list[str],
    n_episodes: int = 25,
):
    """Evaluate a trained policy collecting per-episode metrics.

    Runs *n_episodes* deterministic rollouts and returns four parallel lists:

    * **rewards** – total reward per episode
    * **lengths** – step count per episode
    * **fwd_vels** – mean forward velocity per episode
    * **successes** – 1.0 if any *success_keys* triggered, else 0.0
    """
    import numpy as _np

    rewards: list[float] = []
    lengths: list[float] = []
    fwd_vels: list[float] = []
    successes: list[float] = []

    for _ in range(n_episodes):
        obs = eval_env.reset()
        ep_reward, ep_len = 0.0, 0
        ep_fwd: list[float] = []
        ep_success = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = eval_env.step(action)
            ep_reward += float(reward[0])
            ep_len += 1
            if "forward_vel" in infos[0]:
                ep_fwd.append(float(infos[0]["forward_vel"]))
            if any(infos[0].get(k) for k in success_keys):
                ep_success = 1.0
            done = bool(dones[0])
        rewards.append(ep_reward)
        lengths.append(float(ep_len))
        fwd_vels.append(float(_np.mean(ep_fwd)) if ep_fwd else 0.0)
        successes.append(ep_success)

    return rewards, lengths, fwd_vels, successes


# ── DiagnosticsCallback ─────────────────────────────────────────────────


class DiagnosticsCallback(_BaseCallback):
    """Logs per-component reward breakdowns and training diagnostics to TensorBoard.

    Tracked metrics (under ``diagnostics/`` in TensorBoard):
      - Per-component rewards: reward_forward, reward_alive, reward_energy, etc.
      - Environment state: forward_vel, prey_distance, pelvis_height, tilt_angle
      - Observation statistics: mean, std, max absolute value
      - Action statistics: mean, std
      - VecNormalize running variance for observations and returns
      - Termination reason breakdown (fraction per reason)
      - Reward plateau detection with console warnings

    When *log_dir* is provided, per-rollout averages of INFO_KEYS are also
    saved to ``diagnostics.npz`` so they can be plotted alongside the
    evaluation curves produced by the notebook's ``plot_training_curves``.
    """

    REWARD_KEYS = [
        "reward_forward",
        "reward_alive",
        "reward_energy",
        "reward_tail",
        "reward_posture",
        "reward_nosedive",
        "reward_smoothness",
        "reward_strike",
        "reward_approach",
        "reward_gait",
        "reward_heading",
        "reward_lateral",
    ]
    INFO_KEYS = [
        "forward_vel",
        "prey_distance",
        "pelvis_height",
        "tilt_angle",
        "tail_instability",
        "contact_asymmetry",
        "action_delta",
        "heading_alignment",
        "lateral_vel",
        "forward_z",
        "approach_delta",
        "strike_success",
        "r_foot_contact",
        "l_foot_contact",
    ]

    def __init__(self, plateau_window=10, plateau_threshold=1.0, log_dir=None, verbose=0):
        super().__init__(verbose)
        self.plateau_window = plateau_window
        self.plateau_threshold = plateau_threshold
        self._log_dir = Path(log_dir) if log_dir is not None else None
        self._step_infos = {k: [] for k in self.REWARD_KEYS + self.INFO_KEYS}
        self._rollout_ep_rewards: list[float] = []
        self._rollout_terminations: Counter = Counter()
        self._history_timesteps: list[int] = []
        self._history = {k: [] for k in self.INFO_KEYS}
        self._history_rewards = {k: [] for k in self.REWARD_KEYS}
        self._history_terminations: dict[str, list[float]] = {}
        self._history_term_timesteps: list[int] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            for key in self.REWARD_KEYS + self.INFO_KEYS:
                if key in info:
                    self._step_infos[key].append(float(info[key]))
            if "termination_reason" in info:
                self._rollout_terminations[info["termination_reason"]] += 1
        return True

    def _on_rollout_end(self) -> None:
        for key, values in self._step_infos.items():
            if values:
                self.logger.record(f"diagnostics/{key}", _np.mean(values))

        has_info = any(self._step_infos[k] for k in self.INFO_KEYS)
        if has_info:
            self._history_timesteps.append(self.num_timesteps)
            for key in self.INFO_KEYS:
                vals = self._step_infos[key]
                self._history[key].append(float(_np.mean(vals)) if vals else float("nan"))
            for key in self.REWARD_KEYS:
                vals = self._step_infos[key]
                self._history_rewards[key].append(float(_np.mean(vals)) if vals else float("nan"))
            self._save_diagnostics()

        self._step_infos = {k: [] for k in self.REWARD_KEYS + self.INFO_KEYS}

        total_terminations = sum(self._rollout_terminations.values())
        if total_terminations > 0:
            self._history_term_timesteps.append(self.num_timesteps)
            for reason, count in self._rollout_terminations.items():
                frac = count / total_terminations
                self.logger.record(f"terminations/{reason}", frac)
                self._history_terminations.setdefault(reason, []).append(frac)
            self.logger.record("terminations/total_count", total_terminations)
            self._save_diagnostics()
        self._rollout_terminations.clear()

        if hasattr(self.model, "rollout_buffer") and self.model.rollout_buffer.observations is not None:
            obs = self.model.rollout_buffer.observations
            self.logger.record("diagnostics/obs_mean", float(_np.mean(obs)))
            self.logger.record("diagnostics/obs_std", float(_np.std(obs)))
            self.logger.record("diagnostics/obs_max_abs", float(_np.max(_np.abs(obs))))

        if hasattr(self.model, "rollout_buffer") and self.model.rollout_buffer.actions is not None:
            acts = self.model.rollout_buffer.actions
            self.logger.record("diagnostics/action_mean", float(_np.mean(acts)))
            self.logger.record("diagnostics/action_std", float(_np.std(acts)))

        env = self.training_env
        if hasattr(env, "obs_rms"):
            self.logger.record("diagnostics/vecnorm_obs_var_mean", float(_np.mean(env.obs_rms.var)))
        if hasattr(env, "ret_rms"):
            self.logger.record("diagnostics/vecnorm_ret_var", float(_np.mean(env.ret_rms.var)))

        ep_rewards = [info["episode"]["r"] for info in self.locals.get("infos", []) if "episode" in info]
        if ep_rewards:
            self._rollout_ep_rewards.append(_np.mean(ep_rewards))
            if len(self._rollout_ep_rewards) >= self.plateau_window:
                recent = self._rollout_ep_rewards[-self.plateau_window:]
                variation = max(recent) - min(recent)
                self.logger.record("diagnostics/reward_variation", variation)
                if variation < self.plateau_threshold:
                    print(
                        f"\n*** PLATEAU WARNING: Reward variation over last "
                        f"{self.plateau_window} rollouts is only {variation:.4f}. "
                        f"Consider adjusting learning rate or stopping. ***\n"
                    )

    def _save_diagnostics(self) -> None:
        """Persist accumulated diagnostics to an npz file in the stage dir."""
        if self._log_dir is None:
            return
        save_dict = {"timesteps": _np.array(self._history_timesteps)}
        for key in self.INFO_KEYS:
            save_dict[key] = _np.array(self._history[key])
        for key in self.REWARD_KEYS:
            arr = self._history_rewards[key]
            if len(arr) == len(self._history_timesteps):
                save_dict[key] = _np.array(arr)
        if self._history_term_timesteps:
            save_dict["term_timesteps"] = _np.array(self._history_term_timesteps)
            for reason, fracs in self._history_terminations.items():
                save_dict[f"term_{reason}"] = _np.array(fracs)
        _np.savez(self._log_dir / "diagnostics.npz", **save_dict)


# ── Video recording ──────────────────────────────────────────────────────


def record_stage_video(
    model,
    env_class: type,
    env_kwargs: dict,
    stage: int,
    stage_dir,
    species: str = "dino",
    algorithm: str = "ppo",
    seed: int = 42,
    vecnorm_path: str | None = None,
    max_steps: int = 1000,
):
    """Record and save a video of the trained policy for a given stage.

    When *vecnorm_path* is provided, observations are normalized using the
    saved ``VecNormalize`` running statistics so the policy sees the same
    input distribution it was trained on.

    Requires the ``mediapy`` package (``pip install mediapy``).
    """
    try:
        import mediapy
    except ImportError:
        logger.warning("Skipping video for stage %d (mediapy not installed).", stage)
        return

    sb3 = _ensure_sb3()
    render_env = env_class(render_mode="rgb_array", **env_kwargs)

    vec_normalize = None
    if vecnorm_path and Path(vecnorm_path).exists():
        dummy_env = sb3["DummyVecEnv"]([lambda: env_class(**env_kwargs)])
        vec_normalize = sb3["VecNormalize"].load(vecnorm_path, dummy_env)
        vec_normalize.training = False
        vec_normalize.norm_reward = False

    obs, _ = render_env.reset(seed=seed + 2000 + stage)
    frames = []
    episode_reward = 0.0

    for _ in range(max_steps):
        if vec_normalize is not None:
            obs_input = vec_normalize.normalize_obs(obs)
        else:
            obs_input = obs
        action, _ = model.predict(obs_input, deterministic=True)
        obs, reward, terminated, truncated, info = render_env.step(action)
        frames.append(render_env.render())
        episode_reward += reward
        if terminated or truncated:
            break

    render_env.close()
    if vec_normalize is not None:
        vec_normalize.close()

    video_path = str(Path(stage_dir) / f"{species}_{algorithm.lower()}_stage{stage}.mp4")
    mediapy.write_video(video_path, frames, fps=50)
    logger.info("Stage %d video: reward=%.2f | %d frames", stage, episode_reward, len(frames))
    logger.info("  Saved to: %s", video_path)
    return video_path, frames


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
    from environments.shared.config import save_stage_config
    from environments.shared.curriculum import (
        RewardRampCallback,
        StageWarmupCallback,
        load_vecnorm_stats,
    )
    from environments.shared.wandb_integration import WandbCallback, init_wandb

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
        load_vecnorm_stats(_vecnorm_path, train_env, eval_env)

    # Create or load model
    alg_cls = sb3["SAC"] if algorithm == "sac" else sb3["PPO"]
    alg_kwargs = config["sac_kwargs"].copy() if algorithm == "sac" else config["ppo_kwargs"].copy()
    alg_kwargs["verbose"] = verbose
    alg_kwargs["tensorboard_log"] = str(log_path / "tensorboard")

    if algorithm == "ppo":
        lr_end = alg_kwargs.pop("learning_rate_end", None)
        if lr_end is not None:
            lr_start = alg_kwargs["learning_rate"]
            alg_kwargs["learning_rate"] = linear_schedule(lr_start, lr_end)
            logger.info("Using linear LR schedule: %s -> %s", lr_start, lr_end)

    wandb_run = None
    if use_wandb:
        wandb_run = init_wandb(species=species, stage=stage, config=config)
        logger.info("W&B run initialized.")

    if load_path:
        logger.info("Loading model from: %s", load_path)
        model = alg_cls.load(load_path, env=train_env, **alg_kwargs)
    else:
        logger.info("Creating new %s model...", algorithm.upper())
        model = alg_cls("MlpPolicy", train_env, **alg_kwargs)

    logger.info("Model architecture:")
    logger.info("  Policy: %s", model.policy)
    logger.info("  Learning rate: %s", model.learning_rate)
    logger.info("  Batch size: %s", alg_kwargs["batch_size"])

    # Setup callbacks
    callbacks = []

    eval_callback = sb3["EvalCallback"](
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(log_path),
        eval_freq=eval_freq // n_envs,
        n_eval_episodes=20,
        deterministic=True,
        render=False,
        verbose=max(verbose, 1),
    )
    callbacks.append(eval_callback)

    checkpoint_callback = sb3["CheckpointCallback"](
        save_freq=save_freq // n_envs,
        save_path=str(model_dir),
        name_prefix=f"stage{stage}",
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)

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
    _report_hpt_metrics(species_cfg, model, eval_env, eval_callback, log_path, stage, total_timesteps)

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
    stage: int,
    total_timesteps: int,
):
    """Report metrics to Vertex AI Hypertune (no-op when not installed)."""
    try:
        import hypertune as _hypertune
    except ImportError:
        return

    import numpy as _np

    _hpt = _hypertune.HyperTune()
    _hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag="best_mean_reward",
        metric_value=eval_callback.best_mean_reward,
        global_step=total_timesteps,
    )
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
        _hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag="best_mean_episode_length",
            metric_value=best_mean_ep_length,
            global_step=total_timesteps,
        )
        logger.info(
            "HPT metric reported: best_mean_episode_length=%.1f",
            best_mean_ep_length,
        )

        last_mean_reward = float(mean_rewards_per_eval[-1])
        last_mean_ep_length = float(eval_lengths[-1].mean())
        _hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag="last_mean_reward",
            metric_value=last_mean_reward,
            global_step=total_timesteps,
        )
        _hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag="last_mean_episode_length",
            metric_value=last_mean_ep_length,
            global_step=total_timesteps,
        )
        logger.info(
            "HPT metric reported: last_mean_reward=%.4f, last_mean_episode_length=%.1f",
            last_mean_reward,
            last_mean_ep_length,
        )

    if stage >= 2:
        _, _, fwd_vels, success_flags = eval_policy(model, eval_env, species_cfg.success_keys, n_episodes=10)
        if fwd_vels:
            best_fwd = float(_np.mean(fwd_vels))
            _hpt.report_hyperparameter_tuning_metric(
                hyperparameter_metric_tag="best_mean_forward_vel",
                metric_value=best_fwd,
                global_step=total_timesteps,
            )
            logger.info("HPT metric reported: best_mean_forward_vel=%.4f", best_fwd)
        if stage >= 3 and success_flags:
            best_success = float(_np.mean(success_flags))
            _hpt.report_hyperparameter_tuning_metric(
                hyperparameter_metric_tag="best_mean_success_rate",
                metric_value=best_success,
                global_step=total_timesteps,
            )
            logger.info(
                "HPT metric reported: best_mean_success_rate=%.4f",
                best_success,
            )


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
    from environments.shared.config import (
        save_stage_config,
        upload_curriculum_artifacts,
    )
    from environments.shared.curriculum import (
        CurriculumCallback,
        CurriculumManager,
        RewardRampCallback,
        StageWarmupCallback,
        load_vecnorm_stats,
        thresholds_from_configs,
    )
    from environments.shared.wandb_integration import WandbCallback, init_wandb

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
            load_vecnorm_stats(prev_vecnorm_path, train_env, eval_env)

        alg_cls = sb3["SAC"] if algorithm == "sac" else sb3["PPO"]
        alg_kwargs = config["sac_kwargs"].copy() if algorithm == "sac" else config["ppo_kwargs"].copy()
        alg_kwargs["verbose"] = verbose
        alg_kwargs["tensorboard_log"] = str(stage_dir / "tensorboard")

        if algorithm == "ppo":
            lr_end = alg_kwargs.pop("learning_rate_end", None)
            if lr_end is not None:
                lr_start = alg_kwargs["learning_rate"]
                alg_kwargs["learning_rate"] = linear_schedule(lr_start, lr_end)
                logger.info("Using linear LR schedule: %s -> %s", lr_start, lr_end)

        wandb_run = None
        if use_wandb:
            wandb_run = init_wandb(species=species, stage=stage, config=config)

        if load_path:
            logger.info("Loading model from previous stage: %s", load_path)
            model = alg_cls.load(load_path, env=train_env, **alg_kwargs)
        else:
            model = alg_cls("MlpPolicy", train_env, **alg_kwargs)

        # Build callbacks
        callbacks = []

        eval_callback = sb3["EvalCallback"](
            eval_env,
            best_model_save_path=str(model_dir),
            log_path=str(stage_dir),
            eval_freq=eval_freq // n_envs,
            n_eval_episodes=20,
            deterministic=True,
            render=False,
            verbose=max(verbose, 1),
        )
        callbacks.append(eval_callback)

        checkpoint_callback = sb3["CheckpointCallback"](
            save_freq=save_freq // n_envs,
            save_path=str(model_dir),
            name_prefix=f"stage{stage}",
            save_vecnormalize=True,
        )
        callbacks.append(checkpoint_callback)

        if use_wandb:
            callbacks.append(WandbCallback())

        curriculum_cb = CurriculumCallback(
            curriculum_manager=manager,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=10,
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
        load_path = str(final_path)
        prev_vecnorm_path = str(final_path) + "_vecnorm.pkl"

        logger.info("Stage %d complete. Model saved to %s", stage, final_path)

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

    from environments.shared.config import append_stage_result_csv

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


# ── Evaluation ───────────────────────────────────────────────────────────


def evaluate(
    species_cfg: SpeciesConfig,
    stage_configs: Dict[int, Dict[str, Any]],
    model_path: str,
    n_episodes: int = 10,
    render: bool = True,
    stage: int | None = None,
    algorithm: str = "ppo",
):
    """Evaluate a trained model with full locomotion metrics."""
    from environments.shared.metrics import LocomotionMetrics

    sb3 = _ensure_sb3()

    logger.info("Loading model from: %s", model_path)

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
        env = species_cfg.env_class(render_mode=render_mode, **env_kwargs)
        return sb3["Monitor"](env)

    vec_env = sb3["DummyVecEnv"]([_make_eval_env])

    if Path(vecnorm_path).exists():
        logger.info("Loading normalization stats from: %s", vecnorm_path)
        vec_env = sb3["VecNormalize"].load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        logger.warning("No VecNormalize stats found. Results may differ from training.")

    alg_cls = sb3["SAC"] if algorithm == "sac" else sb3["PPO"]
    model = alg_cls.load(model_path, env=vec_env)

    logger.info(
        "Evaluating for %d episodes (stage %d: %s)...",
        n_episodes,
        stage,
        stage_configs[stage]["name"],
    )

    episode_reports = []

    for ep in range(n_episodes):
        obs = vec_env.reset()
        metrics = LocomotionMetrics()
        total_reward = 0.0
        step = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(action)
            step_reward = float(rewards[0])
            total_reward += step_reward
            step += 1
            metrics.record_step(infos[0], step_reward)

            if dones[0]:
                break

        report = metrics.compute()
        episode_reports.append(report)

        term_reason = report.get("termination_reason", "truncated")
        logger.info(
            "  Episode %d: reward=%.2f, length=%d, fwd_vel=%.3f m/s, tilt=%.2f rad, ended=%s",
            ep + 1,
            total_reward,
            step,
            report.get("mean_forward_velocity", 0.0),
            report.get("mean_tilt_angle", 0.0),
            term_reason,
        )

    vec_env.close()

    agg = LocomotionMetrics.aggregate_episodes(episode_reports)
    _log_eval_results(species_cfg, agg, n_episodes)


def _log_eval_results(
    species_cfg: SpeciesConfig,
    agg: dict,
    n_episodes: int,
):
    """Log formatted evaluation results."""
    logger.info("=" * 60)
    logger.info("Evaluation Results (%d episodes)", n_episodes)
    logger.info("=" * 60)

    logger.info("--- Core Performance ---")
    logger.info(
        "  Reward:       %.2f +/- %.2f",
        agg.get("mean_total_reward", 0),
        agg.get("std_total_reward", 0),
    )
    logger.info(
        "  Ep Length:    %.1f +/- %.1f",
        agg.get("mean_episode_length", 0),
        agg.get("std_episode_length", 0),
    )

    logger.info("--- Velocity ---")
    logger.info(
        "  Forward vel:  %.3f +/- %.3f m/s",
        agg.get("mean_mean_forward_velocity", 0),
        agg.get("std_mean_forward_velocity", 0),
    )
    logger.info("  Max fwd vel:  %.3f m/s", agg.get("mean_max_forward_velocity", 0))
    logger.info("  Consistency:  %.3f", agg.get("mean_velocity_consistency", 0))
    logger.info(
        "  Distance:     %.3f +/- %.3f m",
        agg.get("mean_total_distance", 0),
        agg.get("std_total_distance", 0),
    )

    logger.info("--- Gait Quality ---")
    logger.info("  Symmetry:     %.3f", agg.get("mean_gait_symmetry", 0))
    logger.info("  Stride freq:  %.3f Hz", agg.get("mean_stride_frequency", 0))
    logger.info("  Cost of transport: %.4f", agg.get("mean_cost_of_transport", 0))

    logger.info("--- Balance ---")
    logger.info(
        "  %s: %.3f +/- %.3f m",
        species_cfg.height_label,
        agg.get("mean_mean_pelvis_height", 0),
        agg.get("std_mean_pelvis_height", 0),
    )
    logger.info(
        "  Mean tilt:     %.3f +/- %.3f rad",
        agg.get("mean_mean_tilt_angle", 0),
        agg.get("std_mean_tilt_angle", 0),
    )
    logger.info("  Max tilt:      %.3f rad", agg.get("mean_max_tilt_angle", 0))

    if "mean_initial_prey_distance" in agg:
        logger.info("--- %s ---", species_cfg.stage3_section_label)
        logger.info(
            "  Initial dist:   %.3f m",
            agg.get("mean_initial_prey_distance", 0),
        )
        logger.info("  Final dist:     %.3f m", agg.get("mean_final_prey_distance", 0))
        logger.info("  Min dist:       %.3f m", agg.get("mean_min_prey_distance", 0))
        logger.info("  Time to target: %.3f s", agg.get("mean_time_to_target", -1))
    if "mean_heading_alignment" in agg:
        logger.info("  Heading align:  %.3f", agg.get("mean_heading_alignment", 0))
    if "mean_success_rate" in agg:
        logger.info(
            "  Success rate:   %.1f%%",
            100.0 * agg.get("mean_success_rate", 0),
        )

    term_counts = agg.get("termination_counts")
    if term_counts:
        logger.info("--- Termination Reasons ---")
        for reason, count in sorted(term_counts.items(), key=lambda x: -x[1]):
            pct = 100.0 * count / n_episodes
            logger.info("  %-20s %d (%.0f%%)", reason, count, pct)

    logger.info("=" * 60)


# ── CLI entry point ──────────────────────────────────────────────────────


def main(species_cfg: SpeciesConfig):
    """Parse arguments and dispatch to train/curriculum/evaluate."""
    from environments.shared.config import load_all_stages

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stage_configs = load_all_stages(species_cfg.species)

    parser = argparse.ArgumentParser(description=f"Train {species_cfg.species.title()} with SB3 PPO")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ── train ─────────────────────────────────────────────────────────
    train_parser = subparsers.add_parser("train", help="Train a policy")
    train_parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help=f"Curriculum stage ({species_cfg.stage_descriptions})",
    )
    train_parser.add_argument("--timesteps", type=int, default=500000, help="Total training timesteps")
    train_parser.add_argument("--n-envs", type=int, default=4, help="Number of parallel environments")
    train_parser.add_argument("--load", type=str, default=None, help="Path to model to continue from")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    train_parser.add_argument("--eval-freq", type=int, default=10000, help="Evaluation frequency")
    train_parser.add_argument("--save-freq", type=int, default=50000, help="Checkpoint frequency")
    train_parser.add_argument("--log-dir", type=str, default=None, help="Custom log directory")
    train_parser.add_argument("--subproc", action="store_true", help="Use subprocess vectorization")
    train_parser.add_argument(
        "--verbose",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="Verbose level: 0=eval only, 1=progress bar (default), 2=debug",
    )
    train_parser.add_argument(
        "--algorithm",
        type=str,
        choices=["ppo", "sac"],
        default="ppo",
        help="RL algorithm",
    )
    train_parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    train_parser.add_argument(
        "--override",
        nargs="*",
        default=None,
        metavar="KEY=VALUE",
        help="Override config values, e.g. ppo.learning_rate=1e-4",
    )
    train_parser.add_argument("--output-dir", type=str, default=None, help="Base output directory")

    # ── curriculum ────────────────────────────────────────────────────
    cur_parser = subparsers.add_parser("curriculum", help="Run automated end-to-end curriculum (stages 1-3)")
    cur_parser.add_argument("--n-envs", type=int, default=4)
    cur_parser.add_argument("--seed", type=int, default=42)
    cur_parser.add_argument("--eval-freq", type=int, default=10000)
    cur_parser.add_argument("--save-freq", type=int, default=50000)
    cur_parser.add_argument("--log-dir", type=str, default=None)
    cur_parser.add_argument("--subproc", action="store_true")
    cur_parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=1)
    cur_parser.add_argument("--algorithm", type=str, choices=["ppo", "sac"], default="ppo")
    cur_parser.add_argument("--wandb", action="store_true")
    cur_parser.add_argument("--override", nargs="*", default=None, metavar="KEY=VALUE")
    cur_parser.add_argument("--output-dir", type=str, default=None)
    cur_parser.add_argument("--gcs-bucket", type=str, default=None)
    cur_parser.add_argument("--gcs-project", type=str, default=None)

    # ── eval ──────────────────────────────────────────────────────────
    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained policy")
    eval_parser.add_argument("model_path", type=str, help="Path to trained model")
    eval_parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Curriculum stage (auto-detected if omitted)",
    )
    eval_parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    eval_parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    eval_parser.add_argument("--algorithm", type=str, choices=["ppo", "sac"], default="ppo")

    # ── dispatch ──────────────────────────────────────────────────────
    args = parser.parse_args()

    if args.command == "train" or args.command is None:
        if args.command is None:
            args.stage = 1
            args.timesteps = 500000
            args.n_envs = 4
            args.load = None
            args.seed = 42
            args.eval_freq = 10000
            args.save_freq = 50000
            args.log_dir = None
            args.subproc = False
            args.verbose = 1
            args.algorithm = "ppo"
            args.wandb = False
            args.override = None
            args.output_dir = None

        _apply_overrides(stage_configs, args.override)
        train(
            species_cfg=species_cfg,
            stage_configs=stage_configs,
            stage=args.stage,
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            seed=args.seed,
            load_path=args.load,
            eval_freq=args.eval_freq,
            save_freq=args.save_freq,
            log_dir=args.log_dir,
            use_subproc=args.subproc,
            verbose=args.verbose,
            algorithm=args.algorithm,
            use_wandb=args.wandb,
            output_dir=args.output_dir,
        )

    elif args.command == "curriculum":
        _apply_overrides(stage_configs, args.override)
        train_curriculum(
            species_cfg=species_cfg,
            stage_configs=stage_configs,
            n_envs=args.n_envs,
            seed=args.seed,
            eval_freq=args.eval_freq,
            save_freq=args.save_freq,
            log_dir=args.log_dir,
            use_subproc=args.subproc,
            verbose=args.verbose,
            algorithm=args.algorithm,
            use_wandb=args.wandb,
            output_dir=args.output_dir,
            gcs_bucket=args.gcs_bucket,
            gcs_project=args.gcs_project,
        )

    elif args.command == "eval":
        evaluate(
            species_cfg=species_cfg,
            stage_configs=stage_configs,
            model_path=args.model_path,
            n_episodes=args.episodes,
            render=not args.no_render,
            stage=args.stage,
            algorithm=args.algorithm,
        )
