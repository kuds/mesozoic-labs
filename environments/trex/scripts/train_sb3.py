#!/usr/bin/env python3
"""
Train T-Rex with Stable-Baselines3 PPO.

Supports curriculum learning with three stages:
1. Standing/balance (no forward velocity reward)
2. Walking (moderate speed target)
3. Sprinting + bite (full reward)

Usage:
    # Single-stage training
    python train_sb3.py train --stage 1 --timesteps 1000000
    python train_sb3.py train --stage 2 --timesteps 2000000 --load models/stage1_final.zip
    python train_sb3.py train --stage 3 --timesteps 3000000 --load models/stage2_final.zip

    # Automated end-to-end curriculum (all 3 stages)
    python train_sb3.py curriculum --n-envs 4

    # Evaluate a trained model
    python train_sb3.py eval models/stage3_final.zip --episodes 10
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add repo root to path
_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
except ImportError:
    logger.error("stable-baselines3 not installed. Install with: pip install stable-baselines3[extra]")
    sys.exit(1)

from environments.shared.config import load_all_stages, save_stage_config
from environments.shared.curriculum import (
    CurriculumCallback,
    CurriculumManager,
    thresholds_from_configs,
)
from environments.shared.metrics import LocomotionMetrics
from environments.trex.envs.trex_env import TRexEnv

# Load curriculum configs from TOML files (configs/trex/)
STAGE_CONFIGS = load_all_stages("trex")


def make_env(stage: int, rank: int, seed: int = 0):
    """Create a single environment instance."""

    def _init():
        env_kwargs = STAGE_CONFIGS[stage]["env_kwargs"].copy()
        env = TRexEnv(**env_kwargs)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


def create_vec_env(stage: int, n_envs: int, seed: int = 0, use_subproc: bool = False):
    """Create vectorized environment."""
    if use_subproc and n_envs > 1:
        env = SubprocVecEnv([make_env(stage, i, seed) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(stage, i, seed) for i in range(n_envs)])

    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )

    return env


def train(
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
):
    """Train the T-Rex policy."""

    config = STAGE_CONFIGS[stage]
    logger.info("=" * 60)
    logger.info("Training Stage %d: %s", stage, config["name"])
    logger.info("Description: %s", config["description"])
    logger.info("=" * 60)

    # Setup directories (organised as <species>/<datetime>/)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path: Path
    if log_dir is None:
        log_path = Path(__file__).parent.parent / "logs" / "trex" / f"stage{stage}_{timestamp}"
    else:
        log_path = Path(log_dir)

    log_path.mkdir(parents=True, exist_ok=True)
    model_dir = log_path / "models"
    model_dir.mkdir(exist_ok=True)

    logger.info("Log directory: %s", log_path)
    logger.info("Model directory: %s", model_dir)

    # Save reward weights and hyperparameters for reproducibility
    save_stage_config(
        log_path,
        stage,
        config,
        "PPO",
        extra={"seed": seed, "n_envs": n_envs, "timesteps": total_timesteps},
    )

    logger.info("Creating %d training environments...", n_envs)
    train_env = create_vec_env(stage, n_envs, seed, use_subproc)

    logger.info("Creating evaluation environment...")
    eval_env = create_vec_env(stage, 1, seed + 1000, use_subproc=False)

    ppo_kwargs = config["ppo_kwargs"].copy()
    ppo_kwargs["verbose"] = verbose
    ppo_kwargs["tensorboard_log"] = str(log_path / "tensorboard")

    if load_path:
        logger.info("Loading model from: %s", load_path)
        # Pass all stage hyperparameters so rollout buffer, gamma, etc. are
        # re-initialised correctly for the new stage.
        model = PPO.load(load_path, env=train_env, **ppo_kwargs)
    else:
        logger.info("Creating new PPO model...")
        model = PPO(
            "MlpPolicy",
            train_env,
            **ppo_kwargs,
        )

    logger.info("Model architecture:")
    logger.info("  Policy: %s", model.policy)
    logger.info("  Learning rate: %s", model.learning_rate)
    logger.info("  Batch size: %s", ppo_kwargs["batch_size"])

    callbacks = []

    eval_callback = EvalCallback(
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

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // n_envs,
        save_path=str(model_dir),
        name_prefix=f"stage{stage}",
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)

    callback_list = CallbackList(callbacks)

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

    final_path = model_dir / f"stage{stage}_final"
    logger.info("Saving final model to: %s", final_path)
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


def train_curriculum(
    n_envs: int = 4,
    seed: int = 42,
    eval_freq: int = 10000,
    save_freq: int = 50000,
    log_dir: str | None = None,
    use_subproc: bool = False,
    verbose: int = 1,
):
    """Run the full 3-stage curriculum with automatic advancement."""
    thresholds = thresholds_from_configs(STAGE_CONFIGS)
    manager = CurriculumManager(
        species="trex",
        stage_thresholds=thresholds,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if log_dir is None:
        base_dir = Path(__file__).parent.parent / "logs" / "trex" / f"curriculum_{timestamp}"
    else:
        base_dir = Path(log_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Starting automated curriculum training (stages 1-3)")
    logger.info("Base directory: %s", base_dir)
    logger.info("=" * 60)

    model = None
    load_path = None

    for stage in range(1, 4):
        config = STAGE_CONFIGS[stage]
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

        # Save reward weights and hyperparameters for reproducibility
        save_stage_config(
            stage_dir,
            stage,
            config,
            "PPO",
            extra={"seed": seed, "n_envs": n_envs, "timesteps": total_timesteps},
        )

        train_env = create_vec_env(stage, n_envs, seed, use_subproc)
        eval_env = create_vec_env(stage, 1, seed + 1000, use_subproc=False)

        ppo_kwargs = config["ppo_kwargs"].copy()
        ppo_kwargs["verbose"] = verbose
        ppo_kwargs["tensorboard_log"] = str(stage_dir / "tensorboard")

        if load_path:
            logger.info("Loading model from previous stage: %s", load_path)
            # Pass all stage hyperparameters so rollout buffer, gamma, etc. are
            # re-initialised correctly for the new stage.
            model = PPO.load(load_path, env=train_env, **ppo_kwargs)
        else:
            model = PPO("MlpPolicy", train_env, **ppo_kwargs)

        callbacks = []

        eval_callback = EvalCallback(
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

        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq // n_envs,
            save_path=str(model_dir),
            name_prefix=f"stage{stage}",
            save_vecnormalize=True,
        )
        callbacks.append(checkpoint_callback)

        curriculum_cb = None
        if stage < 3:
            curriculum_cb = CurriculumCallback(
                curriculum_manager=manager,
                eval_env=eval_env,
                eval_freq=eval_freq,
                n_eval_episodes=10,
            )
            callbacks.append(curriculum_cb)

        try:
            model.learn(
                total_timesteps=total_timesteps,
                callback=CallbackList(callbacks),
                progress_bar=verbose >= 1,
            )
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user.")
            break

        final_path = model_dir / f"stage{stage}_final"
        model.save(str(final_path))
        train_env.save(str(final_path) + "_vecnorm.pkl")
        load_path = str(final_path)

        logger.info("Stage %d complete. Model saved to %s", stage, final_path)

        train_env.close()
        eval_env.close()

        if curriculum_cb and curriculum_cb.ready_to_advance:
            manager.advance()
            logger.info("Auto-advanced to stage %d", manager.current_stage)
        elif stage < 3:
            logger.warning(
                "Stage %d timestep budget exhausted without meeting advancement thresholds. Advancing anyway.",
                stage,
            )
            manager.advance()

    logger.info("=" * 60)
    logger.info("Curriculum training complete!")
    logger.info("Results directory: %s", base_dir)
    logger.info("=" * 60)


def evaluate(model_path: str, n_episodes: int = 10, render: bool = True, stage: int | None = None):
    """Evaluate a trained model."""
    logger.info("Loading model from: %s", model_path)

    if stage is None:
        stage = 1
        for s in [1, 2, 3]:
            if f"stage{s}" in model_path:
                stage = s
                break
        logger.info("Auto-detected stage %d from filename", stage)

    env_kwargs = STAGE_CONFIGS[stage]["env_kwargs"].copy()

    vecnorm_path = model_path.replace(".zip", "_vecnorm.pkl")
    if not vecnorm_path.endswith("_vecnorm.pkl"):
        vecnorm_path = model_path + "_vecnorm.pkl"

    render_mode = "human" if render else None

    def _make_eval_env():
        env = TRexEnv(render_mode=render_mode, **env_kwargs)
        return Monitor(env)

    vec_env = DummyVecEnv([_make_eval_env])

    if Path(vecnorm_path).exists():
        logger.info("Loading normalization stats from: %s", vecnorm_path)
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        logger.warning("No VecNormalize stats found.")

    model = PPO.load(model_path, env=vec_env)

    config_name = STAGE_CONFIGS[stage]["name"]
    logger.info("Evaluating for %d episodes (stage %d: %s)...", n_episodes, stage, config_name)

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

    logger.info("=" * 60)
    logger.info("Evaluation Results (%d episodes)", n_episodes)
    logger.info("=" * 60)

    # Core performance
    logger.info("--- Core Performance ---")
    logger.info("  Reward:       %.2f +/- %.2f", agg.get("mean_total_reward", 0), agg.get("std_total_reward", 0))
    logger.info("  Ep Length:    %.1f +/- %.1f", agg.get("mean_episode_length", 0), agg.get("std_episode_length", 0))

    # Velocity
    logger.info("--- Velocity ---")
    logger.info(
        "  Forward vel:  %.3f +/- %.3f m/s",
        agg.get("mean_mean_forward_velocity", 0),
        agg.get("std_mean_forward_velocity", 0),
    )
    logger.info("  Max fwd vel:  %.3f m/s", agg.get("mean_max_forward_velocity", 0))
    logger.info("  Consistency:  %.3f", agg.get("mean_velocity_consistency", 0))
    logger.info("  Distance:     %.3f +/- %.3f m", agg.get("mean_total_distance", 0), agg.get("std_total_distance", 0))

    # Gait quality
    logger.info("--- Gait Quality ---")
    logger.info("  Symmetry:     %.3f", agg.get("mean_gait_symmetry", 0))
    logger.info("  Stride freq:  %.3f Hz", agg.get("mean_stride_frequency", 0))
    logger.info("  Cost of transport: %.4f", agg.get("mean_cost_of_transport", 0))

    # Balance
    logger.info("--- Balance ---")
    logger.info(
        "  Pelvis height: %.3f +/- %.3f m", agg.get("mean_mean_pelvis_height", 0), agg.get("std_mean_pelvis_height", 0)
    )
    logger.info(
        "  Mean tilt:     %.3f +/- %.3f rad", agg.get("mean_mean_tilt_angle", 0), agg.get("std_mean_tilt_angle", 0)
    )
    logger.info("  Max tilt:      %.3f rad", agg.get("mean_max_tilt_angle", 0))

    # Hunting (stage 3)
    if "mean_initial_prey_distance" in agg:
        logger.info("--- Hunting ---")
        logger.info("  Initial dist:   %.3f m", agg.get("mean_initial_prey_distance", 0))
        logger.info("  Final dist:     %.3f m", agg.get("mean_final_prey_distance", 0))
        logger.info("  Min dist:       %.3f m", agg.get("mean_min_prey_distance", 0))
        logger.info("  Time to target: %.3f s", agg.get("mean_time_to_target", -1))
    if "mean_heading_alignment" in agg:
        logger.info("  Heading align:  %.3f", agg.get("mean_heading_alignment", 0))
    if "mean_success_rate" in agg:
        logger.info("  Bite success:   %.1f%%", 100.0 * agg.get("mean_success_rate", 0))

    # Termination reasons
    term_counts = agg.get("termination_counts")
    if term_counts:
        logger.info("--- Termination Reasons ---")
        for reason, count in sorted(term_counts.items(), key=lambda x: -x[1]):
            pct = 100.0 * count / n_episodes
            logger.info("  %-20s %d (%.0f%%)", reason, count, pct)

    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Train T-Rex with SB3 PPO")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a policy")
    train_parser.add_argument(
        "--stage", type=int, choices=[1, 2, 3], default=1, help="Curriculum stage (1=balance, 2=locomotion, 3=bite)"
    )
    train_parser.add_argument("--timesteps", type=int, default=500000, help="Total training timesteps")
    train_parser.add_argument("--n-envs", type=int, default=4, help="Number of parallel environments")
    train_parser.add_argument("--load", type=str, default=None, help="Path to model to continue training from")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    train_parser.add_argument("--eval-freq", type=int, default=10000, help="Evaluation frequency (timesteps)")
    train_parser.add_argument("--save-freq", type=int, default=50000, help="Checkpoint save frequency (timesteps)")
    train_parser.add_argument("--log-dir", type=str, default=None, help="Custom log directory")
    train_parser.add_argument("--subproc", action="store_true", help="Use subprocess vectorization")
    train_parser.add_argument(
        "--verbose",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="Verbose level: 0=eval results only, 1=training stats + progress bar (default), 2=debug",
    )

    # Curriculum command
    cur_parser = subparsers.add_parser("curriculum", help="Run automated end-to-end curriculum (stages 1-3)")
    cur_parser.add_argument("--n-envs", type=int, default=4, help="Number of parallel environments")
    cur_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    cur_parser.add_argument("--eval-freq", type=int, default=10000, help="Evaluation frequency (timesteps)")
    cur_parser.add_argument("--save-freq", type=int, default=50000, help="Checkpoint save frequency (timesteps)")
    cur_parser.add_argument("--log-dir", type=str, default=None, help="Custom log directory")
    cur_parser.add_argument("--subproc", action="store_true", help="Use subprocess vectorization")
    cur_parser.add_argument(
        "--verbose",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="Verbose level: 0=eval results only, 1=training stats + progress bar (default), 2=debug",
    )

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained policy")
    eval_parser.add_argument("model_path", type=str, help="Path to trained model")
    eval_parser.add_argument(
        "--stage", type=int, choices=[1, 2, 3], default=None, help="Curriculum stage (auto-detected if omitted)"
    )
    eval_parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    eval_parser.add_argument("--no-render", action="store_true", help="Disable rendering")

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

        train(
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
        )

    elif args.command == "curriculum":
        train_curriculum(
            n_envs=args.n_envs,
            seed=args.seed,
            eval_freq=args.eval_freq,
            save_freq=args.save_freq,
            log_dir=args.log_dir,
            use_subproc=args.subproc,
            verbose=args.verbose,
        )

    elif args.command == "eval":
        evaluate(
            model_path=args.model_path,
            n_episodes=args.episodes,
            render=not args.no_render,
            stage=args.stage,
        )


if __name__ == "__main__":
    main()
