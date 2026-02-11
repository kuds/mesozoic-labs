#!/usr/bin/env python3
"""
Train velociraptor with Stable-Baselines3 PPO.

Supports curriculum learning with three stages:
1. Standing/balance (no forward velocity reward)
2. Walking (moderate speed target)
3. Sprinting + strike (full reward)

Usage:
    # Single-stage training
    python train_sb3.py train --stage 1 --timesteps 500000
    python train_sb3.py train --stage 2 --timesteps 1000000 --load models/stage1_final.zip
    python train_sb3.py train --stage 3 --timesteps 2000000 --load models/stage2_final.zip

    # Automated end-to-end curriculum (all 3 stages)
    python train_sb3.py curriculum --n-envs 4

    # Quick test run
    python train_sb3.py train --stage 1 --timesteps 10000 --eval-freq 2000
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

import numpy as np

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

from environments.shared.config import load_all_stages
from environments.shared.curriculum import (
    CurriculumCallback,
    CurriculumManager,
    thresholds_from_configs,
)
from environments.velociraptor.envs.raptor_env import RaptorEnv

# Load curriculum configs from TOML files (configs/velociraptor/)
STAGE_CONFIGS = load_all_stages("velociraptor")


def make_env(stage: int, rank: int, seed: int = 0):
    """Create a single environment instance."""

    def _init():
        env_kwargs = STAGE_CONFIGS[stage]["env_kwargs"].copy()
        env = RaptorEnv(**env_kwargs)
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

    # Wrap with observation/reward normalization
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
    """Train the raptor policy."""

    config = STAGE_CONFIGS[stage]
    logger.info("=" * 60)
    logger.info("Training Stage %d: %s", stage, config["name"])
    logger.info("Description: %s", config["description"])
    logger.info("=" * 60)

    # Setup directories (organised as <species>/<datetime>/)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path: Path
    if log_dir is None:
        log_path = Path(__file__).parent.parent / "logs" / "velociraptor" / f"stage{stage}_{timestamp}"
    else:
        log_path = Path(log_dir)

    log_path.mkdir(parents=True, exist_ok=True)
    model_dir = log_path / "models"
    model_dir.mkdir(exist_ok=True)

    logger.info("Log directory: %s", log_path)
    logger.info("Model directory: %s", model_dir)

    # Create environments
    logger.info("Creating %d training environments...", n_envs)
    train_env = create_vec_env(stage, n_envs, seed, use_subproc)

    logger.info("Creating evaluation environment...")
    eval_env = create_vec_env(stage, 1, seed + 1000, use_subproc=False)

    # Create or load model
    ppo_kwargs = config["ppo_kwargs"].copy()
    ppo_kwargs["verbose"] = verbose
    ppo_kwargs["tensorboard_log"] = str(log_path / "tensorboard")

    if load_path:
        logger.info("Loading model from: %s", load_path)
        model = PPO.load(load_path, env=train_env)
        # Update hyperparameters for new stage
        model.learning_rate = ppo_kwargs["learning_rate"]
        model.ent_coef = ppo_kwargs["ent_coef"]
        model.clip_range = ppo_kwargs["clip_range"]
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

    # Setup callbacks
    callbacks = []

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(log_path),
        eval_freq=eval_freq // n_envs,  # Adjusted for vec env
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=max(verbose, 1),
    )
    callbacks.append(eval_callback)

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // n_envs,
        save_path=str(model_dir),
        name_prefix=f"stage{stage}",
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)

    callback_list = CallbackList(callbacks)

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

    # Save final model
    final_path = model_dir / f"stage{stage}_final"
    logger.info("Saving final model to: %s", final_path)
    model.save(str(final_path))
    train_env.save(str(final_path) + "_vecnorm.pkl")

    # Cleanup
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
        species="velociraptor",
        stage_thresholds=thresholds,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if log_dir is None:
        base_dir = Path(__file__).parent.parent / "logs" / "velociraptor" / f"curriculum_{timestamp}"
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

        # Create environments for this stage
        train_env = create_vec_env(stage, n_envs, seed, use_subproc)
        eval_env = create_vec_env(stage, 1, seed + 1000, use_subproc=False)

        # Create or load model
        ppo_kwargs = config["ppo_kwargs"].copy()
        ppo_kwargs["verbose"] = verbose
        ppo_kwargs["tensorboard_log"] = str(stage_dir / "tensorboard")

        if load_path:
            logger.info("Loading model from previous stage: %s", load_path)
            model = PPO.load(load_path, env=train_env)
            model.learning_rate = ppo_kwargs["learning_rate"]
            model.ent_coef = ppo_kwargs["ent_coef"]
            model.clip_range = ppo_kwargs["clip_range"]
        else:
            model = PPO("MlpPolicy", train_env, **ppo_kwargs)

        # Build callbacks
        callbacks = []

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(model_dir),
            log_path=str(stage_dir),
            eval_freq=eval_freq // n_envs,
            n_eval_episodes=5,
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

        # Add curriculum callback for non-final stages
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

        # Save stage checkpoint
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
            # Timestep budget exhausted without meeting threshold
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
    """Evaluate a trained model.

    Args:
        model_path: Path to the saved model (.zip file).
        n_episodes: Number of evaluation episodes.
        render: Whether to render in a window.
        stage: Curriculum stage (1-3). Auto-detected from filename if not provided.
    """
    logger.info("Loading model from: %s", model_path)

    # Determine stage from argument or filename
    if stage is None:
        stage = 1
        for s in [1, 2, 3]:
            if f"stage{s}" in model_path:
                stage = s
                break
        logger.info("Auto-detected stage %d from filename", stage)

    env_kwargs = STAGE_CONFIGS[stage]["env_kwargs"].copy()

    # Build a properly normalized vectorized env for evaluation
    vecnorm_path = model_path.replace(".zip", "_vecnorm.pkl")
    if not vecnorm_path.endswith("_vecnorm.pkl"):
        vecnorm_path = model_path + "_vecnorm.pkl"

    render_mode = "human" if render else None

    def _make_eval_env():
        env = RaptorEnv(render_mode=render_mode, **env_kwargs)
        return Monitor(env)

    vec_env = DummyVecEnv([_make_eval_env])

    if Path(vecnorm_path).exists():
        logger.info("Loading normalization stats from: %s", vecnorm_path)
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        logger.warning("No VecNormalize stats found. Results may differ from training.")

    model = PPO.load(model_path, env=vec_env)

    logger.info("Evaluating for %d episodes (stage %d: %s)...", n_episodes, stage, STAGE_CONFIGS[stage]["name"])

    episode_rewards = []
    episode_lengths = []

    for ep in range(n_episodes):
        obs = vec_env.reset()
        total_reward = 0
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


def main():
    parser = argparse.ArgumentParser(description="Train Velociraptor with SB3 PPO")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a policy")
    train_parser.add_argument(
        "--stage", type=int, choices=[1, 2, 3], default=1, help="Curriculum stage (1=balance, 2=locomotion, 3=strike)"
    )
    train_parser.add_argument("--timesteps", type=int, default=500000, help="Total training timesteps")
    train_parser.add_argument("--n-envs", type=int, default=4, help="Number of parallel environments")
    train_parser.add_argument("--load", type=str, default=None, help="Path to model to continue training from")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    train_parser.add_argument("--eval-freq", type=int, default=10000, help="Evaluation frequency (timesteps)")
    train_parser.add_argument("--save-freq", type=int, default=50000, help="Checkpoint save frequency (timesteps)")
    train_parser.add_argument("--log-dir", type=str, default=None, help="Custom log directory")
    train_parser.add_argument(
        "--subproc", action="store_true", help="Use subprocess vectorization (faster but more memory)"
    )
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
        "--stage",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Curriculum stage (auto-detected from filename if omitted)",
    )
    eval_parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to evaluate")
    eval_parser.add_argument("--no-render", action="store_true", help="Disable rendering")

    # Parse args
    args = parser.parse_args()

    if args.command == "train" or args.command is None:
        # Default to train if no command specified
        if args.command is None:
            # Use defaults
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
