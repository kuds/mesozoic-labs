#!/usr/bin/env python3
"""
Train velociraptor with Stable-Baselines3 PPO.

Supports curriculum learning with three stages:
1. Standing/balance (no forward velocity reward)
2. Walking (moderate speed target)
3. Sprinting + strike (full reward)

Usage:
    python train_sb3.py --stage 1 --timesteps 500000
    python train_sb3.py --stage 2 --timesteps 1000000 --load models/stage1_final.zip
    python train_sb3.py --stage 3 --timesteps 2000000 --load models/stage2_final.zip
    
    # Quick test run
    python train_sb3.py --stage 1 --timesteps 10000 --eval-freq 2000
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import (
        EvalCallback, 
        CheckpointCallback, 
        CallbackList
    )
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.utils import set_random_seed
except ImportError:
    print("ERROR: stable-baselines3 not installed.")
    print("Install with: pip install stable-baselines3[extra]")
    sys.exit(1)

from envs.raptor_env import RaptorEnv


# Curriculum stage configurations
STAGE_CONFIGS = {
    1: {
        "name": "balance",
        "description": "Learn to stand and balance without falling",
        "env_kwargs": {
            "forward_vel_weight": 0.0,      # No forward reward
            "alive_bonus": 1.0,              # Strong alive bonus
            "energy_penalty_weight": 0.0005, # Light energy penalty
            "tail_stability_weight": 0.1,    # Encourage stable tail
            "strike_bonus": 0.0,             # No strike reward yet
            "prey_distance_range": (10.0, 15.0),  # Prey far away (irrelevant)
            "max_episode_steps": 500,
        },
        "ppo_kwargs": {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,  # Higher entropy for exploration
        },
    },
    2: {
        "name": "locomotion",
        "description": "Learn forward walking/running",
        "env_kwargs": {
            "forward_vel_weight": 1.0,       # Forward velocity reward
            "alive_bonus": 0.5,              # Moderate alive bonus
            "energy_penalty_weight": 0.001,
            "tail_stability_weight": 0.05,
            "strike_bonus": 0.0,             # No strike yet
            "prey_distance_range": (8.0, 12.0),
            "max_episode_steps": 1000,
        },
        "ppo_kwargs": {
            "learning_rate": 1e-4,
            "n_steps": 2048,
            "batch_size": 128,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.005,
        },
    },
    3: {
        "name": "strike",
        "description": "Sprint and strike prey with sickle claw",
        "env_kwargs": {
            "forward_vel_weight": 1.0,
            "alive_bonus": 0.1,
            "energy_penalty_weight": 0.001,
            "tail_stability_weight": 0.02,
            "strike_bonus": 500.0,           # Big strike reward!
            "prey_distance_range": (3.0, 8.0),  # Prey closer
            "prey_lateral_range": (-1.5, 1.5),  # Some lateral variation
            "max_episode_steps": 1000,
        },
        "ppo_kwargs": {
            "learning_rate": 5e-5,
            "n_steps": 4096,
            "batch_size": 256,
            "n_epochs": 10,
            "gamma": 0.995,  # Higher gamma for long-horizon strike
            "gae_lambda": 0.95,
            "clip_range": 0.1,  # Tighter clipping for fine-tuning
            "ent_coef": 0.001,
        },
    },
}


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
    load_path: str = None,
    eval_freq: int = 10000,
    save_freq: int = 50000,
    log_dir: str = None,
    use_subproc: bool = False,
):
    """Train the raptor policy."""
    
    config = STAGE_CONFIGS[stage]
    print("=" * 60)
    print(f"Training Stage {stage}: {config['name']}")
    print(f"Description: {config['description']}")
    print("=" * 60)
    
    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if log_dir is None:
        log_dir = Path(__file__).parent.parent / "logs" / f"stage{stage}_{timestamp}"
    else:
        log_dir = Path(log_dir)
    
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir = log_dir / "models"
    model_dir.mkdir(exist_ok=True)
    
    print(f"\nLog directory: {log_dir}")
    print(f"Model directory: {model_dir}")
    
    # Create environments
    print(f"\nCreating {n_envs} training environments...")
    train_env = create_vec_env(stage, n_envs, seed, use_subproc)
    
    print("Creating evaluation environment...")
    eval_env = create_vec_env(stage, 1, seed + 1000, use_subproc=False)
    
    # Create or load model
    ppo_kwargs = config["ppo_kwargs"].copy()
    ppo_kwargs["verbose"] = 1
    ppo_kwargs["tensorboard_log"] = str(log_dir / "tensorboard")
    
    if load_path:
        print(f"\nLoading model from: {load_path}")
        model = PPO.load(load_path, env=train_env)
        # Update hyperparameters for new stage
        model.learning_rate = ppo_kwargs["learning_rate"]
        model.ent_coef = ppo_kwargs["ent_coef"]
        model.clip_range = ppo_kwargs["clip_range"]
    else:
        print("\nCreating new PPO model...")
        model = PPO(
            "MlpPolicy",
            train_env,
            **ppo_kwargs,
        )
    
    # Print model info
    print(f"\nModel architecture:")
    print(f"  Policy: {model.policy}")
    print(f"  Learning rate: {model.learning_rate}")
    print(f"  Batch size: {ppo_kwargs['batch_size']}")
    
    # Setup callbacks
    callbacks = []
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(log_dir),
        eval_freq=eval_freq // n_envs,  # Adjusted for vec env
        n_eval_episodes=5,
        deterministic=True,
        render=False,
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
    print(f"\nStarting training for {total_timesteps:,} timesteps...")
    print("-" * 60)
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    
    # Save final model
    final_path = model_dir / f"stage{stage}_final"
    print(f"\nSaving final model to: {final_path}")
    model.save(str(final_path))
    train_env.save(str(final_path) + "_vecnorm.pkl")
    
    # Cleanup
    train_env.close()
    eval_env.close()
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Final model: {final_path}.zip")
    print(f"VecNormalize stats: {final_path}_vecnorm.pkl")
    print("=" * 60)
    
    return model


def evaluate(model_path: str, n_episodes: int = 10, render: bool = True):
    """Evaluate a trained model."""
    from stable_baselines3.common.evaluation import evaluate_policy
    
    print(f"Loading model from: {model_path}")
    
    # Determine stage from filename (hacky but works)
    stage = 1
    for s in [1, 2, 3]:
        if f"stage{s}" in model_path:
            stage = s
            break
    
    model = PPO.load(model_path)
    
    render_mode = "human" if render else None
    env_kwargs = STAGE_CONFIGS[stage]["env_kwargs"].copy()
    env = RaptorEnv(render_mode=render_mode, **env_kwargs)
    
    # Load VecNormalize stats if available
    vecnorm_path = model_path.replace(".zip", "_vecnorm.pkl")
    if Path(vecnorm_path).exists():
        print(f"Loading normalization stats from: {vecnorm_path}")
        # Note: For proper eval, would need to wrap in VecNormalize
        # This is simplified for visualization
    
    print(f"\nEvaluating for {n_episodes} episodes...")
    
    episode_rewards = []
    episode_lengths = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        step = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(step)
        print(f"  Episode {ep+1}: reward={total_reward:.2f}, length={step}")
    
    env.close()
    
    print(f"\nResults:")
    print(f"  Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Mean length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")


def main():
    parser = argparse.ArgumentParser(description="Train Velociraptor with SB3 PPO")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a policy")
    train_parser.add_argument("--stage", type=int, choices=[1, 2, 3], default=1,
                             help="Curriculum stage (1=balance, 2=locomotion, 3=strike)")
    train_parser.add_argument("--timesteps", type=int, default=500000,
                             help="Total training timesteps")
    train_parser.add_argument("--n-envs", type=int, default=4,
                             help="Number of parallel environments")
    train_parser.add_argument("--load", type=str, default=None,
                             help="Path to model to continue training from")
    train_parser.add_argument("--seed", type=int, default=42,
                             help="Random seed")
    train_parser.add_argument("--eval-freq", type=int, default=10000,
                             help="Evaluation frequency (timesteps)")
    train_parser.add_argument("--save-freq", type=int, default=50000,
                             help="Checkpoint save frequency (timesteps)")
    train_parser.add_argument("--log-dir", type=str, default=None,
                             help="Custom log directory")
    train_parser.add_argument("--subproc", action="store_true",
                             help="Use subprocess vectorization (faster but more memory)")
    
    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained policy")
    eval_parser.add_argument("model_path", type=str, help="Path to trained model")
    eval_parser.add_argument("--episodes", type=int, default=10,
                            help="Number of episodes to evaluate")
    eval_parser.add_argument("--no-render", action="store_true",
                            help="Disable rendering")
    
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
        )
    
    elif args.command == "eval":
        evaluate(
            model_path=args.model_path,
            n_episodes=args.episodes,
            render=not args.no_render,
        )


if __name__ == "__main__":
    main()
