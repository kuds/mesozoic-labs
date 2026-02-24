#!/usr/bin/env python3
"""
Train T-Rex with Stable-Baselines3 (PPO or SAC).

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

    # Use SAC instead of PPO
    python train_sb3.py train --stage 1 --algo SAC --timesteps 500000

    # Enable Weights & Biases logging
    python train_sb3.py curriculum --wandb

    # Evaluate a trained model
    python train_sb3.py eval models/stage3_final.zip --episodes 10
"""

import argparse
import logging
import sys
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

from environments.shared.config import load_all_stages
from environments.shared.train_utils import (
    add_common_args,
    evaluate_cli,
    train_curriculum,
    train_stage,
)
from environments.trex.envs.trex_env import TRexEnv

# ---------------------------------------------------------------------------
# Species-specific constants
# ---------------------------------------------------------------------------

SPECIES = "trex"
ENV_CLASS = TRexEnv
STAGE_CONFIGS = load_all_stages(SPECIES)
CLIP_REWARD = 10.0

REWARD_KEYS = [
    "reward_forward",
    "reward_alive",
    "reward_energy",
    "reward_tail",
    "reward_bite",
    "reward_approach",
    "reward_total",
]
INFO_KEYS = [
    "forward_vel",
    "tilt_angle",
    "pelvis_height",
    "tail_instability",
    "bite_success",
    "prey_distance",
    "approach_delta",
]

_FALLBACK_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train T-Rex with SB3")
    add_common_args(parser)
    args = parser.parse_args()

    if args.command == "train" or args.command is None:
        if args.command is None:
            args.stage = 1
            args.timesteps = 500000
            args.algo = "PPO"
            args.n_envs = 4
            args.load = None
            args.seed = 42
            args.eval_freq = 10000
            args.save_freq = 50000
            args.log_dir = None
            args.subproc = False
            args.wandb = False
            args.verbose = 1

        train_stage(
            env_class=ENV_CLASS,
            species=SPECIES,
            stage_configs=STAGE_CONFIGS,
            stage=args.stage,
            total_timesteps=args.timesteps,
            algorithm=args.algo,
            n_envs=args.n_envs,
            seed=args.seed,
            load_path=args.load,
            eval_freq=args.eval_freq,
            save_freq=args.save_freq,
            log_dir=args.log_dir,
            use_subproc=args.subproc,
            use_wandb=args.wandb,
            clip_reward=CLIP_REWARD,
            reward_keys=REWARD_KEYS,
            info_keys=INFO_KEYS,
            verbose=args.verbose,
            fallback_root=_FALLBACK_ROOT,
        )

    elif args.command == "curriculum":
        train_curriculum(
            env_class=ENV_CLASS,
            species=SPECIES,
            stage_configs=STAGE_CONFIGS,
            algorithm=args.algo,
            n_envs=args.n_envs,
            seed=args.seed,
            eval_freq=args.eval_freq,
            save_freq=args.save_freq,
            log_dir=args.log_dir,
            use_subproc=args.subproc,
            use_wandb=args.wandb,
            clip_reward=CLIP_REWARD,
            reward_keys=REWARD_KEYS,
            info_keys=INFO_KEYS,
            verbose=args.verbose,
            record_video=not args.no_video,
            fallback_root=_FALLBACK_ROOT,
        )

    elif args.command == "eval":
        evaluate_cli(
            env_class=ENV_CLASS,
            stage_configs=STAGE_CONFIGS,
            model_path=args.model_path,
            algorithm=args.algo,
            stage=args.stage,
            n_episodes=args.episodes,
            render=not args.no_render,
        )


if __name__ == "__main__":
    main()
