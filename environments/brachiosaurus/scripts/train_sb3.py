#!/usr/bin/env python3
"""
Train Brachiosaurus with Stable-Baselines3 PPO.

Supports curriculum learning with three stages:
1. Standing/balance (quadrupedal stability)
2. Walking (coordinated quadrupedal gait)
3. Walking + food reaching (full reward with neck control)

Usage:
    # Single-stage training
    python train_sb3.py train --stage 1 --timesteps 1000000
    python train_sb3.py train --stage 2 --timesteps 2000000 --load models/best_model.zip
    python train_sb3.py train --stage 3 --timesteps 3000000 --load models/best_model.zip

    # Automated end-to-end curriculum (all 3 stages)
    python train_sb3.py curriculum --n-envs 4

    # Evaluate a trained model
    python train_sb3.py eval models/stage3_final.zip --episodes 10
"""

import sys
from pathlib import Path

# Add repo root to path
_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from environments.brachiosaurus.envs.brachio_env import BrachioEnv
from environments.shared.train_base import SpeciesConfig, main

SPECIES_CONFIG = SpeciesConfig(
    species="brachiosaurus",
    env_class=BrachioEnv,
    stage_descriptions="1=balance, 2=locomotion, 3=food_reach",
    height_label="Torso height",
    stage3_section_label="Food Reaching",
    success_keys=["food_reached"],
)

if __name__ == "__main__":
    main(SPECIES_CONFIG)
