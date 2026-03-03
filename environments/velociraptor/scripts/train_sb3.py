#!/usr/bin/env python3
"""
Train velociraptor with Stable-Baselines3 PPO.

Supports curriculum learning with three stages:
1. Standing/balance (no forward velocity reward)
2. Walking (moderate speed target)
3. Sprinting + strike (full reward)

Usage:
    # Single-stage training
    python train_sb3.py train --stage 1 --timesteps 1000000
    python train_sb3.py train --stage 2 --timesteps 2000000 --load models/best_model.zip
    python train_sb3.py train --stage 3 --timesteps 3000000 --load models/best_model.zip

    # Automated end-to-end curriculum (all 3 stages)
    python train_sb3.py curriculum --n-envs 4

    # Quick test run
    python train_sb3.py train --stage 1 --timesteps 10000 --eval-freq 2000
"""

import sys
from pathlib import Path

# Add repo root to path
_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from environments.shared.train_base import SpeciesConfig, main
from environments.velociraptor.envs.raptor_env import RaptorEnv

SPECIES_CONFIG = SpeciesConfig(
    species="velociraptor",
    env_class=RaptorEnv,
    stage_descriptions="1=balance, 2=locomotion, 3=strike",
    height_label="Pelvis height",
    stage3_section_label="Hunting",
    success_keys=["strike_success", "bite_success"],
)

if __name__ == "__main__":
    main(SPECIES_CONFIG)
