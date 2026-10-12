#!/usr/bin/env python3
"""
Test the Dibothrosuchus Gymnasium environment.

Usage:
    python test_env.py --embodiment anatomical
    python test_env.py --embodiment anatomical --render
    python test_env.py --embodiment anatomical --episodes 5 --steps 200
"""

import sys
from pathlib import Path

# Add repo root to path
_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from environments.shared.harnesses.env_smoke import run_env_tests

SPECIES = "dibothrosuchus_elaphros"

REWARD_KEYS = [
    "reward_forward",
    "reward_alive",
    "reward_energy",
    "reward_gait",
    "reward_snap",
    "reward_approach",
    "reward_idle",
    "reward_total",
]

if __name__ == "__main__":
    sys.exit(run_env_tests(SPECIES, REWARD_KEYS, "Dibothrosuchus"))
