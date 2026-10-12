#!/usr/bin/env python3
"""
Test the T-Rex Gymnasium environment.

Usage:
    python test_env.py
    python test_env.py --render
    python test_env.py --episodes 5 --steps 200
"""

import sys
from pathlib import Path

# Add repo root to path
_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from environments.shared.harnesses.env_smoke import run_env_tests
from environments.trex.envs.trex_env import TRexEnv

REWARD_KEYS = [
    "reward_forward",
    "reward_alive",
    "reward_energy",
    "reward_tail",
    "reward_bite",
    "reward_idle",
    "reward_total",
]

if __name__ == "__main__":
    sys.exit(run_env_tests(TRexEnv, REWARD_KEYS, "T-Rex"))
