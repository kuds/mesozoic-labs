#!/usr/bin/env python3
"""
Test the T-Rex Gymnasium environment.

Usage:
    python test_env.py
    python test_env.py --render
    python test_env.py --episodes 5 --steps 200
"""

import argparse
import sys
from pathlib import Path

# Add repo root to path
_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from environments.shared.test_env_base import (
    test_basic_functionality,
    test_determinism,
    test_episode_rollout,
    test_observation_bounds,
    test_reward_components,
)
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


def main():
    parser = argparse.ArgumentParser(description="Test T-Rex Gymnasium environment")
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes for rollout test")
    parser.add_argument("--steps", type=int, default=100, help="Max steps per episode")
    args = parser.parse_args()

    all_passed = True

    all_passed &= test_basic_functionality(TRexEnv, render=args.render)
    all_passed &= test_episode_rollout(TRexEnv, args.episodes, args.steps, render=args.render)
    all_passed &= test_reward_components(TRexEnv, REWARD_KEYS, render=args.render)
    all_passed &= test_determinism(TRexEnv)
    all_passed &= test_observation_bounds(TRexEnv)

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
