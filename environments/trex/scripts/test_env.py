#!/usr/bin/env python3
"""
Test the T-Rex Gymnasium environment.

Usage:
    python test_env.py
    python test_env.py --render
    python test_env.py --episodes 5 --steps 200
"""

import argparse
import numpy as np
import sys
from pathlib import Path

# Add repo root to path
_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from environments.trex.envs.trex_env import TRexEnv


def test_basic_functionality(render: bool = False):
    """Test basic env operations."""
    print("=" * 60)
    print("Testing basic environment functionality")
    print("=" * 60)

    render_mode = "human" if render else None
    env = TRexEnv(render_mode=render_mode)

    print(f"\nObservation space: {env.observation_space}")
    print(f"  Shape: {env.observation_space.shape}")
    print(f"  Dtype: {env.observation_space.dtype}")

    print(f"\nAction space: {env.action_space}")
    print(f"  Shape: {env.action_space.shape}")
    print(f"  Low: {env.action_space.low}")
    print(f"  High: {env.action_space.high}")

    print("\n--- Testing reset ---")
    obs, info = env.reset(seed=42)
    print(f"Initial obs shape: {obs.shape}")
    print(f"Initial obs range: [{obs.min():.3f}, {obs.max():.3f}]")

    print("\n--- Testing step (zero action) ---")
    action = np.zeros(env.action_space.shape)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Reward: {reward:.4f}")
    print(f"Terminated: {terminated}, Truncated: {truncated}")

    print("\n--- Testing step (random action) ---")
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Reward: {reward:.4f}")

    env.close()
    print("\nBasic functionality test passed!")
    return True


def test_episode_rollout(num_episodes: int = 3, max_steps: int = 100,
                         render: bool = False):
    """Test running full episodes."""
    print("\n" + "=" * 60)
    print(f"Testing episode rollouts ({num_episodes} episodes, "
          f"{max_steps} steps max)")
    print("=" * 60)

    render_mode = "human" if render else None
    env = TRexEnv(render_mode=render_mode, max_episode_steps=max_steps)

    episode_rewards = []
    episode_lengths = []
    termination_reasons = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=ep)
        total_reward = 0
        step = 0

        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            if terminated or truncated:
                reason = info.get("termination_reason", "truncated")
                termination_reasons.append(reason)
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(step)
        print(f"  Episode {ep + 1}: reward={total_reward:.2f}, "
              f"length={step}, ended={termination_reasons[-1]}")

    env.close()

    print("\nSummary:")
    print(f"  Avg reward: {np.mean(episode_rewards):.2f} "
          f"+/- {np.std(episode_rewards):.2f}")
    print(f"  Avg length: {np.mean(episode_lengths):.1f} "
          f"+/- {np.std(episode_lengths):.1f}")
    reasons, counts = np.unique(termination_reasons, return_counts=True)
    print(f"  Termination reasons: {dict(zip(reasons, counts))}")
    print("\nEpisode rollout test passed!")
    return True


def test_determinism():
    """Test that environment is deterministic given same seed."""
    print("\n" + "=" * 60)
    print("Testing determinism (same seed = same trajectory)")
    print("=" * 60)

    def run_episode(seed):
        env = TRexEnv()
        obs, _ = env.reset(seed=seed)
        np.random.seed(seed)
        trajectory = [obs.copy()]
        for _ in range(50):
            action = np.random.randn(env.action_space.shape[0]).astype(
                np.float32
            )
            action = np.clip(action, -1, 1)
            obs, _, terminated, truncated, _ = env.step(action)
            trajectory.append(obs.copy())
            if terminated or truncated:
                break
        env.close()
        return np.array(trajectory)

    traj1 = run_episode(seed=123)
    traj2 = run_episode(seed=123)

    if np.allclose(traj1, traj2):
        print("Environment is deterministic!")
        return True
    else:
        max_diff = np.abs(traj1 - traj2).max()
        print(f"Trajectories differ! Max difference: {max_diff}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test T-Rex Gymnasium environment"
    )
    parser.add_argument("--render", action="store_true",
                        help="Enable rendering")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes for rollout test")
    parser.add_argument("--steps", type=int, default=100,
                        help="Max steps per episode")
    args = parser.parse_args()

    all_passed = True
    all_passed &= test_basic_functionality(render=args.render)
    all_passed &= test_episode_rollout(
        args.episodes, args.steps, render=args.render
    )
    all_passed &= test_determinism()

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
