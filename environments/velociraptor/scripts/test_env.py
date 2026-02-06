#!/usr/bin/env python3
"""
Test the Raptor Gymnasium environment.

This script verifies:
1. Environment loads without errors
2. Observation and action spaces are valid
3. Step/reset work correctly
4. Reward components are reasonable
5. Termination conditions trigger appropriately

Usage:
    python test_env.py
    python test_env.py --render   # With visualization
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

from environments.velociraptor.envs.raptor_env import RaptorEnv


def test_basic_functionality(render: bool = False):
    """Test basic env operations."""
    print("=" * 60)
    print("Testing basic environment functionality")
    print("=" * 60)

    render_mode = "human" if render else None
    env = RaptorEnv(render_mode=render_mode)

    # Check spaces
    print(f"\nObservation space: {env.observation_space}")
    print(f"  Shape: {env.observation_space.shape}")
    print(f"  Dtype: {env.observation_space.dtype}")

    print(f"\nAction space: {env.action_space}")
    print(f"  Shape: {env.action_space.shape}")
    print(f"  Low: {env.action_space.low}")
    print(f"  High: {env.action_space.high}")

    # Test reset
    print("\n--- Testing reset ---")
    obs, info = env.reset(seed=42)
    print(f"Initial obs shape: {obs.shape}")
    print(f"Initial obs range: [{obs.min():.3f}, {obs.max():.3f}]")
    print(f"Initial info: {info}")

    # Test step with zero action
    print("\n--- Testing step (zero action) ---")
    action = np.zeros(env.action_space.shape)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Reward: {reward:.4f}")
    print(f"Terminated: {terminated}, Truncated: {truncated}")
    print(f"Info: {info}")

    # Test step with random action
    print("\n--- Testing step (random action) ---")
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Random action: {action}")
    print(f"Reward: {reward:.4f}")
    print(f"Info: {info}")

    env.close()
    print("\n✓ Basic functionality test passed!")
    return True


def test_episode_rollout(num_episodes: int = 3, max_steps: int = 100, render: bool = False):
    """Test running full episodes."""
    print("\n" + "=" * 60)
    print(f"Testing episode rollouts ({num_episodes} episodes, {max_steps} steps max)")
    print("=" * 60)

    render_mode = "human" if render else None
    env = RaptorEnv(render_mode=render_mode, max_episode_steps=max_steps)

    episode_rewards = []
    episode_lengths = []
    termination_reasons = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=ep)
        total_reward = 0
        step = 0

        while True:
            # Random policy
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
        print(f"  Episode {ep+1}: reward={total_reward:.2f}, length={step}, ended={termination_reasons[-1]}")

    env.close()

    print("\nSummary:")
    print(f"  Avg reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Avg length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"  Termination reasons: {dict(zip(*np.unique(termination_reasons, return_counts=True)))}")
    print("\n✓ Episode rollout test passed!")
    return True


def test_reward_components(render: bool = False):
    """Analyze reward component distributions."""
    print("\n" + "=" * 60)
    print("Analyzing reward components over 500 random steps")
    print("=" * 60)

    render_mode = "human" if render else None
    env = RaptorEnv(render_mode=render_mode)
    obs, _ = env.reset(seed=42)

    # Collect reward components
    components = {
        "reward_forward": [],
        "reward_alive": [],
        "reward_energy": [],
        "reward_tail": [],
        "reward_strike": [],
        "reward_total": [],
    }

    for _ in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        for key in components:
            if key in info:
                components[key].append(info[key])

        if terminated or truncated:
            obs, _ = env.reset()

    env.close()

    print("\nReward component statistics:")
    print("-" * 50)
    for key, values in components.items():
        values = np.array(values)
        print(f"  {key:20s}: mean={values.mean():8.4f}, std={values.std():8.4f}, "
              f"min={values.min():8.4f}, max={values.max():8.4f}")

    print("\n✓ Reward component analysis complete!")
    return True


def test_determinism():
    """Test that environment is deterministic given same seed."""
    print("\n" + "=" * 60)
    print("Testing determinism (same seed = same trajectory)")
    print("=" * 60)

    def run_episode(seed):
        env = RaptorEnv()
        obs, _ = env.reset(seed=seed)

        # Use deterministic "policy" based on observation
        np.random.seed(seed)

        trajectory = [obs.copy()]
        for _ in range(50):
            action = np.random.randn(env.action_space.shape[0]).astype(np.float32)
            action = np.clip(action, -1, 1)
            obs, _, terminated, truncated, _ = env.step(action)
            trajectory.append(obs.copy())
            if terminated or truncated:
                break

        env.close()
        return np.array(trajectory)

    # Run twice with same seed
    traj1 = run_episode(seed=123)
    traj2 = run_episode(seed=123)

    # Check if identical
    if np.allclose(traj1, traj2):
        print("✓ Environment is deterministic!")
        return True
    else:
        max_diff = np.abs(traj1 - traj2).max()
        print(f"✗ Trajectories differ! Max difference: {max_diff}")
        return False


def test_observation_bounds():
    """Check that observations stay within reasonable bounds."""
    print("\n" + "=" * 60)
    print("Testing observation bounds over 1000 steps")
    print("=" * 60)

    env = RaptorEnv()
    obs, _ = env.reset(seed=42)

    all_obs = [obs]

    for _ in range(1000):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        all_obs.append(obs)

        if terminated or truncated:
            obs, _ = env.reset()

    env.close()

    all_obs = np.array(all_obs)

    print("\nObservation statistics:")
    print(f"  Shape: {all_obs.shape}")
    print(f"  Min: {all_obs.min():.4f}")
    print(f"  Max: {all_obs.max():.4f}")
    print(f"  Mean: {all_obs.mean():.4f}")
    print(f"  Std: {all_obs.std():.4f}")

    # Check for NaN/Inf
    if np.any(np.isnan(all_obs)):
        print("✗ WARNING: NaN values detected in observations!")
        return False
    if np.any(np.isinf(all_obs)):
        print("✗ WARNING: Inf values detected in observations!")
        return False

    # Check for reasonable bounds (arbitrary but sensible)
    if all_obs.max() > 1000 or all_obs.min() < -1000:
        print("⚠ WARNING: Observations have very large values, consider normalization")

    print("\n✓ Observation bounds test passed!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test Raptor Gymnasium environment")
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes for rollout test")
    parser.add_argument("--steps", type=int, default=100, help="Max steps per episode")
    args = parser.parse_args()

    all_passed = True

    all_passed &= test_basic_functionality(render=args.render)
    all_passed &= test_episode_rollout(args.episodes, args.steps, render=args.render)
    all_passed &= test_reward_components(render=args.render)
    all_passed &= test_determinism()
    all_passed &= test_observation_bounds()

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
