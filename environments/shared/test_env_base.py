"""
Shared environment smoke-test functions.

Each species' ``scripts/test_env.py`` calls these functions with its
own environment class, eliminating ~500 lines of duplicated test logic.
"""

import gymnasium as gym
import numpy as np


def test_basic_functionality(env_class, render: bool = False):
    """Test basic env operations (spaces, reset, step)."""
    print("=" * 60)
    print("Testing basic environment functionality")
    print("=" * 60)

    render_mode = "human" if render else None
    env = env_class(render_mode=render_mode)

    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Box)

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
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Reward: {reward:.4f}")
    print(f"Terminated: {terminated}, Truncated: {truncated}")
    print(f"Info: {info}")

    # Test step with random action
    print("\n--- Testing step (random action) ---")
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Reward: {reward:.4f}")

    env.close()
    print("\nBasic functionality test passed!")
    return True


def test_episode_rollout(env_class, num_episodes: int = 3, max_steps: int = 100, render: bool = False):
    """Test running full episodes."""
    print("\n" + "=" * 60)
    print(f"Testing episode rollouts ({num_episodes} episodes, {max_steps} steps max)")
    print("=" * 60)

    render_mode = "human" if render else None
    env = env_class(render_mode=render_mode, max_episode_steps=max_steps)

    episode_rewards = []
    episode_lengths = []
    termination_reasons = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=ep)
        total_reward: float = 0.0
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
        print(f"  Episode {ep + 1}: reward={total_reward:.2f}, length={step}, ended={termination_reasons[-1]}")

    env.close()

    print("\nSummary:")
    print(f"  Avg reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"  Avg length: {np.mean(episode_lengths):.1f} +/- {np.std(episode_lengths):.1f}")
    reasons, counts = np.unique(termination_reasons, return_counts=True)
    print(f"  Termination reasons: {dict(zip(reasons, counts))}")
    print("\nEpisode rollout test passed!")
    return True


def test_reward_components(env_class, reward_keys: list[str], render: bool = False):
    """Analyze reward component distributions over 500 random steps."""
    print("\n" + "=" * 60)
    print("Analyzing reward components over 500 random steps")
    print("=" * 60)

    render_mode = "human" if render else None
    env = env_class(render_mode=render_mode)
    obs, _ = env.reset(seed=42)

    components: dict[str, list[float]] = {key: [] for key in reward_keys}

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
    for key, vals in components.items():
        arr = np.array(vals)
        print(f"  {key:20s}: mean={arr.mean():8.4f}, std={arr.std():8.4f}, min={arr.min():8.4f}, max={arr.max():8.4f}")

    print("\nReward component analysis complete!")
    return True


def test_determinism(env_class):
    """Test that the environment is deterministic given the same seed."""
    print("\n" + "=" * 60)
    print("Testing determinism (same seed = same trajectory)")
    print("=" * 60)

    def run_episode(seed):
        env = env_class()
        obs, _ = env.reset(seed=seed)
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

    traj1 = run_episode(seed=123)
    traj2 = run_episode(seed=123)

    if np.allclose(traj1, traj2):
        print("Environment is deterministic!")
        return True
    else:
        max_diff = np.abs(traj1 - traj2).max()
        print(f"Trajectories differ! Max difference: {max_diff}")
        return False


def test_observation_bounds(env_class):
    """Check that observations stay within reasonable bounds."""
    print("\n" + "=" * 60)
    print("Testing observation bounds over 1000 steps")
    print("=" * 60)

    env = env_class()
    obs, _ = env.reset(seed=42)

    all_obs = [obs]

    for _ in range(1000):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        all_obs.append(obs)

        if terminated or truncated:
            obs, _ = env.reset()

    env.close()

    obs_array = np.array(all_obs)

    print("\nObservation statistics:")
    print(f"  Shape: {obs_array.shape}")
    print(f"  Min: {obs_array.min():.4f}")
    print(f"  Max: {obs_array.max():.4f}")
    print(f"  Mean: {obs_array.mean():.4f}")
    print(f"  Std: {obs_array.std():.4f}")

    if np.any(np.isnan(obs_array)):
        print("WARNING: NaN values detected in observations!")
        return False
    if np.any(np.isinf(obs_array)):
        print("WARNING: Inf values detected in observations!")
        return False

    if obs_array.max() > 1000 or obs_array.min() < -1000:
        print("WARNING: Observations have very large values, consider normalization")

    print("\nObservation bounds test passed!")
    return True
