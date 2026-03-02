"""Cross-species integration tests.

These tests verify that all three species environments work consistently
through the shared infrastructure: Gymnasium registration, config loading,
determinism, and observation validity.
"""

import numpy as np
import pytest

from environments.brachiosaurus.envs.brachio_env import BrachioEnv
from environments.shared.config import load_all_stages, load_stage_config
from environments.trex.envs.trex_env import TRexEnv
from environments.velociraptor.envs.raptor_env import RaptorEnv

# Parametrise across all three species
SPECIES_ENVS = [
    pytest.param(RaptorEnv, "velociraptor", id="velociraptor"),
    pytest.param(TRexEnv, "trex", id="trex"),
    pytest.param(BrachioEnv, "brachiosaurus", id="brachiosaurus"),
]


# ── Gymnasium registration ───────────────────────────────────────────────


class TestGymnasiumRegistration:
    """Verify that gym.make works for all registered species."""

    @pytest.mark.parametrize(
        "gym_id",
        [
            "MesozoicLabs/Raptor-v0",
            "MesozoicLabs/TRex-v0",
            "MesozoicLabs/Brachio-v0",
        ],
    )
    def test_gym_make(self, gym_id):
        import gymnasium as gym

        env = gym.make(gym_id)
        obs, info = env.reset(seed=0)
        assert obs is not None
        assert obs.dtype == np.float32
        env.close()


# ── Config-to-env integration ────────────────────────────────────────────


class TestConfigEnvIntegration:
    """Verify TOML env_kwargs are valid constructor arguments."""

    @pytest.mark.parametrize("env_class, species", SPECIES_ENVS)
    def test_stage1_config_creates_env(self, env_class, species):
        config = load_stage_config(species, 1)
        env = env_class(**config["env_kwargs"])
        obs, _ = env.reset(seed=0)
        assert obs is not None
        env.close()

    @pytest.mark.parametrize("env_class, species", SPECIES_ENVS)
    def test_all_stages_create_env(self, env_class, species):
        configs = load_all_stages(species)
        for stage, config in configs.items():
            env = env_class(**config["env_kwargs"])
            obs, _ = env.reset(seed=0)
            assert obs is not None, f"Stage {stage} failed"
            env.close()


# ── Determinism ──────────────────────────────────────────────────────────


class TestDeterminism:
    @pytest.mark.parametrize("env_class, species", SPECIES_ENVS)
    def test_deterministic_trajectories(self, env_class, species):
        """Same seed must produce identical trajectories."""

        def _run(seed):
            env = env_class()
            obs, _ = env.reset(seed=seed)
            rng = np.random.RandomState(seed)
            trajectory = [obs.copy()]
            for _ in range(30):
                action = rng.randn(env.action_space.shape[0]).astype(np.float32)
                action = np.clip(action, -1, 1)
                obs, _, terminated, truncated, _ = env.step(action)
                trajectory.append(obs.copy())
                if terminated or truncated:
                    break
            env.close()
            return np.array(trajectory)

        t1 = _run(seed=99)
        t2 = _run(seed=99)
        np.testing.assert_array_equal(t1, t2)


# ── Observation validity ─────────────────────────────────────────────────


class TestObservationValidity:
    @pytest.mark.parametrize("env_class, species", SPECIES_ENVS)
    def test_no_nan_inf_in_rollout(self, env_class, species):
        """200-step rollout should produce no NaN or Inf in observations."""
        env = env_class()
        obs, _ = env.reset(seed=42)
        for _ in range(200):
            assert not np.any(np.isnan(obs)), "NaN in observation"
            assert not np.any(np.isinf(obs)), "Inf in observation"
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
        env.close()

    @pytest.mark.parametrize("env_class, species", SPECIES_ENVS)
    def test_obs_dtype_is_float32(self, env_class, species):
        env = env_class()
        obs, _ = env.reset(seed=0)
        assert obs.dtype == np.float32
        env.close()

    @pytest.mark.parametrize("env_class, species", SPECIES_ENVS)
    def test_action_space_bounds(self, env_class, species):
        env = env_class()
        assert np.all(env.action_space.low == -1.0)
        assert np.all(env.action_space.high == 1.0)
        env.close()


# ── Reward consistency ───────────────────────────────────────────────────


class TestRewardConsistency:
    """Verify common reward invariants hold for all species."""

    @pytest.mark.parametrize("env_class, species", SPECIES_ENVS)
    def test_reward_total_matches_step_reward(self, env_class, species):
        """The scalar reward returned by step() must equal info['reward_total']."""
        env = env_class()
        env.reset(seed=42)
        for _ in range(10):
            action = env.action_space.sample()
            _, reward, terminated, truncated, info = env.step(action)
            assert abs(reward - info["reward_total"]) < 1e-6, (
                f"{species}: step reward {reward} != info total {info['reward_total']}"
            )
            if terminated or truncated:
                env.reset()
        env.close()

    @pytest.mark.parametrize("env_class, species", SPECIES_ENVS)
    def test_alive_bonus_positive(self, env_class, species):
        env = env_class()
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert info["reward_alive"] > 0
        env.close()

    @pytest.mark.parametrize("env_class, species", SPECIES_ENVS)
    def test_energy_penalty_zero_for_zero_action(self, env_class, species):
        env = env_class()
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert abs(info["reward_energy"]) < 1e-8
        env.close()

    @pytest.mark.parametrize("env_class, species", SPECIES_ENVS)
    def test_energy_penalty_negative_for_full_action(self, env_class, species):
        env = env_class()
        env.reset(seed=42)
        action = np.ones(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert info["reward_energy"] < 0
        env.close()


# ── Termination ──────────────────────────────────────────────────────────


class TestTermination:
    """Verify all species produce termination_reason in info."""

    @pytest.mark.parametrize("env_class, species", SPECIES_ENVS)
    def test_termination_reason_present_on_terminal(self, env_class, species):
        """Run until termination and check termination_reason is set."""
        env = env_class(max_episode_steps=500)
        env.reset(seed=42)
        for _ in range(500):
            action = env.action_space.sample()
            _, _, terminated, truncated, info = env.step(action)
            if terminated:
                assert "termination_reason" in info, f"{species}: terminated but no termination_reason in info"
                break
        env.close()
