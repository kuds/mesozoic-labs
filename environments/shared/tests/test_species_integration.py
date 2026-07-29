"""Cross-species integration tests.

These tests verify that every species environment works consistently
through the shared infrastructure: Gymnasium registration, config loading,
determinism, observation validity, basic functionality, and common reward
invariants.

Per-species test files should only contain tests for species-unique behavior
(e.g. tail termination for raptor, head termination for T-Rex, food-reach for
brachiosaurus, snout snap for dibothrosuchus).  All shared behavior is tested
here via parametrization.
"""

import importlib

import numpy as np
import pytest

from environments.brachiosaurus.envs.brachio_env import BrachioEnv
from environments.dibothrosuchus.envs.dibothrosuchus_env import DibothrosuchusEnv
from environments.shared.config import load_all_stages, load_stage_config
from environments.trex.envs.trex_env import TRexEnv
from environments.velociraptor.envs.raptor_env import RaptorEnv

# Parametrise across every implemented species
SPECIES_ENVS = [
    pytest.param(RaptorEnv, "velociraptor", id="velociraptor"),
    pytest.param(TRexEnv, "trex", id="trex"),
    pytest.param(BrachioEnv, "brachiosaurus", id="brachiosaurus"),
    pytest.param(DibothrosuchusEnv, "dibothrosuchus", id="dibothrosuchus"),
]

# Expected observation and action dimensions per species
SPECIES_DIMS = {
    "velociraptor": {"obs": 67, "act": 22},
    "trex": {"obs": 61, "act": 21},
    "brachiosaurus": {"obs": 83, "act": 30},
    "dibothrosuchus": {"obs": 77, "act": 27},
}

# Expected reward component keys per species
SPECIES_REWARD_KEYS = {
    "velociraptor": [
        "reward_forward",
        "reward_alive",
        "reward_energy",
        "reward_tail",
        "reward_strike",
        "reward_approach",
        "reward_posture",
        "reward_gait",
        "reward_smoothness",
        "reward_idle",
        "reward_total",
    ],
    "trex": [
        "reward_forward",
        "reward_alive",
        "reward_energy",
        "reward_tail",
        "reward_bite",
        "reward_approach",
        "reward_posture",
        "reward_nosedive",
        "reward_height",
        "reward_gait",
        "reward_smoothness",
        "reward_heading",
        "reward_lateral",
        "reward_idle",
        "reward_total",
    ],
    "brachiosaurus": [
        "reward_forward",
        "reward_alive",
        "reward_energy",
        "reward_gait",
        "reward_food",
        "reward_approach",
        "reward_idle",
        "reward_total",
    ],
    "dibothrosuchus": [
        "reward_forward",
        "reward_alive",
        "reward_energy",
        "reward_gait",
        "reward_gait_symmetry",
        "reward_tail",
        "reward_snap",
        "reward_approach",
        "reward_posture",
        "reward_nosedive",
        "reward_height",
        "reward_smoothness",
        "reward_heading",
        "reward_lateral",
        "reward_idle",
        "reward_total",
    ],
}


# ── Gymnasium registration ───────────────────────────────────────────────


class TestGymnasiumRegistration:
    """Verify that gym.make works for all registered species."""

    @pytest.mark.parametrize(
        "gym_id",
        [
            "MesozoicLabs/Raptor-v0",
            "MesozoicLabs/TRex-v0",
            "MesozoicLabs/Brachio-v0",
            "MesozoicLabs/Dibothrosuchus-v0",
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


# ── Basic functionality ─────────────────────────────────────────────────


class TestBasicFunctionality:
    """Common env lifecycle tests (previously duplicated in per-species files)."""

    @pytest.mark.parametrize("env_class, species", SPECIES_ENVS)
    def test_spaces_are_valid(self, env_class, species):
        env = env_class()
        dims = SPECIES_DIMS[species]
        assert env.observation_space.shape == (dims["obs"],)
        assert env.observation_space.dtype == np.float32
        assert env.action_space.shape == (dims["act"],)
        assert np.all(env.action_space.low == -1.0)
        assert np.all(env.action_space.high == 1.0)
        env.close()

    @pytest.mark.parametrize("env_class, species", SPECIES_ENVS)
    def test_reset_returns_valid_obs(self, env_class, species):
        env = env_class()
        obs, info = env.reset(seed=42)
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == np.float32
        assert not np.any(np.isnan(obs))
        assert not np.any(np.isinf(obs))
        env.close()

    @pytest.mark.parametrize("env_class, species", SPECIES_ENVS)
    def test_step_zero_action(self, env_class, species):
        env = env_class()
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        env.close()

    @pytest.mark.parametrize("env_class, species", SPECIES_ENVS)
    def test_step_random_action(self, env_class, species):
        env = env_class()
        env.reset(seed=42)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape
        assert not np.any(np.isnan(obs))
        env.close()

    @pytest.mark.parametrize("env_class, species", SPECIES_ENVS)
    def test_reward_components_in_info(self, env_class, species):
        env = env_class()
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        for key in SPECIES_REWARD_KEYS[species]:
            assert key in info, f"Missing reward component: {key}"
        env.close()

    @pytest.mark.parametrize("env_class, species", SPECIES_ENVS)
    def test_episode_runs_to_completion(self, env_class, species):
        env = env_class()
        env.reset(seed=0)
        total_steps = 0
        for _ in range(env.max_episode_steps):
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            total_steps += 1
            if terminated or truncated:
                break
        assert total_steps > 0
        env.close()

    @pytest.mark.parametrize("env_class, species", SPECIES_ENVS)
    def test_multiple_resets(self, env_class, species):
        env = env_class()
        for seed in range(3):
            obs, info = env.reset(seed=seed)
            assert obs.shape == env.observation_space.shape
            action = env.action_space.sample()
            obs2, _, _, _, _ = env.step(action)
            assert obs2.shape == env.observation_space.shape
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

    @pytest.mark.parametrize("env_class, species", SPECIES_ENVS)
    def test_approach_reward_zero_on_first_step(self, env_class, species):
        """Approach reward should be zero on the first step (no prior distance)."""
        env = env_class()
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert info["reward_approach"] == 0.0
        assert info["approach_delta"] == 0.0
        env.close()

    @pytest.mark.parametrize("env_class, species", SPECIES_ENVS)
    def test_zero_forward_weight_zeroes_forward_reward(self, env_class, species):
        env = env_class(forward_vel_weight=0.0)
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert info["reward_forward"] == 0.0
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


# ── Edge cases ───────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge case tests for robustness across all species."""

    @pytest.mark.parametrize("env_class, species", SPECIES_ENVS)
    def test_double_close(self, env_class, species):
        """Calling close() twice should not raise."""
        env = env_class()
        env.reset(seed=42)
        env.close()
        env.close()

    @pytest.mark.parametrize("env_class, species", SPECIES_ENVS)
    def test_double_reset(self, env_class, species):
        """Calling reset() twice without stepping should not raise."""
        env = env_class()
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)
        env.close()

    @pytest.mark.parametrize("env_class, species", SPECIES_ENVS)
    def test_clipped_actions_same_as_boundary(self, env_class, species):
        """Actions at exactly +/-1 should work without issues."""
        env = env_class()
        env.reset(seed=42)
        action_max = np.ones(env.action_space.shape, dtype=np.float32)
        obs, reward, _, _, _ = env.step(action_max)
        assert not np.any(np.isnan(obs))
        action_min = -np.ones(env.action_space.shape, dtype=np.float32)
        obs, reward, _, _, _ = env.step(action_min)
        assert not np.any(np.isnan(obs))
        env.close()

    @pytest.mark.parametrize("env_class, species", SPECIES_ENVS)
    def test_set_reward_weight(self, env_class, species):
        """set_reward_weight should update an existing attribute."""
        env = env_class()
        env.reset(seed=42)
        original = env.alive_bonus
        env.set_reward_weight("alive_bonus", 999.0)
        assert env.alive_bonus == 999.0
        env.set_reward_weight("alive_bonus", original)
        env.close()

    @pytest.mark.parametrize("env_class, species", SPECIES_ENVS)
    def test_set_reward_weight_invalid_attr_raises(self, env_class, species):
        """set_reward_weight should raise for non-existent attributes."""
        env = env_class()
        with pytest.raises(AttributeError):
            env.set_reward_weight("nonexistent_weight_xyz", 1.0)
        env.close()


MJX_CONFIG_MODULES = {
    "velociraptor": "environments.velociraptor.mjx_config",
    "trex": "environments.trex.mjx_config",
    "brachiosaurus": "environments.brachiosaurus.mjx_config",
    "dibothrosuchus": "environments.dibothrosuchus.mjx_config",
}

# Stage-config [env] keys the MJX path consumes and the Gymnasium envs do not
# accept.  Mirrors the JAX-only note in the stage-1 configs and the same
# constant in environments/shared/scripts/zero_action_baseline.py.
JAX_ONLY_ENV_KEYS = frozenset({"foot_contact_weight", "foot_contact_gate"})


class TestSB3MJXEnvelopeParity:
    """The Gymnasium env and the MJX registry must agree on the alive envelope.

    ``healthy_z_range`` and ``max_tilt_angle`` are absent from the fingerprinted
    plant interface, so nothing else catches a divergence.  That is how the
    T-Rex came to run with ``healthy_z_range`` (0.5, 1.6) on one path and
    (0.75, 1.6) on the other, and ``max_tilt_angle`` 1.047 against 0.7, while
    the other three species matched.

    A mismatch changes the task rather than just the cutoff: on the MJX path
    ``healthy_z_range`` also scales the alive bonus (``height_frac``), and
    ``max_tilt_angle`` normalises the posture penalty as
    ``(tilt / max_tilt_angle) ** 2``.
    """

    @pytest.mark.parametrize("env_class, species", SPECIES_ENVS)
    def test_healthy_z_range_matches(self, env_class, species):
        importlib.import_module(MJX_CONFIG_MODULES[species])
        from environments.shared.mjx_env import _SPECIES_CONFIGS

        registered = _SPECIES_CONFIGS[species]["healthy_z_range"]
        env = env_class()
        try:
            assert tuple(registered) == tuple(env.healthy_z_range), (
                f"{species}: MJX registers healthy_z_range={tuple(registered)} but the "
                f"Gymnasium env uses {tuple(env.healthy_z_range)}"
            )
        finally:
            env.close()

    @pytest.mark.parametrize("env_class, species", SPECIES_ENVS)
    def test_max_tilt_angle_matches(self, env_class, species):
        importlib.import_module(MJX_CONFIG_MODULES[species])
        from environments.shared.mjx_env import _SPECIES_CONFIGS

        registered = _SPECIES_CONFIGS[species]["max_tilt_angle"]
        env = env_class()
        try:
            assert registered == pytest.approx(env.max_tilt_angle), (
                f"{species}: MJX registers max_tilt_angle={registered} but the Gymnasium env uses {env.max_tilt_angle}"
            )
        finally:
            env.close()

    @pytest.mark.parametrize("stage", (1, 2, 3))
    def test_trex_nosedive_threshold_never_falls_back(self, stage):
        """Every T-Rex stage must resolve the same nosedive gate on both paths.

        ``nosedive_termination_threshold`` is the per-stage tunable pitch gate
        (``max_tilt_angle`` is the absolute backstop).  Only stage 1 sets it in
        TOML.  When a stage leaves it unset the two paths reach for different
        fallbacks: the Gymnasium env takes ``TRexEnv.__init__``'s default, while
        the MJX path takes the generic literal at ``mjx_env.py`` --
        ``weights.get("nosedive_termination_threshold", 0.5)`` -- which is not a
        T-Rex number and is 8.5 degrees stricter.  Stages 2 and 3 diverged that
        way, in exactly the stages whose configs ask for a head-forward running
        posture.

        Asserting that the merged reward weights *carry* the key is what closes
        it: a present key makes the ``.get`` fallback unreachable, so this stays
        valid if that literal ever changes.  Same category as the two envelope
        tests above, which is why it lives beside them.
        """
        importlib.import_module(MJX_CONFIG_MODULES["trex"])
        from environments.shared.mjx_env import _SPECIES_CONFIGS, MJXEnvConfig, canonicalize_env_kwargs

        key = "nosedive_termination_threshold"
        toml_env_kwargs = dict(load_stage_config("trex", stage)["env_kwargs"])

        # Reproduce MJXDinoEnv's registry-then-TOML reward-weight merge.
        config_fields = {field.name for field in MJXEnvConfig.__dataclass_fields__.values()}
        merged = dict(_SPECIES_CONFIGS["trex"].get("reward_weights", {}))
        for name, value in canonicalize_env_kwargs(dict(toml_env_kwargs)).items():
            if name == "reward_weights":
                merged.update(value)
            elif name not in config_fields:
                merged[name] = value

        assert key in merged, (
            f"trex stage {stage}: the MJX reward weights do not carry {key!r}, so the MJX path "
            f"falls back to its generic default while the Gymnasium env uses its own"
        )

        # The Gymnasium env takes the raw TOML keys; two of them are MJX-only.
        env = TRexEnv(**{k: v for k, v in toml_env_kwargs.items() if k not in JAX_ONLY_ENV_KEYS})
        try:
            assert merged[key] == pytest.approx(env.nosedive_termination_threshold), (
                f"trex stage {stage}: MJX resolves {key}={merged[key]} but the Gymnasium env "
                f"uses {env.nosedive_termination_threshold}"
            )
        finally:
            env.close()
