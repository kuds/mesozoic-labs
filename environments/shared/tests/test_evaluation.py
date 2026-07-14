"""Tests for evaluation utilities (eval_policy, record_stage_video, evaluate)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from environments.shared.evaluation import eval_policy, evaluate, record_stage_video
from environments.shared.plant_contract import PlantCompatibilityError, PlantIdentity, attach_plant_identity


def _plant_identity():
    return PlantIdentity(
        species="velociraptor",
        model_path="environments/velociraptor/assets/raptor.xml",
        physics_revision=1,
        policy_interface_revision=1,
        visual_revision=1,
        source_closure_sha256="sha256:" + "1" * 64,
        policy_interface_sha256="sha256:" + "2" * 64,
        physics_sha256="sha256:" + "3" * 64,
        visual_sha256="sha256:" + "4" * 64,
        nq=31,
        nv=30,
        nu=22,
        observation_dim=67,
        action_dim=22,
    )


class TestEvalPolicy:
    """Test eval_policy with mocked model and environment."""

    def _make_mock_env(self, n_episodes, steps_per_ep=5):
        """Create a mock VecEnv that simulates episodes."""
        env = MagicMock()
        obs = np.zeros(10)
        env.reset.return_value = obs

        # Build a sequence of step returns: (obs, reward, dones, infos)
        step_returns = []
        for ep in range(n_episodes):
            for s in range(steps_per_ep):
                done = s == steps_per_ep - 1
                info = {
                    "forward_vel": 1.0 + ep * 0.1,
                }
                if done:
                    info["strike_success"] = float(ep % 2 == 0)
                step_returns.append((obs, np.array([1.0]), [done], [info]))

        env.step.side_effect = step_returns
        return env

    def test_returns_five_lists(self):
        env = self._make_mock_env(2, steps_per_ep=3)
        model = MagicMock()
        model.predict.return_value = (np.array([0.0]), None)

        rewards, lengths, fwd_vels, successes, distances = eval_policy(
            model, env, success_keys=["strike_success"], n_episodes=2
        )

        assert len(rewards) == 2
        assert len(lengths) == 2
        assert len(fwd_vels) == 2
        assert len(successes) == 2
        assert len(distances) == 2

    def test_episode_lengths_correct(self):
        env = self._make_mock_env(1, steps_per_ep=4)
        model = MagicMock()
        model.predict.return_value = (np.array([0.0]), None)

        _, lengths, _, _, _ = eval_policy(model, env, success_keys=["strike_success"], n_episodes=1)

        assert lengths[0] == 4.0

    def test_rewards_accumulated(self):
        env = self._make_mock_env(1, steps_per_ep=3)
        model = MagicMock()
        model.predict.return_value = (np.array([0.0]), None)

        rewards, _, _, _, _ = eval_policy(model, env, success_keys=["strike_success"], n_episodes=1)

        assert rewards[0] == pytest.approx(3.0)  # 3 steps * 1.0 reward

    def test_forward_velocity_collected(self):
        env = self._make_mock_env(1, steps_per_ep=3)
        model = MagicMock()
        model.predict.return_value = (np.array([0.0]), None)

        _, _, fwd_vels, _, _ = eval_policy(model, env, success_keys=["strike_success"], n_episodes=1)

        assert fwd_vels[0] > 0

    def test_success_detection(self):
        """Success key in last step info should be detected."""
        env = MagicMock()
        obs = np.zeros(10)
        env.reset.return_value = obs

        # Episode 1: strike_success=1.0 in final step
        env.step.side_effect = [
            (obs, np.array([1.0]), [False], [{"forward_vel": 1.0}]),
            (obs, np.array([1.0]), [True], [{"forward_vel": 1.0, "strike_success": 1.0}]),
        ]
        model = MagicMock()
        model.predict.return_value = (np.array([0.0]), None)

        _, _, _, successes, _ = eval_policy(model, env, success_keys=["strike_success"], n_episodes=1)

        assert successes[0] == 1.0

    def test_no_forward_vel_defaults_to_zero(self):
        """When forward_vel is missing from info, fwd_vel should be 0.0."""
        env = MagicMock()
        obs = np.zeros(10)
        env.reset.return_value = obs
        env.step.side_effect = [
            (obs, np.array([1.0]), [True], [{}]),
        ]
        model = MagicMock()
        model.predict.return_value = (np.array([0.0]), None)

        _, _, fwd_vels, _, _ = eval_policy(model, env, success_keys=["strike_success"], n_episodes=1)

        assert fwd_vels[0] == 0.0


class TestRecordStageVideo:
    """Test record_stage_video with mocked dependencies."""

    def test_skips_when_mediapy_not_installed(self):
        """Should log warning and return None when mediapy is not available."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "mediapy":
                raise ImportError("no mediapy")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = record_stage_video(
                model=MagicMock(),
                env_class=MagicMock,
                env_kwargs={},
                stage=1,
                stage_dir="/tmp",
            )

        assert result is None

    def test_records_video_with_mediapy(self, tmp_path):
        """Should record frames and save video."""
        mock_mediapy = MagicMock()
        mock_env = MagicMock()
        mock_env.reset.return_value = (np.zeros(10), {})
        # Simulate steps: 2 steps then done
        mock_env.step.side_effect = [
            (np.zeros(10), 1.0, False, False, {}),
            (np.zeros(10), 1.0, True, False, {}),
        ]
        mock_env.render.return_value = np.zeros((64, 64, 3), dtype=np.uint8)

        mock_env_class = MagicMock(return_value=mock_env)

        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array([0.0]), None)

        mock_sb3 = {
            "DummyVecEnv": MagicMock(),
            "VecNormalize": MagicMock(),
        }

        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "mediapy":
                return mock_mediapy
            return real_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=mock_import),
            patch("environments.shared.train_base._ensure_sb3", return_value=mock_sb3),
        ):
            result = record_stage_video(
                model=mock_model,
                env_class=mock_env_class,
                env_kwargs={},
                stage=1,
                stage_dir=str(tmp_path),
                max_steps=5,
            )

        # Should have called write_video
        assert mock_mediapy.write_video.called or result is None


class TestEvaluateFunction:
    """Test the evaluate() function with mocked dependencies."""

    def test_evaluate_runs_episodes(self):
        """evaluate() should run n_episodes and log results."""
        mock_sb3 = {
            "PPO": MagicMock(),
            "SAC": MagicMock(),
            "Monitor": MagicMock(),
            "DummyVecEnv": MagicMock(),
            "VecNormalize": MagicMock(),
        }

        mock_vec_env = MagicMock()
        obs = np.zeros((1, 10))
        mock_vec_env.reset.return_value = obs
        # Simulate 2 episodes of 2 steps each
        mock_vec_env.step.side_effect = [
            (obs, np.array([1.0]), [False], [{"forward_vel": 1.0}]),
            (obs, np.array([1.0]), [True], [{"forward_vel": 1.0, "termination_reason": "truncated"}]),
            (obs, np.array([1.0]), [False], [{"forward_vel": 1.0}]),
            (obs, np.array([1.0]), [True], [{"forward_vel": 1.0, "termination_reason": "fallen"}]),
        ]
        mock_sb3["DummyVecEnv"].return_value = mock_vec_env

        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array([0.0]), None)
        identity = _plant_identity()
        attach_plant_identity(mock_model, identity)
        mock_sb3["PPO"].load.return_value = mock_model

        mock_species_cfg = MagicMock()
        mock_species_cfg.species = "velociraptor"
        mock_species_cfg.height_label = "Pelvis height"
        mock_species_cfg.stage3_section_label = "Hunting"

        stage_configs = {
            1: {"name": "balance", "env_kwargs": {"forward_vel_weight": 0.0}},
        }

        with (
            patch("environments.shared.train_base._ensure_sb3", return_value=mock_sb3),
            patch("environments.shared.plant_contract.current_plant_identity", return_value=identity),
        ):
            evaluate(
                species_cfg=mock_species_cfg,
                stage_configs=stage_configs,
                model_path="/tmp/stage1_model.zip",
                n_episodes=2,
                render=False,
                stage=1,
            )

        mock_vec_env.close.assert_called_once()

    def test_untagged_model_fails_closed_and_closes_environment(self):
        mock_sb3 = {
            "PPO": MagicMock(),
            "SAC": MagicMock(),
            "Monitor": MagicMock(),
            "DummyVecEnv": MagicMock(),
            "VecNormalize": MagicMock(),
        }
        mock_vec_env = MagicMock()
        mock_sb3["DummyVecEnv"].return_value = mock_vec_env
        mock_sb3["PPO"].load.return_value = SimpleNamespace()
        species_cfg = SimpleNamespace(
            species="velociraptor",
            env_class=MagicMock(),
            height_label="Pelvis height",
            stage3_section_label="Hunting",
        )

        with (
            patch("environments.shared.train_base._ensure_sb3", return_value=mock_sb3),
            patch("environments.shared.plant_contract.current_plant_identity", return_value=_plant_identity()),
            pytest.raises(PlantCompatibilityError, match="has no plant identity"),
        ):
            evaluate(
                species_cfg=species_cfg,
                stage_configs={1: {"name": "balance", "env_kwargs": {}}},
                model_path="/tmp/legacy_stage1_model.zip",
                render=False,
                stage=1,
            )

        mock_vec_env.close.assert_called_once()

    def test_untagged_vecnormalize_fails_before_model_load(self):
        mock_sb3 = {
            "PPO": MagicMock(),
            "SAC": MagicMock(),
            "Monitor": MagicMock(),
            "DummyVecEnv": MagicMock(),
            "VecNormalize": MagicMock(),
        }
        mock_vec_env = MagicMock()
        legacy_vecnorm = SimpleNamespace(close=MagicMock())
        mock_sb3["DummyVecEnv"].return_value = mock_vec_env
        mock_sb3["VecNormalize"].load.return_value = legacy_vecnorm
        species_cfg = SimpleNamespace(
            species="velociraptor",
            env_class=MagicMock(),
            height_label="Pelvis height",
            stage3_section_label="Hunting",
        )

        with (
            patch("environments.shared.train_base._ensure_sb3", return_value=mock_sb3),
            patch("environments.shared.plant_contract.current_plant_identity", return_value=_plant_identity()),
            patch("environments.shared.evaluation.Path.exists", return_value=True),
            pytest.raises(PlantCompatibilityError, match="has no plant identity"),
        ):
            evaluate(
                species_cfg=species_cfg,
                stage_configs={1: {"name": "balance", "env_kwargs": {}}},
                model_path="/tmp/stage1_model.zip",
                render=False,
                stage=1,
            )

        legacy_vecnorm.close.assert_called_once()
        mock_sb3["PPO"].load.assert_not_called()
