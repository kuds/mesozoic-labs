"""Tests for evaluation utilities (eval_policy, record_stage_video, evaluate)."""

import json
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from environments.shared.evaluation import eval_policy, eval_policy_quality, evaluate, record_stage_video
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


class TestEvalPolicyQuality:
    def test_exports_raw_stance_means_and_episode_variation(self):
        env = MagicMock()
        env.reset.return_value = np.zeros(10)
        env.step.side_effect = [
            (
                np.zeros(10),
                np.array([1.0]),
                [True],
                [
                    {
                        "bite_success": 0.0,
                        "r_foot_contact": 75.0,
                        "l_foot_contact": 25.0,
                    }
                ],
            ),
            (
                np.zeros(10),
                np.array([1.0]),
                [True],
                [
                    {
                        "bite_success": 0.0,
                        "r_foot_contact": 50.0,
                        "l_foot_contact": 50.0,
                    }
                ],
            ),
        ]
        model = MagicMock()
        model.predict.return_value = (np.array([0.0]), None)

        result = eval_policy_quality(model, env, success_keys=[], n_episodes=2)

        assert result["eval_mean_bilateral_support_duty"] == 1.0
        assert result["eval_mean_r_foot_load_share"] == pytest.approx(0.625)
        assert result["eval_std_r_foot_load_share"] == pytest.approx(0.125)

    def test_omits_stance_metrics_without_trex_marker(self):
        env = MagicMock()
        env.reset.return_value = np.zeros(10)
        env.step.return_value = (
            np.zeros(10),
            np.array([1.0]),
            [True],
            [{"r_foot_contact": 75.0, "l_foot_contact": 25.0}],
        )
        model = MagicMock()
        model.predict.return_value = (np.array([0.0]), None)

        result = eval_policy_quality(model, env, success_keys=[], n_episodes=1)

        assert "eval_mean_bilateral_support_duty" not in result
        assert "eval_mean_r_foot_load_share" not in result


class TestReplaySeed:
    """The replay-env seed must work for integer AND semantic stage refs."""

    def test_integer_stages_keep_the_historical_arithmetic(self):
        from environments.shared.evaluation import replay_seed

        # Bit-for-bit the old seed + 2000 + stage, so every recorded
        # integer-stage replay stays reproducible.
        assert replay_seed(42, 1) == 2043
        assert replay_seed(42, 3) == 2045

    def test_semantic_stages_get_a_stable_offset(self):
        """Regression: seed + 2000 + "recovery" raised TypeError inside the
        best-effort replay recorder, silently costing the recovery stage
        both its replays in the field."""
        from environments.shared.evaluation import replay_seed

        first = replay_seed(42, "recovery")
        assert isinstance(first, int)
        assert first == replay_seed(42, "recovery")  # deterministic
        assert first >= 2042  # decorrelated band, like the integer stages
        assert replay_seed(42, "recovery") != replay_seed(42, "some_other_stage")


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

    def test_records_named_camera_views_and_stance_csv_without_extra_steps(self, tmp_path):
        mock_mediapy = MagicMock()
        mock_env = MagicMock()
        mock_env._camera = SimpleNamespace(azimuth=135.0, elevation=-20.0, distance=3.0)
        mock_env.reset.return_value = (np.zeros(10), {})
        mock_env.step.return_value = (
            np.zeros(10),
            1.0,
            True,
            False,
            {"r_foot_contact": 75.0, "l_foot_contact": 25.0},
        )
        mock_env.render.return_value = np.zeros((64, 64, 3), dtype=np.uint8)
        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array([0.0]), None)

        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "mediapy":
                return mock_mediapy
            return real_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=mock_import),
            patch(
                "environments.shared.train_base._ensure_sb3",
                return_value={"DummyVecEnv": MagicMock(), "VecNormalize": MagicMock()},
            ),
        ):
            record_stage_video(
                model=mock_model,
                env_class=MagicMock(return_value=mock_env),
                env_kwargs={},
                stage=1,
                stage_dir=str(tmp_path),
                species="dino",
                label="best",
                max_steps=1,
                camera_views={
                    "side": {"azimuth": 90.0, "elevation": -8.0, "distance": 3.4},
                    "front": {"azimuth": 180.0, "elevation": -8.0, "distance": 3.4},
                },
                collect_stance_diagnostics=True,
            )

        written_paths = [str(call.args[0]) for call in mock_mediapy.write_video.call_args_list]
        assert any(path.endswith("_best.mp4") for path in written_paths)
        assert any(path.endswith("_best_side.mp4") for path in written_paths)
        assert any(path.endswith("_best_front.mp4") for path in written_paths)
        assert all(len(call.args[1]) == 1 for call in mock_mediapy.write_video.call_args_list)
        assert (tmp_path / "dino_ppo_stage1_best_stance.csv").exists()
        assert mock_model.predict.call_count == 1
        assert mock_env.step.call_count == 1
        assert mock_env._camera.azimuth == 135.0
        assert mock_env._camera.elevation == -20.0
        assert mock_env._camera.distance == 3.0


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
                allow_unnormalized=True,
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
                allow_unnormalized=True,
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


def _mock_sb3():
    return {
        "PPO": MagicMock(),
        "SAC": MagicMock(),
        "Monitor": MagicMock(),
        "DummyVecEnv": MagicMock(),
        "VecNormalize": MagicMock(),
    }


def _species_cfg():
    return SimpleNamespace(
        species="velociraptor",
        env_class=MagicMock(),
        height_label="Pelvis height",
        stage3_section_label="Hunting",
    )


def _two_step_episode(target):
    """Make *target* (a VecEnv mock) play one two-step episode."""
    obs = np.zeros((1, 10))
    target.reset.return_value = obs
    target.step.side_effect = [
        (obs, np.array([1.0]), [False], [{"forward_vel": 1.0}]),
        (obs, np.array([1.0]), [True], [{"forward_vel": 1.0, "termination_reason": "truncated"}]),
    ]


def _tagged_model():
    model = MagicMock()
    model.predict.return_value = (np.array([0.0]), None)
    attach_plant_identity(model, _plant_identity())
    return model


class TestEvaluateSidecarResolution:
    """evaluate() must find both sidecar conventions and fail closed on neither (ER1).

    It used to probe only ``<base>_vecnorm.pkl``, so every periodic
    ``<prefix>_<steps>_steps.zip`` checkpoint — whose sidecar is
    ``<prefix>_vecnormalize_<steps>_steps.pkl`` — was scored on raw
    observations after one warning.
    """

    def test_periodic_checkpoint_loads_its_vecnormalize_sidecar(self, tmp_path):
        model_path = tmp_path / "stage1_500000_steps.zip"
        model_path.touch()
        sidecar = tmp_path / "stage1_vecnormalize_500000_steps.pkl"
        sidecar.touch()
        mock_sb3 = _mock_sb3()
        mock_sb3["DummyVecEnv"].return_value = MagicMock()
        normalized = MagicMock()
        attach_plant_identity(normalized, _plant_identity())
        _two_step_episode(normalized)
        mock_sb3["VecNormalize"].load.return_value = normalized
        mock_sb3["PPO"].load.return_value = _tagged_model()

        with (
            patch("environments.shared.train_base._ensure_sb3", return_value=mock_sb3),
            patch("environments.shared.plant_contract.current_plant_identity", return_value=_plant_identity()),
        ):
            evaluate(
                species_cfg=_species_cfg(),
                stage_configs={1: {"name": "balance", "env_kwargs": {}}},
                model_path=str(model_path),
                n_episodes=1,
                render=False,
                stage=1,
            )

        mock_sb3["VecNormalize"].load.assert_called_once()
        assert mock_sb3["VecNormalize"].load.call_args.args[0] == str(sidecar)
        assert normalized.training is False
        assert normalized.norm_reward is False

    def test_missing_sidecar_fails_closed_before_the_model_loads(self, tmp_path):
        mock_sb3 = _mock_sb3()
        mock_vec_env = MagicMock()
        mock_sb3["DummyVecEnv"].return_value = mock_vec_env

        with (
            patch("environments.shared.train_base._ensure_sb3", return_value=mock_sb3),
            patch("environments.shared.plant_contract.current_plant_identity", return_value=_plant_identity()),
            pytest.raises(FileNotFoundError, match="--allow-unnormalized"),
        ):
            evaluate(
                species_cfg=_species_cfg(),
                stage_configs={1: {"name": "balance", "env_kwargs": {}}},
                model_path=str(tmp_path / "stage1_500000_steps.zip"),
                render=False,
                stage=1,
            )

        mock_vec_env.close.assert_called_once()
        mock_sb3["VecNormalize"].load.assert_not_called()
        mock_sb3["PPO"].load.assert_not_called()

    def test_allow_unnormalized_downgrades_to_a_loud_warning(self, tmp_path, caplog):
        mock_sb3 = _mock_sb3()
        mock_vec_env = MagicMock()
        _two_step_episode(mock_vec_env)
        mock_sb3["DummyVecEnv"].return_value = mock_vec_env
        mock_sb3["PPO"].load.return_value = _tagged_model()

        with (
            patch("environments.shared.train_base._ensure_sb3", return_value=mock_sb3),
            patch("environments.shared.plant_contract.current_plant_identity", return_value=_plant_identity()),
            caplog.at_level(logging.WARNING, logger="environments.shared.evaluation"),
        ):
            evaluate(
                species_cfg=_species_cfg(),
                stage_configs={1: {"name": "balance", "env_kwargs": {}}},
                model_path=str(tmp_path / "stage1_500000_steps.zip"),
                n_episodes=1,
                render=False,
                stage=1,
                allow_unnormalized=True,
            )

        assert any("UNNORMALIZED EVAL" in record.message for record in caplog.records)
        mock_sb3["VecNormalize"].load.assert_not_called()
        mock_vec_env.close.assert_called_once()


class TestEvaluateSeeding:
    """evaluate() must seed its environment (ER3).

    Unseeded, ``reset_noise_scale`` drew from OS entropy, so no two CLI
    evaluations were comparable and checkpoint comparisons were unpaired.
    """

    def _run(self, tmp_path, **kwargs):
        mock_sb3 = _mock_sb3()
        mock_vec_env = MagicMock()
        _two_step_episode(mock_vec_env)
        mock_sb3["DummyVecEnv"].return_value = mock_vec_env
        mock_sb3["PPO"].load.return_value = _tagged_model()
        with (
            patch("environments.shared.train_base._ensure_sb3", return_value=mock_sb3),
            patch("environments.shared.plant_contract.current_plant_identity", return_value=_plant_identity()),
        ):
            evaluate(
                species_cfg=_species_cfg(),
                stage_configs={1: {"name": "balance", "env_kwargs": {}}},
                model_path=str(tmp_path / "stage1_model.zip"),
                n_episodes=1,
                render=False,
                stage=1,
                allow_unnormalized=True,
                **kwargs,
            )
        return mock_vec_env

    def test_the_default_seed_is_a_fixed_constant_independent_of_training(self, tmp_path):
        from environments.shared.evaluation import DEFAULT_EVAL_SEED

        vec_env = self._run(tmp_path)
        vec_env.seed.assert_called_once_with(DEFAULT_EVAL_SEED)
        # Not derived from the CLI's default training seed (42) or the
        # trainer's eval-env offset (seed + 1000).
        assert DEFAULT_EVAL_SEED not in (42, 1042)

    def test_an_explicit_seed_reaches_the_env_before_the_first_reset(self, tmp_path):
        vec_env = self._run(tmp_path, seed=7)
        vec_env.seed.assert_called_once_with(7)
        seed_call = next(i for i, (name, _, _) in enumerate(vec_env.mock_calls) if name == "seed")
        reset_call = next(i for i, (name, _, _) in enumerate(vec_env.mock_calls) if name == "reset")
        assert seed_call < reset_call

    def test_the_evaluation_seed_survives_the_checkpoints_saved_training_seed(self, tmp_path):
        """SB3's load() re-seeds the env with a checkpoint's saved seed; ours must win.

        ``load()`` runs ``_setup_model`` -> ``set_random_seed(self.seed)`` ->
        ``env.seed(<training seed>)`` for any zip that carries a seed
        (``--override ppo.seed=N``, external checkpoints).  A VecEnv seed is
        only queued until the next reset, so seeding before the load let the
        training seed replace the evaluation seed and the panel was paired on
        the training seed instead.
        """
        mock_sb3 = _mock_sb3()
        mock_vec_env = MagicMock()
        _two_step_episode(mock_vec_env)
        mock_sb3["DummyVecEnv"].return_value = mock_vec_env

        def load_like_sb3(path, env=None):
            env.seed(42)  # the checkpoint's saved training seed
            return _tagged_model()

        mock_sb3["PPO"].load.side_effect = load_like_sb3
        with (
            patch("environments.shared.train_base._ensure_sb3", return_value=mock_sb3),
            patch("environments.shared.plant_contract.current_plant_identity", return_value=_plant_identity()),
        ):
            evaluate(
                species_cfg=_species_cfg(),
                stage_configs={1: {"name": "balance", "env_kwargs": {}}},
                model_path=str(tmp_path / "stage1_model.zip"),
                n_episodes=1,
                render=False,
                stage=1,
                allow_unnormalized=True,
                seed=7,
            )
        seeds = [args[0] for name, args, _ in mock_vec_env.mock_calls if name == "seed"]
        assert seeds == [42, 7], seeds
        last_seed = max(i for i, (name, _, _) in enumerate(mock_vec_env.mock_calls) if name == "seed")
        first_reset = next(i for i, (name, _, _) in enumerate(mock_vec_env.mock_calls) if name == "reset")
        assert last_seed < first_reset

    def test_same_seed_reproduces_the_panel_on_a_real_vec_env(self, tmp_path):
        """Two runs with one seed give identical numbers; a new seed does not."""
        gym = pytest.importorskip("gymnasium")
        pytest.importorskip("stable_baselines3")
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        class NoisyEnv(gym.Env):
            observation_space = gym.spaces.Box(-10.0, 10.0, (1,), dtype=np.float32)
            action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)

            def __init__(self, render_mode=None):
                self._t = 0

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                self._t = 0
                return self.np_random.normal(size=1).astype(np.float32), {}

            def step(self, action):
                self._t += 1
                obs = self.np_random.normal(size=1).astype(np.float32)
                return obs, float(obs[0]), self._t >= 3, False, {"forward_vel": 0.0}

        class FakePPO:
            @staticmethod
            def load(path, env=None):
                return _tagged_model()

        sb3 = {
            "PPO": FakePPO,
            "SAC": FakePPO,
            "Monitor": Monitor,
            "DummyVecEnv": DummyVecEnv,
            "VecNormalize": VecNormalize,
        }
        panels = []

        def run(seed):
            with (
                patch("environments.shared.train_base._ensure_sb3", return_value=sb3),
                patch("environments.shared.plant_contract.current_plant_identity", return_value=_plant_identity()),
                patch("environments.shared.plant_contract.validate_environment_plant"),
                patch(
                    "environments.shared.evaluation._log_eval_results",
                    side_effect=lambda cfg, agg, n: panels.append(agg["mean_total_reward"]),
                ),
            ):
                evaluate(
                    species_cfg=SimpleNamespace(species="velociraptor", env_class=NoisyEnv),
                    stage_configs={1: {"name": "balance", "env_kwargs": {}}},
                    model_path=str(tmp_path / "stage1_model.zip"),
                    n_episodes=4,
                    render=False,
                    stage=1,
                    allow_unnormalized=True,
                    seed=seed,
                )

        run(123)
        run(123)
        run(456)
        assert panels[0] == panels[1]
        assert panels[0] != panels[2]


class TestPostEvalEpisodes:
    """train()'s post-training panels must be sizable and skippable (EE4).

    The 50-episode quality panel plus the 30-episode velocity panel ran
    after every run with no knob to skip them, costing smoke, debug and CI
    invocations about half their wall time.
    """

    @staticmethod
    def _report(tmp_path, monkeypatch, post_eval_episodes):
        from environments.shared import train_base

        captured = {}

        def _panels(*args, **kwargs):
            captured.update(kwargs)
            return {"quality_eval_checkpoint": "final_model", "mean_forward_vel": 1.5}

        monkeypatch.setattr(train_base, "_post_training_eval_panels", _panels)
        (tmp_path / "models").mkdir()
        train_base._report_hpt_metrics(
            SimpleNamespace(species="velociraptor", success_keys=["strike_success"]),
            MagicMock(),
            MagicMock(),
            SimpleNamespace(best_mean_reward=12.5),
            tmp_path,
            tmp_path / "models",
            1,
            1000,
            "ppo",
            post_eval_episodes=post_eval_episodes,
        )
        return captured, json.loads((tmp_path / "metrics.json").read_text())

    def test_zero_skips_the_panels_but_still_writes_metrics_json(self, tmp_path, monkeypatch):
        captured, metrics = self._report(tmp_path, monkeypatch, 0)
        assert captured == {}, "the panels ran despite post_eval_episodes=0"
        assert metrics["post_eval_skipped"] is True
        assert metrics["quality_eval_checkpoint"] is None
        assert "mean_forward_vel" not in metrics, "skipped panels must not fabricate numbers"
        assert metrics["best_mean_reward"] == 12.5
        assert metrics["timesteps"] == 1000

    def test_none_keeps_the_historical_50_and_30_episode_panels(self, tmp_path, monkeypatch):
        captured, metrics = self._report(tmp_path, monkeypatch, None)
        assert captured == {"quality_episodes": 50, "velocity_episodes": 30}
        assert "post_eval_skipped" not in metrics
        assert metrics["mean_forward_vel"] == 1.5

    def test_a_positive_value_sizes_both_panels(self, tmp_path, monkeypatch):
        captured, _ = self._report(tmp_path, monkeypatch, 3)
        assert captured == {"quality_episodes": 3, "velocity_episodes": 3}

    def test_the_counts_reach_the_rollout_loops(self, tmp_path, monkeypatch):
        from environments.shared import evaluation, train_base

        seen = {}
        monkeypatch.setattr(
            evaluation, "eval_policy_quality", lambda *a, n_episodes: seen.__setitem__("quality", n_episodes) or {}
        )
        monkeypatch.setattr(
            train_base,
            "eval_policy",
            lambda *a, n_episodes: seen.__setitem__("velocity", n_episodes) or ([], [], [], [], []),
        )
        (tmp_path / "models").mkdir()
        panel = train_base._post_training_eval_panels(
            SimpleNamespace(species="velociraptor", success_keys=[]),
            MagicMock(),
            MagicMock(),
            tmp_path / "models",
            "ppo",
            None,
            quality_episodes=4,
            velocity_episodes=2,
        )
        assert seen == {"quality": 4, "velocity": 2}
        assert panel["quality_eval_checkpoint"] == "final_model"

    def test_train_threads_the_knob_through_to_the_report(self):
        """The parameter exists on train() and reaches _report_hpt_metrics."""
        import ast
        import inspect

        from environments.shared import train_base

        assert "post_eval_episodes" in inspect.signature(train_base.train).parameters
        tree = ast.parse(inspect.getsource(train_base.train))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_report_hpt_metrics"
        ]
        assert calls, "train() no longer calls _report_hpt_metrics"
        assert all(any(kw.arg == "post_eval_episodes" for kw in call.keywords) for call in calls)

    def test_the_cli_exposes_the_knob(self):
        from environments.shared.cli import main

        mock_train = MagicMock()
        mock_load = MagicMock(return_value={1: {"env_kwargs": {}, "ppo_kwargs": {}}})
        cfg = MagicMock()
        cfg.species = "velociraptor"
        cfg.stage_descriptions = "1=balance"
        with (
            patch("environments.shared.config.load_all_stages", mock_load),
            patch("environments.shared.train_base.train", mock_train),
            patch("sys.argv", ["prog", "train", "--stage", "1", "--timesteps", "10", "--post-eval-episodes", "0"]),
        ):
            main(cfg)
        assert mock_train.call_args.kwargs["post_eval_episodes"] == 0


class TestEvalCliFlags:
    """The eval subcommand must expose --allow-unnormalized and --seed (ER1/ER3)."""

    @staticmethod
    def _run_eval(argv):
        from environments.shared.cli import main

        mock_eval = MagicMock()
        cfg = MagicMock()
        cfg.species = "velociraptor"
        cfg.stage_descriptions = "1=balance"
        with (
            patch("environments.shared.config.load_all_stages", MagicMock(return_value={1: {}})),
            patch("environments.shared.evaluation.evaluate", mock_eval),
            patch("sys.argv", ["prog", "eval", "/tmp/model.zip", *argv]),
        ):
            main(cfg)
        return mock_eval.call_args.kwargs

    def test_defaults_fail_closed_on_the_fixed_seed(self):
        from environments.shared.evaluation import DEFAULT_EVAL_SEED

        kwargs = self._run_eval([])
        assert kwargs["allow_unnormalized"] is False
        assert kwargs["seed"] == DEFAULT_EVAL_SEED

    def test_flags_are_forwarded(self):
        kwargs = self._run_eval(["--allow-unnormalized", "--seed", "9"])
        assert kwargs["allow_unnormalized"] is True
        assert kwargs["seed"] == 9


class TestDetectStageFromPath:
    """Stage inference must understand both run-directory generations."""

    def test_legacy_stage_token_wins(self):
        from environments.shared.evaluation import detect_stage_from_path

        assert detect_stage_from_path("/runs/20260817/stage2/models/best_model.zip") == 2

    def test_nn_id_layout_maps_ids_to_legacy_numbers(self):
        from environments.shared.evaluation import detect_stage_from_path

        # 03_locomotion is legacy stage 2 at manifest position 3 — the id
        # decides, never the digits. Regression: this silently fell back
        # to stage 1 and evaluated against the wrong env.
        assert detect_stage_from_path("/runs/x/03_locomotion/models/best_model.zip") == 2
        assert detect_stage_from_path("/runs/x/01_stance/models/best_model.zip") == 1

    def test_semantic_only_stage_passes_through_as_id(self):
        from environments.shared.evaluation import detect_stage_from_path

        assert detect_stage_from_path("/runs/x/02_recovery/models/recovery_final.zip") == "recovery"

    def test_unrecognized_paths_keep_the_historical_default(self):
        from environments.shared.evaluation import detect_stage_from_path

        assert detect_stage_from_path("/tmp/some_model.zip") == 1
