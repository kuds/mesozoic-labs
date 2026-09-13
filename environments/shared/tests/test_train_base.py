"""Tests for shared training infrastructure (train_base.py)."""

import dataclasses
import json
import logging
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from environments.shared.plant_contract import (
    MODEL_IDENTITY_ATTRIBUTE,
    PlantCompatibilityError,
    attach_plant_identity,
    current_plant_identity,
    validate_model_plant,
)
from environments.shared.train_base import (
    SpeciesConfig,
    _apply_overrides,
    _build_core_callbacks,
    _cast_value,
    _create_or_load_model,
    _is_gcs_path,
    _load_vecnorm_into_envs,
    _make_local_tb_dir,
    _prepare_alg_kwargs,
    _save_final_and_sync_tb,
    _select_handoff_checkpoint,
    _sync_tb_to_gcs,
    cosine_schedule,
    linear_schedule,
)

from .reporting_helpers import make_plant_identity as _plant_identity

# ── linear_schedule ──────────────────────────────────────────────────────


class TestLinearSchedule:
    def test_returns_initial_at_start(self):
        sched = linear_schedule(1e-3, 1e-4)
        assert sched(1.0) == pytest.approx(1e-3)

    def test_returns_final_at_end(self):
        sched = linear_schedule(1e-3, 1e-4)
        assert sched(0.0) == pytest.approx(1e-4)

    def test_midpoint(self):
        sched = linear_schedule(1e-3, 1e-4)
        mid = sched(0.5)
        expected = 1e-4 + 0.5 * (1e-3 - 1e-4)
        assert mid == pytest.approx(expected)

    def test_constant_when_initial_equals_final(self):
        sched = linear_schedule(5e-4, 5e-4)
        assert sched(0.0) == pytest.approx(5e-4)
        assert sched(0.5) == pytest.approx(5e-4)
        assert sched(1.0) == pytest.approx(5e-4)


# ── _cast_value and _apply_overrides ─────────────────────────────────────
# These are re-exported from cli.py; comprehensive tests live in test_cli.py.
# We verify that the re-exports are importable from train_base.


class TestReExports:
    def test_cast_value_importable(self):
        assert callable(_cast_value)

    def test_apply_overrides_importable(self):
        assert callable(_apply_overrides)


# ── SpeciesConfig ────────────────────────────────────────────────────────


class TestSpeciesConfig:
    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(SpeciesConfig)

    def test_required_fields(self):
        fields = {f.name for f in dataclasses.fields(SpeciesConfig)}
        expected = {
            "species",
            "env_class",
            "stage_descriptions",
            "height_label",
            "stage3_section_label",
            "success_keys",
        }
        assert expected == fields

    def test_from_velociraptor(self):
        from environments.velociraptor.envs.raptor_env import RaptorEnv

        cfg = SpeciesConfig(
            species="velociraptor",
            env_class=RaptorEnv,
            stage_descriptions="1=balance, 2=locomotion, 3=strike",
            height_label="Pelvis height",
            stage3_section_label="Hunting",
            success_keys=["strike_success"],
        )
        assert cfg.species == "velociraptor"
        assert cfg.env_class is RaptorEnv

    def test_from_trex(self):
        from environments.trex.envs.trex_env import TRexEnv

        cfg = SpeciesConfig(
            species="trex",
            env_class=TRexEnv,
            stage_descriptions="1=balance, 2=locomotion, 3=bite",
            height_label="Pelvis height",
            stage3_section_label="Hunting",
            success_keys=["bite_success"],
        )
        assert cfg.species == "trex"

    def test_from_brachiosaurus(self):
        from environments.brachiosaurus.envs.brachio_env import BrachioEnv

        cfg = SpeciesConfig(
            species="brachiosaurus",
            env_class=BrachioEnv,
            stage_descriptions="1=balance, 2=locomotion, 3=food_reach",
            height_label="Torso height",
            stage3_section_label="Food Reaching",
            success_keys=["food_reached"],
        )
        assert cfg.species == "brachiosaurus"


class TestBuildCoreCallbacks:
    @pytest.mark.parametrize(
        ("stage", "success_threshold", "success_applicable"),
        [(1, 0.0, False), (3, 0.5, True)],
    )
    def test_wires_stage_aware_eval_and_plateau_callbacks(
        self,
        tmp_path,
        stage,
        success_threshold,
        success_applicable,
    ):
        gym = pytest.importorskip("gymnasium")
        pytest.importorskip("stable_baselines3")
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.vec_env import DummyVecEnv

        from environments.shared.eval_diagnostics import (
            StageAwareEvalCallback,
            StageGatePlateauCallback,
        )

        class TinyEnv(gym.Env):
            observation_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
            action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                return np.zeros(1, dtype=np.float32), {}

            def step(self, action):
                return np.zeros(1, dtype=np.float32), 0.0, False, False, {"forward_vel": 0.0}

        eval_env = DummyVecEnv([TinyEnv])
        stage_config = {
            "curriculum_kwargs": {
                "min_avg_reward": 100.0,
                "min_success_rate": success_threshold,
                "diagnostics_plateau_window": 7,
                "diagnostics_plateau_min_relative_variation": 0.02,
            }
        }

        callbacks, eval_callback, _ = _build_core_callbacks(
            {"CheckpointCallback": CheckpointCallback},
            eval_env,
            tmp_path / "models",
            tmp_path / "logs",
            stage,
            1,
            100,
            100,
            0,
            stage_config,
        )

        assert isinstance(eval_callback, StageAwareEvalCallback)
        assert eval_callback.success_applicable is success_applicable
        plateau_callback = next(callback for callback in callbacks if isinstance(callback, StageGatePlateauCallback))
        assert plateau_callback.plateau_window == 7
        assert plateau_callback.min_relative_variation == 0.02
        eval_env.close()

    def test_collapse_early_stop_params_are_configurable(self, tmp_path):
        gym = pytest.importorskip("gymnasium")
        pytest.importorskip("stable_baselines3")
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.vec_env import DummyVecEnv

        from environments.shared.curriculum import EvalCollapseEarlyStopCallback

        class TinyEnv(gym.Env):
            observation_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
            action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                return np.zeros(1, dtype=np.float32), {}

            def step(self, action):
                return np.zeros(1, dtype=np.float32), 0.0, False, False, {"forward_vel": 0.0}

        def _build_collapse_cb(curriculum_kwargs):
            eval_env = DummyVecEnv([TinyEnv])
            callbacks, _, _ = _build_core_callbacks(
                {"CheckpointCallback": CheckpointCallback},
                eval_env,
                tmp_path / "models",
                tmp_path / "logs",
                2,
                1,
                100,
                100,
                0,
                {"curriculum_kwargs": curriculum_kwargs},
            )
            eval_env.close()
            return next(cb for cb in callbacks if isinstance(cb, EvalCollapseEarlyStopCallback))

        # Explicit [curriculum] overrides are honoured.
        cb = _build_collapse_cb(
            {
                "min_avg_reward": 100.0,
                "collapse_min_evals": 20,
                "collapse_patience": 10,
                "collapse_drop_fraction": 0.5,
                "collapse_peak_floor": 42.0,
                "collapse_smoothing_window": 3,
            }
        )
        assert cb.min_evals == 20
        assert cb.patience == 10
        assert cb.drop_fraction == 0.5
        assert cb.peak_floor == 42.0
        assert cb.smoothing_window == 3

        # Defaults are lenient (looser than the old hardcoded 8 / 5 / 0.3) and
        # the smoothing window defaults to 5 evals.  The arming floor no longer
        # falls back to the stage's reward gate: an unconfigured backstop must
        # not abort a run, and coupling it to min_avg_reward meant removing the
        # reward gate silently armed collapse detection on any positive peak.
        # See docs/STAGE1_SPLIT_PLAN.md section 7.4.
        cb_default = _build_collapse_cb({"min_avg_reward": 100.0})
        assert cb_default.min_evals == 12
        assert cb_default.patience == 8
        assert cb_default.drop_fraction == 0.4
        assert cb_default.peak_floor == float("inf")
        assert cb_default.smoothing_window == 5


# ── cosine_schedule ─────────────────────────────────────────────────────


class TestSelectHandoffCheckpoint:
    """The quality eval and next-stage loading must agree on the checkpoint."""

    def test_returns_none_when_nothing_saved(self, tmp_path):
        assert _select_handoff_checkpoint(tmp_path) is None

    def test_ignores_candidates_without_matched_vecnorm(self, tmp_path):
        (tmp_path / "best_model.zip").touch()
        assert _select_handoff_checkpoint(tmp_path) is None

    def test_selects_best_model_when_complete(self, tmp_path):
        (tmp_path / "best_model.zip").touch()
        (tmp_path / "best_model_vecnorm.pkl").touch()
        name, model_path, vecnorm_path = _select_handoff_checkpoint(tmp_path)
        assert name == "best_model"
        assert model_path == str(tmp_path / "best_model")
        assert vecnorm_path == str(tmp_path / "best_model_vecnorm.pkl")

    def test_prefers_robust_best_model(self, tmp_path):
        for stem in ("best_model", "robust_best_model"):
            (tmp_path / f"{stem}.zip").touch()
            (tmp_path / f"{stem}_vecnorm.pkl").touch()
        name, model_path, vecnorm_path = _select_handoff_checkpoint(tmp_path)
        assert name == "robust_best_model"
        assert model_path == str(tmp_path / "robust_best_model")
        assert vecnorm_path == str(tmp_path / "robust_best_model_vecnorm.pkl")


class TestCosineSchedule:
    def test_returns_initial_at_start(self):
        sched = cosine_schedule(1e-3, 1e-4)
        assert sched(1.0) == pytest.approx(1e-3)

    def test_returns_final_at_end(self):
        sched = cosine_schedule(1e-3, 1e-4)
        assert sched(0.0) == pytest.approx(1e-4)

    def test_midpoint_matches_cosine_formula(self):
        sched = cosine_schedule(1e-3, 1e-4)
        mid = sched(0.5)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * 0.5))
        expected = 1e-4 + cosine_decay * (1e-3 - 1e-4)
        assert mid == pytest.approx(expected)

    def test_constant_when_initial_equals_final(self):
        sched = cosine_schedule(5e-4, 5e-4)
        for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
            assert sched(p) == pytest.approx(5e-4)

    def test_monotonically_decreasing(self):
        sched = cosine_schedule(1e-3, 1e-4)
        values = [sched(p) for p in [1.0, 0.75, 0.5, 0.25, 0.0]]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1]

    def test_always_between_bounds(self):
        sched = cosine_schedule(1e-3, 1e-4)
        for p in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            val = sched(p)
            assert 1e-4 <= val <= 1e-3


# ── GCS path utilities ─────────────────────────────────────────────────


class TestIsGcsPath:
    def test_gcs_path_detected(self):
        assert _is_gcs_path("/gcs/my-bucket/runs/run1") is True

    def test_local_path_not_gcs(self):
        assert _is_gcs_path("/home/user/runs/run1") is False

    def test_relative_path_not_gcs(self):
        assert _is_gcs_path("runs/run1") is False

    def test_path_object(self):
        assert _is_gcs_path(Path("/gcs/bucket/tb")) is True
        assert _is_gcs_path(Path("/tmp/tb")) is False


class TestMakeLocalTbDir:
    def test_creates_directory(self, tmp_path):
        local_dir = _make_local_tb_dir("/gcs/bucket/tb_logs")
        assert local_dir.exists()
        assert local_dir.is_dir()

    def test_stable_across_calls(self):
        d1 = _make_local_tb_dir("/gcs/bucket/tb_logs")
        d2 = _make_local_tb_dir("/gcs/bucket/tb_logs")
        assert d1 == d2

    def test_different_paths_get_different_dirs(self):
        d1 = _make_local_tb_dir("/gcs/bucket-a/tb")
        d2 = _make_local_tb_dir("/gcs/bucket-b/tb")
        assert d1 != d2


class TestSyncTbToGcs:
    def test_copies_files(self, tmp_path):
        src = tmp_path / "local_tb"
        src.mkdir()
        (src / "events.out.tfevents.1234").write_text("data")
        (src / "subdir").mkdir()
        (src / "subdir" / "nested.txt").write_text("nested")

        dest = tmp_path / "gcs_tb"
        _sync_tb_to_gcs(src, str(dest))

        assert (dest / "events.out.tfevents.1234").read_text() == "data"
        assert (dest / "subdir" / "nested.txt").read_text() == "nested"
        # Source should be cleaned up
        assert not src.exists()

    def test_noop_when_source_missing(self, tmp_path):
        dest = tmp_path / "gcs_tb"
        # Should not raise
        _sync_tb_to_gcs(tmp_path / "nonexistent", str(dest))
        assert not dest.exists()


# ── _prepare_alg_kwargs ──────────────────────────────────────────────────


class TestPrepareAlgKwargs:
    """Tests for the shared algorithm kwargs setup helper."""

    def _make_config(self, **ppo_overrides):
        ppo = {
            "learning_rate": 3e-4,
            "batch_size": 64,
            "clip_range": 0.2,
        }
        ppo.update(ppo_overrides)
        return {
            "ppo_kwargs": ppo,
            "sac_kwargs": {"learning_rate": 1e-3, "batch_size": 256},
        }

    def test_ppo_basic(self, tmp_path):
        config = self._make_config()
        kwargs, local_tb, gcs_tb = _prepare_alg_kwargs(config, "ppo", 1, tmp_path, True)
        assert kwargs["learning_rate"] == 3e-4
        assert kwargs["batch_size"] == 64
        assert kwargs["verbose"] == 1
        assert local_tb is None  # not GCS
        assert gcs_tb == tmp_path / "tensorboard"

    def test_sac_selects_sac_kwargs(self, tmp_path):
        config = self._make_config()
        kwargs, _, _ = _prepare_alg_kwargs(config, "sac", 0, tmp_path, True)
        assert kwargs["learning_rate"] == 1e-3
        assert kwargs["batch_size"] == 256

    def test_linear_lr_schedule(self, tmp_path):
        config = self._make_config(learning_rate_end=1e-5)
        kwargs, _, _ = _prepare_alg_kwargs(config, "ppo", 1, tmp_path, False)
        # learning_rate should be a callable schedule
        assert callable(kwargs["learning_rate"])
        assert kwargs["learning_rate"](1.0) == pytest.approx(3e-4)
        assert kwargs["learning_rate"](0.0) == pytest.approx(1e-5)
        # learning_rate_end should be consumed (popped)
        assert "learning_rate_end" not in kwargs

    def test_cosine_lr_schedule(self, tmp_path):
        config = self._make_config(learning_rate_end=1e-5, lr_schedule="cosine")
        kwargs, _, _ = _prepare_alg_kwargs(config, "ppo", 1, tmp_path, False)
        assert callable(kwargs["learning_rate"])
        assert kwargs["learning_rate"](1.0) == pytest.approx(3e-4)
        assert kwargs["learning_rate"](0.0) == pytest.approx(1e-5)

    def test_clip_range_annealing(self, tmp_path):
        config = self._make_config(clip_range_end=0.05)
        kwargs, _, _ = _prepare_alg_kwargs(config, "ppo", 1, tmp_path, False)
        assert callable(kwargs["clip_range"])
        assert kwargs["clip_range"](1.0) == pytest.approx(0.2)
        assert kwargs["clip_range"](0.0) == pytest.approx(0.05)

    def test_tb_disabled(self, tmp_path):
        config = self._make_config()
        kwargs, local_tb, _ = _prepare_alg_kwargs(config, "ppo", 1, tmp_path, False)
        assert "tensorboard_log" not in kwargs
        assert local_tb is None

    def test_tb_gcs_buffering(self):
        config = self._make_config()
        gcs_path = Path("/gcs/bucket/run1")
        kwargs, local_tb, gcs_tb = _prepare_alg_kwargs(config, "ppo", 1, gcs_path, True)
        assert local_tb is not None
        assert local_tb.exists()
        assert kwargs["tensorboard_log"] == str(local_tb)

    def test_does_not_mutate_original_config(self, tmp_path):
        config = self._make_config(learning_rate_end=1e-5, clip_range_end=0.05)
        original_ppo = config["ppo_kwargs"].copy()
        _prepare_alg_kwargs(config, "ppo", 1, tmp_path, False)
        # Original config should be unchanged
        assert config["ppo_kwargs"] == original_ppo


# ── _load_vecnorm_into_envs ──────────────────────────────────────────────


class TestLoadVecnormIntoEnvs:
    def test_no_load_path_disables_eval_training(self):
        train_env = MagicMock()
        eval_env = MagicMock()
        _load_vecnorm_into_envs(None, train_env, eval_env, task_load_mode="resume_same_stage")
        assert eval_env.training is False
        assert eval_env.norm_reward is False

    @patch("environments.shared.train_base.logger")
    def test_with_load_path_calls_load_vecnorm(self, mock_logger):
        train_env = MagicMock()
        eval_env = MagicMock()
        with patch("environments.shared.curriculum.load_vecnorm_stats", return_value=True) as mock_load:
            _load_vecnorm_into_envs("/path/to/model.zip", train_env, eval_env, task_load_mode="initialize_next_stage")
        mock_load.assert_called_once_with(
            "/path/to/model_vecnorm.pkl",
            train_env,
            eval_env,
            unsafe_skip_plant_validation=True,
        )

    @patch("environments.shared.train_base.logger")
    def test_strips_zip_extension(self, mock_logger):
        train_env = MagicMock()
        eval_env = MagicMock()
        with patch("environments.shared.curriculum.load_vecnorm_stats", return_value=True) as mock_load:
            _load_vecnorm_into_envs("/path/model.zip", train_env, eval_env, task_load_mode="initialize_next_stage")
        mock_load.assert_called_once_with(
            "/path/model_vecnorm.pkl",
            train_env,
            eval_env,
            unsafe_skip_plant_validation=True,
        )

    @patch("environments.shared.train_base.logger")
    def test_same_stage_resume_carries_ret_rms(self, mock_logger):
        # The reward distribution is unchanged on a same-stage resume, so the
        # loaded ret_rms is the correct one (review TC6).
        train_env = MagicMock()
        eval_env = MagicMock()
        with patch("environments.shared.curriculum.load_vecnorm_stats", return_value=True) as mock_load:
            _load_vecnorm_into_envs("/path/model.zip", train_env, eval_env, task_load_mode="resume_same_stage")
        mock_load.assert_called_once_with(
            "/path/model_vecnorm.pkl",
            train_env,
            eval_env,
            unsafe_skip_plant_validation=True,
            carry_ret_rms=True,
        )

    def test_missing_sidecar_fails_closed(self):
        # Training a loaded policy under fresh normalization statistics
        # collapses it within the first updates — refuse, don't warn (TC5).
        train_env = MagicMock()
        eval_env = MagicMock()
        with patch("environments.shared.curriculum.load_vecnorm_stats", return_value=False):
            with pytest.raises(FileNotFoundError, match="allow-fresh-vecnorm"):
                _load_vecnorm_into_envs("/path/model", train_env, eval_env, task_load_mode="resume_same_stage")

    @patch("environments.shared.train_base.logger")
    def test_missing_sidecar_escape_hatch_warns_and_resets_eval_env(self, mock_logger):
        train_env = MagicMock()
        eval_env = MagicMock()
        with patch("environments.shared.curriculum.load_vecnorm_stats", return_value=False):
            _load_vecnorm_into_envs(
                "/path/model",
                train_env,
                eval_env,
                task_load_mode="resume_same_stage",
                allow_fresh_vecnorm=True,
            )
        assert eval_env.training is False
        assert eval_env.norm_reward is False
        mock_logger.warning.assert_called_once()
        # The old message claimed only the EVAL env was affected; the train
        # env trains under fresh statistics too, and the warning must say so.
        assert "train" in mock_logger.warning.call_args.args[0].lower()

    def test_forwards_plant_contract_and_legacy_override(self):
        train_env = MagicMock()
        eval_env = MagicMock()
        identity = _plant_identity()
        with patch("environments.shared.curriculum.load_vecnorm_stats", return_value=True) as mock_load:
            _load_vecnorm_into_envs(
                "/path/model.zip",
                train_env,
                eval_env,
                plant_identity=identity,
                allow_legacy_plant=True,
                task_load_mode="initialize_next_stage",
            )

        mock_load.assert_called_once_with(
            "/path/model_vecnorm.pkl",
            train_env,
            eval_env,
            current_plant=identity,
            allow_legacy_plant=True,
        )


# ── _create_or_load_model ────────────────────────────────────────────────


class TestCreateOrLoadModel:
    def _make_sb3(self):
        return {
            "PPO": MagicMock(),
            "SAC": MagicMock(),
        }

    def test_creates_new_ppo_model(self):
        sb3 = self._make_sb3()
        env = MagicMock()
        kwargs = {"batch_size": 64, "policy_kwargs": {"net_arch": [256, 256]}}
        _create_or_load_model(sb3, "ppo", kwargs, env)
        sb3["PPO"].assert_called_once_with(
            "MlpPolicy",
            env,
            policy_kwargs={"net_arch": [256, 256]},
            batch_size=64,
        )
        # policy_kwargs should be popped from kwargs
        assert "policy_kwargs" not in kwargs

    def test_creates_new_sac_model(self):
        sb3 = self._make_sb3()
        env = MagicMock()
        kwargs = {"batch_size": 256}
        _create_or_load_model(sb3, "sac", kwargs, env)
        sb3["SAC"].assert_called_once()

    def test_loads_existing_model(self):
        sb3 = self._make_sb3()
        env = MagicMock()
        kwargs = {"batch_size": 64, "policy_kwargs": {"net_arch": [128]}}
        _create_or_load_model(sb3, "ppo", kwargs, env, load_path="/path/model")
        sb3["PPO"].load.assert_called_once_with("/path/model", env=env, batch_size=64)
        # policy_kwargs should NOT be passed to .load()
        call_kwargs = sb3["PPO"].load.call_args
        assert "policy_kwargs" not in call_kwargs.kwargs

    def test_pops_policy_kwargs_even_on_load(self):
        sb3 = self._make_sb3()
        kwargs = {"policy_kwargs": {"net_arch": [64]}, "lr": 1e-3}
        _create_or_load_model(sb3, "ppo", kwargs, MagicMock(), load_path="/p")
        assert "policy_kwargs" not in kwargs

    def test_new_model_is_tagged_with_current_plant(self):
        sb3 = self._make_sb3()
        identity = _plant_identity()

        model = _create_or_load_model(
            sb3,
            "ppo",
            {},
            MagicMock(),
            plant_identity=identity,
        )

        assert getattr(model, MODEL_IDENTITY_ATTRIBUTE) == identity.to_dict()

    def test_legacy_model_fails_closed_by_default(self):
        sb3 = self._make_sb3()
        sb3["PPO"].load.return_value = SimpleNamespace()

        with pytest.raises(PlantCompatibilityError, match="has no plant identity"):
            _create_or_load_model(
                sb3,
                "ppo",
                {},
                MagicMock(),
                load_path="/legacy/model.zip",
                plant_identity=_plant_identity(),
            )

    def test_explicit_legacy_load_is_retagged_for_next_save(self):
        sb3 = self._make_sb3()
        legacy_model = SimpleNamespace()
        sb3["PPO"].load.return_value = legacy_model
        identity = _plant_identity()

        model = _create_or_load_model(
            sb3,
            "ppo",
            {},
            MagicMock(),
            load_path="/legacy/model.zip",
            plant_identity=identity,
            allow_legacy_plant=True,
        )

        assert model is legacy_model
        assert getattr(model, MODEL_IDENTITY_ATTRIBUTE) == identity.to_dict()

    def test_tagged_incompatible_model_is_rejected_even_with_legacy_override(self):
        sb3 = self._make_sb3()
        loaded_model = SimpleNamespace()
        attach_plant_identity(loaded_model, _plant_identity(physics_revision=2))
        sb3["PPO"].load.return_value = loaded_model

        with pytest.raises(PlantCompatibilityError, match="physics_revision"):
            _create_or_load_model(
                sb3,
                "ppo",
                {},
                MagicMock(),
                load_path="/wrong/model.zip",
                plant_identity=_plant_identity(),
                allow_legacy_plant=True,
            )

    def test_sb3_model_and_vecnormalize_round_trip_plant_identity(self, tmp_path):
        pytest.importorskip("stable_baselines3")
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        from environments.velociraptor.envs.raptor_env import RaptorEnv

        identity = current_plant_identity("velociraptor")

        def make_env():
            return Monitor(RaptorEnv())

        vec_env = VecNormalize(DummyVecEnv([make_env]), norm_obs=True, norm_reward=True)
        attach_plant_identity(vec_env, identity)
        model = PPO("MlpPolicy", vec_env, n_steps=8, batch_size=4, n_epochs=1, verbose=0)
        attach_plant_identity(model, identity)
        model_path = tmp_path / "model"
        vecnorm_path = tmp_path / "model_vecnorm.pkl"
        model.save(model_path)
        vec_env.save(vecnorm_path)
        vec_env.close()

        loaded_model = PPO.load(model_path)
        validate_model_plant(loaded_model, identity)

        base_env = DummyVecEnv([make_env])
        loaded_vecnorm = VecNormalize.load(vecnorm_path, base_env)
        try:
            validate_model_plant(loaded_vecnorm, identity)
        finally:
            loaded_vecnorm.close()


# ── _save_final_and_sync_tb ──────────────────────────────────────────────


class TestSaveFinalAndSyncTb:
    def test_saves_model_and_vecnorm(self, tmp_path):
        model = MagicMock()
        train_env = MagicMock()
        model_dir = tmp_path / "models"
        model_dir.mkdir()

        result = _save_final_and_sync_tb(
            model,
            train_env,
            model_dir,
            1,
            None,
            tmp_path / "tb",
        )

        assert result == model_dir / "stage1_final"
        model.save.assert_called_once_with(str(model_dir / "stage1_final"))
        train_env.save.assert_called_once_with(str(model_dir / "stage1_final") + "_vecnorm.pkl")

    def test_syncs_tb_when_local_dir_provided(self, tmp_path):
        model = MagicMock()
        train_env = MagicMock()
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        local_tb = tmp_path / "local_tb"
        local_tb.mkdir()
        (local_tb / "events.out").write_text("data")
        gcs_tb = tmp_path / "gcs_tb"

        _save_final_and_sync_tb(model, train_env, model_dir, 2, local_tb, gcs_tb)

        assert (gcs_tb / "events.out").read_text() == "data"

    def test_tb_sync_failure_does_not_raise(self, tmp_path):
        model = MagicMock()
        train_env = MagicMock()
        model_dir = tmp_path / "models"
        model_dir.mkdir()

        with patch(
            "environments.shared.train_base._sync_tb_to_gcs",
            side_effect=OSError("FUSE error"),
        ):
            # Should not raise
            _save_final_and_sync_tb(
                model,
                train_env,
                model_dir,
                1,
                tmp_path / "local_tb",
                tmp_path / "gcs_tb",
            )


class TestCheckpointRetentionIsWired:
    """The retention callback must actually be installed, not merely exist.

    It is unit-tested in isolation in test_curriculum_checkpoints.py; this
    covers the wiring in ``_build_core_callbacks``, which is the only thing
    that makes it run during training.
    """

    def _build(self, tmp_path, stage_config, n_envs=1, save_freq=100):
        gym = pytest.importorskip("gymnasium")
        pytest.importorskip("stable_baselines3")
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.vec_env import DummyVecEnv

        class TinyEnv(gym.Env):
            observation_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
            action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                return np.zeros(1, dtype=np.float32), {}

            def step(self, action):
                return np.zeros(1, dtype=np.float32), 0.0, False, False, {}

        eval_env = DummyVecEnv([TinyEnv])
        try:
            callbacks, _, _ = _build_core_callbacks(
                {"CheckpointCallback": CheckpointCallback},
                eval_env,
                tmp_path / "models",
                tmp_path / "logs",
                1,
                n_envs,
                100,
                save_freq,
                0,
                stage_config,
            )
        finally:
            eval_env.close()
        return callbacks

    def _retention(self, callbacks):
        from environments.shared.curriculum import CheckpointRetentionCallback

        return next(cb for cb in callbacks if isinstance(cb, CheckpointRetentionCallback))

    def test_it_is_installed_with_the_default_cap(self, tmp_path):
        from environments.shared.curriculum import DEFAULT_MAX_CHECKPOINTS

        retention = self._retention(self._build(tmp_path, {"curriculum_kwargs": {"min_avg_reward": 1.0}}))
        assert retention.max_checkpoints == DEFAULT_MAX_CHECKPOINTS
        assert retention.name_prefix == "stage1"

    def test_it_runs_after_the_checkpoint_callback(self, tmp_path):
        # Otherwise it would prune before the checkpoint that displaced one
        # had been written, on the step they share.
        pytest.importorskip("stable_baselines3")
        from stable_baselines3.common.callbacks import CheckpointCallback as _CC

        from environments.shared.curriculum import CheckpointRetentionCallback

        callbacks = self._build(tmp_path, {"curriculum_kwargs": {"min_avg_reward": 1.0}})
        kinds = [type(cb) for cb in callbacks]
        assert kinds.index(_CC) < kinds.index(CheckpointRetentionCallback)

    def test_it_shares_the_checkpoint_callbacks_cadence(self, tmp_path):
        # Both must fire on the same n_calls, or pruning happens on steps
        # where nothing was written -- a Drive glob for no reason.
        pytest.importorskip("stable_baselines3")
        from stable_baselines3.common.callbacks import CheckpointCallback as _CC

        callbacks = self._build(tmp_path, {"curriculum_kwargs": {"min_avg_reward": 1.0}}, n_envs=4, save_freq=100)
        checkpoint = next(cb for cb in callbacks if isinstance(cb, _CC))
        assert self._retention(callbacks).save_freq == checkpoint.save_freq

    def test_the_config_knob_reaches_it(self, tmp_path):
        retention = self._retention(
            self._build(tmp_path, {"curriculum_kwargs": {"min_avg_reward": 1.0, "max_checkpoints": 2}})
        )
        assert retention.max_checkpoints == 2

    def test_zero_disables_retention_without_disabling_checkpointing(self, tmp_path):
        pytest.importorskip("stable_baselines3")
        from stable_baselines3.common.callbacks import CheckpointCallback as _CC

        callbacks = self._build(tmp_path, {"curriculum_kwargs": {"min_avg_reward": 1.0, "max_checkpoints": 0}})
        assert self._retention(callbacks).max_checkpoints == 0
        assert any(isinstance(cb, _CC) for cb in callbacks)

    def test_more_envs_than_save_freq_does_not_divide_to_zero(self, tmp_path):
        # save_freq // n_envs used to reach CheckpointCallback as 0, and SB3
        # then evaluates `n_calls % 0` -> ZeroDivisionError on the first step.
        pytest.importorskip("stable_baselines3")
        from stable_baselines3.common.callbacks import CheckpointCallback as _CC

        callbacks = self._build(tmp_path, {"curriculum_kwargs": {"min_avg_reward": 1.0}}, n_envs=64, save_freq=8)
        checkpoint = next(cb for cb in callbacks if isinstance(cb, _CC))
        assert checkpoint.save_freq >= 1
        assert self._retention(callbacks).save_freq >= 1


class TestPrepareAlgKwargsWarnsOnAnEmptyTable:
    """Review CF4: the first point that knows the backend says when its table is empty."""

    def test_an_empty_ppo_table_warns(self, tmp_path, caplog):
        from environments.shared.train_base import _prepare_alg_kwargs

        with caplog.at_level(logging.WARNING):
            _prepare_alg_kwargs({"ppo_kwargs": {}, "sac_kwargs": {"learning_rate": 1e-3}}, "ppo", 0, tmp_path, False)
        assert any("no [ppo] table" in record.message for record in caplog.records)

    def test_a_populated_table_is_quiet(self, tmp_path, caplog):
        from environments.shared.train_base import _prepare_alg_kwargs

        with caplog.at_level(logging.WARNING):
            _prepare_alg_kwargs({"ppo_kwargs": {"learning_rate": 1e-3}, "sac_kwargs": {}}, "ppo", 0, tmp_path, False)
        assert not any("table" in record.message for record in caplog.records)


class TestTrainCurriculumWalksTheManifest:
    """Review TC7/OP6: the curriculum iterates the manifest, not ``range(1, 4)``.

    Indexed into ``load_all_stages``' ``[1, "recovery", 2, 3]`` the range
    skipped trex's recovery stage without a log line.  Every heavyweight
    collaborator is replaced so the loop's ORDER, its skip, and its handoff
    are the only things exercised.
    """

    def _run(
        self,
        species,
        tmp_path,
        monkeypatch,
        caplog,
        *,
        learn_side_effect=None,
        trunk_from=None,
        find_ancestor=None,
        retrain_from=None,
        output_dir=None,
        label=None,
        use_wandb=False,
        task_sha256=None,
        plant=None,
        target=None,
    ):
        """*task_sha256* and *plant*, when given, are what every node derives as its task digest and
        what ``current_plant_identity`` answers, so the REAL reuse rule can run against a real trunk
        (a test that leaves ``find_ancestor`` None); otherwise both are inert stand-ins.  *target* is
        forwarded as ``target=`` (decision D-A24).  The REAL ``CurriculumManager`` is used and every
        instance is kept under ``record["managers"]`` so a test can pin where the walk left it."""
        from environments.shared import ancestors as ancestors_module
        from environments.shared import config as config_module
        from environments.shared import curriculum as curriculum_module
        from environments.shared import plant_contract, result_bundle, task_fingerprint, train_base, wandb_integration
        from environments.shared.config import load_all_stages
        from environments.shared.stage_manifest import stage_label

        record: dict = {
            "saved": [],
            "loads": [],
            "parents": [],
            "verdicts": [],
            "parent_run_ids": [],
            "ancestors": [],
            "labels": [],
            "wandb_tags": [],
            "managers": [],
        }
        model = MagicMock()
        model.num_timesteps = 10
        model.learn.side_effect = learn_side_effect

        def write_verdict(stage_dir, **kwargs):
            # D-A22: every verdict the loop writes names the gate it judged
            # under; the recorder digests it the way the real writer does.
            from environments.shared.curriculum.gate_schema import gate_config_sha256

            assert set(kwargs["gate_config"]) == {"gate_kind", "gate_schema_version", "thresholds"}
            record["verdicts"].append(
                {"stage_dir": stage_dir, "gate_sha256": gate_config_sha256(kwargs["gate_config"]), **kwargs}
            )
            return stage_dir / "gate_verdict.json"

        # train_curriculum imports the writer at call time from the package,
        # so the recorder sees every verdict the loop decides to record.
        monkeypatch.setattr(result_bundle, "write_gate_verdict", write_verdict)

        # The reuse rule and the record writer are imported the same way; a
        # test that passes trunk_from supplies the rule's answer per node.
        if find_ancestor is not None:
            monkeypatch.setattr(ancestors_module, "find_certified_ancestor", find_ancestor)

        def record_ancestor(run_dir, ancestor):
            record["ancestors"].append((Path(run_dir), ancestor))
            return Path(run_dir) / "ancestors" / ancestor.stage_id

        monkeypatch.setattr(ancestors_module, "record_ancestor", record_ancestor)

        def create_or_load(sb3, algorithm, alg_kwargs, train_env, load_path, **kwargs):
            record["loads"].append(load_path)
            return model

        def save_config(stage_dir, stage, config, algorithm, **kwargs):
            record["saved"].append((stage, kwargs.get("load_path"), kwargs.get("load_mode")))
            record["parent_run_ids"].append((stage, kwargs.get("parent_run_id")))
            record["labels"].append((stage, kwargs.get("label")))

        def init_wandb(**kwargs):
            # Records the tags and reports W&B as unavailable (None), which
            # is the path the loop takes on a headless worker.
            record["wandb_tags"].append((kwargs["stage"], kwargs.get("tags")))
            return None

        def shaping(config, **kwargs):
            record["parents"].append(kwargs["parent_id"])
            return []

        monkeypatch.setattr(train_base, "_ensure_sb3", lambda: {"CallbackList": list})
        monkeypatch.setattr(
            train_base,
            "current_plant_identity",
            lambda species: SimpleNamespace(to_dict=dict) if plant is None else plant,
        )
        monkeypatch.setattr(train_base, "create_vec_env", lambda *args, **kwargs: MagicMock())
        monkeypatch.setattr(train_base, "_load_vecnorm_into_envs", lambda *args, **kwargs: None)
        monkeypatch.setattr(train_base, "_create_or_load_model", create_or_load)
        monkeypatch.setattr(
            train_base, "_build_core_callbacks", lambda *args, **kwargs: ([], MagicMock(best_mean_reward=1.0), None)
        )
        monkeypatch.setattr(train_base, "_maybe_ent_coef_decay_callback", lambda *args, **kwargs: None)
        monkeypatch.setattr(train_base, "_stage_entry_shaping_callbacks", shaping)
        monkeypatch.setattr(
            train_base,
            "_save_final_and_sync_tb",
            lambda model, train_env, model_dir, stage, *rest: model_dir / f"{stage_label(stage)}_final",
        )
        monkeypatch.setattr(train_base, "_select_handoff_checkpoint", lambda model_dir: None)
        monkeypatch.setattr(train_base, "_record_stage_result", lambda *args, **kwargs: None)
        monkeypatch.setattr(config_module, "save_stage_config", save_config)
        monkeypatch.setattr(config_module, "upload_curriculum_artifacts", lambda *args, **kwargs: None)
        monkeypatch.setattr(wandb_integration, "init_wandb", init_wandb)
        monkeypatch.setattr(curriculum_module, "CurriculumCallback", lambda **kwargs: MagicMock(ready_to_advance=True))
        real_manager = curriculum_module.CurriculumManager

        def make_manager(**kwargs):
            manager = real_manager(**kwargs)
            record["managers"].append(manager)
            return manager

        monkeypatch.setattr(curriculum_module, "CurriculumManager", make_manager)
        monkeypatch.setattr(
            task_fingerprint,
            "derive_stage_task_fingerprint",
            lambda **kwargs: {} if task_sha256 is None else {"task_sha256": task_sha256},
        )
        monkeypatch.setattr(plant_contract, "write_plant_identity", lambda path, identity: None)

        species_cfg = SimpleNamespace(species=species, env_class=object)
        with caplog.at_level(logging.INFO):
            train_base.train_curriculum(
                species_cfg,
                load_all_stages(species),
                n_envs=1,
                seed=1,
                verbose=0,
                use_tensorboard=False,
                output_dir=str(tmp_path if output_dir is None else output_dir),
                trunk_from=trunk_from,
                retrain_from=retrain_from,
                label=label,
                use_wandb=use_wandb,
                target=target,
            )
        return record

    @staticmethod
    def _certified_ancestor(
        trunk: Path,
        stage_id: str = "stance",
        stage_key: str = "1",
        model_sha256: str = "a" * 64,
        *,
        species: str = "velociraptor",
        edits: "dict[str, dict] | None" = None,
        record_config: bool = True,
    ):
        """A certified ancestor under *trunk* whose stage directory records a REAL ``stage_config.json``
        — the species' own config for the stage, with *edits* (``{"ppo_kwargs": {...}, ...}``) overlaid so
        a test can make the ancestor's recorded recipe differ from this run's.  ``record_config=False``
        leaves the directory without one.  Called before ``_run`` so the real writer is used."""
        from environments.shared.ancestors import CertifiedAncestor
        from environments.shared.config import load_all_stages, save_stage_config
        from environments.shared.curriculum.gate_schema import gate_config_sha256, gate_config_view

        stage_dir = trunk / f"01_{stage_id}"
        (stage_dir / "models").mkdir(parents=True, exist_ok=True)
        stage_ref = int(stage_key) if stage_key.isdigit() else stage_key
        recorded = dict(load_all_stages(species)[stage_ref])
        if record_config:
            for section, changes in (edits or {}).items():
                recorded[section] = {**recorded.get(section, {}), **changes}
            save_stage_config(stage_dir, stage_ref, recorded, "PPO", species=species)
        return CertifiedAncestor(
            stage_id=stage_id,
            stage_key=stage_key,
            run_id="trunk-run-id",
            source_run_dir=trunk,
            stage_dir=stage_dir,
            handoff_name="best_model",
            model_stem=str(stage_dir / "models" / "best_model"),
            model_zip=stage_dir / "models" / "best_model.zip",
            model_sha256="sha256:" + model_sha256,
            normalization_path=stage_dir / "models" / "best_model_vecnorm.pkl",
            normalization_sha256="sha256:" + "b" * 64,
            task_sha256="sha256:" + "c" * 64,
            judged_by="test",
            verdict={"passed": True},
            gate_sha256=gate_config_sha256(gate_config_view(recorded.get("curriculum_kwargs", {}))),
        )

    def test_trunk_from_reuses_a_certified_ancestor_and_trains_the_rest(self, tmp_path, monkeypatch, caplog):
        """BEHAVIOR_RECIPES_PLAN §4.2: a node satisfied by an earlier run's certified checkpoint is
        recorded, not trained; its child enters on that handoff with the trunk's run id as lineage;
        a node the rule refuses is trained here with the refusal logged, and its children are then
        trained here without consulting the trunk (nothing there descends from a checkpoint this run
        produced)."""
        from environments.shared.ancestors import AncestorReuseError
        from environments.shared.config import load_all_stages
        from environments.shared.curriculum.gate_schema import gate_config_sha256, gate_config_view

        trunk = tmp_path / "trunk-run"
        ancestor = self._certified_ancestor(trunk)
        asked: list[tuple[str, str | None]] = []

        def find_ancestor(
            run_dir,
            *,
            species,
            entry,
            current_task_sha256,
            plant_identity,
            current_gate_config,
            parent_model_sha256,
            follow_records,
        ):
            asked.append((entry.id, parent_model_sha256))
            assert Path(run_dir) == trunk and species == "velociraptor"
            # D-A22 (rule 7): the rule is handed the node's CURRENT [curriculum]
            # block, so it can refuse a verdict judged under another gate.
            assert current_gate_config == load_all_stages(species)[entry.reference]["curriculum_kwargs"]
            # The trunk is another run by construction, so the curriculum opts
            # in to following its ancestor records (D-A23); the notebook's
            # same-run candidate must not, which is why the library defaults off.
            assert follow_records is True
            if entry.id == "stance":
                return ancestor
            raise AncestorReuseError(f"{entry.id}: no gate_verdict.json in the trunk")

        record = self._run("velociraptor", tmp_path, monkeypatch, caplog, trunk_from=trunk, find_ancestor=find_ancestor)

        # The rule is handed the resolved parent's digest so it can check the
        # chain (rule 4): None for the root, the reused stance's for walk.
        assert asked == [("stance", None), ("locomotion", ancestor.model_sha256)]
        # Stance was reused, never trained; walk and hunt were trained here.
        assert [stage for stage, _, _ in record["saved"]] == [2, 3]
        assert [v["stage_id"] for v in record["verdicts"]] == ["locomotion", "behavior"]
        # D-A22: each verdict digests the block the node was judged under.
        assert [v["gate_sha256"] for v in record["verdicts"]] == [
            gate_config_sha256(gate_config_view(load_all_stages("velociraptor")[stage]["curriculum_kwargs"]))
            for stage in (2, 3)
        ]
        # Walk entered on the ancestor's handoff, crossing the edge, with the
        # trunk's run id recorded as lineage; hunt's parent was trained here.
        assert record["loads"][0] == ancestor.model_stem
        assert record["saved"][0] == (2, ancestor.model_stem, "initialize_next_stage")
        assert record["parent_run_ids"] == [(2, "trunk-run-id"), (3, None)]
        # The reuse left a record in this run, never a checkpoint copy.
        assert [(run_dir, a.stage_id) for run_dir, a in record["ancestors"]] == [(tmp_path, "stance")]
        not_reused = [r for r in caplog.records if r.message.startswith("Not reusing")]
        assert [r.levelno for r in not_reused] == [logging.WARNING, logging.INFO]
        assert "'locomotion'" in not_reused[0].message and "no gate_verdict.json" in not_reused[0].message
        assert "'behavior'" in not_reused[1].message and "it is this run's target" in not_reused[1].message

    def test_a_child_of_a_node_trained_here_is_not_looked_up(self, tmp_path, monkeypatch, caplog):
        """Once a node is trained in this run the trunk is not consulted for its children: no earlier
        run's checkpoint descends from a checkpoint this run just produced, so the chain (rule 4)
        could never hold.  Every node is still accounted for in the log."""
        from environments.shared.ancestors import AncestorReuseError

        trunk = tmp_path / "trunk-run"
        asked: list[str] = []

        def find_ancestor(run_dir, *, entry, **kwargs):
            asked.append(entry.id)
            raise AncestorReuseError("stance: gate_verdict.json records a FAILED gate")

        record = self._run("velociraptor", tmp_path, monkeypatch, caplog, trunk_from=trunk, find_ancestor=find_ancestor)

        assert asked == ["stance"]
        assert [stage for stage, _, _ in record["saved"]] == [1, 2, 3]
        assert record["parent_run_ids"] == [(1, None), (2, None), (3, None)]
        assert record["ancestors"] == []
        not_reused = [r for r in caplog.records if r.message.startswith("Not reusing")]
        assert [r.levelno for r in not_reused] == [logging.WARNING, logging.INFO, logging.INFO]
        assert "'stance'" in not_reused[0].message and "FAILED gate" in not_reused[0].message
        assert (
            "'locomotion'" in not_reused[1].message
            and "parent 'stance' was trained in this run" in not_reused[1].message
        )
        assert "'behavior'" in not_reused[2].message and "it is this run's target" in not_reused[2].message

    def test_trunk_from_reuses_the_whole_chain_but_never_the_target(self, tmp_path, monkeypatch, caplog):
        """A run's target is what it exists to certify: with stance and walk both certified in the
        trunk, hunt is still trained here, on the reused walk, with the trunk as its lineage.  An
        earlier run's certified hunt is that run's deliverable, published from there."""
        trunk = tmp_path / "trunk-run"
        stance = self._certified_ancestor(trunk, "stance", "1", model_sha256="a" * 64)
        locomotion = self._certified_ancestor(trunk, "locomotion", "2", model_sha256="b" * 64)
        asked: list[tuple[str, str | None]] = []

        def find_ancestor(run_dir, *, entry, parent_model_sha256, **kwargs):
            asked.append((entry.id, parent_model_sha256))
            return {"stance": stance, "locomotion": locomotion}[entry.id]

        record = self._run("velociraptor", tmp_path, monkeypatch, caplog, trunk_from=trunk, find_ancestor=find_ancestor)

        # Root-first, each child checked against the digest its parent resolved to.
        assert asked == [("stance", None), ("locomotion", stance.model_sha256)]
        assert [stage for stage, _, _ in record["saved"]] == [3]
        assert [v["stage_id"] for v in record["verdicts"]] == ["behavior"]
        assert record["loads"] == [locomotion.model_stem]
        assert record["saved"] == [(3, locomotion.model_stem, "initialize_next_stage")]
        assert record["parent_run_ids"] == [(3, "trunk-run-id")]
        assert [a.stage_id for _, a in record["ancestors"]] == ["stance", "locomotion"]
        target = [r for r in caplog.records if r.message.startswith("Not reusing 'behavior'")]
        assert len(target) == 1 and target[0].levelno == logging.INFO
        assert "it is this run's target" in target[0].message

    def test_trunk_from_a_middle_run_reuses_stance_from_the_original_run(self, tmp_path, monkeypatch, caplog):
        """Decision D-A23: a run that itself reused stance holds only ``ancestors/stance/``, and the real
        rule, which the curriculum asks to follow records (``follow_records=True`` — the trunk is
        another run by construction), follows that record to the run that certified stance.  The curriculum passes the middle
        run and receives the ORIGINAL: it is what is recorded as this run's ancestor and what walk's
        ``parent_run_id`` names; walk, which the middle run neither trained nor reused, is trained
        here, and hunt is the target."""
        from environments.shared.ancestors import find_certified_ancestor, record_ancestor
        from environments.shared.config import load_all_stages
        from environments.shared.stage_manifest import load_stage_manifest

        from .reporting_helpers import make_plant_identity
        from .test_ancestors import STANCE_TASK, build_trunk_run

        plant = make_plant_identity()
        original = tmp_path / "logs" / "original-run"
        original.mkdir(parents=True)
        # The trunk's stance was judged under the checkout's own velociraptor
        # stance block, so rule 7 (D-A22) matches the gate this run declares.
        stance_curriculum = load_all_stages("velociraptor")[1]["curriculum_kwargs"]
        stance_dir = build_trunk_run(original, species="velociraptor", plant=plant, curriculum=stance_curriculum)
        middle = tmp_path / "logs" / "middle-run"
        middle.mkdir()
        record_ancestor(
            middle,
            find_certified_ancestor(
                original,
                species="velociraptor",
                entry=load_stage_manifest("velociraptor").by_id("stance"),
                current_task_sha256=STANCE_TASK,
                plant_identity=plant,
                current_gate_config=stance_curriculum,
            ),
        )
        assert [child.name for child in middle.iterdir()] == ["ancestors"]

        record = self._run(
            "velociraptor",
            tmp_path,
            monkeypatch,
            caplog,
            trunk_from=middle,
            output_dir=tmp_path / "logs" / "child-run",
            task_sha256=STANCE_TASK,
            plant=plant,
        )

        assert [stage for stage, _, _ in record["saved"]] == [2, 3]
        assert record["parent_run_ids"] == [(2, "original-run"), (3, None)]
        assert record["loads"][0] == str(stance_dir / "models" / "robust_best_model")
        [(child_dir, ancestor)] = record["ancestors"]
        assert child_dir == tmp_path / "logs" / "child-run" and ancestor.stage_id == "stance"
        assert ancestor.run_id == "original-run" and ancestor.source_run_dir == original
        assert ancestor.stage_dir == stance_dir and ancestor.via == (middle,)
        followed = [r for r in caplog.records if r.message.startswith("Following the ancestor record")]
        assert len(followed) == 1 and str(middle) in followed[0].message and str(original) in followed[0].message
        # The D-A21 ignored-edit warning (the fixture records no algorithm
        # block) and the reuse line both name the ORIGINAL run, never the middle.
        reused = [r for r in caplog.records if r.message.startswith("Reusing certified 'stance' from run original-run")]
        assert [r.levelno for r in reused] == [logging.WARNING, logging.INFO]
        assert not [r for r in caplog.records if "from run middle-run" in r.message]
        not_reused = [r for r in caplog.records if r.message.startswith("Not reusing")]
        assert [r.levelno for r in not_reused] == [logging.WARNING, logging.INFO]
        assert (
            "'locomotion'" in not_reused[0].message and "no stage directory for 'locomotion'" in not_reused[0].message
        )
        assert "ancestors/locomotion/ancestor.json to follow" in not_reused[0].message
        assert "'behavior'" in not_reused[1].message and "it is this run's target" in not_reused[1].message

    def test_a_reused_child_needs_its_parent_resolved_first(self, tmp_path, monkeypatch, caplog):
        """The parent check precedes reuse: with locomotion's edge pointing at the never-certified
        recovery node, the trunk is not even consulted for locomotion — a certified walk from
        anywhere is worth nothing on top of a parent this run does not have."""
        from environments.shared.ancestors import AncestorReuseError

        trunk = tmp_path / "trunk-run"
        asked: list[str] = []

        def find_ancestor(run_dir, *, entry, **kwargs):
            asked.append(entry.id)
            raise AncestorReuseError("nothing certified here")

        record = self._run_with_recovery_edge(
            tmp_path, monkeypatch, caplog, trunk_from=trunk, find_ancestor=find_ancestor
        )

        assert asked == ["stance"]
        assert [stage for stage, _, _ in record["saved"]] == [1]
        skipped = [r for r in caplog.records if "Skipping 'locomotion'" in r.message]
        assert len(skipped) == 1 and "declared parent 'recovery' has no certified checkpoint" in skipped[0].message

    def _run_with_recovery_edge(self, tmp_path, monkeypatch, caplog, **run_kwargs):
        """The trex curriculum with locomotion's edge retargeted at the non-advancing recovery node."""
        import shutil

        from environments.shared import config as config_module
        from environments.shared import stage_manifest

        configs = tmp_path / "configs"
        shutil.copytree(stage_manifest._CONFIGS_DIR / "trex", configs / "trex")
        manifest_path = configs / "trex" / "stages.toml"
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
        marker = 'warm_start_from = "stance"       # the 2026-08-23 lineage rule'
        (index,) = [i for i, line in enumerate(lines) if line.startswith(marker)]
        lines[index] = 'warm_start_from = "recovery"'
        manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        monkeypatch.setattr(stage_manifest, "_CONFIGS_DIR", configs)
        monkeypatch.setattr(config_module, "_CONFIGS_DIR", configs)
        return self._run("trex", tmp_path, monkeypatch, caplog, **run_kwargs)

    def test_a_node_whose_parent_has_no_certified_checkpoint_stops_the_curriculum(self, tmp_path, monkeypatch, caplog):
        """Nothing is trained from scratch silently: with locomotion's edge pointing at the
        non-advancing recovery node (plan A2), the CLI curriculum stops after stance and says why."""
        record = self._run_with_recovery_edge(tmp_path, monkeypatch, caplog)

        assert [stage for stage, _, _ in record["saved"]] == [1]
        assert [v["stage_id"] for v in record["verdicts"]] == ["stance"]
        skipped = [r for r in caplog.records if "Skipping 'locomotion'" in r.message]
        assert len(skipped) == 1 and skipped[0].levelno == logging.WARNING
        assert "declared parent 'recovery' has no certified checkpoint" in skipped[0].message
        assert record["loads"] == [None]

    def test_trex_visits_the_advancing_stages_in_manifest_order_and_logs_the_skip(self, tmp_path, monkeypatch, caplog):
        from environments.shared.stage_manifest import load_stage_manifest

        record = self._run("trex", tmp_path, monkeypatch, caplog)

        manifest = load_stage_manifest("trex")
        assert [stage for stage, _, _ in record["saved"]] == [entry.reference for entry in manifest.advancing_stages]
        assert [stage for stage, _, _ in record["saved"]] == [1, 2, 3]
        skips = [r for r in caplog.records if "Skipping non-advancing stage 'recovery'" in r.message]
        assert len(skips) == 1 and skips[0].levelno == logging.WARNING
        assert "train --stage recovery" in skips[0].message
        # Shaping is keyed on the node's declared EDGE (BEHAVIOR_RECIPES_PLAN
        # §4.2), never on its position: stance is a root, locomotion (legacy
        # 2, position 3) enters from stance and behavior from locomotion.
        assert record["parents"] == [None, "stance", "locomotion"]

    def test_the_handoff_skips_over_the_non_advancing_stage(self, tmp_path, monkeypatch, caplog):
        record = self._run("trex", tmp_path, monkeypatch, caplog)

        stance_final = str(tmp_path / "01_stance" / "models" / "stage1_final")
        locomotion_final = str(tmp_path / "03_locomotion" / "models" / "stage2_final")
        assert record["loads"] == [None, stance_final, locomotion_final]
        # And the lineage the next stage's config records is that same handoff.
        assert record["saved"] == [
            (1, None, None),
            (2, stance_final, "initialize_next_stage"),
            (3, locomotion_final, "initialize_next_stage"),
        ]

    def test_a_manifest_less_species_is_unchanged(self, tmp_path, monkeypatch, caplog):
        record = self._run("velociraptor", tmp_path, monkeypatch, caplog)

        assert [stage for stage, _, _ in record["saved"]] == [1, 2, 3]
        # The v2 file declares the edges the synthesizer used to derive.
        assert record["parents"] == [None, "stance", "locomotion"]
        assert not [r for r in caplog.records if "Skipping non-advancing stage" in r.message]

    def test_every_trained_node_records_the_managers_verdict(self, tmp_path, monkeypatch, caplog):
        """Decision D-A5: the in-training verdict is written per node, hash-bound to its handoff."""
        from environments.shared.train_base import CURRICULUM_MANAGER_JUDGED_BY

        record = self._run("velociraptor", tmp_path, monkeypatch, caplog)

        assert [v["stage_id"] for v in record["verdicts"]] == ["stance", "locomotion", "behavior"]
        for verdict in record["verdicts"]:
            assert verdict["passed"] is True and verdict["failures"] == []
            assert verdict["judged_by"] == CURRICULUM_MANAGER_JUDGED_BY
            assert verdict["stage_dir"].name in {"01_stance", "02_locomotion", "03_behavior"}
            assert verdict["checkpoint"].suffix == ".zip" and verdict["normalization"].suffix == ".pkl"

    def test_an_interrupted_node_records_no_verdict_and_stops_the_curriculum(self, tmp_path, monkeypatch, caplog):
        """A Ctrl-C partway through a budget is not a gate failure: no verdict, no handoff, loop stops."""
        record = self._run("velociraptor", tmp_path, monkeypatch, caplog, learn_side_effect=KeyboardInterrupt)

        assert [stage for stage, _, _ in record["saved"]] == [1]
        assert record["verdicts"] == []
        assert [r for r in caplog.records if "no gate verdict recorded" in r.message]

    def test_retrain_from_covers_the_node_and_its_descendants_but_not_its_ancestors(
        self, tmp_path, monkeypatch, caplog
    ):
        """Decision D-A19: ``--retrain-from locomotion`` against a trunk certifying stance AND
        locomotion reuses stance (strictly above the named node), consults the trunk for nothing
        else, and trains locomotion and behavior here — locomotion on the reused stance with the
        trunk's run id as lineage, behavior on the locomotion trained here.  The retrain line is
        the INFO reason for both covered nodes (checked before the target and parent-trained-here
        branches)."""
        trunk = tmp_path / "trunk-run"
        stance = self._certified_ancestor(trunk, "stance", "1", model_sha256="a" * 64)
        locomotion = self._certified_ancestor(trunk, "locomotion", "2", model_sha256="b" * 64)
        asked: list[tuple[str, str | None]] = []

        def find_ancestor(run_dir, *, entry, parent_model_sha256, **kwargs):
            asked.append((entry.id, parent_model_sha256))
            return {"stance": stance, "locomotion": locomotion}[entry.id]

        record = self._run(
            "velociraptor",
            tmp_path,
            monkeypatch,
            caplog,
            trunk_from=trunk,
            find_ancestor=find_ancestor,
            retrain_from="locomotion",
        )

        assert asked == [("stance", None)]
        assert [stage for stage, _, _ in record["saved"]] == [2, 3]
        assert [v["stage_id"] for v in record["verdicts"]] == ["locomotion", "behavior"]
        assert record["saved"][0] == (2, stance.model_stem, "initialize_next_stage")
        assert record["parent_run_ids"] == [(2, "trunk-run-id"), (3, None)]
        assert [a.stage_id for _, a in record["ancestors"]] == ["stance"]
        not_reused = [r for r in caplog.records if r.message.startswith("Not reusing")]
        assert [(r.levelno, r.message) for r in not_reused] == [
            (
                logging.INFO,
                f"Not reusing 'locomotion' from --trunk-from {trunk}: --retrain-from 'locomotion' covers it. "
                "Training it here.",
            ),
            (
                logging.INFO,
                f"Not reusing 'behavior' from --trunk-from {trunk}: --retrain-from 'locomotion' covers it. "
                "Training it here.",
            ),
        ]

    def test_retrain_from_accepts_a_legacy_number_and_naming_the_root_reuses_nothing(
        self, tmp_path, monkeypatch, caplog
    ):
        """The knob resolves through the manifest like ``--stage``: legacy 1 is stance, the root,
        so nothing above it exists to reuse and the trunk is never consulted."""
        trunk = tmp_path / "trunk-run"
        asked: list[str] = []

        def find_ancestor(run_dir, *, entry, **kwargs):
            asked.append(entry.id)
            raise AssertionError("the trunk must not be consulted for a covered node")

        record = self._run(
            "velociraptor", tmp_path, monkeypatch, caplog, trunk_from=trunk, find_ancestor=find_ancestor, retrain_from=1
        )

        assert asked == []
        assert [stage for stage, _, _ in record["saved"]] == [1, 2, 3]
        assert record["parent_run_ids"] == [(1, None), (2, None), (3, None)]
        assert record["ancestors"] == []
        covered = [r for r in caplog.records if "--retrain-from 'stance' covers it" in r.message]
        assert [r.levelno for r in covered] == [logging.INFO] * 3

    @pytest.mark.parametrize(
        ("retrain_from", "match"),
        [
            ("recovery", "non-advancing stage 'recovery'"),
            ("no_such_node", "not a stage of trex"),
            (7, "not a stage of trex"),
        ],
    )
    def test_an_unknown_or_non_advancing_retrain_from_raises_before_any_directory_is_written(
        self, tmp_path, monkeypatch, caplog, retrain_from, match
    ):
        """A retrain-from that the curriculum cannot walk (trex's non-advancing recovery, an unknown
        id, an unknown legacy number) is a ValueError naming the advancing ids, raised before the run
        directory exists — a typo leaves no footprint."""
        run_dir = tmp_path / "fresh-run"

        with pytest.raises(ValueError, match=match) as excinfo:
            self._run(
                "trex",
                tmp_path,
                monkeypatch,
                caplog,
                trunk_from=tmp_path / "trunk",
                retrain_from=retrain_from,
                output_dir=run_dir,
            )

        assert "['stance', 'locomotion', 'behavior']" in str(excinfo.value)
        assert not run_dir.exists()

    # -- decision D-A24: ``target`` walks the target's chain -----------------

    @staticmethod
    def _walk(record, base_dir):
        """The walk a record describes, with paths relative to *base_dir* so two runs compare."""
        return {
            "saved": [
                (stage, load and str(Path(load).relative_to(base_dir)), mode) for stage, load, mode in record["saved"]
            ],
            "loads": [load and str(Path(load).relative_to(base_dir)) for load in record["loads"]],
            "parents": record["parents"],
            "verdicts": [(v["stage_id"], v["stage_dir"].name, v["passed"]) for v in record["verdicts"]],
            "parent_run_ids": record["parent_run_ids"],
            "ancestors": [(run_dir.name, a.stage_id) for run_dir, a in record["ancestors"]],
        }

    def test_target_walk_trains_stance_and_locomotion_only(self, tmp_path, monkeypatch, caplog):
        """Decision D-A24: ``target="walk"`` walks walk's chain — stance, locomotion — and stops:
        both are trained and judged, no third stage directory exists, behavior is skipped with a
        line naming the target, and the banner names the chain.  The manager is still keyed over
        the FULL advancing ladder (3 stages) and is advanced once per node that passed, never past
        the target, so a walk-only run leaves it at stage 2 mid-ladder: ``is_final_stage`` False,
        one more ``advance()`` legal (to 3), a second one a RuntimeError — the semantics the loop
        relies on, pinned here."""
        record = self._run("velociraptor", tmp_path, monkeypatch, caplog, target="walk")

        assert [stage for stage, _, _ in record["saved"]] == [1, 2]
        assert record["loads"] == [None, str(tmp_path / "01_stance" / "models" / "stage1_final")]
        assert record["parents"] == [None, "stance"]
        assert [v["stage_id"] for v in record["verdicts"]] == ["stance", "locomotion"]
        assert all(v["passed"] for v in record["verdicts"])
        assert sorted(p.name for p in tmp_path.iterdir() if p.is_dir()) == ["01_stance", "02_locomotion"]
        banner = [r for r in caplog.records if r.message.startswith("Starting automated curriculum training")]
        assert len(banner) == 1 and banner[0].message.endswith(
            "(target 'locomotion', its chain in manifest order): stance -> locomotion"
        )
        skipped = [r for r in caplog.records if r.message.startswith("Skipping advancing stage 'behavior'")]
        assert len(skipped) == 1 and skipped[0].levelno == logging.INFO
        assert "target 'locomotion' (stance -> locomotion)" in skipped[0].message
        assert not [r for r in caplog.records if "Skipping non-advancing stage" in r.message]
        # The manager: full ladder, advanced exactly once (stance -> 2), left mid-ladder.
        [manager] = record["managers"]
        assert manager.total_stages == 3 and manager.current_stage == 2 and not manager.is_final_stage
        assert [r.message for r in caplog.records if r.message.startswith("Auto-advanced")] == [
            "Auto-advanced to stage 2"
        ]
        assert manager.advance() == 3 and manager.is_final_stage
        with pytest.raises(RuntimeError, match="Cannot advance past final stage 3"):
            manager.advance()

    def test_target_walk_never_reuses_locomotion_from_a_trunk(self, tmp_path, monkeypatch, caplog):
        """The target-is-never-reused rule (D-A18) follows the target: with a trunk certifying stance
        AND locomotion, ``target="walk"`` reuses stance and trains locomotion here as the target,
        with the trunk as its lineage; the trunk is never consulted for locomotion or behavior."""
        trunk = tmp_path / "trunk-run"
        stance = self._certified_ancestor(trunk, "stance", "1", model_sha256="a" * 64)
        locomotion = self._certified_ancestor(trunk, "locomotion", "2", model_sha256="b" * 64)
        asked: list[tuple[str, str | None]] = []

        def find_ancestor(run_dir, *, entry, parent_model_sha256, **kwargs):
            asked.append((entry.id, parent_model_sha256))
            return {"stance": stance, "locomotion": locomotion}[entry.id]

        record = self._run(
            "velociraptor",
            tmp_path,
            monkeypatch,
            caplog,
            trunk_from=trunk,
            find_ancestor=find_ancestor,
            target="walk",
        )

        assert asked == [("stance", None)]
        assert record["saved"] == [(2, stance.model_stem, "initialize_next_stage")]
        assert [v["stage_id"] for v in record["verdicts"]] == ["locomotion"]
        assert record["parent_run_ids"] == [(2, "trunk-run-id")]
        assert [a.stage_id for _, a in record["ancestors"]] == ["stance"]
        not_reused = [r for r in caplog.records if r.message.startswith("Not reusing")]
        assert len(not_reused) == 1 and not_reused[0].levelno == logging.INFO
        assert not_reused[0].message.startswith("Not reusing 'locomotion'")
        assert "it is this run's target" in not_reused[0].message
        [manager] = record["managers"]
        assert manager.current_stage == 2

    def test_target_hunt_is_the_default_and_a_stage_id_or_legacy_number_resolves_like_its_label(
        self, tmp_path, monkeypatch, caplog
    ):
        """``target="hunt"`` (the label), ``"behavior"`` (the deliverable's id) and ``3`` (its legacy
        number) each walk exactly what the default walks; ``"locomotion"`` and ``2`` walk exactly what
        ``"walk"`` walks.  The record projections compare with run directories factored out."""
        walks = {}
        for name, target in (("default", None), ("hunt", "hunt"), ("behavior", "behavior"), ("three", 3)):
            run_dir = tmp_path / name
            record = self._run("velociraptor", tmp_path, monkeypatch, caplog, output_dir=run_dir, target=target)
            walks[name] = self._walk(record, run_dir)
        assert walks["hunt"] == walks["behavior"] == walks["three"] == walks["default"]
        assert [stage for stage, _, _ in walks["default"]["saved"]] == [1, 2, 3]
        banners = [r.message for r in caplog.records if r.message.startswith("Starting automated")]
        assert len(banners) == 4 and all(
            b.endswith("(target 'behavior', its chain in manifest order): stance -> locomotion -> behavior")
            for b in banners
        )
        assert not [r for r in caplog.records if r.message.startswith("Skipping advancing stage")]

        caplog.clear()
        for name, target in (("walk", "walk"), ("locomotion", "locomotion"), ("two", 2)):
            run_dir = tmp_path / name
            record = self._run("velociraptor", tmp_path, monkeypatch, caplog, output_dir=run_dir, target=target)
            walks[name] = self._walk(record, run_dir)
        assert walks["locomotion"] == walks["two"] == walks["walk"]
        assert [stage for stage, _, _ in walks["walk"]["saved"]] == [1, 2]
        assert len([r for r in caplog.records if r.message.startswith("Skipping advancing stage 'behavior'")]) == 3

    @pytest.mark.parametrize("target", ["stand", "recovery"])
    def test_a_target_whose_chain_runs_through_a_non_advancing_stage_raises_before_any_directory_is_written(
        self, tmp_path, monkeypatch, caplog, target
    ):
        """On T-Rex ``stand`` resolves to recovery (its deepest deliverable) and ``recovery`` names it
        outright; either chain is stance -> recovery, which the integer-keyed manager cannot judge.
        The ValueError names the offending node, the advancing ids and the notebook's BEHAVIOR knob,
        and the run directory does not exist afterwards."""
        run_dir = tmp_path / "fresh-run"

        with pytest.raises(ValueError) as excinfo:
            self._run("trex", tmp_path, monkeypatch, caplog, output_dir=run_dir, target=target)

        message = str(excinfo.value)
        assert message.startswith(
            f"target {target!r} resolves to 'recovery', whose chain ['stance', 'recovery'] runs through"
        )
        assert "the non-advancing stage(s) ['recovery']" in message
        assert "advancing stages ['stance', 'locomotion', 'behavior']" in message
        assert f"notebook's BEHAVIOR knob (BEHAVIOR = {target!r})" in message
        assert not run_dir.exists()

    def _run_with_behavior_edge_on_stance(self, tmp_path, monkeypatch, caplog, **run_kwargs):
        """The velociraptor curriculum with behavior's edge retargeted at stance, leaving locomotion —
        still advancing, still numbered 2 — off hunt's chain.  The loader accepts it: legacy numbers
        pin ids and order, not the edges between advancing nodes."""
        import shutil

        from environments.shared import config as config_module
        from environments.shared import stage_manifest

        configs = tmp_path / "configs"
        shutil.copytree(stage_manifest._CONFIGS_DIR / "velociraptor", configs / "velociraptor")
        manifest_path = configs / "velociraptor" / "stages.toml"
        text = manifest_path.read_text(encoding="utf-8")
        assert text.count('warm_start_from = "locomotion"') == 1
        manifest_path.write_text(text.replace('warm_start_from = "locomotion"', 'warm_start_from = "stance"'))
        monkeypatch.setattr(stage_manifest, "_CONFIGS_DIR", configs)
        monkeypatch.setattr(config_module, "_CONFIGS_DIR", configs)
        return self._run("velociraptor", tmp_path, monkeypatch, caplog, **run_kwargs)

    @pytest.mark.parametrize("target", ["hunt", "behavior", 3])
    def test_a_target_whose_chain_skips_a_ladder_stage_raises_before_any_directory_is_written(
        self, tmp_path, monkeypatch, caplog, target
    ):
        """An explicit target's chain must be a PREFIX of the advancing ladder, not merely all-advancing:
        with behavior's edge on stance, hunt's chain is stance -> behavior and the manager — advanced
        once after stance, so at stage 2 of 3 — would judge behavior against locomotion's thresholds.
        Refused naming the skipped stage, the ladder and the notebook, with no run directory."""
        run_dir = tmp_path / "fresh-run"

        with pytest.raises(ValueError) as excinfo:
            self._run_with_behavior_edge_on_stance(tmp_path, monkeypatch, caplog, output_dir=run_dir, target=target)

        message = str(excinfo.value)
        assert message.startswith(f"target {target!r} resolves to 'behavior', whose chain ['stance', 'behavior'] skips")
        assert (
            "the advancing stage(s) ['locomotion'] below it on the ladder ['stance', 'locomotion', 'behavior']"
            in message
        )
        assert "judge the node after the gap against the skipped stage's thresholds" in message
        expected_behavior = target if isinstance(target, str) else "behavior"
        assert f"notebook's BEHAVIOR knob (BEHAVIOR = {expected_behavior!r})" in message
        assert not run_dir.exists()

    def test_the_default_target_still_walks_the_whole_ladder_over_an_off_ladder_edge(
        self, tmp_path, monkeypatch, caplog
    ):
        """``target=None`` is the ladder itself and is not validated as a chain (the walk before D-A24,
        bit for bit): with behavior's edge on stance every ladder node is still walked in order and
        behavior warm-starts from stance's handoff along its declared edge."""
        record = self._run_with_behavior_edge_on_stance(tmp_path, monkeypatch, caplog)

        assert [stage for stage, _, _ in record["saved"]] == [1, 2, 3]
        assert record["parents"] == [None, "stance", "stance"]
        assert record["loads"][2] == str(tmp_path / "01_stance" / "models" / "stage1_final")

    @pytest.mark.parametrize(
        ("target", "match"),
        [
            (
                "fly",
                "target 'fly' does not name a behavior of velociraptor: velociraptor has no behavior 'fly'; recipe labels: \\['stand', 'walk', 'hunt'\\], deliverable ids: \\['stance', 'locomotion', 'behavior'\\]",
            ),
            (7, "target 7 does not name a behavior of velociraptor: velociraptor has no stage with legacy number 7"),
        ],
    )
    def test_an_unknown_target_raises_listing_the_labels(self, tmp_path, monkeypatch, caplog, target, match):
        """An unknown label lists the recipe labels and deliverable ids; an unknown legacy number
        says so; neither leaves a footprint."""
        run_dir = tmp_path / "fresh-run"
        with pytest.raises(ValueError, match=match):
            self._run("velociraptor", tmp_path, monkeypatch, caplog, output_dir=run_dir, target=target)
        assert not run_dir.exists()

    def test_retrain_from_outside_the_targets_chain_raises_before_any_directory_is_written(
        self, tmp_path, monkeypatch, caplog
    ):
        """``retrain_from="behavior"`` with ``target="walk"`` names an advancing node the run never
        walks: refused naming the chain, before the run directory exists."""
        run_dir = tmp_path / "fresh-run"
        with pytest.raises(ValueError) as excinfo:
            self._run(
                "velociraptor",
                tmp_path,
                monkeypatch,
                caplog,
                trunk_from=tmp_path / "trunk",
                retrain_from="behavior",
                target="walk",
                output_dir=run_dir,
            )
        assert str(excinfo.value) == (
            "retrain_from 'behavior' names 'behavior', which is not on the chain this run walks to its target "
            "'locomotion': ['stance', 'locomotion']; it must name one of those"
        )
        assert not run_dir.exists()

    def test_an_occupied_stage_directory_is_refused_before_it_is_written(self, tmp_path, monkeypatch, caplog):
        """Decision D-A20: the curriculum has no resume mode, so an ``output_dir`` whose stage
        directory already records a stage (here a judged ``02_locomotion``) is refused when the walk
        reaches it — after stance trained, before locomotion's config or models directory is
        written — and the message names the directory, the file and both ways forward."""
        from environments.shared.config import StageDirectoryOccupiedError

        occupied = tmp_path / "02_locomotion"
        occupied.mkdir()
        (occupied / "gate_verdict.json").write_text("{}")

        with pytest.raises(StageDirectoryOccupiedError, match="gate_verdict.json") as excinfo:
            self._run("velociraptor", tmp_path, monkeypatch, caplog)

        message = str(excinfo.value)
        assert str(occupied) in message and "--output-dir" in message and "resume_same_stage" in message
        assert not (occupied / "models").exists()
        assert not (occupied / "stage_config.json").exists()
        # Stance, whose directory was fresh, was walked (its models directory is the loop's own
        # mkdir); the refusal stopped the run there, before anything of locomotion's.
        assert (tmp_path / "01_stance" / "models").is_dir()
        assert not (tmp_path / "03_behavior").exists()

    def _reuse_stance(self, tmp_path, monkeypatch, caplog, ancestor):
        """Walk velociraptor with a trunk whose rule certifies stance only; returns the record."""
        from environments.shared.ancestors import AncestorReuseError

        def find_ancestor(run_dir, *, entry, **kwargs):
            if entry.id == "stance":
                return ancestor
            raise AncestorReuseError(f"{entry.id}: not in the trunk")

        return self._run(
            "velociraptor",
            tmp_path,
            monkeypatch,
            caplog,
            trunk_from=tmp_path / "trunk-run",
            find_ancestor=find_ancestor,
        )

    @staticmethod
    def _ignored_edit_warnings(caplog):
        return [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "ignores this run's hyperparameter edit" in r.message
        ]

    def test_reuse_warns_naming_the_keys_of_an_ignored_hyperparameter_edit(self, tmp_path, monkeypatch, caplog):
        """Decision D-A21: reuse carries the ancestor's recorded recipe.  When this run's algorithm
        block or shaping keys differ from the ancestor's ``stage_config.json`` the loop warns — naming
        the node, the ancestor's run, every differing dotted key, and ``--retrain-from`` as the way to
        train it here — and still reuses (an ignored edit is a warning, never a refusal)."""
        ancestor = self._certified_ancestor(
            tmp_path / "trunk-run",
            edits={"ppo_kwargs": {"learning_rate": 1e-5}, "curriculum_kwargs": {"warmup_timesteps": 1}},
        )

        record = self._reuse_stance(tmp_path, monkeypatch, caplog, ancestor)

        warnings = self._ignored_edit_warnings(caplog)
        assert len(warnings) == 1
        message = warnings[0].message
        assert "'stance'" in message and "trunk-run-id" in message
        assert "ppo.learning_rate, shaping.warmup_timesteps" in message
        assert "--retrain-from stance" in message
        # Reused all the same: stance was not trained, walk and hunt were.
        assert [stage for stage, _, _ in record["saved"]] == [2, 3]
        assert [a.stage_id for _, a in record["ancestors"]] == ["stance"]

    def test_reuse_is_silent_when_the_ancestor_recorded_the_same_recipe(self, tmp_path, monkeypatch, caplog):
        """The digest agrees when nothing in the algorithm block or shaping keys changed, so the
        warning stays silent — the common case must not nag."""
        ancestor = self._certified_ancestor(tmp_path / "trunk-run")

        record = self._reuse_stance(tmp_path, monkeypatch, caplog, ancestor)

        assert self._ignored_edit_warnings(caplog) == []
        assert [stage for stage, _, _ in record["saved"]] == [2, 3]

    def test_an_ancestor_without_a_readable_config_is_reported_not_trusted(self, tmp_path, monkeypatch, caplog):
        """An ancestor whose ``stage_config.json`` cannot be read is never assumed to match: the
        warning names ``<unreadable stage_config.json>`` in place of the keys."""
        ancestor = self._certified_ancestor(tmp_path / "trunk-run", record_config=False)

        self._reuse_stance(tmp_path, monkeypatch, caplog, ancestor)

        warnings = self._ignored_edit_warnings(caplog)
        assert len(warnings) == 1 and "<unreadable stage_config.json>" in warnings[0].message

    def test_the_label_reaches_every_trained_nodes_config_and_the_wandb_tags(self, tmp_path, monkeypatch, caplog):
        """Decision D-A21: ``label`` is handed to ``save_stage_config`` for every trained node and
        every W&B run is tagged ``hp:<12 hex of the node's digest>`` plus ``label:<label>``; without a
        label the config gets ``None`` (nothing recorded) and only the ``hp:`` tag is set."""
        from environments.shared.config import hyperparameters_sha256, load_all_stages

        configs = load_all_stages("velociraptor")
        expected_hp = {
            stage: "hp:" + hyperparameters_sha256(configs[stage], "ppo").removeprefix("sha256:")[:12]
            for stage in (1, 2, 3)
        }
        assert all(len(tag) == len("hp:") + 12 for tag in expected_hp.values())

        record = self._run("velociraptor", tmp_path, monkeypatch, caplog, label="lr-sweep-a", use_wandb=True)
        assert record["labels"] == [(1, "lr-sweep-a"), (2, "lr-sweep-a"), (3, "lr-sweep-a")]
        assert record["wandb_tags"] == [(stage, [expected_hp[stage], "label:lr-sweep-a"]) for stage in (1, 2, 3)]

        record = self._run("velociraptor", tmp_path / "unlabelled", monkeypatch, caplog, use_wandb=True)
        assert record["labels"] == [(1, None), (2, None), (3, None)]
        assert record["wandb_tags"] == [(stage, [expected_hp[stage]]) for stage in (1, 2, 3)]


class TestTrainRecordsTheLabel:
    """Decision D-A21 at the single-stage launch path: ``train(..., label=...)`` hands the label to
    ``save_stage_config`` (which records it beside the digest) exactly as given."""

    class ConfigReached(RuntimeError):
        pass

    def _save_config_kwargs(self, tmp_path, monkeypatch, **train_kwargs):
        from environments.shared import config as config_module
        from environments.shared import task_fingerprint, train_base
        from environments.shared.config import load_all_stages

        captured = {}

        def save_config(*args, **kwargs):
            captured.update(kwargs)
            raise self.ConfigReached()

        monkeypatch.setattr(train_base, "current_plant_identity", lambda species: SimpleNamespace(to_dict=dict))
        monkeypatch.setattr(task_fingerprint, "derive_stage_task_fingerprint", lambda **kwargs: {})
        monkeypatch.setattr(train_base, "_ensure_sb3", lambda: {})
        monkeypatch.setattr(config_module, "save_stage_config", save_config)
        with pytest.raises(self.ConfigReached):
            train_base.train(
                SimpleNamespace(species="velociraptor", env_class=object),
                load_all_stages("velociraptor"),
                1,
                total_timesteps=1,
                output_dir=str(tmp_path / "run"),
                use_tensorboard=False,
                verbose=0,
                **train_kwargs,
            )
        return captured

    def test_the_label_is_passed_through_and_defaults_to_none(self, tmp_path, monkeypatch):
        assert self._save_config_kwargs(tmp_path, monkeypatch, label="lr-sweep-a")["label"] == "lr-sweep-a"
        assert self._save_config_kwargs(tmp_path, monkeypatch)["label"] is None

    def test_the_wandb_tags_carry_the_digest_prefix_and_the_label(self):
        from environments.shared.config import hyperparameters_sha256, load_all_stages
        from environments.shared.train_base import _wandb_run_tags

        config = load_all_stages("velociraptor")[1]
        hp_tag = "hp:" + hyperparameters_sha256(config, "ppo").removeprefix("sha256:")[:12]
        assert _wandb_run_tags(config, "ppo", None) == [hp_tag]
        assert _wandb_run_tags(config, "ppo", "  ") == [hp_tag]
        assert _wandb_run_tags(config, "ppo", "lr-sweep-a") == [hp_tag, "label:lr-sweep-a"]


class TestTrainRefusesAnUndeclaredParent:
    """Invariant 4 at the launch path: train() checks the recorded parent stage against the
    manifest edge before it creates a directory, writes a config, or imports SB3."""

    @staticmethod
    def _checkpoint(path, stage):
        import zipfile

        from environments.shared.task_fingerprint import MODEL_TASK_ATTRIBUTE

        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("data", json.dumps({MODEL_TASK_ATTRIBUTE: {"species": "trex", "stage": stage}}))
        return path

    def _train(self, tmp_path, monkeypatch, *, parent_stage, child_stage):
        from environments.shared import task_fingerprint, train_base
        from environments.shared.config import load_all_stages

        class SB3Reached(RuntimeError):
            """The check passed: train() went on to import the backend."""

        monkeypatch.setattr(train_base, "current_plant_identity", lambda species: SimpleNamespace(to_dict=dict))
        monkeypatch.setattr(task_fingerprint, "derive_stage_task_fingerprint", lambda **kwargs: {})
        monkeypatch.setattr(train_base, "_ensure_sb3", lambda: (_ for _ in ()).throw(SB3Reached()))
        parent = self._checkpoint(tmp_path / "parent.zip", parent_stage)
        output_dir = tmp_path / "out"
        output_dir.mkdir(exist_ok=True)
        try:
            train_base.train(
                SimpleNamespace(species="trex", env_class=object),
                load_all_stages("trex"),
                child_stage,
                total_timesteps=1,
                load_path=str(parent),
                task_load_mode="initialize_next_stage",
                output_dir=str(output_dir),
                use_tensorboard=False,
                verbose=0,
            )
        except SB3Reached:
            return "passed-the-check", output_dir
        return "returned", output_dir

    def test_the_declared_parent_is_accepted_and_nothing_else_is(self, tmp_path, monkeypatch):
        from environments.shared.task_fingerprint import TaskFingerprintError

        outcome, output_dir = self._train(tmp_path, monkeypatch, parent_stage=1, child_stage=2)
        assert outcome == "passed-the-check"

        with pytest.raises(TaskFingerprintError, match="declares warm_start_from"):
            self._train(tmp_path, monkeypatch, parent_stage=3, child_stage=2)
        # Refused before a stage directory, a config or an environment existed.
        assert not any(output_dir.iterdir())

    def test_a_root_refuses_a_foreign_parent_but_accepts_itself(self, tmp_path, monkeypatch):
        from environments.shared.task_fingerprint import TaskFingerprintError

        outcome, _ = self._train(tmp_path, monkeypatch, parent_stage=1, child_stage=1)
        assert outcome == "passed-the-check"
        with pytest.raises(TaskFingerprintError, match="is a root node"):
            self._train(tmp_path, monkeypatch, parent_stage=2, child_stage=1)

    def test_a_same_stage_resume_is_not_checked_against_the_edge(self, tmp_path, monkeypatch):
        """resume_same_stage is judged by the exact task hash later, never by the edge here."""
        from environments.shared import task_fingerprint, train_base
        from environments.shared.config import load_all_stages

        calls = []
        monkeypatch.setattr(train_base, "current_plant_identity", lambda species: SimpleNamespace(to_dict=dict))
        monkeypatch.setattr(task_fingerprint, "derive_stage_task_fingerprint", lambda **kwargs: {})
        monkeypatch.setattr(task_fingerprint, "validate_declared_parent", lambda *a, **k: calls.append(k))
        monkeypatch.setattr(train_base, "_ensure_sb3", lambda: (_ for _ in ()).throw(KeyboardInterrupt()))
        parent = self._checkpoint(tmp_path / "parent.zip", 3)
        with pytest.raises(KeyboardInterrupt):
            train_base.train(
                SimpleNamespace(species="trex", env_class=object),
                load_all_stages("trex"),
                2,
                total_timesteps=1,
                load_path=str(parent),
                task_load_mode="resume_same_stage",
                output_dir=str(tmp_path),
                use_tensorboard=False,
                verbose=0,
            )
        assert calls == []


class TestTrainRefusesAnOccupiedStageDirectory:
    """Decision D-A20 at the launch path: train() refuses to write into a stage directory that
    already records a stage (``stage_config.json`` or ``gate_verdict.json``) unless the load is an
    explicit same-stage resume — checked after the directory is resolved and before its config,
    models directory or any environment is written."""

    class ConfigReached(RuntimeError):
        """The guard passed: train() went on to write the stage config."""

    def _train(self, tmp_path, monkeypatch, *, output_dir, load_path=None, task_load_mode="resume_same_stage"):
        from environments.shared import config as config_module
        from environments.shared import task_fingerprint, train_base
        from environments.shared.config import load_all_stages

        def save_config(*args, **kwargs):
            raise self.ConfigReached()

        monkeypatch.setattr(train_base, "current_plant_identity", lambda species: SimpleNamespace(to_dict=dict))
        monkeypatch.setattr(task_fingerprint, "derive_stage_task_fingerprint", lambda **kwargs: {})
        # The declared-parent check (pinned above) runs first and reads the checkpoint; the guard
        # under test is what happens once a load has passed it.
        monkeypatch.setattr(task_fingerprint, "read_checkpoint_task_fingerprint", lambda path: None)
        monkeypatch.setattr(task_fingerprint, "validate_declared_parent", lambda *a, **k: None)
        monkeypatch.setattr(train_base, "_ensure_sb3", lambda: {})
        monkeypatch.setattr(config_module, "save_stage_config", save_config)
        try:
            train_base.train(
                SimpleNamespace(species="velociraptor", env_class=object),
                load_all_stages("velociraptor"),
                1,
                total_timesteps=1,
                load_path=load_path,
                task_load_mode=task_load_mode,
                output_dir=str(output_dir),
                use_tensorboard=False,
                verbose=0,
            )
        except self.ConfigReached:
            return "passed-the-guard"
        return "returned"

    @pytest.mark.parametrize("occupancy_file", ["stage_config.json", "gate_verdict.json"])
    def test_a_recorded_stage_is_refused_from_scratch_and_from_a_parent(self, tmp_path, monkeypatch, occupancy_file):
        from environments.shared.config import StageDirectoryOccupiedError

        occupied = tmp_path / "old-run"
        occupied.mkdir()
        (occupied / occupancy_file).write_text("{}")

        # From scratch: the CLI's default resume_same_stage without --load is not a resume.
        with pytest.raises(StageDirectoryOccupiedError, match=occupancy_file) as excinfo:
            self._train(tmp_path, monkeypatch, output_dir=occupied)
        assert "--output-dir" in str(excinfo.value) and "--load-mode resume_same_stage" in str(excinfo.value)
        # Entering from a parent is a new variant, not a resume, whatever the checkpoint.
        with pytest.raises(StageDirectoryOccupiedError, match="initialize_next_stage"):
            self._train(
                tmp_path,
                monkeypatch,
                output_dir=occupied,
                load_path=str(tmp_path / "stance.zip"),
                task_load_mode="initialize_next_stage",
            )
        # Nothing was written into the refused directory.
        assert sorted(p.name for p in occupied.iterdir()) == [occupancy_file]

    def test_an_explicit_same_stage_resume_is_accepted(self, tmp_path, monkeypatch):
        occupied = tmp_path / "old-run"
        occupied.mkdir()
        (occupied / "stage_config.json").write_text("{}")
        (occupied / "gate_verdict.json").write_text("{}")

        outcome = self._train(
            tmp_path,
            monkeypatch,
            output_dir=occupied,
            load_path=str(occupied / "models" / "stage1_1000_steps.zip"),
            task_load_mode="resume_same_stage",
        )

        assert outcome == "passed-the-guard"

    def test_a_fresh_directory_is_never_refused(self, tmp_path, monkeypatch):
        assert self._train(tmp_path, monkeypatch, output_dir=tmp_path / "fresh") == "passed-the-guard"


class TestTrainResumeKeepsTheEdge:
    """A ``train --load <periodic> --load-mode resume_same_stage`` into a stage directory that
    entered from its parent re-saves ``stage_config.json`` through the real ``save_stage_config``
    and keeps the edge (``parent_checkpoint_sha256``, ``parent_run_id`` ...), recording the periodic
    checkpoint under ``RESUME_LINEAGE_KEYS`` — so a resumed node still chains for reuse (ancestors
    rule 4; BEHAVIOR_RECIPES_PLAN §4.7 branch 3). The notebook's ``train_stage`` re-saves through the
    same function with the same ``load_mode`` argument (pinned in test_sb3_notebook_pins)."""

    class EnvsReached(RuntimeError):
        """The stage config was written; train() went on to build its environments."""

    @staticmethod
    def _zip(path):
        import zipfile

        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("data", json.dumps({}))
            archive.writestr("policy.pth", b"weights")
        return path

    def test_the_kept_edge_survives_the_resume_re_save(self, tmp_path, monkeypatch):
        from environments.shared import task_fingerprint, train_base
        from environments.shared.config import (
            LOAD_LINEAGE_KEYS,
            RESUME_LINEAGE_KEYS,
            load_all_stages,
            save_stage_config,
        )
        from environments.shared.result_bundle import sha256_file

        stage_configs = load_all_stages("velociraptor")
        stage_dir = tmp_path / "run" / "02_locomotion"
        parent = self._zip(tmp_path / "run" / "01_stance" / "models" / "stance_final.zip")
        save_stage_config(
            stage_dir,
            2,
            stage_configs[2],
            "PPO",
            species="velociraptor",
            load_path=str(parent),
            load_mode="initialize_next_stage",
            parent_run_id="20260901_120000",
        )
        entered = json.loads((stage_dir / "stage_config.json").read_text())["run"]
        # The parent zip carries no task fingerprint, so parent_task_sha256 is (rightly) absent.
        edge = {key: entered[key] for key in LOAD_LINEAGE_KEYS if key in entered}
        assert set(edge) == {"load_path", "load_mode", "parent_checkpoint_sha256", "parent_run_id"}
        periodic = self._zip(stage_dir / "models" / "locomotion_100_steps.zip")

        def envs_reached(*args, **kwargs):
            raise self.EnvsReached()

        monkeypatch.setattr(train_base, "current_plant_identity", lambda species: _plant_identity())
        monkeypatch.setattr(task_fingerprint, "derive_stage_task_fingerprint", lambda **kwargs: {})
        monkeypatch.setattr(task_fingerprint, "read_checkpoint_task_fingerprint", lambda path: None)
        monkeypatch.setattr(task_fingerprint, "validate_declared_parent", lambda *a, **k: None)
        monkeypatch.setattr(train_base, "_ensure_sb3", lambda: {})
        monkeypatch.setattr(train_base, "create_vec_env", envs_reached)
        with pytest.raises(self.EnvsReached):
            train_base.train(
                SimpleNamespace(species="velociraptor", env_class=object),
                stage_configs,
                2,
                total_timesteps=1,
                load_path=str(periodic),
                task_load_mode="resume_same_stage",
                output_dir=str(stage_dir),
                use_tensorboard=False,
                verbose=0,
            )

        resumed = json.loads((stage_dir / "stage_config.json").read_text())["run"]
        assert {key: resumed[key] for key in LOAD_LINEAGE_KEYS if key in resumed} == edge
        assert resumed["parent_checkpoint_sha256"] == sha256_file(parent)
        assert {key: resumed[key] for key in RESUME_LINEAGE_KEYS} == {
            "resume_load_path": str(periodic),
            "resume_checkpoint_sha256": sha256_file(periodic),
        }
