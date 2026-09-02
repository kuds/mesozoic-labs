"""Tests for shared training infrastructure (train_base.py)."""

import dataclasses
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
    PlantIdentity,
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


def _plant_identity(**changes):
    values = {
        "species": "velociraptor",
        "model_path": "environments/velociraptor/assets/raptor.xml",
        "physics_revision": 1,
        "policy_interface_revision": 1,
        "visual_revision": 1,
        "source_closure_sha256": "sha256:" + "1" * 64,
        "policy_interface_sha256": "sha256:" + "2" * 64,
        "physics_sha256": "sha256:" + "3" * 64,
        "visual_sha256": "sha256:" + "4" * 64,
        "nq": 31,
        "nv": 30,
        "nu": 22,
        "observation_dim": 67,
        "action_dim": 22,
    }
    values.update(changes)
    return PlantIdentity(**values)


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

    def _run(self, species, tmp_path, monkeypatch, caplog):
        from environments.shared import config as config_module
        from environments.shared import curriculum as curriculum_module
        from environments.shared import plant_contract, task_fingerprint, train_base
        from environments.shared.config import load_all_stages
        from environments.shared.stage_manifest import stage_label

        record: dict = {"saved": [], "loads": [], "positions": []}
        model = MagicMock()
        model.num_timesteps = 10

        def create_or_load(sb3, algorithm, alg_kwargs, train_env, load_path, **kwargs):
            record["loads"].append(load_path)
            return model

        def save_config(stage_dir, stage, config, algorithm, **kwargs):
            record["saved"].append((stage, kwargs.get("load_path"), kwargs.get("load_mode")))

        def shaping(config, **kwargs):
            record["positions"].append(kwargs["stage_position"])
            return []

        monkeypatch.setattr(train_base, "_ensure_sb3", lambda: {"CallbackList": list})
        monkeypatch.setattr(train_base, "current_plant_identity", lambda species: SimpleNamespace(to_dict=dict))
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
        monkeypatch.setattr(curriculum_module, "CurriculumCallback", lambda **kwargs: MagicMock(ready_to_advance=True))
        monkeypatch.setattr(task_fingerprint, "derive_stage_task_fingerprint", lambda **kwargs: {})
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
                output_dir=str(tmp_path),
            )
        return record

    def test_trex_visits_the_advancing_stages_in_manifest_order_and_logs_the_skip(self, tmp_path, monkeypatch, caplog):
        from environments.shared.stage_manifest import load_stage_manifest

        record = self._run("trex", tmp_path, monkeypatch, caplog)

        manifest = load_stage_manifest("trex")
        assert [stage for stage, _, _ in record["saved"]] == [entry.reference for entry in manifest.advancing_stages]
        assert [stage for stage, _, _ in record["saved"]] == [1, 2, 3]
        skips = [r for r in caplog.records if "Skipping non-advancing stage 'recovery'" in r.message]
        assert len(skips) == 1 and skips[0].levelno == logging.WARNING
        assert "train --stage recovery" in skips[0].message
        # Shaping is keyed on the manifest POSITION, so locomotion (legacy 2)
        # enters as position 3 and behavior as position 4.
        assert record["positions"] == [1, 3, 4]

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
        assert record["positions"] == [1, 2, 3]
        assert not [r for r in caplog.records if "Skipping non-advancing stage" in r.message]
