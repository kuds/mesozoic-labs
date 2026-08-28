"""Tests for the checkpoint resume path (review §3.2, findings F3/F4).

Two defects are pinned here:

* F3 — the VecNormalize sidecar loader only probed the curated
  ``<base>_vecnorm.pkl`` name, so resuming from one of SB3's periodic
  ``<prefix>_<steps>_steps.zip`` checkpoints (whose sidecar is
  ``<prefix>_vecnormalize_<steps>_steps.pkl``) silently trained the loaded
  policy under fresh normalization statistics.
* F4 — stage-entry shaping (warm-up + forward-velocity ramp) keyed on stage
  position instead of the load mode, so a ``resume_same_stage`` resume that
  had just passed an exact task-fingerprint identity check trained on a
  ramp-modified task.  Shaping must apply only to
  ``initialize_next_stage`` boundary crossings, and the ramp only when the
  stage uses ``forward_vel_weight`` at all (the train_curriculum guard).
"""

import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from environments.shared import train_base
from environments.shared.train_base import (
    _load_vecnorm_into_envs,
    _resolve_vecnorm_sidecar,
    _stage_entry_shaping_callbacks,
)

# ── VecNormalize sidecar resolution (F3) ─────────────────────────────────


class TestResolveVecnormSidecar:
    """Both sidecar naming conventions must resolve to an existing file."""

    def test_curated_sidecar_is_found(self, tmp_path):
        (tmp_path / "best_model_vecnorm.pkl").touch()
        resolved = _resolve_vecnorm_sidecar(str(tmp_path / "best_model.zip"))
        assert resolved == str(tmp_path / "best_model_vecnorm.pkl")

    def test_zip_extension_is_optional(self, tmp_path):
        (tmp_path / "stage2_final_vecnorm.pkl").touch()
        resolved = _resolve_vecnorm_sidecar(str(tmp_path / "stage2_final"))
        assert resolved == str(tmp_path / "stage2_final_vecnorm.pkl")

    def test_periodic_checkpoint_falls_back_to_sb3_naming(self, tmp_path):
        # CheckpointCallback(save_vecnormalize=True) pairs
        # stage2_5000000_steps.zip with stage2_vecnormalize_5000000_steps.pkl.
        (tmp_path / "stage2_vecnormalize_5000000_steps.pkl").touch()
        resolved = _resolve_vecnorm_sidecar(str(tmp_path / "stage2_5000000_steps.zip"))
        assert resolved == str(tmp_path / "stage2_vecnormalize_5000000_steps.pkl")

    def test_curated_name_wins_over_periodic_when_both_exist(self, tmp_path):
        (tmp_path / "stage2_5000000_steps_vecnorm.pkl").touch()
        (tmp_path / "stage2_vecnormalize_5000000_steps.pkl").touch()
        resolved = _resolve_vecnorm_sidecar(str(tmp_path / "stage2_5000000_steps.zip"))
        assert resolved == str(tmp_path / "stage2_5000000_steps_vecnorm.pkl")

    def test_multi_underscore_prefix_survives_the_pattern(self, tmp_path):
        # The prefix group is greedy: a stage label with underscores must not
        # lose its tail to the step-count group.
        (tmp_path / "stage2_walk_vecnormalize_500000_steps.pkl").touch()
        resolved = _resolve_vecnorm_sidecar(str(tmp_path / "stage2_walk_500000_steps.zip"))
        assert resolved == str(tmp_path / "stage2_walk_vecnormalize_500000_steps.pkl")

    def test_missing_everything_returns_the_curated_name(self, tmp_path):
        # The caller's warning must name the primary probe.
        resolved = _resolve_vecnorm_sidecar(str(tmp_path / "stage2_5000000_steps.zip"))
        assert resolved == str(tmp_path / "stage2_5000000_steps_vecnorm.pkl")

    def test_non_periodic_name_never_probes_sb3_naming(self, tmp_path):
        resolved = _resolve_vecnorm_sidecar(str(tmp_path / "best_model.zip"))
        assert resolved == str(tmp_path / "best_model_vecnorm.pkl")

    def test_pkl_path_passes_through_unchanged(self, tmp_path):
        # train_curriculum hands the sidecar path itself; appending
        # _vecnorm.pkl to it would probe a file that cannot exist.
        sidecar = tmp_path / "stage1_final_vecnorm.pkl"
        sidecar.touch()
        assert _resolve_vecnorm_sidecar(str(sidecar)) == str(sidecar)

    def test_missing_pkl_path_still_passes_through(self, tmp_path):
        # The warning must name the path that was actually probed.
        sidecar = tmp_path / "robust_best_model_vecnorm.pkl"
        assert _resolve_vecnorm_sidecar(str(sidecar)) == str(sidecar)


class TestLoadVecnormIntoEnvsResolution:
    """_load_vecnorm_into_envs must load whichever sidecar exists."""

    @patch("environments.shared.train_base.logger")
    def test_periodic_sidecar_reaches_the_loader(self, mock_logger, tmp_path):
        (tmp_path / "stage2_vecnormalize_5000000_steps.pkl").touch()
        train_env = MagicMock()
        eval_env = MagicMock()
        with patch("environments.shared.curriculum.load_vecnorm_stats", return_value=True) as mock_load:
            _load_vecnorm_into_envs(str(tmp_path / "stage2_5000000_steps.zip"), train_env, eval_env)
        mock_load.assert_called_once_with(
            str(tmp_path / "stage2_vecnormalize_5000000_steps.pkl"),
            train_env,
            eval_env,
            unsafe_skip_plant_validation=True,
        )

    @patch("environments.shared.train_base.logger")
    def test_handoff_sidecar_path_is_not_doubled(self, mock_logger, tmp_path):
        # train_curriculum passes prev_vecnorm_path (already a sidecar);
        # deriving <base>_vecnorm.pkl from it produced *_vecnorm.pkl_vecnorm.pkl.
        sidecar = tmp_path / "stage1_final_vecnorm.pkl"
        sidecar.touch()
        train_env = MagicMock()
        eval_env = MagicMock()
        with patch("environments.shared.curriculum.load_vecnorm_stats", return_value=True) as mock_load:
            _load_vecnorm_into_envs(str(sidecar), train_env, eval_env)
        mock_load.assert_called_once_with(
            str(sidecar),
            train_env,
            eval_env,
            unsafe_skip_plant_validation=True,
        )

    @patch("environments.shared.train_base.logger")
    def test_missing_sidecar_still_warns_and_resets_eval_env(self, mock_logger, tmp_path):
        train_env = MagicMock()
        eval_env = MagicMock()
        with patch("environments.shared.curriculum.load_vecnorm_stats", return_value=False):
            _load_vecnorm_into_envs(str(tmp_path / "stage2_5000000_steps.zip"), train_env, eval_env)
        assert eval_env.training is False
        assert eval_env.norm_reward is False
        mock_logger.warning.assert_called_once()

    def test_real_vecnormalize_stats_round_trip_via_periodic_naming(self, tmp_path):
        """End-to-end: stats saved under SB3's periodic name are carried forward."""
        gym = pytest.importorskip("gymnasium")
        pytest.importorskip("stable_baselines3")
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        class TinyEnv(gym.Env):
            observation_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
            action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                return np.zeros(1, dtype=np.float32), {}

            def step(self, action):
                return np.zeros(1, dtype=np.float32), 0.0, False, False, {}

        source_env = VecNormalize(DummyVecEnv([TinyEnv]), norm_obs=True, norm_reward=True)
        source_env.obs_rms.mean = np.full(1, 0.42)
        source_env.obs_rms.var = np.full(1, 2.5)
        source_env.save(str(tmp_path / "stage2_vecnormalize_5000000_steps.pkl"))
        source_env.close()

        new_train = VecNormalize(DummyVecEnv([TinyEnv]), norm_obs=True, norm_reward=True)
        new_eval = VecNormalize(DummyVecEnv([TinyEnv]), norm_obs=True, norm_reward=True)
        try:
            _load_vecnorm_into_envs(str(tmp_path / "stage2_5000000_steps.zip"), new_train, new_eval)
            np.testing.assert_allclose(new_train.obs_rms.mean, [0.42])
            np.testing.assert_allclose(new_train.obs_rms.var, [2.5])
            np.testing.assert_allclose(new_eval.obs_rms.mean, [0.42])
            assert new_eval.training is False
            assert new_eval.norm_reward is False
        finally:
            new_train.close()
            new_eval.close()


# ── Stage-entry shaping keyed on the load mode (F4) ──────────────────────


def _shaping_config(forward_vel_weight=1.0, **cur_kwargs):
    return {
        "env_kwargs": {"forward_vel_weight": forward_vel_weight},
        "curriculum_kwargs": cur_kwargs,
    }


class TestStageEntryShapingCallbacks:
    """Warm-up/ramp only on initialize_next_stage boundary crossings."""

    def _types(self, callbacks):
        return [type(cb).__name__ for cb in callbacks]

    def test_resume_same_stage_gets_no_shaping(self):
        pytest.importorskip("stable_baselines3")
        callbacks = _stage_entry_shaping_callbacks(
            _shaping_config(),
            task_load_mode="resume_same_stage",
            stage_position=2,
            load_path="/run/models/stage2_5000000_steps.zip",
        )
        assert callbacks == []

    def test_initialize_next_stage_gets_warmup_and_ramp(self):
        pytest.importorskip("stable_baselines3")
        from environments.shared.curriculum import RewardRampCallback, StageWarmupCallback

        callbacks = _stage_entry_shaping_callbacks(
            _shaping_config(),
            task_load_mode="initialize_next_stage",
            stage_position=2,
            load_path="/run/models/robust_best_model.zip",
        )
        assert self._types(callbacks) == ["StageWarmupCallback", "RewardRampCallback"]
        assert isinstance(callbacks[0], StageWarmupCallback)
        assert isinstance(callbacks[1], RewardRampCallback)

    def test_first_stage_gets_no_shaping_even_when_crossing(self):
        pytest.importorskip("stable_baselines3")
        callbacks = _stage_entry_shaping_callbacks(
            _shaping_config(),
            task_load_mode="initialize_next_stage",
            stage_position=1,
            load_path="/run/models/best_model.zip",
        )
        assert callbacks == []

    def test_fresh_model_gets_no_shaping(self):
        pytest.importorskip("stable_baselines3")
        callbacks = _stage_entry_shaping_callbacks(
            _shaping_config(),
            task_load_mode="initialize_next_stage",
            stage_position=2,
            load_path=None,
        )
        assert callbacks == []

    def test_zero_forward_vel_weight_suppresses_the_ramp(self):
        # Recovery mirrors stance with forward_vel_weight = 0.0; ramping
        # 0.1 -> 0.0 would inject a walk incentive the fingerprint denies.
        pytest.importorskip("stable_baselines3")
        callbacks = _stage_entry_shaping_callbacks(
            _shaping_config(forward_vel_weight=0.0),
            task_load_mode="initialize_next_stage",
            stage_position=2,
            load_path="/run/models/robust_best_model.zip",
        )
        assert self._types(callbacks) == ["StageWarmupCallback"]

    def test_curriculum_kwargs_reach_the_callbacks(self):
        pytest.importorskip("stable_baselines3")
        callbacks = _stage_entry_shaping_callbacks(
            _shaping_config(
                forward_vel_weight=0.8,
                warmup_timesteps=50_000,
                warmup_clip_range=0.05,
                ramp_start_value=0.2,
                ramp_timesteps=250_000,
            ),
            task_load_mode="initialize_next_stage",
            stage_position=2,
            load_path="/run/models/best_model.zip",
        )
        warmup, ramp = callbacks
        assert warmup.warmup_timesteps == 50_000
        assert warmup.warmup_clip_range == 0.05
        assert ramp.attr_name == "forward_vel_weight"
        assert ramp.start_value == 0.2
        assert ramp.end_value == 0.8
        assert ramp.ramp_timesteps == 250_000


class TestShapingIsWired:
    """Both launch paths must build shaping through the shared helper.

    An inline copy is how the ramp guard was lost once already (the
    notebook's train_stage cell, 20260821 recovery pilot), so the absence
    of inline construction is itself the invariant.
    """

    def test_train_routes_through_the_helper_keyed_on_its_load_mode(self):
        src = inspect.getsource(train_base.train)
        assert "_stage_entry_shaping_callbacks(" in src
        assert "task_load_mode=task_load_mode" in src
        assert "StageWarmupCallback(" not in src
        assert "RewardRampCallback(" not in src

    def test_train_curriculum_routes_through_the_helper_as_a_boundary_crossing(self):
        src = inspect.getsource(train_base.train_curriculum)
        assert "_stage_entry_shaping_callbacks(" in src
        assert 'task_load_mode="initialize_next_stage"' in src
        assert "StageWarmupCallback(" not in src
        assert "RewardRampCallback(" not in src

    def test_train_resolves_the_sidecar_from_its_load_path(self):
        # _load_vecnorm_into_envs owns the resolution for both launch paths.
        src = inspect.getsource(train_base.train)
        assert "_load_vecnorm_into_envs(" in src
        src_curriculum = inspect.getsource(train_base.train_curriculum)
        assert "_load_vecnorm_into_envs(" in src_curriculum


# ── The resolver is the loader's single naming authority ─────────────────


class TestResolverMatchesCheckpointRetentionNaming:
    """The loader and the pruner must agree on SB3's periodic naming.

    ``CheckpointRetentionCallback`` derives the same sidecar name when it
    prunes; a periodic checkpoint it would pair for deletion must be one the
    resolver can find for loading.
    """

    def test_resolver_finds_what_the_retention_callback_pairs(self, tmp_path):
        from environments.shared.curriculum.checkpoints import _CHECKPOINT_KINDS

        prefix, steps = "stage2", 5000000
        vecnorm_infix = next(infix for infix, ext in _CHECKPOINT_KINDS if ext == "pkl" and "vecnormalize" in infix)
        sidecar = tmp_path / f"{prefix}_{vecnorm_infix}{steps}_steps.pkl"
        sidecar.touch()
        resolved = _resolve_vecnorm_sidecar(str(tmp_path / f"{prefix}_{steps}_steps.zip"))
        assert Path(resolved) == sidecar
