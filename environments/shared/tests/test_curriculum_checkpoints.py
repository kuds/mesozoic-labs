"""Tests for environments.shared.curriculum.checkpoints."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from environments.shared.curriculum import (
    SaveVecNormalizeCallback,
    load_vecnorm_stats,
)
from environments.shared.plant_contract import PlantCompatibilityError


class TestLoadVecnormStats:
    """Test that VecNormalize stats are correctly carried across stages."""

    @pytest.fixture
    def vec_envs(self):
        """Create a pair of VecNormalize-wrapped dummy envs for stage 1."""
        pytest.importorskip("stable_baselines3")
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        from environments.velociraptor.envs.raptor_env import RaptorEnv

        def _make():
            env = RaptorEnv()
            return Monitor(env)

        train = VecNormalize(DummyVecEnv([_make]), norm_obs=True, norm_reward=True)
        eval_ = VecNormalize(DummyVecEnv([_make]), norm_obs=True, norm_reward=True)
        yield train, eval_
        train.close()
        eval_.close()

    def test_stats_carried_forward(self, vec_envs, tmp_path):
        """obs_rms/ret_rms are copied from saved file into fresh envs."""
        import numpy as np
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        from environments.shared.curriculum import load_vecnorm_stats
        from environments.velociraptor.envs.raptor_env import RaptorEnv

        train_env, _ = vec_envs

        # Run a few steps so the running mean/var diverge from defaults
        obs = train_env.reset()
        for _ in range(50):
            action = [train_env.action_space.sample()]
            obs, _, dones, _ = train_env.step(action)
            if dones[0]:
                train_env.reset()

        # Snapshot the trained stats
        saved_obs_mean = train_env.obs_rms.mean.copy()
        saved_obs_var = train_env.obs_rms.var.copy()

        # Save to disk
        save_path = str(tmp_path / "stage1_final_vecnorm.pkl")
        train_env.save(save_path)

        # Create fresh envs (simulating a new stage)
        def _make():
            return Monitor(RaptorEnv())

        new_train = VecNormalize(DummyVecEnv([_make]), norm_obs=True, norm_reward=True)
        new_eval = VecNormalize(DummyVecEnv([_make]), norm_obs=True, norm_reward=True)

        # Before loading, stats should be at defaults (mean≈0, var≈1)
        assert not np.allclose(new_train.obs_rms.mean, saved_obs_mean)

        # Load stats from the "previous stage"
        loaded = load_vecnorm_stats(save_path, new_train, new_eval, unsafe_skip_plant_validation=True)
        assert loaded is True

        # Stats should now match the saved values
        np.testing.assert_array_equal(new_train.obs_rms.mean, saved_obs_mean)
        np.testing.assert_array_equal(new_train.obs_rms.var, saved_obs_var)
        np.testing.assert_array_equal(new_eval.obs_rms.mean, saved_obs_mean)
        np.testing.assert_array_equal(new_eval.obs_rms.var, saved_obs_var)

        # Train env should still be in training mode
        assert new_train.training is True
        assert new_train.norm_reward is True

        # Eval env should NOT be training or normalizing reward
        assert new_eval.training is False
        assert new_eval.norm_reward is False

        new_train.close()
        new_eval.close()

    def test_missing_file_returns_false(self, vec_envs):
        """load_vecnorm_stats returns False when file doesn't exist."""
        from environments.shared.curriculum import load_vecnorm_stats

        train_env, eval_env = vec_envs
        result = load_vecnorm_stats(
            "/nonexistent/path_vecnorm.pkl",
            train_env,
            eval_env,
            unsafe_skip_plant_validation=True,
        )
        assert result is False


class TestSaveVecNormalizeCallback:
    """Test SaveVecNormalizeCallback saves VecNormalize on new best model."""

    def test_saves_vecnormalize_on_step(self, tmp_path):
        """_on_step saves VecNormalize to the configured path."""
        save_path = str(tmp_path / "best_model_vecnorm.pkl")

        mock_vec_env = MagicMock()
        mock_model = MagicMock()
        mock_model.get_vec_normalize_env.return_value = mock_vec_env

        cb = object.__new__(SaveVecNormalizeCallback)
        cb.save_path = save_path
        cb.verbose = 0
        cb.model = mock_model

        result = cb._on_step()

        assert result is True
        mock_model.get_vec_normalize_env.assert_called_once()
        mock_vec_env.save.assert_called_once_with(save_path)

    def test_no_op_without_vecnormalize(self, tmp_path):
        """_on_step is a no-op when there is no VecNormalize wrapper."""
        save_path = str(tmp_path / "best_model_vecnorm.pkl")

        mock_model = MagicMock()
        mock_model.get_vec_normalize_env.return_value = None

        cb = object.__new__(SaveVecNormalizeCallback)
        cb.save_path = save_path
        cb.verbose = 0
        cb.model = mock_model

        result = cb._on_step()

        assert result is True
        assert not (tmp_path / "best_model_vecnorm.pkl").exists()

    def test_raises_without_sb3(self):
        """Constructor raises ImportError when SB3 is unavailable."""
        with patch("environments.shared.curriculum.sb3_compat._SB3_AVAILABLE", False):
            with pytest.raises(ImportError, match="stable-baselines3"):
                SaveVecNormalizeCallback(save_path="/tmp/test.pkl")


class TestLoadVecnormStatsMocked:
    """Test load_vecnorm_stats logic with mocked SB3 dependencies."""

    def _sb3_mock_modules(self):
        mock_vec_env_mod = MagicMock()
        return {
            "stable_baselines3": MagicMock(),
            "stable_baselines3.common": MagicMock(),
            "stable_baselines3.common.vec_env": mock_vec_env_mod,
        }, mock_vec_env_mod

    def test_requires_current_plant_before_file_or_backend_access(self):
        with pytest.raises(PlantCompatibilityError, match="without current_plant"):
            load_vecnorm_stats("/nonexistent/path.pkl", MagicMock())

        with pytest.raises(PlantCompatibilityError, match="without current_plant"):
            load_vecnorm_stats("/nonexistent/path.pkl", MagicMock(), allow_legacy_plant=True)

    def test_unsafe_skip_is_explicit_and_cannot_mix_with_validation(self, caplog):
        result = load_vecnorm_stats(
            "/nonexistent/path.pkl",
            MagicMock(),
            unsafe_skip_plant_validation=True,
        )
        assert result is False
        assert "UNSAFE" in caplog.text

        with pytest.raises(ValueError, match="cannot be combined"):
            load_vecnorm_stats(
                "/nonexistent/path.pkl",
                MagicMock(),
                current_plant=MagicMock(),
                unsafe_skip_plant_validation=True,
            )

    def test_missing_file_returns_false_with_sb3(self):
        mods, _ = self._sb3_mock_modules()
        with patch("environments.shared.curriculum.sb3_compat._SB3_AVAILABLE", True), patch.dict(sys.modules, mods):
            result = load_vecnorm_stats(
                "/nonexistent/path.pkl",
                MagicMock(),
                unsafe_skip_plant_validation=True,
            )
        assert result is False

    def test_loads_and_applies_stats(self, tmp_path):
        fake_pkl = tmp_path / "vecnorm.pkl"
        fake_pkl.write_bytes(b"fake")

        mods, mock_vec_env_mod = self._sb3_mock_modules()
        mock_prev = MagicMock()
        mock_vec_env_mod.VecNormalize.load.return_value = mock_prev

        # Caller-configured flags (e.g. SAC sets norm_reward=False in
        # create_vec_env).  load_vecnorm_stats must leave them alone.
        mock_train = MagicMock()
        mock_train.training = True
        mock_train.norm_reward = False
        mock_eval = MagicMock()

        with patch("environments.shared.curriculum.sb3_compat._SB3_AVAILABLE", True), patch.dict(sys.modules, mods):
            result = load_vecnorm_stats(
                str(fake_pkl),
                mock_train,
                mock_eval,
                unsafe_skip_plant_validation=True,
            )

        assert result is True
        assert mock_train.obs_rms == mock_prev.obs_rms
        assert mock_train.training is True
        assert mock_train.norm_reward is False
        assert mock_eval.training is False
        assert mock_eval.norm_reward is False

    def test_loads_without_eval_env(self, tmp_path):
        fake_pkl = tmp_path / "vecnorm.pkl"
        fake_pkl.write_bytes(b"fake")

        mods, mock_vec_env_mod = self._sb3_mock_modules()
        mock_prev = MagicMock()
        mock_vec_env_mod.VecNormalize.load.return_value = mock_prev

        mock_train = MagicMock()

        with patch("environments.shared.curriculum.sb3_compat._SB3_AVAILABLE", True), patch.dict(sys.modules, mods):
            result = load_vecnorm_stats(
                str(fake_pkl),
                mock_train,
                eval_env=None,
                unsafe_skip_plant_validation=True,
            )

        assert result is True
        assert mock_train.obs_rms == mock_prev.obs_rms

    def test_validates_plant_before_copying_stats(self, tmp_path):
        fake_pkl = tmp_path / "vecnorm.pkl"
        fake_pkl.write_bytes(b"fake")

        mods, mock_vec_env_mod = self._sb3_mock_modules()
        mock_prev = MagicMock()
        mock_prev.obs_rms = "incompatible stats"
        mock_vec_env_mod.VecNormalize.load.return_value = mock_prev
        mock_train = MagicMock()
        mock_train.obs_rms = "original stats"
        current_plant = MagicMock()

        with (
            patch("environments.shared.curriculum.sb3_compat._SB3_AVAILABLE", True),
            patch.dict(sys.modules, mods),
            patch(
                "environments.shared.plant_contract.validate_model_plant",
                side_effect=PlantCompatibilityError("wrong physics plant"),
            ) as mock_validate,
            pytest.raises(PlantCompatibilityError, match="wrong physics plant"),
        ):
            load_vecnorm_stats(
                str(fake_pkl),
                mock_train,
                current_plant=current_plant,
                allow_legacy_plant=True,
            )

        mock_validate.assert_called_once_with(
            mock_prev,
            current_plant,
            artifact=str(fake_pkl),
            allow_legacy=True,
        )
        assert mock_train.obs_rms == "original stats"


class TestRobustBestModelCallback:
    """RobustBestModelCallback saves by risk-adjusted (mean - std) score."""

    @staticmethod
    def _make_cb(model_dir, risk_coef=1.0):
        import numpy as np

        from environments.shared.curriculum import RobustBestModelCallback

        cb = object.__new__(RobustBestModelCallback)
        cb.eval_callback = MagicMock()
        cb.eval_callback.evaluations_results = []
        cb.model_dir = model_dir
        cb.risk_coef = risk_coef
        cb.best_score = -np.inf
        cb._last_seen_n = 0
        cb.num_timesteps = 0
        cb.model = MagicMock()
        cb.model.get_vec_normalize_env.return_value = None
        return cb

    def test_saves_on_first_eval(self, tmp_path):
        cb = self._make_cb(tmp_path)
        cb.eval_callback.evaluations_results = [[100.0, 100.0, 100.0]]

        assert cb._on_step() is True

        cb.model.save.assert_called_once_with(str(tmp_path / "robust_best_model"))
        assert cb.best_score == pytest.approx(100.0)

    def test_high_mean_high_variance_does_not_beat_consistent(self, tmp_path):
        """The 20260709_185946 failure mode: bimodal eval with a higher mean loses."""
        cb = self._make_cb(tmp_path)
        # Consistent checkpoint (like the ~450k eval): mean 1040, small std.
        consistent = [1040.0, 1090.0, 990.0]
        cb.eval_callback.evaluations_results = [consistent]
        cb._on_step()
        assert cb.model.save.call_count == 1

        # Later checkpoint with higher mean but a catastrophic tail
        # (like the 800k "peak"): mean-based selection would switch,
        # risk-adjusted selection must not.
        bimodal = [1400.0, 1380.0, 1350.0, 1300.0, 76.0]
        import numpy as np

        assert np.mean(bimodal) > np.mean(consistent)
        assert np.mean(bimodal) - np.std(bimodal) < cb.best_score
        cb.eval_callback.evaluations_results = [consistent, bimodal]
        cb._on_step()
        assert cb.model.save.call_count == 1  # no new save

    def test_saves_vecnorm_when_present(self, tmp_path):
        cb = self._make_cb(tmp_path)
        vec_env = MagicMock()
        cb.model.get_vec_normalize_env.return_value = vec_env
        cb.eval_callback.evaluations_results = [[50.0, 52.0]]

        cb._on_step()

        vec_env.save.assert_called_once_with(str(tmp_path / "robust_best_model") + "_vecnorm.pkl")

    def test_no_new_eval_is_noop(self, tmp_path):
        cb = self._make_cb(tmp_path)
        cb.eval_callback.evaluations_results = [[100.0]]
        cb._on_step()
        cb._on_step()  # same eval count — no re-save
        assert cb.model.save.call_count == 1

    def test_risk_coef_scales_penalty(self, tmp_path):
        import numpy as np

        cb = self._make_cb(tmp_path, risk_coef=2.0)
        rewards = [100.0, 60.0]
        cb.eval_callback.evaluations_results = [rewards]
        cb._on_step()
        expected = float(np.mean(rewards) - 2.0 * np.std(rewards))
        assert cb.best_score == pytest.approx(expected)

    def test_raises_without_sb3(self):
        from environments.shared.curriculum import RobustBestModelCallback

        with patch("environments.shared.curriculum.sb3_compat._SB3_AVAILABLE", False):
            with pytest.raises(ImportError, match="stable-baselines3"):
                RobustBestModelCallback(eval_callback=MagicMock(), model_dir="/tmp/x")


class TestPublishEvalArtifactsCallback:
    """PublishEvalArtifactsCallback atomically copies evaluations.npz."""

    @staticmethod
    def _make_cb(local_dir, publish_dir):
        from environments.shared.curriculum import PublishEvalArtifactsCallback

        cb = object.__new__(PublishEvalArtifactsCallback)
        cb.eval_callback = MagicMock()
        cb.eval_callback.log_path = str(local_dir / "evaluations")
        cb.eval_callback.evaluations_results = []
        cb.publish_dir = publish_dir
        cb._last_published_n = 0
        return cb

    def test_publishes_after_new_eval(self, tmp_path):
        import numpy as np

        local_dir = tmp_path / "local"
        local_dir.mkdir()
        publish_dir = tmp_path / "stage"
        np.savez(str(local_dir / "evaluations.npz"), results=np.array([[1.0]]))

        cb = self._make_cb(local_dir, publish_dir)
        cb.eval_callback.evaluations_results = [[1.0]]

        assert cb._on_step() is True

        data = np.load(str(publish_dir / "evaluations.npz"))
        assert data["results"].shape == (1, 1)

    def test_no_publish_without_new_eval(self, tmp_path):
        import numpy as np

        local_dir = tmp_path / "local"
        local_dir.mkdir()
        publish_dir = tmp_path / "stage"
        np.savez(str(local_dir / "evaluations.npz"), results=np.array([[1.0]]))

        cb = self._make_cb(local_dir, publish_dir)

        assert cb._on_step() is True  # no evals recorded yet
        assert not (publish_dir / "evaluations.npz").exists()

    def test_publishes_on_training_end(self, tmp_path):
        import numpy as np

        local_dir = tmp_path / "local"
        local_dir.mkdir()
        publish_dir = tmp_path / "stage"
        np.savez(str(local_dir / "evaluations.npz"), results=np.array([[1.0], [2.0]]))

        cb = self._make_cb(local_dir, publish_dir)
        cb._on_training_end()

        data = np.load(str(publish_dir / "evaluations.npz"))
        assert data["results"].shape == (2, 1)

    def test_missing_source_is_noop(self, tmp_path):
        local_dir = tmp_path / "local"
        local_dir.mkdir()
        publish_dir = tmp_path / "stage"

        cb = self._make_cb(local_dir, publish_dir)
        cb.eval_callback.evaluations_results = [[1.0]]

        assert cb._on_step() is True
        assert not (publish_dir / "evaluations.npz").exists()

    def test_raises_without_sb3(self):
        from environments.shared.curriculum import PublishEvalArtifactsCallback

        with patch("environments.shared.curriculum.sb3_compat._SB3_AVAILABLE", False):
            with pytest.raises(ImportError, match="stable-baselines3"):
                PublishEvalArtifactsCallback(eval_callback=MagicMock(), publish_dir="/tmp/x")


class TestPrunePeriodicCheckpoints:
    """Retention for SB3's periodic checkpoints (`prune_periodic_checkpoints`)."""

    def _populate(self, model_dir, steps=(500_000, 1_000_000, 1_500_000, 2_000_000)):
        model_dir.mkdir(parents=True, exist_ok=True)
        for step in steps:
            (model_dir / f"stage1_{step}_steps.zip").write_bytes(b"policy")
            (model_dir / f"stage1_vecnormalize_{step}_steps.pkl").write_bytes(b"stats")
        # The three checkpoints that must never be pruned.
        for keeper in ("best_model", "robust_best_model", "stage1_final"):
            (model_dir / f"{keeper}.zip").write_bytes(b"policy")
            (model_dir / f"{keeper}_vecnorm.pkl").write_bytes(b"stats")

    def test_keeps_the_newest_n_step_points(self, tmp_path):
        from environments.shared.curriculum import prune_periodic_checkpoints

        self._populate(tmp_path)
        removed = prune_periodic_checkpoints(tmp_path, "stage1", max_checkpoints=2)

        assert {p.name for p in removed} == {
            "stage1_500000_steps.zip",
            "stage1_vecnormalize_500000_steps.pkl",
            "stage1_1000000_steps.zip",
            "stage1_vecnormalize_1000000_steps.pkl",
        }
        assert (tmp_path / "stage1_2000000_steps.zip").exists()
        assert (tmp_path / "stage1_1500000_steps.zip").exists()

    def test_never_touches_best_robust_or_final(self, tmp_path):
        from environments.shared.curriculum import prune_periodic_checkpoints

        self._populate(tmp_path)
        prune_periodic_checkpoints(tmp_path, "stage1", max_checkpoints=1)

        for keeper in ("best_model", "robust_best_model", "stage1_final"):
            assert (tmp_path / f"{keeper}.zip").exists()
            assert (tmp_path / f"{keeper}_vecnorm.pkl").exists()

    def test_a_pruned_step_takes_its_vecnormalize_with_it(self, tmp_path):
        # An orphaned .pkl belongs to no policy left in the directory.
        from environments.shared.curriculum import prune_periodic_checkpoints

        self._populate(tmp_path)
        prune_periodic_checkpoints(tmp_path, "stage1", max_checkpoints=1)

        surviving_zips = sorted(p.name for p in tmp_path.glob("stage1_*_steps.zip"))
        surviving_pkls = sorted(p.name for p in tmp_path.glob("stage1_vecnormalize_*_steps.pkl"))
        assert surviving_zips == ["stage1_2000000_steps.zip"]
        assert surviving_pkls == ["stage1_vecnormalize_2000000_steps.pkl"]

    def test_orders_by_step_count_not_mtime(self, tmp_path):
        # On a Drive/GCS-FUSE mount mtime reflects when the upload finished,
        # which is not the order the checkpoints were produced in.
        import os

        from environments.shared.curriculum import prune_periodic_checkpoints

        self._populate(tmp_path, steps=(500_000, 2_000_000))
        # Make the OLD checkpoint look newest by mtime.
        os.utime(tmp_path / "stage1_500000_steps.zip", (10**9, 10**9))
        os.utime(tmp_path / "stage1_2000000_steps.zip", (10**6, 10**6))

        prune_periodic_checkpoints(tmp_path, "stage1", max_checkpoints=1)

        assert (tmp_path / "stage1_2000000_steps.zip").exists()
        assert not (tmp_path / "stage1_500000_steps.zip").exists()

    def test_zero_disables_pruning(self, tmp_path):
        from environments.shared.curriculum import prune_periodic_checkpoints

        self._populate(tmp_path)
        assert prune_periodic_checkpoints(tmp_path, "stage1", max_checkpoints=0) == []
        assert len(list(tmp_path.glob("stage1_*_steps.zip"))) == 4

    def test_leaves_unparsed_names_alone(self, tmp_path):
        from environments.shared.curriculum import prune_periodic_checkpoints

        self._populate(tmp_path, steps=(500_000,))
        stray = tmp_path / "stage1_handwritten_steps.zip"
        stray.write_bytes(b"not mine")
        prune_periodic_checkpoints(tmp_path, "stage1", max_checkpoints=0 + 1)
        assert stray.exists()

    def test_missing_directory_is_not_an_error(self, tmp_path):
        from environments.shared.curriculum import prune_periodic_checkpoints

        assert prune_periodic_checkpoints(tmp_path / "nope", "stage1", max_checkpoints=2) == []


class TestCheckpointRetentionCallback:
    def _callback(self, tmp_path, save_freq=10, max_checkpoints=2):
        pytest.importorskip("stable_baselines3")
        from unittest.mock import MagicMock

        from environments.shared.curriculum import CheckpointRetentionCallback

        cb = CheckpointRetentionCallback(
            model_dir=tmp_path,
            name_prefix="stage1",
            save_freq=save_freq,
            max_checkpoints=max_checkpoints,
        )
        cb.model = MagicMock()
        return cb

    def _populate(self, tmp_path):
        for step in (500_000, 1_000_000, 1_500_000):
            (tmp_path / f"stage1_{step}_steps.zip").write_bytes(b"policy")

    def test_prunes_only_on_the_checkpoint_cadence(self, tmp_path):
        # _on_step runs every training step; globbing a Drive mount six
        # million times a stage would cost more than the storage it saves.
        self._populate(tmp_path)
        cb = self._callback(tmp_path, save_freq=10, max_checkpoints=2)

        for _ in range(9):
            cb.n_calls += 1
            cb._on_step()
        assert len(list(tmp_path.glob("stage1_*_steps.zip"))) == 3, "must not prune between save points"

        cb.n_calls += 1  # n_calls == 10, a save step
        cb._on_step()
        assert len(list(tmp_path.glob("stage1_*_steps.zip"))) == 2

    def test_training_end_prunes_a_stage_that_stopped_between_save_points(self, tmp_path):
        # EvalCollapseEarlyStopCallback can end a stage anywhere.
        self._populate(tmp_path)
        cb = self._callback(tmp_path, save_freq=10, max_checkpoints=1)
        cb.n_calls = 7
        cb._on_training_end()
        assert sorted(p.name for p in tmp_path.glob("stage1_*_steps.zip")) == ["stage1_1500000_steps.zip"]

    def test_rejects_a_save_freq_that_would_prune_every_step(self, tmp_path):
        import pytest

        pytest.importorskip("stable_baselines3")

        from environments.shared.curriculum import CheckpointRetentionCallback

        with pytest.raises(ValueError, match="save_freq"):
            CheckpointRetentionCallback(model_dir=tmp_path, name_prefix="stage1", save_freq=0)
