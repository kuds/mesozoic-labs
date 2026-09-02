"""Wiring pins for train_base's shared setup helpers.

* CO5 -- TensorBoard local-buffering was keyed on ``/gcs/`` only, so on
  Colab + Drive the event files streamed straight to the FUSE mount and a
  hard runtime reclaim lost the un-uploaded tail of the stage's logs.  Any
  remote mount (:func:`_is_remote_mount_path`) must now get the same
  buffer + periodic-sync treatment.
* TC9 -- the budget-anchored advisories (never-beaten-baseline warning,
  collapse warm-up sanity log) re-read the TOML budget instead of the total
  the session actually trains to, so a ``--timesteps`` override or a
  shortened resume mis-timed or suppressed them.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from environments.shared.curriculum import early_stopping
from environments.shared.tb_sync import PeriodicTbSyncCallback
from environments.shared.train_base import _build_core_callbacks, _prepare_alg_kwargs

_CONFIG = {
    "ppo_kwargs": {"learning_rate": 3e-4, "batch_size": 64, "clip_range": 0.2},
    "sac_kwargs": {"learning_rate": 1e-3, "batch_size": 256},
}


class TestRemoteMountTensorBoardBuffering:
    @pytest.mark.parametrize("mount", ["/content/drive/MyDrive/runs/run1", "/gdrive/MyDrive/run1", "/gcs/bucket/run1"])
    def test_every_remote_mount_buffers_locally(self, mount):
        kwargs, local_tb, remote_tb = _prepare_alg_kwargs(_CONFIG, "ppo", 1, Path(mount), True)
        assert local_tb is not None and local_tb.exists()
        assert kwargs["tensorboard_log"] == str(local_tb)
        assert remote_tb == Path(mount) / "tensorboard"

    def test_a_local_path_streams_in_place(self, tmp_path):
        kwargs, local_tb, _ = _prepare_alg_kwargs(_CONFIG, "ppo", 1, tmp_path, True)
        assert local_tb is None
        assert kwargs["tensorboard_log"] == str(tmp_path / "tensorboard")

    def test_drive_buffer_gets_the_periodic_sync_on_the_checkpoint_cadence(self, tmp_path):
        """The same treatment GCS gets: buffer dir in, PeriodicTbSyncCallback out."""
        _, local_tb, remote_tb = _prepare_alg_kwargs(_CONFIG, "ppo", 1, Path("/content/drive/MyDrive/run1"), True)
        callbacks = _build(tmp_path, {"curriculum_kwargs": {"min_avg_reward": 1.0}}, local_tb, remote_tb)
        sync = next(cb for cb in callbacks if isinstance(cb, PeriodicTbSyncCallback))
        assert sync.local_tb_dir == local_tb
        assert Path(sync.gcs_tb_path) == remote_tb
        assert sync.sync_freq == 100


class TestRemoteMountPathResolution:
    """The mount test resolves the path first: a relative path given from the
    mount's parent, or a symlink into the mount, lands on the mount all the
    same, and a prefix test on the unresolved text let both stream straight
    to it (review follow-up on CO5)."""

    @pytest.fixture
    def mount(self, tmp_path, monkeypatch):
        from environments.shared import train_base

        root = tmp_path / "content" / "drive"
        (root / "MyDrive").mkdir(parents=True)
        monkeypatch.setattr(train_base, "_REMOTE_MOUNT_ROOTS", (str(root),))
        return root

    def test_a_relative_path_under_the_mount_is_remote(self, mount, monkeypatch):
        from environments.shared.train_base import _is_remote_mount_path

        monkeypatch.chdir(mount.parent)
        assert _is_remote_mount_path("drive/MyDrive/runs/run1")
        assert _is_remote_mount_path(Path("drive") / "MyDrive")

    def test_a_symlink_into_the_mount_is_remote(self, mount, tmp_path):
        from environments.shared.train_base import _is_remote_mount_path

        link = tmp_path / "runs"
        link.symlink_to(mount / "MyDrive", target_is_directory=True)
        assert _is_remote_mount_path(link / "run1")

    def test_a_local_path_and_a_uri_are_judged_as_before(self, mount, tmp_path):
        from environments.shared.train_base import _is_remote_mount_path

        assert not _is_remote_mount_path(tmp_path / "local" / "run1")
        assert _is_remote_mount_path("/gcs/bucket/run1")
        # A URI is not a filesystem write target: tested as given, never resolved.
        assert not _is_remote_mount_path("gs://bucket/run1")


class TestBufferedResumeContinuesTheRemoteRun:
    """A resume on a fresh runtime must keep writing to the mount's latest
    TensorBoard run directory, not start a PPO_0 beside the pre-crash PPO_1."""

    def test_remote_run_dirs_are_mirrored_into_the_buffer(self, tmp_path):
        from environments.shared.tb_sync import _mirror_remote_run_dirs

        remote = tmp_path / "remote" / "tensorboard"
        (remote / "PPO_1").mkdir(parents=True)
        (remote / "PPO_1" / "events.out.tfevents.1").write_bytes(b"x")
        (remote / "notes.txt").write_text("not a run dir")
        local = tmp_path / "buffer"
        local.mkdir()

        assert _mirror_remote_run_dirs(local, remote) == ["PPO_1"]
        assert (local / "PPO_1").is_dir()
        assert not any((local / "PPO_1").iterdir()), "only the NAME is mirrored, never the event files"
        assert _mirror_remote_run_dirs(local, tmp_path / "missing") == []

    def test_sb3_numbering_continues_from_the_mirrored_names(self, tmp_path):
        pytest.importorskip("stable_baselines3")
        from stable_baselines3.common.utils import configure_logger

        from environments.shared.tb_sync import _mirror_remote_run_dirs

        remote = tmp_path / "remote" / "tensorboard"
        (remote / "PPO_1").mkdir(parents=True)
        empty = tmp_path / "empty-buffer"
        empty.mkdir()
        mirrored = tmp_path / "mirrored-buffer"
        mirrored.mkdir()
        _mirror_remote_run_dirs(mirrored, remote)

        assert Path(configure_logger(0, str(empty), "PPO", reset_num_timesteps=False).dir).name == "PPO_0"
        assert Path(configure_logger(0, str(mirrored), "PPO", reset_num_timesteps=False).dir).name == "PPO_1"

    def test_prepare_alg_kwargs_mirrors_before_handing_the_buffer_to_sb3(self, tmp_path, monkeypatch):
        from environments.shared import train_base

        remote_root = tmp_path / "content" / "drive" / "MyDrive" / "run1"
        (remote_root / "tensorboard" / "PPO_1").mkdir(parents=True)
        monkeypatch.setattr(train_base, "_REMOTE_MOUNT_ROOTS", (str(tmp_path / "content" / "drive"),))

        kwargs, local_tb, remote_tb = _prepare_alg_kwargs(_CONFIG, "ppo", 1, remote_root, True)
        assert remote_tb == remote_root / "tensorboard"
        assert (Path(kwargs["tensorboard_log"]) / "PPO_1").is_dir()
        assert local_tb is not None and local_tb == Path(kwargs["tensorboard_log"])


class TestPostEvalEpisodesRejectsNegatives:
    def test_train_base_rejects_a_negative_count(self):
        from environments.shared.train_base import _validate_post_eval_episodes

        _validate_post_eval_episodes(None)
        _validate_post_eval_episodes(0)
        with pytest.raises(ValueError, match="non-negative"):
            _validate_post_eval_episodes(-1)

    def test_the_cli_type_rejects_a_negative_count(self):
        import argparse

        from environments.shared.cli import _non_negative_int

        assert _non_negative_int("0") == 0
        assert _non_negative_int("12") == 12
        with pytest.raises(argparse.ArgumentTypeError):
            _non_negative_int("-1")
        with pytest.raises(argparse.ArgumentTypeError):
            _non_negative_int("three")

    def test_train_refuses_before_doing_any_work(self, tmp_path):
        from environments.shared import train_base

        with pytest.raises(ValueError, match="post_eval_episodes"):
            train_base.train(
                species_cfg=None,
                stage_configs={},
                stage=1,
                total_timesteps=10,
                output_dir=str(tmp_path),
                post_eval_episodes=-1,
            )


class TestAdvisoriesAnchorOnTheActualBudget:
    def test_collapse_warmup_sanity_log_uses_the_passed_total(self, monkeypatch, caplog):
        monkeypatch.setattr(early_stopping, "EvalCollapseEarlyStopCallback", lambda **kwargs: "callback")
        toml = {"collapse_peak_warmup_timesteps": 5_000_000, "timesteps": 10_000_000, "collapse_peak_floor": 1.0}

        with caplog.at_level(logging.WARNING, logger=early_stopping.logger.name):
            # TOML says 10M (warm-up reachable); this session trains 4M.
            early_stopping.build_eval_collapse_early_stop_callback(None, toml, total_timesteps=4_000_000)
        assert any("disabled for the whole run" in r.message for r in caplog.records)

        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=early_stopping.logger.name):
            # TOML says 1M (warm-up past it); this session trains 10M.
            early_stopping.build_eval_collapse_early_stop_callback(
                None, {**toml, "timesteps": 1_000_000}, total_timesteps=10_000_000
            )
        assert not any("disabled for the whole run" in r.message for r in caplog.records)

    def test_core_callbacks_thread_the_total_to_both_advisories(self, tmp_path, caplog):
        from environments.shared.curriculum import BaselineProgressCallback

        run_dir = tmp_path
        (run_dir / "zero_action_baseline.json").write_text(
            json.dumps({"results": {"trex": {"reward_mean": 3271.77, "episodes": 40}}}), encoding="utf-8"
        )
        stage_config = {
            "curriculum_kwargs": {
                "min_avg_reward": 1.0,
                "timesteps": 10_000_000,
                "collapse_peak_warmup_timesteps": 5_000_000,
                "collapse_peak_floor": 1.0,
            }
        }
        with caplog.at_level(logging.WARNING, logger=early_stopping.logger.name):
            callbacks = _build(run_dir / "01_stance", stage_config, None, None, species="trex", total=2_000_000)

        baseline = next(cb for cb in callbacks if isinstance(cb, BaselineProgressCallback))
        assert baseline.total_timesteps == 2_000_000, "anchored on the TOML budget, not the session's"
        assert any("disabled for the whole run" in r.message for r in caplog.records), (
            "the warm-up sanity log compared against the TOML budget, not the session's"
        )


def _build(stage_dir, stage_config, local_tb_dir, gcs_tb_path, *, species=None, total=None):
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
            Path(stage_dir) / "models",
            Path(stage_dir),
            1,
            1,
            100,
            100,
            0,
            stage_config,
            local_tb_dir=local_tb_dir,
            gcs_tb_path=gcs_tb_path,
            species=species,
            total_timesteps=total,
        )
    finally:
        eval_env.close()
    return callbacks


def test_the_notebook_anchors_the_advisories_on_the_session_target():
    """TC9, notebook side: the call passes the resume-aware target it already computes."""
    repo_root = Path(__file__).resolve().parents[3]
    notebook = json.loads((repo_root / "notebooks" / "sb3_training.ipynb").read_text(encoding="utf-8"))
    cells = ["".join(c.get("source", [])) for c in notebook["cells"] if c.get("cell_type") == "code"]
    calls = [c for c in cells if "_build_core_callbacks(" in c]
    assert len(calls) == 1, "expected exactly one _build_core_callbacks call"
    cell = calls[0]
    call_start = cell.index("_build_core_callbacks(")
    assert "total_timesteps=target_timesteps," in cell[call_start:]
    assert cell.index("target_timesteps = loaded_steps + timesteps") < call_start


def test_the_notebook_binds_every_evidence_csv_to_its_checkpoint():
    """RP3, notebook side: each evaluation CSV records the hash of the checkpoint it rolled."""
    repo_root = Path(__file__).resolve().parents[3]
    notebook = json.loads((repo_root / "notebooks" / "sb3_training.ipynb").read_text(encoding="utf-8"))
    cells = ["".join(c.get("source", [])) for c in notebook["cells"] if c.get("cell_type") == "code"]
    writers = [c for c in cells if "_lib_save_evaluation_episodes(" in c]
    assert len(writers) == 1, "expected the evidence writers in exactly one cell"
    cell = writers[0]
    calls = cell.split("_lib_save_evaluation_episodes(")[1:]
    assert len(calls) == 3, "expected final, selected and selected-fallback evidence writes"
    bound = [call.split(")", 1)[0] for call in calls]
    assert all("checkpoint_path=" in call for call in bound)
    assert sum('checkpoint_path=f"{final_path}.zip"' in call for call in bound) == 2
    assert sum("checkpoint_path=best_model_zip" in call for call in bound) == 1
