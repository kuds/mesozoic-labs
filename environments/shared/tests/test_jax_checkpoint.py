"""Tests for jax_checkpoint module."""

from __future__ import annotations

import pickle

from environments.shared.jax_checkpoint import (
    CheckpointManager,
    load_checkpoint,
    save_checkpoint,
)


class TestSaveLoadCheckpoint:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "ckpt.pkl"
        params = {"w": [1.0, 2.0, 3.0]}
        save_checkpoint(path, params, update=10)
        data = load_checkpoint(path)
        assert data["update"] == 10
        assert data["params"] == params

    def test_with_obs_rms_and_history(self, tmp_path):
        path = tmp_path / "ckpt.pkl"
        save_checkpoint(
            path,
            params={"w": 1.0},
            obs_rms="rms_object",
            update=5,
            history={"reward": [1.0, 2.0]},
            extra={"best_reward": 5.5},
        )
        data = load_checkpoint(path)
        assert data["obs_rms"] == "rms_object"
        assert data["history"] == {"reward": [1.0, 2.0]}
        assert data["best_reward"] == 5.5


class TestCheckpointManager:
    def test_save_and_rotate(self, tmp_path):
        mgr = CheckpointManager(tmp_path, prefix="ckpt", max_keep=2)
        for u in (1, 2, 3):
            mgr.save({"w": u}, update=u)
        # Only the last 2 should remain
        remaining = sorted(p.name for p in tmp_path.iterdir())
        assert remaining == ["ckpt_2.pkl", "ckpt_3.pkl"]

    def test_latest_property(self, tmp_path):
        mgr = CheckpointManager(tmp_path, prefix="ckpt", max_keep=5)
        assert mgr.latest is None
        mgr.save({"w": 0}, update=0)
        assert mgr.latest is not None
        assert mgr.latest.name == "ckpt_0.pkl"

    def test_rediscovers_existing_files_on_construct(self, tmp_path):
        """Bug #9 regression: a fresh CheckpointManager pointed at a
        directory that already has rotated checkpoints must enforce
        max_keep across the OLD plus NEW files, not start counting
        from zero and let old files accumulate forever."""
        # Pre-populate directory with three checkpoints from a prior run
        for u in (1, 2, 3):
            with open(tmp_path / f"ckpt_{u}.pkl", "wb") as f:
                pickle.dump({"params": {"w": u}, "update": u}, f)

        mgr = CheckpointManager(tmp_path, prefix="ckpt", max_keep=2)
        # Saving one new ckpt should rotate the OLDEST pre-existing one out
        mgr.save({"w": 4}, update=4)
        remaining = sorted(p.name for p in tmp_path.iterdir())
        # max_keep=2 — newest two should survive: ckpt_3 and ckpt_4
        assert remaining == ["ckpt_3.pkl", "ckpt_4.pkl"]

    def test_ignores_unrelated_files_on_construct(self, tmp_path):
        """Random other .pkl files in the directory should not corrupt
        the rotation queue."""
        (tmp_path / "params.pkl").write_bytes(b"unrelated")
        (tmp_path / "ckpt_random_other.pkl").write_bytes(b"unrelated")
        with open(tmp_path / "ckpt_5.pkl", "wb") as f:
            pickle.dump({"params": {}, "update": 5}, f)

        mgr = CheckpointManager(tmp_path, prefix="ckpt", max_keep=1)
        mgr.save({"w": 6}, update=6)
        # ckpt_5 must rotate out, but unrelated files survive untouched
        names = sorted(p.name for p in tmp_path.iterdir())
        assert "params.pkl" in names
        assert "ckpt_random_other.pkl" in names
        assert "ckpt_5.pkl" not in names
        assert "ckpt_6.pkl" in names
