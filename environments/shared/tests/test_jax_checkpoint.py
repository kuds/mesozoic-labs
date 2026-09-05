"""Tests for jax_checkpoint module."""

from __future__ import annotations

import pickle

import pytest

from environments.shared.jax_checkpoint import (
    CheckpointManager,
    load_checkpoint,
    restore_train_state,
    save_checkpoint,
)
from environments.shared.plant_contract import (
    MODEL_IDENTITY_ATTRIBUTE,
    PlantCompatibilityError,
)

from .reporting_helpers import make_plant_identity as _plant_identity


class TestSaveLoadCheckpoint:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "ckpt.pkl"
        params = {"w": [1.0, 2.0, 3.0]}
        identity = _plant_identity()
        save_checkpoint(path, params, update=10, plant_identity=identity)
        data = load_checkpoint(path, current_plant=identity)
        assert data["update"] == 10
        assert data["params"] == params

    def test_with_obs_rms_and_history(self, tmp_path):
        path = tmp_path / "ckpt.pkl"
        identity = _plant_identity()
        save_checkpoint(
            path,
            params={"w": 1.0},
            obs_rms="rms_object",
            update=5,
            history={"reward": [1.0, 2.0]},
            extra={"best_reward": 5.5},
            plant_identity=identity,
        )
        data = load_checkpoint(path, current_plant=identity)
        assert data["obs_rms"] == "rms_object"
        assert data["history"] == {"reward": [1.0, 2.0]}
        assert data["best_reward"] == 5.5

    @pytest.mark.parametrize("load_kwargs", [{}, {"allow_legacy_plant": True}])
    def test_omitting_current_plant_fails_before_deserialization(self, tmp_path, load_kwargs):
        path = tmp_path / "untrusted.pkl"
        path.write_bytes(b"not a pickle")

        with pytest.raises(PlantCompatibilityError, match="without current_plant"):
            load_checkpoint(path, **load_kwargs)

    def test_explicit_unsafe_opt_out_loads_without_validation(self, tmp_path, caplog):
        path = tmp_path / "unvalidated.pkl"
        save_checkpoint(path, params={"w": 1.0})

        data = load_checkpoint(path, unsafe_skip_plant_validation=True)

        assert data["params"] == {"w": 1.0}
        assert "UNSAFE" in caplog.text
        assert "without plant compatibility validation" in caplog.text

    @pytest.mark.parametrize("extra_kwargs", [{"current_plant": _plant_identity()}, {"allow_legacy_plant": True}])
    def test_unsafe_opt_out_rejects_validation_options(self, tmp_path, extra_kwargs):
        path = tmp_path / "ckpt.pkl"
        save_checkpoint(path, params={})

        with pytest.raises(ValueError, match="cannot be combined"):
            load_checkpoint(path, unsafe_skip_plant_validation=True, **extra_kwargs)

    def test_plant_identity_round_trip_and_validation(self, tmp_path):
        path = tmp_path / "ckpt.pkl"
        identity = _plant_identity()

        save_checkpoint(path, params={"w": 1.0}, plant_identity=identity)
        data = load_checkpoint(path, current_plant=identity)

        assert data[MODEL_IDENTITY_ATTRIBUTE] == identity.to_dict()

    def test_incompatible_tagged_checkpoint_is_rejected(self, tmp_path):
        path = tmp_path / "ckpt.pkl"
        save_checkpoint(path, params={}, plant_identity=_plant_identity(physics_revision=2))

        with pytest.raises(PlantCompatibilityError, match="physics_revision"):
            load_checkpoint(path, current_plant=_plant_identity(), allow_legacy_plant=True)

    def test_legacy_checkpoint_requires_explicit_override(self, tmp_path):
        path = tmp_path / "legacy.pkl"
        save_checkpoint(path, params={"w": 1.0})
        current = _plant_identity()

        with pytest.raises(PlantCompatibilityError, match="has no plant identity"):
            load_checkpoint(path, current_plant=current)

        assert load_checkpoint(path, current_plant=current, allow_legacy_plant=True)["params"] == {"w": 1.0}

    def test_restore_train_state_forwards_plant_validation(self, tmp_path):
        path = tmp_path / "ckpt.pkl"
        identity = _plant_identity()
        save_checkpoint(path, params={"w": 1.0}, opt_state="opt", update=7, plant_identity=identity)

        params, opt_state, obs_rms, update = restore_train_state(path, current_plant=identity)

        assert params == {"w": 1.0}
        assert opt_state == "opt"
        assert obs_rms is None
        assert update == 7

    def test_restore_train_state_requires_current_plant(self, tmp_path):
        path = tmp_path / "ckpt.pkl"
        save_checkpoint(path, params={"w": 1.0}, opt_state="opt")

        with pytest.raises(PlantCompatibilityError, match="without current_plant"):
            restore_train_state(path)

    def test_restore_train_state_explicit_unsafe_opt_out(self, tmp_path, caplog):
        path = tmp_path / "ckpt.pkl"
        save_checkpoint(path, params={"w": 1.0}, opt_state="opt", update=3)

        params, opt_state, _obs_rms, update = restore_train_state(
            path,
            unsafe_skip_plant_validation=True,
        )

        assert params == {"w": 1.0}
        assert opt_state == "opt"
        assert update == 3
        assert "UNSAFE" in caplog.text

    def test_restore_train_state_can_refuse_a_checkpoint_without_opt_state(self, tmp_path):
        """A best-model / params-only file must not resume a stage: substituting
        ``optimizer.init`` restarts Adam and the LR schedule mid-stage."""
        path = tmp_path / "best.pkl"
        identity = _plant_identity()
        save_checkpoint(path, params={"w": 1.0}, update=7, plant_identity=identity)

        with pytest.raises(ValueError, match="no optimizer state"):
            restore_train_state(path, current_plant=identity, require_opt_state=True)
        # The default keeps the substitution for callers that want a fresh optimizer.
        _params, opt_state, _obs_rms, update = restore_train_state(path, current_plant=identity)
        assert opt_state is None and update == 7

        save_checkpoint(path, params={"w": 1.0}, opt_state="opt", update=7, plant_identity=identity)
        assert restore_train_state(path, current_plant=identity, require_opt_state=True)[1] == "opt"

    def test_extra_cannot_override_reserved_plant_metadata(self, tmp_path):
        with pytest.raises(ValueError, match="reserved key"):
            save_checkpoint(
                tmp_path / "ckpt.pkl",
                params={},
                extra={MODEL_IDENTITY_ATTRIBUTE: {}},
                plant_identity=_plant_identity(),
            )


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

    def test_resaving_an_update_takes_one_slot_and_keeps_the_file_just_written(self, tmp_path):
        """The trainer's final save lands on the last periodic count whenever
        the budget is a multiple of the interval; the re-written file must
        not count twice against max_keep (with max_keep=1 rotation unlinked
        the checkpoint right after writing it and ``latest`` dangled)."""
        mgr = CheckpointManager(tmp_path, prefix="ckpt", max_keep=1)
        mgr.save({"w": 1}, update=100)
        path = mgr.save({"w": 2}, update=100)
        assert path.exists() and mgr.latest == path
        assert [p.name for p in mgr._recent] == ["ckpt_100.pkl"]
        assert load_checkpoint(path, unsafe_skip_plant_validation=True)["params"] == {"w": 2}

        mgr = CheckpointManager(tmp_path / "two", prefix="ckpt", max_keep=2)
        for u in (1, 2, 3, 3):
            mgr.save({"w": u}, update=u)
        assert sorted(p.name for p in (tmp_path / "two").iterdir()) == ["ckpt_2.pkl", "ckpt_3.pkl"]

    def test_manager_tags_every_rotating_checkpoint(self, tmp_path):
        identity = _plant_identity()
        mgr = CheckpointManager(tmp_path, prefix="ckpt", plant_identity=identity)

        path = mgr.save({"w": 1}, update=1)

        assert load_checkpoint(path, current_plant=identity)[MODEL_IDENTITY_ATTRIBUTE] == identity.to_dict()
