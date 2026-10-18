"""The checkpoint archive rule: keep the newest periodic checkpoint of every interval of A steps.

``max_checkpoints`` keeps the newest few periodic checkpoints, so when a
posture set in mid-run the checkpoints from before it were gone.  With
``checkpoint_archive_every_timesteps = A`` the prune also keeps, of the
older ones, the largest step present in each interval ``(k A, (k+1) A]``.
The prune parts are pure and run without stable-baselines3.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from environments.shared.curriculum.checkpoints import (
    CHECKPOINT_ARCHIVE_KEY,
    archived_steps,
    checkpoint_archive_every,
    prune_periodic_checkpoints,
)

PREFIX = "stage1"


def _save(directory: Path, step: int, *, vecnormalize: bool = True) -> None:
    (directory / f"{PREFIX}_{step}_steps.zip").write_bytes(b"policy")
    if vecnormalize:
        (directory / f"{PREFIX}_vecnormalize_{step}_steps.pkl").write_bytes(b"stats")


def _present(directory: Path) -> list[int]:
    """The periodic steps present, by name (an unparsable name is not one)."""
    stems = (path.name[len(PREFIX) + 1 : -len("_steps.zip")] for path in directory.glob(f"{PREFIX}_*_steps.zip"))
    return sorted(int(stem) for stem in stems if stem.isdigit())


def _main_prune(directory: Path, max_checkpoints: int) -> set[str]:
    """A frozen copy of the base's rule (``2e734e29``), on names: what it would delete."""
    if max_checkpoints <= 0:
        return set()
    by_step: dict[int, list[str]] = {}
    for infix, extension in (("", "zip"), ("vecnormalize_", "pkl"), ("replay_buffer_", "pkl")):
        for path in directory.glob(f"{PREFIX}_{infix}*_steps.{extension}"):
            remainder = path.name[len(f"{PREFIX}_{infix}") : -len(f"_steps.{extension}")]
            if remainder.isdigit():
                by_step.setdefault(int(remainder), []).append(path.name)
    doomed = sorted(by_step, reverse=True)[max_checkpoints:]
    return {name for step in doomed for name in by_step[step]}


class TestTheTwelveMillionStepRun:
    """The notebook's stance cadence: a checkpoint every 100,000 steps to 12,000,000, A = 500,000, N = 5."""

    def test_it_keeps_one_per_half_million_and_the_newest_five(self, tmp_path):
        cadence, archive, newest = 100_000, 500_000, 5
        for step in range(cadence, 12_000_000 + 1, cadence):
            _save(tmp_path, step)
            prune_periodic_checkpoints(tmp_path, PREFIX, newest, archive_every_timesteps=archive)
            present = _present(tmp_path)
            saved = range(cadence, step + 1, cadence)
            interval_final = {
                max(s for s in saved if (s - 1) // archive == k) for k in {(s - 1) // archive for s in saved}
            }
            assert set(present) == interval_final | set(sorted(saved)[-newest:]), step
        kept = _present(tmp_path)
        assert kept == [*range(500_000, 11_500_000 + 1, 500_000), *range(11_600_000, 12_000_000 + 1, 100_000)]
        assert len(kept) == 28
        assert len(list(tmp_path.glob(f"{PREFIX}_vecnormalize_*_steps.pkl"))) == 28  # every pair whole

    def test_a_cadence_longer_than_the_interval_keeps_every_checkpoint(self, tmp_path):
        steps = [k * 1_048_576 for k in range(1, 12)]
        for step in steps:
            _save(tmp_path, step)
            prune_periodic_checkpoints(tmp_path, PREFIX, 5, archive_every_timesteps=500_000)
        assert _present(tmp_path) == steps

    def test_the_kept_set_is_a_function_of_the_names_so_a_resume_prunes_alike(self, tmp_path):
        steps = list(range(100_000, 3_000_001, 100_000))
        assert archived_steps(steps, 500_000) == archived_steps(list(reversed(steps)), 500_000)
        assert archived_steps(steps, 500_000) == {500_000, 1_000_000, 1_500_000, 2_000_000, 2_500_000, 3_000_000}
        assert archived_steps({1, 500_000, 500_001}, 500_000) == {500_000, 500_001}
        assert archived_steps(steps, 0) == set()


class TestWithoutAnArchive:
    def test_it_deletes_what_the_base_deleted_on_random_step_sets(self, tmp_path):
        rng = np.random.default_rng(627)
        for trial in range(200):
            directory = tmp_path / str(trial)
            directory.mkdir()
            for step in rng.choice(np.arange(1, 5_000), size=int(rng.integers(0, 15)), replace=False):
                _save(directory, int(step), vecnormalize=bool(rng.integers(0, 2)))
            keep = int(rng.integers(0, 8))
            expected = _main_prune(directory, keep)
            removed = prune_periodic_checkpoints(directory, PREFIX, keep, archive_every_timesteps=0)
            assert {path.name for path in removed} == expected, trial


class TestWhatIsNeverTouched:
    def test_the_handoff_pairs_and_unparsable_names_stay_and_a_sidecar_goes_with_its_zip(self, tmp_path):
        for step in (100, 200, 300, 400, 650):
            _save(tmp_path, step)
        bystanders = [
            "best_model.zip",
            "best_model_vecnorm.pkl",
            "robust_best_model.zip",
            "robust_best_model_vecnorm.pkl",
            "stage1_final.zip",
            "stage1_final_vecnorm.pkl",
            "stage1_x_100_steps.zip",
            "stage1_100x_steps.zip",
        ]
        for name in bystanders:
            (tmp_path / name).write_bytes(b"")
        removed = prune_periodic_checkpoints(tmp_path, PREFIX, 1, archive_every_timesteps=250)
        assert sorted(path.name for path in removed) == [
            "stage1_100_steps.zip",
            "stage1_300_steps.zip",
            "stage1_vecnormalize_100_steps.pkl",
            "stage1_vecnormalize_300_steps.pkl",
        ]
        assert _present(tmp_path) == [200, 400, 650]
        assert all((tmp_path / name).exists() for name in bystanders)
        for step in _present(tmp_path):
            assert (tmp_path / f"{PREFIX}_vecnormalize_{step}_steps.pkl").exists()

    def test_max_checkpoints_zero_still_keeps_everything(self, tmp_path):
        for step in (100, 200, 300):
            _save(tmp_path, step)
        assert prune_periodic_checkpoints(tmp_path, PREFIX, 0, archive_every_timesteps=150) == []
        assert _present(tmp_path) == [100, 200, 300]


class TestTheKey:
    def test_it_is_registered_as_a_retention_key(self):
        from environments.shared.curriculum.gate_schema import _RETENTION_KEYS

        assert CHECKPOINT_ARCHIVE_KEY == "checkpoint_archive_every_timesteps"
        assert {"max_checkpoints", CHECKPOINT_ARCHIVE_KEY} == _RETENTION_KEYS

    def test_a_stage_may_declare_it(self):
        from environments.shared.config import load_stage_config
        from environments.shared.curriculum.gate_schema import validate_gate_config
        from environments.shared.model_id import ModelId

        curriculum = dict(load_stage_config(ModelId("tyrannosaurus_rex", "anatomical"), "stance")["curriculum_kwargs"])
        curriculum[CHECKPOINT_ARCHIVE_KEY] = 500_000
        assert validate_gate_config("stance", curriculum) == curriculum["gate_kind"]

    @pytest.mark.parametrize("value", [True, 500_000.0, -1, "500000", None])
    def test_a_bool_a_non_integer_or_a_negative_value_is_refused(self, value):
        with pytest.raises(ValueError, match=CHECKPOINT_ARCHIVE_KEY):
            checkpoint_archive_every({CHECKPOINT_ARCHIVE_KEY: value})

    def test_absent_it_is_zero(self):
        assert checkpoint_archive_every({}) == 0 and checkpoint_archive_every(None) == 0
        assert checkpoint_archive_every({CHECKPOINT_ARCHIVE_KEY: np.int64(500_000)}) == 500_000

    def test_the_trainer_hands_it_to_the_retention_callback(self, tmp_path, monkeypatch):
        pytest.importorskip("stable_baselines3")
        import tempfile

        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.vec_env import DummyVecEnv

        from environments.shared.curriculum import CheckpointRetentionCallback
        from environments.shared.train_base import _build_core_callbacks

        from .tiny_env_helpers import tiny_env_class

        scratch = tmp_path / "scratch"
        scratch.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(scratch))

        def retention(curriculum):
            eval_env = DummyVecEnv([tiny_env_class()])
            try:
                callbacks, _, _ = _build_core_callbacks(
                    {"CheckpointCallback": CheckpointCallback},
                    eval_env,
                    tmp_path / "models",
                    tmp_path / "logs",
                    1,
                    1,
                    100,
                    100,
                    0,
                    {"curriculum_kwargs": {"min_avg_reward": 1.0, **curriculum}},
                )
            finally:
                eval_env.close()
            return next(callback for callback in callbacks if isinstance(callback, CheckpointRetentionCallback))

        assert retention({}).archive_every_timesteps == 0
        assert retention({CHECKPOINT_ARCHIVE_KEY: 500_000}).archive_every_timesteps == 500_000
        with pytest.raises(ValueError, match=CHECKPOINT_ARCHIVE_KEY):
            retention({CHECKPOINT_ARCHIVE_KEY: 0.5})

    def test_the_callback_prunes_with_it(self, tmp_path):
        pytest.importorskip("stable_baselines3")
        from environments.shared.curriculum import CheckpointRetentionCallback

        callback = CheckpointRetentionCallback(
            model_dir=tmp_path, name_prefix=PREFIX, save_freq=1, max_checkpoints=1, archive_every_timesteps=300
        )
        for step in (100, 200, 300, 400, 500):
            _save(tmp_path, step)
        callback._prune()
        assert _present(tmp_path) == [300, 500]
