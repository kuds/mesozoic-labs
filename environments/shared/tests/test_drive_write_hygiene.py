"""Write hygiene of the files a run keeps on Drive.

Four fixes from the review of how training runs are stored on Google Drive:

* the TensorBoard sync publishes each file atomically, under a temporary
  name TensorBoard never reads, and skips a file that has not changed since
  its last copy (``tb_sync``);
* ``collected_results.csv`` (and the CLI's appended ``curriculum_results.csv``),
  ``stage_summary.txt``, ``training_summary.txt``, the zero-action records and
  the ``ancestors/<stage_id>/`` copies are published atomically, with the
  bytes they had;
* the zero-action preflight writes no record that would differ from the
  newest one only in its time, and keeps an unchanged run copy as it is;
* ``eval_episodes.csv`` is rewritten on the checkpoint cadence and at
  training end instead of after every evaluation, and after an interrupt the
  trainer saves, all three checkpoint-cadence records are rewritten after the
  final save.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from environments.shared.model_id import ModelId

TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")
VELOCIRAPTOR_MONGOLIENSIS = ModelId("velociraptor_mongoliensis", "anatomical")


def _reclaimed_mid_copy(monkeypatch):
    """Make every ``shutil.copyfile`` write half its bytes and then fail, as a runtime reclaimed mid-copy does."""

    def half_then_fail(src, dst, *args, **kwargs):
        data = Path(src).read_bytes()
        Path(dst).write_bytes(data[: len(data) // 2])
        raise OSError("runtime reclaimed mid-copy")

    monkeypatch.setattr(shutil, "copyfile", half_then_fail)


def _reclaimed_before_publish(monkeypatch):
    """Make every ``os.replace`` fail: the runtime is reclaimed before a staged write is published."""

    def fail(*args, **kwargs):
        raise OSError("runtime reclaimed before the publish")

    monkeypatch.setattr(os, "replace", fail)


def _litter(directory: Path) -> list[str]:
    """Temporary files a failed atomic write would have stranded (it removes them)."""
    return sorted(path.name for path in directory.rglob(".*.tmp"))


# ---------------------------------------------------------------------------
# (a) TensorBoard sync
# ---------------------------------------------------------------------------


def _buffer(tmp_path: Path) -> tuple[Path, Path]:
    local = tmp_path / "buffer"
    (local / "PPO_1").mkdir(parents=True)
    (local / "PPO_1" / "events.out.tfevents.1").write_bytes(b"event-records-" * 64)
    (local / "PPO_1" / "events.out.tfevents.2").write_bytes(b"other-run-" * 64)
    return local, tmp_path / "drive" / "tensorboard"


class TestTensorBoardSync:
    def test_a_reclaim_mid_copy_leaves_the_previous_complete_event_file(self, tmp_path, monkeypatch):
        """The local buffer dies with the runtime, so the mount's copy is the only one left: it must stay whole."""
        from environments.shared.tb_sync import _sync_tb_to_gcs

        local, remote = _buffer(tmp_path)
        _sync_tb_to_gcs(local, remote, cleanup=False)
        before = {path.name: path.read_bytes() for path in (remote / "PPO_1").iterdir()}
        for path in (local / "PPO_1").iterdir():
            with path.open("ab") as handle:
                handle.write(b"more-records-" * 64)

        _reclaimed_mid_copy(monkeypatch)
        with pytest.raises(OSError, match="reclaimed"):
            _sync_tb_to_gcs(local, remote, cleanup=False)

        # Whichever file the reclaim hit, the mount still holds every previous copy whole.
        assert {path.name: path.read_bytes() for path in (remote / "PPO_1").iterdir()} == before
        assert _litter(remote) == []

    def test_a_killed_copy_strands_no_file_tensorboard_reads_and_the_next_sync_removes_it(self, tmp_path, monkeypatch):
        """A hard kill (a reclaim) runs no cleanup, so the temporary stays: it must not look like an event file."""
        from environments.shared import file_io
        from environments.shared.tb_sync import _sync_tb_to_gcs

        local, remote = _buffer(tmp_path)
        _sync_tb_to_gcs(local, remote, cleanup=False)
        before = {path.name: path.read_bytes() for path in (remote / "PPO_1").iterdir()}
        for path in (local / "PPO_1").iterdir():
            with path.open("ab") as handle:
                handle.write(b"more-records-" * 64)

        with monkeypatch.context() as killed:
            _reclaimed_mid_copy(killed)
            killed.setattr(file_io.os, "unlink", lambda path: None)  # the process dies: nothing removes the temporary
            with pytest.raises(OSError, match="reclaimed"):
                _sync_tb_to_gcs(local, remote, cleanup=False)

        names = sorted(path.name for path in (remote / "PPO_1").iterdir())
        stranded = [name for name in names if name.endswith(".tmp")]
        assert len(stranded) == 1 and stranded[0].startswith(".tbsync.")
        # TensorBoard reads every file whose name contains "tfevents" (io_wrapper.IsTensorFlowEventsFile).
        assert [name for name in names if "tfevents" in name] == sorted(before)
        assert {name: (remote / "PPO_1" / name).read_bytes() for name in before} == before

        _sync_tb_to_gcs(local, remote, cleanup=False)

        assert sorted(path.name for path in (remote / "PPO_1").iterdir()) == sorted(before)
        assert {path.name: path.read_bytes() for path in (remote / "PPO_1").iterdir()} == {
            path.name: path.read_bytes() for path in (local / "PPO_1").iterdir()
        }

    def test_the_periodic_sync_skips_a_file_unchanged_since_its_last_copy(self, tmp_path):
        pytest.importorskip("stable_baselines3")
        from environments.shared.tb_sync import PeriodicTbSyncCallback

        local, remote = _buffer(tmp_path)
        callback = PeriodicTbSyncCallback(local, remote, sync_freq=100)
        callback.num_timesteps = 100
        callback._on_step()
        changed, unchanged = remote / "PPO_1" / "events.out.tfevents.1", remote / "PPO_1" / "events.out.tfevents.2"
        assert changed.read_bytes() == (local / "PPO_1" / "events.out.tfevents.1").read_bytes()
        # Mark the mount's copies: a file the next sync copies again loses its mark.
        changed.write_bytes(b"mark")
        unchanged.write_bytes(b"mark")
        with (local / "PPO_1" / "events.out.tfevents.1").open("ab") as handle:
            handle.write(b"more-records-" * 64)

        callback.num_timesteps = 200
        callback._on_step()

        assert changed.read_bytes() == (local / "PPO_1" / "events.out.tfevents.1").read_bytes()
        assert unchanged.read_bytes() == b"mark", "an unchanged file was uploaded again"

    def test_a_file_rewritten_at_the_same_size_is_copied_again(self, tmp_path):
        """The skip keys on size AND modification time."""
        from environments.shared.tb_sync import _sync_tb_to_gcs

        local, remote = _buffer(tmp_path)
        synced: dict[str, tuple[int, int]] = {}
        _sync_tb_to_gcs(local, remote, cleanup=False, synced=synced)
        source = local / "PPO_1" / "events.out.tfevents.2"
        source.write_bytes(b"OTHER-RUN-" * 64)
        stat = source.stat()
        os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

        _sync_tb_to_gcs(local, remote, cleanup=False, synced=synced)

        assert (remote / "PPO_1" / "events.out.tfevents.2").read_bytes() == b"OTHER-RUN-" * 64

    def test_the_stage_end_sync_copies_every_file_and_drops_the_buffer(self, tmp_path):
        pytest.importorskip("stable_baselines3")
        from environments.shared.tb_sync import PeriodicTbSyncCallback, _sync_tb_to_gcs

        local, remote = _buffer(tmp_path)
        callback = PeriodicTbSyncCallback(local, remote, sync_freq=100)
        callback.num_timesteps = 100
        callback._on_step()
        expected = {path.name: path.read_bytes() for path in (local / "PPO_1").iterdir()}
        for path in (remote / "PPO_1").iterdir():
            path.write_bytes(b"lost")  # a copy that went missing or stale on the mount

        _sync_tb_to_gcs(local, remote)

        assert {path.name: path.read_bytes() for path in (remote / "PPO_1").iterdir()} == expected
        assert not local.exists()


# ---------------------------------------------------------------------------
# (b) small writes published atomically, bytes unchanged
# ---------------------------------------------------------------------------


def _old_results_csv(rows, path, *, fixed_columns=None, append=False):
    """The pre-change ``write_results_csv`` (plain writes), the reference for the bytes."""
    from environments.shared.reporting.csv_output import _compute_fieldnames

    path = Path(path)
    if append and path.exists():
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            existing_fieldnames = list(reader.fieldnames or [])
            existing_rows = list(reader)
        if [key for row in rows for key in row if key not in existing_fieldnames]:
            fieldnames = _compute_fieldnames(existing_rows + list(rows), fixed_columns)
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(existing_rows)
                writer.writerows(rows)
        else:
            with open(path, "a", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=existing_fieldnames, extrasaction="ignore").writerows(rows)
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_compute_fieldnames(rows, fixed_columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


_ROWS = [
    {"run_id": "r", "stage": 1, "ppo_learning_rate": 3e-4, "best_mean_reward": 12.5, "gate_reason": 'a, "b"'},
    {"run_id": "r", "stage": 2, "ppo_learning_rate": 1e-4, "best_mean_reward": None, "eval_x": "—"},
]


class TestResultsCsv:
    def test_the_batch_write_keeps_its_bytes(self, tmp_path):
        from environments.shared.reporting.csv_output import write_results_csv

        write_results_csv(_ROWS, tmp_path / "new.csv", fixed_columns=["run_id", "stage"])
        _old_results_csv(_ROWS, tmp_path / "old.csv", fixed_columns=["run_id", "stage"])
        assert (tmp_path / "new.csv").read_bytes() == (tmp_path / "old.csv").read_bytes()

    @pytest.mark.parametrize("new_row", [{"run_id": "r", "stage": 3}, {"run_id": "r", "stage": 3, "ppo_gamma": 0.99}])
    def test_an_append_keeps_its_bytes(self, tmp_path, new_row):
        """Appending to the old file, and rewriting it when a new column appears, write what they wrote."""
        from environments.shared.reporting.csv_output import write_results_csv

        for name, writer in (("new.csv", write_results_csv), ("old.csv", _old_results_csv)):
            writer(_ROWS[:1], tmp_path / name, fixed_columns=["run_id", "stage"], append=True)
            writer(_ROWS[1:], tmp_path / name, fixed_columns=["run_id", "stage"], append=True)
            writer([new_row], tmp_path / name, fixed_columns=["run_id", "stage"], append=True)
        assert (tmp_path / "new.csv").read_bytes() == (tmp_path / "old.csv").read_bytes()

    @pytest.mark.parametrize("append", [False, True])
    def test_a_write_that_dies_before_its_publish_leaves_the_previous_file(self, tmp_path, monkeypatch, append):
        from environments.shared.reporting.csv_output import write_results_csv

        path = tmp_path / "collected_results.csv"
        write_results_csv(_ROWS[:1], path, fixed_columns=["run_id", "stage"])
        before = path.read_bytes()

        _reclaimed_before_publish(monkeypatch)
        with pytest.raises(OSError, match="reclaimed"):
            write_results_csv(_ROWS, path, fixed_columns=["run_id", "stage"], append=append)

        assert path.read_bytes() == before
        assert _litter(tmp_path) == []


def _stage_results(**overrides):
    base = {
        "stage": 1,
        "name": "Balance",
        "description": "Stand",
        "timesteps": 1_450_000,
        "duration_seconds": 3114.1,
        "mean_reward": 281.85,
        "std_reward": 154.5,
        "mean_episode_length": 313.0,
        "std_episode_length": 142.0,
    }
    base.update(overrides)
    return base


class TestSummaryTexts:
    def test_the_stage_summary_is_published_atomically(self, tmp_path, monkeypatch):
        from environments.shared.reporting import text_summaries

        path = text_summaries.write_stage_summary(tmp_path, _stage_results(), TYRANNOSAURUS_REX, "PPO")
        before = path.read_bytes()
        assert before.endswith(b"\n") and before.decode("utf-8").startswith("Mesozoic Labs: Stage 1 Summary\n")

        _reclaimed_before_publish(monkeypatch)
        with pytest.raises(OSError, match="reclaimed"):
            text_summaries.write_stage_summary(tmp_path, _stage_results(mean_reward=1.0), TYRANNOSAURUS_REX, "PPO")

        assert path.read_bytes() == before and _litter(tmp_path) == []

    def test_the_training_summary_is_published_atomically(self, tmp_path, monkeypatch):
        from environments.shared.reporting import text_summaries

        path = text_summaries.write_training_summary(tmp_path, [_stage_results()], TYRANNOSAURUS_REX, "PPO", 42, 4)
        before = path.read_bytes()
        assert path.name == "training_summary.txt" and b"Total training time: " in before

        _reclaimed_before_publish(monkeypatch)
        with pytest.raises(OSError, match="reclaimed"):
            text_summaries.write_training_summary(tmp_path, [_stage_results()], TYRANNOSAURUS_REX, "PPO", 43, 4)

        assert path.read_bytes() == before and _litter(tmp_path) == []


class TestAncestorRecordCopies:
    def test_a_reclaim_mid_copy_leaves_no_truncated_record(self, tmp_path, monkeypatch):
        """``ancestor.json`` is written last, so a copy cut short was never rewritten: it must not exist at all."""
        from environments.shared.ancestors import find_certified_ancestor, record_ancestor
        from environments.shared.result_bundle import ANCESTORS_DIRNAME
        from environments.shared.stage_manifest import load_stage_manifest

        from .ancestors_helpers import STANCE_CURRICULUM, STANCE_TASK, build_trunk_run, trunk_plant

        trunk = tmp_path / "20260901_120000"
        trunk.mkdir()
        build_trunk_run(trunk)
        ancestor = find_certified_ancestor(
            trunk,
            model_id=TYRANNOSAURUS_REX,
            entry=load_stage_manifest(TYRANNOSAURUS_REX).by_id("stance"),
            current_task_sha256=STANCE_TASK,
            plant_identity=trunk_plant(),
            current_gate_config=STANCE_CURRICULUM,
        )
        child = tmp_path / "child"

        _reclaimed_mid_copy(monkeypatch)
        with pytest.raises(OSError, match="reclaimed"):
            record_ancestor(child, ancestor)

        target = child / ANCESTORS_DIRNAME / ancestor.stage_id
        assert sorted(path.name for path in target.iterdir()) == []


# ---------------------------------------------------------------------------
# (c) zero-action records
# ---------------------------------------------------------------------------


class _Clock:
    """``datetime`` for the preflight with a ``now()`` a test moves."""

    def __init__(self):
        self.now_value = datetime(2026, 11, 24, 9, 0, 0)

    def advance(self, seconds=3600):
        self.now_value += timedelta(seconds=seconds)

    def as_datetime_class(self):
        clock = self

        class _Frozen(datetime):
            @classmethod
            def now(cls, tz=None):
                return clock.now_value

        return _Frozen


@pytest.fixture
def preflight(tmp_path, monkeypatch):
    """Run the notebook's preflight for the velociraptor (the trained model) and the T. rex, on a moving clock."""
    from .test_zero_action_baseline import _stub_stage

    lengths = {TYRANNOSAURUS_REX: [1000, 1000, 1000, 400], VELOCIRAPTOR_MONGOLIENSIS: [100, 200, 100, 200]}
    baseline_module = _stub_stage(monkeypatch, lengths, {TYRANNOSAURUS_REX: 5000.0, VELOCIRAPTOR_MONGOLIENSIS: None})
    clock = _Clock()
    monkeypatch.setattr(baseline_module, "datetime", clock.as_datetime_class())
    log_base, run_dir = tmp_path / "logs", tmp_path / "run"
    run_dir.mkdir()

    def run():
        baseline_module.preflight(
            [TYRANNOSAURUS_REX, VELOCIRAPTOR_MONGOLIENSIS],
            stage=1,
            episodes=4,
            seed=7,
            model_id=VELOCIRAPTOR_MONGOLIENSIS,
            log_base=log_base,
            run_dir=run_dir,
        )
        clock.advance()

    return SimpleNamespace(
        run=run, log_base=log_base, run_dir=run_dir, lengths=lengths, module=baseline_module, monkeypatch=monkeypatch
    )


def _baseline_files(log_base: Path) -> list[str]:
    return sorted(str(path.relative_to(log_base)) for path in log_base.rglob("*.*"))


class TestZeroActionRecords:
    def test_a_second_session_with_the_same_baseline_writes_nothing(self, preflight, capsys):
        preflight.run()
        files = _baseline_files(preflight.log_base)
        run_copy = preflight.run_dir / "zero_action_baseline.json"
        sealed = run_copy.read_bytes()
        capsys.readouterr()

        preflight.run()

        assert files == [
            "tyrannosaurus_rex/anatomical/zero_action_baselines/20261124_090000.json",
            "velociraptor_mongoliensis/anatomical/zero_action_baselines/20261124_090000.json",
            "velociraptor_mongoliensis/anatomical/zero_action_baselines/20261124_090000.txt",
        ]
        assert _baseline_files(preflight.log_base) == files, "a record differing only in its time was written"
        assert run_copy.read_bytes() == sealed, "the run copy was rewritten with only a new captured_at"
        out = capsys.readouterr().out
        assert out.count("Unchanged: ") == 2 and "no new record" in out
        assert f"Run copy unchanged: {run_copy}" in out and "Copied to run" not in out

    def test_a_changed_baseline_writes_a_new_record_and_refreshes_the_run_copy(self, preflight):
        preflight.run()
        preflight.lengths[VELOCIRAPTOR_MONGOLIENSIS][:] = [300, 300, 300, 300]  # the plant now stands longer

        preflight.run()

        raptor = preflight.log_base / "velociraptor_mongoliensis/anatomical/zero_action_baselines"
        assert sorted(path.name for path in raptor.iterdir()) == [
            "20261124_090000.json",
            "20261124_090000.txt",
            "20261124_100000.json",
            "20261124_100000.txt",
        ]
        # The T. rex measured the same, so it gets no new record.
        trex = preflight.log_base / "tyrannosaurus_rex/anatomical/zero_action_baselines"
        assert sorted(path.name for path in trex.iterdir()) == ["20261124_090000.json"]
        run_copy = (preflight.run_dir / "zero_action_baseline.json").read_text(encoding="utf-8")
        assert run_copy == (raptor / "20261124_100000.json").read_text(encoding="utf-8")
        assert json.loads(run_copy)["captured_at"] == "2026-11-24T10:00:00"

    def test_a_changed_plant_identity_is_a_changed_baseline(self, preflight):
        preflight.run()

        class _NewPlant:
            @staticmethod
            def to_dict():
                return {"plant_revision": 2}

        preflight.monkeypatch.setattr(preflight.module, "current_plant_identity", lambda model_id: _NewPlant())
        preflight.run()

        record = json.loads((preflight.run_dir / "zero_action_baseline.json").read_text(encoding="utf-8"))
        assert record["results"]["velociraptor_mongoliensis"]["anatomical"]["plant_identity"] == {"plant_revision": 2}
        trex = preflight.log_base / "tyrannosaurus_rex/anatomical/zero_action_baselines"
        assert len(list(trex.glob("*.json"))) == 2

    @pytest.mark.parametrize("run_copy_text", [None, "{truncated", '{"captured_at": "x"}'])
    def test_a_missing_or_different_run_copy_is_written(self, preflight, run_copy_text):
        """The model-level record is unchanged, but the run has no copy of it (a new run), or not this one."""
        preflight.run()
        run_copy = preflight.run_dir / "zero_action_baseline.json"
        if run_copy_text is None:
            run_copy.unlink()
        else:
            run_copy.write_text(run_copy_text, encoding="utf-8")

        preflight.run()

        record = json.loads(run_copy.read_text(encoding="utf-8"))
        assert record["captured_at"] == "2026-11-24T10:00:00"
        assert record["results"]["velociraptor_mongoliensis"]["anatomical"]["verdict"] == "NO GATE"

    @pytest.mark.parametrize("form", ["a_nan", "an_int_for_a_float"])
    def test_a_newest_record_that_differs_only_in_form_is_not_the_same(self, preflight, form):
        """Compared as written: a NaN no record of the tool holds, or ``850`` for ``850.0``, is another record."""
        preflight.run()
        trex = preflight.log_base / "tyrannosaurus_rex/anatomical/zero_action_baselines"
        newest = trex / "20261124_090000.json"
        record = json.loads(newest.read_text(encoding="utf-8"))
        assert record["results"]["tyrannosaurus_rex"]["anatomical"]["reward_mean"] == 850.0
        record["results"]["tyrannosaurus_rex"]["anatomical"]["reward_mean"] = float("nan") if form == "a_nan" else 850
        newest.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")

        preflight.run()

        assert sorted(path.name for path in trex.iterdir()) == ["20261124_090000.json", "20261124_100000.json"]
        written = json.loads((trex / "20261124_100000.json").read_text(encoding="utf-8"))
        assert written["results"]["tyrannosaurus_rex"]["anatomical"]["reward_mean"] == 850.0

    @pytest.mark.parametrize("unreadable", ["not_utf8", "a_directory"])
    def test_an_unreadable_newest_table_is_written_again(self, preflight, unreadable):
        preflight.run()
        raptor = preflight.log_base / "velociraptor_mongoliensis/anatomical/zero_action_baselines"
        table = (raptor / "20261124_090000.txt").read_text(encoding="utf-8")
        if unreadable == "not_utf8":
            (raptor / "20261124_090000.txt").write_bytes(b"\xff\xfe not a table")
        else:
            (raptor / "20261124_093000.txt").mkdir()  # the newest name, and no file to read

        preflight.run()

        assert (raptor / "20261124_100000.txt").read_text(encoding="utf-8") == table
        assert not (raptor / "20261124_100000.json").exists(), "the record itself is unchanged"

    def test_the_readers_still_read_a_kept_copy(self, preflight):
        from environments.shared.curriculum.baseline_watch import read_zero_action_baseline

        preflight.run()
        preflight.run()

        assert read_zero_action_baseline(preflight.run_dir, VELOCIRAPTOR_MONGOLIENSIS) == pytest.approx(150.0)

    def test_the_records_keep_their_bytes_and_are_published_atomically(self, preflight, monkeypatch):
        preflight.run()
        record = preflight.log_base / "velociraptor_mongoliensis/anatomical/zero_action_baselines/20261124_090000.json"
        text = record.read_text(encoding="utf-8")
        assert text == json.dumps(json.loads(text), indent=2, sort_keys=True, allow_nan=False)  # no trailing newline
        assert (preflight.run_dir / "zero_action_baseline.json").read_text(encoding="utf-8") == text

        preflight.lengths[VELOCIRAPTOR_MONGOLIENSIS][:] = [300, 300, 300, 300]
        before = {path: path.read_bytes() for path in preflight.run_dir.rglob("*") if path.is_file()}
        _reclaimed_before_publish(monkeypatch)
        with pytest.raises(OSError, match="reclaimed"):
            preflight.run()

        assert {path: path.read_bytes() for path in preflight.run_dir.rglob("*") if path.is_file()} == before
        assert _litter(preflight.log_base) == [] and _litter(preflight.run_dir) == []


# ---------------------------------------------------------------------------
# (d) eval_episodes.csv on the checkpoint cadence
# ---------------------------------------------------------------------------


def _evaluate(callback, *, n_calls: int, timesteps: int) -> None:
    """Drive *callback* through one evaluation of four synthetic statues on call *n_calls*, as SB3 does."""
    from environments.shared import eval_diagnostics as eval_diagnostics_module
    from environments.shared.gait.pad_telemetry import RailFoot, RailLayout

    from .test_eval_diagnostics import _screen_traces, _TraceRecorder
    from .test_gait_pad_telemetry import LAYOUT

    traces = _screen_traces()
    recorder = _TraceRecorder(list(traces))
    rails = RailLayout(feet=(RailFoot(0, "r"), RailFoot(1, "l")))
    callback.n_calls = n_calls

    def evaluate():
        callback._foot_setup = [(None, None, None, LAYOUT)]
        callback._rail_layouts = [rails]
        callback._pad_recorders = {0: recorder}
        callback._pad_quota = {0: len(traces)}
        for trace in traces:
            for step in range(trace.length):
                done = step == trace.length - 1
                info = {"reward_alive": 1.0, "reward_posture": 0.5 if trace.length >= 400 else 0.25}
                if done:
                    info.update(
                        {"TimeLimit.truncated": True}
                        if trace.length >= 400
                        else {"TimeLimit.truncated": False, "termination_reason": "nosedive"}
                    )
                callback._log_success_callback(
                    {"i": 0, "done": done, "info": info, "actions": np.zeros((1, 4)), "reward": 1.0}, {}
                )
        callback.evaluations_timesteps.append(timesteps)
        callback.evaluations_results.append([float(trace.length) for trace in traces])
        callback.evaluations_length.append([trace.length for trace in traces])
        return True

    with patch.object(eval_diagnostics_module._EvalCallback, "_on_step", side_effect=evaluate):
        assert callback._on_step() is True


def _on_disk(path: Path) -> list[int]:
    """The ``timesteps`` of each row of the ledger file at *path* (``[]``: no file)."""
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return [int(row["timesteps"]) for row in csv.DictReader(handle)]


@pytest.fixture
def ledger_callback(tmp_path, monkeypatch):
    """A bare ``StageAwareEvalCallback`` evaluating every 10 calls, its ledger flushed every *flush* calls."""
    pytest.importorskip("stable_baselines3")
    from environments.shared import eval_diagnostics as eval_diagnostics_module
    from environments.shared.episode_ledger import EpisodeLedger
    from environments.shared.gait.pad_telemetry import RailFoot, RailLayout

    from .test_eval_diagnostics import _bare_stage_aware_callback
    from .test_gait_pad_telemetry import LAYOUT

    def build(flush: int | None):
        callback = _bare_stage_aware_callback(success_applicable=False, settle_steps=100)
        callback.eval_freq = 10
        callback.verbose = 0
        callback.n_eval_episodes = 4
        callback.horizon = 400
        callback.control_dt = 0.01
        callback.evaluations_results, callback.evaluations_length, callback.evaluations_timesteps = [], [], []
        callback.evaluations_screen_v2 = []
        callback._ledger = EpisodeLedger(
            tmp_path / "eval_episodes.csv", fixed_columns=eval_diagnostics_module.EVAL_LEDGER_FIXED_COLUMNS
        )
        callback._foot_resolved = True
        callback.foot_pad_keys = LAYOUT.episode_keys()
        callback.foot_rail_keys = RailLayout(feet=(RailFoot(0, "r"), RailFoot(1, "l"))).episode_keys()
        callback.ledger_flush_every_calls = flush
        return callback

    return build


class TestEvalLedgerCadence:
    def test_the_file_is_rewritten_on_the_checkpoint_calls_and_at_training_end(self, ledger_callback, tmp_path):
        path = tmp_path / "eval_episodes.csv"
        callback = ledger_callback(flush=20)

        _evaluate(callback, n_calls=10, timesteps=1000)
        assert _on_disk(path) == [], "rewritten after an evaluation between checkpoints"
        _evaluate(callback, n_calls=20, timesteps=2000)
        assert _on_disk(path) == [1000] * 4 + [2000] * 4  # the checkpoint call: its own evaluation included
        _evaluate(callback, n_calls=30, timesteps=3000)
        assert _on_disk(path) == [1000] * 4 + [2000] * 4
        assert len(callback._ledger) == 12  # every row is still held

        callback._on_training_end()  # a normal end, an early stop and a gate stop all end here

        assert _on_disk(path) == [1000] * 4 + [2000] * 4 + [3000] * 4

    def test_without_a_cadence_every_evaluation_rewrites_it(self, ledger_callback, tmp_path):
        callback = ledger_callback(flush=None)
        _evaluate(callback, n_calls=10, timesteps=1000)
        assert _on_disk(tmp_path / "eval_episodes.csv") == [1000] * 4

    def test_with_the_report_extensions_off_nothing_is_flushed(self, ledger_callback, tmp_path):
        """The inertness switch: off, the callback writes no ledger, on the cadence or at training end."""
        callback = ledger_callback(flush=20)
        _evaluate(callback, n_calls=10, timesteps=1000)
        callback.report_extensions = False

        assert callback.flush_records() is False
        callback._on_training_end()
        assert not (tmp_path / "eval_episodes.csv").exists()

    def test_a_resume_from_the_newest_checkpoint_loses_no_row_it_keeps(self, ledger_callback, tmp_path):
        """A crash after call 30 loses the 3000-step rows, which a resume from the 2000-step checkpoint drops."""
        from environments.shared import eval_diagnostics as eval_diagnostics_module
        from environments.shared.episode_ledger import EpisodeLedger

        callback = ledger_callback(flush=20)
        for n_calls in (10, 20, 30):
            _evaluate(callback, n_calls=n_calls, timesteps=100 * n_calls)
        # The runtime dies here: no training end.
        resumed = EpisodeLedger(None, fixed_columns=eval_diagnostics_module.EVAL_LEDGER_FIXED_COLUMNS)

        assert resumed.seed_prior(tmp_path / "eval_episodes.csv", max_timesteps=2000) == 8
        assert resumed.column("timesteps") == ["1000"] * 4 + ["2000"] * 4

    def test_a_cadence_below_one_call_is_refused(self):
        pytest.importorskip("stable_baselines3")
        from stable_baselines3.common.vec_env import DummyVecEnv

        from environments.shared.eval_diagnostics import StageAwareEvalCallback

        from .tiny_env_helpers import tiny_env_class

        env = DummyVecEnv([tiny_env_class()])
        try:
            with pytest.raises(ValueError, match="ledger_flush_every_calls must be at least 1"):
                StageAwareEvalCallback(env, stage=1, success_applicable=False, ledger_flush_every_calls=0)
        finally:
            env.close()

    def test_the_trainer_flushes_on_the_checkpoint_stride(self, tmp_path, monkeypatch):
        """The eval callback runs before the CheckpointCallback and flushes on its ``save_freq``."""
        pytest.importorskip("stable_baselines3")
        pytest.importorskip("mujoco")
        from stable_baselines3.common.callbacks import CheckpointCallback

        from environments.shared.eval_diagnostics import StageAwareEvalCallback

        from .test_curriculum_rollout_ledger import _core_callbacks

        callbacks = _core_callbacks(tmp_path, monkeypatch, model_id=TYRANNOSAURUS_REX)
        kinds = [type(callback) for callback in callbacks]
        evaluation = callbacks[kinds.index(StageAwareEvalCallback)]
        checkpoint = callbacks[kinds.index(CheckpointCallback)]
        assert kinds.index(StageAwareEvalCallback) < kinds.index(CheckpointCallback)
        assert evaluation.ledger_flush_every_calls == checkpoint.save_freq == 100
        assert evaluation._ledger.path == tmp_path / "logs" / "eval_episodes.csv"

    def test_the_training_ledger_and_the_policy_probe_flush_on_request(self, tmp_path):
        """The two other checkpoint-cadence records answer the trainer's ``flush_records`` too."""
        pytest.importorskip("stable_baselines3")
        from environments.shared.policy_probe import PolicyProbeCallback
        from environments.shared.rollout_ledger import TRAIN_LEDGER_FILENAME, TrainingEpisodeLedgerCallback

        ledger = TrainingEpisodeLedgerCallback(model_id=TYRANNOSAURUS_REX, ledger_dir=tmp_path, flush_every_calls=100)
        ledger._ledger.extend([{"timesteps": 100, "env": 0}, {"timesteps": 200, "env": 1}])
        assert ledger.flush_records() is True
        assert _on_disk(tmp_path / TRAIN_LEDGER_FILENAME) == [100, 200]
        assert ledger.flush_records() is False  # no row since

        probe = object.__new__(PolicyProbeCallback)
        with patch.object(PolicyProbeCallback, "write", return_value=True) as write:
            assert probe.flush_records() is True
        write.assert_called_once_with()


class _CadenceRecord:
    """A callback holding a checkpoint-cadence record: ``flush_records`` logs itself, then raises *raises*."""

    def __init__(self, name: str, events: list, raises: BaseException | None = None):
        self.name, self.events, self.raises = name, events, raises

    def flush_records(self) -> bool:
        self.events.append(f"flush {self.name}")
        if self.raises is not None:
            raise self.raises
        return True


class TestTheTrainerFlushesAfterASavedInterrupt:
    """``learn()`` skips ``on_training_end`` on a KeyboardInterrupt; a saved stage can be re-judged."""

    def _train(self, tmp_path, monkeypatch, *, interrupt=True, eval_raises=None):
        from environments.shared import train_base
        from environments.shared.config import load_all_stages

        from .test_train_base import TestOneStageBody as _StageBody

        harness = _StageBody()
        _model, eval_callback = harness._patch(monkeypatch, interrupt=interrupt)
        events = harness.record["events"]
        records = [
            _CadenceRecord("eval", events, raises=eval_raises),
            _CadenceRecord("train ledger", events),
            _CadenceRecord("probe", events),
        ]
        monkeypatch.setattr(
            train_base, "_build_core_callbacks", lambda *args, **kwargs: (["core", *records], eval_callback, None)
        )

        train_base.train(
            SimpleNamespace(model_id=VELOCIRAPTOR_MONGOLIENSIS, env_class=object),
            load_all_stages(VELOCIRAPTOR_MONGOLIENSIS),
            1,
            total_timesteps=10,
            seed=harness.SEED,
            output_dir=str(tmp_path / "01_stance"),
            use_tensorboard=False,
            verbose=0,
        )
        return events

    def test_every_record_is_rewritten_after_the_final_save(self, tmp_path, monkeypatch):
        """After it: a second Ctrl-C during a rewrite (seconds on a 12M T. rex stance) cannot cost the final pair."""
        events = self._train(tmp_path, monkeypatch)

        flushes = [event for event in events if isinstance(event, str) and event.startswith("flush")]
        assert flushes == ["flush eval", "flush train ledger", "flush probe"]
        assert events.index("learn") < events.index("final save") < events.index("flush eval")

    def test_a_second_ctrl_c_during_a_rewrite_keeps_the_saved_stage(self, tmp_path, monkeypatch, caplog):
        with caplog.at_level("WARNING", logger="environments.shared.train_base"):
            events = self._train(tmp_path, monkeypatch, eval_raises=KeyboardInterrupt())

        assert events.index("final save") < events.index("flush eval")
        assert "flush train ledger" not in events and "flush probe" not in events
        assert "the final checkpoint pair is saved" in caplog.text

    def test_a_failed_rewrite_is_logged_and_the_others_still_run(self, tmp_path, monkeypatch, caplog):
        with caplog.at_level("WARNING", logger="environments.shared.train_base"):
            events = self._train(tmp_path, monkeypatch, eval_raises=OSError("mount gone"))

        assert [event for event in events if isinstance(event, str) and event.startswith("flush")] == [
            "flush eval",
            "flush train ledger",
            "flush probe",
        ]
        assert "could not be rewritten after the interrupt" in caplog.text

    def test_a_stage_that_ends_normally_is_left_to_on_training_end(self, tmp_path, monkeypatch):
        events = self._train(tmp_path, monkeypatch, interrupt=False)

        assert not [event for event in events if isinstance(event, str) and event.startswith("flush")]
