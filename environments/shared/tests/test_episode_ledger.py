"""Tests for environments.shared.episode_ledger: one CSV row per episode, held by column, written atomically."""

from __future__ import annotations

import csv
import logging
import os

import numpy as np
import pytest

from environments.shared.episode_ledger import EpisodeLedger, format_cell


def _read(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_a_cell_is_repr_of_a_float_an_int_as_an_int_and_text_verbatim():
    assert (
        format_cell(0.1) == "0.1" and format_cell(np.float64(0.25)) == "0.25" and format_cell(np.float32(0.5)) == "0.5"
    )
    assert format_cell(float("nan")) == "nan" and format_cell(float("inf")) == "inf"
    assert format_cell(3) == 3 and format_cell(np.int64(7)) == 7 and format_cell(True) == 1
    assert format_cell("max_sole_cop_fore_aft;horizon") == "max_sole_cop_fore_aft;horizon"
    assert format_cell(None) == ""


def test_rows_are_held_by_column_and_a_late_column_reads_blank_on_the_earlier_rows(tmp_path):
    ledger = EpisodeLedger(tmp_path / "episodes.csv", fixed_columns=("timesteps", "episode"))
    ledger.extend([{"timesteps": 100, "episode": 0, "x": 0.5}])
    ledger.extend([{"timesteps": 200, "episode": 0, "y": float("nan")}, {"timesteps": 200, "episode": 1}])
    assert len(ledger) == 3 and ledger.columns == ("timesteps", "episode", "x", "y")
    assert ledger.column("x") == [0.5, "", ""] and ledger.column("timesteps") == [100, 200, 200]
    assert ledger.write() is True
    rows = _read(tmp_path / "episodes.csv")
    assert rows == [
        {"timesteps": "100", "episode": "0", "x": "0.5", "y": ""},
        {"timesteps": "200", "episode": "0", "x": "", "y": "nan"},
        {"timesteps": "200", "episode": "1", "x": "", "y": ""},
    ]


def test_it_writes_only_when_a_row_was_added_and_never_without_a_path(tmp_path):
    path = tmp_path / "episodes.csv"
    ledger = EpisodeLedger(path, fixed_columns=("timesteps",))
    assert ledger.write() is False and not path.exists()  # nothing added: no file at all
    ledger.extend([{"timesteps": 1}])
    assert ledger.write() is True
    stamp = path.stat().st_mtime_ns
    assert ledger.write() is False and path.stat().st_mtime_ns == stamp
    memory = EpisodeLedger(None, fixed_columns=("timesteps",))
    memory.extend([{"timesteps": 1}])
    assert memory.write() is False and len(memory) == 1


def test_a_failed_write_keeps_the_previous_file_and_the_rows_and_warns_once(tmp_path, monkeypatch, caplog):
    path = tmp_path / "episodes.csv"
    ledger = EpisodeLedger(path, fixed_columns=("timesteps",))
    ledger.extend([{"timesteps": 1}])
    ledger.write()
    before = path.read_bytes()

    def refuse(source, destination):
        raise OSError("the Drive mount went away")

    with monkeypatch.context() as patched:
        patched.setattr(os, "replace", refuse)
        with caplog.at_level(logging.WARNING, logger="environments.shared.episode_ledger"):
            for step in (2, 3):
                ledger.extend([{"timesteps": step}])
                assert ledger.write() is False
    assert path.read_bytes() == before
    assert sum("episodes.csv" in record.getMessage() for record in caplog.records) == 1
    assert ledger.write() is True  # the rows were kept: the next write carries them
    assert [row["timesteps"] for row in _read(path)] == ["1", "2", "3"]


@pytest.mark.parametrize(("max_timesteps", "kept"), [(None, 3), (200, 2), (99, 0)])
def test_a_resume_keeps_the_prior_rows_at_or_before_the_resumed_step_verbatim(tmp_path, max_timesteps, kept):
    prior = EpisodeLedger(tmp_path / "episodes.csv", fixed_columns=("timesteps", "episode"))
    prior.extend(
        [
            {"timesteps": 100, "episode": 0, "x": 0.1},
            {"timesteps": 200, "episode": 0, "x": float("nan")},
            {"timesteps": 300, "episode": 0, "x": 1 / 3},
        ]
    )
    prior.write()
    resumed = EpisodeLedger(tmp_path / "resumed.csv", fixed_columns=("timesteps", "episode"))
    assert resumed.seed_prior(tmp_path / "episodes.csv", max_timesteps=max_timesteps) == kept
    assert len(resumed) == kept and resumed.write() is False  # seeding adds no row of this session
    resumed.extend([{"timesteps": 400, "episode": 0, "x": 0.5}])
    resumed.write()
    rows = _read(tmp_path / "resumed.csv")
    assert [row["timesteps"] for row in rows] == ["100", "200", "300"][:kept] + ["400"]
    assert [row["x"] for row in rows] == ["0.1", "nan", repr(1 / 3)][:kept] + ["0.5"]


def test_no_prior_file_or_an_unreadable_one_seeds_nothing(tmp_path):
    ledger = EpisodeLedger(None, fixed_columns=("timesteps",))
    assert ledger.seed_prior(tmp_path / "absent.csv", max_timesteps=None) == 0
    (tmp_path / "bad.csv").write_text("timesteps\nnot a number\n7\n", encoding="utf-8")
    assert ledger.seed_prior(tmp_path / "bad.csv", max_timesteps=None) == 1  # the unreadable row is dropped
    assert ledger.column("timesteps") == ["7"]
