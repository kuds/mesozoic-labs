"""Tests for environments.shared.reporting.runs_summary: the offline-judgement ledger and ``runs_summary.csv``.

The ledger (``results/offline_judgements.csv``) is read by the Drive summary notebook, whose
executed tests are in ``test_drive_summary_notebook.py``; these need neither pandas nor a
notebook.  The first three tests check the committed ledger, its documentation and that a change
to it runs CI, so a pull request that adds rows is checked by them.
"""

from __future__ import annotations

import csv
import math
import os
import re

import numpy as np
import pytest

from environments.shared.paths import REPOSITORY_ROOT
from environments.shared.reporting import (
    CSV_METRIC_COLUMNS,
    OFFLINE_JUDGEMENT_COLUMNS,
    OFFLINE_JUDGEMENTS_PATH,
    OfflineJudgementError,
    load_offline_judgements,
    merge_runs_summary,
    source_counts,
    write_runs_summary,
)
from environments.shared.reporting.runs_summary import RUNS_SUMMARY_FIXED_COLUMNS
from environments.shared.tests.ci_workflow_helpers import glob_matches, path_filters

SHA256 = "sha256:" + "0123456789abcdef" * 4
#: A synthetic judgement: every column filled with a valid value.
VALID = {
    "judged_on": "2026-10-17",
    "species": "tyrannosaurus_rex",
    "embodiment": "anatomical",
    "algorithm": "PPO",
    "run_id": "tyrannosaurus_rex-anatomical-stable-baselines3-ppo-test",
    "run_path": "ppo/20260101_000000",
    "stage": "1",
    "checkpoint_step": "6291456",
    "checkpoint_sha256": SHA256,
    "gate_kind": "stance_quality_heading/v1",
    "verdict": "PASS",
    "passing_count": "39",
    "panel_size": "40",
    "lower_bound": "0.8817",
    "lower_bound_bar": "0.8",
    "binding_bar": "max_sole_cop_fore_aft 0.80",
    "panel_seeds": "3042-3081",
    "judging_commit": "2e734e29",
    "judged_by": "environments/shared/scripts/stance_gate_report.py",
    "evidence": "docs/investigations/example/certificate.json",
    "note": "a synthetic row, with a comma",
    "source": "offline",
}


def _ledger(tmp_path, *rows, header=OFFLINE_JUDGEMENT_COLUMNS):
    """A ledger file: *header*, then each row (a dict over the schema, or a list of cells)."""
    path = tmp_path / "offline_judgements.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow([row.get(column, "") for column in header] if isinstance(row, dict) else row)
    return path


def _refusal(tmp_path, *rows, **kwargs) -> str:
    with pytest.raises(OfflineJudgementError) as raised:
        load_offline_judgements(_ledger(tmp_path, *rows, **kwargs))
    return str(raised.value)


class TestTheCommittedLedger:
    def test_it_is_valid_and_its_header_is_the_schema(self):
        assert OFFLINE_JUDGEMENTS_PATH == REPOSITORY_ROOT / "results" / "offline_judgements.csv"
        first_line = OFFLINE_JUDGEMENTS_PATH.read_text(encoding="utf-8").splitlines()[0]
        assert first_line == ",".join(OFFLINE_JUDGEMENT_COLUMNS)
        assert all(row["source"] == "offline" for row in load_offline_judgements())

    def test_the_results_readme_documents_every_column(self):
        readme = (REPOSITORY_ROOT / "results" / "README.md").read_text(encoding="utf-8")
        documented = re.findall(r"^\| `([a-z0-9_]+)` \|", readme, flags=re.MULTILINE)
        assert documented == list(OFFLINE_JUDGEMENT_COLUMNS)

    def test_a_change_to_it_triggers_the_workflow(self):
        relative = OFFLINE_JUDGEMENTS_PATH.relative_to(REPOSITORY_ROOT).as_posix()
        for patterns in path_filters():
            assert any(glob_matches(pattern, relative) for pattern in patterns), (
                f"{relative} is missing from a python-ci.yml paths filter, so a PR adding rows skips their check"
            )


class TestLoad:
    def test_a_row_is_read_typed(self, tmp_path):
        (row,) = load_offline_judgements(_ledger(tmp_path, VALID))
        assert row == {
            **VALID,
            "stage": 1,
            "checkpoint_step": 6291456,
            "passing_count": 39,
            "panel_size": 40,
            "lower_bound": 0.8817,
            "lower_bound_bar": 0.8,
        }
        assert list(row) == list(OFFLINE_JUDGEMENT_COLUMNS)

    def test_a_semantic_stage_keeps_its_id_and_empty_optional_cells_read_none(self, tmp_path):
        optional = ("run_path", "passing_count", "panel_size", "lower_bound", "lower_bound_bar", "binding_bar", "note")
        sparse = {**VALID, "stage": "recovery", "verdict": "WALKS", **dict.fromkeys(optional, "")}
        (row,) = load_offline_judgements(_ledger(tmp_path, sparse))
        assert row["stage"] == "recovery"
        assert all(row[column] is None for column in optional)
        (row,) = load_offline_judgements(_ledger(tmp_path, {**VALID, "run_id": ""}))
        assert row["run_id"] is None and row["run_path"] == VALID["run_path"]

    @pytest.mark.parametrize("text", ["", "\n\n", "\r\n \r\n"])
    def test_an_empty_file_has_no_rows(self, tmp_path, text):
        path = tmp_path / "offline_judgements.csv"
        path.write_text(text, encoding="utf-8")
        assert load_offline_judgements(path) == []

    def test_a_header_alone_and_blank_lines_have_no_rows(self, tmp_path):
        path = _ledger(tmp_path, [], ["", ""])
        assert load_offline_judgements(path) == []

    def test_cells_are_stripped_and_a_byte_order_mark_is_not_a_column_name(self, tmp_path):
        path = tmp_path / "offline_judgements.csv"
        cells = [f" {VALID[column]} " if VALID[column] and column != "note" else VALID[column] for column in VALID]
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            csv.writer(handle).writerows([OFFLINE_JUDGEMENT_COLUMNS, cells])
        (row,) = load_offline_judgements(path)
        assert row["judged_on"] == "2026-10-17" and row["checkpoint_sha256"] == SHA256

    def test_a_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_offline_judgements(tmp_path / "absent.csv")

    @pytest.mark.parametrize(
        "header,detail",
        [
            ((*OFFLINE_JUDGEMENT_COLUMNS, "seed"), "unknown ['seed']"),
            (OFFLINE_JUDGEMENT_COLUMNS[:-1], "missing ['source']"),
            (tuple("verdikt" if c == "verdict" else c for c in OFFLINE_JUDGEMENT_COLUMNS), "unknown ['verdikt']"),
            (tuple(reversed(OFFLINE_JUDGEMENT_COLUMNS)), "repeated or reordered columns"),
            ((*OFFLINE_JUDGEMENT_COLUMNS, "note"), "repeated or reordered columns"),
        ],
        ids=["unknown", "missing", "misspelt", "reordered", "repeated"],
    )
    def test_the_header_must_be_the_schema_exactly(self, tmp_path, header, detail):
        message = _refusal(tmp_path, header=header)
        assert "the header must be exactly judged_on, species," in message and detail in message

    @pytest.mark.parametrize(
        "column,value,fragment",
        [
            ("judged_on", "17/10/2026", "judged_on must be a date written YYYY-MM-DD"),
            ("judged_on", "2026-02-30", "judged_on must be a date written YYYY-MM-DD"),
            ("judged_on", "20261017", "judged_on must be a date written YYYY-MM-DD"),
            ("species", "tyrannosaurus_imperator", "name no model of the species manifest: Unknown species"),
            ("embodiment", "robot", "tyrannosaurus_rex has no 'robot' embodiment"),
            ("algorithm", "ppo", "algorithm must be the public algorithm label, PPO or SAC, got 'ppo'"),
            ("algorithm", "MirrorKLPPO", "algorithm must be the public algorithm label"),
            ("stage", "stance", "stage must be written as its key, as collected_results.csv writes it: '1'"),
            ("stage", "01", "stage must be written as its key"),
            ("stage", "walk", "stage 'walk' is not a stage of tyrannosaurus_rex/anatomical"),
            ("checkpoint_step", "-1", "checkpoint_step must be a non-negative integer"),
            ("checkpoint_step", "6.3e6", "checkpoint_step must be a non-negative integer"),
            (
                "checkpoint_sha256",
                SHA256.removeprefix("sha256:"),
                "checkpoint_sha256 must be the checkpoint file's digest",
            ),
            ("checkpoint_sha256", SHA256.replace("abc", "ABC"), "sha256:<64 lowercase hex>"),
            ("checkpoint_sha256", SHA256[:-1], "checkpoint_sha256 must be the checkpoint file's digest"),
            ("gate_kind", "stance quality", "gate_kind must be one word"),
            ("verdict", "pass", "verdict must be an upper-case word"),
            ("passing_count", "39.5", "passing_count must be a non-negative integer"),
            ("panel_size", "0", "panel_size must be a positive integer"),
            ("passing_count", "41", "passing_count 41 exceeds panel_size 40"),
            ("lower_bound", "1.2", "lower_bound must be a number from 0 to 1"),
            ("lower_bound", "nan", "lower_bound must be a number from 0 to 1"),
            ("lower_bound_bar", "high", "lower_bound_bar must be a number from 0 to 1"),
            ("judging_commit", "main", "judging_commit must be a commit"),
            ("judging_commit", "2E734E29", "judging_commit must be a commit"),
            ("source", "training", "source must be 'offline', got 'training'"),
            ("judged_by", "", "judged_by is required"),
            ("evidence", " ", "evidence is required"),
            ("panel_seeds", "", "panel_seeds is required"),
        ],
    )
    def test_a_bad_cell_is_refused(self, tmp_path, column, value, fragment):
        assert fragment in _refusal(tmp_path, {**VALID, column: value})

    def test_a_row_names_its_run(self, tmp_path):
        message = _refusal(tmp_path, {**VALID, "run_id": "", "run_path": ""})
        assert "run_id or run_path is required" in message

    def test_a_recovery_stage_is_refused_on_a_model_without_one(self, tmp_path):
        row = {**VALID, "species": "velociraptor_mongoliensis", "stage": "recovery"}
        assert "stage 'recovery' is not a stage of velociraptor_mongoliensis/anatomical" in _refusal(tmp_path, row)

    def test_a_row_of_the_wrong_length_is_refused(self, tmp_path):
        cells = [VALID[column] for column in OFFLINE_JUDGEMENT_COLUMNS]
        message = _refusal(tmp_path, cells[:-1], [*cells, "extra"])
        assert "line 2: 21 cells, the header has 22" in message
        assert "line 3: 23 cells, the header has 22" in message

    def test_a_repeated_row_is_refused(self, tmp_path):
        assert "line 3: repeats line 2" in _refusal(tmp_path, VALID, VALID)

    def test_every_problem_is_listed_with_its_line(self, tmp_path):
        message = _refusal(tmp_path, VALID, {**VALID, "verdict": "pass", "panel_size": "0"}, {**VALID, "source": ""})
        assert message.splitlines()[1:] == [
            "line 3: verdict must be an upper-case word (PASS, FAIL, or the criteria's own outcome), got 'pass'",
            "line 3: panel_size must be a positive integer, got '0'",
            "line 4: source is required",
        ]


TRAINING = {
    "species": "tyrannosaurus_rex",
    "embodiment": "anatomical",
    "algorithm": "PPO",
    "source": "training",
    "run_dir": "ppo/20260101_000000",
    "run_path": "ppo/20260101_000000",
    "stage": 1,
    "stage_name": "stance",
    "ppo_learning_rate": 3e-4,
    "best_mean_reward": np.float64(812.5),
    "mean_forward_vel": float("nan"),
    "stage_passed": False,
    "eval_heading": 0.5,
}
SWEEP = {**TRAINING, "source": "sweep", "run_dir": "sweeps/stage1_ppo_20260315_101500", "trial_id": "t1"}


def _judgement(tmp_path, **cells):
    (row,) = load_offline_judgements(_ledger(tmp_path, {**VALID, **cells}))
    return row


class TestMerge:
    def test_the_table_rows_then_the_ledger_rows(self, tmp_path):
        passed, walked = _judgement(tmp_path), _judgement(tmp_path, verdict="WALKS", stage="2", run_path="")
        failed = _judgement(tmp_path, verdict="FAIL")
        merged = merge_runs_summary([TRAINING, SWEEP], [passed, walked, failed])
        assert [row["source"] for row in merged] == ["training", "sweep", "offline", "offline", "offline"]
        assert [row["stage_passed"] for row in merged] == [False, False, True, None, False]
        assert [row["run_dir"] for row in merged[2:]] == [VALID["run_path"], None, VALID["run_path"]]
        assert merged[2]["stage"] == 1 and merged[3]["stage"] == 2
        assert source_counts(merged) == {"training": 1, "sweep": 1, "offline": 3}

    def test_missing_values_read_none_and_numpy_scalars_plain(self):
        (row,) = merge_runs_summary([TRAINING])
        assert row["mean_forward_vel"] is None
        assert row["best_mean_reward"] == 812.5 and type(row["best_mean_reward"]) is float
        assert merge_runs_summary([{"source": "training", "seed": np.int64(42)}])[0]["seed"] == 42

    def test_the_offline_label_is_the_ledgers_alone(self, tmp_path):
        with pytest.raises(ValueError, match="a table row claims source 'offline'"):
            merge_runs_summary([{**TRAINING, "source": "offline"}])
        with pytest.raises(ValueError, match="an offline judgement must carry source 'offline'"):
            merge_runs_summary([], [{**_judgement(tmp_path), "source": "training"}])

    def test_nothing_merges_to_nothing(self):
        assert merge_runs_summary([], []) == [] and source_counts([]) == {}


def _read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


class TestWrite:
    def test_the_columns_are_the_union_with_the_ledger_block_after_the_identity(self, tmp_path):
        rows = merge_runs_summary([TRAINING, SWEEP], [_judgement(tmp_path)])
        header, written = _read(write_runs_summary(rows, tmp_path / "runs_summary.csv"))
        fixed = len(RUNS_SUMMARY_FIXED_COLUMNS)
        assert header[:fixed] == list(RUNS_SUMMARY_FIXED_COLUMNS)
        block = header[fixed : fixed + 14]
        assert block[:3] == ["judged_on", "checkpoint_step", "checkpoint_sha256"] and block[-1] == "note"
        others = header[fixed + 14 : header.index(CSV_METRIC_COLUMNS[0])]
        assert others == ["ppo_learning_rate", "run_id", "run_path"]
        assert header[-len(CSV_METRIC_COLUMNS) - 1 : -1] == CSV_METRIC_COLUMNS and header[-1] == "eval_heading"
        assert len(header) == len(set(header))
        offline = written[2]
        assert offline["source"] == "offline" and offline["stage_passed"] == "True"
        assert offline["gate_kind"] == "stance_quality_heading/v1" and offline["note"] == VALID["note"]
        assert offline["best_mean_reward"] == "" and offline["ppo_learning_rate"] == ""
        assert written[0]["judged_on"] == "" and written[0]["mean_forward_vel"] == ""
        assert written[0]["best_mean_reward"] == "812.5" and written[0]["stage_passed"] == "False"

    def test_without_an_offline_row_the_ledger_block_is_left_out(self, tmp_path):
        header, _ = _read(write_runs_summary(merge_runs_summary([TRAINING]), tmp_path / "runs_summary.csv"))
        assert "judged_on" not in header and "verdict" not in header
        assert header[: len(RUNS_SUMMARY_FIXED_COLUMNS)] == list(RUNS_SUMMARY_FIXED_COLUMNS)

    def test_no_rows_write_the_header_alone_and_the_directory_is_made(self, tmp_path):
        path = write_runs_summary([], tmp_path / "logs" / "model" / "runs_summary.csv")
        header, written = _read(path)
        assert written == [] and header == [*RUNS_SUMMARY_FIXED_COLUMNS, *CSV_METRIC_COLUMNS]

    def test_a_failed_publish_keeps_the_previous_file_and_strands_no_temporary(self, tmp_path, monkeypatch):
        path = tmp_path / "runs_summary.csv"
        write_runs_summary(merge_runs_summary([TRAINING]), path)
        before = path.read_bytes()

        def reclaimed(src, dst):
            raise OSError("runtime reclaimed")

        monkeypatch.setattr(os, "replace", reclaimed)
        with pytest.raises(OSError, match="runtime reclaimed"):
            write_runs_summary(merge_runs_summary([TRAINING, SWEEP]), path)
        assert path.read_bytes() == before
        assert sorted(entry.name for entry in tmp_path.iterdir()) == ["runs_summary.csv"]

    def test_nan_reaching_the_writer_directly_is_written_empty(self, tmp_path):
        _, written = _read(write_runs_summary([{"source": "training", "seed": math.nan}], tmp_path / "r.csv"))
        assert written[0]["seed"] == ""
