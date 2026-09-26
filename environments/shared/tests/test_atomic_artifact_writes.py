"""Every run-tree record CU-3 converted is published atomically, with its old bytes.

A Colab runtime reclaimed in the middle of an in-place write leaves a
truncated JSON or CSV on the Drive mount, and three sidecar writers used to
strand a fixed ``<name>.json.tmp`` that the result-bundle manifest hashes into
the bundle instead of discarding (its cleanup only matches dot-named
``.*.tmp`` files). Each writer now goes through ``file_io``: here a failing
``os.replace`` stands in for the reclaim, and the destination must keep its
previous content with no temporary file left beside it. The byte checks pin
the format each writer had before (CLEANUP_PLAN_2026_09.md CU-3).
"""

from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from environments.shared.config import save_stage_config
from environments.shared.curriculum.gate_resolver import write_gate_resolution
from environments.shared.plant_contract import write_plant_identity
from environments.shared.recovery_evaluation import (
    EpisodeRecord,
    RecoveryPanelEvidence,
    ShoveRecord,
    write_recovery_evidence,
)
from environments.shared.reporting import stance_report
from environments.shared.reporting.csv_output import save_evaluation_episodes
from environments.shared.task_fingerprint import write_task_fingerprint

from .reporting_helpers import make_plant_identity

STAGE_CONFIG = {
    "name": "balance",
    "description": "Stand upright",
    "env_kwargs": {"alive_bonus": 2.0},
    "ppo_kwargs": {"learning_rate": 3e-4},
    "curriculum_kwargs": {"min_avg_reward": 10.0},
}
PANEL_EPISODE = {
    "episode": 0,
    "seed": 3042,
    "length": 1000,
    "reward": 2.5,
    "reached_horizon": True,
    "unsupported_duty": 0.01,
    "bilateral_support_duty": 0.9,
    "single_support_duty": None,
}


@pytest.fixture
def lost_mount(monkeypatch):
    """Every rename fails, as when the runtime dies before the publish lands."""

    def replace(src, dst):
        raise OSError("runtime reclaimed")

    monkeypatch.setattr(os, "replace", replace)


def _left_behind(directory: Path, keep: set[str]) -> list[str]:
    return sorted(path.name for path in directory.iterdir() if path.name not in keep)


def _write_old(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("old", encoding="utf-8")


class TestPublishedAtomically:
    def test_stage_config(self, tmp_path, lost_mount):
        target = tmp_path / "stage1" / "stage_config.json"
        _write_old(target)
        with pytest.raises(OSError, match="reclaimed"):
            save_stage_config(tmp_path / "stage1", 1, STAGE_CONFIG, "PPO")
        assert target.read_text(encoding="utf-8") == "old"
        assert _left_behind(target.parent, {"stage_config.json"}) == []

    def test_metrics_json(self, tmp_path, monkeypatch, lost_mount):
        from environments.shared import train_base

        target = tmp_path / "metrics.json"
        _write_old(target)
        (tmp_path / "models").mkdir()
        with pytest.raises(OSError, match="reclaimed"):
            train_base._report_hpt_metrics(
                SimpleNamespace(species="velociraptor", success_keys=["strike_success"]),
                MagicMock(),
                MagicMock(),
                SimpleNamespace(best_mean_reward=12.5),
                tmp_path,
                tmp_path / "models",
                1,
                1000,
                "ppo",
                post_eval_episodes=0,
            )
        assert target.read_text(encoding="utf-8") == "old"
        assert _left_behind(tmp_path, {"metrics.json", "models"}) == []

    def test_stance_report_text_json_and_panel(self, tmp_path, monkeypatch, lost_mount):
        monkeypatch.setattr(stance_report, "render_stance_gate_report", lambda report: "report text")
        for name in ("stance_gate_report.txt", "stance_gate_report.json", "stance_panel_selected.csv"):
            _write_old(tmp_path / name)
        with pytest.raises(OSError, match="reclaimed"):
            stance_report.write_stance_gate_report(tmp_path, {"episode_evidence": [PANEL_EPISODE]})
        for name in ("stance_gate_report.txt", "stance_gate_report.json", "stance_panel_selected.csv"):
            assert (tmp_path / name).read_text(encoding="utf-8") == "old", name
        assert (
            _left_behind(tmp_path, {"stance_gate_report.txt", "stance_gate_report.json", "stance_panel_selected.csv"})
            == []
        )

    def test_evaluation_episodes(self, tmp_path, lost_mount):
        target = tmp_path / "evaluation_selected.csv"
        _write_old(target)
        with pytest.raises(OSError, match="reclaimed"):
            save_evaluation_episodes(
                tmp_path,
                rewards=[1.0],
                lengths=[10],
                forward_velocities=[0.1],
                distances=[1.0],
                successes=[True],
                evaluation_seed=7,
                checkpoint_label="selected",
            )
        assert target.read_text(encoding="utf-8") == "old"
        assert _left_behind(tmp_path, {"evaluation_selected.csv"}) == []

    def test_recovery_evidence(self, tmp_path, lost_mount):
        target = tmp_path / "recovery_episodes_policy.csv"
        _write_old(target)
        evidence = RecoveryPanelEvidence(
            controller_id="policy",
            episodes=(EpisodeRecord("policy", 0, 3042, 1000, True, 1, 1, True, 2.5),),
            shoves=(ShoveRecord("policy", 0, 3042, 0, 100, 110, 1.0, 0.0, 50.0, True, 130),),
            safe_set={},
        )
        with pytest.raises(OSError, match="reclaimed"):
            write_recovery_evidence(tmp_path, evidence)
        assert target.read_text(encoding="utf-8") == "old"
        assert _left_behind(tmp_path, {"recovery_episodes_policy.csv"}) == []

    @pytest.mark.parametrize(
        ("name", "write"),
        [
            ("gate_resolution.json", lambda directory: write_gate_resolution(directory, {"a": 1})),
            (
                "task_fingerprint.json",
                lambda directory: write_task_fingerprint(directory / "task_fingerprint.json", {"a": 1}),
            ),
            (
                "plant_identity.json",
                lambda directory: write_plant_identity(directory / "plant_identity.json", make_plant_identity()),
            ),
        ],
    )
    def test_the_sidecars_strand_no_fixed_temporary(self, tmp_path, name, write, lost_mount):
        _write_old(tmp_path / name)
        with pytest.raises(OSError, match="reclaimed"):
            write(tmp_path)
        assert (tmp_path / name).read_text(encoding="utf-8") == "old"
        # The old writers left `<name>.tmp` here, which the bundle manifest hashed.
        assert _left_behind(tmp_path, {name}) == []


class TestBytesUnchanged:
    """The converted writers keep the format they had before CU-3."""

    def test_stage_config_is_indent_2_with_a_trailing_newline(self, tmp_path):
        text = save_stage_config(tmp_path, 1, STAGE_CONFIG, "PPO").read_text(encoding="utf-8")
        assert text == json.dumps(json.loads(text), indent=2) + "\n"

    def test_metrics_json_has_no_trailing_newline(self, tmp_path):
        from environments.shared import train_base

        (tmp_path / "models").mkdir()
        train_base._report_hpt_metrics(
            SimpleNamespace(species="velociraptor", success_keys=["strike_success"]),
            MagicMock(),
            MagicMock(),
            SimpleNamespace(best_mean_reward=12.5),
            tmp_path,
            tmp_path / "models",
            1,
            1000,
            "ppo",
            post_eval_episodes=0,
        )
        text = (tmp_path / "metrics.json").read_text(encoding="utf-8")
        assert text == json.dumps(json.loads(text), indent=2)

    def test_stance_report_json_is_sorted_and_the_text_is_utf8(self, tmp_path, monkeypatch):
        monkeypatch.setattr(stance_report, "render_stance_gate_report", lambda report: "panel — ok")
        stance_report.write_stance_gate_report(tmp_path, {"episode_evidence": [PANEL_EPISODE], "b": 1, "a": 2})
        text = (tmp_path / "stance_gate_report.json").read_text(encoding="utf-8")
        assert text == json.dumps(json.loads(text), indent=2, sort_keys=True, allow_nan=False) + "\n"
        assert (tmp_path / "stance_gate_report.txt").read_bytes() == "panel — ok\n".encode("utf-8")

    def test_evidence_csvs_keep_the_excel_dialect(self, tmp_path):
        stance_report.write_stance_panel_evidence(tmp_path, {"episode_evidence": [PANEL_EPISODE]})
        raw = (tmp_path / "stance_panel_selected.csv").read_bytes()
        header, rows = raw.decode("utf-8").split("\r\n", 1)
        reparsed = list(csv.DictReader(io.StringIO(raw.decode("utf-8"), newline="")))
        rendered = io.StringIO(newline="")
        writer = csv.DictWriter(rendered, fieldnames=header.split(","))
        writer.writeheader()
        writer.writerows(reparsed)
        assert raw == rendered.getvalue().encode("utf-8")
        assert rows.endswith("\r\n")

    @pytest.mark.parametrize("name", ["gate_resolution.json", "task_fingerprint.json"])
    def test_sidecars_are_sorted_with_a_trailing_newline(self, tmp_path, name):
        value = {"b": 1, "a": {"d": 2, "c": 3}}
        if name == "gate_resolution.json":
            path = write_gate_resolution(tmp_path, value)
        else:
            path = write_task_fingerprint(tmp_path / name, value)
        assert path == tmp_path / name
        assert path.read_bytes() == (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
