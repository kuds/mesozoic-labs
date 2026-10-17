"""Executed tests of ``notebooks/google_drive_summary.ipynb`` (cleanup CU-15).

The Drive summary keeps its own run reader (cells 5, 8, 9, 11 and 12; the cleanup plan's
section 4.9 keeps it in the notebook), so no library test reaches it, and a change it depends on
would otherwise surface only in a Colab session. The reader test runs those cells whole and in
order -- cell 9 keeps cell 8's ``scan_run`` as ``_legacy_scan_run`` before replacing it, so
defining one function at a time would not do -- on a logs tree that takes every route of cell 9's
``scan_run``: in today's layout (``<species>/<embodiment>/<algorithm>/<run_id>/NN_<stage id>/``, built by the
bundle test helpers) a complete run, a partial one, a failed one and one whose summary no longer
matches its manifest, and beside them an older flat run and a March 2026 sweep. Two folders pin
the walk itself: an interrupted run, which ``discover_runs`` finds only by its ``NN_<stage id>``
directory (it has no CSV yet) and which audits as a quarantined conflict; and a sweep folder in the
form the sweep command PR-A retired last wrote (``sweeps/<algorithm>_<timestamp>/``), which gives
no row (the 2026-08 gap review's NB2, retired by D-D17 without a fix) and which only
``discover_runs``' ``sweeps/`` branch keeps out of the run walk (the cleanup plan's section 7).

Section 9's cell (24) saves the table to ``runs_summary.csv`` with the repository's offline judgements
(``reporting.runs_summary``, tested by itself in ``test_reporting_runs_summary.py``); its test runs the
cell with runs and a ledger, with each alone, with neither, and with no logs directory.

Cell 5 pip-installs pandas when it cannot import it, so the reader and export tests skip without
pandas: the SB3 job, which lists this module, has it through ``stable-baselines3[extra]``, and the
shared matrix installs only the ``test`` extra. The setup and install cell tests need neither.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from environments.shared.model_id import ModelId
from environments.shared.reporting import save_result_bundle
from environments.shared.result_bundle import initialize_result_bundle
from environments.shared.stage_manifest import load_stage_manifest, stage_dirname

from .notebook_cells import REPO_ROOT, cell_index, code_cell, code_cell_sources
from .result_bundle_helpers import _complete_bundle, _complete_bundle_inputs, _plant_identity

DRIVE_SUMMARY_PATH = REPO_ROOT / "notebooks" / "google_drive_summary.ipynb"
#: The reader cells, in the order the notebook runs them: imports (5), the legacy parser (8), the
#: bundle router (9), the scan (11) and the identity and audit tables (12).
READER_CELL_MARKERS = (
    "import pandas as pd",
    "def parse_run_dir_name(",
    "_legacy_scan_run = scan_run",
    "def discover_runs(",
    "bundle_audit_df = ",
)


@pytest.mark.parametrize("importable,mounted,mounts", [(True, False, True), (True, True, False), (False, False, False)])
def test_the_setup_cell_mounts_drive_only_from_colab_and_only_once(monkeypatch, importable, mounted, mounts):
    """The setup cell's ``if IN_COLAB:`` block up to its Git block, run with google.colab faked or absent."""
    colab_block = next(
        node for node in ast.parse(code_cell(DRIVE_SUMMARY_PATH, "REPO_REF =")).body if isinstance(node, ast.If)
    )
    git_start = next(
        index
        for index, node in enumerate(colab_block.body)
        if isinstance(node, ast.Import) and node.names[0].name == "pathlib"
    )
    calls = []
    colab = types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(mount=calls.append)
    monkeypatch.setitem(sys.modules, "google.colab", colab if importable else None)
    monkeypatch.setattr(os.path, "ismount", lambda path: mounted and path == "/content/drive")
    namespace = {"os": os, "Path": Path}
    head = ast.Module(body=colab_block.body[:git_start], type_ignores=[])
    exec(compile(head, "drive_summary_mount", "exec"), namespace)
    assert calls == (["/content/drive"] if mounts else [])
    assert namespace["LOGS_DIR"] == Path("/content/drive/MyDrive/mesozoic-labs/logs")


@pytest.mark.parametrize("installed", [False, True])
def test_the_install_cell_asks_for_the_package_before_it_puts_the_checkout_on_the_path(
    monkeypatch, tmp_path, installed
):
    """A fresh runtime installs the checkout (and so its dependencies); one that has it skips the install. The
    setup cell leaves ``sys.path`` alone, so the checkout cannot answer ``find_spec`` for an install."""
    import importlib.util

    repo_root = tmp_path / "mesozoic-labs"
    installs = []

    def find_spec(name, *args):
        assert name == "environments" and str(repo_root) not in sys.path, "asked after the path insert"
        return object() if installed else None

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    monkeypatch.setattr(subprocess, "check_call", lambda command, **kwargs: installs.append(command))
    monkeypatch.setattr(sys, "path", list(sys.path))
    source = code_cell(DRIVE_SUMMARY_PATH, '"-m", "pip", "install", "-q", "-e"')
    exec(compile(source, "drive_summary_install", "exec"), {"IN_COLAB": True, "repo_root": repo_root})
    assert installs == ([] if installed else [[sys.executable, "-m", "pip", "install", "-q", "-e", str(repo_root)]])
    assert sys.path[0] == str(repo_root)
    assert "sys.path" not in code_cell(DRIVE_SUMMARY_PATH, "REPO_REF =")


def _failed_bundle(run_dir: Path) -> None:
    """A canonical run none of whose stages passed: a ``failed`` manifest and its CSV, and no summary."""
    stage_results, stage_configs = _complete_bundle_inputs(
        run_dir,
        algorithm="PPO",
        dirname=lambda stage: stage_dirname(ModelId("velociraptor_mongoliensis", "anatomical"), stage),
    )
    for stage_result in stage_results:
        stage_result["gate_passed"] = stage_result["publication_gate_passed"] = False
    save_result_bundle(
        stage_results,
        stage_configs,
        ModelId("velociraptor_mongoliensis", "anatomical"),
        "PPO",
        42,
        run_dir,
        backend="stable-baselines3",
        backend_version="test-backend-1.0",
        parallel_envs=4,
        evaluation_episodes=3,
        evaluation_seeds=[101, 102, 103],
        plant_identity=_plant_identity(ModelId("velociraptor_mongoliensis", "anatomical")),
        run_id="velociraptor_mongoliensis-anatomical-stable-baselines3-ppo-failed",
    )


def _interrupted_run(run_dir: Path) -> None:
    """A run stopped in its first stage: the provenance the SB3 notebook captures before training and
    one ``NN_<stage id>`` directory, but no CSV, summary or artifact manifest yet."""
    initialize_result_bundle(
        run_dir,
        model_id=ModelId("velociraptor_mongoliensis", "anatomical"),
        algorithm="PPO",
        backend="stable-baselines3",
        seed=42,
        evaluation_seeds=[101, 102],
        parallel_envs=4,
        hardware="test-cpu",
        plant_identity=_plant_identity(ModelId("velociraptor_mongoliensis", "anatomical")),
        run_id="velociraptor_mongoliensis-anatomical-stable-baselines3-ppo-interrupted",
    )
    stage_dir = run_dir / stage_dirname(ModelId("velociraptor_mongoliensis", "anatomical"), 1)
    stage_dir.mkdir()
    (stage_dir / "stage_config.json").write_text(json.dumps({"name": "Stage 1", "algorithm": "PPO"}), encoding="utf-8")


def _legacy_run(run_dir: Path) -> None:
    """A run from before result bundles: ``stage1/`` holding a stage config and ``evaluations.npz``."""
    stage_dir = run_dir / "stage1"
    stage_dir.mkdir(parents=True)
    curriculum = {"min_avg_reward": 10.0, "min_eval_episodes": 3, "required_consecutive": 1}
    config = {"name": "Stance", "algorithm": "PPO", "library_version": "0.3.0", "curriculum": curriculum}
    (stage_dir / "stage_config.json").write_text(json.dumps(config), encoding="utf-8")
    np.savez(
        stage_dir / "evaluations.npz",
        timesteps=np.array([5000, 10000]),
        results=np.array([[5.0, 6.0, 7.0], [11.0, 12.0, 13.0]]),
        ep_lengths=np.array([[100, 100, 100], [200, 200, 200]]),
    )


def _legacy_sweep(sweep_dir: Path) -> None:
    """A March 2026 sweep folder: two trials of stage 1 in its ``collected_results.csv``."""
    sweep_dir.mkdir(parents=True)
    (sweep_dir / "collected_results.csv").write_text(
        "trial_id,stage,stage_name,stage_passed,best_mean_reward\nt1,1,Stance,True,42.0\nt2,1,Stance,False,7.0\n",
        encoding="utf-8",
    )


def test_the_reader_cells_read_todays_layout(tmp_path, monkeypatch, capsys, stable_provenance):
    pytest.importorskip("pandas")  # before cell 5, which pip-installs pandas when the import fails

    def _no_install(command, **kwargs):
        raise AssertionError(f"a reader cell ran {command}")

    monkeypatch.setattr(subprocess, "check_call", _no_install)
    logs = tmp_path / "logs"
    valid, partial, conflict, failed = (f"ppo/20261001_0{hour}0000" for hour in range(4))
    legacy, sweep = "ppo_20260301_120000", "sweeps/stage1_ppo_20260315_101500"
    for run in (valid, conflict):
        paths, _, _ = _complete_bundle(
            logs / "velociraptor_mongoliensis" / "anatomical" / run,
            algorithm="PPO",
            backend="stable-baselines3",
            dirname=lambda stage: stage_dirname(ModelId("velociraptor_mongoliensis", "anatomical"), stage),
        )
    # The second run's summary no longer matches its artifact manifest: a canonical conflict.
    paths["summary"].write_text(paths["summary"].read_text(encoding="utf-8") + "\n", encoding="utf-8")
    _complete_bundle(
        logs / "tyrannosaurus_rex" / "anatomical" / partial,
        algorithm="PPO",
        backend="stable-baselines3",
        model_id=ModelId("tyrannosaurus_rex", "anatomical"),
        stage_refs=tuple(
            entry.reference for entry in load_stage_manifest(ModelId("tyrannosaurus_rex", "anatomical")).stages
        ),
        dirname=lambda stage: stage_dirname(ModelId("tyrannosaurus_rex", "anatomical"), stage),
    )
    _failed_bundle(logs / "velociraptor_mongoliensis" / "anatomical" / failed)
    interrupted = "ppo/20260930_120000"
    _interrupted_run(logs / "velociraptor_mongoliensis" / "anatomical" / interrupted)
    _legacy_run(logs / "velociraptor_mongoliensis" / "anatomical" / legacy)
    _legacy_sweep(logs / "velociraptor_mongoliensis" / "anatomical" / sweep)
    # NB2, accepted: no row from a sweep folder without the stage<N>_ prefix, and no run either.
    unread_sweep = logs / "velociraptor_mongoliensis" / "anatomical" / "sweeps" / "ppo_20260828_143000"
    unread_sweep.mkdir()
    (unread_sweep / "collected_results.csv").write_text(
        "trial_id,stage,stage_name,stage_passed,best_mean_reward\nt9,1,Stance,True,99.0\n", encoding="utf-8"
    )
    # A top-level directory that is no species of the manifest (a tree written under a species ID decision D-D33
    # retired looks the same) is named as skipped and never read, though it holds a run.
    _legacy_run(logs / "pre_rename_tree" / legacy)

    sources = code_cell_sources(DRIVE_SUMMARY_PATH)
    indexes = [cell_index(sources, marker) for marker in READER_CELL_MARKERS]
    assert indexes == sorted(indexes), "the reader cells must run in the notebook's order"
    displayed = []
    namespace = {"LOGS_DIR": logs, "Path": Path, "display": displayed.append}
    for index in indexes:
        exec(compile(sources[index], f"google_drive_summary code cell {index}", "exec"), namespace)

    out = capsys.readouterr().out
    assert "Found 13 stage results across 5 runs/sweeps." in out
    assert "Skipped top-level directories (not a species of the manifest; not read): pre_rename_tree" in out
    assert namespace["SKIPPED_DIRS"] == ["pre_rename_tree"]
    assert set(zip(namespace["df"]["species"], namespace["df"]["embodiment"])) == {
        ("velociraptor_mongoliensis", "anatomical"),
        ("tyrannosaurus_rex", "anatomical"),
    }
    columns = ["species", "run_dir", "stage", "stage_name", "stage_passed", "source", "backend", "bundle_status"]
    rows = [tuple(str(value) for value in row) for row in namespace["df"][columns].itertuples(index=False)]
    sb3 = "stable-baselines3"
    expected = [
        *(
            ("velociraptor_mongoliensis", valid, str(s), f"Stage {s}", "True", "training", sb3, "canonical-valid")
            for s in (1, 2, 3)
        ),
        *(
            (
                "tyrannosaurus_rex",
                partial,
                str(s),
                f"Stage {s}",
                str(s != "recovery"),
                "training",
                sb3,
                "canonical-partial",
            )
            for s in (1, "recovery", 2, 3)
        ),
        *(
            ("velociraptor_mongoliensis", failed, str(s), f"Stage {s}", "False", "training", sb3, "failed")
            for s in (1, 2, 3)
        ),
        ("velociraptor_mongoliensis", legacy, "1", "Stance", "True", "training", sb3, "legacy-unverified"),
        ("velociraptor_mongoliensis", sweep, "1", "Stance", "True", "sweep", sb3, "sweep"),
        ("velociraptor_mongoliensis", sweep, "1", "Stance", "False", "sweep", sb3, "sweep"),
    ]
    assert sorted(rows) == sorted(expected)
    df = namespace["df"]
    assert set(df["algorithm"]) == {"PPO"}
    bundled = df[df["bundle_status"].isin(["canonical-valid", "canonical-partial", "failed"])]
    assert set(zip(bundled["run_dir"], bundled["run_id"])) == {
        (valid, "velociraptor_mongoliensis-anatomical-stable-baselines3-ppo-test"),
        (partial, "tyrannosaurus_rex-anatomical-stable-baselines3-ppo-test"),
        (failed, "velociraptor_mongoliensis-anatomical-stable-baselines3-ppo-failed"),
    }
    assert df["best_mean_reward"].notna().all() and df.loc[df["source"] == "training", "timesteps"].notna().all()
    # Cell 9 reads these from summary.json with .get(), so a renamed key would give None, not an error.
    summary_metrics = [
        "best_mean_reward",
        "last_mean_reward",
        "last_mean_episode_length",
        "mean_forward_vel",
        "std_forward_vel",
        "mean_distance_traveled",
        "mean_success_rate",
        "training_duration_seconds",
        "selected_model_mean_reward",
        "selected_model_mean_episode_length",
        "selected_model_mean_forward_vel",
        "selected_model_success_rate",
    ]
    summary_rows = df[df["bundle_status"].isin(["canonical-valid", "canonical-partial"])]
    assert summary_rows[summary_metrics].notna().all().all()
    records = {record["run_path"]: record for record in namespace["BUNDLE_AUDITS"].values()}
    assert {run_path: record["status"] for run_path, record in records.items()} == {
        valid: "canonical-valid",
        partial: "canonical-partial",
        conflict: "canonical-conflict",
        failed: "failed",
        interrupted: "canonical-conflict",  # no artifact manifest yet; no row
        legacy: "legacy-unverified",
    }
    assert [record["run_path"] for record in namespace["QUARANTINED_RUNS"]] == [interrupted, conflict]
    assert records[interrupted]["errors"] == ["canonical bundle is missing artifact_manifest.json"]
    assert len(displayed) == 3, "cell 12 shows the audits, the quarantined runs and the run identities"
    assert set(displayed[-1]["species"]) == {"Tyrannosaurus Rex", "Velociraptor Mongoliensis"}

    # Section 7's example names a run as <algorithm>/<run_id>; its stage directory is found by reference.
    exec(compile(code_cell(DRIVE_SUMMARY_PATH, "def show_full_config("), "show_full_config", "exec"), namespace)
    namespace["show_full_config"]("tyrannosaurus_rex", "anatomical", partial, 1)
    assert json.loads(capsys.readouterr().out.split("---\n", 1)[1])["name"] == "Stage 1"


def test_display_and_export_cells_align_historical_stance_names_without_rewriting_ids(tmp_path, monkeypatch, capsys):
    pd = pytest.importorskip("pandas")
    from environments.shared import reporting
    from environments.shared.species_names import model_ids, species_display_name, species_ids

    monkeypatch.setattr(reporting, "OFFLINE_JUDGEMENTS_PATH", tmp_path / "no_ledger.csv")

    labels = {species: species_display_name(species) for species in species_ids()}

    rows = [
        {
            **model_id.record(),
            "source": source,
            "run_dir": f"ppo/{source}_{index}",
            "algorithm": "PPO",
            "library_version": "test",
            "run_date": "2026-10-08",
            "trial_id": "t1",
            "stage": 1,
            "stage_name": name,
            "stage_passed": index != 0,
            "best_mean_reward": float(index),
        }
        for model_id in model_ids()
        for index, (source, name) in enumerate((("training", "balance"), ("training", "stance"), ("sweep", None)))
    ]
    frame = pd.DataFrame(rows)
    before = frame.copy(deep=True)
    displayed = []
    namespace = {"df": frame, "LOGS_DIR": tmp_path, "display": displayed.append}
    exec(compile(code_cell(DRIVE_SUMMARY_PATH, "import pandas as pd"), "summary_imports", "exec"), namespace)
    for marker in ("Training runs pivot", "styled =", "Settings vs Outcome", "Combined CSV across all models"):
        exec(compile(code_cell(DRIVE_SUMMARY_PATH, marker), marker, "exec"), namespace)
    # Both old and new records occupy the same Stance column in each pivot.
    for pivot in displayed[:2]:
        assert set(pivot["species"]) == set(labels.values())
        assert "stage1 (Stance)" in pivot and not any("balance" in column for column in pivot)
    details = displayed[2].data
    assert set(details["stage_name"]) == {"Stance"}
    assert set(details["species"]) == set(labels.values())
    output = capsys.readouterr().out
    assert all(label in output for label in labels.values())
    pd.testing.assert_frame_equal(frame, before)
    exported = pd.read_csv(tmp_path / "runs_summary.csv")
    assert set(exported["species"]) == set(labels)
    assert set(zip(exported["species"], exported["embodiment"])) == {(m.species, m.embodiment) for m in model_ids()}
    assert set(exported["stage_name"].dropna()) == {"balance", "stance"}
    # One file per model, in its own directory (decision D-D33).
    for model_id in model_ids():
        assert (model_id.path(tmp_path) / "runs_summary.csv").is_file()
        assert (model_id.path(tmp_path) / "sweeps" / "sweeps_summary.csv").is_file()


#: A synthetic offline judgement of a T. rex run: a six-heading certificate's PASS.
_CERTIFICATE = {
    "judged_on": "2026-10-17",
    "species": "tyrannosaurus_rex",
    "embodiment": "anatomical",
    "algorithm": "PPO",
    "run_id": "",
    "run_path": "ppo/20260101_000000",
    "stage": "1",
    "checkpoint_step": "6291456",
    "checkpoint_sha256": "sha256:" + "ab" * 32,
    "gate_kind": "stance_quality_heading/v1",
    "verdict": "PASS",
    "passing_count": "40",
    "panel_size": "40",
    "lower_bound": "0.9278",
    "lower_bound_bar": "0.8",
    "binding_bar": "",
    "panel_seeds": "3042-3081",
    "judging_commit": "2e734e29",
    "judged_by": "environments/shared/scripts/stance_gate_report.py",
    "evidence": "docs/investigations/example/certificate.json",
    "note": "synthetic",
    "source": "offline",
}
#: The certificate, and a pilot's outcome on the same run's next stage.
OFFLINE_ROWS = (_CERTIFICATE, {**_CERTIFICATE, "stage": "2", "gate_kind": "walking_pilot/v1", "verdict": "MIXED"})


def _write_ledger(path: Path) -> Path:
    from environments.shared.reporting import OFFLINE_JUDGEMENT_COLUMNS

    lines = [",".join(OFFLINE_JUDGEMENT_COLUMNS)]
    lines += [",".join(row[column] for column in OFFLINE_JUDGEMENT_COLUMNS) for row in OFFLINE_ROWS]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _training_rows():
    """Two training stages of the T. rex run and one legacy sweep trial of the anatomical Compsognathus."""
    common = {"algorithm": "PPO", "library_version": "test", "run_date": "2026-10-08", "stage_name": "stance"}
    return [
        {
            **ModelId("tyrannosaurus_rex", "anatomical").record(),
            **common,
            "source": "training",
            "run_dir": "ppo/20260101_000000",
            "run_path": "ppo/20260101_000000",
            "stage": stage,
            "stage_passed": passed,
            "best_mean_reward": reward,
        }
        for stage, passed, reward in ((1, True, 812.5), ("recovery", None, float("nan")))
    ] + [
        {
            **ModelId("compsognathus_longipes", "anatomical").record(),
            **common,
            "source": "sweep",
            "run_dir": "sweeps/stage1_ppo_20260315_101500",
            "trial_id": "t1",
            "stage": 1,
            "stage_passed": False,
            "best_mean_reward": 7.0,
            "quality_score": 0.5,
        }
    ]


@pytest.mark.parametrize(
    "runs,ledger,logs,expected",
    [
        (True, True, True, {"training": 2, "sweep": 1, "offline": 2}),
        (False, True, True, {"offline": 2}),
        (True, False, True, {"training": 2, "sweep": 1}),
        (True, True, False, None),
        (False, False, True, None),
    ],
    ids=["runs-and-ledger", "ledger-only", "runs-only", "no-logs-directory", "nothing"],
)
def test_the_export_cell_saves_the_table_with_the_offline_judgements(
    tmp_path, monkeypatch, capsys, runs, ledger, logs, expected
):
    """Section 9 adds the repository's offline judgements to the table and replaces runs_summary.csv on Drive."""
    pd = pytest.importorskip("pandas")
    from environments.shared import reporting

    ledger_path = tmp_path / "repo" / "offline_judgements.csv"
    ledger_path.parent.mkdir()
    if ledger:
        _write_ledger(ledger_path)
    monkeypatch.setattr(reporting, "OFFLINE_JUDGEMENTS_PATH", ledger_path)
    logs_dir = tmp_path / "drive" / "logs"
    if logs:
        logs_dir.mkdir(parents=True)
    displayed = []
    namespace = {
        "df": pd.DataFrame(_training_rows() if runs else []),
        "LOGS_DIR": logs_dir,
        "display": displayed.append,
    }
    exec(compile(code_cell(DRIVE_SUMMARY_PATH, "import pandas as pd"), "summary_imports", "exec"), namespace)
    exec(compile(code_cell(DRIVE_SUMMARY_PATH, "write_runs_summary("), "summary_export", "exec"), namespace)
    out = capsys.readouterr().out

    combined = logs_dir / "runs_summary.csv"
    if not ledger:
        assert f"No offline-judgement ledger at {ledger_path}; no offline rows added." in out
    else:
        assert f"Offline judgements: 2 row(s) from {ledger_path}" in out
        assert list(displayed[0]["verdict"]) == ["PASS", "MIXED"]
        assert set(displayed[0]["species"]) == {"Tyrannosaurus Rex"}  # display labels; the file keeps the IDs
    if expected is None:
        assert ("Logs directory not found" in out) == (not logs) and ("No data to export." in out) == logs
        assert not combined.exists() and logs_dir.exists() == logs
        return
    counts = ", ".join(f"{count} {source}" for source, count in expected.items())
    assert f"Combined summary ({sum(expected.values())} rows: {counts}) written to: {combined}" in out
    table = pd.read_csv(combined, dtype=str, keep_default_na=False)  # the cells as written
    assert dict(table["source"].value_counts()) == expected
    assert sorted(entry.name for entry in logs_dir.rglob("*") if entry.is_file() and entry.name.startswith(".")) == []
    tyrannosaurus = pd.read_csv(ModelId("tyrannosaurus_rex", "anatomical").path(logs_dir) / "runs_summary.csv")
    assert list(tyrannosaurus["source"]) == [s for s in ("training", "training", "offline", "offline") if s in expected]
    if runs:
        sweeps = pd.read_csv(
            ModelId("compsognathus_longipes", "anatomical").path(logs_dir) / "sweeps" / "sweeps_summary.csv"
        )
        assert list(sweeps["trial_id"]) == ["t1"] and "judged_on" not in sweeps
    if ledger:
        offline = table[table["source"] == "offline"]
        assert list(offline["verdict"]) == ["PASS", "MIXED"] and list(offline["stage"]) == ["1", "2"]
        assert list(offline["stage_passed"]) == ["True", ""] and list(offline["best_mean_reward"]) == ["", ""]
        assert list(offline["run_dir"]) == ["ppo/20260101_000000"] * 2
        assert list(offline["checkpoint_step"]) == ["6291456"] * 2 and list(offline["binding_bar"]) == ["", ""]
    else:
        assert "judged_on" not in table and "verdict" not in table
