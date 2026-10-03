"""Executed tests of ``notebooks/google_drive_summary.ipynb`` (cleanup CU-15).

The Drive summary keeps its own run reader (cells 5, 8, 9, 11 and 12; the cleanup plan's
section 4.9 keeps it in the notebook), so no library test reaches it, and a change it depends on
would otherwise surface only in a Colab session. The reader test runs those cells whole and in
order -- cell 9 keeps cell 8's ``scan_run`` as ``_legacy_scan_run`` before replacing it, so
defining one function at a time would not do -- on a logs tree that takes every route of cell 9's
``scan_run``: in today's layout (``<species>/<algorithm>/<run_id>/NN_<stage id>/``, built by the
bundle test helpers) a complete run, a partial one, a failed one and one whose summary no longer
matches its manifest, and beside them an older flat run and a March 2026 sweep. Two folders pin
the walk itself: an interrupted run, which ``discover_runs`` finds only by its ``NN_<stage id>``
directory (it has no CSV yet) and which audits as a quarantined conflict; and a sweep folder in the
form the sweep command PR-A retired last wrote (``sweeps/<algorithm>_<timestamp>/``), which gives
no row (the 2026-08 gap review's NB2, retired by D-D17 without a fix) and which only
``discover_runs``' ``sweeps/`` branch keeps out of the run walk (the cleanup plan's section 7).

Cell 5 pip-installs pandas when it cannot import it, so the reader test skips without pandas: the
SB3 job, which lists this module, has it through ``stable-baselines3[extra]``, and the shared
matrix installs only the ``test`` extra. The setup and install cell tests need neither.
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
        run_dir, algorithm="PPO", dirname=lambda stage: stage_dirname("velociraptor", stage)
    )
    for stage_result in stage_results:
        stage_result["gate_passed"] = stage_result["publication_gate_passed"] = False
    save_result_bundle(
        stage_results,
        stage_configs,
        "velociraptor",
        "PPO",
        42,
        run_dir,
        backend="stable-baselines3",
        backend_version="test-backend-1.0",
        parallel_envs=4,
        evaluation_episodes=3,
        evaluation_seeds=[101, 102, 103],
        plant_identity=_plant_identity("velociraptor"),
        run_id="velociraptor-stable-baselines3-ppo-failed",
    )


def _interrupted_run(run_dir: Path) -> None:
    """A run stopped in its first stage: the provenance the SB3 notebook captures before training and
    one ``NN_<stage id>`` directory, but no CSV, summary or artifact manifest yet."""
    initialize_result_bundle(
        run_dir,
        species="velociraptor",
        algorithm="PPO",
        backend="stable-baselines3",
        seed=42,
        evaluation_seeds=[101, 102],
        parallel_envs=4,
        hardware="test-cpu",
        plant_identity=_plant_identity("velociraptor"),
        run_id="velociraptor-stable-baselines3-ppo-interrupted",
    )
    stage_dir = run_dir / stage_dirname("velociraptor", 1)
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
            logs / "velociraptor" / run,
            algorithm="PPO",
            backend="stable-baselines3",
            dirname=lambda stage: stage_dirname("velociraptor", stage),
        )
    # The second run's summary no longer matches its artifact manifest: a canonical conflict.
    paths["summary"].write_text(paths["summary"].read_text(encoding="utf-8") + "\n", encoding="utf-8")
    _complete_bundle(
        logs / "trex" / partial,
        algorithm="PPO",
        backend="stable-baselines3",
        species="trex",
        stage_refs=tuple(entry.reference for entry in load_stage_manifest("trex").stages),
        dirname=lambda stage: stage_dirname("trex", stage),
    )
    _failed_bundle(logs / "velociraptor" / failed)
    interrupted = "ppo/20260930_120000"
    _interrupted_run(logs / "velociraptor" / interrupted)
    _legacy_run(logs / "velociraptor" / legacy)
    _legacy_sweep(logs / "velociraptor" / sweep)
    # NB2, accepted: no row from a sweep folder without the stage<N>_ prefix, and no run either.
    unread_sweep = logs / "velociraptor" / "sweeps" / "ppo_20260828_143000"
    unread_sweep.mkdir()
    (unread_sweep / "collected_results.csv").write_text(
        "trial_id,stage,stage_name,stage_passed,best_mean_reward\nt9,1,Stance,True,99.0\n", encoding="utf-8"
    )

    sources = code_cell_sources(DRIVE_SUMMARY_PATH)
    indexes = [cell_index(sources, marker) for marker in READER_CELL_MARKERS]
    assert indexes == sorted(indexes), "the reader cells must run in the notebook's order"
    displayed = []
    namespace = {"LOGS_DIR": logs, "Path": Path, "display": displayed.append}
    for index in indexes:
        exec(compile(sources[index], f"google_drive_summary code cell {index}", "exec"), namespace)

    assert "Found 13 stage results across 5 runs/sweeps." in capsys.readouterr().out
    columns = ["species", "run_dir", "stage", "stage_name", "stage_passed", "source", "backend", "bundle_status"]
    rows = [tuple(str(value) for value in row) for row in namespace["df"][columns].itertuples(index=False)]
    sb3 = "stable-baselines3"
    expected = [
        *(("velociraptor", valid, str(s), f"Stage {s}", "True", "training", sb3, "canonical-valid") for s in (1, 2, 3)),
        *(
            ("trex", partial, str(s), f"Stage {s}", str(s != "recovery"), "training", sb3, "canonical-partial")
            for s in (1, "recovery", 2, 3)
        ),
        *(("velociraptor", failed, str(s), f"Stage {s}", "False", "training", sb3, "failed") for s in (1, 2, 3)),
        ("velociraptor", legacy, "1", "Stance", "True", "training", sb3, "legacy-unverified"),
        ("velociraptor", sweep, "1", "Stance", "True", "sweep", sb3, "sweep"),
        ("velociraptor", sweep, "1", "Stance", "False", "sweep", sb3, "sweep"),
    ]
    assert sorted(rows) == sorted(expected)
    df = namespace["df"]
    assert set(df["algorithm"]) == {"PPO"}
    bundled = df[df["bundle_status"].isin(["canonical-valid", "canonical-partial", "failed"])]
    assert set(zip(bundled["run_dir"], bundled["run_id"])) == {
        (valid, "velociraptor-stable-baselines3-ppo-test"),
        (partial, "trex-stable-baselines3-ppo-test"),
        (failed, "velociraptor-stable-baselines3-ppo-failed"),
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

    # Section 7's example names a run as <algorithm>/<run_id>; its stage directory is found by reference.
    exec(compile(code_cell(DRIVE_SUMMARY_PATH, "def show_full_config("), "show_full_config", "exec"), namespace)
    namespace["show_full_config"]("trex", partial, 1)
    assert json.loads(capsys.readouterr().out.split("---\n", 1)[1])["name"] == "Stage 1"
