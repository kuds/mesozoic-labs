"""The read-side run reports and the per-model run index (the maintainer's decision of 2026-11-26).

``reporting/run_index.py`` reads a run and ``scripts/run_report.py`` writes its report folder and the model's index
under ``<model>/reports/``, outside every run.  These tests pin, over runs the generation and bundle test helpers
build in every layout generation and for several models (the Compsognathus robot, a quadruped and the bipeds):

* the classification of every file a run holds, with no ``unclassified`` row, from the files a real 64-step
  notebook stage wrote (2026-11-27) and from what ``generate_stage_artifacts`` and ``save_result_bundle`` write;
* each run state, including the ``stage<N>/`` and flat layouts;
* sizes and digests copied from ``artifact_manifest.json``, with every file hasher patched to fail;
* a report of a run with a reused ancestor linking the trunk run's report, and the index's grouping and order;
* the script's refusals, its whole-folder replacement (the old folder kept when a build fails), and that it writes
  nothing inside a run;
* that the declared-but-absent probe rule is the one ``stage_artifacts._run_stance_probes`` rolls by;
* the notebook wiring: the SB3 helper never raises, ``save_run_bundle`` and the verify cell write the report, the
  storage cell prints the index's first lines, and the Drive summary notebook never walks ``reports/`` and rebuilds
  every model's reports, best-effort.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

from environments.shared import result_bundle
from environments.shared.model_id import ModelId
from environments.shared.reporting import run_index, save_result_bundle, stage_artifacts
from environments.shared.result_bundle import initialize_result_bundle, write_gate_verdict
from environments.shared.scripts import run_report
from environments.shared.species_names import model_ids
from environments.shared.stage_manifest import load_stage_manifest, stage_dirname

from .notebook_cells import REPO_ROOT, code_cell, exec_top_level_def
from .result_bundle_helpers import (
    _complete_bundle,
    _legacy_stage_dirname,
    _plant_identity,
    _reused_trunk_bundle_inputs,
    _stage_config,
)

pytestmark = pytest.mark.usefixtures("stable_provenance")

SB3_NOTEBOOK = REPO_ROOT / "notebooks" / "sb3_training.ipynb"
DRIVE_SUMMARY = REPO_ROOT / "notebooks" / "google_drive_summary.ipynb"
VELOCIRAPTOR_MONGOLIENSIS = ModelId("velociraptor_mongoliensis", "anatomical")
TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")
BRACHIOSAURUS_ALTITHORAX = ModelId("brachiosaurus_altithorax", "anatomical")
DIBOTHROSUCHUS_ELAPHROS = ModelId("dibothrosuchus_elaphros", "anatomical")
COMPSOGNATHUS_LONGIPES_ROBOT = ModelId("compsognathus_longipes", "robot")
ALLOSAURUS_FRAGILIS = ModelId("allosaurus_fragilis", "anatomical")
#: A biped, the robot and the two quadrupeds.
SEVERAL_MODELS = (TYRANNOSAURUS_REX, COMPSOGNATHUS_LONGIPES_ROBOT, BRACHIOSAURUS_ALTITHORAX, DIBOTHROSUCHUS_ELAPHROS)

#: Every file a real 64-step notebook stance stage held after ``train_stage`` and ``generate_stage_artifacts``
#: (graphs and replays on) on 2026-11-27, for the Compsognathus robot (stance_quality/v1 and its heading probe),
#: the T. rex (the six-heading certificate, its filtered and impulse probes) and the Brachiosaurus
#: (reward_and_length/v1); the twenty training-history pages and the replays are added per model below.
REAL_STAGE_FILES = (
    "diagnostics.npz",
    "eval_episodes.csv",
    "evaluation_final.csv",
    "evaluation_selected.csv",
    "evaluations.npz",
    "figures/behavioral_metrics.png",
    "figures/foot_contacts.png",
    "figures/gate_progress.png",
    "figures/locomotion_health.png",
    "figures/stance_diagnostics.png",
    "figures/training_curves.png",
    "figures/training_history/training_history.html",
    "figures/training_history/training_history.json",
    "gate_progress.npz",
    "gate_verdict.json",
    "handoff_in_training.json",
    "models/best_model.zip",
    "models/best_model_vecnorm.pkl",
    "models/robust_best_model.zip",
    "models/robust_best_model_vecnorm.pkl",
    "models/stage1_64_steps.zip",
    "models/stage1_final.zip",
    "models/stage1_final_vecnorm.pkl",
    "models/stage1_vecnormalize_64_steps.pkl",
    "plant_identity.json",
    "policy_probe.npz",
    "progress.json",
    "stage_config.json",
    "stage_summary.txt",
    "stance_gate_probe_filtered.json",
    "stance_gate_probe_filtered.txt",
    "stance_gate_probe_impulse.json",
    "stance_gate_probe_impulse.txt",
    "stance_gate_report.json",
    "stance_gate_report.txt",
    "stance_heading_panel_selected.csv",
    "stance_heading_probe.json",
    "stance_heading_probe.txt",
    "stance_heading_report.json",
    "stance_heading_report.txt",
    "stance_panel_selected.csv",
    "task_fingerprint.json",
    "tensorboard/PPO_1/events.out.tfevents.1795744964.vm.1270.0",
    "train_episodes.csv",
)
#: The files no 2026-11-27 smoke stage wrote but a writer does (module comments of run_index.ARTIFACT_ROLES).
OTHER_STAGE_FILES = (
    "metrics.json",
    "diagnostics.prior_20261121T175147Z.npz",
    "hop_watch.json",
    "gate_resolution.json",
    "recovery_policy_panel_policy.csv",
    "widen_report.json",
    "stance_gate_probe_constant.txt",
    "stance_gate_probe_constant.json",
    "stance_gate_probe_release.txt",
    "stance_gate_probe_release.json",
    "models/stage1_100000_steps.zip",
    "models/stage1_vecnormalize_100000_steps.pkl",
    # The retired in-run K3 panel's files, which the three stance6_r10LTSB screen runs keep.
    "k3_panel.csv",
    "k3_panel.json",
    "k3_panel.prior_20261025T154400Z.csv",
    "k3_panel.prior_20261025T154400Z.json",
)
#: A stage directory's root before stage_layout (2026-08): figures, replays and stance CSVs beside the evidence.
LEGACY_FLAT_STAGE_FILES = (
    "training_curves.png",
    "velociraptor_mongoliensis_anatomical_ppo_stage1_best.mp4",
    "velociraptor_mongoliensis_anatomical_ppo_stage1_best_stance.csv",
    "stage_result.json",
)
RUN_FILES = (
    "provenance.json",
    "trunk_run.json",
    "plant_identity.json",
    "collected_results.csv",
    "summary.json",
    "artifact_manifest.json",
    "training_summary.txt",
    "zero_action_baseline.json",
    "ancestors/stance/ancestor.json",
    "ancestors/stance/gate_verdict.json",
    "ancestors/stance/stage_config.json",
    "ancestors/stance/task_fingerprint.json",
    "ancestors/stance/plant_identity.json",
)


def _replays(model_id: ModelId, label: str = "stage1") -> list[str]:
    prefix = f"replays/{model_id.species}_{model_id.embodiment}_ppo_{label}"
    return [
        f"{prefix}_{role}{view}" for role in ("selected", "final") for view in (".mp4", "_side.mp4", "_front.mp4")
    ] + [f"{prefix}_{role}_stance.csv" for role in ("selected", "final")]


def _history_pages() -> list[str]:
    pytest.importorskip("matplotlib")
    from environments.shared.reporting import training_history

    return [f"figures/training_history/{name}.png" for name in training_history.page_names()]


def _files(root: Path) -> dict[str, tuple[bytes, int]]:
    return {
        path.relative_to(root).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _provenance(run_dir: Path, model_id: ModelId, *, captured_at: str | None = None) -> None:
    initialize_result_bundle(
        run_dir,
        model_id=model_id,
        algorithm="PPO",
        backend="stable-baselines3",
        seed=42,
        evaluation_seeds=[101, 102],
        parallel_envs=4,
        hardware="test-cpu",
        plant_identity=_plant_identity(model_id),
        run_id=f"{model_id.species}-{model_id.embodiment}-{run_dir.name}",
        captured_at=captured_at,
    )


def _verdict(
    stage_dir: Path,
    model_id: ModelId,
    stage: "int | str",
    *,
    passed: bool,
    curriculum: "dict[str, Any] | None" = None,
    judged_by: str = stage_artifacts.GATE_VERDICT_JUDGED_BY,
) -> Path:
    """``gate_verdict.json`` as *judged_by* (``generate_stage_artifacts``) writes it, bound to a selected pair in
    ``models/``, beside a ``stage_config.json`` recording *curriculum* (a reward-and-length gate by default)."""
    entry = load_stage_manifest(model_id).resolve(stage)
    models = stage_dir / "models"
    models.mkdir(parents=True, exist_ok=True)
    (models / "robust_best_model.zip").write_bytes(b"checkpoint")
    (models / "robust_best_model_vecnorm.pkl").write_bytes(b"statistics")
    config = _stage_config(stage, "PPO", curriculum=curriculum or {"min_avg_reward": 1.0})
    (stage_dir / "stage_config.json").write_text(json.dumps({**config, **model_id.record()}), encoding="utf-8")
    from environments.shared.curriculum.gate_schema import gate_config_view

    return write_gate_verdict(
        stage_dir,
        model_id=model_id,
        stage=entry.reference,
        stage_id=entry.id,
        gate_kind=config["curriculum_kwargs"].get("gate_kind", "reward_and_length/v1"),
        gate_schema_version=config["curriculum_kwargs"].get("gate_schema_version", 1),
        passed=passed,
        failures=[] if passed else ["min_avg_reward: evidence=0.5 threshold=1.0"],
        task_sha256="sha256:" + "7" * 64,
        judged_by=judged_by,
        checkpoint=models / "robust_best_model.zip",
        normalization=models / "robust_best_model_vecnorm.pkl",
        gate_config=gate_config_view(config["curriculum_kwargs"]),
        stage_result={"stage": entry.reference, "gate_passed": passed},
    )


def _run(logs: Path, model_id: ModelId, tree: str = "ppo", run_id: str = "20261121_175147") -> Path:
    run_dir = model_id.path(logs) / tree / run_id
    run_dir.mkdir(parents=True)
    return run_dir


# ---------------------------------------------------------------------------------------------- classification
@pytest.mark.parametrize("model_id", model_ids(), ids=str)
def test_every_file_a_real_stage_and_run_hold_is_classified(model_id):
    stage = stage_dirname(model_id, 1)
    stage_files = [*REAL_STAGE_FILES, *OTHER_STAGE_FILES, *_history_pages(), *_replays(model_id)]
    paths = [*RUN_FILES, *(f"{stage}/{name}" for name in stage_files)]
    paths += [f"stage1/{name}" for name in (*REAL_STAGE_FILES, *LEGACY_FLAT_STAGE_FILES)]
    classified = {path: run_index.classify(path, model_id=model_id) for path in paths}
    assert not [path for path, (_, role) in classified.items() if role == run_index.UNCLASSIFIED]
    assert {role for _, role in classified.values()} <= set(run_index.ROLES)
    assert classified[f"{stage}/gate_verdict.json"] == ("stance", "verdict")
    assert classified[f"{stage}/stance_heading_report.txt"] == ("stance", "certificate")
    assert classified[f"{stage}/stance_heading_panel_selected.csv"] == ("stance", "evidence")
    assert classified[f"{stage}/stance_gate_probe_impulse.txt"] == ("stance", "probe")
    assert classified[f"{stage}/models/robust_best_model_vecnorm.pkl"] == ("stance", "normalization")
    assert classified[f"{stage}/models/stage1_vecnormalize_64_steps.pkl"] == ("stance", "normalization")
    assert classified[f"{stage}/models/stage1_final.zip"] == ("stance", "checkpoint")
    assert classified[f"{stage}/figures/training_history/01_outcomes.png"] == ("stance", "training_history")
    assert classified[f"{stage}/{_replays(model_id)[-1]}"] == ("stance", "replay")
    assert classified[f"{stage}/handoff_in_training.json"] == ("stance", "report_only")
    assert classified[f"{stage}/k3_panel.json"] == classified[f"{stage}/k3_panel.csv"] == ("stance", "report_only")
    assert classified["stage1/training_curves.png"] == ("stance", "figure")
    assert classified["stage1/velociraptor_mongoliensis_anatomical_ppo_stage1_best_stance.csv"] == ("stance", "replay")
    assert classified["ancestors/stance/gate_verdict.json"] == ("stance", "ancestor_record")
    assert classified["artifact_manifest.json"] == (None, "bundle")
    assert classified["zero_action_baseline.json"] == (None, "baseline")


def test_the_recovery_stage_and_open_ids_classify_by_the_models_manifest():
    assert run_index.classify("02_recovery/gate_resolution.json", model_id=TYRANNOSAURUS_REX) == ("recovery", "config")
    assert run_index.classify("recovery/models/recovery_final.zip", model_id=TYRANNOSAURUS_REX) == (
        "recovery",
        "checkpoint",
    )
    assert run_index.classify("stage2/evaluations.npz", model_id=BRACHIOSAURUS_ALTITHORAX) == ("locomotion", "ledger")
    assert run_index.classify("09_no_such_node/stage_config.json", model_id=TYRANNOSAURUS_REX)[1] == "unclassified"


def test_every_file_the_generation_tests_write_is_classified(tmp_path):
    """``generate_stage_artifacts`` on a trainer-shaped stage (test_stage_artifacts_generation's fixture, graphs on),
    then the bundle and the reused-trunk helpers in both layouts, for several models: no unclassified row."""
    pytest.importorskip("matplotlib")
    from .test_stage_artifacts_generation import _judge_velociraptor_mongoliensis_stage, _write_ledgers

    logs = tmp_path / "logs"
    judged = _run(logs, VELOCIRAPTOR_MONGOLIENSIS)
    _provenance(judged, VELOCIRAPTOR_MONGOLIENSIS)
    stage_dir = judged / stage_dirname(VELOCIRAPTOR_MONGOLIENSIS, 1)
    stage_dir.mkdir()
    _write_ledgers(stage_dir)
    _judge_velociraptor_mongoliensis_stage(stage_dir)
    runs = [judged]
    for model_id in SEVERAL_MODELS:
        for name, dirname in (
            ("nested", lambda stage, model_id=model_id: stage_dirname(model_id, stage)),
            ("legacy", _legacy_stage_dirname),
        ):
            run_dir = _run(logs, model_id, run_id=f"20261001_00000{len(runs) % 10}_{name}")
            _complete_bundle(run_dir, algorithm="PPO", backend="stable-baselines3", dirname=dirname, model_id=model_id)
            runs.append(run_dir)
    trunked = _run(logs, VELOCIRAPTOR_MONGOLIENSIS, run_id="20261002_000000")
    results, configs = _reused_trunk_bundle_inputs(trunked)
    save_result_bundle(
        results,
        configs,
        VELOCIRAPTOR_MONGOLIENSIS,
        "PPO",
        42,
        trunked,
        backend="stable-baselines3",
        backend_version="test-backend-1.0",
        parallel_envs=4,
        evaluation_episodes=3,
        evaluation_seeds=[101, 102, 103],
        plant_identity=_plant_identity(),
        run_id="child-run",
    )
    runs.append(trunked)
    for run_dir in runs:
        index = run_index.build_run_index(run_dir)
        assert index["files"], run_dir
        assert not [entry["path"] for entry in index["files"] if entry["role"] == "unclassified"], run_dir
        assert not [warning for warning in index["warnings"] if "no writer names" in warning], run_dir
    roles = {entry["path"]: entry["role"] for entry in run_index.build_run_index(judged)["files"]}
    assert roles["01_stance/figures/gate_progress.png"] == "figure"
    assert roles["01_stance/figures/training_history/training_history.html"] == "training_history"
    assert roles["01_stance/gate_verdict.json"] == "verdict" and roles["provenance.json"] == "provenance"


def test_a_file_no_writer_names_is_unclassified_and_a_warning(tmp_path):
    run_dir = _run(tmp_path / "logs", COMPSOGNATHUS_LONGIPES_ROBOT)
    _provenance(run_dir, COMPSOGNATHUS_LONGIPES_ROBOT)
    stage_dir = run_dir / "01_stance"
    _verdict(stage_dir, COMPSOGNATHUS_LONGIPES_ROBOT, 1, passed=True)
    for name in ("01_stance/notes.txt", "scratch.bin", "01_stance/.gate_verdict.json.abc.tmp", ".DS_Store"):
        (run_dir / name).write_bytes(b"x")
    index = run_index.build_run_index(run_dir)
    roles = {entry["path"]: entry["role"] for entry in index["files"]}
    assert ".DS_Store" not in roles, "the manifest's litter is not a file of the run"
    assert [path for path, role in roles.items() if role == "unclassified"] == [
        "01_stance/.gate_verdict.json.abc.tmp",
        "01_stance/notes.txt",
        "scratch.bin",
    ]
    assert any(warning.startswith("3 file(s) no writer names: ") for warning in index["warnings"])


# ---------------------------------------------------------------------------------------------- states
def _state_run(state: str, logs: Path) -> tuple[Path, str]:
    """A run in *state*, and the layout it is in."""
    model_id = COMPSOGNATHUS_LONGIPES_ROBOT if state in ("complete", "empty") else BRACHIOSAURUS_ALTITHORAX
    nested = lambda stage: stage_dirname(model_id, stage)  # noqa: E731
    if state == "complete":
        run_dir = _run(logs, model_id)
        _complete_bundle(run_dir, algorithm="PPO", backend="stable-baselines3", dirname=nested, model_id=model_id)
        return run_dir, "NN_<stage_id>"
    if state == "partial":
        refs = tuple(entry.reference for entry in load_stage_manifest(TYRANNOSAURUS_REX).stages)
        run_dir = _run(logs, TYRANNOSAURUS_REX)
        _complete_bundle(
            run_dir,
            algorithm="PPO",
            backend="stable-baselines3",
            dirname=lambda stage: stage_dirname(TYRANNOSAURUS_REX, stage),
            model_id=TYRANNOSAURUS_REX,
            stage_refs=refs,
        )
        return run_dir, "NN_<stage_id>"
    if state == "failed":
        from .test_drive_summary_notebook import _failed_bundle

        run_dir = _run(logs, VELOCIRAPTOR_MONGOLIENSIS)
        _failed_bundle(run_dir)
        return run_dir, "NN_<stage_id>"
    if state == "bundle_unreadable":
        run_dir = _run(logs, model_id)
        _complete_bundle(run_dir, algorithm="PPO", backend="stable-baselines3", dirname=nested, model_id=model_id)
        (run_dir / "artifact_manifest.json").write_text("{not json", encoding="utf-8")
        return run_dir, "NN_<stage_id>"
    if state == "judged_no_bundle":
        # The stage<N>/ layout of 2026-08.
        run_dir = _run(logs, model_id)
        _provenance(run_dir, model_id)
        _verdict(run_dir / "stage1", model_id, 1, passed=True)
        return run_dir, "stage<N>"
    if state == "post_training_cut":
        run_dir = _run(logs, model_id)
        _provenance(run_dir, model_id)
        stage_dir = run_dir / nested(1)
        (stage_dir / "models").mkdir(parents=True)
        (stage_dir / "models" / "stage1_final.zip").write_bytes(b"checkpoint")
        (stage_dir / "evaluation_final.csv").write_text("episode,reward\n0,1.0\n", encoding="utf-8")
        return run_dir, "NN_<stage_id>"
    if state == "training_cut":
        # A flat run from before 2026-03-31: <algo>_<timestamp>/ in the model directory, no provenance.
        run_dir = model_id.path(logs) / "ppo_20260301_120000"
        stage_dir = run_dir / "stage1"
        (stage_dir / "models").mkdir(parents=True)
        (stage_dir / "stage_config.json").write_text(json.dumps({"name": "Stance"}), encoding="utf-8")
        (stage_dir / "evaluations.npz").write_bytes(b"npz")
        (stage_dir / "models" / "best_model.zip").write_bytes(b"checkpoint")
        (run_dir / "collected_results.csv").write_text("stage,stage_passed\n1,True\n", encoding="utf-8")
        return run_dir, "stage<N>"
    assert state == "empty"
    # What the storage cell writes before training, and a node reused from another run: nothing trained here.
    run_dir = _run(logs, model_id)
    _provenance(run_dir, model_id)
    from .result_bundle_helpers import _write_ancestor_record

    _write_ancestor_record(run_dir, 1, model_id=model_id)
    return run_dir, "none"


@pytest.mark.parametrize("state", run_index.RUN_STATES)
def test_each_state_is_read_from_the_files(tmp_path, state):
    run_dir, layout = _state_run(state, tmp_path / "logs")
    assert run_index.run_state(run_dir) == state
    index = run_index.build_run_index(run_dir)
    assert index["state"] == state and index["summary"]["layout"] == layout
    assert index["schema"] == run_index.RUN_INDEX_SCHEMA
    page = run_index.render_run_report(run_dir)
    assert f"**State: {state}.** {run_index.STATE_DESCRIPTIONS[state].capitalize()}." in page
    assert "cut off during" not in page and "cut off after" not in page, "a state names the files a run holds"
    if state == "partial":
        assert "at least one deliverable it records is certified, and its target or another" in page
    if state == "bundle_unreadable":
        assert all(entry["hash_source"] == "none" for entry in index["files"])
        assert any("cannot read artifact manifest" in warning for warning in index["warnings"])


def test_a_flat_run_and_a_named_run_are_dated_and_indexed(tmp_path):
    run_dir, _ = _state_run("training_cut", tmp_path / "logs")
    model_dir = run_dir.parent
    assert run_index.model_dir_of(run_dir) == model_dir
    assert run_index.report_dir_for(run_dir, model_dir) == model_dir / "reports" / "ppo_20260301_120000"
    [row] = run_index.build_model_index(model_dir)
    assert row["tree"] == "" and row["layout"] == "stage<N>" and row["started"] == "2026-03-01T12:00:00"
    page = run_index.render_run_report(run_dir)
    assert (
        "no tree: in the model directory (a flat run from before 2026-03-31, or a command-line curriculum run)" in page
    )
    assert "layout `stage<N>`" in page
    assert (
        "Its stage directories (`stage<N>`) are of a layout older than `gate_verdict.json`: the state says which files "
        "the run holds, not that its training was cut off."
    ) in page


@pytest.mark.parametrize(
    ("model_id", "name", "dirname", "layout"),
    (
        (BRACHIOSAURUS_ALTITHORAX, "curriculum_20261120_101500", "01_stance", "NN_<stage_id>"),
        (COMPSOGNATHUS_LONGIPES_ROBOT, "ppo_20260819_101500", "stance", "bare <stage_id>"),
        (TYRANNOSAURUS_REX, "ppo_20260301_120000", "stage1", "stage<N>"),
    ),
    ids=("command-line curriculum", "bare ids", "stage<N>"),
)
def test_the_layout_is_the_stage_directories_generation(tmp_path, model_id, name, dirname, layout):
    """Where a run sits (a tree, or the model directory) is its tree; its stage directories' names, its layout."""
    run_dir = model_id.path(tmp_path / "logs") / name
    (run_dir / dirname / "models").mkdir(parents=True)
    (run_dir / dirname / "stage_config.json").write_text("{}", encoding="utf-8")
    (run_dir / dirname / "models" / "best_model.zip").write_bytes(b"checkpoint")
    index = run_index.build_run_index(run_dir)
    assert index["state"] == "training_cut" and index["summary"]["layout"] == layout
    assert index["summary"]["tree"] == "" and index["summary"]["kind"] is None
    page = run_index.render_run_report(run_dir)
    assert (
        f"no tree: in the model directory (a flat run from before 2026-03-31, or a command-line curriculum run); "
        f"layout `{layout}`" in page
    )
    assert ("are of a layout older than `gate_verdict.json`" in page) is (layout != "NN_<stage_id>")


# ---------------------------------------------------------------------------------------------- digests
def test_sizes_and_digests_come_from_the_manifest_and_nothing_is_hashed(tmp_path, monkeypatch):
    run_dir = _run(tmp_path / "logs", DIBOTHROSUCHUS_ELAPHROS)
    paths, _, _ = _complete_bundle(
        run_dir,
        algorithm="PPO",
        backend="stable-baselines3",
        dirname=lambda stage: stage_dirname(DIBOTHROSUCHUS_ELAPHROS, stage),
        model_id=DIBOTHROSUCHUS_ELAPHROS,
    )
    manifest = {entry["path"]: entry for entry in json.loads(paths["artifact_manifest"].read_text())["files"]}
    # A file written after the last bundle (a later session's copy of the zero-action baseline).
    (run_dir / "zero_action_baseline.json").write_text("{}\n", encoding="utf-8")

    def refuse(*args, **kwargs):
        raise AssertionError("the index hashed a file")

    original = result_bundle.sha256_file
    for module in list(sys.modules.values()):
        if (module.__name__ or "").startswith("environments.") and getattr(module, "sha256_file", None) is original:
            monkeypatch.setattr(module, "sha256_file", refuse)
    import hashlib

    monkeypatch.setattr(hashlib, "file_digest", refuse, raising=False)
    index = run_index.build_run_index(run_dir)
    entries = {entry["path"]: entry for entry in index["files"]}
    assert set(manifest) <= set(entries)
    for path, recorded in manifest.items():
        assert entries[path]["sha256"] == recorded["sha256"] and entries[path]["size"] == recorded["size_bytes"]
        assert entries[path]["hash_source"] == "manifest"
    assert entries["zero_action_baseline.json"] == {
        "path": "zero_action_baseline.json",
        "stage": None,
        "role": "baseline",
        "size": 3,
        "sha256": None,
        "hash_source": "none",
    }
    assert entries["artifact_manifest.json"]["hash_source"] == "none"
    assert run_report.write_run_report(run_dir)["report"].is_file()


def test_the_manifest_disagreeing_with_the_files_is_a_warning(tmp_path):
    run_dir = _run(tmp_path / "logs", VELOCIRAPTOR_MONGOLIENSIS)
    _complete_bundle(
        run_dir,
        algorithm="PPO",
        backend="stable-baselines3",
        dirname=lambda stage: stage_dirname(VELOCIRAPTOR_MONGOLIENSIS, stage),
    )
    (run_dir / "01_stance" / "evaluation_final.csv").unlink()
    with (run_dir / "collected_results.csv").open("a", encoding="utf-8") as handle:
        handle.write("a row added by hand\n")
    (run_dir / "02_locomotion" / "models" / "best_model.pkl").unlink()
    warnings = run_index.build_run_index(run_dir)["warnings"]
    assert "artifact_manifest.json lists 2 file(s) that no longer exist: " in " ".join(warnings)
    assert any("01_stance/evaluation_final.csv" in warning for warning in warnings)
    assert any(warning.startswith("1 file(s) changed size since artifact_manifest.json") for warning in warnings)


# ---------------------------------------------------------------------------------------------- reports and index
def test_a_reused_node_links_the_trunk_runs_report(tmp_path):
    logs = tmp_path / "logs"
    run_dir = _run(logs, VELOCIRAPTOR_MONGOLIENSIS, run_id="20261102_000000")
    results, configs = _reused_trunk_bundle_inputs(run_dir)
    save_result_bundle(
        results,
        configs,
        VELOCIRAPTOR_MONGOLIENSIS,
        "PPO",
        42,
        run_dir,
        backend="stable-baselines3",
        backend_version="test-backend-1.0",
        parallel_envs=4,
        evaluation_episodes=3,
        evaluation_seeds=[101, 102, 103],
        plant_identity=_plant_identity(),
        run_id="child-run",
    )
    trunk = run_dir.parent / "trunk-run"
    assert (trunk / "stage1" / "gate_verdict.json").is_file(), "the helper wrote the trunk's judged stance"
    model_dir = VELOCIRAPTOR_MONGOLIENSIS.path(logs)
    written = run_report.write_model_reports(model_dir)
    assert not written.failed
    page = (model_dir / "reports" / "ppo" / run_dir.name / "report.md").read_text(encoding="utf-8")
    trunk_link = "reused from [ppo/trunk-run/stage1](<../trunk-run/report.md>)"
    assert f"| stance | — | reused (PASS) | reward_and_length/v1 | {trunk_link} |" in page
    assert (model_dir / "reports" / "ppo" / "trunk-run" / "report.md").is_file(), "the link resolves"
    assert "| locomotion | stage2 | not judged | — | — |" in page
    assert "**State: complete.**" in page and "Target deliverable: `behavior`." in page
    assert "| locomotion | yes | no | 1 | 1 | stage2/models/best_model.pkl |" in page
    trunk_page = (model_dir / "reports" / "ppo" / "trunk-run" / "report.md").read_text(encoding="utf-8")
    assert "| stance | stage1 | PASS | reward_and_length/v1 | stage1/models/best_model.zip |" in trunk_page


def test_the_index_groups_by_state_newest_first(tmp_path):
    logs = tmp_path / "logs"
    model_id = COMPSOGNATHUS_LONGIPES_ROBOT
    model_dir = model_id.path(logs)
    for run_id, captured in (
        ("older", "2026-10-01T12:00:00.000000+00:00"),
        ("newer", "2026-10-02T12:00:00.000000+00:00"),
    ):
        run_dir = _run(logs, model_id, run_id=run_id)
        _complete_bundle(
            run_dir,
            algorithm="PPO",
            backend="stable-baselines3",
            dirname=lambda stage: stage_dirname(model_id, stage),
            model_id=model_id,
        )
        provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
        recorded = provenance["captured_at"]
        assert len(recorded) == len(captured), "the same length keeps the manifest's size"
        (run_dir / "provenance.json").write_text(
            (run_dir / "provenance.json").read_text(encoding="utf-8").replace(recorded, captured), encoding="utf-8"
        )
    for run_id, captured, passed in (
        ("judged_late", "2026-11-02T00:00:00+00:00", True),
        ("judged_early", "2026-11-01T00:00:00+00:00", False),
    ):
        run_dir = _run(logs, model_id, tree="ppo_screen", run_id=run_id)
        _provenance(run_dir, model_id, captured_at=captured)
        _verdict(run_dir / "01_stance", model_id, 1, passed=passed)
    quick = _run(logs, model_id, tree="ppo_quick_test", run_id="20261103_000000")
    (quick / "01_stance" / "models").mkdir(parents=True)
    (quick / "01_stance" / "models" / "stage1_50000_steps.zip").write_bytes(b"checkpoint")
    (quick / "01_stance" / "stage_config.json").write_text("{}", encoding="utf-8")
    _run(logs, model_id, run_id="stance screen LTSB s52").joinpath("provenance.json").write_text("{}", encoding="utf-8")
    # Not runs: the reports tree, the zero-action baselines, a tree's hidden temporary.
    (model_dir / "zero_action_baselines").mkdir()
    (model_dir / "ppo" / ".ipynb_checkpoints").mkdir()
    rows = run_index.build_model_index(model_dir)
    assert [(row["run_id"], row["state"]) for row in rows] == [
        ("newer", "complete"),
        ("older", "complete"),
        ("judged_late", "judged_no_bundle"),
        ("judged_early", "judged_no_bundle"),
        ("20261103_000000", "training_cut"),
        ("stance screen LTSB s52", "empty"),
    ]
    assert rows[0]["certified"] == ["stance", "locomotion", "behavior"] and rows[0]["provisional"] is False
    assert rows[0]["replicates"] == 1 and rows[0]["handoff"] == f"{stage_dirname(model_id, 3)}/models/best_model.pkl"
    assert rows[2]["passed"] == ["stance"] and rows[2]["handoff"] == "01_stance/models/robust_best_model.zip"
    assert rows[2]["tree"] == "ppo_screen" and rows[2]["kind"] == "screen" and rows[4]["kind"] == "quick_test"
    assert all("warnings" not in row for row in rows), "a row reads less than a report: it carries no warning count"
    written = run_report.write_model_reports(model_dir)
    assert not written.failed and len(written.reports) == 6
    page = (model_dir / "reports" / "INDEX.md").read_text(encoding="utf-8")
    sections = [line for line in page.splitlines() if line.startswith("## ")]
    assert sections == ["## complete", "## judged_no_bundle", "## training_cut", "## empty"]
    assert page.index("| newer |") < page.index("| older |") < page.index("| judged_late |")
    assert "| judged_late | ppo_screen | judged_no_bundle | none published (passed: stance) |" in page
    assert "[report](<ppo/stance screen LTSB s52/report.md>)" in page
    document = json.loads((model_dir / "reports" / "index.json").read_text(encoding="utf-8"))
    assert document["schema"] == run_index.MODEL_INDEX_SCHEMA and document["model"] == model_id.record()
    assert [row["run_id"] for row in document["runs"]] == [row["run_id"] for row in rows]
    run_document = json.loads((model_dir / "reports" / "ppo" / "newer" / "index.json").read_text(encoding="utf-8"))
    assert run_document["schema"] == run_index.RUN_INDEX_SCHEMA and run_document["state"] == "complete"
    assert "summary" not in run_document


def test_the_certificate_and_probe_lines_come_from_the_text_reports(tmp_path, monkeypatch):
    """A six-heading certificate stage whose stage_config.json declares the impulse and hold probes: the impulse
    probe ran, the hold probe did not. The JSON is never opened."""
    from .stance_heading_helpers import HEADING_CURRICULUM

    run_dir = _run(tmp_path / "logs", TYRANNOSAURUS_REX)
    _provenance(run_dir, TYRANNOSAURUS_REX)
    stage_dir = run_dir / "01_stance"
    stage_dir.mkdir()
    curriculum = {**HEADING_CURRICULUM, "stance_probe_impulse_speeds": [1.0], "stance_probe_hold_constant": True}
    _verdict(stage_dir, TYRANNOSAURUS_REX, 1, passed=False, curriculum=curriculum)
    (stage_dir / "stance_heading_report.txt").write_text(
        "policy              robust_best_model.zip\n\n"
        "seeds clean at every heading     38 / 40   LCB95 0.7612   (>= 0.8000, min_eval_episodes 40: the statistic)\n"
        "\nthe 0 deg cell in detail:\nclean episodes  40 / 40\n\nGATE: FAIL\n  - clean_seed_lcb 0.7612 < 0.8000\n",
        encoding="utf-8",
    )
    (stage_dir / "stance_heading_report.json").write_text("not read", encoding="utf-8")
    (stage_dir / "stance_gate_probe_impulse.txt").write_text(
        "PROBE: a lateral impulse applied to the root partway\nthrough each episode.\nmore\n", encoding="utf-8"
    )
    opened: list[str] = []
    real_read_text = Path.read_text

    def read_text(self, *args, **kwargs):
        opened.append(self.name)
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    page = run_index.render_run_report(run_dir)
    assert "stance_heading_report.json" not in opened
    assert (
        "- `stance` certificate, 01_stance/stance_heading_report.txt: GATE: FAIL; seeds clean at every heading 38 / 40 "
        "LCB95 0.7612 (>= 0.8000, min_eval_episodes 40: the statistic)"
    ) in page.replace(
        "[01_stance/stance_heading_report.txt](<01_stance/stance_heading_report.txt>)",
        "01_stance/stance_heading_report.txt",
    )
    assert "PROBE: a lateral impulse applied to the root partway through each episode." in page
    assert "declares the constant hold probe, and it holds no stance_gate_probe_constant.txt" in page
    assert "`stance_gate_report.py --hold-constant --out-dir <a scratch directory>` rolls it again" in page
    assert "impulse probe, and it holds no" not in page
    assert "`stance` failed its gate:\n- min_avg_reward: evidence=0.5 threshold=1.0" in page


@pytest.mark.parametrize("model_id", (TYRANNOSAURUS_REX, COMPSOGNATHUS_LONGIPES_ROBOT, ALLOSAURUS_FRAGILIS), ids=str)
def test_only_a_stage_generate_stage_artifacts_judged_expects_its_probes(tmp_path, model_id):
    """The committed stance stage of each model that declares probes: judged by the command-line curriculum (which
    writes the certificate and rolls no probe), it gets a note and no warning; judged by generate_stage_artifacts, a
    warning per declared probe it lacks."""
    from environments.shared import train_base
    from environments.shared.config import load_stage_config

    for name in ("CURRICULUM_MANAGER_JUDGED_BY", "STANCE_V2_POST_TRAINING_JUDGED_BY"):
        assert getattr(train_base, name).startswith(run_index._COMMAND_LINE_JUDGED_BY)
    assert train_base.STANCE_HEADING_POST_TRAINING_JUDGED_BY.startswith(run_index._COMMAND_LINE_JUDGED_BY)
    curriculum = load_stage_config(model_id, "stance")["curriculum_kwargs"]
    declared = stage_artifacts.declared_stance_probes(curriculum)
    assert declared, "the committed stance stage declares a probe"
    heading = curriculum["gate_kind"] == "stance_quality_heading/v1"
    certificate = "stance_heading_report.txt" if heading else "stance_gate_report.txt"
    command_line = (
        train_base.STANCE_HEADING_POST_TRAINING_JUDGED_BY if heading else train_base.CURRICULUM_MANAGER_JUDGED_BY
    )
    for judged_by in (command_line, stage_artifacts.GATE_VERDICT_JUDGED_BY):
        run_dir = _run(tmp_path / judged_by.replace("/", "_").replace(".", "_"), model_id)
        _provenance(run_dir, model_id)
        stage_dir = run_dir / stage_dirname(model_id, "stance")
        _verdict(stage_dir, model_id, "stance", passed=True, curriculum=curriculum, judged_by=judged_by)
        (stage_dir / certificate).write_text("GATE: PASS\n", encoding="utf-8")
        index = run_index.build_run_index(run_dir)
        missing = [warning for warning in index["warnings"] if "probe, and it holds no" in warning]
        page = run_index.render_run_report(run_dir)
        if judged_by == command_line:
            assert missing == [] and index["summary"]["nodes"][0]["missing_probes"] == []
            assert (
                f"- `stance` probes: none, as expected: its stage_config.json declares the {', '.join(declared)} "
                f"probe(s), and the command-line curriculum that judged it ({judged_by}) rolls no probe."
            ) in page
        else:
            assert len(missing) == len(declared) and index["summary"]["nodes"][0]["missing_probes"] == list(declared)
            assert "rolls no probe" not in page


# ---------------------------------------------------------------------------------------------- the shared probe rule
#: A ``[curriculum]`` block's switches for all five probes.
_EVERY_PROBE = {
    "stance_probe_filter_hz": [5.0],
    "stance_probe_hold_constant": True,
    "stance_probe_release_ablation": True,
    "stance_probe_impulse_speeds": [1.0],
    "stance_probe_spawn_yaw_deg": [90.0],
}
_WRITER_LABELS = {
    "_write_filtered_action_probe": "filtered action",
    "_write_constant_hold_probe": "constant hold",
    "_write_constant_hold_ablation": "release ablation",
    "_write_impulse_probe": "impulse",
    "_write_heading_probe": "heading",
}


def _rolled(tmp_path: Path, stage_config: dict[str, Any], monkeypatch) -> tuple[str, ...]:
    """The probes ``_run_stance_probes`` gets past each writer's switch for (the first stance_report call each makes
    after it is recorded, then raised, so nothing rolls)."""
    from environments.shared.curriculum.stance_heading_gate import STANCE_HEADING_GATE_KIND
    from environments.shared.reporting import stance_report

    from .stance_heading_helpers import heading_report
    from .test_stance_gate_report import _minimal_stance_report

    reached: list[str] = []

    def record(*args, **kwargs):
        reached.append(_WRITER_LABELS[sys._getframe(1).f_code.co_name])
        raise RuntimeError("recorded, not rolled")

    for name in ("build_stance_gate_report", "constant_hold_actions", "impulse_variants", "spawn_yaw_variants"):
        monkeypatch.setattr(stance_report, name, record)
    stage_dir = tmp_path / f"stage_{len(list(tmp_path.iterdir()))}"
    stage_dir.mkdir()
    certificate = stage_config["curriculum_kwargs"].get("gate_kind") == STANCE_HEADING_GATE_KIND
    report = heading_report(stage_dir) if certificate else _minimal_stance_report()
    models = stage_dir / "models"
    models.mkdir(exist_ok=True)
    for name in ("robust_best_model.zip", "robust_best_model_vecnorm.pkl"):
        if not (models / name).exists():
            (models / name).write_bytes(b"pair")
    stage_artifacts._run_stance_probes(
        model_id=TYRANNOSAURUS_REX,
        stage=1,
        stage_config=stage_config,
        stage_dir=stage_dir,
        model_dir=models,
        report=report,
    )
    return tuple(reached)


def test_the_declared_probes_are_the_ones_the_stage_rolls(tmp_path, monkeypatch):
    """For every committed stance stage of the seven models and a block declaring each probe alone: the probes
    ``declared_stance_probes`` names are the ones ``_run_stance_probes`` gets past each writer's switch for."""
    from environments.shared.config import load_all_stages
    from environments.shared.curriculum.gate_schema import STANCE_GATE_KINDS

    from .stance_heading_helpers import HEADING_CURRICULUM

    configs = []
    for model_id in model_ids():
        for config in load_all_stages(model_id).values():
            if config["curriculum_kwargs"].get("gate_kind") in STANCE_GATE_KINDS:
                configs.append(config)
    switches = {
        "stance_probe_filter_hz": [5.0, "nonsense"],
        "stance_probe_hold_constant": True,
        "stance_probe_release_ablation": True,
        "stance_probe_impulse_speeds": [0.0, 1.0],
        "stance_probe_spawn_yaw_deg": [0.0, 90.0],
    }
    v1 = {"gate_kind": "stance_quality/v1", "min_eval_episodes": 40}
    configs += [{"curriculum_kwargs": {**v1, key: value}, "env_kwargs": {}} for key, value in switches.items()]
    configs += [{"curriculum_kwargs": {**v1, key: [0.0]}, "env_kwargs": {}} for key in switches]
    configs.append({"curriculum_kwargs": {**HEADING_CURRICULUM, **switches}, "env_kwargs": {}})
    seen = set()
    for config in configs:
        declared = stage_artifacts.declared_stance_probes(config["curriculum_kwargs"])
        assert _rolled(tmp_path, config, monkeypatch) == declared, config["curriculum_kwargs"]
        seen.update(declared)
    assert seen == set(stage_artifacts.STANCE_PROBE_LABELS)
    assert stage_artifacts.declared_stance_probes({"gate_kind": "reward_and_length/v1", **switches}) == ()


def test_the_runner_calls_every_writer_and_each_writers_switch_decides(tmp_path, monkeypatch):
    """``_run_stance_probes`` is training's code and stays as it was: it calls every writer, whatever the stage
    declares, and each writer's own switch decides (``declared_stance_probes`` only reads that rule back, and
    ``test_the_declared_probes_are_the_ones_the_stage_rolls`` pins that the two agree)."""
    from .test_stance_gate_report import _minimal_stance_report

    called: list[str] = []
    for name, writer_label in _WRITER_LABELS.items():
        monkeypatch.setattr(stage_artifacts, name, lambda _label=writer_label, **kwargs: called.append(_label))
    (tmp_path / "models").mkdir()
    for name in ("robust_best_model.zip", "robust_best_model_vecnorm.pkl"):
        (tmp_path / "models" / name).write_bytes(b"pair")
    curriculum = {"gate_kind": "stance_quality/v1", "min_eval_episodes": 40}
    assert stage_artifacts.declared_stance_probes(curriculum) == ()
    stage_artifacts._run_stance_probes(
        model_id=TYRANNOSAURUS_REX,
        stage=1,
        stage_config={"curriculum_kwargs": curriculum, "env_kwargs": {}},
        stage_dir=tmp_path,
        model_dir=tmp_path / "models",
        report=_minimal_stance_report(),
    )
    assert called == list(stage_artifacts.STANCE_PROBE_LABELS)


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("stance_probe_filter_hz", ["5 Hz"]),
        ("stance_probe_impulse_speeds", ["1 m/s"]),
        ("stance_probe_spawn_yaw_deg", ["ninety"]),
    ),
)
def test_a_probe_value_that_is_no_number_declares_nothing_and_is_logged_at_stage_end(
    tmp_path, caplog, key, value
):
    """A mistyped probe value (a unit inside the string) declares no probe, so the report expects none, and the
    probe's writer still logs it when the stage ends, as it did before the report read the rule."""
    from .test_stance_gate_report import _minimal_stance_report

    (tmp_path / "models").mkdir()
    for name in ("robust_best_model.zip", "robust_best_model_vecnorm.pkl"):
        (tmp_path / "models" / name).write_bytes(b"pair")
    curriculum = {"gate_kind": "stance_quality/v1", "min_eval_episodes": 40, key: value}
    assert stage_artifacts.declared_stance_probes(curriculum) == ()
    with caplog.at_level(logging.WARNING):
        stage_artifacts._run_stance_probes(
            model_id=TYRANNOSAURUS_REX,
            stage=1,
            stage_config={"curriculum_kwargs": curriculum, "env_kwargs": {}},
            stage_dir=tmp_path,
            model_dir=tmp_path / "models",
            report=_minimal_stance_report(),
        )
    assert f"{key} entry is not a number: {value[0]!r}; ignoring it" in caplog.text
    assert not list(tmp_path.glob("stance_*probe*")), "nothing is probed"


def test_the_probe_labels_are_the_runners_and_name_the_writers_files(tmp_path, monkeypatch, caplog):
    from environments.shared.reporting import stance_report

    from .test_stance_gate_report import _minimal_stance_report

    for name in _WRITER_LABELS:
        monkeypatch.setattr(stage_artifacts, name, lambda **kwargs: (_ for _ in ()).throw(RuntimeError("drift")))
    (tmp_path / "models").mkdir()
    for name in ("robust_best_model.zip", "robust_best_model_vecnorm.pkl"):
        (tmp_path / "models" / name).write_bytes(b"pair")
    with caplog.at_level(logging.WARNING):
        stage_artifacts._run_stance_probes(
            model_id=TYRANNOSAURUS_REX,
            stage=1,
            stage_config={"curriculum_kwargs": {"gate_kind": "stance_quality/v1", **_EVERY_PROBE}},
            stage_dir=tmp_path,
            model_dir=tmp_path / "models",
            report=_minimal_stance_report(),
        )
    logged = [record.getMessage() for record in caplog.records if record.getMessage().startswith("Stance probe (")]
    assert [message.split("(")[1].split(")")[0] for message in logged] == list(stage_artifacts.STANCE_PROBE_LABELS)
    assert tuple(run_index.PROBE_REPORTS) == stage_artifacts.STANCE_PROBE_LABELS
    # Each writer's text file is the one PROBE_REPORTS names.
    for name in ("render_constant_hold_probe", "render_constant_hold_ablation"):
        monkeypatch.setattr(stance_report, name, lambda reports, **kwargs: ("text", {}))
    for name in ("render_impulse_probe", "render_heading_probe"):
        monkeypatch.setattr(stance_report, name, lambda policy, statue, **kwargs: ("text", {}))
    report = _minimal_stance_report()
    written = {
        "filtered action": stance_report.write_action_filter_sweep(
            tmp_path, [{**report, "filter_actions_hz": 5.0}], probe_episodes=1
        )["action_filter_sweep_txt"],
        "constant hold": stance_report.write_constant_hold_probe(tmp_path, [report], probe_episodes=1)[
            "constant_hold_probe_txt"
        ],
        "release ablation": stance_report.write_constant_hold_ablation(tmp_path, [report], probe_episodes=1)[
            "constant_hold_ablation_txt"
        ],
        "impulse": stance_report.write_impulse_probe(tmp_path, [report], [report], probe_episodes=1)[
            "impulse_probe_txt"
        ],
        "heading": stance_report.write_heading_probe(tmp_path, [report], [report], probe_episodes=1)[
            "heading_probe_txt"
        ],
    }
    assert {label: path.name for label, path in written.items()} == run_index.PROBE_REPORTS


# ---------------------------------------------------------------------------------------------- the script
def test_the_script_refuses_before_writing_anything(tmp_path, capsys, monkeypatch):
    logs = tmp_path / "logs"
    model_dir = COMPSOGNATHUS_LONGIPES_ROBOT.path(logs)
    not_a_run = model_dir / "ppo" / "nothing_here"
    not_a_run.mkdir(parents=True)
    with pytest.raises(run_report.RunReportError, match="is no run directory.*pass LOG_BASE/<species>/<embodiment>/"):
        run_report.write_run_report(not_a_run)
    elsewhere = _run(tmp_path / "elsewhere", COMPSOGNATHUS_LONGIPES_ROBOT)
    _provenance(elsewhere, COMPSOGNATHUS_LONGIPES_ROBOT)
    with pytest.raises(run_report.RunReportError, match="is not a run inside the model directory"):
        run_report.write_run_report(elsewhere, model_dir=model_dir)
    with pytest.raises(run_report.RunReportError, match="is a run, not a model directory: pass the run as RUN_DIR"):
        run_report.write_run_report(elsewhere, model_dir=elsewhere)
    # A directory inside a run, passed as the run: its stage directory.
    _verdict(elsewhere / "01_stance", COMPSOGNATHUS_LONGIPES_ROBOT, 1, passed=True)
    with pytest.raises(run_report.RunReportError, match="this looks like a directory inside a run: pass the run"):
        run_report.write_run_report(elsewhere / "01_stance", model_dir=model_dir)
    # A model directory inside a run: its reports/ would land in the run, whose manifest hashes every file.
    nested_model = elsewhere / "copied" / "compsognathus_longipes" / "robot"
    nested = _run(elsewhere / "copied", COMPSOGNATHUS_LONGIPES_ROBOT)
    _provenance(nested, COMPSOGNATHUS_LONGIPES_ROBOT)
    before = _files(elsewhere)
    with pytest.raises(run_report.RunReportError, match="inside the run directory"):
        run_report.write_run_report(nested, model_dir=nested_model)
    with pytest.raises(run_report.RunReportError, match="inside the run directory"):
        run_report.write_model_reports(nested_model)
    assert _files(elsewhere) == before
    assert not (model_dir / "reports").exists() and not (nested_model / "reports").exists()
    with pytest.raises(run_report.RunReportError, match="is no directory: pass --model-dir"):
        run_report.write_model_reports(model_dir / "missing")
    with pytest.raises(SystemExit, match="is no run directory"):
        run_report.main([str(not_a_run)])
    with pytest.raises(SystemExit):
        run_report.main([])
    capsys.readouterr()


def test_a_wrong_model_directory_is_refused(tmp_path, monkeypatch, capsys):
    """A run tree, a stage directory of a run without provenance.json, and a model directory inside such a run: none
    is a model directory, and reports/ there would sit inside a tree the trunk and replicate scans read, or a run."""
    logs = tmp_path / "logs"
    model_dir = VELOCIRAPTOR_MONGOLIENSIS.path(logs)
    flat, _ = _state_run("training_cut", logs)  # a Brachiosaurus flat run without provenance.json
    legacy = VELOCIRAPTOR_MONGOLIENSIS.path(logs) / "ppo_20260225_143021"
    (legacy / "stage1" / "models").mkdir(parents=True)
    (legacy / "stage1" / "stage_summary.txt").write_text("summary\n", encoding="utf-8")
    (legacy / "stage1" / "models" / "stage1_final.zip").write_bytes(b"checkpoint")
    inside = legacy / "copy" / "velociraptor_mongoliensis" / "anatomical"
    inside.mkdir(parents=True)
    run_dir = _run(logs, VELOCIRAPTOR_MONGOLIENSIS)
    _provenance(run_dir, VELOCIRAPTOR_MONGOLIENSIS)
    for wrong, message in (
        (model_dir / "ppo", "is a run tree, not a model directory: pass its parent"),
        (legacy / "stage1", "is not a model directory LOG_BASE/<species>/<embodiment>"),
        (flat / "stage1", "is not a model directory LOG_BASE/<species>/<embodiment>"),
        (inside, f"lies inside the run directory {legacy}"),
    ):
        before = _files(tmp_path)
        with pytest.raises(SystemExit, match=message):
            run_report.main(["--model-dir", str(wrong)])
        with pytest.raises(SystemExit, match=message):
            run_report.main([str(run_dir), "--model-dir", str(wrong)])
        assert _files(tmp_path) == before, wrong
    assert not list(tmp_path.rglob("reports"))
    # Relative paths are resolved: a bare run id typed inside the tree, and `--model-dir .`.
    monkeypatch.chdir(run_dir.parent)
    assert run_report.main([run_dir.name]) == 0
    assert not (run_dir.parent / "reports").exists()
    index = model_dir / "reports" / "INDEX.md"
    assert index.read_text(encoding="utf-8").startswith("# Runs of Velociraptor mongoliensis (Anatomical)\n")
    monkeypatch.chdir(model_dir)
    assert run_report.main(["--model-dir", "."]) == 0
    document = json.loads((model_dir / "reports" / "index.json").read_text(encoding="utf-8"))
    assert document["model"] == VELOCIRAPTOR_MONGOLIENSIS.record()
    assert [(row["run_id"], row["state"]) for row in document["runs"]] == [
        (legacy.name, "training_cut"),
        (run_dir.name, "empty"),
    ]
    capsys.readouterr()


def test_a_report_folder_is_replaced_whole_and_kept_when_a_build_fails(tmp_path, monkeypatch, capsys):
    logs = tmp_path / "logs"
    run_dir = _run(logs, BRACHIOSAURUS_ALTITHORAX)
    _provenance(run_dir, BRACHIOSAURUS_ALTITHORAX)
    _verdict(run_dir / "01_stance", BRACHIOSAURUS_ALTITHORAX, 1, passed=False)
    run_before = _files(run_dir)
    assert run_report.main([str(run_dir)]) == 0
    folder = BRACHIOSAURUS_ALTITHORAX.path(logs) / "reports" / "ppo" / run_dir.name
    out = capsys.readouterr().out
    assert f"written: {folder / 'report.md'}" in out and f"written: {folder.parent.parent / 'INDEX.md'}" in out
    # Nothing changed: nothing is rewritten (Drive keeps the files, and no old copy goes to its trash).
    kept = _files(folder.parent.parent)
    assert run_report.main([str(run_dir)]) == 0
    out = capsys.readouterr().out
    assert f"unchanged: {folder / 'report.md'}" in out and f"unchanged: {folder.parent.parent / 'INDEX.md'}" in out
    assert _files(folder.parent.parent) == kept
    # Another tool's output in the folder is carried into the rebuilt one, and named.
    (folder / "stale.txt").write_text("another tool's output", encoding="utf-8")
    (folder / "rerun").mkdir()
    (folder / "rerun" / "stance_gate_report.txt").write_text("GATE: PASS\n", encoding="utf-8")
    (folder / "report.md").write_text("an older page\n", encoding="utf-8")
    written = run_report.write_run_report(run_dir)
    assert "report" not in written.unchanged and written.kept == ["rerun", "stale.txt"]
    assert f"kept in {folder} (not written by run_report, carried into the rebuilt folder): rerun, stale.txt" in (
        capsys.readouterr().out
    )
    assert sorted(path.name for path in folder.iterdir()) == ["index.json", "report.md", "rerun", "stale.txt"]
    assert (folder / "rerun" / "stance_gate_report.txt").read_text(encoding="utf-8") == "GATE: PASS\n"
    assert (folder / "report.md").read_text(encoding="utf-8").startswith("# Run ppo/")
    kept = _files(folder)

    def fail(*args, **kwargs):
        raise RuntimeError("the build failed")

    monkeypatch.setattr(run_index, "render_run_report", fail)
    with pytest.raises(RuntimeError, match="the build failed"):
        run_report.write_run_report(run_dir)
    assert _files(folder) == kept
    monkeypatch.undo()
    (folder / "report.md").write_text("an older page\n", encoding="utf-8")
    kept = _files(folder)
    real_write_text = Path.write_text

    def fail_on_the_index(self, text, *args, **kwargs):
        if self.name == "index.json" and self.parent.name.endswith(".tmp"):
            raise OSError("disk full")
        return real_write_text(self, text, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_on_the_index)
    with pytest.raises(OSError, match="disk full"):
        run_report.write_run_report(run_dir)
    assert _files(folder) == kept
    monkeypatch.undo()
    real_rename = Path.rename

    def fail_the_new_folder(self, target):
        if self.name.endswith(".tmp"):
            raise OSError("Drive went away")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", fail_the_new_folder)
    with pytest.raises(OSError, match="Drive went away"):
        run_report.write_run_report(run_dir)
    monkeypatch.undo()
    assert _files(folder) == kept, "the old folder is put back"
    assert [path.name for path in folder.parent.iterdir()] == [run_dir.name], "no temporary is left behind"
    assert _files(run_dir) == run_before, "the run is only read"
    capsys.readouterr()


def test_a_report_folder_and_its_temporaries_survive_a_lost_runtime(tmp_path, capsys):
    """What a runtime lost mid-replace leaves (a staging folder, the old folder renamed aside, an index temporary)
    is cleared by the next build, and the old folder is put back first when the report folder is missing."""
    logs = tmp_path / "logs"
    run_dir = _run(logs, DIBOTHROSUCHUS_ELAPHROS)
    _provenance(run_dir, DIBOTHROSUCHUS_ELAPHROS)
    _verdict(run_dir / "01_stance", DIBOTHROSUCHUS_ELAPHROS, 1, passed=True)
    run_report.write_run_report(run_dir)
    folder = DIBOTHROSUCHUS_ELAPHROS.path(logs) / "reports" / "ppo" / run_dir.name
    (folder / "notes.txt").write_text("mine", encoding="utf-8")
    # Lost between renaming the old folder aside and renaming the new one in.
    folder.rename(folder.parent / f".{run_dir.name}.old-0123abcd")
    staging = folder.parent / f".{run_dir.name}.ab_c1234.tmp"
    staging.mkdir()
    (staging / "report.md").write_text("half", encoding="utf-8")
    stray = folder.parent.parent / ".INDEX.md.q1w2e3r4.tmp"
    stray.write_text("half", encoding="utf-8")
    other = folder.parent / ".another_run.ab_c1234.tmp"
    other.mkdir()
    run_report.write_run_report(run_dir)
    assert sorted(path.name for path in folder.parent.iterdir()) == [".another_run.ab_c1234.tmp", run_dir.name]
    assert (folder / "notes.txt").read_text(encoding="utf-8") == "mine", "the old folder was put back first"
    assert not stray.exists()
    capsys.readouterr()


def test_report_folders_get_the_default_mode(tmp_path):
    previous = os.umask(0o022)
    try:
        run_dir = _run(tmp_path / "logs", COMPSOGNATHUS_LONGIPES_ROBOT)
        _provenance(run_dir, COMPSOGNATHUS_LONGIPES_ROBOT)
        written = run_report.write_run_report(run_dir)
    finally:
        os.umask(previous)
    assert stat.S_IMODE(written["report"].parent.stat().st_mode) == 0o755
    assert stat.S_IMODE(written["report"].stat().st_mode) == 0o644
    assert stat.S_IMODE(written["model_index"].stat().st_mode) == 0o644


def test_model_mode_keeps_going_when_one_run_fails(tmp_path, monkeypatch, capsys):
    logs = tmp_path / "logs"
    model_dir = DIBOTHROSUCHUS_ELAPHROS.path(logs)
    for run_id in ("20261101_000000", "20261102_000000"):
        run_dir = _run(logs, DIBOTHROSUCHUS_ELAPHROS, run_id=run_id)
        _provenance(run_dir, DIBOTHROSUCHUS_ELAPHROS)
        _verdict(run_dir / "01_stance", DIBOTHROSUCHUS_ELAPHROS, 1, passed=True)
    assert run_report.main(["--model-dir", str(model_dir)]) == 0
    kept = _files(model_dir / "reports" / "ppo" / "20261101_000000")
    real = run_index.build_run_index

    def fail_one(run_dir, **kwargs):
        if Path(run_dir).name == "20261101_000000":
            raise ValueError("unforeseen")
        return real(run_dir, **kwargs)

    monkeypatch.setattr(run_index, "build_run_index", fail_one)
    capsys.readouterr()
    assert run_report.main(["--model-dir", str(model_dir)]) == 1
    out = capsys.readouterr().out
    assert "not written (its previous report, if any, is kept): " in out and "ValueError: unforeseen" in out
    assert "rerun `python -m environments.shared.scripts.run_report " in out
    assert f"unchanged: {model_dir / 'reports' / 'ppo' / '20261102_000000' / 'report.md'}" in out
    assert _files(model_dir / "reports" / "ppo" / "20261101_000000") == kept
    assert "| 20261101_000000 | ppo | judged_no_bundle |" in (model_dir / "reports" / "INDEX.md").read_text()


def test_a_trunk_report_not_built_yet_is_named_not_linked(tmp_path, capsys):
    logs = tmp_path / "logs"
    run_dir = _run(logs, VELOCIRAPTOR_MONGOLIENSIS, run_id="20261102_000000")
    _provenance(run_dir, VELOCIRAPTOR_MONGOLIENSIS)
    from .result_bundle_helpers import _write_ancestor_record

    _write_ancestor_record(run_dir, 1, model_id=VELOCIRAPTOR_MONGOLIENSIS)
    trunk = run_dir.parent / "trunk-run"
    written = run_report.write_run_report(run_dir)
    page = written["report"].read_text(encoding="utf-8")
    not_built = "reused from ppo/trunk-run/stage1 (its report is not built yet) |"
    assert f"| stance | — | reused (PASS) | reward_and_length/v1 | {not_built}" in page
    run_report.write_run_report(trunk)
    written = run_report.write_run_report(run_dir)
    assert "report" not in written.unchanged, "the page changed once the trunk's report was built"
    assert "reused from [ppo/trunk-run/stage1](<../trunk-run/report.md>)" in written["report"].read_text("utf-8")
    capsys.readouterr()


def test_a_run_the_index_cannot_read_is_listed_as_unreadable_with_its_error(tmp_path, monkeypatch):
    import errno

    logs = tmp_path / "logs"
    model_id = COMPSOGNATHUS_LONGIPES_ROBOT
    model_dir = model_id.path(logs)
    broken = _run(logs, model_id, run_id="20261101_000000")
    _complete_bundle(
        broken,
        algorithm="PPO",
        backend="stable-baselines3",
        dirname=lambda stage: stage_dirname(model_id, stage),
        model_id=model_id,
    )
    fine = _run(logs, model_id, run_id="20261102_000000")
    _provenance(fine, model_id)
    _verdict(fine / "01_stance", model_id, 1, passed=True)
    real = run_index.read_gate_verdict

    def drive_read_error(stage_dir):
        if broken in Path(stage_dir).parents:
            raise OSError(errno.EIO, "Input/output error")
        return real(stage_dir)

    monkeypatch.setattr(run_index, "read_gate_verdict", drive_read_error)
    rows = run_index.build_model_index(model_dir)
    assert [(row["run_id"], row["state"]) for row in rows] == [
        ("20261102_000000", "judged_no_bundle"),
        ("20261101_000000", "unreadable"),
    ]
    assert rows[1]["error"] == "OSError: [Errno 5] Input/output error"
    page = run_index.render_model_index(model_dir, rows)
    assert "2 run(s): 1 judged_no_bundle, 1 unreadable." in page
    assert [line for line in page.splitlines() if line.startswith("## ")] == ["## judged_no_bundle", "## unreadable"]
    assert (
        "| 20261101_000000 | ppo | unreadable | — | — | — | — | not built; the index could not read the run: OSError: "
        "[Errno 5] Input/output error |"
    ) in page


def test_each_record_is_read_once_and_an_index_row_reads_less(tmp_path, monkeypatch):
    logs = tmp_path / "logs"
    model_id = TYRANNOSAURUS_REX
    runs = []
    for run_id in ("20261101_000000", "20261102_000000"):
        run_dir = _run(logs, model_id, run_id=run_id)
        _provenance(run_dir, model_id)
        _verdict(run_dir / "01_stance", model_id, 1, passed=True)
        (run_dir / "01_stance" / "stance_gate_report.txt").write_text("GATE: PASS\n", encoding="utf-8")
        runs.append(run_dir)
    opened: list[str] = []
    real_read_text = Path.read_text

    def read_text(self, *args, **kwargs):
        opened.append(self.name)
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    run_index.build_run_index(runs[0])
    assert opened.count("provenance.json") == 1 and opened.count("stage_config.json") == 1
    assert opened.count("stance_gate_report.txt") == 1
    opened.clear()
    run_index.build_model_index(model_id.path(logs))
    assert opened.count("provenance.json") == 2
    assert opened.count("stage_config.json") == 0 and opened.count("stance_gate_report.txt") == 0


def _unreadable_case(case: str, run_dir: Path, model_id: ModelId) -> str:
    """Make one record of the run at *run_dir* unreadable, missing or wrong; the warning the report gives."""
    stage_dir = run_dir / "01_stance"
    if case == "checkpoint deleted":
        _verdict(stage_dir, model_id, 1, passed=True)
        (stage_dir / "models" / "robust_best_model.zip").unlink()
        return (
            "01_stance/gate_verdict.json names its checkpoint 01_stance/models/robust_best_model.zip, which no longer "
            "exists (the verdict is bound to that file: re-judge the stage before reusing it)"
        )
    if case == "trunk handoff gone":
        from .result_bundle_helpers import _write_ancestor_record

        _write_ancestor_record(run_dir, 1, model_id=model_id)
        (run_dir.parent / "trunk-run" / "stage1" / "models" / "best_model.zip").unlink()
        return "ancestors/stance/ancestor.json names the trunk's model_path "
    if case == "ancestor record corrupt":
        from .result_bundle_helpers import _write_ancestor_record

        record_dir, _ = _write_ancestor_record(run_dir, 1, model_id=model_id)
        (record_dir / "ancestor.json").write_text("[]", encoding="utf-8")
        return "ancestors/stance/ancestor.json is missing or cannot be read"
    if case == "verdict corrupt":
        _verdict(stage_dir, model_id, 1, passed=True)
        (stage_dir / "gate_verdict.json").write_text("{not json", encoding="utf-8")
        return "01_stance/gate_verdict.json: "
    if case == "provenance corrupt":
        (run_dir / "provenance.json").write_text("{not json", encoding="utf-8")
        return "provenance.json cannot be read, or holds no JSON object"
    if case == "provenance names no model":
        (run_dir / "provenance.json").write_text(json.dumps({"species": "nobody"}), encoding="utf-8")
        return "provenance.json names no model: "
    if case == "certificate not UTF-8":
        _verdict(stage_dir, model_id, 1, passed=True)
        (stage_dir / "stance_gate_report.txt").write_bytes(b"\xff\xfeGATE: PASS")
        return "01_stance/stance_gate_report.txt cannot be read: "
    if case == "certificate a directory":
        _verdict(stage_dir, model_id, 1, passed=True)
        (stage_dir / "stance_gate_report.txt").mkdir()
        return ""
    if case == "stage config wrong type":
        _verdict(stage_dir, model_id, 1, passed=True)
        (stage_dir / "stage_config.json").write_text("[1, 2]", encoding="utf-8")
        return ""
    if case == "trunk record corrupt":
        (run_dir / "trunk_run.json").write_text("{not json", encoding="utf-8")
        return "trunk_run.json: "
    assert case == "manifest without a file list"
    (run_dir / "artifact_manifest.json").write_text(json.dumps({"status": "partial"}), encoding="utf-8")
    return "artifact_manifest.json records no file list; its files are listed without digests"


@pytest.mark.parametrize(
    "case",
    (
        "checkpoint deleted",
        "trunk handoff gone",
        "ancestor record corrupt",
        "verdict corrupt",
        "provenance corrupt",
        "provenance names no model",
        "certificate not UTF-8",
        "certificate a directory",
        "stage config wrong type",
        "trunk record corrupt",
        "manifest without a file list",
    ),
)
def test_an_unreadable_or_missing_record_is_a_warning_and_the_report_is_written(tmp_path, case, capsys):
    model_id = BRACHIOSAURUS_ALTITHORAX
    run_dir = _run(tmp_path / "logs", model_id)
    _provenance(run_dir, model_id)
    expected = _unreadable_case(case, run_dir, model_id)
    before = _files(run_dir)
    written = run_report.write_run_report(run_dir)
    warnings = json.loads(written["index"].read_text(encoding="utf-8"))["warnings"]
    page = written["report"].read_text(encoding="utf-8")
    if expected:
        assert [warning for warning in warnings if warning.startswith(expected)], warnings
        assert f"- {expected}" in page
    if case == "verdict corrupt":
        assert "| stance | 01_stance | unreadable |" in page
    if case == "certificate not UTF-8":
        assert "certificate, [01_stance/stance_gate_report.txt](<" in page and "): unreadable" in page
    assert _files(run_dir) == before, "the run is only read"
    capsys.readouterr()


def _helpers_namespace(run_dir: Path, model_id: ModelId, **names: Any) -> dict[str, Any]:
    namespace: dict[str, Any] = {"RUN_DIR": run_dir, "MODEL_ID": model_id, "Path": Path, **names}
    exec_top_level_def(code_cell(SB3_NOTEBOOK, "def train_stage("), "write_run_report_quietly", namespace)
    return namespace


@pytest.mark.parametrize("model_id", (TYRANNOSAURUS_REX, COMPSOGNATHUS_LONGIPES_ROBOT), ids=str)
def test_the_notebook_helper_writes_the_report_and_never_raises(tmp_path, monkeypatch, capsys, model_id):
    logs = tmp_path / "logs"
    run_dir = _run(logs, model_id)
    _provenance(run_dir, model_id)
    _verdict(run_dir / "01_stance", model_id, 1, passed=True)
    before = _files(run_dir)
    namespace = _helpers_namespace(run_dir, model_id, LOG_BASE=logs)
    namespace["write_run_report_quietly"]()
    report = model_id.path(logs) / "reports" / "ppo" / run_dir.name / "report.md"
    assert capsys.readouterr().out.splitlines() == [
        f"Run report: {report} (model index: {model_id.path(logs) / 'reports' / 'INDEX.md'})"
    ]
    # Without the storage cell's LOG_BASE, with the tool refusing, and with it raising: one warning line each.
    for names, patch in (
        ({}, None),
        ({"LOG_BASE": logs, "RUN_DIR": run_dir / "01_stance"}, None),
        ({"LOG_BASE": logs}, RuntimeError("a library without the tool")),
    ):
        namespace = _helpers_namespace(names.pop("RUN_DIR", run_dir), model_id, **names)
        if patch is not None:
            monkeypatch.setattr(
                run_report, "write_run_report", lambda *a, error=patch, **k: (_ for _ in ()).throw(error)
            )
        namespace["write_run_report_quietly"]()
        [line] = capsys.readouterr().out.splitlines()
        assert line.startswith("WARNING: the run report was not written (")
        assert line.endswith(
            "the run is unaffected. Rebuild it later with `python -m environments.shared.scripts.run_report "
            "<RUN_DIR>`, or by running the Drive summary notebook."
        )
        monkeypatch.undo()
    assert _files(run_dir) == before


def test_save_run_bundle_writes_the_report_after_the_bundle(tmp_path, capsys):
    logs = tmp_path / "logs"
    run_dir = _run(logs, VELOCIRAPTOR_MONGOLIENSIS)
    _provenance(run_dir, VELOCIRAPTOR_MONGOLIENSIS)
    _verdict(run_dir / "01_stance", VELOCIRAPTOR_MONGOLIENSIS, 1, passed=True)
    order: list[str] = []
    namespace = _helpers_namespace(run_dir, VELOCIRAPTOR_MONGOLIENSIS, LOG_BASE=logs)
    real = namespace["write_run_report_quietly"]
    namespace.update(
        TARGET_NODE=load_stage_manifest(VELOCIRAPTOR_MONGOLIENSIS).resolve("stance"),
        STAGE_CONFIGS={},
        ALGORITHM="ppo",
        SEED=42,
        HARDWARE_LABEL="local",
        N_ENVS=4,
        CHECKPOINT_SELECTION_SEED=1042,
        EVALUATION_SEED=3042,
        PUBLICATION_SEED_START=3042,
        PLANT_IDENTITY=type("Identity", (), {"to_dict": staticmethod(dict)})(),
        _ACTIVE_RUN_ID=run_dir.name,
        repo_root=None,
        discover_replicates_for_run=lambda *args, **kwargs: {},
        _lib_save_result_bundle=lambda *args, **kwargs: order.append("bundle") or {},
        write_run_report_quietly=lambda: (order.append("report"), real()),
    )
    exec_top_level_def(code_cell(SB3_NOTEBOOK, "def train_stage("), "save_run_bundle", namespace)
    namespace["save_run_bundle"]([], model_id=VELOCIRAPTOR_MONGOLIENSIS)
    assert order == ["bundle", "report"]
    assert (VELOCIRAPTOR_MONGOLIENSIS.path(logs) / "reports" / "ppo" / run_dir.name / "report.md").is_file()
    capsys.readouterr()


def test_the_verify_cell_rewrites_the_report_last():
    tree = ast.parse(code_cell(SB3_NOTEBOOK, "# Verify the bundle without rewriting immutable artifacts."))
    last = tree.body[-1]
    assert isinstance(last, ast.Expr) and ast.unparse(last.value) == "write_run_report_quietly()"


def test_the_storage_cell_prints_the_first_lines_of_the_model_index(tmp_path, monkeypatch, capsys):
    from .test_run_id_rules import _run_storage, _storage_namespace

    namespace, _ = _storage_namespace(tmp_path, monkeypatch, COMPSOGNATHUS_LONGIPES_ROBOT)
    _run_storage(namespace)
    assert "Run index" not in capsys.readouterr().out
    index = COMPSOGNATHUS_LONGIPES_ROBOT.path(tmp_path / "logs") / "reports" / "INDEX.md"
    index.parent.mkdir(parents=True)
    index.write_text("\n".join(f"line {number}" for number in range(1, 16)) + "\n", encoding="utf-8")
    namespace, _ = _storage_namespace(tmp_path, monkeypatch, COMPSOGNATHUS_LONGIPES_ROBOT)
    _run_storage(namespace)
    out = capsys.readouterr().out
    assert f"Run index ({index}), first lines:\nline 1\n" in out
    assert "line 12\n... (3 more lines)" in out and "line 13" not in out
    index.write_bytes(b"\xff\xfe not UTF-8")
    namespace, _ = _storage_namespace(tmp_path, monkeypatch, COMPSOGNATHUS_LONGIPES_ROBOT)
    _run_storage(namespace)
    assert f"Run index:     {index} cannot be read (" in capsys.readouterr().out


# ---------------------------------------------------------------------------------------------- the Drive summary
def test_the_drive_walker_never_enters_the_reports_tree(tmp_path, monkeypatch):
    from environments.shared.stage_manifest import stage_ref_from_dirname

    model_dir = TYRANNOSAURUS_REX.path(tmp_path / "logs")
    run_dir = _run(tmp_path / "logs", TYRANNOSAURUS_REX)
    _provenance(run_dir, TYRANNOSAURUS_REX)
    _verdict(run_dir / "01_stance", TYRANNOSAURUS_REX, 1, passed=True)
    run_report.write_model_reports(model_dir)
    # A report folder holding what makes a run directory a run, as a hand-copied one could.
    (model_dir / "reports" / "ppo" / run_dir.name / "collected_results.csv").write_text("stage\n1\n", encoding="utf-8")
    namespace: dict[str, Any] = {"stage_ref_from_dirname": stage_ref_from_dirname, "re": __import__("re")}
    namespace["datetime"] = __import__("datetime").datetime
    exec_top_level_def(code_cell(DRIVE_SUMMARY, "def parse_run_dir_name("), "parse_sweep_dir_name", namespace)
    walker = code_cell(DRIVE_SUMMARY, "def discover_runs(")
    for name in ("_is_run_dir", "_is_sweep_dir", "discover_runs"):
        exec_top_level_def(walker, name, namespace)
    listed: list[Path] = []
    real_iterdir = Path.iterdir

    def iterdir(self):
        listed.append(self)
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", iterdir)
    runs, sweeps = namespace["discover_runs"](model_dir)
    assert runs == [run_dir] and sweeps == []
    assert not [path for path in listed if (model_dir / "reports") in (path, *path.parents)]


def test_the_drive_summary_cell_rebuilds_every_models_reports_best_effort(tmp_path, monkeypatch, capsys):
    pd = pytest.importorskip("pandas")
    from environments.shared import reporting

    monkeypatch.setattr(reporting, "OFFLINE_JUDGEMENTS_PATH", tmp_path / "no_ledger.csv")
    logs = tmp_path / "drive" / "logs"
    for model_id in (COMPSOGNATHUS_LONGIPES_ROBOT, BRACHIOSAURUS_ALTITHORAX):
        run_dir = _run(logs, model_id)
        _provenance(run_dir, model_id)
        _verdict(run_dir / "01_stance", model_id, 1, passed=True)
    before = _files(COMPSOGNATHUS_LONGIPES_ROBOT.path(logs) / "ppo")

    def run_cells() -> str:
        namespace = {"df": pd.DataFrame([]), "LOGS_DIR": logs, "display": lambda value: None}
        exec(compile(code_cell(DRIVE_SUMMARY, "import pandas as pd"), "summary_imports", "exec"), namespace)
        exec(compile(code_cell(DRIVE_SUMMARY, "write_runs_summary("), "summary_export", "exec"), namespace)
        return str(capsys.readouterr().out)

    out = run_cells()
    for model_id, name in (
        (COMPSOGNATHUS_LONGIPES_ROBOT, "Compsognathus longipes (Robot)"),
        (BRACHIOSAURUS_ALTITHORAX, "Brachiosaurus altithorax (Anatomical)"),
    ):
        index = model_id.path(logs) / "reports" / "INDEX.md"
        assert f"  {name} run reports: 1 written, 0 unchanged, index {index}\n" in out
        assert "| 20261121_175147 | ppo | judged_no_bundle |" in index.read_text(encoding="utf-8")
    assert _files(COMPSOGNATHUS_LONGIPES_ROBOT.path(logs) / "ppo") == before, "the runs are only read"
    real = run_report.write_model_reports

    def fail_robot(model_dir):
        if Path(model_dir).name == "robot":
            raise OSError("Drive went away")
        return real(model_dir)

    monkeypatch.setattr(run_report, "write_model_reports", fail_robot)
    out = run_cells()
    robot = COMPSOGNATHUS_LONGIPES_ROBOT.path(logs)
    assert (
        "  WARNING: Compsognathus longipes (Robot) run reports not written (OSError: Drive went away); rerun this "
        f"cell, or `python -m environments.shared.scripts.run_report --model-dir {robot}`"
    ) in out
    assert "  Brachiosaurus altithorax (Anatomical) run reports: 0 written, 1 unchanged, index " in out
    monkeypatch.undo()
    monkeypatch.setattr(reporting, "OFFLINE_JUDGEMENTS_PATH", tmp_path / "no_ledger.csv")
    # A run whose report fails is named, with its error.
    real_build = run_index.build_run_index

    def fail_the_robot_run(run_dir, **kwargs):
        if "robot" in Path(run_dir).parts:
            raise ValueError("unforeseen")
        return real_build(run_dir, **kwargs)

    monkeypatch.setattr(run_index, "build_run_index", fail_the_robot_run)
    out = run_cells()
    robot_run = robot / "ppo" / "20261121_175147"
    assert f"run reports: 0 written, 0 unchanged, index {robot / 'reports' / 'INDEX.md'}; 1 not written:\n" in out
    assert f"    {robot_run}: ValueError: unforeseen (its previous report, if any, is kept)\n" in out
