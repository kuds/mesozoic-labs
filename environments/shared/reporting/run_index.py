"""What a run directory holds, read back without touching it: the per-run report and the per-model index.

The maintainer chose "reports and index" on 2026-11-26: a derived tree beside
the run trees of each model, never inside a run, never hashed, rebuilt whole::

    LOG_BASE/<species>/<embodiment>/
    ├── <algo>/<run_id>/ ...             the runs (environments/shared/run_tree.py)
    ├── <algo>_screen/<run_id>/ ...
    ├── <algo>_quick_test/<run_id>/ ...
    └── reports/
        ├── INDEX.md                     one row per run of the model, grouped by state, newest first
        ├── index.json                   the same rows (MODEL_INDEX_SCHEMA)
        └── <algo>[_screen|_quick_test]/<run_id>/
            ├── report.md                the one page for a run
            └── index.json               every file of the run (RUN_INDEX_SCHEMA)

A run's report folder mirrors the run's path under the model directory, so a
run directly in the model directory (a flat ``<algo>_<YYYYMMDD_HHMMSS>/`` run
from before 2026-03-31, or a command-line ``curriculum_<timestamp>/`` run,
``train_base.train_curriculum``'s default) has its folder at
``reports/<run name>/``.  ``scripts/run_report.py`` writes the tree; this
module only reads.  The tool owns ``report.md`` and ``index.json`` of each
report folder and the two index files; anything else in a report folder is
carried into each rebuilt folder, and a folder of ``reports/`` that is no
run's report folder is never touched.

Placement rule (part of the same decision): a stage directory's root holds
only files some reader names by a fixed path, plus the ledgers; report-only
output goes in ``figures/``, ``replays/`` or this out-of-run ``reports/`` tree;
a new stage-root file needs a fixed-path reader.  Nothing inside a run, and no
manifest rule, changes: this module opens files for reading only.

Run states (:func:`run_state`), from the files a run holds:

* ``complete``, ``partial`` and ``failed``: the status its
  ``artifact_manifest.json`` records (``result_bundle.read_bundle_status``);
  ``failed`` is a bundle in which no deliverable is certified;
* ``bundle_unreadable``: an ``artifact_manifest.json`` that cannot be read or
  records another status, so the bundle's status is unknown;
* without a manifest, by the most advanced stage directory:
  ``judged_no_bundle`` (a stage holds ``gate_verdict.json``),
  ``post_training_cut`` (evaluation CSVs, ``evaluation_selected.csv`` or
  ``evaluation_final.csv``, but no verdict), ``training_cut`` (a checkpoint in
  ``models/``, no evaluation) and ``empty`` (nothing trained: at most the
  records written before training, or the ``ancestors/`` records of reused
  nodes).

The model index adds one group, ``unreadable``, for a run whose records
raised something unforeseen while the index read them; its row shows the
error.

The states describe files, not history.  A run's ``layout`` is the
generation of its stage directories: ``NN_<stage_id>`` (from 2026-08-20),
``stage<N>`` or ``bare <stage_id>`` (both older than ``gate_verdict.json``,
so a finished run of theirs reads as ``training_cut`` or
``post_training_cut``, and its report says so), or ``none``.

A node a run holds as an ``ancestors/<stage_id>/`` record is reused, whatever
else the run holds: when the run also holds its own stage directory for the
node (its own verdict failed or no longer passed the reuse rule, and the chain
loop reused the trunk's certified node), the record stands for the node, as
the bundle writer reads it, and the report names that directory as kept but
not published and reads nothing else from it.

The report expects a stance stage's declared probes
(``stage_artifacts.declared_stance_probes``) only when
``generate_stage_artifacts`` judged it (its verdict's ``judged_by``), since
that is the one path that rolls them; the command-line curriculum's judge
writes the certificate and rolls no probe.

A run of the certification tree that ``run_exclusions.excluded_run_reason``
names (the nine r10 stance screens, and runs built on them) never counts, and
its row and page say so.

Cost: every reader here opens only small records (``provenance.json`` once
per run, ``trunk_run.json``, ``stage_config.json`` once per stage,
``gate_verdict.json``, ``ancestor.json``, the certificate and probe ``.txt``
reports) and the artifact manifest, and lists the run's files.  A model
index row reads less: the manifest's status, ``provenance.json``, each
stage's ``gate_verdict.json``, the names under ``ancestors/``, and for a run
of the certification tree what ``run_exclusions.excluded_run_reason`` reads
(``provenance.json`` and each stage's ``stage_config.json`` and
``gate_verdict.json``).  Sizes and digests come from
``artifact_manifest.json`` when the run has one; nothing is ever hashed.  A
partial, cut-off, legacy or empty run is a state the report names, with the
next step for an unfinished one; a file that cannot be read is a warning in
the report, never an error.
"""

from __future__ import annotations

import fnmatch
import os
import re
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from ..file_io import read_json_object
from ..model_id import ModelId
from ..result_bundle.constants import (
    ANCESTOR_RECORD_NAME,
    ANCESTORS_DIRNAME,
    DEFAULT_MANIFEST_NAME,
    DEFAULT_PROVENANCE_NAME,
    TRUNK_RECORD_NAME,
)
from ..result_bundle.errors import ResultBundleError
from ..result_bundle.gate_verdict import GATE_VERDICT_FILENAME, read_gate_verdict

# The manifest's own litter rule, so the index lists exactly what the manifest would hash.
from ..result_bundle.manifest import _is_ignored_litter, read_bundle_status
from ..run_exclusions import excluded_commit_reason, excluded_stage_records_reason
from ..run_tree import run_kind_of, run_tree_names
from ..stage_manifest import LEGACY_STAGE_IDS, StageManifest, StageManifestError, load_stage_manifest
from ..stage_manifest import stage_ref_from_dirname as _stage_ref_from_dirname
from . import stage_layout

#: The derived tree beside a model's run trees.
REPORTS_DIRNAME = "reports"
#: A run's page, in its report folder.
REPORT_FILENAME = "report.md"
#: The model's index page and its JSON twin, in ``reports/``; a run's file index is also ``index.json``.
INDEX_MD_FILENAME = "INDEX.md"
INDEX_JSON_FILENAME = "index.json"
RUN_INDEX_SCHEMA = "mesozoic.run-index/v1"
MODEL_INDEX_SCHEMA = "mesozoic.model-index/v1"

COMPLETE = "complete"
PARTIAL = "partial"
FAILED = "failed"
BUNDLE_UNREADABLE = "bundle_unreadable"
JUDGED_NO_BUNDLE = "judged_no_bundle"
POST_TRAINING_CUT = "post_training_cut"
TRAINING_CUT = "training_cut"
EMPTY = "empty"
#: Every run state, in the order the model index groups them.
RUN_STATES = (COMPLETE, PARTIAL, FAILED, JUDGED_NO_BUNDLE, POST_TRAINING_CUT, TRAINING_CUT, BUNDLE_UNREADABLE, EMPTY)
#: No run state: the model index's group for a run whose records raised while the index read them.
UNREADABLE = "unreadable"
#: The model index's groups, in order: every run state, then :data:`UNREADABLE`.
INDEX_STATES = (*RUN_STATES, UNREADABLE)
#: What each state means, from the files alone, as the report prints it (the module docstring has the rules).
STATE_DESCRIPTIONS = {
    COMPLETE: "the bundle is complete: its target and every deliverable it records are certified",
    PARTIAL: "the bundle is partial: at least one deliverable it records is certified, and its target or another "
    "deliverable it records is not",
    FAILED: "the bundle is failed: no deliverable it records is certified",
    BUNDLE_UNREADABLE: "the run's artifact_manifest.json cannot be read or records another status, so the bundle's "
    "status is unknown",
    JUDGED_NO_BUNDLE: "a stage holds gate_verdict.json, and the run holds no bundle (artifact_manifest.json)",
    POST_TRAINING_CUT: "a stage holds an evaluation CSV, and no stage holds a gate verdict nor the run a bundle",
    TRAINING_CUT: "a stage holds a checkpoint, and no stage holds an evaluation CSV or a gate verdict nor the run a "
    "bundle",
    EMPTY: "no stage holds a checkpoint and the run holds no bundle: nothing trained here",
    UNREADABLE: "the index could not read the run's records; its row shows the error",
}
#: A run's layout: the generation of its stage directories (:func:`_layout`).
LAYOUT_NN = "NN_<stage_id>"
LAYOUT_STAGE_N = "stage<N>"
LAYOUT_BARE = "bare <stage_id>"
LAYOUT_NONE = "none"
#: The layouts older than ``gate_verdict.json``: a finished run of theirs holds no verdict.
_OLDER_LAYOUTS = (LAYOUT_STAGE_N, LAYOUT_BARE)
_NN_DIRNAME = re.compile(r"[0-9]{2}_[a-z][a-z0-9_]*")
_STAGE_N_DIRNAME = re.compile(r"stage[0-9]+")
#: The prefix of every ``judged_by`` the command-line curriculum records (train_base's
#: CURRICULUM_MANAGER_JUDGED_BY, STANCE_V2_POST_TRAINING_JUDGED_BY and STANCE_HEADING_POST_TRAINING_JUDGED_BY);
#: it rolls no stance probe.
_COMMAND_LINE_JUDGED_BY = "train_base.train_curriculum/"
#: What finishes a run in each unfinished state, for a run of a notebook tree (``{run_id}`` is its directory name).
NEXT_STEPS = {
    JUDGED_NO_BUNDLE: "Next: the SB3 notebook's publish cell (section 6, `PUBLISH_RUN = True`, with "
    '`RUN_ID = "{run_id}"`) writes the training summary and the bundle from the verdicts the run holds; a chain loop '
    "that trains or judges a further node of this run writes them too",
    POST_TRAINING_CUT: 'Next: re-enter the run (`RUN_ID = "{run_id}"`) and run the chain loop: it judges a node that '
    "was trained but never gated (an intact final pair and no verdict), then saves the bundle",
    TRAINING_CUT: 'Next: re-enter the run (`RUN_ID = "{run_id}"`): the chain loop judges a node whose final pair is '
    "intact, and section 5's RESUME cell (`RESUME_STAGE`) first finishes one cut off before its final pair",
    BUNDLE_UNREADABLE: "Next: nothing writes into this run while its bundle's status is unknown: the bundle writer, "
    "and every cell that writes into a run, refuse a manifest they cannot read (`result_bundle.read_bundle_status`). "
    "Restore `artifact_manifest.json` (Drive keeps a file's earlier versions), then rebuild this page",
}
#: The ``stance_gate_report.py`` options that roll each probe again, by its label (the values are the stage's own).
_PROBE_OPTIONS = {
    "filtered action": "--filter-actions {values}",
    "constant hold": "--hold-constant",
    "release ablation": "--hold-release-ablation",
    "impulse": "--impulse-probe --impulse-speeds {values}",
    "heading": "--spawn-yaw-offsets {values}",
}
#: The folder name each re-rolled probe goes in, under ``reports/<run path>.probes/<stage dir>/``.
_PROBE_FOLDERS = {
    "filtered action": "filtered",
    "constant hold": "constant",
    "release ablation": "release",
    "impulse": "impulse",
    "heading": "heading",
}
#: The suffix of the folder of ``reports/`` a missing-probe warning names for one's own re-rolled probes
#: (``reports/<run path>.probes/``): no run's report folder, so ``scripts/run_report.py`` never touches it.
PROBES_SUFFIX = ".probes"

#: What each role means.  Every file a run holds is one of these, or ``unclassified``.
ROLES = {
    "bundle": "the result bundle's run-level files",
    "provenance": "what the run is and where it came from",
    "fingerprint": "the plant and task identities a node was trained on",
    "config": "the node's recorded configuration, and the gate resolution frozen before it trained",
    "verdict": "the node's gate verdict",
    "certificate": "the certification report the gate judged",
    "evidence": "the per-episode evidence the bundle audit recomputes",
    "ledger": "the records training writes as it goes",
    "baseline": "the zero-action baseline the stage-1 gate is calibrated against",
    "checkpoint": "a policy checkpoint",
    "normalization": "a checkpoint's VecNormalize statistics",
    "tensorboard": "TensorBoard event files",
    "figure": "a stage figure",
    "training_history": "the stage-end training-history report",
    "replay": "a replay video or its per-frame stance CSV",
    "probe": "a report-only stance probe",
    "summary_text": "a human-readable summary",
    "ancestor_record": "the record of a node reused from another run",
    "report_only": "a sidecar no reader names by a fixed path",
}
UNCLASSIFIED = "unclassified"

#: Every file name a run or stage directory holds, in every layout generation: ``(scope, pattern, role)``.
#: ``scope`` is ``run`` (the run directory), ``stage`` (a stage directory: ``NN_<stage_id>/`` from 2026-08-20,
#: ``stage<N>/`` or a bare reserved id before) or ``ancestor`` (``ancestors/<stage_id>/``).  A pattern without a
#: slash names a file in the scope's root; one with a slash a path below it, where ``*`` also crosses directories.
#: The first match wins.  Built from the writers, which the comments name; a name none of them writes is
#: ``unclassified``, a warning in the report, never an error.
ARTIFACT_ROLES: tuple[tuple[str, str, str], ...] = (
    # Run level.  provenance.json: result_bundle/provenance.initialize_result_bundle (the storage cell, before
    # training) and update_provenance (the bundle writer).  trunk_run.json: result_bundle/trunk_record.
    ("run", "provenance.json", "provenance"),
    ("run", "trunk_run.json", "provenance"),
    # reporting/bundles.save_result_bundle: plant_identity.json, collected_results.csv (csv_output.save_results_csv),
    # summary.json (whenever a deliverable is certified) and artifact_manifest.json, written last
    # (result_bundle/manifest.write_artifact_manifest).  train_base.train_curriculum also writes the run's
    # plant_identity.json (the command-line curriculum).
    ("run", "plant_identity.json", "fingerprint"),
    ("run", "collected_results.csv", "bundle"),
    ("run", "summary.json", "bundle"),
    ("run", "artifact_manifest.json", "bundle"),
    # reporting/text_summaries.write_training_summary.
    ("run", "training_summary.txt", "summary_text"),
    # scripts/zero_action_baseline.preflight copies the model's baseline into the run; curriculum/baseline_watch
    # reads it during training.
    ("run", "zero_action_baseline.json", "baseline"),
    # train_base.train_curriculum (the command-line curriculum): one row per stage.
    ("run", "curriculum_results.csv", "ledger"),
    # environments/shared/ancestors.record_ancestor: ancestor.json and verbatim copies of the ancestor stage's
    # ANCESTOR_COPIED_FILES (gate_verdict.json, stage_config.json, task_fingerprint.json, plant_identity.json).
    ("ancestor", ANCESTOR_RECORD_NAME, "ancestor_record"),
    ("ancestor", GATE_VERDICT_FILENAME, "ancestor_record"),
    ("ancestor", "stage_config.json", "ancestor_record"),
    ("ancestor", "task_fingerprint.json", "ancestor_record"),
    ("ancestor", "plant_identity.json", "ancestor_record"),
    # Stage level.  config.save_stage_config (train_base.train, and scripts/widen_checkpoint for a widened root).
    ("stage", "stage_config.json", "config"),
    ("stage", "plant_identity.json", "fingerprint"),
    ("stage", "task_fingerprint.json", "fingerprint"),
    # harnesses/freeze_recovery_gate: a frozen-null gate's resolution, frozen before the node trains.
    ("stage", "gate_resolution.json", "config"),
    # result_bundle/gate_verdict.write_gate_verdict, called by reporting/stage_artifacts._write_stage_gate_verdict
    # (and scripts/backfill_gate_verdict.py).
    ("stage", GATE_VERDICT_FILENAME, "verdict"),
    # reporting/stance_report.write_stance_gate_report and write_stance_heading_report, called by
    # stage_artifacts._write_stance_gate_report / _write_stance_heading_certificate.
    ("stage", "stance_gate_report.txt", "certificate"),
    ("stage", "stance_gate_report.json", "certificate"),
    ("stage", "stance_heading_report.txt", "certificate"),
    ("stage", "stance_heading_report.json", "certificate"),
    # Their per-episode evidence (stance_report.write_stance_panel_evidence, _write_stance_heading_panel_evidence);
    # csv_output.save_evaluation_episodes (train_base, stage_artifacts.evaluate_stage_checkpoints and
    # _write_task_success_evidence); recovery_evaluation.write_recovery_evidence (recovery_<panel>_<controller>.csv).
    ("stage", "stance_panel_selected.csv", "evidence"),
    ("stage", "stance_heading_panel_selected.csv", "evidence"),
    ("stage", "evaluation_selected.csv", "evidence"),
    ("stage", "evaluation_final.csv", "evidence"),
    ("stage", "recovery_*.csv", "evidence"),
    # The report-only probes stage_artifacts._run_stance_probes rolls last (stance_report.write_action_filter_sweep,
    # write_constant_hold_probe, write_constant_hold_ablation, write_impulse_probe, write_heading_probe).
    ("stage", "stance_gate_probe_*.txt", "probe"),
    ("stage", "stance_gate_probe_*.json", "probe"),
    ("stage", "stance_heading_probe.txt", "probe"),
    ("stage", "stance_heading_probe.json", "probe"),
    # Sidecars no reader names: stage_artifacts._write_handoff_sidecar; scripts/widen_checkpoint's report; the
    # stage_result.json the JAX stage writer (save_jax_stage_artifacts) wrote until decision D-D17 retired it.
    ("stage", "handoff_in_training.json", "report_only"),
    ("stage", "widen_report.json", "report_only"),
    ("stage", "stage_result.json", "report_only"),
    # The in-run K3 noise panel (environments/shared/k3_panel.py, decision D-D48), retired by the maintainer's
    # decision of 2026-11-26: k3_panel.csv, k3_panel.json and the k3_panel.prior_<stamp> copies a resume set aside.
    # The runs it ran in keep them; nothing reads them.
    ("stage", "k3_panel.csv", "report_only"),
    ("stage", "k3_panel.json", "report_only"),
    ("stage", "k3_panel.prior_*", "report_only"),
    # reporting/text_summaries.write_stage_summary.
    ("stage", "stage_summary.txt", "summary_text"),
    # Written during training: SB3's EvalCallback (published by train_base's PublishEvalArtifactsCallback);
    # diagnostics.DiagnosticsCallback (and the prior file it copies aside on a resume); eval_diagnostics'
    # StageGatePlateauCallback and its evaluation ledger; rollout_ledger; policy_probe; curriculum/progress_report;
    # curriculum/hop_watch; train_base's post-training panels (metrics.json, the command-line trainer).
    ("stage", "evaluations.npz", "ledger"),
    ("stage", "diagnostics.npz", "ledger"),
    ("stage", "diagnostics.prior_*.npz", "ledger"),
    ("stage", "gate_progress.npz", "ledger"),
    ("stage", "eval_episodes.csv", "ledger"),
    ("stage", "train_episodes.csv", "ledger"),
    ("stage", "policy_probe.npz", "ledger"),
    ("stage", "progress.json", "ledger"),
    ("stage", "hop_watch.json", "ledger"),
    ("stage", "metrics.json", "ledger"),
    # models/: SB3's EvalCallback and SaveVecNormalizeCallback (best_model), curriculum/checkpoints'
    # RobustBestModelCallback (robust_best_model), SB3's CheckpointCallback (<label>_<N>_steps.zip and
    # <label>_vecnormalize_<N>_steps.pkl), train_base's final save (<label>_final.zip, <label>_final_vecnorm.pkl).
    # A checkpoint resolves as .zip or .pkl (reporting/bundles._resolve_model_artifact); the statistics first.
    ("stage", "models/*_vecnorm.pkl", "normalization"),
    ("stage", "models/*_vecnormalize_*_steps.pkl", "normalization"),
    ("stage", "models/*.zip", "checkpoint"),
    ("stage", "models/*.pkl", "checkpoint"),
    # SB3's tensorboard_log (train_base), synced by tb_sync.
    ("stage", "tensorboard/*", "tensorboard"),
    # stage_layout: figures/training_history/ (stage_artifacts._write_training_history), figures/ (the
    # visualization plots generate_stage_artifacts draws) and replays/ (evaluation.record_stage_video).
    ("stage", f"{stage_layout.FIGURES_DIRNAME}/{stage_layout.TRAINING_HISTORY_DIRNAME}/*", "training_history"),
    ("stage", f"{stage_layout.FIGURES_DIRNAME}/*", "figure"),
    ("stage", f"{stage_layout.REPLAYS_DIRNAME}/*.mp4", "replay"),
    ("stage", f"{stage_layout.REPLAYS_DIRNAME}/*.csv", "replay"),
    # Before stage_layout, the same artifacts sat in the stage root (stage_layout.iter_replay_files reads them).
    ("stage", "*.mp4", "replay"),
    ("stage", "*_stance.csv", "replay"),
    ("stage", "*.png", "figure"),
)

#: The certification reports, by the gate kinds that write them (their text forms are read, never the JSON).
CERTIFICATE_REPORTS = ("stance_heading_report.txt", "stance_gate_report.txt")
#: The text report each stance probe's writer leaves in the stage directory, by the label
#: ``stage_artifacts.declared_stance_probes`` names it with (stance_report.write_action_filter_sweep,
#: write_constant_hold_probe, write_constant_hold_ablation, write_impulse_probe, write_heading_probe).
PROBE_REPORTS = {
    "filtered action": "stance_gate_probe_filtered.txt",
    "constant hold": "stance_gate_probe_constant.txt",
    "release ablation": "stance_gate_probe_release.txt",
    "impulse": "stance_gate_probe_impulse.txt",
    "heading": "stance_heading_probe.txt",
}
#: Directories of a model directory that hold no run tree: this tree, ``sweeps/`` (the March 2026 sweeps, retired
#: by D-D17; the Drive summary notebook reads them apart) and the zero-action baseline records.
_NOT_RUN_TREES = frozenset({REPORTS_DIRNAME, "sweeps", "zero_action_baselines"})
#: The trees a notebook run of either algorithm lives in (run_tree.run_tree_names).
RUN_TREES = frozenset(run_tree_names("ppo") + run_tree_names("sac"))
#: A stage directory a reader recognises by its contents (the Drive summary notebook's ``_is_run_dir`` rule).
_STAGE_MARKERS = ("stage_config.json", "evaluations.npz", "stage_summary.txt", GATE_VERDICT_FILENAME)
_EVALUATION_CSVS = ("evaluation_selected.csv", "evaluation_final.csv")
_TIMESTAMP = re.compile(r"(\d{8})_(\d{6})")
#: How many paths a warning names before it counts the rest, and how many gate failures a report lists.
_NAMED = 5
_FAILURES = 8


@lru_cache(maxsize=16)
def _manifest_of(model_id: "ModelId | None") -> "StageManifest | None":
    """The model's stage manifest, or ``None`` for no model or one without a manifest (read once per model)."""
    if model_id is None:
        return None
    try:
        return load_stage_manifest(model_id)
    except (StageManifestError, OSError, ValueError):
        return None


@lru_cache(maxsize=256)
def _stage_id(name: str, model_id: "ModelId | None" = None) -> "str | None":
    """The stage id a run-directory child named *name* holds, in any layout generation, or ``None``.

    ``stage_manifest.stage_ref_from_dirname`` recognises the name: with the
    model's manifest, any id it declares; without one, the reserved ids.
    """
    manifest = _manifest_of(model_id)
    try:
        ref = _stage_ref_from_dirname(name, model_id=model_id if manifest is not None else None)
    except StageManifestError:
        ref = _stage_ref_from_dirname(name)
    if ref is None:
        return None
    if manifest is not None:
        try:
            return manifest.resolve(ref).id
        except (StageManifestError, KeyError, ValueError):
            pass
    return LEGACY_STAGE_IDS.get(ref, str(ref)) if isinstance(ref, int) else str(ref)


def _matches(pattern: str, relative: str) -> bool:
    if "/" not in pattern:
        return "/" not in relative and fnmatch.fnmatchcase(relative, pattern)
    return fnmatch.fnmatchcase(relative, pattern)


def _role_in(scope: str, relative: str) -> str:
    for rule_scope, pattern, role in ARTIFACT_ROLES:
        if rule_scope == scope and _matches(pattern, relative):
            return role
    return UNCLASSIFIED


def classify(path: str, *, model_id: "ModelId | None" = None) -> tuple["str | None", str]:
    """The stage id and the role of the file at *path*, a POSIX path relative to its run directory.

    The stage id is ``None`` for a run-level file.  A name no writer writes
    is :data:`UNCLASSIFIED`.
    """
    parts = PurePosixPath(path).parts
    if len(parts) == 1:
        return None, _role_in("run", parts[0])
    if parts[0] == ANCESTORS_DIRNAME and len(parts) == 3:
        return parts[1], _role_in("ancestor", parts[2])
    stage_id = _stage_id(parts[0], model_id)
    if stage_id is None:
        return None, UNCLASSIFIED
    return stage_id, _role_in("stage", "/".join(parts[1:]))


def is_run_dir(path: "str | Path") -> bool:
    """Whether *path* is a run directory, of any layout generation.

    A run holds ``provenance.json`` (every notebook run, from the storage
    cell on), ``artifact_manifest.json`` or ``collected_results.csv``, or a
    child named like a stage directory (``NN_<stage_id>``, ``stage<N>`` or a
    bare reserved id) that holds a stage's records, as the Drive summary
    notebook recognises a run from before result bundles.
    """
    root = Path(path)
    if not root.is_dir():
        return False
    if any(
        (root / name).is_file() for name in (DEFAULT_PROVENANCE_NAME, DEFAULT_MANIFEST_NAME, "collected_results.csv")
    ):
        return True
    try:
        children = sorted(root.iterdir())
    except OSError:
        return False
    # The name first: it costs no filesystem call (Drive FUSE).
    return any(
        _stage_ref_from_dirname(child.name) is not None
        and child.is_dir()
        and (any((child / name).exists() for name in _STAGE_MARKERS) or (child / "models").is_dir())
        for child in children
    )


def model_dir_of(run_dir: "str | Path") -> Path:
    """The model directory a run lies in, by the layout: ``<model>/<tree>/<run_id>`` or ``<model>/<run>``.

    The run's parent when it is no run tree of the notebook (``<algo>``,
    ``<algo>_screen`` or ``<algo>_quick_test`` for PPO or SAC): a flat run
    from before 2026-03-31, or a command-line curriculum run.  The path is
    resolved first, so a relative or ``.`` path reads the directories it
    lands in; nothing is read.
    """
    run = Path(run_dir).resolve()
    return run.parent.parent if run.parent.name in RUN_TREES else run.parent


def report_dir_for(run_dir: "str | Path", model_dir: "str | Path") -> Path:
    """The run's report folder: ``<model_dir>/reports/`` and the run's path under *model_dir*.

    Raises ``ValueError`` for a run that is not inside *model_dir*, or is
    *model_dir* itself.
    """
    run, model = Path(run_dir).resolve(), Path(model_dir).resolve()
    relative = run.relative_to(model)
    if not relative.parts or relative.parts[0] == REPORTS_DIRNAME:
        raise ValueError(f"{run} is no run under {model}")
    return model / REPORTS_DIRNAME / relative


def discover_runs(model_dir: "str | Path") -> list[Path]:
    """Every run of the model at *model_dir*: each run of each tree, and each flat run from before 2026-03-31.

    The ``reports/`` tree, ``sweeps/`` and ``zero_action_baselines/`` are no
    run trees, and a hidden directory is skipped.  Sorted by path.
    """
    root = Path(model_dir)
    runs: list[Path] = []
    if not root.is_dir():
        return runs
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name in _NOT_RUN_TREES or entry.name.startswith("."):
            continue
        if is_run_dir(entry):
            runs.append(entry)
            continue
        try:
            children = sorted(entry.iterdir())
        except OSError:
            continue
        runs.extend(child for child in children if not child.name.startswith(".") and is_run_dir(child))
    return runs


def _model_id_of(
    run_dir: Path, model_dir: "Path | None", provenance: "Mapping[str, Any] | None", warnings: list[str]
) -> "ModelId | None":
    """The model *provenance* (the run's ``provenance.json``) names, else the one its model directory's path names."""
    if provenance is not None:
        try:
            return ModelId.from_record(provenance)
        except ValueError as exc:
            warnings.append(f"provenance.json names no model: {exc}")
    directory = model_dir if model_dir is not None else model_dir_of(run_dir)
    try:
        return ModelId(directory.parent.name, directory.name)
    except ValueError:
        return None


def _stage_dirs(run_dir: Path, model_id: "ModelId | None") -> list[tuple[str, Path]]:
    """``(stage id, directory)`` for every child of *run_dir* named like a stage directory, in name order."""
    try:
        children = sorted(run_dir.iterdir())
    except OSError:
        return []
    found = []
    for child in children:
        if child.name == ANCESTORS_DIRNAME:
            continue
        stage_id = _stage_id(child.name, model_id)
        if stage_id is not None and child.is_dir():
            found.append((stage_id, child))
    return found


def _holds_checkpoint(stage_dir: Path) -> bool:
    models = stage_dir / "models"
    try:
        names = [path.name for path in models.iterdir() if path.is_file()] if models.is_dir() else []
    except OSError:
        return False
    return any(_role_in("stage", f"models/{name}") == "checkpoint" for name in names)


def _state(run_dir: Path, stage_dirs: Sequence[tuple[str, Path]], warnings: list[str]) -> "tuple[str, str | None]":
    """The run's state and the status its manifest records (``None`` without one)."""
    try:
        status = read_bundle_status(run_dir)
    except ResultBundleError as exc:
        warnings.append(_without_run_path(exc, run_dir.resolve()))
        return BUNDLE_UNREADABLE, None
    if status is not None:
        return status, status
    directories = [directory for _, directory in stage_dirs]
    if any((directory / GATE_VERDICT_FILENAME).is_file() for directory in directories):
        return JUDGED_NO_BUNDLE, None
    if any((directory / name).is_file() for directory in directories for name in _EVALUATION_CSVS):
        return POST_TRAINING_CUT, None
    if any(_holds_checkpoint(directory) for directory in directories):
        return TRAINING_CUT, None
    return EMPTY, None


def run_state(run_dir: "str | Path") -> str:
    """The state of the run at *run_dir*: one of :data:`RUN_STATES` (the module docstring has the rules)."""
    run = Path(run_dir).resolve()
    provenance = read_json_object(run / DEFAULT_PROVENANCE_NAME)
    return _state(run, _stage_dirs(run, _model_id_of(run, None, provenance, [])), [])[0]


def _layout(stage_dirs: Sequence[tuple[str, Path]]) -> str:
    """The generation of the run's stage directories: one of the ``LAYOUT_*`` names, :data:`LAYOUT_NONE` without one.

    ``NN_<stage_id>`` from 2026-08-21 (``stage_manifest.stage_dirname``),
    ``stage<N>`` before, and a bare reserved id (``stance/``) for the
    generation ``stage_manifest.stage_ref_from_dirname`` also reads.  Where
    the run sits (a tree, or the model directory) is its ``tree``.
    """
    names = [directory.name for _, directory in stage_dirs]
    if any(_NN_DIRNAME.fullmatch(name) for name in names):
        return LAYOUT_NN
    if any(_STAGE_N_DIRNAME.fullmatch(name) for name in names):
        return LAYOUT_STAGE_N
    return LAYOUT_BARE if names else LAYOUT_NONE


def _started(provenance: "Mapping[str, Any] | None", run_dir: Path) -> "str | None":
    """When the run started: its provenance's ``captured_at``, else the timestamp its directory's name carries."""
    captured = provenance.get("captured_at") if provenance is not None else None
    if isinstance(captured, str) and captured:
        return captured
    match = _TIMESTAMP.search(run_dir.name)
    if match is None:
        return None
    day, time = match.groups()
    return f"{day[:4]}-{day[4:6]}-{day[6:]}T{time[:2]}:{time[2:4]}:{time[4:]}"


def _key_id(key: Any, manifest: "StageManifest | None") -> str:
    """A serialized stage key (``"1"``, ``"recovery"``) as its stage id."""
    text = str(key)
    if manifest is not None:
        try:
            return manifest.by_legacy_number(int(text)).id if text.isdigit() else manifest.resolve(text).id
        except (StageManifestError, KeyError, ValueError):
            pass
    return LEGACY_STAGE_IDS.get(int(text), text) if text.isdigit() else text


def _read_text(path: Path, warnings: list[str], what: str) -> "str | None":
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, ValueError) as exc:
        warnings.append(f"{what} cannot be read: {exc}")
        return None


def _certificate_line(text: str) -> str:
    """The certificate's verdict and headline statistic, from its text form: ``GATE: PASS; seeds clean ...``."""
    lines = [line.strip() for line in text.splitlines()]
    gate = [line for line in lines if line.startswith("GATE: ")]
    verdict = gate[-1] if gate else "no GATE line"
    for prefix in ("seeds clean at every heading", "clean episodes", "mean_unsupported_duty"):
        headline = next((line for line in lines if line.startswith(prefix)), None)
        if headline is not None:
            return f"{verdict}; {' '.join(headline.split())}"
    return verdict


def _probe_line(text: str) -> str:
    """A probe's banner, its first sentence: what it modified (its tables are in the file)."""
    words: list[str] = []
    for line in (line.strip() for line in text.splitlines()):
        if not line:
            if words:
                break
            continue
        words.append(line)
        if line.endswith((".", ":")):
            break
    first = " ".join(words)
    return first if len(first) <= 200 else first[:197] + "..."


def _cached(cache: "dict[Path, Mapping[str, Any] | None] | None", path: Path) -> "Mapping[str, Any] | None":
    """The JSON object at *path* (``file_io.read_json_object``), read once per :func:`_run_summary` through *cache*."""
    if cache is None:
        return read_json_object(path)
    if path not in cache:
        cache[path] = read_json_object(path)
    return cache[path]


def _recorded_curriculum(
    stage_dir: Path, cache: "dict[Path, Mapping[str, Any] | None] | None" = None
) -> "Mapping[str, Any] | None":
    """The ``[curriculum]`` block the stage's ``stage_config.json`` records (read once per stage), or ``None``."""
    record = _cached(cache, stage_dir / "stage_config.json") or {}
    block = record.get("curriculum", record.get("curriculum_kwargs"))
    return block if isinstance(block, Mapping) else None


def _declared_probes(curriculum: "Mapping[str, Any] | None") -> tuple[str, ...]:
    """The probes a recorded ``[curriculum]`` block declares (``stage_artifacts.declared_stance_probes``)."""
    from .stage_artifacts import declared_stance_probes

    return declared_stance_probes(curriculum) if curriculum is not None else ()


def _relative_to_run(path_text: Any, stage_dir: Path, run_dir: Path) -> "tuple[str | None, Path | None]":
    """A path a verdict records (stage-directory-relative or absolute) as shown run-relative, and as found here."""
    if not isinstance(path_text, str) or not path_text:
        return None, None
    path = Path(path_text)
    here = path if path.is_absolute() else stage_dir / path
    try:
        shown = here.resolve().relative_to(run_dir.resolve()).as_posix()
    except ValueError:
        shown = str(path)
    return shown, here


def _blank_node(stage_id: str, directory: "str | None", verdict: str) -> dict[str, Any]:
    return {
        "id": stage_id,
        "dir": directory,
        "verdict": verdict,
        "passed": None,
        "gate_kind": None,
        "judged_by": None,
        "handoff": None,
        "normalization": None,
        "failures": [],
        "reused_from": None,
        "own_dir": None,
        "own_verdict": None,
        "certificate": None,
        "probes": [],
        "missing_probes": [],
        "probe_note": None,
    }


def _without_run_path(error: Exception, run_dir: Path) -> str:
    """*error*'s message with the run directory's absolute path dropped, so each path it names reads run-relative."""
    return str(error).replace(f"{run_dir}{os.sep}", "")


def _own_node(
    stage_id: str,
    stage_dir: Path,
    run_dir: Path,
    warnings: list[str],
    *,
    detail: bool = True,
    rerun: "Mapping[str, Any] | None" = None,
    cache: "dict[Path, Mapping[str, Any] | None] | None" = None,
) -> dict[str, Any]:
    """A node trained in this run, from its stage directory; without *detail*, from ``gate_verdict.json`` alone.

    *rerun* is what a missing-probe warning's command names (:func:`_probe_commands`); *cache* keeps the records
    read (:func:`_cached`), the verdict as ``read_gate_verdict`` gave it.
    """
    curriculum = _recorded_curriculum(stage_dir, cache) if detail else None
    recorded_kind = curriculum.get("gate_kind") if curriculum is not None else None
    node = _blank_node(stage_id, stage_dir.name, "not judged")
    node["gate_kind"] = str(recorded_kind) if recorded_kind is not None else None
    try:
        verdict = read_gate_verdict(stage_dir)
    except ResultBundleError as exc:
        warnings.append(
            f"{_without_run_path(exc, run_dir)}: the node reads as not judged here, and nothing can reuse it; re-judge "
            f"the stage, e.g. `python -m environments.shared.scripts.backfill_gate_verdict <run>/{stage_dir.name} "
            "--force` (it refuses a run whose bundle is complete)"
        )
        verdict = None
        node["verdict"] = "unreadable"
    if cache is not None:
        cache[stage_dir / GATE_VERDICT_FILENAME] = verdict
    if verdict is not None:
        node.update(
            verdict="PASS" if verdict["passed"] else "FAIL",
            passed=verdict["passed"],
            gate_kind=verdict.get("gate_kind") or node["gate_kind"],
            judged_by=verdict.get("judged_by"),
            failures=[str(failure) for failure in verdict.get("failures") or ()],
        )
        for field in ("checkpoint", "normalization"):
            shown, here = _relative_to_run(verdict.get(field), stage_dir, run_dir)
            node["handoff" if field == "checkpoint" else "normalization"] = shown
            if detail and here is not None and not here.is_file():
                warnings.append(
                    f"{stage_dir.name}/{GATE_VERDICT_FILENAME} names its {field} {shown}, which no longer exists "
                    "(the verdict is bound to that file: re-judge the stage before reusing it)"
                )
    if not detail:
        return node
    for name in CERTIFICATE_REPORTS:
        if (stage_dir / name).is_file():
            text = _read_text(stage_dir / name, warnings, f"{stage_dir.name}/{name}")
            node["certificate"] = {
                "path": f"{stage_dir.name}/{name}",
                "line": _certificate_line(text) if text is not None else "unreadable",
            }
            break
    for label, name in PROBE_REPORTS.items():
        if (stage_dir / name).is_file():
            text = _read_text(stage_dir / name, warnings, f"{stage_dir.name}/{name}")
            node["probes"].append(
                {"label": label, "path": f"{stage_dir.name}/{name}", "line": _probe_line(text) if text else ""}
            )
    if node["certificate"] is not None:
        _expect_probes(node, stage_dir, curriculum, warnings, rerun)
    return node


def _probe_values(label: str, curriculum: Mapping[str, Any]) -> str:
    """The stage's own values for a probe's ``stance_gate_report.py`` option, comma-separated."""
    from .stage_artifacts import _probe_cutoffs, _probe_offsets, _probe_speeds

    if label == "filtered action":
        values = _probe_cutoffs(curriculum.get("stance_probe_filter_hz"), warn=False)
    elif label == "impulse":
        values = _probe_speeds(curriculum.get("stance_probe_impulse_speeds"), warn=False)
    elif label == "heading":
        values = _probe_offsets(curriculum.get("stance_probe_spawn_yaw_deg"), warn=False)
    else:
        return ""
    return ",".join(f"{value:g}" for value in values)


def _probe_commands(
    node: Mapping[str, Any], labels: Sequence[str], curriculum: Mapping[str, Any], rerun: Mapping[str, Any]
) -> list[str]:
    """One ``stance_gate_report.py`` command per probe in *labels*, on the stage's handoff pair, written outside the run.

    Paths start at ``<model>``, the model directory; ``--out-dir`` is a folder of
    ``reports/<run path>.probes/`` (:data:`PROBES_SUFFIX`), which ``run_report`` never touches.
    """
    model_id: "ModelId | None" = rerun.get("model_id")
    run_path = rerun["run_path"]
    stage = rerun.get("stage")
    model = (
        f"--species {model_id.species} --embodiment {model_id.embodiment}" if model_id else "--species S --embodiment E"
    )
    pair = " ".join(
        f"--{option} <model>/{run_path}/{path}" if path else f"--{option} <{option}>"
        for option, path in (("model", node["handoff"]), ("vecnorm", node["normalization"]))
    )
    commands = []
    for label in labels:
        values = _probe_values(label, curriculum)
        # The script filters at one cutoff per call: the stage's sweep is one call per cutoff.
        runs = [(f"{_PROBE_FOLDERS[label]}_{value}hz", value) for value in values.split(",")]
        if label != "filtered action" or not values:
            runs = [(_PROBE_FOLDERS[label], values)]
        for folder, value in runs:
            options = _PROBE_OPTIONS[label].format(values=value)
            out_dir = f"<model>/{REPORTS_DIRNAME}/{run_path}{PROBES_SUFFIX}/{node['dir']}/{folder}"
            commands.append(
                f"`python -m environments.shared.scripts.stance_gate_report {model}"
                + (f" --stage {stage}" if stage is not None else "")
                + f" {pair} {options} --out-dir {out_dir}`"
            )
    return commands


def _listed(items: Sequence[str], last: str = "and") -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} {last} {items[-1]}"


def _expect_probes(
    node: dict[str, Any],
    stage_dir: Path,
    curriculum: "Mapping[str, Any] | None",
    warnings: list[str],
    rerun: "Mapping[str, Any] | None" = None,
) -> None:
    """The declared probes a stance stage holding a certificate lacks, by what judged it.

    Only ``generate_stage_artifacts`` rolls the probes, last
    (``stage_artifacts._run_stance_probes``), so they are expected only on a
    stage its verdict says it judged: one warning names every probe it lacks,
    with a command per probe that rolls it again outside the run.  The
    command-line curriculum's judge writes the certificate and rolls none: a
    note, not a warning.  A stage with no verdict, or one another tool
    judged, expects nothing.
    """
    from .stage_artifacts import GATE_VERDICT_JUDGED_BY

    declared = _declared_probes(curriculum)
    judged_by = node["judged_by"]
    if judged_by == GATE_VERDICT_JUDGED_BY:
        present = {probe["label"] for probe in node["probes"]}
        node["missing_probes"] = missing = [label for label in declared if label not in present]
        if not missing:
            return
        warning = (
            f"{stage_dir.name}: its stage_config.json declares the {_listed(missing)} "
            f"probe{'s' if len(missing) > 1 else ''}, and it holds no "
            f"{_listed([PROBE_REPORTS[label] for label in missing], 'or')}. The probes run last, each on its own, so "
            "a runtime lost during them, or a probe that failed, leaves its report out, and nothing writes into the "
            "stage directory again, so this warning stays"
        )
        if rerun is not None and curriculum is not None:
            model_id = rerun.get("model_id")
            where = (
                f"LOG_BASE/{model_id.species}/{model_id.embodiment}" if model_id else "LOG_BASE/<species>/<embodiment>"
            )
            warning += (
                f". To roll one again as report-only output of your own, outside the run (`<model>` is {where}; the "
                "script scores the checkout's stage file with its own episode counts): "
                + "; ".join(_probe_commands(node, missing, curriculum, rerun))
                + " (a folder per call: each call also writes its own certification report beside the probe)"
            )
        warnings.append(warning)
    elif declared and isinstance(judged_by, str) and judged_by.startswith(_COMMAND_LINE_JUDGED_BY):
        node["probe_note"] = (
            f"its stage_config.json declares the {', '.join(declared)} probe(s), and the command-line curriculum "
            f"that judged it ({judged_by}) rolls no probe"
        )


def _trunk_report_path(source_run: str) -> str:
    """The trunk run's path under its model directory, by the layout: ``<tree>/<run_id>``, or ``<run>`` for a run
    directly in the model directory (a command-line ``curriculum_<timestamp>`` run, or a flat one)."""
    source = Path(source_run)
    return "/".join(source.parts[-2:]) if source.parent.name in RUN_TREES else source.name


def _trunk_file(value: str, source_run: Any, run_dir: Path) -> "tuple[Path | None, Path | None]":
    """Where a trunk file an ancestor record names is on this machine, and the sibling path tried.

    As recorded, else at the same place below the trunk run beside this run
    (``ancestors._resolve_source_run_dir``'s rule, for a run read where it
    was not written: Drive for desktop, a renamed or shared logs folder);
    ``None`` first when neither exists.
    """
    recorded = Path(value)
    if recorded.is_file():
        return recorded, None
    if not isinstance(source_run, str) or not source_run:
        return None, None
    source = Path(source_run)
    try:
        below = recorded.relative_to(source)
    except ValueError:
        return None, None
    sibling = run_dir.parent / source.name / below
    return (sibling if sibling.is_file() else None), sibling


def _reused_node(
    stage_id: str, record_dir: Path, run_dir: Path, warnings: list[str], *, detail: bool = True
) -> dict[str, Any]:
    """A node reused from another run, from its ``ancestors/<stage_id>/`` record; without *detail*, from its name."""
    node = _blank_node(stage_id, None, "reused")
    if not detail:
        return node
    record = read_json_object(record_dir / ANCESTOR_RECORD_NAME)
    where = f"{ANCESTORS_DIRNAME}/{stage_id}"
    if record is None:
        warnings.append(f"{where}/{ANCESTOR_RECORD_NAME} is missing or cannot be read")
        return node
    copied = read_json_object(record_dir / GATE_VERDICT_FILENAME) or {}
    passed = copied.get("passed")
    node["passed"] = passed if isinstance(passed, bool) else None
    node["gate_kind"] = copied.get("gate_kind")
    source_run = record.get("source_run_dir")
    source_stage = record.get("source_stage_dir")
    recorded_handoff = record.get("handoff")
    handoff: Mapping[str, Any] = recorded_handoff if isinstance(recorded_handoff, Mapping) else {}
    node["reused_from"] = {
        "run_id": record.get("parent_run_id"),
        "run": _trunk_report_path(source_run) if isinstance(source_run, str) and source_run else None,
        "stage_dir": Path(source_stage).name if isinstance(source_stage, str) and source_stage else None,
        "source_run_dir": source_run if isinstance(source_run, str) else None,
    }
    node["handoff"] = handoff.get("model_path")
    for field in ("model_path", "normalization_path"):
        value = handoff.get(field)
        if not isinstance(value, str) or not value:
            continue
        found, sibling = _trunk_file(value, source_run, run_dir)
        if found is None:
            beside = f" nor at {sibling}, in the trunk run beside this run" if sibling is not None else ""
            warnings.append(
                f"{where}/{ANCESTOR_RECORD_NAME} names the trunk's {field} {value}, which is neither there{beside}: "
                "the trunk run is not on this machine, so the reuse cannot be checked here and a run trunked from "
                "this one cannot reuse the node here (read the run where its trunk run is)"
            )
    return node


def _nodes(
    run_dir: Path,
    model_id: "ModelId | None",
    stage_dirs: Sequence[tuple[str, Path]],
    warnings: list[str],
    *,
    detail: bool = True,
    rerun: "Mapping[str, Any] | None" = None,
    cache: "dict[Path, Mapping[str, Any] | None] | None" = None,
) -> list[dict[str, Any]]:
    """The run's nodes in the manifest's order: its own stages and the ones reused from a trunk.

    An ``ancestors/<stage_id>/`` record stands for its node, as the bundle
    writer reads it (``result_bundle.load_ancestor_records``): a stage
    directory of the run for the same node is named as kept but not
    published (``own_dir``, with its verdict), and nothing else is read
    from it.  Without *detail* a reused node is known by its record's name
    alone.
    """
    own = dict(stage_dirs)
    records: dict[str, Path] = {}
    ancestors = run_dir / ANCESTORS_DIRNAME
    if ancestors.is_dir():
        try:
            records = {path.name: path for path in sorted(ancestors.iterdir()) if path.is_dir()}
        except OSError as exc:
            warnings.append(f"{ANCESTORS_DIRNAME}/ cannot be listed: {exc}")
    manifest = _manifest_of(model_id)
    order = [entry.id for entry in manifest.stages] if manifest is not None else []
    ids = [stage_id for stage_id in order if stage_id in own or stage_id in records]
    ids += [stage_id for stage_id in [*own, *records] if stage_id not in ids]
    nodes = []
    for stage_id in dict.fromkeys(ids):
        if stage_id in records:
            node = _reused_node(stage_id, records[stage_id], run_dir, warnings, detail=detail)
            if stage_id in own:
                node["own_dir"] = own[stage_id].name
                kept = _cached(cache, own[stage_id] / GATE_VERDICT_FILENAME) if detail else None
                passed = kept.get("passed") if kept is not None else None
                node["own_verdict"] = ("PASS" if passed else "FAIL") if isinstance(passed, bool) else None
            nodes.append(node)
        else:
            stage = None
            if rerun is not None and manifest is not None:
                try:
                    reference = manifest.resolve(stage_id).reference
                    stage = reference if isinstance(reference, int) else None
                except (StageManifestError, KeyError, ValueError):
                    stage = None
            nodes.append(
                _own_node(
                    stage_id,
                    own[stage_id],
                    run_dir,
                    warnings,
                    detail=detail,
                    rerun={**rerun, "stage": stage} if rerun is not None else None,
                    cache=cache,
                )
            )
    return nodes


def _exclusion(
    run_dir: Path, provenance: "Mapping[str, Any] | None", cache: "dict[Path, Mapping[str, Any] | None]"
) -> "str | None":
    """Why the run never counts (``run_exclusions.excluded_run_reason``'s answer), from the records read once.

    Rule (a) on its ``provenance.json``, then rules (b) to (d) on each
    immediate subdirectory's ``stage_config.json`` and ``gate_verdict.json``
    (``run_exclusions.excluded_stage_records_reason``), each read through
    *cache*.
    """
    if provenance is not None:
        reason = excluded_commit_reason(provenance.get("repository_commit"))
        if reason is not None:
            return f"{DEFAULT_PROVENANCE_NAME}: {reason}"
    try:
        children = sorted(child for child in run_dir.iterdir() if child.is_dir())
    except OSError:
        return None
    for child in children:
        config = _cached(cache, child / "stage_config.json")
        reason = excluded_stage_records_reason(child.name, config, _cached(cache, child / GATE_VERDICT_FILENAME))
        if reason is not None:
            return reason
    return None


def _deliverables(provenance: "Mapping[str, Any] | None", manifest: "StageManifest | None") -> list[dict[str, Any]]:
    """The deliverables the bundle writer recorded in ``provenance.json`` (empty before the first bundle)."""
    recorded = provenance.get("deliverables") if provenance is not None else None
    if not isinstance(recorded, Mapping):
        return []
    rows = []
    for key, record in recorded.items():
        if not isinstance(record, Mapping):
            continue
        recorded_replication = record.get("replication")
        replication: Mapping[str, Any] = recorded_replication if isinstance(recorded_replication, Mapping) else {}
        rows.append(
            {
                "key": str(key),
                "id": _key_id(key, manifest),
                "certified": record.get("certified") is True,
                "provisional": record.get("provisional") is True,
                "replicates": replication.get("count"),
                "certification_seeds": record.get("certification_seeds"),
                "handoff": record.get("model_path"),
            }
        )
    return rows


def _run_summary(run_dir: Path, model_dir: Path, *, detail: bool = True) -> dict[str, Any]:
    """Everything a report reads about a run, from its small records only; without *detail*, what an index row reads.

    Without *detail* (:func:`build_model_index`), each stage is read from
    its ``gate_verdict.json`` alone and a reused node from its record's
    name: no ``stage_config.json``, certificate, probe, ``ancestor.json`` or
    ``trunk_run.json``, and no handoff path is checked, so ``warnings`` is
    not the report's.  Deliverables count as certified only behind a sealed
    bundle (``complete``, ``partial`` or ``failed``): a save that stopped
    before its manifest leaves them recorded in ``provenance.json`` alone.
    """
    warnings: list[str] = []
    provenance = read_json_object(run_dir / DEFAULT_PROVENANCE_NAME)
    if provenance is None and (run_dir / DEFAULT_PROVENANCE_NAME).exists():
        warnings.append(f"{DEFAULT_PROVENANCE_NAME} cannot be read, or holds no JSON object")
    model_id = _model_id_of(run_dir, model_dir, provenance, warnings)
    manifest = _manifest_of(model_id)
    stage_dirs = _stage_dirs(run_dir, model_id)
    state, status = _state(run_dir, stage_dirs, warnings)
    relative = run_dir.resolve().relative_to(model_dir.resolve())
    rerun = {"model_id": model_id, "run_path": relative.as_posix()} if detail else None
    cache: dict[Path, Mapping[str, Any] | None] = {}
    nodes = _nodes(run_dir, model_id, stage_dirs, warnings, detail=detail, rerun=rerun, cache=cache)
    deliverables = _deliverables(provenance, manifest)
    sealed = status is not None
    primary_key = provenance.get("primary_deliverable") if provenance is not None else None
    target_key = provenance.get("target_deliverable") if provenance is not None else None
    primary = next((row for row in deliverables if row["key"] == str(primary_key)), None) if sealed else None
    certified = [row for row in deliverables if row["certified"]] if sealed else []
    passed = [node for node in nodes if node["dir"] is not None and node["passed"] is True]
    handoff = primary["handoff"] if primary is not None else None
    if handoff is None and passed:
        handoff = passed[-1]["handoff"]
    kind = run_kind_of(run_dir) if relative.parent.parts else None
    excluded = _exclusion(run_dir, provenance, cache) if kind == "certify" else None
    trunk = None
    if detail and (run_dir / TRUNK_RECORD_NAME).exists():
        from ..result_bundle.trunk_record import read_trunk_record

        try:
            trunk = read_trunk_record(run_dir)
        except ResultBundleError as exc:
            warnings.append(f"{TRUNK_RECORD_NAME}: {exc}")
    return {
        "run_id": run_dir.name,
        "provenance_run_id": provenance.get("run_id") if provenance is not None else None,
        "path": relative.as_posix(),
        "tree": relative.parent.as_posix() if relative.parent.parts else "",
        "kind": kind,
        "excluded": excluded,
        "provenance": provenance is not None,
        "layout": _layout(stage_dirs),
        "model": model_id.record() if model_id is not None else None,
        "state": state,
        "manifest_status": status,
        "started": _started(provenance, run_dir),
        "target": _key_id(target_key, manifest) if target_key is not None else None,
        "trunk": trunk,
        "nodes": nodes,
        "deliverables": deliverables,
        "certified": [row["id"] for row in certified],
        "passed": [node["id"] for node in passed],
        "provisional": any(row["provisional"] for row in certified) if certified else None,
        "replicates": primary["replicates"] if primary is not None else None,
        "handoff": handoff,
        "warnings": warnings,
    }


def _listing(run_dir: Path, warnings: list[str]) -> dict[str, int]:
    """Every file under *run_dir* (run-relative POSIX path -> size), skipping the manifest's litter."""
    files: dict[str, int] = {}

    def failed(error: OSError) -> None:
        warnings.append(f"cannot list {error.filename}: {error.strerror}")

    for directory, _subdirectories, names in os.walk(run_dir, onerror=failed):
        for name in names:
            path = Path(directory) / name
            if _is_ignored_litter(path, run_dir):
                continue
            try:
                files[path.relative_to(run_dir).as_posix()] = path.stat().st_size
            except OSError as exc:
                warnings.append(f"cannot read the size of {path.relative_to(run_dir).as_posix()}: {exc}")
    return dict(sorted(files.items()))


def _manifest_entries(run_dir: Path, warnings: list[str]) -> "dict[str, Mapping[str, Any]] | None":
    """The manifest's entries by path (through ``file_io.read_json_object``), or ``None`` with a warning."""
    manifest = read_json_object(run_dir / DEFAULT_MANIFEST_NAME)
    files = manifest.get("files") if manifest is not None else None
    if not isinstance(files, list):
        warnings.append(f"{DEFAULT_MANIFEST_NAME} records no file list; its files are listed without digests")
        return None
    return {str(entry["path"]): entry for entry in files if isinstance(entry, Mapping) and "path" in entry}


def _named(paths: Sequence[str]) -> str:
    shown = ", ".join(paths[:_NAMED])
    return shown + (f" and {len(paths) - _NAMED} more" if len(paths) > _NAMED else "")


def build_run_index(run_dir: "str | Path", *, model_dir: "str | Path | None" = None) -> dict[str, Any]:
    """Every file of the run at *run_dir*, with its stage, role, size and digest, and the run's summary.

    *model_dir* defaults to :func:`model_dir_of`.  A file the run's
    ``artifact_manifest.json`` lists takes its ``size`` and ``sha256`` from
    it (``hash_source`` ``"manifest"``); any other file (a run without a
    manifest, or a file written after the last bundle) its size from the
    listing, ``sha256`` ``None`` and ``hash_source`` ``"none"``.  Nothing is
    hashed.  ``warnings`` names unclassified files, files the manifest lists
    that no longer exist or changed size, declared probes a stage lacks, the
    handoff checkpoint and normalization a verdict or an ancestor record
    names that no longer exist, and records that cannot be read.  Schema
    :data:`RUN_INDEX_SCHEMA`.
    """
    run = Path(run_dir).resolve()
    model = Path(model_dir).resolve() if model_dir is not None else model_dir_of(run)
    summary = _run_summary(run, model)
    warnings = summary["warnings"]
    model_id = ModelId.from_record(summary["model"]) if summary["model"] is not None else None
    listed = _listing(run, warnings)
    manifest = _manifest_entries(run, warnings) if summary["manifest_status"] is not None else None
    entries = []
    for path, size in listed.items():
        stage, role = classify(path, model_id=model_id)
        recorded = manifest.get(path) if manifest is not None else None
        entry: dict[str, Any] = {"path": path, "stage": stage, "role": role}
        if recorded is not None:
            entry.update(size=recorded.get("size_bytes"), sha256=recorded.get("sha256"), hash_source="manifest")
        else:
            entry.update(size=size, sha256=None, hash_source="none")
        entries.append(entry)
    unclassified = [entry["path"] for entry in entries if entry["role"] == UNCLASSIFIED]
    if unclassified:
        warnings.append(
            f"{len(unclassified)} file(s) no writer names: {_named(unclassified)} (when a writer does write one, add "
            "its name and the writer to run_index.ARTIFACT_ROLES)"
        )
    if manifest is not None:
        gone = sorted(path for path in manifest if path not in listed and path != DEFAULT_MANIFEST_NAME)
        check = "check the bundle with result_bundle.validate_result_bundle(<run>, require_complete=False)"
        if gone:
            warnings.append(
                f"{DEFAULT_MANIFEST_NAME} lists {len(gone)} file(s) that no longer exist: {_named(gone)} ({check})"
            )
        resized = [
            path for path, size in listed.items() if path in manifest and manifest[path].get("size_bytes") != size
        ]
        if resized:
            warnings.append(
                f"{len(resized)} file(s) changed size since {DEFAULT_MANIFEST_NAME} was written: {_named(resized)} "
                f"({check})"
            )
    return {
        "schema": RUN_INDEX_SCHEMA,
        "run": summary["path"],
        "run_id": summary["provenance_run_id"] or summary["run_id"],
        "model": summary["model"],
        "state": summary["state"],
        "manifest_status": summary["manifest_status"],
        "files": entries,
        "warnings": list(warnings),
        "summary": summary,
    }


def _cell(value: Any) -> str:
    text = "—" if value in (None, "", []) else ", ".join(value) if isinstance(value, list) else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _link(text: str, target: "str | Path") -> str:
    destination = Path(target).as_posix() if isinstance(target, Path) else target
    return f"[{text}](<{destination}>)"


def _sentence(text: str) -> str:
    """*text* with its first character upper-cased and the rest as written (``str.capitalize`` lowers the rest)."""
    return text[:1].upper() + text[1:]


def _relative(target: Path, start: "Path | None") -> str:
    if start is None:
        return target.as_posix()
    return Path(os.path.relpath(target, start)).as_posix()


def render_run_report(
    run_dir: "str | Path",
    *,
    model_dir: "str | Path | None" = None,
    report_dir: "str | Path | None" = None,
    index: "Mapping[str, Any] | None" = None,
) -> str:
    """The run's ``report.md``: its state, a row per node, the certificate and probe lines, its files, its warnings.

    *index* is :func:`build_run_index`'s result (built when omitted).  Links
    are relative to *report_dir*, the folder the page is written to (run-
    relative paths when it is ``None``); a node reused from another run links
    to the trunk run's report in *model_dir*'s ``reports/`` tree.
    """
    run = Path(run_dir).resolve()
    model = Path(model_dir).resolve() if model_dir is not None else model_dir_of(run)
    built = dict(index) if index is not None else build_run_index(run, model_dir=model)
    summary = built["summary"]
    start = Path(report_dir) if report_dir is not None else None

    def to_run(path: str) -> str:
        return _relative(run / path, start) if start is not None else path

    title = summary["path"]
    try:
        from ..species_names import model_display_name

        display = model_display_name(ModelId.from_record(summary["model"])) if summary["model"] else "unknown model"
    except (ValueError, KeyError):
        display = f"{summary['model']['species']} {summary['model']['embodiment']}" if summary["model"] else "unknown"
    if summary["tree"]:
        kind = {"certify": "certification", "screen": "screen", "quick_test": "quick test"}[summary["kind"]]
        where = f"tree `{summary['tree']}` ({kind})"
    else:
        where = "no tree: in the model directory (a flat run from before 2026-03-31, or a command-line curriculum run)"
    lines = [
        f"# Run {title}",
        "",
        f"{display}; run id `{built['run_id']}`; {where}; layout `{summary['layout']}`"
        + (f"; started {summary['started']}" if summary["started"] else ""),
        "",
        f"**State: {summary['state']}.** {_sentence(STATE_DESCRIPTIONS[summary['state']])}.",
    ]
    if summary["state"] in (TRAINING_CUT, POST_TRAINING_CUT) and summary["layout"] in _OLDER_LAYOUTS:
        lines.append(
            f"Its stage directories (`{summary['layout']}`) are of a layout older than `{GATE_VERDICT_FILENAME}`: "
            "the state says which files the run holds, not that its training was cut off."
        )
    elif summary["state"] in NEXT_STEPS and (summary["kind"] is not None or summary["state"] == BUNDLE_UNREADABLE):
        lines.append(NEXT_STEPS[summary["state"]].format(run_id=summary["run_id"]) + ".")
    if summary["excluded"]:
        lines.append(
            f"**Never counts:** {summary['excluded']} (`environments/shared/run_exclusions.py`): not as certified, "
            "as a seed replicate or as a reusable ancestor, whatever its bundle records."
        )
    if summary["target"] is not None:
        lines.append(f"Target deliverable: `{summary['target']}`.")
    if summary["trunk"] is not None:
        lines.append(f"Trunk run (`{TRUNK_RECORD_NAME}`): `{summary['trunk'] or 'none'}`.")
    lines += [
        "",
        "Built from the run's files by `python -m environments.shared.scripts.run_report`, which reads the run and "
        "writes nothing in it; this page is rebuilt whole.",
        "",
        "## Nodes",
        "",
    ]
    if summary["nodes"]:
        lines += ["| Node | Directory | Verdict | Gate kind | Handoff |", "|---|---|---|---|---|"]
        for node in summary["nodes"]:
            if node["reused_from"] is not None:
                origin = node["reused_from"]
                shown = f"{origin['run'] or origin['run_id'] or 'another run'}/{origin['stage_dir'] or node['id']}"
                if origin["run"] is not None and start is not None:
                    trunk_report = model / REPORTS_DIRNAME / origin["run"] / REPORT_FILENAME
                    if trunk_report.is_file():
                        shown = _link(shown, _relative(trunk_report, start))
                    else:
                        shown += " (its report is not built yet)"
                handoff = f"reused from {shown}"
                verdict = "reused" + ("" if node["passed"] is None else f" ({'PASS' if node['passed'] else 'FAIL'})")
            else:
                handoff = _cell(node["handoff"])
                verdict = node["verdict"]
            directory = f"{node['own_dir']} (not published)" if node.get("own_dir") else node["dir"]
            lines.append(
                f"| {_cell(node['id'])} | {_cell(directory)} | {verdict} | {_cell(node['gate_kind'])} | {handoff} |"
            )
        for node in summary["nodes"]:
            if node.get("own_dir"):
                own_verdict = f" (its verdict there: {node['own_verdict']})" if node.get("own_verdict") else ""
                lines += [
                    "",
                    f"`{node['id']}`: `{ANCESTORS_DIRNAME}/{node['id']}/` stands for this node, and the bundle records "
                    f"it as reused. The run's own `{node['own_dir']}/`{own_verdict} stays in the run but is not "
                    "published; this page reads nothing else from it.",
                ]
        for node in summary["nodes"]:
            failures = node["failures"]
            if not failures:
                continue
            lines += ["", f"`{node['id']}` failed its gate:"] + [f"- {failure}" for failure in failures[:_FAILURES]]
            if len(failures) > _FAILURES:
                lines.append(f"- and {len(failures) - _FAILURES} more ({node['dir']}/{GATE_VERDICT_FILENAME})")
    else:
        lines.append("No stage directory and no `ancestors/` record.")
    reports = [
        (node["id"], "certificate", node["certificate"]) for node in summary["nodes"] if node["certificate"] is not None
    ] + [(node["id"], f"{probe['label']} probe", probe) for node in summary["nodes"] for probe in node["probes"]]
    lines += ["", "## Certificates and probes", ""]
    if reports:
        lines += [
            f"- `{stage}` {what}, {_link(item['path'], to_run(item['path']))}: {item['line']}"
            for stage, what, item in reports
        ]
    else:
        lines.append("None: no stage holds a stance certificate or probe report.")
    lines += [
        f"- `{node['id']}` probes: none, as expected: {node['probe_note']}."
        for node in summary["nodes"]
        if node["probe_note"]
    ]
    lines += ["", "## Deliverables", ""]
    if summary["deliverables"] and summary["manifest_status"] is None:
        lines += [
            f"Recorded in `{DEFAULT_PROVENANCE_NAME}` by a bundle save that sealed no bundle (the run's "
            f"`{DEFAULT_MANIFEST_NAME}` is missing or cannot be read), so none of these is published:",
            "",
        ]
    if summary["deliverables"]:
        lines += [
            "| Deliverable | Certified | Provisional | Replicates | Seeds | Handoff |",
            "|---|---|---|---|---|---|",
        ]
        for row in summary["deliverables"]:
            lines.append(
                f"| {_cell(row['id'])} | {'yes' if row['certified'] else 'no'} | "
                f"{'yes' if row['provisional'] else 'no'} | {_cell(row['replicates'])} | "
                f"{_cell(row['certification_seeds'])} | {_cell(row['handoff'])} |"
            )
    else:
        held = {entry["path"] for entry in built["files"]}
        if summary["provenance"]:
            lines.append(
                f"None recorded: the run's `{DEFAULT_PROVENANCE_NAME}` holds no deliverables (no bundle was written)."
            )
        else:
            text = f"None recorded: the run holds no `{DEFAULT_PROVENANCE_NAME}`, so no bundle records deliverables."
            if "collected_results.csv" in held:
                text += " Its `collected_results.csv` holds its stage results, as a run from before result bundles did."
            if "curriculum_results.csv" in held:
                text += " Its `curriculum_results.csv` holds a row per stage (the command-line curriculum writes no bundle)."
            lines.append(text)
    files = built["files"]
    lines += ["", "## Files", ""]
    for heading, roles in (
        ("Figures", ("figure",)),
        ("Training history", ("training_history",)),
        ("Replays", ("replay",)),
    ):
        chosen = [entry["path"] for entry in files if entry["role"] in roles]
        if heading == "Training history":
            chosen = [path for path in chosen if path.endswith(".html")] or chosen
        lines.append(f"**{heading}:** " + (", ".join(_link(path, to_run(path)) for path in chosen) or "none"))
        lines.append("")
    hashed = sum(1 for entry in files if entry["hash_source"] == "manifest")
    later = [entry for entry in files if entry["hash_source"] != "manifest" and entry["path"] != DEFAULT_MANIFEST_NAME]
    if hashed:
        lines.append(
            f"`{INDEX_JSON_FILENAME}` beside this page lists the run's {len(files)} files: {hashed} with the size and "
            f"digest `{DEFAULT_MANIFEST_NAME}` records"
            + (", the manifest itself" if later else ", and the manifest itself")
            + (f", and {len(later)} it does not list (written after it), without a digest" if later else "")
            + "; nothing is re-hashed."
        )
    else:
        lines.append(
            f"`{INDEX_JSON_FILENAME}` beside this page lists the run's {len(files)} files with their sizes and no "
            f"digest: the run holds no `{DEFAULT_MANIFEST_NAME}` it can read, and nothing is hashed."
        )
    lines += ["", "## Warnings", ""]
    lines += [f"- {warning}" for warning in built["warnings"]] or ["None."]
    return "\n".join(lines) + "\n"


def _report_of(path: str, model_dir: Path) -> "str | None":
    """The run's ``report.md`` relative to ``reports/``, or ``None`` while it has none."""
    report = Path(path) / REPORT_FILENAME
    return report.as_posix() if (model_dir / REPORTS_DIRNAME / report).is_file() else None


def _row(summary: Mapping[str, Any], model_dir: Path) -> dict[str, Any]:
    return {
        "run_id": summary["run_id"],
        "path": summary["path"],
        "tree": summary["tree"],
        "kind": summary["kind"],
        "layout": summary["layout"],
        "state": summary["state"],
        "started": summary["started"],
        "target": summary["target"],
        "certified": summary["certified"],
        "passed": summary["passed"],
        "provisional": summary["provisional"],
        "replicates": summary["replicates"],
        "handoff": summary["handoff"],
        "excluded": summary["excluded"],
        "report": _report_of(summary["path"], model_dir),
    }


def _unreadable_row(run_dir: Path, model_dir: Path, error: Exception) -> dict[str, Any]:
    relative = run_dir.resolve().relative_to(model_dir.resolve())
    return {
        "run_id": run_dir.name,
        "path": relative.as_posix(),
        "tree": relative.parent.as_posix() if relative.parent.parts else "",
        "kind": None,
        "layout": None,
        "state": UNREADABLE,
        "started": None,
        "target": None,
        "certified": [],
        "passed": [],
        "provisional": None,
        "replicates": None,
        "handoff": None,
        "excluded": None,
        "report": _report_of(relative.as_posix(), model_dir),
        "error": f"{type(error).__name__}: {error}",
    }


def build_model_index(
    model_dir: "str | Path", *, summaries: "Mapping[str, Mapping[str, Any]] | None" = None
) -> list[dict[str, Any]]:
    """One row per run of the model at *model_dir* (:func:`discover_runs`), grouped by state, newest first.

    Groups follow :data:`INDEX_STATES`; within a group, the run that started
    last comes first (its provenance's ``captured_at``, else the timestamp
    in its name), and a run with no start time after them, by name.
    ``report`` is the run's ``report.md`` relative to ``reports/``, or
    ``None`` while it has none.  Each run is read without detail (the
    module docstring's cost), except one whose summary *summaries* gives
    already (by its path under *model_dir*: a report just built).  A run
    whose records raise anything unforeseen is a :data:`UNREADABLE` row
    with ``error`` set, never an exception.
    """
    model = Path(model_dir).resolve()
    known = summaries or {}
    rows = []
    for run in discover_runs(model):
        try:
            relative = run.relative_to(model).as_posix()
            summary = known[relative] if relative in known else _run_summary(run, model, detail=False)
            rows.append(_row(summary, model))
        except Exception as exc:  # noqa: BLE001 - one run must never cost the index the others
            rows.append(_unreadable_row(run, model, exc))
    rows.sort(key=lambda row: row["path"])
    rows.sort(key=lambda row: row["started"] or "", reverse=True)
    rows.sort(key=lambda row: (INDEX_STATES.index(row["state"]), row["started"] is None))
    return rows


def _model_record(model: Path) -> "dict[str, str] | None":
    try:
        return ModelId(model.parent.name, model.name).record()
    except ValueError:
        return None


def model_index_document(model_dir: "str | Path", rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The model's ``reports/index.json``: :data:`MODEL_INDEX_SCHEMA`, the model and its rows."""
    return {
        "schema": MODEL_INDEX_SCHEMA,
        "model": _model_record(Path(model_dir).resolve()),
        "runs": [dict(row) for row in rows],
    }


def render_model_index(model_dir: "str | Path", rows: Sequence[Mapping[str, Any]]) -> str:
    """The model's ``reports/INDEX.md``: a table per state, newest first, each run linked to its report."""
    model = Path(model_dir).resolve()
    try:
        from ..species_names import model_display_name

        title = model_display_name(ModelId(model.parent.name, model.name))
    except (ValueError, KeyError):
        title = model.name
    counts = {state: sum(1 for row in rows if row["state"] == state) for state in INDEX_STATES}
    tally = ", ".join(f"{count} {state}" for state, count in counts.items() if count) or "none"
    lines = [
        f"# Runs of {title}",
        "",
        f"{len(rows)} run(s): {tally}. Built by `python -m environments.shared.scripts.run_report` from the runs under "
        "this model's directory, which it only reads; rebuilt whole. Each state is defined in "
        "`environments/shared/reporting/run_index.py`.",
    ]
    for state in INDEX_STATES:
        group = [row for row in rows if row["state"] == state]
        if not group:
            continue
        lines += [
            "",
            f"## {state}",
            "",
            "| Run | Tree | State | Certified | Provisional | Replicates | Handoff | Report |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for row in group:
            certified = _cell(row["certified"])
            if not row["certified"] and row["passed"]:
                certified = f"none published (passed: {_cell(row['passed'])})"
            if row.get("excluded"):
                certified += "; never counts (run_exclusions.py)"
            provisional = {None: "—", True: "yes", False: "no"}[row["provisional"]]
            report = _link("report", row["report"]) if row["report"] else "not built"
            if row.get("error"):
                report += f"; the index could not read the run: {_cell(row['error'])}"
            lines.append(
                f"| {_cell(row['run_id'])} | {_cell(row['tree'] or '(model directory)')} | {row['state']} | "
                f"{certified} | {provisional} | {_cell(row['replicates'])} | {_cell(row['handoff'])} | {report} |"
            )
    return "\n".join(lines) + "\n"
