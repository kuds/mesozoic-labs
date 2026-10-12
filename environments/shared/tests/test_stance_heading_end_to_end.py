"""Yaw zero passes, -90 and -45 fall: every consumer of the six-heading certificate refuses it (decision D-D32).

The acceptance case the heading design asks for: a stance that
``stance_quality/v2`` certifies at the training heading -- seed 52's shape,
``20261009_155723``, 40/40 clean at 0 deg -- falls on every seed when the
whole scene is turned -90 or -45 degrees.  With the certificate declared,
the stage directory holds that certificate AND a v2 PASS beside it, and:

* the post-stage gate (``_apply_stage_gate``) records the stage FAILED,
  naming -90 and -45, and ``gate_verdict.json`` records it so;
* reuse refuses the FAILED verdict (rule 2), and refuses a FORGED passed
  verdict with the right gate digest on rule 8, which re-derives the
  certificate (``test_ancestors.py`` covers rule 8's passing, missing and
  edited cases);
* publication refuses a summary that claims ``stage_passed`` for it, from
  the grid CSV alone;
* the backfill tool writes the FAIL the rows re-derive to, never a PASS, and
  refuses a directory that holds only the v2 report
  (``test_backfill_gate_verdict.py``);
* the command-line curriculum's post-training judge fails the node, so it
  never resolves as its child's parent;
* and a v2 PASS cannot be promoted by attaching a heading report beside it.

The certificates are the synthetic ones of :mod:`.stance_heading_helpers`
(real rows, no rollout).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from environments.shared.ancestors import AncestorReuseError, find_certified_ancestor
from environments.shared.curriculum.gate_schema import gate_config_view
from environments.shared.curriculum.stance_heading_gate import STANCE_HEADING_PANEL_CSV
from environments.shared.model_id import ModelId
from environments.shared.reporting.stage_artifacts import _apply_stage_gate
from environments.shared.reporting.stance_report import write_stance_gate_report, write_stance_heading_report
from environments.shared.result_bundle import (
    ResultBundleError,
    read_gate_verdict,
    sha256_file,
    write_gate_verdict,
)
from environments.shared.result_bundle.evidence import _validate_stance_heading_panel_evidence
from environments.shared.stage_manifest import load_stage_manifest

from .ancestors_helpers import JUDGED_BY, trunk_plant
from .stance_heading_helpers import HEADING_CURRICULUM, heading_report, heading_trunk_run, seed52_like_report
from .stance_v2_helpers import HORIZON, TASK_SHA256, V2_CURRICULUM, v2_report

TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")


def _find(run_dir: Path, **overrides: Any):
    kwargs: dict[str, Any] = dict(
        model_id=TYRANNOSAURUS_REX,
        entry=load_stage_manifest(TYRANNOSAURUS_REX).by_id("stance"),
        current_task_sha256=TASK_SHA256,
        plant_identity=trunk_plant(),
        current_gate_config=HEADING_CURRICULUM,
    )
    kwargs.update(overrides)
    return find_certified_ancestor(run_dir, **kwargs)


def _forge_pass(stage_dir: Path) -> None:
    """A PASSED verdict for the heading block, hash-bound to the handoff pair: everything rules 2-7 check."""
    models = stage_dir / "models"
    write_gate_verdict(
        stage_dir,
        model_id=TYRANNOSAURUS_REX,
        stage=1,
        stage_id="stance",
        gate_kind=HEADING_CURRICULUM["gate_kind"],
        gate_schema_version=1,
        passed=True,
        failures=[],
        task_sha256=TASK_SHA256,
        judged_by=JUDGED_BY,
        checkpoint=models / "robust_best_model.zip",
        normalization=models / "robust_best_model_vecnorm.pkl",
        gate_config=gate_config_view(HEADING_CURRICULUM),
    )


def test_yaw_zero_passes_but_minus_90_fails_and_handoff_publication_and_reuse_all_refuse_it(tmp_path, monkeypatch):
    run_dir = tmp_path / "20261012_120000"
    stage_dir = heading_trunk_run(run_dir, passed=False, verdict=False)
    certificate = seed52_like_report(stage_dir)
    # The 0 deg cell alone would certify: 40/40 clean, the v2 decision PASS.
    zero = next(heading for heading in certificate["headings"] if heading["yaw_deg"] == 0.0)
    assert zero["result"]["passed"] is True and zero["result"]["n_clean"] == 40
    write_stance_heading_report(stage_dir, certificate)
    # ... and a v2 PASS on the same handoff pair and task sits beside it.
    v2_pass = v2_report(stage_dir, V2_CURRICULUM)
    assert v2_pass["passed"] is True
    write_stance_gate_report(stage_dir, v2_pass)

    # Handoff: the post-stage gate fails the stage, naming the two headings that fell, and records it.
    stage_results: dict[str, Any] = {"best_model_reward": 1e9}
    _apply_stage_gate(
        stage=1,
        stage_config={"curriculum_kwargs": HEADING_CURRICULUM},
        stage_results=stage_results,
        stance_report=certificate,
        stage_dir=stage_dir,
        model_id=TYRANNOSAURUS_REX,
    )
    assert stage_results["gate_passed"] is False and stage_results["publication_gate_passed"] is False
    named = {failure.split(":", 1)[0] for failure in stage_results["gate_failures"]}
    assert {"stage 1 heading -90", "stage 1 heading -45"} <= named
    assert not any(failure.startswith("stage 1 heading +0") for failure in stage_results["gate_failures"])
    assert stage_results["stance_heading_clean_seeds"] == 0 and stage_results["stance_heading_n_seeds"] == 40
    verdict = read_gate_verdict(stage_dir)
    assert verdict is not None and verdict["passed"] is False
    assert verdict["stage_result"]["stance_heading_clean_seed_lcb"] == 0.0
    assert verdict["gate"] == gate_config_view(HEADING_CURRICULUM)

    # Reuse: the FAILED verdict is refused (rule 2) ...
    with pytest.raises(AncestorReuseError, match="records a FAILED gate"):
        _find(run_dir)
    # ... and a forged PASS, hash-bound and under the right gate digest, is refused on rule 8.
    _forge_pass(stage_dir)
    with pytest.raises(AncestorReuseError, match=r"does not re-derive to a passing .*\(reuse rule 8\)") as excinfo:
        _find(run_dir)
    assert "heading -90" in str(excinfo.value) and "heading -45" in str(excinfo.value)

    # Publication: a summary that claims the stage passed is refused from the grid CSV alone.
    with pytest.raises(ResultBundleError, match=r"publication gate fails stance_quality_heading/v1.*0/40 seeds clean"):
        _validate_stance_heading_panel_evidence(
            stage_dir / STANCE_HEADING_PANEL_CSV,
            HEADING_CURRICULUM,
            env_kwargs={"max_episode_steps": HORIZON},
            stage=1,
            panel_seed_start=3042,
            certified_hash=sha256_file(stage_dir / "models" / "robust_best_model.zip"),
            certified_normalization=sha256_file(stage_dir / "models" / "robust_best_model_vecnorm.pkl"),
            model_id=TYRANNOSAURUS_REX,
            task_sha256=TASK_SHA256,
        )

    # Backfill: the FAIL the rows re-derive to, never a PASS.
    from environments.shared.scripts.backfill_gate_verdict import backfill_gate_verdict

    backfill_gate_verdict(stage_dir, force=True)
    backfilled = read_gate_verdict(stage_dir)
    assert backfilled is not None and backfilled["passed"] is False
    assert any("heading -90" in failure for failure in backfilled["failures"])

    # The command-line curriculum: the post-training judge fails the node, so it never resolves as a parent.
    from environments.shared import train_base
    from environments.shared.reporting import stage_artifacts

    monkeypatch.setattr(stage_artifacts, "_write_stance_gate_report", lambda **kwargs: certificate)
    passed, failures, stage_result = train_base._post_training_stance_verdict(
        TYRANNOSAURUS_REX,
        1,
        {"curriculum_kwargs": HEADING_CURRICULUM},
        stage_dir=stage_dir,
        model_dir=stage_dir / "models",
    )
    assert passed is False and any("heading -45" in failure for failure in failures)
    assert stage_result["stance_heading_clean_seeds"] == 0


def test_a_v2_pass_cannot_be_promoted_by_attaching_a_heading_report(tmp_path):
    """A v2 stage's verdict is v2's; a heading stage's needs its own certificate, whatever v2 report is beside it."""
    from environments.shared.reporting import evaluate_stage_gate

    stage_dir = tmp_path / "stage"
    v2_pass = v2_report(stage_dir, V2_CURRICULUM)
    write_stance_gate_report(stage_dir, v2_pass)
    # Under the heading kind the v2 PASS is refused by name.
    passed, failures = evaluate_stage_gate(HEADING_CURRICULUM, {}, stage=1, stance_report=v2_pass, stage_dir=stage_dir)
    assert passed is False and "certifies yaw zero only and cannot be promoted" in failures[0]
    # A heading report attached to the v2 report, or relabelled as one, is still a kind-v2 (schema /v5) report.
    attached = {**v2_pass, "headings": heading_report(tmp_path / "other")["headings"]}
    passed, failures = evaluate_stage_gate(HEADING_CURRICULUM, {}, stage=1, stance_report=attached, stage_dir=stage_dir)
    assert passed is False and "certifies yaw zero only" in failures[0]
    # And under v2, a certificate is not a v2 report either: each kind admits only its own.
    certificate = heading_report(tmp_path / "third")
    passed, failures = evaluate_stage_gate(V2_CURRICULUM, {}, stage=1, stance_report=certificate, stage_dir=stage_dir)
    # The kind-v2 report schema was re-pinned from /v3 to /v5 (decision D-D33).
    assert passed is False and any("not 'mesozoic.stance-gate-report/v5'" in failure for failure in failures)


def test_the_cli_curriculum_judges_a_heading_node_after_training(tmp_path, monkeypatch):
    """The command-line curriculum rolls the certificate through the post-stage writer and judges it (D-D32)."""
    from environments.shared import train_base
    from environments.shared.reporting import stage_artifacts

    calls = []

    def fake_write(*, model_id, stage, stage_config, stage_dir, model_dir):
        calls.append((model_id, stage, stage_config["curriculum_kwargs"]["gate_kind"], model_dir))
        return heading_report(stage_dir)

    monkeypatch.setattr(stage_artifacts, "_write_stance_gate_report", fake_write)
    config = {
        "name": "Balance",
        "description": "Stand",
        "curriculum_kwargs": dict(HEADING_CURRICULUM),
        "env_kwargs": {},
    }
    passed, failures, stage_result = train_base._post_training_stance_verdict(
        TYRANNOSAURUS_REX, 1, config, stage_dir=tmp_path, model_dir=tmp_path / "models"
    )
    assert calls == [(TYRANNOSAURUS_REX, 1, "stance_quality_heading/v1", tmp_path / "models")]
    assert (passed, failures) == (True, [])
    assert stage_result["gate_kind"] == "stance_quality_heading/v1" and stage_result["gate_passed"] is True
    assert stage_result["stance_heading_clean_seeds"] == 40 and "stance_clean_count" not in stage_result
    # No certificate is a named FAIL, never the yaw-zero panel's verdict.
    monkeypatch.setattr(stage_artifacts, "_write_stance_gate_report", lambda **kwargs: None)
    passed, failures, stage_result = train_base._post_training_stance_verdict(
        TYRANNOSAURUS_REX, 1, config, stage_dir=tmp_path, model_dir=tmp_path / "models"
    )
    assert passed is False and "no six-heading certificate" in failures[0]
    assert not any(key.startswith("stance_heading") for key in stage_result)
