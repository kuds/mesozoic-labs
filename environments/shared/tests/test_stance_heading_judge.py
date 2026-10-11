"""The judge of the six-heading stance certificate (``stance_quality_heading/v1``, decision D-D32).

``reporting.gates`` admits a certificate only when it describes this gate,
protocol, grid, seed block, measurement, task and handoff pair, and then
re-derives the verdict from every (seed, heading) row: a claimed ``passed``
is never the evidence, a v2 report or the report-only probe is never
promoted, and a stage with no certificate never falls back to the yaw-zero
panel.  Every certificate here is the synthetic one of
:mod:`.stance_heading_helpers`; ``test_stance_heading_report.py`` rolls
real ones.
"""

from __future__ import annotations

import copy
import json
import math
from typing import Any, Callable

import pytest

from environments.shared.curriculum import stance_heading_gate
from environments.shared.reporting import evaluate_recorded_gate, evaluate_stage_gate
from environments.shared.reporting.gates import stance_heading_report_refusals
from environments.shared.reporting.stage_artifacts import stance_heading_stage_result

from .stance_heading_helpers import FALL, HEADING_CURRICULUM, heading_report, seed52_like_report
from .stance_v2_helpers import V2_CURRICULUM, v2_report


def judge(report: Any, stage_dir: Any, **curriculum: Any) -> tuple[bool, list[str]]:
    return evaluate_stage_gate(
        {**HEADING_CURRICULUM, **curriculum},
        {"best_model_reward": 1e9},
        stage=1,
        stance_report=report,
        stage_dir=stage_dir,
    )


def refused(report: Any, stage_dir: Any, fragment: str, **curriculum: Any) -> None:
    passed, failures = judge(report, stage_dir, **curriculum)
    assert passed is False
    assert any(fragment in failure for failure in failures), failures
    assert stance_heading_report_refusals({**HEADING_CURRICULUM, **curriculum}, report, stage=1, stage_dir=stage_dir)


def test_the_judge_certifies_a_valid_certificate_by_re_deriving_it(tmp_path, monkeypatch):
    report = heading_report(tmp_path)
    assert report["passed"] is True
    assert judge(report, tmp_path) == (True, [])
    assert stance_heading_report_refusals(HEADING_CURRICULUM, report, stage=1, stage_dir=tmp_path) == []
    # Through JSON, as the judge reads it from stance_heading_report.json.
    assert judge(json.loads(json.dumps(report)), tmp_path) == (True, [])

    def consulted(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("consulted")

    monkeypatch.setattr(stance_heading_gate, "evaluate_stance_heading_gate", consulted)
    with pytest.raises(RuntimeError, match="consulted"):
        judge(report, tmp_path)


def test_no_certificate_is_refused_and_never_falls_back_to_yaw_zero(tmp_path):
    passed, failures = judge(None, tmp_path)
    assert passed is False
    assert "no six-heading certificate (stance_heading_report.json) was produced" in failures[0]
    assert "never falls back to the yaw-zero stance_quality/v2 panel" in failures[0]
    # A v2 PASS on the same handoff pair and task -- the yaw-zero panel -- is not the certificate.
    v2_pass = v2_report(tmp_path, V2_CURRICULUM)
    assert v2_pass["passed"] is True
    refused(v2_pass, tmp_path, "certifies yaw zero only and cannot be promoted")
    # Relabelled as the heading kind it is still a v3 report.
    refused({**v2_pass, "gate_kind": HEADING_CURRICULUM["gate_kind"]}, tmp_path, "certifies yaw zero only")


def test_a_heading_probe_record_is_refused_as_report_only(tmp_path):
    from environments.shared.reporting.stance_report import render_heading_probe

    from .test_stance_heading_probe import STANDS, _sweeps

    _, probe = render_heading_probe(*_sweeps(STANDS), probe_episodes=8)
    refused(probe, tmp_path, "is the report-only heading probe (mesozoic.stance-heading-probe/v1)")


def _heading(report: dict[str, Any], yaw: float) -> dict[str, Any]:
    return next(heading for heading in report["headings"] if heading["yaw_deg"] == yaw)


def _drop_heading(report: dict[str, Any]) -> None:
    report["heading_grid_deg"] = report["heading_grid_deg"][:-1]
    report["headings"] = report["headings"][:-1]


#: (mutation of a valid certificate, the refusal it must draw).
CORRUPTIONS: dict[str, tuple[Callable[[dict[str, Any]], None], str]] = {
    "schema": (lambda r: r.update(schema="mesozoic.stance-heading-report/v0"), "has schema"),
    "scored-kind": (lambda r: r.update(scored_gate_kind="stance_quality/v2"), "was scored by 'stance_quality/v2'"),
    "declared-kind": (lambda r: r.update(gate_kind="stance_quality/v2"), "records declared gate_kind"),
    "protocol": (lambda r: r.update(protocol="animal_yaw/v1"), "rolled under protocol 'animal_yaw/v1'"),
    "grid-missing-heading": (_drop_heading, "missing +180"),
    "grid-plus-minus-180": (
        lambda r: r.update(heading_grid_deg=[*r["heading_grid_deg"], -180.0]),
        "which are one heading",
    ),
    "grid-reordered": (
        lambda r: r.update(heading_grid_deg=[45.0, -90.0, -45.0, 0.0, 90.0, 180.0]),
        "reordered",
    ),
    "cells-reordered": (lambda r: r.update(headings=list(reversed(r["headings"]))), "not one per heading"),
    "seeds-fresh-audit": (
        lambda r: r.update(seeds=[7042 + index for index in range(len(r["seeds"]))]),
        "not the certification panel 3042",
    ),
    "seeds-shuffled": (lambda r: r.update(seeds=[r["seeds"][1], r["seeds"][0], *r["seeds"][2:]]), "rolled seeds"),
    "stochastic": (lambda r: r.update(policy_deterministic=False), "policy_deterministic=False"),
    "normalization": (lambda r: r.update(normalization={"mode": "none"}), "normalised observations with 'none'"),
    "plant": (lambda r: r.update(checkpoint_plant_validated=False), "checkpoint_plant_validated=False"),
    "measurement": (
        lambda r: r["measurement"]["constants"].update(spawn_grace_s=0.2),
        "was measured under constants",
    ),
    "statue-reused": (
        lambda r: _heading(r, 90.0)["statue"].update(reused_policy_panel=True),
        "certificate's +90 deg statue panel is not a separately rolled one",
    ),
    "statue-policy": (
        lambda r: _heading(r, -45.0)["statue"].update(policy="random actions"),
        "certificate's -45 deg statue panel scored 'random actions'",
    ),
    "gate-digest": (lambda r: r.update(gate_sha256="sha256:" + "0" * 64), "records gate_sha256"),
    "probe-marker": (
        lambda r: r.update(spawn_yaw={"kind": "spawn_yaw/v1", "yaw_deg": 90.0, "whole_scene": True}),
        "is a probe (spawn_yaw)",
    ),
    "horizon": (lambda r: r.update(horizon=500), "scored a 500-step horizon"),
    "task": (lambda r: r.update(task_sha256="sha256:" + "0" * 64), "rolled task"),
    "verdict-type": (lambda r: r.update(passed="yes"), "carries no boolean verdict"),
}


@pytest.mark.parametrize("name", sorted(CORRUPTIONS))
def test_a_certificate_that_does_not_describe_this_gate_is_refused(tmp_path, name):
    report = heading_report(tmp_path)
    mutate, fragment = CORRUPTIONS[name]
    mutate(report)
    refused(report, tmp_path, fragment)


def test_stale_model_or_normalizer_hashes_are_refused(tmp_path):
    report = heading_report(tmp_path)
    (tmp_path / "models" / "robust_best_model.zip").write_bytes(b"retrained weights")
    refused(report, tmp_path, "scored checkpoint")
    report = heading_report(tmp_path / "other")
    (tmp_path / "other" / "models" / "robust_best_model_vecnorm.pkl").write_bytes(b"other statistics")
    refused(report, tmp_path / "other", "scored VecNormalize sidecar")
    refused(heading_report(tmp_path / "third"), None, "no stage_dir was given")


def test_altered_thresholds_are_refused(tmp_path):
    report = heading_report(tmp_path)
    passed, failures = judge(report, tmp_path, min_all_feet_support=0.99)
    assert passed is False
    assert any("scored under other thresholds" in failure and "min_all_feet_support" in failure for failure in failures)
    assert any("records gate_sha256" in failure for failure in failures)
    # The grouped bar is the same key: a certificate scored at 0.80 does not certify a 0.85 gate.
    refused(report, tmp_path, "min_clean_stance_lcb: judged at 0.8, configured 0.85", min_clean_stance_lcb=0.85)


def test_another_measurement_version_is_refused(tmp_path):
    from environments.shared.gait.constants import measurement_sha256

    report = heading_report(tmp_path)
    report["measurement"]["constants"]["measurement_version"] = "floor-truth/v0"
    report["measurement_sha256"] = measurement_sha256(report["measurement"])
    refused(report, tmp_path, "was measured under constants")
    report = heading_report(tmp_path / "edited")
    report["measurement"]["frame_skip"] = 10
    refused(report, tmp_path / "edited", "measurement_sha256")


def test_nonfinite_cells_or_missing_statue_rows_are_refused(tmp_path):
    def with_row(mutate: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        report = heading_report(tmp_path)
        mutate(report)
        return report

    # A row's heading that is not a finite number of degrees.
    report = with_row(lambda r: _heading(r, 45.0)["episode_evidence"][3].update(yaw_deg=math.nan))
    refused(report, tmp_path, "cannot be re-derived")
    # A row recorded at another heading than its cell.
    report = with_row(lambda r: _heading(r, 45.0)["episode_evidence"][3].update(yaw_deg=-45.0))
    refused(report, tmp_path, "of the +45 deg cell records heading -45.0")
    # An unmeasured metric (JSON null) under a clean flag that says it was measured.
    report = with_row(lambda r: _heading(r, 90.0)["episode_evidence"][0].update(all_feet_support=None))
    refused(report, tmp_path, "the +90 deg cell's clean flags differ")
    # A missing row, and a row off the seed list.
    report = with_row(lambda r: _heading(r, 180.0)["episode_evidence"].pop())
    refused(report, tmp_path, "records 39 rows, not one per seed (40)")
    report = with_row(lambda r: _heading(r, 0.0)["episode_evidence"][5].update(seed=7047))
    refused(report, tmp_path, "ran on seed 7047, not 3047")
    # A statue-relative criterion is declared: every heading needs its statue rows, reducing to its reference.
    report = with_row(lambda r: _heading(r, -90.0).update(statue=None))
    refused(report, tmp_path, "the -90 deg cell rolled no statue panel")
    report = with_row(lambda r: _heading(r, -90.0)["statue"].update(episode_evidence=[]))
    refused(report, tmp_path, "statue panel at -90 deg records 0 rows")
    report = with_row(lambda r: _heading(r, 45.0)["statue"]["reference"].update(mean_reward=1.0))
    refused(report, tmp_path, "the +45 deg cell's recorded statue reference")


@pytest.mark.parametrize("value", [1, -1.5, True, math.nan, "rows", {"seed": 3042}], ids=repr)
def test_a_cell_whose_rows_are_not_a_list_is_refused_not_raised(tmp_path, value):
    """A hand-edited cell is a refusal: the judge, reuse rule 8 and the backfill tool read it through here."""
    report = heading_report(tmp_path)
    _heading(report, -90.0)["episode_evidence"] = value
    refused(report, tmp_path, "certificate cannot be re-derived")


def test_a_recorded_verdict_its_rows_contradict_is_refused(tmp_path):
    report = seed52_like_report(tmp_path)
    assert report["passed"] is False and report["result"]["n_clean_seeds"] == 0
    laundered = copy.deepcopy(report)
    laundered.update(passed=True, failures=[])
    laundered["result"]["n_clean_seeds"] = 40
    for heading in laundered["headings"]:
        heading["result"]["n_clean"] = 40
        for row in heading["episode_evidence"]:
            row.update(clean=True, reasons=[])
    refused(laundered, tmp_path, "disagrees with the one re-derived from its own rows")
    # Only the per-heading count edited.
    edited = copy.deepcopy(report)
    _heading(edited, -90.0)["result"]["n_clean"] = 40
    refused(edited, tmp_path, "40 clean episodes recorded at -90 deg, 0 re-derived")
    # Only the clean-seed count edited, the verdict left as it re-derives: the count is checked on its own.
    edited = copy.deepcopy(report)
    edited["result"]["n_clean_seeds"] = 37
    refused(edited, tmp_path, "37 clean seeds recorded, 0 re-derived")
    clean = heading_report(tmp_path / "clean")
    clean["result"]["n_clean_seeds"] = 39
    refused(clean, tmp_path / "clean", "39 clean seeds recorded, 40 re-derived")


def test_a_claimed_passed_field_is_not_evidence(tmp_path):
    """The rows are the evidence: flipping ``passed`` alone is a refusal, and a valid FAIL fails on its reasons."""
    report = seed52_like_report(tmp_path)
    refused({**report, "passed": True}, tmp_path, "passed True recorded, False re-derived")
    clean = heading_report(tmp_path / "clean")
    refused({**clean, "passed": False}, tmp_path / "clean", "passed False recorded, True re-derived")


def test_a_failing_certificate_fails_with_its_re_derived_reasons(tmp_path):
    report = seed52_like_report(tmp_path)
    passed, failures = judge(report, tmp_path)
    assert passed is False
    assert failures[0] == (
        "stage 1 clean_seed_lcb 0.0000 < 0.8000 (0/40 seeds clean at every heading of -90, -45, +0, +45, +90, +180)"
    )
    assert any(failure.startswith("stage 1 heading -90: ") for failure in failures)
    assert any(failure.startswith("stage 1 heading -45: ") for failure in failures)
    assert not any(failure.startswith(("stage 1 heading +0", "stage 1 heading +45")) for failure in failures)
    # Admitted, so its numbers travel with the FAIL.
    assert stance_heading_report_refusals(HEADING_CURRICULUM, report, stage=1, stage_dir=tmp_path) == []
    numbers = stance_heading_stage_result(report)
    assert numbers["stance_heading_clean_seeds"] == 0 and numbers["stance_heading_n_seeds"] == 40
    assert numbers["stance_heading_clean_seed_lcb"] == 0.0


def test_three_falls_at_one_heading_fail_its_rail_with_the_grouped_bound_met(tmp_path):
    block = {**HEADING_CURRICULUM, "min_full_horizon_fraction": 0.95}
    report = heading_report(tmp_path, block, defects={(index, 90.0): FALL for index in range(3)})
    passed, failures = judge(report, tmp_path, min_full_horizon_fraction=0.95)
    assert passed is False and failures == ["stage 1 heading +90: full_horizon_fraction 0.9250 < 0.9500"]
    assert report["result"]["n_clean_seeds"] == 37


def test_a_heading_history_never_certifies(tmp_path):
    counted = [{"stance_clean_count": 40, "n_stance_samples": 40, "mean_reward": 1e9, "n_episodes": 40}] * 5
    assert evaluate_recorded_gate(HEADING_CURRICULUM, counted) is None
    assert evaluate_recorded_gate(HEADING_CURRICULUM, []) is None
