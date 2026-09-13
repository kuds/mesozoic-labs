"""The ``ancestors/<stage_id>/`` reader: records of nodes reused from another run.

BEHAVIOR_RECIPES_PLAN §4.2 (Phase A).  The writer is
``environments.shared.ancestors.record_ancestor``; the helper here reproduces
its layout by hand so the reader is pinned independently of the writer.
Every malformation fails closed — absence of a file inside a record is an
error, never a pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from environments.shared.curriculum.gate_schema import gate_config_sha256
from environments.shared.result_bundle import ResultBundleError
from environments.shared.result_bundle.ancestors import (
    ANCESTOR_RECORD_KEYS,
    load_ancestor_records,
    project_ancestor_records,
)
from environments.shared.result_schema import ANCESTOR_RECORD_FIELDS

from .result_bundle_helpers import _ANCESTOR_TASK_SHA256, _TRUNK_RUN_ID, _write_ancestor_record


def _record_path(record_dir: Path) -> Path:
    return record_dir / "ancestor.json"


def _rewrite(record_dir: Path, mutate) -> None:
    record = json.loads(_record_path(record_dir).read_text(encoding="utf-8"))
    mutate(record)
    _record_path(record_dir).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_no_ancestors_directory_is_an_empty_result(tmp_path: Path) -> None:
    run_dir = tmp_path / "child"
    run_dir.mkdir()
    assert load_ancestor_records(run_dir, species="velociraptor") == {}


def test_records_load_and_project_to_provenance_fields(tmp_path: Path) -> None:
    run_dir = tmp_path / "child"
    run_dir.mkdir()
    _, checkpoint_hash = _write_ancestor_record(run_dir, 1)

    records = load_ancestor_records(run_dir, species="velociraptor")

    assert list(records) == ["1"]
    record = records["1"]
    assert set(record) == set(ANCESTOR_RECORD_KEYS)
    assert record["stage_id"] == "stance"
    assert record["run_id"] == _TRUNK_RUN_ID
    assert record["model_hash"] == checkpoint_hash
    assert record["passed"] is True
    assert record["task_sha256"] == _ANCESTOR_TASK_SHA256
    assert record["gate_kind"] == "reward_and_length/v1"
    projected = project_ancestor_records(records)
    assert set(projected["1"]) == set(ANCESTOR_RECORD_FIELDS)
    assert projected["1"]["model_hash"] == checkpoint_hash


def test_a_failed_verdict_loads_as_not_passed(tmp_path: Path) -> None:
    run_dir = tmp_path / "child"
    run_dir.mkdir()
    _write_ancestor_record(run_dir, 1, passed=False)
    assert load_ancestor_records(run_dir, species="velociraptor")["1"]["passed"] is False


def test_a_record_whose_hashes_disagree_with_its_verdict_is_refused(tmp_path: Path) -> None:
    run_dir = tmp_path / "child"
    run_dir.mkdir()
    record_dir, _ = _write_ancestor_record(run_dir, 1)
    _rewrite(record_dir, lambda record: record["handoff"].__setitem__("model_sha256", "sha256:" + "0" * 64))

    with pytest.raises(ResultBundleError, match="hashes the handoff checkpoint"):
        load_ancestor_records(run_dir, species="velociraptor")


def test_a_record_for_an_unknown_stage_id_is_refused(tmp_path: Path) -> None:
    run_dir = tmp_path / "child"
    run_dir.mkdir()
    record_dir, _ = _write_ancestor_record(run_dir, 1)
    record_dir.rename(record_dir.parent / "sprint")

    with pytest.raises(ResultBundleError, match="does not declare"):
        load_ancestor_records(run_dir, species="velociraptor")


def test_a_record_missing_its_gate_verdict_is_refused(tmp_path: Path) -> None:
    run_dir = tmp_path / "child"
    run_dir.mkdir()
    record_dir, _ = _write_ancestor_record(run_dir, 1)
    (record_dir / "gate_verdict.json").unlink()

    with pytest.raises(ResultBundleError, match="absence is never a pass"):
        load_ancestor_records(run_dir, species="velociraptor")


def test_a_record_missing_its_stage_config_is_refused(tmp_path: Path) -> None:
    run_dir = tmp_path / "child"
    run_dir.mkdir()
    record_dir, _ = _write_ancestor_record(run_dir, 1)
    (record_dir / "stage_config.json").unlink()

    with pytest.raises(ResultBundleError, match="missing its stage_config.json"):
        load_ancestor_records(run_dir, species="velociraptor")


def test_a_task_fingerprint_that_disagrees_between_record_and_config_is_refused(tmp_path: Path) -> None:
    run_dir = tmp_path / "child"
    run_dir.mkdir()
    record_dir, _ = _write_ancestor_record(run_dir, 1)
    config_path = record_dir / "stage_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["task_fingerprint"]["task_sha256"] = "sha256:" + "1" * 64
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ResultBundleError, match="records task_sha256"):
        load_ancestor_records(run_dir, species="velociraptor")


def test_a_copied_verdict_whose_gate_digest_disagrees_with_its_recorded_gate_is_refused(tmp_path: Path) -> None:
    """D-A22 / D-B8: the copied verdict must agree with itself — ``gate_sha256`` is the digest of
    ``gate`` — so a verdict edited after judging fails the child's bundle, through the same reader
    reuse uses.  The ancestor's recorded ``[curriculum]`` block is deliberately NOT compared: a
    directory re-judged under an edited gate is legitimate."""
    run_dir = tmp_path / "child"
    run_dir.mkdir()
    record_dir, _ = _write_ancestor_record(run_dir, 1)
    verdict_path = record_dir / "gate_verdict.json"
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    assert verdict["gate"]["thresholds"] == {"min_avg_reward": 1.0} and verdict["gate_sha256"].startswith("sha256:")
    verdict["gate"]["thresholds"]["min_avg_reward"] = 0.5
    verdict_path.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ResultBundleError, match="not the digest of the recorded gate block"):
        load_ancestor_records(run_dir, species="velociraptor")

    # A verdict judged under a block other than the one the copied config records still loads.
    verdict["gate_sha256"] = gate_config_sha256(verdict["gate"])
    verdict_path.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert load_ancestor_records(run_dir, species="velociraptor")["1"]["passed"] is True


def test_a_phase_a_record_without_gate_sha256_still_loads(tmp_path: Path) -> None:
    """A child of a pre-D-A22 reuse stays auditable: the copied verdict carries neither field."""
    run_dir = tmp_path / "child"
    run_dir.mkdir()
    record_dir, checkpoint_hash = _write_ancestor_record(run_dir, 1)
    verdict_path = record_dir / "gate_verdict.json"
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    phase_a = {key: value for key, value in verdict.items() if key not in {"gate", "gate_sha256"}}
    verdict_path.write_text(json.dumps(phase_a, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    records = load_ancestor_records(run_dir, species="velociraptor")
    assert set(records["1"]) == set(ANCESTOR_RECORD_KEYS)
    assert records["1"]["model_hash"] == checkpoint_hash and records["1"]["passed"] is True


def test_a_record_without_a_parent_run_id_is_refused(tmp_path: Path) -> None:
    run_dir = tmp_path / "child"
    run_dir.mkdir()
    record_dir, _ = _write_ancestor_record(run_dir, 1)
    _rewrite(record_dir, lambda record: record.__setitem__("parent_run_id", ""))

    with pytest.raises(ResultBundleError, match="parent_run_id"):
        load_ancestor_records(run_dir, species="velociraptor")
