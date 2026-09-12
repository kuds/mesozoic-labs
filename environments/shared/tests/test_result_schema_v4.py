"""Result schema v4: publication per deliverable (BEHAVIOR_RECIPES_PLAN §4.3, Phase A).

The rules the bundle writer, the summary and the audit share: which
deliverables a run certified (its own gate plus every ``warm_start_from``
ancestor's), which one is published (the target when certified, else the
deepest certified), and the target-aware bundle status.  The v2/v3 pins in
``test_result_summaries.py`` are untouched: below schema 4 every rule runs
verbatim, and ``require_publishable`` is a synonym of ``require_complete``.
"""

from __future__ import annotations

import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from environments.shared import stage_manifest
from environments.shared.result_schema import (
    CANONICAL_RUNTIME_PROVENANCE_FIELDS,
    CANONICAL_RUNTIME_PROVENANCE_FIELDS_V4,
    ResultSchemaError,
    bundle_status_for,
    certified_deliverables,
    primary_deliverable_key,
    validate_result_summary,
)
from environments.shared.stage_manifest import load_stage_manifest

from .test_result_summaries import _canonical_summary


def _entries(species: str, *ids: str) -> list[tuple[str, Any]]:
    manifest = load_stage_manifest(species)
    return [(entry.key, entry) for entry in manifest.stages if not ids or entry.id in ids]


# ── The shared rules ─────────────────────────────────────────────────────


def test_certified_requires_the_whole_chain() -> None:
    # trex: stance -> recovery, stance -> locomotion -> behavior; every node a deliverable.
    entries = _entries("trex")
    certified = certified_deliverables(
        entries, {"1": True, "recovery": False, "2": True, "3": False}, None, species="trex"
    )
    assert certified == {"1": True, "recovery": False, "2": True, "3": False}
    # A failed root certifies nothing below it, whatever the children's own verdicts.
    certified = certified_deliverables(
        entries, {"1": False, "recovery": True, "2": True, "3": True}, None, species="trex"
    )
    assert certified == {"1": False, "recovery": False, "2": False, "3": False}
    # Records carrying the verdict (a summary's stage rows) read the same as bools.
    certified = certified_deliverables(
        entries,
        {
            "1": {"stage_passed": True},
            "recovery": {"stage_passed": False},
            "2": {"stage_passed": True},
            "3": {"stage_passed": True},
        },
        None,
        species="trex",
    )
    assert certified == {"1": True, "recovery": False, "2": True, "3": True}


def test_an_absent_ancestor_never_reads_as_passed() -> None:
    # Walk and hunt recorded without stance: their chain is incomplete.
    entries = _entries("velociraptor", "locomotion", "behavior")
    assert certified_deliverables(entries, {"2": True, "3": True}, None, species="velociraptor") == {
        "2": False,
        "3": False,
    }


def test_ancestor_records_complete_a_chain() -> None:
    entries = _entries("velociraptor", "locomotion", "behavior")
    stages = {"2": True, "3": True}
    assert certified_deliverables(entries, stages, {"1": {"passed": True}}, species="velociraptor") == {
        "2": True,
        "3": True,
    }
    assert certified_deliverables(entries, stages, {"1": {"passed": False}}, species="velociraptor") == {
        "2": False,
        "3": False,
    }
    # A stage cannot be both trained here and reused from another run.
    with pytest.raises(ResultSchemaError, match="both as a trained stage and as a reused ancestor"):
        certified_deliverables(
            _entries("velociraptor"), {"1": True, "2": True, "3": True}, {"1": True}, species="velociraptor"
        )


def test_primary_is_the_target_when_certified_else_the_deepest_certified() -> None:
    deliverables = {"1": True, "recovery": False, "2": True, "3": False}
    # The hunt target failed: walk, the deepest certified, is published.
    assert primary_deliverable_key(deliverables, species="trex", target="3") == "2"
    assert primary_deliverable_key(deliverables, species="trex", target=3) == "2"
    # Walk targeted and certified: walk.
    assert primary_deliverable_key(deliverables, species="trex", target="2") == "2"
    # Only the stand recipe certified: recovery is deeper than stance along its chain.
    assert primary_deliverable_key({"1": True, "recovery": True, "2": False}, species="trex", target=3) == "recovery"
    # Nothing certified: no primary, so no summary.
    assert primary_deliverable_key({"1": False, "2": False}, species="trex", target=3) is None


def test_bundle_status_is_target_aware() -> None:
    assert bundle_status_for({"1": True, "2": True}, species="velociraptor", target="2") == "complete"
    # The same run aimed at hunt is partial: its target is absent.
    assert bundle_status_for({"1": True, "2": True}, species="velociraptor", target="3") == "partial"
    assert bundle_status_for({"1": True, "2": True}, species="velociraptor", target=None) == "partial"
    assert bundle_status_for({"1": True, "2": False, "3": False}, species="velociraptor", target="3") == "partial"
    # Every present deliverable certified and the target present: complete.
    assert bundle_status_for({"1": True, "2": True, "3": True}, species="velociraptor", target="3") == "complete"
    assert bundle_status_for({"1": False, "2": False, "3": False}, species="velociraptor", target="3") == "failed"


def test_a_run_with_no_deliverable_yet_but_every_stage_passing_is_partial() -> None:
    # The pre-Phase-A rule under a v1 / synthesized manifest (one deliverable,
    # the last advancing node): a passing stance-only run is partial, not failed.
    assert bundle_status_for({}, species="velociraptor", target="3", stages={"1": True}) == "partial"
    assert bundle_status_for({}, species="velociraptor", target="3", stages={"1": False}) == "failed"


# ── The v4 canonical summary ─────────────────────────────────────────────


def _canonical_summary_v4() -> dict[str, Any]:
    summary = deepcopy(_canonical_summary())
    summary["schema_version"] = 4
    provenance = summary["provenance"]
    provenance["deliverables"] = {
        key: {
            "model_path": checkpoint["model_path"],
            "model_hash": checkpoint["model_hash"],
            "normalization_hash": checkpoint["normalization_hash"],
            "gate_kind": "reward_and_length/v1",
            "certified": True,
            "replication": {"count": 1, "runs": [{"run_id": provenance["run_id"], "training_seed": summary["seed"]}]},
        }
        for key, checkpoint in provenance["selected_checkpoints"].items()
    }
    provenance["primary_deliverable"] = "3"
    provenance["target_deliverable"] = "3"
    return summary


def _fail_stage(summary: dict[str, Any], key: str) -> None:
    summary["stages"][key]["stage_passed"] = False
    summary["stages"][key]["publication_gate_passed"] = False


def test_v4_canonical_summary_validates_complete() -> None:
    summary = _canonical_summary_v4()
    validate_result_summary(summary, expected_species="velociraptor", require_complete=True, canonical_provenance=True)
    validate_result_summary(summary, expected_species="velociraptor", require_complete=False, require_publishable=True)


def test_v4_runtime_fields_extend_the_v3_tuple_without_changing_it() -> None:
    assert CANONICAL_RUNTIME_PROVENANCE_FIELDS_V4[: len(CANONICAL_RUNTIME_PROVENANCE_FIELDS)] == (
        CANONICAL_RUNTIME_PROVENANCE_FIELDS
    )
    assert set(CANONICAL_RUNTIME_PROVENANCE_FIELDS_V4) - set(CANONICAL_RUNTIME_PROVENANCE_FIELDS) == {
        "deliverables",
        "primary_deliverable",
        "target_deliverable",
    }


@pytest.mark.parametrize("field", ["deliverables", "primary_deliverable", "target_deliverable"])
def test_v4_canonical_provenance_requires_deliverables_primary_and_target(field: str) -> None:
    summary = _canonical_summary_v4()
    del summary["provenance"][field]

    with pytest.raises(ResultSchemaError, match=rf"canonical provenance.*missing fields.*{field}"):
        validate_result_summary(summary, canonical_provenance=True)


def test_v4_primary_must_be_certified_never_the_failed_leaf() -> None:
    """Plan invariant 5: a failed hunt publishes walk; the leaf never headlines."""
    summary = _canonical_summary_v4()
    _fail_stage(summary, "3")
    provenance = summary["provenance"]
    provenance["deliverables"]["3"]["certified"] = False
    provenance["primary_deliverable"] = "2"
    provenance["selected_model_path"] = provenance["selected_checkpoints"]["2"]["model_path"]
    provenance["model_hash"] = provenance["selected_checkpoints"]["2"]["model_hash"]
    summary["bundle_status"] = "partial"
    summary["final_avg_reward"] = summary["stages"]["2"]["final_eval_reward"]

    validate_result_summary(
        summary,
        expected_species="velociraptor",
        require_complete=False,
        require_publishable=True,
        canonical_provenance=True,
    )
    with pytest.raises(ResultSchemaError, match="bundle_status 'partial'"):
        validate_result_summary(summary, expected_species="velociraptor", require_complete=True)

    # Naming the failed leaf as the primary is refused.
    laundered = deepcopy(summary)
    laundered["provenance"]["primary_deliverable"] = "3"
    with pytest.raises(ResultSchemaError, match="primary_deliverable.*must be '2'"):
        validate_result_summary(laundered, require_complete=False, require_publishable=True, canonical_provenance=True)

    # So is headlining the leaf's checkpoint.
    headlined = deepcopy(summary)
    headlined["provenance"]["selected_model_path"] = provenance["selected_checkpoints"]["3"]["model_path"]
    with pytest.raises(ResultSchemaError, match="selected_model_path.*primary deliverable"):
        validate_result_summary(headlined, require_complete=False, require_publishable=True, canonical_provenance=True)


def test_v4_certified_flag_cannot_be_laundered() -> None:
    summary = _canonical_summary_v4()
    _fail_stage(summary, "3")
    # The leaf failed but is recorded as certified.
    with pytest.raises(ResultSchemaError, match="recorded as certified, but its recorded verdicts"):
        validate_result_summary(summary, require_complete=False, require_publishable=True)

    # The mirror image: a node whose whole chain passed cannot be hidden either.
    summary = _canonical_summary_v4()
    summary["provenance"]["deliverables"]["3"]["certified"] = False
    with pytest.raises(ResultSchemaError, match="recorded as uncertified although"):
        validate_result_summary(summary, require_complete=False, require_publishable=True)


def test_v4_bundle_status_must_match_the_deliverables_and_target() -> None:
    summary = _canonical_summary_v4()
    summary["bundle_status"] = "partial"
    with pytest.raises(ResultSchemaError, match="bundle_status.*must be 'complete'"):
        validate_result_summary(summary, require_complete=False, require_publishable=True)

    summary = _canonical_summary_v4()
    summary["bundle_status"] = "certified"
    with pytest.raises(ResultSchemaError, match="bundle_status.*must be one of"):
        validate_result_summary(summary, require_complete=False)


def test_v4_summary_with_no_certified_deliverable_is_not_publishable() -> None:
    summary = _canonical_summary_v4()
    for key in ("1", "2", "3"):
        _fail_stage(summary, key)
        summary["provenance"]["deliverables"][key]["certified"] = False
    summary["provenance"]["primary_deliverable"] = None
    summary["bundle_status"] = "failed"

    # A non-publishable stage subset still validates as such.
    validate_result_summary(summary, expected_species="velociraptor", require_complete=False)
    with pytest.raises(ResultSchemaError, match="records no certified deliverable"):
        validate_result_summary(summary, require_complete=False, require_publishable=True)
    with pytest.raises(ResultSchemaError):
        validate_result_summary(summary, require_complete=True)


def test_v4_deliverables_must_record_every_deliverable_stage_the_summary_records() -> None:
    summary = _canonical_summary_v4()
    del summary["provenance"]["deliverables"]["2"]
    with pytest.raises(ResultSchemaError, match="must record every deliverable stage.*missing \\['2'\\]"):
        validate_result_summary(summary, require_complete=False, require_publishable=True)


def test_v4_walk_only_summary_is_complete_when_walk_is_the_target() -> None:
    summary = _canonical_summary_v4()
    del summary["stages"]["3"]
    provenance = summary["provenance"]
    del provenance["selected_checkpoints"]["3"]
    del provenance["deliverables"]["3"]
    provenance["target_deliverable"] = "2"
    provenance["primary_deliverable"] = "2"
    provenance["selected_model_path"] = provenance["selected_checkpoints"]["2"]["model_path"]
    provenance["model_hash"] = provenance["selected_checkpoints"]["2"]["model_hash"]
    summary["final_avg_reward"] = summary["stages"]["2"]["final_eval_reward"]
    summary["total_timesteps"] = sum(int(stage["timesteps"]) for stage in summary["stages"].values())
    summary["total_training_time_seconds"] = round(
        sum(float(stage.get("training_time_seconds") or 0.0) for stage in summary["stages"].values()), 1
    )

    validate_result_summary(summary, expected_species="velociraptor", require_complete=True, canonical_provenance=True)

    # The same two stages aimed at hunt are partial: the target is absent.
    aimed_at_hunt = deepcopy(summary)
    aimed_at_hunt["provenance"]["target_deliverable"] = "3"
    aimed_at_hunt["bundle_status"] = "partial"
    validate_result_summary(
        aimed_at_hunt,
        expected_species="velociraptor",
        require_complete=False,
        require_publishable=True,
        canonical_provenance=True,
    )
    with pytest.raises(ResultSchemaError, match="bundle_status 'partial'"):
        validate_result_summary(aimed_at_hunt, expected_species="velociraptor", require_complete=True)


def test_v4_without_publication_flags_accepts_a_summary_without_deliverables() -> None:
    # A Colab run checked after each node, before anything is publishable.
    summary = _canonical_summary_v4()
    del summary["provenance"]["deliverables"]
    del summary["provenance"]["primary_deliverable"]
    del summary["provenance"]["target_deliverable"]
    validate_result_summary(summary, expected_species="velociraptor", require_complete=False)
    with pytest.raises(ResultSchemaError, match="without provenance.deliverables"):
        validate_result_summary(
            summary, expected_species="velociraptor", require_complete=False, require_publishable=True
        )


def test_require_publishable_equals_require_complete_below_v4() -> None:
    summary = deepcopy(_canonical_summary())
    assert summary["schema_version"] < 4
    del summary["stages"]["3"]
    summary["total_timesteps"] = sum(int(stage["timesteps"]) for stage in summary["stages"].values())
    summary["final_avg_reward"] = summary["stages"]["2"]["final_eval_reward"]

    validate_result_summary(summary, expected_species="velociraptor", require_complete=False)
    with pytest.raises(ResultSchemaError, match="must contain every advancing stage"):
        validate_result_summary(summary, expected_species="velociraptor", require_complete=True)
    with pytest.raises(ResultSchemaError, match="must contain every advancing stage"):
        validate_result_summary(
            summary, expected_species="velociraptor", require_complete=False, require_publishable=True
        )


# ── The v4 provenance shape validators ───────────────────────────────────


def _canonical_publishable(summary: dict[str, Any]) -> dict[str, Any]:
    return validate_result_summary(
        summary,
        expected_species="velociraptor",
        require_complete=False,
        require_publishable=True,
        canonical_provenance=True,
    )


def _walk_only_v4() -> dict[str, Any]:
    """The v4 canonical summary reduced to stance and walk, walk targeted."""
    summary = _canonical_summary_v4()
    del summary["stages"]["3"]
    provenance = summary["provenance"]
    del provenance["selected_checkpoints"]["3"]
    del provenance["deliverables"]["3"]
    provenance["target_deliverable"] = "2"
    provenance["primary_deliverable"] = "2"
    provenance["selected_model_path"] = provenance["selected_checkpoints"]["2"]["model_path"]
    provenance["model_hash"] = provenance["selected_checkpoints"]["2"]["model_hash"]
    summary["final_avg_reward"] = summary["stages"]["2"]["final_eval_reward"]
    summary["total_timesteps"] = sum(int(stage["timesteps"]) for stage in summary["stages"].values())
    summary["total_training_time_seconds"] = round(
        sum(float(stage.get("training_time_seconds") or 0.0) for stage in summary["stages"].values()), 1
    )
    return summary


def _ancestor_record(**overrides: Any) -> dict[str, Any]:
    record = {
        "run_id": "trunk-run",
        "model_hash": "sha256:" + "e" * 64,
        "normalization_hash": "sha256:" + "f" * 64,
        "gate_kind": "reward_and_length/v1",
        "passed": True,
        "task_sha256": "sha256:" + "7" * 64,
    }
    record.update(overrides)
    return record


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: r.__setitem__("extra", True), "must carry exactly the fields"),
        (lambda r: r.pop("replication"), "must carry exactly the fields"),
        (lambda r: r["replication"].__setitem__("count", 2), "replication.runs must be a list of 2 run records"),
        (lambda r: r["replication"].__setitem__("runs", []), "replication.runs must be a list of 1 run records"),
        (lambda r: r["replication"]["runs"][0].__setitem__("extra", 1), "must carry exactly run_id and training_seed"),
        (
            lambda r: r["replication"]["runs"][0].__setitem__("training_seed", -1),
            "training_seed must be a non-negative integer",
        ),
        (lambda r: r.__setitem__("certified", "yes"), "certified must be a boolean"),
        (lambda r: r.__setitem__("model_path", "/abs/best_model.zip"), "must be a normalized relative POSIX path"),
        (lambda r: r.__setitem__("model_hash", "sha256:" + "0" * 64), "does not match provenance.selected_checkpoints"),
        (
            lambda r: r.__setitem__("normalization_hash", "sha256:" + "1" * 64),
            "does not match provenance.selected_checkpoints",
        ),
        (lambda r: r.__setitem__("normalization_hash", None), "normalization_hash must be sha256"),
    ],
    ids=[
        "extra-field",
        "missing-field",
        "replication-count-mismatch",
        "replication-runs-empty",
        "run-record-extra-key",
        "negative-seed",
        "certified-not-bool",
        "absolute-model-path",
        "model-hash-disagrees",
        "normalization-hash-disagrees",
        "sb3-normalization-hash-null",
    ],
)
def test_v4_deliverable_record_shape_is_fail_closed(mutate, message: str) -> None:
    summary = _canonical_summary_v4()
    mutate(summary["provenance"]["deliverables"]["3"])
    with pytest.raises(ResultSchemaError, match=message):
        _canonical_publishable(summary)


def test_v4_rejects_a_deliverable_key_the_manifest_does_not_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under a synthesized manifest only the last advancing node is a deliverable."""
    configs = tmp_path / "configs"
    shutil.copytree(stage_manifest._CONFIGS_DIR / "velociraptor", configs / "velociraptor")
    (configs / "velociraptor" / "stages.toml").unlink()
    monkeypatch.setattr(stage_manifest, "_CONFIGS_DIR", configs)
    assert [entry.id for entry in load_stage_manifest("velociraptor").deliverables] == ["behavior"]

    summary = _canonical_summary_v4()
    with pytest.raises(ResultSchemaError, match="does not flag as a deliverable"):
        _canonical_publishable(summary)

    for key in ("1", "2"):
        del summary["provenance"]["deliverables"][key]
    _canonical_publishable(summary)


def test_v4_certified_deliverable_needs_its_chain_present() -> None:
    """A certified walk with no stance in the summary and no ancestor record is refused."""
    summary = _walk_only_v4()
    del summary["stages"]["1"]
    provenance = summary["provenance"]
    del provenance["selected_checkpoints"]["1"]
    del provenance["deliverables"]["1"]
    summary["total_timesteps"] = sum(int(stage["timesteps"]) for stage in summary["stages"].values())
    summary["total_training_time_seconds"] = round(
        sum(float(stage.get("training_time_seconds") or 0.0) for stage in summary["stages"].values()), 1
    )
    with pytest.raises(ResultSchemaError, match="no selected checkpoint or ancestor record for its chain ancestors"):
        _canonical_publishable(summary)

    # An ancestor record for stance satisfies the chain ...
    provenance["ancestors"] = {"1": _ancestor_record()}
    _canonical_publishable(summary)

    # ... unless the record did not pass, in which case the certified flag is laundering.
    provenance["ancestors"]["1"]["passed"] = False
    with pytest.raises(ResultSchemaError, match="recorded as certified, but its recorded verdicts"):
        _canonical_publishable(summary)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: r.pop("task_sha256"), "must carry exactly the fields"),
        (lambda r: r.__setitem__("judged_by", "x"), "must carry exactly the fields"),
        (lambda r: r.__setitem__("passed", "yes"), "passed must be a boolean"),
        (lambda r: r.__setitem__("model_hash", "not-a-digest"), "model_hash must be sha256"),
        (lambda r: r.__setitem__("run_id", ""), "run_id must be a non-empty string"),
    ],
    ids=["missing-field", "extra-field", "passed-not-bool", "bad-hash", "empty-run-id"],
)
def test_v4_ancestor_record_shape_is_fail_closed(mutate, message: str) -> None:
    summary = _walk_only_v4()
    del summary["stages"]["1"]
    provenance = summary["provenance"]
    del provenance["selected_checkpoints"]["1"]
    del provenance["deliverables"]["1"]
    summary["total_timesteps"] = sum(int(stage["timesteps"]) for stage in summary["stages"].values())
    summary["total_training_time_seconds"] = round(
        sum(float(stage.get("training_time_seconds") or 0.0) for stage in summary["stages"].values()), 1
    )
    provenance["ancestors"] = {"1": _ancestor_record()}
    _canonical_publishable(summary)
    mutate(provenance["ancestors"]["1"])
    with pytest.raises(ResultSchemaError, match=message):
        _canonical_publishable(summary)


def test_v4_a_stage_cannot_be_both_trained_and_a_reused_ancestor() -> None:
    summary = _canonical_summary_v4()
    summary["provenance"]["ancestors"] = {"1": _ancestor_record()}
    with pytest.raises(ResultSchemaError, match="cannot be both trained here and reused"):
        _canonical_publishable(summary)
