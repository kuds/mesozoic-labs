"""Result schema v4: publication per deliverable (BEHAVIOR_RECIPES_PLAN §4.3, Phase A).

The rules the bundle writer, the summary and the audit share: which
deliverables a run certified (its own gate plus every ``warm_start_from``
ancestor's), which one is published (the target when certified, else the
deepest certified), and the target-aware bundle status.

Schema v5 (decision D-D33) carries the v4 rules unchanged and adds
``embodiment`` to the canonical runtime provenance; the reader accepts schema 5
alone (decision D-D63 retired schema 2), so the summaries here are written at
v5 (the writers' version) and the ``test_v4_*`` names keep the rule set they
pin.

Since decision D-D59 every committed chain ends at walk (stance -> locomotion,
with recovery a sibling branch off stance on the T-Rex), so the leaf whose
failure the rules are pinned on is the walk, ``"2"``, where it was the hunt,
``"3"``.
"""

from __future__ import annotations

import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from environments.shared import stage_manifest
from environments.shared.model_id import ModelId
from environments.shared.result_schema import (
    CANONICAL_RUNTIME_PROVENANCE_FIELDS,
    DELIVERABLE_RECORD_FIELDS,
    OPTIONAL_DELIVERABLE_RECORD_FIELDS,
    RESULT_SCHEMA_VERSION,
    ResultSchemaError,
    bundle_status_for,
    certified_deliverables,
    primary_deliverable_key,
    validate_result_summary,
)
from environments.shared.stage_manifest import load_stage_manifest

from .result_bundle_helpers import _canonical_summary

VELOCIRAPTOR_MONGOLIENSIS = ModelId("velociraptor_mongoliensis", "anatomical")
TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")


def _entries(model_id: ModelId, *ids: str) -> list[tuple[str, Any]]:
    manifest = load_stage_manifest(model_id)
    return [(entry.key, entry) for entry in manifest.stages if not ids or entry.id in ids]


# ── The shared rules ─────────────────────────────────────────────────────


def test_certified_requires_the_whole_chain() -> None:
    # The Tyrannosaurus rex: stance -> recovery, stance -> locomotion; every node a deliverable.
    entries = _entries(TYRANNOSAURUS_REX)
    certified = certified_deliverables(
        entries,
        {"1": True, "recovery": False, "2": True},
        None,
        model_id=TYRANNOSAURUS_REX,
    )
    assert certified == {"1": True, "recovery": False, "2": True}
    # A failed root certifies nothing below it, whatever the children's own verdicts.
    certified = certified_deliverables(
        entries,
        {"1": False, "recovery": True, "2": True},
        None,
        model_id=TYRANNOSAURUS_REX,
    )
    assert certified == {"1": False, "recovery": False, "2": False}
    # Records carrying the verdict (a summary's stage rows) read the same as bools.
    certified = certified_deliverables(
        entries,
        {
            "1": {"stage_passed": True},
            "recovery": {"stage_passed": False},
            "2": {"stage_passed": True},
        },
        None,
        model_id=TYRANNOSAURUS_REX,
    )
    assert certified == {"1": True, "recovery": False, "2": True}


def test_an_absent_ancestor_never_reads_as_passed() -> None:
    # Walk recorded without stance: its chain is incomplete.
    entries = _entries(VELOCIRAPTOR_MONGOLIENSIS, "locomotion")
    assert certified_deliverables(entries, {"2": True}, None, model_id=VELOCIRAPTOR_MONGOLIENSIS) == {"2": False}


def test_ancestor_records_complete_a_chain() -> None:
    entries = _entries(VELOCIRAPTOR_MONGOLIENSIS, "locomotion")
    stages = {"2": True}
    assert certified_deliverables(entries, stages, {"1": {"passed": True}}, model_id=VELOCIRAPTOR_MONGOLIENSIS) == {
        "2": True,
    }
    assert certified_deliverables(entries, stages, {"1": {"passed": False}}, model_id=VELOCIRAPTOR_MONGOLIENSIS) == {
        "2": False,
    }
    # A stage cannot be both trained here and reused from another run.
    with pytest.raises(ResultSchemaError, match="both as a trained stage and as a reused ancestor"):
        certified_deliverables(
            _entries(VELOCIRAPTOR_MONGOLIENSIS),
            {"1": True, "2": True},
            {"1": True},
            model_id=VELOCIRAPTOR_MONGOLIENSIS,
        )


def test_primary_is_the_target_when_certified_else_the_deepest_certified() -> None:
    deliverables = {"1": True, "recovery": False, "2": False}
    # The walk target failed: stance, the deepest certified, is published.
    assert primary_deliverable_key(deliverables, model_id=TYRANNOSAURUS_REX, target="2") == "1"
    assert primary_deliverable_key(deliverables, model_id=TYRANNOSAURUS_REX, target=2) == "1"
    # Walk targeted and certified: walk.
    assert primary_deliverable_key({"1": True, "2": True}, model_id=TYRANNOSAURUS_REX, target="2") == "2"
    # Only the stance and recover recipes certified: recovery is deeper than stance along its chain.
    assert (
        primary_deliverable_key({"1": True, "recovery": True, "2": False}, model_id=TYRANNOSAURUS_REX, target=2)
        == "recovery"
    )
    # Nothing certified: no primary, so no summary.
    assert primary_deliverable_key({"1": False, "2": False}, model_id=TYRANNOSAURUS_REX, target=2) is None


def test_bundle_status_is_target_aware() -> None:
    assert bundle_status_for({"1": True, "2": True}, model_id=VELOCIRAPTOR_MONGOLIENSIS, target="2") == "complete"
    # A stance-only run aimed at walk is partial: its target is absent.
    assert bundle_status_for({"1": True}, model_id=VELOCIRAPTOR_MONGOLIENSIS, target="2") == "partial"
    assert bundle_status_for({"1": True}, model_id=VELOCIRAPTOR_MONGOLIENSIS, target="1") == "complete"
    assert bundle_status_for({"1": True, "2": True}, model_id=VELOCIRAPTOR_MONGOLIENSIS, target=None) == "partial"
    assert bundle_status_for({"1": True, "2": False}, model_id=VELOCIRAPTOR_MONGOLIENSIS, target="2") == "partial"
    assert (
        bundle_status_for(
            {"1": False, "2": False},
            model_id=VELOCIRAPTOR_MONGOLIENSIS,
            target="2",
        )
        == "failed"
    )


def test_a_run_with_no_deliverable_yet_but_every_stage_passing_is_partial() -> None:
    # The pre-Phase-A rule under a v1 / synthesized manifest (one deliverable,
    # the last advancing node): a passing stance-only run is partial, not failed.
    assert bundle_status_for({}, model_id=VELOCIRAPTOR_MONGOLIENSIS, target="2", stages={"1": True}) == "partial"
    assert bundle_status_for({}, model_id=VELOCIRAPTOR_MONGOLIENSIS, target="2", stages={"1": False}) == "failed"


# ── The v4 canonical summary, written at v5 ──────────────────────────────


def _canonical_summary_v5() -> dict[str, Any]:
    """The canonical summary under the deliverable rules, at the writers' schema (v4 before decision D-D33): every
    stage a certified deliverable on the checkpoint it selected, walk the target and the primary."""
    summary = _canonical_summary()
    assert summary["schema_version"] == RESULT_SCHEMA_VERSION
    provenance = summary["provenance"]
    for key, checkpoint in provenance["selected_checkpoints"].items():
        record = provenance["deliverables"][key]
        assert [record[field] for field in ("model_path", "model_hash", "normalization_hash")] == [
            checkpoint[field] for field in ("model_path", "model_hash", "normalization_hash")
        ]
        assert record["certified"] is True and record["replication"]["runs"] == [
            {"run_id": provenance["run_id"], "training_seed": summary["seed"]}
        ]
    assert (provenance["primary_deliverable"], provenance["target_deliverable"]) == ("2", "2")
    return summary


def _fail_stage(summary: dict[str, Any], key: str) -> None:
    summary["stages"][key]["stage_passed"] = False
    summary["stages"][key]["publication_gate_passed"] = False


def test_v4_canonical_summary_validates_complete() -> None:
    summary = _canonical_summary_v5()
    validate_result_summary(
        summary,
        expected_model_id=VELOCIRAPTOR_MONGOLIENSIS,
        require_complete=True,
        canonical_provenance=True,
    )
    validate_result_summary(
        summary,
        expected_model_id=VELOCIRAPTOR_MONGOLIENSIS,
        require_complete=False,
        require_publishable=True,
    )


def test_the_canonical_runtime_fields_are_schema_5s() -> None:
    """One tuple, schema 5's (decision D-D63 retired schema 2, whose v3 tuple it was built on): schema 3's fields with
    ``embodiment`` after ``species`` (decision D-D33), then the deliverable keys schema 4 added."""
    assert RESULT_SCHEMA_VERSION == 5
    assert CANONICAL_RUNTIME_PROVENANCE_FIELDS == (
        "run_id",
        "captured_at",
        "species",
        "embodiment",
        "algorithm",
        "backend",
        "backend_version",
        "repository_dirty",
        "repository_patch_sha256",
        "training_seed",
        "seed_roles",
        "evaluation_protocols",
        "evaluation_seeds",
        "parallel_envs",
        "hardware",
        "python_version",
        "platform",
        "dependency_versions",
        "plant_identity",
        "selected_checkpoints",
        "selected_model_path",
        "deliverables",
        "primary_deliverable",
        "target_deliverable",
    )


@pytest.mark.parametrize("field", ["deliverables", "primary_deliverable", "target_deliverable", "embodiment"])
def test_v4_canonical_provenance_requires_deliverables_primary_and_target(field: str) -> None:
    summary = _canonical_summary_v5()
    del summary["provenance"][field]

    with pytest.raises(ResultSchemaError, match=rf"canonical provenance.*missing fields.*{field}"):
        validate_result_summary(summary, canonical_provenance=True)


def test_v4_primary_must_be_certified_never_the_failed_leaf() -> None:
    """Plan invariant 5: a failed walk publishes stance; the leaf never headlines."""
    summary = _canonical_summary_v5()
    _fail_stage(summary, "2")
    provenance = summary["provenance"]
    provenance["deliverables"]["2"]["certified"] = False
    provenance["primary_deliverable"] = "1"
    provenance["selected_model_path"] = provenance["selected_checkpoints"]["1"]["model_path"]
    provenance["model_hash"] = provenance["selected_checkpoints"]["1"]["model_hash"]
    summary["bundle_status"] = "partial"
    summary["final_avg_reward"] = summary["stages"]["1"]["final_eval_reward"]

    validate_result_summary(
        summary,
        expected_model_id=VELOCIRAPTOR_MONGOLIENSIS,
        require_complete=False,
        require_publishable=True,
        canonical_provenance=True,
    )
    with pytest.raises(ResultSchemaError, match="bundle_status 'partial'"):
        validate_result_summary(summary, expected_model_id=VELOCIRAPTOR_MONGOLIENSIS, require_complete=True)

    # Naming the failed leaf as the primary is refused.
    laundered = deepcopy(summary)
    laundered["provenance"]["primary_deliverable"] = "2"
    with pytest.raises(ResultSchemaError, match="primary_deliverable.*must be '1'"):
        validate_result_summary(laundered, require_complete=False, require_publishable=True, canonical_provenance=True)

    # So is headlining the leaf's checkpoint.
    headlined = deepcopy(summary)
    headlined["provenance"]["selected_model_path"] = provenance["selected_checkpoints"]["2"]["model_path"]
    with pytest.raises(ResultSchemaError, match="selected_model_path.*primary deliverable"):
        validate_result_summary(headlined, require_complete=False, require_publishable=True, canonical_provenance=True)


def test_v4_certified_flag_cannot_be_laundered() -> None:
    summary = _canonical_summary_v5()
    _fail_stage(summary, "2")
    # The leaf failed but is recorded as certified.
    with pytest.raises(ResultSchemaError, match="recorded as certified, but its recorded verdicts"):
        validate_result_summary(summary, require_complete=False, require_publishable=True)

    # The mirror image: a node whose whole chain passed cannot be hidden either.
    summary = _canonical_summary_v5()
    summary["provenance"]["deliverables"]["2"]["certified"] = False
    with pytest.raises(ResultSchemaError, match="recorded as uncertified although"):
        validate_result_summary(summary, require_complete=False, require_publishable=True)


def test_v4_bundle_status_must_match_the_deliverables_and_target() -> None:
    summary = _canonical_summary_v5()
    summary["bundle_status"] = "partial"
    with pytest.raises(ResultSchemaError, match="bundle_status.*must be 'complete'"):
        validate_result_summary(summary, require_complete=False, require_publishable=True)

    summary = _canonical_summary_v5()
    summary["bundle_status"] = "certified"
    with pytest.raises(ResultSchemaError, match="bundle_status.*must be one of"):
        validate_result_summary(summary, require_complete=False)


def test_v4_summary_with_no_certified_deliverable_is_not_publishable() -> None:
    summary = _canonical_summary_v5()
    for key in ("1", "2"):
        _fail_stage(summary, key)
        summary["provenance"]["deliverables"][key]["certified"] = False
    summary["provenance"]["primary_deliverable"] = None
    summary["bundle_status"] = "failed"

    # A non-publishable stage subset still validates as such.
    validate_result_summary(summary, expected_model_id=VELOCIRAPTOR_MONGOLIENSIS, require_complete=False)
    with pytest.raises(ResultSchemaError, match="records no certified deliverable"):
        validate_result_summary(summary, require_complete=False, require_publishable=True)
    with pytest.raises(ResultSchemaError):
        validate_result_summary(summary, require_complete=True)


def test_v4_deliverables_must_record_every_deliverable_stage_the_summary_records() -> None:
    summary = _canonical_summary_v5()
    del summary["provenance"]["deliverables"]["2"]
    with pytest.raises(ResultSchemaError, match="must record every deliverable stage.*missing \\['2'\\]"):
        validate_result_summary(summary, require_complete=False, require_publishable=True)


def test_v4_stance_only_summary_is_complete_when_stance_is_the_target() -> None:
    """The walk-only case of the three-stage chains before decision D-D59, one node shorter: the stance stage alone
    is a complete bundle when stance is the target, and partial when the same stage was aimed at walk."""
    summary = _stance_only_v4()

    validate_result_summary(
        summary,
        expected_model_id=VELOCIRAPTOR_MONGOLIENSIS,
        require_complete=True,
        canonical_provenance=True,
    )

    # The same stage aimed at walk is partial: the target is absent.
    aimed_at_walk = deepcopy(summary)
    aimed_at_walk["provenance"]["target_deliverable"] = "2"
    aimed_at_walk["bundle_status"] = "partial"
    validate_result_summary(
        aimed_at_walk,
        expected_model_id=VELOCIRAPTOR_MONGOLIENSIS,
        require_complete=False,
        require_publishable=True,
        canonical_provenance=True,
    )
    with pytest.raises(ResultSchemaError, match="bundle_status 'partial'"):
        validate_result_summary(aimed_at_walk, expected_model_id=VELOCIRAPTOR_MONGOLIENSIS, require_complete=True)


def test_v4_without_publication_flags_accepts_a_summary_without_deliverables() -> None:
    # A Colab run checked after each node, before anything is publishable.
    summary = _canonical_summary_v5()
    del summary["provenance"]["deliverables"]
    del summary["provenance"]["primary_deliverable"]
    del summary["provenance"]["target_deliverable"]
    validate_result_summary(summary, expected_model_id=VELOCIRAPTOR_MONGOLIENSIS, require_complete=False)
    with pytest.raises(ResultSchemaError, match="without provenance.deliverables"):
        validate_result_summary(
            summary,
            expected_model_id=VELOCIRAPTOR_MONGOLIENSIS,
            require_complete=False,
            require_publishable=True,
        )


# ── The v4 provenance shape validators ───────────────────────────────────


def _canonical_publishable(summary: dict[str, Any]) -> dict[str, Any]:
    return validate_result_summary(
        summary,
        expected_model_id=VELOCIRAPTOR_MONGOLIENSIS,
        require_complete=False,
        require_publishable=True,
        canonical_provenance=True,
    )


def _walk_only_v4() -> dict[str, Any]:
    """The v4 canonical summary: stance and walk, walk targeted (the whole chain since decision D-D59)."""
    return _canonical_summary_v5()


def _stance_only_v4() -> dict[str, Any]:
    """The v4 canonical summary reduced to stance, stance targeted."""
    summary = _canonical_summary_v5()
    del summary["stages"]["2"]
    provenance = summary["provenance"]
    del provenance["selected_checkpoints"]["2"]
    del provenance["deliverables"]["2"]
    provenance["target_deliverable"] = "1"
    provenance["primary_deliverable"] = "1"
    provenance["selected_model_path"] = provenance["selected_checkpoints"]["1"]["model_path"]
    provenance["model_hash"] = provenance["selected_checkpoints"]["1"]["model_hash"]
    summary["bundle_status"] = "complete"
    summary["final_avg_reward"] = summary["stages"]["1"]["final_eval_reward"]
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
        # Seed replication (D-B11/D-B16): the label and the bar are typed,
        # the label must be the count's own reading of the bar, and the
        # runs are a set of distinct runs on distinct seeds.
        (lambda r: r.__setitem__("provisional", "no"), "provisional must be a boolean"),
        (lambda r: r.__setitem__("certification_seeds", 0), "certification_seeds must be a positive integer"),
        (lambda r: r.__setitem__("certification_seeds", True), "certification_seeds must be a positive integer"),
        (
            lambda r: r.update({"provisional": True, "certification_seeds": 1}),
            "provisional must equal replication.count < certification_seeds",
        ),
        (
            lambda r: r.update({"provisional": False, "certification_seeds": 2}),
            "provisional must equal replication.count < certification_seeds",
        ),
        (
            lambda r: _with_second_run(
                r, run_id="other-run", training_seed=r["replication"]["runs"][0]["training_seed"]
            ),
            "replication.runs must carry distinct run ids and training seeds",
        ),
        (
            lambda r: _with_second_run(r, run_id=r["replication"]["runs"][0]["run_id"], training_seed=4242),
            "replication.runs must carry distinct run ids and training seeds",
        ),
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
        "provisional-not-bool",
        "certification-seeds-zero",
        "certification-seeds-bool",
        "provisional-true-at-n-1",
        "provisional-false-below-n",
        "duplicate-training-seed",
        "duplicate-run-id",
    ],
)
def test_v4_deliverable_record_shape_is_fail_closed(mutate, message: str) -> None:
    summary = _canonical_summary_v5()
    mutate(summary["provenance"]["deliverables"]["2"])
    with pytest.raises(ResultSchemaError, match=message):
        _canonical_publishable(summary)


def _with_second_run(record: dict[str, Any], *, run_id: str, training_seed: int) -> None:
    first = record["replication"]["runs"][0]
    record["replication"] = {"count": 2, "runs": [first, {"run_id": run_id, "training_seed": training_seed}]}


def test_v4_replication_with_two_runs_validates() -> None:
    """Seed replication (plan §4.5, D-B16): a record may list this run plus a replicate on another seed,
    and the D-B11 pair beside it — certification_seeds 2 with the count 2 is NOT provisional."""
    summary = _canonical_summary_v5()
    record = summary["provenance"]["deliverables"]["2"]
    _with_second_run(record, run_id="velociraptor_mongoliensis-anatomical-seed-43", training_seed=43)
    record.update({"certification_seeds": 2, "provisional": False})
    summary["provenance"]["deliverables"]["1"].update({"certification_seeds": 2, "provisional": True})

    validated = _canonical_publishable(summary)

    records = validated["provenance"]["deliverables"]
    assert records["2"]["replication"]["count"] == 2
    assert records["2"]["replication"]["runs"][1] == {
        "run_id": "velociraptor_mongoliensis-anatomical-seed-43",
        "training_seed": 43,
    }
    assert (records["2"]["certification_seeds"], records["2"]["provisional"]) == (2, False)
    assert (records["1"]["certification_seeds"], records["1"]["provisional"]) == (2, True)


def test_v4_deliverable_record_optional_fields_validate_and_are_kept() -> None:
    """Decision D-A21: a deliverable record may carry ``hyperparameters_sha256`` and ``label`` —
    any subset of OPTIONAL_DELIVERABLE_RECORD_FIELDS on top of exactly the required fields — and
    the validated summary keeps them (type-checked; writers record the label already stripped).
    Decisions D-B11/D-B16 add ``provisional`` and ``certification_seeds`` to the optional set;
    ``widened_from_run_id`` marks a widened root (its training curve is the parent run's)."""
    assert OPTIONAL_DELIVERABLE_RECORD_FIELDS == (
        "hyperparameters_sha256",
        "label",
        "widened_from_run_id",
        "provisional",
        "certification_seeds",
    )
    assert not set(OPTIONAL_DELIVERABLE_RECORD_FIELDS) & set(DELIVERABLE_RECORD_FIELDS)
    digest = "sha256:" + "d" * 64
    summary = _canonical_summary_v5()
    summary["provenance"]["deliverables"]["2"].update({"hyperparameters_sha256": digest, "label": "lr-sweep-a"})
    summary["provenance"]["deliverables"]["1"]["label"] = "stance-only"

    validated = _canonical_publishable(summary)

    records = validated["provenance"]["deliverables"]
    assert records["2"]["hyperparameters_sha256"] == digest and records["2"]["label"] == "lr-sweep-a"
    assert records["1"]["label"] == "stance-only" and "hyperparameters_sha256" not in records["1"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("hyperparameters_sha256", "d" * 64, "hyperparameters_sha256 must be sha256:<64 lowercase hex>"),
        ("hyperparameters_sha256", None, "hyperparameters_sha256 must be sha256:<64 lowercase hex>"),
        ("label", "", "label must be a non-empty string"),
        ("label", 7, "label must be a non-empty string"),
    ],
    ids=["digest-bare-hex", "digest-null", "label-empty", "label-not-a-string"],
)
def test_v4_wrong_typed_optional_deliverable_fields_are_refused(field: str, value: Any, message: str) -> None:
    """Optional means absent-or-valid: a present field of the wrong shape fails closed."""
    summary = _canonical_summary_v5()
    summary["provenance"]["deliverables"]["2"][field] = value
    with pytest.raises(ResultSchemaError, match=message):
        _canonical_publishable(summary)


def test_v4_an_unknown_deliverable_field_is_still_refused_beside_the_optional_ones() -> None:
    """The optional set widens the record by exactly two names; anything else is still an error."""
    summary = _canonical_summary_v5()
    summary["provenance"]["deliverables"]["2"].update(
        {"hyperparameters_sha256": "sha256:" + "d" * 64, "label": "lr-sweep-a", "notes": "free text"}
    )
    with pytest.raises(ResultSchemaError, match="must carry exactly the fields"):
        _canonical_publishable(summary)


def test_v4_rejects_a_deliverable_key_the_manifest_does_not_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under a v1 manifest only the last advancing node is a deliverable."""
    configs = tmp_path / "configs"
    model_dir = VELOCIRAPTOR_MONGOLIENSIS.path(configs)
    shutil.copytree(VELOCIRAPTOR_MONGOLIENSIS.path(stage_manifest._CONFIGS_DIR), model_dir)
    (model_dir / "stages.toml").write_text(
        f'schema = "{stage_manifest.STAGE_MANIFEST_SCHEMA_V1}"\n'
        '[[stages]]\nid = "stance"\nconfig = "stance.toml"\nlegacy_number = 1\n'
        '[[stages]]\nid = "locomotion"\nconfig = "locomotion.toml"\nlegacy_number = 2\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(stage_manifest, "_CONFIGS_DIR", configs)
    assert [entry.id for entry in load_stage_manifest(VELOCIRAPTOR_MONGOLIENSIS).deliverables] == ["locomotion"]

    summary = _canonical_summary_v5()
    with pytest.raises(ResultSchemaError, match="does not flag as a deliverable"):
        _canonical_publishable(summary)

    del summary["provenance"]["deliverables"]["1"]
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
    summary = _canonical_summary_v5()
    summary["provenance"]["ancestors"] = {"1": _ancestor_record()}
    with pytest.raises(ResultSchemaError, match="cannot be both trained here and reused"):
        _canonical_publishable(summary)


# ── task_success/v1 stage-row keys: retired, now unvalidated (decision D-D59) ───


@pytest.mark.parametrize(
    "updates",
    [
        {"selected_model_success_lcb": 0.5006, "selected_model_success_count": 20, "selected_model_n_episodes": 30},
        # Shapes the schema refused while the gate existed: no reader consumes the keys any more, so a stage
        # row carrying them is treated like any other stage row with a key the schema does not name.
        {"selected_model_success_lcb": 1.2},
        {"selected_model_success_count": 31, "selected_model_n_episodes": 30},
    ],
)
def test_the_retired_task_success_stage_keys_are_unvalidated(updates: dict[str, Any]) -> None:
    summary = _canonical_summary()
    summary["stages"]["2"].update(updates)
    validate_result_summary(summary, canonical_provenance=True)


def test_a_canonical_stage_row_may_leave_the_success_rates_null() -> None:
    """A model that declares no success keys records no rate (decision D-D59's amendment): null, not a zero.

    The keys stay required, and every other selected or final metric must still be finite.
    """
    from environments.shared.result_schema import NULLABLE_SUCCESS_RATE_METRICS

    assert NULLABLE_SUCCESS_RATE_METRICS == {"mean_success_rate", "selected_model_success_rate"}
    summary = _canonical_summary_v5()
    for stage in summary["stages"].values():
        stage["mean_success_rate"] = None
        stage["selected_model_success_rate"] = None
    validate_result_summary(summary, expected_model_id=VELOCIRAPTOR_MONGOLIENSIS, canonical_provenance=True)

    missing = deepcopy(summary)
    del missing["stages"]["2"]["mean_success_rate"]
    with pytest.raises(ResultSchemaError, match="mean_success_rate is required for canonical"):
        validate_result_summary(missing, canonical_provenance=True)
    other = deepcopy(summary)
    other["stages"]["2"]["selected_model_distance"] = None
    with pytest.raises(ResultSchemaError, match="selected_model_distance must be a finite number for canonical"):
        validate_result_summary(other, canonical_provenance=True)


def test_v4_canonical_best_eval_reward_is_null_only_for_a_widened_root() -> None:
    """A widened root (widen_checkpoint) never trained in its run, so its training curve — the best
    EvalCallback panel ``best_eval_reward`` summarizes — is the parent run's. The 2026-09-20 widen
    session died at its first bundle write on exactly this rule (the judged stance had no
    evaluations.npz), so the exemption is keyed on the deliverable record naming the parent run."""
    summary = _canonical_summary_v5()
    for key in ("best_eval_reward", "best_eval_std", "best_eval_step"):
        summary["stages"]["1"][key] = None
    with pytest.raises(ResultSchemaError, match="best_eval_reward must be a finite number.*widened_from_run_id"):
        validate_result_summary(summary, canonical_provenance=True)

    summary["provenance"]["deliverables"]["1"]["widened_from_run_id"] = "20260815_205206"
    validate_result_summary(
        summary,
        expected_model_id=VELOCIRAPTOR_MONGOLIENSIS,
        require_complete=True,
        canonical_provenance=True,
    )

    # Per stage: a null on a stage that trained here is still missing evidence.
    trained_null = deepcopy(summary)
    trained_null["stages"]["2"]["best_eval_reward"] = None
    with pytest.raises(ResultSchemaError, match="best_eval_reward must be a finite number for canonical stage 2"):
        validate_result_summary(trained_null, canonical_provenance=True)

    # best_eval_reward alone: the judged panel and the final evaluation stay required for the widened root.
    for metric in ("final_eval_reward", "selected_model_reward"):
        unjudged = deepcopy(summary)
        unjudged["stages"]["1"][metric] = None
        with pytest.raises(ResultSchemaError, match=f"{metric} must be a finite number for canonical stage 1"):
            validate_result_summary(unjudged, canonical_provenance=True)

    # The marker is a non-empty run id; an empty one is malformed and grants nothing.
    blank = deepcopy(summary)
    blank["provenance"]["deliverables"]["1"]["widened_from_run_id"] = ""
    with pytest.raises(ResultSchemaError, match="widened_from_run_id"):
        validate_result_summary(blank, canonical_provenance=True)
