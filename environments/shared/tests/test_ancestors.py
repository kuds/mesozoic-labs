"""Tests for environments.shared.ancestors — certified-ancestor reuse (BEHAVIOR_RECIPES_PLAN §4.2).

The reuse rule is fail-closed on five independent checks (invariant 6): a
candidate whose gate did not pass, whose plant identity mismatches, or whose
recorded task hash differs from the current stage config is refused, as is a
checkpoint rewritten after judging.  The record a child run keeps under
``ancestors/<stage_id>/`` holds JSON sidecars only — never a checkpoint.

SB3-free by construction: checkpoints are ``_sb3_style_zip`` archives (a JSON
``data`` member beside fake weights), the idiom ``test_config.py`` uses.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from environments.shared.ancestors import (
    ANCESTOR_COPIED_FILES,
    AncestorReuseError,
    CertifiedAncestor,
    find_certified_ancestor,
    record_ancestor,
    run_id_for,
)
from environments.shared.plant_contract import MODEL_IDENTITY_ATTRIBUTE
from environments.shared.result_bundle import (
    ANCESTOR_RECORD_NAME,
    ANCESTOR_RECORD_SCHEMA,
    ANCESTORS_DIRNAME,
    read_gate_verdict,
    sha256_file,
    write_gate_verdict,
)
from environments.shared.stage_manifest import load_stage_manifest
from environments.shared.task_fingerprint import (
    MODEL_TASK_ATTRIBUTE,
    TASK_FINGERPRINT_SCHEMA,
    TASK_FINGERPRINT_SCHEMA_V1,
)

from .reporting_helpers import make_plant_identity

#: The task digest the trunk fixture's stance node was judged under; the
#: curriculum tests derive the same digest for stage 1 so reuse can match.
STANCE_TASK = "sha256:" + "1" * 64
OTHER_TASK = "sha256:" + "2" * 64
JUDGED_BY = "reporting.stage_artifacts.generate_stage_artifacts"


def _sb3_style_zip(path: Path, data: dict[str, Any]) -> Path:
    """An SB3 checkpoint archive's shape: a JSON ``data`` member beside the weights."""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data", json.dumps(data))
        archive.writestr("policy.pth", b"weights")
    return path


def trunk_plant():
    """The plant every trunk fixture is tagged with (a fake trex)."""
    return make_plant_identity(species="trex", model_path="environments/trex/assets/trex.xml")


def build_trunk_run(
    run_dir: Path,
    *,
    stage_dirname: str = "01_stance",
    stage: "int | str" = 1,
    stage_id: str = "stance",
    task_sha256: str = STANCE_TASK,
    handoff: str = "robust_best_model",
    plant=None,
    tag_identity: bool = True,
    fingerprint_schema: str = TASK_FINGERPRINT_SCHEMA,
    curriculum: "dict[str, Any] | None" = None,
    verdict: bool = True,
    passed: bool = True,
) -> Path:
    """A stage directory shaped like a judged run's: handoff pair, sidecars, verdict."""
    plant = plant or trunk_plant()
    stage_dir = run_dir / stage_dirname
    models = stage_dir / "models"
    models.mkdir(parents=True, exist_ok=True)
    fingerprint = {
        "schema": fingerprint_schema,
        "species": "trex",
        "stage": stage,
        "backend": "stable-baselines3",
        "task_sha256": task_sha256,
    }
    data: dict[str, Any] = {MODEL_TASK_ATTRIBUTE: fingerprint}
    if tag_identity:
        data[MODEL_IDENTITY_ATTRIBUTE] = plant.to_dict()
    zip_path = _sb3_style_zip(models / f"{handoff}.zip", data)
    vecnorm = models / f"{handoff}_vecnorm.pkl"
    vecnorm.write_bytes(b"vecnorm-stats")
    (stage_dir / "task_fingerprint.json").write_text(json.dumps(fingerprint, indent=2) + "\n", encoding="utf-8")
    curriculum = curriculum if curriculum is not None else {"gate_kind": "stance_quality/v1", "gate_schema_version": 1}
    (stage_dir / "stage_config.json").write_text(
        json.dumps(
            {
                "species": "trex",
                "stage": stage,
                "name": stage_id,
                "description": "",
                "reward_weights": {"forward_vel_weight": 0.0},
                "curriculum": curriculum,
                "task_fingerprint": fingerprint,
                "plant_identity": plant.to_dict(),
                "run": {"seed": 1, "timesteps": 1000},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (stage_dir / "plant_identity.json").write_text(json.dumps(plant.to_dict(), indent=2) + "\n", encoding="utf-8")
    if verdict:
        write_gate_verdict(
            stage_dir,
            species="trex",
            stage=stage,
            stage_id=stage_id,
            gate_kind=curriculum.get("gate_kind", "stance_quality/v1"),
            gate_schema_version=curriculum.get("gate_schema_version", 1),
            passed=passed,
            failures=[] if passed else ["unsupported_duty_ucb 0.2153 > 0.0200"],
            task_sha256=task_sha256,
            judged_by=JUDGED_BY,
            checkpoint=zip_path,
            normalization=vecnorm,
        )
    return stage_dir


@pytest.fixture
def trunk_run(tmp_path):
    run = tmp_path / "20260901_120000"
    run.mkdir()
    build_trunk_run(run)
    return run


def _find(run_dir, **overrides):
    kwargs: dict[str, Any] = dict(
        species="trex",
        entry=load_stage_manifest("trex").by_id("stance"),
        current_task_sha256=STANCE_TASK,
        plant_identity=trunk_plant(),
    )
    kwargs.update(overrides)
    return find_certified_ancestor(run_dir, **kwargs)


class TestFindCertifiedAncestor:
    @pytest.mark.parametrize("dirname", ["01_stance", "stage1", "stance"])
    def test_a_certified_trunk_node_is_found_across_every_directory_layout(self, tmp_path, dirname):
        run = tmp_path / "run"
        run.mkdir()
        stage_dir = build_trunk_run(run, stage_dirname=dirname)

        ancestor = _find(run)

        assert isinstance(ancestor, CertifiedAncestor)
        assert ancestor.stage_dir == stage_dir
        assert ancestor.stage_id == "stance" and ancestor.stage_key == "1"
        assert ancestor.run_id == "run"
        assert ancestor.handoff_name == "robust_best_model"
        assert ancestor.model_stem == str(stage_dir / "models" / "robust_best_model")
        assert ancestor.model_zip == stage_dir / "models" / "robust_best_model.zip"
        assert ancestor.model_sha256 == sha256_file(ancestor.model_zip)
        assert ancestor.normalization_path == stage_dir / "models" / "robust_best_model_vecnorm.pkl"
        assert ancestor.normalization_sha256 == sha256_file(ancestor.normalization_path)
        assert ancestor.task_sha256 == STANCE_TASK
        assert ancestor.judged_by == JUDGED_BY
        assert ancestor.verdict == read_gate_verdict(stage_dir)

    def test_refuses_a_missing_stage_directory(self, tmp_path):
        with pytest.raises(AncestorReuseError, match="no stage directory for 'stance'"):
            _find(tmp_path)
        with pytest.raises(AncestorReuseError, match="not a run directory"):
            _find(tmp_path / "missing")

    def test_refuses_a_failed_verdict(self, tmp_path):
        build_trunk_run(tmp_path, passed=False)
        with pytest.raises(AncestorReuseError, match="FAILED gate.*unsupported_duty_ucb"):
            _find(tmp_path)

    def test_refuses_a_missing_verdict(self, tmp_path):
        build_trunk_run(tmp_path, verdict=False)
        with pytest.raises(AncestorReuseError, match="no gate_verdict.json"):
            _find(tmp_path)

    def test_refuses_a_malformed_verdict(self, tmp_path):
        stage_dir = build_trunk_run(tmp_path)
        (stage_dir / "gate_verdict.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(AncestorReuseError, match="not readable JSON"):
            _find(tmp_path)

    def test_refuses_a_null_checkpoint_hash(self, tmp_path):
        stage_dir = build_trunk_run(tmp_path)
        write_gate_verdict(
            stage_dir,
            species="trex",
            stage=1,
            stage_id="stance",
            gate_kind="stance_quality/v1",
            gate_schema_version=1,
            passed=True,
            failures=[],
            task_sha256=STANCE_TASK,
            judged_by=JUDGED_BY,
            checkpoint=None,
            normalization=None,
        )
        with pytest.raises(AncestorReuseError, match="no complete handoff pair"):
            _find(tmp_path)

    def test_refuses_a_verdict_for_another_node(self, tmp_path):
        build_trunk_run(tmp_path, stage_id="recovery")
        with pytest.raises(AncestorReuseError, match="judged stage 'recovery', not 'stance'"):
            _find(tmp_path)

    def test_refuses_a_task_hash_that_differs_from_the_current_config(self, tmp_path):
        build_trunk_run(tmp_path)
        with pytest.raises(AncestorReuseError, match="judged under task"):
            _find(tmp_path, current_task_sha256=OTHER_TASK)
        with pytest.raises(AncestorReuseError, match="no current task fingerprint"):
            _find(tmp_path, current_task_sha256=None)

    def test_refuses_when_the_directory_record_disagrees_with_the_verdict(self, tmp_path):
        stage_dir = build_trunk_run(tmp_path)
        (stage_dir / "task_fingerprint.json").write_text(
            json.dumps({"task_sha256": OTHER_TASK}) + "\n", encoding="utf-8"
        )
        with pytest.raises(AncestorReuseError, match="disagrees with the current task"):
            _find(tmp_path)

    def test_refuses_a_checkpoint_rewritten_after_judging(self, tmp_path):
        stage_dir = build_trunk_run(tmp_path)
        _sb3_style_zip(
            stage_dir / "models" / "robust_best_model.zip",
            {MODEL_TASK_ATTRIBUTE: {"task_sha256": STANCE_TASK}, "rewritten": True},
        )
        with pytest.raises(AncestorReuseError, match="rewritten after judging"):
            _find(tmp_path)

    def test_refuses_a_sidecar_rewritten_after_judging(self, tmp_path):
        stage_dir = build_trunk_run(tmp_path)
        (stage_dir / "models" / "robust_best_model_vecnorm.pkl").write_bytes(b"other-stats")
        with pytest.raises(AncestorReuseError, match="VecNormalize sidecar was rewritten"):
            _find(tmp_path)

    def test_refuses_a_plant_mismatch(self, tmp_path):
        build_trunk_run(tmp_path)
        other_plant = make_plant_identity(
            species="trex", model_path="environments/trex/assets/trex.xml", physics_sha256="sha256:" + "9" * 64
        )
        with pytest.raises(AncestorReuseError, match="incompatible with the current trex plant"):
            _find(tmp_path, plant_identity=other_plant)

    def test_refuses_an_untagged_legacy_checkpoint(self, tmp_path):
        build_trunk_run(tmp_path, tag_identity=False)
        with pytest.raises(AncestorReuseError, match="has no plant identity"):
            _find(tmp_path)

    def test_refuses_when_the_handoff_pair_is_incomplete(self, tmp_path):
        stage_dir = build_trunk_run(tmp_path)
        (stage_dir / "models" / "robust_best_model_vecnorm.pkl").unlink()
        with pytest.raises(AncestorReuseError, match="no complete handoff pair"):
            _find(tmp_path)

    def test_a_v1_schema_fingerprint_is_not_valved_for_reuse(self, tmp_path):
        """Invariant 6: reuse compares task digests exactly; the dated schema-v1
        valve resume_same_stage applies is deliberately not extended here."""
        v1_digest = "sha256:" + "f" * 64
        build_trunk_run(tmp_path, fingerprint_schema=TASK_FINGERPRINT_SCHEMA_V1, task_sha256=v1_digest)
        with pytest.raises(AncestorReuseError, match="schema-v1 fingerprint valve does not apply"):
            _find(tmp_path, current_task_sha256=STANCE_TASK)

    def test_a_reuse_is_refused_before_the_handoff_is_hashed_when_the_verdict_fails(self, tmp_path):
        # Rule order: the verdict is judged before any file is hashed, so a
        # failed node with a missing sidecar is reported as failed, not as
        # an incomplete pair.
        stage_dir = build_trunk_run(tmp_path, passed=False)
        (stage_dir / "models" / "robust_best_model_vecnorm.pkl").unlink()
        with pytest.raises(AncestorReuseError, match="FAILED gate"):
            _find(tmp_path)


class TestRunIdFor:
    def test_parent_run_id_prefers_provenance_json_over_the_directory_name(self, tmp_path):
        run = tmp_path / "20260901_120000"
        run.mkdir()
        assert run_id_for(run) == "20260901_120000"
        (run / "provenance.json").write_text(json.dumps({"run_id": "trex-sb3-ppo-abc123"}), encoding="utf-8")
        assert run_id_for(run) == "trex-sb3-ppo-abc123"
        build_trunk_run(run)
        assert _find(run).run_id == "trex-sb3-ppo-abc123"

    def test_a_provenance_without_a_run_id_is_an_error_not_a_guess(self, tmp_path):
        run = tmp_path / "run"
        run.mkdir()
        (run / "provenance.json").write_text(json.dumps({"species": "trex"}), encoding="utf-8")
        with pytest.raises(AncestorReuseError, match="records no run_id"):
            run_id_for(run)
        (run / "provenance.json").write_text("{broken", encoding="utf-8")
        with pytest.raises(AncestorReuseError, match="unreadable"):
            run_id_for(run)


class TestRecordAncestor:
    def test_copies_the_records_and_never_the_checkpoint(self, trunk_run, tmp_path):
        child = tmp_path / "child"
        child.mkdir()
        ancestor = _find(trunk_run)

        target = record_ancestor(child, ancestor)

        assert target == child / ANCESTORS_DIRNAME / "stance"
        assert sorted(p.name for p in target.iterdir()) == sorted((ANCESTOR_RECORD_NAME, *ANCESTOR_COPIED_FILES))
        assert not [p for p in (child / ANCESTORS_DIRNAME).rglob("*") if p.suffix in {".zip", ".pkl"}]
        for name in ANCESTOR_COPIED_FILES:
            assert (target / name).read_bytes() == (ancestor.stage_dir / name).read_bytes()
        record = json.loads((target / ANCESTOR_RECORD_NAME).read_text(encoding="utf-8"))
        assert record["schema"] == ANCESTOR_RECORD_SCHEMA
        assert record["stage_id"] == "stance" and record["stage_key"] == "1"
        assert record["parent_run_id"] == "20260901_120000"
        assert record["source_run_dir"] == str(trunk_run)
        assert record["source_stage_dir"] == str(ancestor.stage_dir)
        assert record["handoff"] == {
            "name": "robust_best_model",
            "model_path": str(ancestor.model_zip.resolve()),
            "model_sha256": ancestor.model_sha256,
            "normalization_path": str(ancestor.normalization_path.resolve()),
            "normalization_sha256": ancestor.normalization_sha256,
        }
        assert record["task_sha256"] == STANCE_TASK
        assert record["judged_by"] == JUDGED_BY
        assert record["reused_at"].endswith("+00:00")
        # The copied verdict is the one the reuse rule accepted.
        assert read_gate_verdict(target) == ancestor.verdict

    def test_is_idempotent(self, trunk_run, tmp_path):
        child = tmp_path / "child"
        ancestor = _find(trunk_run)
        first = record_ancestor(child, ancestor)
        before = (first / ANCESTOR_RECORD_NAME).read_text(encoding="utf-8")
        assert record_ancestor(child, ancestor) == first
        assert (first / ANCESTOR_RECORD_NAME).read_text(encoding="utf-8") == before

    def test_a_conflicting_record_is_an_error(self, trunk_run, tmp_path):
        child = tmp_path / "child"
        ancestor = _find(trunk_run)
        target = record_ancestor(child, ancestor)
        record = json.loads((target / ANCESTOR_RECORD_NAME).read_text(encoding="utf-8"))
        record["handoff"]["model_sha256"] = OTHER_TASK
        (target / ANCESTOR_RECORD_NAME).write_text(json.dumps(record), encoding="utf-8")
        with pytest.raises(AncestorReuseError, match="different ancestor for 'stance'"):
            record_ancestor(child, ancestor)

    def test_the_record_is_written_last_and_only_when_the_copies_exist(self, trunk_run, tmp_path):
        child = tmp_path / "child"
        ancestor = _find(trunk_run)
        (ancestor.stage_dir / "plant_identity.json").unlink()
        target = record_ancestor(child, ancestor)
        # An absent optional sidecar is simply not copied; the record is still complete.
        assert not (target / "plant_identity.json").exists()
        assert (target / "gate_verdict.json").is_file() and (target / ANCESTOR_RECORD_NAME).is_file()
