"""Tests for environments.shared.ancestors — certified-ancestor reuse (BEHAVIOR_RECIPES_PLAN §4.2).

The reuse rule is fail-closed on seven independent checks (invariant 6): a
candidate whose gate did not pass, whose plant identity mismatches, or whose
recorded task hash differs from the current stage config is refused, as is a
checkpoint rewritten after judging, a non-root candidate is reused only
on top of the very parent checkpoint it was trained from (rule 4, the
chain), and a verdict judged under another gate configuration — or before
the gate was recorded at all — is refused naming the thresholds that differ
(rule 7, decision D-A22).  The record a child run keeps under ``ancestors/<stage_id>/`` holds
JSON sidecars only — never a checkpoint.

SB3-free by construction: checkpoints are ``_sb3_style_zip`` archives (a JSON
``data`` member beside fake weights), the idiom ``test_config.py`` uses.
"""

from __future__ import annotations

import json
import logging
import shutil
import zipfile
from pathlib import Path
from typing import Any

import pytest

from environments.shared.ancestors import (
    ANCESTOR_COPIED_FILES,
    ANCESTOR_RECORD_HOP_LIMIT,
    AncestorReuseError,
    CertifiedAncestor,
    find_certified_ancestor,
    record_ancestor,
    run_id_for,
)
from environments.shared.curriculum.gate_schema import gate_config_sha256, gate_config_view
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
LOCOMOTION_TASK = "sha256:" + "3" * 64
JUDGED_BY = "reporting.stage_artifacts.generate_stage_artifacts"

#: The ``[curriculum]`` block the trunk fixture's stance node records and is
#: judged under, and the CURRENT block ``_find`` compares it against (rule
#: 7): the gate keys of ``stance_quality/v1`` plus a few keys that are not
#: the gate (schedule and collapse), which must never enter the digest.
STANCE_CURRICULUM: "dict[str, Any]" = {
    "gate_kind": "stance_quality/v1",
    "gate_schema_version": 1,
    "timesteps": 11_000_000,
    "min_full_horizon_fraction": 0.95,
    "max_unsupported_duty": 0.02,
    "max_unsupported_duty_ucb": 0.02,
    "settle_steps": 200,
    "min_eval_episodes": 40,
    "min_avg_reward": 2100.0,
    "required_consecutive": 3,
    "collapse_patience": 10,
}
#: The locomotion child's block: an unregistered kind (the fixture's, not a
#: real one), which the view projects through every known threshold key.
LOCOMOTION_CURRICULUM: "dict[str, Any]" = {
    "gate_kind": "locomotion/v1",
    "gate_schema_version": 1,
    "timesteps": 8_000_000,
    "min_avg_reward": 100.0,
    "min_avg_forward_vel": 2.0,
    "required_consecutive": 3,
}


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
    lineage: "dict[str, Any] | None" = None,
    species: str = "trex",
    judged_under: "dict[str, Any] | None" = None,
    seed: int = 1,
    algorithm: "str | None" = None,
    hyperparameters: "dict[str, Any] | None" = None,
    record_recipe_digest: bool = False,
) -> Path:
    """A stage directory shaped like a judged run's: handoff pair, sidecars, verdict.

    *lineage* is merged into the ``stage_config.json`` run block: the load
    keys ``save_stage_config`` records for a node that entered from a parent
    (``load_mode``, ``parent_checkpoint_sha256``, ...).  None is a node
    trained from scratch, which records no load keys at all.  *species*
    names the species every record is stamped with; the default plant is
    the trex fake, so pass a matching *plant* for another species.
    *curriculum* is the block the stage records AND is judged under (its
    ``gate_config_view`` is the verdict's ``gate``), :data:`STANCE_CURRICULUM`
    by default; *judged_under* judges the verdict under another block than
    the directory records (a directory re-judged after a threshold edit,
    decision D-B8).  *seed* is the run block's training seed; *algorithm*
    and *hyperparameters* record the ``"algorithm"`` / ``"hyperparameters"``
    blocks ``save_stage_config`` writes (absent by default, as before), and
    *record_recipe_digest* adds the D-A21 ``hyperparameters_sha256`` to the
    run block — the shapes replicate discovery (test_replication.py) tells
    apart.
    """
    plant = plant or trunk_plant()
    stage_dir = run_dir / stage_dirname
    models = stage_dir / "models"
    models.mkdir(parents=True, exist_ok=True)
    fingerprint = {
        "schema": fingerprint_schema,
        "species": species,
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
    curriculum = curriculum if curriculum is not None else STANCE_CURRICULUM
    stage_config: dict[str, Any] = {
        "species": species,
        "stage": stage,
        "name": stage_id,
        "description": "",
        "reward_weights": {"forward_vel_weight": 0.0},
        "curriculum": curriculum,
        "task_fingerprint": fingerprint,
        "plant_identity": plant.to_dict(),
        "run": {"seed": seed, "timesteps": 1000, **(lineage or {})},
    }
    if algorithm is not None:
        from environments.shared.config import hyperparameters_sha256

        stage_config["algorithm"] = algorithm
        stage_config["hyperparameters"] = dict(hyperparameters or {})
        if record_recipe_digest:
            stage_config["run"]["hyperparameters_sha256"] = hyperparameters_sha256(
                {f"{algorithm.lower()}_kwargs": dict(hyperparameters or {}), "curriculum_kwargs": curriculum},
                algorithm,
            )
    (stage_dir / "stage_config.json").write_text(json.dumps(stage_config, indent=2) + "\n", encoding="utf-8")
    (stage_dir / "plant_identity.json").write_text(json.dumps(plant.to_dict(), indent=2) + "\n", encoding="utf-8")
    if verdict:
        write_gate_verdict(
            stage_dir,
            species=species,
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
            gate_config=gate_config_view(judged_under if judged_under is not None else curriculum),
        )
    return stage_dir


def strip_gate_record(stage_dir: Path) -> Path:
    """Rewrite *stage_dir*'s verdict as a pre-D-A22 file: no ``gate``, no ``gate_sha256``."""
    path = stage_dir / "gate_verdict.json"
    verdict = json.loads(path.read_text(encoding="utf-8"))
    phase_a = {key: value for key, value in verdict.items() if key not in {"gate", "gate_sha256"}}
    path.write_text(json.dumps(phase_a, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


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
        current_gate_config=STANCE_CURRICULUM,
    )
    kwargs.update(overrides)
    return find_certified_ancestor(run_dir, **kwargs)


def _follow(run_dir, **overrides):
    """``_find`` for a candidate that is ANOTHER run: opts in to following its ancestor records."""
    return _find(run_dir, follow_records=True, **overrides)


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
        assert ancestor.gate_sha256 == gate_config_sha256(gate_config_view(STANCE_CURRICULUM))
        assert ancestor.gate_sha256 == ancestor.verdict["gate_sha256"]

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
            gate_config=gate_config_view(STANCE_CURRICULUM),
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

    # -- Rule 7 (decision D-A22): the gate the verdict was judged under. --

    def test_refuses_a_verdict_with_no_gate_sha256(self, tmp_path):
        """A verdict judged before D-A22 certifies an unknown gate; both re-judge paths are named."""
        strip_gate_record(build_trunk_run(tmp_path))
        with pytest.raises(AncestorReuseError, match="records no gate_sha256") as excinfo:
            _find(tmp_path)
        message = str(excinfo.value)
        assert "generate_stage_artifacts" in message
        assert "scripts/backfill_gate_verdict.py --force [--gate current]" in message

    def test_refuses_when_no_current_gate_configuration_is_given(self, tmp_path):
        build_trunk_run(tmp_path)
        with pytest.raises(AncestorReuseError, match="no current gate configuration"):
            _find(tmp_path, current_gate_config=None)
        # The keyword is REQUIRED: a caller that forgets it cannot ask at all.
        with pytest.raises(TypeError, match="current_gate_config"):
            find_certified_ancestor(  # type: ignore[call-arg]
                tmp_path,
                species="trex",
                entry=load_stage_manifest("trex").by_id("stance"),
                current_task_sha256=STANCE_TASK,
                plant_identity=trunk_plant(),
            )

    def test_refuses_a_verdict_judged_under_a_different_gate_configuration_naming_the_thresholds(self, tmp_path):
        stage_dir = build_trunk_run(tmp_path)
        recorded = read_gate_verdict(stage_dir)["gate_sha256"]
        current = {**STANCE_CURRICULUM, "max_unsupported_duty_ucb": 0.05}
        with pytest.raises(
            AncestorReuseError, match="differing thresholds: max_unsupported_duty_ucb: judged at"
        ) as excinfo:
            _find(tmp_path, current_gate_config=current)
        message = str(excinfo.value)
        assert f"was judged under gate {recorded} (stance_quality/v1)" in message
        assert (
            f"'stance' declares gate {gate_config_sha256(gate_config_view(current))} (stance_quality/v1) now" in message
        )
        assert "max_unsupported_duty_ucb: judged at 0.02, configured 0.05 now" in message
        named = message.split("differing thresholds: ", 1)[1].split(";", 1)[0]
        assert named == "max_unsupported_duty_ucb: judged at 0.02, configured 0.05 now"
        assert "--gate current" in message and "generate_stage_artifacts" in message
        # A dropped threshold, another schema version, another kind: each is named or called out.
        with pytest.raises(AncestorReuseError, match="min_avg_reward: judged at 2100.0, configured None now"):
            _find(tmp_path, current_gate_config={k: v for k, v in STANCE_CURRICULUM.items() if k != "min_avg_reward"})
        with pytest.raises(AncestorReuseError, match=r"none nameable \(gate kind or schema version differs\)"):
            _find(tmp_path, current_gate_config={**STANCE_CURRICULUM, "gate_schema_version": 2})
        with pytest.raises(AncestorReuseError, match=r"\(recovery_quality/v1\) now"):
            _find(tmp_path, current_gate_config={**STANCE_CURRICULUM, "gate_kind": "recovery_quality/v1"})

    def test_a_non_gate_key_edit_does_not_refuse_reuse(self, tmp_path):
        """Schedule, collapse, shaping, retention and the JAX override table are not the gate (D-B7)."""
        build_trunk_run(tmp_path)
        current = {
            **STANCE_CURRICULUM,
            "timesteps": 1_000,
            "collapse_patience": 3,
            "collapse_min_evals": 2,
            "warmup_timesteps": 5,
            "max_checkpoints": 1,
            "jax": {"min_avg_reward": 1.0},
        }
        assert _find(tmp_path, current_gate_config=current).stage_id == "stance"

    def test_a_numeric_retype_does_not_refuse_reuse(self, tmp_path):
        """``100`` and ``100.0`` state the same threshold (D-B7)."""
        build_trunk_run(tmp_path, curriculum={**STANCE_CURRICULUM, "min_avg_reward": 2100, "settle_steps": 200.0})
        assert _find(tmp_path).stage_id == "stance"
        assert (
            _find(tmp_path, current_gate_config={**STANCE_CURRICULUM, "min_eval_episodes": 40.0}).stage_id == "stance"
        )

    def test_a_directory_re_judged_under_the_current_gate_is_reusable_although_it_trained_under_another(self, tmp_path):
        """D-B8: only the gate the verdict was judged under counts.  A directory that trained under
        one block and was re-judged under the current one (edit a threshold, re-judge, never retrain)
        is reusable under the current gate — and not under the block it trained under."""
        trained_under = {**STANCE_CURRICULUM, "max_unsupported_duty_ucb": 0.05}
        stage_dir = build_trunk_run(tmp_path, curriculum=trained_under, judged_under=STANCE_CURRICULUM)
        recorded = json.loads((stage_dir / "stage_config.json").read_text(encoding="utf-8"))["curriculum"]
        assert recorded == trained_under
        assert read_gate_verdict(stage_dir)["gate_sha256"] == gate_config_sha256(gate_config_view(STANCE_CURRICULUM))

        found = _find(tmp_path)
        assert found.stage_id == "stance" and found.gate_sha256 == read_gate_verdict(stage_dir)["gate_sha256"]
        with pytest.raises(AncestorReuseError, match="max_unsupported_duty_ucb: judged at 0.02, configured 0.05 now"):
            _find(tmp_path, current_gate_config=trained_under)

    def test_a_verdict_whose_gate_digest_disagrees_with_its_recorded_gate_is_refused(self, tmp_path):
        """A verdict edited after judging is malformed (the reader refuses it), surfaced through rule 2."""
        stage_dir = build_trunk_run(tmp_path)
        path = stage_dir / "gate_verdict.json"
        verdict = json.loads(path.read_text(encoding="utf-8"))
        verdict["gate"]["thresholds"]["max_unsupported_duty_ucb"] = 0.5
        path.write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
        with pytest.raises(AncestorReuseError, match="not the digest of the recorded gate block"):
            _find(tmp_path)
        # Nor does re-stamping the digest help: it must match the CURRENT gate too.
        verdict["gate_sha256"] = gate_config_sha256(verdict["gate"])
        path.write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
        with pytest.raises(AncestorReuseError, match="max_unsupported_duty_ucb: judged at 0.5, configured 0.02 now"):
            _find(tmp_path)

    def test_rule_seven_is_applied_before_the_handoff_is_hashed(self, tmp_path):
        """Rule order: a gate mismatch is named even when the checkpoint was also rewritten."""
        stage_dir = build_trunk_run(tmp_path)
        (stage_dir / "models" / "robust_best_model.zip").write_bytes(b"rewritten")
        with pytest.raises(AncestorReuseError, match="differing thresholds: max_unsupported_duty_ucb"):
            _find(tmp_path, current_gate_config={**STANCE_CURRICULUM, "max_unsupported_duty_ucb": 0.05})
        strip_gate_record(stage_dir)
        with pytest.raises(AncestorReuseError, match="records no gate_sha256"):
            _find(tmp_path)

    def test_rule_seven_is_applied_before_the_chain(self, tmp_path):
        """Rule order: a non-root candidate with both a wrong parent and a stale gate gets the actionable
        gate refusal (re-judge), not the chain's — rule 7 sits between the task and the chain."""
        chained = tmp_path / "chained"
        chained.mkdir()
        other_stance = "sha256:" + "f" * 64
        locomotion_dir, stance_sha256 = build_chained_trunk(chained, parent_sha256=other_stance)
        edited = {**LOCOMOTION_CURRICULUM, "min_avg_forward_vel": 3.0}
        with pytest.raises(AncestorReuseError) as excinfo:
            _find_locomotion(chained, parent_model_sha256=stance_sha256, current_gate_config=edited)
        assert "differing thresholds: min_avg_forward_vel: judged at 2.0, configured 3.0 now" in str(excinfo.value)
        assert "parent_checkpoint_sha256" not in str(excinfo.value) and other_stance not in str(excinfo.value)
        strip_gate_record(locomotion_dir)
        with pytest.raises(AncestorReuseError, match="records no gate_sha256"):
            _find_locomotion(chained, parent_model_sha256=stance_sha256)
        # With the gate restored the chain rule is what refuses this candidate.
        build_chained_trunk(chained, parent_sha256=other_stance)
        with pytest.raises(AncestorReuseError, match=f"descends from 'stance' checkpoint {other_stance}"):
            _find_locomotion(chained, parent_model_sha256=stance_sha256)

    def test_a_digest_without_a_gate_block_is_refused_without_inventing_thresholds(self, tmp_path):
        """The reader allows gate_sha256 beside no gate block; the mismatch then says the thresholds
        cannot be named rather than listing every key as 'judged at None'."""
        stage_dir = build_trunk_run(tmp_path)
        path = stage_dir / "gate_verdict.json"
        verdict = json.loads(path.read_text(encoding="utf-8"))
        del verdict["gate"]
        path.write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
        assert _find(tmp_path).gate_sha256 == verdict["gate_sha256"]
        with pytest.raises(AncestorReuseError) as excinfo:
            _find(tmp_path, current_gate_config={**STANCE_CURRICULUM, "max_unsupported_duty_ucb": 0.05})
        message = str(excinfo.value)
        assert "not nameable (the verdict records its gate digest but no gate block)" in message
        assert "judged at None" not in message


def build_chained_trunk(
    run_dir: Path, *, parent_sha256: "str | None" = None, **locomotion_overrides
) -> tuple[Path, str]:
    """A stance root plus a locomotion child that recorded entering from it.

    Returns the locomotion stage directory and the stance handoff's digest —
    the ``parent_checkpoint_sha256`` the child recorded unless
    *parent_sha256* overrides it (a child trained from some OTHER stance).
    """
    stance_dir = build_trunk_run(run_dir)
    stance_sha256 = sha256_file(stance_dir / "models" / "robust_best_model.zip")
    lineage = {
        "load_path": str(stance_dir / "models" / "robust_best_model.zip"),
        "load_mode": "initialize_next_stage",
        "parent_checkpoint_sha256": parent_sha256 or stance_sha256,
        "parent_task_sha256": STANCE_TASK,
    }
    kwargs: dict[str, Any] = dict(
        stage_dirname="03_locomotion",
        stage=2,
        stage_id="locomotion",
        task_sha256=LOCOMOTION_TASK,
        curriculum=LOCOMOTION_CURRICULUM,
        lineage=lineage,
    )
    kwargs.update(locomotion_overrides)
    return build_trunk_run(run_dir, **kwargs), stance_sha256


def _find_locomotion(run_dir, **overrides):
    kwargs: dict[str, Any] = dict(
        entry=load_stage_manifest("trex").by_id("locomotion"),
        current_task_sha256=LOCOMOTION_TASK,
        current_gate_config=LOCOMOTION_CURRICULUM,
    )
    kwargs.update(overrides)
    return _find(run_dir, **kwargs)


class TestChainRule:
    """Rule 4: a certified child is reusable only on top of the parent checkpoint it was trained from.

    Two runs that both certified stance produced two different checkpoints;
    a walk descends from exactly one of them.  Ids are never a substitute
    for the digests, and reuse proceeds root-first.
    """

    def test_a_child_is_reused_on_top_of_the_parent_it_was_trained_from(self, tmp_path):
        locomotion_dir, stance_sha256 = build_chained_trunk(tmp_path)

        ancestor = _find_locomotion(tmp_path, parent_model_sha256=stance_sha256)

        assert ancestor.stage_id == "locomotion" and ancestor.stage_dir == locomotion_dir
        assert ancestor.task_sha256 == LOCOMOTION_TASK

    def test_a_child_trained_from_another_parent_checkpoint_is_refused(self, tmp_path):
        """Same ids, same task, a different stance checkpoint: the digest decides, not the id."""
        other_stance = "sha256:" + "f" * 64
        _, stance_sha256 = build_chained_trunk(tmp_path, parent_sha256=other_stance)

        with pytest.raises(AncestorReuseError, match=f"descends from 'stance' checkpoint {other_stance}"):
            _find_locomotion(tmp_path, parent_model_sha256=stance_sha256)

    def test_a_child_that_recorded_no_load_is_refused(self, tmp_path):
        """A locomotion trained from scratch (or before lineage was recorded) proves no chain."""
        _, stance_sha256 = build_chained_trunk(tmp_path, lineage=None)

        with pytest.raises(AncestorReuseError, match="records no initialize_next_stage load"):
            _find_locomotion(tmp_path, parent_model_sha256=stance_sha256)

    def test_a_child_that_entered_under_another_load_mode_is_refused(self, tmp_path):
        _, stance_sha256 = build_chained_trunk(tmp_path)
        locomotion_config = tmp_path / "03_locomotion" / "stage_config.json"
        record = json.loads(locomotion_config.read_text(encoding="utf-8"))
        record["run"]["load_mode"] = "resume_same_stage"
        locomotion_config.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

        with pytest.raises(AncestorReuseError, match="load_mode='resume_same_stage'"):
            _find_locomotion(tmp_path, parent_model_sha256=stance_sha256)

    def test_a_child_whose_parent_is_unresolved_is_refused(self, tmp_path):
        """Reuse is root-first: with no certified stance resolved, the walk's chain cannot be checked."""
        build_chained_trunk(tmp_path)

        with pytest.raises(AncestorReuseError, match="warm-starts from 'stance', which has no resolved certified"):
            _find_locomotion(tmp_path, parent_model_sha256=None)

    def test_an_unreadable_stage_config_reads_as_no_recorded_parent(self, tmp_path):
        """The task still resolves from task_fingerprint.json; the chain then fails closed."""
        _, stance_sha256 = build_chained_trunk(tmp_path)
        (tmp_path / "03_locomotion" / "stage_config.json").write_text("{not json", encoding="utf-8")

        with pytest.raises(AncestorReuseError, match="records no initialize_next_stage load"):
            _find_locomotion(tmp_path, parent_model_sha256=stance_sha256)

    def test_the_chain_is_checked_before_the_handoff_is_hashed(self, tmp_path):
        """Rule order: a wrong parent is named even when the checkpoint was also rewritten."""
        other_stance = "sha256:" + "f" * 64
        locomotion_dir, stance_sha256 = build_chained_trunk(tmp_path, parent_sha256=other_stance)
        (locomotion_dir / "models" / "robust_best_model.zip").write_bytes(b"rewritten")

        with pytest.raises(AncestorReuseError, match="descends from 'stance' checkpoint"):
            _find_locomotion(tmp_path, parent_model_sha256=stance_sha256)

    def test_a_root_that_entered_from_a_parent_is_refused(self, tmp_path):
        """A stance that warm-started from something contradicts the manifest, which makes it a root."""
        build_trunk_run(
            tmp_path,
            lineage={
                "load_path": "elsewhere/best_model.zip",
                "load_mode": "initialize_next_stage",
                "parent_checkpoint_sha256": "sha256:" + "e" * 64,
            },
        )

        with pytest.raises(AncestorReuseError, match="'stance' is a root node in the current manifest"):
            _find(tmp_path)

    def test_a_resumed_root_is_not_a_child(self, tmp_path):
        """resume_same_stage continues the same node; it is not a parent and the root stays reusable."""
        build_trunk_run(
            tmp_path,
            lineage={
                "load_path": "earlier/robust_best_model.zip",
                "load_mode": "resume_same_stage",
                "parent_checkpoint_sha256": "sha256:" + "e" * 64,
            },
        )

        assert _find(tmp_path).stage_id == "stance"

    def test_a_parent_digest_for_a_root_is_a_caller_error(self, trunk_run):
        with pytest.raises(ValueError, match="'stance' is a root node"):
            _find(trunk_run, parent_model_sha256="sha256:" + "a" * 64)


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


def build_middle_run(run_dir: Path, source_run: Path, **find_overrides) -> Path:
    """A run that REUSED stance from *source_run* and trained nothing of it.

    Holds exactly what ``record_ancestor`` writes — ``ancestors/stance/``
    with ``ancestor.json`` and the copied sidecars — and no stage directory
    or checkpoint for stance at all, which is what a run trunked from
    *source_run* looks like.  Returns the record directory.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    return record_ancestor(run_dir, _find(source_run, **find_overrides))


def _repoint_record(record_dir: Path, source_run: Path) -> Path:
    """Rewrite a middle run's record to name *source_run* — a hand-edited or moved layout."""
    record_path = record_dir / ANCESTOR_RECORD_NAME
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["source_run_dir"] = str(source_run)
    record["source_stage_dir"] = str(source_run / "01_stance")
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record_path


def _rejudge_rewritten_stance(stage_dir: Path) -> None:
    """Rewrite the stance handoff and judge it again, so the source is reusable on its own
    but is no longer the checkpoint an earlier record bound its reuse to."""
    fingerprint = json.loads((stage_dir / "task_fingerprint.json").read_text(encoding="utf-8"))
    zip_path = _sb3_style_zip(
        stage_dir / "models" / "robust_best_model.zip",
        {MODEL_TASK_ATTRIBUTE: fingerprint, MODEL_IDENTITY_ATTRIBUTE: trunk_plant().to_dict(), "rewritten": True},
    )
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
        checkpoint=zip_path,
        normalization=stage_dir / "models" / "robust_best_model_vecnorm.pkl",
        # The same block build_trunk_run records: the re-judged source stays
        # reusable on its own (rule 7 passes) so the record's handoff-digest
        # binding is what refuses it.
        gate_config=gate_config_view(STANCE_CURRICULUM),
    )


class TestTrunksCompose:
    """Decision D-A23: rule 1 follows a run's ``ancestors/<stage_id>/ancestor.json`` to the run that
    certified the node, so a run that reused a node can serve as a trunk for it.  The result
    describes the SOURCE; the record binds the reuse to one checkpoint pair; the source must be
    a run directory this machine can see, as recorded or beside the run that holds the record.
    Following is opt-in (``follow_records=True``, what ``train_curriculum --trunk-from`` passes):
    a caller that tries its own run directory first must not follow the record it wrote itself."""

    def test_a_record_is_followed_only_when_the_caller_opts_in(self, tmp_path):
        """The default refuses a run that holds only a record, with the pre-D-A23 wording plus the
        record it is not following.  The case that matters is the notebook's: RUN_DIR reused stance
        from TRUNK_DIR in an earlier pass and holds ``ancestors/stance/`` for it; on a re-run RUN_DIR
        is tried first and must refuse — following would return the TRUNK's stance with RUN_DIR as
        the candidate, which the loop would take for this run's own node — and TRUNK_DIR then
        resolves directly, cross-run, as before."""
        trunk = tmp_path / "trunk"
        trunk.mkdir()
        stage_dir = build_trunk_run(trunk)
        run_dir = tmp_path / "run"
        build_middle_run(run_dir, trunk)

        with pytest.raises(AncestorReuseError) as excinfo:
            _find(run_dir)
        message = str(excinfo.value)
        assert message.startswith(f"{run_dir} has no stage directory for 'stance' (looked for ")
        assert (
            "it holds ancestors/stance/ancestor.json for a reuse it made itself, which is not followed here" in message
        )
        assert "follow_records=False" in message
        assert "followed the ancestor record" not in message

        direct = _find(trunk)
        assert direct.source_run_dir == trunk and direct.stage_dir == stage_dir and direct.via == ()
        # The same run, asked to follow, resolves to the trunk and says so.
        followed = _follow(run_dir)
        assert followed == CertifiedAncestor(**{**direct.__dict__, "via": (run_dir,)})

    def test_a_record_names_its_source_absolutely_so_it_follows_from_any_cwd(self, tmp_path, monkeypatch, caplog):
        """A record made from a relative trunk path (``--trunk-from logs/<run>``, the documented
        spelling) stores the source resolved, like the handoff paths beside it, so following it
        from another working directory — with the middle run copied elsewhere, so the sibling
        fallback cannot help — reaches the original run directly."""
        logs = tmp_path / "logs"
        original = logs / "original"
        original.mkdir(parents=True)
        stage_dir = build_trunk_run(original)
        monkeypatch.chdir(tmp_path)
        record_dir = build_middle_run(Path("logs") / "middle", Path("logs") / "original")
        record = json.loads((record_dir / ANCESTOR_RECORD_NAME).read_text(encoding="utf-8"))
        assert record["source_run_dir"] == str(original.resolve())
        assert record["source_stage_dir"] == str(stage_dir.resolve())
        assert Path(record["source_run_dir"]).is_absolute()

        elsewhere = tmp_path / "elsewhere"
        shutil.copytree(logs / "middle", elsewhere / "middle")
        assert not (elsewhere / "original").exists()
        monkeypatch.chdir(elsewhere)
        with caplog.at_level(logging.INFO):
            found = _follow(Path("middle"))

        assert found.source_run_dir == original.resolve() and found.stage_dir == stage_dir.resolve()
        assert found.run_id == "original" and found.via == (Path("middle"),)
        assert "using its sibling" not in caplog.text

    def test_a_trunk_of_a_trunk_resolves_to_the_run_that_certified_the_node(self, tmp_path, caplog):
        original = tmp_path / "logs" / "20260901_120000"
        original.mkdir(parents=True)
        (original / "provenance.json").write_text(json.dumps({"run_id": "trex-sb3-ppo-original"}), encoding="utf-8")
        stage_dir = build_trunk_run(original)
        middle = tmp_path / "logs" / "20260902_120000"
        build_middle_run(middle, original)
        assert not any(child.is_dir() and child.name != ANCESTORS_DIRNAME for child in middle.iterdir())

        with caplog.at_level(logging.INFO):
            found = _follow(middle)

        direct = _follow(original)
        assert found == CertifiedAncestor(**{**direct.__dict__, "via": (middle,)})
        assert found.run_id == "trex-sb3-ppo-original"
        assert found.source_run_dir == original and found.stage_dir == stage_dir
        assert found.via == (middle,)
        assert direct.via == ()
        followed = [r for r in caplog.records if r.message.startswith("Following the ancestor record")]
        assert len(followed) == 1 and str(middle) in followed[0].message and str(original) in followed[0].message

        # A child trunked from the middle run records the ORIGINAL run as its parent.
        child = tmp_path / "logs" / "20260903_120000"
        with caplog.at_level(logging.INFO):
            record_dir = record_ancestor(child, found)
        record = json.loads((record_dir / ANCESTOR_RECORD_NAME).read_text(encoding="utf-8"))
        assert record["parent_run_id"] == "trex-sb3-ppo-original"
        assert record["source_run_dir"] == str(original) and record["source_stage_dir"] == str(stage_dir)
        assert "via" not in record
        assert f"resolved via {middle}" in caplog.text

    def test_a_stage_directory_beside_a_record_is_preferred(self, tmp_path):
        original = tmp_path / "original"
        original.mkdir()
        build_trunk_run(original)
        middle = tmp_path / "middle"
        build_middle_run(middle, original)
        own = build_trunk_run(middle)

        found = _follow(middle)

        assert found.stage_dir == own and found.source_run_dir == middle and found.run_id == "middle"
        assert found.via == ()

    def test_a_record_bound_to_a_checkpoint_the_source_no_longer_holds_is_refused(self, tmp_path):
        original = tmp_path / "original"
        original.mkdir()
        stage_dir = build_trunk_run(original)
        middle = tmp_path / "middle"
        record_dir = build_middle_run(middle, original)
        bound = json.loads((record_dir / ANCESTOR_RECORD_NAME).read_text(encoding="utf-8"))["handoff"]["model_sha256"]

        _rejudge_rewritten_stance(stage_dir)
        rewritten = _follow(original)
        assert rewritten.model_sha256 != bound

        with pytest.raises(AncestorReuseError) as excinfo:
            _follow(middle)
        message = str(excinfo.value)
        assert message.startswith(f"followed the ancestor record in {middle} to {original}: ")
        assert rewritten.model_sha256 in message and bound in message
        assert "rewritten since" in message
        # The refusal is the record's handoff-digest binding (D-A23), not a
        # gate one: the re-judged source was judged under the same gate.
        assert "bound the reuse to handoff.model_sha256" in message
        assert "judged under gate" not in message and "gate_sha256" not in message

    def test_a_record_whose_sidecar_digest_disagrees_is_refused(self, tmp_path):
        original = tmp_path / "original"
        original.mkdir()
        build_trunk_run(original)
        middle = tmp_path / "middle"
        record_path = build_middle_run(middle, original) / ANCESTOR_RECORD_NAME
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record["handoff"]["normalization_sha256"] = OTHER_TASK
        record_path.write_text(json.dumps(record), encoding="utf-8")

        with pytest.raises(AncestorReuseError, match="handoff.normalization_sha256 sha256:2{64}"):
            _follow(middle)

    def test_a_missing_source_refuses_naming_both_paths(self, tmp_path):
        original = tmp_path / "old" / "original"
        original.mkdir(parents=True)
        build_trunk_run(original)
        middle = tmp_path / "new" / "middle"
        build_middle_run(middle, original)
        shutil.rmtree(tmp_path / "old")

        with pytest.raises(AncestorReuseError) as excinfo:
            _follow(middle)
        message = str(excinfo.value)
        assert "a run directory this machine cannot see" in message
        assert str(original) in message and str(tmp_path / "new" / "original") in message

    def test_the_sibling_fallback_finds_a_moved_log_base(self, tmp_path, caplog):
        old_base = tmp_path / "old" / "logs" / "trex" / "ppo"
        original = old_base / "20260901_120000"
        original.mkdir(parents=True)
        build_trunk_run(original)
        build_middle_run(old_base / "20260902_120000", original)
        new_base = tmp_path / "new" / "logs" / "trex" / "ppo"
        new_base.parent.mkdir(parents=True)
        shutil.move(str(old_base), str(new_base))
        assert not original.exists()

        with caplog.at_level(logging.INFO):
            found = _follow(new_base / "20260902_120000")

        assert found.source_run_dir == new_base / "20260901_120000"
        assert found.stage_dir == new_base / "20260901_120000" / "01_stance"
        assert found.run_id == "20260901_120000"
        assert found.via == (new_base / "20260902_120000",)
        assert "using its sibling" in caplog.text

    def test_a_two_hop_chain_resolves_through_both_records(self, tmp_path):
        """A record made by ``record_ancestor`` already names the certifying run, so a chain longer
        than one hop is a hand-edited layout; it still resolves, outermost hop first in ``via``."""
        a = tmp_path / "a"
        a.mkdir()
        build_trunk_run(a)
        b = tmp_path / "b"
        build_middle_run(b, a)
        c = tmp_path / "c"
        _repoint_record(build_middle_run(c, a), b)

        found = _follow(c)

        assert found.run_id == "a" and found.source_run_dir == a and found.stage_dir == a / "01_stance"
        assert found.via == (c, b)

    def test_a_cycle_refuses(self, tmp_path):
        a = tmp_path / "a"
        a.mkdir()
        build_trunk_run(a)
        b = tmp_path / "b"
        c = tmp_path / "c"
        _repoint_record(build_middle_run(b, a), c)
        _repoint_record(build_middle_run(c, a), b)

        with pytest.raises(AncestorReuseError, match="already on the followed path .*which is a cycle") as excinfo:
            _follow(c)
        assert str(excinfo.value).startswith(f"followed the ancestor record in {c} to {b}: ")

        # A record naming its own run is the one-hop cycle.
        _repoint_record(b / ANCESTORS_DIRNAME / "stance", b)
        with pytest.raises(AncestorReuseError, match="which is a cycle"):
            _follow(b)

    def test_the_hop_limit_refuses(self, tmp_path):
        source = tmp_path / "run00"
        source.mkdir()
        build_trunk_run(source)
        runs = [source]
        for hop in range(1, ANCESTOR_RECORD_HOP_LIMIT + 2):
            run = tmp_path / f"run{hop:02d}"
            _repoint_record(build_middle_run(run, source), runs[-1])
            runs.append(run)

        # Exactly the limit is followed; one more is refused, naming the path.
        assert _follow(runs[ANCESTOR_RECORD_HOP_LIMIT]).via == tuple(runs[ANCESTOR_RECORD_HOP_LIMIT:0:-1])
        with pytest.raises(AncestorReuseError, match=f"past the limit of {ANCESTOR_RECORD_HOP_LIMIT}"):
            _follow(runs[ANCESTOR_RECORD_HOP_LIMIT + 1])

    @pytest.mark.parametrize(
        "break_source, reason",
        [
            (lambda run: build_trunk_run(run, passed=False), "FAILED gate"),
            (
                lambda run: (run / "01_stance" / "models" / "robust_best_model_vecnorm.pkl").unlink(),
                "no complete handoff pair",
            ),
            (lambda run: shutil.rmtree(run / "01_stance"), "no stage directory for 'stance'"),
            (lambda run: strip_gate_record(run / "01_stance"), "records no gate_sha256"),
        ],
    )
    def test_every_refusal_at_the_source_is_prefixed_with_the_hop_taken(self, tmp_path, break_source, reason):
        original = tmp_path / "original"
        original.mkdir()
        build_trunk_run(original)
        middle = tmp_path / "middle"
        build_middle_run(middle, original)
        break_source(original)

        with pytest.raises(AncestorReuseError, match=reason) as excinfo:
            _follow(middle)
        assert str(excinfo.value).startswith(f"followed the ancestor record in {middle} to {original}: ")

    def test_a_gate_refusal_at_the_source_is_prefixed_with_the_hop_taken(self, tmp_path):
        """Rule 7 at the source: a stance reused through a record is still checked against THIS
        run's gate, and refused naming the threshold that differs."""
        original = tmp_path / "original"
        original.mkdir()
        build_trunk_run(original)
        middle = tmp_path / "middle"
        build_middle_run(middle, original)

        assert _follow(middle).via == (middle,)
        current = {**STANCE_CURRICULUM, "min_full_horizon_fraction": 0.99}
        with pytest.raises(AncestorReuseError, match="min_full_horizon_fraction: judged at 0.95") as excinfo:
            _follow(middle, current_gate_config=current)
        assert str(excinfo.value).startswith(f"followed the ancestor record in {middle} to {original}: ")

    def test_the_chain_rule_is_applied_at_the_source(self, tmp_path):
        """Rule 4 at the source: a walk reused through a record is still checked against the digest
        resolved for its parent in THIS run, and refused on any other stance."""
        original = tmp_path / "original"
        original.mkdir()
        _, stance_sha256 = build_chained_trunk(original)
        middle = tmp_path / "middle"
        middle.mkdir()
        record_ancestor(middle, _find_locomotion(original, parent_model_sha256=stance_sha256))

        found = _find_locomotion(middle, parent_model_sha256=stance_sha256, follow_records=True)
        assert found.source_run_dir == original and found.via == (middle,)
        with pytest.raises(AncestorReuseError, match="descends from 'stance' checkpoint") as excinfo:
            _find_locomotion(middle, parent_model_sha256="sha256:" + "d" * 64, follow_records=True)
        assert str(excinfo.value).startswith("followed the ancestor record in ")

    @pytest.mark.parametrize(
        "corrupt, reason",
        [
            (lambda record: "{broken", "not readable JSON"),
            (lambda record: json.dumps([record]), "must hold a JSON object"),
            (lambda record: json.dumps({**record, "schema": "mesozoic.ancestor-record/v0"}), "declares schema"),
            (
                lambda record: json.dumps({**record, "stage_id": "recovery"}),
                "records stage_id 'recovery', not 'stance'",
            ),
            (
                lambda record: json.dumps({k: v for k, v in record.items() if k != "source_run_dir"}),
                "no source_run_dir",
            ),
            (lambda record: json.dumps({**record, "handoff": "sha256"}), "no handoff object"),
            (
                lambda record: json.dumps({**record, "handoff": {**record["handoff"], "model_sha256": None}}),
                "no handoff.model_sha256",
            ),
        ],
    )
    def test_a_record_that_cannot_be_read_fails_closed(self, tmp_path, corrupt, reason):
        original = tmp_path / "original"
        original.mkdir()
        build_trunk_run(original)
        middle = tmp_path / "middle"
        record_path = build_middle_run(middle, original) / ANCESTOR_RECORD_NAME
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record_path.write_text(corrupt(record), encoding="utf-8")

        with pytest.raises(AncestorReuseError, match=reason) as excinfo:
            _follow(middle)
        assert "cannot be followed" in str(excinfo.value)

    def test_a_run_with_neither_directory_nor_record_says_so(self, tmp_path):
        with pytest.raises(
            AncestorReuseError, match="no stage directory for 'stance'.*and no ancestors/stance/ancestor.json to follow"
        ):
            _follow(tmp_path)


# ---------------------------------------------------------------------------
# Automatic trunk selection (decision D-A25)
# ---------------------------------------------------------------------------

BEHAVIOR_TASK = "sha256:" + "4" * 64


def _trex_chain():
    manifest = load_stage_manifest("trex")
    return tuple(manifest.by_id(stage_id) for stage_id in ("stance", "locomotion", "behavior"))


def _trex_stage_configs(**stance_curriculum_overrides):
    return {
        1: {"env_kwargs": {}, "curriculum_kwargs": {**STANCE_CURRICULUM, **stance_curriculum_overrides}},
        2: {"env_kwargs": {}, "curriculum_kwargs": LOCOMOTION_CURRICULUM},
        3: {"env_kwargs": {}, "curriculum_kwargs": {"gate_kind": "task_success/v1", "timesteps": 10}},
    }


@pytest.fixture
def fixture_tasks(monkeypatch):
    """Derive each chain node's task digest as the trunk fixtures were judged under."""
    from environments.shared import task_fingerprint as task_fingerprint_module

    digests = {1: STANCE_TASK, 2: LOCOMOTION_TASK, 3: BEHAVIOR_TASK}
    monkeypatch.setattr(
        task_fingerprint_module,
        "derive_stage_task_fingerprint",
        lambda **kwargs: {"task_sha256": digests[kwargs["stage"]]},
    )


def _select(log_dir, **overrides):
    from environments.shared.ancestors import select_trunk

    kwargs: dict[str, Any] = dict(
        species="trex",
        chain=_trex_chain(),
        stage_configs=_trex_stage_configs(),
        plant_identity=trunk_plant(),
    )
    kwargs.update(overrides)
    return select_trunk(log_dir, **kwargs)


class TestSelectTrunk:
    """Decision D-A25: ``TRUNK_FROM = "auto"`` picks the sibling run covering the most of the chain."""

    def test_the_run_covering_the_most_of_the_chain_wins_over_a_newer_shallower_one(self, tmp_path, fixture_tasks):
        older = tmp_path / "20260901_000000"
        deeper = tmp_path / "20260905_000000"
        newest = tmp_path / "20260910_000000"
        build_trunk_run(older)
        build_chained_trunk(deeper)
        build_trunk_run(newest)

        selection = _select(tmp_path)

        assert selection.considered == ("stance", "locomotion")
        assert [candidate.run_dir.name for candidate in selection.candidates] == [
            "20260910_000000",
            "20260905_000000",
            "20260901_000000",
        ], "scanned greatest directory name first (the newest timestamp id)"
        assert [candidate.coverage for candidate in selection.candidates] == [1, 2, 1]
        assert selection.run_dir == deeper and selection.selected is selection.candidates[1]
        assert [ancestor.stage_id for ancestor in selection.selected.covered] == ["stance", "locomotion"]
        # Each child was chained onto the ancestor found for its parent (rule 4).
        stance_sha256 = sha256_file(deeper / "01_stance" / "models" / "robust_best_model.zip")
        assert selection.selected.covered[0].model_sha256 == stance_sha256
        assert selection.selected.refusal is None
        # The shallower runs name the node they could not satisfy and why.
        newest_candidate = selection.candidates[0]
        assert newest_candidate.refusal is not None
        assert newest_candidate.refusal[0] == "locomotion" and "no stage directory" in newest_candidate.refusal[1]
        # Replication is reported per covered node (the fixtures record no recipe digest: counted alone).
        assert [(node.stage_id, node.distinct_seeds, node.required_seeds) for node in selection.support] == [
            ("stance", 1, 1),
            ("locomotion", 1, 1),
        ]
        assert not any(node.provisional for node in selection.support)
        text = selection.describe()
        assert "selected run 20260905_000000" in text and "covers stance, locomotion" in text
        assert "also usable: run 20260910_000000" in text and "also usable: run 20260901_000000" in text
        assert "TRUNK_FROM = '<run id>'" in text

    def test_the_newest_run_wins_a_tie(self, tmp_path, fixture_tasks):
        build_trunk_run(tmp_path / "20260901_000000")
        build_trunk_run(tmp_path / "20260910_000000")

        selection = _select(tmp_path)

        assert selection.run_dir == tmp_path / "20260910_000000"
        assert [candidate.coverage for candidate in selection.candidates] == [1, 1]

    def test_this_run_is_excluded_and_every_refusal_is_named(self, tmp_path, fixture_tasks):
        this_run = tmp_path / "20260917_000000"
        build_trunk_run(this_run)
        usable = tmp_path / "20260901_000000"
        build_trunk_run(usable)
        failed = tmp_path / "20260905_000000"
        build_trunk_run(failed, passed=False)
        other_task = tmp_path / "20260906_000000"
        build_trunk_run(other_task, task_sha256=OTHER_TASK)
        (tmp_path / "behaviors").mkdir()  # the direction/terrain output tree: no stage directory at all

        selection = _select(tmp_path, exclude=(this_run,))

        assert selection.run_dir == usable
        assert this_run not in [candidate.run_dir for candidate in selection.candidates]
        refusals = {candidate.run_dir.name: candidate.refusal for candidate in selection.candidates}
        # The usable run covers stance and names locomotion as the node it could not satisfy.
        assert refusals["20260901_000000"][0] == "locomotion"
        assert refusals["20260905_000000"][0] == "stance" and "FAILED gate" in refusals["20260905_000000"][1]
        assert "judged under task" in refusals["20260906_000000"][1]
        assert "no stage directory" in refusals["behaviors"][1]
        text = selection.describe()
        assert "refused 20260905_000000: stance: " in text and "FAILED gate" in text
        assert "refused behaviors: stance: " in text
        assert "    locomotion: not covered: " in text, "the selected run names the node it could not satisfy"

    def test_no_usable_run_selects_nothing_and_says_so(self, tmp_path, fixture_tasks):
        build_trunk_run(tmp_path / "20260905_000000", passed=False)

        selection = _select(tmp_path)

        assert selection.run_dir is None and selection.selected is None and selection.support == ()
        assert "no run covers the root 'stance': every node trains here" in selection.describe()
        empty = _select(tmp_path / "missing")
        assert empty.run_dir is None and empty.candidates == ()

    def test_retrain_from_limits_the_nodes_consulted(self, tmp_path, fixture_tasks):
        deeper = tmp_path / "20260905_000000"
        build_chained_trunk(deeper)
        stance, locomotion, _ = _trex_chain()

        selection = _select(tmp_path, retrain_from=locomotion)
        assert selection.considered == ("stance",)
        assert selection.run_dir == deeper and [a.stage_id for a in selection.selected.covered] == ["stance"]

        selection = _select(tmp_path, retrain_from=stance)
        assert selection.considered == () and selection.run_dir is None
        assert "nothing (the root is trained here)" in selection.describe()

    def test_the_widen_parameter_left_with_the_notebook_knob(self):
        """Decision D-D14: the notebook's WIDEN_FROM was select_trunk's only ``widen_from`` caller; both are gone.

        A root widened on the command line is judged in its new run, and the notebook's resolve cell refuses a
        resolved trunk until it holds a verdict (``result_bundle.refuse_trunk_over_unjudged_widened_root``)."""
        import inspect

        from environments.shared.ancestors import select_trunk

        assert "widen_from" not in inspect.signature(select_trunk).parameters

    def test_a_run_that_reused_its_stance_is_followed_to_the_source(self, tmp_path, fixture_tasks):
        source = tmp_path / "20260901_000000"
        build_trunk_run(source)
        middle = tmp_path / "20260910_000000"
        build_middle_run(middle, source)

        selection = _select(tmp_path)

        # Both cover stance; the newer (middle) run wins the tie and resolves to the source (D-A23).
        assert selection.run_dir == middle
        ancestor = selection.selected.covered[0]
        assert ancestor.source_run_dir == source and ancestor.via == (middle,)
        assert selection.support[0].run_id == run_id_for(source)

    def test_replication_of_a_covered_node_is_counted_among_the_source_siblings(self, tmp_path, fixture_tasks):
        recipe = dict(algorithm="ppo", hyperparameters={"n_steps": 8}, record_recipe_digest=True)
        build_trunk_run(tmp_path / "20260901_000000", seed=7, **recipe)
        build_trunk_run(tmp_path / "20260905_000000", seed=9, **recipe)
        build_trunk_run(tmp_path / "20260906_000000", seed=9, **recipe)  # repeats a seed: not a replicate

        selection = _select(tmp_path, stage_configs=_trex_stage_configs(certification_seeds=3))

        assert selection.run_dir == tmp_path / "20260906_000000"
        (support,) = selection.support
        assert (support.training_seed, support.distinct_seeds, support.required_seeds) == (9, 2, 3)
        assert support.provisional
        assert "replication 2 of 3 required seed(s) - provisional" in selection.describe()

    def test_an_older_interface_parent_is_reported_as_a_widen_candidate(self, tmp_path, fixture_tasks):
        from environments.shared.command_frame import COMMAND_WIDTH

        current = trunk_plant()
        older_plant = make_plant_identity(
            species="trex",
            model_path="environments/trex/assets/trex.xml",
            policy_interface_revision=current.policy_interface_revision - 2,
            observation_dim=current.observation_dim - COMMAND_WIDTH,
        )
        older = tmp_path / "20260815_205206"
        # Judged under its own (older) task, as a real pre-bump run is: rule 3 refuses it.
        build_trunk_run(older, plant=older_plant, task_sha256=OTHER_TASK)
        failed_older = tmp_path / "20260810_000000"
        build_trunk_run(failed_older, plant=older_plant, task_sha256=OTHER_TASK, passed=False)
        # Older interface but behind a physics bump: the widen gate would refuse it, so no hint.
        physics_bumped = make_plant_identity(
            species="trex",
            model_path="environments/trex/assets/trex.xml",
            policy_interface_revision=current.policy_interface_revision - 2,
            observation_dim=current.observation_dim - COMMAND_WIDTH,
            physics_sha256="sha256:" + "f" * 64,
        )
        build_trunk_run(tmp_path / "20260601_000000", plant=physics_bumped, task_sha256=OTHER_TASK)
        # Older interface but not exactly COMMAND_WIDTH narrower: not widenable either.
        wrong_width = make_plant_identity(
            species="trex",
            model_path="environments/trex/assets/trex.xml",
            policy_interface_revision=current.policy_interface_revision - 1,
            observation_dim=current.observation_dim - 7,
        )
        build_trunk_run(tmp_path / "20260501_000000", plant=wrong_width, task_sha256=OTHER_TASK)

        selection = _select(tmp_path, plant_identity=current)

        assert selection.run_dir is None
        assert [
            (parent.run_dir.name, parent.stage_id, parent.revision_gap) for parent in selection.older_interface
        ] == [("20260815_205206", "stance", 2)]
        text = selection.describe()
        assert "older-interface parent: run 20260815_205206" in text
        assert f"policy interface r{current.policy_interface_revision - 2}" in text
        # D-D14: the hint names the command-line tool and the bound this parent needs, never the deleted knobs.
        assert "python -m environments.shared.scripts.widen_checkpoint --max-revision-gap 2" in text
        assert "WIDEN_FROM" not in text and "WIDEN_MAX_REVISION_GAP" not in text

    def test_an_older_interface_parent_identified_only_by_its_checkpoint_is_still_reported(
        self, tmp_path, fixture_tasks
    ):
        from environments.shared.command_frame import COMMAND_WIDTH

        current = trunk_plant()
        older_plant = make_plant_identity(
            species="trex",
            model_path="environments/trex/assets/trex.xml",
            policy_interface_revision=current.policy_interface_revision - 1,
            observation_dim=current.observation_dim - COMMAND_WIDTH,
        )
        older = tmp_path / "20260815_205206"
        stage_dir = build_trunk_run(older, plant=older_plant, task_sha256=OTHER_TASK)
        (stage_dir / "plant_identity.json").unlink()  # the archive's own identity attribute remains

        selection = _select(tmp_path, plant_identity=current)

        assert [parent.run_dir.name for parent in selection.older_interface] == ["20260815_205206"]

    def test_an_unreadable_checkpoint_refuses_that_run_only(self, tmp_path, fixture_tasks):
        """No neighbour aborts the scan: rule 5 hashes the handoff pair, and a checkpoint the mount cannot
        read is listed as refused while a usable sibling is still selected."""
        usable = tmp_path / "20260901_000000"
        build_trunk_run(usable)
        broken = tmp_path / "20260910_000000"
        stage_dir = build_trunk_run(broken)
        archive = stage_dir / "models" / "robust_best_model.zip"
        archive.unlink()
        archive.mkdir()  # exists, so the handoff selector offers it; hashing it raises IsADirectoryError

        selection = _select(tmp_path)

        assert selection.run_dir == usable
        refusal = next(c.refusal for c in selection.candidates if c.run_dir == broken)
        assert refusal[0] == "stance" and refusal[1].startswith("unreadable:")
        assert "refused 20260910_000000: stance: unreadable:" in selection.describe()

    def test_a_run_of_another_algorithm_species_or_backend_is_refused_before_the_rules(self, tmp_path, fixture_tasks):
        ppo = tmp_path / "20260901_000000"
        build_trunk_run(ppo, algorithm="ppo", hyperparameters={"n_steps": 8})
        sac = tmp_path / "20260910_000000"
        build_trunk_run(sac, algorithm="sac", hyperparameters={"batch_size": 8})
        jax = tmp_path / "20260911_000000"
        build_trunk_run(jax)
        (jax / "provenance.json").write_text(
            json.dumps({"run_id": "jax-run", "species": "trex", "algorithm": "PPO", "backend": "jax-mjx"})
        )
        other_species = tmp_path / "20260912_000000"
        build_trunk_run(other_species)
        (other_species / "provenance.json").write_text(
            json.dumps(
                {"run_id": "raptor", "species": "velociraptor", "algorithm": "PPO", "backend": "stable-baselines3"}
            )
        )

        selection = _select(tmp_path, algorithm="ppo")

        assert selection.run_dir == ppo, "the newest SAC run is refused although every rule would accept it"
        refusals = {c.run_dir.name: c.refusal for c in selection.candidates}
        assert "records algorithm SAC, not PPO" in refusals["20260910_000000"][1]
        assert "backend 'jax-mjx'" in refusals["20260911_000000"][1]
        assert "species 'velociraptor'" in refusals["20260912_000000"][1]
        # Without an algorithm to compare, the SAC run's records decide nothing and it wins on recency
        # among the same-species, same-backend runs.
        assert _select(tmp_path).run_dir == sac

    def test_retrain_from_at_the_root_consults_no_run(self, tmp_path, fixture_tasks):
        build_trunk_run(tmp_path / "20260905_000000")
        stance, _, _ = _trex_chain()

        selection = _select(tmp_path, retrain_from=stance)

        assert selection.candidates == () and selection.run_dir is None
        assert selection.skipped == "RETRAIN_FROM='stance' covers the root: every node trains here"
        assert "refused" not in selection.describe()

    def test_limit_bounds_the_scan_and_bad_inputs_are_refused_loudly(self, tmp_path, fixture_tasks):
        build_trunk_run(tmp_path / "20260901_000000")
        newest = tmp_path / "20260910_000000"
        build_trunk_run(newest, passed=False)
        (newest / "provenance.json").write_text("{not json")  # unreadable: named by its directory

        selection = _select(tmp_path, limit=1)
        assert [c.run_dir.name for c in selection.candidates] == ["20260910_000000"]
        assert selection.candidates[0].run_id == "20260910_000000"
        assert selection.run_dir is None

        with pytest.raises(ValueError, match="needs the target's chain"):
            _select(tmp_path, chain=())
        manifest = load_stage_manifest("trex")
        with pytest.raises(ValueError, match="is not on the chain"):
            _select(tmp_path, retrain_from=manifest.by_id("recovery"))

    def test_describe_caps_the_refusals_it_prints_but_keeps_every_candidate(self, tmp_path, fixture_tasks):
        for index in range(23):
            build_trunk_run(tmp_path / f"202608{index + 1:02d}_000000", passed=False)

        selection = _select(tmp_path)

        assert len(selection.candidates) == 23 and selection.run_dir is None
        text = selection.describe()
        assert text.count("  refused 202608") == 20
        assert "... and 3 more refused run(s); TRUNK_SELECTION.candidates lists every run" in text

    def test_replication_of_a_followed_ancestor_is_counted_in_the_logs_tree(self, tmp_path, fixture_tasks):
        recipe = dict(algorithm="ppo", hyperparameters={"n_steps": 8}, record_recipe_digest=True)
        source = tmp_path / "20260901_000000"
        build_trunk_run(source, seed=7, **recipe)
        build_trunk_run(tmp_path / "20260905_000000", seed=9, **recipe)  # a distinct-seed sibling in the logs tree
        middle = tmp_path / "20260910_000000"
        build_middle_run(middle, source)

        selection = _select(tmp_path)

        assert selection.run_dir == middle
        (support,) = selection.support
        assert support.run_id == run_id_for(source) and support.distinct_seeds == 2

    def test_the_selection_derives_each_node_exactly_as_the_loop_does(self, tmp_path, monkeypatch):
        """The task digest handed to the rule comes from the current stage config through the shared
        derivation with the plant identity — the sources ``train_base.train`` records it from."""
        from environments.shared import task_fingerprint as task_fingerprint_module

        derived: list[dict[str, Any]] = []

        def derive(**kwargs):
            derived.append(kwargs)
            return {"task_sha256": {1: STANCE_TASK, 2: LOCOMOTION_TASK}[kwargs["stage"]]}

        monkeypatch.setattr(task_fingerprint_module, "derive_stage_task_fingerprint", derive)
        configs = _trex_stage_configs()
        configs[1]["env_kwargs"] = {"push_prob": 0.25}
        build_trunk_run(tmp_path / "20260901_000000")

        _select(tmp_path, stage_configs=configs)

        assert [d["stage"] for d in derived] == [1, 2]
        assert derived[0] == {
            "species": "trex",
            "stage": 1,
            "backend": "stable-baselines3",
            "env_kwargs": {"push_prob": 0.25},
            "plant_identity": trunk_plant().to_dict(),
        }
