"""Shared builders for the certified-ancestor test modules.

``test_ancestors.py``, ``test_replication.py`` and ``test_train_base.py``
build their trunks with these: a stage directory shaped like a judged run's
(handoff pair, sidecars, gate verdict) under the fixed task digests and
``[curriculum]`` blocks below.  SB3-free: a checkpoint is a zip holding a
JSON ``data`` member beside fake weights.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from environments.shared.curriculum.gate_schema import gate_config_view
from environments.shared.plant_contract import MODEL_IDENTITY_ATTRIBUTE
from environments.shared.result_bundle import sha256_file, write_gate_verdict
from environments.shared.task_fingerprint import MODEL_TASK_ATTRIBUTE, TASK_FINGERPRINT_SCHEMA

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
