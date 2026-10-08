"""Quiet checkpoints keep their exact identities when disabled pushes are added."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from environments.shared.task_fingerprint import (
    TaskFingerprintError,
    compute_task_fingerprint,
    validate_recorded_task,
)

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "compsognathus_pre_recovery_tasks.json").read_text())


def recompute(record, **changes):
    return compute_task_fingerprint(
        species=record["species"],
        stage=record["stage"],
        backend=record["backend"],
        env_kwargs={**record["env"], **changes},
        plant_identity=record["plant"],
        perturbation_manifest=record["perturbation"],
    )


#: The constructor kwargs the T-Rex physics-r8 task revision added (plant_versions note 13), at the
#: legacy defaults every task that does not set them resolves to.
TREX_R8_TASK_KEYS = {
    "neck_posture_reference": "keyframe",
    "foot_flatness_weight": 0.0,
    "foot_flatness_tolerance_deg": 3.0,
    "stance_width_weight": 0.0,
    "stance_width_tolerance_m": 0.05,
}
#: And the ones the stance follow-up added (decision D-D27), at their legacy defaults.
TREX_FOLLOWUP_TASK_KEYS = {
    "stance_width_reference": "keyframe",
    "stance_width_settle_steps": 200,
    "foot_terms_min_support_force": 0.0,
    "floor_impact_weight": 0.0,
    "floor_impact_threshold_bw": 1.4,
    "airborne_substep_weight": 0.0,
    "action_penalty_source": "filtered",
}


#: The stance-quality constructor kwargs CompsognathusBiologicalEnv added with the anatomical physics-r2 /
#: policy-interface-r3 revision (plant_versions note 15, D-D26), at the inert defaults every task that does not
#: set them resolves to.  The robot's constructor is CompsognathusEnv's, unchanged.
COMPSOGNATHUS_R2_TASK_KEYS = {
    "bilateral_support_weight": 0.0,
    "foot_contact_saturation_force": 4.4,
    "support_force_aggregation": "min",
    "support_conditioned_alive_fraction": 0.0,
    "foot_flatness_weight": 0.0,
    "foot_flatness_tolerance_deg": 2.0,
    "stance_width_weight": 0.0,
    "stance_width_tolerance_m": 0.03,
    "stance_width_reference": "settled",
    "stance_width_settle_steps": 200,
    "foot_terms_min_support_force": 0.0,
    "leg_home_pose_weight": 0.0,
    "leg_home_pose_tolerance": 0.15,
    "support_geom_coverage_weight": 0.0,
    "floor_impact_weight": 0.0,
    "floor_impact_threshold_bw": 2.5,
    "airborne_substep_weight": 0.0,
    "action_saturation_weight": 0.0,
    "action_saturation_threshold": 0.9,
    "action_jerk_weight": 0.0,
}
#: Species whose recorded tasks a later plant revision retired, with the constructor kwargs that revision added.
RETIRED_TASK_KEYS = {
    "trex": {**TREX_R8_TASK_KEYS, **TREX_FOLLOWUP_TASK_KEYS},
    "compsognathus": COMPSOGNATHUS_R2_TASK_KEYS,
}


@pytest.mark.parametrize("record", FIXTURE["fingerprints"], ids=lambda record: f"{record['species']}-{record['stage']}")
def test_pre_recovery_quiet_checkpoint_identity_is_preserved(record):
    current = recompute(record)
    if record["species"] in RETIRED_TASK_KEYS:
        # The trex records are physics-r7 tasks, retired with every trex checkpoint by physics r8 (its task
        # revision and the D-D27 stance follow-up added constructor kwargs), and the anatomical compsognathus
        # records physics-r1 tasks, retired by its physics r2: each added kwarg enters each task at its legacy
        # default (no carve-out exists for them), and nothing else in the payload moved.  The robot's records
        # stay exact below.
        added = RETIRED_TASK_KEYS[record["species"]]
        assert {key: value for key, value in current["env"].items() if key not in record["env"]} == added
        assert {key: value for key, value in current["env"].items() if key in record["env"]} == record["env"]
        unchanged = {key: value for key, value in record.items() if key not in ("env", "task_sha256")}
        assert {key: value for key, value in current.items() if key not in ("env", "task_sha256")} == unchanged
        assert current["task_sha256"] != record["task_sha256"]
        return
    assert current == record
    assert validate_recorded_task(record, current, mode="resume_same_stage") is None


@pytest.mark.parametrize("record", FIXTURE["fingerprints"][:6])
def test_real_quiet_task_changes_still_refuse_resume(record):
    changed = recompute(record, alive_bonus=record["env"]["alive_bonus"] + 0.5)
    with pytest.raises(TaskFingerprintError, match="env"):
        validate_recorded_task(record, changed, mode="resume_same_stage")


@pytest.mark.parametrize("record", FIXTURE["fingerprints"][:6])
@pytest.mark.parametrize("multiple", [0.0, 1.0])
def test_explicit_push_settings_remain_part_of_task_identity(record, multiple):
    changed = recompute(record, perturbation_capture_velocity_multiple=multiple)
    assert changed["env"]["perturbation_capture_velocity_multiple"] == multiple
    assert changed["task_sha256"] != record["task_sha256"]
