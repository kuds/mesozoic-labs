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


@pytest.mark.parametrize("record", FIXTURE["fingerprints"], ids=lambda record: f"{record['species']}-{record['stage']}")
def test_pre_recovery_quiet_checkpoint_identity_is_preserved(record):
    current = recompute(record)
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
