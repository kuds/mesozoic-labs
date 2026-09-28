"""Tests for environments.shared.plant_contract.validation."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import mujoco
import pytest

from environments.shared.plant_contract import (
    MODEL_IDENTITY_ATTRIBUTE,
    PlantCompatibilityError,
    attach_plant_identity,
    current_plant_identity,
    validate_compiled_plant,
    validate_environment_plant,
    validate_model_plant,
    validate_recorded_identity,
    write_plant_identity,
)
from environments.velociraptor.envs.raptor_env import RaptorEnv


def test_runtime_environment_binding_rejects_control_cadence_override():
    current = current_plant_identity("velociraptor", verify_generated=False)
    env = RaptorEnv(frame_skip=4, reset_noise_scale=0.0)
    try:
        with pytest.raises(PlantCompatibilityError, match="policy_interface_sha256"):
            validate_environment_plant(env, current)
    finally:
        env.close()


def test_compiled_backend_binding_rejects_different_model(raptor_layers):
    source, _interface, _version, _original = raptor_layers
    changed_source = source.replace('mass="4.0" material="body_mat"', 'mass="4.1" material="body_mat"', 1)
    current = current_plant_identity("velociraptor", verify_generated=False)

    with pytest.raises(PlantCompatibilityError, match="physics_sha256"):
        validate_compiled_plant(mujoco.MjModel.from_xml_string(changed_source), current)


def test_missing_checkpoint_identity_fails_closed_unless_explicitly_allowed():
    current = current_plant_identity("velociraptor", verify_generated=False)

    with pytest.raises(PlantCompatibilityError, match="has no plant identity"):
        validate_recorded_identity(None, current)

    validate_recorded_identity(None, current, allow_legacy=True)


def test_malformed_checkpoint_identity_is_a_compatibility_error():
    current = current_plant_identity("velociraptor", verify_generated=False)
    recorded = current.to_dict()
    recorded["physics_sha256"] = "not-a-digest"

    with pytest.raises(PlantCompatibilityError, match="invalid plant identity metadata"):
        validate_recorded_identity(recorded, current)


def test_non_mapping_model_identity_is_not_treated_as_legacy():
    current = current_plant_identity("velociraptor", verify_generated=False)
    model = SimpleNamespace(**{MODEL_IDENTITY_ATTRIBUTE: "corrupt"})

    with pytest.raises(PlantCompatibilityError, match="invalid plant identity metadata"):
        validate_model_plant(model, current, allow_legacy=True)


def test_visual_and_source_changes_do_not_invalidate_policy_checkpoint():
    current = current_plant_identity("velociraptor", verify_generated=False)
    recorded = replace(
        current,
        visual_revision=current.visual_revision + 1,
        source_closure_sha256="sha256:" + "1" * 64,
        visual_sha256="sha256:" + "2" * 64,
    )

    validate_recorded_identity(recorded.to_dict(), current)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("physics_revision", 99, "physics_revision"),
        ("policy_interface_revision", 99, "policy_interface_revision"),
        ("physics_sha256", "sha256:" + "3" * 64, "physics_sha256"),
        ("policy_interface_sha256", "sha256:" + "4" * 64, "policy_interface_sha256"),
    ],
)
def test_policy_or_physics_mismatch_is_rejected(field, value, message):
    current = current_plant_identity("velociraptor", verify_generated=False)
    recorded = replace(current, **{field: value})

    with pytest.raises(PlantCompatibilityError, match=message):
        validate_recorded_identity(recorded.to_dict(), current)


def test_identity_attaches_to_model_and_round_trips_to_json(tmp_path):
    current = current_plant_identity("velociraptor", verify_generated=False)
    model = SimpleNamespace()

    attach_plant_identity(model, current)
    validate_model_plant(model, current)
    assert getattr(model, MODEL_IDENTITY_ATTRIBUTE)["physics_sha256"] == current.physics_sha256

    path = write_plant_identity(tmp_path / "plant_identity.json", current)
    assert json.loads(path.read_text()) == current.to_dict()
