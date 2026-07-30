"""Tests for the layered MuJoCo plant-safety contract."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import replace
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

import environments.shared.plant_contract as plant_contract
from environments.brachiosaurus.envs.brachio_env import BrachioEnv
from environments.shared.plant_contract import (
    BUNDLED_MANIFEST_PATH,
    GENERATED_MANIFEST_PATH,
    MODEL_IDENTITY_ATTRIBUTE,
    PlantCompatibilityError,
    PlantContractError,
    PlantVersion,
    attach_plant_identity,
    check_plant_manifest,
    current_plant_identity,
    fingerprint_model_layers,
    load_plant_versions,
    validate_compiled_plant,
    validate_environment_plant,
    validate_mjx_environment_plant,
    validate_model_plant,
    validate_recorded_identity,
    write_plant_identity,
)
from environments.shared.plant_contract.digests import (
    _canonical_float,
    _normalized_python_tokens,
    _semantic_digest,
)
from environments.shared.plant_contract.manifest import _enforce_revision_changes
from environments.shared.plant_contract.physics_layer import _option_payload
from environments.shared.plant_contract.policy_layer import _policy_interface_payload
from environments.shared.plant_contract.source_layer import _source_payload
from environments.trex.envs.trex_env import TRexEnv
from environments.velociraptor.envs.raptor_env import RaptorEnv

CANONICAL_MUJOCO_VERSION = load_plant_versions()[0]


@pytest.fixture(scope="module")
def raptor_layers():
    """Return original model source, interface shell, version, and hashes."""
    env = RaptorEnv(reset_noise_scale=0.0)
    try:
        source = (GENERATED_MANIFEST_PATH.parent.parent / "environments/velociraptor/assets/raptor.xml").read_text()
        interface = env
        version = PlantVersion(
            species="velociraptor",
            physics_revision=1,
            policy_interface_revision=1,
            visual_revision=1,
            observation_schema="bipedal-target/v1",
        )
        layers = fingerprint_model_layers(env.model, interface, version)
        yield source, interface, version, layers
    finally:
        env.close()


@pytest.mark.skipif(
    mujoco.__version__ != CANONICAL_MUJOCO_VERSION,
    reason="exact manifest checks run only with the canonical MuJoCo version",
)
def test_committed_manifest_is_current_and_covers_all_species():
    manifest = check_plant_manifest()

    assert manifest["fingerprint_tool_version"] == plant_contract.FINGERPRINT_TOOL_VERSION == 2
    assert manifest["generated_with"]["float_significant_digits"] == 12
    assert set(manifest["plants"]) == {"velociraptor", "trex", "brachiosaurus", "dibothrosuchus"}
    for entry in manifest["plants"].values():
        assert entry["policy_interface"]["revision"] >= 1
        assert entry["physics"]["revision"] >= 1
        assert entry["visual"]["revision"] >= 1


@pytest.mark.parametrize("value", [0.1, -0.1, 1.234567890123, 1e-9, 1e9, -math.pi])
def test_portable_float_canonicalization_collapses_compiler_ulp_noise(value):
    expected = _canonical_float(value)

    for direction in (-math.inf, math.inf):
        perturbed = value
        for _ in range(4):
            perturbed = math.nextafter(perturbed, direction)
        assert _canonical_float(perturbed) == expected


def test_portable_float_canonicalization_preserves_meaningful_changes():
    assert _canonical_float(1.0) != _canonical_float(1.000000001)

    base = np.array([0.1, -math.pi, 1e-9, 1e9], dtype=np.float64)
    perturbed = base.copy()
    directions = np.where(perturbed < 0.0, -np.inf, np.inf)
    for _ in range(4):
        perturbed = np.nextafter(perturbed, directions)

    assert _semantic_digest("portable-array-test/v1", base) == _semantic_digest("portable-array-test/v1", perturbed)

    changed = base.copy()
    changed[0] *= 1.0 + 1e-9
    assert _semantic_digest("portable-array-test/v1", base) != _semantic_digest("portable-array-test/v1", changed)


def test_portable_float_canonicalization_handles_special_values():
    assert _canonical_float(0.0) == _canonical_float(-0.0) == "0"
    assert _canonical_float(math.inf) == "+inf"
    assert _canonical_float(-math.inf) == "-inf"
    with pytest.raises(PlantContractError, match="cannot encode NaN"):
        _canonical_float(math.nan)


def test_source_closure_remains_byte_exact(monkeypatch, tmp_path):
    model_path = tmp_path / "model.xml"
    model_path.write_text('<mujoco model="test"><worldbody/></mujoco>\n', encoding="utf-8")
    monkeypatch.setattr(plant_contract.constants, "REPOSITORY_ROOT", tmp_path)

    original = _source_payload(model_path)
    original_digest = _semantic_digest(plant_contract.SOURCE_SCHEMA, original)
    model_path.write_text('<mujoco model="test"><!-- byte change --><worldbody/></mujoco>\n', encoding="utf-8")
    changed = _source_payload(model_path)
    changed_digest = _semantic_digest(plant_contract.SOURCE_SCHEMA, changed)

    assert original["dependencies"][0]["content_sha256"] != changed["dependencies"][0]["content_sha256"]
    assert original_digest != changed_digest


def test_bundled_runtime_manifest_matches_repository_manifest():
    assert BUNDLED_MANIFEST_PATH.read_bytes() == GENERATED_MANIFEST_PATH.read_bytes()


def test_runtime_identity_falls_back_to_bundled_manifest(monkeypatch, tmp_path):
    monkeypatch.setattr(plant_contract.constants, "SPECIES_MANIFEST_PATH", tmp_path / "missing-species.toml")
    monkeypatch.setattr(plant_contract.constants, "PLANT_VERSIONS_PATH", tmp_path / "missing-versions.toml")
    monkeypatch.setattr(plant_contract.constants, "GENERATED_MANIFEST_PATH", tmp_path / "missing-manifest.json")

    identity = plant_contract.current_plant_identity("velociraptor")

    assert identity.species == "velociraptor"
    assert identity.nu == identity.action_dim == 22


@pytest.mark.parametrize(
    ("species", "observation_dim", "action_dim"),
    [("velociraptor", 67, 22), ("trex", 61, 21), ("brachiosaurus", 83, 30)],
)
def test_current_identity_matches_each_executable_environment(species, observation_dim, action_dim):
    # Identity generation requires exact SB3/MJX observation parity.
    identity = current_plant_identity(species, verify_generated=False)

    assert identity.observation_dim == observation_dim
    assert identity.action_dim == identity.nu == action_dim


def test_visual_only_material_change_only_changes_visual_layer(raptor_layers):
    source, interface, version, original = raptor_layers
    changed_source = source.replace('rgba="0.6 0.5 0.4 1"', 'rgba="0.55 0.48 0.38 1"', 1)
    assert changed_source != source
    changed_model = mujoco.MjModel.from_xml_string(changed_source)

    changed = fingerprint_model_layers(changed_model, interface, version)

    assert changed["policy_interface_sha256"] == original["policy_interface_sha256"]
    assert changed["physics_sha256"] == original["physics_sha256"]
    assert changed["visual_sha256"] != original["visual_sha256"]


def test_mass_change_changes_physics_without_changing_interface(raptor_layers):
    source, interface, version, original = raptor_layers
    changed_source = source.replace('mass="4.0" material="body_mat"', 'mass="4.1" material="body_mat"', 1)
    assert changed_source != source
    changed_model = mujoco.MjModel.from_xml_string(changed_source)

    changed = fingerprint_model_layers(changed_model, interface, version)

    assert changed["policy_interface_sha256"] == original["policy_interface_sha256"]
    assert changed["physics_sha256"] != original["physics_sha256"]


def test_home_control_change_updates_policy_and_physics_fingerprints(raptor_layers):
    source, interface, version, original = raptor_layers
    changed_source = source.replace('ctrl="0.663225', 'ctrl="0.650000', 1)
    assert changed_source != source
    changed_model = mujoco.MjModel.from_xml_string(changed_source)

    changed = fingerprint_model_layers(changed_model, interface, version)

    assert changed["policy_interface_sha256"] != original["policy_interface_sha256"]
    assert changed["physics_sha256"] != original["physics_sha256"]
    assert changed["visual_sha256"] == original["visual_sha256"]


def test_joint_actuator_force_cap_changes_physics_fingerprint(raptor_layers):
    source, interface, version, original = raptor_layers
    changed_source = source.replace(
        'name="r_hip_pitch" class="leg_joint" type="hinge"',
        'name="r_hip_pitch" class="leg_joint" type="hinge" actuatorfrcrange="-5 5"',
        1,
    )
    assert changed_source != source
    changed_model = mujoco.MjModel.from_xml_string(changed_source)
    joint_id = mujoco.mj_name2id(changed_model, mujoco.mjtObj.mjOBJ_JOINT, "r_hip_pitch")
    assert changed_model.jnt_actfrclimited[joint_id]

    changed = fingerprint_model_layers(changed_model, interface, version)

    assert changed["policy_interface_sha256"] == original["policy_interface_sha256"]
    assert changed["physics_sha256"] != original["physics_sha256"]
    assert changed["visual_sha256"] == original["visual_sha256"]


def test_active_contact_override_option_changes_physics_fingerprint(raptor_layers):
    source, interface, version, _original = raptor_layers
    base_source = source.replace(
        '<flag warmstart="enable"/>',
        '<flag warmstart="enable" override="enable"/>',
        1,
    )
    option = '<option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast">'
    changed_source = base_source.replace(option, option[:-1] + ' o_margin="0.1">', 1)
    base = fingerprint_model_layers(mujoco.MjModel.from_xml_string(base_source), interface, version)
    changed = fingerprint_model_layers(mujoco.MjModel.from_xml_string(changed_source), interface, version)

    assert changed["policy_interface_sha256"] == base["policy_interface_sha256"]
    assert changed["physics_sha256"] != base["physics_sha256"]
    assert changed["visual_sha256"] == base["visual_sha256"]


def test_all_public_mujoco_options_are_fingerprinted(raptor_layers):
    _source, interface, _version, _original = raptor_layers
    expected = {
        name
        for name in dir(interface.model.opt)
        if not name.startswith("_") and not callable(getattr(interface.model.opt, name))
    }

    assert set(_option_payload(interface.model)) == expected


def test_heightfield_samples_change_physics_and_visual_fingerprints(raptor_layers):
    source, interface, version, _original = raptor_layers
    base_source = source.replace(
        "<asset>",
        '<asset>\n    <hfield name="terrain" nrow="2" ncol="2" size="50 50 1 0.1" elevation="0 0 0 0"/>',
        1,
    ).replace(
        '<geom name="floor" type="plane" size="50 50 0.1" material="grid_mat" conaffinity="1" contype="1"/>',
        '<geom name="floor" type="hfield" hfield="terrain" size="50 50 0.1" '
        'material="grid_mat" conaffinity="1" contype="1"/>',
        1,
    )
    changed_source = base_source.replace('elevation="0 0 0 0"', 'elevation="0 0 0 0.5"', 1)
    base = fingerprint_model_layers(mujoco.MjModel.from_xml_string(base_source), interface, version)
    changed = fingerprint_model_layers(mujoco.MjModel.from_xml_string(changed_source), interface, version)

    assert changed["policy_interface_sha256"] == base["policy_interface_sha256"]
    assert changed["physics_sha256"] != base["physics_sha256"]
    assert changed["visual_sha256"] != base["visual_sha256"]


def test_noncolliding_fluid_geom_changes_physics_fingerprint():
    env = BrachioEnv(reset_noise_scale=0.0)
    try:
        source = (
            GENERATED_MANIFEST_PATH.parent.parent / "environments/brachiosaurus/assets/brachiosaurus.xml"
        ).read_text()
        option = '<option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast">'
        base_source = source.replace(option, option[:-1] + ' density="1">', 1)
        changed_source = base_source.replace(
            'mass="8.0" material="body_mat" contype="0" conaffinity="0"/>',
            'mass="8.0" material="body_mat" contype="0" conaffinity="0" fluidshape="ellipsoid" fluidcoef="1 1 1 1 1"/>',
            1,
        )
        version = PlantVersion(
            species="brachiosaurus",
            physics_revision=1,
            policy_interface_revision=1,
            visual_revision=1,
            observation_schema="quadrupedal-target/v1",
        )
        base = fingerprint_model_layers(mujoco.MjModel.from_xml_string(base_source), env, version)
        changed = fingerprint_model_layers(mujoco.MjModel.from_xml_string(changed_source), env, version)
    finally:
        env.close()

    assert changed["policy_interface_sha256"] == base["policy_interface_sha256"]
    assert changed["physics_sha256"] != base["physics_sha256"]
    assert changed["visual_sha256"] == base["visual_sha256"]


def test_camera_resolution_changes_only_visual_fingerprint(raptor_layers):
    source, interface, version, _original = raptor_layers
    base_source = source.replace(
        "<worldbody>",
        '<worldbody>\n    <camera name="contract_camera" pos="0 0 2" resolution="640 480"/>',
        1,
    )
    changed_source = base_source.replace('resolution="640 480"', 'resolution="800 600"', 1)
    base = fingerprint_model_layers(mujoco.MjModel.from_xml_string(base_source), interface, version)
    changed = fingerprint_model_layers(mujoco.MjModel.from_xml_string(changed_source), interface, version)

    assert changed["policy_interface_sha256"] == base["policy_interface_sha256"]
    assert changed["physics_sha256"] == base["physics_sha256"]
    assert changed["visual_sha256"] != base["visual_sha256"]


def test_control_range_change_changes_interface_and_physics(raptor_layers):
    source, interface, version, original = raptor_layers
    changed_source = source.replace('ctrlrange="-1.0472 1.5708"', 'ctrlrange="-1.0 1.5708"', 1)
    assert changed_source != source
    changed_model = mujoco.MjModel.from_xml_string(changed_source)

    changed = fingerprint_model_layers(changed_model, interface, version)

    assert changed["policy_interface_sha256"] != original["policy_interface_sha256"]
    assert changed["physics_sha256"] != original["physics_sha256"]


def test_touch_site_shape_change_changes_interface_and_visual_layers(raptor_layers):
    source, interface, version, original = raptor_layers
    changed_source = source.replace(
        '<site name="r_foot" pos="0.05 0 0" size="0.08" group="4"/>',
        '<site name="r_foot" pos="0.05 0 0" size="0.085" group="4"/>',
        1,
    )
    assert changed_source != source
    changed_model = mujoco.MjModel.from_xml_string(changed_source)

    changed = fingerprint_model_layers(changed_model, interface, version)

    assert changed["policy_interface_sha256"] != original["policy_interface_sha256"]
    assert changed["physics_sha256"] == original["physics_sha256"]
    assert changed["visual_sha256"] != original["visual_sha256"]


def test_interface_code_tokens_ignore_formatting_but_detect_semantic_changes():
    original = """\
def normalize(distance):
    # Formatting and comments are not interface semantics.
    return distance / (distance + 1e-8)
"""
    reformatted = """\
def normalize(distance):
    return distance / (distance + 1e-8)  # same calculation
"""
    changed = original.replace("1e-8", "1e-6")

    assert _normalized_python_tokens(original) == _normalized_python_tokens(reformatted)
    assert _normalized_python_tokens(original) != _normalized_python_tokens(changed)

    documented = '''\
def label(value):
    """Documentation does not execute."""
    return {"kind": "distance"}["kind"], value
'''
    requoted = """\
def label(value):
    '''Different documentation is also ignored.'''
    return {'kind': 'distance'}['kind'], value
"""
    assert _normalized_python_tokens(documented) == _normalized_python_tokens(requoted)


def test_interface_code_tokens_reject_cross_version_f_strings():
    with pytest.raises(PlantContractError, match="f-strings are not supported"):
        _normalized_python_tokens('def label(value):\n    return f"value={value}"\n')


def test_cached_observation_body_mapping_changes_interface_fingerprint(raptor_layers):
    _source, env, version, original = raptor_layers
    original_pelvis_id = env.pelvis_id
    env.pelvis_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "head")
    try:
        changed = fingerprint_model_layers(env.model, env, version)
    finally:
        env.pelvis_id = original_pelvis_id

    assert changed["policy_interface_sha256"] != original["policy_interface_sha256"]
    assert changed["physics_sha256"] == original["physics_sha256"]
    assert changed["visual_sha256"] == original["visual_sha256"]


@pytest.mark.parametrize(("env_class", "species"), ((RaptorEnv, "velociraptor"), (TRexEnv, "trex")))
def test_biped_policy_contract_records_home_residual_action_mapping(env_class, species):
    env = env_class(reset_noise_scale=0.0)
    try:
        version = load_plant_versions()[1][species]
        payload = _policy_interface_payload(env.model, env, version, require_backend_parity=True)
        mapping = payload["action_mapping"]

        assert mapping["mode"] == "home-keyframe-residual/v1"
        assert mapping["origin"]["keyframe"] == "home"
        np.testing.assert_allclose(mapping["origin"]["ctrl"], env.model.key_ctrl[env.home_keyframe_id])
        assert set(payload["interface_implementations"]["jax_action_mapping"]) == {"scale_action_around_nominal_jax"}
        assert set(payload["interface_implementations"]["home_reset"]["jax"]) == {"reset_mujoco_data_to_home"}
        assert payload["jax_interface"]["action_mapping"] == "home-keyframe-residual/v1"
    finally:
        env.close()


def test_brachio_retains_midpoint_action_mapping_contract():
    env = BrachioEnv(reset_noise_scale=0.0)
    try:
        version = load_plant_versions()[1]["brachiosaurus"]
        payload = _policy_interface_payload(env.model, env, version, require_backend_parity=True)
    finally:
        env.close()

    assert payload["action_mapping"] == "clip[-1,1]-then-affine-to-ordered-ctrlrange/v1"
    assert set(payload["interface_implementations"]["jax_action_mapping"]) == {"scale_action_jax"}
    assert "action_mapping" not in payload["jax_interface"]


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


def test_mjx_runtime_binding_rejects_interface_override(raptor_layers):
    _source, interface, _version, _original = raptor_layers
    current = current_plant_identity("velociraptor", verify_generated=False)
    config = SimpleNamespace(
        species="velociraptor",
        frame_skip=5,
        body_ids={"pelvis": 2},
        sensor_foot_indices=(10, 11),
        sensor_gyro_start=0,
        sensor_accel_start=3,
        sensor_quat_start=6,
        action_mapping="home-keyframe-residual/v1",
    )
    runtime_env = SimpleNamespace(mj_model=interface.model, config=config, action_dim=22)
    validate_mjx_environment_plant(runtime_env, current)

    config.frame_skip = 4
    with pytest.raises(PlantCompatibilityError, match="frame_skip"):
        validate_mjx_environment_plant(runtime_env, current)

    config.frame_skip = 5
    config.action_mapping = "midpoint/v1"
    with pytest.raises(PlantCompatibilityError, match="action_mapping"):
        validate_mjx_environment_plant(runtime_env, current)


def test_human_revision_counters_do_not_change_semantic_fingerprints(raptor_layers):
    _source, interface, version, original = raptor_layers
    env = RaptorEnv(reset_noise_scale=0.0)
    try:
        revised = fingerprint_model_layers(
            env.model,
            interface,
            replace(
                version,
                physics_revision=version.physics_revision + 1,
                policy_interface_revision=version.policy_interface_revision + 1,
                visual_revision=version.visual_revision + 1,
            ),
        )
    finally:
        env.close()

    assert revised == original


def test_changed_layer_requires_revision_increment():
    old = {
        "plants": {
            "velociraptor": {
                "policy_interface": {"sha256": "interface-a", "revision": 1},
                "physics": {"sha256": "physics-a", "revision": 1},
                "visual": {"sha256": "visual-a", "revision": 1},
            }
        }
    }
    new = copy.deepcopy(old)
    new["plants"]["velociraptor"]["physics"]["sha256"] = "physics-b"

    with pytest.raises(PlantContractError, match="physics fingerprint changed"):
        _enforce_revision_changes(old, new)

    new["plants"]["velociraptor"]["physics"]["revision"] = 2
    _enforce_revision_changes(old, new)


def test_layer_revision_cannot_roll_back_when_fingerprint_is_unchanged():
    old = {
        "plants": {
            "velociraptor": {
                "policy_interface": {"sha256": "interface-a", "revision": 2},
                "physics": {"sha256": "physics-a", "revision": 2},
                "visual": {"sha256": "visual-a", "revision": 2},
            }
        }
    }
    new = copy.deepcopy(old)
    new["plants"]["velociraptor"]["visual"]["revision"] = 1

    with pytest.raises(PlantContractError, match="visual_revision cannot decrease"):
        _enforce_revision_changes(old, new)


@pytest.mark.skipif(
    mujoco.__version__ != CANONICAL_MUJOCO_VERSION,
    reason="exact manifest checks run only with the canonical MuJoCo version",
)
def test_manifest_check_enforces_revisions_against_external_baseline(tmp_path):
    baseline = json.loads(GENERATED_MANIFEST_PATH.read_text(encoding="utf-8"))
    baseline["plants"]["velociraptor"]["physics"]["sha256"] = "sha256:" + "0" * 64
    baseline_path = tmp_path / "base-plant-manifest.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    with pytest.raises(PlantContractError, match="physics fingerprint changed"):
        check_plant_manifest(baseline_path=baseline_path)


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
