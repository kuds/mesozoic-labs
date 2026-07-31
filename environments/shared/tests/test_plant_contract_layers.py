"""Tests for the four plant-contract layer payloads.

Each case perturbs one aspect of the model and asserts which layer
fingerprints move and which stay put — the separation is the contract, so
the assertions necessarily span source, policy, physics, and visual."""

from __future__ import annotations

from dataclasses import replace

import mujoco
import numpy as np
import pytest

from environments.brachiosaurus.envs.brachio_env import BrachioEnv
from environments.shared.plant_contract import (
    GENERATED_MANIFEST_PATH,
    PlantVersion,
    fingerprint_model_layers,
    load_plant_versions,
)
from environments.shared.plant_contract.physics_layer import _option_payload
from environments.shared.plant_contract.policy_layer import _policy_interface_payload
from environments.trex.envs.trex_env import TRexEnv
from environments.velociraptor.envs.raptor_env import RaptorEnv


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
