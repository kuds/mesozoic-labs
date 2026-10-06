"""Tests for the floor-truth morphology: the explicit six-species registry, and the measurement manifest.

The anchors are the current plants (``61a3424``, mujoco 3.10.0): body weights
840.91 / 132.44 / 9.81 / 15.55 / 84.86 / 1719.69 N, every sole frame normal
``-z`` and forward ``+x``, brachiosaurus' authored sole tilt 19.22 degrees
(fore) and 15.78 (hind).  The registry is explicit for all six species, and
:data:`GENERIC_ONLY` / :data:`REGISTRY_ONLY` name every place it departs from
the generic rule -- the test fails if a departure appears, or disappears,
without this table changing with it.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from environments.shared.config import load_stage_config
from environments.shared.gait import constants
from environments.shared.gait.constants import (
    MEASUREMENT_MANIFEST_SCHEMA,
    MEASUREMENT_VERSION,
    measurement_constants,
    measurement_manifest,
    measurement_sha256,
)
from environments.shared.gait.morphology import SUPPORT_REGISTRY, FootRegistryEntry, Morphology, MorphologyError
from environments.shared.record_fields import is_sha256_digest
from environments.shared.result_bundle.hashing import canonical_json_sha256
from environments.shared.species_registry import get_species_config

SPECIES = ("trex", "velociraptor", "compsognathus", "compsognathus_robot", "dibothrosuchus", "brachiosaurus")
BODY_WEIGHT_N = {
    "trex": 840.91,
    "velociraptor": 132.44,
    "compsognathus": 9.81,
    "compsognathus_robot": 15.55,
    "dibothrosuchus": 84.86,
    "brachiosaurus": 1719.69,
}
FEET = {
    "trex": ("r", "l"),
    "velociraptor": ("r", "l"),
    "compsognathus": ("r", "l"),
    "compsognathus_robot": ("r", "l"),
    "dibothrosuchus": ("fr", "fl", "rr", "rl"),
    "brachiosaurus": ("fr", "fl", "rr", "rl"),
}
ROOT = {"dibothrosuchus": "torso", "brachiosaurus": "torso"}  # pelvis elsewhere

#: Per foot, the support geoms the generic rule takes and the registry drops ...
GENERIC_ONLY = {
    "trex": ("{s}_metatarsus_geom",),  # the capsule above the plantar box: 0 statue load
    "velociraptor": (),
    "compsognathus": (),
    "compsognathus_robot": ("{side}_roll_cheek_-1", "{side}_roll_cheek_1"),  # a cheek down is a rolled foot
    "dibothrosuchus": (),
    "brachiosaurus": ("{s}_meta_geom",),  # the metapodial capsule in the touch group: 0 statue load
}
#: ... and the ones the registry adds that the generic rule misses.
REGISTRY_ONLY = {
    "trex": (),
    "velociraptor": ("{s}_toe_d4_geom", "{s}_metatarsus_geom"),  # 0.18 and 0.26 of the statue's foot load
    "compsognathus": (),
    "compsognathus_robot": (),
    "dibothrosuchus": (),
    "brachiosaurus": (),
}
SOLE = {
    "trex": "{s}_plantar_geom",
    "velociraptor": None,
    "compsognathus": "{s}_plantar_pad",
    "compsognathus_robot": "{side}_sole",
    "dibothrosuchus": "{s}_foot_geom",
    "brachiosaurus": "{s}_foot_geom",
}


def _names(templates, label):
    side = {"r": "right", "l": "left"}[label[-1]]
    return {template.format(s=label, side=side) for template in templates}


@pytest.fixture(scope="module")
def stance_env():
    """One stance env per species, built on first use and shared by the module's tests."""
    built = {}

    def get(species):
        if species not in built:
            built[species] = get_species_config(species).env_class(
                **dict(load_stage_config(species, "stance")["env_kwargs"])
            )
        return built[species]

    yield get
    for env in built.values():
        env.close()


# ── the registry ─────────────────────────────────────────────────────────────


def test_every_species_has_an_explicit_registry_entry_with_a_pinned_reference_site():
    assert set(SUPPORT_REGISTRY) == set(SPECIES)
    for species, entry in SUPPORT_REGISTRY.items():
        assert isinstance(entry, FootRegistryEntry)
        assert entry.support and entry.foot_site, species
        assert entry.sole == SOLE[species]
    with pytest.raises(TypeError):
        SUPPORT_REGISTRY["trex"] = SUPPORT_REGISTRY["velociraptor"]  # type: ignore[index]


@pytest.mark.parametrize("species", SPECIES)
def test_the_registry_departs_from_the_generic_rule_exactly_where_documented(stance_env, species):
    env = stance_env(species)
    registered = Morphology.from_env(env, species)
    generic = Morphology.from_env(env, species, registry={})
    assert registered.registry_entry is SUPPORT_REGISTRY[species]
    assert generic.registry_entry is None
    assert registered.labels == generic.labels == FEET[species]
    described, generic_described = registered.describe(env.model), generic.describe(env.model)
    for foot, generic_foot in zip(described["feet"], generic_described["feet"]):
        label = foot["label"]
        support, generic_support = set(foot["support_geoms"]), set(generic_foot["support_geoms"])
        assert generic_support - support == _names(GENERIC_ONLY[species], label), label
        assert support - generic_support == _names(REGISTRY_ONLY[species], label), label
        # Every registered geom is a colliding geom of that foot's leg, and the sole one of them.
        assert support <= set(foot["leg_geoms"])
        assert foot["sole_geom"] == (None if SOLE[species] is None else _names([SOLE[species]], label).pop())
        # The registry pins the reference site and the sole the generic rule would choose anyway.
        assert foot["reference_site"] == generic_foot["reference_site"]
        assert foot["sole_geom"] == generic_foot["sole_geom"]


@pytest.mark.parametrize("species", SPECIES)
def test_legs_are_whole_limbs_rooted_below_the_animal_root(stance_env, species):
    env = stance_env(species)
    morphology = Morphology.from_env(env, species)
    model = env.model
    assert model.body(morphology.root_body).name == ROOT.get(species, "pelvis")
    tops = []
    for foot in morphology.feet:
        (top,) = [body for body in foot.leg_bodies if int(model.body_parentid[body]) not in foot.leg_bodies]
        assert int(model.body_parentid[top]) == morphology.root_body
        tops.append(model.body(top).name)
    if species == "compsognathus_robot":
        assert tops == ["hip_roll_assembly_2", "hip_roll_assembly"]
    else:
        assert tops == [f"{label}_thigh" for label in FEET[species]]
    legs = [set(foot.leg_geoms) for foot in morphology.feet]
    assert all(not (a & b) for i, a in enumerate(legs) for b in legs[i + 1 :])
    assert set().union(*legs) <= set(morphology.animal_geoms)
    assert not set(morphology.floor_geoms) & set(morphology.animal_geoms)


@pytest.mark.parametrize("species", SPECIES)
def test_body_weight_is_the_root_subtree_weight(stance_env, species):
    env = stance_env(species)
    morphology = Morphology.from_env(env, species)
    assert morphology.body_weight_n == pytest.approx(BODY_WEIGHT_N[species], abs=0.01)
    assert morphology.frame_skip == env.frame_skip and morphology.dt == env.dt
    assert morphology.n_feet == len(FEET[species]) and morphology.is_quadruped == (len(FEET[species]) == 4)


@pytest.mark.parametrize("species", [species for species in SPECIES if SOLE[species] is not None])
def test_sole_frames_face_the_floor_and_point_forward(stance_env, species):
    morphology = Morphology.from_env(stance_env(species), species)
    for foot in morphology.feet:
        assert (foot.sole_normal_axis, foot.sole_normal_sign) == (2, -1.0), foot.label
        assert (foot.sole_forward_axis, foot.sole_forward_sign) == (0, 1.0), foot.label
        assert (foot.sole_lateral_axis, foot.sole_lateral_sign) == (1, 1.0), foot.label
        assert foot.side == foot.label[-1].upper()
    tilts = [foot.home_sole_tilt_deg for foot in morphology.feet]
    if species == "brachiosaurus":
        assert tilts == pytest.approx([19.22, 19.22, 15.78, 15.78], abs=0.01)
    else:
        assert tilts == pytest.approx([0.0] * len(tilts), abs=1e-4)  # acos amplifies rounding near flat


def test_a_foot_without_a_sole_has_no_sole_frame(stance_env):
    for foot in Morphology.from_env(stance_env("velociraptor"), "velociraptor").feet:
        assert foot.sole_geom == foot.sole_type == -1
        assert np.isnan(foot.home_sole_tilt_deg)


def test_an_alias_reaches_the_registry_and_an_unknown_name_gets_the_generic_rule(stance_env):
    env = stance_env("velociraptor")
    assert Morphology.from_env(env, "raptor").registry_entry is SUPPORT_REGISTRY["velociraptor"]
    unknown = Morphology.from_env(env, "not-a-species")
    assert unknown.species == "not-a-species" and unknown.registry_entry is None
    assert [len(foot.support_geoms) for foot in unknown.feet] == [1, 1]  # toe_d3 alone


def test_building_a_morphology_leaves_the_env_untouched(stance_env):
    env = stance_env("trex")
    env.reset(seed=3042)
    for _ in range(3):
        env.step(np.zeros(env.model.nu))
    before = (env.data.qpos.copy(), env.data.qvel.copy(), env.data.time, env._substep_probe_hook)
    morphology = Morphology.from_env(env, "trex")
    np.testing.assert_array_equal(env.data.qpos, before[0])
    np.testing.assert_array_equal(env.data.qvel, before[1])
    assert env.data.time == before[2] and env._substep_probe_hook is before[3]
    for table in (morphology.geom_foot, morphology.geom_support, morphology.is_floor, morphology.is_animal):
        assert table.shape == (env.model.ngeom,) and not table.flags.writeable


# ── refusals ─────────────────────────────────────────────────────────────────

_TREX = SUPPORT_REGISTRY["trex"]


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        (replace(_TREX, support=(*_TREX.support, "{s}_no_such_geom")), "is not in the model"),
        (replace(_TREX, support=(*_TREX.support, "l_toe_d2_geom")), "is not a colliding geom of leg"),
        (replace(_TREX, support=(*_TREX.support, "{s}_toe_d2_geom")), "listed twice"),
        (replace(_TREX, sole="{s}_metatarsus_geom"), "is not one of foot"),
        (replace(_TREX, foot_site=None), "pins no foot_site"),
        (replace(_TREX, foot_site="{s}_no_such_site"), "foot site"),
        (replace(_TREX, foot_site="l_foot"), "is not on leg"),
    ],
)
def test_a_bad_registry_entry_is_refused(stance_env, entry, message):
    with pytest.raises(MorphologyError, match=message):
        Morphology.from_env(stance_env("trex"), "trex", registry={"trex": entry})


def test_an_env_without_foot_sensors_is_refused(stance_env):
    env = stance_env("trex")
    groups = env._foot_sensor_groups
    try:
        env._foot_sensor_groups = ()
        with pytest.raises(MorphologyError, match="_foot_sensor_groups"):
            Morphology.from_env(env, "trex")
        env._foot_sensor_groups = ((10_000,),)
        with pytest.raises(MorphologyError, match="starts no sensor"):
            Morphology.from_env(env, "trex")
        env._foot_sensor_groups = (groups[0], groups[0])
        with pytest.raises(MorphologyError, match="share a label"):
            Morphology.from_env(env, "trex")
    finally:
        env._foot_sensor_groups = groups
    assert Morphology.from_env(env, "trex").n_feet == 2


def test_an_invalid_home_keyframe_is_refused(stance_env):
    env = stance_env("trex")
    keyframe = env._reset_keyframe_id
    env._reset_keyframe_id = env.model.nkey
    try:
        with pytest.raises(MorphologyError, match="keyframe"):
            Morphology.from_env(env, "trex")
    finally:
        env._reset_keyframe_id = keyframe


# ── the measurement manifest ─────────────────────────────────────────────────


def test_the_manifest_records_the_constants_the_registry_and_the_resolved_names(stance_env):
    env = stance_env("trex")
    manifest = measurement_manifest(Morphology.from_env(env, "trex"), env.model)
    assert manifest["schema"] == MEASUREMENT_MANIFEST_SCHEMA
    assert manifest["constants"] == measurement_constants()
    assert manifest["constants"] == {
        "measurement_version": MEASUREMENT_VERSION,
        "contact_threshold_n": 0.1,
        "down_substep_fraction": 0.5,
        "debounce_s": 0.02,
        "spawn_grace_s": 0.1,
        "load_window_s": 1.0,
        "saturation_abs": 0.99,
        "sole_rolled_deg": 2.0,
        "support_geom_down_fraction": 0.5,
    }
    assert MEASUREMENT_VERSION == "floor-truth/v1"
    assert (manifest["species"], manifest["frame_skip"], manifest["dt"]) == ("trex", 5, 0.01)
    assert manifest["body_weight_n"] == pytest.approx(840.91, abs=0.01)
    assert manifest["registry"] == {
        "support": list(_TREX.support),
        "sole": "{s}_plantar_geom",
        "foot_site": "{s}_foot",
    }
    right = manifest["feet"][0]
    assert right == {
        "label": "r",
        "sensors": ["r_foot_touch", "r_toe_d2_touch", "r_toe_d3_touch", "r_toe_d4_touch"],
        "reference_site": "r_foot",
        "support_geoms": ["r_plantar_geom", "r_toe_d2_geom", "r_toe_d3_geom", "r_toe_d4_geom"],
        "sole_geom": "r_plantar_geom",
        "sole_normal": "-z",
        "sole_forward": "+x",
        "sole_lateral": "+y",
    }
    generic = measurement_manifest(Morphology.from_env(env, "trex", registry={}), env.model)
    assert generic["registry"] is None


def test_the_manifest_digest_is_the_canonical_json_digest_and_survives_a_json_round_trip(stance_env):
    env = stance_env("compsognathus_robot")
    manifest = measurement_manifest(Morphology.from_env(env, "compsognathus_robot"), env.model)
    digest = measurement_sha256(manifest)
    assert is_sha256_digest(digest)
    assert digest == canonical_json_sha256(manifest)
    assert measurement_sha256(json.loads(json.dumps(manifest, indent=2))) == digest
    assert (
        measurement_sha256(measurement_manifest(Morphology.from_env(env, "compsognathus_robot"), env.model)) == digest
    )


def test_a_constant_or_a_registry_edit_moves_the_manifest_digest(stance_env, monkeypatch):
    env = stance_env("trex")
    morphology = Morphology.from_env(env, "trex")
    digest = measurement_sha256(measurement_manifest(morphology, env.model))
    monkeypatch.setattr(constants, "CONTACT_THRESHOLD_N", 0.2)
    assert measurement_sha256(measurement_manifest(morphology, env.model)) != digest
    monkeypatch.undo()
    assert measurement_sha256(measurement_manifest(morphology, env.model)) == digest
    fewer_toes = replace(_TREX, support=_TREX.support[:-1])
    edited = Morphology.from_env(env, "trex", registry={"trex": fewer_toes})
    assert measurement_sha256(measurement_manifest(edited, env.model)) != digest
    # Order matters too: the per-geom metric tuples (support_geom_duty, support_geom_share) are positional.
    reordered = replace(_TREX, support=tuple(reversed(_TREX.support)))
    moved = Morphology.from_env(env, "trex", registry={"trex": reordered})
    assert measurement_sha256(measurement_manifest(moved, env.model)) != digest


def test_the_manifest_rounds_away_cross_architecture_noise_in_the_body_weight(stance_env):
    env = stance_env("trex")
    morphology = Morphology.from_env(env, "trex")
    noisy = replace(morphology, body_weight_n=morphology.body_weight_n * (1 + 4e-16))
    assert noisy.body_weight_n != morphology.body_weight_n
    assert measurement_manifest(noisy, env.model) == measurement_manifest(morphology, env.model)
