"""Tests for the floor-truth morphology: the explicit six-model registry, and the measurement manifest.

The anchors are the current plants (``61a3424``, mujoco 3.10.0): body weights
840.91 / 132.44 / 9.81 / 15.55 / 84.86 / 1719.69 N, every sole frame normal
``-z`` and forward ``+x``, brachiosaurus' authored sole tilt 19.22 degrees
(fore) and 15.78 (hind).  The registry is explicit for all six models, and
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
from environments.shared.gait.morphology import FootRegistryEntry, Morphology, MorphologyError, support_entry
from environments.shared.model_id import ModelId
from environments.shared.record_fields import is_sha256_digest
from environments.shared.result_bundle.hashing import canonical_json_sha256
from environments.shared.species_names import model_ids
from environments.shared.species_registry import get_species_config

TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")
VELOCIRAPTOR_MONGOLIENSIS = ModelId("velociraptor_mongoliensis", "anatomical")
COMPSOGNATHUS_LONGIPES = ModelId("compsognathus_longipes", "anatomical")
COMPSOGNATHUS_LONGIPES_ROBOT = ModelId("compsognathus_longipes", "robot")
DIBOTHROSUCHUS_ELAPHROS = ModelId("dibothrosuchus_elaphros", "anatomical")
BRACHIOSAURUS_ALTITHORAX = ModelId("brachiosaurus_altithorax", "anatomical")
MODEL_IDS = (
    TYRANNOSAURUS_REX,
    VELOCIRAPTOR_MONGOLIENSIS,
    COMPSOGNATHUS_LONGIPES,
    COMPSOGNATHUS_LONGIPES_ROBOT,
    DIBOTHROSUCHUS_ELAPHROS,
    BRACHIOSAURUS_ALTITHORAX,
)
BODY_WEIGHT_N = {
    TYRANNOSAURUS_REX: 840.91,
    VELOCIRAPTOR_MONGOLIENSIS: 132.44,
    COMPSOGNATHUS_LONGIPES: 9.81,
    COMPSOGNATHUS_LONGIPES_ROBOT: 15.55,
    DIBOTHROSUCHUS_ELAPHROS: 84.86,
    BRACHIOSAURUS_ALTITHORAX: 1719.69,
}
FEET = {
    TYRANNOSAURUS_REX: ("r", "l"),
    VELOCIRAPTOR_MONGOLIENSIS: ("r", "l"),
    COMPSOGNATHUS_LONGIPES: ("r", "l"),
    COMPSOGNATHUS_LONGIPES_ROBOT: ("r", "l"),
    DIBOTHROSUCHUS_ELAPHROS: ("fr", "fl", "rr", "rl"),
    BRACHIOSAURUS_ALTITHORAX: ("fr", "fl", "rr", "rl"),
}
ROOT = {
    TYRANNOSAURUS_REX: "pelvis",
    VELOCIRAPTOR_MONGOLIENSIS: "pelvis",
    COMPSOGNATHUS_LONGIPES: "pelvis",
    COMPSOGNATHUS_LONGIPES_ROBOT: "pelvis",
    DIBOTHROSUCHUS_ELAPHROS: "torso",
    BRACHIOSAURUS_ALTITHORAX: "torso",
}

#: Per foot, the support geoms the generic rule takes and the registry drops ...
GENERIC_ONLY = {
    TYRANNOSAURUS_REX: ("{s}_metatarsus_geom",),  # the capsule above the plantar box: 0 statue load
    # Digit II's sickle claw hangs off the metatarsus, a touch-site body since physics r3 (D-D25): it
    # is held clear of the floor and bears no statue load.
    VELOCIRAPTOR_MONGOLIENSIS: ("{s}_claw_geom",),
    COMPSOGNATHUS_LONGIPES: (),
    COMPSOGNATHUS_LONGIPES_ROBOT: ("{side}_roll_cheek_-1", "{side}_roll_cheek_1"),  # a cheek down is a rolled foot
    DIBOTHROSUCHUS_ELAPHROS: (),
    BRACHIOSAURUS_ALTITHORAX: ("{s}_meta_geom",),  # the metapodial capsule in the touch group: 0 statue load
}
#: ... and the ones the registry adds that the generic rule misses.
REGISTRY_ONLY = {
    TYRANNOSAURUS_REX: (),
    # Before physics r3 the toe_d3 site was the foot's only touch site, so the generic rule missed digit IV
    # and the metatarsus (0.18 and 0.26 of the r2 statue's foot load); the r3 sensors made it see both.
    VELOCIRAPTOR_MONGOLIENSIS: (),
    COMPSOGNATHUS_LONGIPES: (),
    COMPSOGNATHUS_LONGIPES_ROBOT: (),
    DIBOTHROSUCHUS_ELAPHROS: (),
    BRACHIOSAURUS_ALTITHORAX: (),
}
SOLE = {
    TYRANNOSAURUS_REX: "{s}_plantar_geom",
    VELOCIRAPTOR_MONGOLIENSIS: None,
    COMPSOGNATHUS_LONGIPES: "{s}_plantar_pad",
    COMPSOGNATHUS_LONGIPES_ROBOT: "{side}_sole",
    DIBOTHROSUCHUS_ELAPHROS: "{s}_foot_geom",
    BRACHIOSAURUS_ALTITHORAX: "{s}_foot_geom",
}


def _names(templates, label):
    side = {"r": "right", "l": "left"}[label[-1]]
    return {template.format(s=label, side=side) for template in templates}


@pytest.fixture(scope="module")
def stance_env():
    """One stance env per model, built on first use and shared by the module's tests."""
    built = {}

    def get(model_id):
        if model_id not in built:
            built[model_id] = get_species_config(model_id).env_class(
                **dict(load_stage_config(model_id, "stance")["env_kwargs"])
            )
        return built[model_id]

    yield get
    for env in built.values():
        env.close()


# ── the registry ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name", ["MODEL_IDS", "BODY_WEIGHT_N", "FEET", "ROOT", "GENERIC_ONLY", "REGISTRY_ONLY", "SOLE"]
)
def test_every_per_model_table_covers_every_model(name):
    """A new model fails here, naming the table, instead of going unpinned (review M4)."""
    assert set(globals()[name]) == set(model_ids()), f"{name} must pin every manifest model"


def test_every_model_has_an_explicit_registry_entry_with_a_pinned_reference_site():
    assert set(MODEL_IDS) == set(model_ids())
    for model_id in MODEL_IDS:
        entry = support_entry(model_id)
        assert isinstance(entry, FootRegistryEntry)
        assert entry.support and entry.foot_site, model_id
        assert entry.sole == SOLE[model_id]


@pytest.mark.parametrize("model_id", MODEL_IDS, ids=lambda m: f"{m.species}-{m.embodiment}")
def test_the_registry_departs_from_the_generic_rule_exactly_where_documented(stance_env, model_id):
    env = stance_env(model_id)
    registered = Morphology.from_env(env, model_id)
    generic = Morphology.from_env(env, model_id, registry={})
    assert registered.registry_entry == support_entry(model_id)
    assert generic.registry_entry is None
    assert registered.labels == generic.labels == FEET[model_id]
    described, generic_described = registered.describe(env.model), generic.describe(env.model)
    for foot, generic_foot in zip(described["feet"], generic_described["feet"]):
        label = foot["label"]
        support, generic_support = set(foot["support_geoms"]), set(generic_foot["support_geoms"])
        assert generic_support - support == _names(GENERIC_ONLY[model_id], label), label
        assert support - generic_support == _names(REGISTRY_ONLY[model_id], label), label
        # Every registered geom is a colliding geom of that foot's leg, and the sole one of them.
        assert support <= set(foot["leg_geoms"])
        assert foot["sole_geom"] == (None if SOLE[model_id] is None else _names([SOLE[model_id]], label).pop())
        # The registry pins the reference site and the sole the generic rule would choose anyway.
        assert foot["reference_site"] == generic_foot["reference_site"]
        assert foot["sole_geom"] == generic_foot["sole_geom"]


@pytest.mark.parametrize("model_id", MODEL_IDS, ids=lambda m: f"{m.species}-{m.embodiment}")
def test_legs_are_whole_limbs_rooted_below_the_animal_root(stance_env, model_id):
    env = stance_env(model_id)
    morphology = Morphology.from_env(env, model_id)
    model = env.model
    assert model.body(morphology.root_body).name == ROOT[model_id]
    tops = []
    for foot in morphology.feet:
        (top,) = [body for body in foot.leg_bodies if int(model.body_parentid[body]) not in foot.leg_bodies]
        assert int(model.body_parentid[top]) == morphology.root_body
        tops.append(model.body(top).name)
    if model_id == COMPSOGNATHUS_LONGIPES_ROBOT:
        assert tops == ["hip_roll_assembly_2", "hip_roll_assembly"]
    else:
        assert tops == [f"{label}_thigh" for label in FEET[model_id]]
    legs = [set(foot.leg_geoms) for foot in morphology.feet]
    assert all(not (a & b) for i, a in enumerate(legs) for b in legs[i + 1 :])
    assert set().union(*legs) <= set(morphology.animal_geoms)
    assert not set(morphology.floor_geoms) & set(morphology.animal_geoms)


@pytest.mark.parametrize("model_id", MODEL_IDS, ids=lambda m: f"{m.species}-{m.embodiment}")
def test_body_weight_is_the_root_subtree_weight(stance_env, model_id):
    env = stance_env(model_id)
    morphology = Morphology.from_env(env, model_id)
    assert morphology.body_weight_n == pytest.approx(BODY_WEIGHT_N[model_id], abs=0.01)
    assert morphology.frame_skip == env.frame_skip and morphology.dt == env.dt
    assert morphology.n_feet == len(FEET[model_id]) and morphology.is_quadruped == (len(FEET[model_id]) == 4)


@pytest.mark.parametrize(
    "model_id", [m for m in MODEL_IDS if SOLE[m] is not None], ids=lambda m: f"{m.species}-{m.embodiment}"
)
def test_sole_frames_face_the_floor_and_point_forward(stance_env, model_id):
    morphology = Morphology.from_env(stance_env(model_id), model_id)
    for foot in morphology.feet:
        assert (foot.sole_normal_axis, foot.sole_normal_sign) == (2, -1.0), foot.label
        assert (foot.sole_forward_axis, foot.sole_forward_sign) == (0, 1.0), foot.label
        assert (foot.sole_lateral_axis, foot.sole_lateral_sign) == (1, 1.0), foot.label
        assert foot.side == foot.label[-1].upper()
    tilts = [foot.home_sole_tilt_deg for foot in morphology.feet]
    if model_id == BRACHIOSAURUS_ALTITHORAX:
        assert tilts == pytest.approx([19.22, 19.22, 15.78, 15.78], abs=0.01)
    else:
        assert tilts == pytest.approx([0.0] * len(tilts), abs=1e-4)  # acos amplifies rounding near flat


def test_a_foot_without_a_sole_has_no_sole_frame(stance_env):
    for foot in Morphology.from_env(stance_env(VELOCIRAPTOR_MONGOLIENSIS), VELOCIRAPTOR_MONGOLIENSIS).feet:
        assert foot.sole_geom == foot.sole_type == -1
        assert np.isnan(foot.home_sole_tilt_deg)


def test_an_unknown_model_is_refused_and_an_empty_registry_gets_the_generic_rule(stance_env):
    env = stance_env(VELOCIRAPTOR_MONGOLIENSIS)
    with pytest.raises(ValueError, match="Unknown species 'not_a_species'"):
        Morphology.from_env(env, ModelId("not_a_species", "anatomical"))
    generic = Morphology.from_env(env, VELOCIRAPTOR_MONGOLIENSIS, registry={})
    assert generic.model_id == VELOCIRAPTOR_MONGOLIENSIS and generic.registry_entry is None
    # Digit III, digit IV, the metatarsus and the claw it carries: the touch-site bodies since physics r3.
    assert [len(foot.support_geoms) for foot in generic.feet] == [4, 4]


def test_building_a_morphology_leaves_the_env_untouched(stance_env):
    env = stance_env(TYRANNOSAURUS_REX)
    env.reset(seed=3042)
    for _ in range(3):
        env.step(np.zeros(env.model.nu))
    before = (env.data.qpos.copy(), env.data.qvel.copy(), env.data.time, env._substep_probe_hook)
    morphology = Morphology.from_env(env, TYRANNOSAURUS_REX)
    np.testing.assert_array_equal(env.data.qpos, before[0])
    np.testing.assert_array_equal(env.data.qvel, before[1])
    assert env.data.time == before[2] and env._substep_probe_hook is before[3]
    for table in (morphology.geom_foot, morphology.geom_support, morphology.is_floor, morphology.is_animal):
        assert table.shape == (env.model.ngeom,) and not table.flags.writeable


# ── refusals ─────────────────────────────────────────────────────────────────

_TYRANNOSAURUS_REX_ENTRY = support_entry(TYRANNOSAURUS_REX)
assert _TYRANNOSAURUS_REX_ENTRY is not None, "the T-Rex declares FOOT_SUPPORT"


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        (
            replace(_TYRANNOSAURUS_REX_ENTRY, support=(*_TYRANNOSAURUS_REX_ENTRY.support, "{s}_no_such_geom")),
            "is not in the model",
        ),
        (
            replace(_TYRANNOSAURUS_REX_ENTRY, support=(*_TYRANNOSAURUS_REX_ENTRY.support, "l_toe_d2_geom")),
            "is not a colliding geom of leg",
        ),
        (
            replace(_TYRANNOSAURUS_REX_ENTRY, support=(*_TYRANNOSAURUS_REX_ENTRY.support, "{s}_toe_d2_geom")),
            "listed twice",
        ),
        (replace(_TYRANNOSAURUS_REX_ENTRY, sole="{s}_metatarsus_geom"), "is not one of foot"),
        (replace(_TYRANNOSAURUS_REX_ENTRY, foot_site=None), "pins no foot_site"),
        (replace(_TYRANNOSAURUS_REX_ENTRY, foot_site="{s}_no_such_site"), "foot site"),
        (replace(_TYRANNOSAURUS_REX_ENTRY, foot_site="l_foot"), "is not on leg"),
    ],
)
def test_a_bad_registry_entry_is_refused(stance_env, entry, message):
    with pytest.raises(MorphologyError, match=message):
        Morphology.from_env(stance_env(TYRANNOSAURUS_REX), TYRANNOSAURUS_REX, registry={TYRANNOSAURUS_REX: entry})


def test_an_env_without_foot_sensors_is_refused(stance_env):
    env = stance_env(TYRANNOSAURUS_REX)
    groups = env._foot_sensor_groups
    try:
        env._foot_sensor_groups = ()
        with pytest.raises(MorphologyError, match="_foot_sensor_groups"):
            Morphology.from_env(env, TYRANNOSAURUS_REX)
        env._foot_sensor_groups = ((10_000,),)
        with pytest.raises(MorphologyError, match="starts no sensor"):
            Morphology.from_env(env, TYRANNOSAURUS_REX)
        env._foot_sensor_groups = (groups[0], groups[0])
        with pytest.raises(MorphologyError, match="share a label"):
            Morphology.from_env(env, TYRANNOSAURUS_REX)
    finally:
        env._foot_sensor_groups = groups
    assert Morphology.from_env(env, TYRANNOSAURUS_REX).n_feet == 2


def test_an_invalid_home_keyframe_is_refused(stance_env):
    env = stance_env(TYRANNOSAURUS_REX)
    keyframe = env._reset_keyframe_id
    env._reset_keyframe_id = env.model.nkey
    try:
        with pytest.raises(MorphologyError, match="keyframe"):
            Morphology.from_env(env, TYRANNOSAURUS_REX)
    finally:
        env._reset_keyframe_id = keyframe


# ── the measurement manifest ─────────────────────────────────────────────────


def test_the_manifest_records_the_constants_the_registry_and_the_resolved_names(stance_env):
    env = stance_env(TYRANNOSAURUS_REX)
    manifest = measurement_manifest(Morphology.from_env(env, TYRANNOSAURUS_REX), env.model)
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
    assert (manifest["species"], manifest["embodiment"], manifest["frame_skip"], manifest["dt"]) == (
        "tyrannosaurus_rex",
        "anatomical",
        5,
        0.01,
    )
    assert manifest["body_weight_n"] == pytest.approx(840.91, abs=0.01)
    assert manifest["registry"] == {
        "support": list(_TYRANNOSAURUS_REX_ENTRY.support),
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
    generic = measurement_manifest(Morphology.from_env(env, TYRANNOSAURUS_REX, registry={}), env.model)
    assert generic["registry"] is None


def test_the_manifest_digest_is_the_canonical_json_digest_and_survives_a_json_round_trip(stance_env):
    env = stance_env(COMPSOGNATHUS_LONGIPES_ROBOT)
    manifest = measurement_manifest(Morphology.from_env(env, COMPSOGNATHUS_LONGIPES_ROBOT), env.model)
    digest = measurement_sha256(manifest)
    assert is_sha256_digest(digest)
    assert digest == canonical_json_sha256(manifest)
    assert measurement_sha256(json.loads(json.dumps(manifest, indent=2))) == digest
    assert (
        measurement_sha256(measurement_manifest(Morphology.from_env(env, COMPSOGNATHUS_LONGIPES_ROBOT), env.model))
        == digest
    )


def test_a_constant_or_a_registry_edit_moves_the_manifest_digest(stance_env, monkeypatch):
    env = stance_env(TYRANNOSAURUS_REX)
    morphology = Morphology.from_env(env, TYRANNOSAURUS_REX)
    digest = measurement_sha256(measurement_manifest(morphology, env.model))
    monkeypatch.setattr(constants, "CONTACT_THRESHOLD_N", 0.2)
    assert measurement_sha256(measurement_manifest(morphology, env.model)) != digest
    monkeypatch.undo()
    assert measurement_sha256(measurement_manifest(morphology, env.model)) == digest
    fewer_toes = replace(_TYRANNOSAURUS_REX_ENTRY, support=_TYRANNOSAURUS_REX_ENTRY.support[:-1])
    edited = Morphology.from_env(env, TYRANNOSAURUS_REX, registry={TYRANNOSAURUS_REX: fewer_toes})
    assert measurement_sha256(measurement_manifest(edited, env.model)) != digest
    # Order matters too: the per-geom metric tuples (support_geom_duty, support_geom_share) are positional.
    reordered = replace(_TYRANNOSAURUS_REX_ENTRY, support=tuple(reversed(_TYRANNOSAURUS_REX_ENTRY.support)))
    moved = Morphology.from_env(env, TYRANNOSAURUS_REX, registry={TYRANNOSAURUS_REX: reordered})
    assert measurement_sha256(measurement_manifest(moved, env.model)) != digest


def test_the_manifest_rounds_away_cross_architecture_noise_in_the_body_weight(stance_env):
    env = stance_env(TYRANNOSAURUS_REX)
    morphology = Morphology.from_env(env, TYRANNOSAURUS_REX)
    noisy = replace(morphology, body_weight_n=morphology.body_weight_n * (1 + 4e-16))
    assert noisy.body_weight_n != morphology.body_weight_n
    assert measurement_manifest(noisy, env.model) == measurement_manifest(morphology, env.model)
