"""Tests for environments.shared.plant_contract.manifest."""

from __future__ import annotations

import copy
import json
from typing import Any

import mujoco
import pytest

import environments.shared.plant_contract as plant_contract
from environments.shared.model_id import ModelId
from environments.shared.plant_contract import (
    BUNDLED_MANIFEST_PATH,
    GENERATED_MANIFEST_PATH,
    PlantContractError,
    check_plant_manifest,
    current_plant_identity,
    load_plant_versions,
)
from environments.shared.plant_contract.manifest import _enforce_revision_changes
from environments.shared.species_names import model_ids

VELOCIRAPTOR_MONGOLIENSIS = ModelId("velociraptor_mongoliensis", "anatomical")
ALLOSAURUS_FRAGILIS = ModelId("allosaurus_fragilis", "anatomical")
TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")
BRACHIOSAURUS_ALTITHORAX = ModelId("brachiosaurus_altithorax", "anatomical")
DIBOTHROSUCHUS_ELAPHROS = ModelId("dibothrosuchus_elaphros", "anatomical")
COMPSOGNATHUS_LONGIPES = ModelId("compsognathus_longipes", "anatomical")
COMPSOGNATHUS_LONGIPES_ROBOT = ModelId("compsognathus_longipes", "robot")

CANONICAL_MUJOCO_VERSION = load_plant_versions()[0]

# Per-model pins, keyed by the model (review M4); each names the models it leaves out and why, and
# test_the_per_model_tables_cover_every_model fails on a model in neither.  Widths carry the Phase C 3-dim command
# segment (BEHAVIOR_RECIPES_PLAN §4.6): 67/61/83/77 -> 70/64/86/80, and 43 / 53 -> 46 / 56 for the two Compsognathus
# plants; velociraptor left the frozen MJX core at policy-interface revision 11, and trex at revision 14 with the
# heading-free observation (64 -> 63), which also took the anatomical compsognathus from 56 to 55 (projected gravity
# replaces the 4-wide quaternion), and the velociraptor from 70 to 69 at revision 13 (2026-11-06) and to 71 at
# revision 14, the life-size plant (note 22, 2026-11-08: six per-site foot channels; its action 22 -> 17).
# Brachiosaurus and dibothrosuchus, the last two, left it at revisions 11 and 10 (note 23), when the core was
# removed: every model is SB3-only.  The target-free revision dropped the 4 target channels from all seven and made
# the brachiosaurus, the dibothrosuchus and the Compsognathus robot heading-free as well (quaternion 4 -> projected
# gravity 3): 45/55/46/71/63/86/80 -> 41/51/41/67/59/81/75.
#: Each SB3-only model's observation width.
SB3_ONLY_OBSERVATION_WIDTH = {
    ALLOSAURUS_FRAGILIS: 41,
    COMPSOGNATHUS_LONGIPES: 51,
    COMPSOGNATHUS_LONGIPES_ROBOT: 41,
    VELOCIRAPTOR_MONGOLIENSIS: 67,
    TYRANNOSAURUS_REX: 59,
    BRACHIOSAURUS_ALTITHORAX: 81,
    DIBOTHROSUCHUS_ELAPHROS: 75,
}
#: Each SB3-only model's action width, resolved from the bundled manifest alone.
BUNDLED_ACTION_DIM = {
    ALLOSAURUS_FRAGILIS: 10,
    VELOCIRAPTOR_MONGOLIENSIS: 17,
    TYRANNOSAURUS_REX: 15,
    COMPSOGNATHUS_LONGIPES: 14,
    COMPSOGNATHUS_LONGIPES_ROBOT: 12,
    BRACHIOSAURUS_ALTITHORAX: 30,
    DIBOTHROSUCHUS_ELAPHROS: 27,
}
#: (observation_dim, action_dim) of the identity generated from each executable environment.  Exempt: both
#: Compsognathus models, whose widths SB3_ONLY_OBSERVATION_WIDTH and BUNDLED_ACTION_DIM pin.
IDENTITY_DIMS = {
    ALLOSAURUS_FRAGILIS: (41, 10),
    VELOCIRAPTOR_MONGOLIENSIS: (67, 17),
    TYRANNOSAURUS_REX: (59, 15),
    BRACHIOSAURUS_ALTITHORAX: (81, 30),
    DIBOTHROSUCHUS_ELAPHROS: (75, 27),
}
_TABLES = {
    "SB3_ONLY_OBSERVATION_WIDTH": (SB3_ONLY_OBSERVATION_WIDTH, frozenset()),
    "BUNDLED_ACTION_DIM": (BUNDLED_ACTION_DIM, frozenset()),
    "IDENTITY_DIMS": (IDENTITY_DIMS, frozenset({COMPSOGNATHUS_LONGIPES, COMPSOGNATHUS_LONGIPES_ROBOT})),
}


@pytest.mark.parametrize("name", list(_TABLES))
def test_the_per_model_tables_cover_every_model(name):
    """A new model fails here, naming the table, instead of going unpinned (review M4)."""
    table, exempt = _TABLES[name]
    assert not set(table) & exempt, f"{name} pins a model it exempts"
    assert set(table) | exempt == set(model_ids()), f"{name} must pin every model it does not exempt"


@pytest.mark.skipif(
    mujoco.__version__ != CANONICAL_MUJOCO_VERSION,
    reason="exact manifest checks run only with the canonical MuJoCo version",
)
def test_committed_manifest_is_current_and_covers_all_models():
    manifest = check_plant_manifest()

    assert manifest["fingerprint_tool_version"] == plant_contract.FINGERPRINT_TOOL_VERSION == 2
    assert manifest["generated_with"]["float_significant_digits"] == 12
    from environments.shared.species_names import model_ids

    # plant-manifest/v2 nests every plant as plants[species][embodiment] (decision D-D33).
    plants = {
        ModelId(model_id, embodiment): entry
        for model_id, embodiments in manifest["plants"].items()
        for embodiment, entry in embodiments.items()
    }
    assert set(plants) == set(model_ids())
    for entry in plants.values():
        assert entry["policy_interface"]["revision"] >= 1
        assert entry["physics"]["revision"] >= 1
        assert entry["visual"]["revision"] >= 1


def test_bundled_runtime_manifest_matches_repository_manifest():
    assert BUNDLED_MANIFEST_PATH.read_bytes() == GENERATED_MANIFEST_PATH.read_bytes()


def _readable(value: object) -> str | None:
    """A model's parameter id is ``<species>-<embodiment>``; any other value keeps pytest's own."""
    return f"{value.species}-{value.embodiment}" if isinstance(value, ModelId) else None


@pytest.mark.parametrize("model_id", list(SB3_ONLY_OBSERVATION_WIDTH), ids=lambda m: f"{m.species}-{m.embodiment}")
def test_sb3_only_species_do_not_claim_mjx_parity(model_id):
    from environments.shared.plant_contract import policy_layer
    from environments.shared.species_registry import get_species_config

    assert not hasattr(policy_layer, "_jax_policy_interface_payload")
    with get_species_config(model_id).env_class() as env:
        assert env.supported_training_backends == ("stable-baselines3",)
        payload = policy_layer._policy_interface_payload(env.model, env, load_plant_versions()[1][model_id])
    # The constant record of the removed backend, kept so no SB3-only digest moved with the core (note 23).
    assert payload["jax_interface"] == {"supported": False, "backend": "jax-mjx"}
    assert payload["backend_observation_equal"] is None
    implementations = payload["interface_implementations"]
    assert implementations["supported_training_backends"] == ["stable-baselines3"]
    assert "sb3_sensor_cache" in implementations
    assert not {"backend_neutral_observation", "jax_action_mapping", "jax_observation_callers"} & set(implementations)
    assert set(implementations.get("home_reset", {"sb3": None})) == {"sb3"}
    assert payload["observation"]["shape"] == [SB3_ONLY_OBSERVATION_WIDTH[model_id]]


@pytest.mark.parametrize("model_id", list(SB3_ONLY_OBSERVATION_WIDTH), ids=lambda m: f"{m.species}-{m.embodiment}")
@pytest.mark.parametrize("backends", [("stable-baselines3", "jax-mjx"), ("jax-mjx",)], ids=["dual", "mjx-only"])
def test_a_declared_jax_mjx_backend_is_refused(model_id, backends, monkeypatch):
    """jax-mjx left with the frozen MJX interface core (plant_versions note 23): no layer accepts it."""
    from environments.shared.plant_contract import policy_layer
    from environments.shared.species_registry import get_species_config

    with get_species_config(model_id).env_class() as env:
        monkeypatch.setattr(env, "supported_training_backends", backends, raising=False)
        with pytest.raises(PlantContractError, match="declares the jax-mjx training backend, which was removed"):
            policy_layer._policy_interface_payload(env.model, env, load_plant_versions()[1][model_id])
        with pytest.raises(PlantContractError, match="jax-mjx"):
            plant_contract.fingerprint_model_layers(env.model, env, load_plant_versions()[1][model_id])


def test_any_backend_but_stable_baselines3_is_refused():
    from types import SimpleNamespace

    from environments.shared.plant_contract.policy_layer import declared_training_backends

    assert declared_training_backends(SimpleNamespace(), VELOCIRAPTOR_MONGOLIENSIS) == ("stable-baselines3",)
    for backends in ((), ("torchrl",), ("stable-baselines3", "stable-baselines3")):
        with pytest.raises(PlantContractError, match="invalid training backends"):
            declared_training_backends(SimpleNamespace(supported_training_backends=backends), VELOCIRAPTOR_MONGOLIENSIS)


@pytest.mark.parametrize(("model_id", "action_dim"), list(BUNDLED_ACTION_DIM.items()), ids=_readable)
def test_runtime_identity_falls_back_to_bundled_manifest(model_id, action_dim, monkeypatch, tmp_path):
    """The bundled manifest resolves every model.

    It records no training backends, so each model is checked against the
    environment's own supported_training_backends (Stable-Baselines3 alone
    on every model since plant_versions note 23).
    """
    monkeypatch.setattr(plant_contract.constants, "SPECIES_MANIFEST_PATH", tmp_path / "missing-species.toml")
    monkeypatch.setattr(plant_contract.constants, "PLANT_VERSIONS_PATH", tmp_path / "missing-versions.toml")
    monkeypatch.setattr(plant_contract.constants, "GENERATED_MANIFEST_PATH", tmp_path / "missing-manifest.json")

    identity = plant_contract.current_plant_identity(model_id)

    assert identity.model_id == model_id
    assert identity.nu == identity.action_dim == action_dim


@pytest.mark.parametrize(
    ("model_id", "observation_dim", "action_dim"),
    [(model_id, *dims) for model_id, dims in IDENTITY_DIMS.items()],
    ids=_readable,
)
def test_current_identity_matches_each_executable_environment(model_id, observation_dim, action_dim):
    identity = current_plant_identity(model_id, verify_generated=False)

    assert identity.observation_dim == observation_dim
    assert identity.action_dim == identity.nu == action_dim


# The revision check on synthetic manifests.  v1 keyed a plant by its species
# string; v2 nests it as plants[species][embodiment] (decision D-D33).  The v1
# keys here are made up rather than the retired species IDs.
MANIFEST_V1 = "mesozoic.plant-manifest/v1"
MANIFEST_V2 = "mesozoic.plant-manifest/v2"
MONGOLIENSIS = VELOCIRAPTOR_MONGOLIENSIS
FRAGILIS = ModelId("allosaurus_fragilis", "anatomical")
FRAGILIS_ROBOT = ModelId("allosaurus_fragilis", "robot")
# The same plant after an epithet correction.
JIMMADSENI = ModelId("allosaurus_jimmadseni", "anatomical")


@pytest.fixture
def v2_is_current(monkeypatch):
    """The revision check reads v2 as the current schema, the one a v1 baseline may cross to."""
    monkeypatch.setattr(plant_contract.constants, "PLANT_MANIFEST_SCHEMA", MANIFEST_V2)


def _plant(physics: str = "physics-a", *, revision: int = 1) -> dict[str, Any]:
    """The three layers of one synthetic manifest entry, each at ``revision``."""
    return {
        "policy_interface": {"sha256": "interface-a", "revision": revision},
        "physics": {"sha256": physics, "revision": revision},
        "visual": {"sha256": "visual-a", "revision": revision},
    }


def _v1(plants: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {"schema": MANIFEST_V1, "plants": copy.deepcopy(plants)}


def _v2(plants: dict[ModelId, dict[str, Any]]) -> dict[str, Any]:
    nested: dict[str, dict[str, Any]] = {}
    for model_id, entry in plants.items():
        nested.setdefault(model_id.species, {})[model_id.embodiment] = copy.deepcopy(entry)
    return {"schema": MANIFEST_V2, "plants": nested}


def test_changed_layer_requires_revision_increment(v2_is_current):
    old = _v2({MONGOLIENSIS: _plant()})
    new = copy.deepcopy(old)
    new["plants"]["velociraptor_mongoliensis"]["anatomical"]["physics"]["sha256"] = "physics-b"

    with pytest.raises(PlantContractError, match="physics fingerprint changed"):
        _enforce_revision_changes(old, new)

    new["plants"]["velociraptor_mongoliensis"]["anatomical"]["physics"]["revision"] = 2
    _enforce_revision_changes(old, new)


def test_layer_revision_cannot_roll_back_when_fingerprint_is_unchanged(v2_is_current):
    old = _v2({MONGOLIENSIS: _plant(revision=2)})
    new = copy.deepcopy(old)
    new["plants"]["velociraptor_mongoliensis"]["anatomical"]["visual"]["revision"] = 1

    with pytest.raises(PlantContractError, match="visual_revision cannot decrease"):
        _enforce_revision_changes(old, new)


def test_v1_baseline_re_keyed_to_v2_pairs_by_physics_digest(v2_is_current):
    old = _v1({"allosaurus": _plant("physics-a", revision=3), "allosaurus_robot": _plant("physics-b", revision=2)})
    new = _v2({FRAGILIS: _plant("physics-a", revision=3), FRAGILIS_ROBOT: _plant("physics-b", revision=2)})

    _enforce_revision_changes(old, new)

    # Each pair is then checked layer by layer: a moved digest needs its revision raised.
    new["plants"]["allosaurus_fragilis"]["robot"]["policy_interface"]["sha256"] = "interface-b"
    with pytest.raises(
        PlantContractError,
        match=r"allosaurus_robot \(now allosaurus_fragilis/robot\) policy_interface fingerprint changed",
    ):
        _enforce_revision_changes(old, new)
    new["plants"]["allosaurus_fragilis"]["robot"]["policy_interface"]["revision"] = 3
    _enforce_revision_changes(old, new)


def test_physics_change_and_lowered_revision_across_a_re_key_are_refused(v2_is_current):
    # Compared by key alone, this re-keyed plant used to pass vacuously.
    old = _v1({"allosaurus": _plant("physics-a", revision=4)})
    new = _v2({FRAGILIS: _plant("physics-b", revision=1)})

    with pytest.raises(PlantContractError, match="allosaurus left the plant manifest"):
        _enforce_revision_changes(old, new)


@pytest.mark.parametrize(
    ("old_plants", "new_plants"),
    [
        # Two baseline plants with one digest.
        (
            {"allosaurus": _plant("physics-a"), "allosaurus_twin": _plant("physics-a")},
            {FRAGILIS: _plant("physics-a"), FRAGILIS_ROBOT: _plant("physics-b")},
        ),
        # Two current plants with one digest.
        (
            {"allosaurus": _plant("physics-a"), "allosaurus_robot": _plant("physics-b")},
            {FRAGILIS: _plant("physics-a"), FRAGILIS_ROBOT: _plant("physics-a")},
        ),
    ],
    ids=["baseline", "current"],
)
def test_a_physics_digest_shared_by_re_keyed_plants_is_refused(v2_is_current, old_plants, new_plants):
    with pytest.raises(PlantContractError, match="physics-a is shared by more than one re-keyed plant"):
        _enforce_revision_changes(_v1(old_plants), _v2(new_plants))


def test_an_unmatched_baseline_plant_is_refused(v2_is_current):
    old = _v1({"allosaurus": _plant("physics-a"), "allosaurus_robot": _plant("physics-b")})
    new = _v2({FRAGILIS: _plant("physics-a")})

    with pytest.raises(PlantContractError, match="allosaurus_robot left the plant manifest"):
        _enforce_revision_changes(old, new)


def test_same_schema_re_key_with_unchanged_physics_pairs_by_digest(v2_is_current):
    old = _v2({FRAGILIS: _plant(revision=2)})
    new = _v2({JIMMADSENI: _plant(revision=2)})

    _enforce_revision_changes(old, new)

    new["plants"]["allosaurus_jimmadseni"]["anatomical"]["visual"]["sha256"] = "visual-b"
    with pytest.raises(
        PlantContractError,
        match=r"allosaurus_fragilis/anatomical \(now allosaurus_jimmadseni/anatomical\) visual fingerprint changed",
    ):
        _enforce_revision_changes(old, new)


def test_same_schema_re_key_cannot_restart_the_counters(v2_is_current):
    old = _v2({FRAGILIS: _plant(revision=3)})
    new = _v2({JIMMADSENI: _plant(revision=1)})

    with pytest.raises(PlantContractError, match="policy_interface_revision cannot decrease from 3 to 1"):
        _enforce_revision_changes(old, new)


def test_same_schema_removal_is_refused_and_addition_passes(v2_is_current):
    both = _v2({FRAGILIS: _plant("physics-a"), FRAGILIS_ROBOT: _plant("physics-b")})
    anatomical_only = _v2({FRAGILIS: _plant("physics-a")})

    with pytest.raises(
        PlantContractError,
        match="allosaurus_fragilis/robot left the plant manifest.*retired only by an explicit decision",
    ):
        _enforce_revision_changes(both, anatomical_only)
    # A current plant without a baseline is new.
    _enforce_revision_changes(anatomical_only, both)


@pytest.mark.parametrize(
    ("old_schema", "new_schema", "message"),
    [
        (MANIFEST_V1, MANIFEST_V1, "current plant manifest must use schema mesozoic.plant-manifest/v2"),
        (MANIFEST_V2, MANIFEST_V1, "current plant manifest must use schema mesozoic.plant-manifest/v2"),
        ("mesozoic.plant-manifest/v0", MANIFEST_V2, "only mesozoic.plant-manifest/v1 against .*/v2 crosses schemas"),
    ],
    ids=["v1-baseline-v1-current", "v2-baseline-v1-current", "v0-baseline-v2-current"],
)
def test_only_the_v1_to_v2_transition_crosses_schemas(v2_is_current, old_schema, new_schema, message):
    old = {"schema": old_schema, "plants": {"allosaurus": _plant()}}
    new = {"schema": new_schema, "plants": {"allosaurus": _plant()}}

    with pytest.raises(PlantContractError, match=message):
        _enforce_revision_changes(old, new)


@pytest.mark.skipif(
    mujoco.__version__ != CANONICAL_MUJOCO_VERSION,
    reason="exact manifest checks run only with the canonical MuJoCo version",
)
def test_manifest_check_enforces_revisions_against_external_baseline(tmp_path):
    baseline = json.loads(GENERATED_MANIFEST_PATH.read_text(encoding="utf-8"))
    baseline["plants"]["velociraptor_mongoliensis"]["anatomical"]["physics"]["sha256"] = "sha256:" + "0" * 64
    baseline_path = tmp_path / "base-plant-manifest.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    with pytest.raises(PlantContractError, match="physics fingerprint changed"):
        check_plant_manifest(baseline_path=baseline_path)
