"""The model identity (species, embodiment) and every form built from it (decision D-D33)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from gymnasium.envs.registration import parse_env_id

from environments.shared.model_id import EMBODIMENT_LABELS, EMBODIMENTS, ModelId

# Written out rather than derived, so a change to a naming rule fails here by
# name: (species, embodiment, species base class, environment class, Gymnasium ID).
SIX_MODELS = [
    (
        "velociraptor_mongoliensis",
        "anatomical",
        "VelociraptorMongoliensisEnv",
        "VelociraptorMongoliensisAnatomicalEnv",
        "MesozoicLabs/VelociraptorMongoliensis-Anatomical-v0",
    ),
    (
        "tyrannosaurus_rex",
        "anatomical",
        "TyrannosaurusRexEnv",
        "TyrannosaurusRexAnatomicalEnv",
        "MesozoicLabs/TyrannosaurusRex-Anatomical-v0",
    ),
    (
        "brachiosaurus_altithorax",
        "anatomical",
        "BrachiosaurusAltithoraxEnv",
        "BrachiosaurusAltithoraxAnatomicalEnv",
        "MesozoicLabs/BrachiosaurusAltithorax-Anatomical-v0",
    ),
    (
        "dibothrosuchus_elaphros",
        "anatomical",
        "DibothrosuchusElaphrosEnv",
        "DibothrosuchusElaphrosAnatomicalEnv",
        "MesozoicLabs/DibothrosuchusElaphros-Anatomical-v0",
    ),
    (
        "compsognathus_longipes",
        "anatomical",
        "CompsognathusLongipesEnv",
        "CompsognathusLongipesAnatomicalEnv",
        "MesozoicLabs/CompsognathusLongipes-Anatomical-v0",
    ),
    (
        "compsognathus_longipes",
        "robot",
        "CompsognathusLongipesEnv",
        "CompsognathusLongipesRobotEnv",
        "MesozoicLabs/CompsognathusLongipes-Robot-v0",
    ),
]
ANATOMICAL = ModelId("compsognathus_longipes", "anatomical")
ROBOT = ModelId("compsognathus_longipes", "robot")


def test_embodiments_are_a_closed_vocabulary_with_one_label_each():
    assert EMBODIMENTS == ("anatomical", "robot")
    assert tuple(EMBODIMENT_LABELS) == EMBODIMENTS
    assert EMBODIMENT_LABELS == {"anatomical": "Anatomical", "robot": "Robot"}


@pytest.mark.parametrize(
    ("species", "embodiment", "species_class", "env_class", "gym_id"),
    SIX_MODELS,
    ids=[f"{row[0]}-{row[1]}" for row in SIX_MODELS],
)
def test_every_form_is_built_from_the_pair(species, embodiment, species_class, env_class, gym_id):
    model_id = ModelId(species, embodiment)

    assert str(model_id) == f"{species}/{embodiment}"
    assert model_id.path(Path("configs")) == Path("configs", species, embodiment)
    assert model_id.package == f"environments.{species}.{embodiment}"
    assert model_id.species_pascal + "Env" == species_class
    assert model_id.pascal + "Env" == model_id.env_class_name == env_class
    assert model_id.gym_id == gym_id
    assert model_id.asset_name == f"{species}_{embodiment}.xml"
    assert model_id.asset_path(Path("environments")) == Path(
        "environments", species, embodiment, "assets", f"{species}_{embodiment}.xml"
    )
    assert model_id.anchor == f"{species}-{embodiment}"
    # A valid Gymnasium ID: namespace, name, version.
    assert parse_env_id(model_id.gym_id) == ("MesozoicLabs", gym_id.split("/")[1].removesuffix("-v0"), 0)


@pytest.mark.parametrize(
    ("species", "embodiment"),
    [row[:2] for row in SIX_MODELS],
    ids=[f"{row[0]}-{row[1]}" for row in SIX_MODELS],
)
def test_record_round_trip_carries_two_fields(species, embodiment):
    model_id = ModelId(species, embodiment)

    assert model_id.record() == {"species": species, "embodiment": embodiment}
    assert ModelId.from_record(model_id.record()) == model_id
    assert ModelId.from_record({**model_id.record(), "stage": "stance"}) == model_id


def test_identity_is_a_frozen_hashable_value_ordered_by_species_then_embodiment():
    assert ModelId("compsognathus_longipes", "robot") == ROBOT
    assert len({ANATOMICAL, ROBOT, ModelId("compsognathus_longipes", "anatomical")}) == 2
    assert sorted([ROBOT, ModelId("brachiosaurus_altithorax", "anatomical"), ANATOMICAL]) == [
        ModelId("brachiosaurus_altithorax", "anatomical"),
        ANATOMICAL,
        ROBOT,
    ]
    with pytest.raises(dataclasses.FrozenInstanceError):
        ROBOT.embodiment = "anatomical"  # type: ignore[misc]
    # The joined form is a message, not the identity.
    joined: object = str(ROBOT)
    assert ROBOT != joined


@pytest.mark.parametrize(
    "species",
    [
        "trex",  # an alias: no alias resolves
        "tyrannosaurus_rex/anatomical",  # a joined identity is never parsed
        "Tyrannosaurus_Rex",  # no normalisation: case
        "tyrannosaurus-rex",  # no normalisation: separator
        "Tyrannosaurus Rex",  # a display name
        "tyrannosaurus_rex ",
        "tyrannosaurus",  # a genus alone
        "",
    ],
)
def test_species_must_be_an_exact_binomial(species):
    with pytest.raises(ValueError, match="binomial"):
        ModelId(species, "anatomical")


@pytest.mark.parametrize("embodiment", ["Anatomical", "Robot", "anatomic", "robot_rev_b", "", "anatomical "])
def test_embodiment_must_be_in_the_vocabulary(embodiment):
    with pytest.raises(ValueError, match="embodiment must be one of anatomical, robot"):
        ModelId("tyrannosaurus_rex", embodiment)


def test_there_is_no_default_embodiment():
    with pytest.raises(TypeError):
        ModelId("tyrannosaurus_rex")  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ({"species": "tyrannosaurus_rex"}, "no 'embodiment' field"),
        ({"embodiment": "anatomical"}, "no 'species' field"),
        ({"species": "tyrannosaurus_rex", "embodiment": None}, "'embodiment' must be a string"),
        ({"species": "tyrannosaurus_rex/anatomical", "embodiment": "anatomical"}, "binomial"),
        ({"species": "trex", "embodiment": "anatomical"}, "binomial"),
    ],
)
def test_from_record_refuses_a_missing_field_a_joined_string_or_an_alias(record, message):
    with pytest.raises(ValueError, match=message):
        ModelId.from_record(record)


def test_matches_compares_both_fields():
    assert ROBOT.matches(ROBOT.record())
    assert ROBOT.matches({**ROBOT.record(), "schema": "any"})
    # The other embodiment of the same species is another model.
    assert not ROBOT.matches(ANATOMICAL.record())
    assert not ANATOMICAL.matches(ROBOT.record())
    assert not ROBOT.matches({"species": "tyrannosaurus_rex", "embodiment": "robot"})


@pytest.mark.parametrize("missing", ["species", "embodiment"])
def test_matches_refuses_a_record_missing_a_field_instead_of_matching(missing):
    record = {key: value for key, value in ROBOT.record().items() if key != missing}

    for model_id in (ANATOMICAL, ROBOT):
        with pytest.raises(ValueError, match=f"no '{missing}' field"):
            model_id.matches(record)
