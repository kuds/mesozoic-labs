"""Protect the separate fore/hind soles, digit arrangement, and free-body reference walk."""

import xml.etree.ElementTree as ET

import pytest

from examples.triceratops_horridus.check_anatomy import check_reference_walk, inspect_scene
from examples.triceratops_horridus.check_terrain import inspect_course
from examples.triceratops_horridus.check_turns import check_turn, run_turn
from examples.triceratops_horridus.courses import COURSES, TURN_COURSES
from examples.triceratops_horridus.measure_proportions import measurements
from examples.triceratops_horridus.reference_walk import MODEL_PATH


def test_foot_geometry_and_limb_proportions():
    model = ET.parse(MODEL_PATH)
    for side in ("left", "right"):
        for limb, digits, unguals in (("fore", 5, 3), ("hind", 4, 4)):
            prefix = f"{side}_{limb}_"
            names = {geom.get("name") for geom in model.iter("geom") if geom.get("name", "").startswith(prefix)}
            assert {n for n in names if n.startswith(prefix + "digit_")} == {
                f"{prefix}digit_{i}" for i in range(1, digits + 1)
            }
            assert {n for n in names if n.startswith(prefix + "ungual_")} == {
                f"{prefix}ungual_{i}" for i in range(1, unguals + 1)
            }
    dimensions = measurements(MODEL_PATH)
    for axis in ("length", "width"):
        assert dimensions[f"Hind contact pad envelope, {axis}"] > dimensions[f"Fore contact pad envelope, {axis}"]
    assert 0.65 < dimensions["Lower hindlimb (pivot span)"] / dimensions["Femur (pivot span)"] < 0.72


def test_adult_stands_and_repeats_the_reference_walk():
    check_reference_walk(inspect_scene(MODEL_PATH)["walk"])


@pytest.mark.parametrize("direction", ("left", "right"))
def test_adult_turns_and_leaves_on_the_new_heading(direction):
    check_turn(run_turn(TURN_COURSES[direction]))


@pytest.mark.parametrize("name", ("uphill", "downhill", "cross_left", "cross_right", "uneven", "uneven_mirrored"))
def test_adult_stands_and_traverses_the_course(name):
    result = inspect_course(COURSES[name])
    assert result["passed"], result
