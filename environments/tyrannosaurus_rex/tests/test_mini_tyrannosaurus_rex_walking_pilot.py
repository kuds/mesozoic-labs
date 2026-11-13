"""Integrity guards for the manual diagnostic, not a passing walking gate."""

import shutil
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from examples.mini_tyrannosaurus_rex.walking_pilot import MODEL_PATH, WalkingPilot


@pytest.mark.parametrize("weak_limits", [(-0.12, 0.30), (-0.30, 0.12)])
def test_pilot_refuses_to_increase_either_direction_of_one_weak_leg_motor(weak_limits):
    pilot = WalkingPilot()
    pilot.model.actuator_forcerange[pilot.model.actuator("left_knee_servo").id] = weak_limits
    saved = pilot.model.actuator_forcerange.copy()
    with pytest.raises(ValueError, match="weakest loaded leg limit"):
        pilot.run(duration=2.1, cap=0.20)
    assert pilot.data.time == 0
    np.testing.assert_array_equal(pilot.model.actuator_forcerange, saved)


@pytest.mark.parametrize("fixed_fixture", [False, True], ids=["world", "fixed-fixture"])
def test_pilot_refuses_an_extra_world_weld_even_when_display_support_is_off(tmp_path, fixed_fixture):
    root = ET.parse(MODEL_PATH).getroot()
    equality = root.find("equality")
    world = root.find("worldbody")
    assert equality is not None and world is not None
    anchor = "world"
    if fixed_fixture:
        ET.SubElement(world, "body", name="test_fixture")
        anchor = "test_fixture"
    ET.SubElement(equality, "weld", name="hidden_test_support", body1="torso", body2=anchor, active="true")
    path = tmp_path / "model.xml"
    path.write_text(ET.tostring(root, encoding="unicode"))
    shutil.copytree(MODEL_PATH.parent / "assets", tmp_path / "assets")
    pilot = WalkingPilot(path)
    assert not pilot.model.eq_active0[pilot.model.equality("display_support").id]
    with pytest.raises(ValueError, match="world weld"):
        pilot.run(duration=2.1, cap=0.20)
    assert pilot.data.time == 0
