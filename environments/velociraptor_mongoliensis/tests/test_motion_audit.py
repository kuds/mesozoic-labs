"""Ground clearance uses the skinned surface and keeps the physics trajectory intact."""

from argparse import Namespace
from xml.etree import ElementTree

from environments.velociraptor_mongoliensis.anatomical.env import VelociraptorMongoliensisAnatomicalEnv
from environments.velociraptor_mongoliensis.scripts.audit_motion import run


def test_skin_probe_measures_foot_without_changing_stance(tmp_path):
    args = Namespace(mode="stand", noise=0.0, xml=None, seconds=0.2, warmup=0.1, skin_clearance=False)
    unprobed = run(args, 0)
    args.skin_clearance = True
    probed = run(args, 0)
    skin = probed.pop("min_rendered_skin_floor_clearance_m")
    assert probed == unprobed
    assert 0.01 < skin["ungual_II_horn"] < skin["hallux_claw_horn"]
    assert skin["hallux_scale"] > 0.03
    assert skin["digit_II_scale"] > 0
    for name in ("ball_scale", "digit_3_scale", "digit_4_scale", "ungual_3_horn", "ungual_4_horn"):
        assert abs(skin[name]) < 0.01

    tree = ElementTree.parse(VelociraptorMongoliensisAnatomicalEnv.MODEL_PATH)
    for parent in tree.iter():
        for element in parent.findall("skin"):
            parent.remove(element)
    args.xml = tmp_path / "skinless.xml"
    tree.write(args.xml)
    assert run(args, 0)["min_rendered_skin_floor_clearance_m"] == {}
