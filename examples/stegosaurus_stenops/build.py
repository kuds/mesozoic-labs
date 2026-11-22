"""Assemble the standalone reference MJCF. Run from the repository root."""

from pathlib import Path
from xml.etree import ElementTree as ET

from .axial import add_axial
from .limbs import LEGS, add_limbs

MODEL_PATH = Path(__file__).with_name("model.xml")


def build_model() -> ET.ElementTree:
    root = ET.Element("mujoco", model="Stegosaurus stenops")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(root, "option", timestep=".002", integrator="implicitfast", cone="elliptic")
    ET.SubElement(root, "statistic", center="-.8 0 1.14", extent="3.8")
    defaults = ET.SubElement(root, "default")
    ET.SubElement(defaults, "geom", friction="1 .008 .001", condim="3", solref=".012 1", solimp=".9 .98 .002")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", azimuth="105", elevation="-9", offwidth="1440", offheight="960")
    ET.SubElement(visual, "headlight", ambient=".36 .36 .36", diffuse=".39 .38 .36", specular=".04 .04 .04")
    ET.SubElement(visual, "rgba", haze=".80 .82 .79 1")
    ET.SubElement(visual, "quality", shadowsize="4096", offsamples="4")
    ET.SubElement(visual, "map", fogstart=".9", fogend="5")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        name="sky",
        type="skybox",
        builtin="gradient",
        rgb1=".73 .77 .75",
        rgb2=".88 .89 .85",
        width="128",
        height="768",
    )
    ET.SubElement(
        asset,
        "texture",
        name="tiles",
        type="2d",
        builtin="checker",
        width="256",
        height="256",
        rgb1=".51 .52 .46",
        rgb2=".53 .54 .48",
    )
    ET.SubElement(
        asset,
        "material",
        name="floor_mat",
        texture="tiles",
        texrepeat="1 1",
        texuniform="true",
        specular=".05",
        shininess=".05",
    )
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(
        world,
        "light",
        name="key",
        type="directional",
        pos="3 -3 6",
        dir="-.21 .32 -1",
        diffuse=".70 .67 .60",
        specular=".14 .13 .11",
    )
    ET.SubElement(
        world,
        "light",
        name="fill",
        type="directional",
        pos="-2 3 4",
        dir=".3 -.4 -1",
        diffuse=".32 .35 .38",
        specular="0 0 0",
        castshadow="false",
    )
    ET.SubElement(
        world, "geom", name="ground", type="plane", size="0 0 .1", contype="1", conaffinity="2", material="floor_mat"
    )
    torso = ET.SubElement(world, "body", name="torso", pos="0 0 1.2")
    ET.SubElement(torso, "freejoint", name="root")
    actuator = ET.SubElement(root, "actuator")
    add_axial(asset, torso, actuator)
    add_limbs(torso, actuator, asset)
    ET.indent(root)
    return ET.ElementTree(root)


def write_model(directory: Path) -> None:
    """Write the model and its limb surface data beside it, for any MuJoCo viewer."""
    tree = build_model()
    root = tree.getroot()
    assert root is not None
    assets = root.find("asset")
    assert assets is not None
    limb_skins = ET.Element("mujocoinclude")
    names = {f"{leg}_hide" for leg in LEGS}
    for skin in assets.findall("skin"):
        if skin.get("name") in names:
            assets.remove(skin)
            limb_skins.append(skin)
    ET.SubElement(assets, "include", file="limb_skins.xml")
    for name, element in (("limb_skins.xml", limb_skins), (MODEL_PATH.name, root)):
        ET.indent(element)
        (directory / name).write_bytes(ET.tostring(element, encoding="utf-8", xml_declaration=True) + b"\n")


if __name__ == "__main__":
    write_model(MODEL_PATH.parent)
