"""Promote the frozen A/forward study to a self-contained M1 training asset.

Runtime training never imports examples or generates XML. The committed asset is
the canonical plant; this deterministic builder records its engineering source.
"""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "examples/mini_tyrannosaurus_rex/leg_study/forward/model.xml"
OUTPUT = ROOT / "environments/tyrannosaurus_rex/robot/assets/tyrannosaurus_rex_robot.xml"


def model_text() -> str:
    root = ET.fromstring(SOURCE.read_bytes())
    root.set("model", "tyrannosaurus_rex_robot")
    torso = root.find(".//body[@name='torso']")
    world = root.find("worldbody")
    assert torso is not None and world is not None
    torso.set("name", "pelvis")
    silhouette = torso.find("geom[@name='proposed_torso_silhouette']")
    assert silhouette is not None
    torso.remove(silhouette)
    # A 10 micrometre initial sole penetration makes the home support polygon
    # well-defined for the push calibration (exact tangency has no contacts).
    home = root.find("keyframe/key[@name='home']")
    assert home is not None
    qpos = home.attrib["qpos"].split()
    qpos[2] = "0.11299"
    home.set("qpos", " ".join(qpos))
    # Full hardware and installation reserves stay intact. Payload markers
    # denote inertia allowances, not collision-qualified head/tail geometry.
    for geom in root.findall(".//geom"):
        if geom.get("name", "").endswith("_mass_marker"):
            geom.set("group", "1")
            geom.set("rgba", "0.55 0.61 0.65 0.22")
    imu = torso.find("site[@name='body_imu']")
    assert imu is not None
    imu.set("name", "imu")
    for side, short in (("right", "r"), ("left", "l")):
        foot = root.find(f".//body[@name='{side}_foot']")
        assert foot is not None
        ET.SubElement(
            foot, "site", name=f"{short}_foot", type="box", pos=".007 0 -.02575", size=".036 .018 .004", rgba="0 1 0 0"
        )
    target = ET.SubElement(world, "body", name="prey", mocap="true", pos=".7 0 .113")
    ET.SubElement(
        target,
        "geom",
        name="prey_geom",
        type="sphere",
        size=".015",
        rgba="1 .5 .1 .4",
        contype="0",
        conaffinity="0",
        group="1",
    )
    sensors = root.find("sensor")
    if sensors is not None:
        root.remove(sensors)
    sensors = ET.SubElement(root, "sensor")
    ET.SubElement(sensors, "gyro", name="pelvis_gyro", site="imu")
    ET.SubElement(sensors, "accelerometer", name="pelvis_accel", site="imu")
    ET.SubElement(sensors, "framequat", name="pelvis_orientation", objtype="site", objname="imu")
    for short in ("r", "l"):
        ET.SubElement(sensors, "touch", name=f"{short}_foot_touch", site=f"{short}_foot")
    custom = ET.SubElement(root, "custom")
    for name, value in {
        "hardware_milestone": "m01_skeleton",
        "hardware_revision": "hw_r01",
        "mass_basis": "639.1g chassis allocation plus 115g fixed future payload allowances; unmeasured",
        "engineering_source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    }.items():
        ET.SubElement(custom, "text", name=name, data=value)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(model_text())
    print(OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
