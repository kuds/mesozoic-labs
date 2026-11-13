"""Optional 825 g, 15-motor tail-yaw study derived from the 800 g base MJCF.

The rear module adds an assigned 18 g XL330 and 7 g mounting allowance, while
moving the already-budgeted 70 g battery 23 mm forward. Both changes update the
torso inertia; the existing engineering reserve remains untouched. Proxies do
not validate horn, bearing, connector, cable, or fabrication clearances.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path

import mujoco
import numpy as np

MODEL_PATH = Path(__file__).with_name("model.xml")
TAIL_LIMIT = math.radians(25)
BATTERY_MASS = 0.070
BATTERY_SHIFT = np.array([0.023, 0.0, 0.0])
TAIL_SCALE = 0.90


def _required(root: ET.Element, path: str) -> ET.Element:
    element = root.find(path)
    if element is None:
        raise ValueError(f"Tail option needs the base-model element: {path}")
    return element


def _format(values: Iterable[float]) -> str:
    return " ".join(f"{value:.12g}" for value in values)


def _box_inertia(mass: float, dimensions: np.ndarray) -> np.ndarray:
    return np.diag(mass / 12 * (dimensions @ dimensions - dimensions * dimensions))


def _update_torso_inertia(torso: ET.Element) -> None:
    inertial = _required(torso, "inertial")
    raw = np.fromstring(inertial.attrib["fullinertia"], sep=" ")
    tensor = np.array([[raw[0], raw[3], raw[4]], [raw[3], raw[1], raw[5]], [raw[4], raw[5], raw[2]]])
    pieces = [(float(inertial.attrib["mass"]), np.fromstring(inertial.attrib["pos"], sep=" "), tensor)]
    battery = _required(torso, "geom[@name='schematic_battery_proxy']")
    battery_position = np.fromstring(battery.attrib["pos"], sep=" ")
    battery_dimensions = 2 * np.fromstring(battery.attrib["size"], sep=" ")
    # Subtract the original battery contribution before adding its new position.
    for mass, position in ((-BATTERY_MASS, battery_position), (BATTERY_MASS, battery_position + BATTERY_SHIFT)):
        pieces.append((mass, position, _box_inertia(mass, battery_dimensions)))
    for mass, position, dimensions in (
        (0.018, [-0.0755, 0, -0.0295], [0.034, 0.020, 0.026]),
        (0.007, [-0.073, 0, -0.005], [0.060, 0.034, 0.050]),
    ):
        pieces.append((mass, np.array(position), _box_inertia(mass, np.array(dimensions))))
    total = sum(piece[0] for piece in pieces)
    center = sum((mass * position for mass, position, _ in pieces), start=np.zeros(3)) / total
    combined = np.zeros((3, 3))
    for mass, position, local in pieces:
        offset = position - center
        combined += local + mass * ((offset @ offset) * np.eye(3) - np.outer(offset, offset))
    inertial.set("mass", f"{total:.12g}")
    inertial.set("pos", _format(center))
    inertial.set("fullinertia", _format(combined.flat[i] for i in (0, 4, 8, 1, 2, 5)))


def load_model(base_path: Path = MODEL_PATH) -> mujoco.MjModel:
    """Build the tail option in memory; the base XML and its assets stay unchanged."""
    root = ET.parse(base_path).getroot()
    torso = _required(root, ".//body[@name='torso']")
    tail = _required(torso, "body[@name='tail']")
    key = _required(root, "keyframe/key[@name='home']")
    positions, controls = key.attrib["qpos"].split(), key.attrib["ctrl"].split()
    if (
        torso.findall("body")[0] is not tail
        or tail.find("joint") is not None
        or len(positions) != 22
        or len(controls) != 14
    ):
        raise ValueError("Tail option requires the 800 g base model's fixed-tail and 14-actuator layout.")
    _update_torso_inertia(torso)
    tail.set("pos", "-.083 0 .015")
    # The smaller shell retains the conservative 15 g assigned mass. Geometry,
    # COM and inertia scale together; the physical mass remains unmeasured.
    tail_inertia = _required(tail, "inertial")
    tail_inertia.set("pos", _format(TAIL_SCALE * np.fromstring(tail_inertia.attrib["pos"], sep=" ")))
    tail_inertia.set("diaginertia", _format(TAIL_SCALE**2 * np.fromstring(tail_inertia.attrib["diaginertia"], sep=" ")))
    for geom in tail.findall("geom"):
        for attribute in ("pos", "fromto", "size"):
            if attribute in geom.attrib:
                geom.set(attribute, _format(TAIL_SCALE * np.fromstring(geom.attrib[attribute], sep=" ")))
    limits = _format((-TAIL_LIMIT, TAIL_LIMIT))
    ET.SubElement(
        tail,
        "joint",
        name="tail_yaw",
        axis="0 0 1",
        range=limits,
        damping=".001",
        frictionloss=".0002",
        armature=".000002",
    )
    for collision in (False, True):
        attributes = {
            "name": "tail_yaw_servo_case" + ("_collision" if collision else ""),
            "type": "box",
            "pos": "-.0755 0 -.0295",
            "size": ".017 .010 .013",
        }
        attributes.update({"class": "collision"} if collision else {"material": "servo_graphite", "group": "2"})
        ET.SubElement(torso, "geom", attributes)
    ET.SubElement(
        torso,
        "geom",
        name="tail_drive_extension",
        type="cylinder",
        pos="-.083 0 0",
        size=".002 .015",
        material="bracket_silver",
        group="2",
    )
    ET.SubElement(
        torso,
        "geom",
        name="tail_root_bearing",
        type="cylinder",
        pos="-.083 0 .012",
        size=".008 .003",
        material="bracket_graphite",
        group="2",
    )
    for side, y in (("right", -0.014), ("left", 0.014)):
        ET.SubElement(
            torso,
            "geom",
            name=f"tail_support_{side}",
            type="capsule",
            fromto=f"-.038 {y} -.015 -.083 {y} .009",
            size=".002",
            material="bracket_silver",
            group="2",
        )
        ET.SubElement(
            torso,
            "geom",
            name=f"tail_mount_plate_{side}",
            type="box",
            pos=f"-.0755 {y} -.0295",
            size=".018 .0015 .014",
            material="bracket_graphite",
            group="2",
        )
    ET.SubElement(
        torso,
        "geom",
        name="tail_mount_crossbar",
        type="capsule",
        fromto="-.083 -.014 .009 -.083 .014 .009",
        size=".002",
        material="bracket_silver",
        group="2",
    )
    for name in ("schematic_battery_proxy", "battery_sled", "battery_sled_rail_-1", "battery_sled_rail_1"):
        geom = _required(torso, f"geom[@name='{name}']")
        geom.set("pos", _format(np.fromstring(geom.attrib["pos"], sep=" ") + BATTERY_SHIFT))
    ET.SubElement(
        _required(root, "actuator"),
        "position",
        name="tail_yaw_servo",
        joint="tail_yaw",
        ctrlrange=limits,
        kp=".1",
        kv=".008",
        forcerange="-.02 .02",
    )
    for tag, suffix in (("jointpos", "position"), ("jointvel", "velocity")):
        ET.SubElement(_required(root, "sensor"), tag, name=f"tail_yaw_{suffix}", joint="tail_yaw")
    positions.insert(7, "0")
    key.set("qpos", " ".join(positions))
    key.set("ctrl", " ".join([*controls, "0"]))
    _required(root, "custom/numeric[@name='forward_platform_revision']").set("data", "11")
    root.set("model", "mini_tyrannosaurus_rex_tail_yaw_option")
    assets = {
        include.attrib["file"]: (base_path.parent / include.attrib["file"]).read_bytes()
        for include in root.iter("include")
    }
    for name, contents in assets.items():
        asset_root = ET.fromstring(contents)
        tail_mesh = asset_root.find(".//mesh[@name='baby_tail']")
        if tail_mesh is not None:
            tail_mesh.set("scale", _format(TAIL_SCALE * np.fromstring(tail_mesh.get("scale", "1 1 1"), sep=" ")))
            assets[name] = ET.tostring(asset_root)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"), assets=assets)
    if not np.isclose(np.sum(model.body_mass), 0.825, rtol=0, atol=1e-9):
        raise ValueError("Tail option mass differs from 825 g; reconcile the base-model component ledger first.")
    return model
