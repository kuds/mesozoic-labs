"""Reflect lower leg hardware, not the torso or the forward-facing feet.

Both candidates retain the A layout's complete parts, mass budget, actuator
count, link lengths and angular travel magnitude. The rear candidate reflects
the thigh and shin geometry/inertia in x, with opposite pitch coordinates.
This is a new bracket arrangement, not a claim that a stock servo is mirrored.
The box envelopes do not model asymmetric plugs, horns or fastening features.
"""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from environments.shared.robot_building import values
from examples.mini_tyrannosaurus_rex.layout_study.build import HERE as LAYOUT_ROOT

HERE = Path(__file__).resolve().parent
BASE = LAYOUT_ROOT / "a_low_spine" / "model.xml"
VARIANTS = {"forward": 1, "rearward": -1}
REFLECTION = np.diag([-1.0, 1.0, 1.0])
PITCH = ("hip_pitch", "knee", "ankle")


def reflect_element(node: ET.Element) -> None:
    """Reflect a symmetric primitive or aggregate inertia through the yz plane."""
    if "pos" in node.attrib:
        p = np.fromstring(node.attrib["pos"], sep=" ")
        node.set("pos", values(REFLECTION @ p))
    if "fromto" in node.attrib:
        p = np.fromstring(node.attrib["fromto"], sep=" ").reshape(2, 3)
        node.set("fromto", values((p @ REFLECTION).ravel()))
    if "quat" in node.attrib:
        q = np.fromstring(node.attrib["quat"], sep=" ")
        matrix = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        q = Rotation.from_matrix(REFLECTION @ matrix @ REFLECTION).as_quat()
        node.set("quat", values(q[[3, 0, 1, 2]]))
    if "fullinertia" in node.attrib:
        inertia = np.fromstring(node.attrib["fullinertia"], sep=" ")
        inertia[3:5] *= -1  # Ixy and Ixz; diagonal terms and Iyz stay the same.
        node.set("fullinertia", values(inertia))


def build(name: str, directory: Path | None = None) -> dict[str, Any]:
    sign = VARIANTS[name]
    directory = directory or HERE / name
    directory.mkdir(parents=True, exist_ok=True)
    root = ET.parse(BASE).getroot()
    root.set("model", f"a_torso_{name}_legs")
    if sign == -1:
        for side in ("left", "right"):
            thigh = root.find(f".//body[@name='{side}_thigh']")
            assert thigh is not None
            # All descendants except the foot's contents are reflected. The
            # ankle position changes, while sole, pedestal and foot inertia
            # keep their original orientation and forward toe extension.
            for node in thigh.iter():
                if node is thigh or node.tag == "joint":
                    continue
                foot = thigh.find(f".//body[@name='{side}_foot']")
                assert foot is not None
                if node is not foot and node in list(foot.iter()):
                    continue
                reflect_element(node)
            for suffix in PITCH:
                joint = root.find(f".//joint[@name='{side}_{suffix}']")
                assert joint is not None
                low, high = np.fromstring(joint.attrib["range"], sep=" ")
                joint.set("range", values((-high, -low)))
                actuator = root.find(f"actuator/position[@joint='{side}_{suffix}']")
                assert actuator is not None
                actuator.set("ctrlrange", values((-high, -low)))
    ET.indent(root, space="  ")
    (directory / "model.xml").write_text(ET.tostring(root, encoding="unicode") + "\n")
    record = dict(
        name=name,
        fold_sign=sign,
        source_commit="b7f8cb64ed2cb73f2a041a17e874b6bff1470e03",
        source_model="../layout_study/a_low_spine/model.xml",
        source_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),
        changed="Thigh and shin positions, symmetric geometry and aggregate inertias reflected in x; knee range sign reversed"
        if sign == -1
        else "No physical changes from A",
        unchanged="Torso hardware and payloads, hips, forward-facing feet, motor sizes, link lengths, masses, control gains, torque caps and range magnitudes",
        limitations="Primitive packing model; revised brackets, plug orientation, cable sweeps, horn/idler attachments and shell openings need CAD",
        hardware_ready=False,
    )
    (directory / "variant.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


if __name__ == "__main__":
    for variant in VARIANTS:
        build(variant)
        print(variant, "built")
