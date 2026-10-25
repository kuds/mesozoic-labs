"""Adapt silhouette C to the mini's retained revision-10 mechanism.

Generate visual/collision surfaces and explicit, unweighed head inertias. The
motor envelopes, pivots, limits and body mass allocations remain full size.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).parent
BASELINE = HERE / "baseline_rev10" / "model.xml"


def values(items: Any) -> str:
    if isinstance(items, str):
        return items
    if isinstance(items, (int, float, np.floating)):
        number = float(items)
        return "0" if abs(number) < 1e-14 else f"{number:.12g}"
    return " ".join(values(item) for item in items)


def find(root: ET.Element, tag: str, name: str) -> ET.Element:
    item = root.find(f".//{tag}[@name='{name}']")
    if item is None:
        raise ValueError(f"Missing {tag}: {name}")
    return item


def geom(parent: ET.Element, name: str, **attrs: Any) -> ET.Element:
    return ET.SubElement(parent, "geom", {"name": name, **{key.rstrip("_"): values(v) for key, v in attrs.items()}})


def rounded_mesh(half: tuple[float, float, float], center: tuple[float, float, float], *, wedge: bool = False):
    """The medium C's rounded surface and taper, fitted to mini dimensions."""
    cols, rings, exponent = 40, 24, 0.55
    vertices: list[tuple[float, ...]] = [(0.0, 0.0, -1.0)]
    for row in range(1, rings):
        latitude = -math.pi / 2 + row * math.pi / rings
        for col in range(cols):
            longitude = col * 2 * math.pi / cols
            raw = (
                math.cos(latitude) * math.cos(longitude),
                math.cos(latitude) * math.sin(longitude),
                math.sin(latitude),
            )
            vertices.append(tuple(math.copysign(abs(x) ** exponent, x) for x in raw))
    vertices.append((0.0, 0.0, 1.0))
    points = np.array(vertices)
    if wedge:
        points[:, 1] *= 0.96 - 0.10 * points[:, 0]
        points[:, 2] -= 0.12 * (points[:, 0] + 1)
    points = points * half + center
    faces = [(0, 1 + (i + 1) % cols, 1 + i) for i in range(cols)]
    for row in range(rings - 2):
        for col in range(cols):
            a, b = 1 + row * cols + col, 1 + row * cols + (col + 1) % cols
            faces.extend(((a, b, a + cols), (b, b + cols, a + cols)))
    last = 1 + (rings - 2) * cols
    faces.extend((last + i, last + (i + 1) % cols, len(points) - 1) for i in range(cols))
    return points, np.array(faces)


def surface_component(name: str, mass: float, vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    """Area-weighted thin-surface moments, not a selected material or thickness."""
    triangles = vertices[faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    weights = areas / areas.sum()
    center = np.sum(weights[:, None] * triangles.mean(axis=1), axis=0)
    sums = triangles.sum(axis=1)
    second = (np.einsum("tvi,tvj->tij", triangles, triangles) + np.einsum("ti,tj->tij", sums, sums)) / 12
    covariance = np.sum(weights[:, None, None] * second, axis=0) - np.outer(center, center)
    inertia = mass * (np.trace(covariance) * np.eye(3) - covariance)
    return dict(
        name=name,
        mass_kg=mass,
        pos_m=center.tolist(),
        inertia_kg_m2=inertia.tolist(),
        basis="uniform_surface_allowance",
    )


def box_component(name: str, mass: float, pos: tuple[float, float, float], half: tuple[float, float, float]):
    squared = np.square(half)
    inertia = np.diag(mass / 3 * (squared.sum() - squared))
    return dict(name=name, mass_kg=mass, pos_m=list(pos), inertia_kg_m2=inertia.tolist(), basis="box_allowance")


def assign_inertia(body: ET.Element, components: list[dict[str, Any]]) -> None:
    mass = sum(row["mass_kg"] for row in components)
    center = sum(row["mass_kg"] * np.array(row["pos_m"]) for row in components) / mass
    inertia = np.zeros((3, 3))
    for row in components:
        offset = np.array(row["pos_m"]) - center
        inertia += np.array(row["inertia_kg_m2"]) + row["mass_kg"] * (
            offset @ offset * np.eye(3) - np.outer(offset, offset)
        )
    element = body.find("inertial")
    assert element is not None
    element.attrib.clear()
    element.set("mass", values(mass))
    element.set("pos", values(center))
    element.set("fullinertia", values(inertia[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]))


def build(directory: Path = HERE) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    root = ET.parse(BASELINE, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))).getroot()
    root[0].text = (
        " Silhouette C: rounded tapered skull, side eyes and an open-backed cream jaw.\n"
        "    Revision-10 mechanism, full motor cases and 800 g allocation retained.\n"
        "    Head moments regenerated from explicit unweighed component allowances.\n"
        "    Leg position/velocity gains reduced for drift-free unsupported stance.\n"
        "    Not fabrication CAD or a validated walking robot. "
    )
    for include in root.iter("include"):
        source = (BASELINE.parent / include.get("file", "")).resolve()
        include.set("file", os.path.relpath(source, directory))
    # The revision-10 gains produced torque clipping and 4--8 cm of creep in a
    # 30 s hold at the 0.10 Nm sensitivity cap. Lower gains remove that observed
    # drift without changing torque caps, geometry, mass or adding support.
    # These are simulation gains, not identified XL330 hardware settings.
    default_servo = root.find("default/position")
    assert default_servo is not None
    for servo in root.findall("actuator/position"):
        if any(part in servo.get("name", "") for part in ("hip_", "knee", "ankle")):
            servo.set("kp", values(float(servo.get("kp", default_servo.attrib["kp"])) * 0.5))
            servo.set("kv", values(float(servo.get("kv", default_servo.attrib["kv"])) * 0.25))
    custom = root.find("custom")
    asset = root.find("asset")
    assert custom is not None and asset is not None
    ET.SubElement(custom, "text", name="character_design", data="silhouette_c_mini")
    palette = {
        "sage": (0.29, 0.38, 0.22, 1),
        "cream": (0.78, 0.71, 0.55, 1),
        "head_brown": (0.29, 0.38, 0.22, 1),
        "brow_brown": (0.16, 0.22, 0.12, 1),
        "claw_brown": (0.22, 0.27, 0.24, 1),
        "eyes": (0.022, 0.030, 0.027, 1),
    }
    for name, rgba in palette.items():
        find(root, "material", name).set("rgba", values(rgba))

    head, jaw = find(root, "body", "head_pitch_body"), find(root, "body", "jaw_body")
    mesh_specs = {
        "mini_c_skull": ((0.058, 0.049, 0.026), (0.027, 0, 0.031), True),
        "mini_c_chin": ((0.016, 0.039, 0.0085), (0.0684, 0, -0.008), False),
        "mini_c_left_rail": ((0.035, 0.006, 0.009), (0.020, 0.035, -0.0075), False),
        "mini_c_right_rail": ((0.035, 0.006, 0.009), (0.020, -0.035, -0.0075), False),
    }
    meshes = {}
    for name, (half, center, wedge) in mesh_specs.items():
        vertices, faces = rounded_mesh(half, center, wedge=wedge)
        meshes[name] = (vertices, faces)
        ET.SubElement(asset, "mesh", name=name, vertex=values(vertices.ravel()), face=" ".join(map(str, faces.ravel())))
    # Matching convex surfaces cover the skull and each jaw piece separately;
    # the central neck opening is retained instead of convexifying the whole U.
    for name, mesh in (
        ("head_shell", "mini_c_skull"),
        ("head_collision", "mini_c_skull"),
        ("lower_jaw", "mini_c_chin"),
        ("jaw_collision", "mini_c_chin"),
    ):
        element = find(root, "geom", name)
        for attr in ("pos", "size"):
            element.attrib.pop(attr, None)
        element.set("type", "mesh")
        element.set("mesh", mesh)
    find(root, "geom", "lower_jaw").set("material", "cream")
    jaw.remove(find(root, "geom", "chin_panel"))
    for side in ("left", "right"):
        geom(jaw, side + "_jaw_rail", type="mesh", mesh="mini_c_" + side + "_rail", material="cream")
        geom(jaw, side + "_jaw_rail_collision", type="mesh", mesh="mini_c_" + side + "_rail", class_="collision")
        for suffix in ("brow_rear", "mouth_corner", "mouth_line"):
            head.remove(find(root, "geom", side + "_" + suffix))
        sign = 1 if side == "left" else -1
        eye = find(root, "geom", side + "_eye")
        eye.set("pos", values((0.003, sign * 0.050, 0.031)))
        eye.set("size", ".0115 .003 .013")
        eye.set("euler", values((0, 0, -sign * 0.22)))
        geom(
            head,
            side + "_eye_socket",
            type="ellipsoid",
            pos=(0, sign * 0.045, 0.031),
            size=(0.0175, 0.004, 0.0185),
            material="brow_brown",
            euler=(0, 0, -sign * 0.22),
        )
        geom(
            head,
            side + "_eye_white",
            type="ellipsoid",
            pos=(0.001, sign * 0.048, 0.031),
            size=(0.015, 0.003, 0.016),
            material="cream",
            euler=(0, 0, -sign * 0.22),
        )
        glint = find(root, "geom", side + "_eye_glint")
        glint.set("pos", values((0.007, sign * 0.053, 0.037)))
        glint.set("size", ".0022")
        tooth = find(root, "geom", side + "_rounded_tooth")
        tooth.set("fromto", values((0.063, sign * 0.033, 0.006, 0.063, sign * 0.033, -0.003)))
        tooth.set("size", ".0022")
        nostril = find(root, "geom", side + "_nostril")
        nostril.set("pos", values((0.083, sign * 0.021, 0.027)))
        nostril.set("size", ".002 .003 .002")

    # Explicit pairs keep parent/child filtering from concealing a closed-jaw
    # overlap. Distances are checked independently in the character report.
    contact = root.find("contact")
    assert contact is not None
    for name in ("jaw_collision", "left_jaw_rail_collision", "right_jaw_rail_collision"):
        ET.SubElement(contact, "pair", geom1="head_collision", geom2=name, condim="4")

    # These allocations reconcile to existing head bodies; they are not extra
    # ledger items or a claim that a printable shell has this mass.
    upper = [
        surface_component("upper_shell_eyes_teeth", 0.028, *meshes["mini_c_skull"]),
        box_component("jaw_motor", 0.018, (0.016, -0.014, 0.026), (0.010, 0.013, 0.017)),
        box_component("head_imu", 0.003, (0.027, 0, 0.018), (0.012, 0.010, 0.003)),
        box_component("head_mounts", 0.006, (0.004, 0, 0.010), (0.014, 0.022, 0.008)),
    ]
    lower = [
        surface_component("front_chin", 0.006, *meshes["mini_c_chin"]),
        surface_component("left_jaw_rail", 0.002, *meshes["mini_c_left_rail"]),
        surface_component("right_jaw_rail", 0.002, *meshes["mini_c_right_rail"]),
        box_component("jaw_hinge_linkage", 0.002, (0, 0, 0), (0.006, 0.020, 0.006)),
    ]
    assign_inertia(head, upper)
    assign_inertia(jaw, lower)
    ET.indent(root, space="  ")
    path = directory / "model.xml"
    path.write_text(ET.tostring(root, encoding="unicode") + "\n")
    allocation = {
        "design": "silhouette_c_mini",
        "base_mechanism": "revision_10",
        "scope": "Unweighed allocations within the existing ledger; surfaces are not hollow-shell CAD.",
        "target_total_mass_g": 800,
        "moving_yaw_subtree_mass_g": 91,
        "retained_yaw_carrier_mass_g": 24,
        "head_pitch_body": upper,
        "jaw_body": lower,
        "mass_budget_addition_g": 0,
        "camera_mass_included": False,
        "inertia_method": "Uniform mesh-surface moments plus hardware boxes, combined by parallel-axis theorem.",
    }
    (directory / "head_allocation.json").write_text(json.dumps(allocation, indent=2, allow_nan=False) + "\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    print(build(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
