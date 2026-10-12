"""Surface finishes on the original robot legs, without removing hardware.

Narrow colour fields are clipped to the existing cover surfaces. Their 80 um
render offset represents paint, not a second load-bearing cover. All original
collision meshes and aggregate inertials remain intact. No CAD mass savings
are credited to these visual changes.
"""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


def numbers(values):
    return " ".join(f"{float(v):.12g}" for v in np.asarray(values).ravel())


def clip_polygon(points, axis, bound, sign):
    """Clip a triangle/polygon to sign * (coordinate - bound) >= 0."""
    clipped = []
    for a, b in zip(points, np.roll(points, -1, axis=0), strict=True):
        da, db = sign * (a[axis] - bound), sign * (b[axis] - bound)
        if da >= 0:
            clipped.append(a)
        if (da >= 0) != (db >= 0):
            clipped.append(a + da / (da - db) * (b - a))
    return np.asarray(clipped)


def cover_colour_field(asset, body, name, triangles, side, material, y_bounds):
    vertices, faces = [], []
    for triangle in triangles:
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal /= norm
        # Local -Z faces outward on BOTH mirrored cover bodies.
        if normal[2] * side < 0.65:
            continue
        points = triangle.copy()
        for axis, bound, sign in ((0, -0.008, 1), (0, 0.028, -1), (1, y_bounds[0], 1), (1, y_bounds[1], -1)):
            points = clip_polygon(points, axis, bound, sign)
            if len(points) < 3:
                break
        if len(points) < 3:
            continue
        # Closed thin prisms make valid meshes while following the old surface.
        start, count = len(vertices), len(points)
        vertices.extend(points + normal * 0.00002)
        vertices.extend(points + normal * 0.00008)
        for i in range(1, count - 1):
            faces.extend(((start, start + i + 1, start + i), (start + count, start + count + i, start + count + i + 1)))
        for i in range(count):
            j = (i + 1) % count
            faces.extend(((start + i, start + j, start + count + j), (start + i, start + count + j, start + count + i)))
    if not vertices:
        raise ValueError(f"No outer cover surface selected for {name}")
    ET.SubElement(asset, "mesh", name=name, vertex=numbers(vertices), face=" ".join(str(i) for f in faces for i in f))
    ET.SubElement(body, "geom", name=name, type="mesh", mesh=name, material=material, contype="0", conaffinity="0")


def apply_leg_finish(root):
    asset = root.find("asset")
    for material, rgba in (
        ("frame_mat", ".22 .26 .27 1"),
        ("servo_mat", ".065 .075 .08 1"),
        ("sole_mat", ".10 .12 .12 1"),
    ):
        asset.find(f"material[@name='{material}']").set("rgba", rgba)
    for geom in root.findall(".//geom"):
        mesh = geom.get("mesh", "")
        if mesh in ("left_cache", "right_cache"):
            geom.set("material", "leg_cover_mat")
        elif mesh in ("drive_palonier", "passive_palonier", "left_roll_to_pitch", "right_roll_to_pitch"):
            geom.set("material", "shell_alloy_mat")
        elif geom.get("name", "").endswith("ankle_roll_servo"):
            geom.set("material", "servo_mat")

    # Use MuJoCo's imported vertices and original transforms, not guessed STL
    # axes, to attach the finishes exactly to their existing rigid bodies.
    reference = copy.deepcopy(root)
    reference.find("compiler").set("meshdir", str(Path(__file__).resolve().parents[1] / "assets/meshes"))
    model = mujoco.MjModel.from_xml_string(ET.tostring(reference, encoding="unicode"))
    for side, name, sign in (("left", "knee_and_ankle_assembly", -1), ("right", "knee_and_ankle_assembly_3", -1)):
        body = root.find(f".//body[@name='{name}']")
        g = model.geom(f"{name}_mesh_0").id
        mesh = model.geom_dataid[g]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, model.geom_quat[g])
        vertices = model.mesh_vert[start : start + count] @ rotation.reshape(3, 3).T + model.geom_pos[g]
        start, count = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
        triangles = vertices[model.mesh_face[start : start + count]]
        for label, material, limits in (
            ("upper", "shell_olive_mat", (-0.007, 0.024)),
            ("lower", "shell_alloy_mat", (-0.077, -0.018)),
        ):
            y_bounds = limits if side == "left" else (-limits[1], -limits[0])
            cover_colour_field(asset, body, f"{side}_cover_finish_{label}", triangles, sign, material, y_bounds)

        # Three painted toe marks remain within the existing rectangular sole.
        foot = root.find(f".//body[@name='{side}_rolling_foot']")
        for digit, y in enumerate((-0.028, 0, 0.028), start=2):
            ET.SubElement(
                foot,
                "geom",
                name=f"{side}_toe_finish_{digit}",
                type="box",
                pos=numbers([0.032, y, -0.02994]),
                size=numbers([0.015 if digit == 3 else 0.011, 0.006, 0.00004]),
                material="shell_olive_mat",
                contype="0",
                conaffinity="0",
            )
