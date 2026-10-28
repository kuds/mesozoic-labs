"""Hollow torso skin with individually inspectable convex wall patches."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

import numpy as np

from environments.shared.robot_building import element, values
from examples.mini_tyrannosaurus_rex.head_design import surface_component

WALL_M = 0.0012
DENSITY_KG_M3 = 1050.0  # provisional polymer, not a selected print/material process


def apply_torso(root: ET.Element) -> dict[str, Any]:
    """Replace the solid torso proxy with a thin, open mechanism bay shell.

    Every visible outer triangle has a matching convex wall prism for independent
    geometric checking. Large joint apertures are deliberately open: an enclosing
    convex collision hull would incorrectly fill the electronics cavity.
    """
    torso = root.find(".//body[@name='torso']")
    asset = root.find("asset")
    assert torso is not None and asset is not None
    obsolete = {"torso_shell", "belly_panel", "back_access_panel", "torso_collision", "chest_collision"}
    for geom in list(torso.findall("geom")):
        if geom.get("name") in obsolete or geom.get("name", "").startswith("rear_hatch_fastener"):
            torso.remove(geom)
    # x, vertical center, halfwidth, halfheight. Boxier rear accommodates a
    # transverse 85 mm computer allowance; the front narrows into the neck.
    profile = np.array(
        [
            [-0.115, 0.040, 0.025, 0.025],
            [-0.110, 0.030, 0.051, 0.040],
            [-0.095, 0.030, 0.055, 0.040],
            [-0.040, 0.030, 0.061, 0.040],
            [-0.028, 0.030, 0.058, 0.040],
            [0.018, 0.031, 0.057, 0.039],
            [0.042, 0.025, 0.046, 0.035],
            [0.064, 0.005, 0.046, 0.021],
            [0.112, 0.005, 0.046, 0.021],
            [0.120, 0.005, 0.038, 0.020],
            [0.128, 0.005, 0.004, 0.010],
        ]
    )
    cols = 20
    angles = np.arange(cols) * 2 * np.pi / cols

    def rings(inset: float) -> np.ndarray:
        yz = np.stack(
            (
                np.sign(np.cos(angles)) * abs(np.cos(angles)) ** 0.35,
                np.sign(np.sin(angles)) * abs(np.sin(angles)) ** 0.35,
            ),
            axis=1,
        )
        result = []
        for x, z, w, h in profile:
            result.append(np.column_stack((np.full(cols, x), yz[:, 0] * (w - inset), z + yz[:, 1] * (h - inset))))
        return np.array(result)

    outer = rings(0)
    outer_points: list[list[float]] = []
    outer_faces: list[list[int]] = []
    names: list[str] = []
    omissions: list[dict[str, Any]] = []
    for row in range(len(profile) - 1):
        for col in range(cols):
            nxt = (col + 1) % cols
            q = outer[[row, row, row + 1, row + 1], [col, nxt, nxt, col]]
            mid = q.mean(axis=0)
            # Rear battery lid remains removable. Front underside is open to
            # the hip stack; sides open at the shared shoulder shaft.
            opening = -0.034 < mid[0] < 0.042 and mid[2] < 0.028
            opening |= 0.012 < mid[0] < 0.050 and abs(mid[1]) > 0.035 and abs(mid[2] - 0.048) < 0.019
            opening |= -0.070 < mid[0] < 0.043 and abs(mid[1]) < 0.043 and mid[2] > 0.070
            opening |= -0.040 < mid[0] < 0.026 and abs(mid[1]) < 0.026 and mid[2] > 0.060
            if opening:
                omissions.append(dict(row=row, col=col, reason="open hip/shoulder/neck mechanism aperture"))
                continue
            start = len(outer_points)
            outer_points.extend(q.tolist())
            outer_faces.extend([[start, start + 1, start + 2], [start, start + 2, start + 3]])
            for part, indices in enumerate(((0, 1, 2), (0, 2, 3))):
                triangle = q[list(indices)]
                normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
                normal /= np.linalg.norm(normal)
                vertices = np.vstack((triangle, triangle - WALL_M * normal))
                name = f"juv_torso_wall_{row:02}_{col:02}_{part}"
                faces = [[0, 1, 2], [3, 5, 4], [0, 3, 4], [0, 4, 1], [1, 4, 5], [1, 5, 2], [2, 5, 3], [2, 3, 0]]
                element(asset, "mesh", name=name, vertex=vertices.ravel(), face=np.array(faces).ravel())
                element(
                    torso,
                    "geom",
                    name=name,
                    type="mesh",
                    mesh=name,
                    group=3,
                    contype=1,
                    conaffinity=1,
                    rgba=(0.7, 0.25, 0.1, 0.2),
                )
                names.append(name)
    vertices, faces = np.array(outer_points), np.array(outer_faces)
    element(asset, "mesh", name="juv_torso_skin", vertex=vertices.ravel(), face=faces.ravel())
    element(
        torso,
        "geom",
        name="juv_torso_skin",
        type="mesh",
        mesh="juv_torso_skin",
        material="sage",
        group=1,
        contype=0,
        conaffinity=0,
    )
    triangles = vertices[faces]
    area = float(
        np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1).sum() / 2
    )
    component = surface_component("torso_skin_1p2mm", area * WALL_M * DENSITY_KG_M3, vertices, faces)
    component["basis"] = "outer triangle area x 1.2 mm x assumed 1050 kg/m3; fasteners/frame separately allocated"
    tail = torso.find("body[@name='tail']")
    assert tail is not None
    tail.set("pos", values((-0.115, 0, 0.040)))
    return dict(
        shell_geoms=names,
        skin_geom="juv_torso_skin",
        shell_mass_component=component,
        wall_thickness_m=WALL_M,
        assumed_density_kg_m3=DENSITY_KG_M3,
        surface_area_m2=area,
        open_patches=omissions,
        profile_m=profile.tolist(),
        shell_limitations=[
            "Rear cap, panel flanges, fasteners and internal mounting frame are allocated in mass but need fabrication CAD.",
            "The hip, shoulder and yaw apertures are open; joint covers must remain separate moving parts.",
        ],
    )
