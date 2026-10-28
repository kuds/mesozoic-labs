"""Juvenile-theropod exterior on the mini's retained full-size mechanism.

The meshes are visible concept surfaces, not manufacturing CAD. Every new
external portion also has an identically shaped inspection/collision geom.
Moving fairings remain on one rigid body and leave the original pivots intact.
"""

from __future__ import annotations

import copy
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

from ..head_design import assign_inertia, box_component, find, geom, surface_component, values


def _loft(
    profile: list[tuple[float, float, float, float]], *, side: float = 0.0, subdivisions: int = 5, cols: int = 32
):
    """Closed smooth x-stations: x, z, y-halfwidth, z-halfheight."""
    sections = []
    for left, right in zip(profile[:-1], profile[1:], strict=True):
        for t in np.linspace(0, 1, subdivisions, endpoint=False):
            w = float(t * t * (3 - 2 * t))
            sections.append(
                (left[0] + t * (right[0] - left[0]), *[left[i] + w * (right[i] - left[i]) for i in (1, 2, 3)])
            )
    sections.append(profile[-1])
    vertices = []
    for x, z, width, height in sections:
        for angle in np.arange(cols) * (2 * math.pi / cols):
            c, s = math.cos(angle), math.sin(angle)
            vertices.append(
                (x, side + width * math.copysign(abs(c) ** 0.70, c), z + height * math.copysign(abs(s) ** 0.70, s))
            )
    faces = []
    for row in range(len(sections) - 1):
        for i in range(cols):
            a, b = row * cols + i, row * cols + (i + 1) % cols
            faces.extend(((a, b, a + cols), (b, b + cols, a + cols)))
    vertices.extend(((profile[0][0], side, profile[0][1]), (profile[-1][0], side, profile[-1][1])))
    end = (len(sections) - 1) * cols
    for i in range(cols):
        faces.extend(((len(vertices) - 2, (i + 1) % cols, i), (len(vertices) - 1, end + i, end + (i + 1) % cols)))
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)


def _skull_walls(vertices: np.ndarray, faces: np.ndarray, *, cols: int, thickness: float = 0.0012):
    """True thin walls; open drive bay below and a forward camera aperture.

    Each visible triangle is the outer face of an independently convex prism.
    Thus MuJoCo never convexifies the skull's empty installation cavity.
    """
    vertices = vertices.tolist()
    front_center = len(vertices) - 1
    front_row = front_center - 1 - cols
    kept = []
    for face in faces.tolist():
        points = np.asarray([vertices[i] for i in face])
        if front_center in face:
            continue
        # Open access under the retained jaw motor and nod shaft. The boundary
        # follows the actual triangle edges and is present in the visible mesh.
        if points[:, 0].mean() < 0.045 and points[:, 2].mean() < 0.020:
            continue
        kept.append(tuple(face))
    inner_start = len(vertices)
    for angle in np.arange(cols) * 2 * math.pi / cols:
        vertices.append((0.108, 0.006 * math.cos(angle), 0.027 + 0.006 * math.sin(angle)))
    for i in range(cols):
        a, b = front_row + i, front_row + (i + 1) % cols
        c, d = inner_start + i, inner_start + (i + 1) % cols
        kept.extend(((c, a, b), (c, b, d)))
    vertices_array = np.array(vertices)
    smooth_normals = np.zeros_like(vertices_array)
    for face in kept:
        outer = vertices_array[list(face)]
        area_normal = np.cross(outer[1] - outer[0], outer[2] - outer[0])
        smooth_normals[list(face)] += area_normal
    lengths = np.linalg.norm(smooth_normals, axis=1, keepdims=True)
    smooth_normals /= np.maximum(lengths, 1e-15)
    prism_faces = np.array(((0, 1, 2), (5, 4, 3), (0, 3, 4), (0, 4, 1), (1, 4, 5), (1, 5, 2), (2, 5, 3), (2, 3, 0)))
    walls = []
    visual_vertices, visual_faces, visual_normals = [], [], []
    for face in kept:
        outer = vertices_array[list(face)]
        normal = np.cross(outer[1] - outer[0], outer[2] - outer[0])
        normal /= np.linalg.norm(normal)
        wall = np.concatenate((outer, outer - thickness * normal))
        offset = len(visual_vertices)
        visual_vertices.extend(wall)
        visual_faces.extend(prism_faces + offset)
        for index, triangle in enumerate(prism_faces):
            if index == 0:
                visual_normals.extend(smooth_normals[list(face)])
            elif index == 1:
                visual_normals.extend(-smooth_normals[list(face)][::-1])
            else:
                points = wall[triangle]
                side_normal = np.cross(points[1] - points[0], points[2] - points[0])
                side_normal /= np.linalg.norm(side_normal)
                visual_normals.extend((side_normal,) * 3)
        walls.append((wall, prism_faces))
    return (np.array(visual_vertices), np.array(visual_faces)), walls, np.array(visual_normals)


def _mesh(
    asset: ET.Element, name: str, vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray | None = None
) -> None:
    if normals is not None:
        vertices = vertices[faces].reshape(-1, 3)
        faces = np.arange(len(vertices)).reshape(-1, 3)
    element = ET.SubElement(
        asset, "mesh", name=name, vertex=values(vertices.ravel()), face=" ".join(map(str, faces.ravel()))
    )
    if normals is not None:
        element.set("normal", values(normals.ravel()))


def _outboard_cover(center: tuple[float, float, float], half: tuple[float, float, float], sign: int):
    """Convex half-ellipsoid with a flat mounting back on an outboard face."""
    cols, rings = 32, 10
    vertices = [(center[0], center[1] + sign * half[1], center[2])]
    for theta in np.linspace(0, math.pi / 2, rings + 1)[1:]:
        for angle in np.arange(cols) * 2 * math.pi / cols:
            vertices.append(
                (
                    center[0] + half[0] * math.sin(theta) * math.cos(angle),
                    center[1] + sign * half[1] * math.cos(theta),
                    center[2] + half[2] * math.sin(theta) * math.sin(angle),
                )
            )
    faces = [(0, 1 + i, 1 + (i + 1) % cols) for i in range(cols)]
    for row in range(rings - 1):
        for i in range(cols):
            a, b = 1 + row * cols + i, 1 + row * cols + (i + 1) % cols
            faces.extend(((a, a + cols, b), (b, a + cols, b + cols)))
    vertices.append(center)
    last = 1 + (rings - 1) * cols
    faces.extend((len(vertices) - 1, last + (i + 1) % cols, last + i) for i in range(cols))
    if sign > 0:
        faces = [(a, c, b) for a, b, c in faces]
    return np.array(vertices), np.array(faces)


def _inertia_component(body: ET.Element) -> dict[str, Any]:
    inertial = body.find("inertial")
    if inertial is None:
        raise ValueError(f"Missing explicit inertia on {body.get('name')}")
    if "fullinertia" in inertial.attrib:
        xx, yy, zz, xy, xz, yz = map(float, inertial.attrib["fullinertia"].split())
        tensor = [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]
    else:
        if inertial.get("quat", "1 0 0 0") != "1 0 0 0":
            raise ValueError("A rotated principal inertia needs an explicit conversion before augmentation.")
        tensor = np.diag(list(map(float, inertial.attrib["diaginertia"].split()))).tolist()
    return dict(
        name="retained_mechanism_allocation",
        mass_kg=float(inertial.attrib["mass"]),
        pos_m=list(map(float, inertial.get("pos", "0 0 0").split())),
        inertia_kg_m2=tensor,
        basis="retained_explicit_body_inertia",
    )


def apply_appearance(root: ET.Element, output_dir: Path) -> dict[str, Any]:
    """Apply the external candidate; return mass and geometry inspection metadata.

    ``output_dir`` is accepted for the build API; meshes are embedded in the
    XML so the generated appearance requires no additional asset files.
    Torso shape and hardware placement are owned by the packaging builder.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if root.find(".//mesh[@name='juv_skull_mesh']") is not None:
        raise ValueError("Apply the juvenile appearance once to a fresh silhouette-C model.")
    asset = root.find("asset")
    assert asset is not None
    shells: dict[str, list[str]] = {}
    collisions: dict[str, list[str]] = {}
    allocations: dict[str, list[dict[str, Any]]] = {}
    deltas: dict[str, float] = {}
    interfaces: list[dict[str, Any]] = []
    rigid_mounts: list[dict[str, Any]] = []

    def register(body: ET.Element, visual: str, collision: str) -> None:
        name = body.attrib["name"]
        shells.setdefault(name, []).append(visual)
        collisions.setdefault(name, []).append(collision)

    def primitive(body: ET.Element, name: str, *, material: str = "sage", **attrs: Any) -> None:
        geom(body, name, material=material, group=1, **attrs)
        geom(body, name + "_collision", class_="collision", **attrs)
        register(body, name, name + "_collision")

    def mesh_geom(body: ET.Element, name: str, mesh_name: str, material: str, collision_name: str | None = None):
        collision_name = collision_name or name + "_collision"
        for existing_name in (name, collision_name):
            old = root.find(f".//geom[@name='{existing_name}']")
            if old is not None:
                body.remove(old)
        geom(body, name, type="mesh", mesh=mesh_name, material=material, group=1)
        geom(body, collision_name, type="mesh", mesh=mesh_name, class_="collision")
        register(body, name, collision_name)

    def augment(body: ET.Element, additions: list[dict[str, Any]]) -> None:
        retained = _inertia_component(body)
        rows = [retained, *additions]
        assign_inertia(body, rows)
        allocations[body.attrib["name"]] = rows
        deltas[body.attrib["name"]] = sum(row["mass_kg"] for row in additions)

    # Keep the family palette, with restrained dark sockets rather than a large
    # white eye surround. Colors are linear-light values used by the renderer.
    for name, rgba in {
        "sage": (0.29, 0.38, 0.22, 1),
        "head_brown": (0.29, 0.38, 0.22, 1),
        "cream": (0.78, 0.71, 0.55, 1),
        "brow_brown": (0.20, 0.27, 0.16, 1),
    }.items():
        find(root, "material", name).set("rgba", values(rgba))

    head = find(root, "body", "head_pitch_body")
    jaw = find(root, "body", "jaw_body")
    skull = _loft(
        [
            (-0.037, 0.034, 0.010, 0.014),
            (-0.028, 0.036, 0.037, 0.021),
            (-0.010, 0.037, 0.044, 0.023),
            (0.020, 0.036, 0.045, 0.026),
            (0.045, 0.032, 0.038, 0.023),
            (0.080, 0.029, 0.034, 0.023),
            (0.100, 0.028, 0.029, 0.023),
            (0.108, 0.028, 0.019, 0.017),
        ],
        subdivisions=1,
        cols=16,
    )
    outer_skull = skull
    skull, skull_walls, skull_normals = _skull_walls(*outer_skull, cols=16)
    _mesh(asset, "juv_skull_mesh", *skull, normals=skull_normals)
    for existing_name in ("head_shell", "head_collision"):
        old = head.find(f"geom[@name='{existing_name}']")
        if old is not None:
            head.remove(old)
    geom(head, "head_shell", type="mesh", mesh="juv_skull_mesh", material="head_brown", group=1)
    shells.setdefault("head_pitch_body", []).append("head_shell")
    for i, wall in enumerate(skull_walls):
        name = f"juv_skull_wall_{i:03d}"
        _mesh(asset, name + "_mesh", *wall)
        geom(head, name, type="mesh", mesh=name + "_mesh", class_="collision")
        collisions.setdefault("head_pitch_body", []).append(name)
    # A U-shaped jaw retains a real central neck/drive opening. It is not
    # convexified into one block, which would conceal the necessary opening.
    chin = _loft(
        [
            (0.078, -0.003, 0.032, 0.005),
            (0.089, -0.004, 0.035, 0.007),
            (0.103, -0.002, 0.027, 0.006),
            (0.108, 0.000, 0.014, 0.004),
        ]
    )
    _mesh(asset, "juv_chin_mesh", *chin)
    mesh_geom(jaw, "lower_jaw", "juv_chin_mesh", "cream", "jaw_collision")
    jaw_rows = [surface_component("front_chin", 0.004, *chin)]
    for side, sign in (("left", 1), ("right", -1)):
        rail = _loft(
            [
                (-0.011, -0.004, 0.005, 0.005),
                (0.008, -0.003, 0.007, 0.009),
                (0.040, -0.004, 0.006, 0.008),
                (0.077, -0.003, 0.006, 0.006),
                (0.089, -0.003, 0.004, 0.005),
            ],
            side=sign * 0.035,
        )
        # The rear cheeks flare around the full nod motor and yoke; the front
        # tapers inward to join the chin. A straight y=35 mm rail fouls the yoke.
        rail[0][:, 1] += sign * np.interp(rail[0][:, 0], (-0.011, 0.040, 0.077, 0.089), (0.007, 0.007, 0, -0.004))
        mesh_name = f"juv_{side}_jaw_rail_mesh"
        _mesh(asset, mesh_name, *rail)
        mesh_geom(jaw, side + "_jaw_rail", mesh_name, "cream", side + "_jaw_rail_collision")
        jaw_rows.append(surface_component(side + "_jaw_rail", 0.003, *rail))
        for suffix in ("eye_white", "eye_socket", "rounded_tooth"):
            old = head.find(f"geom[@name='{side}_{suffix}']")
            if old is not None:
                head.remove(old)
        eye = find(root, "geom", side + "_eye")
        eye.set("pos", values((-0.006, sign * 0.046, 0.038)))
        eye.set("size", ".008 .0025 .009")
        eye.set("euler", "0 0 0")
        glint = find(root, "geom", side + "_eye_glint")
        glint.set("pos", values((-0.003, sign * 0.0484, 0.041)))
        glint.set("size", ".0012")
        primitive(
            head, f"juv_{side}_brow", type="ellipsoid", pos=(-0.006, sign * 0.043, 0.0505), size=(0.022, 0.008, 0.0065)
        )
        primitive(
            head, f"juv_{side}_cheek", type="ellipsoid", pos=(-0.014, sign * 0.045, 0.026), size=(0.015, 0.008, 0.008)
        )
        nostril = find(root, "geom", side + "_nostril")
        nostril.set("pos", values((0.100, sign * 0.023, 0.032)))
        nostril.set("size", ".003 .0015 .002")
        for i, x in enumerate((0.052, 0.068, 0.084)):
            primitive(
                head,
                f"juv_{side}_tooth_{i}",
                material="cream",
                type="capsule",
                fromto=(x, sign * 0.029, 0.025, x + 0.001, sign * 0.029, 0.010),
                size=0.0017,
            )
        # These decorative eye surfaces project less than the surrounding brow;
        # its outer envelope is included in independent clearance checks.
    upper_rows = [surface_component("upper_skull_surface", 0.024, *skull)]
    for side, sign in (("left", 1), ("right", -1)):
        upper_rows.extend(
            (
                box_component(f"{side}_brow", 0.0008, (-0.006, sign * 0.043, 0.0505), (0.022, 0.008, 0.0065)),
                box_component(f"{side}_cheek", 0.0005, (-0.014, sign * 0.045, 0.026), (0.015, 0.008, 0.008)),
                box_component(f"{side}_eye", 0.0005, (-0.006, sign * 0.046, 0.038), (0.008, 0.0025, 0.009)),
                box_component(f"{side}_teeth", 0.0002, (0.068, sign * 0.029, 0.0175), (0.017, 0.0017, 0.0092)),
            )
        )
    old_allocation = json.loads(Path(__file__).parent.parent.joinpath("head_allocation.json").read_text())
    upper_rows += copy.deepcopy(old_allocation["head_pitch_body"][1:])
    jaw_rows.append(box_component("jaw_hinge_linkage", 0.002, (0, 0, 0), (0.006, 0.020, 0.006)))
    for body, rows in ((head, upper_rows), (jaw, jaw_rows)):
        old_mass = float(body.find("inertial").attrib["mass"])
        assign_inertia(body, rows)
        allocations[body.attrib["name"]] = rows
        deltas[body.attrib["name"]] = sum(row["mass_kg"] for row in rows) - old_mass

    # Preserve the explicit head/jaw contacts despite MuJoCo's parent filter.
    # The old filled convex head is replaced by actual wall pieces, not merely
    # hidden from contact or from installation-envelope checks.
    contact = root.find("contact")
    assert contact is not None
    for pair in list(contact.findall("pair")):
        if "head_collision" in (pair.get("geom1"), pair.get("geom2")):
            contact.remove(pair)
    for first in collisions["head_pitch_body"]:
        for second in collisions["jaw_body"]:
            ET.SubElement(contact, "pair", geom1=first, geom2=second, condim="4")

    neck = find(root, "body", "head_yaw_body")
    neck_rows = []
    for name, pos, half in (
        ("juv_neck_back", (-0.022, 0, 0.0145), (0.004, 0.033, 0.0115)),
        ("juv_neck_front", (0.022, 0, 0.0055), (0.006, 0.023, 0.0025)),
        ("juv_neck_left", (-0.020, 0.030, 0.0145), (0.007, 0.003, 0.0115)),
        ("juv_neck_right", (-0.020, -0.030, 0.0145), (0.007, 0.003, 0.0115)),
    ):
        primitive(
            neck, name, type="ellipsoid", pos=pos, size=half, material="cream" if name.endswith("front") else "sage"
        )
        neck_rows.append(box_component(name + "_allowance", 0.00075, pos, half))
    augment(neck, neck_rows)
    interfaces.append(
        dict(
            first_body="torso",
            second_body="head_yaw_body",
            purpose="Open yaw seam; candidate torso neck must stop below the rotating carrier.",
        )
    )
    interfaces.append(
        dict(
            first_body="head_yaw_body",
            second_body="head_pitch_body",
            purpose="Open nod axis; no fairing bridges the hinge.",
        )
    )

    # Partial outboard covers seat at the existing cases' lateral faces, with
    # a 2.5 mm installation allowance. They do not fill the internal mechanisms.
    for side, sign in (("left", 1), ("right", -1)):
        for body in root.iter("body"):
            for item in list(body.findall("geom")):
                name = item.get("name", "")
                if name.startswith(side + "_") and ("partial_fairing" in name or "fairing_fastener" in name):
                    body.remove(item)
        for part, pos, half in (
            ("hip_roll_body", (0.000, sign * 0.0450, -0.0075), (0.016, 0.006, 0.019)),
            ("thigh", (0.010, sign * 0.0300, -0.050), (0.016, 0.006, 0.021)),
            ("shin", (-0.0175, sign * 0.0300, -0.0425), (0.021, 0.006, 0.014)),
        ):
            body = find(root, "body", f"{side}_{part}")
            name = f"juv_{side}_{part}_shell"
            cover = _outboard_cover(pos, half, sign)
            _mesh(asset, name + "_mesh", *cover)
            mesh_geom(body, name, name + "_mesh", "sage")
            augment(body, [surface_component(name + "_allowance", 0.001, *cover)])
            interfaces.append(
                dict(
                    first_body=body.attrib["name"],
                    second_body=body.attrib["name"],
                    purpose="Outboard cover follows the existing motor case face; 2.5 mm allowance reserves standoffs and no pivot is enclosed.",
                )
            )
        foot = find(root, "body", side + "_foot")
        toe_rows = []
        for i, y in enumerate((-0.010, 0, 0.010)):
            pos = (0.028, y, -0.0185)
            name = f"juv_{side}_toe_{i}"
            dome = _outboard_cover((0, 0, 0), (0.013, 0.0053, 0.0045), 1)
            # Swap lateral and vertical coordinates, reversing winding. The
            # mounting plane is exactly the retained baseplate's top face.
            dome = (dome[0][:, (0, 2, 1)] + pos, dome[1][:, (0, 2, 1)])
            _mesh(asset, name + "_mesh", *dome)
            mesh_geom(foot, name, name + "_mesh", "sage")
            rigid_mounts.append(
                dict(
                    first=name + "_collision",
                    second=side + "_foot_baseplate",
                    clearance_m=0.0,
                    reason="Rigid toe overlay flat bottom is seated on the retained baseplate top face; touching is the designed bonded/mounted interface, penetration is not permitted.",
                )
            )
            primitive(
                foot,
                name + "_claw",
                type="ellipsoid",
                pos=(0.0395, y, -0.0145),
                size=(0.0025, 0.004, 0.0020),
                material="cream",
            )
            toe_rows.append(surface_component(name + "_shell_allowance", 0.00035, *dome))
            toe_rows.append(
                box_component(name + "_claw_allowance", 0.00015, (0.0395, y, -0.0145), (0.0025, 0.004, 0.0020))
            )
        augment(foot, toe_rows)
        interfaces.append(
            dict(
                first_body=side + "_foot",
                second_body=side + "_foot",
                purpose="Toe covers seat on the retained baseplate; sole collision footprint unchanged.",
            )
        )

    tail = find(root, "body", "tail")
    for item in list(tail.findall("geom")):
        if item.get("name", "").startswith("tail_"):
            tail.remove(item)
    tail_mesh = _loft(
        [
            (-0.165, 0.006, 0.0015, 0.0015),
            (-0.140, 0.005, 0.004, 0.004),
            (-0.090, 0.002, 0.010, 0.010),
            (-0.040, 0, 0.016, 0.016),
            (0, 0, 0.021, 0.020),
        ]
    )
    _mesh(asset, "juv_tail_mesh", *tail_mesh)
    mesh_geom(tail, "juv_tail_shell", "juv_tail_mesh", "sage")
    tail_rows = [surface_component("lightweight_fixed_tail_surface", 0.015, *tail_mesh)]
    old_tail_mass = float(tail.find("inertial").attrib["mass"])
    assign_inertia(tail, tail_rows)
    allocations["tail"] = tail_rows
    deltas["tail"] = 0.015 - old_tail_mass
    interfaces.append(
        dict(
            first_body="torso",
            second_body="tail",
            purpose="Fixed-tail attachment at the root; mounting interface is intentionally coincident.",
        )
    )

    # Existing two-finger arms and shared transmission are retained. The arm
    # shell remains light; it is not qualified as a fall-recovery load path.
    return {
        "design": "juvenile_theropod",
        "shell_geoms_by_body": shells,
        "collision_geoms_by_body": collisions,
        "inspection_geoms_by_body": copy.deepcopy(collisions),
        "intentional_interfaces": interfaces,
        "rigid_mount_interfaces": rigid_mounts,
        "body_component_allocations": allocations,
        "mass_deltas_kg": deltas,
        "total_added_mass_kg": sum(deltas.values()),
        "mesh_source": "Inline deterministic vertex/triangle arrays in the generated MJCF.",
        "head_cavity": {
            "wall_thickness_m": 0.0012,
            "outer_loft_vertices_m": outer_skull[0].tolist(),
            "outer_loft_faces": outer_skull[1].tolist(),
            "loft_columns": 16,
            "loft_row_count": 8,
            "wall_geoms": [f"juv_skull_wall_{i:03d}" for i in range(len(skull_walls))],
            "camera_aperture": {"center_m": [0.108, 0, 0.027], "radius_m": 0.006, "axis": "+x"},
            "underside_access": "Surface triangles with mean x < 45 mm and mean z < 20 mm are omitted; opening is visible and checkable.",
            "validation": "Check every hardware reserve against every wall prism plus exterior features; verify component corners within the loft cross-section except intentional underside access. No solid convex skull is used.",
        },
        "limitations": [
            "Appearance meshes represent concept geometry; skull walls model 1.2 mm nominal thickness, while fastening, ventilation and manufacturing CAD remain unqualified.",
            "Skull and jaw replacement retain their existing assigned masses; new neck/leg/toe covers add explicit 12 g allowances.",
            "Motor centers, joint axes/ranges, actuator settings and original sole collision footprints are unchanged by appearance generation.",
            "A real 12 mm nose aperture clears the camera optical axis; the actual lens field of view, window material, ribbon bends and lens-to-window mount remain unqualified.",
            "Body-local box inertia surrogates are used for small new covers, with explicit assigned mass rather than inferred material density.",
        ],
    }
