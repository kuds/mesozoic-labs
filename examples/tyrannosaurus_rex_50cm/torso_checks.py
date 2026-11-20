"""Check the continuous torso surface against its collision and hardware bounds.

These are geometric checks on the compiled planning model. They do not establish
wall thickness, manufactured tolerances, strength or service-removal access.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

from .torso_shell import COLLISIONS

GEOMETRY_TOLERANCE_M = 1e-7
WALL_ALLOWANCE_M = 0.003


def _mesh(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> tuple[np.ndarray, np.ndarray]:
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if geom < 0 or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
        raise ValueError(f"Torso geometry must reference a compiled mesh: {name}.")
    mesh = model.geom_dataid[geom]
    start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    face_start, face_count = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
    vertices = model.mesh_vert[start : start + count] @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
    return vertices, model.mesh_face[face_start : face_start + face_count].copy()


def _components(adjacent: dict[int, set[int]]) -> int:
    unseen = set(adjacent)
    components = 0
    while unseen:
        components += 1
        pending = [unseen.pop()]
        while pending:
            for neighbor in adjacent[pending.pop()]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    pending.append(neighbor)
    return components


def _topology(vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    """A single closed, consistently wound triangle surface has no open seams."""
    directed = Counter((int(a), int(b)) for face in faces for a, b in zip(face, np.roll(face, -1), strict=True))
    edges: Counter[tuple[int, int]] = Counter()
    adjacent: dict[int, set[int]] = {}
    for (a, b), count in directed.items():
        edges[min(a, b), max(a, b)] += count
        adjacent.setdefault(a, set()).add(b)
        adjacent.setdefault(b, set()).add(a)
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for index, face in enumerate(faces):
        for a, b in zip(face, np.roll(face, -1), strict=True):
            key = min(int(a), int(b)), max(int(a), int(b))
            edge_faces.setdefault(key, []).append(index)
    neighbors: dict[int, set[int]] = {index: set() for index in range(len(faces))}
    for incident in edge_faces.values():
        for face in incident:
            neighbors[face].update(incident)
    components = _components(neighbors)
    triangles = vertices[faces]
    double_area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    boundary = sum(count != 2 for count in edges.values())
    winding = sum(directed[a, b] != 1 or directed[b, a] != 1 for a, b in edges)
    unused = len(vertices) - len(adjacent)
    degenerate = int(np.count_nonzero(double_area <= 1e-14))
    return {
        "name": "continuous_closed_surface",
        "status": "passed" if components == 1 and not (boundary or winding or unused or degenerate) else "failed",
        "connected_components": components,
        "connected_vertex_components": _components(adjacent),
        "nonmanifold_edges": boundary,
        "inconsistently_wound_edges": winding,
        "unused_vertices": unused,
        "degenerate_triangles": degenerate,
    }


def _surface_intersects_box(triangles: np.ndarray, halfsize: np.ndarray) -> bool:
    """Exact triangle/AABB separating-axis test, with touching counted as contact.

    Triangles are already transformed into the box's frame. In addition to box
    axes and triangle normals, all nine edge/box-axis crosses are required.
    """
    edges = np.roll(triangles, -1, axis=1) - triangles
    basis = np.eye(3)
    axes = np.concatenate(
        (
            np.broadcast_to(basis, (len(triangles), 3, 3)),
            np.cross(edges[:, 0], edges[:, 1])[:, None, :],
            np.cross(edges[:, :, None, :], basis[None, None, :, :]).reshape(-1, 9, 3),
        ),
        axis=1,
    )
    projections = np.einsum("nvc,nac->nva", triangles, axes)
    radius = np.abs(axes) @ halfsize
    # Scale numerical slack with axis length; cross-product axes are not unit.
    slack = np.linalg.norm(axes, axis=2) * GEOMETRY_TOLERANCE_M
    separated = (projections.min(axis=1) > radius + slack) | (projections.max(axis=1) < -radius - slack)
    return bool(np.any(~np.any(separated, axis=1)))


def _inside_surface(point: np.ndarray, triangles: np.ndarray) -> bool:
    """Require odd ray parity in three oblique directions on the actual surface."""
    edge1, edge2 = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    delta = point - triangles[:, 0]
    counts = []
    for direction in (np.array([1.0, 0.371, 0.193]), np.array([-0.217, 1.0, 0.413]), np.array([0.123, -0.173, 1.0])):
        cross = np.cross(direction, edge2)
        determinant = np.einsum("ij,ij->i", edge1, cross)
        active = np.abs(determinant) > 1e-14
        inverse = np.zeros(len(triangles))
        inverse[active] = 1 / determinant[active]
        u = np.einsum("ij,ij->i", delta, cross) * inverse
        q = np.cross(delta, edge1)
        v = (q @ direction) * inverse
        distance = np.einsum("ij,ij->i", edge2, q) * inverse
        hit = active & (u >= -1e-10) & (v >= -1e-10) & (u + v <= 1 + 1e-10) & (distance > GEOMETRY_TOLERANCE_M)
        ordered = np.sort(distance[hit])
        # One crossing on a shared triangle edge is one surface crossing.
        counts.append(int(bool(len(ordered))) + int(np.count_nonzero(np.diff(ordered) > GEOMETRY_TOLERANCE_M)))
    return all(count % 2 == 1 for count in counts)


def check_torso(model: mujoco.MjModel, manifest: dict[str, Any]) -> dict[str, Any]:
    """Check seams, conservative collision coverage and entire hardware reserves.

    A closed surface, an interior center and no surface intersection with an
    inflated box establish containment of that whole connected box. Checking
    only its corners would not suffice for this nonconvex outer torso.
    """
    if not isinstance(manifest.get("assemblies"), list) or not manifest["assemblies"]:
        raise ValueError("Torso validation requires declared hardware assemblies.")
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    vertices, faces = _mesh(model, data, "torso_shell")
    triangles = vertices[faces]
    checks = [_topology(vertices, faces)]
    shell_body = model.geom_bodyid[model.geom("torso_shell").id]
    covered = np.zeros(len(faces), dtype=bool)
    for name in COLLISIONS:
        proxy_vertices, _ = _mesh(model, data, name)
        same_body = model.geom_bodyid[model.geom(name).id] == shell_body
        checks.append({"name": name + ":rigid_frame", "status": "passed" if same_body else "failed"})
        hull = ConvexHull(proxy_vertices)
        equations = hull.equations
        inside = np.all(vertices @ equations[:, :3].T + equations[:, 3] <= GEOMETRY_TOLERANCE_M, axis=1)
        covered |= np.all(inside[faces], axis=1)
    checks.append(
        {
            "name": "visible_surface_collision_coverage",
            "status": "passed" if covered.all() else "failed",
            "uncovered_triangles": int(np.count_nonzero(~covered)),
            "surface_triangles": len(faces),
            "criterion": "Every triangle lies in one declared convex collision section.",
        }
    )
    for assembly in manifest["assemblies"]:
        geom = model.geom(assembly["geometry"]).id
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError("Torso hardware reservations must be boxes.")
        rotation = data.geom_xmat[geom].reshape(3, 3)
        center = data.geom_xpos[geom]
        local = (triangles - center) @ rotation
        contact = _surface_intersects_box(local, model.geom_size[geom] + WALL_ALLOWANCE_M)
        interior = _inside_surface(center, triangles)
        same_body = model.geom_bodyid[geom] == shell_body
        checks.append(
            {
                "name": assembly["id"] + ":outer_shell_containment",
                "status": "passed" if interior and not contact and same_body else "failed",
                "center_inside": interior,
                "inflated_box_intersects_surface": contact,
                "same_rigid_body": bool(same_body),
                "wall_allowance_m": WALL_ALLOWANCE_M,
            }
        )
    passed = all(check["status"] == "passed" for check in checks)
    return {
        "schema_version": 1,
        "status": "incomplete" if passed else "failed",
        "known_limits_passed": passed,
        "checks": checks,
        "coverage": {
            "shell_vertices": len(vertices),
            "shell_triangles": len(faces),
            "convex_sections": len(COLLISIONS),
            "hardware_reserves": len(manifest["assemblies"]),
        },
        "limitations": [
            "Topology, collision coverage and containment use the compiled planning surface, not fabrication CAD.",
            "The 3 mm inflated-box allowance is a geometric planning margin, not a manufactured shell-wall specification.",
            "Mounting strength, shell thickness, tolerances, wiring and service-removal paths remain unqualified.",
        ],
    }
