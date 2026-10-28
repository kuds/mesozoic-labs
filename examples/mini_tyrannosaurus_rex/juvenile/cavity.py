"""Whole-box enclosure in closed design lofts, separate from physical wall gaps.

The reference volumes close the design's intentional joint/service openings.
They establish whether installed parts stay inside the intended exterior, not
whether a manufactured shell is watertight. Actual wall prisms and their 2 mm
hardware clearance are checked separately by the clearance validator.
"""

from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Any

import mujoco
import numpy as np


def _torso_loft(profile: list[list[float]], columns: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Recreate the exact polygonal outer rings, including reference end caps."""
    stations = np.asarray(profile, dtype=float)
    if stations.ndim != 2 or stations.shape[1] != 4 or len(stations) < 2:
        raise ValueError("Torso reference requires x, z, halfwidth, halfheight stations")
    if not np.isfinite(stations).all() or np.any(np.diff(stations[:, 0]) <= 0) or np.any(stations[:, 2:] <= 0):
        raise ValueError("Torso reference requires finite increasing stations and positive dimensions")
    angles = np.arange(columns) * 2 * np.pi / columns
    yz = np.column_stack(
        (np.sign(np.cos(angles)) * abs(np.cos(angles)) ** 0.35, np.sign(np.sin(angles)) * abs(np.sin(angles)) ** 0.35)
    )
    vertices = [(x, y * width, z + height * vertical) for x, z, width, height in stations for y, vertical in yz]
    faces: list[tuple[int, int, int]] = []
    for row in range(len(stations) - 1):
        for col in range(columns):
            a, b = row * columns + col, row * columns + (col + 1) % columns
            # Match torso.py's actual outer-triangle diagonal.
            faces.extend(((a, b, b + columns), (a, b + columns, a + columns)))
    vertices.extend(((stations[0, 0], 0, stations[0, 1]), (stations[-1, 0], 0, stations[-1, 1])))
    rear, front = len(vertices) - 2, len(vertices) - 1
    last = (len(stations) - 1) * columns
    for col in range(columns):
        nxt = (col + 1) % columns
        faces.extend(((rear, nxt, col), (front, last + col, last + nxt)))
    return np.asarray(vertices), np.asarray(faces, dtype=int)


def _checked_mesh(vertices: Any, faces: Any) -> tuple[np.ndarray, np.ndarray]:
    points, triangles = np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Reference vertices must be finite three-dimensional points")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles):
        raise ValueError("Reference faces must be triangles")
    if triangles.min() < 0 or triangles.max() >= len(points):
        raise ValueError("Reference triangle index out of range")
    edges = Counter(tuple(sorted((int(a), int(b)))) for face in triangles for a, b in zip(face, np.roll(face, -1)))
    if any(count != 2 for count in edges.values()):
        raise ValueError("Enclosure reference must be a closed two-manifold mesh")
    directed = Counter((int(a), int(b)) for face in triangles for a, b in zip(face, np.roll(face, -1)))
    if any(count != directed[(b, a)] for (a, b), count in directed.items()):
        raise ValueError("Enclosure reference must have consistent face winding")
    geometry = points[triangles]
    if np.any(
        np.linalg.norm(np.cross(geometry[:, 1] - geometry[:, 0], geometry[:, 2] - geometry[:, 0]), axis=1) < 1e-14
    ):
        raise ValueError("Enclosure reference has a degenerate triangle")
    return points, triangles


def _inside(point: np.ndarray, triangles: np.ndarray) -> tuple[bool, float]:
    """Solid-angle winding of a consistently oriented closed surface."""
    relative = triangles - point
    a, b, c = relative[:, 0], relative[:, 1], relative[:, 2]
    lengths = np.linalg.norm(relative, axis=2)
    numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
    denominator = (
        lengths.prod(axis=1)
        + np.einsum("ij,ij->i", a, b) * lengths[:, 2]
        + np.einsum("ij,ij->i", b, c) * lengths[:, 0]
        + np.einsum("ij,ij->i", c, a) * lengths[:, 1]
    )
    winding = float(np.arctan2(numerator, denominator).sum() / (2 * math.pi))
    return bool(abs(abs(winding) - 1.0) < 1e-6), winding


def _triangle_box_intersections(triangles: np.ndarray, halfsize: np.ndarray) -> list[int]:
    """Complete triangle/AABB separating-axis test in the box's local frame."""
    edges = np.roll(triangles, -1, axis=1) - triangles
    normal = np.cross(edges[:, 0], edges[:, 1])
    basis = np.eye(3)
    axes = np.concatenate(
        (
            np.broadcast_to(basis, (len(triangles), 3, 3)),
            normal[:, None],
            np.cross(edges[:, :, None], basis[None, None]).reshape(-1, 9, 3),
        ),
        axis=1,
    )
    norms = np.linalg.norm(axes, axis=2)
    axes = axes / np.maximum(norms[:, :, None], 1e-15)
    projections = np.einsum("tvi,tai->tav", triangles, axes)
    radii = np.abs(axes) @ halfsize
    separated = (projections.min(axis=2) > radii + 1e-10) | (projections.max(axis=2) < -radii - 1e-10)
    separated &= norms > 1e-12
    return [int(index) for index in np.flatnonzero(~np.any(separated, axis=1))]


def apply_cavities(
    root: ET.Element, hardware: dict[str, Any], torso_metadata: dict[str, Any], appearance_metadata: dict[str, Any]
) -> dict[str, Any]:
    """Declare references from exterior geometry, independent of part locations.

    No contact geometry, mass, fabricated cavity boxes or build-time pass verdicts
    are added. The validator rereads compiled installation positions/rotations.
    """
    torso_vertices, torso_faces = _torso_loft(torso_metadata["profile_m"])
    # Fail if the torso generator changes its sampling or triangle diagonal
    # without updating this complete reference. Open patches may be omitted,
    # but every remaining visible triangle must belong to the reference.
    skin = root.find("asset/mesh[@name='juv_torso_skin']")
    if skin is None:
        raise ValueError("Missing torso skin for enclosure-reference reconciliation")
    skin_vertices = np.fromstring(skin.attrib["vertex"], sep=" ").reshape(-1, 3)
    skin_faces = np.fromstring(skin.attrib["face"], sep=" ", dtype=int).reshape(-1, 3)

    def ordered(triangle: np.ndarray) -> np.ndarray:
        return np.array(sorted(tuple(point) for point in triangle))

    reference_triangles = np.array([ordered(triangle) for triangle in torso_vertices[torso_faces]])
    # MJCF uses ten significant digits. Compare geometric coordinates directly;
    # decimal-bin rounding can disagree at a boundary despite sub-nanometre error.
    if any(
        not np.any(np.max(np.abs(reference_triangles - ordered(triangle)), axis=(1, 2)) <= 1e-9)
        for triangle in skin_vertices[skin_faces]
    ):
        raise ValueError("Torso enclosure reference no longer matches actual outer skin triangles")
    head = appearance_metadata.get("head_cavity", {})
    references = {
        "torso_outer_loft": dict(
            owner_body="torso",
            vertices_m=torso_vertices.tolist(),
            triangles=torso_faces.tolist(),
            basis="Exact 20-column torso outer profile and triangle diagonals, with intentional apertures capped for enclosure only.",
        ),
    }
    if head.get("outer_loft_vertices_m") is not None and head.get("outer_loft_faces") is not None:
        references["skull_outer_loft"] = dict(
            owner_body="head_pitch_body",
            vertices_m=head["outer_loft_vertices_m"],
            triangles=head["outer_loft_faces"],
            basis="Actual closed polygonal skull loft before underside and lens-aperture openings.",
        )
    for reference in references.values():
        _checked_mesh(reference["vertices_m"], reference["triangles"])
        if root.find(f".//body[@name='{reference['owner_body']}']") is None:
            raise ValueError(f"Missing enclosure owner {reference['owner_body']}")
        reference["geometry_sha256"] = hashlib.sha256(
            json.dumps([reference["vertices_m"], reference["triangles"]], separators=(",", ":")).encode()
        ).hexdigest()
    owner_references = {row["owner_body"]: name for name, row in references.items()}
    checks = [
        dict(hardware=item["name"], envelope_geom=item["envelope_geom"], reference=owner_references[item["owner_body"]])
        for item in hardware["hardware"]
        if item["owner_body"] in owner_references
    ]
    return dict(
        schema_version=1,
        references=references,
        checks=checks,
        scope="Whole installation boxes inside the closed exterior reference; physical wall gaps are independently checked.",
        limitations=[
            "Reference caps establish exterior containment, not a fabricated closure over intentional joint and service openings.",
            "Installed boxes must separately clear actual wall material and mechanisms by the declared clearance margin.",
        ],
    )


def exterior_containment_screen(model: mujoco.MjModel, manifest: dict[str, Any]) -> dict[str, Any]:
    """Prove each connected box stays inside its closed reference surface.

    A box whose center is inside and whose volume meets no boundary triangle is
    entirely inside, including every corner, edge and face. This is not a finite
    point sample. Convexifying the whole reference is deliberately avoided.
    """
    specification = manifest.get("exterior_containment", {})
    references = specification.get("references", {})
    specs = specification.get("checks", [])
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    meshes = {name: _checked_mesh(row["vertices_m"], row["triangles"]) for name, row in references.items()}
    results = []
    for spec in specs:
        reference = references[spec["reference"]]
        vertices, faces = meshes[spec["reference"]]
        body = model.body(reference["owner_body"]).id
        geom = model.geom(spec["envelope_geom"]).id
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError("Exterior enclosure requires actual installation box geometry")
        if model.body_weldid[body] != model.body_weldid[model.geom_bodyid[geom]]:
            raise ValueError("Exterior enclosure requires rigid co-mounting; moving parts require a pose sweep")
        reference_rotation = data.xmat[body].reshape(3, 3)
        center = reference_rotation.T @ (data.geom_xpos[geom] - data.xpos[body])
        rotation = reference_rotation.T @ data.geom_xmat[geom].reshape(3, 3)
        inside, winding = _inside(center, vertices[faces])
        triangles_in_box = ((vertices - center) @ rotation)[faces]
        intersections = _triangle_box_intersections(triangles_in_box, model.geom_size[geom])
        results.append(
            dict(
                **spec,
                owner_body=reference["owner_body"],
                center_inside=inside,
                winding_number=winding,
                intersecting_reference_triangles=len(intersections),
                witness_triangles=intersections[:8],
                passed=inside and not intersections,
            )
        )
    expected = {row["envelope_geom"] for row in manifest.get("hardware", [])}
    inspected = {row["envelope_geom"] for row in results}
    missing = sorted(expected - inspected)
    return dict(
        scope="Whole installation box enclosure by closed loft winding and complete triangle/OBB SAT; independent physical-wall clearance required.",
        checks=results,
        missing_envelopes=missing,
        complete=bool(results) and not missing,
        passed=bool(results) and not missing and all(row["passed"] for row in results),
    )
