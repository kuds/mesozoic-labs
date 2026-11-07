"""Conservative clearance evidence for compiled convex MuJoCo geometry.

MuJoCo 3.10 can return a zero distance for separated convex geoms when the
distance search radius is large. A positive support-plane gap independently
certifies separation. It is a lower bound, never an exact closest distance or
a penetration estimate. No collision filters or solver options are changed.
"""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np


def _intervals(model: mujoco.MjModel, data: mujoco.MjData, geom: int, axes: np.ndarray) -> np.ndarray:
    """Exact support intervals along unit world axes for supported convex geoms."""
    local = axes @ data.geom_xmat[geom].reshape(3, 3)
    center = axes @ data.geom_xpos[geom]
    size, kind = model.geom_size[geom], model.geom_type[geom]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        projected = model.mesh_vert[start : start + count] @ local.T
        return np.column_stack((center + projected.min(axis=0), center + projected.max(axis=0)))
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        radius = np.full(len(axes), size[0])
    elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        radius = size[0] + size[1] * np.abs(local[:, 2])
    elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        radius = np.linalg.norm(local * size, axis=1)
    elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        radius = size[0] * np.linalg.norm(local[:, :2], axis=1) + size[1] * np.abs(local[:, 2])
    elif kind == mujoco.mjtGeom.mjGEOM_BOX:
        radius = np.abs(local) @ size
    else:
        raise ValueError("Support-plane clearance requires a convex primitive or mesh.")
    return np.column_stack((center - radius, center + radius))


def _certificate(
    model: mujoco.MjModel, data: mujoco.MjData, first: int, second: int, axes: np.ndarray
) -> dict[str, Any] | None:
    norms = np.linalg.norm(axes, axis=1)
    axes = axes[norms > 1e-12] / norms[norms > 1e-12, None]
    if not len(axes):
        return None
    left, right = _intervals(model, data, first, axes), _intervals(model, data, second, axes)
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Nonfinite geometry support interval.")
    gaps = np.maximum(right[:, 0] - left[:, 1], left[:, 0] - right[:, 1])
    index = int(np.argmax(gaps))
    # Subtract a floating-point allowance rather than promoting roundoff into
    # certified clearance at a touching face or exactly at a planning limit.
    padding = 64 * np.finfo(float).eps * max(1.0, float(np.max(np.abs([left, right]))))
    bound = float(gaps[index] - padding)
    if bound <= 0:
        return None
    return {
        "distance_lower_bound_m": bound,
        "axis_world": axes[index].tolist(),
        "first_interval_m": left[index].tolist(),
        "second_interval_m": right[index].tolist(),
        "roundoff_allowance_m": float(padding),
    }


def _mesh_face_axes(model: mujoco.MjModel, data: mujoco.MjData, geom: int) -> np.ndarray:
    """Use actual triangle directions; each resulting gap still checks all vertices."""
    mesh = model.geom_dataid[geom]
    start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    face_start, face_count = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
    vertices = model.mesh_vert[start : start + count].astype(float)
    triangles = vertices[model.mesh_face[face_start : face_start + face_count]]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    normals = normals[lengths > 1e-12] / lengths[lengths > 1e-12, None]
    normals = normals @ data.geom_xmat[geom].reshape(3, 3).T
    # Parallel/opposite normals describe the same projection interval. Rounded
    # directions remain valid test axes: no triangle plane is itself trusted
    # as a hull boundary, including for nonconvex meshes or reversed winding.
    signs = np.sign(normals[np.arange(len(normals)), np.argmax(np.abs(normals), axis=1)])
    return np.unique(np.round(normals * signs[:, None], 12), axis=0)


def separation_lower_bound(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    first: int,
    second: int,
    *,
    stop_at_m: float | None = None,
) -> dict[str, Any] | None:
    """Return a positive separation certificate, or None when no axis proves it.

    Candidate axes need not include the closest-point direction: every positive
    projection gap is a valid Euclidean lower bound. Overlapping projections
    never prove overlap. If supplied, ``stop_at_m`` allows early return once a
    certificate meets that margin; the bound need not be the strongest one.
    """
    first_axes = data.geom_xmat[first].reshape(3, 3).T
    second_axes = data.geom_xmat[second].reshape(3, 3).T
    directions = [np.eye(3), first_axes, second_axes, (data.geom_xpos[second] - data.geom_xpos[first])[None, :]]
    if model.geom_type[first] == model.geom_type[second] == mujoco.mjtGeom.mjGEOM_BOX:
        directions.append(np.cross(first_axes[:, None, :], second_axes[None, :, :]).reshape(-1, 3))
    best = _certificate(model, data, first, second, np.concatenate(directions))
    if best is not None and stop_at_m is not None and best["distance_lower_bound_m"] >= stop_at_m:
        return best
    # Oblique mesh faces can separate shapes while all body-frame projections
    # overlap. Bound temporary vertex-by-direction matrices to 64 directions and
    # stop as soon as a sufficient certificate is found during threshold checks.
    for geom in (first, second):
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        axes = _mesh_face_axes(model, data, geom)
        for start in range(0, len(axes), 64):
            proof = _certificate(model, data, first, second, axes[start : start + 64])
            if proof is not None and (best is None or proof["distance_lower_bound_m"] > best["distance_lower_bound_m"]):
                best = proof
                if stop_at_m is not None and best["distance_lower_bound_m"] >= stop_at_m:
                    return best
    return best


def clearance_distance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    first: int,
    second: int,
    *,
    required_clearance_m: float,
    search_distance_m: float,
) -> dict[str, Any]:
    """Keep native distance evidence unless a positive plane gap contradicts it."""
    native = float(mujoco.mj_geomDistance(model, data, first, second, search_distance_m, None))
    if not math.isfinite(native):
        raise ValueError(f"Nonfinite geometry distance for {model.geom(first).name} / {model.geom(second).name}.")
    result = {
        "min_distance_m": native,
        "native_distance_m": native,
        "distance_is_lower_bound": native >= search_distance_m,
        "distance_method": "mujoco",
    }
    # Most pairs have ample clearance. Check support planes only where a
    # native result would fail the declared criterion, including negative ones.
    if native < required_clearance_m:
        proof = separation_lower_bound(model, data, first, second, stop_at_m=required_clearance_m)
        if proof is not None and proof["distance_lower_bound_m"] > max(native, 0.0):
            result.update(
                min_distance_m=proof["distance_lower_bound_m"],
                distance_is_lower_bound=True,
                distance_method="support_plane_lower_bound",
                separation_plane=proof,
            )
    return result
