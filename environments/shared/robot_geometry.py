"""Compiled MuJoCo geometry measurements for standalone robot concepts.

These helpers do not choose a plant, modify its geometry or set acceptance gates.
Visual groups 1 and 2 follow the repository's concept-model convention.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np


def visual_bounds(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    """World bounds of actual visual surfaces, excluding floor and packaging boxes."""
    lower, upper = [], []
    for g in range(model.ngeom):
        if model.geom_group[g] not in (1, 2):
            continue
        rotation = data.geom_xmat[g].reshape(3, 3)
        pos, size, kind = data.geom_xpos[g], model.geom_size[g], model.geom_type[g]
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[g]
            a, n = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            points = model.mesh_vert[a : a + n] @ rotation.T + pos
            lower.append(points.min(axis=0))
            upper.append(points.max(axis=0))
            continue
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            extent = np.abs(rotation) @ size
        elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            extent = np.sqrt(np.sum((rotation * size) ** 2, axis=1))
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            extent = size[0] * np.linalg.norm(rotation[:, :2], axis=1) + size[1] * np.abs(rotation[:, 2])
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            extent = size[0] + size[1] * np.abs(rotation[:, 2])
        else:
            extent = np.full(3, size[0])
        lower.append(pos - extent)
        upper.append(pos + extent)
    return np.min(lower, axis=0), np.max(upper, axis=0)


def internal_contacts(model: mujoco.MjModel, data: mujoco.MjData, *, floor_geom: str = "floor") -> list[dict[str, Any]]:
    """Return contacts not involving the declared floor; no acceptance threshold."""
    floor = model.geom(floor_geom).id
    return [
        dict(pair=[model.geom(c.geom1).name, model.geom(c.geom2).name], penetration_m=max(0.0, -float(c.dist)))
        for c in data.contact[: data.ncon]
        if floor not in (c.geom1, c.geom2)
    ]
