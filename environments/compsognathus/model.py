"""Small model-only API, shared by the viewer and prototype validation.

Controls are absolute joint-angle targets in actuator order. No learned policy,
task reward, ground-truth observation contract or Gym registration is provided.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from . import MODEL_PATHS

PARAMETERS = json.loads((Path(__file__).parent / "data/model_parameters.json").read_text())


def robot_body_ids(model: mujoco.MjModel) -> list[int]:
    """Descendants of pelvis only: never count the world or mocap target."""
    root = model.body("pelvis").id
    included = {root}
    for body in range(root + 1, model.nbody):
        if int(model.body_parentid[body]) in included:
            included.add(body)
    return sorted(included)


def load_model(variant: str, mass_scale: float = 1.0):
    if variant not in MODEL_PATHS:
        raise ValueError(f"Unknown model {variant!r}; choose {tuple(MODEL_PATHS)}")
    if not np.isfinite(mass_scale) or mass_scale <= 0:
        raise ValueError("mass_scale must be finite and positive")
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATHS[variant]))
    data = mujoco.MjData(model)
    if mass_scale != 1:
        ids = robot_body_ids(model)
        model.body_mass[ids] *= mass_scale
        model.body_inertia[ids] *= mass_scale
        mujoco.mj_setConst(model, data)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    return model, data


def set_motors_enabled(model: mujoco.MjModel, enabled: bool):
    """Toggle solver-level actuation, preserving gains and other disable flags."""
    flag = int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    if enabled:
        model.opt.disableflags &= ~flag
    else:
        model.opt.disableflags |= flag


def model_bounds(model: mujoco.MjModel, data: mujoco.MjData):
    """World-space visual AABB of the animal/robot, excluding floor and target."""
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    bodies = set(robot_body_ids(model))
    for geom in range(model.ngeom):
        if int(model.geom_bodyid[geom]) not in bodies:
            continue
        pos = data.geom_xpos[geom]
        rot = data.geom_xmat[geom].reshape(3, 3)
        size = model.geom_size[geom]
        kind = model.geom_type[geom]
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[geom]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            points = model.mesh_vert[start : start + count] @ rot.T + pos
            geom_low, geom_high = points.min(axis=0), points.max(axis=0)
        else:
            if kind == mujoco.mjtGeom.mjGEOM_BOX:
                extent = np.abs(rot) @ size
            elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                extent = np.sqrt((rot**2) @ (size**2))
            elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
                extent = np.abs(rot[:, 2]) * size[1] + size[0]
            elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
                extent = np.abs(rot[:, 2]) * size[1] + np.sqrt(np.maximum(0, 1 - rot[:, 2] ** 2)) * size[0]
            elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                extent = np.full(3, size[0])
            else:
                raise ValueError(f"Unhandled bound type {kind} for {model.geom(geom).name}")
            geom_low, geom_high = pos - extent, pos + extent
        low, high = np.minimum(low, geom_low), np.maximum(high, geom_high)
    return low, high


def floor_contacts(model: mujoco.MjModel, data: mujoco.MjData):
    """Names, positions and normal forces for active floor constraints."""
    floor = model.geom("floor").id
    contacts = []
    for i in range(data.ncon):
        contact = data.contact[i]
        if floor not in (contact.geom1, contact.geom2) or contact.efc_address < 0:
            continue
        other = contact.geom2 if contact.geom1 == floor else contact.geom1
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        contacts.append(
            {
                "geom": model.geom(other).name,
                "position": contact.pos.copy(),
                "normal_force": float(force[0]),
                "distance": float(contact.dist),
            }
        )
    return contacts
