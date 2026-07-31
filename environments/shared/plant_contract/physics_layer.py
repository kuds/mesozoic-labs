"""Physics layer: compiled dynamics, contacts, actuators, and reset state.

Fails closed on MJCF features the payload does not yet classify, so a new
physics feature cannot slip past the contract unrecorded."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from .digests import _array_digest
from .errors import PlantContractError
from .identity import PlantVersion
from .introspection import _fields


def _option_payload(model: mujoco.MjModel) -> dict[str, Any]:
    # MuJoCo adds option fields over time.  The contract pins an exact MuJoCo
    # version, so fingerprint every public data attribute instead of relying
    # on an allow-list that can silently miss contact overrides or solver
    # controls introduced by a newer release.
    return {
        name: value
        for name in dir(model.opt)
        if not name.startswith("_") and not callable(value := getattr(model.opt, name))
    }


def _reject_unclassified_features(model: mujoco.MjModel) -> None:
    unsupported = []
    for count_name in ("nflex", "nplugin"):
        if int(getattr(model, count_name, 0)):
            unsupported.append(f"{count_name}={getattr(model, count_name)}")
    for field in ("body_plugin", "geom_plugin", "actuator_plugin", "sensor_plugin"):
        values = np.asarray(getattr(model, field, []))
        if values.size and np.any(values >= 0):
            unsupported.append(field)
    if unsupported:
        raise PlantContractError(
            "plant fingerprint does not yet classify these MuJoCo features: " + ", ".join(unsupported)
        )


def _physics_geom_ids(model: mujoco.MjModel) -> np.ndarray:
    """Return geoms that can affect contacts, fluids, or tendon dynamics."""
    geom_ids = set(np.flatnonzero((model.geom_contype != 0) | (model.geom_conaffinity != 0)).tolist())
    if model.npair:
        geom_ids.update(int(value) for value in model.pair_geom1)
        geom_ids.update(int(value) for value in model.pair_geom2)
    fluid_rows = np.flatnonzero(np.any(np.asarray(model.geom_fluid) != 0.0, axis=1))
    geom_ids.update(int(value) for value in fluid_rows)
    if float(model.opt.density) != 0.0 or float(model.opt.viscosity) != 0.0 or model.ntendon:
        # The default fluid model and tendon wrapping can depend on otherwise
        # collision-disabled geometry, so classify all geoms conservatively.
        geom_ids.update(range(model.ngeom))
    return np.asarray(sorted(geom_ids), dtype=np.int64)


def _physics_payload(model: mujoco.MjModel, version: PlantVersion) -> dict[str, Any]:
    _reject_unclassified_features(model)
    dynamics_geom_ids = _physics_geom_ids(model)
    payload: dict[str, Any] = {
        "options": _option_payload(model),
        "dimensions": {
            name: int(getattr(model, name))
            for name in (
                "nq",
                "nv",
                "nu",
                "na",
                "nbody",
                "njnt",
                "ngeom",
                "nsite",
                "nsensor",
                "ntendon",
                "neq",
                "npair",
                "nexclude",
                "nkey",
                "nmocap",
            )
        },
        "bodies": _fields(
            model,
            (
                "body_parentid",
                "body_mocapid",
                "body_pos",
                "body_quat",
                "body_ipos",
                "body_iquat",
                "body_mass",
                "body_inertia",
                "body_gravcomp",
            ),
        ),
        "joints": _fields(
            model,
            (
                "jnt_type",
                "jnt_bodyid",
                "jnt_pos",
                "jnt_axis",
                "jnt_limited",
                "jnt_range",
                "jnt_stiffness",
                "jnt_stiffnesspoly",
                "jnt_actfrclimited",
                "jnt_actfrcrange",
                "jnt_actgravcomp",
                "jnt_margin",
                "jnt_solref",
                "jnt_solimp",
            ),
        ),
        "dofs": _fields(
            model,
            (
                "dof_jntid",
                "dof_armature",
                "dof_damping",
                "dof_dampingpoly",
                "dof_frictionloss",
                "dof_solref",
                "dof_solimp",
            ),
        ),
        "dynamics_geoms": {
            "ids": dynamics_geom_ids,
            **_fields(
                model,
                (
                    "geom_bodyid",
                    "geom_type",
                    "geom_dataid",
                    "geom_pos",
                    "geom_quat",
                    "geom_size",
                    "geom_contype",
                    "geom_conaffinity",
                    "geom_condim",
                    "geom_friction",
                    "geom_priority",
                    "geom_solmix",
                    "geom_solref",
                    "geom_solimp",
                    "geom_margin",
                    "geom_gap",
                    "geom_fluid",
                ),
                dynamics_geom_ids,
            ),
        },
        "actuators": _fields(
            model,
            (
                "actuator_trntype",
                "actuator_trnid",
                "actuator_dyntype",
                "actuator_gaintype",
                "actuator_biastype",
                "actuator_actearly",
                "actuator_dynprm",
                "actuator_gainprm",
                "actuator_biasprm",
                "actuator_gear",
                "actuator_cranklength",
                "actuator_length0",
                "actuator_armature",
                "actuator_damping",
                "actuator_dampingpoly",
                "actuator_delay",
                "actuator_ctrllimited",
                "actuator_ctrlrange",
                "actuator_forcelimited",
                "actuator_forcerange",
                "actuator_actlimited",
                "actuator_actrange",
                "actuator_lengthrange",
            ),
        ),
        "initialization": {
            "qpos0": model.qpos0,
            "qpos_spring": model.qpos_spring,
            **_fields(model, ("key_time", "key_qpos", "key_qvel", "key_act", "key_ctrl", "key_mpos", "key_mquat")),
        },
    }
    if model.npair:
        payload["contact_pairs"] = _fields(
            model,
            (
                "pair_geom1",
                "pair_geom2",
                "pair_dim",
                "pair_friction",
                "pair_solref",
                "pair_solreffriction",
                "pair_solimp",
                "pair_margin",
                "pair_gap",
            ),
        )
    if model.nexclude:
        payload["contact_exclusions"] = _fields(model, ("exclude_signature",))
    if model.neq:
        payload["equalities"] = _fields(
            model,
            ("eq_type", "eq_objtype", "eq_obj1id", "eq_obj2id", "eq_active0", "eq_solref", "eq_solimp", "eq_data"),
        )
    if model.ntendon:
        payload["tendons"] = _fields(
            model,
            (
                "tendon_adr",
                "tendon_num",
                "tendon_limited",
                "tendon_range",
                "tendon_stiffness",
                "tendon_stiffnesspoly",
                "tendon_damping",
                "tendon_dampingpoly",
                "tendon_frictionloss",
                "tendon_armature",
                "tendon_actfrclimited",
                "tendon_actfrcrange",
                "tendon_margin",
                "tendon_solref_lim",
                "tendon_solimp_lim",
                "tendon_solref_fri",
                "tendon_solimp_fri",
                "tendon_lengthspring",
                "wrap_type",
                "wrap_objid",
                "wrap_prm",
            ),
        )
    if model.nhfield:
        payload["heightfields"] = {
            **_fields(
                model,
                ("hfield_adr", "hfield_nrow", "hfield_ncol", "hfield_size"),
            ),
            "data": _array_digest("mesozoic.physics-heightfield-data/v1", model.hfield_data),
        }

    collision_mesh_ids = sorted(
        {
            int(model.geom_dataid[index])
            for index in dynamics_geom_ids
            if int(model.geom_type[index]) == int(mujoco.mjtGeom.mjGEOM_MESH)
        }
    )
    if collision_mesh_ids:
        mesh_payload = []
        for mesh_id in collision_mesh_ids:
            vert_adr = int(model.mesh_vertadr[mesh_id])
            vert_num = int(model.mesh_vertnum[mesh_id])
            face_adr = int(model.mesh_faceadr[mesh_id])
            face_num = int(model.mesh_facenum[mesh_id])
            mesh_payload.append(
                {
                    "mesh_id": mesh_id,
                    "vertices": _array_digest(
                        "mesozoic.collision-mesh-vertices/v1", model.mesh_vert[vert_adr : vert_adr + vert_num]
                    ),
                    "faces": _array_digest(
                        "mesozoic.collision-mesh-faces/v1", model.mesh_face[face_adr : face_adr + face_num]
                    ),
                    "scale": model.mesh_scale[mesh_id],
                    "pos": model.mesh_pos[mesh_id],
                    "quat": model.mesh_quat[mesh_id],
                }
            )
        payload["collision_meshes"] = mesh_payload
    return payload
