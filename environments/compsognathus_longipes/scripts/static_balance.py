"""Inspect loaded foot support and fixed-target static force balance on a horizontal floor.

Run from the repository root with
`python -m environments.compsognathus_longipes.scripts.static_balance`.
Both samples use zero velocity and the authored home controls. The second pose
is obtained by letting the free model settle before re-evaluating it at rest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

from environments.compsognathus_longipes import MODEL_PATHS
from environments.compsognathus_longipes.model import floor_contacts, load_model, pelvis_subtree_body_ids

# Numerical reporting tolerances, in N for translations and Nm for rotations.
FORCE_ATOL = 1e-4
TORQUE_ATOL = 1e-5
LOAD_THRESHOLD_N = 1e-6
DIRECTIONS = {"forward": (1, 0), "rearward": (-1, 0), "left": (0, 1), "right": (0, -1)}


def support_polygon(points, com_xy):
    """Euclidean signed distance to the loaded hull, positive inside.

    Directional distances are available travel along world-axis rays from a
    supported projection. A point or line has zero interior area.
    """
    points = np.unique(np.asarray(points, dtype=float).reshape(-1, 2), axis=0)
    com_xy = np.asarray(com_xy, dtype=float)
    vertices = points
    directional = {name: None for name in DIRECTIONS}
    margin = None
    inside = False
    if len(points) >= 3 and np.linalg.matrix_rank(points - points[0]) == 2:
        hull = ConvexHull(points)
        vertices = points[hull.vertices]
        normals = hull.equations[:, :2]
        slack = -(normals @ com_xy + hull.equations[:, 2])
        inside = np.min(slack) >= -1e-12
        if inside:
            margin = float(max(0, np.min(slack)))
            for name, direction in DIRECTIONS.items():
                projections = normals @ direction
                facing = projections > 1e-12
                directional[name] = float(max(0, np.min(slack[facing] / projections[facing])))
    if len(vertices) and not inside:
        edges = np.roll(vertices, -1, axis=0) - vertices
        lengths_squared = np.sum(edges**2, axis=1)
        t = np.divide(
            np.sum((com_xy - vertices) * edges, axis=1),
            lengths_squared,
            out=np.zeros(len(vertices)),
            where=lengths_squared > 0,
        )
        closest = vertices + np.clip(t, 0, 1)[:, None] * edges
        margin = -float(np.min(np.linalg.norm(closest - com_xy, axis=1)))
    return {"hull_xy_m": vertices.tolist(), "signed_margin_m": margin, "directional_clearance_m": directional}


def foot_support(model, data, embodiment, com_xy):
    """Assign actual loaded contacts to the corresponding sole's body subtree."""
    labels = ("r", "l") if embodiment == "anatomical" else ("right", "left")
    foot_bodies = {
        int(model.site(f"{label}_sole").bodyid[0]): side for label, side in zip(labels, ("right", "left"), strict=True)
    }
    points = {"right": [], "left": []}
    contacts = []
    other_floor_contacts = []
    for contact in floor_contacts(model, data):
        if contact["normal_force"] <= LOAD_THRESHOLD_N:
            continue
        body = int(model.geom(contact["geom"]).bodyid[0])
        while body and body not in foot_bodies:
            body = model.body_parentid[body]
        if not body:
            other_floor_contacts.append(contact["geom"])
            continue
        side = foot_bodies[body]
        points[side].append(contact["position"][:2])
        contacts.append(
            dict(side=side, geom=contact["geom"], xyz_m=contact["position"].tolist(), normal_N=contact["normal_force"])
        )
    return {
        "contacts": contacts,
        "normal_load_N": {
            side: sum(entry["normal_N"] for entry in contacts if entry["side"] == side) for side in ("left", "right")
        },
        "other_loaded_floor_geoms": sorted(set(other_floor_contacts)),
        "both_feet": support_polygon(points["left"] + points["right"], com_xy),
        "left_foot_alone_at_same_com": support_polygon(points["left"], com_xy),
        "right_foot_alone_at_same_com": support_polygon(points["right"], com_xy),
    }


def constraint_contribution(model, data, rows):
    generalized = np.zeros(model.nv)
    if np.any(rows):
        mujoco.mj_mulJacTVec(model, data, generalized, np.where(rows, data.efc_force, 0.0))
    return generalized


def dof_labels(model):
    names, units = [], []
    for i in range(model.njnt):
        joint = model.joint(i)
        kind = model.jnt_type[i]
        if kind == mujoco.mjtJoint.mjJNT_FREE:
            names.extend(f"{joint.name}_{direction}" for direction in ("x", "y", "z", "rx", "ry", "rz"))
            units.extend(["N"] * 3 + ["Nm"] * 3)
        elif kind == mujoco.mjtJoint.mjJNT_BALL:
            names.extend(f"{joint.name}_{direction}" for direction in ("rx", "ry", "rz"))
            units.extend(["Nm"] * 3)
        else:
            names.append(joint.name)
            units.append("N" if kind == mujoco.mjtJoint.mjJNT_SLIDE else "Nm")
    return names, np.array(units)


def at_rest(model, data, embodiment):
    """Gravity and all native reactions in each generalized coordinate."""
    if np.any(data.qvel):
        raise ValueError("Static audit requires velocities of zero")
    mujoco.mj_forward(model, data)
    root = model.body("pelvis").id
    com = data.subtree_com[root]
    active = data.qfrc_actuator
    equality = constraint_contribution(model, data, data.efc_type == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))
    contact = constraint_contribution(
        model, data, data.efc_type >= int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS)
    )
    other = data.qfrc_constraint - equality - contact
    residual = active + data.qfrc_passive + data.qfrc_constraint - data.qfrc_bias
    names, units = dof_labels(model)
    forces = np.abs(residual[units == "N"])
    torques = np.abs(residual[units == "Nm"])
    max_force = float(forces.max(initial=0.0))
    max_torque = float(torques.max(initial=0.0))
    channels = {}
    for i in range(model.nu):
        start = data.moment_rowadr[i]
        span = slice(start, start + data.moment_rownnz[i])
        lo, hi = model.actuator_forcerange[i]
        force = float(data.actuator_force[i])
        limit = hi if force >= 0 else -lo
        channels[model.actuator(i).name] = {
            "transmission_force": force,
            "force_limits": [float(lo), float(hi)] if model.actuator_forcelimited[i] else None,
            "fraction_of_directional_cap": float(abs(force) / limit)
            if model.actuator_forcelimited[i] and limit
            else None,
            "generalized_contribution": {
                names[dof]: {
                    "unit": str(units[dof]),
                    "applied": float(moment * force),
                    "limits": sorted((float(moment * lo), float(moment * hi)))
                    if model.actuator_forcelimited[i]
                    else None,
                }
                for dof, moment in zip(data.moment_colind[span], data.actuator_moment[span], strict=True)
            },
        }
    support = foot_support(model, data, embodiment, com[:2])
    supported = all(support[f"{side}_foot_alone_at_same_com"]["hull_xy_m"] for side in ("left", "right"))
    margin = support["both_feet"]["signed_margin_m"]
    return {
        "mass_kg": float(model.body_subtreemass[root]),
        "weight_N": float(-model.opt.gravity[2] * model.body_subtreemass[root]),
        "com_m": com.tolist(),
        "support": support,
        "actuators": channels,
        "dofs": {
            name: {
                "unit": str(units[i]),
                "gravity": float(-data.qfrc_bias[i]),
                "actuation": float(active[i]),
                "passive": float(data.qfrc_passive[i]),
                "contacts": float(contact[i]),
                "equality": float(equality[i]),
                "other_constraints": float(other[i]),
                "residual": float(residual[i]),
            }
            for i, name in enumerate(names)
        },
        "max_force_residual_N": max_force,
        "max_torque_residual_Nm": max_torque,
        "in_static_double_support_at_fixed_targets": bool(
            supported
            and margin is not None
            and margin > 0
            and not support["other_loaded_floor_geoms"]
            and max_force <= FORCE_ATOL
            and max_torque <= TORQUE_ATOL
        ),
    }


def fixed_pose_tail_mass_sensitivity(model, data, hull_xy):
    """Transfer mass between the four tail links and the trunk at their fixed inertial centres."""
    trunk = model.body("pelvis").id
    tail = [model.body(f"tail_{i}").id for i in range(1, 5)]
    bodies = pelvis_subtree_body_ids(model)
    tail_mass = float(sum(model.body_mass[tail]))
    cases = []
    for fraction in (-0.25, 0.0, 0.25):
        masses = model.body_mass.copy()
        masses[tail] *= 1 + fraction
        masses[trunk] -= fraction * tail_mass
        total_mass = float(masses[bodies].sum())
        com = np.average(data.xipos[bodies], weights=masses[bodies], axis=0)
        support = support_polygon(hull_xy, com[:2])
        cases.append(
            {
                "tail_mass_change_fraction": fraction,
                "tail_mass_kg": float(masses[tail].sum()),
                "trunk_own_mass_kg": float(masses[trunk]),
                "total_mass_kg": total_mass,
                "com_m": com.tolist(),
                "signed_margin_m": support["signed_margin_m"],
            }
        )
    return {
        "method": (
            "At the settled posture, vary each tail link's mass by the same fraction and transfer the "
            "opposite total to the pelvis body's own mass. All body inertial centres and original loaded "
            "foot contacts stay fixed. Each case measures the distance from its projected COM to that same support polygon."
        ),
        "cases": cases,
    }


def analyze(embodiment, settle_s=10.0):
    if not np.isfinite(settle_s) or settle_s <= 0:
        raise ValueError("settle_s must be finite and positive")
    model, data = load_model(embodiment)
    home_qpos = data.qpos.copy()
    home = at_rest(model, data, embodiment)
    mujoco.mj_step(model, data, nstep=round(settle_s / model.opt.timestep))
    names, units = dof_labels(model)
    displacement = np.zeros(model.nv)
    mujoco.mj_differentiatePos(model, displacement, 1.0, home_qpos, data.qpos)
    motion = {
        "elapsed_s": float(data.time),
        "max_translation_speed_m_s": float(np.abs(data.qvel[units == "N"]).max(initial=0.0)),
        "max_angular_speed_rad_s": float(np.abs(data.qvel[units == "Nm"]).max(initial=0.0)),
        "change_from_home_m_or_rad": dict(zip(names, displacement.tolist(), strict=True)),
    }
    data.qvel[:] = 0.0
    settled = at_rest(model, data, embodiment)
    result = {
        "mjcf_sha256": hashlib.sha256(MODEL_PATHS[embodiment].read_bytes()).hexdigest(),
        "authored_home": home,
        "fixed_target_settling": motion,
        "settled_pose_at_rest": settled,
    }
    if embodiment == "anatomical":
        hull = settled["support"]["both_feet"]["hull_xy_m"]
        result["fixed_pose_tail_mass_sensitivity"] = fixed_pose_tail_mass_sensitivity(model, data, hull)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embodiment", choices=tuple(MODEL_PATHS), action="append")
    parser.add_argument("--settle-s", type=float, default=10.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        "mujoco_version": mujoco.__version__,
        "force_tolerance_N": FORCE_ATOL,
        "torque_tolerance_Nm": TORQUE_ATOL,
        "loaded_contact_threshold_N": LOAD_THRESHOLD_N,
        "models": {name: analyze(name, args.settle_s) for name in (args.embodiment or MODEL_PATHS)},
    }
    result = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(result)
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
