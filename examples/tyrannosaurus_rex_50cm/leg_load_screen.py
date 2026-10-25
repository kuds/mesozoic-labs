"""Compare static leg support loads with an idealized motor-mass relocation.

This is a symmetric two-foot gravity balance, not a transmission design or a
fall-recovery trial. The remote-drive alternative adds no mechanism mass or
losses; its motors deliberately share abstract torso mass locations.
"""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

from environments.shared.robot_validation import require_unsupported

from .recovery_checks import LEG_JOINTS, prepare_crouch

MOVED_MOTORS = tuple(
    f"{side}_{axis}_motor" for side in ("left", "right") for axis in ("knee", "ankle_pitch", "ankle_roll")
)
TORQUE_BUDGET_FRACTION = 0.8


def _motor_components(model: mujoco.MjModel, components: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Require the supplied mass/COM ledger to describe the compiled robot."""
    root = model.body("torso").id
    masses = np.zeros(model.nbody)
    moments = np.zeros((model.nbody, 3))
    named: dict[str, dict[str, Any]] = {}
    for part in components:
        name = str(part["name"])
        if name in named:
            raise ValueError(f"Duplicate ledger component: {name}.")
        body = model.body(part["body"]).id
        mass, position = float(part["mass_kg"]), np.asarray(part["pos_m"], dtype=float)
        if not math.isfinite(mass) or mass <= 0 or position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError(f"Invalid component mass or position: {name}.")
        if model.body_rootid[body] != root:
            raise ValueError(f"Component {name} is outside the robot subtree.")
        named[name] = part
        masses[body] += mass
        moments[body] += mass * position
    robot = np.flatnonzero(model.body_rootid == root)
    if not np.allclose(masses[robot], model.body_mass[robot], rtol=0, atol=1e-8):
        raise ValueError("Component ledger body masses do not match the compiled model.")
    occupied = robot[masses[robot] > 0]
    if not np.allclose(moments[occupied] / masses[occupied, None], model.body_ipos[occupied], rtol=0, atol=1e-8):
        raise ValueError("Component ledger centers of mass do not match the compiled model.")
    if not math.isclose(float(masses.sum()), float(model.body_subtreemass[root]), abs_tol=1e-8, rel_tol=0):
        raise ValueError("Compiled subtree mass is stale or inconsistent with the component ledger.")
    if any(name not in named for name in MOVED_MOTORS):
        raise ValueError("The comparison requires all six named knee and ankle motor components.")
    return [named[name] for name in MOVED_MOTORS]


def _support_loads(
    model: mujoco.MjModel, data: mujoco.MjData, gravity_bias: np.ndarray, center: np.ndarray
) -> dict[str, Any]:
    root = model.body("torso").id
    support = np.zeros(model.nv)
    feet: dict[str, Any] = {}
    floor_z = float(data.geom_xpos[model.geom("floor").id, 2])
    for side in ("left", "right"):
        geom = model.geom(side + "_sole_collision").id
        rotation = data.geom_xmat[geom].reshape(3, 3)
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_BOX or not np.allclose(
            rotation[:, 2], [0, 0, 1], atol=1e-10, rtol=0
        ):
            raise ValueError("Static support comparison requires horizontal box soles.")
        point = data.geom_xpos[geom].copy()
        point[0], point[2] = center[0], floor_z
        local = rotation.T @ (point - data.geom_xpos[geom])
        margin = model.geom_size[geom, :2] - np.abs(local[:2])
        jacobian = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacobian, None, point, int(model.geom_bodyid[geom]))
        force = np.array([0.0, 0.0, model.body_subtreemass[root] * 9.81 / 2])
        support += jacobian.T @ force
        feet[side] = {
            "assumed_center_of_pressure_m": point.tolist(),
            "assumed_vertical_force_n": float(force[2]),
            "cop_margin_xy_m": margin.tolist(),
            "sole_bottom_z_m": float(data.geom_xpos[geom, 2] - model.geom_size[geom, 2]),
        }
    required = gravity_bias - support
    root_dof = int(model.joint("root").dofadr[0])
    residual = float(np.linalg.norm(required[root_dof : root_dof + 6]))
    torques: dict[str, float] = {}
    fractions: dict[str, float] = {}
    for name in LEG_JOINTS:
        joint, actuator = model.joint(name).id, model.actuator(name + "_servo").id
        if (
            model.actuator_trntype[actuator] != mujoco.mjtTrn.mjTRN_JOINT
            or model.actuator_trnid[actuator, 0] != joint
            or not np.array_equal(model.actuator_gear[actuator], [1, 0, 0, 0, 0, 0])
            or model.jnt_actfrclimited[joint]
        ):
            raise ValueError("Static support comparison requires direct unit-gear leg actuators without extra caps.")
        torque = float(required[model.jnt_dofadr[joint]])
        cap = float(model.actuator_forcerange[actuator, 1 if torque >= 0 else 0])
        torques[name], fractions[name] = torque, abs(torque / cap)
    if not np.isfinite(required).all() or not np.isfinite(center).all():
        raise ValueError("Static support calculation produced nonfinite values.")
    return {
        "whole_body_com_m": center.tolist(),
        "feet": feet,
        "required_leg_torque_nm": torques,
        "torque_cap_fraction": fractions,
        "maximum_torque_fraction": max(fractions.values()),
        "free_root_equilibrium_residual_norm": residual,
        "checks": {
            "cop_inside_both_soles": all(min(foot["cop_margin_xy_m"]) >= 0 for foot in feet.values()),
            "soles_at_floor": all(abs(foot["sole_bottom_z_m"] - floor_z) <= 1e-9 for foot in feet.values()),
            "free_root_equilibrium": residual <= 1e-8,
            "within_gravity_torque_budget": max(fractions.values()) <= TORQUE_BUDGET_FRACTION,
        },
    }


def compare_motor_locations(
    model: mujoco.MjModel, components: list[dict[str, Any]], *, knee_angle: float = 1.1
) -> dict[str, Any]:
    """Compare the actual layout with moving six motor masses to the torso.

    ``components`` is the full ``components.json["components"]`` list. Each
    scenario uses explicit equal vertical ground forces at the sagittal COM.
    Failed center-of-pressure/equilibrium checks make that assumed support
    infeasible even when individual actuator torques remain below their caps.
    """
    motors = _motor_components(model, components)
    home = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, home, model.key("home").id)
    require_unsupported(model, home)
    if np.any(home.eq_active):
        raise ValueError("Static support comparison requires uncoupled joints.")
    home.qvel[:] = 0
    mujoco.mj_forward(model, home)
    crouch = prepare_crouch(model, knee_angle=knee_angle)
    root, root_address = model.body("torso").id, int(model.joint("root").qposadr[0])
    rows = []
    for pose, data in (("home", home), ("deep_crouch", crouch)):
        bias, center = data.qfrc_bias.copy(), data.subtree_com[root].copy()
        actual = _support_loads(model, data, bias, center)
        for part in motors:
            body, mass = model.body(part["body"]).id, float(part["mass_kg"])
            old = data.xpos[body] + data.xmat[body].reshape(3, 3) @ np.asarray(part["pos_m"])
            side = 1 if part["name"].startswith("left") else -1
            new = data.xpos[root] + data.xmat[root].reshape(3, 3) @ np.array([0.015, side * 0.060, 0.040])
            old_jac, new_jac = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, old_jac, None, old, body)
            mujoco.mj_jac(model, data, new_jac, None, new, root)
            bias -= (new_jac - old_jac).T @ (mass * model.opt.gravity)
            center += mass * (new - old) / model.body_subtreemass[root]
        rows.append(
            {
                "pose": pose,
                "root_height_m": float(data.qpos[root_address + 2]),
                "joint_angles_rad": {name: float(data.qpos[model.joint(name).qposadr[0]]) for name in LEG_JOINTS},
                "actual_layout": actual,
                "ideal_torso_relocation": _support_loads(model, data, bias, center),
            }
        )
    return {
        "scope": "Symmetric two-foot quasi-static gravity balance; no transmission or recovery validation.",
        "model_mass_kg": float(model.body_subtreemass[root]),
        "relocated_motor_mass_kg": sum(float(part["mass_kg"]) for part in motors),
        "relocated_motor_names": list(MOVED_MOTORS),
        "allowed_torque_fraction": TORQUE_BUDGET_FRACTION,
        "assumptions": [
            "Both soles are horizontal; each foot supplies half the body weight vertically, with sagittal "
            "center of pressure at the scenario's whole-body center of mass and lateral position at sole center.",
            "The crouch uses hip and ankle pitch offsets of minus half the knee angle. The torso stays upright.",
            "The alternative moves knee/ankle motor mass to torso-local points (0.015, +/-0.060, 0.040)m; "
            "coincident motors are abstract mass locations, not a packaged or clearance-checked mechanism.",
            "The alternative assumes zero transmission mass, losses, compliance or backlash and unchanged "
            "joint torque caps. Required output torque still has to pass through a real transmission.",
            "There is no acceleration, friction, thermal model or ground-contact solver. Infeasible support "
            "checks remain failures; this calculation does not demonstrate standing up or recovery after a fall.",
        ],
        "poses": rows,
    }
