"""Offline gravity preload for a reproducible fixed-target standing keyframe.

Nonnegative vertical foot loads balance total weight and COM moments. Their
Jacobians give nominal joint torques. tau/kp offsets position targets without
changing the home joint angles, gains, force caps, or introducing root forces.
This is a model-based nominal preload, not online balance feedback.
"""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import minimize

from environments.compsognathus.model import floor_contacts


def gravity_targets(model, data):
    contacts = floor_contacts(model, data)
    points = np.array([c["position"] for c in contacts])
    root = model.body("pelvis").id
    com = data.subtree_com[root]
    weight = model.body_subtreemass[root] * -model.opt.gravity[2]
    constraints = np.vstack([np.ones(len(points)), points[:, 0] - com[0], points[:, 1] - com[1]])
    desired = np.array([weight, 0, 0])
    result = minimize(
        lambda forces: np.sum(forces**2),
        np.full(len(points), weight / len(points)),
        jac=lambda forces: 2 * forces,
        constraints={
            "type": "eq",
            "fun": lambda forces: constraints @ forces - desired,
            "jac": lambda forces: constraints,
        },
        bounds=[(0, None)] * len(points),
        method="SLSQP",
        options={"ftol": 1e-12},
    )
    if not result.success or np.max(np.abs(constraints @ result.x - desired)) > 1e-8:
        raise ValueError(f"Cannot balance home on nonnegative vertical foot loads: {result.message}")
    generalized_contact = np.zeros(model.nv)
    for contact, force in zip(contacts, result.x, strict=True):
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        body = int(model.geom_bodyid[model.geom(contact["geom"]).id])
        mujoco.mj_jac(model, data, jp, jr, contact["position"], body)
        generalized_contact += jp[2] * force
    torque = data.qfrc_bias - data.qfrc_passive - generalized_contact
    if np.max(np.abs(torque[:6])) > 1e-8:
        raise ValueError("Nominal support would require an external pelvis wrench")
    joints = model.actuator_trnid[:, 0]
    selected = torque[model.jnt_dofadr[joints]]
    targets = data.qpos[model.jnt_qposadr[joints]] + selected / model.actuator_gainprm[:, 0]
    if np.any(np.abs(selected) >= np.max(np.abs(model.actuator_forcerange), axis=1)):
        raise ValueError("Gravity preload exceeds an actuator torque limit")
    if np.any(targets < model.actuator_ctrlrange[:, 0]) or np.any(targets > model.actuator_ctrlrange[:, 1]):
        raise ValueError("Gravity preload exceeds a position target limit")
    return targets


def write_gravity_targets(path):
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    targets = gravity_targets(model, data)
    root = ET.parse(path)
    root.find("keyframe/key[@name='home']").set("ctrl", " ".join(f"{v:.12g}" for v in targets))
    ET.indent(root, space="  ")
    root.write(path, encoding="utf-8", xml_declaration=True)
