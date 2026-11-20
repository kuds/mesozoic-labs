"""Unsupported-robot checks for standalone MuJoCo engineering experiments.

These helpers inspect a single free-root robot and explicit floor/foot geometry.
They do not change a model, implement a controller or certify a gait. Registered
environment evaluations continue to use ``shared.gait`` and ``FOOT_SUPPORT``.
"""

from __future__ import annotations

import mujoco
import numpy as np


def require_unsupported(model: mujoco.MjModel, data: mujoco.MjData, *, root_joint: str = "root") -> None:
    """Reject artificial assistance without silently disabling it.

    Internal closed-chain constraints are permitted; constraints to anything
    outside the robot's free-root subtree are not. Call after resetting the
    actual trial data, before modifying equality activation for a display demo.
    """
    root = model.joint(root_joint).id
    if model.jnt_type[root] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("Unsupported stance requires a free root joint.")
    root_body = int(model.jnt_bodyid[root])
    if not np.allclose(model.opt.gravity, [0, 0, -9.81], rtol=0, atol=1e-12):
        raise ValueError("Unsupported stance requires Earth gravity along -Z.")
    disabled = mujoco.mjtDisableBit.mjDSBL_GRAVITY | mujoco.mjtDisableBit.mjDSBL_CONTACT
    if model.opt.disableflags & disabled:
        raise ValueError("Gravity and contact physics must be enabled.")
    if np.any(model.body_gravcomp):
        raise ValueError("Body gravity compensation would invalidate unsupported stance.")
    if np.any(data.xfrc_applied) or np.any(data.qfrc_applied):
        raise ValueError("Applied external forces would invalidate unsupported stance.")
    for eq in range(model.neq):
        if not data.eq_active[eq] or model.eq_type[eq] not in (
            mujoco.mjtEq.mjEQ_WELD,
            mujoco.mjtEq.mjEQ_CONNECT,
        ):
            continue
        bodies = [int(model.eq_obj1id[eq]), int(model.eq_obj2id[eq])]
        if model.eq_objtype[eq] == mujoco.mjtObj.mjOBJ_SITE:
            bodies = [int(model.site_bodyid[site]) if site >= 0 else 0 for site in bodies]
        if any(body <= 0 or model.body_rootid[body] != root_body for body in bodies):
            raise ValueError("An active world weld or external fixture constraint invalidates unsupported stance.")
    if not model.actuator_forcelimited.all() or not np.isfinite(model.actuator_forcerange).all():
        raise ValueError("Every actuator needs a finite force limit.")
    if np.any(model.actuator_forcerange[:, 0] >= 0) or np.any(model.actuator_forcerange[:, 1] <= 0):
        raise ValueError("Every actuator needs positive and negative force capacity.")


class GroundSupportProbe:
    """Measure only declared sole-to-floor support, at each physics substep.

    Any robot contact with the environment except those pairs is unexpected:
    this includes the head on the floor and a foot or torso on a fixture. Sole
    contact with another foot is internal, and contributes no ground support.
    The declared floors must be fixed and horizontal (normal forces are then
    vertical loads). Foot count and names are supplied by the caller.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        feet: dict[str, tuple[str, ...]],
        floor_geoms: tuple[str, ...] = ("floor",),
        root_joint: str = "root",
    ) -> None:
        self.model = model
        root_body = int(model.jnt_bodyid[model.joint(root_joint).id])
        self.robot_geoms = set(np.flatnonzero(model.body_rootid[model.geom_bodyid] == root_body).tolist())
        self.floor_geoms = {model.geom(name).id for name in floor_geoms}
        if not self.floor_geoms or self.floor_geoms & self.robot_geoms:
            raise ValueError("Declare fixed external floor geometry.")
        scratch = mujoco.MjData(model)
        mujoco.mj_forward(model, scratch)
        for geom in self.floor_geoms:
            body = int(model.geom_bodyid[geom])
            if model.body_weldid[body] != 0 or not np.allclose(
                scratch.geom_xmat[geom].reshape(3, 3)[:, 2], [0, 0, 1], atol=1e-12, rtol=0
            ):
                raise ValueError("Support loads require stationary horizontal floors.")
        self.foot_names = tuple(feet)
        self.foot_geoms: dict[int, int] = {}
        if not feet or any(not names for names in feet.values()):
            raise ValueError("Declare at least one support geom per foot.")
        for index, names in enumerate(feet.values()):
            for name in names:
                geom = model.geom(name).id
                if geom not in self.robot_geoms or geom in self.foot_geoms:
                    raise ValueError("Each support geom must belong to exactly one robot foot.")
                self.foot_geoms[geom] = index
        self._force = np.zeros(6)

    def sample(self, data: mujoco.MjData) -> tuple[np.ndarray, float]:
        """Return per-foot vertical support and total unexpected normal force (N)."""
        loads = np.zeros(len(self.foot_names))
        unexpected = 0.0
        for index, contact in enumerate(data.contact[: data.ncon]):
            first, second = map(int, contact.geom)
            if (first in self.robot_geoms) == (second in self.robot_geoms):
                continue
            robot, external = (first, second) if first in self.robot_geoms else (second, first)
            mujoco.mj_contactForce(self.model, data, index, self._force)
            normal = max(0.0, float(self._force[0]))
            if external in self.floor_geoms and robot in self.foot_geoms:
                # Project the contact normal; a box floor's side cannot count as
                # vertical support just because its geom is a declared floor.
                vertical_fraction = max(0.0, float(contact.frame[2]) * (1 if robot == second else -1))
                loads[self.foot_geoms[robot]] += normal * vertical_fraction
                unexpected += normal * (1 - vertical_fraction)
            else:
                unexpected += normal
        return loads, unexpected
