"""Straight, slow reference walk for the anatomical Velociraptor on level ground.

Construct ReferenceWalk after env.reset(). Pass action(env.data, env.dt) to
env.step(). Only a private kinematics workspace receives prescribed positions.
The live root, passive digits and all floor contacts evolve in MuJoCo.
Run the species' audit_motion module with --mode walk to replay and measure it.
"""

from __future__ import annotations

import mujoco
import numpy as np

from environments.velociraptor_mongoliensis.anatomical.env import VelociraptorMongoliensisAnatomicalEnv

PERIOD = 0.70
STRIDE = 0.18
STANCE_FRACTION = 0.65
SWING_HEIGHT = 0.032
LATERAL_SWAY = 0.028
WALKING_CROUCH = 0.012
MEDIAL_STEP = 0.020
START = 0.80
SETTLE = 1.0
SIDES = ("r", "l")
FOOTFALL_PHASE = np.array([0.0, 0.5])


class ReferenceWalk:
    """Use the leg servos to place alternating feet and support the measured trunk."""

    def __init__(self, env: VelociraptorMongoliensisAnatomicalEnv):
        model = self.model = env.model
        self.reference = mujoco.MjData(model)
        self.key = model.key("home")
        self.start_time = env.data.time
        self.origin: np.ndarray | None = None
        mujoco.mj_resetDataKeyframe(model, self.reference, self.key.id)
        mujoco.mj_forward(model, self.reference)
        self.pads = np.array([model.geom(f"{side}_mtp_pad").id for side in SIDES])
        self.home_feet = self.reference.geom_xpos[self.pads].copy()
        self.pelvis = model.body("pelvis").id
        self.mass = model.body_subtreemass[self.pelvis]
        actuated = model.actuator_trnid[:, 0]
        self.qpos = model.jnt_qposadr[actuated]
        self.dofs = model.jnt_dofadr[actuated]
        self.legs = [
            np.array([model.joint(f"{side}_{j}").id for j in ("hip_pitch", "hip_roll", "knee", "ankle")])
            for side in SIDES
        ]

    def _targets(self, time: float):
        cycles = max(0.0, (time - START) / PERIOD)
        lap, phase = np.divmod(cycles - FOOTFALL_PHASE, 1.0)
        swing = np.clip((phase - STANCE_FRACTION) / (1 - STANCE_FRACTION), 0, 1)
        blend = swing**3 * (10 - 15 * swing + 6 * swing**2)
        touchdown = lap + FOOTFALL_PHASE
        landing = STRIDE * (touchdown + STANCE_FRACTION / 2)
        before = np.where(touchdown > 0, landing, 0)
        after = np.where(touchdown + 1 > 0, landing + STRIDE, 0)
        feet = self.home_feet.copy()
        feet[:, 0] += before + (after - before) * blend
        feet[:, 2] += SWING_HEIGHT * np.sin(np.pi * swing) ** 2
        # Move each track medially over successive landings, beginning in
        # the reset stance. Both anchors remain world-fixed during support.
        before_width = np.clip((touchdown - 3) / 3, 0, 1)
        after_width = np.clip((touchdown - 2) / 3, 0, 1)
        feet[:, 1] += MEDIAL_STEP * np.array([1, -1]) * (before_width + blend * (after_width - before_width))

        ramp = min(time / START, 1.0)
        angle = 2 * np.pi * (cycles - STANCE_FRACTION / 2)
        sway = -LATERAL_SWAY * np.cos(angle)
        root = self.key.qpos[:3] + [STRIDE * cycles, sway * ramp, -WALKING_CROUCH * ramp]
        frequency = 2 * np.pi / PERIOD
        if time > START:
            velocity = np.array([STRIDE / PERIOD, LATERAL_SWAY * frequency * np.sin(angle), 0.0])
        else:
            velocity = np.array([0.0, sway / START, 0.0])
        lateral_accel = -1.1 * frequency**2 * sway if time >= START else 0.0
        return feet, root, velocity, lateral_accel, (phase < STANCE_FRACTION) | (time < START)

    def _place_legs(self, data, feet, root, velocity):
        model, ref = self.model, self.reference
        ref.qpos[:3] = root
        ref.qpos[:2] += 0.85 * (root[:2] - data.qpos[:2]) + 0.055 * (velocity[:2] - data.qvel[:2])
        ref.qpos[3:7] = [1.0, -0.7 * data.qpos[4], -0.7 * data.qpos[5], 0.0]
        ref.qpos[3:7] /= np.linalg.norm(ref.qpos[3:7])
        for _ in range(2):
            mujoco.mj_kinematics(model, ref)
            mujoco.mj_comPos(model, ref)
            for pad, joints, target in zip(self.pads, self.legs, feet, strict=True):
                dofs, qpos = model.jnt_dofadr[joints], model.jnt_qposadr[joints]
                linear = np.zeros((3, model.nv))
                angular = np.zeros_like(linear)
                mujoco.mj_jacGeom(model, ref, linear, angular, pad)
                frame = ref.xmat[model.geom_bodyid[pad]]
                pitch = np.arctan2(-frame[6], frame[0])
                jac = np.vstack([linear[:, dofs], angular[1, dofs]])
                # Match the MTP path and the home metatarsal inclination. Digits
                # III and IV are passive and remain free to flex against the floor.
                error = np.r_[target - ref.geom_xpos[pad], -pitch]
                delta = np.linalg.lstsq(jac, error, rcond=1e-6)[0]
                ref.qpos[qpos] = np.clip(ref.qpos[qpos] + np.clip(delta, -0.16, 0.16), *model.jnt_range[joints].T)

    def _support(self, data, active, wrench):
        """Resolve a desired root wrench into joint torques using the support feet."""
        model = self.model
        force = wrench[:3]
        moment = data.xmat[self.pelvis].reshape(3, 3) @ wrench[3:]
        center = data.geom_xpos[self.pads]
        # The world center of pressure producing this pitch and roll moment,
        # including the moment arm of the horizontal reaction at ground level.
        cop = data.qpos[:2] + (np.array([-moment[1], moment[0]]) - data.qpos[2] * force[:2]) / max(
            force[2], 0.5 * self.mass * 9.81
        )
        width = max(center[1, 1] - center[0, 1], 0.001)
        left = np.clip((cop[1] - center[0, 1]) / width, 0, 1)
        shares = np.array([1 - left, left]) * active
        if shares.sum() < 1e-4:
            shares = active.astype(float)
        shares /= shares.sum()
        mean = shares @ center[:, :2]
        # The MTP pad and distal III/IV pads bound the usable pressure line.
        dx, dy = np.clip(cop - mean, [-0.010, -0.010], [0.105, 0.010])
        arm = np.r_[mean + [dx, 0.0] - data.qpos[:2], -data.qpos[2]]
        yaw_residual = moment[2] - np.cross(arm, force)[2]
        limit = 0.6 * min(shares) * max(0.0, force[2])
        differential = np.clip(yaw_residual / width, -limit, limit)
        support = np.zeros(model.nv)
        for side in np.flatnonzero(active):
            contact = center[side] + [dx, dy, -center[side, 2]]
            jac = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jac, None, contact, model.geom_bodyid[self.pads[side]])
            share = force * shares[side]
            share[0] += differential * (1 - 2 * side)
            support += jac.T @ share
        return support

    def action(self, data: mujoco.MjData, dt: float) -> np.ndarray:
        """Return one normalized residual action for the next control interval."""
        model = self.model
        time = data.time - self.start_time - SETTLE
        if time < 0:
            return np.zeros(model.nu)
        # A noisy reset settles at a displaced pelvis. Use that location as
        # the origin of the straight track, keeping the world +x direction.
        if self.origin is None:
            self.origin = data.qpos[:2].copy()
        feet, root, velocity, lateral_accel, active = self._targets(time)
        root[:2] += self.origin
        feet[:, :2] += self.origin
        previous = self.reference.qpos[self.qpos].copy()
        self._place_legs(data, feet, root, velocity)
        desired = self.reference.qpos[self.qpos]
        wrench = data.qfrc_bias[:6].copy()
        wrench[:3] += self.mass * (
            np.array([10, 25, 55]) * (root - data.qpos[:3]) + np.array([5, 7, 10]) * (velocity - data.qvel[:3])
        )
        wrench[1] += self.mass * lateral_accel
        inertia = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, data, inertia)
        wrench[3:] += inertia[3:6, 3:6] @ (-140 * data.qpos[4:7] * np.sign(data.qpos[3]) - 12 * data.qvel[3:6])
        support = self._support(data, active, wrench)
        torque = (data.qfrc_bias - data.qfrc_spring - support)[self.dofs]
        torque += model.dof_damping[self.dofs] * (desired - previous) / dt
        ctrl = np.clip(desired + torque / model.actuator_gainprm[:, 0], *model.actuator_ctrlrange.T)
        delta = ctrl - self.key.ctrl
        span = np.where(
            delta < 0, self.key.ctrl - model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1] - self.key.ctrl
        )
        return np.asarray(delta / span)
