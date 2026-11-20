"""Level-ground, four-beat reference walk driven only by the limb motors."""

from __future__ import annotations

import math

import mujoco
import numpy as np

from .limbs import JOINTS, LEGS, ROOT_HEIGHT, link_angles, morphology

PERIOD = 1.85
STRIDE = 0.46
DUTY = 0.76
START = 1.0
# Right hind, right fore, left hind, left fore, expressed in LEGS order.
TOUCHDOWN = np.array((0.75, 0.25, 0.50, 0.0))


class WalkController:
    """Track world-fixed stance targets and support the free torso through the feet."""

    def __init__(self, model):
        self.model = model
        self.homes = np.array([morphology(leg)[0][:2] for leg in LEGS])
        self.sites = [model.site(leg + "_sole").id for leg in LEGS]
        joint_ids = np.array([[model.joint(f"{leg}_{name}").id for name in JOINTS] for leg in LEGS])
        self.qidx = model.jnt_qposadr[joint_ids]
        self.vidx = model.jnt_dofadr[joint_ids]
        self.motor = np.array([[model.actuator(f"{leg}_{name}").id for name in JOINTS] for leg in LEGS])
        self.rest = [link_angles(0, z - h, lengths, bend) for (_, _, z), lengths, h, bend in map(morphology, LEGS)]
        self.mass = model.body_subtreemass[model.body("torso").id]
        self.kp = np.array([[9000, 7500, 6000, 2400]] * 2 + [[14000, 12500, 10500, 3400]] * 2)
        self.kd = np.array([[105, 90, 68, 25]] * 2 + [[220, 190, 145, 35]] * 2)
        self.previous = None
        self.walk_start = None
        self.origin = None
        self.jac = np.zeros((4, 3, model.nv))
        self.inertia = np.zeros((model.nv, model.nv))

    def _trajectory(self, time):
        """Smooth foot arcs join fixed ground targets with zero end velocity."""
        cycles = max(0.0, (time - START) / PERIOD)
        lap, phase = np.divmod(cycles - TOUCHDOWN, 1.0)
        swing = np.clip((phase - DUTY) / (1 - DUTY), 0.0, 1.0)
        touchdown = lap + TOUCHDOWN
        landing = STRIDE * (touchdown + DUTY / 2)
        before = np.where(touchdown > 0, landing, 0)
        after = np.where(touchdown + 1 > 0, landing + STRIDE, 0)
        progress = swing**3 * (10 - 15 * swing + 6 * swing**2)
        feet = np.column_stack((self.homes, np.zeros(4)))
        feet[:, 0] += before + (after - before) * progress
        feet[:, 2] += 0.073 * (4 * swing * (1 - swing)) ** 2
        # Shift toward the planted hindlimb when its opposite number swings.
        angle = 2 * math.pi * (cycles - 0.85)
        ramp = np.clip(time / START, 0.0, 1.0)
        sway = 0.10
        # A small flexed-elbow posture leaves reach for forward progression.
        # Move the trunk forward over the forefeet once the first steps begin.
        settle = 0.015
        transition = np.clip((cycles - 1 / 3) / 0.75, 0.0, 1.0)
        advance = 0.06 * transition**2 * (3 - 2 * transition)
        root = np.array((STRIDE * cycles + advance, sway * math.cos(angle) * ramp, ROOT_HEIGHT - settle * ramp))
        velocity = np.zeros(3)
        if time >= START:
            velocity[:2] = [
                STRIDE / PERIOD + 0.06 * 6 * transition * (1 - transition) / (0.75 * PERIOD),
                -sway * math.sin(angle) * 2 * math.pi / PERIOD,
            ]
        elif time >= 0:
            velocity[1:] = [sway * math.cos(angle) / START, -settle / START]
        weights = np.ones(4)
        if time >= START:
            # Transfer load over the final and initial 4% of stance.
            weights = np.where(phase < DUTY, np.clip(np.minimum(phase, DUTY - phase) / 0.04, 0.001, 1), 0.0)
        # Tip the toes down just after liftoff and up on approach. Both
        # angle and angular velocity return to zero where the sole plants.
        foot_pitch = 0.19 * (4 * swing * (1 - swing)) ** 2 * np.sin(2 * math.pi * swing)
        return feet, root, velocity, weights, foot_pitch

    def _targets(self, feet, root):
        targets = []
        for i, leg in enumerate(LEGS):
            (x, y, z), lengths, h, bend = morphology(leg)
            down = root[2] + z - ROOT_HEIGHT - feet[i, 2]
            dy = feet[i, 1] - root[1] - y
            roll = math.atan2(dy, down)
            angles = link_angles(feet[i, 0] - root[0] - x, math.hypot(dy, down) - h, lengths, bend)
            shoulder, absolute_shin = np.subtract(self.rest[i], angles)
            targets.append((roll, shoulder, absolute_shin - shoulder, -absolute_shin))
        return np.array(targets)

    def _support(self, data, weights, wrench):
        for i, site in enumerate(self.sites):
            mujoco.mj_jacSite(self.model, data, self.jac[i], None, site)
        active = np.flatnonzero(weights > 0)
        # The unactuated base coordinates directly map point forces to a base
        # wrench. Remove negative vertical reactions (the ground cannot pull).
        while len(active):
            jac = self.jac[active].reshape(-1, self.model.nv)
            weight = np.repeat(weights[active], 3)
            forces = weight * np.linalg.lstsq(jac[:, :6].T * weight, wrench, rcond=1.0e-5)[0]
            normal = forces[2::3]
            lightest = int(np.argmin(normal))
            if normal[lightest] >= 0 or len(active) <= 2:
                forces[2::3] = np.maximum(0, normal)
                return jac.T @ forces
            active = np.delete(active, lightest)
        return np.zeros(self.model.nv)

    def step(self, data, walking=True):
        """Write joint motor commands for one physics tick, then call mj_step externally."""
        if self.origin is None:
            self.origin = data.qpos[:2].copy()
        if walking and self.walk_start is None:
            self.walk_start = data.time
        time = data.time - self.walk_start if walking else -1.0
        feet, root, velocity, weights, foot_pitch = self._trajectory(time)
        if not walking:
            if self.walk_start is not None:
                self.origin = data.qpos[:2].copy()
                self.previous = None
            self.walk_start = None
            feet[:, :2] = self.homes
            feet[:, 2] = 0
        desired = self._targets(feet, root)
        desired[:, 3] += foot_pitch
        if self.previous is None:
            self.previous = desired.copy()
        qdot = (desired - self.previous) / self.model.opt.timestep
        self.previous = desired
        root[:2] += self.origin
        wrench = data.qfrc_bias[:6].copy()
        wrench[:3] += self.mass * (
            np.array((8, 22, 65)) * (root - data.qpos[:3]) + np.array((5, 8, 13)) * (velocity - data.qvel[:3])
        )
        mujoco.mj_fullM(self.model, data, self.inertia)
        wrench[3:] += self.inertia[3:6, 3:6] @ (-110 * data.qpos[4:7] * np.sign(data.qpos[3]) - 12 * data.qvel[3:6])
        reaction = self._support(data, weights, wrench)
        torque = (data.qfrc_bias - reaction)[self.vidx]
        torque += self.kp * (desired - data.qpos[self.qidx]) + self.kd * (qdot - data.qvel[self.vidx])
        limits = self.model.actuator_ctrlrange[self.motor]
        data.ctrl[self.motor] = np.clip(torque, limits[:, :, 0], limits[:, :, 1])
