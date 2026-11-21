"""Free-root, flat-ground biped reference driven by residual leg servos."""

from __future__ import annotations

import mujoco
import numpy as np

from environments.shared.base_env import BaseDinoEnv

SIDES = ("r", "l")
FOOTFALL_PHASE = np.array([0.0, 0.5])


class BipedReferenceWalk:
    """Use the leg servos to place alternating feet and support the measured trunk."""

    period: float
    stride: float
    stance_fraction: float
    swing_height: float
    lateral_sway: float
    walking_crouch: float
    medial_step: float
    start = 0.80
    settle = 1.0
    rocker_radius = 0.0
    pad_suffix: str
    cop_min: tuple[float, float]
    cop_max: tuple[float, float]
    lateral_accel_gain = 1.1
    position_lead = 0.85
    velocity_lead = 0.055
    orientation_lead = -0.7
    yaw_lead = 0.0
    angular_stiffness = 140.0
    angular_damping = 12.0
    lateral_stiffness = 25.0
    lateral_damping = 7.0

    def __init__(self, env: BaseDinoEnv):
        self.env = env
        model = self.model = env.model
        self.reference = mujoco.MjData(model)
        self.key = model.key("home")
        self.start_time = env.data.time
        self.origin: np.ndarray | None = None
        mujoco.mj_resetDataKeyframe(model, self.reference, self.key.id)
        mujoco.mj_forward(model, self.reference)
        self.pads = np.array([model.geom(f"{side}_{self.pad_suffix}").id for side in SIDES])
        self.sole_pitch = np.zeros(len(self.pads))
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

    def _distance(self, cycles):
        return self.stride * cycles

    def _next_landing(self, touchdown, landing):
        return landing + self.stride

    def _speed(self, cycles):
        return self.stride / self.period

    def _foot_pitch(self, phase, cycles):
        return np.zeros_like(phase)

    def _targets(self, time: float):
        cycles = max(0.0, (time - self.start) / self.period)
        lap, phase = np.divmod(cycles - FOOTFALL_PHASE, 1.0)
        swing = np.clip((phase - self.stance_fraction) / (1 - self.stance_fraction), 0, 1)
        blend = swing**3 * (10 - 15 * swing + 6 * swing**2)
        touchdown = lap + FOOTFALL_PHASE
        landing = self._distance(touchdown + self.stance_fraction / 2)
        before = np.where(touchdown > 0, landing, 0)
        after = np.where(touchdown + 1 > 0, self._next_landing(touchdown, landing), 0)
        feet = self.home_feet.copy()
        feet[:, 0] += before + (after - before) * blend
        self.sole_pitch = self._foot_pitch(phase, cycles)
        # Rotate the pad target about a point anterior to it during toe-off.
        feet[:, 2] += self.rocker_radius * np.sin(self.sole_pitch)
        feet[:, 0] += self.rocker_radius * (1 - np.cos(self.sole_pitch))
        feet[:, 2] += self.swing_height * np.sin(np.pi * swing) ** 2
        # Move each track medially over successive landings, beginning in
        # the reset stance. Both anchors remain world-fixed during support.
        before_width = np.clip((touchdown - 3) / 3, 0, 1)
        after_width = np.clip((touchdown - 2) / 3, 0, 1)
        feet[:, 1] += self.medial_step * np.array([1, -1]) * (before_width + blend * (after_width - before_width))

        ramp = min(time / self.start, 1.0)
        angle = 2 * np.pi * (cycles - self.stance_fraction / 2)
        sway = -self.lateral_sway * np.cos(angle)
        root = self.key.qpos[:3] + [self._distance(cycles), sway * ramp, -self.walking_crouch * ramp]
        frequency = 2 * np.pi / self.period
        if time > self.start:
            velocity = np.array([self._speed(cycles), self.lateral_sway * frequency * np.sin(angle), 0.0])
        else:
            velocity = np.array([0.0, sway / self.start, 0.0])
        lateral_accel = -self.lateral_accel_gain * frequency**2 * sway if time >= self.start else 0.0
        return feet, root, velocity, lateral_accel, (phase < self.stance_fraction) | (time < self.start)

    def _place_legs(self, data, feet, root, velocity):
        model, ref = self.model, self.reference
        ref.qpos[:3] = root
        ref.qpos[:2] += self.position_lead * (root[:2] - data.qpos[:2]) + self.velocity_lead * (
            velocity[:2] - data.qvel[:2]
        )
        ref.qpos[3:7] = [
            1.0,
            self.orientation_lead * data.qpos[4],
            self.orientation_lead * data.qpos[5],
            self.yaw_lead * data.qpos[6],
        ]
        ref.qpos[3:7] /= np.linalg.norm(ref.qpos[3:7])
        for _ in range(2):
            mujoco.mj_kinematics(model, ref)
            mujoco.mj_comPos(model, ref)
            for pad, joints, target, angle in zip(self.pads, self.legs, feet, self.sole_pitch, strict=True):
                dofs, qpos = model.jnt_dofadr[joints], model.jnt_qposadr[joints]
                linear = np.zeros((3, model.nv))
                angular = np.zeros_like(linear)
                mujoco.mj_jacGeom(model, ref, linear, angular, pad)
                frame = ref.xmat[model.geom_bodyid[pad]]
                pitch = np.arctan2(-frame[6], frame[0])
                jac = np.vstack([linear[:, dofs], angular[1, dofs]])
                # Match the plantar path and sole pitch. The passive digits remain free.
                error = np.r_[target - ref.geom_xpos[pad], angle - pitch]
                delta = np.linalg.lstsq(jac, error, rcond=1e-6)[0]
                ref.qpos[qpos] = np.clip(ref.qpos[qpos] + np.clip(delta, -0.16, 0.16), *model.jnt_range[joints].T)

    def _load_shares(self, planned):
        return planned

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
        shares = self._load_shares(shares)
        mean = shares @ center[:, :2]
        # Bound the reaction point by the load-bearing plantar surface.
        dx, dy = np.clip(cop - mean, self.cop_min, self.cop_max)
        arm = np.r_[mean + [dx, 0.0] - data.qpos[:2], -data.qpos[2]]
        yaw_residual = moment[2] - np.cross(arm, force)[2]
        limit = 0.6 * min(shares) * max(0.0, force[2])
        differential = np.clip(yaw_residual / width, -limit, limit)
        support = np.zeros(model.nv)
        for side in np.flatnonzero(shares):
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
        time = data.time - self.start_time - self.settle
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
            np.array([10, self.lateral_stiffness, 55]) * (root - data.qpos[:3])
            + np.array([5, self.lateral_damping, 10]) * (velocity - data.qvel[:3])
        )
        wrench[1] += self.mass * lateral_accel
        inertia = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, data, inertia)
        wrench[3:] += inertia[3:6, 3:6] @ (
            -self.angular_stiffness * data.qpos[4:7] * np.sign(data.qpos[3]) - self.angular_damping * data.qvel[3:6]
        )
        support = self._support(data, active, wrench)
        torque = (data.qfrc_bias - data.qfrc_spring - support)[self.dofs]
        torque += model.dof_damping[self.dofs] * (desired - previous) / dt
        ctrl = np.clip(desired + torque / model.actuator_gainprm[:, 0], *model.actuator_ctrlrange.T)
        delta = ctrl - self.key.ctrl
        span = np.where(
            delta < 0, self.key.ctrl - model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1] - self.key.ctrl
        )
        return np.asarray(delta / span)
