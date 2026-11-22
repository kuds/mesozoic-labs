"""Straight, slow, level-ground reference for the free-root Allosaurus.

Initialize after reset and forward kinematics. Every 10 ms send control(data)
to the ten joint motors. The controller reads the live state and returns
torques in N·m. All motion and contact forces are integrated by MuJoCo.
"""

from __future__ import annotations

import mujoco
import numpy as np

SIDES = ("r", "l")
PERIOD = 1.0
STRIDE = 0.35
MEDIAL_STEP = 0.008
PRESSURE_FORWARD = 0.065
DUTY = 0.81
LIFT = 0.060
START = 1.2
CONTROL_INTERVAL = 0.010
PLAN_INTERVAL = 0.005


def _smoothstep(x):
    x = np.clip(x, 0, 1)
    return x**3 * (10 - 15 * x + 6 * x**2)


def _progress(cycles):
    cycles = np.maximum(0, cycles)
    return STRIDE * (cycles + 0.8 * np.expm1(-cycles / 0.8))


class ReferenceWalk:
    """Use alternating ground-fixed stance anchors and whole-body inverse dynamics."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, *, walking: bool = True):
        self.model = model
        self.start = data.time
        self.walking = walking
        joints = model.actuator_trnid[:, 0]
        self.qpos = model.jnt_qposadr[joints]
        self.dofs = model.jnt_dofadr[joints]
        self.passive = np.setdiff1d(np.arange(model.nv), self.dofs)
        self.neutral = model.qpos0[self.qpos]
        self.sites = [model.site(f"{s}_foot").id for s in SIDES]
        self.pelvis = model.body("pelvis").id
        self.mass = model.body_subtreemass[self.pelvis]
        self.pads = [[model.geom(f"{side}_pad{suffix}").id for suffix in ("", "_d2", "_d3", "_d4")] for side in SIDES]
        self.placement_offset = np.zeros((2, 2))
        # Plan the course in the initial pelvis heading. Keep the pressure
        # and placement calculations in world coordinates for ground contact.
        forward = data.xmat[self.pelvis].reshape(3, 3)[:2, 0]
        yaw = np.arctan2(forward[1], forward[0])
        self.course = np.array([[np.cos(yaw), np.sin(yaw), 0.0], [-np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
        self.heading_quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        # Reset noise perturbs the two joints and pads differently. Anchor the
        # repeating stride to the symmetric home pose in this heading.
        key = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, key, model.key("home").id)
        mujoco.mj_forward(model, key)
        self.home = (key.site_xpos[self.sites] - key.xpos[self.pelvis]) @ self.course
        self.home[:, :2] += data.xpos[self.pelvis, :2]
        self.home[:, 2] = 0
        self.com = key.subtree_com[self.pelvis] @ self.course
        self.com[:2] += data.xpos[self.pelvis, :2] - key.xpos[self.pelvis, :2] @ self.course[:2, :2]
        self._plan_balance()

    def _foot_path(self, time):
        cycles = np.maximum(0, (time - START) / PERIOD) if self.walking else np.zeros_like(time)
        offsets = np.array([0.0, 0.5])
        touchdown = np.floor(cycles[..., None] - offsets) + offsets
        phase = cycles[..., None] - touchdown
        swing = np.clip((phase - DUTY) / (1 - DUTY), 0, 1)
        before = np.where(touchdown > 0, _progress(touchdown + DUTY / 2), 0)
        after = np.where(touchdown + 1 > 0, _progress(touchdown + 1 + DUTY / 2), 0)
        feet = np.broadcast_to(self.home, touchdown.shape + (3,)).copy()
        blend = _smoothstep(swing)
        feet += (before + (after - before) * blend)[..., None] * self.course[0]
        # Step gradually inward from the wider bilateral reset stance.
        medial_before = _smoothstep((touchdown - 0.25) / 2.5)
        medial_after = _smoothstep((touchdown + 0.75) / 2.5)
        medial = MEDIAL_STEP * (medial_before + (medial_after - medial_before) * blend)
        feet += (medial * np.array([1, -1]))[..., None] * self.course[1]
        feet[..., 2] += LIFT * 16 * (swing * (1 - swing)) ** 2
        ramp = _smoothstep((time - 0.2) / (START - 0.2)) if self.walking else np.zeros_like(time)
        shares = np.where(phase < 0.5, _smoothstep(phase / (DUTY - 0.5)), 1 - _smoothstep((phase - 0.5) / (DUTY - 0.5)))
        return feet, 0.5 + ramp[..., None] * (shares - 0.5), swing

    def _plan_balance(self):
        # The bounded constant-height inverted pendulum solution is the center
        # of pressure convolved with a two-sided exponential. Previewing the
        # load transfer lets the trunk move before the opposite foot lifts.
        self.first_time = -6.0
        time = np.arange(self.first_time, 21, PLAN_INTERVAL)
        feet, shares, _ = self._foot_path(time)
        pressure = (feet[:, :, :2] * shares[:, :, None]).sum(axis=1)
        if self.walking:
            # As walking begins, carry weight forward from the metatarsal
            # cushion onto the pads of the three weight-bearing digits.
            advance = PRESSURE_FORWARD * _smoothstep((time - 0.2) / 1.5)
            pressure += advance[:, None] * self.course[0, :2]
        initial_offset = (self.com[:2] - self.home[:, :2].mean(axis=0)) @ self.course[0, :2]
        pressure += (initial_offset * (1 - _smoothstep((time - 0.2) / 0.9)))[:, None] * self.course[0, :2]
        omega2 = -self.model.opt.gravity[2] / self.com[2]
        decay = np.exp(-np.sqrt(omega2) * PLAN_INTERVAL)
        past, future = pressure.copy(), pressure.copy()
        for i in range(1, len(time)):
            past[i] = decay * past[i - 1] + (1 - decay) * pressure[i]
        for i in range(len(time) - 2, -1, -1):
            future[i] = decay * future[i + 1] + (1 - decay) * pressure[i]
        self.plan = (past + future - (1 - decay) * pressure) / (1 + decay)
        self.velocity = np.gradient(self.plan, PLAN_INTERVAL, axis=0)
        self.acceleration = (self.plan - pressure) * omega2

    def _targets(self, time):
        h = PLAN_INTERVAL
        feet, shares, swing = self._foot_path(np.array([time - h, time, time + h]))
        # Reuse the steady-state cycle for arbitrarily long walks. The
        # exponential startup has converged within this part of the table.
        laps = int((time - 12) / PERIOD) if time > 15 else 0
        index = int((time - laps * PERIOD - self.first_time) / h)
        center = np.r_[self.plan[index], self.com[2]]
        if self.walking:
            center += laps * STRIDE * self.course[0]
        return (
            feet[1],
            (feet[2] - feet[0]) / (2 * h),
            (feet[2] - 2 * feet[1] + feet[0]) / h**2,
            shares[1],
            swing[1],
            center,
            np.r_[self.velocity[index], 0],
            np.r_[self.acceleration[index], 0],
        )

    def control(self, data: mujoco.MjData) -> np.ndarray:
        """Return signed joint torques. Hold the result for CONTROL_INTERVAL."""
        m = self.model
        n = m.nv
        feet, foot_velocity, foot_accel, shares, swing, center, velocity, accel = self._targets(data.time - self.start)
        com_jac = np.zeros((3, n))
        mujoco.mj_jacSubtreeCom(m, data, com_jac, self.pelvis)
        error = center - data.subtree_com[self.pelvis]
        velocity_error = velocity - com_jac @ data.qvel
        # Move an upcoming foot toward the measured capture point. Its target
        # stops moving at touchdown, so the correction cannot drag a stance foot.
        correction = -0.8 * error[:2] - 0.25 * velocity_error[:2]
        self.placement_offset += np.clip(
            0.20 * np.sin(np.pi * swing[:, None]) ** 2 * (correction - self.placement_offset), -0.012, 0.012
        )
        feet[:, :2] += self.placement_offset

        identity = np.eye(n)
        rows = [com_jac, identity[3:6], identity[self.dofs]]
        inverse = np.empty(4)
        orientation_error = np.empty(4)
        mujoco.mju_negQuat(inverse, data.qpos[3:7])
        mujoco.mju_mulQuat(orientation_error, inverse, self.heading_quat)
        targets = [
            accel + 45 * error + 13 * velocity_error,
            290 * np.sign(orientation_error[0]) * orientation_error[1:] - 23 * data.qvel[3:6],
            50 * (self.neutral - data.qpos[self.qpos]) - 8 * data.qvel[self.dofs],
        ]
        weights = [3, 3, 5, 4, 4, 3] + [0.2] * len(self.dofs)
        contact_rows: list[np.ndarray] = []
        prior = []
        for i, (site, share, geoms) in enumerate(zip(self.sites, shares, self.pads, strict=True)):
            jac = np.zeros((3, n))
            angular = np.zeros_like(jac)
            mujoco.mj_jacSite(m, data, jac, angular, site)
            body = m.site_bodyid[site]
            position = data.site_xpos[site]
            jac_dot, angular_dot = np.zeros_like(jac), np.zeros_like(jac)
            mujoco.mj_jacDot(m, data, jac_dot, angular_dot, position, body)
            foot_forward = data.site_xmat[site].reshape(3, 3)[:, 0]
            pitch = np.arctan2(-foot_forward[2], foot_forward @ self.course[0])
            lateral = self.course[1]
            rows.extend([self.course @ jac, (lateral @ angular)[None, :]])
            targets.extend(
                [
                    self.course
                    @ (
                        foot_accel[i]
                        + 280 * (feet[i] - position)
                        + 31 * (foot_velocity[i] - jac @ data.qvel)
                        - jac_dot @ data.qvel
                    ),
                    [-200 * pitch - 26 * (lateral @ angular @ data.qvel) - lateral @ angular_dot @ data.qvel],
                ]
            )
            weights.extend([6, 5, 9, 2] if share > 0.001 else [5, 4, 6, 2])
            if share <= 1e-5:
                continue
            # Sample the rear cushion and the distal part of each pad. These
            # force generators are used only inside the inverse dynamics solve.
            for geom in geoms:
                point = data.geom_xpos[geom].copy()
                point[2] = 0
                if geom == geoms[0]:
                    point -= 0.045 * self.course[0]
                else:
                    point[:2] += 0.65 * (point[:2] - data.geom_xpos[geoms[0], :2])
                reaction = np.zeros_like(jac)
                mujoco.mj_jac(m, data, reaction, None, point, body)
                contact_rows.extend(reaction)
                prior.extend([0, 0, -m.opt.gravity[2] * share / len(geoms)])

        mass = np.empty((n, n))
        mujoco.mj_fullM(m, data, mass)
        bias = data.qfrc_bias - data.qfrc_passive
        tasks = np.vstack(rows) * np.asarray(weights)[:, None]
        goal = np.concatenate(targets) * weights
        support = np.asarray(contact_rows).T
        # Solve joint accelerations and ground reaction per kilogram together.
        # Remove candidate pads that would have to pull on the floor. The
        # floating base and passive tail have no actuator terms in their rows.
        active = np.arange(support.shape[1])
        penalty = np.tile([0.22, 0.22, 0.18], len(prior) // 3)
        for _ in range(len(prior) // 3):
            reaction = support[:, active]
            k = len(active)
            system = np.block(
                [
                    [tasks, np.zeros((len(goal), k))],
                    [45 * mass[self.passive] / self.mass, -45 * reaction[self.passive]],
                    [np.zeros((k, n)), np.diag(penalty[active])],
                ]
            )
            target = np.r_[goal, -45 * bias[self.passive] / self.mass, penalty[active] * np.asarray(prior)[active]]
            solution = np.linalg.lstsq(system, target, rcond=1e-6)[0]
            forces = solution[n:]
            pulls = np.flatnonzero((active % 3 == 2) & (forces < -0.005))
            if not len(pulls) or k == 3:
                break
            worst_pad = active[pulls[np.argmin(forces[pulls])]] // 3
            active = active[active // 3 != worst_pad]
        torque = (mass @ solution[:n] + bias - self.mass * (reaction @ forces))[self.dofs]
        return np.asarray(np.clip(torque, *m.actuator_ctrlrange.T))

    def action(self, data: mujoco.MjData) -> np.ndarray:
        """Return the reference torque as a normalized Gym home-keyframe residual."""
        home = self.model.key("home").ctrl
        lower, upper = self.model.actuator_ctrlrange.T
        delta = self.control(data) - home
        span = np.where(delta < 0, home - lower, upper - home)
        return np.asarray(np.divide(delta, span, out=np.zeros_like(delta), where=span > 0))
