"""A slow lateral-sequence reference walk for the adult anatomical MuJoCo scene."""

from pathlib import Path

import mujoco
import numpy as np

MODEL_PATH = Path(__file__).with_name("anatomical.xml")
FEET = ("left_fore", "right_fore", "left_hind", "right_hind")
# Touchdown phases: left hind, left fore, right hind, right fore.
TOUCHDOWN = np.array([0.25, 0.75, 0.0, 0.50])
PERIOD = 2.4  # seconds per complete stride
DUTY = 0.76
STRIDE = 0.65  # metres
LIFT = 0.045
SWAY = 0.145
CONTROL_DT = 0.01


class ReferenceWalk:
    """Reference walking through the sixteen limb motors, with optional terrain and turning."""

    def __init__(self, model, terrain=None):
        self.model = model
        self.terrain = terrain
        self.path_xy = None
        self.applied_curvature = 0.0
        self.last_time = None
        self.trunk = model.body("trunk").id
        self.sites = [model.site(foot + "_sole").id for foot in FEET]
        self.motors = np.array(
            [[i for i in range(model.nu) if model.actuator(i).name.startswith(foot + "_")] for foot in FEET]
        )
        self.joints = model.actuator_trnid[self.motors, 0]
        self.dofs = model.jnt_dofadr[self.joints]
        self.qpos = model.jnt_qposadr[self.joints]
        self.ik = mujoco.MjData(model)
        mujoco.mj_forward(model, self.ik)
        self.home = self.ik.site_xpos[self.sites] - self.ik.xpos[self.trunk]
        self.home[:, 2] += 0.005  # slight crouch leaves room to extend into contact
        self.jac_pos = np.empty((3, model.nv))
        self.jac_rot = np.empty_like(self.jac_pos)
        self.weight = -model.opt.gravity[2] * model.body_mass.sum()

    def _turn_placements(self, data, curvature, yaw, fore_aft):
        """Place each foot tangent to its own arc about the commanded turn center."""
        xy = data.xpos[self.trunk, :2]
        if self.path_xy is None:
            self.path_xy = xy.copy()
            self.last_xy = xy.copy()
            self.path_heading = yaw
        # Advance the intended path by measured travel. A slower stride then follows
        # the same radius without accumulating a timing-dependent heading error.
        tangent = np.array([np.cos(self.path_heading), np.sin(self.path_heading)])
        distance = (xy - self.last_xy) @ tangent
        change = curvature * distance
        midpoint = self.path_heading + change / 2
        self.path_xy += distance * np.array([np.cos(midpoint), np.sin(midpoint)])
        self.path_heading += change
        self.last_xy = xy.copy()
        heading_error = np.arctan2(np.sin(yaw - self.path_heading), np.cos(yaw - self.path_heading))
        normal = np.array([-np.sin(self.path_heading), np.cos(self.path_heading)])
        lateral_error = (xy - self.path_xy) @ normal
        corrected = curvature - np.clip(heading_error + 0.4 * lateral_error, -0.2, 0.2)
        # Local velocity of a body point: v_foot / v_forward = (1 - k*y, k*x).
        # Inside feet take shorter steps. Fore and hind feet move oppositely in Y.
        stride = STRIDE * fore_aft
        return np.column_stack((stride * (1 - corrected * self.home[:, 1]), stride * corrected * self.home[:, 0]))

    def update(self, data, *, curvature=0.0):
        """Command curvature in inverse meters. Zero after a turn follows the exit tangent."""
        m = self.model
        if not np.isfinite(curvature):
            raise ValueError("Curvature must be finite")
        if self.last_time is not None and data.time < self.last_time:
            self.path_xy = None
            self.applied_curvature = 0.0
            self.last_time = None
            mujoco.mj_resetData(m, self.ik)
            mujoco.mj_forward(m, self.ik)
            if self.terrain is not None:
                self.terrain.reset()
        elapsed = CONTROL_DT if self.last_time is None else max(0.0, data.time - self.last_time)
        self.last_time = data.time
        # Ease into and out of a bend so a mid-stance foot does not jump to a new target.
        self.applied_curvature += min(elapsed / 0.35, 1.0) * (curvature - self.applied_curvature)
        t = max(0.0, data.time - 0.8)
        amplitude = min(t / 0.6, 1.0)
        phase = (t / PERIOD - TOUCHDOWN) % 1.0
        swing = np.clip((phase - DUTY) / (1.0 - DUTY), 0.0, 1.0)
        # Quintic interpolation matches the stance velocity at both ends of swing.
        travel = swing**3 * (10 - 15 * swing + 6 * swing**2)
        fore_aft = DUTY / 2 - phase + travel
        # Mild toe-up at placement and toe-down at release reduces ankle effort.
        sole_pitch = -0.024 * amplitude * fore_aft / (DUTY / 2)
        yaw = np.arctan2(data.xmat[self.trunk, 3], data.xmat[self.trunk, 0])
        targets = self.home.copy()
        if self.applied_curvature != 0.0 or self.path_xy is not None:
            targets[:, :2] += amplitude * self._turn_placements(data, self.applied_curvature, yaw, fore_aft)
        else:
            steer = np.clip(0.7 * yaw + 0.2 * data.xpos[self.trunk, 1], -0.10, 0.10)
            strides = STRIDE + np.array([steer, -steer, steer, -steer])
            targets[:, 0] += amplitude * strides * fore_aft
        # A modest additional lateral transfer on bends helps unload the inner foot cleanly.
        sway = SWAY + min(0.075 * abs(self.applied_curvature), 0.025)
        body_sway = amplitude * sway * np.cos(2 * np.pi * (t / PERIOD - 0.045))
        targets[:, 1] += body_sway
        targets[:, 2] += amplitude * LIFT * np.sin(np.pi * swing) ** 2

        if self.terrain is not None:
            self.terrain.adapt(self, data, targets, sole_pitch, swing, amplitude, body_sway)

        # Solve sole translation and pitch on a scratch pose. The simulated root is free.
        ik = self.ik
        for _ in range(3):
            mujoco.mj_kinematics(m, ik)
            mujoco.mj_comPos(m, ik)
            for i, site in enumerate(self.sites):
                mujoco.mj_jacSite(m, ik, self.jac_pos, self.jac_rot, site)
                dofs = self.dofs[i]
                jac = np.vstack([self.jac_pos[:, dofs], self.jac_rot[1, dofs]])
                delta = targets[i] - (ik.site_xpos[site] - ik.xpos[self.trunk])
                pitch = np.arctan2(-ik.site_xmat[site, 6], ik.site_xmat[site, 8])
                step = np.linalg.solve(jac, np.r_[delta, sole_pitch[i] - pitch])
                ik.qpos[self.qpos[i]] += np.clip(step, -0.12, 0.12)

        reference = ik.qpos[self.qpos]
        effort = (
            m.jnt_stiffness[self.joints] * reference
            + 100_000 * (reference - data.qpos[self.qpos])
            - 1_000 * data.qvel[self.dofs]
        )
        # Quasistatic feedforward shares weight over the planned stance feet.
        # Walking contacts and the resulting ground reaction are solved by MuJoCo.
        support = (phase < DUTY) | (amplitude < 0.95)
        offsets = data.site_xpos[self.sites, :2] - data.subtree_com[self.trunk, :2]
        moments = np.vstack([np.ones(4), offsets.T])
        shares = np.zeros(4)
        shares[support] = np.linalg.lstsq(moments[:, support], [1, 0, 0], rcond=None)[0]
        shares = np.maximum(shares, 0)
        if self.terrain is not None:
            self.terrain.support_shares(shares, support)
        shares /= shares.sum()
        for i, site in enumerate(self.sites):
            mujoco.mj_jacSite(m, data, self.jac_pos, None, site)
            effort[i] -= self.jac_pos[2, self.dofs[i]] * shares[i] * self.weight
        limits = m.actuator_ctrlrange[self.motors]
        data.ctrl[self.motors] = np.clip(effort / m.actuator_gear[self.motors, 0], limits[:, :, 0], limits[:, :, 1])


def main():
    import argparse
    import time

    import mujoco.viewer

    from .courses import COURSES, EXTENDED_COURSES, TURN_COURSES
    from .terrain_walk import TerrainFootPlacement

    turns = {f"turn_{direction}": course for direction, course in TURN_COURSES.items()}
    terrains = {**COURSES, **EXTENDED_COURSES}
    parser = argparse.ArgumentParser(description="Watch the adult walking on a measured reference course.")
    parser.add_argument("--course", choices=["flat", *turns, *terrains], default="flat")
    name = parser.parse_args().course
    terrain = terrains.get(name)
    turn = turns.get(name)
    if terrain is None:
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        walk = ReferenceWalk(model)
    else:
        model = terrain.model(MODEL_PATH)
        data = mujoco.MjData(model)
        terrain.reset(model, data)
        walk = ReferenceWalk(model, terrain=TerrainFootPlacement(terrain))
    steps = round(CONTROL_DT / model.opt.timestep)
    dt = steps * model.opt.timestep
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 8.2
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -12
        while viewer.is_running():
            start = time.monotonic()
            curvature = turn.curvature_at(round(data.time / dt) * dt) if turn else 0.0
            walk.update(data, curvature=curvature)
            mujoco.mj_step(model, data, nstep=steps)
            mujoco.mj_forward(model, data)
            viewer.cam.lookat[:] = data.subtree_com[walk.trunk]
            viewer.sync()
            time.sleep(max(0.0, dt - (time.monotonic() - start)))


if __name__ == "__main__":
    main()
