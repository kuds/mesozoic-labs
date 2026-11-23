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
    """A flat-ground control, acting only through the sixteen original limb motors."""

    def __init__(self, model):
        self.model = model
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

    def update(self, data):
        m = self.model
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
        steer = np.clip(0.7 * yaw + 0.2 * data.xpos[self.trunk, 1], -0.10, 0.10)
        strides = STRIDE + np.array([steer, -steer, steer, -steer])
        targets = self.home.copy()
        targets[:, 0] += amplitude * strides * fore_aft
        targets[:, 1] += amplitude * SWAY * np.cos(2 * np.pi * (t / PERIOD - 0.045))
        targets[:, 2] += amplitude * LIFT * np.sin(np.pi * swing) ** 2

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
        shares /= shares.sum()
        for i, site in enumerate(self.sites):
            mujoco.mj_jacSite(m, data, self.jac_pos, None, site)
            effort[i] -= self.jac_pos[2, self.dofs[i]] * shares[i] * self.weight
        limits = m.actuator_ctrlrange[self.motors]
        data.ctrl[self.motors] = np.clip(effort / m.actuator_gear[self.motors, 0], limits[:, :, 0], limits[:, :, 1])


def main():
    import time

    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    walk = ReferenceWalk(model)
    steps = round(CONTROL_DT / model.opt.timestep)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 8.2
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -12
        while viewer.is_running():
            start = time.monotonic()
            walk.update(data)
            mujoco.mj_step(model, data, nstep=steps)
            mujoco.mj_forward(model, data)
            viewer.cam.lookat[:] = data.subtree_com[walk.trunk]
            viewer.sync()
            time.sleep(max(0.0, steps * model.opt.timestep - (time.monotonic() - start)))


if __name__ == "__main__":
    main()
