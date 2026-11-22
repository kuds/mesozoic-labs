"""Flat-ground quadruped reference for five-joint legs driven by residual position servos.

The IK workspace is independent of the live animal. The controller only outputs
normalized actions. Contact, slip, weight transfer and root motion come from MuJoCo.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from environments.shared.base_env import BaseDinoEnv
from environments.shared.gait.events import debounce, rising_edges, steps_for
from environments.shared.gait.recorder import EpisodeTrace
from environments.shared.gait.stance_track import yaw_pitch_roll

FEET = ("fr", "fl", "rr", "rl")


@dataclass(frozen=True)
class WalkCycle:
    period: float  # seconds per stride
    stride: float  # metres per stride
    stance_fraction: float
    ipsilateral_phase: float  # fore touchdown after the hind foot on the same side
    lift: float  # mid-swing height, metres
    sway: float  # lateral root excursion, metres
    sway_phase: float
    start: float  # seconds to shift weight before advancing
    lowering: float = 0.0  # additional limb flexion during walking, metres


class QuadrupedReferenceWalk:
    """Track ground-fixed stance feet and distribute body weight across the supporting legs."""

    cycle: WalkCycle
    forward_position_gain = 5.0

    def __init__(self, env: BaseDinoEnv):
        model = env.model
        self.model = model
        self.ref = mujoco.MjData(model)
        self.key = model.key("home")
        mujoco.mj_resetDataKeyframe(model, self.ref, self.key.id)
        mujoco.mj_forward(model, self.ref)
        self.feet = np.array([model.geom(f"{leg}_foot_geom").id for leg in FEET])
        self.sole_pitch = np.zeros(len(self.feet))
        self.homes = self.ref.geom_xpos[self.feet].copy()
        self.homes[:, 2] = model.geom_size[self.feet, 2] - 0.0005  # level soles, soft contact
        joints = model.actuator_trnid[:, 0]
        self.actuated_qpos = model.jnt_qposadr[joints]
        self.actuated_dofs = model.jnt_dofadr[joints]
        self.legs = [
            np.array([model.joint(f"{leg}_{joint}").id for joint in ("hip_pitch", "hip_roll", "knee", "ankle", "toe")])
            for leg in FEET
        ]
        self.mass = model.body_subtreemass[model.body("torso").id]

    def targets(self, time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        c = self.cycle
        touchdown_phase = np.array([0.5 + c.ipsilateral_phase, c.ipsilateral_phase, 0.5, 0.0])
        cycles = max(0.0, (time - c.start) / c.period)
        lap, phase = np.divmod(cycles - touchdown_phase, 1.0)
        swing = np.clip((phase - c.stance_fraction) / (1 - c.stance_fraction), 0, 1)
        touchdown = lap + touchdown_phase
        # Place each foot ahead of the torso by half its stance excursion. The
        # first swing starts where that foot stood at reset.
        landing_x = c.stride * (touchdown + c.stance_fraction / 2)
        before = np.where(touchdown > 0, landing_x, 0)
        after = np.where(touchdown + 1 > 0, landing_x + c.stride, 0)
        feet = self.homes.copy()
        feet[:, 0] += before + (after - before) * swing**3 * (10 - 15 * swing + 6 * swing**2)
        feet[:, 2] += c.lift * 16 * (swing * (1 - swing)) ** 2

        ramp = min(1.0, time / c.start)
        angle = 2 * np.pi * (cycles - c.sway_phase)
        root = self.key.qpos[:3] + [c.stride * cycles, c.sway * np.sin(angle) * ramp, -c.lowering * ramp]
        if time < c.start:
            velocity = np.array([0, c.sway * np.sin(angle) / c.start, -c.lowering / c.start])
        else:
            velocity = np.array([c.stride / c.period, c.sway * np.cos(angle) * 2 * np.pi / c.period, 0])
        return feet, root, velocity, phase

    def _place_legs(self, feet: np.ndarray, root: np.ndarray) -> None:
        """Two warm-started IK iterations with specified sole pitches and a weak posture preference."""
        model, ref = self.model, self.ref
        ref.qpos[:3] = root
        for _ in range(2):
            mujoco.mj_kinematics(model, ref)
            mujoco.mj_comPos(model, ref)
            for geom, target, joints, sole_pitch in zip(self.feet, feet, self.legs, self.sole_pitch, strict=True):
                dofs, q = model.jnt_dofadr[joints], model.jnt_qposadr[joints]
                linear = np.zeros((3, model.nv))
                angular = np.zeros_like(linear)
                mujoco.mj_jacGeom(model, ref, linear, angular, geom)
                pitch = np.arctan2(-ref.geom_xmat[geom, 6], ref.geom_xmat[geom, 0])
                jacobian = np.vstack((linear[:, dofs], 0.3 * angular[1, dofs], 0.03 * np.eye(len(q))))
                error = np.r_[
                    target - ref.geom_xpos[geom], 0.3 * (sole_pitch - pitch), 0.03 * (self.key.qpos[q] - ref.qpos[q])
                ]
                delta = np.linalg.lstsq(jacobian, error, rcond=1e-4)[0]
                ref.qpos[q] = np.clip(ref.qpos[q] + np.clip(delta, -0.1, 0.1), *model.jnt_range[joints].T)

    def _support_torque(self, data: mujoco.MjData, stance: np.ndarray, wrench: np.ndarray) -> np.ndarray:
        """Resolve the desired root force and moment at the planned stance feet."""
        jacobians = []
        for geom in self.feet[stance]:
            jacobian = np.zeros((3, self.model.nv))
            point = data.geom_xpos[geom].copy()
            point[2] = 0.0
            mujoco.mj_jac(self.model, data, jacobian, None, point, self.model.geom_bodyid[geom])
            jacobians.append(jacobian)
        while True:
            full = np.vstack(jacobians)
            forces = np.linalg.lstsq(full[:, :6].T, wrench, rcond=1e-4)[0].reshape(-1, 3)
            lightest = int(np.argmin(forces[:, 2]))
            if len(jacobians) <= 2 or forces[lightest, 2] >= 0:
                forces[:, 2] = np.maximum(0, forces[:, 2])
                return np.asarray(full.T @ forces.ravel())
            jacobians.pop(lightest)

    def action(self, data: mujoco.MjData, dt: float) -> np.ndarray:
        model = self.model
        feet, root, velocity, phase = self.targets(data.time)
        previous = self.ref.qpos[self.actuated_qpos].copy()
        self._place_legs(feet, root)
        desired = self.ref.qpos[self.actuated_qpos]
        wrench = data.qfrc_bias[:6].copy()
        wrench[:3] += self.mass * (
            np.array([self.forward_position_gain, 17, 45]) * (root - data.qpos[:3])
            + np.array([4, 6, 9]) * (velocity - data.qvel[:3])
        )
        inertia = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, data, inertia)
        wrench[3:] += inertia[3:6, 3:6] @ (-100 * data.qpos[4:7] * np.sign(data.qpos[3]) - 8 * data.qvel[3:6])
        support = self._support_torque(
            data, (phase < self.cycle.stance_fraction) | (data.time < self.cycle.start), wrench
        )
        dofs = self.actuated_dofs
        torque = (data.qfrc_bias - data.qfrc_spring - support)[dofs] + model.dof_damping[dofs] * (
            desired - previous
        ) / dt
        ctrl = np.clip(desired + torque / model.actuator_gainprm[:, 0], *model.actuator_ctrlrange.T)
        residual = ctrl - self.key.ctrl
        span = np.where(
            residual < 0, self.key.ctrl - model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1] - self.key.ctrl
        )
        return np.asarray(np.divide(residual, span, out=np.zeros_like(ctrl), where=span > 0))


def measure_walk(trace: EpisodeTrace, clearance: np.ndarray, window_start: float, debounce_seconds: float) -> dict:
    """Summarize a completed walk after startup. Touchdown requires a measurable load on the foot."""
    if trace.terminated:
        return {"completed": False, "reason": trace.termination_reason, "seconds": trace.length * trace.dt}
    window = slice(round(window_start / trace.dt), None)
    root = trace.root_pos[window]
    loaded = trace.leg_floor_mean[window] > 0.01 * trace.body_weight_n
    down = np.stack([debounce(column, steps_for(debounce_seconds, trace.dt)) for column in loaded.T], axis=1)
    strikes = {leg: rising_edges(down[:, i]) for i, leg in enumerate(trace.labels)}
    events = sorted((int(t), leg) for leg, times in strikes.items() for t in times)
    phases = []
    strides = []
    for side in ("r", "l"):
        hind, fore = strikes["r" + side], strikes["f" + side]
        for start, end in zip(hind[:-1], hind[1:], strict=True):
            same_side = fore[(start < fore) & (fore < end)]
            phases.extend((same_side - start) / (end - start))
            strides.append((end - start) * trace.dt)
    angles = np.rad2deg(yaw_pitch_roll(trace.root_quat[window]))
    assert trace.leg_slip_speed is not None and trace.site_pos is not None
    return {
        "completed": True,
        "measurement_seconds": len(root) * trace.dt,
        "forward_speed_mps": (root[-1, 0] - root[0, 0]) / ((len(root) - 1) * trace.dt),
        "duty_factor": dict(zip(trace.labels, down.mean(axis=0), strict=True)),
        "touchdowns": [leg for _, leg in events],
        "median_stride_seconds": float(np.median(strides)) if strides else None,
        "median_ipsilateral_phase": float(np.median(phases)) if phases else None,
        "minimum_loaded_feet": int(loaded.sum(axis=1).min()),
        "airborne_substeps": int(trace.feet_airborne_substeps[window].sum()),
        "nonleg_peak_load_n": float(trace.nonleg_floor_max[window].max()),
        "peak_off_sole_step_mean_n": float(trace.offsupport_floor_mean[window].max()),
        "median_stride_length_m": {
            leg: float(np.median(np.diff(trace.foot_pos[window, i, 0][strikes[leg]])))
            if len(strikes[leg]) > 1
            else None
            for i, leg in enumerate(trace.labels)
        },
        "peak_swing_clearance_m": dict(zip(FEET, clearance[window].max(axis=0), strict=True)),
        "loaded_slip_p95_mps": float(
            np.nanpercentile(np.where(loaded[:, None, :], trace.leg_slip_speed[window], np.nan), 95)
        ),
        "root_bob_m": float(np.ptp(root[:, 2])),
        "max_abs_pitch_roll_deg": np.abs(angles[:, 1:]).max(axis=0).tolist(),
        "head_tail_tip_vertical_range_m": np.ptp(trace.site_pos[window, :, 2], axis=0).tolist(),
    }
