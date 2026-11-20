"""Replay a slow, straight, four-beat reference walk through the Gymnasium environment.

Run from the repository root with
`python -m environments.brachiosaurus_altithorax.scripts.reference_walk`.
The controller sends normalized actuator actions. Its separate IK workspace can
move freely while the live torso and feet move only through MuJoCo dynamics.
"""

from __future__ import annotations

import argparse
import json

import mujoco
import numpy as np

from environments.brachiosaurus_altithorax.anatomical.env import BrachiosaurusAltithoraxAnatomicalEnv
from environments.shared.gait.events import debounce, rising_edges, steps_for
from environments.shared.gait.morphology import Morphology
from environments.shared.gait.recorder import EpisodeTrace, SubstepContactRecorder
from environments.shared.gait.stance_track import yaw_pitch_roll
from environments.shared.model_id import ModelId

MODEL = ModelId("brachiosaurus_altithorax", "anatomical")
FEET = ("fr", "fl", "rr", "rl")
# Lateral sequence with diagonal couplets. Left hind starts the stride,
# its forefoot follows at phase 0.35 and the opposite pair follows half a cycle later.
TOUCHDOWN_PHASE = np.array([0.85, 0.35, 0.50, 0.00])
PERIOD = 1.40  # seconds
STRIDE = 0.45  # model metres
STANCE_FRACTION = 0.68
START = 0.80  # shift weight before the first step
WINDOW_START = START + 3 * PERIOD


class ReferenceWalk:
    """Track ground-fixed stance feet and distribute body weight across the supporting legs."""

    def __init__(self, env: BrachiosaurusAltithoraxAnatomicalEnv):
        model = env.model
        self.model = model
        self.ref = mujoco.MjData(model)
        self.key = model.key("home")
        mujoco.mj_resetDataKeyframe(model, self.ref, self.key.id)
        mujoco.mj_forward(model, self.ref)
        self.feet = np.array([model.geom(f"{leg}_foot_geom").id for leg in FEET])
        self.homes = self.ref.geom_xpos[self.feet].copy()
        self.homes[:, 2] = model.geom_size[self.feet, 2] - 0.0005  # level soles, soft contact
        joints = model.actuator_trnid[:, 0]
        self.actuated_qpos = model.jnt_qposadr[joints]
        self.actuated_dofs = model.jnt_dofadr[joints]
        self.legs = [
            np.array([model.joint(f"{leg}_{joint}").id for joint in ("hip_pitch", "hip_roll", "knee", "ankle", "toe")])
            for leg in FEET
        ]
        self.mass = model.body_subtreemass[env.torso_id]

    def targets(self, time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cycles = max(0.0, (time - START) / PERIOD)
        lap, phase = np.divmod(cycles - TOUCHDOWN_PHASE, 1.0)
        swing = np.clip((phase - STANCE_FRACTION) / (1 - STANCE_FRACTION), 0, 1)
        touchdown = lap + TOUCHDOWN_PHASE
        # Place each foot ahead of the torso by half its stance excursion. The
        # first swing starts where that foot stood at reset.
        landing_x = STRIDE * (touchdown + STANCE_FRACTION / 2)
        before = np.where(touchdown > 0, landing_x, 0)
        after = np.where(touchdown + 1 > 0, landing_x + STRIDE, 0)
        feet = self.homes.copy()
        feet[:, 0] += before + (after - before) * swing**3 * (10 - 15 * swing + 6 * swing**2)
        feet[:, 2] += 0.050 * 16 * (swing * (1 - swing)) ** 2

        ramp = min(1.0, time / START)
        angle = 2 * np.pi * (cycles - 0.26)
        root = self.key.qpos[:3] + [STRIDE * cycles, 0.025 * np.sin(angle) * ramp, 0.0]
        if time < START:
            velocity = np.array([0, 0.025 * np.sin(angle) / START, 0.0])
        else:
            velocity = np.array([STRIDE / PERIOD, 0.025 * np.cos(angle) * 2 * np.pi / PERIOD, 0])
        return feet, root, velocity, phase

    def _place_legs(self, feet: np.ndarray, root: np.ndarray) -> None:
        """Two warm-started damped IK iterations per tick, with level soles and a weak posture preference."""
        model, ref = self.model, self.ref
        ref.qpos[:3] = root
        for _ in range(2):
            mujoco.mj_kinematics(model, ref)
            mujoco.mj_comPos(model, ref)
            for geom, target, joints in zip(self.feet, feet, self.legs, strict=True):
                dofs, q = model.jnt_dofadr[joints], model.jnt_qposadr[joints]
                linear = np.zeros((3, model.nv))
                angular = np.zeros_like(linear)
                mujoco.mj_jacGeom(model, ref, linear, angular, geom)
                pitch = np.arctan2(-ref.geom_xmat[geom, 6], ref.geom_xmat[geom, 0])
                jacobian = np.vstack((linear[:, dofs], 0.3 * angular[1, dofs], 0.03 * np.eye(len(q))))
                error = np.r_[target - ref.geom_xpos[geom], -0.3 * pitch, 0.03 * (self.key.qpos[q] - ref.qpos[q])]
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
            np.array([5, 17, 45]) * (root - data.qpos[:3]) + np.array([4, 6, 9]) * (velocity - data.qvel[:3])
        )
        inertia = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, data, inertia)
        wrench[3:] += inertia[3:6, 3:6] @ (-100 * data.qpos[4:7] * np.sign(data.qpos[3]) - 8 * data.qvel[3:6])
        support = self._support_torque(data, (phase < STANCE_FRACTION) | (data.time < START), wrench)
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


def run_walk(seed: int = 0, reset_noise: float = 0.0) -> tuple[EpisodeTrace, np.ndarray, np.ndarray]:
    """Measure nine strides plus startup using actual floor contacts at every physics substep."""
    env = BrachiosaurusAltithoraxAnatomicalEnv(
        reset_noise_scale=reset_noise,
        food_distance_range=(50, 50),
        food_lateral_range=(0, 0),
        food_approach_weight=0,
        food_reach_bonus=0,
    )
    try:
        env.max_episode_steps = round((START + 9 * PERIOD) / env.dt)
        recorder = SubstepContactRecorder(env, Morphology.from_env(env, MODEL), balance=True)
        recorder.record_slip().record_sites([env.head_tip_site_id, env.tail_tip_site_id]).attach()
        env.reset(seed=seed)
        walk = ReferenceWalk(env)
        clearance = []
        knee_q = [env.model.jnt_qposadr[env.model.joint(f"{leg}_knee").id] for leg in FEET]
        flexion = []
        while True:
            action = walk.action(env.data, env.dt)
            _, reward, terminated, truncated, info = env.step(action)
            recorder.end_step(action, reward)
            flexion.append(-np.rad2deg(env.data.qpos[knee_q]))
            ids = walk.feet
            # Support function of an oriented ellipsoid along the world vertical.
            vertical = env.data.geom_xmat[ids].reshape(-1, 3, 3)[:, 2, :]
            clearance.append(env.data.geom_xpos[ids, 2] - np.linalg.norm(vertical * env.model.geom_size[ids], axis=1))
            if terminated or truncated:
                trace = recorder.end_episode(
                    terminated=terminated, truncated=truncated, termination_reason=info.get("termination_reason", "")
                )
                return trace, np.array(clearance), np.array(flexion)
    finally:
        env.close()


def measure(trace: EpisodeTrace, clearance: np.ndarray, flexion: np.ndarray) -> dict:
    """Summarize six strides after startup. Touchdown requires a measurable load on the foot."""
    if trace.terminated:
        return {"completed": False, "reason": trace.termination_reason, "seconds": trace.length * trace.dt}
    window = slice(round(WINDOW_START / trace.dt), None)
    root = trace.root_pos[window]
    loaded = trace.leg_floor_mean[window] > 0.01 * trace.body_weight_n
    down = np.stack([debounce(column, steps_for(0.05, trace.dt)) for column in loaded.T], axis=1)
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
        "median_loaded_flexion_deg": {
            leg: float(np.median(flexion[window, i][loaded[:, trace.labels.index(leg)]])) for i, leg in enumerate(FEET)
        },
        "peak_swing_clearance_m": dict(zip(FEET, clearance[window].max(axis=0), strict=True)),
        "loaded_slip_p95_mps": float(
            np.nanpercentile(np.where(loaded[:, None, :], trace.leg_slip_speed[window], np.nan), 95)
        ),
        "root_bob_m": float(np.ptp(root[:, 2])),
        "max_abs_pitch_roll_deg": np.abs(angles[:, 1:]).max(axis=0).tolist(),
        "head_tail_tip_vertical_range_m": np.ptp(trace.site_pos[window, :, 2], axis=0).tolist(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reset-noise", type=float, default=0.0)
    args = parser.parse_args()
    print(json.dumps(measure(*run_walk(args.seed, args.reset_noise)), indent=2))
