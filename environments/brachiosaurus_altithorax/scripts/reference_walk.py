"""Replay a slow, straight, four-beat reference walk through the Gymnasium environment.

Run from the repository root with
`python -m environments.brachiosaurus_altithorax.scripts.reference_walk`.
The controller sends normalized actuator actions. Its separate IK workspace can
move freely while the live torso and feet move only through MuJoCo dynamics.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from environments.brachiosaurus_altithorax.anatomical.env import BrachiosaurusAltithoraxAnatomicalEnv
from environments.shared.gait.morphology import Morphology
from environments.shared.gait.recorder import EpisodeTrace, SubstepContactRecorder
from environments.shared.gait.reference_walk import FEET, QuadrupedReferenceWalk, WalkCycle, measure_walk
from environments.shared.model_id import ModelId

MODEL = ModelId("brachiosaurus_altithorax", "anatomical")
# Lateral sequence with diagonal couplets. Left hind starts the stride,
# its forefoot follows at phase 0.35 and the opposite pair follows half a cycle later.
TOUCHDOWN_PHASE = np.array([0.85, 0.35, 0.50, 0.00])
PERIOD = 1.40  # seconds
STRIDE = 0.45  # model metres
STANCE_FRACTION = 0.68
START = 0.80  # shift weight before the first step
WINDOW_START = START + 3 * PERIOD


class ReferenceWalk(QuadrupedReferenceWalk):
    cycle = WalkCycle(
        period=PERIOD,
        stride=STRIDE,
        stance_fraction=STANCE_FRACTION,
        ipsilateral_phase=float(TOUCHDOWN_PHASE[1]),
        lift=0.050,
        sway=0.025,
        sway_phase=0.26,
        start=START,
    )


def run_walk(seed: int = 0, reset_noise: float = 0.0) -> tuple[EpisodeTrace, np.ndarray, np.ndarray]:
    """Measure nine strides plus startup using actual floor contacts at every physics substep."""
    env = BrachiosaurusAltithoraxAnatomicalEnv(reset_noise_scale=reset_noise)
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
    report = measure_walk(trace, clearance, WINDOW_START, 0.05)
    if report["completed"]:
        window = slice(round(WINDOW_START / trace.dt), None)
        loaded = trace.leg_floor_mean[window] > 0.01 * trace.body_weight_n
        report["median_loaded_flexion_deg"] = {
            leg: float(np.median(flexion[window, i][loaded[:, trace.labels.index(leg)]])) for i, leg in enumerate(FEET)
        }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reset-noise", type=float, default=0.0)
    args = parser.parse_args()
    print(json.dumps(measure(*run_walk(args.seed, args.reset_noise)), indent=2))
