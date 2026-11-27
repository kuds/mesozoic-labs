"""Replay and measure a straight, level-ground, four-beat walk.

Run from the repository root with
`python -m environments.dibothrosuchus_elaphros.scripts.reference_walk`.
This deterministic actuator reference tests whether the plant can perform a
plausible slow walk. Timing and lateral weight transfer are modeling choices.
"""

from __future__ import annotations

import argparse
import json

import mujoco
import numpy as np

from environments.dibothrosuchus_elaphros.anatomical.env import DibothrosuchusElaphrosAnatomicalEnv
from environments.shared.gait.morphology import Morphology
from environments.shared.gait.recorder import EpisodeTrace, SubstepContactRecorder
from environments.shared.gait.reference_walk import FEET, QuadrupedReferenceWalk, WalkCycle, measure_walk
from environments.shared.model_id import ModelId

MODEL = ModelId("dibothrosuchus_elaphros", "anatomical")
PERIOD = 0.80
START = 0.50
WINDOW_START = START + 3 * PERIOD


class ReferenceWalk(QuadrupedReferenceWalk):
    """Left hind, left fore, right hind, right fore, with alternating weight transfer."""

    # Use forward velocity feedback. A position error accumulated from many
    # short real strides would gradually push one forefoot off the ground.
    forward_position_gain = 0.0
    cycle = WalkCycle(
        period=PERIOD,
        stride=0.14,
        stance_fraction=0.70,
        ipsilateral_phase=0.28,
        lift=0.018,
        sway=0.017,
        # 0.08 until physics r4: with the 20 g snout_snap box gone the lighter
        # head moves 0.4% of the weight off the forefeet, and from about the
        # tenth stride the right forefoot then loaded on the same control step
        # as the left hind foot. 0.04, 0.05, 0.06 and 0.07 all keep the four
        # beats apart over the regression test's 20 strides (0.10 does not);
        # 0.06 has a passing value on each side.
        sway_phase=0.06,
        start=START,
        lowering=0.007,
    )

    def __init__(self, env: DibothrosuchusElaphrosAnatomicalEnv):
        super().__init__(env)
        # Local +Z is the long axis of each distal digit capsule. Measure
        # the forward distance from the central pad to the end of digit III.
        digits = np.array([env.model.geom(f"{leg}_digit_3").id for leg in FEET])
        tip_x = self.ref.geom_xpos[digits, 0] + abs(self.ref.geom_xmat[digits, 2]) * env.model.geom_size[digits, 1]
        self.toe_lever = tip_x - self.homes[:, 0]

    def targets(self, time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        feet, root, velocity, phase = super().targets(time)
        stance = self.cycle.stance_fraction
        # Lift the heel about digit III over the last quarter of stance.
        # Ease the pad back to level by mid-swing before placing it again.
        rise = np.clip((phase - 0.75 * stance) / (0.25 * stance), 0, 1)
        fall = np.clip((phase - stance) / (0.5 * (1 - stance)), 0, 1)
        roll = rise**2 * (3 - 2 * rise) - fall**2 * (3 - 2 * fall)
        self.sole_pitch = np.deg2rad(10) * roll * min(1.0, time / START)
        feet[:, 2] += self.toe_lever * np.sin(self.sole_pitch)
        feet[:, 0] += self.toe_lever * (1 - np.cos(self.sole_pitch))
        return feet, root, velocity, phase


def run_walk(seed: int = 0, reset_noise: float = 0.0, strides: int = 9) -> tuple[EpisodeTrace, np.ndarray]:
    """Advance a freely moving animal and measure the lowest point of each entire support foot."""
    if strides < 5:
        raise ValueError("The walk needs three startup strides and at least two measured strides.")
    # The progress terms measure along the spawn's heading, +x at the home keyframe.
    env = DibothrosuchusElaphrosAnatomicalEnv(reset_noise_scale=reset_noise)
    try:
        env.max_episode_steps = round((START + strides * PERIOD) / env.dt)
        morph = Morphology.from_env(env, MODEL)
        recorder = SubstepContactRecorder(env, morph, balance=True)
        recorder.record_slip().record_sites([env.snout_tip_site_id, env.tail_tip_site_id]).attach()
        env.reset(seed=seed)
        walk = ReferenceWalk(env)
        clearance = []
        while True:
            action = walk.action(env.data, env.dt)
            _, reward, terminated, truncated, info = env.step(action)
            recorder.end_step(action, reward)
            clearance.append(
                [
                    min(
                        mujoco.mj_geomDistance(env.model, env.data, g, env.floor_geom_id, 1.0, None)
                        for g in foot.support_geoms
                    )
                    for foot in morph.feet
                ]
            )
            if terminated or truncated:
                trace = recorder.end_episode(
                    terminated=terminated, truncated=truncated, termination_reason=info.get("termination_reason", "")
                )
                return trace, np.array(clearance)
    finally:
        env.close()


def measure(trace: EpisodeTrace, clearance: np.ndarray) -> dict:
    """Measure steady strides from contact forces and actual foot poses."""
    report = measure_walk(trace, clearance, WINDOW_START, 0.025)
    if report["completed"]:
        window = slice(round(WINDOW_START / trace.dt), None)
        loaded = trace.leg_floor_mean[window] > 0.01 * trace.body_weight_n
        # The recorder defines positive pitch as toe up. Quantify heel rise
        # on a *loaded* foot rather than accepting a freely pitched swing foot.
        pitch = np.where(loaded, -trace.sole_pitch_deg[window], np.nan)
        report["loaded_heel_up_pitch_p95_deg"] = dict(
            zip(trace.labels, np.nanpercentile(pitch, 95, axis=0), strict=True)
        )
        # Compare the combined digit impulse with all floor load on each leg.
        leg_impulse = trace.leg_floor_mean[window].sum(axis=0) * trace.dt
        support_impulse = trace.support_floor_mean[window].sum(axis=0) * trace.dt
        digit_share = {}
        for leg, total in zip(trace.labels, leg_impulse, strict=True):
            digits = [i for i, name in enumerate(trace.support_geom_names) if name.startswith(f"{leg}_digit_")]
            digit_share[leg] = float(support_impulse[digits].sum() / total) if total > 0 else None
        report["digital_load_fraction"] = digit_share
        feet_y = trace.foot_pos[window, :, 1]
        report["minimum_track_width_m"] = {
            group: float(np.min(feet_y[:, trace.labels.index(left)] - feet_y[:, trace.labels.index(right)]))
            for group, left, right in (("fore", "fl", "fr"), ("hind", "rl", "rr"))
        }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reset-noise", type=float, default=0.0)
    parser.add_argument("--strides", type=int, default=9, help="Total strides, including three for startup")
    args = parser.parse_args()
    print(json.dumps(measure(*run_walk(args.seed, args.reset_noise, args.strides)), indent=2))
