"""Measure a free standing and walking trial from solved ground contacts, in SI units."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from environments.shared.gait.events import steps_for
from environments.shared.gait.stance_track import convex_hull, signed_margin, yaw_pitch_roll

from .build import MODEL_PATH
from .walk import WalkController

# Measurement order (not footfall order).
FEET = ("lf", "rf", "lh", "rh")
LATERAL_SEQUENCE = ("rh", "rf", "lh", "lf")
LOAD_FRACTION = 0.01


@dataclass
class Sample:
    time: float
    root: np.ndarray
    attitude: np.ndarray
    feet: np.ndarray
    load: np.ndarray
    clearance: np.ndarray
    skin_clearance: np.ndarray
    visible_clearance: np.ndarray
    slip: np.ndarray
    com_margin: float
    nonfoot_load: float
    nonfoot_clearance: float
    landmarks: np.ndarray
    bend_deg: np.ndarray
    motor_limit_fraction: float
    joint_limit_excess: float


class FloorProbe:
    """Measure actual contact velocity, including the rotation of the foot about its ankle."""

    def __init__(self, model):
        self.model = model
        self.ground = model.geom("ground").id
        self.root = model.body("torso").id
        self.bodies = [model.body(f"{foot}_foot").id for foot in FEET]
        self.upper = [model.body(f"{foot}_upper").id for foot in FEET]
        self.lower = [model.body(f"{foot}_lower").id for foot in FEET]

        collides = (model.geom_contype & model.geom_conaffinity[self.ground] != 0) | (
            model.geom_conaffinity & model.geom_contype[self.ground] != 0
        )
        self.soles = [np.flatnonzero((model.geom_bodyid == body) & collides) for body in self.bodies]
        if any(not len(sole) for sole in self.soles):
            raise ValueError("Each foot needs at least one ground collider")
        self.foot_index = {int(geom): i for i, sole in enumerate(self.soles) for geom in sole}
        self.visible_feet = [np.flatnonzero((model.geom_bodyid == body) & ~collides) for body in self.bodies]
        self.nonfoot = np.flatnonzero(~np.isin(model.geom_bodyid, [0, *self.bodies]))
        self.landmarks = [model.site(name).id for name in ("snout", "tail_tip")]
        # MuJoCo skins are skinned vertices, separate from visual/physical geoms. Read
        # the deformed render mesh without creating an OpenGL context. Vertices bound
        # predominantly to a foot belong to its cushion. All other vertices are body/leg.
        self.scene = mujoco.MjvScene(model, maxgeom=model.ngeom + 100)
        self.view_options = mujoco.MjvOption()
        self.camera = mujoco.MjvCamera()
        self.foot_vertices = []
        for body in self.bodies:
            (bone,) = np.flatnonzero(model.skin_bonebodyid == body)
            skin = np.searchsorted(model.skin_boneadr, bone, side="right") - 1
            first = model.skin_bonevertadr[bone]
            span = slice(first, first + model.skin_bonevertnum[bone])
            ids = model.skin_bonevertid[span][model.skin_bonevertweight[span] > 0.5]
            self.foot_vertices.append(ids + model.skin_vertadr[skin])
        self.nonfoot_vertices = np.setdiff1d(np.arange(model.nskinvert), np.concatenate(self.foot_vertices))
        self.hinges = np.flatnonzero((model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE) & model.jnt_limited.astype(bool))
        self.weight = mujoco.mj_getTotalmass(model) * abs(model.opt.gravity[2])
        self.jac = np.empty((3, model.nv))
        self.wrench = np.empty(6)

    def clearance(self, data, geoms):
        """Lowest surface, including visual geometry that does not generate solver contacts."""
        # A geom cannot reach below its bounding sphere. Stop exact distance queries once
        # the remaining lower bounds exceed the best distance already found.
        bounds = data.geom_xpos[geoms, 2] - self.model.geom_rbound[geoms]
        lowest = 5.0
        for index in np.argsort(bounds):
            if bounds[index] >= lowest:
                break
            distance = mujoco.mj_geomDistance(self.model, data, self.ground, int(geoms[index]), 5.0, None)
            lowest = min(lowest, distance)
        return lowest

    def sample(self, data):
        model = self.model
        support = []
        load = np.zeros(len(FEET))
        weighted_slip = np.zeros(len(FEET))
        nonfoot_load = 0.0
        for index, contact in enumerate(data.contact):
            first, second = contact.geom
            if self.ground not in (first, second):
                continue
            other = int(second if first == self.ground else first)
            mujoco.mj_contactForce(model, data, index, self.wrench)
            normal = max(0.0, self.wrench[0])
            if other not in self.foot_index:
                nonfoot_load += normal
                continue
            foot = self.foot_index[other]
            load[foot] += normal
            if normal > 1.0:
                support.append(contact.pos[:2].copy())
            mujoco.mj_jac(model, data, self.jac, None, contact.pos, self.bodies[foot])
            # Ground is stationary and level, so the point's world XY velocity is its slip.
            weighted_slip[foot] += normal * np.linalg.norm(self.jac[:2] @ data.qvel)
        mujoco.mjv_updateScene(
            model, data, self.view_options, None, self.camera, mujoco.mjtCatBit.mjCAT_DYNAMIC, self.scene
        )
        vertex_z = self.scene.skinvert.reshape(-1, 3)[:, 2]
        skin_clearance = np.array([vertex_z[vertices].min() for vertices in self.foot_vertices])
        visible = np.minimum(skin_clearance, [self.clearance(data, geoms) for geoms in self.visible_feet])
        clearance = np.minimum(visible, [self.clearance(data, geoms) for geoms in self.soles])
        margin = signed_margin(convex_hull(np.asarray(support).reshape(-1, 2)), data.subtree_com[self.root, :2])
        # Signed included flexion in the limb plane; zero is straight, negative is hyperextension.
        proximal = data.xpos[self.lower] - data.xpos[self.upper]
        distal = data.xpos[self.bodies] - data.xpos[self.lower]
        hinge_axis = data.xmat[self.lower].reshape(-1, 3, 3)[:, :, 1]
        sine = np.sum(np.cross(proximal, distal) * hinge_axis, axis=1)
        cosine = np.sum(proximal * distal, axis=1)
        bend = np.rad2deg(np.arctan2(sine, cosine)) * np.array([-1, -1, 1, 1])
        q = data.qpos[model.jnt_qposadr[self.hinges]]
        low, high = model.jnt_range[self.hinges].T
        limits = np.where(data.ctrl >= 0, model.actuator_ctrlrange[:, 1], -model.actuator_ctrlrange[:, 0])
        return Sample(
            float(data.time),
            data.xpos[self.root].copy(),
            data.xquat[self.root].copy(),
            data.xpos[self.bodies].copy(),
            load / self.weight,
            np.asarray(clearance),
            skin_clearance,
            visible,
            np.divide(weighted_slip, load, out=np.zeros_like(load), where=load > 0),
            margin,
            nonfoot_load / self.weight,
            min(self.clearance(data, self.nonfoot), float(vertex_z[self.nonfoot_vertices].min())),
            data.site_xpos[self.landmarks, 2].copy(),
            bend,
            float(np.max(np.abs(data.ctrl) / limits)),
            float(np.max(np.maximum(low - q, q - high), initial=0.0)),
        )


def trial(model: mujoco.MjModel, *, walking: bool, seconds: float, roll_deg: float = 0.0):
    """Start each trial from the model's neutral state, optionally releasing it from a small roll."""
    data = mujoco.MjData(model)
    probe = FloorProbe(model)
    root = model.joint("root")
    if root.type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("Validation requires a free root")
    start = int(root.qposadr[0])
    if roll_deg:
        angle = np.deg2rad(roll_deg) / 2
        quat = np.array([np.cos(angle), np.sin(angle), 0.0, 0.0])
        initial = data.qpos[start + 3 : start + 7].copy()
        mujoco.mju_mulQuat(data.qpos[start + 3 : start + 7], initial, quat)
        # Release the tilted animal with its lowest pad touching the floor.
        # Rotating about the pelvis without this lift would spawn two feet underground.
        mujoco.mj_forward(model, data)
        data.qpos[start + 2] -= min(probe.clearance(data, sole) for sole in probe.soles)
    mujoco.mj_forward(model, data)
    controller = WalkController(model)
    measured = mujoco.MjData(model)
    samples = []
    interval = steps_for(0.01, model.opt.timestep)
    for step in range(steps_for(seconds, model.opt.timestep)):
        controller.step(data, walking=walking)
        if step % interval == 0:
            # mj_step leaves positions and contacts from the preceding integration step.
            # Re-evaluate at these qpos/qvel/ctrl so force, clearance and slip share one instant.
            mujoco.mj_copyData(measured, model, data)
            mujoco.mj_forward(model, measured)
            samples.append(probe.sample(measured))
        mujoco.mj_step(model, data)
    if np.any(data.warning.number) or not np.isfinite(data.qpos).all():
        raise RuntimeError(f"MuJoCo reported a simulation warning or divergence: {data.warning.number}")
    return samples


def _per_foot(values):
    return {foot: float(value) for foot, value in zip(FEET, values, strict=True)}


def touchdowns(load, clearance):
    """A strike needs both prior measurable flight and a subsequent load above 1% BW.

    Lightly loaded pads can cross the force threshold several times during one stance.
    Re-arm only when the lowest surface of the entire foot clears the floor by 5 mm.
    """
    armed = False
    events = []
    for index, (force, gap) in enumerate(zip(load, clearance, strict=True)):
        if gap > 0.005:
            armed = True
        elif armed and force > LOAD_FRACTION:
            events.append(index)
            armed = False
    return np.asarray(events, dtype=int)


def summarize(samples: list[Sample], *, walking: bool, start: float):
    frames = [s for s in samples if s.time >= start - 1e-8]
    if len(frames) < 2:
        raise ValueError("Measurement window needs at least two samples")
    t = np.array([s.time for s in frames])
    root = np.array([s.root for s in frames])
    angles = np.rad2deg(yaw_pitch_roll(np.array([s.attitude for s in frames])))
    all_angles = np.rad2deg(yaw_pitch_roll(np.array([s.attitude for s in samples])))
    feet = np.array([s.feet for s in frames])
    load = np.array([s.load for s in frames])
    raw_down = load > LOAD_FRACTION
    margin = np.array([s.com_margin for s in frames])
    margins = margin[np.isfinite(margin)]
    body_delta = root[-1] - root[0]
    travel_time = t[-1] - t[0]
    landmarks = np.array([s.landmarks for s in frames])
    # Coordinates of the feet in the transverse axis of the body (left positive).
    yaw = np.deg2rad(angles[:, 0])
    lateral = np.column_stack((-np.sin(yaw), np.cos(yaw)))
    foot_y = np.einsum("tfi,ti->tf", feet[:, :, :2] - root[:, None, :2], lateral)
    total = load.sum()
    summary = {
        "window_s": [float(t[0]), float(t[-1])],
        "forward_distance_m": float(body_delta[0]),
        "forward_speed_mps": float(body_delta[0] / travel_time),
        "lateral_range_m": float(np.ptp(root[:, 1])),
        "root_bob_m": float(np.ptp(root[:, 2])),
        "max_abs_pitch_roll_deg": np.max(np.abs(angles[:, 1:]), axis=0).tolist(),
        "whole_trial_max_abs_pitch_roll_deg": np.max(np.abs(all_angles[:, 1:]), axis=0).tolist(),
        "max_abs_yaw_deg": float(np.max(np.abs(angles[:, 0]))),
        "peak_motor_limit_fraction": float(max(s.motor_limit_fraction for s in frames)),
        "whole_trial_peak_motor_limit_fraction": float(max(s.motor_limit_fraction for s in samples)),
        "motor_saturated_sample_fraction": float(np.mean([s.motor_limit_fraction >= 0.9999 for s in frames])),
        "whole_trial_max_joint_limit_excess_deg": float(np.rad2deg(max(s.joint_limit_excess for s in samples))),
        "mean_total_support_bw": float(load.sum(axis=1).mean()),
        "mean_foot_share": _per_foot(load.sum(axis=0) / total if total else np.zeros(4)),
        "hind_load_fraction": float(load[:, 2:].sum() / total) if total else 0.0,
        "min_loaded_feet": int(raw_down.sum(axis=1).min()),
        "support_count_fraction": {str(n): float((raw_down.sum(axis=1) == n).mean()) for n in range(5)},
        "com_support_margin_min_m": float(margins.min()) if margins.size else None,
        "com_inside_loaded_hull_fraction": float((margin >= 0).mean()),
        "support_hull_defined_fraction": float(np.isfinite(margin).mean()),
        "whole_trial_nonfoot_peak_bw": float(max(s.nonfoot_load for s in samples)),
        "steady_min_visible_sole_height_m": _per_foot(np.min([s.visible_clearance for s in frames], axis=0)),
        "steady_min_foot_surface_height_m": _per_foot(np.min([s.clearance for s in frames], axis=0)),
        "steady_min_pad_skin_height_m": _per_foot(np.min([s.skin_clearance for s in frames], axis=0)),
        "whole_trial_min_pad_skin_height_m": _per_foot(np.min([s.skin_clearance for s in samples], axis=0)),
        "whole_trial_nonfoot_clearance_m": float(min(s.nonfoot_clearance for s in samples)),
        "min_track_width_fore_hind_m": np.min(foot_y[:, ::2] - foot_y[:, 1::2], axis=0).tolist(),
        "snout_tail_min_height_m": landmarks.min(axis=0).tolist(),
        "snout_tail_vertical_range_m": np.ptp(landmarks, axis=0).tolist(),
    }
    if not walking:
        summary["max_foot_drift_m"] = float(np.linalg.norm(feet[:, :, :2] - feet[0, :, :2], axis=2).max())
        return summary

    clearance = np.array([s.clearance for s in frames])
    slip = np.array([s.slip for s in frames])
    strikes = [touchdowns(load[:, i], clearance[:, i]) for i in range(len(FEET))]
    events = sorted((int(index), FEET[i]) for i, times in enumerate(strikes) for index in times)
    labels = [foot for _, foot in events]
    summary["touchdowns"] = labels
    summary["lateral_sequence"] = bool(labels) and all(
        b == LATERAL_SEQUENCE[(LATERAL_SEQUENCE.index(a) + 1) % 4] for a, b in zip(labels, labels[1:])
    )
    intervals = np.diff([t[k] for k, _ in events])
    summary["touchdown_interval_range_s"] = [float(intervals.min()), float(intervals.max())] if intervals.size else None
    same_side = []
    for hind, fore in ((2, 0), (3, 1)):
        for a, b in zip(strikes[hind][:-1], strikes[hind][1:], strict=True):
            following = strikes[fore][(strikes[fore] > a) & (strikes[fore] < b)]
            same_side.extend((following - a) / (b - a))
    summary["hind_to_same_side_fore_phase_range"] = (
        [float(min(same_side)), float(max(same_side))] if same_side else None
    )
    summary["feet"] = {}
    bend = np.array([s.bend_deg for s in frames])
    for i, (foot, strikes_i) in enumerate(zip(FEET, strikes, strict=True)):
        duties, loaded_duties, peaks = [], [], []
        for a, b in zip(strikes_i[:-1], strikes_i[1:], strict=True):
            # Track first/last floor contact separately from materially weight-bearing time.
            duties.append(float((load[a:b, i] > 0.0001).mean()))
            loaded_duties.append(float(raw_down[a:b, i].mean()))
            peaks.append(float(clearance[a:b, i].max()))
        loaded_slip = slip[:, i][raw_down[:, i]]
        summary["feet"][foot] = {
            "touchdowns": len(strikes_i),
            "flexion_range_deg": [float(bend[:, i].min()), float(bend[:, i].max())],
            "loaded_flexion_median_deg": float(np.median(bend[raw_down[:, i], i])) if loaded_slip.size else None,
            "duty_factor_range": [min(duties), max(duties)] if duties else None,
            "weight_bearing_duty_range": [min(loaded_duties), max(loaded_duties)] if loaded_duties else None,
            "median_stride_length_m": float(np.median(np.diff(feet[strikes_i, i, 0]))) if duties else None,
            "median_stride_period_s": float(np.median(np.diff(t[strikes_i]))) if duties else None,
            "min_peak_swing_clearance_m": min(peaks) if peaks else None,
            "loaded_slip_p95_mps": float(np.percentile(loaded_slip, 95)) if loaded_slip.size else None,
        }
    return summary


def check(standing: dict, walk: dict) -> dict[str, bool]:
    """Engineering regression tolerances for a straight, slow walk on the model's level floor."""
    stats = list(walk["feet"].values())
    return {
        "stand_all_four_bear_weight": standing["min_loaded_feet"] == 4,
        "stand_weight_supported": abs(standing["mean_total_support_bw"] - 1.0) < 0.02,
        "stand_hind_biased": 0.55 < standing["hind_load_fraction"] < 0.95,
        "stand_com_inside_support": (standing["com_support_margin_min_m"] or 0.0) > 0.025,
        "stand_feet_hold_position": standing["max_foot_drift_m"] < 0.025,
        "upright_throughout": max(
            *standing["whole_trial_max_abs_pitch_roll_deg"], *walk["whole_trial_max_abs_pitch_roll_deg"]
        )
        < 15.0,
        "floor_load_only_on_feet": max(standing["whole_trial_nonfoot_peak_bw"], walk["whole_trial_nonfoot_peak_bw"])
        < 0.001,
        "body_and_legs_clear_floor": min(
            standing["whole_trial_nonfoot_clearance_m"], walk["whole_trial_nonfoot_clearance_m"]
        )
        > 0.01,
        "visible_pads_clear_floor_within_5mm": min(
            *standing["steady_min_visible_sole_height_m"].values(), *walk["steady_min_visible_sole_height_m"].values()
        )
        > -0.005,
        "joints_stay_within_soft_limit_tolerance": max(
            standing["whole_trial_max_joint_limit_excess_deg"], walk["whole_trial_max_joint_limit_excess_deg"]
        )
        < 1.0,
        "walk_motors_avoid_sustained_saturation": walk["motor_saturated_sample_fraction"] < 0.01,
        "walk_makes_forward_progress": walk["forward_speed_mps"] > 0.08,
        "walk_stays_on_heading": walk["max_abs_yaw_deg"] < 12 and walk["lateral_range_m"] < 0.35,
        "walk_feet_do_not_cross": min(walk["min_track_width_fore_hind_m"]) > 0,
        "walk_no_flight_or_single_support": walk["min_loaded_feet"] >= 2,
        "walk_repeats_lateral_sequence": walk["lateral_sequence"] and min(f["touchdowns"] for f in stats) >= 3,
        "walk_distinct_four_beats": bool(
            walk["touchdown_interval_range_s"] and walk["touchdown_interval_range_s"][0] > 0.05
        ),
        "walk_every_foot_advances": all((f["median_stride_length_m"] or 0) > 0.12 for f in stats),
        "walk_every_swing_clears_floor": all((f["min_peak_swing_clearance_m"] or 0) > 0.01 for f in stats),
        "walk_majority_stance_with_swing": all(
            f["duty_factor_range"] and 0.5 < f["duty_factor_range"][0] <= f["duty_factor_range"][1] < 0.95
            for f in stats
        ),
        "walk_loaded_slip_bounded": all(
            f["loaded_slip_p95_mps"] is not None and f["loaded_slip_p95_mps"] < 0.08 for f in stats
        ),
    }


def run_validation(model_path=MODEL_PATH, *, walk_seconds: float = 20, roll_deg: float = 0.0):
    model = mujoco.MjModel.from_xml_path(str(model_path))
    stand_samples = trial(model, walking=False, seconds=5, roll_deg=roll_deg)
    walk_samples = trial(model, walking=True, seconds=walk_seconds, roll_deg=roll_deg)
    standing = summarize(stand_samples, walking=False, start=2.0)
    walk = summarize(walk_samples, walking=True, start=4.0)
    checks = check(standing, walk)
    return {
        "mujoco_version": mujoco.__version__,
        "mass_kg": mujoco.mj_getTotalmass(model),
        "initial_roll_deg": roll_deg,
        "standing": standing,
        "walk": walk,
        "checks": checks,
        "passed": all(checks.values()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--walk-seconds", type=float, default=20.0)
    parser.add_argument("--roll-deg", type=float, default=0.0, help="Additional initial root roll in both trials.")
    args = parser.parse_args()
    if not np.isfinite(args.walk_seconds) or args.walk_seconds < 10 or not np.isfinite(args.roll_deg):
        parser.error("Use a finite initial roll and at least 10 seconds of walking.")
    result = run_validation(args.model, walk_seconds=args.walk_seconds, roll_deg=args.roll_deg)
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if result["passed"] else 1)
