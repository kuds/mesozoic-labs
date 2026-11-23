"""Measure a free-body straight entry, left or right arc, and straight exit."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from examples.shared.footfalls import swing_events
from examples.shared.ground_contacts import support_margin
from examples.shared.turn_metrics import circular_path_metrics

from .courses import TURN_COURSES, foot_contacts
from .reference_walk import CONTROL_DT, FEET, MODEL_PATH, ReferenceWalk


def run_turn(course, *, straight_baseline=False, trace_path=None):
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    walk = ReferenceWalk(model)
    contacts = foot_contacts(model)
    steps = round(CONTROL_DT / model.opt.timestep)
    dt = steps * model.opt.timestep
    frames = round(course.duration_s / dt) + 1
    times = np.arange(frames) * dt
    positions = np.empty((frames, 3))
    yaw = np.empty(frames)
    tilt = np.empty(frames)
    soles = np.empty((frames, len(FEET), 3))
    loads = np.empty((frames, len(FEET)))
    gaps = np.empty_like(loads)
    slip = np.empty(frames)
    margins = np.empty(frames)
    clipped = np.empty(frames)
    other_load = np.empty(frames)

    for i, t in enumerate(times):
        positions[i] = data.xpos[walk.trunk]
        rot = data.xmat[walk.trunk]
        yaw[i] = np.arctan2(rot[3], rot[0])
        tilt[i] = np.degrees(np.arccos(np.clip(rot[8], -1, 1)))
        soles[i] = data.site_xpos[walk.sites]
        sample = contacts.measure(model, data)
        loads[i], gaps[i] = sample.loads, sample.clearance
        slip[i], other_load[i] = sample.slip, sample.other_load
        points = sample.contact_xy
        # The t=0 reset precedes contact settling. An unsupported sample in the
        # scored entry/turn/exit window fails the loaded-foot gate below.
        margins[i] = support_margin(points, data.subtree_com[walk.trunk, :2]) if points else -np.inf
        clipped[i] = np.mean(np.abs(data.ctrl[walk.motors]) > 0.999)
        if i < frames - 1:
            walk.update(data, curvature=0.0 if straight_baseline else course.curvature_at(t))
            mujoco.mj_step(model, data, nstep=steps)
            mujoco.mj_forward(model, data)
    assert np.isfinite(positions).all() and np.isfinite(yaw).all() and not np.any(data.warning.number)
    if trace_path is not None:
        np.savez_compressed(trace_path, time=times, trunk=positions, yaw=yaw, soles=soles, loads=loads, feet=FEET)

    entry = round(course.entry_s / dt)
    exit_at = round((course.entry_s + course.turn_s) / dt)
    arc = slice(entry, exit_at + 1)
    yaw = np.unwrap(yaw)
    path = circular_path_metrics(positions[arc, :2], yaw[arc], course.radius_m)
    # Use the same axis at both ends to distinguish leaving along the tangent
    # from returning toward the initial +X heading after releasing the turn.
    exit_delta = positions[-1, :2] - positions[exit_at, :2]
    exit_tangent = np.array([np.cos(yaw[exit_at]), np.sin(yaw[exit_at])])
    loaded = loads[arc] > 0.01 * walk.weight
    side = np.array([1, -1, 1, -1])
    normal = np.column_stack((-np.sin(yaw[arc]), np.cos(yaw[arc])))
    lateral = ((soles[arc, :, :2] - positions[arc, None, :2]) * normal[:, None, :]).sum(axis=2)
    strides = []
    all_events = []
    smallest_peaks = []
    shallow = []
    swings = []
    for i in range(len(FEET)):
        landings, peaks, dips = swing_events(loaded[:, i], gaps[arc, i])
        assert len(landings) >= 2, f"{FEET[i]} did not repeatedly clear the floor"
        strides.append(np.median(np.linalg.norm(np.diff(soles[arc, i, :2][landings], axis=0), axis=1)))
        all_events.extend((frame, i) for frame in landings)
        smallest_peaks.append(np.min(peaks))
        shallow.append(dips)
        swings.append(len(peaks))
    expected = [FEET.index(foot) for foot in ("left_hind", "left_fore", "right_hind", "right_fore")]
    ordered = [i for _, i in sorted(all_events)]
    follows = [b == expected[(expected.index(a) + 1) % len(FEET)] for a, b in zip(ordered, ordered[1:])]
    inside = np.sign(course.radius_m) == side
    counts = loaded.sum(axis=1)
    return {
        "radius_command_m": course.radius_m,
        "entry_s": course.entry_s,
        "turn_s": course.turn_s,
        "exit_s": course.exit_s,
        "path": path,
        "exit_heading_change_deg": np.degrees(yaw[-1] - yaw[exit_at]),
        "exit_forward_distance_m": exit_delta @ exit_tangent,
        "exit_cross_track_m": exit_delta @ [-exit_tangent[1], exit_tangent[0]],
        "complete_swings": dict(zip(FEET, swings)),
        "footfall_sequence_agreement": np.mean(follows),
        "threshold_dips_without_flight": int(sum(shallow)),
        "minimum_swing_peak_clearance_m": min(smallest_peaks),
        "minimum_same_side_sole_distance_m": (lateral * side).min(),
        "mean_inside_stride_m": np.mean(np.asarray(strides)[inside]),
        "mean_outside_stride_m": np.mean(np.asarray(strides)[~inside]),
        "minimum_loaded_feet": int(counts.min()),
        "three_or_four_loaded_fraction": np.mean(counts >= 3),
        "minimum_support_margin_m": margins[arc].min(),
        "com_in_support_fraction": np.mean(margins[arc] >= 0),
        "mean_load_weighted_pad_slip_m_s": slip[arc].mean(),
        "limb_motor_saturation_fraction": clipped[arc].mean(),
        "max_trunk_tilt_deg_whole_course": tilt.max(),
        "max_nonfoot_ground_load_n_whole_course": other_load.max(),
        "minimum_loaded_feet_through_exit": int((loads[entry:] > 0.01 * walk.weight).sum(axis=1).min()),
    }


def check_turn(result):
    p = result["path"]
    sign = np.sign(result["radius_command_m"])
    assert 80 < sign * p["heading_change_deg"] < 105, "Body missed the quarter turn"
    assert 80 < sign * p["swept_angle_deg"] < 105, "Position missed the quarter turn"
    assert p["rms_radial_error_m"] < 0.20 and p["max_radial_error_m"] < 0.40
    assert p["forward_distance_m"] > 7.2
    assert abs(result["exit_heading_change_deg"]) < 5 and abs(result["exit_cross_track_m"]) < 0.18
    assert result["exit_forward_distance_m"] > 2
    assert min(result["complete_swings"].values()) >= 12 and result["footfall_sequence_agreement"] > 0.97
    assert result["minimum_swing_peak_clearance_m"] > 0.018 and result["threshold_dips_without_flight"] <= 6
    assert result["minimum_same_side_sole_distance_m"] > 0.25
    assert result["mean_outside_stride_m"] > result["mean_inside_stride_m"] + 0.07
    assert result["minimum_loaded_feet_through_exit"] >= 2 and result["three_or_four_loaded_fraction"] > 0.98
    assert result["minimum_support_margin_m"] > -0.09
    assert result["mean_load_weighted_pad_slip_m_s"] < 0.03 and result["limb_motor_saturation_fraction"] < 0.075
    assert result["max_trunk_tilt_deg_whole_course"] < 3 and result["max_nonfoot_ground_load_n_whole_course"] < 1.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("left", "right", "both"), default="both")
    parser.add_argument("--baseline", action="store_true", help="Measure the original straight command on each turn")
    parser.add_argument("--trace-dir", type=Path, help="Save each trial's actual pose, sole centers and measured loads")
    args = parser.parse_args()
    if args.trace_dir:
        args.trace_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, course in TURN_COURSES.items():
        if args.case in ("both", name):
            trace = args.trace_dir / f"turn_{name}.npz" if args.trace_dir else None
            results[name] = run_turn(course, straight_baseline=args.baseline, trace_path=trace)
            if not args.baseline:
                check_turn(results[name])
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
