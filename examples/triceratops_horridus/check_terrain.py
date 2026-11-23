"""Check the adult's free-body straight walk on signed inclines and rounded uneven ground."""

import argparse
import json

import mujoco
import numpy as np

from examples.shared.footfalls import swing_events
from examples.shared.ground_contacts import support_margin

from .courses import COURSES, EXTENDED_COURSES, foot_contacts
from .reference_walk import CONTROL_DT, FEET, MODEL_PATH, PERIOD, ReferenceWalk


def inspect_course(course, *, baseline=False, cycles=None, path=MODEL_PATH, trace_path=None):
    """Run the same spawn, duration and physical contacts for each controller on a course."""
    if cycles is None:
        cycles = 15 if course.mounds else 12
    if cycles < 3:
        raise ValueError("At least three strides are needed to score a walking trial")
    model = course.model(path)
    data = mujoco.MjData(model)
    course.reset(model, data)
    if baseline:
        walk = ReferenceWalk(model)
    else:
        from .terrain_walk import TerrainFootPlacement

        walk = ReferenceWalk(model, terrain=TerrainFootPlacement(course))
    probe = foot_contacts(model, course.surface_names())
    sites = [model.site(f + "_sole").id for f in FEET]
    root = model.body("trunk").id
    weight = -model.opt.gravity[2] * model.body_mass.sum()
    load_threshold = 0.01 * weight
    heading = np.array([1.0, 0.0, np.tan(np.radians(course.pitch_deg))])
    heading /= np.linalg.norm(heading)
    neutral_height = data.xpos[root, 2] - course.height(*data.xpos[root, :2])
    # The static trial releases the animal from rest, independently of walking.
    mujoco.mj_step(model, data, nstep=round(5.0 / model.opt.timestep))
    mujoco.mj_forward(model, data)
    support = probe.measure(model, data)
    stance = {
        "loaded_feet": int(np.count_nonzero(support.loads > load_threshold)),
        "support_margin_m": support_margin(support.contact_xy, data.subtree_com[root, :2])
        if support.contact_xy
        else None,
    }
    course.reset(model, data)
    spawn_xy = data.xpos[root, :2].copy()
    steps = round(CONTROL_DT / model.opt.timestep)
    dt = steps * model.opt.timestep
    warmup = 0.8 + 2 * PERIOD
    records = []
    trace = []
    bad_contact = max_abs_drift = max_tilt = max_relief = 0.0
    minimum_separation = float("inf")
    min_height_fraction = 1.0
    mounds_loaded = set()
    early_stop = None
    for frame in range(round((warmup + cycles * PERIOD) / dt)):
        walk.update(data)
        mujoco.mj_step(model, data, nstep=steps)
        mujoco.mj_forward(model, data)
        if np.any(data.warning.number) or not np.isfinite(data.qvel).all():
            early_stop = "numerical warning"
            break
        sample = probe.measure(model, data)
        xyz = data.xpos[root]
        relative_height = (xyz[2] - course.height(*xyz[:2])) / neutral_height
        min_height_fraction = min(min_height_fraction, float(relative_height))
        tilt = np.degrees(np.arccos(np.clip(data.xmat[root, 8], -1, 1)))
        max_tilt = max(max_tilt, tilt)
        max_abs_drift = max(max_abs_drift, abs(xyz[1] - spawn_xy[1]))
        bad_contact = max(bad_contact, sample.other_load)
        # A projected sole on the animal's own side stays positive after applying the side sign.
        rot = data.xmat[root].reshape(3, 3)
        yaw = np.arctan2(rot[1, 0], rot[0, 0])
        lateral = np.array([-np.sin(yaw), np.cos(yaw)])
        foot_xy = data.site_xpos[sites, :2]
        if trace_path is not None:
            trace.append((data.time, data.qpos.copy(), data.site_xpos[sites].copy(), sample.loads, sample.clearance))
        separations = ((foot_xy - xyz[:2]) @ lateral) * [1, -1, 1, -1]
        minimum_separation = min(minimum_separation, separations.min())
        mounds_loaded.update((np.flatnonzero(sample.surface_loads[1:] > load_threshold) + 1).tolist())
        if course.mounds:
            max_relief = max(max_relief, course.height(*foot_xy.T).max())
        if not sample.contact_xy or relative_height < 0.6 or tilt > 45:
            early_stop = "unsupported or fallen"
            break
        if (frame + 1) * dt >= warmup - dt / 2:
            records.append(
                (
                    xyz.copy(),
                    sample.loads,
                    sample.clearance,
                    sample.slip,
                    support_margin(sample.contact_xy, data.subtree_com[root, :2]),
                    np.mean(np.abs(data.ctrl[walk.motors]) >= 0.999),
                )
            )
    if trace_path is not None and trace:
        time, poses, soles, loads, gaps = map(np.asarray, zip(*trace))
        np.savez_compressed(trace_path, time=time, qpos=poses, soles=soles, loads=loads, clearance=gaps, feet=FEET)
    if len(records) < 2:
        return {"passed": False, "reason": early_stop or "insufficient samples", "stance": stance}
    pos, loads, clearance, slip, margin, clipping = map(np.asarray, zip(*records))
    elapsed = dt * (len(records) - 1)
    loaded = loads > load_threshold
    supporting = loaded.sum(axis=1)
    feet = {}
    landings = []
    for i, foot in enumerate(FEET):
        events, peaks, shallow = swing_events(loaded[:, i], clearance[:, i])
        landings.extend((int(step), foot) for step in events)
        feet[foot] = {
            "swings": len(events),
            "shallow_reloads": shallow,
            "duty_factor": loaded[:, i].mean(),
            "minimum_swing_clearance_m": peaks.min() if len(peaks) else 0.0,
        }
    order = ("left_hind", "left_fore", "right_hind", "right_fore")
    sequence = [f for _, f in sorted(landings)]
    agreement = sum(b == order[(order.index(a) + 1) % 4] for a, b in zip(sequence, sequence[1:])) / max(
        1, len(sequence) - 1
    )
    distance = (pos[-1] - pos[0]) @ heading
    # Furthest exposed edge of the ellipsoid/plane intersection along world +X.
    exit_distance = None
    if course.mounds:
        edge = max(m.x + m.x_radius * np.sqrt(1 - (m.center[2] / m.z_radius) ** 2) for m in course.mounds)
        exit_distance = data.site_xpos[sites[2:], 0].min() - edge
    report = {
        "stance": stance,
        "scored_strides": cycles,
        "hind_soles_past_course_end_m": exit_distance,
        "duration_s": elapsed,
        "forward_progress_m": distance,
        "forward_speed_m_s": distance / elapsed,
        "elevation_change_m": pos[-1, 2] - pos[0, 2],
        "maximum_lateral_drift_m": max_abs_drift,
        "maximum_trunk_tilt_deg": max_tilt,
        "minimum_trunk_height_fraction": min_height_fraction,
        "minimum_signed_foot_side_distance_m": minimum_separation,
        "maximum_other_contact_force_N": bad_contact,
        "minimum_loaded_feet": int(supporting.min()),
        "three_or_four_loaded_fraction": (supporting >= 3).mean(),
        "minimum_com_margin_m": margin.min(),
        "com_inside_support_fraction": (margin >= 0).mean(),
        "mean_pad_slip_m_s": slip.mean(),
        "limb_motor_clipping_fraction": clipping.mean(),
        "minimum_pad_separation_m": clearance.min(),
        "contact_sequence_agreement": agreement,
        "loaded_mounds": sorted(mounds_loaded),
        "maximum_mound_height_under_soles_m": max_relief if course.mounds else None,
        "feet": feet,
    }
    # The courses challenge the level controller. Require sustained progress and true unloaded
    # steps over the surface, including contact with the raised ground on the uneven course.
    checks = {
        "completed": early_stop is None,
        "stance": stance["loaded_feet"] == 4 and stance["support_margin_m"] > 0.2,
        "progress": distance / elapsed > 0.20,
        "holds_line": max_abs_drift < 0.45,
        "balance": supporting.min() >= 2 and (supporting >= 3).mean() > 0.98 and margin.min() > -0.12,
        "upright": max_tilt < 15 and min_height_fraction > 0.8,
        "foot_placement": minimum_separation > 0.12 and bad_contact < 1.0,
        "sliding": slip.mean() < 0.04,
        "pad_penetration": clearance.min() > -0.01,
        "motor_clipping": clipping.mean() < 0.10,
        "sequence": agreement > 0.97 and all(cycles - 1 <= f["swings"] <= cycles + 1 for f in feet.values()),
        "swing_clearance": all(f["minimum_swing_clearance_m"] > 0.010 for f in feet.values()),
        "planted_reloads": all(f["shallow_reloads"] <= 5 for f in feet.values()),
        "foot_loading": all(0.55 < f["duty_factor"] < 0.93 for f in feet.values()),
        "raised_contacts": not course.mounds
        or len(mounds_loaded) >= (len(course.mounds) if cycles >= 15 else min(4, len(course.mounds), cycles - 1)),
        "exits_course": exit_distance is None or cycles < 15 or exit_distance > 0,
    }
    report["failed_checks"] = [k for k, v in checks.items() if not v]
    report["passed"] = not report["failed_checks"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    available = {**COURSES, **EXTENDED_COURSES}
    parser.add_argument(
        "--course",
        choices=available,
        action="append",
        help="Repeat to select courses. Defaults to the six standard routes.",
    )
    parser.add_argument(
        "--baseline", action="store_true", help="Run the original level-floor controller on these same courses"
    )
    parser.add_argument("--cycles", type=int, help="Override the 12 scored strides on slopes or 15 on uneven ground")
    parser.add_argument(
        "--trace", help="Save a single course’s 100 Hz times, poses, sole positions, loads and clearances"
    )
    args = parser.parse_args()
    if args.trace and len(args.course or COURSES) != 1:
        parser.error("--trace requires one --course")
    results = {
        name: inspect_course(available[name], baseline=args.baseline, cycles=args.cycles, trace_path=args.trace)
        for name in (args.course or COURSES)
    }
    print(json.dumps(results, indent=2))
    if not args.baseline and not all(row["passed"] for row in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
