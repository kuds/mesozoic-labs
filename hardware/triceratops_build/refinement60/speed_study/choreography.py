"""Replay a preplanned six-heading four-beat show and score actual contacts, yaw and motion."""

import argparse
import copy
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "motion"))
import run  # noqa: E402

crawl, mj, trajectory, steering = run.crawl, run.mj, run.trajectory, run.steering
MODELS = {128: ROOT / "models/refined/physics.xml", 160: ROOT / "models/mid160/physics.xml"}
SCHEDULE = steering.PROGRAMS["six-headings"]
DURATION = steering.DURATION["six-headings"]
DEFAULT_CADENCE = 1.25
TRACTION_METHOD = (
    "C5c integrates each sole's horizontal velocity at contacting ground points at 1 kHz, then checks peak "
    "displacement from 100 ms after each scheduled touchdown to the next lift across the whole course; limit "
    "10 mm. C6 uses the existing 100 Hz saturation fraction (any actuator at >=99% of its own limit); "
    "limit 1% of the run. All-gates requires the six actual one-second heading holds plus C1/C3/C4/C5c/C6."
)


def cast_path(t, body, feet, phases, params):
    """Bend the straight centerline and keep each stance anchor fixed in world coordinates."""
    yaw = np.zeros(len(t))
    for previous, current in zip(SCHEDULE, SCHEDULE[1:]):
        progress = np.clip((t - current[0]) / 5.0, 0, 1)
        # Trapezoidal yaw rate, with a half-second ramp on entry and exit.
        e = 0.1
        fraction = np.where(
            progress < e,
            progress**2 / (2 * e),
            np.where(progress > 1 - e, 1 - e - (1 - progress) ** 2 / (2 * e), progress - e / 2),
        ) / (1 - e)
        yaw += np.deg2rad(current[1] - previous[1]) * fraction
    sine, cosine = np.sin(yaw), np.cos(yaw)
    rot = np.stack([cosine, -sine, sine, cosine], axis=1).reshape(-1, 2, 2)
    mid = (yaw[1:] + yaw[:-1]) / 2
    center = np.zeros_like(body)
    center[1:] = np.cumsum(np.diff(body[:, :1], axis=0) * np.stack([np.cos(mid), np.sin(mid)], axis=1), axis=0)
    feet_world = feet.copy()
    beat = params["shift_s"] + params["swing_s"]
    for leg in range(4):
        previous = feet[0, leg, :2].copy()
        last_end = 0
        for phase in (ph for ph in phases if ph["leg_index"] == leg):
            a, b = phase["swing_start"], phase["swing_end"]
            start, end = np.searchsorted(t, [a, b])
            ref = min(np.searchsorted(t, b + 1.6 * beat), len(t) - 1)
            # Place the sole for the middle of its upcoming stance. This uses the known show script.
            landing = center[ref] + rot[ref] @ (feet[end, leg, :2] - [body[ref, 0], 0])
            feet_world[last_end:start, leg, :2] = previous
            for i in range(start, end):
                v = (t[i] - a) / (b - a)
                w = crawl.min_jerk((v - params["lift_frac"] / 2) / (1 - params["lift_frac"]))
                feet_world[i, leg, :2] = previous + w * (landing - previous)
            previous, last_end = landing, end
        feet_world[last_end:, leg, :2] = previous
    torso = np.zeros_like(body)
    phase_index = 0
    for i, ti in enumerate(t):
        while phase_index + 1 < len(phases) and ti > phases[phase_index]["swing_end"]:
            phase_index += 1
        active = phases[phase_index]
        previous = phases[max(0, phase_index - 1)]
        w = (
            crawl.min_jerk((ti - previous["swing_end"]) / (active["swing_start"] - previous["swing_end"]))
            if phase_index
            else 1.0
        )
        for ph, blend in ((active, w), (previous, 1 - w)):
            others = [j for j in range(4) if j != ph["leg_index"]]
            triangle = feet[i, others, :2]
            weights = np.linalg.solve(np.vstack([triangle.T, np.ones(3)]), [*body[i], 1])
            torso[i] += blend * (weights @ feet_world[i, others, :2])
    feet_world[:, :, :2] = np.einsum("tij,tkj->tki", rot.transpose(0, 2, 1), feet_world[:, :, :2] - torso[:, None, :])
    return feet_world


def course_plan(model, cadence):
    kin = crawl.Kin(model)
    source = json.loads((ROOT / "motion/reference_plan.json").read_text())
    steps, params, shape = trajectory.refined_recipe(source, kin)
    factor = 0.8 / cadence
    while len(steps) * (params["swing_s"] + params["shift_s"]) * factor < DURATION + 7:
        for _ in range(4):
            previous, earlier = steps[-4], steps[-8]
            extra = copy.deepcopy(previous)
            extra["k"] += 4
            for label in ("from", "to", "body"):
                extra[label][0] += previous[label][0] - earlier[label][0]
            steps.append(extra)
    t, body, feet, phases = trajectory.exhibit_waypoints(crawl, kin, steps, params, shape)
    t *= factor
    for ph in phases:
        ph["swing_start"] *= factor
        ph["swing_end"] *= factor
    for key in ("shift_s", "swing_s", "stand_s"):
        params[key] *= factor
    local = cast_path(t, body, feet, phases, params)
    angles = {c: np.zeros(3) for c in crawl.CORNERS}
    targets = [np.zeros(12)]
    misses, max_error = 0, 0.0
    for i in range(1, np.searchsorted(t, DURATION + 0.02)):
        angles, ok, _ = kin.pose(0, 0, dict(zip(crawl.CORNERS, local[i])), angles)
        if not ok:
            misses += 1
            max_error = max(
                max_error,
                max(np.linalg.norm(local[i, j] - kin.d.xpos[kin.foot[c]]) for j, c in enumerate(crawl.CORNERS)),
            )
        targets.append(np.concatenate([angles[c] for c in crawl.CORNERS]))
    motion = (steps, t[: len(targets)], np.array(targets), [ph for ph in phases if ph["swing_end"] <= DURATION])
    return (
        motion,
        params,
        {"samples": len(targets), "clipped_samples": misses, "max_ankle_residual_mm": 1000 * max_error},
    )


def trial(task, traction=False):
    length, seed, stiff, output, cadence = task
    path = MODELS[length]
    stem = output / str(length) / f"{cadence:g}x-{seed}-{'stiff' if stiff else 'nominal'}"
    stem = stem.with_name(stem.name.replace(".", "p"))
    stem.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        if stiff:
            tree = ET.parse(path)
            tree.find("option").set("impratio", "10")
            path = Path(tmp) / "physics.xml"
            tree.write(path)
        model = mj.MjModel.from_xml_path(str(path))
        motion, params, ik = course_plan(model, cadence)
        sensors = [model.sensor(c + "_normal_load").adr[0] for c in crawl.CORNERS]
        neck = [model.actuator(j).id for j in ("neck_yaw", "neck_pitch")]
        slider = run.feet60_study.Slider(model) if traction else None
        n = 0
        samples = np.zeros((round(motion[1][-1] / model.opt.timestep), 5))
        poses, pose_times = [], []
        record = seed == 11 and not stiff

        def hook(m, data):
            nonlocal n
            if slider is not None:
                slider(m, data)
            if n % 10 == 0:
                data.ctrl[neck] = trajectory.glance(data.time / 0.8, crawl)
            samples[n, 0], samples[n, 1:] = data.time, data.sensordata[sensors]
            if record and n % 10 == 0:
                poses.append(data.qpos.copy())
                pose_times.append(data.time)
            n += 1

        raw = crawl.run(path, seed, params, step_hook=hook, motion_plan=motion)
    log = raw["log"]
    t = np.array(log["t"])
    yaw = np.deg2rad(log["yaw"])
    trace = dict(
        t=t,
        xyz=np.array(log["base"]),
        yaw_rad=yaw - yaw[0],
        slip_mps=np.full((len(t), 4), np.nan),
        roll_pitch_rad=np.zeros((len(t), 2)),
    )
    evaluation = steering.evaluate(trace, "six-headings", yaw[0])
    rules, info = crawl.judge(raw)
    load = np.array(log["load"])
    weak = []
    for ph in motion[3]:
        leg = ph["leg_index"]
        selected = load[(t >= ph["swing_start"]) & (t <= ph["swing_end"])]
        if np.mean(selected[:, leg] < 0.05 * raw["weight_n"]) < 0.8 or np.any(
            np.delete(selected, leg, axis=1) < 0.05 * raw["weight_n"]
        ):
            weak.append([ph["leg"], round(ph["swing_start"], 2)])
    # The comparison covers from the first changed command at 3 s through the last second at 40 s.
    moving = (samples[:, 0] >= 3) & (samples[:, 0] <= DURATION + 1e-8)
    f = samples[moving, 1:]
    segments = evaluation["segments"]
    # The joint sequence has a footfall cadence instead of an operator m/s request.
    for segment in segments:
        segment["requested_speed_mps"] = None
    scored = segments[1:]
    duration = [s["observed_until_s"] - s["start_s"] for s in scored]
    summary = dict(
        upper_mm=length,
        seed=seed,
        floor="stiff" if stiff else "nominal",
        cadence=cadence,
        ik=ik,
        holds_s=[s["first_1s_in_5deg_band_s"] for s in scored],
        final_heading_deg=[round(s["final_actual_heading_deg"], 2) for s in scored],
        commands_complete=len(scored) == len(SCHEDULE) - 1
        and all(s["completed"] and s["first_1s_in_5deg_band_s"] is not None for s in scored),
        measured_forward_mm_s=1000 * np.average([s["projected_mean_speed_mps"] for s in scored], weights=duration),
        max_tilt_deg=info["max_tilt_deg"],
        min_support_margin_mm=1000 * info["min_margin_measured_m"],
        upright=rules["C1"],
        strict_four_beat=rules["C4"],
        swing_count=len(motion[3]),
        failed_swings=weak,
        three_loaded_pct=100 * np.mean(np.sum(f >= 0.05 * raw["weight_n"], axis=1) >= 3),
        airborne_pct=100 * np.mean(np.max(f, axis=1) < 0.05 * raw["weight_n"]),
        peak_by_foot_N=f.max(0).tolist(),
        peak_10ms_N=run.force_transients(samples[moving, 0], f, crawl.CORNERS, 0.001, None)["mean_10ms_peak_N"],
    )
    if slider is not None:
        # End the last stance at any scheduled lift during the truncated final gait cycle.
        phases = list(motion[3])
        next_index = len(phases)
        next_start = phases[-1]["swing_end"] + params["shift_s"]
        if next_index < len(motion[0]) and next_start <= DURATION:
            leg = motion[0][next_index]["leg"]
            phases.append(
                {
                    "leg_index": crawl.CORNERS.index(leg),
                    "swing_start": next_start,
                    "swing_end": next_start + params["swing_s"],
                }
            )
        times = np.asarray(slider.t)
        stop = np.searchsorted(times, DURATION + 1e-8, side="right")
        slide, event = run.stance_slides(times[:stop], np.asarray(slider.hist)[:stop], phases)
        # C2 is specific to straight travel. The directional course is judged by the six actual yaw holds.
        gates = {
            "C1": rules["C1"],
            "six_headings": summary["commands_complete"],
            "C3": rules["C3"],
            "C4": rules["C4"],
            "C5c": slide <= 0.010,
            "C6": rules["C6"],
        }
        summary.update(
            max_planted_slide_mm=1000 * slide,
            worst_slide_event=event,
            actuator_saturation_pct=100 * info["saturated_fraction"],
            traction_gates=gates,
            all_gates=all(gates.values()),
        )
    if record:
        pt = np.array(pose_times)
        commands = np.array([next(a for start, a, _ in reversed(SCHEDULE) if start <= ti) for ti in pt])
        np.savez_compressed(
            stem.with_suffix(".npz"),
            t=pt,
            qpos=poses,
            command_deg=commands,
            yaw_rad=np.interp(pt, t, yaw - yaw[0]),
            physics_t_s=samples[:, 0],
            physics_foot_load_N=samples[:, 1:],
        )
    run.write_json(
        stem.with_suffix(".json"),
        {
            **summary,
            "segments": segments,
            "settings": {"program": "six-headings", "seed": seed, "walk_cadence": cadence},
        },
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "speed_study/choreography.json")
    parser.add_argument(
        "--traction",
        action="store_true",
        help="Run the exhibit planted-sole slide and actuator saturation gates in addition to heading/contact scoring.",
    )
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    parser.add_argument("--cadences", type=float, nargs="+", default=[DEFAULT_CADENCE])
    a = parser.parse_args()
    if any(not np.isfinite(c) or c <= 0 for c in a.cadences) or a.jobs < 1:
        parser.error("use positive finite cadence multipliers and worker count")
    cases = [
        (length, seed, stiff, a.records, cadence)
        for cadence in a.cadences
        for seed in a.seeds
        for stiff in (False, True)
        for length in MODELS
    ]
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        rows = list(pool.map(partial(trial, traction=a.traction), cases))
    run.write_json(
        a.out,
        {
            "method": "Preplanned native exhibit geometry at the listed cadence multipliers; target schedule is known in advance. No yaw measurements drive the joint trajectories. Triangles use the baseline barycentric trunk location. Body heading is scored from the floating chassis. Force/contact and mean projected speed cover 3 to 40 s; C4 and upright tests cover the full run.",
            "foot_order": crawl.CORNERS,
            **({"traction_method": TRACTION_METHOD} if a.traction else {}),
            "runs": rows,
        },
    )


if __name__ == "__main__":
    main()
