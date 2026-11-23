"""Simulate standing, four-beat walking or heading commands on the 60 cm Triceratops."""

import argparse
import json
import math
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import heading as steering
import mujoco as mj
import numpy as np
import trajectory
from gentle import GentleTurn
from scipy.ndimage import gaussian_filter1d

HERE = Path(__file__).resolve().parent
BUILD = HERE.parents[1]
for dependency in (BUILD / "sim", BUILD / "walk", BUILD / "rl", BUILD.parents[1]):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

import crawl  # noqa: E402
import feet60_study  # noqa: E402
import walk60  # noqa: E402


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False, default=lambda x: x.item()) + "\n")


def leg_indices(m):
    legs = ["left_fore", "right_fore", "left_hind", "right_hind"]
    axes = ["roll", "pitch", "flexion"]
    return legs, [m.actuator(f"{leg}_{ax}").id for leg in legs for ax in axes]


def static(path):
    m = mj.MjModel.from_xml_path(str(path))
    d = mj.MjData(m)
    mj.mj_resetDataKeyframe(m, d, 0)
    legs, ids = leg_indices(m)
    adr = [m.sensor(leg + "_normal_load").adr[0] for leg in legs]
    body = m.body("chassis").id
    vals = []
    for i in range(5000):
        mj.mj_step(m, d)
        if i >= 4000:
            vals.append([*d.sensordata[adr], *d.actuator_force[ids], *d.subtree_com[body], *d.xpos[body]])
    y = np.mean(vals, axis=0)
    return dict(
        mass_kg=float(m.body_mass.sum()),
        mean_foot_N=y[:4].tolist(),
        fore_share=float(y[:2].sum() / y[:4].sum()),
        torque_Nm=y[4:16].tolist(),
        com_xyz_m=y[16:19].tolist(),
        body_xyz_m=y[19:].tolist(),
    )


def leg_metrics(load, tau, rpm, tilt, w):
    return dict(
        mean_foot_N=load.mean(0).tolist(),
        peak_foot_N=load.max(0).tolist(),
        p95_foot_N=np.percentile(load, 95, axis=0).tolist(),
        max_single_bodyweights=float(load.max() / w),
        roll_pitch_abs_max_deg=np.max(np.abs(tilt), axis=0).tolist(),
        fore_share=float(np.sum(load[:, :2]) / np.sum(load)),
        torque_absmax_Nm=np.max(abs(tau), axis=0).tolist(),
        torque_rms_Nm=np.sqrt(np.mean(tau * tau, axis=0)).tolist(),
        rpm_absmax=np.max(abs(rpm), axis=0).tolist(),
    )


def walk(a, path):
    m = mj.MjModel.from_xml_path(str(path))
    motion, P, T = trajectory.exhibit_plan(crawl, m, a.gait)
    _, tplan, _, phases = motion
    legs, ids = leg_indices(m)
    body = m.body("chassis").id
    neck = [m.actuator(j).id for j in ("neck_yaw", "neck_pitch")]
    soles = [m.geom(j + "_sole").id for j in legs]
    loads = [m.sensor(j + "_normal_load").adr[0] for j in legs]
    slider = feet60_study.Slider(m)
    frames = []
    n = 0
    ns = math.ceil((tplan[-1] + 0.1) / m.opt.timestep)
    signal = np.zeros((ns, 32))
    w = float(m.body_mass.sum() * 9.81)

    def hook(m, d):
        nonlocal n
        if n % 10 == 0:
            d.ctrl[neck] = trajectory.glance(d.time / T, crawl)
        slider(m, d)
        R = d.xmat[body].reshape(3, 3)
        sig = signal[n]
        sig[:4] = d.sensordata[loads]
        sig[4:16] = d.actuator_force[ids]
        sig[16:28] = d.actuator_velocity[ids] * 60 / (2 * np.pi)
        sig[28:30] = np.degrees([np.arctan2(R[2, 1], R[2, 2]), np.arcsin(-np.clip(R[2, 0], -1, 1))])
        sig[30] = d.time
        if a.record and n % 5 == 0:
            frames.append(
                (
                    d.time,
                    d.qpos.copy(),
                    d.sensordata[loads].copy(),
                    d.xpos[body].copy(),
                    d.geom_xpos[soles].copy(),
                    d.subtree_com[body].copy(),
                )
            )
        n += 1

    r = crawl.run(path, a.seed, P, step_hook=hook, motion_plan=motion)
    rules, info = crawl.judge(r)
    t = np.array(slider.t)
    hist = np.array(slider.hist)
    windows = {i: [[0.0, None]] for i in range(4)}
    for ph in phases:
        i = ph["leg_index"]
        windows[i][-1][1] = ph["swing_start"]
        windows[i].append([ph["swing_end"], None])
    slide = []
    events = []
    for i, span in windows.items():
        span[-1][1] = t[-1]
        for start, end in span:
            a0 = np.searchsorted(t, start + 0.1)
            b = np.searchsorted(t, end, side="right")
            if b > a0:
                vv = np.linalg.norm(hist[a0:b, i] - hist[a0, i], axis=1)
                slide.append(float(vv.max()))
                events.append([i, float(start), float(end), float(t[a0 + np.argmax(vv)])])
    rules["C5c"] = max(slide) <= 0.010
    lt = np.array(r["log"]["t"])
    base = np.array(r["log"]["base"])
    lo, hi = phases[2]["swing_start"], phases[-1]["swing_end"]
    ss = (lt >= lo) & (lt <= hi)
    sig = signal[:n]
    ss0 = (sig[:, 30] >= lo) & (sig[:, 30] <= hi)
    load = sig[ss0, :4]
    result = dict(
        rules=rules,
        steady_seconds=[lo, hi],
        min_support_margin_mm=1000 * info["min_margin_measured_m"],
        worst_slide_event=events[int(np.argmax(slide))],
        max_slide_mm=1000 * max(slide),
        saturated_pct=100 * info["saturated_fraction"],
        pass_all=all(rules[k] for k in ("C1", "C2", "C3", "C4", "C5c", "C6")),
        drift_mm=1000 * info["drift_m"],
        yaw_change_deg=info["yaw_change_deg"],
        speed_mps=info["speed_mps"],
        travel_m=info["travel_m"],
        side_span_mm=1000 * np.ptp(np.percentile(base[ss, 1], [1, 99])),
        below_5mm_s_pct=100 * np.mean(np.gradient(gaussian_filter1d(base[:, 0], 5), lt)[ss] < 0.005),
        min_height_ratio=info["min_height_ratio"],
        three_or_four_support_pct=100 * np.mean(np.sum(load > 0.05 * w, axis=1) >= 3),
        **leg_metrics(load, sig[ss0, 4:16], sig[ss0, 16:28], sig[ss0, 28:30], w),
    )
    if a.record:
        keys = ("t", "qpos", "foot_load_N", "body_xyz_m", "feet_xyz_m", "com_xyz_m")
        np.savez_compressed(a.out.with_suffix(".npz"), **dict(zip(keys, map(np.array, zip(*frames)))))
    return result


def force_transients(time, force, legs, timestep, csv_path):
    """Inspect 1 kHz raw and 10 ms mean maxima across all four feet in the scoring interval."""
    count = int(round(0.010 / timestep))
    if count < 1 or not np.isclose(count * timestep, 0.010):
        raise ValueError("The 10 ms window requires an integer number of physics steps")
    if len(time) < count or not np.allclose(np.diff(time), timestep, rtol=1e-8, atol=1e-12):
        raise ValueError("The force trace must have a continuous 10 ms interval")
    means = np.stack([np.convolve(channel, np.ones(count) / count, mode="valid") for channel in force.T], axis=1)
    raw_row, raw_leg = np.unravel_index(np.argmax(force), force.shape)
    mean_start, mean_leg = np.unravel_index(np.argmax(means), means.shape)
    mean_end = mean_start + count - 1
    if csv_path is not None:
        indices = np.arange(len(time))
        context = int(round(0.025 / timestep))
        keep = (abs(indices - raw_row) <= context) | (
            (indices >= mean_start - context) & (indices <= mean_end + context)
        )
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            csv_path,
            np.column_stack((time[keep], force[keep])),
            delimiter=",",
            comments="",
            fmt="%.9f",
            header="t_s,LF_N,RF_N,LH_N,RH_N",
        )
    return {
        "sample_period_ms": timestep * 1000,
        "raw_peak_N": force[raw_row, raw_leg],
        "raw_peak_foot": legs[raw_leg],
        "raw_peak_t_s": time[raw_row],
        "raw_peak_all_feet_N": force[raw_row].tolist(),
        "mean_10ms_peak_N": means[mean_start, mean_leg],
        "mean_10ms_peak_foot": legs[mean_leg],
        "mean_10ms_first_sample_s": time[mean_start],
        "mean_10ms_last_sample_s": time[mean_end],
        "mean_sample_count": count,
    }


def course(a, path):
    speed = steering.PROGRAMS[a.program][0][2]
    duration = steering.DURATION[a.program]
    e = walk60.TrikeEnv(
        walk60.plant_config(
            model_path=str(path),
            command_mode="heading",
            episode_seconds=duration,
            domain_randomization=a.random,
            observation_noise=True,
            cmd_yaw_gain=a.yaw_gain,
        )
    )
    obs, reset = e.reset(seed=a.seed, options={"delay_steps": 10, "command_target": (0.0, speed)})
    p = walk60.WalkParams(duty=0.70)
    turn = None
    if a.turn_profile != "original":
        p = p.replace(lift_m=0.028) if speed > 0 else p.replace(lift_m=0.012, duty=0.64)
        turn = GentleTurn(e, gyro_bias_dps=a.gyro_bias_dps, stock_yaw=a.turn_profile == "shaped-stock-yaw")
    c = walk60.TrotController(e, p)
    prog = [(t, d, speed) for t, d, _ in steering.PROGRAMS[a.program]]
    m = e.model
    body = m.body("chassis").id
    legs, ids = leg_indices(m)
    lad = [m.sensor(j + "_normal_load").adr[0] for j in legs]
    soles = [m.geom(j + "_sole").id for j in legs]
    signal = np.zeros((math.ceil((duration + e.dt) / m.opt.timestep), 32))
    n = 0

    def stats(e):
        nonlocal n
        d = e.data
        R = d.xmat[body].reshape(3, 3)
        sig = signal[n]
        sig[:4] = d.sensordata[lad]
        sig[4:16] = d.actuator_force[ids]
        sig[16:28] = d.actuator_velocity[ids] * 60 / (2 * np.pi)
        sig[28:30] = np.degrees([np.arctan2(R[2, 1], R[2, 2]), np.arcsin(-np.clip(R[2, 0], -1, 1))])
        sig[30] = d.time
        n += 1

    e.substep_hook = stats
    seg = 0
    headings = e._heading
    buf = []

    def frame():
        elapsed = e._step_count * e.dt
        _, direction, speed = prog[seg]
        return (
            elapsed,
            *steering.observe(e, direction, speed),
            e.data.sensordata[lad].copy(),
            e.data.subtree_com[body].copy(),
            e.data.geom_xpos[soles].copy(),
        )

    while True:
        elapsed = e._step_count * e.dt
        if seg + 1 < len(prog) and elapsed >= prog[seg + 1][0] - 1e-8:
            seg += 1
            obs = e.set_command_target(headings + np.deg2rad(prog[seg][1]), speed)
        buf.append(frame())
        if turn is not None:
            obs = turn.apply(obs, prog[seg][1], speed, c)
        obs, _, term, trunc, info = e.step(c.act(obs))
        if term or trunc:
            buf.append(frame())
            break
    keys = (
        "t",
        "xyz",
        "yaw_rad",
        "command_deg",
        "command_mps",
        "roll_pitch_rad",
        "slip_mps",
        "qpos",
        "foot_load_N",
        "com_xyz_m",
        "feet_xyz_m",
    )
    tr = dict(zip(keys, map(np.array, zip(*buf))))
    if a.record:
        np.savez_compressed(
            a.out.with_suffix(".npz"),
            **{k: v[::2] for k, v in tr.items()},
            initial_heading_rad=headings,
            physics_t_s=signal[:n, 30],
            physics_foot_load_N=signal[:n, :4],
        )
    result = steering.evaluate(tr, a.program, headings)
    h = signal[:n]
    h = h[h[:, 30] > 2.0] if n > 2000 else h
    w = m.body_mass.sum() * 9.81
    result.update(
        terminated=bool(term),
        termination=info["termination"],
        initial_heading_rad=float(headings),
        friction=reset.get("friction"),
        saturation_pct_by_axis=(
            100 * np.mean(np.abs(h[:, 4:16]) >= 0.99 * m.actuator_forcerange[ids, 1], axis=0)
        ).tolist(),
        **leg_metrics(h[:, :4], h[:, 4:16], h[:, 16:28], h[:, 28:30], w),
    )
    enough_samples = len(h) >= round(0.010 / m.opt.timestep)
    result["force_transients"] = (
        force_transients(h[:, 30], h[:, :4], legs, m.opt.timestep, a.force_trace) if enough_samples else None
    )
    result["time_in_air_pct"] = 100 * np.mean(np.sum(h[:, :4] > 2, axis=1) == 0)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program", choices=("stand", "walk", "course", "six-headings", "pivot", "pivot-right"))
    parser.add_argument("--model", type=Path, default=HERE.parent / "models/refined/physics.xml")
    parser.add_argument(
        "--gait",
        choices=("reference", "refined"),
        default="refined",
        help="Four-beat reference used by walk. Steering uses its separate 70%%-stance controller.",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--record", action="store_true", help="Save time, pose and load arrays alongside the JSON for moving routines."
    )
    parser.add_argument(
        "--stiff", action="store_true", help="Use the stiffer floor contact setting on a four-beat walk."
    )
    parser.add_argument("--random", action="store_true", help="Randomize the moving-course or pivot plant and floor.")
    parser.add_argument(
        "--yaw-gain", type=float, default=1.5, help="Heading adapter gain per second for course and pivots."
    )
    parser.add_argument("--out", type=Path, required=True, help="Output prefix for JSON and optional NPZ.")
    parser.add_argument(
        "--force-trace",
        type=Path,
        metavar="CSV",
        help="Write all four normal forces at physics rate around the maximum instantaneous load and 10 ms average for a steering trial.",
    )
    parser.add_argument(
        "--turn-profile",
        choices=("original", "gentle", "shaped-stock-yaw"),
        default="original",
        help="Steering controller for course and pivots.",
    )
    parser.add_argument(
        "--gyro-bias-dps",
        type=float,
        default=0.0,
        help="Inject a constant bias in the gentle relative-yaw estimator (degrees/s).",
    )
    a = parser.parse_args()
    if a.gyro_bias_dps and a.turn_profile != "gentle":
        parser.error("--gyro-bias-dps applies to the gentle turn profile")
    if a.turn_profile != "original" and a.program in ("stand", "walk"):
        parser.error("--turn-profile applies only to steering")
    if a.force_trace and a.program in ("stand", "walk"):
        parser.error("--force-trace applies to steering programs")
    if a.stiff and a.program != "walk":
        parser.error("--stiff applies to walk")
    if a.random and a.program in ("stand", "walk"):
        parser.error("--random applies to heading trials")
    if a.seed is None:
        a.seed = 11 if a.program == "walk" else 101
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        path = a.model
        if a.stiff:
            tree = ET.parse(path)
            if any("file" in element.attrib for element in tree.iter()):
                parser.error("--stiff requires the self-contained physics.xml, which has no relative assets")
            tree.find("option").set("impratio", "10")
            path = Path(temp) / "physics.xml"
            tree.write(path)
        if a.program == "stand":
            result = static(path)
        elif a.program == "walk":
            result = walk(a, path)
        else:
            result = course(a, path)
    result["settings"] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    write_json(a.out.with_suffix(".json"), result)


if __name__ == "__main__":
    main()
