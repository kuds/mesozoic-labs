"""Measure a four-beat steering schedule against all six heading changes, using actual foot loads."""

import argparse
import hashlib
import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = {128: ROOT / "models/refined/physics.xml", 160: ROOT / "models/mid160/physics.xml"}


def ordered_support(track, schedule, weight, model_path):
    """Apply the straight walk's 10 mm triangle and three planted feet checks to complete turns' swings."""
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    legs = ("left_fore", "right_fore", "left_hind", "right_hind")
    ankles = [model.joint(leg + "_ankle").id for leg in legs]
    times, poses = track["t"], track["qpos"]
    loaded = track["foot_load_N"] >= 0.05 * weight
    centers = track["com_xyz_m"][:, :2]
    period, duty, stand = (schedule[k] for k in ("period_s", "duty", "stand_s"))
    minimum, good, swings = float("inf"), 0, 0
    for leg, name in enumerate(legs):
        offset = schedule["offsets"][name]
        others = [i for i in range(4) if i != leg]
        for cycle in range(-1, int(times[-1] / period) + 1):
            start = stand + (cycle + offset + duty) * period
            end = start + (1 - duty) * period
            if start < 3.0 or end > times[-1]:
                continue
            indices = np.flatnonzero((times >= start) & (times <= end))
            if not len(indices):
                continue
            swings += 1
            good += int((~loaded[indices, leg]).mean() >= 0.8 and loaded[indices][:, others].all())
            for i in indices:
                data.qpos[:] = poses[i]
                mujoco.mj_kinematics(model, data)
                tri = data.xanchor[np.array(ankles)[others], :2]
                edges = np.roll(tri, -1, axis=0) - tri
                rel = centers[i] - tri
                signed = edges[:, 0] * rel[:, 1] - edges[:, 1] * rel[:, 0]
                orient = np.sign(edges[0, 0] * edges[1, 1] - edges[0, 1] * edges[1, 0])
                minimum = min(minimum, float(np.min(orient * signed / np.linalg.norm(edges, axis=1))))
    return good, swings, round(1000 * minimum, 2) if swings else None


def measure(task, output, read_existing):
    length, speed, seed, randomized = task
    tag = f"{length}-{round(1000 * speed):03d}-{seed}-{'random' if randomized else 'nominal'}"
    prefix = output / tag
    cmd = [
        sys.executable,
        str(ROOT / "motion/run.py"),
        "six-headings",
        "--turn-profile",
        "gentle",
        "--steering-footfall",
        "four-beat",
        "--model",
        str(MODELS[length]),
        "--speed-mps",
        str(speed),
        "--seed",
        str(seed),
        "--record",
        "--out",
        str(prefix),
    ]
    if randomized:
        cmd.append("--random")
    if not read_existing:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    row = json.loads(prefix.with_suffix(".json").read_text())
    settings = row["settings"]
    if (settings["speed_mps"], settings["seed"], settings["random"], settings["steering_footfall"]) != (
        speed,
        seed,
        randomized,
        "four-beat",
    ) or Path(settings["model"]).parts[-2:] != MODELS[length].parts[-2:]:
        raise ValueError(f"Unexpected model or command settings in {tag}")
    segments = row["segments"][1:]
    with np.load(prefix.with_suffix(".npz")) as track:
        t = track["physics_t_s"]
        mask = t > 3.0
        force = track["physics_foot_load_N"][mask]
        # An unloaded foot and three load-bearing feet during the central half of the planned swing.
        # Read the simulator's actual posture/contact response, separate from the planned gait sequence.
        weight = max(row["peak_foot_N"]) / row["max_single_bodyweights"]
        loaded = force > 0.05 * weight
        p = row["footfall_schedule"]
        offsets = np.array([p["offsets"][s] for s in ("left_fore", "right_fore", "left_hind", "right_hind")])
        leg_phase = ((t[mask, None] - p["stand_s"]) / p["period_s"] - offsets) % 1.0
        swing = (leg_phase - p["duty"]) / (1.0 - p["duty"])
        middle = (swing >= 0.25) & (swing < 0.75)
        orderly = (~loaded & (loaded.sum(axis=1) == 3)[:, None] & middle).sum()
        correspondence = 100 * orderly / middle.sum() if middle.any() else None
        supported, swings, margin = ordered_support(track, p, weight, MODELS[length])
    if segments:
        elapsed = sum(s["observed_until_s"] - s["start_s"] for s in segments)
        projected = (
            sum(s["projected_mean_speed_mps"] * (s["observed_until_s"] - s["start_s"]) for s in segments) / elapsed
        )
    else:
        projected = None
    peak = row["force_transients"]
    return dict(
        length_mm=length,
        command_mps=speed,
        seed=seed,
        random=randomized,
        finished=not row["terminated"] and row["elapsed_s"] >= 40,
        elapsed_s=row["elapsed_s"],
        bearing_holds=[s["first_1s_in_5deg_band_s"] for s in segments],
        end_error_deg=[
            round(math.remainder(s["final_actual_heading_deg"] - s["requested_heading_deg"], 360), 2) for s in segments
        ],
        projected_speed_after_setup_mps=round(projected, 4) if projected is not None else None,
        minimum_ankle_triangle_margin_mm=margin,
        complete_swings=swings,
        swings_passing_ordered_contact=supported,
        peak_roll_pitch_abs_deg=[round(z, 2) for z in row["abs_peak_roll_pitch_deg"]],
        three_plus_support_pct=round(100 * float((loaded.sum(axis=1) >= 3).mean()), 2) if len(loaded) else None,
        scheduled_mid_swing_measured_pct=round(float(correspondence), 2) if correspondence is not None else None,
        loaded_slip_p95_mps=row["loaded_slip_after3s_p95_mps"],
        per_foot_peak_N=[round(z, 2) for z in row["peak_foot_N"]],
        peak_10ms_N=round(peak["mean_10ms_peak_N"], 2) if peak else None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speeds", type=float, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103])
    parser.add_argument(
        "--read-existing", action="store_true", help="Analyze existing JSON and full pose/force recordings."
    )
    parser.add_argument("--nominal-only", action="store_true")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.jobs < 1 or any(not math.isfinite(speed) or speed <= 0 for speed in args.speeds):
        parser.error("use positive worker and speed values")
    tags = {round(1000 * speed) for speed in args.speeds}
    if len(tags) != len(args.speeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("use unique seeds and speeds separated by at least one millimeter per second")
    args.records.mkdir(parents=True, exist_ok=True)
    tasks = product(MODELS, args.speeds, args.seeds, [False] if args.nominal_only else [False, True])
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        rows = list(pool.map(lambda t: measure(t, args.records, args.read_existing), tasks))
    data = dict(
        model_sha256={n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in MODELS.items()},
        foot_order=["LF", "RF", "LH", "RH"],
        schedule_deg=[15, 35, 10, -15, -35, 0],
        method="Noisy-gyro closed-loop six-heading course. Speed projects motion along each target after the 3 s setup. "
        "Contact fractions use actual 1 kHz normal forces exceeding 5% of modeled weight. Complete scheduled "
        "swings are also scored on the 50 Hz pose trace: the swing foot must be unloaded for at least 80% of "
        "samples and all other feet loaded throughout. The margin is signed horizontal COM distance into the "
        "intended stance ankle triangle. Raw and 10 ms forces follow the ordinary post-2 s steering scoring.",
        runs=rows,
    )
    args.out.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
