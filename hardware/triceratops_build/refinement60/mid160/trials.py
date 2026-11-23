"""Run matched 60 cm exhibit walks, gentle pivots and six-heading moving courses."""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {128: ROOT / "models/refined/physics.xml", 160: ROOT / "models/mid160/physics.xml"}
FILM = {("walk", 11, False), ("pivot-right", 102, False), ("six-headings", 103, False)}
FEET = ["LF", "RF", "LH", "RH"]


def cases():
    yield "stand", 11, False
    # Run the three filmed trials first so their poses can be inspected immediately.
    yield from sorted(FILM)
    for seed in (11, 12, 13):
        if ("walk", seed, False) not in FILM:
            yield "walk", seed, False
    for seed in (101, 102, 103):
        for program, randomized in (
            ("pivot", False),
            ("pivot-right", False),
            ("six-headings", False),
            ("six-headings", True),
        ):
            if (program, seed, randomized) not in FILM:
                yield program, seed, randomized


def run_one(records, length, case):
    program, seed, randomized = case
    condition = "random" if randomized else "nominal"
    prefix = records / str(length) / f"{program}-{seed}-{condition}"
    cmd = [
        sys.executable,
        str(ROOT / "motion/run.py"),
        program,
        "--model",
        str(MODELS[length]),
        "--seed",
        str(seed),
        "--out",
        str(prefix),
    ]
    if case in FILM:
        cmd.append("--record")
    if program not in ("stand", "walk"):
        cmd.extend(["--turn-profile", "gentle"])
    if randomized:
        cmd.append("--random")
    subprocess.run(cmd, check=True)
    data = json.loads(prefix.with_suffix(".json").read_text())
    row = dict(program=program, seed=seed, randomized=randomized)
    if program == "stand":
        fields = ("mass_kg", "mean_foot_N", "fore_share", "com_xyz_m")
    else:
        fields = ("peak_foot_N", "mean_foot_N", "fore_share", "roll_pitch_abs_max_deg")
        force = data.get("force_transients")
        if force is not None:
            row.update(
                peak_10ms_by_foot_N=force["mean_10ms_peak_by_foot_N"],
                raw_peak_time_s=force["raw_peak_t_s"],
                peak_10ms_window_end_s=force["mean_10ms_last_sample_s"],
            )
        row["front_elbow_rms_max_Nm"] = max(data["torque_rms_Nm"][2:6:3])
        if program == "walk":
            fields += (
                "pass_all",
                "rules",
                "speed_mps",
                "min_support_margin_mm",
                "max_slide_mm",
                "steady_seconds",
                "side_span_mm",
                "three_or_four_support_pct",
                "saturated_pct",
            )
        else:
            fields += (
                "terminated",
                "termination",
                "elapsed_s",
                "loaded_slip_after3s_p95_mps",
                "time_in_air_pct",
                "friction",
            )
            changes = data["segments"][1:]
            row["target_deg"] = [s["requested_heading_deg"] for s in changes]
            row["hold_start_after_command_s"] = [s["first_1s_in_5deg_band_s"] for s in changes]
            row["final_actual_heading_deg"] = [s["final_actual_heading_deg"] for s in changes]
            row["end_ground_track_error_deg"] = [s["ground_track_final2s_signed_error_deg"] for s in changes]
            row["setup_completed"] = data["segments"][0]["completed"]
            row["completed_changes"] = sum(s["completed"] for s in changes)
            row["max_departure_from_command_position_m"] = max(
                (s["max_planar_departure_m"] for s in changes), default=None
            )
    row.update({k: data[k] for k in fields})
    return length, row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, nargs="+", choices=MODELS, default=[128, 160])
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument(
        "--records", type=Path, help="Keep individual outputs and the three filmed NPZs per model here."
    )
    parser.add_argument("--out", type=Path, default=ROOT / "mid160/results.json")
    args = parser.parse_args()
    if args.jobs < 1 or len(args.length) != len(set(args.length)):
        parser.error("use a positive worker count and specify each length once")
    report = {
        "foot_order": FEET,
        "scoring": "Pivots and six-headings use gentle steering, 100 Hz control, 1 kHz physics and 10 ms leg-command delay. "
        "Moving trials use 0.25 m/s requested forward speed. Each change requires true body yaw within "
        "±5° continuously for 1 s. Hold times exclude the initial straight setup. Foot force is the normal "
        "touch sensor in the original runner's steady walk or post-2 s steering window. Each 10 ms value "
        "is the largest average of 10 consecutive samples on that foot.",
        "models": {
            length: {"physics_sha256": hashlib.sha256(MODELS[length].read_bytes()).hexdigest(), "trials": []}
            for length in args.length
        },
    }
    with tempfile.TemporaryDirectory() as temp, ThreadPoolExecutor(max_workers=args.jobs) as pool:
        jobs = [
            pool.submit(run_one, args.records or Path(temp), length, case)
            for case, length in product(cases(), args.length)
        ]
        for job in jobs:
            length, row = job.result()
            report["models"][length]["trials"].append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
