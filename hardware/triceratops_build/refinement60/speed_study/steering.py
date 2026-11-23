"""Reproduce matched six-heading diagonal-gait speed trials and their compact evidence."""

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
MODELS = {128: ROOT / "models/refined/physics.xml", 160: ROOT / "models/mid160/physics.xml"}
SPEEDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.575, 0.60]
TARGETS = [15, 35, 10, -15, -35, 0]
FOOT_ORDER = ["left_fore", "right_fore", "left_hind", "right_hind"]


def trial(raw, length, speed, seed, randomized, read_existing):
    name = f"{length}-{round(speed * 1000):04d}-{seed}-{'random' if randomized else 'nominal'}"
    prefix = raw / name
    if not read_existing:
        command = [
            sys.executable,
            str(ROOT / "motion/run.py"),
            "six-headings",
            "--speed-mps",
            str(speed),
            "--turn-profile",
            "gentle",
            "--seed",
            str(seed),
            "--model",
            str(MODELS[length]),
            "--out",
            str(prefix),
        ]
        if randomized:
            command.append("--random")
        subprocess.run(command, check=True)
    result = json.loads(prefix.with_suffix(".json").read_text())
    settings = result["settings"]
    if settings.get("steering_footfall", "diagonal") != "diagonal":
        raise ValueError(f"Unexpected steering footfall in {name}")
    if (settings["speed_mps"], settings["seed"], settings["random"], settings["turn_profile"]) != (
        speed,
        seed,
        randomized,
        "gentle",
    ) or Path(settings["model"]).parent.name != MODELS[length].parent.name:
        raise ValueError(f"Unexpected settings in {name}")

    # A partial or unreached segment is a missed command, including trials that trip the tilt stop.
    segments = result["segments"]
    turns = segments[1:]
    if [s["requested_heading_deg"] for s in turns] != TARGETS[: len(turns)]:
        raise ValueError(f"Unexpected headings in {name}")
    if any(s["requested_speed_mps"] != speed for s in segments):
        raise ValueError(f"Unexpected speed metadata in {name}")
    acquired = [s["completed"] and s["first_1s_in_5deg_band_s"] is not None for s in turns]
    end_errors = [
        abs(math.remainder(s["final_actual_heading_deg"] - s["requested_heading_deg"], 360)) if s["completed"] else None
        for s in turns
    ]
    duration = result["elapsed_s"]
    mean_speed = (
        sum(s["projected_mean_speed_mps"] * (s["observed_until_s"] - s["start_s"]) for s in segments) / duration
    )
    force = result["force_transients"]
    track_errors = [
        abs(s["ground_track_final2s_signed_error_deg"])
        for s in turns
        if s["completed"] and s["ground_track_final2s_signed_error_deg"] is not None
    ]
    return {
        "length_mm": length,
        "command_mps": speed,
        "seed": seed,
        "randomized": randomized,
        "elapsed_s": duration,
        "stop": result["termination"],
        "finished": not result["terminated"] and duration == 40,
        "holds": acquired + [False] * (6 - len(acquired)),
        "end_errors_deg": end_errors + [None] * (6 - len(end_errors)),
        "actual_mps": round(mean_speed, 5),
        "peak_by_foot_N": [round(n, 2) for n in result["peak_foot_N"]],
        "peak_10ms_by_foot_N": [round(n, 2) for n in force["mean_10ms_peak_by_foot_N"]] if force else None,
        "roll_pitch_abs_max_deg": [round(n, 3) for n in result["abs_peak_roll_pitch_deg"]],
        "support_count_pct": [round(n, 4) for n in result["support_count_pct"]],
        "loaded_slip_p95_mps": result["loaded_slip_after3s_p95_mps"],
        "max_abs_ground_track_error_deg": round(max(track_errors), 3) if track_errors else None,
        "max_ik_residual_mm": round(1000 * result["max_ik_residual_m"], 3),
        "max_axis_saturation_pct": round(max(result["saturation_pct_by_axis"]), 3),
    }


def summarize(rows):
    finished = [r for r in rows if r["finished"]]
    acquired = [r for r in finished if all(r["holds"])]
    end_errors = [e for r in rows for e in r["end_errors_deg"] if e is not None]
    slips = [r["loaded_slip_p95_mps"] for r in rows if r["loaded_slip_p95_mps"] is not None]
    return dict(
        length_mm=rows[0]["length_mm"],
        command_mps=rows[0]["command_mps"],
        trials=len(rows),
        finished=len(finished),
        all_six_held=len(acquired),
        total_holds=sum(sum(r["holds"]) for r in rows),
        all_six_end_within_5=sum(all(e is not None and e <= 5 for e in r["end_errors_deg"]) for r in finished),
        worst_scored_end_error_deg=max(end_errors, default=None),
        mean_speed_completed_mps=round(mean(r["actual_mps"] for r in finished), 5) if finished else None,
        any_foot_peak_N=max(max(r["peak_by_foot_N"]) for r in rows),
        right_fore_peak_N=max(r["peak_by_foot_N"][1] for r in rows),
        any_foot_peak_10ms_N=max(
            (max(r["peak_10ms_by_foot_N"]) for r in rows if r["peak_10ms_by_foot_N"]), default=None
        ),
        worst_roll_or_pitch_deg=max(max(r["roll_pitch_abs_max_deg"]) for r in rows),
        worst_flight_pct=max(r["support_count_pct"][0] for r in rows),
        worst_loaded_slip_p95_mps=max(slips, default=None),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speeds", type=float, nargs="+", default=SPEEDS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103])
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--raw-dir", type=Path, help="Keep individual runner JSON results here.")
    parser.add_argument(
        "--read-existing", action="store_true", help="Summarize the specified raw directory without rerunning."
    )
    parser.add_argument("--out", required=True, type=Path)
    a = parser.parse_args()
    if a.jobs < 1 or (a.read_existing and a.raw_dir is None):
        parser.error("Use positive --jobs and provide --raw-dir to read existing trials.")
    with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor(max_workers=a.jobs) as pool:
        raw = a.raw_dir or Path(tmp)
        raw.mkdir(parents=True, exist_ok=True)
        cases = product(MODELS, a.speeds, a.seeds, (False, True))
        futures = [pool.submit(trial, raw, *case, a.read_existing) for case in cases]
        rows = [job.result() for job in futures]
    report = {
        "physics_sha256": {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in MODELS.items()},
        "schedule_deg": TARGETS,
        "foot_order": FOOT_ORDER,
        "seeds": a.seeds,
        "method": "40 s six-heading course including straight setup, 0.70 s stride and 70% requested stance, gentle gyro steering and 10 ms motor delay. Maxima use all trials including terminated runs. Actual speed projects displacement onto the requested direction for each segment and includes setup; terminal trials use observed time in the raw rows and are excluded from the completed-run mean. Foot normal loads are sampled at 1 kHz after 2 s (from launch if terminated by 2 s); support counts use >2 N per foot. Slip is evaluated after 3 s. Each completed requested heading must stay within ±5° continuously for 1 s. End errors and track error refer only to completed commands.",
        "summary": [
            summarize([r for r in rows if r["length_mm"] == length and r["command_mps"] == speed])
            for length, speed in product(MODELS, a.speeds)
        ],
        "trials": rows,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    # Leave each trial on one line so the evidence stays readable and easy to diff.
    header = {key: value for key, value in report.items() if key not in ("trials", "summary")}
    lines = [json.dumps(header, indent=2, ensure_ascii=False)[:-2] + ","]
    for group in ("summary", "trials"):
        lines.append(f'  "{group}": [')
        lines.append(",\n".join("    " + json.dumps(row, allow_nan=False) for row in report[group]))
        lines.append("  ]," if group == "summary" else "  ]")
    a.out.write_text("\n".join(lines) + "\n}\n")


if __name__ == "__main__":
    main()
