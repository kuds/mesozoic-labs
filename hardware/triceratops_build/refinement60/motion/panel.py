"""Run the matched 60 cm standing, walking, moving-course and pivot trials."""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent


def trial_matrix(group, steering_seeds=(101, 102, 103)):
    """Return routine, seed and variation flags. Use independent generator processes for each trial."""
    if group in ("all", "walk"):
        yield "stand", 11, []
        for seed in (11, 12, 13, 14, 21, 22, 23, 24):
            for flags in ([], ["--stiff"]):
                yield "walk", seed, flags
    if group in ("all", "steering"):
        for seed in steering_seeds:
            for flags in ([], ["--random"]):
                yield "course", seed, flags
            yield "pivot", seed, []
            yield "pivot-right", seed, []


def process(args, trial):
    mode, seed, flags = trial
    condition = flags[0][2:] if flags else "nominal"
    stem = args.out / f"{mode}-{seed}-{condition}"
    cmd = [
        sys.executable,
        str(HERE / "run.py"),
        mode,
        "--model",
        str(args.model),
        "--gait",
        args.gait,
        "--yaw-gain",
        str(args.yaw_gain),
        "--seed",
        str(seed),
        "--out",
        str(stem),
        *flags,
    ]
    if mode not in ("stand", "walk"):
        cmd.extend(["--turn-profile", args.turn_profile, "--gyro-bias-dps", str(args.gyro_bias_dps)])
    subprocess.run(cmd, check=True)
    return mode, json.loads(stem.with_suffix(".json").read_text())


def report(results):
    summary = {}
    walks = [r for name, r in results if name == "walk"]
    if walks:
        summary["walk"] = {
            "trials": len(walks),
            "passes": sum(w["pass_all"] for w in walks),
            "worst_contact_slide_mm": max(w["max_slide_mm"] for w in walks),
            "minimum_support_margin_mm": min(w["min_support_margin_mm"] for w in walks),
            "worst_foot_load_N": max(max(w["peak_foot_N"]) for w in walks),
            "mean_lateral_span_mm": sum(w["side_span_mm"] for w in walks) / len(walks),
        }
    for mode in ("course", "pivot", "pivot-right"):
        runs = [r for name, r in results if name == mode]
        if runs:
            summary[mode] = {
                "trials": len(runs),
                "upright": sum(not r["terminated"] for r in runs),
                "all_headings_acquired": sum(
                    len(r["segments"]) == (4 if mode == "course" else 2)
                    and all(s["completed"] and s["first_1s_in_5deg_band_s"] is not None for s in r["segments"])
                    for r in runs
                ),
                "worst_foot_load_N": max(max(r["peak_foot_N"]) for r in runs),
                "worst_10ms_mean_N": max(
                    (r["force_transients"]["mean_10ms_peak_N"] for r in runs if r["force_transients"]), default=None
                ),
                "worst_loaded_slip_p95_mps": max(
                    (r["loaded_slip_after3s_p95_mps"] for r in runs if r["loaded_slip_after3s_p95_mps"] is not None),
                    default=None,
                ),
            }
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=HERE.parent / "models/refined/physics.xml")
    p.add_argument("--gait", choices=("reference", "refined"), default="refined")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--group", choices=("walk", "steering", "all"), default="all")
    p.add_argument("--jobs", type=int, default=1, help="Number of independent simulator processes")
    p.add_argument("--yaw-gain", type=float, default=1.5)
    p.add_argument("--turn-profile", choices=("original", "gentle", "shaped-stock-yaw"), default="original")
    p.add_argument("--steering-seeds", type=int, nargs="+", default=[101, 102, 103])
    p.add_argument("--gyro-bias-dps", type=float, default=0.0)
    args = p.parse_args()
    if len(set(args.steering_seeds)) != len(args.steering_seeds):
        p.error("--steering-seeds must be unique")
    if args.gyro_bias_dps and args.turn_profile != "gentle":
        p.error("--gyro-bias-dps applies only to the gentle controller")
    if args.jobs < 1:
        p.error("--jobs must be positive")
    args.out.mkdir(exist_ok=True, parents=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as workers:
        summary = report(
            list(workers.map(lambda trial: process(args, trial), trial_matrix(args.group, args.steering_seeds)))
        )
    result = json.dumps(summary, indent=2, allow_nan=False)
    (args.out / "summary.json").write_text(result + "\n")
    print(result)


if __name__ == "__main__":
    main()
