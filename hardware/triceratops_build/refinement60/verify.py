"""Exercise the staged model and its normal motion entry points without a graphics context."""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def verify(output):
    output.mkdir(parents=True, exist_ok=True)
    model = ROOT / "models" / "refined"
    subprocess.run(
        [sys.executable, str(ROOT / "geometry" / "check_models.py"), str(model), str(output / "model.json")],
        check=True,
    )
    trials = [
        ("stand", "stand", []),
        ("walk", "walk", ["--seed", "11"]),
        ("walk-stiff", "walk", ["--seed", "21", "--stiff"]),
        ("left", "pivot", ["--seed", "101"]),
        ("right", "pivot-right", ["--seed", "102", "--force-trace", str(output / "right.csv")]),
        ("gentle-right", "pivot-right", [
            "--seed", "102", "--turn-profile", "gentle", "--force-trace", str(output / "gentle-right.csv"),
        ]),
        ("course", "course", ["--seed", "101"]),
        ("gentle-course", "course", ["--seed", "103", "--random", "--turn-profile", "gentle"]),
        ("gentle-six", "six-headings", ["--seed", "103", "--turn-profile", "gentle"]),
    ]
    for name, mode, flags in trials:
        prefix = output / name
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "motion" / "run.py"),
                mode,
                "--model",
                str(model / "physics.xml"),
                "--out",
                str(prefix),
                *flags,
            ],
            check=True,
        )
        row = json.loads(prefix.with_suffix(".json").read_text())
        if mode == "stand":
            if min(row["mean_foot_N"]) <= 0 or not math.isclose(
                sum(row["mean_foot_N"]), row["mass_kg"] * 9.81, rel_tol=1e-3
            ):
                raise ValueError(f"{name}: static load sum or foot contact failed")
        elif mode == "walk":
            if not row["pass_all"]:
                raise ValueError(f"{name}: {row['rules']}")
        else:
            headings = [item["requested_heading_deg"] for item in row["segments"]]
            expected = {
                "course": [0, 35, -35, 0], "six-headings": [0, 15, 35, 10, -15, -35, 0],
                "pivot": [0, 40], "pivot-right": [0, -40],
            }[mode]
            if (
                row["terminated"]
                or headings != expected
                or any(not item["completed"] or item["first_1s_in_5deg_band_s"] is None for item in row["segments"])
            ):
                raise ValueError(f"{name}: heading acquisition or run completion failed")

    # Relative within-runner regression on the published seed. These are simulated load reductions.
    original = json.loads((output / "right.json").read_text())["force_transients"]
    gentle = json.loads((output / "gentle-right.json").read_text())["force_transients"]
    for metric in ("raw_peak_N", "mean_10ms_peak_N"):
        if gentle[metric] > 0.70 * original[metric]:
            raise ValueError(f"seed 102 pivot failed 30% load reduction for {metric}: {gentle[metric]} vs {original[metric]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Directory for the measured JSON results")
    verify(parser.parse_args().output.resolve())
