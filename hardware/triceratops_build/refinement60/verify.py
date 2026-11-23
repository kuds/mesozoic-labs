"""Exercise the staged model and its normal motion entry points without a graphics context."""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent


def check_comparison_geometry(original, middle):
    """Keep the comparison on the same chassis, stance, lower links and drive limits."""
    stances = []
    limits = []
    for folder, upper_length in ((original, 0.128), (middle, 0.160)):
        model = mujoco.MjModel.from_xml_path(str(folder / "physics.xml"))
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("neutral").id)
        mujoco.mj_kinematics(model, data)
        stance = []
        for side in ("left", "right"):
            prefix = f"{side}_fore_"
            shoulder, elbow, ankle = [data.xanchor[model.joint(prefix + j).id] for j in ("pitch", "flexion", "ankle")]
            np.testing.assert_allclose(np.linalg.norm(shoulder - elbow), upper_length, atol=1e-6)
            np.testing.assert_allclose(np.linalg.norm(elbow - ankle), 0.184, atol=1e-6)
            stance.extend((shoulder.copy(), ankle.copy(), data.geom(prefix + "sole").xpos.copy()))
        stances.append(stance)
        limits.append(model.actuator_forcerange.copy())
    np.testing.assert_allclose(stances[0], stances[1], atol=1e-6)
    np.testing.assert_array_equal(limits[0], limits[1])


def check_rehearsed_walk(output):
    report = output / "rehearsed-course.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "speed_study" / "choreography.py"),
            "--traction",
            "--cadences",
            "1.5",
            "--seeds",
            "12",
            "--jobs",
            "2",
            "--records",
            str(output / "rehearsed"),
            "--out",
            str(report),
        ],
        check=True,
    )
    rows = json.loads(report.read_text())["runs"]
    expected = {(upper, floor) for upper in (128, 160) for floor in ("nominal", "stiff")}
    if {(r["upper_mm"], r["floor"]) for r in rows} != expected:
        raise ValueError("Rehearsed trials must cover both lengths and both contact settings")
    for row in rows:
        if (
            not row["commands_complete"]
            or len(row["holds_s"]) != 6
            or any(t is None for t in row["holds_s"])
            or not row["upright"]
            or row["min_support_margin_mm"] < 10
            or not row["traction_gates"]["C6"]
            or row["measured_forward_mm_s"] < 25
        ):
            raise ValueError(f"Rehearsed 1.50× course failed at {row['upper_mm']} mm on {row['floor']}")


def verify(output):
    output.mkdir(parents=True, exist_ok=True)
    original = ROOT / "models" / "refined"
    middle = ROOT / "models" / "mid160"
    for folder, report in ((original, "model.json"), (middle, "model-160.json")):
        subprocess.run(
            [sys.executable, str(ROOT / "geometry" / "check_models.py"), str(folder), str(output / report)],
            check=True,
        )
    check_comparison_geometry(original, middle)
    check_rehearsed_walk(output)
    trials = [
        ("stand", "stand", []),
        ("walk", "walk", ["--seed", "11"]),
        ("walk-stiff", "walk", ["--seed", "21", "--stiff"]),
        ("left", "pivot", ["--seed", "101"]),
        ("right", "pivot-right", ["--seed", "102", "--force-trace", str(output / "right.csv")]),
        (
            "gentle-right",
            "pivot-right",
            [
                "--seed",
                "102",
                "--turn-profile",
                "gentle",
                "--force-trace",
                str(output / "gentle-right.csv"),
            ],
        ),
        ("course", "course", ["--seed", "101"]),
        ("gentle-course", "course", ["--seed", "103", "--random", "--turn-profile", "gentle"]),
        ("gentle-six", "six-headings", ["--seed", "103", "--turn-profile", "gentle"]),
    ]
    matched = [(original, name, mode, flags) for name, mode, flags in trials]
    matched.extend(
        [
            (original, "128-exhibit-150", "walk", ["--seed", "11", "--walk-cadence", "1.5"]),
            (middle, "160-exhibit-150", "walk", ["--seed", "11", "--walk-cadence", "1.5"]),
            (
                middle,
                "160-directional-030",
                "six-headings",
                ["--seed", "103", "--turn-profile", "gentle", "--speed-mps", "0.30"],
            ),
            (
                middle,
                "160-staggered-008",
                "six-headings",
                [
                    "--seed",
                    "103",
                    "--turn-profile",
                    "gentle",
                    "--steering-footfall",
                    "four-beat",
                    "--speed-mps",
                    "0.08",
                ],
            ),
            (middle, "160-stand", "stand", []),
            (middle, "160-walk", "walk", ["--seed", "11"]),
            (middle, "160-right", "pivot-right", ["--seed", "102", "--turn-profile", "gentle"]),
            (middle, "160-six", "six-headings", ["--seed", "103", "--random", "--turn-profile", "gentle"]),
        ]
    )
    for model, name, mode, flags in matched:
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
            if "--steering-footfall" in flags:
                schedule = row["footfall_schedule"]
                if len(set(schedule["offsets"].values())) != 4 or schedule["duty"] < 0.75:
                    raise ValueError(f"{name}: four separate swing phases were not scheduled")
            if "--speed-mps" in flags:
                requested = float(flags[flags.index("--speed-mps") + 1])
                if any(s["requested_speed_mps"] != requested for s in row["segments"]):
                    raise ValueError(f"{name}: speed override is missing from heading measurements")
            headings = [item["requested_heading_deg"] for item in row["segments"]]
            expected = {
                "course": [0, 35, -35, 0],
                "six-headings": [0, 15, 35, 10, -15, -35, 0],
                "pivot": [0, 40],
                "pivot-right": [0, -40],
            }[mode]
            if (
                row["terminated"]
                or headings != expected
                or any(not item["completed"] or item["first_1s_in_5deg_band_s"] is None for item in row["segments"])
            ):
                raise ValueError(f"{name}: heading acquisition or run completion failed")

    for default, faster in (("walk", "128-exhibit-150"), ("160-walk", "160-exhibit-150")):
        original_speed = json.loads((output / f"{default}.json").read_text())["speed_mps"]
        fast_speed = json.loads((output / f"{faster}.json").read_text())["speed_mps"]
        if fast_speed < 1.35 * original_speed:
            raise ValueError(f"{faster}: cadence override failed to increase observed speed")

    reference_run = json.loads((output / "gentle-right.json").read_text())
    longer_run = json.loads((output / "160-right.json").read_text())
    if longer_run["peak_foot_N"][1] >= reference_run["peak_foot_N"][1]:
        raise ValueError("The 160 mm foreleg no longer reduces the matched right-forefoot pivot peak.")

    # Relative within-runner regression on the published seed. These are simulated load reductions.
    original = json.loads((output / "right.json").read_text())["force_transients"]
    gentle = json.loads((output / "gentle-right.json").read_text())["force_transients"]
    for metric in ("raw_peak_N", "mean_10ms_peak_N"):
        if gentle[metric] > 0.70 * original[metric]:
            raise ValueError(
                f"seed 102 pivot failed 30% load reduction for {metric}: {gentle[metric]} vs {original[metric]}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Directory for the measured JSON results")
    verify(parser.parse_args().output.resolve())
