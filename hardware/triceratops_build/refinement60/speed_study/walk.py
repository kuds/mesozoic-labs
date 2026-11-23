"""Compare the straight four-beat walk at fixed stride and progressively faster footfalls."""

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {128: ROOT / "models/refined/physics.xml", 160: ROOT / "models/mid160/physics.xml"}


def trial(job, root, record):
    length, cadence, seed, floor = job
    name = f"{cadence:g}".replace(".", "p") + f"x-{seed}-{floor}"
    dest = root / str(length) / name
    args = [
        sys.executable,
        str(ROOT / "motion/run.py"),
        "walk",
        "--model",
        str(MODELS[length]),
        "--walk-cadence",
        str(cadence),
        "--seed",
        str(seed),
        "--out",
        str(dest),
    ]
    if floor == "stiff":
        args.append("--stiff")
    if record and seed == 11 and floor == "nominal":
        args.append("--record")
    subprocess.run(args, check=True)
    r = json.loads(dest.with_suffix(".json").read_text())
    changes = [k for k in ("C1", "C2", "C3", "C4", "C5c", "C6") if not r["rules"][k]]
    return dict(
        upper_mm=length,
        cadence=cadence,
        seed=seed,
        floor=floor,
        pass_all=r["pass_all"],
        failed_criteria=changes,
        speed_mm_s=round(1000 * r["speed_mps"], 3),
        footfall_interval_s=round(r["footfall_interval_s"], 4),
        min_support_margin_mm=round(r["min_support_margin_mm"], 3),
        max_slide_mm=round(r["max_slide_mm"], 3),
        three_or_four_support_pct=round(r["three_or_four_support_pct"], 2),
        body_lateral_span_mm=round(r["side_span_mm"], 3),
        upright=r["rules"]["C1"],
        four_beat=r["rules"]["C4"],
        tilt_peak_deg=round(r["max_tilt_deg"], 3),
        actuator_saturation_pct=round(r["saturated_pct"], 3),
        peak_N_by_foot=[round(v, 3) for v in r["peak_foot_N"]],
        peak_10ms_N_by_foot=[round(v, 3) for v in r["force_transients"]["mean_10ms_peak_by_foot_N"]],
        front_elbow_rms_max_Nm=round(max(r["torque_rms_Nm"][i] for i in (2, 5)), 4),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cadences",
        type=float,
        nargs="+",
        default=[1, 1.25, 1.4, 1.45, 1.475, 1.5, 1.525, 1.55, 1.575, 1.6, 1.625, 1.75, 1.875, 2, 2.5],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 12, 13])
    parser.add_argument("--floors", nargs="+", choices=("nominal", "stiff"), default=["nominal", "stiff"])
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--records", type=Path, help="Keep all trial JSONs and nominal seed 11 NPZs here.")
    parser.add_argument(
        "--holdout", action="store_true", help="Repeat 1.25x on seeds 21–23 and 1.4x, 1.45x, 1.5x on seeds 21–22."
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.holdout:
        jobs = [
            *itertools.product(MODELS, [1.25], (21, 22, 23), args.floors),
            *itertools.product(MODELS, [1.4, 1.45, 1.5], (21, 22), args.floors),
        ]
    else:
        jobs = itertools.product(MODELS, args.cadences, args.seeds, args.floors)
    output = args.out or ROOT / "speed_study" / ("walk-holdout.json" if args.holdout else "walk.json")
    with tempfile.TemporaryDirectory() as temp, ThreadPoolExecutor(max_workers=args.jobs) as pool:
        rows = list(pool.map(lambda j: trial(j, args.records or Path(temp), bool(args.records)), jobs))
    document = {
        "models_sha256": {length: hashlib.sha256(path.read_bytes()).hexdigest() for length, path in MODELS.items()},
        "foot_order": ["LF", "RF", "LH", "RH"],
        "scoring": "The straight exhibit plan keeps the 96 mm stride and all plantar and trunk waypoints. "
        "Its reference time and swing/stance clock are divided by the cadence multiplier. Physics, actuator "
        "limits, judge thresholds and neck-gesture timing are unchanged. Walk speed is net displacement divided "
        "by elapsed time less the two standing holds. Force and elbow RMS cover the steady sequence following "
        "two setup swings. Balance and contact sequence are judged across the entire trial. Each 10 ms force "
        "peak averages ten consecutive 1 kHz samples on a single foot. C1 checks posture and numerical health, "
        "C2 progress/drift, C3 a minimum 10 mm support margin, C4 ordered unloaded swings and three loaded stance "
        "feet, C5c at most 10 mm contact slide and C6 at most 1% actuator saturation.",
        "runs": sorted(rows, key=lambda r: (r["upper_mm"], r["cadence"], r["seed"], r["floor"])),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
