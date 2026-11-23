"""Evaluate the scripted walk on the capsule model at seven terrain points."""

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

BUILD = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=BUILD, help="Triceratops build directory")
    parser.add_argument("--out", type=Path, required=True, help="Destination for terrain.json")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    sys.path.insert(0, str(args.build.resolve() / "walk"))
    import sweep_walk60 as walk_sweep

    config = walk_sweep.env_config()
    config["model_path"] = "trike60_C_closed_capsule.xml"
    points = {"flat": walk_sweep.S.make_point()}
    for kind, values in [("step", (0.01, 0.02, 0.03)), ("ramp", (3.0, 5.0, 8.0))]:
        for value in values:
            points[f"{kind}_{value}"] = walk_sweep.S.make_point(terrain=(kind, value))
    jobs = ((name, point, seed) for seed in range(200000, 200008) for name, point in points.items())
    episodes = defaultdict(list)
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=walk_sweep._init_worker,
        initargs=(config,),
    ) as pool:
        for name, record in pool.map(walk_sweep._task, jobs):
            episodes[name].append(record)
    axes = [
        f"{leg}_{axis}"
        for leg in ("left_fore", "right_fore", "left_hind", "right_hind")
        for axis in ("roll", "pitch", "flexion")
    ]
    result = {name: walk_sweep.S.summarise_point(point, episodes[name], axes) for name, point in points.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
