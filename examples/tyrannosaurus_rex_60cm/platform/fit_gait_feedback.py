"""Bounded residual-policy improvement on sensor/phase features, with full-strength PPO."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import fit_gait_residual as first
from .evaluate_gait import digest, run_episode

SEEDS = (62241, 62242)


def with_feedback(obs, learned, coefficients):
    roll, yaw, ankle, lift, gyro = coefficients
    action = first.adapted_action(obs, learned, (roll, yaw, 0.0), gain=1.0)
    vx, _, omega = obs[-3:]
    activity = np.clip(max(vx / 0.08, abs(omega) / 0.20), 0, 1)
    # Correct lateral foot angle during transfer, and adjust swing-leg
    # shortening while preserving the sagittal sole orientation.
    for offset, side in ((0, 1), (6, -1)):
        action[offset + 5] += activity * (ankle * obs[36] + gyro * obs[31])
        correction = activity * lift * max(0.0, side * obs[36])
        action[offset + 3] += correction
        action[offset + 4] -= 1.25 * correction  # 0.20 knee / 0.16 ankle action scale
    return np.clip(action, -1, 1)


def score(job):
    weights, cases, seed = job
    rows = []
    for surface, sign in cases:
        env = first._env_cls(surface=surface, max_episode_steps=1000)
        result, _ = run_episode(
            env,
            lambda obs: with_feedback(obs, first._actor(obs), weights),
            surface,
            sign,
            seed,
            "development",
            first._protocol,
            False,
        )
        env.close()
        rows.append(result)
    # Favor reduced transition acceleration once walking is established.
    losses = [
        first.trial_loss(r) + 2.5 * max(0.0, (r["metrics"]["transition_accel_p95_mps2"] or 10) - 2.5) for r in rows
    ]
    return {"coefficients": weights, "loss": float(np.mean(losses) + 0.4 * max(losses)), "cases": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--grid", required=True, type=Path)
    parser.add_argument("--workers", default=2, type=int)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    grid = json.loads(args.grid.read_text())
    screen = (("hard", 1), ("carpet", -1))
    with ProcessPoolExecutor(max_workers=args.workers, initializer=first.initialize, initargs=(args.bundle,)) as pool:
        runs = list(pool.map(score, ((w, screen, SEEDS[0]) for w in grid)))
        top = sorted(runs, key=lambda r: r["loss"])[:4]
        finalists = list(dict.fromkeys([(0.0,) * 5] + [tuple(r["coefficients"]) for r in top]))
        verification = list(pool.map(score, ((w, (*screen, ("hard", -1), ("carpet", 1)), SEEDS[1]) for w in finalists)))
    record = {
        "schema": "mesozoic.gait-feedback-fit/v1",
        "gain": 1.0,
        "training_seeds": SEEDS,
        "source_sha256": {
            p.name: digest(p)
            for p in (Path(__file__), Path(first.__file__), Path(__file__).with_name("evaluate_gait.py"))
        },
        "frozen_PPO_bundle_sha256": digest(args.bundle / "bundle.json"),
        "grid_sha256": digest(args.grid),
        "screen": runs,
        "verification": verification,
        "best": min(verification, key=lambda row: row["loss"])["coefficients"],
    }
    args.out.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print("Best training coefficients", record["best"])


if __name__ == "__main__":
    main()
