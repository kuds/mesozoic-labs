"""Fit speed-dependent ankle and swing corrections while enforcing minimum swing height."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import fit_gait_residual as first
from .evaluate_gait import digest, run_episode
from .fit_gait_feedback import with_feedback

SEEDS = (62351, 62352)


def corrected_action(obs, learned, weights):
    # At cruise this equals the unattenuated fitted walking action. While the
    # command is small, reduce the foot roll/lift excursion to soften transfers.
    roll, yaw, ankle, lift, slow_ankle, slow_lift = weights
    vx = float(obs[-3])
    relief = 1.0 - np.clip(vx / 0.12, 0.0, 1.0) ** 4
    return with_feedback(obs, learned, (roll, yaw, ankle + relief * slow_ankle, lift + relief * slow_lift, 0.0))


def score(job):
    weights, cases, seed = job
    rows = []
    for surface, direction in cases:
        env = first._env_cls(surface=surface, max_episode_steps=1000)
        result, _ = run_episode(
            env,
            lambda obs: corrected_action(obs, first._actor(obs), weights),
            surface,
            direction,
            seed,
            "development",
            first._protocol,
            False,
        )
        rows.append(result)
        env.close()
    losses = []
    # This trial explicitly treats inadequate swing height as infeasible, even
    # if deliberate foot scraping would make starts/stops appear smoother.
    for r in rows:
        m = r["metrics"]
        loss = first.trial_loss(r)
        loss += 2000.0 * max(0.0, 0.012 - (m["median_clearance_m"] or 0.0))
        loss += 8.0 * max(0.0, (m["transition_accel_p95_mps2"] or 10.0) - 2.45)
        loss += 20.0 * max(0.0, 0.08 - (m["forward_mps"] or 0.0))
        losses.append(loss)
    return {"coefficients": weights, "loss": float(np.mean(losses) + max(losses)), "cases": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    screen = (("hard", 1), ("carpet", -1))
    grid = json.loads(args.grid.read_text())
    with ProcessPoolExecutor(max_workers=args.workers, initializer=first.initialize, initargs=(args.bundle,)) as pool:
        runs = list(pool.map(score, ((w, screen, SEEDS[0]) for w in grid)))
        top = sorted(runs, key=lambda row: row["loss"])[:5]
        finalists = list(dict.fromkeys([(0.0,) * 6] + [tuple(r["coefficients"]) for r in top]))
        verification = list(pool.map(score, ((w, (*screen, ("hard", -1), ("carpet", 1)), SEEDS[1]) for w in finalists)))
    record = {
        "schema": "mesozoic.gait-transition-fit/v1",
        "gain": 1.0,
        "training_seeds": SEEDS,
        "source_sha256": {
            p.name: digest(p)
            for p in (
                Path(__file__),
                Path(first.__file__),
                Path(__file__).with_name("fit_gait_feedback.py"),
                Path(__file__).with_name("evaluate_gait.py"),
            )
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
