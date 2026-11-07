"""Fit a small command-conditioned correction to a frozen PPO actor on training seeds.

This is a bounded black-box policy search. Candidates receive complete unsupported
rollouts on the same command course. Development and confirmation seeds are reserved
for independent evaluation and cannot be supplied here.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .evaluate_gait import PROTOCOL, digest, run_episode

TRAIN_SEED = 62131
GAIN = 0.05


def adapted_action(obs, learned, coefficients, gain=GAIN):
    roll, yaw, pitch = coefficients
    wave, cosine = obs[36:38]
    vx, _, omega = obs[-3:]
    activity = np.clip(max(vx / 0.08, abs(omega) / 0.20), 0, 1)
    action = np.array(learned) * gain
    for offset, side in ((0, 1), (6, -1)):
        stride = -cosine * side
        action[offset] += activity * yaw * (omega / 0.5) * stride
        action[offset + 1] += activity * roll * wave
        action[offset + 5] -= activity * roll * wave
        forward = activity * pitch * stride * (vx / 0.12)
        action[offset + 2] += forward
        action[offset + 4] -= forward
    return np.clip(action, -1, 1)


def trial_loss(row):
    # Full course completion is the first priority. Penalize tracking error and
    # oscillations, including lateral acceleration during commanded speed ramps.
    m = row["metrics"]
    loss = 8 * (15 - m["duration_s"]) + 12 * max(0, 0.012 - (m["median_clearance_m"] or 0))
    loss += 12 * abs((m["forward_mps"] or 0) - 0.12)
    loss += 2.5 * abs((m["turn_radps"] or 0) - 0.30)
    loss += 0.70 * (m["transition_accel_p95_mps2"] or 10)
    loss += 0.05 * (m["tilt_p95_deg"] or 30)
    loss += 0.05 * max(0, 12 - m["alternations"])
    loss += 0.3 * max(0, m["substep_peak_foot_force_bw"] - 3)
    return loss


def score(job):
    coefficients, surfaces, seed = job
    reports = []
    for surface, sign in surfaces:
        env = _env_cls(surface=surface, max_episode_steps=1000)
        r, _ = run_episode(
            env,
            lambda obs: adapted_action(obs, _actor(obs), coefficients),
            surface,
            sign,
            seed,
            "development",
            _protocol,
            False,
        )
        reports.append(r)
        env.close()
    losses = [trial_loss(row) for row in reports]
    return {"coefficients": list(coefficients), "loss": float(np.mean(losses) + 0.4 * max(losses)), "cases": reports}


def initialize(bundle):
    global _actor, _env_cls, _protocol
    import torch

    from environments.tyrannosaurus_rex.intermediate.env import TyrannosaurusRexIntermediateEnv
    from environments.tyrannosaurus_rex.intermediate.inference import load_policy

    _env_cls = TyrannosaurusRexIntermediateEnv
    torch.set_num_threads(1)
    _actor = load_policy(bundle)
    _protocol = json.loads(PROTOCOL.read_text())


def search(grid, bundle, workers):
    # Use one paired crossing on each surface, then score finalists in both turn
    # directions over a fresh seed, including zero adapter as control.
    screen = (("hard", 1), ("carpet", -1))
    verify = (("hard", -1), ("carpet", 1), *screen)
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize, initargs=(bundle,)) as pool:
        trials = list(pool.map(score, ((c, screen, TRAIN_SEED) for c in grid)))
        winners = sorted(trials, key=lambda row: row["loss"])[:5]
        finalists = list(dict.fromkeys([(0.0, 0.0, 0.0)] + [tuple(w["coefficients"]) for w in winners]))
        verification = list(pool.map(score, ((c, verify, TRAIN_SEED + 1) for c in finalists)))
    return trials, verification


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--grid", required=True, type=Path, help="JSON array of triples (roll,yaw,pitch)")
    p.add_argument("--workers", default=2, type=int)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    trials, verification = search(json.loads(args.grid.read_text()), args.bundle, args.workers)
    record = {
        "schema": "mesozoic.gait-residual-fit/v1",
        "learner": "bounded direct policy search",
        "gain": GAIN,
        "training_seeds": [TRAIN_SEED, TRAIN_SEED + 1],
        "frozen_PPO_bundle_sha256": digest(args.bundle / "bundle.json"),
        "source_sha256": digest(Path(__file__)),
        "evaluation_sha256": digest(Path(__file__).with_name("evaluate_gait.py")),
        "grid_sha256": digest(args.grid),
        "screen": trials,
        "verification": verification,
        "best": min(verification, key=lambda row: row["loss"])["coefficients"],
    }
    args.out.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print("Best training coefficients", record["best"])


if __name__ == "__main__":
    main()
