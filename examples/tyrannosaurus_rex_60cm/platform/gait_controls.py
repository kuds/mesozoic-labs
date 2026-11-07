"""Paired development controls for a frozen, fitted walking policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from environments.shared.robot_gait_adapter import adapt
from environments.tyrannosaurus_rex.intermediate.env import TyrannosaurusRexIntermediateEnv
from environments.tyrannosaurus_rex.intermediate.inference import _load_verified, load_policy

from .evaluate_gait import PROTOCOL, digest, run_episode


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--bundle", required=True, type=Path)
    args.add_argument("--out", required=True, type=Path)
    cfg = args.parse_args()
    if cfg.out.exists():
        raise FileExistsError(cfg.out)
    torch.set_num_threads(1)
    record = json.loads((cfg.bundle / "bundle.json").read_text())
    policy = load_policy(cfg.bundle)
    actor, stats, _, _ = _load_verified(cfg.bundle)
    coefficients = record["adapter"]["coefficients"]
    seed = 62094
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        random_policy = type(actor.policy)(
            actor.observation_space, actor.action_space, lr_schedule=lambda _: 0.0, **actor.policy_kwargs
        )
    random_policy.set_training_mode(False)

    def raw_ppo(obs):
        return actor.predict(stats.normalize_obs(obs), deterministic=True)[0]

    def zero(obs):
        return adapt(obs, np.zeros(14, dtype=np.float32), coefficients)

    def untrained(obs):
        action = random_policy.predict(stats.normalize_obs(obs), deterministic=True)[0]
        return adapt(obs, action, coefficients)

    protocol = json.loads(PROTOCOL.read_text())
    runs = {}
    for name, predict in (
        ("trained_with_fit", policy), ("trained_only", raw_ppo), ("fit_only", zero), ("random_with_fit", untrained)
    ):
        rows = []
        for surface in protocol["surface"]:
            env = TyrannosaurusRexIntermediateEnv(surface=surface, max_episode_steps=1000)
            for direction in protocol["turn_sign"]:
                for episode_seed in protocol["development_seeds"]:
                    result, _ = run_episode(env, predict, surface, direction, episode_seed,
                                            "development", protocol, False)
                    rows.append({**result, "surface": surface, "sign": direction, "seed": episode_seed})
            env.close()
        runs[name] = rows
    report = {
        "schema": "mesozoic.walking-controls/v1", "random_policy_seed": seed,
        "bundle_sha256": digest(cfg.bundle / "bundle.json"), "protocol_sha256": digest(PROTOCOL),
        "implementation_sha256": digest(Path(__file__)), "evaluator_sha256": digest(PROTOCOL.with_name("evaluate_gait.py")),
        "normalization": "All MLPs receive the identical frozen training observation statistics. Random uses initial SB3 policy means.",
        "episodes": runs,
    }
    cfg.out.parent.mkdir(exist_ok=True, parents=True)
    cfg.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for name, rows in runs.items():
        print(name, sum(row["passed"] for row in rows), "/", len(rows))


if __name__ == "__main__":
    main()
