"""Spike training script for velociraptor via mjlab.

Usage (once mjlab is installed)::

    python environments/velociraptor/scripts/train_mjlab.py \\
        --stage 1 --num-envs 4096 --timesteps 20_000_000

Pilot success criteria (see ROADMAP Phase 5, mjlab pilot):

1. Reaches Stage 1 balance reward >= 1900 in <= 30 min on a single L4 GPU
   (vs SB3 PPO baseline of 2:57:25).
2. envs/sec throughput >= 2x the existing MJXDinoEnv path.
3. LOC to express env + curriculum <= 60% of current MJXDinoEnv LOC.
"""

from __future__ import annotations

import argparse
import sys


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="mjlab training spike for velociraptor")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--timesteps", type=int, default=20_000_000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Import here so `--help` works without the optional dep installed.
    import environments.velociraptor.mjlab_config  # noqa: F401  # registers species
    from environments.shared.mjlab_env import make_mjlab_env

    env = make_mjlab_env(
        species="velociraptor",
        stage=args.stage,
        num_envs=args.num_envs,
        device=args.device,
    )

    # Training loop is handed off to mjlab's built-in rl-games / rsl-rl
    # integration. The exact runner selection is decided in the pilot PR.
    raise NotImplementedError(
        "Pilot PR: run mjlab's built-in PPO runner with curriculum hook. "
        f"Env constructed for stage={args.stage}, num_envs={args.num_envs}, "
        f"device={args.device}. env={env!r}"
    )


if __name__ == "__main__":
    sys.exit(main())
