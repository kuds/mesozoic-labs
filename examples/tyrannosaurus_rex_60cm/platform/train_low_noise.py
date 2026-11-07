"""Bounded PPO comparison starting on the walking task with smaller exploration."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from environments.shared.config import load_all_stages
from environments.shared.model_id import ModelId
from environments.shared.species_registry import get_species_config
from environments.shared.train_base import train


def main() -> None:
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--output", type=Path, required=True)
    args.add_argument("--steps", type=int, default=98304)
    args.add_argument("--seed", type=int, default=484)
    config = args.parse_args()
    if config.steps <= 0 or config.steps % 1024:
        args.error("--steps must be a positive multiple of 1024")
    torch.set_num_threads(1)
    model = ModelId("tyrannosaurus_rex", "intermediate")
    stages = load_all_stages(model)
    # Compare a fresh policy with a quieter exploration distribution against
    # transfer from stance. The environment and task fingerprint are identical.
    ppo = stages[2]["ppo_kwargs"]
    ppo.update(learning_rate=0.00005, n_epochs=3, clip_range=0.12)
    ppo["policy_kwargs"]["log_std_init"] = -2.7
    train(
        get_species_config(model),
        stages,
        2,
        config.steps,
        n_envs=4,
        seed=config.seed,
        eval_freq=16384,
        save_freq=32768,
        output_dir=str(config.output),
        verbose=0,
        use_tensorboard=False,
        post_eval_episodes=8,
    )


if __name__ == "__main__":
    main()
