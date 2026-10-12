"""Statue stance-quality baseline at a candidate Stage 1a operating point.

Companion to ``zero_action_baseline.py``, which scores the do-nothing policy on
REWARD.  Stage 1a cannot gate on reward: summing the positive stage-1 weights
gives a ceiling of 3.35/step for T-Rex and the statue collects 97.0% of it with
energy and smoothness at exactly zero, so the statue IS the optimum and every
active policy scores below it.  See
docs/PLANT_VALIDATION_AND_STAGE1_OBJECTIVE.md section 9.

What 1a can gate on is stance QUALITY, and these are the numbers a policy has to
MATCH rather than beat.  Usage (``--model-id`` repeats; with none, the four
anatomical models of :data:`MODEL_IDS`, in that order)::

    python -m environments.shared.scripts.stance_quality_baseline [noise] [episodes] [--model-id SPECIES EMBODIMENT]
"""

from __future__ import annotations

import argparse
import sys
import tomllib
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from environments.shared.constants import PUBLICATION_SEED_START
from environments.shared.model_id import ModelId
from environments.shared.species_names import add_model_id_argument, parsed_model_ids
from environments.shared.species_registry import get_species_config

warnings.filterwarnings("ignore")

#: Report order: the table prints its rows in this order, not the manifest's.
#: The panel this tool has always measured, written out as models (decision D-D33).
MODEL_IDS: tuple[ModelId, ...] = (
    ModelId("tyrannosaurus_rex", "anatomical"),
    ModelId("velociraptor_mongoliensis", "anatomical"),
    ModelId("brachiosaurus_altithorax", "anatomical"),
    ModelId("dibothrosuchus_elaphros", "anatomical"),
)

CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs"
HORIZON = 1000
CONTACT_THRESHOLD = 0.1


def foot_forces(info: dict[str, Any]) -> list[float]:
    """Per-foot contact forces, in a stable order, for however many feet exist."""
    return [float(info[key]) for key in sorted(k for k in info if k.endswith("_foot_contact"))]


def measure(env_cls: type, model_id: ModelId, noise: float, seeds: range) -> dict[str, float]:
    """Run the zero-action policy and summarise its stance rather than its return."""
    from environments.shared.stage_manifest import load_stage_manifest

    stance_file = load_stage_manifest(model_id).resolve(1).config_file
    with open(model_id.path(CONFIG_ROOT) / stance_file, "rb") as handle:
        config = dict(tomllib.load(handle)["env"])
    config["reset_noise_scale"] = noise
    env = env_cls(**config)
    neutral = np.zeros(env.action_space.shape, dtype=np.float32)

    full_horizon = 0
    standing_returns: list[float] = []
    unsupported: list[float] = []
    all_feet: list[float] = []
    switches: list[float] = []

    for seed in seeds:
        env.reset(seed=int(seed))
        supported_count: list[int] = []
        steps = 0
        episode_return = 0.0
        info: dict[str, Any] = {}
        for _ in range(HORIZON):
            _, reward, terminated, truncated, info = env.step(neutral)
            steps += 1
            episode_return += float(reward)
            supported_count.append(sum(1 for force in foot_forces(info) if force > CONTACT_THRESHOLD))
            if terminated or truncated:
                break

        counts = np.array(supported_count)
        n_feet = len(foot_forces(info))
        unsupported.append(float((counts == 0).mean()))
        all_feet.append(float((counts == n_feet).mean()))
        fully_supported = (counts == n_feet).astype(int)
        switch_rate = float(np.abs(np.diff(fully_supported)).mean() / env.dt) if len(fully_supported) > 1 else 0.0
        switches.append(switch_rate)
        if steps >= HORIZON:
            full_horizon += 1
            standing_returns.append(episode_return)

    return {
        "full_horizon": full_horizon / len(seeds),
        "all_feet": float(np.mean(all_feet)),
        "unsupported": float(np.mean(unsupported)),
        "switches": float(np.mean(switches)),
        "standing": float(np.mean(standing_returns)) if standing_returns else float("nan"),
        "standing_sd": float(np.std(standing_returns)) if standing_returns else float("nan"),
    }


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("noise", nargs="?", type=float, default=0.05, help="reset_noise_scale (default 0.05)")
    parser.add_argument("episodes", nargs="?", type=int, default=40, help="episodes per model (default 40)")
    add_model_id_argument(parser, default=MODEL_IDS, what="measure")
    args = parser.parse_args(argv[1:])
    noise, episodes = args.noise, args.episodes
    seeds = range(PUBLICATION_SEED_START, PUBLICATION_SEED_START + episodes)

    print(f"statue stance quality at reset_noise={noise}, {episodes} episodes")
    print(f"{'model':>36} {'full-hz':>8} {'all-feet':>9} {'unsup':>7} {'switch/s':>9} {'standing reward':>18}")
    for model_id in parsed_model_ids(parser, args):
        result = measure(get_species_config(model_id).env_class, model_id, noise, seeds)
        standing_sd = result["standing_sd"]
        spread = f"+/-{standing_sd:.1f}" if standing_sd == standing_sd else ""
        print(
            f"{str(model_id):>36} {result['full_horizon']:>7.0%} {result['all_feet']:>9.3f} "
            f"{result['unsupported']:>7.3f} {result['switches']:>9.2f} "
            f"{result['standing']:>12.1f} {spread:>5}"
        )
    print("  a 1a policy must MATCH these, not beat them -- the statue is the quality ceiling")


if __name__ == "__main__":
    main(sys.argv)
