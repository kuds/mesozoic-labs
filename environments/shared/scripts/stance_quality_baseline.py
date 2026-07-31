"""Statue stance-quality baseline at a candidate Stage 1a operating point.

Companion to ``zero_action_baseline.py``, which scores the do-nothing policy on
REWARD.  Stage 1a cannot gate on reward: summing the positive stage-1 weights
gives a ceiling of 3.35/step for T-Rex and the statue collects 97.0% of it with
energy and smoothness at exactly zero, so the statue IS the optimum and every
active policy scores below it.  See
docs/PLANT_VALIDATION_AND_STAGE1_OBJECTIVE.md section 9.

What 1a can gate on is stance QUALITY, and these are the numbers a policy has to
MATCH rather than beat.  Usage::

    python -m environments.shared.scripts.stance_quality_baseline [noise] [episodes]
"""

from __future__ import annotations

import sys
import tomllib
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from environments.brachiosaurus.envs.brachio_env import BrachioEnv
from environments.dibothrosuchus.envs.dibothrosuchus_env import DibothrosuchusEnv
from environments.trex.envs.trex_env import TRexEnv
from environments.velociraptor.envs.raptor_env import RaptorEnv

warnings.filterwarnings("ignore")

SPECIES: list[tuple[type, str]] = [
    (TRexEnv, "trex"),
    (RaptorEnv, "velociraptor"),
    (BrachioEnv, "brachiosaurus"),
    (DibothrosuchusEnv, "dibothrosuchus"),
]

CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs"
HORIZON = 1000
CONTACT_THRESHOLD = 0.1


def foot_forces(info: dict[str, Any]) -> list[float]:
    """Per-foot contact forces, in a stable order, for however many feet exist."""
    return [float(info[key]) for key in sorted(k for k in info if k.endswith("_foot_contact"))]


def measure(env_cls: type, species: str, noise: float, seeds: range) -> dict[str, float]:
    """Run the zero-action policy and summarise its stance rather than its return."""
    with open(CONFIG_ROOT / species / "stage1_balance.toml", "rb") as handle:
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
    noise = float(argv[1]) if len(argv) > 1 else 0.05
    episodes = int(argv[2]) if len(argv) > 2 else 40
    seeds = range(3042, 3042 + episodes)

    print(f"statue stance quality at reset_noise={noise}, {episodes} episodes")
    print(f"{'species':>15} {'full-hz':>8} {'all-feet':>9} {'unsup':>7} {'switch/s':>9} {'standing reward':>18}")
    for env_cls, species in SPECIES:
        result = measure(env_cls, species, noise, seeds)
        standing_sd = result["standing_sd"]
        spread = f"+/-{standing_sd:.1f}" if standing_sd == standing_sd else ""
        print(
            f"{species:>15} {result['full_horizon']:>7.0%} {result['all_feet']:>9.3f} "
            f"{result['unsupported']:>7.3f} {result['switches']:>9.2f} "
            f"{result['standing']:>12.1f} {spread:>5}"
        )
    print("  a 1a policy must MATCH these, not beat them -- the statue is the quality ceiling")


if __name__ == "__main__":
    main(sys.argv)
