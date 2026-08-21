"""Score the do-nothing policy on a stage, as the floor every run must clear.

Residual actions are centred on each species' home keyframe, so ``action = 0``
commands exactly the nominal stance.  On a passively stable plant that is a
real policy, and on a balance stage it can be a *strong* one: the T-Rex
stage-1 zero-action score beat both trained policies from the 2026-07-23/24
runs on reward, on mean-minus-std, and on full-horizon survival.  Six million
PPO steps produced something worse than ``np.zeros(21)`` and nothing in the
pipeline noticed, because no run was ever compared against this floor.

Run it before and after any plant or task change: the baseline re-anchors
automatically, so it stays a valid comparison across model revisions.  A
trained stage-1 policy that does not beat it on all three of reward,
mean-minus-std, and full-horizon share has not learned to balance.

It is also how the stage-1 reset noise gets calibrated.  Too little and the
plant survives on its own, leaving nothing for a policy to earn (T-Rex at
``reset_noise_scale = 0.05`` scores 93% full-horizon); too much and the task
stops being learnable.

Usage::

    python environments/shared/scripts/zero_action_baseline.py trex
    python environments/shared/scripts/zero_action_baseline.py trex:2 velociraptor
    python environments/shared/scripts/zero_action_baseline.py trex --sweep-noise
"""

import argparse
import importlib
import sys
import tomllib
from pathlib import Path

import numpy as np

_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

SPECIES_ENVS = {
    "trex": ("environments.trex.envs.trex_env", "TRexEnv"),
    "velociraptor": ("environments.velociraptor.envs.raptor_env", "RaptorEnv"),
    "brachiosaurus": ("environments.brachiosaurus.envs.brachio_env", "BrachioEnv"),
    "dibothrosuchus": ("environments.dibothrosuchus.envs.dibothrosuchus_env", "DibothrosuchusEnv"),
}

#: Legacy stage numbers this script accepts; files resolve via the manifest.
KNOWN_STAGES = frozenset({1, 2, 3})

# Keys the TOML [env] block carries for the MJX path only; the Gymnasium envs
# do not accept them.  Mirrors the JAX-only note in the stage-1 configs.
JAX_ONLY_ENV_KEYS = frozenset({"foot_contact_weight", "foot_contact_gate"})

NOISE_SWEEP = (0.01, 0.05, 0.10, 0.15, 0.20)


def build_env(species: str, stage: int):
    """Instantiate a species env from its committed stage TOML."""
    module_name, class_name = SPECIES_ENVS[species]
    env_class = getattr(importlib.import_module(module_name), class_name)
    from environments.shared.stage_manifest import load_stage_manifest

    entry = load_stage_manifest(species).resolve(stage)
    config_path = Path(_repo_root) / "configs" / species / entry.config_file
    env_section = tomllib.loads(config_path.read_text()).get("env", {})
    kwargs = {
        key: (tuple(value) if isinstance(value, list) else value)
        for key, value in env_section.items()
        if key not in JAX_ONLY_ENV_KEYS
    }
    return env_class(**kwargs)


def score(env, episodes: int, seed: int, noise: float | None = None) -> dict:
    """Roll out the zero action for *episodes* and summarise the outcome."""
    if noise is not None:
        env.reset_noise_scale = noise
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    rewards: list[float] = []
    lengths: list[int] = []
    terminations: dict[str, int] = {}

    for index in range(episodes):
        env.reset(seed=seed + index)
        total, steps = 0.0, 0
        while True:
            _, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            steps += 1
            if terminated or truncated:
                reason = info.get("termination_reason", "truncated")
                terminations[reason] = terminations.get(reason, 0) + 1
                break
        rewards.append(total)
        lengths.append(steps)

    rewards_arr = np.asarray(rewards)
    lengths_arr = np.asarray(lengths)
    horizon = env.max_episode_steps
    standing = lengths_arr >= horizon
    return {
        "reward_mean": float(rewards_arr.mean()),
        "reward_std": float(rewards_arr.std()),
        # The gate a trained policy must clear: a fat failure tail shows up here
        # while it hides in the mean.
        "reward_mean_minus_std": float(rewards_arr.mean() - rewards_arr.std()),
        # Conditioned on the statue not falling over.  This is the number a
        # stage-1 gate has to beat to mean anything, and it is much higher than
        # the unconditional mean: the reward pays per step, so the episodes the
        # do-nothing policy survives are worth far more than its average.  A
        # gate set above ``reward_mean`` but below this one is cleared by a
        # policy that has learned only "do not fall".
        #
        # ``None``, not NaN, when the statue never reaches the horizon: this
        # dict is serialised straight into the baseline records the notebook
        # writes to Drive, and NaN is not valid JSON.
        "reward_mean_standing": float(rewards_arr[standing].mean()) if standing.any() else None,
        "reward_std_standing": float(rewards_arr[standing].std()) if standing.any() else None,
        "n_standing": int(standing.sum()),
        "length_mean": float(lengths_arr.mean()),
        "full_horizon_share": float(standing.mean()),
        "terminations": terminations,
        "horizon": horizon,
        "episodes": int(rewards_arr.size),
        "reset_noise_scale": float(env.reset_noise_scale),
    }


def gate_margin(gate: float | None, floor: float | None) -> float | None:
    """Return the gate's margin over a measured floor when both exist."""
    if gate is None or floor is None:
        return None
    return float(gate - floor)


def _standing_text(result: dict) -> str:
    """Format the standing score, which is unavailable when nothing stood."""
    if result["n_standing"] == 0:
        return "-"
    return f"{result['reward_mean_standing']:.1f}"


def report(species: str, stage: int, episodes: int, seed: int, sweep: bool) -> None:
    env = build_env(species, stage)
    try:
        label = f"{species} stage {stage}"
        if sweep:
            print(f"\n{label}: zero-action baseline vs reset_noise_scale ({episodes} episodes)")
            print(f"  {'noise':>7}{'reward':>12}{'mean-std':>11}{'standing':>11}{'ep length':>11}{'full-horizon':>14}")
            for noise in NOISE_SWEEP:
                result = score(env, episodes, seed, noise=noise)
                print(
                    f"  {noise:>7}{result['reward_mean']:>12.1f}{result['reward_mean_minus_std']:>11.1f}"
                    f"{_standing_text(result):>11}"
                    f"{result['length_mean']:>11.1f}{result['full_horizon_share']:>13.0%}"
                )
            return

        result = score(env, episodes, seed)
        print(f"\n{label}: zero-action baseline ({episodes} episodes, reset_noise={env.reset_noise_scale})")
        print(f"  reward             {result['reward_mean']:.2f} +/- {result['reward_std']:.2f}")
        print(f"  reward mean-std    {result['reward_mean_minus_std']:.2f}")
        standing = (
            "never reached the horizon"
            if result["n_standing"] == 0
            else f"{result['reward_mean_standing']:.2f} +/- {result['reward_std_standing']:.2f}"
        )
        print(
            f"  reward standing    {standing}"
            f"   ({result['n_standing']} of {result['episodes']} episodes reached the horizon)"
        )
        print(f"  episode length     {result['length_mean']:.1f} of {result['horizon']}")
        print(f"  full-horizon share {result['full_horizon_share']:.0%}")
        print(f"  terminations       {result['terminations']}")
        print("  a trained policy must beat all three of reward, mean-std, and full-horizon share")
        print("  and 'reward standing' is what it must beat to have learned more than 'do not fall'")
    finally:
        env.close()


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "targets",
        nargs="*",
        default=["trex", "velociraptor", "brachiosaurus", "dibothrosuchus"],
        help="species, optionally with a stage suffix (e.g. trex:2). Default: all species, stage 1.",
    )
    parser.add_argument("--episodes", type=int, default=30, help="episodes per measurement (default 30)")
    parser.add_argument("--seed", type=int, default=3042, help="first evaluation seed (default 3042)")
    parser.add_argument("--sweep-noise", action="store_true", help="sweep reset_noise_scale instead of one point")
    args = parser.parse_args(argv)

    for target in args.targets:
        species, _, stage_text = target.partition(":")
        if species not in SPECIES_ENVS:
            raise SystemExit(f"unknown species {species!r}; choose from {sorted(SPECIES_ENVS)}")
        stage = int(stage_text) if stage_text else 1
        if stage not in KNOWN_STAGES:
            raise SystemExit(f"unknown stage {stage}; choose from {sorted(KNOWN_STAGES)}")
        report(species, stage, args.episodes, args.seed, args.sweep_noise)


if __name__ == "__main__":
    main(sys.argv[1:])
