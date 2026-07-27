#!/usr/bin/env python3
"""Report how far the policy's raw actions leave the declared action bound.

``BaseDinoEnv.step`` clips for the physics but not for the reward::

    ctrl = self._scale_action(action)              # clipped to [-1, 1] -> plant
    reward, info = self._get_reward_info(action)   # RAW -> reward

``reward_energy`` is ``sum(action**2) / n`` and ``reward_action_smoothness`` is
``sum((action - prev)**2) / (n * 4)``.  Both normalisers assume the action is
inside ``[-1, 1]`` -- ``reward_energy``'s docstring says so outright -- and
nothing enforces it.  SB3's Gaussian policy is unbounded, so the assumption is
not merely theoretical: T-Rex stage-1 run ``20260727_130726`` logged
``diagnostics/action_abs_max`` averaging 6.11 with a peak of 7.77, and
``diagnostics/action_saturation`` rising monotonically from 0.32 to 0.68 across
6M steps.

This script quantifies the gap for a given policy without changing any
behaviour: it records the raw actions, then evaluates both penalty terms twice
over the *same* trajectory -- once on the raw action as the env does today, and
once on ``clip(action, -1, 1)`` -- and reports the difference.  Use it to size
the confound before deciding whether to clip, and afterwards to confirm how much
of any reward change was the fix rather than the behaviour.

Two modes:

* ``--model`` given: measures a trained checkpoint.  Answers "how much reward is
  this policy being charged for magnitude the plant never sees?"
* ``--model`` omitted: probes with a Gaussian policy of configurable ``--std``,
  which is what an *untrained* policy looks like at initialisation.  Answers
  "would a start-of-training check have caught this?"

  It would, immediately.  SB3's default ``log_std_init = 0`` gives std 1.0, and
  this script at ``--std 1.0`` measures **32.0% of components outside the
  bound** -- which matches the 0.3218 that run ``20260727_130726`` logged to
  ``diagnostics/action_saturation`` at step 8,192, its very first diagnostic
  write.  Nothing has to be learned for the condition to appear: it follows
  from pairing an unbounded Gaussian policy with a ``Box(-1, 1)`` action space,
  so it is present at initialisation and a startup probe is enough to catch it.

Usage::

    python environments/shared/scripts/action_bound_report.py trex 1 \\
        --model /path/to/run/stage1/models/robust_best_model.zip

    python environments/shared/scripts/action_bound_report.py trex 1 --std 1.0
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

from environments.shared.reward_functions import (  # noqa: E402
    reward_action_smoothness,
    reward_energy,
)

SPECIES_ENVS = {
    "trex": ("environments.trex.envs.trex_env", "TRexEnv"),
    "velociraptor": ("environments.velociraptor.envs.raptor_env", "RaptorEnv"),
    "brachiosaurus": ("environments.brachiosaurus.envs.brachio_env", "BrachioEnv"),
    "dibothrosuchus": ("environments.dibothrosuchus.envs.dibothrosuchus_env", "DibothrosuchusEnv"),
}
STAGE_STEMS = {
    1: "stage1_balance",
    2: "stage2_locomotion",
    3: {
        "trex": "stage3_bite",
        "velociraptor": "stage3_strike",
        "brachiosaurus": "stage3_food_reach",
        "dibothrosuchus": "stage3_snap",
    },
}
# Keys the TOML [env] block carries for the MJX path only, as in
# zero_action_baseline.py and joint_excursion_report.py.
JAX_ONLY_ENV_KEYS = frozenset({"foot_contact_weight", "foot_contact_gate"})
# Matches DiagnosticsCallback.action_saturation_threshold, so the "saturated"
# column here is directly comparable to diagnostics/action_saturation.
SATURATION_THRESHOLD = 0.99


def stage_env_section(species: str, stage: int) -> dict:
    stem = STAGE_STEMS[stage]
    if isinstance(stem, dict):
        stem = stem[species]
    config_path = Path(_repo_root) / "configs" / species / f"{stem}.toml"
    # Annotated local: tomllib returns Any, and mypy will not carry the
    # declared return type back through a bare return of it.
    section: dict = tomllib.loads(config_path.read_text()).get("env", {})
    return section


def build_env(species: str, stage: int):
    module_name, class_name = SPECIES_ENVS[species]
    env_class = getattr(importlib.import_module(module_name), class_name)
    kwargs = {
        key: (tuple(value) if isinstance(value, list) else value)
        for key, value in stage_env_section(species, stage).items()
        if key not in JAX_ONLY_ENV_KEYS
    }
    return env_class(**kwargs)


def penalties(actions: list[np.ndarray], n_actuators: int, energy_w: float, smooth_w: float):
    """Total energy and smoothness penalty over one episode's actions."""
    energy = 0.0
    smooth = 0.0
    prev = None
    for action in actions:
        energy += float(reward_energy(action, n_actuators, energy_w))
        reward, _ = reward_action_smoothness(action, prev, n_actuators, smooth_w)
        smooth += float(reward)
        prev = action
    return energy, smooth


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("species", choices=sorted(SPECIES_ENVS))
    parser.add_argument("stage", type=int, choices=(1, 2, 3))
    parser.add_argument("--model", default=None, help="SB3 .zip policy; omit to probe an untrained Gaussian")
    parser.add_argument("--vecnorm", default=None, help="VecNormalize .pkl (default: alongside the model)")
    parser.add_argument("--std", type=float, default=1.0, help="Gaussian std when --model is omitted")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=3042)
    args = parser.parse_args(argv)

    env = build_env(args.species, args.stage)
    n_actuators = int(env.action_space.shape[0])
    env_section = stage_env_section(args.species, args.stage)
    energy_w = float(env_section.get("energy_penalty_weight", env.energy_penalty_weight))
    smooth_w = float(env_section.get("smoothness_weight", env.smoothness_weight))

    model = None
    normalizer = None
    if args.model:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        model = PPO.load(args.model, device="cpu")
        vecnorm_path = args.vecnorm
        if vecnorm_path is None:
            guess = Path(args.model).with_name(Path(args.model).stem + "_vecnorm.pkl")
            vecnorm_path = str(guess) if guess.exists() else None
        if vecnorm_path:
            normalizer = VecNormalize.load(vecnorm_path, DummyVecEnv([lambda: build_env(args.species, args.stage)]))
            normalizer.training = False
            normalizer.norm_reward = False

    rng = np.random.default_rng(args.seed)
    raw_energy = raw_smooth = clip_energy = clip_smooth = 0.0
    returns: list[float] = []
    all_actions: list[np.ndarray] = []

    for index in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + index)
        episode_actions: list[np.ndarray] = []
        total = 0.0
        while True:
            if model is None:
                action = rng.normal(0.0, args.std, size=n_actuators).astype(np.float32)
            else:
                obs_input = normalizer.normalize_obs(obs) if normalizer is not None else obs
                action, _ = model.predict(obs_input, deterministic=True)
            action = np.asarray(action, dtype=np.float64).reshape(-1)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_actions.append(action)
            all_actions.append(action)
            total += float(reward)
            if terminated or truncated:
                break
        returns.append(total)
        e_raw, s_raw = penalties(episode_actions, n_actuators, energy_w, smooth_w)
        clipped = [np.clip(a, -1.0, 1.0) for a in episode_actions]
        e_clip, s_clip = penalties(clipped, n_actuators, energy_w, smooth_w)
        raw_energy += e_raw
        raw_smooth += s_raw
        clip_energy += e_clip
        clip_smooth += s_clip

    env.close()

    acts = np.concatenate(all_actions)
    n_ep = args.episodes
    source = f"policy {Path(args.model).name}" if args.model else f"untrained Gaussian, std={args.std}"

    print(f"\n{args.species} stage {args.stage}: {source}")
    print(f"{len(all_actions)} steps over {n_ep} episodes, {n_actuators} actuators")
    print(f"mean return {np.mean(returns):.2f} +/- {np.std(returns):.2f}\n")

    print("Raw action statistics (the action space is Box(-1, 1)):")
    print(f"  mean                {acts.mean():+.4f}")
    print(f"  std                 {acts.std():.4f}")
    print(f"  abs max             {np.abs(acts).max():.4f}")
    print(f"  E[a^2]              {np.mean(acts**2):.4f}     (<= 1.0 if the bound held)")
    print(f"  |a| > 1             {np.mean(np.abs(acts) > 1.0) * 100:.1f}% of components")
    print(
        f"  |a| >= {SATURATION_THRESHOLD}          {np.mean(np.abs(acts) >= SATURATION_THRESHOLD) * 100:.1f}%"
        "  (comparable to diagnostics/action_saturation)\n"
    )

    print(f"{'penalty':<14}{'raw (as shipped)':>20}{'on clipped action':>20}{'difference':>14}")
    for label, raw, clip in (("energy", raw_energy, clip_energy), ("smoothness", raw_smooth, clip_smooth)):
        print(f"{label:<14}{raw / n_ep:>20.2f}{clip / n_ep:>20.2f}{(raw - clip) / n_ep:>14.2f}")
    total_delta = (raw_energy + raw_smooth - clip_energy - clip_smooth) / n_ep
    print(
        f"{'TOTAL':<14}{(raw_energy + raw_smooth) / n_ep:>20.2f}"
        f"{(clip_energy + clip_smooth) / n_ep:>20.2f}{total_delta:>14.2f}"
    )

    mean_return = float(np.mean(returns))
    print(f"\nPer episode, clipping before the reward would change the return by {-total_delta:+.2f}")
    if mean_return > 1.0:
        print(f"  = {abs(total_delta) / mean_return * 100:.1f}% of the {mean_return:.2f} mean return")
    else:
        print(f"  (share of return not reported: the mean return is {mean_return:.2f}, so the ratio is")
        print("   meaningless.  Expected in --std probe mode, where the policy falls immediately.)")
    print("\nThe zero-action baseline is unaffected either way: at action = 0 both")
    print("penalties are identically zero, so min_avg_reward does not need re-deriving.")


if __name__ == "__main__":
    main(sys.argv[1:])
