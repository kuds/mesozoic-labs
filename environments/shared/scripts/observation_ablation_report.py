#!/usr/bin/env python3
"""Measure how much a trained policy depends on each slice of its observation.

The observation vector is the most expensive thing in this repo to change: per
``configs/plant_versions.toml`` a layout, width or ordering change bumps
``policy_interface_revision`` for *every* species and invalidates every
checkpoint.  Deciding whether a slice is worth keeping therefore needs a
number, not an argument, and the number is "what does the return do when the
policy stops seeing it?".

The motivating case is observation provenance.  Some slices are things a
physical robot could produce from a joint encoder, an IMU or a contact switch;
others are quantities only the simulator knows.  For the T-Rex the second group
is ``qvel[0:3]`` -- the root body's linear velocity in the WORLD frame, which
no sensor produces and which a feedforward policy cannot reconstruct from the
rest of the vector -- and the four world-frame target direction/distance slots.
A policy that scores the same without them can have them removed cheaply; one
that collapses is built on ground truth.

Three interventions, which answer different questions:

* ``mean``   -- replace the slice with the VecNormalize running mean, i.e. set
  it to zero AFTER normalisation.  This removes the *information* while leaving
  the input distribution centred, and is the ablation to quote.  It is what
  "the policy no longer receives this signal" means.
* ``zero``   -- set the slice to 0.0 in RAW units, before normalisation.  For a
  velocity slice this is not an ablation but a lie ("you are stationary"), and
  it is reported separately because the two are easy to confuse.
* ``body``   -- rotate a world-frame 3-vector into the body frame using the
  pelvis quaternion already in the observation.  This does not remove
  information; it asks whether the policy depends on the *frame* rather than
  the quantity, which is the difference between a slot a robot could fill and
  one it could not.

Everything is measured against a ``baseline`` condition that applies no
intervention.  Check the baseline against the run's own recorded evaluation
before trusting any of the deltas -- if the harness does not reproduce the
number in ``stage_summary.txt`` then nothing below it means anything.

Usage::

    python environments/shared/scripts/observation_ablation_report.py trex 1 \\
        --model /path/to/run/stage1/models/robust_best_model.zip

    # The VecNormalize sidecar is found automatically under either naming
    # convention (curated ``<stem>_vecnorm.pkl`` or SB3's periodic
    # ``<prefix>_vecnormalize_<steps>_steps.pkl``); pass --vecnorm to
    # override.  A checkpoint with no sidecar is fatal unless
    # --allow-unnormalized is given.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from environments.shared.config import SPECIES_NAMES, build_env
from environments.shared.constants import PUBLICATION_SEED_START

# Named observation slices per species, as (start, stop, quat_start_or_None).
# quat_start is the index of the pelvis/torso quaternion, needed only by the
# "body" intervention; None means the slice is not a world-frame 3-vector.
# Derived by reading each species' _get_obs, not by assuming a shared layout.
SLICES: dict[str, dict[str, tuple[int, int, int | None]]] = {
    "trex": {
        "root_linvel": (49, 52, 42),
        "target_dir": (57, 60, 42),
        "target_dist": (60, 61, None),
    },
}
# Slices the simulator can supply but a physical robot could not, per species.
# "privileged" ablates all of them at once.
PRIVILEGED_SLICES = {"trex": ("root_linvel", "target_dir", "target_dist")}
UNNORMALIZED_BANNER = "UNNORMALIZED EVAL — results are not comparable to training-time metrics"


def resolve_vecnorm_path(model_path: str, vecnorm_arg: str | None, allow_unnormalized: bool) -> str | None:
    """The VecNormalize sidecar to evaluate with, or ``None`` for a deliberately unnormalised run.

    An explicit ``--vecnorm`` wins.  Otherwise the trainer's own resolver
    probes both sidecar conventions -- the ``<stem>_vecnorm.pkl`` guess this
    script used to make can never match SB3's periodic
    ``<prefix>_vecnormalize_<steps>_steps.pkl``, so every periodic checkpoint
    was silently scored on raw observations.  No sidecar is fatal unless
    ``--allow-unnormalized``: the ``mean`` intervention is defined in
    normalised space, so without statistics every delta below is meaningless.
    """
    if vecnorm_arg is not None:
        return vecnorm_arg
    from environments.shared.train_base import _resolve_vecnorm_sidecar

    candidate = _resolve_vecnorm_sidecar(model_path)
    if Path(candidate).exists():
        return candidate
    if allow_unnormalized:
        return None
    raise SystemExit(
        f"no VecNormalize sidecar found for {model_path} (probed {candidate}). A policy evaluated on "
        "unnormalised observations is a different policy; pass --vecnorm, or --allow-unnormalized "
        "to proceed deliberately."
    )


def quat_to_rotation(quat: np.ndarray) -> np.ndarray:
    """Body->world rotation matrix from a MuJoCo (w, x, y, z) quaternion."""
    w, x, y, z = (float(component) for component in quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def rollout(env, model, normalizer, condition, episodes: int, seed: int) -> dict:
    """Evaluate *model* under one intervention and summarise the outcome."""
    name, mode, spans = condition
    horizon = env.max_episode_steps
    rewards: list[float] = []
    lengths: list[int] = []
    terminations: dict[str, int] = {}

    for index in range(episodes):
        obs, _ = env.reset(seed=seed + index)
        total, steps = 0.0, 0
        while True:
            raw = np.asarray(obs, dtype=np.float64).copy()

            # Raw-space interventions happen before normalisation, because
            # that is where a real sensor would go wrong.
            for start, stop, quat_start in spans:
                if mode == "zero":
                    raw[start:stop] = 0.0
                elif mode == "body":
                    if quat_start is None or stop - start != 3:
                        raise ValueError(f"{name}: 'body' needs a world-frame 3-vector and a quaternion index")
                    rotation = quat_to_rotation(raw[quat_start : quat_start + 4])
                    raw[start:stop] = rotation.T @ raw[start:stop]

            normalized = normalizer.normalize_obs(raw.astype(np.float32)) if normalizer is not None else raw

            # Mean substitution is exactly zero in normalised space, which is
            # why it must be applied here rather than on the raw vector.
            if mode == "mean":
                normalized = np.asarray(normalized, dtype=np.float32).copy()
                for start, stop, _quat in spans:
                    normalized[start:stop] = 0.0

            action, _ = model.predict(normalized, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(np.clip(action, -1.0, 1.0))
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
    return {
        "name": name,
        "reward_mean": float(rewards_arr.mean()),
        "reward_std": float(rewards_arr.std()),
        "length_mean": float(lengths_arr.mean()),
        "full_horizon_share": float((lengths_arr >= horizon).mean()),
        "terminations": terminations,
    }


def build_conditions(species: str, requested: list[str]) -> list[tuple]:
    """Expand the named slices into (label, mode, spans) conditions."""
    table = SLICES[species]
    privileged = [table[key] for key in PRIVILEGED_SLICES[species]]

    conditions: list[tuple] = [("baseline", "none", [])]
    for key in requested:
        if key == "privileged":
            conditions.append(("all privileged -> mean", "mean", privileged))
            continue
        span = table[key]
        conditions.append((f"{key} -> mean", "mean", [span]))
        conditions.append((f"{key} -> zero (raw)", "zero", [span]))
        if span[2] is not None and span[1] - span[0] == 3:
            conditions.append((f"{key} -> body frame", "body", [span]))
    return conditions


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("species", choices=SPECIES_NAMES)
    parser.add_argument("stage", type=int, choices=(1, 2, 3))
    parser.add_argument("--model", required=True, help="SB3 .zip policy")
    parser.add_argument("--vecnorm", default=None, help="VecNormalize .pkl (default: alongside the model)")
    parser.add_argument(
        "--allow-unnormalized",
        action="store_true",
        help="proceed without a VecNormalize sidecar (the report then scores a different policy)",
    )
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument(
        "--seed", type=int, default=PUBLICATION_SEED_START, help="first evaluation seed (publication protocol)"
    )
    parser.add_argument(
        "--slices",
        nargs="*",
        default=["root_linvel", "target_dir", "target_dist", "privileged"],
        help="named slices to ablate",
    )
    args = parser.parse_args(argv)

    if args.species not in SLICES:
        raise SystemExit(f"no observation slice table for {args.species!r}; add one to SLICES")

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    env = build_env(args.species, args.stage)
    model = PPO.load(args.model, device="cpu")

    vecnorm_path = resolve_vecnorm_path(args.model, args.vecnorm, args.allow_unnormalized)
    normalizer = None
    if vecnorm_path:
        normalizer = VecNormalize.load(vecnorm_path, DummyVecEnv([lambda: build_env(args.species, args.stage)]))
        normalizer.training = False
        normalizer.norm_reward = False

    print(f"\n{args.species} stage {args.stage}: {Path(args.model).name}")
    print(f"{args.episodes} episodes, seeds {args.seed}..{args.seed + args.episodes - 1}, deterministic")
    print(f"VecNormalize: {'loaded' if normalizer is not None else 'NONE — ' + UNNORMALIZED_BANNER}\n")

    results = []
    for condition in build_conditions(args.species, args.slices):
        results.append(rollout(env, model, normalizer, condition, args.episodes, args.seed))
    env.close()

    baseline = results[0]["reward_mean"]
    width = max(len(result["name"]) for result in results)
    print(f"  {'condition':<{width}}  {'return':>18}  {'delta':>16}  {'ep len':>8}  {'full-hz':>8}")
    for result in results:
        delta = result["reward_mean"] - baseline
        share = 100.0 * delta / baseline if baseline else float("nan")
        delta_text = "--" if result is results[0] else f"{delta:+9.2f} ({share:+5.1f}%)"
        print(
            f"  {result['name']:<{width}}  {result['reward_mean']:>9.2f} +/- {result['reward_std']:<6.2f}"
            f"  {delta_text:>16}  {result['length_mean']:>8.1f}  {result['full_horizon_share']:>7.0%}"
        )

    print("\nTerminations:")
    for result in results:
        print(f"  {result['name']:<{width}}  {result['terminations']}")

    print(
        "\nQuote the '-> mean' rows: they remove the information while leaving the\n"
        "input centred.  '-> zero (raw)' asserts a false measurement rather than\n"
        "removing one, and '-> body frame' changes the frame without removing\n"
        "anything.  Check 'baseline' against the run's stage_summary.txt before\n"
        "reading any delta."
    )


if __name__ == "__main__":
    main(sys.argv[1:])
