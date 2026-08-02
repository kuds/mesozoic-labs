"""Score a saved checkpoint against its stage's ``stance_quality/v1`` gate.

Answers one question: **would this policy advance?** It rolls the checkpoint
on the stage's own configured environment, reduces the episodes with the same
:mod:`~environments.shared.curriculum.stance_gate` code the training loop runs,
and prints the verdict criterion by criterion.

Why this exists as a script rather than a notebook cell: the gate's verdict is
the thing runs are judged on, and a verdict computed by hand-rolled
reimplementation is not the same verdict.  Everything here funnels through
``summarize_stance_panel`` and ``evaluate_stance_gate``, so what this prints is
what the curriculum would decide.

It also covers the case the training loop cannot: **an already-finished run**.
Run ``20260801_203206`` completed stage 1 under the old ``reward_and_length/v1``
gate and advanced on reward 2313.2 and length 1000 -- both of which a
zero-action statue clears.  Whether it would have passed the stance gate is a
question only a rollout can answer, and the checkpoint is a 4 MB binary that
lives next to the run, not in the repository.

Usage::

    python environments/shared/scripts/stance_gate_report.py trex \\
        --model  path/to/robust_best_model.zip \\
        --vecnorm path/to/robust_best_model_vecnorm.pkl

    # the do-nothing reference, for the same panel and seeds
    python environments/shared/scripts/stance_gate_report.py trex --zero-action

Thresholds come from the stage TOML, so the report re-anchors automatically
when the gate is retuned.  ``--episodes`` defaults to the stage's own
``min_eval_episodes``, because that is the panel size the bound's power is
specified at; overriding it downward makes the bound weaker than the gate
claims, and the report says so.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from environments.shared.config import load_stage_config  # noqa: E402
from environments.shared.curriculum.stance_gate import (  # noqa: E402
    STANCE_GATE_KIND,
    StanceGateThresholds,
    evaluate_stance_gate,
    stance_panel_from_episode_duties,
)
from environments.shared.species_registry import SPECIES_FACTORIES  # noqa: E402
from environments.shared.stance_diagnostics import derive_stance_info  # noqa: E402


def _load_policy(model_path: str, vecnorm_path: str | None, env_factory: Any):
    """Return ``(predict_fn, describe)`` for a saved SB3 checkpoint.

    Observation normalisation is applied from the saved ``VecNormalize``
    statistics rather than re-estimated: a policy evaluated on unnormalised
    observations is a *different policy*, and would score arbitrarily badly
    for reasons that have nothing to do with stance.  That is why a missing
    or unreadable file is fatal here instead of a warning.

    Loaded through ``VecNormalize.load`` with a throwaway ``DummyVecEnv``,
    matching the sibling report scripts -- the loader needs a live env to
    rebuild the wrapper, and hand-unpickling reconstructs a partial object
    whose ``__setstate__`` expectations drift with the SB3 version.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    model = PPO.load(model_path, device="cpu")

    if vecnorm_path is None:
        return (lambda obs: model.predict(obs, deterministic=True)[0]), "no obs normalisation"

    try:
        normalizer = VecNormalize.load(vecnorm_path, DummyVecEnv([env_factory]))
    except Exception as exc:  # noqa: BLE001 - the message matters more than the type
        # A truncated or text-mode-copied .pkl is the usual cause, and the raw
        # UnpicklingError/KeyError gives no hint that the file rather than the
        # code is at fault.
        raise SystemExit(
            f"cannot read VecNormalize statistics from {vecnorm_path}: "
            f"{type(exc).__name__}: {exc}. Re-copy the file in binary mode. "
            "Running without --vecnorm would evaluate the policy on unnormalised "
            "observations — a different policy — so this is fatal, not a warning."
        ) from exc

    obs_rms = normalizer.obs_rms
    if isinstance(obs_rms, dict):
        # SB3 keeps a per-key RunningMeanStd for Dict observation spaces. Every
        # species here exposes a flat Box, so a dict means the checkpoint was
        # trained against a different observation contract than the stage
        # config builds — normalising with the wrong statistics would silently
        # score a different policy.
        raise SystemExit(
            f"{vecnorm_path} holds per-key statistics for a Dict observation space "
            f"(keys: {sorted(obs_rms)}), but this stage builds a flat Box observation."
        )
    mean = np.asarray(obs_rms.mean, dtype=np.float64)
    var = np.asarray(obs_rms.var, dtype=np.float64)
    epsilon = float(normalizer.epsilon)
    clip = float(normalizer.clip_obs)

    def predict(obs: np.ndarray) -> np.ndarray:
        normalized = np.clip((obs - mean) / np.sqrt(var + epsilon), -clip, clip).astype(np.float32)
        # np.asarray, not a bare return: SB3's predict() is untyped when
        # stable-baselines3 is absent (as it is in the lint job), so the value
        # is Any there and typed here. Coercing makes the annotation true in
        # both environments instead of only the one that happens to run mypy.
        return np.asarray(model.predict(normalized, deterministic=True)[0])

    return predict, f"VecNormalize stats applied (clip {clip:g})"


def run_panel(
    species: str,
    stage: int,
    *,
    predict: Any,
    episodes: int,
    seed: int,
    settle_steps: int,
    horizon: int,
    env_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Roll ``episodes`` deterministic episodes and reduce them for the gate."""
    env_class = SPECIES_FACTORIES[species]().env_class
    env = env_class(**env_kwargs)

    lengths: list[float] = []
    rewards: list[float] = []
    duties: list[float] = []
    bilateral_duties: list[float] = []
    single_duties: list[float] = []
    terminations: dict[str, int] = {}
    # Per-term totals, so a failing panel can be read for *why*. A policy that
    # keeps paying the airborne penalty is being funded by some other term;
    # which one is not guessable from the aggregate score.
    components: dict[str, float] = {}

    for index in range(episodes):
        obs, _ = env.reset(seed=seed + index)
        total = 0.0
        steps = 0
        unsupported = 0
        bilateral = 0
        single = 0
        measured = 0
        while True:
            obs, reward, terminated, truncated, info = env.step(predict(obs))
            total += float(reward)
            for key, value in info.items():
                if key.startswith("reward_") and key != "reward_total":
                    components[key] = components.get(key, 0.0) + float(value)
            stance = derive_stance_info(info)
            if stance and steps >= settle_steps:
                measured += 1
                unsupported += int(stance["unsupported_duty"])
                # The gate bounds unsupported duty only. Reporting the full
                # three-way split shows where recovered airborne time actually
                # went: a policy that trades flight for SINGLE support drives
                # the gated number down without ever planting both feet.
                bilateral += int(stance["bilateral_support_duty"])
                single += int(stance["single_support_duty"])
            steps += 1
            if terminated or truncated:
                reason = info.get("termination_reason", "terminated" if terminated else "truncated")
                terminations[reason] = terminations.get(reason, 0) + 1
                break
        lengths.append(float(steps))
        rewards.append(total)
        duties.append(unsupported / measured if measured else float("nan"))
        if measured:
            bilateral_duties.append(bilateral / measured)
            single_duties.append(single / measured)

    panel = stance_panel_from_episode_duties(
        episode_lengths=lengths,
        episode_duties=duties,
        episode_rewards=rewards,
        horizon=horizon,
    )
    return {
        "panel": panel,
        "lengths": np.asarray(lengths),
        "rewards": np.asarray(rewards),
        "terminations": terminations,
        "components": {key: value / episodes for key, value in components.items()},
        "bilateral_duty": float(np.mean(bilateral_duties)) if bilateral_duties else float("nan"),
        "single_duty": float(np.mean(single_duties)) if single_duties else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("species", help="Species id, e.g. trex")
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--model", help="Path to a saved SB3 checkpoint (.zip)")
    parser.add_argument("--vecnorm", help="Path to the checkpoint's VecNormalize .pkl")
    parser.add_argument(
        "--zero-action",
        action="store_true",
        help="Score the do-nothing policy instead of a checkpoint (the reference the gate is calibrated against)",
    )
    parser.add_argument("--episodes", type=int, default=None, help="Defaults to the stage's min_eval_episodes")
    parser.add_argument("--seed", type=int, default=3042, help="First evaluation seed (default: publication seed)")
    parser.add_argument("--config", help="Explicit stage TOML path")
    args = parser.parse_args()

    if not args.zero_action and not args.model:
        parser.error("pass --model, or --zero-action for the do-nothing reference")

    stage_config = load_stage_config(args.species, args.stage, config_path=args.config)
    curriculum = stage_config["curriculum_kwargs"]
    env_kwargs = dict(stage_config["env_kwargs"])
    horizon = int(env_kwargs.get("max_episode_steps", 1000))

    gate_kind = curriculum.get("gate_kind")
    if gate_kind != STANCE_GATE_KIND:
        print(f"NOTE: {args.species} stage {args.stage} declares gate_kind {gate_kind!r}, not {STANCE_GATE_KIND!r}.")
        print("      The stance criteria below are reported but are not what this stage advances on.")

    thresholds = StanceGateThresholds(
        min_full_horizon_fraction=float(curriculum.get("min_full_horizon_fraction", 0.0)),
        max_unsupported_duty=float(curriculum.get("max_unsupported_duty", float("inf"))),
        max_unsupported_duty_ucb=float(curriculum.get("max_unsupported_duty_ucb", float("inf"))),
        settle_steps=int(curriculum.get("settle_steps", 0)),
        min_eval_episodes=int(curriculum.get("min_eval_episodes", 40)),
        min_avg_reward=float(curriculum.get("min_avg_reward", -float("inf"))),
        required_consecutive=int(curriculum.get("required_consecutive", 3)),
    )
    episodes = args.episodes or thresholds.min_eval_episodes

    env_class = SPECIES_FACTORIES[args.species]().env_class

    if args.zero_action:
        probe = env_class(**env_kwargs)
        zero = np.zeros(probe.action_space.shape[0], dtype=np.float32)
        probe.close()

        def predict(_obs: np.ndarray) -> np.ndarray:
            return zero

        description = "zero action (do-nothing reference)"
    else:
        predict, description = _load_policy(args.model, args.vecnorm, lambda: env_class(**env_kwargs))
        description = f"{Path(args.model).name} — {description}"

    result = run_panel(
        args.species,
        args.stage,
        predict=predict,
        episodes=episodes,
        seed=args.seed,
        settle_steps=thresholds.settle_steps,
        horizon=horizon,
        env_kwargs=env_kwargs,
    )
    panel = result["panel"]
    passed, failures = evaluate_stance_gate(panel, thresholds)

    print()
    print(f"policy              {description}")
    print(f"stage               {args.species} stage {args.stage} ({gate_kind})")
    print(f"panel               {episodes} episodes, seeds {args.seed}-{args.seed + episodes - 1}")
    print(
        f"settle_steps        {thresholds.settle_steps} (duty measured over the remaining {horizon - thresholds.settle_steps})"
    )
    print()
    print(f"reward                 {result['rewards'].mean():9.1f} +/- {result['rewards'].std():.1f}")
    print(f"episode length         {result['lengths'].mean():9.1f}")
    print(
        f"full_horizon_fraction  {panel.full_horizon_fraction:9.4f}   (>= {thresholds.min_full_horizon_fraction:.4f})"
    )
    print(f"mean_unsupported_duty  {panel.mean_unsupported_duty:9.4f}   (<= {thresholds.max_unsupported_duty:.4f})")
    print(f"unsupported_duty_ucb   {panel.unsupported_duty_ucb:9.4f}   (<= {thresholds.max_unsupported_duty_ucb:.4f})")
    print(f"duty episodes          {panel.n_duty_episodes:9d}   (full-horizon episodes only)")
    # Not gated, but decisive for reading a falling unsupported duty: the
    # three shares sum to 1, so a policy can cut flight without ever planting
    # both feet.
    print(f"  bilateral support    {result['bilateral_duty']:9.4f}   (statue 0.998, not gated)")
    print(f"  single support       {result['single_duty']:9.4f}   (statue 0.002, not gated)")
    print(f"terminations           {result['terminations']}")
    components = result["components"]
    if components:
        print()
        print("reward per episode, by term (largest magnitude first):")
        for key in sorted(components, key=lambda k: -abs(components[k])):
            value = components[key]
            if abs(value) > 0.05:
                print(f"  {key:28s} {value:10.2f}")

    print()
    print(f"GATE: {'PASS' if passed else 'FAIL'}")
    for failure in failures:
        print(f"  - {failure}")
    if args.episodes and args.episodes != thresholds.min_eval_episodes:
        print()
        print(
            f"WARNING: --episodes {args.episodes} differs from the stage's min_eval_episodes "
            f"{thresholds.min_eval_episodes}. The bound's power is specified at the latter; "
            "this panel does not certify what the gate claims."
        )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
