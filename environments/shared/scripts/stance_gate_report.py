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
import json
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


#: Bumped when the JSON report's field meanings change.
REPORT_SCHEMA = "mesozoic.stance-gate-report/v1"


def thresholds_from_curriculum(curriculum: dict[str, Any]) -> StanceGateThresholds:
    """Read the stance thresholds a stage declares.

    Ceilings default to ``+inf`` and floors to ``0.0`` so a stage that does not
    gate on stance still produces a readable report rather than a spuriously
    strict one -- ``0.0`` would be the tightest possible ceiling, not "absent".
    """
    return StanceGateThresholds(
        min_full_horizon_fraction=float(curriculum.get("min_full_horizon_fraction", 0.0)),
        max_unsupported_duty=float(curriculum.get("max_unsupported_duty", float("inf"))),
        max_unsupported_duty_ucb=float(curriculum.get("max_unsupported_duty_ucb", float("inf"))),
        settle_steps=int(curriculum.get("settle_steps", 0)),
        min_eval_episodes=int(curriculum.get("min_eval_episodes", 40)),
        min_avg_reward=float(curriculum.get("min_avg_reward", -float("inf"))),
        required_consecutive=int(curriculum.get("required_consecutive", 3)),
    )


def build_stance_gate_report(
    species: str,
    stage: int,
    *,
    stage_config: dict[str, Any],
    model_path: str | None = None,
    vecnorm_path: str | None = None,
    zero_action: bool = False,
    episodes: int | None = None,
    seed: int = 3042,
) -> dict[str, Any]:
    """Roll a policy and return its gate verdict as a serializable dict.

    Importable so the training pipeline can emit the same report it would
    produce by hand, rather than a second implementation that could disagree.
    """
    curriculum = stage_config["curriculum_kwargs"]
    env_kwargs = dict(stage_config["env_kwargs"])
    horizon = int(env_kwargs.get("max_episode_steps", 1000))
    thresholds = thresholds_from_curriculum(curriculum)
    panel_episodes = episodes or thresholds.min_eval_episodes

    env_class = SPECIES_FACTORIES[species]().env_class
    if zero_action:
        probe = env_class(**env_kwargs)
        zero = np.zeros(probe.action_space.shape[0], dtype=np.float32)
        probe.close()

        def predict(_obs: np.ndarray) -> np.ndarray:
            return zero

        description = "zero action (do-nothing reference)"
    else:
        if model_path is None:
            raise ValueError("model_path is required unless zero_action is set")
        predict, note = _load_policy(model_path, vecnorm_path, lambda: env_class(**env_kwargs))
        description = f"{Path(model_path).name} — {note}"

    result = run_panel(
        species,
        stage,
        predict=predict,
        episodes=panel_episodes,
        seed=seed,
        settle_steps=thresholds.settle_steps,
        horizon=horizon,
        env_kwargs=env_kwargs,
    )
    panel = result["panel"]
    passed, failures = evaluate_stance_gate(panel, thresholds)

    return {
        "schema": REPORT_SCHEMA,
        "species": species,
        "stage": stage,
        "gate_kind": curriculum.get("gate_kind"),
        "policy": description,
        "episodes": panel_episodes,
        "seed": seed,
        "settle_steps": thresholds.settle_steps,
        "horizon": horizon,
        "passed": passed,
        "failures": failures,
        "thresholds": {
            "min_full_horizon_fraction": thresholds.min_full_horizon_fraction,
            "max_unsupported_duty": thresholds.max_unsupported_duty,
            "max_unsupported_duty_ucb": thresholds.max_unsupported_duty_ucb,
            "min_avg_reward": thresholds.min_avg_reward,
            "min_eval_episodes": thresholds.min_eval_episodes,
        },
        "metrics": {
            "reward_mean": float(result["rewards"].mean()),
            "reward_std": float(result["rewards"].std()),
            "episode_length_mean": float(result["lengths"].mean()),
            "full_horizon_fraction": panel.full_horizon_fraction,
            "mean_unsupported_duty": panel.mean_unsupported_duty,
            "unsupported_duty_ucb": panel.unsupported_duty_ucb,
            "n_duty_episodes": panel.n_duty_episodes,
            # Not gated. Reported because the three shares sum to 1, so a
            # falling unsupported duty does not by itself mean the feet are
            # being planted -- the time can land in single support instead.
            "bilateral_support_duty": result["bilateral_duty"],
            "single_support_duty": result["single_duty"],
        },
        "terminations": result["terminations"],
        "reward_components": result["components"],
    }


def render_stance_gate_report(report: dict[str, Any]) -> str:
    """Render a report dict as the human-readable text form."""
    thresholds = report["thresholds"]
    metrics = report["metrics"]
    lines: list[str] = []
    if report["gate_kind"] != STANCE_GATE_KIND:
        lines.append(
            f"NOTE: {report['species']} stage {report['stage']} declares gate_kind "
            f"{report['gate_kind']!r}, not {STANCE_GATE_KIND!r}. The stance criteria "
            "below are reported but are not what this stage advances on."
        )
        lines.append("")
    measured = report["horizon"] - report["settle_steps"]
    lines += [
        f"policy              {report['policy']}",
        f"stage               {report['species']} stage {report['stage']} ({report['gate_kind']})",
        f"panel               {report['episodes']} episodes, "
        f"seeds {report['seed']}-{report['seed'] + report['episodes'] - 1}",
        f"settle_steps        {report['settle_steps']} (duty measured over the remaining {measured})",
        "",
        f"reward                 {metrics['reward_mean']:9.1f} +/- {metrics['reward_std']:.1f}",
        f"episode length         {metrics['episode_length_mean']:9.1f}",
        f"full_horizon_fraction  {metrics['full_horizon_fraction']:9.4f}   "
        f"(>= {thresholds['min_full_horizon_fraction']:.4f})",
        f"mean_unsupported_duty  {metrics['mean_unsupported_duty']:9.4f}   "
        f"(<= {thresholds['max_unsupported_duty']:.4f})",
        f"unsupported_duty_ucb   {metrics['unsupported_duty_ucb']:9.4f}   "
        f"(<= {thresholds['max_unsupported_duty_ucb']:.4f})",
        f"duty episodes          {metrics['n_duty_episodes']:9d}   (full-horizon episodes only)",
        f"  bilateral support    {metrics['bilateral_support_duty']:9.4f}   (statue 0.998, not gated)",
        f"  single support       {metrics['single_support_duty']:9.4f}   (statue 0.002, not gated)",
        f"terminations           {report['terminations']}",
    ]
    components = report["reward_components"]
    if components:
        lines += ["", "reward per episode, by term (largest magnitude first):"]
        for key in sorted(components, key=lambda k: -abs(components[k])):
            if abs(components[key]) > 0.05:
                lines.append(f"  {key:28s} {components[key]:10.2f}")
    lines += ["", f"GATE: {'PASS' if report['passed'] else 'FAIL'}"]
    lines += [f"  - {failure}" for failure in report["failures"]]
    return "\n".join(lines)


def write_stance_gate_report(stage_dir: "str | Path", report: dict[str, Any]) -> dict[str, Path]:
    """Write the report beside the stage's other artifacts.

    Two forms on purpose: the text is what a human opens next to
    ``stage_summary.txt``, and the JSON is what later tooling can read without
    parsing prose -- including the per-episode duty evidence the publication
    gate currently refuses stance-gated bundles for lacking.
    """
    directory = Path(stage_dir)
    directory.mkdir(parents=True, exist_ok=True)
    text_path = directory / "stance_gate_report.txt"
    json_path = directory / "stance_gate_report.json"
    text_path.write_text(render_stance_gate_report(report) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"stance_gate_report_txt": text_path, "stance_gate_report_json": json_path}


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
    parser.add_argument("--out-dir", help="Also write stance_gate_report.{txt,json} to this directory")
    args = parser.parse_args()

    if not args.zero_action and not args.model:
        parser.error("pass --model, or --zero-action for the do-nothing reference")

    stage_config = load_stage_config(args.species, args.stage, config_path=args.config)
    report = build_stance_gate_report(
        args.species,
        args.stage,
        stage_config=stage_config,
        model_path=args.model,
        vecnorm_path=args.vecnorm,
        zero_action=args.zero_action,
        episodes=args.episodes,
        seed=args.seed,
    )
    print()
    print(render_stance_gate_report(report))

    if args.out_dir:
        for path in write_stance_gate_report(args.out_dir, report).values():
            print(f"written: {path}")

    declared = report["thresholds"]["min_eval_episodes"]
    if args.episodes and args.episodes != declared:
        print()
        print(
            f"WARNING: --episodes {args.episodes} differs from the stage's min_eval_episodes "
            f"{declared}. The bound's power is specified at the latter; this panel does not "
            "certify what the gate claims."
        )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
