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
import csv
import json
import math
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


class StanceGateReportError(RuntimeError):
    """The report could not be produced for a reason the caller should see.

    Deliberately a plain ``Exception`` subclass rather than ``SystemExit``,
    which is what these paths used to raise.  ``SystemExit`` derives from
    ``BaseException``, so it sailed straight through the ``except Exception``
    guard in ``reporting.stage_artifacts._write_stance_gate_report`` that
    exists precisely so a diagnostic cannot sink a finished training run --
    a truncated VecNormalize ``.pkl`` aborted artifact generation before the
    graphs and videos were written.  ``main`` converts this to ``SystemExit``
    for CLI use, which is where that behaviour belongs.
    """


def _load_policy(
    model_path: str,
    vecnorm_path: str | None,
    env_factory: Any,
    *,
    plant_identity: Any = None,
    allow_legacy_plant: bool = False,
):
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
    if plant_identity is not None:
        from environments.shared.plant_contract import validate_model_plant

        validate_model_plant(model, plant_identity, artifact=model_path, allow_legacy=allow_legacy_plant)

    if vecnorm_path is None:
        return (lambda obs: model.predict(obs, deterministic=True)[0]), "no obs normalisation"

    try:
        normalizer = VecNormalize.load(vecnorm_path, DummyVecEnv([env_factory]))
    except Exception as exc:  # noqa: BLE001 - the message matters more than the type
        # A truncated or text-mode-copied .pkl is the usual cause, and the raw
        # UnpicklingError/KeyError gives no hint that the file rather than the
        # code is at fault.
        raise StanceGateReportError(
            f"cannot read VecNormalize statistics from {vecnorm_path}: "
            f"{type(exc).__name__}: {exc}. Re-copy the file in binary mode. "
            "Running without --vecnorm would evaluate the policy on unnormalised "
            "observations — a different policy — so this is fatal, not a warning."
        ) from exc

    if plant_identity is not None:
        from environments.shared.plant_contract import validate_model_plant

        validate_model_plant(normalizer, plant_identity, artifact=vecnorm_path, allow_legacy=allow_legacy_plant)

    obs_rms = normalizer.obs_rms
    if isinstance(obs_rms, dict):
        # SB3 keeps a per-key RunningMeanStd for Dict observation spaces. Every
        # species here exposes a flat Box, so a dict means the checkpoint was
        # trained against a different observation contract than the stage
        # config builds — normalising with the wrong statistics would silently
        # score a different policy.
        raise StanceGateReportError(
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
    *,
    predict: Any,
    episodes: int,
    seed: int,
    settle_steps: int,
    horizon: int,
    env_kwargs: dict[str, Any],
    plant_identity: Any = None,
) -> dict[str, Any]:
    """Roll ``episodes`` deterministic episodes and reduce them for the gate.

    The rollout environment is validated against *plant_identity* when one is
    supplied, and always closed -- a MuJoCo env holds native handles, and the
    artifact path builds one of these per stage.

    Returns the panel plus the per-episode evidence it was reduced from, so a
    caller can serialize the measurements rather than only their summary.
    """
    env_class = SPECIES_FACTORIES[species]().env_class
    env = env_class(**env_kwargs)
    if plant_identity is not None:
        from environments.shared.plant_contract import validate_environment_plant

        try:
            validate_environment_plant(env, plant_identity, artifact=f"{species} stance gate report environment")
        except BaseException:
            env.close()
            raise

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

    try:
        _roll_episodes(
            env,
            predict=predict,
            episodes=episodes,
            seed=seed,
            settle_steps=settle_steps,
            lengths=lengths,
            rewards=rewards,
            duties=duties,
            bilateral_duties=bilateral_duties,
            single_duties=single_duties,
            terminations=terminations,
            components=components,
        )
    finally:
        env.close()

    panel = stance_panel_from_episode_duties(
        episode_lengths=lengths,
        episode_duties=duties,
        episode_rewards=rewards,
        horizon=horizon,
    )

    def _full_horizon_mean(values: list[float]) -> float:
        """Mean over the same episodes the gate measures duty on.

        The ungated shares used to be averaged over every episode that
        measured anything, while the gated duty uses full-horizon episodes
        only. The three then did not sum to 1 whenever an episode failed
        early -- measured 0.90 with 5 failures in 40 -- which is exactly the
        identity the report cites as the reason for printing bilateral at
        all. Worse, the drift ran in the direction that hides the problem:
        the flailing episodes were folded into bilateral and single but
        excluded from unsupported.
        """
        kept = [value for value, length in zip(values, lengths) if length >= horizon and not np.isnan(value)]
        return float(np.mean(kept)) if kept else float("nan")

    return {
        "panel": panel,
        "lengths": np.asarray(lengths),
        "rewards": np.asarray(rewards),
        "terminations": terminations,
        "components": {key: value / episodes for key, value in components.items()},
        "bilateral_duty": _full_horizon_mean(bilateral_duties),
        "single_duty": _full_horizon_mean(single_duties),
        # The measurements the panel was reduced from. Kept rather than
        # discarded so the report can serialize the evidence, not only its
        # summary: an aggregate cannot be re-checked, and duty is the one
        # column the publication evidence CSV does not carry.
        "episodes": [
            {
                "episode": index,
                "seed": seed + index,
                "length": length,
                "reward": reward,
                "reached_horizon": length >= horizon,
                "unsupported_duty": duty,
                "bilateral_support_duty": bilateral,
                "single_support_duty": single,
            }
            for index, (length, reward, duty, bilateral, single) in enumerate(
                zip(lengths, rewards, duties, bilateral_duties, single_duties)
            )
        ],
    }


def _roll_episodes(
    env: Any,
    *,
    predict: Any,
    episodes: int,
    seed: int,
    settle_steps: int,
    lengths: list[float],
    rewards: list[float],
    duties: list[float],
    bilateral_duties: list[float],
    single_duties: list[float],
    terminations: dict[str, int],
    components: dict[str, float],
) -> None:
    """Roll the panel, appending per-episode measurements to the given lists.

    Split out only so ``run_panel`` can wrap it in the ``finally`` that closes
    the environment; the accumulation is unchanged.
    """
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
        # All three shares stay positionally aligned with `lengths`, so the
        # full-horizon filter below applies to them identically.
        duties.append(unsupported / measured if measured else float("nan"))
        bilateral_duties.append(bilateral / measured if measured else float("nan"))
        single_duties.append(single / measured if measured else float("nan"))


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
    allow_legacy_plant: bool = False,
) -> dict[str, Any]:
    """Roll a policy and return its gate verdict as a serializable dict.

    Importable so the training pipeline can emit the same report it would
    produce by hand, rather than a second implementation that could disagree.

    The checkpoint and the rollout environment are validated against the
    species' current plant identity: a verdict measured on a different plant
    than the one in the tree is not a verdict about this stage, and every
    other artifact path in the repository already refuses that pairing. Pass
    *allow_legacy_plant* to score a checkpoint that predates the contract --
    which is a real use for this script, since it exists partly to judge
    already-finished runs -- and the report records that it was done.

    ``episodes`` of ``None`` means the stage's own ``min_eval_episodes``,
    the panel size the bound's power is specified at. ``0`` is rejected
    rather than silently treated as absent.
    """
    if episodes is not None and episodes < 1:
        raise ValueError(f"episodes must be at least 1 if given, got {episodes}")
    curriculum = stage_config["curriculum_kwargs"]
    env_kwargs = dict(stage_config["env_kwargs"])
    horizon = int(env_kwargs.get("max_episode_steps", 1000))
    thresholds = thresholds_from_curriculum(curriculum)
    panel_episodes = thresholds.min_eval_episodes if episodes is None else episodes

    from environments.shared.plant_contract import current_plant_identity

    plant_identity = current_plant_identity(species)

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
        predict, note = _load_policy(
            model_path,
            vecnorm_path,
            lambda: env_class(**env_kwargs),
            plant_identity=plant_identity,
            allow_legacy_plant=allow_legacy_plant,
        )
        description = f"{Path(model_path).name} — {note}"

    result = run_panel(
        species,
        predict=predict,
        episodes=panel_episodes,
        seed=seed,
        settle_steps=thresholds.settle_steps,
        horizon=horizon,
        env_kwargs=env_kwargs,
        plant_identity=plant_identity,
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
        # Whether the CHECKPOINT's plant identity was checked against the
        # tree. `None` for --zero-action, which has no checkpoint to check.
        # The rollout ENVIRONMENT is always validated regardless, so this
        # field is deliberately narrow rather than a blanket "plant_validated"
        # that would claim something false in the zero-action case.
        "checkpoint_plant_validated": None if zero_action else not allow_legacy_plant,
        "terminations": result["terminations"],
        "reward_components": result["components"],
        # The per-episode measurements the panel was reduced from. This is
        # the duty evidence `result_bundle.evidence` refuses stance-gated
        # bundles for lacking -- note it does NOT by itself lift that
        # refusal, which reads `evaluation_selected.csv`; teaching that
        # function to consume this is the remaining step.
        "episode_evidence": result["episodes"],
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


def _json_safe(value: Any) -> Any:
    """Replace non-finite floats with ``null``, recursively.

    ``json.dumps`` writes bare ``NaN`` and ``Infinity`` tokens, which Python
    reads back but RFC 8259 does not permit -- ``jq``, ``JSON.parse`` and Go's
    ``encoding/json`` all reject them.  That bit exactly when it hurt most:
    the gate's "unmeasurable" sentinel is ``+inf`` and the ungated shares are
    ``NaN``, so it was the FAILING panel, the one worth inspecting, that
    produced the unparseable file -- while the docstring below promises this
    form exists so later tooling can read it without parsing prose.

    ``null`` rather than a string, so a consumer's numeric parse fails loudly
    instead of silently comparing ``"inf"``.  The human-readable text form
    still prints ``inf``, and ``failures`` says which criterion was
    unmeasurable.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_stance_gate_report(stage_dir: "str | Path", report: dict[str, Any]) -> dict[str, Path]:
    """Write the report beside the stage's other artifacts.

    Two forms on purpose: the text is what a human opens next to
    ``stage_summary.txt``, and the JSON is what later tooling can read without
    parsing prose.

    Three forms, and the third is load-bearing. ``stance_panel_selected.csv``
    is the per-episode duty evidence ``result_bundle.evidence`` needs to
    certify a stance-gated stage: it used to refuse such bundles outright
    because ``evaluation_selected.csv`` records reward and length but no duty,
    and certifying on the reward rail alone would pass the statue. The refusal
    was correct and it also made the milestone unreachable -- a run that
    genuinely cleared the stance gate still could not produce a publishable
    bundle. This file is what lifts it, and it is a CSV rather than the JSON's
    ``episode_evidence`` so the auditor recomputes the panel from the same
    kind of per-episode record it already uses for reward and length.

    The panel is a different evaluation from ``evaluation_selected.csv``: 40
    episodes at the panel seeds versus 30 at the publication seed. That is
    deliberate -- ``min_eval_episodes`` is the sample size the bound's power
    is specified at -- so it gets its own file rather than columns bolted
    onto a file with a different episode count.
    """
    directory = Path(stage_dir)
    directory.mkdir(parents=True, exist_ok=True)
    text_path = directory / "stance_gate_report.txt"
    json_path = directory / "stance_gate_report.json"
    text_path.write_text(render_stance_gate_report(report) + "\n", encoding="utf-8")
    # allow_nan=False turns any non-finite value _json_safe missed into a
    # ValueError here rather than an unparseable artifact on Drive.
    json_path.write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    written = {"stance_gate_report_txt": text_path, "stance_gate_report_json": json_path}
    panel_path = write_stance_panel_evidence(directory, report)
    if panel_path is not None:
        written["stance_panel_csv"] = panel_path
    return written


#: Column order of ``stance_panel_selected.csv``.  ``unsupported_duty`` is
#: empty for an episode whose duty could not be measured (one shorter than the
#: settling window); the auditor treats empty as unmeasured rather than zero,
#: because zero is the statue's value and the best possible score.
STANCE_PANEL_FIELDNAMES = (
    "episode",
    "panel_seed",
    "length",
    "reward",
    "reached_horizon",
    "unsupported_duty",
    "bilateral_support_duty",
    "single_support_duty",
)


def write_stance_panel_evidence(stage_dir: "str | Path", report: dict[str, Any]) -> "Path | None":
    """Write the panel's per-episode measurements as publication evidence.

    Returns ``None`` when the report carries no ``episode_evidence`` -- an
    older report, or one built before the field existed. The caller records
    only what was written, and the auditor's own refusal covers the absence,
    so a missing file fails the bundle rather than passing it.
    """
    episodes = report.get("episode_evidence")
    if not episodes:
        return None
    output = Path(stage_dir) / "stance_panel_selected.csv"
    with output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(STANCE_PANEL_FIELDNAMES))
        writer.writeheader()
        for episode in episodes:
            duty = episode.get("unsupported_duty")
            writer.writerow(
                {
                    "episode": episode["episode"],
                    "panel_seed": episode["seed"],
                    "length": int(episode["length"]),
                    "reward": float(episode["reward"]),
                    "reached_horizon": bool(episode["reached_horizon"]),
                    "unsupported_duty": "" if duty is None else float(duty),
                    "bilateral_support_duty": _optional_float_cell(episode.get("bilateral_support_duty")),
                    "single_support_duty": _optional_float_cell(episode.get("single_support_duty")),
                }
            )
    return output


def _optional_float_cell(value: Any) -> str | float:
    return "" if value is None else float(value)


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
    parser.add_argument(
        "--allow-legacy-plant",
        action="store_true",
        help="Score a checkpoint that predates the plant contract. The verdict is then about a plant "
        "that may differ from the one in the tree, and the report records plant_validated=false.",
    )
    parser.add_argument("--seed", type=int, default=3042, help="First evaluation seed (default: publication seed)")
    parser.add_argument("--config", help="Explicit stage TOML path")
    parser.add_argument("--out-dir", help="Also write stance_gate_report.{txt,json} to this directory")
    args = parser.parse_args()

    if not args.zero_action and not args.model:
        parser.error("pass --model, or --zero-action for the do-nothing reference")
    # `--episodes 0` used to fall through `episodes or min_eval_episodes` to the
    # stage default AND skip the under-powered-panel warning below, so it
    # silently produced a full-size panel while reading as an override.
    if args.episodes is not None and args.episodes < 1:
        parser.error(f"--episodes must be at least 1, got {args.episodes}")

    stage_config = load_stage_config(args.species, args.stage, config_path=args.config)
    try:
        report = build_stance_gate_report(
            args.species,
            args.stage,
            stage_config=stage_config,
            model_path=args.model,
            vecnorm_path=args.vecnorm,
            zero_action=args.zero_action,
            episodes=args.episodes,
            seed=args.seed,
            allow_legacy_plant=args.allow_legacy_plant,
        )
    except StanceGateReportError as exc:
        # The CLI boundary is where a diagnosable failure becomes an exit
        # status; inside the library it stays an ordinary exception.
        raise SystemExit(str(exc)) from exc
    print()
    print(render_stance_gate_report(report))

    if args.out_dir:
        for path in write_stance_gate_report(args.out_dir, report).values():
            print(f"written: {path}")

    declared = report["thresholds"]["min_eval_episodes"]
    if args.episodes is not None and args.episodes != declared:
        print()
        print(
            f"WARNING: --episodes {args.episodes} differs from the stage's min_eval_episodes "
            f"{declared}. The bound's power is specified at the latter; this panel does not "
            "certify what the gate claims."
        )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
