"""Post-training stage artifact generation.

The shared entry-points that the training notebook, the sweep trial worker,
and the JAX/MJX trainer all call so that stage artifacts stay consistent
across backends."""

from __future__ import annotations

import json as _json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from . import bundles, csv_output, stage_layout, text_summaries

if TYPE_CHECKING:
    from ..plant_contract import PlantIdentity

logger = logging.getLogger(__name__)

#: Episodes per cutoff in the action-filter probe sweep. Far below the gate's
#: ``min_eval_episodes`` on purpose: the probe certifies nothing, and the
#: effect it measures is enormous -- 96 steps against a 1000-step horizon on
#: T-Rex stage 1 (issue #491) -- so it does not need the sample size the
#: gate's bound has its power specified at. Keeping it small is what makes a
#: multi-cutoff sweep cost about what the old single 40-episode probe did.
_PROBE_EPISODES = 10

#: Episodes per row in the two sweep probes. Smaller than ``_PROBE_EPISODES``
#: because both roll far more rows -- 13 for the ablation, 14 for the impulse
#: sweep once the statue control doubles it -- and both measure effects that are
#: a fall or a full horizon rather than a shift in a mean.
_ABLATION_EPISODES = 8
_IMPULSE_EPISODES = 8


def build_stage_results_from_eval_data(
    stage_dir: "str | Path",
    stage: int,
    stage_config: dict[str, Any],
    timesteps: int,
    duration_seconds: float = 0.0,
) -> dict[str, Any]:
    """Build a ``stage_results`` dict from on-disk evaluation artifacts.

    Reads ``evaluations.npz`` (written by SB3's ``EvalCallback``) and
    ``metrics.json`` to reconstruct the same results dict that the
    training notebook's ``train_stage`` produces.  This allows sweep
    trials and any other post-hoc consumers to build a consistent
    results dict without re-running evaluation.

    If *duration_seconds* is 0 and a ``metrics.json`` exists, the duration
    is read from ``training_duration_seconds`` in that file.

    Fields that require a live policy evaluation (``mean_forward_vel``,
    ``mean_success_rate``, ``best_model_*``) default to ``0.0`` / ``""``
    and can be updated by the caller after running ``eval_policy``.
    """
    import numpy as _np

    stage_dir = Path(stage_dir)
    model_dir = stage_dir / "models"

    # ── Parse evaluations.npz ───────────────────────────────────────────
    eval_npz = stage_dir / "evaluations.npz"
    mean_reward = 0.0
    std_reward = 0.0
    mean_length = 0.0
    std_length = 0.0
    best_eval_reward: float | str = ""
    best_eval_std: float | str = ""
    best_eval_length: float | str = ""
    best_eval_std_length: float | str = ""
    best_eval_timestep: int | str = ""

    if eval_npz.exists():
        eval_data = _np.load(str(eval_npz))
        eval_rewards = eval_data["results"]
        eval_lengths = eval_data["ep_lengths"]
        eval_timesteps = eval_data["timesteps"]

        mean_per_eval = eval_rewards.mean(axis=1)
        best_idx = int(mean_per_eval.argmax())

        best_eval_reward = round(float(mean_per_eval[best_idx]), 2)
        best_eval_std = round(float(eval_rewards[best_idx].std()), 2)
        best_eval_length = round(float(eval_lengths[best_idx].mean()), 1)
        best_eval_std_length = round(float(eval_lengths[best_idx].std()), 1)
        best_eval_timestep = int(eval_timesteps[best_idx])

        # Use last eval as "final" metrics
        mean_reward = float(mean_per_eval[-1])
        std_reward = float(eval_rewards[-1].std())
        mean_length = float(eval_lengths[-1].mean())
        std_length = float(eval_lengths[-1].std())

    # ── Duration and provenance from sidecars ───────────────────────────
    metrics: dict[str, Any] = {}
    metrics_path = stage_dir / "metrics.json"
    if metrics_path.exists():
        metrics = _json.loads(metrics_path.read_text())
        if duration_seconds == 0.0:
            duration_seconds = metrics.get("training_duration_seconds", 0.0)
    plant_identity = metrics.get("plant_identity")
    if not isinstance(plant_identity, Mapping):
        saved_config_path = stage_dir / "stage_config.json"
        if saved_config_path.exists():
            saved_config = _json.loads(saved_config_path.read_text())
            plant_identity = saved_config.get("plant_identity")

    # The SELECTED checkpoint, so the `model_path` this records is the one the
    # replay shows and the evidence CSV is evidence for. It used to hardcode
    # `best_model` while the replay, the stance gate report and the next-stage
    # handoff all resolved through `_select_handoff_checkpoint`, which prefers
    # the risk-adjusted `robust_best_model`. This path feeds the sweep trial
    # worker, where nothing else re-derives it.
    #
    # Falls back to `best_model` rather than skipping when no candidate is
    # complete: unlike the replay and the gate report, this function only
    # *describes* a run, and an empty model_path would lose information the
    # caller can still check for itself.
    from environments.shared.train_base import _select_handoff_checkpoint

    handoff = _select_handoff_checkpoint(model_dir)
    if handoff is None:
        best_model_path = model_dir / "best_model"
        vecnorm_path = str(model_dir / "best_model_vecnorm.pkl")
    else:
        _, selected_path, vecnorm_path = handoff
        best_model_path = Path(selected_path)
    sim_dt = stage_config.get("env_kwargs", {}).get("sim_dt", 0.01)

    result = {
        "stage": stage,
        "name": stage_config["name"],
        "description": stage_config["description"],
        "timesteps": timesteps,
        "duration_seconds": duration_seconds,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_episode_length": mean_length,
        "std_episode_length": std_length,
        "mean_forward_vel": 0.0,
        "std_forward_vel": 0.0,
        "mean_success_rate": 0.0,
        "best_eval_reward": best_eval_reward,
        "best_eval_std": best_eval_std,
        "best_eval_length": best_eval_length,
        "best_eval_std_length": best_eval_std_length,
        "best_eval_timestep": best_eval_timestep,
        "sim_dt": sim_dt,
        "model_path": str(best_model_path),
        "vecnorm_path": vecnorm_path,
    }
    if isinstance(plant_identity, Mapping):
        result["plant_identity"] = dict(plant_identity)
    return result


def _write_stance_gate_report(
    *,
    species: str,
    stage: int,
    stage_config: dict[str, Any],
    stage_dir: Path,
    model_dir: Path,
) -> dict[str, Any] | None:
    """Score the selected checkpoint against the stance gate, into the run dir.

    Returns the report dict, or ``None`` when no panel was rolled.  The return
    value is what :func:`_apply_stage_gate` certifies a ``stance_quality/v1``
    stage from, so ``None`` fails that stage closed.

    Only for stages that actually declare ``stance_quality/v1``: rolling a
    40-episode panel costs a few minutes, which is nothing beside a multi-hour
    stage but is pure waste for a stage the criteria do not govern.

    Runs here rather than in the notebook so it happens for every SB3 run and
    every sweep trial without anyone remembering, and lands beside
    ``stage_summary.txt`` in the run directory -- which is on Drive, so the
    verdict survives a lost runtime.

    ``stance_report_episodes`` in ``[curriculum]`` overrides the panel size;
    ``0`` skips the report entirely. It exists because "a few minutes" is per
    stage AND per sweep trial, and a sweep of fifty trials pays it fifty
    times for a verdict nobody reads until a trial is shortlisted. Overriding
    downward makes the bound weaker than the gate claims -- the panel size is
    what its power is specified at -- so the log says so when it happens.

    Deliberately non-fatal to *artifact generation*: losing it must never cost
    a completed training run its summary, replays and graphs, and the
    checkpoint may legitimately be absent (a stage stopped before its first
    evaluation produced one).  It is emphatically fatal to *certification* —
    a stance-gated stage with no panel fails, because the alternative is
    certifying stance quality nobody measured.
    """
    from environments.shared.curriculum.stance_gate import STANCE_GATE_KIND

    curriculum = stage_config.get("curriculum_kwargs", {})
    if curriculum.get("gate_kind") != STANCE_GATE_KIND:
        return None

    declared_episodes = int(curriculum.get("min_eval_episodes", 40))
    report_episodes = curriculum.get("stance_report_episodes")
    report_episodes = declared_episodes if report_episodes is None else int(report_episodes)
    if report_episodes < 1:
        logger.info(
            "Stance gate report skipped for stage %d: stance_report_episodes = %d",
            stage,
            report_episodes,
        )
        return None
    if report_episodes != declared_episodes:
        logger.warning(
            "Stance gate report for stage %d rolls %d episodes, not the stage's min_eval_episodes "
            "%d. The bound's power is specified at the latter; this panel does not certify what "
            "the gate claims.",
            stage,
            report_episodes,
            declared_episodes,
        )

    # The SELECTED checkpoint, through the one selector — the same call the
    # replay and the next-stage handoff make. This used to be a third private
    # copy of the preference order that also passed `vecnorm_path=None` when
    # the statistics were missing, silently scoring the policy on unnormalised
    # observations: a different policy, reported as this one's gate verdict.
    from environments.shared.train_base import _select_handoff_checkpoint

    handoff = _select_handoff_checkpoint(model_dir)
    if handoff is None:
        logger.warning(
            "Stance gate report skipped for stage %d: no checkpoint in %s has its "
            "matched _vecnorm.pkl, and scoring without the observation statistics "
            "would report a verdict for a different policy.",
            stage,
            model_dir,
        )
        return None

    selected_name, selected_path, selected_vecnorm = handoff
    try:
        from environments.shared.scripts.stance_gate_report import (
            build_stance_gate_report,
            write_stance_gate_report,
        )

        logger.info("Stance gate report scoring stage %d checkpoint: %s", stage, selected_name)
        report = build_stance_gate_report(
            species,
            stage,
            stage_config=stage_config,
            model_path=f"{selected_path}.zip",
            vecnorm_path=selected_vecnorm,
            episodes=report_episodes,
        )
        written = write_stance_gate_report(stage_dir, report)
        logger.info(
            "Stance gate report: %s (duty %.4f, bilateral %.4f) -> %s",
            "PASS" if report["passed"] else "FAIL",
            report["metrics"]["mean_unsupported_duty"],
            report["metrics"]["bilateral_support_duty"],
            written["stance_gate_report_txt"],
        )
        return report
    except Exception:  # noqa: BLE001 - a diagnostic must not sink the run
        logger.warning("Stance gate report failed for stage %d", stage, exc_info=True)
    return None


def _run_stance_probes(
    *,
    species: str,
    stage: int,
    stage_config: dict[str, Any],
    stage_dir: Path,
    model_dir: Path,
    report: dict[str, Any] | None,
) -> None:
    """Run the stance probe battery against the selected checkpoint.

    Pure diagnostics, so they run LAST in :func:`generate_stage_artifacts` --
    after the gate verdict is recorded and the summary, graphs and replays are
    written. They used to run inside :func:`_write_stance_gate_report`'s
    ``try``, between "report written" and "verdict recorded", which put ~300
    probe episodes of exposure in front of everything a finished run cannot
    afford to lose -- and meant a probe-wiring exception (one the helpers'
    internal handlers cannot see, e.g. a signature drift at a call site) was
    caught by the report's handler and turned an already-written PASS into a
    recorded FAIL.

    *report* is the certification report the probes annotate; ``None`` (no
    panel was measured) means there is nothing to probe. Each probe call
    carries its own handler so a failure costs that probe alone -- including
    caller-side failures, which is what makes the CHANGELOG's "individually
    non-fatal" claim true at this level too.
    """
    if report is None:
        return

    curriculum = stage_config.get("curriculum_kwargs", {})
    declared_episodes = int(curriculum.get("min_eval_episodes", 40))
    report_episodes = curriculum.get("stance_report_episodes")
    report_episodes = declared_episodes if report_episodes is None else int(report_episodes)

    # The same selector the report itself used; re-resolved here so the probes
    # keep describing the one policy every other artifact describes.
    from environments.shared.train_base import _select_handoff_checkpoint

    handoff = _select_handoff_checkpoint(model_dir)
    if handoff is None:
        logger.warning(
            "Stance probes skipped for stage %d: no checkpoint in %s has its matched _vecnorm.pkl",
            stage,
            model_dir,
        )
        return
    _selected_name, selected_path, selected_vecnorm = handoff

    probe_calls = (
        (
            "filtered action",
            lambda: _write_filtered_action_probe(
                species=species,
                stage=stage,
                stage_config=stage_config,
                stage_dir=stage_dir,
                model_path=f"{selected_path}.zip",
                vecnorm_path=selected_vecnorm,
                episodes=report_episodes,
            ),
        ),
        (
            "constant hold",
            lambda: _write_constant_hold_probe(
                species=species,
                stage=stage,
                stage_config=stage_config,
                stage_dir=stage_dir,
                model_path=f"{selected_path}.zip",
                vecnorm_path=selected_vecnorm,
                episodes=report_episodes,
                measured=report,
            ),
        ),
        (
            "release ablation",
            lambda: _write_constant_hold_ablation(
                species=species,
                stage=stage,
                stage_config=stage_config,
                stage_dir=stage_dir,
                model_path=f"{selected_path}.zip",
                vecnorm_path=selected_vecnorm,
                measured=report,
            ),
        ),
        (
            "impulse",
            lambda: _write_impulse_probe(
                species=species,
                stage=stage,
                stage_config=stage_config,
                stage_dir=stage_dir,
                model_path=f"{selected_path}.zip",
                vecnorm_path=selected_vecnorm,
                settle_steps=int(report["settle_steps"]),
            ),
        ),
    )
    for label, run_probe in probe_calls:
        try:
            run_probe()
        except Exception:  # noqa: BLE001 - a diagnostic must not sink the run
            logger.warning("Stance probe (%s) failed for stage %d", label, stage, exc_info=True)


def _probe_cutoffs(raw: Any) -> list[float]:
    """Normalise ``stance_probe_filter_hz`` to a sorted list of cutoffs.

    Accepts a single number or a list, because the useful reading is a curve
    rather than a point. A single cutoff answers a yes/no that is already
    known to be "no" on this plant -- the checkpoint that PASSED the gate
    falls at every cutoff from 5 to 35 Hz against a 100 Hz control rate
    (issue #491) -- so its PASS/FAIL carries no information. The scalar that
    can actually move is how long the filtered policy survives, and reading
    that against cutoff shows *how much* high-frequency content the policy
    depends on rather than merely that it depends on some.
    """
    values = raw if isinstance(raw, (list, tuple)) else [raw]
    cutoffs: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            logger.warning("stance_probe_filter_hz entry is not a number: %r; ignoring it", value)
            continue
        if number > 0:
            cutoffs.append(number)
    return sorted(set(cutoffs))


def _write_filtered_action_probe(
    *,
    species: str,
    stage: int,
    stage_config: dict[str, Any],
    stage_dir: Path,
    model_path: str,
    vecnorm_path: str | None,
    episodes: int,
) -> None:
    """Re-score the selected checkpoint with its actions low-passed, over a sweep.

    Measures how much of its own high-frequency command content the policy
    needs in order to stand. A first-order low-pass between policy and plant
    attenuates the tremor while leaving balance correction -- a ~1.1-1.4 Hz
    phenomenon on this plant -- essentially untouched, so survival under the
    filter is a direct read on whether the tremor is load-bearing.

    That question is settled for today's policies and the answer is "yes"
    (issue #491), which is why this logs a CURVE. Reported per cutoff, the
    survival length is a regression metric: 96 steps at 5 Hz is the current
    T-Rex stage-1 baseline, and a policy reaching the horizon there would be
    one that could survive a real actuator's bandwidth limit.

    Off unless ``stance_probe_filter_hz`` is set. Each cutoff costs a panel,
    so the probe deliberately rolls far fewer episodes than the gate report:
    it certifies nothing, and the effect it measures is enormous (96 steps
    against 1000), so it does not need the sample size the bound's power is
    specified at.

    Writes ``stance_gate_probe_filtered.{txt,json}`` and deliberately NOT
    ``stance_panel_selected.csv``; the probe scored a modified policy and must
    never supply the evidence a bundle is certified from.
    """
    cutoffs = _probe_cutoffs(stage_config.get("curriculum_kwargs", {}).get("stance_probe_filter_hz"))
    if not cutoffs:
        return
    probe_episodes = max(1, min(episodes, _PROBE_EPISODES))
    entries: list[dict[str, Any]] = []
    try:
        from environments.shared.scripts.stance_gate_report import (
            build_stance_gate_report,
            write_action_filter_sweep,
        )

        for cutoff in cutoffs:
            probe = build_stance_gate_report(
                species,
                stage,
                stage_config=stage_config,
                model_path=model_path,
                vecnorm_path=vecnorm_path,
                episodes=probe_episodes,
                filter_actions_hz=cutoff,
            )
            entries.append(probe)
            logger.info(
                "Filtered action probe %.4g Hz: episode length %.1f, full-horizon %.4f, reward %.1f. "
                "MODIFIED policy -- not a gate verdict.",
                cutoff,
                probe["metrics"]["episode_length_mean"],
                probe["metrics"]["full_horizon_fraction"],
                probe["metrics"]["reward_mean"],
            )
    except Exception:  # noqa: BLE001 - a diagnostic must not sink the run
        logger.warning("Filtered action probe failed for stage %d", stage, exc_info=True)
    # Written even if a later cutoff raised: a partial curve is still a curve,
    # and discarding the cutoffs that succeeded would lose the measurement to
    # a failure in one of them.
    if entries:
        try:
            from environments.shared.scripts.stance_gate_report import write_action_filter_sweep

            written = write_action_filter_sweep(stage_dir, entries, probe_episodes=probe_episodes)
            logger.info("Filtered action probe sweep -> %s", written["action_filter_sweep_txt"])
        except Exception:  # noqa: BLE001 - a diagnostic must not sink the run
            logger.warning("Could not write the filtered action probe sweep", exc_info=True)


def _write_constant_hold_probe(
    *,
    species: str,
    stage: int,
    stage_config: dict[str, Any],
    stage_dir: Path,
    model_path: str,
    vecnorm_path: str | None,
    episodes: int,
    measured: dict[str, Any],
) -> None:
    """Re-score the checkpoint with its action frozen to the constant it averages.

    The question the filtered probe leaves open. Low-passing a policy still
    lets it respond, just slowly, so a fall under the filter proves the policy
    needs *bandwidth* without saying whether it needs *feedback*. Cutting the
    feedback outright and commanding the policy's own post-settle mean
    separates the two, and the two have opposite fixes: a pose that stands
    under a constant means the tremor is waste the action penalties failed to
    suppress, while a pose that falls means the tremor is the only thing
    holding the animal up and penalising it harder would be actively wrong.

    Off unless ``stance_probe_hold_constant`` is set. Reuses the gate report's
    already-measured per-actuator DC rather than rolling a measurement panel of
    its own, so the whole probe costs the variant panels and nothing else.

    Writes ``stance_gate_probe_constant.{txt,json}`` and deliberately NOT
    ``stance_panel_selected.csv``.
    """
    curriculum = stage_config.get("curriculum_kwargs", {})
    if not curriculum.get("stance_probe_hold_constant"):
        return
    probe_episodes = max(1, min(episodes, _PROBE_EPISODES))
    entries: list[dict[str, Any]] = []
    try:
        from environments.shared.scripts.stance_gate_report import (
            build_stance_gate_report,
            constant_hold_actions,
            constant_hold_variants,
        )

        hold = constant_hold_actions(measured)
        variants = constant_hold_variants(
            hold,
            settle_steps=int(measured["settle_steps"]),
            horizon=int(measured["horizon"]),
        )
        for variant in variants:
            probe = build_stance_gate_report(
                species,
                stage,
                stage_config=stage_config,
                model_path=model_path,
                vecnorm_path=vecnorm_path,
                episodes=probe_episodes,
                hold_constant=variant,
            )
            entries.append(probe)
            logger.info(
                "Constant-hold probe %s: episode length %.1f, full-horizon %.4f, reward %.1f. "
                "MODIFIED policy -- not a gate verdict.",
                variant.label,
                probe["metrics"]["episode_length_mean"],
                probe["metrics"]["full_horizon_fraction"],
                probe["metrics"]["reward_mean"],
            )
    except Exception:  # noqa: BLE001 - a diagnostic must not sink the run
        logger.warning("Constant-hold probe failed for stage %d", stage, exc_info=True)
    # Written even if a later variant raised, for the same reason the filter
    # sweep is: the variants that succeeded are still a measurement, and the
    # controls are what make the others readable.
    if entries:
        try:
            from environments.shared.scripts.stance_gate_report import write_constant_hold_probe

            written = write_constant_hold_probe(stage_dir, entries, probe_episodes=probe_episodes)
            logger.info("Constant-hold probe -> %s", written["constant_hold_probe_txt"])
        except Exception:  # noqa: BLE001 - a diagnostic must not sink the run
            logger.warning("Could not write the constant-hold probe", exc_info=True)


def _write_constant_hold_ablation(
    *,
    species: str,
    stage: int,
    stage_config: dict[str, Any],
    stage_dir: Path,
    model_path: str,
    vecnorm_path: str | None,
    measured: dict[str, Any],
) -> None:
    """Ablate the held pose one actuator group at a time, into the run dir.

    Answers *which* joints make the pose unholdable, where the constant-hold
    probe only answers *whether* it is. Each group is tested twice -- released
    (is it necessary?) and held alone (is it sufficient?) -- because either
    side alone lets a conspicuous group masquerade as a cause. On the T-Rex
    that is not hypothetical: the tail is the most extreme thing in the DC
    table and is provably inert, while the toes carry the whole effect.

    Off unless ``stance_probe_release_ablation`` is set. Costs 2 panels per
    actuator group plus 3 fixed rows -- 13 on the T-Rex -- so it is the most
    expensive of the three probes and is opt-in per stage.
    """
    curriculum = stage_config.get("curriculum_kwargs", {})
    if not curriculum.get("stance_probe_release_ablation"):
        return
    entries: list[dict[str, Any]] = []
    try:
        from environments.shared.scripts.stance_gate_report import (
            build_stance_gate_report,
            constant_hold_actions,
            constant_hold_release_variants,
        )

        hold = constant_hold_actions(measured)
        for variant in constant_hold_release_variants(
            hold, measured, horizon=int(measured["horizon"])
        ):
            probe = build_stance_gate_report(
                species,
                stage,
                stage_config=stage_config,
                model_path=model_path,
                vecnorm_path=vecnorm_path,
                episodes=_ABLATION_EPISODES,
                hold_constant=variant,
            )
            entries.append(probe)
            logger.info(
                "Release ablation %s: episode length %.1f, full-horizon %.4f. "
                "MODIFIED policy -- not a gate verdict.",
                variant.label,
                probe["metrics"]["episode_length_mean"],
                probe["metrics"]["full_horizon_fraction"],
            )
    except Exception:  # noqa: BLE001 - a diagnostic must not sink the run
        logger.warning("Release ablation failed for stage %d", stage, exc_info=True)
    if entries:
        try:
            from environments.shared.scripts.stance_gate_report import write_constant_hold_ablation

            written = write_constant_hold_ablation(
                stage_dir, entries, probe_episodes=_ABLATION_EPISODES
            )
            logger.info("Release ablation -> %s", written["constant_hold_ablation_txt"])
        except Exception:  # noqa: BLE001 - a diagnostic must not sink the run
            logger.warning("Could not write the release ablation", exc_info=True)


def _write_impulse_probe(
    *,
    species: str,
    stage: int,
    stage_config: dict[str, Any],
    stage_dir: Path,
    model_path: str,
    vecnorm_path: str | None,
    settle_steps: int,
) -> None:
    """Shove the animal mid-episode and record whether it recovers.

    The only measurement in the pipeline that tracks *balance* rather than
    *not falling over*. Stage 1 declares no in-episode disturbance, so every
    other criterion -- duty, full-horizon share, reward -- is satisfied by a
    statue, and nothing else can tell a policy that learned to recover from one
    that learned to stand still.

    Rolls the zero-action statue over the same sweep as the control. That is
    not optional: the statue cannot respond to anything, so its survival is the
    plant's *passive* robustness and only the policy's margin above it is
    attributable to control. Reporting the policy's numbers alone would credit
    the plant's own stability to the policy.

    Off unless ``stance_probe_impulse_speeds`` is set. Costs 2 panels per
    (speed, direction) plus the two zero controls -- 14 on the default sweep --
    because the statue side doubles it.
    """
    curriculum = stage_config.get("curriculum_kwargs", {})
    raw = curriculum.get("stance_probe_impulse_speeds")
    if not raw:
        return
    values = raw if isinstance(raw, (list, tuple)) else [raw]
    speeds: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            logger.warning("stance_probe_impulse_speeds entry is not a number: %r; ignoring it", value)
            continue
        if number > 0:
            speeds.append(number)
    if not speeds:
        return
    try:
        from environments.shared.scripts.stance_gate_report import (
            build_stance_gate_report,
            impulse_variants,
            write_impulse_probe,
        )

        variants = impulse_variants(speeds, step=settle_steps)

        def _sweep(zero_action: bool) -> list[dict[str, Any]]:
            return [
                build_stance_gate_report(
                    species,
                    stage,
                    stage_config=stage_config,
                    model_path=None if zero_action else model_path,
                    vecnorm_path=None if zero_action else vecnorm_path,
                    zero_action=zero_action,
                    episodes=_IMPULSE_EPISODES,
                    impulse=variant,
                )
                for variant in variants
            ]

        policy_sweep = _sweep(zero_action=False)
        statue_sweep = _sweep(zero_action=True)
        written = write_impulse_probe(
            stage_dir, policy_sweep, statue_sweep, probe_episodes=_IMPULSE_EPISODES
        )
        envelopes = {
            row["impulse"]["axis_label"]: row["metrics"]["full_horizon_fraction"]
            for row in policy_sweep
            if row["impulse"]["speed"] > 0
        }
        logger.info(
            "Impulse recovery probe -> %s (policy full-horizon by direction: %s). "
            "MODIFIED task -- not a gate verdict.",
            written["impulse_probe_txt"],
            envelopes,
        )
    except Exception:  # noqa: BLE001 - a diagnostic must not sink the run
        logger.warning("Impulse recovery probe failed for stage %d", stage, exc_info=True)


def _apply_stage_gate(
    *,
    stage: int,
    stage_config: dict[str, Any],
    stage_results: dict[str, Any],
    stance_report: dict[str, Any] | None,
) -> None:
    """Record this stage's gate verdict onto *stage_results*, in place.

    Runs here, in the one entry point both the notebook and the sweep trial
    worker already call, because the alternative is what actually happened:
    each caller kept a private checklist, the notebook's drifted out of step
    with ``gate_kind``, and run ``20260802_203215`` recorded
    ``publication_gate_passed = True`` beside a ``GATE: FAIL`` stance report
    and advanced to stage 2 on it.

    Sets ``gate_passed``, ``publication_gate_passed`` and ``gate_failures``;
    callers enforce them.  Never raises: a stage that cannot be certified is
    recorded as failing, which is the fail-closed reading, and an exception
    here would instead cost the run the artifacts written around it.
    """
    from .gates import evaluate_stage_gate

    try:
        passed, failures = evaluate_stage_gate(
            stage_config.get("curriculum_kwargs", {}),
            stage_results,
            stage=stage,
            stance_report=stance_report,
        )
    except Exception as exc:  # noqa: BLE001 - the docstring's promise, kept
        # "Never raises" has to be enforced, not asserted. `evaluate_stage_gate`
        # coerces config values (`float(min_avg_reward)`) and iterates a
        # report's `failures`, so a malformed TOML value or a hand-edited
        # report reaches here as TypeError/ValueError. Letting it propagate
        # would cost a completed multi-hour run the artifacts written around
        # this call; recording it as a failure keeps the fail-closed reading
        # AND the artifacts.
        logger.warning("Stage %d curriculum gate could not be evaluated", stage, exc_info=True)
        passed, failures = False, [f"stage {stage} gate evaluation raised {type(exc).__name__}: {exc}"]
    stage_results["gate_passed"] = passed
    stage_results["publication_gate_passed"] = passed
    stage_results["gate_failures"] = failures
    if passed:
        logger.info("Stage %d curriculum gate: PASS", stage)
    else:
        logger.warning("Stage %d curriculum gate: FAIL — %s", stage, "; ".join(failures))


def generate_stage_artifacts(
    species_cfg,
    stage_config: dict[str, Any],
    stage: int,
    algorithm: str,
    stage_dir: "str | Path",
    seed: int,
    stage_results: dict[str, Any] | None = None,
    timesteps: int = 0,
    record_videos: bool = True,
    generate_graphs: bool = True,
    allow_legacy_plant: bool = False,
) -> dict[str, Any]:
    """Write stage summary, record replay videos, and generate training graphs.

    This is the single shared entry-point for generating post-training
    artifacts.  Both the training notebook and the sweep trial worker
    call this function so that the artifacts are always consistent.

    When *stage_results* is ``None``, a results dict is built from on-disk
    eval data via :func:`build_stage_results_from_eval_data`.  Callers
    that already have richer metrics (e.g. the notebook, which runs a
    full 30-episode eval) should pass their own *stage_results*.

    When *generate_graphs* is ``True`` (the default), training curves and
    diagnostic graphs are saved to the stage directory.  Requires
    ``matplotlib``.

    Also evaluates the stage's declared curriculum gate and records the
    verdict onto *stage_results* as ``gate_passed`` /
    ``publication_gate_passed`` / ``gate_failures``.  It happens here, not in
    each caller, so no trainer can advance on a checklist that has drifted
    away from ``gate_kind`` — see :func:`_apply_stage_gate`.  Callers must
    enforce the verdict; this function records it.

    Returns the (possibly enriched) *stage_results* dict.
    """
    stage_dir = Path(stage_dir)
    model_dir = stage_dir / "models"
    species = species_cfg.species

    if stage_results is None:
        stage_results = build_stage_results_from_eval_data(
            stage_dir,
            stage,
            stage_config,
            timesteps=timesteps,
        )

    stance_report = _write_stance_gate_report(
        species=species,
        stage=stage,
        stage_config=stage_config,
        stage_dir=stage_dir,
        model_dir=model_dir,
    )
    # Before the replays and graphs below, which are best-effort and can be
    # skipped: the verdict must not depend on whether matplotlib imported.
    _apply_stage_gate(
        stage=stage,
        stage_config=stage_config,
        stage_results=stage_results,
        stance_report=stance_report,
    )

    # After the gate, not before it: the summary now states the verdict and
    # every criterion that failed, and writing it first would print a stage
    # summary that says nothing about the decision the stage turns on.
    text_summaries.write_stage_summary(stage_dir, stage_results, species, algorithm)
    logger.info("Stage summary written to: %s", stage_dir / "stage_summary.txt")

    # Figures and replays render into local scratch and publish to the stage
    # directory in one pass on exit.  The plots and videos below read their
    # inputs from `stage_dir` (evaluations.npz, diagnostics.npz, models/) and
    # only their *outputs* are staged — see stage_layout.staged_artifacts for
    # why writing an mp4 straight onto a Drive mount is the expensive part.
    with stage_layout.staged_artifacts(stage_dir) as staging:
        figures_out = stage_layout.figures_dir(staging, create=True)
        replays_out = stage_layout.replays_dir(staging, create=True)

        # ── Generate training graphs ────────────────────────────────────
        if generate_graphs:
            try:
                from environments.shared.visualization import (
                    plot_diagnostics_graphs,
                    plot_foot_contacts,
                    plot_stance_diagnostics,
                    plot_training_curves,
                )

                stage_dirs = [(stage, stage_dir)]
                stage_configs = {stage: stage_config}

                plot_training_curves(
                    stage_dirs,
                    stage_configs,
                    species,
                    algorithm,
                    save_path=figures_out / "training_curves.png",
                    show=False,
                )
                plot_diagnostics_graphs(
                    stage_dirs,
                    stage_configs,
                    species,
                    algorithm,
                    save_dir=figures_out,
                    show=False,
                )
                plot_foot_contacts(
                    stage_dirs,
                    stage_configs,
                    species,
                    algorithm,
                    save_path=figures_out / "foot_contacts.png",
                    show=False,
                )
                plot_stance_diagnostics(
                    stage_dirs,
                    stage_configs,
                    species,
                    algorithm,
                    save_path=figures_out / "stance_diagnostics.png",
                    show=False,
                )
            except ImportError:
                logger.warning("Skipping graph generation (matplotlib not installed).")
            except Exception:
                logger.warning("Graph generation failed.", exc_info=True)

        if record_videos:
            stage_results = _record_stage_replays(
                species_cfg=species_cfg,
                stage_config=stage_config,
                stage=stage,
                algorithm=algorithm,
                stage_dir=stage_dir,
                replays_out=replays_out,
                model_dir=model_dir,
                seed=seed,
                stage_results=stage_results,
                allow_legacy_plant=allow_legacy_plant,
            )

    # Probes run dead last, outside the staging context: they are the most
    # expensive artifact step (~300 episodes at the trex stage-1 settings) and
    # nothing downstream reads them, so a runtime lost mid-probe costs the
    # probes alone -- never the recorded verdict, summary, graphs or replays
    # above. Ordering is pinned by the wiring tests.
    _run_stance_probes(
        species=species,
        stage=stage,
        stage_config=stage_config,
        stage_dir=stage_dir,
        model_dir=model_dir,
        report=stance_report,
    )

    return stage_results


def _record_stage_replays(
    *,
    species_cfg,
    stage_config: dict[str, Any],
    stage: int,
    algorithm: str,
    stage_dir: Path,
    replays_out: Path,
    model_dir: Path,
    seed: int,
    stage_results: dict[str, Any],
    allow_legacy_plant: bool,
) -> dict[str, Any]:
    """Record the best and final replays into *replays_out*.

    Split out of :func:`generate_stage_artifacts` only so the staging
    context there stays readable; the behaviour is unchanged apart from
    the videos landing under ``replays/`` instead of the stage root.
    """
    species = species_cfg.species

    # ── Record replay videos for the selected and final checkpoints ──────
    from ..plant_contract import PlantCompatibilityError, current_plant_identity, validate_model_plant

    try:
        from environments.shared.evaluation import TREX_STAGE1_CAMERA_VIEWS, record_stage_video
        from environments.shared.train_base import _ensure_sb3, _select_handoff_checkpoint

        sb3 = _ensure_sb3()
        env_kwargs = stage_config["env_kwargs"].copy()
        alg_cls = sb3["SAC"] if algorithm == "sac" else sb3["PPO"]
        plant_identity = current_plant_identity(species)

        final_path = model_dir / f"stage{stage}_final"
        final_vecnorm_path = str(final_path) + "_vecnorm.pkl"
        replay_diagnostics = species.lower() == "trex" and stage == 1
        replay_camera_views = TREX_STAGE1_CAMERA_VIEWS if replay_diagnostics else None

        # The SELECTED checkpoint, via the same selector that decides the
        # next-stage handoff and that `evaluation_selected.csv` is evidence
        # for. This used to hardcode `best_model` and label the replay
        # "best", while the selector prefers the risk-adjusted
        # `robust_best_model` — so on any run where both exist (they
        # normally do) the video and the evidence CSV in the same folder
        # described DIFFERENT POLICIES, with nothing in either name saying
        # so. `stance_gate_report` picks the risk-adjusted one too. One
        # selector, one label.
        #
        # Returning None means no candidate has its matched VecNormalize
        # statistics. Recording anyway would replay the policy on
        # unnormalised observations — a different policy again — so the
        # replay is skipped and says why, rather than producing footage
        # that misrepresents the checkpoint.
        handoff = _select_handoff_checkpoint(model_dir)
        if handoff is None:
            logger.warning(
                "Stage %d selected-checkpoint replay skipped: neither robust_best_model nor "
                "best_model in %s has its matched _vecnorm.pkl, and replaying without the "
                "observation statistics would show a different policy.",
                stage,
                model_dir,
            )
        else:
            selected_name, selected_path, selected_vecnorm = handoff
            logger.info("Stage %d selected checkpoint for replay: %s", stage, selected_name)
            selected_model = alg_cls.load(selected_path)
            validate_model_plant(
                selected_model,
                plant_identity,
                artifact=f"{selected_path}.zip",
                allow_legacy=allow_legacy_plant,
            )
            record_stage_video(
                selected_model,
                env_class=species_cfg.env_class,
                env_kwargs=env_kwargs,
                stage=stage,
                stage_dir=stage_dir,
                output_dir=replays_out,
                species=species,
                algorithm=algorithm,
                seed=seed,
                vecnorm_path=selected_vecnorm,
                label="selected",
                plant_identity=plant_identity,
                allow_legacy_plant=allow_legacy_plant,
                camera_views=replay_camera_views,
                collect_stance_diagnostics=replay_diagnostics,
            )

        if (Path(str(final_path) + ".zip")).exists():
            final_model = alg_cls.load(str(final_path))
            validate_model_plant(
                final_model,
                plant_identity,
                artifact=str(final_path) + ".zip",
                allow_legacy=allow_legacy_plant,
            )
            record_stage_video(
                final_model,
                env_class=species_cfg.env_class,
                env_kwargs=env_kwargs,
                stage=stage,
                stage_dir=stage_dir,
                output_dir=replays_out,
                species=species,
                algorithm=algorithm,
                seed=seed,
                vecnorm_path=final_vecnorm_path,
                label="final",
                plant_identity=plant_identity,
                allow_legacy_plant=allow_legacy_plant,
                camera_views=replay_camera_views,
                collect_stance_diagnostics=replay_diagnostics,
            )
    except PlantCompatibilityError:
        raise
    except Exception:
        logger.warning("Video recording failed.", exc_info=True)

    return stage_results


def save_jax_stage_artifacts(
    species: str,
    stage: int,
    stage_config: dict[str, Any],
    stage_results: dict[str, Any],
    stage_dir: "str | Path",
    run_dir: "str | Path",
    eval_results: Any,
    params: Any,
    obs_rms: Any,
    *,
    final_eval_results: Any | None = None,
    seed: int = 42,
    num_envs: int = 2048,
    reward_cfg: dict[str, float] | None = None,
    best_params: Any | None = None,
    best_reward: float = 0.0,
    best_update: int = 0,
    evaluation_seed: int = 42,
    backend_version: str | None = None,
    plant_identity: PlantIdentity | None = None,
) -> dict[str, Path]:
    """Save all post-training artifacts for a JAX/MJX training stage.

    Orchestrates the same artifact generation that the SB3 path performs
    via :func:`generate_stage_artifacts`, but using JAX-native checkpoint
    formats and without requiring an SB3 ``SpeciesConfig``.

    Artifacts saved:

    * ``stage_summary.txt`` — human-readable stage summary
    * ``stage_config.json`` — frozen config snapshot
    * ``collected_results.csv`` — one row per stage (append-safe)
    * ``diagnostics.npz`` — per-step evaluation diagnostics
    * ``best_model.pkl`` — best checkpoint (params + obs stats)
    * ``stage{N}_final.pkl`` — final checkpoint
    * ``training_summary.txt`` — run-level summary

    Args:
        species: Species identifier (e.g. ``"velociraptor"``).
        stage: Curriculum stage number (1, 2, or 3).
        stage_config: Config dict from :func:`config.load_stage_config`.
        stage_results: Results dict with keys like ``mean_reward``,
            ``timesteps``, ``duration_seconds``, ``model_path``, etc.
        stage_dir: Directory for this stage's output files.
        run_dir: Parent run directory (for CSV and training summary).
        eval_results: :class:`jax_eval.EvalResults` instance with
            selected-checkpoint episode and per-step diagnostic data.
        final_eval_results: Episode evidence for the terminal parameters.
            Defaults to *eval_results* only for backward-compatible callers
            whose selected and terminal parameters are identical.
        params: Final JAX network parameters.
        obs_rms: Observation normalisation statistics.
        seed: Random seed used for training.
        num_envs: Number of parallel environments.
        reward_cfg: Reward weight dict (included in config snapshot).
        best_params: Best-performing parameters (falls back to *params*).
        best_reward: Best evaluation reward achieved during training.
        best_update: Update number at which *best_params* was recorded.
        evaluation_seed: Seed used for the fixed publication evaluation.
        backend_version: Optional explicit JAX version for portable tests or
            environments where package metadata cannot be detected.
        plant_identity: Optional precomputed plant identity.  The current
            identity is resolved and verified when omitted.

    Returns:
        Dict mapping artifact name to its file path.
    """
    import numpy as _np

    from ..config import save_stage_config
    from ..jax_checkpoint import save_checkpoint
    from ..plant_contract import current_plant_identity, write_plant_identity
    from ..result_bundle import (
        ResultBundleError,
        initialize_result_bundle,
        load_provenance,
        validate_result_bundle,
    )

    stage_dir = Path(stage_dir)
    run_dir = Path(run_dir)
    if stage not in {1, 2, 3}:
        raise ValueError("stage must be 1, 2, or 3")
    if stage_dir.parent.resolve() != run_dir.resolve():
        raise ValueError("stage_dir must be run_dir/stage<N> for a portable result bundle")
    if stage_dir.name != f"stage{stage}":
        raise ValueError(f"stage_dir must be named stage{stage}")
    if plant_identity is None:
        plant_identity = current_plant_identity(species)

    manifest_path = run_dir / "artifact_manifest.json"
    if manifest_path.exists():
        try:
            existing_manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, _json.JSONDecodeError) as exc:
            raise ResultBundleError(f"cannot read existing artifact manifest: {exc}") from exc
        if isinstance(existing_manifest, Mapping) and existing_manifest.get("status") == "complete":
            validate_result_bundle(run_dir, require_complete=True)
            raise ResultBundleError("completed result bundle is immutable; use a new run_id to rewrite a stage")

    existing_stages: set[int] = set()
    for existing_result_path in sorted(run_dir.glob("stage*/stage_result.json")):
        try:
            saved_result = _json.loads(existing_result_path.read_text(encoding="utf-8"))
            saved_stage = int(saved_result["stage"])
        except (OSError, _json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ResultBundleError(f"cannot read prior JAX stage record {existing_result_path}") from exc
        if existing_result_path.parent.name != f"stage{saved_stage}" or saved_stage not in {1, 2, 3}:
            raise ResultBundleError(f"mislabeled prior JAX stage record: {existing_result_path}")
        if saved_stage in existing_stages:
            raise ResultBundleError(f"duplicate prior JAX stage record for stage {saved_stage}")
        existing_stages.add(saved_stage)
    combined_stages = existing_stages | {stage}
    expected_stages = set(range(1, max(combined_stages) + 1))
    if combined_stages != expected_stages or max(combined_stages) > stage:
        raise ResultBundleError(f"JAX stages must be saved as a contiguous prefix; found {sorted(combined_stages)}")

    episode_fields = ("rewards", "lengths", "forward_vels", "distances", "successes")
    has_episode_evidence = all(hasattr(eval_results, field) for field in episode_fields)
    terminal_eval_results = final_eval_results if final_eval_results is not None else eval_results
    has_final_evidence = all(hasattr(terminal_eval_results, field) for field in episode_fields)
    evaluation_episode_count = len(getattr(eval_results, "rewards", [])) if has_episode_evidence else 30
    if not has_episode_evidence:
        raise ResultBundleError("JAX selected evaluation evidence is missing episode arrays")
    if evaluation_episode_count <= 0:
        raise ResultBundleError("JAX selected evaluation evidence must contain at least one episode")
    if not has_final_evidence:
        raise ResultBundleError("JAX terminal evaluation evidence is missing episode arrays")
    for label, evidence in (("selected", eval_results), ("final", terminal_eval_results)):
        evidence_lengths = {len(getattr(evidence, field)) for field in episode_fields}
        if evidence_lengths != {evaluation_episode_count}:
            raise ResultBundleError(f"JAX {label} evaluation evidence sequences must have equal lengths")
    selected_rewards = _np.asarray(eval_results.rewards, dtype=float)
    selected_lengths = _np.asarray(eval_results.lengths, dtype=float)
    selected_forward_velocities = _np.asarray(eval_results.forward_vels, dtype=float)
    selected_distances = _np.asarray(eval_results.distances, dtype=float)
    selected_successes = _np.asarray(eval_results.successes, dtype=float)
    final_rewards = _np.asarray(terminal_eval_results.rewards, dtype=float)
    final_lengths = _np.asarray(terminal_eval_results.lengths, dtype=float)
    final_forward_velocities = _np.asarray(terminal_eval_results.forward_vels, dtype=float)
    final_distances = _np.asarray(terminal_eval_results.distances, dtype=float)
    final_successes = _np.asarray(terminal_eval_results.successes, dtype=float)
    stage_results.update(
        {
            "mean_reward": round(float(final_rewards.mean()), 2),
            "std_reward": round(float(final_rewards.std()), 2),
            "mean_episode_length": round(float(final_lengths.mean()), 1),
            "std_episode_length": round(float(final_lengths.std()), 1),
            "mean_forward_vel": round(float(final_forward_velocities.mean()), 3),
            "std_forward_vel": round(float(final_forward_velocities.std()), 3),
            "mean_distance_traveled": round(float(final_distances.mean()), 3),
            "mean_success_rate": round(float(final_successes.mean()), 4),
            "best_model_reward": round(float(selected_rewards.mean()), 2),
            "best_model_std_reward": round(float(selected_rewards.std()), 2),
            "best_model_length": round(float(selected_lengths.mean()), 1),
            "best_model_std_length": round(float(selected_lengths.std()), 1),
            "best_model_fwd_vel": round(float(selected_forward_velocities.mean()), 3),
            "best_model_std_fwd_vel": round(float(selected_forward_velocities.std()), 3),
            "best_model_distance": round(float(selected_distances.mean()), 3),
            "best_model_success_rate": round(float(selected_successes.mean()), 4),
        }
    )

    try:
        captured_provenance = load_provenance(run_dir)
    except ResultBundleError:
        captured_provenance = {}
    seed_roles = captured_provenance.get("seed_roles") or {
        "training": seed,
        "publication_evaluation": evaluation_seed,
    }
    initialize_result_bundle(
        run_dir,
        species=species,
        algorithm="JAX_PPO",
        backend="jax-mjx",
        seed=seed,
        evaluation_seeds=[evaluation_seed],
        evaluation_episodes=evaluation_episode_count,
        seed_roles=seed_roles,
        parallel_envs=num_envs,
        hardware=str(captured_provenance.get("hardware") or "Google Colab"),
        plant_identity=plant_identity.to_dict(),
        run_id=captured_provenance.get("run_id"),
    )
    captured_provenance = load_provenance(run_dir)
    if manifest_path.exists():
        manifest_path.unlink()

    model_dir = stage_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    write_plant_identity(run_dir / "plant_identity.json", plant_identity)
    stage_results["plant_identity"] = plant_identity.to_dict()

    # 1. Stage summary text file
    summary_path = text_summaries.write_stage_summary(stage_dir, stage_results, species, "JAX/MJX PPO")
    paths["stage_summary"] = summary_path
    logger.info("Stage summary saved: %s", summary_path)

    # 2. Stage config snapshot
    config_path = save_stage_config(
        stage_dir,
        stage=stage,
        stage_config=stage_config,
        algorithm="jax_ppo",
        species=species,
        extra={
            "seed": seed,
            "num_envs": num_envs,
            "reward_cfg": reward_cfg or {},
        },
        plant_identity=plant_identity,
    )
    paths["stage_config"] = config_path
    logger.info("Stage config saved: %s", config_path)

    # 3. Per-episode evidence for both the selected and terminal parameters.
    evaluation_path = csv_output.save_evaluation_episodes(
        stage_dir,
        rewards=eval_results.rewards,
        lengths=eval_results.lengths,
        forward_velocities=eval_results.forward_vels,
        distances=eval_results.distances,
        successes=eval_results.successes,
        evaluation_seed=evaluation_seed,
        checkpoint_label="selected",
    )
    paths["evaluation_episodes"] = evaluation_path
    final_evaluation_path = csv_output.save_evaluation_episodes(
        stage_dir,
        rewards=terminal_eval_results.rewards,
        lengths=terminal_eval_results.lengths,
        forward_velocities=terminal_eval_results.forward_vels,
        distances=terminal_eval_results.distances,
        successes=terminal_eval_results.successes,
        evaluation_seed=evaluation_seed,
        checkpoint_label="final",
    )
    paths["final_evaluation_episodes"] = final_evaluation_path

    # 4. Diagnostics NPZ from eval results
    diag_data: dict[str, Any] = {
        # JAX diagnostics are a contiguous evaluation trace rather than
        # rollout snapshots. Expose a compatible step axis for dashboards.
        "timesteps": _np.arange(len(eval_results.diag_tilt), dtype=int),
        "tilt_angle": _np.array(eval_results.diag_tilt),
        "forward_vel": _np.array(eval_results.diag_fwd_vel),
        "pelvis_height": _np.array(eval_results.diag_pelvis_h),
        "energy": _np.array(eval_results.diag_energy),
    }
    if eval_results.diag_l_foot:
        diag_data["l_foot_contact"] = _np.array(eval_results.diag_l_foot)
        diag_data["r_foot_contact"] = _np.array(eval_results.diag_r_foot)
    for comp_name, comp_vals in eval_results.diag_reward_components.items():
        diag_data[f"reward_{comp_name}"] = _np.array(comp_vals)
    for diagnostic_name, diagnostic_vals in getattr(eval_results, "diag_reward_diagnostics", {}).items():
        diag_data[diagnostic_name] = _np.array(diagnostic_vals)
    if species.lower() == "trex" and stage == 1 and eval_results.diag_l_foot and eval_results.diag_r_foot:
        from ..stance_diagnostics import derive_stance_info

        derived_rows = [
            derive_stance_info(
                {
                    "r_foot_contact": right_force,
                    "l_foot_contact": left_force,
                }
            )
            for right_force, left_force in zip(
                eval_results.diag_r_foot,
                eval_results.diag_l_foot,
                strict=True,
            )
        ]
        if derived_rows:
            for diagnostic_name in derived_rows[0]:
                diag_data[diagnostic_name] = _np.array(
                    [row[diagnostic_name] for row in derived_rows],
                    dtype=float,
                )

    diag_path = stage_dir / "diagnostics.npz"
    _np.savez(diag_path, **diag_data)
    paths["diagnostics"] = diag_path
    logger.info("Diagnostics saved: %s", diag_path)

    # 5. Model checkpoints (best + final)
    effective_best = best_params if best_params is not None else params
    best_model_path = model_dir / "best_model.pkl"
    save_checkpoint(
        best_model_path,
        effective_best,
        obs_rms=obs_rms,
        extra={"best_reward": best_reward, "best_update": best_update},
        plant_identity=plant_identity,
    )
    paths["best_model"] = best_model_path

    final_model_path = model_dir / f"stage{stage}_final.pkl"
    save_checkpoint(final_model_path, params, obs_rms=obs_rms, plant_identity=plant_identity)
    paths["final_model"] = final_model_path
    logger.info("Models saved: %s, %s", best_model_path, final_model_path)

    # 6. Persist one idempotent stage record for cross-session curricula.
    stage_results["model_path"] = best_model_path.resolve().relative_to(run_dir.resolve()).as_posix()
    persisted_keys = (
        "stage",
        "name",
        "description",
        "timesteps",
        "duration_seconds",
        "mean_reward",
        "std_reward",
        "mean_episode_length",
        "std_episode_length",
        "mean_forward_vel",
        "std_forward_vel",
        "mean_distance_traveled",
        "mean_success_rate",
        "best_eval_reward",
        "best_eval_std",
        "best_eval_length",
        "best_eval_std_length",
        "best_eval_timestep",
        "selection_training_return",
        "selection_training_update",
        "gate_passed",
        "publication_gate_passed",
        # The reasons, not only the boolean. A persisted `false` with no
        # explanation forces a re-run to find out which criterion failed,
        # and for a stance gate the panel that produced it is gone.
        "gate_failures",
        "best_model_reward",
        "best_model_std_reward",
        "best_model_length",
        "best_model_std_length",
        "best_model_fwd_vel",
        "best_model_std_fwd_vel",
        "best_model_distance",
        "best_model_success_rate",
        "model_path",
        "plant_identity",
    )
    persisted_result = {key: stage_results[key] for key in persisted_keys if key in stage_results}
    stage_result_path = stage_dir / "stage_result.json"
    stage_result_path.write_text(_json.dumps(persisted_result, indent=2, sort_keys=True) + "\n")
    paths["stage_result"] = stage_result_path

    accumulated_results: list[dict[str, Any]] = []
    accumulated_configs: dict[int, dict[str, Any]] = {}
    for existing_result_path in sorted(run_dir.glob("stage*/stage_result.json")):
        saved_result = _json.loads(existing_result_path.read_text())
        saved_stage = int(saved_result["stage"])
        accumulated_results.append(saved_result)
        saved_config_path = existing_result_path.parent / "stage_config.json"
        saved_config = _json.loads(saved_config_path.read_text())
        accumulated_configs[saved_stage] = {
            "name": saved_config.get("name", f"Stage {saved_stage}"),
            "description": saved_config.get("description", f"Curriculum stage {saved_stage}"),
            "env_kwargs": saved_config.get("reward_weights", {}),
            "jax_kwargs": saved_config.get("hyperparameters", {}),
            "curriculum_kwargs": saved_config.get("curriculum", {}),
        }
    accumulated_results.sort(key=lambda result: int(result["stage"]))

    # 7. Training summary (run-level, regenerated from every saved stage)
    training_summary_path = text_summaries.write_training_summary(
        run_dir,
        accumulated_results,
        species,
        algorithm="JAX/MJX PPO",
        seed=seed,
        n_envs=num_envs,
    )
    paths["training_summary"] = training_summary_path
    logger.info("Training summary saved: %s", training_summary_path)

    # 8. Canonical CSV/provenance/manifest; summary.json appears at Stage 3.
    bundle_paths = bundles.save_result_bundle(
        accumulated_results,
        accumulated_configs,
        species,
        "JAX_PPO",
        seed,
        run_dir,
        backend="jax-mjx",
        backend_version=backend_version,
        parallel_envs=num_envs,
        hardware=str(captured_provenance.get("hardware") or "Google Colab"),
        evaluation_episodes=evaluation_episode_count,
        evaluation_seeds=[evaluation_seed],
        seed_roles=captured_provenance.get("seed_roles"),
        plant_identity=plant_identity.to_dict(),
        run_id=captured_provenance.get("run_id"),
    )
    paths.update(bundle_paths)
    logger.info("Canonical JAX result bundle saved: %s", run_dir)

    return paths
