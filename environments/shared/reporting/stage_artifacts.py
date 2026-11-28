"""Post-training stage artifact generation.

:func:`generate_stage_artifacts` is the SB3 training notebook's post-training
step for every node it trains or judges (the chain loop and the manual node
cell), so every such stage records the same artifacts: its certificate,
verdict, summary, figures, replays and probes.  Two other callers use parts of
this module: the gate backfill tool (``scripts/backfill_gate_verdict.py``)
re-derives a verdict through :func:`build_stage_results_from_eval_data` and the
stance stage-result helpers and writes ``gate_verdict.json`` alone, and the
command-line curriculum (``train_base``) rolls a floor-truth stance certificate
through ``_write_stance_gate_report`` and a walk's ``command_tracking/v1``
report and figures through ``_write_command_tracking_report`` and
:func:`write_command_tracking_figures`."""

from __future__ import annotations

import contextlib
import json as _json
import logging
import math
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from ..curriculum.checkpoints import select_handoff_checkpoint
from ..curriculum.stance_gate import DEFAULT_MIN_EVAL_EPISODES_STANCE
from ..model_id import ModelId
from ..policy_loading import load_sb3_model
from . import csv_output, stage_layout, text_summaries

if TYPE_CHECKING:
    from ..curriculum.stance_heading_gate import StanceHeadingResult

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
#: Episodes per row in the heading probe (``stance_probe_spawn_yaw_deg``): a
#: fall or a full horizon again, over 1 + 2 x offsets rows, policy and statue.
_HEADING_EPISODES = 8
#: Episodes in each of :func:`evaluate_stage_checkpoints`' two panels, the
#: final checkpoint's and the selected one's; the stage summary states it.
EVALUATION_PANEL_EPISODES = 30


def build_stage_results_from_eval_data(
    stage_dir: "str | Path",
    stage: int,
    stage_config: dict[str, Any],
    timesteps: int,
    duration_seconds: float = 0.0,
    sim_dt: float | None = None,
) -> dict[str, Any]:
    """Build a ``stage_results`` dict from on-disk evaluation artifacts.

    Reads ``evaluations.npz`` (written by SB3's ``EvalCallback``) and
    ``metrics.json`` to reconstruct the same results dict that the
    training notebook's ``train_stage`` produces.  This lets the gate
    backfill tool and any other post-hoc consumer build a consistent
    results dict without re-running evaluation.

    If *duration_seconds* is 0 and a ``metrics.json`` exists, the duration
    is read from ``training_duration_seconds`` in that file.

    The post-training velocity/success panel's numbers (``mean_forward_vel``,
    ``std_forward_vel``, ``mean_distance_traveled``, ``mean_success_rate``)
    are read from ``metrics.json`` too.  When the panel was skipped or failed
    the keys are OMITTED rather than zeroed, so the gate reads them as
    unmeasured; ``best_model_*`` default to ``""`` and can be filled by the
    caller after running ``eval_policy``.

    *sim_dt* is the env's control step, which the summaries multiply episode
    lengths by to print sim time. Pass the env's ``dt``, as
    :func:`generate_stage_artifacts` and :func:`evaluate_stage_checkpoints`
    do; without it the value falls back to ``env_kwargs["sim_dt"]`` or 0.01 s,
    which is wrong for the compsognathus pair (0.02 s).
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
    # The velocity/success panel (train_base._post_training_eval_panels)
    # writes these for the handoff checkpoint.  They used to be hardcoded to
    # 0.0 here although this file was already open, so every stage-2/3 sweep
    # trial failed its velocity gate at "0.00 m/s" whatever it had learned
    # (review ER4).  A skipped (post_eval_skipped) or failed panel leaves a
    # key absent or null; it is then left OUT so gates._gate_metric reads it
    # as unmeasured rather than as a fabricated zero.  Deferred import: the
    # curriculum package reaches back into reporting at import time.
    from environments.shared.curriculum.gate_schema import finite_gate_metric

    measured_panel = {
        key: value
        for key in ("mean_forward_vel", "std_forward_vel", "mean_distance_traveled", "mean_success_rate")
        if (value := finite_gate_metric(metrics.get(key))) is not None
    }
    plant_identity = metrics.get("plant_identity")
    if not isinstance(plant_identity, Mapping):
        saved_config_path = stage_dir / "stage_config.json"
        if saved_config_path.exists():
            saved_config = _json.loads(saved_config_path.read_text(encoding="utf-8"))
            plant_identity = saved_config.get("plant_identity")

    # The SELECTED checkpoint, so the `model_path` this records is the one the
    # replay shows and the evidence CSV is evidence for. It used to hardcode
    # `best_model` while the replay, the stance gate report and the next-stage
    # handoff all resolved through `select_handoff_checkpoint`, which prefers
    # the risk-adjusted `robust_best_model`. This path feeds the gate backfill
    # tool and `generate_stage_artifacts` called without `stage_results`,
    # where nothing else re-derives it.
    #
    # Falls back to `best_model` rather than skipping when no candidate is
    # complete: unlike the replay and the gate report, this function only
    # *describes* a run, and an empty model_path would lose information the
    # caller can still check for itself.
    handoff = select_handoff_checkpoint(model_dir)
    if handoff is None:
        best_model_path = model_dir / "best_model"
        vecnorm_path = str(model_dir / "best_model_vecnorm.pkl")
    else:
        _, selected_path, vecnorm_path = handoff
        best_model_path = Path(selected_path)
    if sim_dt is None:
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
        **measured_panel,
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
    model_id: ModelId,
    stage: "int | str",
    stage_config: dict[str, Any],
    stage_dir: Path,
    model_dir: Path,
) -> dict[str, Any] | None:
    """Score the selected checkpoint against the stance gate, into the run dir.

    Returns the report dict, or ``None`` when no panel was rolled.  The return
    value is what :func:`_apply_stage_gate` certifies a stance-gated stage
    from, so ``None`` fails that stage closed.

    Only for stages that actually declare a stance kind
    (``gate_schema.STANCE_GATE_KINDS``: ``stance_quality/v1``, or
    ``stance_quality/v2``, whose report is scored on floor truth with a
    statue panel when its criteria need one, or the six-heading certificate
    ``stance_quality_heading/v1``, D-D32, rolled by
    :func:`_write_stance_heading_certificate`): rolling a 40-episode panel
    costs a few minutes, which is nothing beside a multi-hour stage but is
    pure waste for a stage the criteria do not govern.

    Runs here rather than in the notebook so it happens for every SB3 run
    without anyone remembering, and lands beside
    ``stage_summary.txt`` in the run directory -- which is on Drive, so the
    verdict survives a lost runtime.

    ``stance_report_episodes`` in ``[curriculum]`` overrides the panel size;
    ``0`` skips the report entirely. It exists because "a few minutes" is per
    stage, which a smoke or debug run need not pay (the tuning sweeps,
    retired by D-D17, paid it once per trial). Overriding
    downward makes the bound weaker than the gate claims -- the panel size is
    what its power is specified at -- so the log says so when it happens.

    Deliberately non-fatal to *artifact generation*: losing it must never cost
    a completed training run its summary, replays and graphs, and the
    checkpoint may legitimately be absent (a stage stopped before its first
    evaluation produced one).  It is emphatically fatal to *certification* —
    a stance-gated stage with no panel fails, because the alternative is
    certifying stance quality nobody measured.
    """
    from environments.shared.curriculum.gate_schema import STANCE_GATE_KINDS

    curriculum = stage_config.get("curriculum_kwargs", {})
    if curriculum.get("gate_kind") not in STANCE_GATE_KINDS:
        return None

    declared_episodes = int(curriculum.get("min_eval_episodes", DEFAULT_MIN_EVAL_EPISODES_STANCE))
    report_episodes = curriculum.get("stance_report_episodes")
    report_episodes = declared_episodes if report_episodes is None else int(report_episodes)
    if report_episodes < 1:
        logger.info(
            "Stance gate report skipped for stage %s: stance_report_episodes = %d",
            stage,
            report_episodes,
        )
        return None
    if report_episodes != declared_episodes:
        logger.warning(
            "Stance gate report for stage %s rolls %d episodes, not the stage's min_eval_episodes "
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
    handoff = select_handoff_checkpoint(model_dir)
    if handoff is None:
        logger.warning(
            "Stance gate report skipped for stage %s: no checkpoint in %s has its "
            "matched _vecnorm.pkl, and scoring without the observation statistics "
            "would report a verdict for a different policy.",
            stage,
            model_dir,
        )
        return None

    selected_name, selected_path, selected_vecnorm = handoff
    from ..curriculum.stance_heading_gate import STANCE_HEADING_GATE_KIND

    if curriculum.get("gate_kind") == STANCE_HEADING_GATE_KIND:
        heading_report = _write_stance_heading_certificate(
            model_id=model_id,
            stage=stage,
            stage_config=stage_config,
            stage_dir=stage_dir,
            selected_name=selected_name,
            model_path=f"{selected_path}.zip",
            vecnorm_path=selected_vecnorm,
            episodes=report_episodes,
        )
        # Beside the certificate, after it returned, whatever it returned: never read by the gate.
        _write_handoff_sidecar(stage_dir, selected_name, f"{selected_path}.zip", stage_config)
        return heading_report
    report: dict[str, Any] | None = None
    try:
        from environments.shared.reporting.stance_report import (
            build_stance_gate_report,
            write_stance_gate_report,
        )

        logger.info("Stance gate report scoring stage %s checkpoint: %s", stage, selected_name)
        report = build_stance_gate_report(
            model_id,
            stage,
            stage_config=stage_config,
            model_path=f"{selected_path}.zip",
            vecnorm_path=selected_vecnorm,
            episodes=report_episodes,
        )
        written = write_stance_gate_report(stage_dir, report)
        if "result" in report:
            # stance_quality/v2: the clean count and its bound are the verdict.
            logger.info(
                "Stance gate report: %s (%d/%d episodes clean, LCB95 %.4f) -> %s",
                "PASS" if report["passed"] else "FAIL",
                report["result"]["n_clean"],
                report["result"]["n_episodes"],
                report["result"]["clean_lcb"],
                written["stance_gate_report_txt"],
            )
        else:
            logger.info(
                "Stance gate report: %s (duty %.4f, bilateral %.4f) -> %s",
                "PASS" if report["passed"] else "FAIL",
                report["metrics"]["mean_unsupported_duty"],
                report["metrics"]["bilateral_support_duty"],
                written["stance_gate_report_txt"],
            )
    except Exception:  # noqa: BLE001 - a diagnostic must not sink the run
        logger.warning("Stance gate report failed for stage %s", stage, exc_info=True)
        report = None
    # Beside the report, after its attempt returned, whatever it returned: never read by the gate.
    _write_handoff_sidecar(stage_dir, selected_name, f"{selected_path}.zip", stage_config)
    return report


#: The stage-end sidecar that says which in-training evaluation the handoff checkpoint was saved at, and its
#: score; beside the certificate, never inside it (``_write_handoff_sidecar``).
HANDOFF_SIDECAR_FILENAME = "handoff_in_training.json"
#: Each handoff's selection rule: ``RobustBestModelCallback``'s ``mean - risk_coef x std`` at the ``risk_coef``
#: train_base uses (its default, 1.0), and SB3's mean for ``best_model``.  On a live-command stage the robust
#: rule is the tracker's family-aware one instead (:func:`_handoff_in_training`, decision D-D62).
_HANDOFF_SELECTION_RULES: dict[str, tuple[str, float]] = {
    "robust_best_model": ("mean - 1.0 x std", 1.0),
    "best_model": ("mean", 0.0),
}
_HANDOFF_KEYS: tuple[str, ...] = (
    "selected",
    "timesteps",
    "mean_reward",
    "std_reward",
    "episodes",
    "selection_rule",
    "selection_score",
    "first_scoring_timesteps",
    "step_agrees",
)


def _live_stop_probability(stage_config: "Mapping[str, Any] | None") -> float | None:
    """A live-command stage's stop probability (its stand family's weight); ``None`` under command_mode "none".

    Read from the stage config's ``env_kwargs``: the ``command_config`` a
    loaded stage holds, or the record a recorded one keeps.  Raises
    ``LookupError`` for a live stage whose config holds no stop probability.
    """
    from ..command_frame import COMMAND_MODE_NONE

    env_kwargs = (stage_config or {}).get("env_kwargs") or {}
    if str(env_kwargs.get("command_mode", COMMAND_MODE_NONE)) == COMMAND_MODE_NONE:
        return None
    command = env_kwargs.get("command_config")
    value = (
        command.get("stop_probability") if isinstance(command, Mapping) else getattr(command, "stop_probability", None)
    )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LookupError("the live-command stage's config records no command stop_probability")
    return float(value)


def _handoff_in_training(
    stage_dir: "str | Path",
    selected_name: str,
    model_zip: "str | Path",
    stage_config: "Mapping[str, Any] | None" = None,
) -> dict[str, Any]:
    """Which in-training evaluation the handoff checkpoint was saved at, and its score there.  Never raises.

    ``timesteps`` is the checkpoint's own ``num_timesteps`` (SB3 stores it
    in the zip's ``data`` member at save; read with
    ``task_fingerprint.read_checkpoint_attribute``, no torch load).  The evaluation
    of that exact step in the stage's ``evaluations.npz`` gives
    ``mean_reward``, ``std_reward`` (ddof 0, as ``RobustBestModelCallback``)
    and ``episodes``; ``selection_rule`` and ``selection_score`` are the
    handoff's own (``mean - 1.0 x std`` for ``robust_best_model``, ``mean``
    for ``best_model``).  As a cross-check, ``first_scoring_timesteps`` is
    the step of the first evaluation whose score equals the history's best
    (``progress_report._first_evaluation_scoring``, ``progress.json``'s
    reading), and ``step_agrees`` whether it is the checkpoint's step.

    On a live-command stage (*stage_config*'s ``command_mode`` is not
    ``"none"``, decision D-D62) ``robust_best_model`` is scored by the
    tracker's family-aware rule
    (:func:`~environments.shared.curriculum.command_evaluation.family_robust_score`)
    on the families each evaluation rolled, which the record carries
    (``evaluations.npz``'s ``command_families``), weighted at the stage's
    stop probability: the number ``RobustBestModelCallback`` compared, bit
    for bit, named ``family-weighted mean - 1.0 x std (cruise 0.85, stand
    0.15)``.  A live record without its families scores nothing (an
    ``error``).  Under command_mode "none", or without *stage_config*, the
    block is the pooled one, exactly as before.

    On any failure (a truncated zip, a zip without ``data``, no
    ``evaluations.npz``, no evaluation at the checkpoint's step) it returns
    the same keys, ``None`` where unknown, plus ``"error"``.
    """
    out: dict[str, Any] = dict.fromkeys(_HANDOFF_KEYS)
    out["selected"] = str(selected_name)
    try:
        import numpy as np

        from ..curriculum.command_evaluation import (
            family_robust_score,
            family_selection_rule,
            family_weights,
            recorded_command_families,
        )
        from ..curriculum.progress_report import _first_evaluation_scoring
        from ..task_fingerprint import read_checkpoint_attribute

        recorded = read_checkpoint_attribute(model_zip, "num_timesteps")
        if recorded is None:
            raise LookupError(f"{Path(model_zip).name} records no num_timesteps (not a readable SB3 zip)")
        timesteps = int(recorded)
        out["timesteps"] = timesteps
        rule = _HANDOFF_SELECTION_RULES.get(str(selected_name))
        stop_probability = _live_stop_probability(stage_config)
        with np.load(Path(stage_dir) / "evaluations.npz") as record:
            steps = [int(value) for value in record["timesteps"]]
            results = [np.asarray(row, dtype=float) for row in record["results"]]
            families = None if stop_probability is None else recorded_command_families(record)
        if timesteps not in steps:
            raise LookupError(f"evaluations.npz holds no evaluation at the checkpoint's {timesteps:,} steps")
        rewards = results[steps.index(timesteps)]
        mean, std = float(rewards.mean()), float(rewards.std())
        out.update(mean_reward=mean, std_reward=std, episodes=int(rewards.size))
        if rule is not None and stop_probability is not None and str(selected_name) == "robust_best_model":
            # A live stage: the tracker's family-aware score of each evaluation, on the families it rolled.
            risk = rule[1]
            if families is None:
                raise LookupError(
                    "evaluations.npz records no command families, and a live-command stage's robust score is "
                    "family-aware (decision D-D62)"
                )
            weights = family_weights(stop_probability)

            def family_score(values: Any, labels: Any) -> float:
                return family_robust_score(values, labels, weights, risk_coef=risk)

            scores = [family_score(row, labels) for row, labels in zip(results, families)]
            out["selection_rule"] = family_selection_rule(stop_probability, risk)
            out["selection_score"] = scores[steps.index(timesteps)]
            out["first_scoring_timesteps"] = _first_evaluation_scoring(
                results, steps, max(scores), family_score, families
            )
            out["step_agrees"] = out["first_scoring_timesteps"] == timesteps
        elif rule is not None:
            name, risk = rule

            def score(values: Any) -> float:
                return float(np.mean(values) - risk * np.std(values))

            best = max(score(row) for row in results)
            out["selection_rule"] = name
            out["selection_score"] = score(rewards)
            out["first_scoring_timesteps"] = _first_evaluation_scoring(results, steps, best, score)
            out["step_agrees"] = out["first_scoring_timesteps"] == timesteps
    except Exception as exc:  # noqa: BLE001 - a sidecar must never cost the certificate or the run
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def _write_handoff_sidecar(
    stage_dir: "str | Path",
    selected_name: str,
    model_zip: "str | Path",
    stage_config: "Mapping[str, Any] | None" = None,
) -> None:
    """Write ``handoff_in_training.json`` beside the certificate and log one line; never raises.

    Called after the certificate attempt has returned, on every stance
    path and on the command-tracking path (with the stage's config, whose
    command mode says how ``robust_best_model`` was scored), and never read
    by ``_apply_stage_gate``: the certificate's bytes, its verdict and
    ``gate_verdict.json`` are what they are without it.
    """
    try:
        from ..file_io import atomic_write_json

        block = _handoff_in_training(stage_dir, selected_name, model_zip, stage_config)
        atomic_write_json(Path(stage_dir) / HANDOFF_SIDECAR_FILENAME, block)

        def shown(value: Any, spec: str = "") -> str:
            return "n/a" if value is None else format(value, spec)

        logger.info(
            "Handoff in training: %s, saved at %s steps: evaluation mean %s, std %s over %s episodes (%s = %s)",
            block["selected"],
            shown(block["timesteps"], ","),
            shown(block["mean_reward"], ".1f"),
            shown(block["std_reward"], ".1f"),
            shown(block["episodes"]),
            shown(block["selection_rule"]),
            shown(block["selection_score"], ".1f"),
        )
    except Exception:  # noqa: BLE001 - a sidecar must never cost the certificate or the run
        logger.warning("Could not write %s; the certificate is unaffected.", HANDOFF_SIDECAR_FILENAME, exc_info=True)


def _write_stance_heading_certificate(
    *,
    model_id: ModelId,
    stage: "int | str",
    stage_config: dict[str, Any],
    stage_dir: Path,
    selected_name: str,
    model_path: str,
    vecnorm_path: str,
    episodes: int,
) -> dict[str, Any] | None:
    """Roll the six-heading certificate on the selected handoff pair (D-D32), into the run dir.

    :func:`_write_stance_gate_report`'s twin for ``stance_quality_heading/v1``,
    with its handoff selection, its ``stance_report_episodes`` override (the
    seeds per heading) and its non-fatal contract: an exception -- a lost
    runtime partway through the grid, a rollout that raised at one heading --
    is logged and gives ``None``, which :func:`_apply_stage_gate` fails
    closed.  Never a fallback to the 0 degree cell: no ``stance_gate_report.*``
    or ``stance_panel_selected.csv`` is written for such a stage, so nothing
    the judge or publication reads can stand in for the grid.

    About 13 minutes on the T. rex (r9) and 21-22 on the anatomical
    compsognathus at 40 seeds: 6 policy and 6 statue panels.  Logged per
    heading by the builder.
    """
    try:
        from environments.shared.reporting.stance_report import (
            build_stance_heading_report,
            write_stance_heading_report,
        )

        logger.info("Six-heading stance certificate scoring stage %s checkpoint: %s", stage, selected_name)
        report = build_stance_heading_report(
            model_id,
            stage,
            stage_config=stage_config,
            model_path=model_path,
            vecnorm_path=vecnorm_path,
            episodes=episodes,
        )
        written = write_stance_heading_report(stage_dir, report)
        result = report["result"]
        logger.info(
            "Six-heading stance certificate: %s (%d/%d seeds clean at every heading, LCB95 %.4f) -> %s",
            "PASS" if report["passed"] else "FAIL",
            result["n_clean_seeds"],
            result["n_seeds"],
            result["clean_seed_lcb"],
            written["stance_heading_report_txt"],
        )
        return report
    except Exception:  # noqa: BLE001 - the certificate must not sink the run; its absence fails the gate
        logger.warning(
            "Six-heading stance certificate failed for stage %s; the stage fails its gate (no yaw-zero fallback)",
            stage,
            exc_info=True,
        )
    return None


def _command_reference_pair(stage_dir: Path, curriculum: Mapping[str, Any]) -> "tuple[str, str] | None":
    """The follow profile's reference pair: the parent checkpoint the stage entered on, when the gate declares one.

    Only for a stage declaring ``min_follow_reference_delta_lcb``; read from
    the stage's recorded entry edge (``stage_config.json``'s
    ``run.load_path``, with its ``_vecnorm.pkl`` sidecar beside it), and
    only when the checkpoint still hashes to the recorded
    ``run.parent_checkpoint_sha256``.  ``None`` otherwise (logged), which
    the gate then fails closed on: a declared comparison with no reference
    panel certifies nothing.
    """
    from ..config import read_recorded_stage_config
    from ..result_bundle.hashing import sha256_file

    if curriculum.get("min_follow_reference_delta_lcb") is None:
        return None
    record = read_recorded_stage_config(stage_dir) or {}
    run = record.get("run") if isinstance(record.get("run"), Mapping) else {}
    load_path = run.get("load_path") if isinstance(run, Mapping) else None
    recorded = run.get("parent_checkpoint_sha256") if isinstance(run, Mapping) else None
    if not isinstance(load_path, str) or not load_path:
        logger.warning("Stage %s declares a paired reference but records no parent checkpoint it entered on", stage_dir)
        return None
    zip_path = Path(load_path)
    vecnorm_path = Path(str(zip_path)[: -len(".zip")] + "_vecnorm.pkl") if zip_path.suffix == ".zip" else None
    if vecnorm_path is None or not zip_path.is_file() or not vecnorm_path.is_file():
        logger.warning("Stage %s's reference pair %s is not complete on disk; no reference panel", stage_dir, zip_path)
        return None
    if sha256_file(zip_path) != recorded:
        logger.warning("Stage %s's reference %s no longer hashes to the recorded parent digest", stage_dir, zip_path)
        return None
    return str(zip_path), str(vecnorm_path)


def _write_command_tracking_report(
    *,
    model_id: ModelId,
    stage: "int | str",
    stage_config: dict[str, Any],
    stage_dir: Path,
    model_dir: Path,
) -> dict[str, Any] | None:
    """Roll the ``command_tracking/v1`` panel on the selected handoff pair (D-D61), into the run dir.

    Only for a stage that declares the kind.  The handoff is
    ``select_handoff_checkpoint``'s, which for this kind is the pair the
    handoff choice (``models/handoff_choice.json``) chose by the gated
    statistic; the judge refuses a report whose pair is not that choice.
    The follow profile's reference, when the gate declares one, is the
    parent checkpoint the stage entered on (:func:`_command_reference_pair`).
    Non-fatal to artifact generation, like the stance reports, and fatal to
    certification: ``None`` (no pair, or a rollout that raised) fails the
    stage closed in :func:`_apply_stage_gate`.
    """
    from ..curriculum.command_tracking_gate import COMMAND_TRACKING_GATE_KIND

    curriculum = stage_config.get("curriculum_kwargs", {})
    if curriculum.get("gate_kind") != COMMAND_TRACKING_GATE_KIND:
        return None
    handoff = select_handoff_checkpoint(model_dir)
    if handoff is None:
        logger.warning(
            "Command-tracking report skipped for stage %s: no checkpoint in %s has its matched _vecnorm.pkl",
            stage,
            model_dir,
        )
        return None
    selected_name, selected_path, selected_vecnorm = handoff
    report: dict[str, Any] | None = None
    try:
        from .command_tracking_report import build_command_tracking_report, write_command_tracking_report

        reference = _command_reference_pair(stage_dir, curriculum)
        logger.info("Command-tracking report scoring stage %s checkpoint: %s", stage, selected_name)
        report = build_command_tracking_report(
            model_id,
            stage,
            stage_config=stage_config,
            model_path=f"{selected_path}.zip",
            vecnorm_path=selected_vecnorm,
            reference_model_path=None if reference is None else reference[0],
            reference_vecnorm_path=None if reference is None else reference[1],
        )
        written = write_command_tracking_report(stage_dir, report)
        logger.info(
            "Command-tracking report (%s): %s -> %s",
            report["profile"],
            "PASS" if report["passed"] else "FAIL",
            written["command_tracking_report_txt"],
        )
    except Exception:  # noqa: BLE001 - the report must not sink the run; its absence fails the gate
        logger.warning("Command-tracking report failed for stage %s; the stage fails its gate", stage, exc_info=True)
        report = None
    _write_handoff_sidecar(stage_dir, selected_name, f"{selected_path}.zip", stage_config)
    return report


def ensure_command_handoff_choice(
    *,
    model_id: ModelId,
    stage: "int | str",
    stage_config: Mapping[str, Any],
    model_dir: "str | Path",
    final_stem: "str | Path | None" = None,
) -> "dict[str, Any] | None":
    """Make a ``command_tracking/v1`` stage's handoff choice when it has no valid one under its current gate.

    A no-op (``None``) for every other kind and for a stage whose
    ``models/handoff_choice.json`` is valid, was made under the gate and
    task the stage declares now, and still ranks exactly the stage's pairs
    (``checkpoints.handoff_choice_candidates_problem``: none rewritten or
    added since); otherwise the choice is made
    (``command_tracking_report.choose_command_handoff``) and its record
    returned.  Called before anything reads the handoff: at the end of
    training (``train_base``) and by :func:`evaluate_stage_checkpoints`, so a
    node judged under an edited gate is handed on by that gate's statistic.
    Never raises: a choice that fails is logged, and the judge then refuses
    the stage for want of one.
    """
    from ..curriculum.checkpoints import handoff_choice_candidates_problem, recorded_handoff_choice
    from ..curriculum.command_tracking_gate import COMMAND_TRACKING_GATE_KIND
    from ..curriculum.gate_schema import gate_config_sha256, gate_config_view

    curriculum = stage_config.get("curriculum_kwargs", {})
    if curriculum.get("gate_kind") != COMMAND_TRACKING_GATE_KIND:
        return None
    try:
        from ..task_fingerprint import stage_task_fingerprint
        from .command_tracking_report import choose_command_handoff

        choice, _ = recorded_handoff_choice(model_dir)
        gate = gate_config_sha256(gate_config_view(curriculum))
        task = stage_task_fingerprint(model_id, stage, env_kwargs=dict(stage_config["env_kwargs"]))["task_sha256"]
        if (
            choice is not None
            and choice.get("gate_sha256") == gate
            and choice.get("task_sha256") == task
            and handoff_choice_candidates_problem(choice, model_dir) is None
        ):
            return None
        return choose_command_handoff(
            model_id, stage, stage_config=stage_config, model_dir=model_dir, final_stem=final_stem
        )
    except Exception:  # noqa: BLE001 - a failed choice is refused by the judge, never a crash after training
        logger.warning("Stage %s's handoff could not be chosen by the gated statistic", stage, exc_info=True)
        return None


def remove_stale_command_tracking_outputs(
    *,
    stage: "int | str",
    stage_config: Mapping[str, Any],
    stage_dir: "str | Path",
    figures_out: "str | Path | None" = None,
) -> None:
    """Remove the trace and command figures an earlier judgement left behind (D-D61).  Never raises.

    A no-op for every kind but ``command_tracking/v1``.  Removes
    ``command_tracking_trace.csv`` from the stage directory and both command
    figures from the stage directory's published ``figures/`` and, when
    given, from *figures_out*: on the notebook path the figures are drawn
    into a staging directory that is only copied over the stage directory,
    so the published copies must be removed here, or a figure of an earlier
    judgement would stand beside this judgement's verdict.
    ``generate_stage_artifacts`` calls it before its figure block, whether or
    not graphs are generated, and :func:`write_command_tracking_figures`
    calls it before drawing.
    """
    from ..curriculum.command_tracking_gate import COMMAND_TRACKING_GATE_KIND
    from ..visualization import COMMAND_GATE_PROGRESS_FIGURE, COMMAND_TRACKING_FIGURE
    from .command_tracking_trace import COMMAND_TRACKING_TRACE_CSV

    if stage_config.get("curriculum_kwargs", {}).get("gate_kind") != COMMAND_TRACKING_GATE_KIND:
        return
    stage_dir = Path(stage_dir)
    figure_dirs = [stage_layout.figures_dir(stage_dir)]
    if figures_out is not None and Path(figures_out) != figure_dirs[0]:
        figure_dirs.append(Path(figures_out))
    stale_paths = [stage_dir / COMMAND_TRACKING_TRACE_CSV]
    for directory in figure_dirs:
        stale_paths.extend((directory / COMMAND_TRACKING_FIGURE, directory / COMMAND_GATE_PROGRESS_FIGURE))
    for stale in stale_paths:
        try:
            stale.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove %s, left by an earlier judgement of stage %s", stale, stage)


def write_command_tracking_figures(
    *,
    model_id: ModelId,
    stage: "int | str",
    stage_config: Mapping[str, Any],
    stage_dir: "str | Path",
    figures_out: "str | Path",
    report: "Mapping[str, Any] | None",
    verdict: bool | None = None,
) -> list[Path]:
    """A ``command_tracking/v1`` stage's figures (D-D61), after its verdict: the paths drawn.  Never raises.

    A no-op (``[]``) for every other kind.  *report* is the report this
    judgement built (``_write_command_tracking_report``'s return) and
    *verdict* the stage's verdict on it.  First any trace and command
    figures an earlier judgement left behind are removed
    (:func:`remove_stale_command_tracking_outputs`: in the stage directory
    and in *figures_out*), so the figures never describe another judgement
    than this one; with *report* ``None``
    (the panel could not be rolled, and the gate failed closed) nothing
    more is drawn.  Then the chart episodes are rolled again with their step
    rows (``command_tracking_trace``: the panel's first cruise seed and first
    stand seed) on the pair *report* scored, when its zip and
    ``_vecnorm.pkl`` still hash to the report's handoff digests, and written
    to ``command_tracking_trace.csv`` in the stage directory, each row
    stamped with that checkpoint's digest; then
    ``figures/command_tracking.png`` (commanded against achieved over those
    episodes, and the tracking fraction over training) and
    ``figures/command_gate_progress.png`` (the gated statistics of every
    handoff candidate and of the certified handoff against the bars, titled
    with *verdict*) are drawn into *figures_out*.  Each step is in its own
    guard, so a failure costs that step's file only.  A figure whose draw
    raises or is interrupted (``BaseException``) has its partial file at
    *figures_out* removed before the error is logged or goes on, so a
    staging directory's publish (which runs in a ``finally``) never copies
    a truncated png into the stage's ``figures/``.  Called inside
    ``generate_stage_artifacts``' figure block and by the CLI curriculum's
    post-training judge, after the verdict is formed: nothing here is read
    by the gate.
    """
    from ..curriculum.command_tracking_gate import COMMAND_TRACKING_GATE_KIND
    from ..visualization import COMMAND_GATE_PROGRESS_FIGURE, COMMAND_TRACKING_FIGURE

    curriculum = stage_config.get("curriculum_kwargs", {})
    if curriculum.get("gate_kind") != COMMAND_TRACKING_GATE_KIND:
        return []
    stage_dir = Path(stage_dir)
    figures_out = Path(figures_out)
    drawn: list[Path] = []
    remove_stale_command_tracking_outputs(
        stage=stage, stage_config=stage_config, stage_dir=stage_dir, figures_out=figures_out
    )
    if not isinstance(report, Mapping):
        logger.warning(
            "Stage %s has no command-tracking report from this judgement, so no command figure is drawn", stage
        )
        return drawn
    handoff = report.get("handoff")
    checkpoint = handoff.get("checkpoint") if isinstance(handoff, Mapping) else None
    if isinstance(checkpoint, str) and checkpoint.endswith(".zip"):
        try:
            from ..result_bundle.hashing import sha256_file
            from .command_tracking_trace import roll_command_trace, write_command_tracking_trace

            assert isinstance(handoff, Mapping)
            zip_path = stage_dir / "models" / checkpoint
            vecnorm_path = stage_dir / "models" / f"{checkpoint[: -len('.zip')]}_vecnorm.pkl"
            if (
                not zip_path.is_file()
                or not vecnorm_path.is_file()
                or sha256_file(zip_path) != handoff.get("checkpoint_sha256")
                or sha256_file(vecnorm_path) != handoff.get("normalization_sha256")
            ):
                logger.warning(
                    "Stage %s's command-tracking trace skipped: %s is no longer the pair its report scored",
                    stage,
                    zip_path,
                )
            else:
                rows = roll_command_trace(
                    model_id,
                    stage,
                    stage_config=stage_config,
                    report=report,
                    model_path=str(zip_path),
                    vecnorm_path=str(vecnorm_path),
                )
                logger.info("Command-tracking trace: %s", write_command_tracking_trace(stage_dir, rows))
        except Exception:  # noqa: BLE001 - the trace costs its own file and figure, never the verdict
            logger.warning(
                "The command-tracking trace failed for stage %s; the verdict is unaffected", stage, exc_info=True
            )
    try:
        from ..visualization import plot_command_gate_progress, plot_command_tracking
    except ImportError:
        logger.warning("Skipping the command-tracking figures (matplotlib not installed).")
        return drawn
    figures_out.mkdir(parents=True, exist_ok=True)
    for name, plot in (
        (COMMAND_TRACKING_FIGURE, plot_command_tracking),
        (COMMAND_GATE_PROGRESS_FIGURE, plot_command_gate_progress),
    ):
        save_path = figures_out / name
        try:
            kwargs: dict[str, Any] = {"model_id": model_id, "save_path": save_path, "show": False}
            if plot is plot_command_tracking:
                kwargs["curriculum_kwargs"] = dict(curriculum)
            else:
                kwargs["verdict"] = verdict
            try:
                figure = plot(stage_dir, **kwargs)
            except BaseException:
                # A savefig that raised or was interrupted (Ctrl-C, a Colab interrupt) part-way can leave a truncated
                # png at save_path.  Remove it before the error goes on, so neither a staging publish (whose finally
                # still copies what staging holds) nor a straight-through write leaves it in the stage's figures/.
                # The sweep above already removed this name's earlier copies, so the figure is simply absent.
                with contextlib.suppress(OSError):
                    save_path.unlink(missing_ok=True)
                raise
            if figure is not None:
                drawn.append(save_path)
        except ImportError:
            logger.warning("Skipping %s (matplotlib not installed).", name)
        except Exception:  # noqa: BLE001 - one figure costs that file only
            logger.warning(
                "The %s figure failed for stage %s; the other figures are unaffected.", name, stage, exc_info=True
            )
    return drawn


def _run_stance_probes(
    *,
    model_id: ModelId,
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

    On a six-heading certificate stage (D-D32) the probes annotate the
    certificate's 0 degree cell (``stance_report.heading_zero_panel``: the
    v2 panel at the training heading), and the report-only heading probe is
    not rolled -- the certificate judged every heading.
    """
    if report is None:
        return

    curriculum = stage_config.get("curriculum_kwargs", {})
    from ..curriculum.stance_heading_gate import STANCE_HEADING_GATE_KIND

    certificate = curriculum.get("gate_kind") == STANCE_HEADING_GATE_KIND
    if certificate:
        try:
            from environments.shared.reporting.stance_report import heading_zero_panel

            report = heading_zero_panel(report)
        except Exception:  # noqa: BLE001 - a diagnostic must not sink the run
            logger.warning("Stance probes skipped for stage %s: no 0 deg cell to annotate", stage, exc_info=True)
            return
    declared_episodes = int(curriculum.get("min_eval_episodes", DEFAULT_MIN_EVAL_EPISODES_STANCE))
    report_episodes = curriculum.get("stance_report_episodes")
    report_episodes = declared_episodes if report_episodes is None else int(report_episodes)

    # The same selector the report itself used; re-resolved here so the probes
    # keep describing the one policy every other artifact describes.
    handoff = select_handoff_checkpoint(model_dir)
    if handoff is None:
        logger.warning(
            "Stance probes skipped for stage %s: no checkpoint in %s has its matched _vecnorm.pkl",
            stage,
            model_dir,
        )
        return
    _selected_name, selected_path, selected_vecnorm = handoff

    probe_calls = (
        (
            "filtered action",
            lambda: _write_filtered_action_probe(
                model_id=model_id,
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
                model_id=model_id,
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
                model_id=model_id,
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
                model_id=model_id,
                stage=stage,
                stage_config=stage_config,
                stage_dir=stage_dir,
                model_path=f"{selected_path}.zip",
                vecnorm_path=selected_vecnorm,
                settle_steps=int(report["settle_steps"]),
            ),
        ),
        (
            "heading",
            lambda: (
                None
                if certificate
                else _write_heading_probe(
                    model_id=model_id,
                    stage=stage,
                    stage_config=stage_config,
                    stage_dir=stage_dir,
                    model_path=f"{selected_path}.zip",
                    vecnorm_path=selected_vecnorm,
                )
            ),
        ),
    )
    for label, run_probe in probe_calls:
        try:
            run_probe()
        except Exception:  # noqa: BLE001 - a diagnostic must not sink the run
            logger.warning("Stance probe (%s) failed for stage %s", label, stage, exc_info=True)


def _probe_cutoffs(raw: Any, *, warn: bool = True) -> list[float]:
    """Normalise ``stance_probe_filter_hz`` to a sorted list of cutoffs.

    Accepts a single number or a list, because the useful reading is a curve
    rather than a point. A single cutoff answers a yes/no that is already
    known to be "no" on this plant -- the checkpoint that PASSED the gate
    falls at every cutoff from 5 to 35 Hz against a 100 Hz control rate
    (issue #491) -- so its PASS/FAIL carries no information. The scalar that
    can actually move is how long the filtered policy survives, and reading
    that against cutoff shows *how much* high-frequency content the policy
    depends on rather than merely that it depends on some.

    *warn* ``False`` drops the warning for an entry that is not a number
    (:func:`declared_stance_probes`, which a report tool calls).
    """
    values = raw if isinstance(raw, (list, tuple)) else [raw]
    cutoffs: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            if warn:
                logger.warning("stance_probe_filter_hz entry is not a number: %r; ignoring it", value)
            continue
        if number > 0:
            cutoffs.append(number)
    return sorted(set(cutoffs))


def _probe_speeds(raw: Any, *, warn: bool = True) -> list[float]:
    """The positive shove speeds ``stance_probe_impulse_speeds`` names, in its order; empty when unset.

    :func:`_write_impulse_probe`'s switch, shared with :func:`declared_stance_probes`.
    """
    if not raw:
        return []
    values = raw if isinstance(raw, (list, tuple)) else [raw]
    speeds: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            if warn:
                logger.warning("stance_probe_impulse_speeds entry is not a number: %r; ignoring it", value)
            continue
        if number > 0:
            speeds.append(number)
    return speeds


def _probe_offsets(raw: Any, *, warn: bool = True) -> list[float]:
    """The finite, non-zero spawn turns ``stance_probe_spawn_yaw_deg`` names, in its order; empty when unset.

    :func:`_write_heading_probe`'s switch; ``stance_report.declares_heading_probe``
    (:func:`declared_stance_probes`) reads the same values.  *warn* ``False``
    drops the warning for an entry that is not a number (a report tool's call).
    """
    if not raw:
        return []
    values = raw if isinstance(raw, (list, tuple)) else [raw]
    offsets: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            if warn:
                logger.warning("stance_probe_spawn_yaw_deg entry is not a number: %r; ignoring it", value)
            continue
        if math.isfinite(number) and number != 0.0:
            offsets.append(number)
    return offsets


#: The labels :func:`_run_stance_probes` logs its probes under, in the order it rolls them.
STANCE_PROBE_LABELS = ("filtered action", "constant hold", "release ablation", "impulse", "heading")


def declared_stance_probes(curriculum: Mapping[str, Any]) -> tuple[str, ...]:
    """The probes :func:`_run_stance_probes` rolls on a stage whose ``[curriculum]`` block is *curriculum*.

    By their :data:`STANCE_PROBE_LABELS`, in that order: ``stance_probe_filter_hz``
    names a positive cutoff (:func:`_probe_cutoffs`), ``stance_probe_hold_constant``
    and ``stance_probe_release_ablation`` are set, ``stance_probe_impulse_speeds``
    names a positive speed (:func:`_probe_speeds`), and the heading probe is due
    (``stance_report.declares_heading_probe``: a finite non-zero
    ``stance_probe_spawn_yaw_deg``, never under the six-heading certificate, whose
    gate judged every heading).  Empty for a stage whose gate is no stance kind
    (``gate_schema.STANCE_GATE_KINDS``): it rolls no certification report, so
    nothing is probed.  :func:`_run_stance_probes` calls every writer and each
    writer's own switch decides (a value that is not a number is logged there);
    these are the probes those switches let through
    (``tests/test_run_index.py`` pins that they agree), rolled only from
    :func:`generate_stage_artifacts`, last, after the verdict, the summary, the
    figures and the replays: a runtime lost before the end of the stage leaves
    them out, and the command-line curriculum's judge
    (``train_base._post_training_stance_verdict``) writes the certification
    report and rolls no probe.  Reads *curriculum* only and logs nothing
    (``run_index`` names a declared probe a stage lacks).
    """
    from ..curriculum.gate_schema import STANCE_GATE_KINDS
    from .stance_report import declares_heading_probe

    if curriculum.get("gate_kind") not in STANCE_GATE_KINDS:
        return ()
    switches = {
        "filtered action": bool(_probe_cutoffs(curriculum.get("stance_probe_filter_hz"), warn=False)),
        "constant hold": bool(curriculum.get("stance_probe_hold_constant")),
        "release ablation": bool(curriculum.get("stance_probe_release_ablation")),
        "impulse": bool(_probe_speeds(curriculum.get("stance_probe_impulse_speeds"), warn=False)),
        "heading": declares_heading_probe(curriculum),
    }
    return tuple(label for label in STANCE_PROBE_LABELS if switches[label])


def _write_filtered_action_probe(
    *,
    model_id: ModelId,
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
        from environments.shared.reporting.stance_report import (
            build_stance_gate_report,
            write_action_filter_sweep,
        )

        for cutoff in cutoffs:
            probe = build_stance_gate_report(
                model_id,
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
        logger.warning("Filtered action probe failed for stage %s", stage, exc_info=True)
    # Written even if a later cutoff raised: a partial curve is still a curve,
    # and discarding the cutoffs that succeeded would lose the measurement to
    # a failure in one of them.
    if entries:
        try:
            from environments.shared.reporting.stance_report import write_action_filter_sweep

            written = write_action_filter_sweep(stage_dir, entries, probe_episodes=probe_episodes)
            logger.info("Filtered action probe sweep -> %s", written["action_filter_sweep_txt"])
        except Exception:  # noqa: BLE001 - a diagnostic must not sink the run
            logger.warning("Could not write the filtered action probe sweep", exc_info=True)


def _write_constant_hold_probe(
    *,
    model_id: ModelId,
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
        from environments.shared.reporting.stance_report import (
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
                model_id,
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
        logger.warning("Constant-hold probe failed for stage %s", stage, exc_info=True)
    # Written even if a later variant raised, for the same reason the filter
    # sweep is: the variants that succeeded are still a measurement, and the
    # controls are what make the others readable.
    if entries:
        try:
            from environments.shared.reporting.stance_report import write_constant_hold_probe

            written = write_constant_hold_probe(stage_dir, entries, probe_episodes=probe_episodes)
            logger.info("Constant-hold probe -> %s", written["constant_hold_probe_txt"])
        except Exception:  # noqa: BLE001 - a diagnostic must not sink the run
            logger.warning("Could not write the constant-hold probe", exc_info=True)


def _write_constant_hold_ablation(
    *,
    model_id: ModelId,
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
        from environments.shared.reporting.stance_report import (
            build_stance_gate_report,
            constant_hold_actions,
            constant_hold_release_variants,
        )

        hold = constant_hold_actions(measured)
        for variant in constant_hold_release_variants(hold, measured, horizon=int(measured["horizon"])):
            probe = build_stance_gate_report(
                model_id,
                stage,
                stage_config=stage_config,
                model_path=model_path,
                vecnorm_path=vecnorm_path,
                episodes=_ABLATION_EPISODES,
                hold_constant=variant,
            )
            entries.append(probe)
            logger.info(
                "Release ablation %s: episode length %.1f, full-horizon %.4f. MODIFIED policy -- not a gate verdict.",
                variant.label,
                probe["metrics"]["episode_length_mean"],
                probe["metrics"]["full_horizon_fraction"],
            )
    except Exception:  # noqa: BLE001 - a diagnostic must not sink the run
        logger.warning("Release ablation failed for stage %s", stage, exc_info=True)
    if entries:
        try:
            from environments.shared.reporting.stance_report import write_constant_hold_ablation

            written = write_constant_hold_ablation(stage_dir, entries, probe_episodes=_ABLATION_EPISODES)
            logger.info("Release ablation -> %s", written["constant_hold_ablation_txt"])
        except Exception:  # noqa: BLE001 - a diagnostic must not sink the run
            logger.warning("Could not write the release ablation", exc_info=True)


def _write_impulse_probe(
    *,
    model_id: ModelId,
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
    speeds = _probe_speeds(curriculum.get("stance_probe_impulse_speeds"))
    if not speeds:
        return
    try:
        from environments.shared.reporting.stance_report import (
            build_stance_gate_report,
            impulse_variants,
            write_impulse_probe,
        )

        variants = impulse_variants(speeds, step=settle_steps)

        def _sweep(zero_action: bool) -> list[dict[str, Any]]:
            return [
                build_stance_gate_report(
                    model_id,
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
        written = write_impulse_probe(stage_dir, policy_sweep, statue_sweep, probe_episodes=_IMPULSE_EPISODES)
        envelopes = {
            row["impulse"]["axis_label"]: row["metrics"]["full_horizon_fraction"]
            for row in policy_sweep
            if row["impulse"]["speed"] > 0
        }
        logger.info(
            "Impulse recovery probe -> %s (policy full-horizon by direction: %s). MODIFIED task -- not a gate verdict.",
            written["impulse_probe_txt"],
            envelopes,
        )
    except Exception:  # noqa: BLE001 - a diagnostic must not sink the run
        logger.warning("Impulse recovery probe failed for stage %s", stage, exc_info=True)


def _write_heading_probe(
    *,
    model_id: ModelId,
    stage: int,
    stage_config: dict[str, Any],
    stage_dir: Path,
    model_path: str,
    vecnorm_path: str | None,
) -> None:
    """Re-roll the policy and the statue from spawns turned about vertical; write stance_heading_probe.{txt,json}.

    Report only (decision D-D27).  The stance stage always spawns facing +x,
    so nothing in the v2 gate can see a stance that reads its world heading;
    both physics-r8 trex stances fell when spawned 90 degrees off, where the
    statue stands.  Each offset is rolled twice -- the animal turned alone,
    and with the episode's reference heading (only a world-frame observation
    would then differ: every plant declares the elliptic friction cone, and
    every model observes no world heading) -- plus the unturned control, for the policy and
    the zero-action statue.  A stage that declares the six-heading certificate
    (D-D32) never rolls it: :func:`_run_stance_probes` skips it there, and
    config load refuses the key beside that kind.

    Off unless ``stance_probe_spawn_yaw_deg`` is set.  Costs 2 x (1 + 2 x
    offsets) panels of ``_HEADING_EPISODES``.
    """
    curriculum = stage_config.get("curriculum_kwargs", {})
    offsets = _probe_offsets(curriculum.get("stance_probe_spawn_yaw_deg"))
    if not offsets:
        return
    try:
        from environments.shared.reporting.stance_report import (
            build_stance_gate_report,
            spawn_yaw_variants,
            stage_heading_probe_warnings,
            write_heading_probe,
        )

        variants = spawn_yaw_variants(offsets)

        def _sweep(zero_action: bool) -> list[dict[str, Any]]:
            return [
                build_stance_gate_report(
                    model_id,
                    stage,
                    stage_config=stage_config,
                    model_path=None if zero_action else model_path,
                    vecnorm_path=None if zero_action else vecnorm_path,
                    zero_action=zero_action,
                    episodes=_HEADING_EPISODES,
                    spawn_yaw=variant,
                )
                for variant in variants
            ]

        policy_sweep = _sweep(zero_action=False)
        statue_sweep = _sweep(zero_action=True)
        written = write_heading_probe(stage_dir, policy_sweep, statue_sweep, probe_episodes=_HEADING_EPISODES)
        logger.info(
            "Heading probe -> %s. MODIFIED start -- report only, not a gate verdict.",
            written["heading_probe_txt"],
        )
        # The maintainer's decision of 2026-10-11: warn, never block, on a whole-scene
        # row under v2's 0.95 rail. Read back from the file the handoff will read.
        for message in stage_heading_probe_warnings(stage_dir, declared=False):
            logger.warning("Stage %s heading probe: %s", stage, message)
    except Exception:  # noqa: BLE001 - a diagnostic must not sink the run
        logger.warning("Heading probe failed for stage %s", stage, exc_info=True)


#: What ``generate_stage_artifacts``' verdict records as ``judged_by``
#: (decision D-A5): the evidence-backed post-stage judgement.
GATE_VERDICT_JUDGED_BY = "reporting.stage_artifacts.generate_stage_artifacts"


def _write_stage_gate_verdict(
    *,
    stage_dir: Path,
    model_id: ModelId,
    stage: "int | str",
    curriculum: Mapping[str, Any],
    passed: bool,
    failures: list[str],
    stage_results: dict[str, Any],
) -> None:
    """Write the per-node ``gate_verdict.json`` beside the handoff pair it judged.

    The handoff resolved here is the same one ``_write_stance_gate_report``
    scored and the notebook's evidence CSVs bind (all through
    ``select_handoff_checkpoint``), so the verdict is hash-bound to the
    checkpoint the evidence describes.  ``stage_id`` is the manifest id when
    the model's stage manifest resolves *stage*, else the stage reference
    spelled out.  The model is required (decision D-D33): a verdict always
    names the model it judged.
    """
    from ..curriculum.gate_schema import gate_config_view
    from ..result_bundle import write_gate_verdict
    from ..stage_manifest import StageManifestError, load_stage_manifest
    from .gates import _current_task_sha256

    stage_id = str(stage)
    try:
        stage_id = load_stage_manifest(model_id).resolve(stage).id
    except StageManifestError:
        logger.warning("Stage %s is not in the %s manifest; recording the verdict by reference", stage, model_id)
    handoff = select_handoff_checkpoint(stage_dir / "models")
    # Recorded as declared: null for a stage that declares no gate kind.
    gate_kind: Any = curriculum.get("gate_kind")
    write_gate_verdict(
        stage_dir,
        model_id=model_id,
        stage=stage,
        stage_id=stage_id,
        gate_kind=gate_kind,
        gate_schema_version=curriculum.get("gate_schema_version"),
        passed=passed,
        failures=failures,
        task_sha256=_current_task_sha256(stage_dir),
        judged_by=GATE_VERDICT_JUDGED_BY,
        checkpoint=Path(handoff[1] + ".zip") if handoff is not None else None,
        normalization=Path(handoff[2]) if handoff is not None else None,
        # D-A22: the verdict records the gate it is judged under — the block
        # handed in, which is the CURRENT config (the notebook's JUDGE branch
        # judges a directory under the config of the session, not the one
        # its stage_config.json recorded; see _warn_if_judged_under_another_gate).
        gate_config=gate_config_view(curriculum),
        stage_result=stage_results,
    )


def _recorded_curriculum_block(stage_dir: Path) -> "Mapping[str, Any] | None":
    """The ``[curriculum]`` block *stage_dir*'s ``stage_config.json`` recorded, else None.

    Read under either spelling (``curriculum`` as ``save_stage_config``
    writes it, ``curriculum_kwargs`` as the loaded config carries it), the
    way the bundle's evidence reader and the backfill tool read it.  None
    for an absent, unreadable or block-less file.
    """
    from ..config import read_recorded_stage_config

    record = read_recorded_stage_config(stage_dir)
    if record is None:
        return None
    block = record.get("curriculum", record.get("curriculum_kwargs"))
    return block if isinstance(block, Mapping) else None


def _warn_if_judged_under_another_gate(stage: "int | str", stage_dir: Path, curriculum: Mapping[str, Any]) -> None:
    """Decision D-B8: say so when the gate judged under is not the one the stage trained under.

    Re-judging a directory under an edited gate is the intended path (edit
    a threshold, re-judge, never retrain), so this is a WARNING and never a
    refusal: the verdict records the gate it is judged under (D-A22) and
    reuse rule 7 compares against that, not against the recorded block.
    Names the thresholds that differ, so the log says what changed.
    """
    from ..curriculum.gate_schema import gate_config_differences, gate_config_sha256, gate_config_view

    recorded = _recorded_curriculum_block(stage_dir)
    if recorded is None:
        return
    recorded_view = gate_config_view(recorded)
    current_view = gate_config_view(curriculum)
    if gate_config_sha256(recorded_view) == gate_config_sha256(current_view):
        return
    differences = gate_config_differences(recorded_view["thresholds"], current_view)
    logger.warning(
        "Stage %s is judged under a gate configuration that differs from the block its stage_config.json "
        "recorded (%s): the verdict certifies the gate it is judged under (decision D-A22), not the one the "
        "stage trained under, and reuse rule 7 compares against this gate",
        stage,
        "; ".join(differences) if differences else "the gate kind or schema version differs",
    )


def stance_v2_stage_result(stance_report: "Mapping[str, Any] | None") -> dict[str, Any]:
    """The ``stage_result`` numbers a ``stance_quality/v2`` verdict was judged on, from its report.

    ``stance_clean_count``, ``stance_n_episodes``, ``stance_clean_fraction``
    (the fraction the bound bounds, which the catalog headlines beside it),
    ``stance_clean_lcb`` and ``stance_statue_mean_reward`` (``None`` when no
    statue panel was rolled), which ``gate_verdict._PERSISTED_STAGE_RESULT_KEYS``
    persists.
    Empty for no report or one not scored as v2.  Shared by the post-stage
    verdict here and the CLI curriculum's post-training v2 judge.
    """
    from ..curriculum.stance_gate_v2 import STANCE_GATE_V2_KIND

    if not isinstance(stance_report, Mapping) or stance_report.get("scored_gate_kind") != STANCE_GATE_V2_KIND:
        return {}
    result = stance_report.get("result")
    if not isinstance(result, Mapping):
        return {}
    return {
        "stance_clean_count": result.get("n_clean"),
        "stance_n_episodes": result.get("n_episodes"),
        "stance_clean_fraction": result.get("clean_fraction"),
        "stance_clean_lcb": result.get("clean_lcb"),
        "stance_statue_mean_reward": result.get("statue_mean_reward"),
    }


def stance_heading_stage_result(
    report: "Mapping[str, Any] | None", *, rederived: "StanceHeadingResult | None" = None
) -> dict[str, Any]:
    """The ``stage_result`` numbers a six-heading certificate (D-D32) was judged on, from its report.

    ``stance_heading_n_seeds``, ``stance_heading_clean_seeds`` (seeds clean
    at every heading), ``stance_heading_clean_seed_fraction``,
    ``stance_heading_clean_seed_lcb`` (the grouped bound) and
    ``stance_heading_statue_mean_reward_min`` (the lowest of the statue
    references' full-horizon mean rewards across the headings, ``None``
    when no statue panel was rolled), which
    ``gate_verdict._PERSISTED_STAGE_RESULT_KEYS`` persists.  Empty for no
    report or one not scored as the certificate.  With *rederived* (the
    judge's re-derivation of an admitted certificate) the seed counts,
    fraction and bound are its numbers, never the report's ``result``
    block: the judge cross-checks only the verdict and the clean counts
    there, so a bound edited in the report is not what a verdict records.
    """
    from ..curriculum.stance_heading_gate import STANCE_HEADING_GATE_KIND

    if not isinstance(report, Mapping) or report.get("scored_gate_kind") != STANCE_HEADING_GATE_KIND:
        return {}
    result: Mapping[str, Any] | None = report.get("result")
    if rederived is not None:
        result = {
            "n_seeds": rederived.n_seeds,
            "n_clean_seeds": rederived.n_clean_seeds,
            "clean_seed_fraction": rederived.clean_seed_fraction,
            "clean_seed_lcb": rederived.clean_seed_lcb,
        }
    if not isinstance(result, Mapping):
        return {}
    statue_rewards = [
        ((heading.get("statue") or {}).get("reference") or {}).get("mean_reward")
        for heading in report.get("headings") or ()
        if isinstance(heading, Mapping)
    ]
    finite = [float(value) for value in statue_rewards if isinstance(value, (int, float)) and math.isfinite(value)]
    return {
        "stance_heading_n_seeds": result.get("n_seeds"),
        "stance_heading_clean_seeds": result.get("n_clean_seeds"),
        "stance_heading_clean_seed_fraction": result.get("clean_seed_fraction"),
        "stance_heading_clean_seed_lcb": result.get("clean_seed_lcb"),
        "stance_heading_statue_mean_reward_min": min(finite) if finite and len(finite) == len(statue_rewards) else None,
    }


def admitted_stance_heading_stage_result(
    curriculum: Mapping[str, Any],
    report: "Mapping[str, Any] | None",
    *,
    stage: "int | str",
    stage_dir: "str | Path | None",
) -> dict[str, Any]:
    """:func:`stance_heading_stage_result` of a certificate the judge ADMITS, else ``{}``.

    Admitted means ``reporting.gates.stance_heading_report_refusals`` finds
    nothing, so a refused certificate's numbers never sit in a verdict's
    ``stage_result`` beside the refusal; the numbers copied are the
    re-derived ones.  Never raises.
    """
    try:
        from .gates import _admit_stance_heading_report

        refusals, rederived = _admit_stance_heading_report(curriculum, report, stage=stage, stage_dir=stage_dir)
        if refusals or rederived is None:
            return {}
    except Exception:  # noqa: BLE001 - a copy of the numbers must never cost the artifacts
        logger.warning("Stage %s heading certificate could not be re-checked for its numbers", stage, exc_info=True)
        return {}
    return stance_heading_stage_result(report, rederived=rederived)


def admitted_stance_v2_stage_result(
    curriculum: Mapping[str, Any],
    stance_report: "Mapping[str, Any] | None",
    *,
    stage: "int | str",
    stage_dir: "str | Path | None",
) -> dict[str, Any]:
    """:func:`stance_v2_stage_result` of a report the judge ADMITS, else ``{}``.

    Admitted means ``reporting.gates.stance_v2_report_refusals`` finds
    nothing: scored as v2, under this gate and measurement, for this
    stage's handoff pair.  So a refused report's numbers never sit in a
    verdict's ``stage_result`` beside the refusal as though they were the
    stage's.  Never raises: a re-check that fails copies nothing.
    """
    try:
        from .gates import stance_v2_report_refusals

        if stance_v2_report_refusals(curriculum, stance_report, stage=stage, stage_dir=stage_dir):
            return {}
    except Exception:  # noqa: BLE001 - a copy of the numbers must never cost the artifacts
        logger.warning("Stage %s stance report could not be re-checked for its numbers", stage, exc_info=True)
        return {}
    return stance_v2_stage_result(stance_report)


def admitted_command_tracking_stage_result(
    curriculum: Mapping[str, Any],
    report: "Mapping[str, Any] | None",
    *,
    stage: "int | str",
    stage_dir: "str | Path | None",
) -> dict[str, Any]:
    """``command_tracking_report.command_tracking_stage_result`` of a report the judge ADMITS, else ``{}``.

    The numbers copied are the re-derived ones
    (``reporting.gates._admit_command_tracking_report``), so a refused
    report's numbers never sit in a verdict's ``stage_result`` beside the
    refusal.  Never raises.
    """
    try:
        from .command_tracking_report import command_tracking_stage_result
        from .gates import _admit_command_tracking_report

        refusals, rederived = _admit_command_tracking_report(curriculum, report, stage=stage, stage_dir=stage_dir)
        if refusals or rederived is None:
            return {}
    except Exception:  # noqa: BLE001 - a copy of the numbers must never cost the artifacts
        logger.warning("Stage %s command-tracking report could not be re-checked for its numbers", stage, exc_info=True)
        return {}
    return command_tracking_stage_result(report, rederived=rederived)


def _apply_stage_gate(
    *,
    stage: int,
    stage_config: dict[str, Any],
    stage_results: dict[str, Any],
    stance_report: dict[str, Any] | None,
    stage_dir: "str | Path | None" = None,
    recovery_successes_by_seed: "dict[int, bool] | None" = None,
    model_id: ModelId,
    command_tracking_report: "dict[str, Any] | None" = None,
) -> None:
    """Record this stage's gate verdict onto *stage_results*, in place.

    Runs here, in the one entry point the notebook calls for every node it
    trains or judges,
    because the alternative is what actually happened:
    each caller kept a private checklist, the notebook's drifted out of step
    with ``gate_kind``, and run ``20260802_203215`` recorded
    ``publication_gate_passed = True`` beside a ``GATE: FAIL`` stance report
    and advanced to stage 2 on it.

    Sets ``gate_passed``, ``publication_gate_passed`` and ``gate_failures``;
    callers enforce them.  When *stage_dir* is given the same verdict is
    also written as the stage's ``gate_verdict.json`` (decision D-A5),
    hash-bound to the handoff pair, which is what lets another run reuse the
    node.  Never raises: a stage that cannot be certified is recorded as
    failing, which is the fail-closed reading, and an exception here — the
    verdict file included — would instead cost the run the artifacts written
    around it.
    """
    from .gates import evaluate_stage_gate

    try:
        passed, failures = evaluate_stage_gate(
            stage_config.get("curriculum_kwargs", {}),
            stage_results,
            stage=stage,
            stance_report=stance_report,
            # recovery_quality/v1 judges solely through the stage directory's
            # frozen gate_resolution.json plus the pushed panel's per-seed
            # successes; forwarding both is what makes the frozen gate
            # REACHABLE from this shared entry point (gap review EE2 — before
            # this, every recovery verdict was the fail-closed no-stage_dir
            # refusal, regardless of any frozen resolution on disk).
            stage_dir=stage_dir,
            recovery_successes_by_seed=recovery_successes_by_seed,
            # command_tracking/v1 (D-D61): its report and the stage directory
            # that binds it to the chosen handoff pair.
            command_tracking_report=command_tracking_report,
        )
    except Exception as exc:  # noqa: BLE001 - the docstring's promise, kept
        # "Never raises" has to be enforced, not asserted. `evaluate_stage_gate`
        # coerces config values (`float(min_avg_reward)`) and iterates a
        # report's `failures`, so a malformed TOML value or a hand-edited
        # report reaches here as TypeError/ValueError. Letting it propagate
        # would cost a completed multi-hour run the artifacts written around
        # this call; recording it as a failure keeps the fail-closed reading
        # AND the artifacts.
        logger.warning("Stage %s curriculum gate could not be evaluated", stage, exc_info=True)
        passed, failures = False, [f"stage {stage} gate evaluation raised {type(exc).__name__}: {exc}"]
    curriculum = stage_config.get("curriculum_kwargs", {})
    from ..curriculum.stance_gate_v2 import STANCE_GATE_V2_KIND

    if curriculum.get("gate_kind") == STANCE_GATE_V2_KIND:
        # The same for the floor-truth stance gate: the clean count, the
        # panel size, the bound and the statue's reference reward travel with
        # the verdict into gate_verdict.json's stage_result -- copied only
        # from a report the judge admits (scored as v2, under this gate and
        # measurement, for this handoff), so a refused report's numbers never
        # sit beside the refusal as though they were the stage's.
        stage_results.update(
            admitted_stance_v2_stage_result(curriculum, stance_report, stage=stage, stage_dir=stage_dir)
        )
    from ..curriculum.stance_heading_gate import STANCE_HEADING_GATE_KIND

    if curriculum.get("gate_kind") == STANCE_HEADING_GATE_KIND:
        # And for the six-heading certificate (D-D32): the grouped count and
        # bound, from a certificate the judge admits only.
        stage_results.update(
            admitted_stance_heading_stage_result(curriculum, stance_report, stage=stage, stage_dir=stage_dir)
        )
    from ..curriculum.command_tracking_gate import COMMAND_TRACKING_GATE_KIND

    if curriculum.get("gate_kind") == COMMAND_TRACKING_GATE_KIND:
        # And for command_tracking/v1 (D-D61): each family's counts and bounds,
        # re-derived from a report the judge admits only.
        stage_results.update(
            admitted_command_tracking_stage_result(
                curriculum, command_tracking_report, stage=stage, stage_dir=stage_dir
            )
        )
    # The verdict travels with the gate it was earned under, so a summary
    # re-served after the gate changes can say so instead of re-serving a
    # bare boolean beneath the current gate's description (review SS5).
    stage_results["gate_kind"] = curriculum.get("gate_kind")
    stage_results["gate_schema_version"] = curriculum.get("gate_schema_version")
    stage_results["gate_passed"] = passed
    stage_results["publication_gate_passed"] = passed
    stage_results["gate_failures"] = failures
    if passed:
        logger.info("Stage %s curriculum gate: PASS", stage)
    else:
        logger.warning("Stage %s curriculum gate: FAIL — %s", stage, "; ".join(failures))
    if stage_dir is not None:
        try:
            _warn_if_judged_under_another_gate(stage, Path(stage_dir), curriculum)
        except Exception:  # noqa: BLE001 - a diagnostic must never cost the artifacts
            logger.warning("Stage %s: could not compare the judged gate with the recorded one", stage, exc_info=True)
        try:
            _write_stage_gate_verdict(
                stage_dir=Path(stage_dir),
                model_id=model_id,
                stage=stage,
                curriculum=curriculum,
                passed=passed,
                failures=failures,
                stage_results=stage_results,
            )
        except Exception:  # noqa: BLE001 - the verdict file must never cost the artifacts
            logger.warning("Stage %s gate verdict could not be written to %s", stage, stage_dir, exc_info=True)


def evaluate_stage_checkpoints(
    species_cfg,
    stage_config,
    stage,
    algorithm,
    stage_dir,
    *,
    final_path,
    final_vecnorm_path,
    timesteps,
    duration_seconds,
    plant_identity,
    evaluation_seed,
    model=None,
):
    """Evaluate a trained node's final and selected checkpoints and build its stage_results.

    The notebook's evaluation of a node, after ``train_stage`` trains it or
    when the chain loop's JUDGE branch judges a node whose budget the
    RESUME cell spent but whose gate was never judged (no
    ``gate_verdict.json``). ``model`` is the in-memory final model right
    after training; ``None`` loads ``<final_path>.zip`` back, validates its
    plant and takes its cumulative step counter as ``timesteps``. Each
    checkpoint is rolled for ``EVALUATION_PANEL_EPISODES`` episodes on
    ``evaluation_seed`` under the VecNormalize sidecar it is paired with, and
    each panel is written as evidence bound to that checkpoint and sidecar;
    the handoff is the shared selector's (:func:`select_handoff_checkpoint`).
    A success rate (``mean_success_rate``, ``best_model_success_rate``) is
    recorded only for a model that declares success keys
    (``species_cfg.success_keys``): without them the panels record no success
    flag, and the rate is left out rather than written as a zero. Prints the report the
    notebook shows. Moved verbatim from the notebook's infrastructure cell
    with its globals as parameters, and unannotated like it (its ``""``
    placeholders become floats), so mypy leaves its body alone.

    Returns (model, handoff_stem, final_model_path, handoff_vecnorm_path, stage_results).
    """
    import numpy as np

    from ..curriculum import load_vecnorm_stats
    from ..evaluation import eval_policy
    from ..plant_contract import validate_model_plant
    from ..train_base import create_vec_env

    algorithm = algorithm.lower()

    def _eval_forward_vel(model, vecnorm_path, n_episodes):
        """Roll ``model`` on a fresh evaluation env under ``vecnorm_path``'s statistics."""
        eval_env = create_vec_env(
            species_cfg,
            {stage: stage_config},
            stage,
            1,
            evaluation_seed,
            algorithm=algorithm,
            gamma=stage_config.get(f"{algorithm}_kwargs", {}).get("gamma"),
            plant_identity=plant_identity,
        )
        if vecnorm_path and Path(vecnorm_path).exists():
            load_vecnorm_stats(
                vecnorm_path,
                eval_env,
                current_plant=plant_identity,
                command_mode=str(stage_config.get("env_kwargs", {}).get("command_mode", "none")),
            )
        eval_env.training = False
        eval_env.norm_reward = False
        try:
            return eval_policy(model, eval_env, species_cfg.success_keys, n_episodes=n_episodes)
        finally:
            eval_env.close()

    stage_dir = Path(stage_dir)
    model_dir = stage_dir / "models"
    final_path = Path(final_path)
    final_vecnorm_path = str(final_vecnorm_path)
    vecnorm_save_path = final_vecnorm_path
    if model is None:
        model = load_sb3_model(str(final_path), algorithm=algorithm)
        validate_model_plant(model, plant_identity, artifact=f"{final_path}.zip")
        timesteps = int(getattr(model, "num_timesteps", 0)) or timesteps

    # Evaluate the final model with its matching VecNormalize stats.
    episode_rewards, episode_lengths, episode_fwd_vels, episode_successes, episode_distances = _eval_forward_vel(
        model,
        final_vecnorm_path,
        n_episodes=EVALUATION_PANEL_EPISODES,
    )
    final_eval_path = csv_output.save_evaluation_episodes(
        stage_dir,
        rewards=episode_rewards,
        lengths=episode_lengths,
        forward_velocities=episode_fwd_vels,
        distances=episode_distances,
        successes=episode_successes,
        evaluation_seed=evaluation_seed,
        checkpoint_label="final",
        checkpoint_path=f"{final_path}.zip",
        normalization_path=final_vecnorm_path,
    )
    print(f"Final-model episode evidence saved to: {final_eval_path}")
    mean_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))
    mean_length = float(np.mean(episode_lengths))
    std_length = float(np.std(episode_lengths))
    mean_fwd_vel = float(np.mean(episode_fwd_vels))
    std_fwd_vel = float(np.std(episode_fwd_vels))
    mean_distance = float(np.mean(episode_distances))
    # The training envs are closed by now: a bare env of this node's task
    # answers for its control timestep.
    _probe_env = species_cfg.env_class(**stage_config.get("env_kwargs", {}))
    sim_dt = float(_probe_env.dt)
    _probe_env.close()
    print(f"Eval (final model): mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Eval: mean_length={mean_length:.1f} +/- {std_length:.1f} steps ({mean_length * sim_dt:.2f}s sim time)")
    print(f"Eval: mean_forward_vel={mean_fwd_vel:.2f} +/- {std_fwd_vel:.2f} m/s")
    print(f"Eval: mean_distance_traveled={mean_distance:.2f} m")
    # Only a model with success keys records success flags (eval_policy).
    live_success: dict[str, float] = {}
    if episode_successes:
        live_success["mean_success_rate"] = float(np.mean(episode_successes))
        print(f"Eval: success_rate={live_success['mean_success_rate']:.0%}")

    # Build base results dict from on-disk eval data (evaluations.npz),
    # then enrich with the live 30-episode evaluation metrics above.
    stage_results = build_stage_results_from_eval_data(
        stage_dir,
        stage,
        stage_config,
        timesteps=timesteps,
        duration_seconds=duration_seconds,
        sim_dt=sim_dt,
    )
    best_eval_reward = stage_results["best_eval_reward"]
    best_eval_std = stage_results["best_eval_std"]
    best_eval_timestep = stage_results["best_eval_timestep"]
    if best_eval_reward != "":
        print(f"Best model eval:  mean_reward={best_eval_reward} +/- {best_eval_std} (at {best_eval_timestep:,} steps)")

    # Override with richer live-eval metrics
    stage_results.update(
        {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "mean_episode_length": mean_length,
            "std_episode_length": std_length,
            "mean_forward_vel": mean_fwd_vel,
            "std_forward_vel": std_fwd_vel,
            "mean_distance_traveled": mean_distance,
            **live_success,
            "sim_dt": sim_dt,
        }
    )

    # The SELECTED checkpoint: next-stage loading, the evidence CSV below, the
    # replay video, and the stance gate report must all describe the same
    # policy. `select_handoff_checkpoint` is the one selector they share --
    # the notebook used to carry a private copy of the preference order, which
    # is how the replay ended up showing `best_model` while
    # `evaluation_selected.csv` was evidence for `robust_best_model`.
    #
    # It prefers the risk-adjusted robust_best_model (highest mean - std eval)
    # over SB3's mean-reward best_model: a high mean can be propped up by a good
    # mode while a fat failure tail is already growing (run 20260709_185946).
    # It requires the matched VecNormalize stats, so it cannot return a hybrid.
    #
    # A command_tracking/v1 stage (D-D61) hands on the pair its gated
    # statistic chooses: made here when the stage has no valid choice under
    # its current gate (the JUDGE branch of a node trained before, or judged
    # under an edited gate), before anything below reads the handoff.
    ensure_command_handoff_choice(
        model_id=species_cfg.model_id,
        stage=stage,
        stage_config=stage_config,
        model_dir=model_dir,
        final_stem=final_path,
    )
    _handoff = select_handoff_checkpoint(model_dir)
    if _handoff is None:
        best_model_zip = model_dir / "best_model.zip"
        best_vecnorm_candidate = str(model_dir / "best_model_vecnorm.pkl")
        if best_model_zip.exists():
            raise FileNotFoundError(
                f"Selected checkpoint {best_model_zip} is missing its matched VecNormalize state "
                f"{best_vecnorm_candidate}; refusing to evaluate or export a hybrid checkpoint."
            )
    else:
        _selected_name, _selected_path, best_vecnorm_candidate = _handoff
        best_model_zip = model_dir / f"{_selected_name}.zip"
        print(f"Selected checkpoint: {_selected_name}")
    best_model_reward, best_model_std_reward = "", ""
    best_model_length, best_model_std_length = "", ""
    best_model_fwd_vel, best_model_std_fwd_vel = "", ""
    best_model_distance = ""
    best_model_success_rate = ""
    if best_model_zip.exists():
        best_path = model_dir / best_model_zip.stem
        model = load_sb3_model(str(best_path), algorithm=algorithm)
        validate_model_plant(model, plant_identity, artifact=str(best_model_zip))
        output_path = str(best_path)
        print(f"Loaded best model for next-stage: {best_path}.zip")

        # Guaranteed by select_handoff_checkpoint, which only returns a
        # candidate whose matched statistics exist. Kept as an assertion
        # because exporting a hybrid checkpoint is silent and unrecoverable.
        if not Path(best_vecnorm_candidate).exists():
            raise FileNotFoundError(
                f"Selected checkpoint {best_model_zip} is missing its matched VecNormalize state "
                f"{best_vecnorm_candidate}; refusing to evaluate or export a hybrid checkpoint."
            )
        vecnorm_save_path = best_vecnorm_candidate
        print(f"Using matched VecNormalize for selected checkpoint: {vecnorm_save_path}")

        print(f"Evaluating best model ({EVALUATION_PANEL_EPISODES} episodes)...")
        bm_rewards, bm_lengths, bm_fwd_vels, bm_successes, bm_distances = _eval_forward_vel(
            model,
            vecnorm_save_path,
            n_episodes=EVALUATION_PANEL_EPISODES,
        )
        selected_eval_path = csv_output.save_evaluation_episodes(
            stage_dir,
            rewards=bm_rewards,
            lengths=bm_lengths,
            forward_velocities=bm_fwd_vels,
            distances=bm_distances,
            successes=bm_successes,
            evaluation_seed=evaluation_seed,
            checkpoint_label="selected",
            checkpoint_path=best_model_zip,
            normalization_path=vecnorm_save_path,
        )
        print(f"Selected-model episode evidence saved to: {selected_eval_path}")
        best_model_reward = round(float(np.mean(bm_rewards)), 2)
        best_model_std_reward = round(float(np.std(bm_rewards)), 2)
        best_model_length = round(float(np.mean(bm_lengths)), 1)
        best_model_std_length = round(float(np.std(bm_lengths)), 1)
        best_model_fwd_vel = round(float(np.mean(bm_fwd_vels)), 2)
        best_model_std_fwd_vel = round(float(np.std(bm_fwd_vels)), 2)
        best_model_distance = round(float(np.mean(bm_distances)), 2)
        print(f"Best model eval:  mean_reward={best_model_reward} +/- {best_model_std_reward}")
        print(f"Best model eval:  mean_length={best_model_length} +/- {best_model_std_length}")
        print(f"Best model eval:  mean_fwd_vel={best_model_fwd_vel} +/- {best_model_std_fwd_vel} m/s")
        print(f"Best model eval:  mean_distance={best_model_distance} m")
        if bm_successes:
            best_model_success_rate = round(float(np.mean(bm_successes)), 2)
            print(f"Best model eval:  success_rate={best_model_success_rate:.0%}")
    else:
        output_path = str(final_path)
        selected_eval_path = csv_output.save_evaluation_episodes(
            stage_dir,
            rewards=episode_rewards,
            lengths=episode_lengths,
            forward_velocities=episode_fwd_vels,
            distances=episode_distances,
            successes=episode_successes,
            evaluation_seed=evaluation_seed,
            checkpoint_label="selected",
            checkpoint_path=f"{final_path}.zip",
            normalization_path=final_vecnorm_path,
        )
        print(f"Selected-model episode evidence saved to: {selected_eval_path}")
        best_model_reward = round(float(np.mean(episode_rewards)), 2)
        best_model_std_reward = round(float(np.std(episode_rewards)), 2)
        best_model_length = round(float(np.mean(episode_lengths)), 1)
        best_model_std_length = round(float(np.std(episode_lengths)), 1)
        best_model_fwd_vel = round(float(np.mean(episode_fwd_vels)), 3)
        best_model_std_fwd_vel = round(float(np.std(episode_fwd_vels)), 3)
        best_model_distance = round(float(np.mean(episode_distances)), 3)
        if episode_successes:
            best_model_success_rate = round(float(np.mean(episode_successes)), 4)

    # The curriculum gate is NOT evaluated here: `generate_stage_artifacts`
    # judges it through the one shared `reporting.gates.evaluate_stage_gate`
    # and records `gate_passed` / `publication_gate_passed` / `gate_failures`
    # onto this dict; the chain loop enforces it. The notebook's own checklist
    # knew nothing about `gate_kind`, so when Tyrannosaurus rex stage 1 moved
    # to stance_quality/v1 it quietly degraded to a reward comparison, which
    # the zero-action statue clears by 68% (run 20260802_203215 advanced to
    # stage 2 beside a stance_gate_report.txt reading GATE: FAIL).

    # Update stage_results with model paths and best-model eval
    stage_results.update(
        {
            "model_path": output_path,
            "final_model_path": str(final_path),
            "vecnorm_path": vecnorm_save_path,
            "final_vecnorm_path": final_vecnorm_path,
            "best_model_reward": best_model_reward,
            "best_model_std_reward": best_model_std_reward,
            "best_model_length": best_model_length,
            "best_model_std_length": best_model_std_length,
            "best_model_fwd_vel": best_model_fwd_vel,
            "best_model_std_fwd_vel": best_model_std_fwd_vel,
            "best_model_distance": best_model_distance,
            "best_model_success_rate": best_model_success_rate,
        }
    )

    return model, output_path, str(final_path), vecnorm_save_path, stage_results


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
    recovery_successes_by_seed: "dict[int, bool] | None" = None,
) -> dict[str, Any]:
    """Write stage summary, record replay videos, and generate training graphs.

    This is the single shared entry-point for generating post-training
    artifacts.  The training notebook calls it for every node it trains or
    judges, so the artifacts are always consistent.

    When *stage_results* is ``None``, a results dict is built from on-disk
    eval data via :func:`build_stage_results_from_eval_data`.  Callers
    that already have richer metrics (e.g. the notebook, which runs a
    full 30-episode eval) should pass their own *stage_results*.

    When *generate_graphs* is ``True`` (the default), training curves and
    diagnostic graphs are saved to the stage directory.  Requires
    ``matplotlib``.

    For a ``recovery_quality/v1`` stage, pass *recovery_successes_by_seed*
    (``RecoveryPanelEvidence.successes_by_seed()`` from the post-training
    pushed panel); the gate is judged against the stage directory's frozen
    ``gate_resolution.json`` and refuses without both.

    A ``command_tracking/v1`` stage (D-D61) rolls its command panel here
    (:func:`_write_command_tracking_report`) on the handoff pair chosen by
    the gated statistic (``models/handoff_choice.json``, made at the end of
    training or by :func:`evaluate_stage_checkpoints`), and is judged from it.

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
    model_id = species_cfg.model_id

    if stage_results is None:
        # The summaries print episode length × sim_dt as sim time, so take the
        # node's own control step from a bare env of its task (the compsognathus
        # pair steps at 0.02 s, not the 0.01 s default).
        probe_env = species_cfg.env_class(**stage_config.get("env_kwargs", {}))
        try:
            sim_dt = float(probe_env.dt)
        finally:
            probe_env.close()
        stage_results = build_stage_results_from_eval_data(
            stage_dir,
            stage,
            stage_config,
            timesteps=timesteps,
            sim_dt=sim_dt,
        )

    stance_report = _write_stance_gate_report(
        model_id=model_id,
        stage=stage,
        stage_config=stage_config,
        stage_dir=stage_dir,
        model_dir=model_dir,
    )
    # command_tracking/v1 (D-D61): the panel on the chosen handoff pair; None for every other kind.
    command_report = _write_command_tracking_report(
        model_id=model_id,
        stage=stage,
        stage_config=stage_config,
        stage_dir=stage_dir,
        model_dir=model_dir,
    )
    # Before the replays and graphs below, which are best-effort and can be
    # skipped: the verdict must not depend on whether matplotlib imported.
    # This also writes the stage's gate_verdict.json (the model names the id).
    _apply_stage_gate(
        stage=stage,
        stage_config=stage_config,
        stage_results=stage_results,
        stance_report=stance_report,
        stage_dir=stage_dir,
        recovery_successes_by_seed=recovery_successes_by_seed,
        model_id=model_id,
        command_tracking_report=command_report,
    )

    # After the gate, not before it: the summary now states the verdict and
    # every criterion that failed, and writing it first would print a stage
    # summary that says nothing about the decision the stage turns on.
    text_summaries.write_stage_summary(
        stage_dir, stage_results, model_id, algorithm, evaluation_episodes=EVALUATION_PANEL_EPISODES
    )
    logger.info("Stage summary written to: %s", stage_dir / "stage_summary.txt")

    # A command_tracking/v1 stage's trace and figures from an earlier judgement
    # go first, before the staged publish and whether or not graphs are drawn:
    # publishing only copies what was staged, so it never removes a stale one.
    remove_stale_command_tracking_outputs(stage=stage, stage_config=stage_config, stage_dir=stage_dir)

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
                stage_configs: dict[int | str, dict[str, Any]] = {stage: stage_config}

                plot_training_curves(
                    stage_dirs,
                    stage_configs,
                    model_id,
                    algorithm,
                    save_path=figures_out / "training_curves.png",
                    show=False,
                )
                plot_diagnostics_graphs(
                    stage_dirs,
                    stage_configs,
                    model_id,
                    algorithm,
                    save_dir=figures_out,
                    show=False,
                )
                plot_foot_contacts(
                    stage_dirs,
                    stage_configs,
                    model_id,
                    algorithm,
                    save_path=figures_out / "foot_contacts.png",
                    show=False,
                )
                plot_stance_diagnostics(
                    stage_dirs,
                    stage_configs,
                    model_id,
                    algorithm,
                    save_path=figures_out / "stance_diagnostics.png",
                    show=False,
                )
            except ImportError:
                logger.warning("Skipping graph generation (matplotlib not installed).")
            except Exception:
                logger.warning("Graph generation failed.", exc_info=True)
            # In its own guard, after the four above: a failure costs gate_progress.png only.
            try:
                from environments.shared.visualization import plot_gate_progress

                plot_gate_progress(
                    stage_dir,
                    curriculum_kwargs=stage_config.get("curriculum_kwargs", {}),
                    model_id=model_id,
                    save_path=figures_out / "gate_progress.png",
                    show=False,
                    live_command=str(stage_config.get("env_kwargs", {}).get("command_mode", "none")) != "none",
                )
            except ImportError:
                logger.warning("Skipping the gate progress figure (matplotlib not installed).")
            except Exception:
                logger.warning("The gate progress figure failed; the other figures are unaffected.", exc_info=True)
            # A command_tracking/v1 stage's (D-D61) trace and figures, after the gate progress figure; a no-op for
            # every other kind, and never raises: a failure costs its own files only.
            write_command_tracking_figures(
                model_id=model_id,
                stage=stage,
                stage_config=stage_config,
                stage_dir=stage_dir,
                figures_out=figures_out,
                report=command_report,
                verdict=stage_results.get("gate_passed")
                if isinstance(stage_results.get("gate_passed"), bool)
                else None,
            )
            # In its own guard, after the gate progress figure: a failure costs the training history only.
            try:
                _write_training_history(stage_dir, staging)
            except ImportError:
                logger.warning("Skipping the training history (matplotlib not installed).")
            except Exception:
                logger.warning("The training history failed; the other figures are unaffected.", exc_info=True)

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
    # nothing downstream gates on them (only the report-only heading-probe
    # handoff warning reads one, and warns that heading robustness is
    # unmeasured when a declared probe is missing; a six-heading certificate
    # stage rolls no heading probe, its gate above judged every heading), so a runtime lost
    # mid-probe keeps the recorded verdict, summary, graphs and replays above.
    # It also costs what the caller writes after this returns: the notebook's
    # training summary and result bundle. A re-entered run takes the node's
    # result from its verdict, and the next node's bundle save, or the publish
    # cell (PUBLISH_RUN) when no node follows, writes them. Ordering is pinned
    # by the wiring tests.
    _run_stance_probes(
        model_id=model_id,
        stage=stage,
        stage_config=stage_config,
        stage_dir=stage_dir,
        model_dir=model_dir,
        report=stance_report,
    )

    return stage_results


def _write_training_history(stage_dir: Path, staging: Path) -> Path | None:
    """Draw the stage's training history from its ledgers into *staging*'s ``figures/training_history/``.

    The pages ``scripts/plot_training_history.py STAGE_DIR`` draws with its
    defaults (:mod:`.training_history`), written by the library itself: that
    script refuses an ``--out`` inside a run directory to keep manual writes
    out of a run, while this is the stage's own artifact step, published with
    the other figures before the bundle is saved, so the save declares them.
    The report is written to a private temporary directory and moved into the
    staging tree in one rename only once every file of it is complete, so a
    failure or an interrupt part-way publishes none of it.  A stage directory
    holding neither ledger (a run from before them) gets none: every page
    would be empty.  Returns the staged directory, or ``None`` when skipped.
    Report only: nothing reads the files back.
    """
    from . import training_history

    ledgers = (training_history.EVAL_EPISODES_FILENAME, training_history.TRAIN_EPISODES_FILENAME)
    if not any((stage_dir / name).is_file() for name in ledgers):
        logger.info("No training history drawn: %s holds neither %s nor %s.", stage_dir, *ledgers)
        return None
    private = Path(tempfile.mkdtemp(prefix="mesozoic-training-history-"))
    try:
        run = training_history.load_run_history(stage_dir)
        title = f"{run.model_display or 'training history'}: {run.label}"
        report = private / stage_layout.TRAINING_HISTORY_DIRNAME
        # A page that fails alone is logged by the module and listed in the JSON's errors; the rest are kept.
        training_history.write_report([run], report, title=title)
        destination = stage_layout.training_history_dir(staging)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(destination, ignore_errors=True)
        shutil.move(str(report), str(destination))
    finally:
        shutil.rmtree(private, ignore_errors=True)
    logger.info("Training history drawn: %s", stage_layout.training_history_dir(stage_dir))
    return destination


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
    model_id = species_cfg.model_id

    # ── Record replay videos for the selected and final checkpoints ──────
    from ..plant_contract import PlantCompatibilityError, current_plant_identity, validate_model_plant
    from ..stage_manifest import stage_label

    try:
        from environments.shared.evaluation import record_stage_video, stance_camera_views
        from environments.shared.policy_loading import _ensure_sb3

        sb3 = _ensure_sb3()
        env_kwargs = stage_config["env_kwargs"].copy()
        alg_cls = sb3["SAC"] if algorithm == "sac" else sb3["PPO"]
        plant_identity = current_plant_identity(model_id)

        final_path = model_dir / f"{stage_label(stage)}_final"
        final_vecnorm_path = str(final_path) + "_vecnorm.pkl"
        # Every model's stance and recovery replays share the same views and
        # per-frame CSV. Framing follows each model's camera size; the old
        # T. rex distance stays compatible with its existing comparisons.
        replay_diagnostics = stage in (1, "stance", "recovery")
        replay_camera_views = stance_camera_views(model_id) if replay_diagnostics else None

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
        handoff = select_handoff_checkpoint(model_dir)
        if handoff is None:
            logger.warning(
                "Stage %s selected-checkpoint replay skipped: neither robust_best_model nor "
                "best_model in %s has its matched _vecnorm.pkl, and replaying without the "
                "observation statistics would show a different policy.",
                stage,
                model_dir,
            )
        else:
            selected_name, selected_path, selected_vecnorm = handoff
            logger.info("Stage %s selected checkpoint for replay: %s", stage, selected_name)
            selected_model = load_sb3_model(selected_path, algorithm=alg_cls)
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
                model_id=model_id,
                algorithm=algorithm,
                seed=seed,
                vecnorm_path=selected_vecnorm,
                label="selected",
                plant_identity=plant_identity,
                allow_legacy_plant=allow_legacy_plant,
                camera_views=replay_camera_views,
                collect_stance_diagnostics=replay_diagnostics,
            )

        if Path(str(final_path) + ".zip").exists() and not Path(final_vecnorm_path).is_file():
            logger.warning(
                "Stage %s final-checkpoint replay skipped: %s has no matched _vecnorm.pkl; "
                "replaying without the observation statistics would show a different policy.",
                stage,
                final_path.name,
            )
        elif Path(str(final_path) + ".zip").exists():
            final_model = load_sb3_model(str(final_path), algorithm=alg_cls)
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
                model_id=model_id,
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
