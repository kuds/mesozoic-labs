"""The ``command_tracking/v1`` post-stage report, and the handoff chosen by the gated statistic.

The report
----------
:func:`build_command_tracking_report` rolls a stage's command panel
(:func:`~environments.shared.command_tracking_panel.roll_command_panel`, on
the panel seeds from
:data:`~environments.shared.constants.PUBLICATION_SEED_START`) on a saved
checkpoint under its frozen VecNormalize statistics, deterministically,
judges it with
:func:`~environments.shared.curriculum.command_tracking_gate.evaluate_command_tracking_gate`
and records everything the post-stage judge (``reporting.gates``) and
publication (``result_bundle.evidence``) bind and re-derive: the identity
(``schema``, ``scored_gate_kind``, ``profile``, ``protocol``), the plan, the
rollout (``policy_deterministic``, ``normalization``, ``horizon``,
``control_dt``), the measurement, the verdict (``passed``, ``failures``,
``result``), a descriptive summary, the binding (``thresholds`` as the gate
view projects them, ``gate_sha256``, the ``handoff`` digests,
``task_sha256``, the plant identity, ``checkpoint_plant_validated``) and
every episode's record with its classification.  The follow profile's
reference panel, when one was rolled, is recorded beside it on the same
seeds.  :func:`write_command_tracking_report` writes the text, the JSON and
the panel CSV (``curriculum.command_tracking_evidence``), in that order.

The handoff choice
------------------
Training saves up to three checkpoints a stage could hand on: the
risk-adjusted ``robust_best_model``, SB3's mean-reward ``best_model`` and the
final pair.  The stance and recovery stages hand on
``select_handoff_checkpoint``'s pair, chosen by evaluation reward, and their
gate then certifies or refuses that one pair.  Under a live command the
reward is a poor proxy for the gate: it mixes the tracking terms with
every other term, and a checkpoint can earn more while settling fewer
commands.  So a ``command_tracking/v1`` stage chooses its handoff by the
gated statistic (:func:`choose_command_handoff`): each complete candidate is
rolled on a SELECTION panel planned from :data:`HANDOFF_SELECTION_SEED_START`,
a seed block disjoint from the certification panel, and ranked by
:func:`~environments.shared.curriculum.command_tracking_gate.handoff_selection_key`
(passed first, then the profile's certifying bound).  The choice is written
to ``models/handoff_choice.json``, which ``select_handoff_checkpoint``
honors, so every reader of the handoff follows it; the chosen pair is then
certified on the certification panel, whose draws played no part in the
choice.  Choosing and certifying on one panel would certify the best of
three draws on the same seeds.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..constants import PUBLICATION_SEED_START
from ..curriculum.checkpoints import REWARD_HANDOFF_NAMES
from ..curriculum.command_tracking_evidence import (
    COMMAND_HANDOFF_SELECTED_KEY,
    COMMAND_TRACKING_PANEL_CSV,
    COMMAND_TRACKING_PANEL_FIELDNAMES,
    COMMAND_TRACKING_REPORT_JSON,
    COMMAND_TRACKING_REPORT_SCHEMA,
    COMMAND_TRACKING_REPORT_TXT,
    COMMAND_TRACKING_STAGE_RESULT_KEYS,
    panel_rows,
)
from ..curriculum.command_tracking_gate import (
    COMMAND_PANEL_PROTOCOL,
    COMMAND_TRACKING_GATE_KIND,
    CRUISE_FAMILY,
    FOLLOW_PROFILE,
    HANDOFF_SELECTION_RULES,
    STAND_FAMILY,
    WALK_PROFILE,
    CommandTrackingEpisode,
    CommandTrackingResult,
    command_config_from,
    evaluate_command_tracking_gate,
    handoff_selection_key,
    summarize_command_panel,
    thresholds_from_curriculum,
)
from ..curriculum.gate_schema import gate_config_sha256, gate_config_view
from ..file_io import atomic_write_csv, atomic_write_json, atomic_write_text
from ..model_id import ModelId

logger = logging.getLogger(__name__)

#: The first seed of the handoff choice's selection panels: the selection block (the recovery calibration's
#: ``SELECTION_SEED``), disjoint from training (``seed + rank``, ``seed + 1000``), the certification panel
#: (3042-) and the fresh audit (7042-), so the draws that choose the handoff never certify it.
HANDOFF_SELECTION_SEED_START = 2042
#: The reward-chosen handoff names a choice considers, in ``select_handoff_checkpoint``'s order; the stage's
#: final pair follows them.
HANDOFF_CANDIDATE_NAMES: tuple[str, ...] = REWARD_HANDOFF_NAMES


class CommandTrackingReportError(RuntimeError):
    """The panel cannot be rolled or judged for a reason the caller should see (an ``Exception``, never a SystemExit)."""


def require_current_protocol(report: Mapping[str, Any]) -> None:
    """Refuse a report rolled under another measurement protocol than :data:`COMMAND_PANEL_PROTOCOL`, naming both.

    Checked before any of its records is parsed: a ``command_panel/v1``
    report's in-tolerance flags were judged per step with no window-fill
    allowance (decision D-D64), so its records answer another question, and
    each record's own measurement would be refused one by one anyway.
    Raises :class:`CommandTrackingReportError`.
    """
    protocol = report.get("protocol") if isinstance(report, Mapping) else None
    if protocol != COMMAND_PANEL_PROTOCOL:
        raise CommandTrackingReportError(
            f"the command-tracking report was rolled under protocol {protocol!r}, not {COMMAND_PANEL_PROTOCOL!r}; "
            "re-roll its panel under the current protocol"
        )


def _policy(
    model_id: ModelId,
    env_kwargs: Mapping[str, Any],
    *,
    model_path: str | None,
    vecnorm_path: str | None,
    zero_action: bool,
    plant_identity: Any,
    allow_legacy_plant: bool,
) -> tuple[Any, str, dict[str, Any]]:
    """``(predict, description, normalization)`` for a saved checkpoint, or the zero-action statue."""
    from ..command_tracking_panel import zero_action_predict
    from ..species_registry import env_class as model_env_class
    from .stance_report import StanceGateReportError, _load_policy

    env_class = model_env_class(model_id)
    if zero_action:
        probe = env_class(**dict(env_kwargs))
        try:
            action_dim = int(probe.action_space.shape[0])
        finally:
            probe.close()
        return zero_action_predict(action_dim), "zero action (the statue)", {"mode": "none (zero action)"}
    if model_path is None:
        raise ValueError("model_path is required unless zero_action is set")
    try:
        predict, note = _load_policy(
            model_path,
            vecnorm_path,
            lambda: env_class(**dict(env_kwargs)),
            plant_identity=plant_identity,
            allow_legacy_plant=allow_legacy_plant,
        )
    except StanceGateReportError as exc:
        raise CommandTrackingReportError(str(exc)) from exc
    record = getattr(predict, "normalization", None)
    normalization = dict(record) if isinstance(record, Mapping) else {"mode": "unrecorded"}
    return predict, f"{Path(model_path).name} — {note}", normalization


def _pair_record(model_path: str | None, vecnorm_path: str | None) -> dict[str, Any]:
    from ..result_bundle.hashing import sha256_file

    return {
        "checkpoint": None if model_path is None else Path(model_path).name,
        "checkpoint_sha256": None if model_path is None else sha256_file(model_path),
        "normalization_sha256": None if vecnorm_path is None else sha256_file(vecnorm_path),
    }


def _evidence(episodes: Any, result: CommandTrackingResult | None) -> list[dict[str, Any]]:
    """Each episode's record, with its classification when *result* judged it (``None`` for a reference)."""
    if result is None:
        return [{"record": episode.as_dict()} for episode in episodes]
    return [
        {"record": episode.as_dict(), "success": not reasons, "reasons": list(reasons)}
        for episode, reasons in zip(episodes, result.episode_reasons)
    ]


def build_command_tracking_report(
    model_id: ModelId,
    stage: "int | str",
    *,
    stage_config: Mapping[str, Any],
    model_path: str | None = None,
    vecnorm_path: str | None = None,
    zero_action: bool = False,
    reference_model_path: str | None = None,
    reference_vecnorm_path: str | None = None,
    seed_start: int = PUBLICATION_SEED_START,
    allow_legacy_plant: bool = False,
) -> dict[str, Any]:
    """Roll and judge a ``command_tracking/v1`` panel on a checkpoint (or the statue), as a report.

    *stage_config* is the loaded stage config (``env_kwargs`` with its live
    command, ``curriculum_kwargs`` declaring the kind).  The policy is
    loaded through ``stance_report._load_policy`` (the frozen VecNormalize
    statistics, SB3's deterministic predict) and checked against the
    current plant; with *reference_model_path* the follow profile's
    reference is rolled on the same plan.  Checked before anything is
    rolled: the stage declares the kind and its thresholds read strictly,
    and its command is live.  Raises :class:`CommandTrackingReportError`
    for a stage that cannot be rolled.
    """
    from ..command_tracking_panel import CommandPanelError, plan_command_panel, roll_command_panel
    from ..plant_contract import current_plant_identity
    from ..task_fingerprint import stage_task_fingerprint

    curriculum = stage_config["curriculum_kwargs"]
    if curriculum.get("gate_kind") != COMMAND_TRACKING_GATE_KIND:
        raise ValueError(
            f"stage {stage} declares gate_kind {curriculum.get('gate_kind')!r}, not {COMMAND_TRACKING_GATE_KIND!r}"
        )
    thresholds = thresholds_from_curriculum(curriculum)
    env_kwargs = dict(stage_config["env_kwargs"])
    try:
        command_config = command_config_from(env_kwargs.get("command_config"))
    except (TypeError, ValueError) as exc:
        raise CommandTrackingReportError(f"stage {stage}'s command config is unreadable: {exc}") from exc
    if command_config is None:
        raise CommandTrackingReportError(
            f"stage {stage} declares {COMMAND_TRACKING_GATE_KIND} but runs command_mode 'none': there is no command "
            "to track"
        )
    plant_identity = current_plant_identity(model_id)
    task_sha256 = stage_task_fingerprint(model_id, stage, env_kwargs=env_kwargs, plant_identity=plant_identity)[
        "task_sha256"
    ]
    predict, description, normalization = _policy(
        model_id,
        env_kwargs,
        model_path=model_path,
        vecnorm_path=vecnorm_path,
        zero_action=zero_action,
        plant_identity=plant_identity,
        allow_legacy_plant=allow_legacy_plant,
    )
    plan = plan_command_panel(thresholds, command_config, seed_start=seed_start)
    try:
        rollout = roll_command_panel(
            model_id, stage_config, predict=predict, thresholds=thresholds, plan=plan, plant_identity=plant_identity
        )
        reference_rollout = None
        reference_description = reference_normalization = None
        if reference_model_path is not None:
            reference_predict, reference_description, reference_normalization = _policy(
                model_id,
                env_kwargs,
                model_path=reference_model_path,
                vecnorm_path=reference_vecnorm_path,
                zero_action=False,
                plant_identity=plant_identity,
                allow_legacy_plant=allow_legacy_plant,
            )
            reference_rollout = roll_command_panel(
                model_id,
                stage_config,
                predict=reference_predict,
                thresholds=thresholds,
                plan=plan,
                plant_identity=plant_identity,
            )
    except CommandPanelError as exc:
        raise CommandTrackingReportError(f"stage {stage}'s command panel cannot be rolled: {exc}") from exc
    reference_block: dict[str, Any] | None = None
    if reference_rollout is not None:
        reference_block = {
            "policy": reference_description,
            "normalization": reference_normalization,
            "handoff": _pair_record(reference_model_path, reference_vecnorm_path),
        }
    episodes = rollout.episodes
    return assemble_command_tracking_report(
        model_id,
        stage,
        curriculum=curriculum,
        command_config=command_config,
        plan=plan,
        episodes=episodes,
        measurement=rollout.measurement,
        horizon=(episodes[0].horizon if episodes else int(env_kwargs.get("max_episode_steps", 1000))),
        control_dt=(
            episodes[0].dt
            if episodes
            else float(env_kwargs.get("timestep", 0.002)) * int(env_kwargs.get("frame_skip", 5))
        ),
        policy=description,
        normalization=normalization,
        handoff=(
            {"checkpoint": None, "checkpoint_sha256": None, "normalization_sha256": None}
            if zero_action
            else _pair_record(model_path, vecnorm_path)
        ),
        task_sha256=task_sha256,
        plant_identity=plant_identity.to_dict(),
        checkpoint_plant_validated=None if zero_action else not allow_legacy_plant,
        reference=reference_block,
        reference_episodes=None if reference_rollout is None else reference_rollout.episodes,
    )


def assemble_command_tracking_report(
    model_id: ModelId,
    stage: "int | str",
    *,
    curriculum: Mapping[str, Any],
    command_config: Any,
    plan: Any,
    episodes: Sequence[CommandTrackingEpisode],
    measurement: Any,
    horizon: int,
    control_dt: float,
    policy: str,
    normalization: Mapping[str, Any],
    handoff: Mapping[str, Any],
    task_sha256: str | None,
    plant_identity: Mapping[str, Any] | None,
    checkpoint_plant_validated: bool | None,
    reference: Mapping[str, Any] | None = None,
    reference_episodes: Sequence[CommandTrackingEpisode] | None = None,
) -> dict[str, Any]:
    """The report for a rolled panel: the gate's verdict on *episodes* and everything the readers bind.

    :func:`build_command_tracking_report`'s second half, with no env: the
    verdict is
    :func:`~environments.shared.curriculum.command_tracking_gate.evaluate_command_tracking_gate`
    on *episodes* (and *reference_episodes*, recorded under *reference*,
    whose ``policy``, ``normalization`` and ``handoff`` the caller gives);
    *plan* is the ``command_tracking_panel.CommandPanelPlan`` the episodes
    were rolled on, *measurement* their ``PanelMeasurement``.  The layout is
    the one ``reporting.gates`` admits and ``command_tracking_evidence.panel_rows``
    writes.
    """
    from ..config import env_kwargs_record

    thresholds = thresholds_from_curriculum(curriculum)
    result = evaluate_command_tracking_gate(
        episodes,
        thresholds,
        command_config=command_config,
        reference=None if reference_episodes is None else list(reference_episodes),
    )
    view = gate_config_view(curriculum)
    reference_block: dict[str, Any] | None = None
    if reference is not None:
        reference_block = {
            **dict(reference),
            "episode_evidence": _evidence(reference_episodes or (), None),
        }
    return {
        "schema": COMMAND_TRACKING_REPORT_SCHEMA,
        **model_id.record(),
        "stage": stage,
        "gate_kind": curriculum.get("gate_kind"),
        "scored_gate_kind": COMMAND_TRACKING_GATE_KIND,
        "profile": thresholds.profile,
        "protocol": COMMAND_PANEL_PROTOCOL,
        "seed_start": int(plan.seed_start),
        "plan": plan.as_dict(),
        "policy": policy,
        "policy_deterministic": True,
        "normalization": dict(normalization),
        "horizon": int(horizon),
        "control_dt": float(control_dt),
        "command_config": env_kwargs_record({"command_config": command_config_from(command_config)})["command_config"],
        "measurement": measurement.as_dict(),
        "passed": result.passed,
        "failures": list(result.failures),
        "result": result.as_dict(),
        "summary": summarize_command_panel(episodes),
        "thresholds": view["thresholds"],
        "gate_sha256": gate_config_sha256(view),
        "handoff": dict(handoff),
        "task_sha256": task_sha256,
        "plant_identity": None if plant_identity is None else dict(plant_identity),
        "checkpoint_plant_validated": checkpoint_plant_validated,
        "episode_evidence": _evidence(episodes, result),
        "reference": reference_block,
    }


def _shown(value: Any, spec: str) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return format(value, spec)


def _window_line(report: Mapping[str, Any]) -> str:
    """The tracking window the panel was judged on and the cruise family's report-only longest stop (decision D-D64)."""
    measurement = report.get("measurement") or {}
    window = measurement.get("tracking_window_s") if isinstance(measurement, Mapping) else None
    shown = "per step" if window == 0 else (f"{_shown(window, 'g')} s (a moving command's errors are window means)")
    summary = report.get("summary") or {}
    cruise = ((summary.get("families") or {}).get(CRUISE_FAMILY) or {}) if isinstance(summary, Mapping) else {}
    longest = cruise.get("longest_stop_s") if isinstance(cruise, Mapping) else None
    if isinstance(longest, Mapping):
        shown += (
            f"; cruise longest stop {_shown(longest.get('mean'), '.3f')} s mean, {_shown(longest.get('max'), '.3f')} s "
            "max (report-only)"
        )
    return shown


def render_command_tracking_report(report: Mapping[str, Any]) -> str:
    """The text form of a command-tracking report: the binding, each family's numbers, the failing episodes, the verdict."""
    result = report.get("result") or {}
    thresholds = report.get("thresholds") or {}
    plan = report.get("plan") or {}
    seeds = list(plan.get("seeds") or ())
    lines = [
        f"policy              {report.get('policy')}",
        f"stage               {report.get('species')} / {report.get('embodiment')} — stage {report.get('stage')} "
        f"({report.get('scored_gate_kind')}, profile {report.get('profile')})",
        f"protocol            {report.get('protocol')}; {len(seeds)} seeds from {report.get('seed_start')} "
        f"(scanned {plan.get('scanned')}); horizon {report.get('horizon')} steps at {report.get('control_dt')} s",
        f"measurement         {report.get('measurement')}",
        f"tracking window     {_window_line(report)}",
        f"task                {report.get('task_sha256')}",
        f"gate                {report.get('gate_sha256')}",
        "",
    ]
    for family in result.get("families") or ():
        name = family.get("family")
        line = (
            f"{name:<8} {family.get('n_success', 0):4d} / {family.get('n_episodes', 0):<4d} successful "
            f"({_shown(family.get('success_fraction'), '.3f')}); survived {family.get('n_survived', 0)}"
        )
        if family.get("success_lcb") is not None:
            line += (
                f"; success LCB95 {_shown(family.get('success_lcb'), '.4f')}, survival LCB95 "
                f"{_shown(family.get('survival_lcb'), '.4f')}"
            )
        lines.append(line)
        failures = family.get("criterion_failures") or {}
        if failures:
            lines.append(
                "         episodes failing each criterion: " + ", ".join(f"{k} {v}" for k, v in failures.items())
            )
        exposure = family.get("command_exposure") or {}
        for label, block in exposure.items():
            lines.append(
                f"         {label}: {block.get('successful_episodes')} / {block.get('exposed_episodes')} exposed "
                f"episodes settled ({_shown(block.get('success_fraction'), '.3f')})"
            )
    if result.get("n_pairs") is not None:
        lines.append(
            f"reference: {result.get('n_pairs')} paired seeds, success difference LCB95 "
            f"{_shown(result.get('paired_delta_lcb'), '.4f')} (>= {thresholds.get('min_follow_reference_delta_lcb')})"
        )
    failing = [entry for entry in report.get("episode_evidence") or () if not entry.get("success")]
    if failing:
        lines += ["", f"episodes that failed ({len(failing)}; first {min(len(failing), 10)} shown):"]
        for entry in failing[:10]:
            record = entry.get("record") or {}
            lines.append(
                f"  seed {record.get('seed')} ({record.get('family')}): " + "; ".join(entry.get("reasons") or ())
            )
    lines += ["", f"GATE: {'PASS' if report.get('passed') else 'FAIL'}"]
    lines += [f"  - {failure}" for failure in report.get("failures") or ()]
    return "\n".join(lines)


def write_command_tracking_report(stage_dir: "str | Path", report: Mapping[str, Any]) -> dict[str, Path]:
    """Write the report beside the stage's other artifacts: the text, the JSON, then the panel CSV, each atomically.

    The CSV is publication's evidence, so a write that stops part-way
    leaves a report publication refuses for want of it, never a CSV without
    the report it came from.
    """
    if report.get("schema") != COMMAND_TRACKING_REPORT_SCHEMA:
        raise ValueError(f"not a {COMMAND_TRACKING_REPORT_SCHEMA} report ({report.get('schema')!r})")
    require_current_protocol(report)
    directory = Path(stage_dir)
    directory.mkdir(parents=True, exist_ok=True)
    rows = panel_rows(report)
    text_path = directory / COMMAND_TRACKING_REPORT_TXT
    json_path = directory / COMMAND_TRACKING_REPORT_JSON
    atomic_write_text(text_path, render_command_tracking_report(report) + "\n")
    atomic_write_json(json_path, dict(report), sort_keys=True, allow_nan=False)
    csv_path = atomic_write_csv(directory / COMMAND_TRACKING_PANEL_CSV, COMMAND_TRACKING_PANEL_FIELDNAMES, rows)
    return {
        "command_tracking_report_txt": text_path,
        "command_tracking_report_json": json_path,
        "command_tracking_panel_csv": Path(csv_path),
    }


def command_tracking_stage_result(
    report: Mapping[str, Any] | None, *, rederived: CommandTrackingResult | None = None
) -> dict[str, Any]:
    """The ``stage_result`` numbers a ``command_tracking/v1`` verdict was judged on (:data:`COMMAND_TRACKING_STAGE_RESULT_KEYS`).

    Read off *rederived* (the judge's re-derivation of an admitted report)
    when given, never the report's own ``result`` block; empty for no report
    or one not scored as this kind.  A family the profile does not judge
    records ``None``.  ``command_handoff_selected`` is the handoff name the
    report scored.
    """
    if not isinstance(report, Mapping) or report.get("scored_gate_kind") != COMMAND_TRACKING_GATE_KIND:
        return {}
    values: dict[str, Any] = dict.fromkeys(COMMAND_TRACKING_STAGE_RESULT_KEYS)
    values["command_profile"] = report.get("profile")
    values[COMMAND_HANDOFF_SELECTED_KEY] = ((report.get("handoff") or {}).get("checkpoint") or "").removesuffix(
        ".zip"
    ) or None
    if rederived is None:
        return {**values, "command_n_episodes": len(report.get("episode_evidence") or ())}

    def finite(value: float | None) -> float | None:
        return None if value is None or not math.isfinite(value) else float(value)

    values["command_n_episodes"] = rederived.n_episodes
    families = {family.family: family for family in rederived.families}
    if rederived.profile == WALK_PROFILE:
        cruise, stand = families.get(CRUISE_FAMILY), families.get(STAND_FAMILY)
        if cruise is not None:
            values.update(
                command_cruise_episodes=cruise.n_episodes,
                command_cruise_successes=cruise.n_success,
                command_cruise_success_lcb=finite(cruise.success_lcb),
                command_cruise_survival_lcb=finite(cruise.survival_lcb),
            )
        if stand is not None:
            values.update(
                command_stand_episodes=stand.n_episodes,
                command_stand_successes=stand.n_success,
                command_stand_success_fraction=finite(stand.success_fraction),
                command_stand_falls=stand.n_episodes - stand.n_survived,
            )
    else:
        follow = families.get(FOLLOW_PROFILE)
        if follow is not None:
            values.update(
                command_follow_episodes=follow.n_episodes,
                command_follow_successes=follow.n_success,
                command_follow_success_lcb=finite(follow.success_lcb),
                command_follow_survival_lcb=finite(follow.survival_lcb),
            )
        values["command_paired_delta_lcb"] = finite(rederived.paired_delta_lcb)
    return values


# ── the handoff chosen by the gated statistic ────────────────────────────────


def shared_panel_seeds(selection: Sequence[Any], certification: Sequence[Any]) -> list[int]:
    """The seeds a handoff choice's selection panel and a certification panel share, ascending (empty: disjoint)."""
    return sorted(set(int(seed) for seed in selection) & set(int(seed) for seed in certification))


def handoff_candidates(model_dir: "str | Path", final_stem: "str | Path | None" = None) -> list[tuple[str, str, str]]:
    """The complete pairs a handoff choice ranks: ``robust_best_model``, ``best_model``, then the final pair.

    Each ``(name, stem, vecnorm path)``, only when both its ``.zip`` and its
    ``_vecnorm.pkl`` exist; *final_stem* is the final checkpoint's stem
    (``models/<stage_label>_final``), skipped when ``None``.
    """
    directory = Path(model_dir)
    stems = [directory / name for name in HANDOFF_CANDIDATE_NAMES]
    if final_stem is not None:
        stems.append(Path(final_stem))
    candidates: list[tuple[str, str, str]] = []
    for stem in stems:
        if stem.parent.resolve() != directory.resolve():
            raise ValueError(f"the candidate {stem} is not in {directory}; a handoff is a pair in the stage's models/")
        zip_path, vecnorm_path = Path(f"{stem}.zip"), Path(f"{stem}_vecnorm.pkl")
        if zip_path.is_file() and vecnorm_path.is_file() and stem.name not in {name for name, _, _ in candidates}:
            candidates.append((stem.name, str(stem), str(vecnorm_path)))
    return candidates


def choose_command_handoff(
    model_id: ModelId,
    stage: "int | str",
    *,
    stage_config: Mapping[str, Any],
    model_dir: "str | Path",
    final_stem: "str | Path | None" = None,
    seed_start: int = HANDOFF_SELECTION_SEED_START,
) -> dict[str, Any] | None:
    """Choose a ``command_tracking/v1`` stage's handoff by the gated statistic and record it; ``None`` when none is complete.

    Every complete candidate (:func:`handoff_candidates`) is rolled on the
    selection panel planned from *seed_start* (the stage's own thresholds
    and command), judged by the gate (the follow profile's paired reference
    is not rolled here: the reference is one policy for every candidate,
    and the success bound ranks them), and ranked by
    :func:`~environments.shared.curriculum.command_tracking_gate.handoff_selection_key`;
    the first candidate in order wins a tie.  The record (schema
    ``checkpoints.HANDOFF_CHOICE_SCHEMA``) names the winner with its pair's
    digests and keeps every candidate's digests and numbers, the rule, the
    plan, the gate digest and the task, and is written to
    ``models/handoff_choice.json`` (``checkpoints.write_handoff_choice``),
    which ``select_handoff_checkpoint`` then honors.  A candidate whose
    panel cannot be rolled is recorded with its error and ranks last; when
    no candidate can be rolled, nothing is recorded and
    :class:`CommandTrackingReportError` is raised, so the stage has no
    choice and its judge refuses it for want of one.  The same happens,
    before any candidate is rolled, when the selection panel shares a seed
    with the certification panel planned from
    :data:`~environments.shared.constants.PUBLICATION_SEED_START`
    (:func:`shared_panel_seeds`): the planner scans up to ``MAX_PANEL_SCAN``
    seeds for a rare stand family, so a low stand rate or a large panel can
    run one block into the other, and a choice made on the certification
    draws would certify the best of three.
    """
    from ..command_tracking_panel import plan_command_panel
    from ..curriculum.checkpoints import HANDOFF_CHOICE_SCHEMA, write_handoff_choice
    from ..task_fingerprint import stage_task_fingerprint

    candidates = handoff_candidates(model_dir, final_stem)
    if not candidates:
        return None
    curriculum = stage_config["curriculum_kwargs"]
    thresholds = thresholds_from_curriculum(curriculum)
    command_config = command_config_from(stage_config["env_kwargs"].get("command_config"))
    if command_config is None:
        raise CommandTrackingReportError(f"stage {stage} runs command_mode 'none': there is no command to track")
    plan = plan_command_panel(thresholds, command_config, seed_start=seed_start)
    certification = plan_command_panel(thresholds, command_config, seed_start=PUBLICATION_SEED_START)
    shared = shared_panel_seeds(plan.seeds, certification.seeds)
    if shared:
        raise CommandTrackingReportError(
            f"stage {stage}: the selection panel from seed {seed_start} shares {len(shared)} seed(s) with the "
            f"certification panel from seed {PUBLICATION_SEED_START} (first {shared[0]}); a handoff chosen on the "
            "certification draws would certify the best of several, so no handoff is chosen"
        )
    scored: list[dict[str, Any]] = []
    ranked: list[tuple[tuple[float, ...], int, str]] = []
    for index, (name, stem, vecnorm) in enumerate(candidates):
        entry: dict[str, Any] = {"name": name, **_pair_record(f"{stem}.zip", vecnorm)}
        try:
            report = build_command_tracking_report(
                model_id,
                stage,
                stage_config=stage_config,
                model_path=f"{stem}.zip",
                vecnorm_path=vecnorm,
                seed_start=seed_start,
            )
            require_current_protocol(report)
            result = report["result"]
            key = handoff_selection_key(
                evaluate_command_tracking_gate(
                    [_episode(entry_record) for entry_record in report["episode_evidence"]],
                    thresholds,
                    command_config=command_config,
                )
            )
            entry.update(
                passed=report["passed"], key=list(key), families=result["families"], failures=report["failures"]
            )
        except Exception as exc:  # noqa: BLE001 - one candidate that cannot be rolled ranks last, and says why
            logger.warning("Handoff choice for stage %s: candidate %s could not be scored", stage, name, exc_info=True)
            key = (-math.inf,)
            entry.update(passed=False, key=None, error=f"{type(exc).__name__}: {exc}")
        scored.append(entry)
        ranked.append((key, -index, name))
    if all(entry.get("key") is None for entry in scored):
        raise CommandTrackingReportError(
            f"stage {stage}: no candidate's selection panel could be rolled ("
            + "; ".join(f"{entry['name']}: {entry.get('error')}" for entry in scored)
            + "), so no handoff is chosen"
        )
    best_key, _, best_name = max(ranked)
    winner = next(entry for entry in scored if entry["name"] == best_name)
    record = {
        "schema": HANDOFF_CHOICE_SCHEMA,
        **model_id.record(),
        "stage": stage,
        "gate_kind": COMMAND_TRACKING_GATE_KIND,
        "gate_sha256": gate_config_sha256(gate_config_view(curriculum)),
        "profile": thresholds.profile,
        "rule": HANDOFF_SELECTION_RULES[thresholds.profile],
        "selection_seed_start": int(seed_start),
        "plan": plan.as_dict(),
        "task_sha256": stage_task_fingerprint(model_id, stage, env_kwargs=dict(stage_config["env_kwargs"]))[
            "task_sha256"
        ],
        "selected": best_name,
        "checkpoint_sha256": winner["checkpoint_sha256"],
        "normalization_sha256": winner["normalization_sha256"],
        "candidates": scored,
    }
    write_handoff_choice(model_dir, _json_ready(record))
    logger.info(
        "Handoff chosen by the gated statistic for stage %s: %s (%s; key %s) among %s",
        stage,
        best_name,
        record["rule"],
        list(best_key),
        ", ".join(name for name, _, _ in candidates),
    )
    return record


def _episode(entry: Mapping[str, Any]) -> CommandTrackingEpisode:
    return CommandTrackingEpisode.from_dict(entry["record"])


def _json_ready(value: Any) -> Any:
    """*value* with every non-finite float as ``None`` (the record is strict JSON)."""
    if isinstance(value, Mapping):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
