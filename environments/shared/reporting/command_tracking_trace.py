"""The ``command_tracking/v1`` chart episodes: their step trace, rolled after the verdict (decision D-D61).

The command-tracking figure (``visualization.plot_command_tracking``) draws
commanded against achieved speed and yaw rate over the steps of panel
episodes.  The report keeps one record per episode and no step rows, so
this module rolls the chart episodes again, after the verdict, with the
step rows kept:

* :func:`chart_plan` picks them from the certification panel's plan: the
  first seed of each reset family in plan order (the first episode whose
  reset drew a cruise command and the first whose reset drew a stand, in
  either profile);
* :func:`roll_command_trace` rolls that sub-plan on the pair the report
  scored, under its frozen VecNormalize statistics and the same
  deterministic predict
  (``command_tracking_panel.roll_command_panel(keep_steps=True)``), and
  reduces each step to :data:`COMMAND_TRACKING_TRACE_FIELDNAMES`;
* :func:`write_command_tracking_trace` writes the rows to
  :data:`COMMAND_TRACKING_TRACE_CSV` in the stage directory, and
  :func:`read_command_tracking_trace` reads them back for the figure.

A panel episode is a pure function of its reset seed and the policy, so
the re-rolled episode is the one the gate judged: each trace row records
whether the episode's reduced record equals the report's record for that
seed (``record_matches_report``), and the figure says so.  Each row also
carries the checkpoint digest of the pair it was rolled on
(``checkpoint_sha256``, the report's handoff), and the figure draws only the
rows whose digest is the report's: a trace left from another pair is never
drawn as this one's.  The trace is
report-only.  Nothing judges, binds or publishes it, and a failure here
costs the trace and the figure, never the verdict.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..constants import PUBLICATION_SEED_START
from ..curriculum.command_tracking_gate import (
    COMMAND_TRACKING_GATE_KIND,
    CommandTrackingEpisode,
    PanelMeasurement,
    command_config_from,
    sample_in_tolerance,
    speed_along_heading,
    thresholds_from_curriculum,
)
from ..file_io import atomic_write_csv
from ..model_id import ModelId

#: The chart episodes' step trace, in the stage directory beside the command-tracking report.
COMMAND_TRACKING_TRACE_CSV = "command_tracking_trace.csv"
#: One row per step of a chart episode.  ``in_tolerance`` is the gate's per-step test
#: (``command_tracking_gate.sample_in_tolerance``) at the report's measurement, and
#: ``speed_along_heading_mps`` the root's speed along the requested heading, the quantity the gate's speed ratio
#: averages; ``heading_error_rad`` is the env's (0 while no heading is commanded).
COMMAND_TRACKING_TRACE_FIELDNAMES: tuple[str, ...] = (
    "episode_seed",
    "family",
    "checkpoint_sha256",
    "record_matches_report",
    "step",
    "time_s",
    "desired_speed",
    "command_v_x",
    "command_v_y",
    "command_yaw_rate",
    "actual_speed",
    "speed_along_heading_mps",
    "actual_yaw_rate",
    "heading_error_rad",
    "in_tolerance",
    "course_progress_m",
    "reward",
    "terminated",
)
#: The trace's columns read back as text, as booleans, and as integers; every other column is a float.
_TEXT_COLUMNS = frozenset({"family", "checkpoint_sha256"})
_BOOL_COLUMNS = frozenset({"record_matches_report", "in_tolerance", "terminated"})
_INT_COLUMNS = frozenset({"episode_seed", "step"})


def chart_plan(plan: Any) -> Any:
    """The chart episodes of a ``command_tracking_panel.CommandPanelPlan``: the first seed of each reset family, in order."""
    from ..command_tracking_panel import CommandPanelPlan

    seeds: list[int] = []
    families: list[str] = []
    for seed, family in zip(plan.seeds, plan.families):
        if family not in families:
            seeds.append(int(seed))
            families.append(family)
    return CommandPanelPlan(
        profile=plan.profile,
        seed_start=plan.seed_start,
        seeds=tuple(seeds),
        families=tuple(families),
        scanned=plan.scanned,
    )


def _number(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number


def _canonical(record: Mapping[str, Any]) -> str:
    """*record* as comparable text, without ``episode``: a chart episode's position in its sub-plan is not the panel's."""
    return json.dumps({key: value for key, value in record.items() if key != "episode"}, sort_keys=True, allow_nan=True)


def trace_rows(
    episodes: Sequence[CommandTrackingEpisode],
    steps: Sequence[Sequence[Mapping[str, Any]]],
    measurement: PanelMeasurement,
    *,
    report: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """The trace rows of rolled chart *episodes* and their step rows (``CommandPanelRollout.steps``).

    ``record_matches_report`` is whether the episode's record (``as_dict``)
    equals *report*'s record for the same seed in every field but
    ``episode`` (the record's position in its plan); ``False`` without one.
    ``checkpoint_sha256`` is *report*'s handoff checkpoint digest, the pair
    the trace is rolled on (empty without a report).
    """
    if len(episodes) != len(steps):
        raise ValueError(f"{len(episodes)} chart episodes with {len(steps)} step sequences")
    recorded: dict[int, str] = {}
    for entry in (report or {}).get("episode_evidence") or ():
        record = entry.get("record") if isinstance(entry, Mapping) else None
        if isinstance(record, Mapping) and isinstance(record.get("seed"), int):
            recorded[int(record["seed"])] = _canonical(record)
    handoff = (report or {}).get("handoff")
    digest = handoff.get("checkpoint_sha256") if isinstance(handoff, Mapping) else None
    rows: list[dict[str, Any]] = []
    for episode, episode_steps in zip(episodes, steps):
        matches = recorded.get(episode.seed) == _canonical(episode.as_dict())
        for row in episode_steps:
            rows.append(
                {
                    "episode_seed": episode.seed,
                    "family": episode.family,
                    "checkpoint_sha256": digest if isinstance(digest, str) else "",
                    "record_matches_report": matches,
                    "step": int(row["step"]),
                    "time_s": _number(row.get("time_s")),
                    "desired_speed": _number(row.get("desired_speed")),
                    "command_v_x": _number(row.get("command_v_x")),
                    "command_v_y": _number(row.get("command_v_y")),
                    "command_yaw_rate": _number(row.get("command_yaw_rate")),
                    "actual_speed": _number(row.get("actual_speed")),
                    "speed_along_heading_mps": speed_along_heading(row),
                    "actual_yaw_rate": _number(row.get("actual_yaw_rate")),
                    "heading_error_rad": _number(row.get("heading_error_rad")),
                    "in_tolerance": sample_in_tolerance(row, measurement),
                    "course_progress_m": _number(row.get("course_progress_m")),
                    "reward": _number(row.get("reward")),
                    "terminated": bool(row.get("terminated")),
                }
            )
    return rows


def roll_command_trace(
    model_id: ModelId,
    stage: "int | str",
    *,
    stage_config: Mapping[str, Any],
    report: Mapping[str, Any] | None = None,
    model_path: str | None = None,
    vecnorm_path: str | None = None,
    zero_action: bool = False,
    seed_start: int = PUBLICATION_SEED_START,
    allow_legacy_plant: bool = False,
) -> list[dict[str, Any]]:
    """Roll the chart episodes of a ``command_tracking/v1`` stage's panel with their step rows kept: the trace rows.

    The panel is planned from *seed_start* under the stage's own thresholds
    and command (the plan the report was rolled on), and :func:`chart_plan`
    picks its chart episodes.  The policy is loaded as the report loads it
    (``command_tracking_report._policy``: the frozen VecNormalize statistics,
    SB3's deterministic predict, the current plant), or the statue with
    *zero_action*.  Raises ``ValueError`` for a stage that does not declare
    the kind and ``command_tracking_report.CommandTrackingReportError`` for
    one that cannot be rolled.
    """
    from ..command_tracking_panel import CommandPanelError, plan_command_panel, roll_command_panel
    from ..plant_contract import current_plant_identity
    from .command_tracking_report import CommandTrackingReportError, _policy

    curriculum = stage_config["curriculum_kwargs"]
    if curriculum.get("gate_kind") != COMMAND_TRACKING_GATE_KIND:
        raise ValueError(
            f"stage {stage} declares gate_kind {curriculum.get('gate_kind')!r}, not {COMMAND_TRACKING_GATE_KIND!r}"
        )
    thresholds = thresholds_from_curriculum(curriculum)
    env_kwargs = dict(stage_config["env_kwargs"])
    command_config = command_config_from(env_kwargs.get("command_config"))
    if command_config is None:
        raise CommandTrackingReportError(f"stage {stage} runs command_mode 'none': there is no command to chart")
    plant_identity = current_plant_identity(model_id)
    predict, _description, _normalization = _policy(
        model_id,
        env_kwargs,
        model_path=model_path,
        vecnorm_path=vecnorm_path,
        zero_action=zero_action,
        plant_identity=plant_identity,
        allow_legacy_plant=allow_legacy_plant,
    )
    plan = chart_plan(plan_command_panel(thresholds, command_config, seed_start=seed_start))
    try:
        rollout = roll_command_panel(
            model_id,
            stage_config,
            predict=predict,
            thresholds=thresholds,
            plan=plan,
            plant_identity=plant_identity,
            keep_steps=True,
        )
    except CommandPanelError as exc:
        raise CommandTrackingReportError(f"stage {stage}'s chart episodes cannot be rolled: {exc}") from exc
    assert rollout.steps is not None
    return trace_rows(rollout.episodes, rollout.steps, rollout.measurement, report=report)


def write_command_tracking_trace(stage_dir: "str | Path", rows: Sequence[Mapping[str, Any]]) -> Path:
    """Write the trace rows to :data:`COMMAND_TRACKING_TRACE_CSV` in *stage_dir*, atomically (floats as ``repr``)."""

    def cell(column: str, value: Any) -> Any:
        if column in _BOOL_COLUMNS:
            return "true" if value else "false"
        if isinstance(value, float):
            return "nan" if math.isnan(value) else repr(value)
        return value

    rendered = [{column: cell(column, row[column]) for column in COMMAND_TRACKING_TRACE_FIELDNAMES} for row in rows]
    return atomic_write_csv(Path(stage_dir) / COMMAND_TRACKING_TRACE_CSV, COMMAND_TRACKING_TRACE_FIELDNAMES, rendered)


def read_command_tracking_trace(path: "str | Path") -> list[dict[str, Any]] | None:
    """The trace rows of a :data:`COMMAND_TRACKING_TRACE_CSV`, or ``None`` when there is no file.

    For the figure: a cell that does not parse reads NaN (``False`` for a
    flag), and a file whose header is not :data:`COMMAND_TRACKING_TRACE_FIELDNAMES`
    raises ``ValueError``.
    """
    file = Path(path)
    if not file.is_file():
        return None
    with file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != COMMAND_TRACKING_TRACE_FIELDNAMES:
            raise ValueError(
                f"{file.name} has the columns {reader.fieldnames}, not {list(COMMAND_TRACKING_TRACE_FIELDNAMES)}"
            )
        rows: list[dict[str, Any]] = []
        for raw in reader:
            row: dict[str, Any] = {}
            for column in COMMAND_TRACKING_TRACE_FIELDNAMES:
                text = raw.get(column) or ""
                if column in _TEXT_COLUMNS:
                    row[column] = text
                elif column in _BOOL_COLUMNS:
                    row[column] = text.strip().lower() == "true"
                elif column in _INT_COLUMNS:
                    try:
                        row[column] = int(text)
                    except ValueError:
                        row[column] = -1
                else:
                    row[column] = _number(text)
            rows.append(row)
    return rows


def trace_rows_of(rows: Sequence[Mapping[str, Any]], report: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    """The trace *rows* rolled on *report*'s handoff checkpoint (their ``checkpoint_sha256`` its digest); none without."""
    handoff = (report or {}).get("handoff")
    digest = handoff.get("checkpoint_sha256") if isinstance(handoff, Mapping) else None
    if not isinstance(digest, str) or not digest:
        return []
    return [row for row in rows if row.get("checkpoint_sha256") == digest]


def trace_episodes(rows: Sequence[Mapping[str, Any]]) -> list[tuple[int, str, list[Mapping[str, Any]]]]:
    """The trace *rows* grouped by episode, in file order: ``(seed, family, rows)``."""
    episodes: list[tuple[int, str, list[Mapping[str, Any]]]] = []
    for row in rows:
        seed = int(row["episode_seed"])
        if not episodes or episodes[-1][0] != seed:
            episodes.append((seed, str(row["family"]), []))
        episodes[-1][2].append(row)
    return episodes
