"""The ``command_tracking/v1`` evidence: the report's names, the panel CSV's layout and its strict reader.

A stage gated on ``command_tracking/v1`` is judged after its training on a
panel rolled on its handoff pair
(``reporting.command_tracking_report.build_command_tracking_report``).  The
panel is written beside the stage's other artifacts as three files:

* :data:`COMMAND_TRACKING_REPORT_JSON` (schema
  :data:`COMMAND_TRACKING_REPORT_SCHEMA`), the report the post-stage judge
  (``reporting.gates.evaluate_stage_gate``) admits and re-derives from its
  per-episode records;
* :data:`COMMAND_TRACKING_REPORT_TXT`, its text form;
* :data:`COMMAND_TRACKING_PANEL_CSV`, one row per episode
  (:data:`COMMAND_TRACKING_PANEL_FIELDNAMES`), which publication
  (``result_bundle.evidence``) re-derives the verdict from alone.

Every CSV row is one episode record
(:class:`~.command_tracking_gate.CommandTrackingEpisode`: every scalar field,
then each gait metric under ``gait_<metric>``), its recorded classification
(``success`` and ``reasons``) and the stamps that bind the panel: the
profile, the measurement (protocol, the five tolerances, ``yaw_rate_max``,
the gait measurement), the handoff pair's digests, the reference pair's when
the follow profile rolled one, the task and gate digests, the panel's first
seed, horizon and control step, and whether the policy was rolled
deterministically.  A reference row (``role`` ``reference``) is never
classified.  Floats are written so they read back bit for bit (``repr``;
``nan`` for an unmeasured value), and :func:`read_command_tracking_panel`
refuses a file it cannot read exactly.

This module is pure: the standard library and the gate module.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from ..gait.locomotion_metrics import LOCOMOTION_METRIC_FIELDS
from .command_tracking_gate import (
    TOLERANCE_NAMES,
    CommandTrackingEpisode,
    PanelMeasurement,
)

#: The report a command-tracking panel writes, and its files in the stage directory.
COMMAND_TRACKING_REPORT_SCHEMA = "mesozoic.command-tracking-report/v1"
COMMAND_TRACKING_REPORT_JSON = "command_tracking_report.json"
COMMAND_TRACKING_REPORT_TXT = "command_tracking_report.txt"
COMMAND_TRACKING_PANEL_CSV = "command_tracking_panel_selected.csv"

#: The ``stage_result`` keys a command-tracking verdict records (``gate_verdict._PERSISTED_STAGE_RESULT_KEYS``),
#: which the summary stage row, the results CSV and the catalog headline read: the profile, each judged family's
#: episode and success counts and bounds, the follow profile's paired reference bound, and the handoff name the
#: panel scored.  ``reporting.command_tracking_report.command_tracking_stage_result`` fills them.
COMMAND_TRACKING_STAGE_RESULT_KEYS: tuple[str, ...] = (
    "command_profile",
    "command_n_episodes",
    "command_cruise_episodes",
    "command_cruise_successes",
    "command_cruise_success_lcb",
    "command_cruise_survival_lcb",
    "command_stand_episodes",
    "command_stand_successes",
    "command_stand_success_fraction",
    "command_stand_falls",
    "command_follow_episodes",
    "command_follow_successes",
    "command_follow_success_lcb",
    "command_follow_survival_lcb",
    "command_paired_delta_lcb",
    "command_handoff_selected",
)
#: The stage-result key naming the handoff the panel scored; a summary row leaves it out, since the provenance's
#: certified checkpoint already states it.
COMMAND_HANDOFF_SELECTED_KEY = "command_handoff_selected"
#: The keys a summary stage row carries (``reporting.summaries``), and the numeric ones publication compares
#: against the results CSV's columns of the same names (``result_bundle.evidence.compare_summary_to_csv``).
COMMAND_TRACKING_SUMMARY_KEYS: tuple[str, ...] = tuple(
    key for key in COMMAND_TRACKING_STAGE_RESULT_KEYS if key != COMMAND_HANDOFF_SELECTED_KEY
)
COMMAND_TRACKING_NUMERIC_SUMMARY_KEYS: tuple[str, ...] = tuple(
    key for key in COMMAND_TRACKING_SUMMARY_KEYS if key != "command_profile"
)

#: A CSV row's role: the scored policy's episode (classified) or the follow profile's reference (never classified).
POLICY_ROLE = "policy"
REFERENCE_ROLE = "reference"

#: The episode record's scalar fields, in record order (the measurement is stamped, the gait metrics spread).
EPISODE_COLUMNS: tuple[str, ...] = tuple(
    spec.name for spec in fields(CommandTrackingEpisode) if spec.name not in ("measurement", "gait_metrics")
)
#: One column per locomotion metric, ``gait_<metric>``, in ``LocomotionEpisodeMetrics`` order.
GAIT_COLUMN_PREFIX = "gait_"
GAIT_COLUMNS: tuple[str, ...] = tuple(f"{GAIT_COLUMN_PREFIX}{name}" for name in LOCOMOTION_METRIC_FIELDS)
#: The measurement's stamps: the protocol, the gait measurement, ``yaw_rate_max`` and each tolerance.
MEASUREMENT_COLUMNS: tuple[str, ...] = (
    "protocol",
    "gait_measurement",
    "yaw_rate_max",
    *(f"tolerance_{name}" for name in TOLERANCE_NAMES),
)
#: The panel-wide stamps that bind the rows: what was scored, under which task and gate, on which panel.
STAMP_COLUMNS: tuple[str, ...] = (
    "profile",
    *MEASUREMENT_COLUMNS,
    "checkpoint_sha256",
    "normalization_sha256",
    "reference_checkpoint_sha256",
    "reference_normalization_sha256",
    "task_sha256",
    "gate_sha256",
    "seed_start",
    "horizon",
    "control_dt",
    "policy_deterministic",
)
COMMAND_TRACKING_PANEL_FIELDNAMES: tuple[str, ...] = (
    "role",
    *EPISODE_COLUMNS,
    *GAIT_COLUMNS,
    "success",
    "reasons",
    *STAMP_COLUMNS,
)

_INT_COLUMNS = frozenset(
    {
        "episode",
        "seed",
        "length",
        "horizon",
        "planned_event_count",
        "eligible_events",
        "settled_events",
        "eligible_stops",
        "settled_stops",
        "eligible_turns",
        "settled_turns",
    }
)
_BOOL_COLUMNS = frozenset({"full_horizon", "fall", "first_event_settled"})
_TEXT_COLUMNS = frozenset({"family", "termination_reason"})


def float_cell(value: Any) -> str:
    """A float column's cell: ``repr`` round-trips exactly; an unmeasured value (``None``, NaN) is ``nan``."""
    if value is None:
        return "nan"
    number = float(value)
    return "nan" if math.isnan(number) else repr(number)


def _episode_cells(record: Mapping[str, Any]) -> dict[str, Any]:
    """One episode record's (``CommandTrackingEpisode.as_dict()``) columns."""
    cells: dict[str, Any] = {}
    for name in EPISODE_COLUMNS:
        value = record[name]
        if name in _INT_COLUMNS:
            cells[name] = int(value)
        elif name in _BOOL_COLUMNS:
            cells[name] = bool(value)
        elif name in _TEXT_COLUMNS:
            cells[name] = "" if value is None else str(value)
        else:
            cells[name] = float_cell(value)
    gait = record.get("gait_metrics") or {}
    unknown = sorted(set(gait) - set(LOCOMOTION_METRIC_FIELDS))
    if unknown:
        raise ValueError(f"the episode record carries gait metrics the panel layout has no column for: {unknown}")
    for name in LOCOMOTION_METRIC_FIELDS:
        cells[f"{GAIT_COLUMN_PREFIX}{name}"] = float_cell(gait.get(name)) if name in gait else ""
    return cells


def _measurement_cells(measurement: Mapping[str, Any]) -> dict[str, Any]:
    tolerances = measurement.get("tolerances") or {}
    return {
        "protocol": measurement.get("protocol") or "",
        "gait_measurement": measurement.get("gait_measurement") or "",
        "yaw_rate_max": float_cell(measurement.get("yaw_rate_max")),
        **{f"tolerance_{name}": float_cell(tolerances.get(name)) for name in TOLERANCE_NAMES},
    }


def panel_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The panel CSV's rows for a command-tracking *report*: its policy episodes, then its reference's.

    Every row stamps the report's binding (:data:`STAMP_COLUMNS`); a policy
    row carries its recorded ``success`` and ``reasons`` (``"; "``-joined),
    a reference row neither.  Raises ``KeyError`` / ``ValueError`` for a
    report without the records it is built from.
    """
    handoff = report.get("handoff") or {}
    reference = report.get("reference") or None
    reference_handoff = (reference or {}).get("handoff") or {}
    stamps = {
        "profile": report["profile"],
        **_measurement_cells(report["measurement"]),
        "checkpoint_sha256": handoff.get("checkpoint_sha256") or "",
        "normalization_sha256": handoff.get("normalization_sha256") or "",
        "reference_checkpoint_sha256": reference_handoff.get("checkpoint_sha256") or "",
        "reference_normalization_sha256": reference_handoff.get("normalization_sha256") or "",
        "task_sha256": report.get("task_sha256") or "",
        "gate_sha256": report.get("gate_sha256") or "",
        "seed_start": int(report["seed_start"]),
        "horizon": int(report["horizon"]),
        "control_dt": float_cell(report["control_dt"]),
        "policy_deterministic": bool(report.get("policy_deterministic")),
    }
    rows: list[dict[str, Any]] = []
    for entry in report["episode_evidence"]:
        rows.append(
            {
                "role": POLICY_ROLE,
                **_episode_cells(entry["record"]),
                "success": bool(entry["success"]),
                "reasons": "; ".join(entry.get("reasons") or ()),
                **stamps,
            }
        )
    for entry in [] if reference is None else reference["episode_evidence"]:
        rows.append({"role": REFERENCE_ROLE, **_episode_cells(entry["record"]), "success": "", "reasons": "", **stamps})
    return rows


@dataclass(frozen=True)
class CommandTrackingPanelEvidence:
    """A ``command_tracking_panel_selected.csv`` read back: the episodes to re-judge and the stamps that bind them.

    ``policy`` and ``policy_success`` are aligned, in file order; the stamps
    are each the one value every row records (an empty stamp reads as
    ``None``).  ``measurement`` is the stamped one every record carries.
    """

    policy: tuple[CommandTrackingEpisode, ...]
    policy_success: tuple[bool, ...]
    reference: tuple[CommandTrackingEpisode, ...]
    profile: str
    measurement: PanelMeasurement
    checkpoint_sha256: str | None
    normalization_sha256: str | None
    reference_checkpoint_sha256: str | None
    reference_normalization_sha256: str | None
    task_sha256: str | None
    gate_sha256: str | None
    seed_start: int
    horizon: int
    control_dt: float
    policy_deterministic: bool


def _cell_bool(text: str | None, *, column: str, where: str) -> bool:
    normalized = (text or "").strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{where} has an unparsable {column} cell {text!r}")


def _cell_whole(text: str | None, *, column: str, where: str) -> int:
    stripped = (text or "").strip()
    sign = stripped[1:] if stripped[:1] == "-" else stripped
    if not sign.isdigit():
        raise ValueError(f"{where} has an unparsable {column} cell {text!r}")
    return int(stripped)


def _cell_float(text: str | None, *, column: str, where: str) -> float:
    stripped = (text or "").strip()
    try:
        return float(stripped)
    except ValueError as exc:
        raise ValueError(f"{where} has an unparsable {column} cell {text!r}") from exc


def _single_stamp(rows: Sequence[Mapping[str, str]], column: str, *, path: Path) -> str | None:
    values = {(row.get(column) or "").strip() for row in rows}
    if len(values) != 1:
        raise ValueError(f"{path} records more than one {column} ({sorted(values)}); one panel has one binding")
    value = values.pop()
    return value or None


def _episode_from_row(row: Mapping[str, str], measurement: PanelMeasurement, *, where: str) -> CommandTrackingEpisode:
    values: dict[str, Any] = {}
    for name in EPISODE_COLUMNS:
        text = row.get(name)
        if name in _INT_COLUMNS:
            values[name] = _cell_whole(text, column=name, where=where)
        elif name in _BOOL_COLUMNS:
            values[name] = _cell_bool(text, column=name, where=where)
        elif name == "family":
            values[name] = (text or "").strip()
        elif name == "termination_reason":
            values[name] = (text or "").strip() or None
        else:
            values[name] = _cell_float(text, column=name, where=where)
    gait: list[tuple[str, float]] = []
    for name in LOCOMOTION_METRIC_FIELDS:
        text = (row.get(f"{GAIT_COLUMN_PREFIX}{name}") or "").strip()
        if text:
            gait.append((name, _cell_float(text, column=f"{GAIT_COLUMN_PREFIX}{name}", where=where)))
    values["measurement"] = measurement.as_dict()
    values["gait_metrics"] = dict(gait)
    # Through the record's own strict reader, so a row and a JSON record are refused for the same reasons.
    record = {key: (None if isinstance(value, float) and math.isnan(value) else value) for key, value in values.items()}
    return CommandTrackingEpisode.from_dict(record)


def read_command_tracking_panel(csv_path: "str | Path") -> CommandTrackingPanelEvidence:
    """Read a ``command_tracking_panel_selected.csv`` back into the gate's inputs, strictly.

    Publication re-derives the verdict from what this returns, so a file
    with no rows, a missing column of :data:`COMMAND_TRACKING_PANEL_FIELDNAMES`,
    a row whose role is neither ``policy`` nor ``reference``, an unparsable
    cell, a reference row that records a classification, no policy row, or a
    stamp that differs between rows is a ``ValueError``.  An empty gait cell
    is a metric the episode was not measured on; ``nan`` is an unmeasured
    value.
    """
    path = Path(csv_path)
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    missing = [column for column in COMMAND_TRACKING_PANEL_FIELDNAMES if column not in columns]
    if missing:
        raise ValueError(f"{path} is not a command_tracking/v1 panel: it lacks the column(s) {missing}")
    if not rows:
        raise ValueError(f"{path} holds no episode rows")
    stamps = {column: _single_stamp(rows, column, path=path) for column in STAMP_COLUMNS}
    tolerances = {
        name: _cell_float(stamps[f"tolerance_{name}"], column=f"tolerance_{name}", where=str(path))
        for name in TOLERANCE_NAMES
    }
    measurement = PanelMeasurement.from_dict(
        {
            "protocol": stamps["protocol"],
            "gait_measurement": stamps["gait_measurement"],
            "yaw_rate_max": _cell_float(stamps["yaw_rate_max"], column="yaw_rate_max", where=str(path)),
            "tolerances": tolerances,
        }
    )
    policy: list[CommandTrackingEpisode] = []
    success: list[bool] = []
    reference: list[CommandTrackingEpisode] = []
    for index, row in enumerate(rows):
        where = f"{path} row {index}"
        role = (row.get("role") or "").strip()
        if role not in (POLICY_ROLE, REFERENCE_ROLE):
            raise ValueError(f"{where} has role {role!r}, not {POLICY_ROLE!r} or {REFERENCE_ROLE!r}")
        try:
            episode = _episode_from_row(row, measurement, where=where)
        except ValueError as exc:
            raise ValueError(f"{where}: {exc}") from exc
        if role == POLICY_ROLE:
            policy.append(episode)
            success.append(_cell_bool(row.get("success"), column="success", where=where))
        else:
            if (row.get("success") or "").strip() or (row.get("reasons") or "").strip():
                raise ValueError(f"{where} is a reference row with a classification; the reference is never classified")
            reference.append(episode)
    if not policy:
        raise ValueError(f"{path} holds no policy row")
    profile = stamps["profile"]
    if profile is None:
        raise ValueError(f"{path} stamps no profile")
    return CommandTrackingPanelEvidence(
        policy=tuple(policy),
        policy_success=tuple(success),
        reference=tuple(reference),
        profile=profile,
        measurement=measurement,
        checkpoint_sha256=stamps["checkpoint_sha256"],
        normalization_sha256=stamps["normalization_sha256"],
        reference_checkpoint_sha256=stamps["reference_checkpoint_sha256"],
        reference_normalization_sha256=stamps["reference_normalization_sha256"],
        task_sha256=stamps["task_sha256"],
        gate_sha256=stamps["gate_sha256"],
        seed_start=_cell_whole(stamps["seed_start"], column="seed_start", where=str(path)),
        horizon=_cell_whole(stamps["horizon"], column="horizon", where=str(path)),
        control_dt=_cell_float(stamps["control_dt"], column="control_dt", where=str(path)),
        policy_deterministic=_cell_bool(stamps["policy_deterministic"], column="policy_deterministic", where=str(path)),
    )
