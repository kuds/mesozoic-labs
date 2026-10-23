"""A stage's training history read back from its ledgers, drawn as pages of graphs (report only).

A stage directory holds one row per evaluation episode (``eval_episodes.csv``)
and one per training episode (``train_episodes.csv``), beside
``stage_config.json``, ``progress.json``, ``plant_identity.json`` and the
checkpoints in ``models/`` (``docs/TRAINING_ARTIFACT_PRESENTATION.md`` defines
every column).  This module reduces them per evaluation and draws them against
the training step, for one run or overlaid for several:

- the outcome counts: episodes, falls, clean episodes (``screen_v2_clean``),
  saturation-flagged episodes (``screen_v2_reasons`` naming
  ``max_actuator_saturation_fraction``) and the mean return;
- the rails' own per-episode numbers (``max_sole_cop_fore_aft``,
  ``max_foot_cop_fore_aft``, ``max_sole_tilt_deg``,
  ``max_sole_corner_lift_m``) and, per rail the stage declares a bar for, the
  count and fraction of full-horizon episodes over it
  (``<rail>_over_bar``, ``<rail>_over_bar_fraction``);
- per foot, the signed whole-foot and pad (sole) centres of pressure
  fore-aft, the digit share, the toe moment margin, the shear and the
  friction-cone use, and the count of full-horizon episodes with either
  foot's window-mean CoP outside the reward band's edges;
- the whole-body balance (``com_fore_aft_mean_m``, ``root_pitch_mean_deg``,
  the XcoM's toe and heel margin minima) and ``actuator_force_frac_max``;
- the achieved left-minus-right hip pitch, knee and ankle offsets
  (``leg_q_<left>_minus_<right>_<joint>_deg``; ``eval_episodes.csv`` records no
  hip roll difference);
- the training falls per window of ``train_episodes.csv``;
- the per-actuator policy standard deviation ``exp(log_std)`` at the
  checkpoint zips in ``models/``.

Every per-episode value is reduced over the evaluation's full-horizon episodes,
the set ``gate_progress.npz`` averages over; the outcome counts are over all its
episodes.  A missing file or column is an absent series, listed in the JSON
summary (``mesozoic.training-history/v1``), never an error.

An evaluation is *flagged* when at least :data:`SATURATION_FLAG_EPISODES` of
its episodes list ``max_actuator_saturation_fraction`` among their failed
screen bars, and :data:`SATURATION_ALARM_IN_A_ROW` flagged evaluations in a
row are an *alarm* (the stance4 and stance5 scorers' report-only rule, set on
evaluations of :data:`SATURATION_RULE_EPISODES` episodes).

Counts are never smoothed.  A comparison draws them as one strip per run (a
cell per evaluation, coloured by the count) and smooths only the continuous
series, with each evaluation's value drawn faintly behind.

Report only: it reads the stage directories and writes only the files its
caller names (``scripts/plot_training_history.py`` refuses an output inside a
run directory).  Nothing reads its output back.  From a checkpoint zip it reads
only the ``log_std`` tensor of ``policy.pth`` (``torch.load(...,
weights_only=True)``) and, for a named checkpoint, ``num_timesteps`` from the
JSON ``data`` member (:func:`~environments.shared.task_fingerprint.read_checkpoint_attribute`);
it unpickles nothing.  Figures are drawn on matplotlib ``Figure`` objects
without pyplot, so no figure registry or interactive backend is touched.
"""

from __future__ import annotations

import base64
import csv
import html
import io
import logging
import math
import os
import re
import tempfile
import textwrap
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..file_io import atomic_write_json, atomic_write_text, read_json_object
from ..model_id import ModelId

logger = logging.getLogger(__name__)

#: The JSON summary's schema string.
SCHEMA = "mesozoic.training-history/v1"
EVAL_EPISODES_FILENAME = "eval_episodes.csv"
TRAIN_EPISODES_FILENAME = "train_episodes.csv"
PROGRESS_FILENAME = "progress.json"
STAGE_CONFIG_FILENAME = "stage_config.json"
PLANT_IDENTITY_FILENAME = "plant_identity.json"
MODELS_DIRNAME = "models"
#: The files a report writes into its output directory.
HTML_FILENAME = "training_history.html"
JSON_FILENAME = "training_history.json"

#: The screen bar whose failures are counted as saturation flags.
SATURATION_REASON = "max_actuator_saturation_fraction"
#: An evaluation is flagged when at least this many of its episodes fail :data:`SATURATION_REASON`.
SATURATION_FLAG_EPISODES = 5
#: This many flagged evaluations in a row are an alarm, at the last of them.
SATURATION_ALARM_IN_A_ROW = 2
#: The evaluation size the flag rule was set on (the stance4 and stance5 ledgers' 10 episodes).
SATURATION_RULE_EPISODES = 10
#: The training-fall window: ``progress.json``'s ``training_rollouts.window_timesteps``.
TRAINING_WINDOW_TIMESTEPS = 262_144
#: The per-actuator std reads: at most this many periodic checkpoints (evenly spaced by step), within this budget.
POLICY_STD_MAX_PERIODIC = 12
POLICY_STD_BUDGET_S = 10.0
#: Two periodic checkpoints are joined by a solid line when their gap is at most this many times the run's spacing.
POLICY_STD_JOIN_GAP = 1.5

#: Per-foot columns of ``eval_episodes.csv`` (``<foot>_<suffix>``) reduced per evaluation.
FOOT_SUFFIXES: tuple[str, ...] = (
    "foot_cop_fore_aft",
    "sole_cop_fore_aft",
    "digit_share",
    "toe_margin_nm",
    "sole_tilt_deg",
    "sole_corner_lift_m",
    "shear_over_normal",
    "cone_use_mean",
    "cone_use_max",
)
#: The rails' per-episode numbers: each episode's maximum over the feet (the gate's own statistics).
RAIL_COLUMNS: tuple[str, ...] = (
    "max_sole_cop_fore_aft",
    "max_foot_cop_fore_aft",
    "max_sole_tilt_deg",
    "max_sole_corner_lift_m",
)
#: Whole-body columns reduced per evaluation.
BODY_COLUMNS: tuple[str, ...] = (
    *RAIL_COLUMNS,
    "com_fore_aft_mean_m",
    "root_pitch_mean_deg",
    "xcom_toe_margin_min_m",
    "xcom_heel_margin_min_m",
    "actuator_force_frac_max",
)
#: The sagittal joints whose left-minus-right achieved offset ``eval_episodes.csv`` records.
LR_JOINTS: tuple[str, ...] = ("hip_pitch", "knee", "ankle")
_LR_COLUMN = re.compile(
    r"^leg_q_(?P<left>[A-Za-z0-9]+)_minus_(?P<right>[A-Za-z0-9]+)_(?P<joint>hip_pitch|knee|ankle)_deg$"
)
_FOOT_COLUMN = re.compile(r"^(?P<foot>[A-Za-z0-9]+)_(?:foot|sole)_cop_fore_aft$")
#: The declared bars recorded and drawn, from ``stage_config.json``'s ``curriculum``.
BAR_KEYS: tuple[str, ...] = (
    "max_sole_cop_fore_aft",
    "max_foot_cop_fore_aft",
    "max_sole_tilt_deg",
    "max_sole_corner_lift_m",
    "max_actuator_saturation_fraction",
    "min_clean_stance_lcb",
)
#: The fore-aft CoP band's reward settings (``reward_weights``), recorded when present.
BAND_PREFIX = "stance_cop_band_"
#: The band edges counted against: ``(count key, foot column suffix, lower key, upper key)``.
BAND_COUNTS: tuple[tuple[str, str, str, str], ...] = (
    ("foot_cop_outside_band", "foot_cop_fore_aft", "lower", "upper"),
    ("sole_cop_outside_pad_band", "sole_cop_fore_aft", "pad_lower", "pad_upper"),
)
#: Actuators drawn on the std pages' first panel: each name stem containing one of these.
KEY_ACTUATORS: tuple[str, ...] = ("hip_pitch", "hip_roll", "knee", "ankle", "tail_3")
#: What the ledgers do not hold, said on every page set.
NOT_RECORDED: tuple[str, ...] = (
    "Hip roll: eval_episodes.csv records the left-minus-right achieved offset of hip pitch, knee and ankle only "
    "(leg_q_<left>_minus_<right>_<joint>_deg); there is no hip roll difference to draw.",
    "The evaluation episodes are the deterministic policy's, at yaw 0 on the evaluation env's own resets. The "
    "judge's perturbation panels (its joint-noise falls, K3) and its verdicts are not in these files. The training "
    "falls are the stochastic training policy's.",
    "The actuator saturation fraction itself is not recorded per episode, and screen_v2_reasons does not name the "
    "saturating actuator: an episode only lists max_actuator_saturation_fraction when it exceeded the stage's bar.",
    "The CoP band is paid per window step on each loaded foot's CoP; these files keep each episode's window mean per "
    "foot only, so the outside-band count reads that mean against the band's edges.",
    "The commanded leg offsets (leg_dc_*) and the per-foot heel moment margin (<foot>_heel_margin_nm) are kept in "
    "gate_progress.npz only, not in eval_episodes.csv.",
    "Per-actuator std is read at the checkpoints kept in models/ only; between them nothing is recorded here "
    "(policy_probe.npz holds it per update where the run kept that file).",
)

# The categorical palette (fixed order, never cycled), chart ink and status red.
_SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_AXIS = "#c3c2b7"
_SURFACE = "#fcfcfb"
_CRITICAL = "#d03b3b"
_HANDOFF = "#008300"
#: The palette less the handoff green, for lines drawn beside the handoff mark.
_OTHER_SERIES = tuple(colour for colour in _SERIES if colour != _HANDOFF)
_ZERO_CELL = "#ecebe6"
_GROUP_STYLES = ("-", "--", ":", "-.")
#: Sequential ramps for the count strips: blue (the clean count) and orange (every problem count), light to dark.
_RAMPS = {
    "blue": ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"),
    "orange": ("#fde4d6", "#f9c3a6", "#f4a079", "#eb6834", "#c94f1d", "#9e3c14", "#70290c"),
}
#: The std page's colour per key actuator (blue, orange, aqua, yellow, violet).
_KEY_COLOURS = dict(zip(KEY_ACTUATORS, (_SERIES[0], _SERIES[1], _SERIES[2], _SERIES[3], _SERIES[6])))
#: At most this many runs on one comparison page (one categorical hue each).
MAX_RUNS = len(_SERIES)
#: The PNG resolution: 96 dpi draws a point size as a browser does (10.5 pt = 14 px), and the HTML shows each page
#: at its native width, never upscaled; it keeps a dozen reports of ten pages under about 30 MB.
DEFAULT_DPI = 96
#: A training-fall strip's colour scale tops out here (a window at or above it is the darkest cell).
TRAINING_FALL_CAP = 0.10


def _number(text: Any) -> float:
    """A finite float from a CSV cell or JSON value, else NaN."""
    if isinstance(text, bool):
        return math.nan
    try:
        value = float(text)
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]] | None:
    """``(header, rows)`` of a CSV file, or ``None`` when it is absent or unreadable."""
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = [row for row in reader]
            return list(reader.fieldnames or []), rows
    except (OSError, UnicodeDecodeError, csv.Error):
        return None


def reason_keys(reasons: str | None) -> set[str]:
    """The bar names a ``screen_v2_reasons`` cell lists (``;``-joined, each optionally ``name: detail``)."""
    return {part.split(":", 1)[0].strip() for part in (reasons or "").split(";") if part.strip()}


def _stats(values: Sequence[float]) -> dict[str, float]:
    """Mean, standard deviation (ddof 0), minimum, maximum and count of the finite *values*."""
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if not array.size:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan, "n": 0.0}
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "n": float(array.size),
    }


def foot_labels(header: Sequence[str]) -> list[str]:
    """The foot labels the header measures a centre of pressure for, in header order (``["r", "l"]`` on the T. rex)."""
    feet: list[str] = []
    for column in header:
        match = _FOOT_COLUMN.match(column)
        if match is not None and match.group("foot") != "max" and match.group("foot") not in feet:
            feet.append(match.group("foot"))
    return feet


def lr_columns(header: Sequence[str]) -> dict[str, str]:
    """``{joint: column}`` of the left-minus-right achieved offsets the header holds."""
    found: dict[str, str] = {}
    for column in header:
        match = _LR_COLUMN.match(column)
        if match is not None and match.group("joint") not in found:
            found[match.group("joint")] = column
    return {joint: found[joint] for joint in LR_JOINTS if joint in found}


def reduced_columns(header: Sequence[str]) -> list[str]:
    """The per-episode columns of the header that are reduced per evaluation, in a fixed order."""
    present = set(header)
    columns = [f"{foot}_{suffix}" for foot in foot_labels(header) for suffix in FOOT_SUFFIXES]
    columns += list(BODY_COLUMNS) + list(lr_columns(header).values())
    return [column for column in columns if column in present]


def expected_columns(feet: Sequence[str]) -> list[str]:
    """The columns this report reads from ``eval_episodes.csv`` (per foot for *feet*)."""
    columns = ["timesteps", "return", "full_horizon", "cause", "screen_v2_clean", "screen_v2_reasons"]
    columns += [f"{foot}_{suffix}" for foot in feet for suffix in FOOT_SUFFIXES]
    columns += list(BODY_COLUMNS)
    return columns


def _full_horizon_rows(
    header: Sequence[str], rows: Sequence[Mapping[str, str]]
) -> tuple[list[int], list[list[Mapping[str, str]]], list[list[Mapping[str, str]]], list[float], bool]:
    """Per evaluation step: its rows, its full-horizon rows and its fall count; and whether falls are known.

    An episode's full horizon comes from ``full_horizon``, else from ``cause``
    (``truncated``); with neither, every row is kept as the reduced set and the
    fall count is NaN.
    """
    present = set(header)
    by_step: dict[int, list[Mapping[str, str]]] = {}
    for row in rows:
        step = _number(row.get("timesteps"))
        if math.isfinite(step):
            by_step.setdefault(int(step), []).append(row)
    steps = sorted(by_step)

    def full_horizon(row: Mapping[str, str]) -> bool | None:
        if "full_horizon" in present:
            value = _number(row.get("full_horizon"))
            if math.isfinite(value):
                return value == 1.0
        if "cause" in present and (row.get("cause") or "").strip():
            return (row.get("cause") or "").strip() == "truncated"
        return None

    groups, fh_rows, falls = [], [], []
    for step in steps:
        group = by_step[step]
        flags = [full_horizon(row) for row in group]
        known = [flag for flag in flags if flag is not None]
        groups.append(group)
        falls.append(float(sum(1 for flag in known if not flag)) if known else math.nan)
        fh_rows.append([row for row, flag in zip(group, flags) if flag] if known else list(group))
    return steps, groups, fh_rows, falls, "full_horizon" in present or "cause" in present


def _over_bar(fh_rows: Sequence[Sequence[Mapping[str, str]]], column: str, bar: float) -> tuple[np.ndarray, ...]:
    """Per evaluation: the full-horizon episodes with *column* above *bar*, and those that measured it."""
    over, measured = [], []
    for group in fh_rows:
        values = [_number(row.get(column)) for row in group]
        finite = [value for value in values if math.isfinite(value)]
        over.append(float(sum(1 for value in finite if value > bar)))
        measured.append(float(len(finite)))
    over_array, measured_array = np.asarray(over, dtype=float), np.asarray(measured, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = np.where(measured_array > 0, over_array / np.where(measured_array > 0, measured_array, 1.0), np.nan)
    return over_array, measured_array, fraction


def reduce_evaluations(
    header: Sequence[str],
    rows: Sequence[Mapping[str, str]],
    *,
    bars: Mapping[str, float | None] | None = None,
) -> dict[str, Any]:
    """Per evaluation (one ``timesteps`` value) of ``eval_episodes.csv``'s rows.

    Returns ``timesteps`` (int), the counts ``episodes``, ``full_horizon``,
    ``falls`` (episodes shorter than the horizon), ``clean``
    (``screen_v2_clean`` 1 over the episodes the screen judged), ``saturated``
    (judged episodes listing :data:`SATURATION_REASON`) and ``mean_return`` as
    arrays, NaN where the evaluation could not supply them; per rail of
    :data:`RAIL_COLUMNS` whose bar *bars* declares, ``<rail>_over_bar`` (the
    full-horizon episodes whose own value exceeds the bar) and
    ``<rail>_over_bar_fraction`` (over those that measured it); and
    ``columns``: per reduced column its ``mean``, ``std``, ``min``, ``max``
    and ``n`` over the evaluation's full-horizon episodes.  A count whose
    column is absent is left out.
    """
    present = set(header)
    steps, groups, fh_rows, falls, fh_known = _full_horizon_rows(header, rows)
    out: dict[str, Any] = {"timesteps": np.asarray(steps, dtype=np.int64), "columns": {}}
    if not steps:
        return out
    episodes, clean, saturated, mean_return = [], [], [], []
    for group in groups:
        episodes.append(float(len(group)))
        judged = group
        if "screen_v2_clean" in present:
            values = [_number(row.get("screen_v2_clean")) for row in group]
            judged = [row for row, value in zip(group, values) if math.isfinite(value)]
            clean.append(float(sum(1 for value in values if value == 1.0)) if judged else math.nan)
        if "screen_v2_reasons" in present:
            saturated.append(
                float(sum(1 for row in judged if SATURATION_REASON in reason_keys(row.get("screen_v2_reasons"))))
                if judged
                else math.nan
            )
        if "return" in present:
            mean_return.append(_stats([_number(row.get("return")) for row in group])["mean"])
    out["episodes"] = np.asarray(episodes, dtype=float)
    if fh_known:
        out["falls"] = np.asarray(falls, dtype=float)
        out["full_horizon"] = np.asarray([float(len(group)) for group in fh_rows], dtype=float)
    if "screen_v2_clean" in present:
        out["clean"] = np.asarray(clean, dtype=float)
    if "screen_v2_reasons" in present:
        out["saturated"] = np.asarray(saturated, dtype=float)
    if "return" in present:
        out["mean_return"] = np.asarray(mean_return, dtype=float)
    for rail in RAIL_COLUMNS:
        bar = (bars or {}).get(rail)
        if rail in present and bar is not None and math.isfinite(bar):
            over, _, fraction = _over_bar(fh_rows, rail, float(bar))
            out[f"{rail}_over_bar"] = over
            out[f"{rail}_over_bar_fraction"] = fraction
    for column in reduced_columns(header):
        stats = [_stats([_number(row.get(column)) for row in group]) for group in fh_rows]
        out["columns"][column] = {key: np.asarray([s[key] for s in stats], dtype=float) for key in stats[0]}
    return out


def band_counts(
    fh_rows: Sequence[Sequence[Mapping[str, str]]], feet: Sequence[str], edges: Mapping[str, float]
) -> dict[str, np.ndarray]:
    """Per evaluation, the full-horizon episodes with either foot's window-mean CoP outside the band's edges.

    ``foot_cop_outside_band`` reads ``<foot>_foot_cop_fore_aft`` against
    ``edges["lower"]``/``edges["upper"]``, ``sole_cop_outside_pad_band``
    ``<foot>_sole_cop_fore_aft`` against ``pad_lower``/``pad_upper``.  A foot
    with no value (unloaded over the window) is not outside.  A count is left
    out when its edges are not in *edges*.
    """
    out: dict[str, np.ndarray] = {}
    for key, suffix, low_key, high_key in BAND_COUNTS:
        low, high = edges.get(low_key, math.nan), edges.get(high_key, math.nan)
        if not (math.isfinite(low) and math.isfinite(high)) or not feet:
            continue
        counts = []
        for group in fh_rows:
            outside = 0
            for row in group:
                values = [_number(row.get(f"{foot}_{suffix}")) for foot in feet]
                outside += any(math.isfinite(value) and not low <= value <= high for value in values)
            counts.append(float(outside))
        out[key] = np.asarray(counts, dtype=float)
    return out


def reduce_training(
    header: Sequence[str], rows: Sequence[Mapping[str, str]], *, window: int = TRAINING_WINDOW_TIMESTEPS
) -> dict[str, Any] | None:
    """Training episodes and falls per window of ``train_episodes.csv``, or ``None`` without its columns.

    Window *k* holds the episodes that ended at a step in ``(k * window, (k + 1) * window]``;
    ``window_end`` is its upper edge, except the last window's, which is its
    last episode's step (the run stopped inside it).  A fall is an episode that
    did not reach the time limit (``cause`` not ``truncated``), as the ledger's
    own summaries count it.
    """
    if "timesteps" not in header or "cause" not in header:
        return None
    counts: dict[int, list[int]] = {}
    last = 0
    for row in rows:
        step = _number(row.get("timesteps"))
        cause = (row.get("cause") or "").strip()
        if not math.isfinite(step) or not cause:
            continue
        index = max(0, int(math.ceil(step / window)) - 1)
        entry = counts.setdefault(index, [0, 0])
        entry[0] += 1
        entry[1] += int(cause != "truncated")
        last = max(last, int(step))
    if not counts:
        return None
    indices = range(min(counts), max(counts) + 1)
    episodes = np.asarray([counts.get(k, [0, 0])[0] for k in indices], dtype=float)
    falls = np.asarray([counts.get(k, [0, 0])[1] for k in indices], dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = np.where(episodes > 0, falls / np.where(episodes > 0, episodes, 1.0), np.nan)
    ends = np.asarray([(k + 1) * window for k in indices], dtype=np.int64)
    ends[-1] = min(int(ends[-1]), last)
    return {
        "window_timesteps": int(window),
        "window_end": ends,
        "last_episode_step": int(last),
        "episodes": episodes,
        "falls": falls,
        "fall_fraction": fraction,
    }


def saturation_alarms(
    timesteps: Sequence[int],
    saturated: Sequence[float],
    *,
    flag_episodes: int = SATURATION_FLAG_EPISODES,
    in_a_row: int = SATURATION_ALARM_IN_A_ROW,
    episodes: Sequence[float] | None = None,
) -> dict[str, Any]:
    """The saturation flags and alarms of one run's evaluations (report only).

    An evaluation is flagged when at least *flag_episodes* of its episodes list
    :data:`SATURATION_REASON`; a run of at least *in_a_row* flagged evaluations
    in a row (adjacent evaluations of the ledger) is an alarm, raised at its
    *in_a_row*-th evaluation.  An evaluation with an unknown count (NaN) is not
    flagged and ends a run.  The same rule as the stance4 and stance5
    scorers' ``saturation_alarm``, which was set on evaluations of
    :data:`SATURATION_RULE_EPISODES` episodes: with *episodes* given, the
    result lists the evaluation sizes seen and says when any differs.
    """
    evaluations = [(int(step), float(count)) for step, count in zip(timesteps, saturated)]
    flagged = [step for step, count in evaluations if math.isfinite(count) and count >= flag_episodes]
    runs: list[dict[str, int]] = []
    current: list[int] = []
    for step, count in [*evaluations, (None, math.nan)]:
        if step is not None and math.isfinite(count) and count >= flag_episodes:
            current.append(step)
            continue
        if len(current) >= in_a_row:
            runs.append(
                {"from": current[0], "to": current[-1], "evaluations": len(current), "alarm_at": current[in_a_row - 1]}
            )
        current = []
    first_any = next((step for step, count in evaluations if math.isfinite(count) and count > 0), None)
    result: dict[str, Any] = {
        "rule": (
            f"an evaluation is flagged when >= {flag_episodes} of its episodes list {SATURATION_REASON} in "
            f"screen_v2_reasons (the scorers' rule, set on {SATURATION_RULE_EPISODES}-episode evaluations); "
            f"{in_a_row} flagged evaluations in a row (adjacent evaluations) are an alarm, raised at evaluation "
            f"{in_a_row} of the run"
        ),
        "flag_episodes": int(flag_episodes),
        "in_a_row": int(in_a_row),
        "evaluations": len(evaluations),
        "first_saturated_episode_at": first_any,
        "first_flagged_at": flagged[0] if flagged else None,
        "flagged": flagged,
        "alarm": bool(runs),
        "runs": runs,
        "first_alarm_at": runs[0]["alarm_at"] if runs else None,
    }
    if episodes is not None:
        sizes = sorted({int(n) for n in episodes if math.isfinite(n)})
        result["episodes_per_evaluation"] = sizes
        if any(size != SATURATION_RULE_EPISODES for size in sizes):
            result["note"] = (
                f"the rule was set on {SATURATION_RULE_EPISODES}-episode evaluations; these hold "
                f"{', '.join(str(size) for size in sizes)}, and the >= {flag_episodes} threshold is applied unchanged"
            )
    return result


def checkpoint_kind(path: Path) -> tuple[str, int | None] | None:
    """``(kind, step from the name)`` of a checkpoint zip, or ``None`` for any other file.

    ``robust_best_model.zip`` and ``best_model.zip`` are ``robust_best`` and
    ``best``; ``<label>_final.zip`` is ``final``; a periodic checkpoint
    ``<label>_<steps>_steps.zip`` is ``periodic`` with its step.
    """
    from ..policy_loading import _PERIODIC_CHECKPOINT_RE

    if path.suffix != ".zip":
        return None
    stem = path.stem
    if stem == "robust_best_model":
        return "robust_best", None
    if stem == "best_model":
        return "best", None
    if stem.endswith("_final"):
        return "final", None
    match = _PERIODIC_CHECKPOINT_RE.fullmatch(stem)
    if match is not None:
        return "periodic", int(match.group(2))
    return None


def read_log_std(path: Path) -> np.ndarray | None:
    """``exp(log_std)`` of a checkpoint zip's ``policy.pth``, or ``None`` when its state dict has no ``log_std``.

    Only ``policy.pth`` is read, with ``torch.load(..., weights_only=True)``,
    which refuses any pickled object but tensors and plain containers.
    """
    import torch

    with zipfile.ZipFile(path) as archive:
        payload = archive.read("policy.pth")
    state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    log_std = state.get("log_std") if isinstance(state, Mapping) else None
    if log_std is None:
        return None
    std: np.ndarray = np.exp(np.asarray(log_std.detach().cpu().numpy(), dtype=float).reshape(-1))
    return std


def _spaced(items: Sequence[Any], count: int) -> list[Any]:
    """At most *count* of *items*, evenly spaced, the first and last included."""
    if len(items) <= count:
        return list(items)
    picks = sorted({int(round(i)) for i in np.linspace(0, len(items) - 1, count)})
    return [items[i] for i in picks]


def list_checkpoints(models_dir: Path, *, progress: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Every checkpoint zip kept in *models_dir*: ``{"name", "kind", "timesteps", "step_source"}``, by step.

    A periodic checkpoint's step is its name's; a named one's is
    ``num_timesteps`` from the zip's JSON ``data`` member, else
    ``progress.json``'s ``checkpoints`` record, else ``None``.
    """
    from ..task_fingerprint import read_checkpoint_attribute

    if not models_dir.is_dir():
        return []
    recorded = (progress or {}).get("checkpoints") if isinstance(progress, Mapping) else None
    recorded = recorded if isinstance(recorded, Mapping) else {}
    found: list[dict[str, Any]] = []
    for path in sorted(models_dir.iterdir()):
        kind = checkpoint_kind(path) if path.is_file() else None
        if kind is None:
            continue
        step, source = kind[1], "file name"
        if kind[0] != "periodic":
            step = read_checkpoint_attribute(path, "num_timesteps")
            source = "zip data (JSON num_timesteps)"
            if isinstance(step, bool) or not isinstance(step, int):
                fallback = recorded.get(f"{kind[0]}_model_timesteps") if kind[0] != "final" else None
                step = fallback if isinstance(fallback, int) and not isinstance(fallback, bool) else None
                source = "progress.json checkpoints" if step is not None else "unknown"
        found.append({"name": path.name, "kind": kind[0], "timesteps": step, "step_source": source, "path": path})
    found.sort(key=lambda c: (c["timesteps"] is None, c["timesteps"] or 0, c["name"]))
    return found


def read_policy_std(
    models_dir: Path,
    *,
    progress: Mapping[str, Any] | None = None,
    max_periodic: int = POLICY_STD_MAX_PERIODIC,
    budget_s: float = POLICY_STD_BUDGET_S,
) -> dict[str, Any]:
    """The per-actuator policy std at the checkpoints in *models_dir* (design section 3.6).

    The named checkpoints are read first (``robust_best``, ``best``, ``final``),
    then at most *max_periodic* periodic ones evenly spaced by step.  The reads
    stop after *budget_s* seconds of wall clock.  Steps as
    :func:`list_checkpoints` gives them.  ``available`` counts the checkpoints
    chosen, ``read`` those read.
    """
    result: dict[str, Any] = {"available": 0, "read": 0, "checkpoints": [], "skipped": None, "errors": []}
    if not models_dir.is_dir():
        result["skipped"] = f"no {MODELS_DIRNAME}/ directory"
        return result
    kept = list_checkpoints(models_dir, progress=progress)
    order = {"robust_best": 0, "best": 1, "final": 2}
    named = sorted((c for c in kept if c["kind"] != "periodic"), key=lambda c: (order[c["kind"]], c["name"]))
    periodic = [c for c in kept if c["kind"] == "periodic"]
    chosen = named + _spaced(periodic, max_periodic)
    result["available"] = len(chosen)
    if not chosen:
        result["skipped"] = f"no checkpoint zip in {MODELS_DIRNAME}/"
        return result
    try:
        import torch  # noqa: F401
    except ImportError:
        result["skipped"] = "torch is not importable"
        return result
    started = time.monotonic()
    for entry in chosen:
        if time.monotonic() - started > budget_s:
            result["skipped"] = f"stopped after {budget_s:g} s of reads"
            break
        try:
            std = read_log_std(entry["path"])
        except Exception as exc:  # noqa: BLE001 - a bad zip costs its own point only
            result["errors"].append(f"{entry['name']}: {type(exc).__name__}: {exc}")
            continue
        if std is None:
            result["errors"].append(f"{entry['name']}: policy.pth has no log_std (not a Gaussian PPO policy)")
            continue
        point = {key: value for key, value in entry.items() if key != "path"}
        point["std"] = std
        result["checkpoints"].append(point)
        result["read"] += 1
    result["checkpoints"].sort(key=lambda c: (c["timesteps"] is None, c["timesteps"] or 0, c["name"]))
    return result


def plant_check(stage: Path, model_id: ModelId | None, record: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Whether the checkout's physics is the run's: the stage's ``plant_identity.json`` against the manifest.

    Reads ``physics_sha256`` and ``physics_revision`` from the stage's
    ``plant_identity.json`` (else ``stage_config.json``'s ``plant_identity``)
    and the checkout's ``configs/plant_manifest.generated.json`` entry for
    *model_id*.  ``status`` is ``match``, ``differs`` or ``unchecked`` (with
    ``why``).
    """
    from ..paths import REPOSITORY_ROOT

    identity = read_json_object(stage / PLANT_IDENTITY_FILENAME)
    source = PLANT_IDENTITY_FILENAME
    recorded = (record or {}).get("plant_identity")
    if identity is None and isinstance(recorded, Mapping):
        identity, source = dict(recorded), f"{STAGE_CONFIG_FILENAME} plant_identity"
    if identity is None:
        return {"status": "unchecked", "why": f"the stage records no {PLANT_IDENTITY_FILENAME}"}
    if model_id is None:
        return {"status": "unchecked", "why": "no model recorded"}
    manifest = read_json_object(REPOSITORY_ROOT / "configs" / "plant_manifest.generated.json") or {}
    entry = ((manifest.get("plants") or {}).get(model_id.species) or {}).get(model_id.embodiment)
    physics = entry.get("physics") if isinstance(entry, Mapping) else None
    if not isinstance(physics, Mapping):
        return {"status": "unchecked", "why": "the checkout's plant manifest has no entry for this model"}
    run = {"physics_sha256": identity.get("physics_sha256"), "physics_revision": identity.get("physics_revision")}
    checkout = {"physics_sha256": physics.get("sha256"), "physics_revision": physics.get("revision")}
    status = "match" if run["physics_sha256"] and run["physics_sha256"] == checkout["physics_sha256"] else "differs"
    return {"status": status, "source": source, "run": run, "checkout": checkout}


def actuator_names(
    model_id: ModelId | None, count: int, *, plant: Mapping[str, Any] | None = None
) -> tuple[list[str] | None, str]:
    """The env's actuator names in action order, from the checkout's MJCF of *model_id*, and where they came from.

    ``(None, why)`` when the model is unknown, MuJoCo is not importable, the
    MJCF does not compile, its actuator count is not *count*, or *plant*
    (:func:`plant_check`) says the checkout's physics differs from the run's:
    the actuators are then labelled by index.
    """
    if model_id is None:
        return None, "no model recorded"
    if plant is not None and plant.get("status") == "differs":
        run, checkout = plant.get("run") or {}, plant.get("checkout") or {}
        return None, (
            f"the checkout's physics (revision {checkout.get('physics_revision')}) differs from the run's "
            f"{plant.get('source', PLANT_IDENTITY_FILENAME)} (revision {run.get('physics_revision')}); actuators "
            "are labelled by index"
        )
    try:
        import mujoco

        from ..paths import ENVIRONMENTS_ROOT

        path = model_id.asset_path(ENVIRONMENTS_ROOT)
        model = mujoco.MjModel.from_xml_path(str(path))
    except Exception as exc:  # noqa: BLE001 - names are a label only
        return None, f"no actuator names ({type(exc).__name__}: {exc})"
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"actuator[{i}]" for i in range(model.nu)]
    if len(names) != count:
        return None, f"the checkout's MJCF has {len(names)} actuators, the policy {count}"
    source = f"the checkout's MJCF ({path.relative_to(ENVIRONMENTS_ROOT.parent).as_posix()}), in actuator order"
    if plant is None or plant.get("status") != "match":
        why = (plant or {}).get("why", "no plant identity checked")
        return names, f"{source}; unchecked against the run's plant ({why})"
    return names, f"{source}; its physics sha256 matches the run's {plant.get('source', PLANT_IDENTITY_FILENAME)}"


def _bars(record: Mapping[str, Any]) -> dict[str, float | None]:
    curriculum = record.get("curriculum")
    curriculum = curriculum if isinstance(curriculum, Mapping) else {}
    return {key: (_number(curriculum.get(key)) if key in curriculum else None) for key in BAR_KEYS}


def _band(record: Mapping[str, Any]) -> dict[str, float] | None:
    for block in (record.get("reward_weights"), (record.get("task_fingerprint") or {}).get("env")):
        if isinstance(block, Mapping):
            band = {key: _number(value) for key, value in block.items() if str(key).startswith(BAND_PREFIX)}
            if band:
                return dict(sorted(band.items()))
    return None


def _edges(band: Mapping[str, float] | None) -> dict[str, float]:
    """The band's four edges (``lower``, ``upper``, ``pad_lower``, ``pad_upper``) that are finite."""
    edges = {
        name: (band or {}).get(f"{BAND_PREFIX}{name}", math.nan)
        for name in ("lower", "upper", "pad_lower", "pad_upper")
    }
    return {name: float(value) for name, value in edges.items() if math.isfinite(value)}


@dataclass
class RunHistory:
    """One stage directory's reduced training history."""

    stage_dir: Path
    label: str
    group: str | None = None
    files: dict[str, bool] = field(default_factory=dict)
    model_id: ModelId | None = None
    model_display: str | None = None
    seed: int | None = None
    run_id: str | None = None
    status: str | None = None
    progress_timesteps: int | None = None
    progress_written_at: str | None = None
    feet: list[str] = field(default_factory=list)
    lr: dict[str, str] = field(default_factory=dict)
    evaluations: dict[str, Any] = field(
        default_factory=lambda: {"timesteps": np.zeros(0, dtype=np.int64), "columns": {}}
    )
    training: dict[str, Any] | None = None
    saturation: dict[str, Any] = field(default_factory=dict)
    bars: dict[str, float | None] = field(default_factory=dict)
    band: dict[str, float] | None = None
    #: The edges the outside-band counts read: the run's own band, else a reference (``band_edges_source`` says).
    band_edges: dict[str, float] = field(default_factory=dict)
    band_edges_source: str | None = None
    checkpoint_steps: dict[str, int | None] = field(default_factory=dict)
    saved_checkpoints: list[dict[str, Any]] = field(default_factory=list)
    policy_std: dict[str, Any] = field(default_factory=dict)
    plant: dict[str, Any] = field(default_factory=dict)
    actuators: list[str] | None = None
    actuator_source: str = ""
    absent: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    #: Per evaluation, its full-horizon rows of eval_episodes.csv (kept for the band counts; never serialised).
    fh_rows: list[list[Mapping[str, str]]] = field(default_factory=list, repr=False)

    @property
    def band_active(self) -> bool:
        """True when the run's reward pays the fore-aft CoP band (``stance_cop_band_weight`` above 0)."""
        weight = (self.band or {}).get(f"{BAND_PREFIX}weight", math.nan)
        return math.isfinite(weight) and weight > 0.0

    @property
    def steps(self) -> np.ndarray:
        """The evaluation steps."""
        return np.asarray(self.evaluations.get("timesteps", np.zeros(0, dtype=np.int64)))

    @property
    def last_step(self) -> int:
        """The latest step this run's files show: an evaluation, a training episode, a checkpoint or progress.json."""
        candidates = [int(self.steps[-1])] if self.steps.size else []
        if self.training is not None:
            candidates.append(int(self.training["window_end"][-1]))
        candidates += [step for step in self.checkpoint_steps.values() if isinstance(step, int)]
        candidates += [c["timesteps"] for c in self.saved_checkpoints if isinstance(c.get("timesteps"), int)]
        if self.progress_timesteps is not None:
            candidates.append(self.progress_timesteps)
        return max(candidates) if candidates else 0

    @property
    def final_step(self) -> int | None:
        """The final checkpoint's step, else the step ``progress.json`` recorded once the stage ended."""
        if self.checkpoint_steps.get("final") is not None:
            return self.checkpoint_steps["final"]
        if self.status == "ended":
            return self.progress_timesteps
        return None

    @property
    def episodes_per_evaluation(self) -> list[int]:
        """The evaluation sizes seen, sorted."""
        episodes = self.series("episodes")
        return [] if episodes is None else sorted({int(n) for n in episodes if math.isfinite(n)})

    def column(self, column: str, stat: str = "mean") -> np.ndarray | None:
        """One reduced statistic of a column, or ``None`` when the column is absent."""
        entry = self.evaluations.get("columns", {}).get(column)
        return None if entry is None else entry[stat]

    def series(self, name: str) -> np.ndarray | None:
        """A per-evaluation count (``clean``, ``falls``, ...), or ``None`` when absent."""
        value = self.evaluations.get(name)
        return None if value is None else np.asarray(value, dtype=float)


def set_band_edges(run: RunHistory, edges: Mapping[str, float], source: str) -> None:
    """Count *run*'s full-horizon episodes outside *edges* (``lower``, ``upper``, ``pad_lower``, ``pad_upper``)."""
    for key, *_ in BAND_COUNTS:
        run.evaluations.pop(key, None)
    run.band_edges = {name: float(value) for name, value in edges.items() if math.isfinite(value)}
    run.band_edges_source = source if run.band_edges else None
    if run.band_edges and run.fh_rows:
        run.evaluations.update(band_counts(run.fh_rows, run.feet, run.band_edges))


def reference_band(runs: Sequence[RunHistory]) -> dict[str, float] | None:
    """The band edges every run that pays the band shares, or ``None`` (none pays it, or they differ)."""
    found = {tuple(sorted(_edges(run.band).items())) for run in runs if run.band_active and _edges(run.band)}
    return dict(found.pop()) if len(found) == 1 else None


def apply_reference_band(runs: Sequence[RunHistory], edges: Mapping[str, float] | None = None) -> None:
    """Give each run without band edges of its own the *edges* (default: :func:`reference_band`) to count against."""
    edges = dict(edges) if edges is not None else reference_band(runs)
    if not edges:
        return
    text = ", ".join(f"{name} {value:+g}" for name, value in edges.items())
    for run in runs:
        if not run.band_edges:
            set_band_edges(run, edges, f"reference edges ({text}); this run does not pay the band")


def load_run_history(
    stage_dir: "str | Path",
    *,
    label: str | None = None,
    group: str | None = None,
    read_std: bool = True,
    training_window: int = TRAINING_WINDOW_TIMESTEPS,
) -> RunHistory:
    """Read and reduce one stage directory's files; a missing file or column is recorded as absent, never raised."""
    stage = Path(stage_dir)
    run = RunHistory(stage_dir=stage, label=label or stage.resolve().parent.name or stage.name, group=group)
    names = (EVAL_EPISODES_FILENAME, TRAIN_EPISODES_FILENAME, STAGE_CONFIG_FILENAME, PROGRESS_FILENAME)
    run.files = {name: (stage / name).is_file() for name in names}
    run.files[PLANT_IDENTITY_FILENAME] = (stage / PLANT_IDENTITY_FILENAME).is_file()
    run.files[f"{MODELS_DIRNAME}/"] = (stage / MODELS_DIRNAME).is_dir()
    run.absent += [name for name, present in run.files.items() if not present]

    record = read_json_object(stage / STAGE_CONFIG_FILENAME) or {}
    progress = read_json_object(stage / PROGRESS_FILENAME) or {}
    species, embodiment = record.get("species", progress.get("species")), record.get("embodiment")
    embodiment = embodiment if embodiment is not None else progress.get("embodiment")
    if isinstance(species, str) and isinstance(embodiment, str):
        run.model_id = ModelId(species, embodiment)
        try:
            from ..species_names import model_display_name

            run.model_display = model_display_name(run.model_id)
        except Exception:  # noqa: BLE001 - a display name only
            run.model_display = f"{species} ({embodiment})"
    run_block: Mapping[str, Any] = record["run"] if isinstance(record.get("run"), Mapping) else {}
    seed = run_block.get("seed", progress.get("seed"))
    run.seed = seed if isinstance(seed, int) and not isinstance(seed, bool) else None
    run_id = progress.get("run_id") or run_block.get("label")
    run.run_id = run_id if isinstance(run_id, str) else None
    run.status = progress.get("status") if isinstance(progress.get("status"), str) else None
    timesteps = progress.get("timesteps")
    run.progress_timesteps = timesteps if isinstance(timesteps, int) and not isinstance(timesteps, bool) else None
    written = progress.get("written_at")
    run.progress_written_at = written if isinstance(written, str) else None
    run.bars = _bars(record)
    run.band = _band(record)
    if not record:
        run.absent.append("stage_config.json: no bars or band recorded")

    table = _read_csv(stage / EVAL_EPISODES_FILENAME)
    if table is not None:
        header, rows = table
        run.feet = foot_labels(header)
        run.lr = lr_columns(header)
        run.evaluations = reduce_evaluations(header, rows, bars=run.bars)
        run.fh_rows = _full_horizon_rows(header, rows)[2]
        run.absent += [
            f"{EVAL_EPISODES_FILENAME}: column {column}"
            for column in expected_columns(run.feet or ["r", "l"])
            if column not in header
        ]
        if len(run.lr) < len(LR_JOINTS):
            run.absent += [
                f"{EVAL_EPISODES_FILENAME}: left-minus-right {joint} (leg_q_*_minus_*_{joint}_deg)"
                for joint in LR_JOINTS
                if joint not in run.lr
            ]
        if not run.steps.size:
            run.absent.append(f"{EVAL_EPISODES_FILENAME}: no row with a timesteps value")
    elif run.files[EVAL_EPISODES_FILENAME]:
        run.errors.append(f"{EVAL_EPISODES_FILENAME} could not be read")
    if _edges(run.band):
        state = "paid" if run.band_active else "not paid: weight 0"
        set_band_edges(run, _edges(run.band), f"the run's own band ({STAGE_CONFIG_FILENAME}; {state})")
    saturated = run.series("saturated")
    if saturated is not None:
        episodes = run.series("episodes")
        run.saturation = saturation_alarms(
            run.steps.tolist(), saturated.tolist(), episodes=None if episodes is None else episodes.tolist()
        )
    else:
        run.saturation = {"computed": False, "why": "no screen_v2_reasons column"}

    table = _read_csv(stage / TRAIN_EPISODES_FILENAME)
    if table is not None:
        run.training = reduce_training(*table, window=training_window)
        if run.training is None:
            run.absent.append(f"{TRAIN_EPISODES_FILENAME}: no timesteps/cause rows")

    run.saved_checkpoints = [
        {key: value for key, value in entry.items() if key != "path"}
        for entry in list_checkpoints(stage / MODELS_DIRNAME, progress=progress)
    ]
    for entry in run.saved_checkpoints:
        if entry["kind"] != "periodic":
            run.checkpoint_steps.setdefault(entry["kind"], entry["timesteps"])
    run.plant = plant_check(stage, run.model_id, record)
    if read_std:
        run.policy_std = read_policy_std(stage / MODELS_DIRNAME, progress=progress)
        if run.policy_std.get("skipped"):
            run.absent.append(f"policy std: {run.policy_std['skipped']}")
        run.errors += run.policy_std.get("errors", [])
        widths = {len(entry["std"]) for entry in run.policy_std["checkpoints"]}
        if len(widths) == 1:
            run.actuators, run.actuator_source = actuator_names(run.model_id, widths.pop(), plant=run.plant)
        elif widths:
            run.actuator_source = f"checkpoints of different widths {sorted(widths)}"
    else:
        run.policy_std = {"available": 0, "read": 0, "checkpoints": [], "skipped": "not requested", "errors": []}
    checkpoints: Mapping[str, Any] = progress["checkpoints"] if isinstance(progress.get("checkpoints"), Mapping) else {}
    for kind in ("robust_best", "best"):
        step = checkpoints.get(f"{kind}_model_timesteps")
        if run.checkpoint_steps.get(kind) is None and isinstance(step, int) and not isinstance(step, bool):
            run.checkpoint_steps[kind] = step
    return run


# ---------------------------------------------------------------------------------------------- marks


def resolve_marks(marks: Iterable["int | str"], runs: Sequence[RunHistory]) -> list[int]:
    """The marked steps, sorted: integers as given, and ``final`` as each run's final step (when it has one)."""
    steps: set[int] = set()
    for mark in marks:
        if isinstance(mark, str) and mark.strip().lower() == "final":
            steps.update(step for run in runs if (step := run.final_step) is not None)
        else:
            steps.add(int(mark))
    return sorted(steps)


# ---------------------------------------------------------------------------------------------- JSON


def _sig(value: Any) -> Any:
    """A JSON value: six significant digits, NaN and infinities as ``null``, arrays as lists."""
    if isinstance(value, np.ndarray):
        return [_sig(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_sig(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _sig(item) for key, item in value.items()}
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return float(f"{number:.6g}") if math.isfinite(number) else None
    return value


def _at(run: RunHistory, step: int) -> int | None:
    """The index of the evaluation at *step*, or ``None``."""
    hits = np.flatnonzero(run.steps == step)
    return int(hits[0]) if hits.size else None


#: The checkpoint table's per-evaluation counts, in column order.
TABLE_COUNTS: tuple[str, ...] = (
    "episodes",
    "full_horizon",
    "clean",
    "falls",
    "saturated",
    "max_sole_cop_fore_aft_over_bar",
    "foot_cop_outside_band",
    "sole_cop_outside_pad_band",
    "max_sole_tilt_deg_over_bar",
    "max_sole_corner_lift_m_over_bar",
)


def checkpoint_rows(run: RunHistory, marks: Sequence[int]) -> list[dict[str, Any]]:
    """The values at the marked steps, ``robust_best``, ``best`` and the last evaluation, one row each.

    A row reads the evaluation at exactly that step; a mark with no evaluation
    there (the final checkpoint, after the last evaluation) reads the last
    evaluation at or before it and says so.  The std columns read the
    checkpoint saved at that step when one was read.
    """
    if not run.steps.size:
        return []
    final = run.final_step
    wanted: list[tuple[str, int, str]] = [
        (f"J {step:,}" + (" (final)" if step == final else ""), step, "final" if step == final else "periodic")
        for step in marks
        if step <= run.last_step
    ]
    for kind in ("robust_best", "best"):
        step = run.checkpoint_steps.get(kind)
        if isinstance(step, int):
            wanted.append((kind, step, kind))
    wanted.append(("last evaluation", int(run.steps[-1]), "periodic"))
    points = run.policy_std.get("checkpoints", [])
    rows = []
    for name, step, prefer in wanted:
        index = _at(run, step)
        note = ""
        if index is None:
            earlier = np.flatnonzero(run.steps <= step)
            if not earlier.size:
                continue
            index = int(earlier[-1])
            note = f"evaluation at {int(run.steps[index]):,} (none at this step)"
        row: dict[str, Any] = {"at": name, "timesteps": step, "evaluation_timesteps": int(run.steps[index])}
        if note:
            row["note"] = note
        for count in TABLE_COUNTS:
            series = run.series(count)
            if series is not None:
                row[count] = series[index]
        for column in [
            f"{foot}_{suffix}" for foot in run.feet for suffix in ("foot_cop_fore_aft", "sole_cop_fore_aft")
        ]:
            values = run.column(column)
            if values is not None:
                row[column] = values[index]
        for column in ("max_sole_cop_fore_aft", "max_sole_tilt_deg", "max_sole_corner_lift_m"):
            for stat, suffix in (("mean", ""), ("max", "_worst")):
                values = run.column(column, stat)
                if values is not None:
                    row[f"{column}{suffix}"] = values[index]
        for column in ("actuator_force_frac_max", "com_fore_aft_mean_m"):
            values = run.column(column)
            if values is not None:
                row[column] = values[index]
        for joint, column in run.lr.items():
            values = run.column(column)
            if values is not None:
                row[f"l_minus_r_{joint}_deg"] = values[index]
        at_step = [point for point in points if point["timesteps"] == step]
        entry = next((point for point in at_step if point["kind"] == prefer), at_step[0] if at_step else None)
        if entry is not None and run.actuators:
            row["std_checkpoint"] = entry["name"]
            for i, name_ in enumerate(run.actuators):
                if any(key in name_ for key in KEY_ACTUATORS):
                    row[f"std_{_stem(name_)}"] = entry["std"][i]
        rows.append(row)
    return rows


def run_record(run: RunHistory, marks: Sequence[int]) -> dict[str, Any]:
    """One run's block of the JSON summary."""
    evaluations: dict[str, Any] = {key: value for key, value in run.evaluations.items() if key != "columns"}
    evaluations["columns"] = run.evaluations.get("columns", {})
    std = dict(run.policy_std)
    std["actuators"] = run.actuators
    std["actuator_names_source"] = run.actuator_source
    record: dict[str, Any] = _sig(
        {
            "label": run.label,
            "group": run.group,
            "stage_dir": str(run.stage_dir),
            "files": run.files,
            "model": ({**run.model_id.record(), "display": run.model_display} if run.model_id is not None else None),
            "seed": run.seed,
            "run_id": run.run_id,
            "status": run.status,
            "progress_timesteps": run.progress_timesteps,
            "progress_written_at": run.progress_written_at,
            "last_step": run.last_step,
            "final_step": run.final_step,
            "episodes_per_evaluation": run.episodes_per_evaluation,
            "bars": run.bars,
            "band": run.band,
            "band_active": run.band_active,
            "band_edges_counted": run.band_edges,
            "band_edges_source": run.band_edges_source,
            "feet": run.feet,
            "left_minus_right_columns": run.lr,
            "evaluations": evaluations,
            "training": run.training,
            "saturation": run.saturation,
            "checkpoint_steps": run.checkpoint_steps,
            "saved_checkpoints": run.saved_checkpoints,
            "plant": run.plant,
            "policy_std": std,
            "checkpoint_table": checkpoint_rows(run, marks),
            "absent": run.absent,
            "errors": run.errors,
        }
    )
    return record


def summary(runs: Sequence[RunHistory], *, title: str, marks: Sequence["int | str"], smooth: int) -> dict[str, Any]:
    """The JSON summary (``mesozoic.training-history/v1``) of *runs*."""
    resolved = resolve_marks(marks, runs)
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "title": title,
        "report_only": "reads the stage directories only; nothing reads this file back",
        "marks": {"requested": [str(mark) for mark in marks], "resolved": resolved},
        "rules": {
            "counts": "episodes, full_horizon, falls (shorter than the horizon), clean and saturated over all the "
            "evaluation's episodes; clean and saturated over the episodes the screen judged",
            "columns": "each per-episode column's mean, std (ddof 0), min, max and n over the evaluation's "
            "full-horizon episodes",
            "over_bar": "<rail>_over_bar: the evaluation's full-horizon episodes whose own <rail> value (the episode's "
            "maximum over the feet; for the CoP rails the window mean of |CoP| with an unloaded step counted as 1) "
            "exceeds the stage's declared bar; <rail>_over_bar_fraction: that count over the full-horizon episodes "
            "that measured the rail",
            "outside_band": "foot_cop_outside_band (sole_cop_outside_pad_band): the evaluation's full-horizon "
            "episodes with either foot's window-mean whole-foot (pad) CoP outside the band's edges, an unloaded foot "
            "not counted; the edges are the run's own or, where it records none, the reference edges "
            "band_edges_source names",
            "saturation": saturation_alarms([], [])["rule"],
            "training_falls": f"train_episodes.csv episodes whose cause is not truncated, per window of "
            f"{TRAINING_WINDOW_TIMESTEPS:,} steps (k w, (k + 1) w]; the last window ends at its last episode",
            "comparison_smoothing_evaluations": smooth,
            "comparison_smoothing": "comparisons draw the continuous series as a centred rolling mean over that many "
            "evaluations, each evaluation's value faint behind it; counts are never smoothed",
            "numbers": "six significant digits; null where not measured",
        },
        "not_recorded": list(NOT_RECORDED),
        "runs": [run_record(run, resolved) for run in runs],
    }


# ---------------------------------------------------------------------------------------------- figures


@dataclass(frozen=True)
class Quantity:
    """One quantity drawn against the step.

    ``kind``: ``column`` (a reduced column, ``{foot}`` for each foot),
    ``series`` (a continuous evaluation series), ``count`` (a per-evaluation
    count: a line on one run's page, a strip per run on a comparison, never
    smoothed) or ``training`` (the training-fall fraction per window, likewise).
    """

    key: str
    title: str
    unit: str
    kind: str = "column"
    spread: str | None = "std"  # one run: "std" band, "max" or "min" worst-episode line, or None
    scale: float = 1.0
    per_foot: bool = False
    ramp: str = "orange"  # a count strip's sequential ramp
    annotate: bool = False  # write each non-zero count (sparse counts only)


@dataclass(frozen=True)
class PageSpec:
    """One page: its file stem, title, caption and quantities, one per panel (per foot per panel when compared).

    The caption may name ``{episodes}`` (the evaluation sizes), ``{band}``
    (the band edges counted) and ``{saturation_note}``.
    """

    name: str
    title: str
    caption: str
    quantities: tuple[Quantity, ...]


_FH = "mean over the evaluation's full-horizon episodes"
#: What a one-run panel draws besides the mean, said in its title.
_SPREAD_NOTES = {
    "std": " (shaded: +/- 1 std over those episodes)",
    "max": " (dashed: the worst episode)",
    "min": " (dashed: the lowest episode)",
}
_PER_EPISODE = (
    "the bar applies to each episode's own value, so the count of episodes over it is drawn below; a mean under "
    "the bar does not mean every episode is"
)
PAGES: tuple[PageSpec, ...] = (
    PageSpec(
        "01_outcomes",
        "Outcomes: clean episodes, falls and return",
        "Counts are over each evaluation's episodes ({episodes}). Clean is the in-training stance_quality/v2 screen "
        "at yaw 0 without the statue-relative criteria (never a verdict). Evaluation episodes are the deterministic "
        "policy's; the training falls come from train_episodes.csv (the stochastic policy). Counts are never "
        "smoothed: a comparison draws one strip per run, a cell per evaluation (per window for training falls).",
        (
            Quantity("clean", "Clean episodes per evaluation (screen_v2_clean = 1)", "episodes", "count", ramp="blue"),
            Quantity(
                "falls",
                "Falls per evaluation (episodes shorter than the horizon; each fall count is written beside it)",
                "episodes",
                "count",
                annotate=True,
            ),
            Quantity(
                "fall_fraction",
                f"Training falls: fraction of the training episodes ending in each {TRAINING_WINDOW_TIMESTEPS:,}-step "
                "window that did not reach the time limit",
                "fraction",
                "training",
            ),
            Quantity("mean_return", "Mean evaluation return (all episodes)", "return", "series", None),
        ),
    ),
    PageSpec(
        "02_saturation",
        "Actuator saturation and force",
        "A saturation flag is an episode whose screen_v2_reasons lists max_actuator_saturation_fraction: some actuator "
        "was commanded at |action| >= 0.99 on more than the stage's bar fraction of its post-settle window steps (the "
        "metric is the largest per-actuator fraction). An evaluation with >= 5 such episodes is flagged (the scorers' "
        "rule, set on 10-episode evaluations{saturation_note}); two flagged evaluations in a row are an alarm (dotted "
        "line; on a comparison's strip, a triangle). actuator_force_frac_max is about force, not the action command.",
        (
            Quantity(
                "saturated",
                "Saturation-flagged episodes per evaluation (>= 5 flags the evaluation)",
                "episodes",
                "count",
            ),
            Quantity(
                "actuator_force_frac_max",
                "actuator_force_frac_max: the highest window mean of |F| / limit over the force-limited actuators "
                f"(limit 1.0; the saturation flags are about the action command, not force); {_FH}",
                "fraction of limit",
                spread="max",
            ),
        ),
    ),
    PageSpec(
        "03_foot_cop",
        "Whole-foot centre of pressure, per foot",
        "The signed whole-foot CoP fore-aft: every floor contact of the limb, in half-lengths of the declared support "
        "span: 0 the middle of the span, -1 the heel line, +1 the toe line, + toward the toe. The reward band (dotted, "
        "{band}) is paid per window step on each loaded foot's CoP, not on this signed mean over episodes, in which "
        "toe and heel episodes cancel: the count of episodes with either foot's window mean outside the band is drawn "
        "below it. max_foot_cop_fore_aft is each episode's maximum over the feet of the window mean of |CoP| (an "
        "unloaded step counted as 1); no stage declares its bar.",
        (
            Quantity(
                "{foot}_foot_cop_fore_aft",
                f"Whole-foot CoP fore-aft, signed; {_FH}",
                "half-lengths (+ toe)",
                per_foot=True,
            ),
            Quantity(
                "foot_cop_outside_band",
                "Full-horizon episodes with either foot's whole-foot window-mean CoP outside the band {band}",
                "episodes",
                "count",
            ),
            Quantity(
                "max_foot_cop_fore_aft",
                f"max_foot_cop_fore_aft per episode (report-only, no bar); {_FH}",
                "half-lengths",
                spread="max",
            ),
        ),
    ),
    PageSpec(
        "04_pad_cop",
        "Pad (sole) centre of pressure and the pad rail, per foot",
        "The signed fore-aft pad CoP in pad half-lengths (+1 the toe edge, -1 the heel edge). The pad rail "
        "max_sole_cop_fore_aft is each episode's maximum over the feet of the window mean of |pad CoP|, an unloaded "
        "step counted as 1: " + _PER_EPISODE + ". The signed mean is not the rail's statistic (toe and heel episodes, "
        "and the two feet, cancel in it), so the rail's bar is drawn on the rail's own panel only. Dotted: the pad "
        "band's edges ({pad_band}), paid per window step.",
        (
            Quantity(
                "{foot}_sole_cop_fore_aft",
                f"Pad CoP fore-aft, signed; {_FH}",
                "pad half-lengths (+ toe)",
                per_foot=True,
            ),
            Quantity(
                "max_sole_cop_fore_aft",
                f"Pad rail: max_sole_cop_fore_aft per episode; {_FH}",
                "pad half-lengths",
                spread="max",
            ),
            Quantity(
                "max_sole_cop_fore_aft_over_bar",
                "Full-horizon episodes over the pad rail's bar (their max_sole_cop_fore_aft above {bar})",
                "episodes",
                "count",
            ),
            Quantity(
                "sole_cop_outside_pad_band",
                "Full-horizon episodes with either foot's pad window-mean CoP outside the pad band {pad_band}",
                "episodes",
                "count",
            ),
        ),
    ),
    PageSpec(
        "05_digits_margins",
        "Digit share and toe moment margin, per foot",
        "Digit share: the post-settle floor force on the digits over the whole limb's. Toe margin: the floor-load "
        "moment left before the whole-foot CoP reaches the toe line (N m).",
        (
            Quantity("{foot}_digit_share", f"Digit share; {_FH}", "fraction of limb force", per_foot=True),
            Quantity("{foot}_toe_margin_nm", f"Toe moment margin; {_FH}", "N m", per_foot=True),
        ),
    ),
    PageSpec(
        "06_balance",
        "Whole-body balance",
        "XcoM margins: how far the extrapolated centre of mass is inside the declared toe and heel lines (m, + "
        "inside), each episode's window minimum. CoM: the centre of mass ahead of the feet's reference point. "
        "Pitch: the root's forward axis above horizontal.",
        (
            Quantity(
                "xcom_toe_margin_min_m",
                f"XcoM toe margin, window minimum; {_FH}",
                "m (+ inside)",
                spread="min",
            ),
            Quantity(
                "xcom_heel_margin_min_m",
                f"XcoM heel margin, window minimum; {_FH}",
                "m (+ inside)",
                spread="min",
            ),
            Quantity("com_fore_aft_mean_m", f"CoM ahead of the feet, window mean; {_FH}", "m (+ forward)"),
            Quantity("root_pitch_mean_deg", f"Root pitch, window mean; {_FH}", "degrees (+ nose up)"),
        ),
    ),
    PageSpec(
        "07_sole_roll",
        "Sole tilt and corner lift (the gate's roll rails)",
        "Each episode's maximum over the feet of the sole's post-settle mean tilt and corner lift, the numbers the "
        "gate's roll rails read; dashed horizontal: the stage's bar. "
        + _PER_EPISODE[0].upper()
        + _PER_EPISODE[1:]
        + ".",
        (
            Quantity(
                "max_sole_tilt_deg",
                f"Sole tilt, max over the feet; {_FH}",
                "degrees",
                spread="max",
            ),
            Quantity(
                "max_sole_corner_lift_m",
                f"Sole corner lift, max over the feet; {_FH}",
                "mm",
                spread="max",
                scale=1000.0,
            ),
            Quantity(
                "max_sole_tilt_deg_over_bar",
                "Full-horizon episodes over the tilt bar (their max_sole_tilt_deg above {bar})",
                "episodes",
                "count",
            ),
            Quantity(
                "max_sole_corner_lift_m_over_bar",
                "Full-horizon episodes over the corner-lift bar (their max_sole_corner_lift_m above {bar})",
                "episodes",
                "count",
            ),
        ),
    ),
    PageSpec(
        "08_floor_forces",
        "Floor shear and friction-cone use, per foot",
        "Shear over normal: the floor's forward force over its normal force on the limb, summed over the loaded "
        "window steps (+ the floor pushes forward). Cone use: the horizontal floor force over mu F_z, the window mean.",
        (
            Quantity("{foot}_shear_over_normal", f"Shear over normal; {_FH}", "ratio (+ forward)", per_foot=True),
            Quantity(
                "{foot}_cone_use_mean", f"Friction-cone use, window mean; {_FH}", "fraction of cone", per_foot=True
            ),
        ),
    ),
    PageSpec(
        "09_asymmetry",
        "Left-right asymmetry of the legs: hip pitch, knee and ankle (hip roll is not recorded)",
        "The achieved joint angle offset from the home pose, left minus right, each side signed about the animal's "
        "leftward axis (leg_q_<left>_minus_<right>_<joint>_deg); a mirror-symmetric posture reads 0. Hip roll is "
        "not recorded in eval_episodes.csv, so it has no panel.",
        tuple(
            Quantity(
                f"lr:{joint}",
                f"{joint.replace('_', ' ').capitalize()}: left minus right; {_FH}",
                "degrees",
            )
            for joint in LR_JOINTS
        ),
    ),
)


def _stem(name: str) -> str:
    return name[: -len("_act")] if name.endswith("_act") else name


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    """A centred rolling mean over *window* evaluations of the finite values, the window cut at the ends.

    NaN where the window holds no finite value; the output is as long as the input.
    """
    values = np.asarray(values, dtype=float)
    if window <= 1 or values.size == 0:
        return values
    finite = np.isfinite(values)
    total = np.concatenate(([0.0], np.cumsum(np.where(finite, values, 0.0))))
    count = np.concatenate(([0.0], np.cumsum(finite.astype(float))))
    index = np.arange(values.size)
    low = np.clip(index - (window - 1) // 2, 0, values.size)
    high = np.clip(index + window // 2 + 1, 0, values.size)
    sums, counts = total[high] - total[low], count[high] - count[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / np.where(counts > 0, counts, 1.0), np.nan)


def _values(
    run: RunHistory, quantity: Quantity, foot: str | None, stat: str = "mean"
) -> tuple[np.ndarray, np.ndarray] | None:
    """``(x in millions of steps, y)`` of one quantity for one run, or ``None`` when absent."""
    if quantity.kind == "training":
        if run.training is None:
            return None
        return run.training["window_end"] / 1e6, np.asarray(run.training[quantity.key], dtype=float)
    if quantity.kind in ("count", "series"):
        series = run.series(quantity.key)
        return None if series is None else (run.steps / 1e6, series)
    key = quantity.key
    if key.startswith("lr:"):
        column = run.lr.get(key[3:])
        if column is None:
            return None
        key = column
    elif foot is not None:
        key = key.format(foot=foot)
    values = run.column(key, stat)
    return None if values is None else (run.steps / 1e6, values * quantity.scale)


def _common(values: Iterable[float | None]) -> list[float]:
    return sorted({float(v) for v in values if v is not None and math.isfinite(v)})


def _fmt_steps(step: float) -> str:
    """A step count as millions, ``4.19M``."""
    return f"{step / 1e6:.2f}M" if step >= 1e6 else f"{step / 1e3:.0f}k"


def _band_text(runs: Sequence[RunHistory], lower: str, upper: str) -> str:
    """The band edges counted, as ``[-0.6, +0.4]``, or why there are none."""
    pairs = {(run.band_edges.get(lower), run.band_edges.get(upper)) for run in runs if run.band_edges}
    pairs = {pair for pair in pairs if None not in pair}
    if not pairs:
        return "(no band edges recorded: none counted)"
    if len(pairs) > 1:
        return "(each run's own edges)"
    low, high = pairs.pop()
    suffix = ""
    if any(run.band_edges_source and "reference" in run.band_edges_source for run in runs):
        suffix = (
            "; this run does not pay the band (reference edges)"
            if len(runs) == 1
            else "; runs that do not pay the band are counted against the same edges"
        )
    return f"[{low:+g}, {high:+g}]{suffix}"


def _bar_text(runs: Sequence[RunHistory], key: str, scale: float = 1.0, unit: str = "") -> str:
    values = _common(run.bars.get(key) for run in runs)
    if not values:
        return "the bar (none declared)"
    if len(values) > 1:
        return "each run's bar"
    value = values[0]
    if scale != 1.0:
        return f"{value * scale:g} {unit} ({value:g} m)"
    return f"{value:g}{(' ' + unit) if unit else ''}"


def _quantity_title(quantity: Quantity, runs: Sequence[RunHistory]) -> str:
    rail = quantity.key[: -len("_over_bar")] if quantity.key.endswith("_over_bar") else None
    bar = ""
    if rail == "max_sole_corner_lift_m":
        bar = _bar_text(runs, rail, 1000.0, "mm")
    elif rail == "max_sole_tilt_deg":
        bar = _bar_text(runs, rail, unit="degrees")
    elif rail is not None:
        bar = _bar_text(runs, rail)
    return quantity.title.format(
        bar=bar,
        band=_band_text(runs, "lower", "upper"),
        pad_band=_band_text(runs, "pad_lower", "pad_upper"),
        foot="{foot}",
    )


def _reference_lines(runs: Sequence[RunHistory], quantity: Quantity) -> list[tuple[float, str, str]]:
    """``(y, label, linestyle)`` of the bars, band edges, limits and zero lines a panel draws."""
    lines: list[tuple[float, str, str]] = []
    key = quantity.key

    def band_lines(lower: str, upper: str, what: str) -> None:
        edged = [run for run in runs if run.band_edges]
        for name in (lower, upper):
            for value in _common(run.band_edges.get(name) for run in edged):
                paid = any(run.band_active for run in edged)
                lines.append((value, f"{what} {value:+g}" + ("" if paid else " (not paid)"), ":"))

    if key == "{foot}_foot_cop_fore_aft":
        band_lines("lower", "upper", "band edge")
    elif key == "{foot}_sole_cop_fore_aft":
        band_lines("pad_lower", "pad_upper", "pad band edge")
    elif key in ("max_sole_cop_fore_aft", "max_foot_cop_fore_aft", "max_sole_tilt_deg", "max_sole_corner_lift_m"):
        for value in _common(run.bars.get(key) for run in runs):
            if key == "max_sole_corner_lift_m":
                text = f"bar {value * quantity.scale:g} mm ({value:g} m)"
            elif key == "max_sole_tilt_deg":
                text = f"bar {value:g} degrees"
            else:
                text = f"{key} bar {value:g}"
            lines.append((value * quantity.scale, text, "--"))
    elif key == "saturated":
        lines.append((float(SATURATION_FLAG_EPISODES), f"flag: >= {SATURATION_FLAG_EPISODES} episodes", "--"))
    elif key == "actuator_force_frac_max":
        lines.append((1.0, "limit 1.0", "--"))
    if key.endswith(("_fore_aft", "margin_min_m", "com_fore_aft_mean_m", "shear_over_normal")) and not key.startswith(
        "max_"
    ):
        lines.append((0.0, "", "-"))
    elif key.startswith("lr:"):
        lines.append((0.0, "", "-"))
    return lines


# Layout, in inches: every page's axes span the same x range, so a step falls at the same place on every page.
_LEFT_IN = 1.0
_RIGHT_IN = 2.8
_TICKS_IN = 0.34
_HEAD_IN = 0.16
_HEAD_LINE_IN = 0.2
_MARK_ROW_IN = 0.26
_LINE_IN = 2.0
_COUNT_IN = 1.4
_STRIP_ROW_IN = 0.25
_TITLE_PT = 10.5
_TICK_PT = 9.5


def _wrap(text: str, width_in: float, size: float = _TITLE_PT) -> str:
    return textwrap.fill(text, width=max(20, int(width_in * 72 / (size * 0.56))))


@dataclass
class _Panel:
    quantity: Quantity | None
    foot: str | None
    title: str
    unit: str
    height: float
    strip: bool = False


def _new_figure(
    panels: Sequence[_Panel],
    *,
    width: float,
    dpi: int,
    title: str,
    legend_rows: int,
    columns: int = 1,
    left: float = _LEFT_IN,
    right: float = _RIGHT_IN,
) -> tuple[Any, list[Any]]:
    """A figure of *panels* stacked (or in *columns*) at fixed margins in inches; returns it and the axes."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    gap = 0.75 if columns > 1 else 0.0
    axes_width = (width - left - right - gap * (columns - 1)) / columns
    rows = [panels[i : i + columns] for i in range(0, len(panels), columns)]
    heads = [_HEAD_IN + _HEAD_LINE_IN * max(_wrap(p.title, axes_width).count("\n") + 1 for p in row) for row in rows]
    title = "\n".join(_wrap(line, width - left - 0.3, 12.0) for line in title.split("\n"))
    title_in = 0.2 + 0.24 * (title.count("\n") + 1)
    bottom_in = 0.5 + 0.24 * legend_rows
    heights = [max(p.height for p in row) for row in rows]
    total = title_in + _MARK_ROW_IN + sum(h + head + _TICKS_IN for h, head in zip(heights, heads)) + bottom_in
    figure = Figure(figsize=(width, total), dpi=dpi)
    FigureCanvasAgg(figure)
    figure.patch.set_facecolor("white")
    figure.text(left / width, 1 - 0.14 / total, title, ha="left", va="top", fontsize=12, color=_INK)
    axes: list[Any] = []
    y = total - title_in - _MARK_ROW_IN
    for row, height, head in zip(rows, heights, heads):
        y -= head + height
        for column, _panel in enumerate(row):
            x = left + column * (axes_width + gap)
            axis = figure.add_axes(
                (x / width, y / total, axes_width / width, height / total), sharex=axes[0] if axes else None
            )
            axes.append(axis)
        y -= _TICKS_IN
    # Read by _colourbar and _figure_legend.
    figure._history_layout = {"width": width, "height": total, "left": left, "right": right, "bottom": bottom_in}
    return figure, axes


def _style_axis(axis: Any, panel: _Panel, *, top: bool) -> None:
    width_in = axis.get_position().width * axis.figure.get_figwidth()
    axis.set_title(
        _wrap(panel.title, width_in),
        loc="left",
        fontsize=_TITLE_PT,
        color=_INK,
        pad=(4 + _MARK_ROW_IN * 72) if top else 5,
    )
    if not panel.strip:
        axis.set_ylabel(panel.unit, fontsize=_TICK_PT, color=_INK_2)
    axis.grid(True, color=_GRID, linewidth=0.7, axis="x" if panel.strip else "both")
    axis.set_facecolor(_SURFACE)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(_AXIS)
    axis.tick_params(colors=_INK_2, labelsize=_TICK_PT, labelbottom=True)


def _absent(axis: Any, text: str) -> None:
    axis.text(0.5, 0.5, text, transform=axis.transAxes, ha="center", va="center", fontsize=9.5, color=_MUTED, wrap=True)


@dataclass
class _Context:
    runs: Sequence[RunHistory]
    marks: Sequence[int]
    comparison: bool
    smooth: int
    xmax: float
    colours: list[str]
    styles: list[Any]
    colour_note: str = ""


def _strip_cmap(ramp: str, top: int) -> tuple[Any, Any]:
    """A discrete colour map for counts 0..*top*: 0 neutral, then the ramp light to dark."""
    from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap

    steps = _RAMPS[ramp]
    ramp_map = LinearSegmentedColormap.from_list(ramp, steps)
    # Zero is neutral: light on a problem count (nothing wrong), a darker grey on the clean count (none clean).
    zero = _ZERO_CELL if ramp == "orange" else "#cfccc2"
    colours = [zero] + [ramp_map(i / max(1, top - 1)) for i in range(top)]
    return ListedColormap(colours), BoundaryNorm(np.arange(-0.5, top + 1.0, 1.0), top + 1)


def _training_cmap() -> tuple[Any, Any]:
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    return LinearSegmentedColormap.from_list("training", (_ZERO_CELL, *_RAMPS["orange"])), Normalize(
        0.0, TRAINING_FALL_CAP, clip=True
    )


def _spacing(x: np.ndarray, default: float) -> float:
    gaps = np.diff(np.asarray(x, dtype=float))
    gaps = gaps[gaps > 0]
    return float(np.median(gaps)) if gaps.size else default


def _draw_strips(axis: Any, context: _Context, quantity: Quantity) -> bool:
    """One row per run, one cell per evaluation (per window for training falls), coloured by the count."""
    from matplotlib.patches import Rectangle
    from matplotlib.transforms import blended_transform_factory

    training = quantity.kind == "training"
    sizes = [n for run in context.runs for n in run.episodes_per_evaluation]
    top = max(sizes) if sizes else 10
    cmap, norm = _training_cmap() if training else _strip_cmap(quantity.ramp, top)
    drawn = False
    for row, run in enumerate(context.runs):
        values = _values(run, quantity, None)
        if values is None:
            axis.text(
                0.005,
                row,
                "not recorded",
                transform=blended_transform_factory(axis.transAxes, axis.transData),
                va="center",
                fontsize=8.5,
                color=_MUTED,
            )
            continue
        drawn = True
        x, y = values
        finite = np.isfinite(y)
        if training and run.training is not None:
            # Windows are contiguous: each starts where the one before ends (the last ends at its last episode).
            starts = np.concatenate(([max(0.0, x[0] - run.training["window_timesteps"] / 1e6)], x[:-1]))
            widths = x - starts
            axis.bar(
                starts[finite],
                0.8,
                width=widths[finite],
                bottom=row - 0.4,
                align="edge",
                color=cmap(norm(y[finite])),
                linewidth=0,
            )
        else:
            width = _spacing(x, 0.1)
            axis.bar(
                x[finite], 0.8, width=width, bottom=row - 0.4, align="center", color=cmap(norm(y[finite])), linewidth=0
            )
        if quantity.annotate:
            for xi, yi in zip(x[finite], y[finite]):
                if yi > 0:
                    colour = "white" if yi >= 0.5 * top else _INK
                    axis.text(xi, row, f"{yi:g}", ha="center", va="center", fontsize=7.5, color=colour)
        if quantity.key == "saturated":
            width = _spacing(x, 0.1)
            for step in run.saturation.get("flagged", []):
                axis.add_patch(
                    Rectangle(
                        (step / 1e6 - width / 2, row - 0.4),
                        width,
                        0.8,
                        fill=False,
                        edgecolor=_INK,
                        linewidth=0.8,
                        zorder=3,
                    )
                )
            for entry in run.saturation.get("runs", []):
                axis.plot(
                    entry["alarm_at"] / 1e6,
                    row,
                    marker="v",
                    markersize=7,
                    markerfacecolor="white",
                    markeredgecolor=_INK,
                    markeredgewidth=1.0,
                    linestyle="none",
                    zorder=4,
                )
    axis.set_ylim(len(context.runs) - 0.5, -0.75)
    axis.set_yticks(range(len(context.runs)))
    axis.set_yticklabels([])
    axis.tick_params(axis="y", length=0)
    # Row names on the right, in ink, each beside a swatch of the run's line.
    transform = blended_transform_factory(axis.transAxes, axis.transData)
    for row, run in enumerate(context.runs):
        axis.plot(
            [1.008, 1.04],
            [row, row],
            transform=transform,
            color=context.colours[row],
            linestyle=context.styles[row],
            linewidth=2.0,
            clip_on=False,
        )
        axis.text(1.048, row, run.label, transform=transform, va="center", ha="left", fontsize=8.5, color=_INK)
    _colourbar(axis, cmap, norm, training=training, top=top)
    return drawn


def _colourbar(axis: Any, cmap: Any, norm: Any, *, training: bool, top: int) -> None:
    from matplotlib.cm import ScalarMappable

    figure = axis.figure
    layout = figure._history_layout
    box = axis.get_position()
    x = box.x1 + 1.9 / layout["width"]
    cax = figure.add_axes((x, box.y0, 0.11 / layout["width"], box.height))
    bar = figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    bar.outline.set_visible(False)
    cax.tick_params(labelsize=8, colors=_INK_2, length=2)
    if training:
        bar.set_ticks([0.0, TRAINING_FALL_CAP / 2, TRAINING_FALL_CAP])
        bar.set_ticklabels(["0", f"{TRAINING_FALL_CAP / 2:g}", f">= {TRAINING_FALL_CAP:g}"])
    else:
        ticks = list(range(0, top + 1, 2 if top > 6 else 1))
        bar.set_ticks(ticks)
        bar.set_ticklabels([str(t) for t in ticks])
    cax.set_title("fraction" if training else "episodes", fontsize=8, color=_INK_2, loc="left")


def _draw_line_single(axis: Any, context: _Context, quantity: Quantity) -> bool:
    run = context.runs[0]
    feet = run.feet if quantity.per_foot else [None]
    drawn = False
    for foot_index, which in enumerate(feet):
        values = _values(run, quantity, which)
        if values is None:
            continue
        x, y = values
        drawn = True
        colour = _SERIES[foot_index % len(_SERIES)] if quantity.per_foot else _SERIES[0]
        label = f"foot {which}" if which is not None else ("mean" if quantity.spread else None)
        axis.plot(x, y, color=colour, linewidth=1.7, label=label, marker="o" if x.size < 40 else None, markersize=3)
        if quantity.spread == "std":
            spread = _values(run, quantity, which, "std")
            if spread is not None:
                axis.fill_between(x, y - spread[1], y + spread[1], color=colour, alpha=0.15, linewidth=0)
        elif quantity.spread in ("max", "min"):
            worst = _values(run, quantity, which, quantity.spread)
            if worst is not None:
                axis.plot(
                    x,
                    worst[1],
                    color=colour,
                    linewidth=1.0,
                    linestyle="--",
                    alpha=0.85,
                    label=f"worst episode{f' ({which})' if which else ''}",
                )
    return drawn


def _draw_count_single(axis: Any, context: _Context, quantity: Quantity) -> bool:
    run = context.runs[0]
    values = _values(run, quantity, None)
    if values is None:
        return False
    x, y = values
    axis.plot(x, y, color=_SERIES[0], linewidth=1.3, marker="o", markersize=3.2, label="_nolegend_")
    finite = y[np.isfinite(y)]
    peak = float(finite.max()) if finite.size else 0.0
    episodes = max(run.episodes_per_evaluation or [10])
    if quantity.key in ("clean", "saturated"):
        axis.set_ylim(-0.4, episodes + 0.4)
    else:
        axis.set_ylim(-0.3, max(3.0, peak + 1.0))
    if quantity.annotate:
        for xi, yi in zip(x, y):
            if np.isfinite(yi) and yi > 0:
                axis.annotate(
                    f"{yi:g}",
                    (xi, yi),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=_INK,
                )
    if quantity.key == "saturated":
        hits = [i for step in run.saturation.get("flagged", []) if (i := _at(run, step)) is not None]
        axis.plot(
            x[hits], y[hits], linestyle="none", marker="o", markersize=5, color=_CRITICAL, label="flagged evaluation"
        )
    return True


def _draw_training_single(axis: Any, context: _Context, quantity: Quantity) -> bool:
    run = context.runs[0]
    values = _values(run, quantity, None)
    if values is None:
        return False
    x, y = values
    axis.plot(x, y, color=_SERIES[0], linewidth=1.3, marker="o", markersize=3, label="_nolegend_")
    rest = y[1:][np.isfinite(y[1:])]
    top = max(0.02, 1.2 * float(rest.max())) if rest.size else 1.0
    axis.set_ylim(0.0, top)
    if y.size and np.isfinite(y[0]) and y[0] > top:
        axis.annotate(
            f"first window {y[0]:.2f} (start-up; off scale)",
            (x[0], top),
            xytext=(8, -2),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=8.5,
            color=_INK_2,
        )
        axis.plot([x[0]], [top], marker="^", color=_SERIES[0], clip_on=False, markersize=6)
    return True


def _draw_line_compare(axis: Any, context: _Context, quantity: Quantity, foot: str | None) -> list[tuple]:
    """Each run's smoothed line (raw values faint behind); returns the end-label entries."""
    ends: list[tuple] = []
    for index, run in enumerate(context.runs):
        values = _values(run, quantity, foot)
        if values is None:
            continue
        x, y = values
        colour, style = context.colours[index], context.styles[index]
        axis.plot(x, y, color=colour, linewidth=0.8, alpha=0.22, label="_nolegend_")
        smooth = _smooth(y, context.smooth)
        axis.plot(x, smooth, color=colour, linestyle=style, linewidth=1.8, label="_nolegend_")
        finite = np.flatnonzero(np.isfinite(smooth))
        if finite.size:
            ends.append((float(x[finite[-1]]), float(smooth[finite[-1]]), run.label, colour, style))
    return ends


def _end_labels(axis: Any, ends: Sequence[tuple]) -> None:
    """Each run's name at the right of the axes, at its line's last value, nudged apart."""
    if not ends:
        return
    to_axes = axis.transScale + axis.transLimits
    height_in = axis.get_position().height * axis.figure.get_figheight()
    gap = 0.16 / height_in
    entries = sorted(
        ((float(to_axes.transform((x, y))[1]), label, colour, style) for x, y, label, colour, style in ends),
        key=lambda e: e[0],
    )
    positions = [min(1.02, max(-0.02, e[0])) for e in entries]
    for k in range(1, len(positions)):
        positions[k] = max(positions[k], positions[k - 1] + gap)
    overflow = positions[-1] - 1.04
    if overflow > 0:
        positions = [p - overflow for p in positions]
        for k in range(len(positions) - 2, -1, -1):
            positions[k] = min(positions[k], positions[k + 1] - gap)
    for (_, label, colour, style), y in zip(entries, positions):
        axis.plot(
            [1.008, 1.04], [y, y], transform=axis.transAxes, color=colour, linestyle=style, linewidth=2.0, clip_on=False
        )
        axis.text(1.048, y, label, transform=axis.transAxes, va="center", ha="left", fontsize=8.5, color=_INK)


def _draw_references(axis: Any, context: _Context, quantity: Quantity) -> None:
    from matplotlib.transforms import blended_transform_factory

    transform = blended_transform_factory(axis.transAxes, axis.transData)
    for level, label, style in _reference_lines(context.runs, quantity):
        colour = _AXIS if style == "-" else _INK_2
        axis.axhline(level, color=colour, linestyle=style, linewidth=1.1 if style != "-" else 0.9, zorder=1)
        if label:
            axis.text(
                0.006,
                level,
                label,
                transform=transform,
                ha="left",
                va="bottom",
                fontsize=8.5,
                color=_INK_2,
                zorder=5,
                bbox={"facecolor": _SURFACE, "alpha": 0.8, "edgecolor": "none", "pad": 0.6},
            )


def _draw_marks(axis: Any, context: _Context, *, top: bool, strip: bool = False) -> None:
    """The J checkpoints, the handoff and best checkpoints (one run), the checkpoints kept and the alarms.

    One run's pages mark every alarm (each run of flagged evaluations); a
    comparison's line panels mark each run's first, in its colour (its strips
    mark every alarm in the run's row).
    """
    from matplotlib.transforms import blended_transform_factory

    transform = blended_transform_factory(axis.transData, axis.transAxes)
    # A half-width panel (the std comparison) has no room for "J 4.19M" beside "J 5.24M".
    wide = axis.get_position().width * axis.figure.get_figwidth() > 6.0
    for step in context.marks:
        x = step / 1e6
        if x <= context.xmax * 1.001:
            axis.axvline(x, color=_INK_2, linewidth=1.0, alpha=0.85, zorder=2.5)
            if top:
                axis.text(
                    x,
                    1.012,
                    f"J {_fmt_steps(step)}" if wide else f"J{step / 1e6:.1f}",
                    transform=transform,
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    color=_INK,
                    fontweight="bold",
                )
    if not context.comparison:
        run = context.runs[0]
        for kind, style in (("robust_best", "--"), ("best", ":")):
            saved = run.checkpoint_steps.get(kind)
            if isinstance(saved, int) and (kind == "robust_best" or saved != run.checkpoint_steps.get("robust_best")):
                axis.axvline(saved / 1e6, color=_HANDOFF, linestyle=style, linewidth=1.3, zorder=1)
        kept = [c["timesteps"] / 1e6 for c in run.saved_checkpoints if isinstance(c.get("timesteps"), int)]
        if kept:
            axis.plot(
                kept,
                [0.0] * len(kept),
                transform=transform,
                linestyle="none",
                marker="^",
                markersize=6,
                color=_INK_2,
                clip_on=False,
                zorder=4,
            )
    if strip:
        return
    for index, run in enumerate(context.runs):
        alarms = run.saturation.get("runs", []) if run.saturation.get("alarm") else []
        for number, entry in enumerate(alarms[:1] if context.comparison else alarms):
            x = entry["alarm_at"] / 1e6
            colour = context.colours[index] if context.comparison else _CRITICAL
            axis.axvline(x, color=colour, linestyle=(0, (1.5, 1.5)), linewidth=1.9, zorder=2)
            if top:
                text = f"{run.label} alarm" if context.comparison else ("first alarm" if number == 0 else "alarm")
                axis.text(
                    x,
                    0.97,
                    f"{text} {_fmt_steps(entry['alarm_at'])} ",
                    transform=transform,
                    rotation=90,
                    ha="right",
                    va="top",
                    fontsize=8,
                    color=_INK_2,
                    zorder=5,
                    bbox={"facecolor": _SURFACE, "alpha": 0.75, "edgecolor": "none", "pad": 0.4},
                )


def _legend(axis: Any) -> None:
    """The panel's own legend, in the right margin, so that it hides no data."""
    handles, labels = axis.get_legend_handles_labels()
    if labels:
        axis.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(1.008, 1.0), frameon=False, borderaxespad=0.0)


def _run_handles(context: _Context) -> tuple[list[Any], list[str]]:
    from matplotlib.lines import Line2D

    handles: list[Any] = []
    labels: list[str] = []
    for index, run in enumerate(context.runs):
        handles.append(Line2D([], [], color=context.colours[index], linestyle=context.styles[index], linewidth=2))
        labels.append(f"{run.label}" + (f" ({run.group})" if run.group else ""))
    handles.append(Line2D([], [], color=_INK_2, linewidth=1.0))
    labels.append("J checkpoint")
    if any(run.saturation.get("alarm") for run in context.runs):
        handles.append(Line2D([], [], color=_INK_2, linestyle=(0, (1.5, 1.5)), linewidth=1.9))
        labels.append("first saturation alarm (run colour); strip triangles: every alarm")
    return handles, labels


def _single_handles(context: _Context) -> tuple[list[Any], list[str]]:
    from matplotlib.lines import Line2D

    run = context.runs[0]
    handles: list[Any] = [Line2D([], [], color=_INK_2, linewidth=1.0)]
    labels = ["J checkpoint"]
    if isinstance(run.checkpoint_steps.get("robust_best"), int):
        handles.append(Line2D([], [], color=_HANDOFF, linestyle="--", linewidth=1.3))
        labels.append(f"robust_best (handoff) {run.checkpoint_steps['robust_best']:,}")
    best = run.checkpoint_steps.get("best")
    if isinstance(best, int) and best != run.checkpoint_steps.get("robust_best"):
        handles.append(Line2D([], [], color=_HANDOFF, linestyle=":", linewidth=1.3))
        labels.append(f"best {best:,}")
    if run.saturation.get("alarm"):
        handles.append(Line2D([], [], color=_CRITICAL, linestyle=(0, (1.5, 1.5)), linewidth=1.9))
        count = len(run.saturation["runs"])
        labels.append(
            f"saturation alarm (first {run.saturation['first_alarm_at']:,}"
            + (f"; {count} runs of flagged evaluations)" if count > 1 else ")")
        )
    if run.saved_checkpoints:
        handles.append(Line2D([], [], linestyle="none", marker="^", color=_INK_2, markersize=6))
        labels.append(f"checkpoint kept in models/ ({len(run.saved_checkpoints)})")
    return handles, labels


def _figure_legend(figure: Any, handles: list[Any], labels: list[str], ncol: int) -> None:
    layout = figure._history_layout
    figure.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(layout["left"] / layout["width"], 0.06 / layout["height"]),
        ncol=ncol,
        fontsize=9,
        frameon=False,
    )


def _png(figure: Any, dpi: int) -> bytes:
    """The figure as PNG bytes, quantised to a 256-colour palette when Pillow is importable (line art keeps)."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi, facecolor="white")
    data = buffer.getvalue()
    try:
        from PIL import Image
    except ImportError:
        return data
    image = Image.open(io.BytesIO(data)).convert("RGB")
    quantised = image.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    out = io.BytesIO()
    quantised.save(out, format="PNG", optimize=True)
    return out.getvalue() if len(out.getvalue()) < len(data) else data


@dataclass
class Page:
    """One rendered page."""

    name: str
    title: str
    caption: str
    png: bytes


def _panels(spec: PageSpec, context: _Context) -> list[_Panel]:
    panels: list[_Panel] = []
    for quantity in spec.quantities:
        title = _quantity_title(quantity, context.runs)
        strip = context.comparison and quantity.kind in ("count", "training")
        if strip:
            height = _STRIP_ROW_IN * len(context.runs) + 0.3
        elif quantity.kind in ("count", "training"):
            height = _COUNT_IN
        else:
            height = _LINE_IN
        if context.comparison and quantity.kind in ("column", "series"):
            spacing = _spacing(np.concatenate([run.steps for run in context.runs]), 0.0) if context.runs else 0.0
            title += (
                f" [centred rolling mean over {context.smooth} evaluations"
                + (f", about {_fmt_steps(spacing * context.smooth)} steps" if spacing else "")
                + "; faint: each evaluation]"
                if context.smooth > 1
                else ""
            )
        elif not context.comparison and quantity.spread in _SPREAD_NOTES and quantity.kind == "column":
            title += _SPREAD_NOTES[quantity.spread]
        if strip and quantity.kind == "training":
            title += f" [one cell per window; colour capped at {TRAINING_FALL_CAP:g}]"
        elif strip:
            title += " [one cell per evaluation, never smoothed]"
        if quantity.per_foot and context.comparison:
            feet: list[str] = []
            for run in context.runs:
                feet += [foot for foot in run.feet if foot not in feet]
            panels += [
                _Panel(quantity, foot, f"Foot {foot}: {title}", quantity.unit, height) for foot in (feet or ["r", "l"])
            ]
        else:
            panels.append(_Panel(quantity, None, title, quantity.unit, height, strip))
    return panels


def _page_figure(spec: PageSpec, context: _Context, *, title: str, dpi: int, width: float) -> Any:
    panels = _panels(spec, context)
    handles, labels = _run_handles(context) if context.comparison else _single_handles(context)
    ncol = min(4, len(labels))
    figure, axes = _new_figure(
        panels, width=width, dpi=dpi, title=f"{title}\n{spec.title}", legend_rows=math.ceil(len(labels) / ncol)
    )
    for index, (axis, panel) in enumerate(zip(axes, panels)):
        quantity = panel.quantity
        assert quantity is not None
        _style_axis(axis, panel, top=index == 0)
        axis.set_xlim(0.0, context.xmax * 1.01 if context.xmax > 0 else 1.0)
        ends: list[tuple] = []
        if panel.strip:
            drawn = _draw_strips(axis, context, quantity)
        elif context.comparison:
            ends = _draw_line_compare(axis, context, quantity, panel.foot)
            drawn = bool(ends)
        elif quantity.kind == "count":
            drawn = _draw_count_single(axis, context, quantity)
        elif quantity.kind == "training":
            drawn = _draw_training_single(axis, context, quantity)
        else:
            drawn = _draw_line_single(axis, context, quantity)
        if not drawn:
            column = quantity.key.replace("{foot}", panel.foot or "<foot>")
            _absent(axis, f"not recorded in these files ({column})")
        elif not panel.strip:
            _draw_references(axis, context, quantity)
            if quantity.key == "actuator_force_frac_max":
                axis.set_ylim(0.0, 1.08)
        _draw_marks(axis, context, top=index == 0, strip=panel.strip)
        if context.comparison:
            _end_labels(axis, ends)
        else:
            _legend(axis)
    axes[-1].set_xlabel("training steps (millions)", fontsize=_TICK_PT + 0.5, color=_INK_2)
    _figure_legend(figure, handles, labels, ncol)
    return figure


def _log_ticks(axis: Any) -> None:
    """Plain decimal tick labels on a log axis (``0.2``, not ``2x10^-1``)."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    axis.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 1.5, 2.0, 3.0, 5.0, 7.0)))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.3g}"))
    axis.yaxis.set_minor_formatter(NullFormatter())


def _std_segments(points: Sequence[Mapping[str, Any]]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Index pairs to join: solid between adjacent periodic checkpoints at most :data:`POLICY_STD_JOIN_GAP` times
    the run's periodic spacing apart, a faint dotted connector across every other gap."""
    steps = [p["timesteps"] for p in points]
    periodic = [s for p, s in zip(points, steps) if p["kind"] == "periodic"]
    spacing = _spacing(np.asarray(periodic, dtype=float), 0.0) if len(periodic) > 1 else 0.0
    solid, dotted = [], []
    for i in range(len(points) - 1):
        both = points[i]["kind"] == "periodic" and points[i + 1]["kind"] == "periodic"
        if both and spacing > 0 and steps[i + 1] - steps[i] <= POLICY_STD_JOIN_GAP * spacing:
            solid.append((i, i + 1))
        else:
            dotted.append((i, i + 1))
    return solid, dotted


def _plot_std(
    axis: Any, points: Sequence[Mapping[str, Any]], values: np.ndarray, colour: str, style: Any, label: str | None
) -> None:
    markers = {"periodic": "o", "robust_best": "D", "best": "^", "final": "s"}
    x = np.asarray([p["timesteps"] for p in points], dtype=float) / 1e6
    solid, dotted = _std_segments(points)
    for i, j in dotted:
        axis.plot(x[[i, j]], values[[i, j]], color=colour, linestyle=":", linewidth=0.9, alpha=0.5)
    first = True
    for i, j in solid:
        axis.plot(
            x[[i, j]],
            values[[i, j]],
            color=colour,
            linestyle=style,
            linewidth=1.6,
            label=label if first else "_nolegend_",
        )
        first = False
    for kind, marker in markers.items():
        picks = [k for k, p in enumerate(points) if p["kind"] == kind]
        axis.plot(
            x[picks],
            values[picks],
            linestyle="none",
            marker=marker,
            markersize=5,
            color=colour,
            label=label if first and picks else "_nolegend_",
        )
        first = first and not picks


def _pairs(names: Sequence[str]) -> dict[str, tuple[int | None, int | None]]:
    """``{joint: (right index, left index)}`` of the actuator names (``r_``/``l_`` or ``right_``/``left_``)."""
    found: dict[str, list[int | None]] = {}
    for i, name in enumerate(names):
        stem = _stem(name)
        match = re.match(r"^(r|l|right|left)_(.+)$", stem)
        joint = match.group(2) if match else stem
        entry = found.setdefault(joint, [None, None])
        side = 1 if match and match.group(1) in ("l", "left") else 0
        entry[side] = i
    return {joint: (entry[0], entry[1]) for joint, entry in found.items()}


def _left_equals_right(stds: np.ndarray, pairs: Mapping[str, tuple[int | None, int | None]]) -> bool | None:
    """True when every left actuator's std equals its right one's at every checkpoint; ``None`` with no pair."""
    both = [(right, left) for right, left in pairs.values() if right is not None and left is not None]
    if not both or not stds.size:
        return None
    return all(np.allclose(stds[:, right], stds[:, left], rtol=1e-6, atol=0.0) for right, left in both)


def _std_figure(context: _Context, *, title: str, dpi: int, width: float) -> Any:
    """Per-actuator std at the checkpoints: one run's actuators, or one panel per key actuator across runs."""
    from matplotlib.lines import Line2D

    names = next((run.actuators for run in context.runs if run.actuators), None)
    markers = {"periodic": "o", "robust_best": "D", "best": "^", "final": "s"}
    marker_handles = [Line2D([], [], linestyle="none", marker=m, color=_INK_2) for m in markers.values()]
    marker_labels = [f"{kind} checkpoint" for kind in markers]
    connector = "solid: adjacent periodic checkpoints; dotted: a longer gap (nothing read between)"
    if not context.comparison or names is None:
        run = context.runs[0]
        points = [p for p in run.policy_std.get("checkpoints", []) if p["timesteps"] is not None]
        stds = np.asarray([p["std"] for p in points], dtype=float) if points else np.zeros((0, 0))
        labels = (
            [_stem(n) for n in run.actuators]
            if run.actuators
            else [f"actuator {i}" for i in range(stds.shape[1] if stds.size else 0)]
        )
        pairs = _pairs(labels) if run.actuators else {}
        equal = _left_equals_right(stds, pairs)
        read = f"std from {run.policy_std.get('read', 0)} of {run.policy_std.get('available', 0)} checkpoints"
        side_note = (
            "left = right at every checkpoint read, so one line per pair"
            if equal
            else "right solid, left dashed"
            if equal is False
            else ""
        )
        key = [i for i, n in enumerate(labels) if any(k in n for k in KEY_ACTUATORS)] if run.actuators else []
        rest = [i for i in range(len(labels)) if i not in key]
        groups = (
            (key, f"Legs and tail_3 ({side_note}; colour per joint); {read}"),
            (rest, f"The other actuators ({side_note}); {read}" if side_note else f"The other actuators; {read}"),
        )
        panels = [_Panel(None, None, heading, "std (action units, log)", 3.0) for _, heading in groups]
        handles, legend_labels = _single_handles(context)
        handles += marker_handles
        legend_labels += marker_labels
        ncol = 4
        figure, axes = _new_figure(
            panels,
            width=width,
            dpi=dpi,
            title=f"{title}\nPer-actuator policy std, exp(log_std), at the checkpoints ({connector})",
            legend_rows=math.ceil(len(legend_labels) / ncol),
        )
        for axis_index, (axis, panel, (indices, _)) in enumerate(zip(axes, panels, groups)):
            _style_axis(axis, panel, top=axis_index == 0)
            axis.set_xlim(0.0, context.xmax * 1.01 if context.xmax > 0 else 1.0)
            if not points or not indices:
                _absent(axis, "no per-actuator std: " + (run.policy_std.get("skipped") or "no actuator names"))
                _draw_marks(axis, context, top=axis_index == 0)
                continue
            joints: list[str] = []
            for i in indices:
                stem = labels[i]
                is_left = bool(re.match(r"^(l|left)_", stem))
                joint = re.sub(r"^(r|l|right|left)_", "", stem)
                if is_left and equal:
                    continue
                if joint not in joints:
                    joints.append(joint)
                fixed = next((k for k in KEY_ACTUATORS if k in joint), None) if axis_index == 0 else None
                colour = _KEY_COLOURS[fixed] if fixed else _OTHER_SERIES[joints.index(joint) % len(_OTHER_SERIES)]
                pair = pairs.get(joint, (None, None))
                label = f"{joint} (r = l)" if equal and None not in pair else stem
                _plot_std(axis, points, stds[:, i], colour, "--" if is_left else "-", label)
            axis.set_yscale("log")
            _log_ticks(axis)
            _draw_marks(axis, context, top=axis_index == 0)
            _legend(axis)
        axes[-1].set_xlabel("training steps (millions)", fontsize=_TICK_PT + 0.5, color=_INK_2)
        _figure_legend(figure, handles, legend_labels, ncol)
        return figure

    pairs = _pairs([_stem(n) for n in names])
    all_stds = [
        np.asarray([p["std"] for p in run.policy_std.get("checkpoints", [])], dtype=float)
        for run in context.runs
        if run.actuators == names and run.policy_std.get("checkpoints")
    ]
    verdicts = [_left_equals_right(stds, pairs) for stds in all_stds]
    equal = bool(verdicts) and all(v is True for v in verdicts)

    def order(index: int) -> tuple[int, int, int]:
        stem = _stem(names[index])
        joint = next(i for i, k in enumerate(KEY_ACTUATORS) if k in stem)
        return joint, 1 if re.match(r"^(l|left)_", stem) else 0, index

    chosen = sorted(
        (
            i
            for i, n in enumerate(names)
            if any(k in n for k in KEY_ACTUATORS) and not (equal and re.match(r"^(l|left)_", _stem(n)))
        ),
        key=order,
    )
    reads = "; ".join(
        f"{run.label} {run.policy_std.get('read', 0)}/{run.policy_std.get('available', 0)}" for run in context.runs
    )
    panels = [
        _Panel(None, None, _stem(names[i]) + (" (left = right in every run)" if equal else ""), "std (log)", 2.2)
        for i in chosen
    ]
    handles, labels = _run_handles(context)
    handles += marker_handles
    labels += marker_labels
    ncol = 4
    figure, axes = _new_figure(
        panels,
        width=width,
        dpi=dpi,
        title=f"{title}\nPer-actuator policy std, exp(log_std), at each run's checkpoints ({connector})\n"
        f"std from checkpoints read/available: {reads}",
        legend_rows=math.ceil(len(labels) / ncol),
        columns=2,
        right=0.4,
    )
    for position, (axis, panel, index) in enumerate(zip(axes, panels, chosen)):
        _style_axis(axis, panel, top=position < 2)
        axis.set_xlim(0.0, context.xmax * 1.01 if context.xmax > 0 else 1.0)
        for run_index, run in enumerate(context.runs):
            points = [p for p in run.policy_std.get("checkpoints", []) if p["timesteps"] is not None]
            if not points or run.actuators != names:
                continue
            values = np.asarray([p["std"][index] for p in points], dtype=float)
            _plot_std(axis, points, values, context.colours[run_index], context.styles[run_index], None)
        axis.set_yscale("log")
        _log_ticks(axis)
        _draw_marks(axis, context, top=position < 2)
    for axis in axes[-2:]:
        axis.set_xlabel("training steps (millions)", fontsize=_TICK_PT + 0.5, color=_INK_2)
    _figure_legend(figure, handles, labels, ncol)
    return figure


def _colours(runs: Sequence[RunHistory]) -> tuple[list[str], str]:
    """One colour per run: by seed when every run records one and no (seed, group) repeats, else by position."""
    seeds = [run.seed for run in runs]
    keys = [(run.seed, run.group) for run in runs]
    if all(seed is not None for seed in seeds) and len(set(keys)) == len(keys) and len(set(seeds)) <= len(_SERIES):
        order = sorted(set(seeds))  # type: ignore[type-var]
        return [_SERIES[order.index(seed)] for seed in seeds], "colour per seed, line style per group"
    return [_SERIES[i % len(_SERIES)] for i in range(len(runs))], "colour per run, line style per group"


def _make_context(runs: Sequence[RunHistory], *, marks: Sequence[int], smooth: int) -> _Context:
    """The drawing context of *runs*: one run's pages, or a comparison (colours, line styles, smoothing)."""
    if len(runs) > MAX_RUNS:
        raise ValueError(f"at most {MAX_RUNS} runs per comparison (one categorical colour each), not {len(runs)}")
    comparison = len(runs) > 1
    groups: list[str] = []
    for run in runs:
        if (run.group or "") not in groups:
            groups.append(run.group or "")
    colours, colour_note = _colours(runs)
    return _Context(
        runs=runs,
        marks=marks,
        comparison=comparison,
        smooth=max(1, int(smooth)) if comparison else 1,
        xmax=max((run.last_step for run in runs), default=0) / 1e6,
        colours=colours,
        styles=[_GROUP_STYLES[groups.index(run.group or "") % len(_GROUP_STYLES)] for run in runs],
        colour_note=colour_note,
    )


def render_pages(
    runs: Sequence[RunHistory],
    *,
    title: str,
    marks: Sequence[int],
    smooth: int = 5,
    dpi: int = DEFAULT_DPI,
    width: float = 12.0,
) -> tuple[list[Page], list[str]]:
    """Every page of *runs*: one run's page set, or the comparison set when there are several.

    Returns the pages and the errors (one per page that failed; a failure costs that page only).
    """
    import matplotlib

    context = _make_context(runs, marks=marks, smooth=smooth)
    comparison, colour_note = context.comparison, context.colour_note
    sizes = sorted({n for run in runs for n in run.episodes_per_evaluation})
    episodes = (
        "no evaluation recorded"
        if not sizes
        else f"{sizes[0]} episodes per evaluation"
        if len(sizes) == 1
        else f"{sizes[0]} to {sizes[-1]} episodes per evaluation"
    )
    odd = [size for size in sizes if size != SATURATION_RULE_EPISODES]
    saturation_note = f"; these evaluations hold {', '.join(map(str, sizes))}" if odd else ""
    fields = {
        "episodes": episodes,
        "band": _band_text(runs, "lower", "upper"),
        "pad_band": _band_text(runs, "pad_lower", "pad_upper"),
        "saturation_note": saturation_note,
    }
    pages: list[Page] = []
    errors: list[str] = []
    with matplotlib.rc_context({"font.family": "DejaVu Sans", "figure.facecolor": "white"}):
        for spec in PAGES:
            try:
                figure = _page_figure(spec, context, title=title, dpi=dpi, width=width)
                caption = spec.caption.format(**fields)
                if comparison:
                    caption += f" Runs: {colour_note}; each run's name sits at its line's end."
                pages.append(Page(spec.name, spec.title, caption, _png(figure, dpi)))
            except Exception as exc:  # noqa: BLE001 - a failure costs this page only
                logger.warning("training history page %s failed: %s", spec.name, exc)
                errors.append(f"{spec.name}: {type(exc).__name__}: {exc}")
        try:
            figure = _std_figure(context, title=title, dpi=dpi, width=width)
            sources = sorted({run.actuator_source for run in runs if run.actuator_source})
            caption = (
                "exp(log_std) of each checkpoint's policy.pth (robust_best, best, final and up to "
                f"{POLICY_STD_MAX_PERIODIC} periodic checkpoints kept in models/), in action units, log scale. Solid "
                "lines join adjacent periodic checkpoints only; a dotted connector crosses a longer gap, where "
                "nothing was read. Actuator names: " + ("; ".join(sources) or "none") + "."
            )
            pages.append(Page("10_policy_std", "Per-actuator policy std", caption, _png(figure, dpi)))
        except Exception as exc:  # noqa: BLE001
            logger.warning("training history page 10_policy_std failed: %s", exc)
            errors.append(f"10_policy_std: {type(exc).__name__}: {exc}")
    return pages, errors


# ---------------------------------------------------------------------------------------------- HTML

_CSS = """
:root { --bg:#f9f9f7; --card:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --line:#e1e0d9;
  --accent:#2a78d6; --bad:#d03b3b; color-scheme: light; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#0d0d0d; --card:#1a1a19;
  --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781; --line:#2c2c2a; --accent:#3987e5; --bad:#e66767;
  color-scheme: dark; } }
:root[data-theme="dark"] { --bg:#0d0d0d; --card:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781;
  --line:#2c2c2a; --accent:#3987e5; --bad:#e66767; color-scheme: dark; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width:1260px; margin:0 auto; padding:24px 16px 48px; }
h1 { font-size:24px; margin:0 0 4px; } h2 { font-size:18px; margin:32px 0 8px; } h3 { font-size:15px; margin:20px 0 6px; }
p, li { color:var(--ink2); max-width:80ch; }
.sub { color:var(--ink2); margin:0 0 16px; }
.scroll { overflow-x:auto; }
table { border-collapse:collapse; font-size:13px; font-variant-numeric:tabular-nums; }
th, td { text-align:right; padding:4px 8px; border-bottom:1px solid var(--line); white-space:nowrap; }
th:first-child, td:first-child { text-align:left; }
th { color:var(--ink2); font-weight:600; }
.alarm { color:var(--bad); font-weight:600; }
figure { margin:20px 0; background:#ffffff; border:1px solid var(--line); border-radius:8px; padding:8px; }
figure img { max-width:100%; height:auto; display:block; margin:0 auto; }
figcaption { color:#52514e; font-size:13.5px; padding:6px 4px 2px; max-width:110ch; }
code { font-size:12.5px; }
nav a { color:var(--accent); margin-right:12px; font-size:13.5px; }
"""


def _std_order(key: str) -> tuple[int, int, str]:
    """A std column's place: by joint in :data:`KEY_ACTUATORS`' order, right before left."""
    stem = key[len("std_") :]
    joint = next((i for i, k in enumerate(KEY_ACTUATORS) if k in stem), len(KEY_ACTUATORS))
    return joint, 1 if re.match(r"^(l|left)_", stem) else 0, stem


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "–"
        if abs(value) >= 1000 or (value == int(value) and abs(value) >= 1):
            return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:g}"
        return f"{value:.{digits}g}"
    if isinstance(value, int):
        return f"{value:,}"
    return html.escape(str(value))


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


#: The checkpoint table's columns after the counts: ``(row key, header, scale)``; the per-foot ones are added per run.
_TABLE_VALUES: tuple[tuple[str, str, float], ...] = (
    ("max_sole_cop_fore_aft", "pad rail mean", 1.0),
    ("max_sole_cop_fore_aft_worst", "pad rail worst", 1.0),
    ("max_sole_tilt_deg", "tilt", 1.0),
    ("max_sole_tilt_deg_worst", "tilt worst", 1.0),
    ("max_sole_corner_lift_m", "lift mm", 1000.0),
    ("max_sole_corner_lift_m_worst", "lift worst mm", 1000.0),
)
_COUNT_HEADERS = {
    "episodes": "episodes",
    "full_horizon": "full horizon",
    "clean": "clean",
    "falls": "falls",
    "saturated": "sat. flags",
    "max_sole_cop_fore_aft_over_bar": "over pad bar",
    "foot_cop_outside_band": "outside band",
    "sole_cop_outside_pad_band": "outside pad band",
    "max_sole_tilt_deg_over_bar": "over tilt bar",
    "max_sole_corner_lift_m_over_bar": "over lift bar",
}


def render_html(document: Mapping[str, Any], pages: Sequence[Page], *, json_name: str = JSON_FILENAME) -> str:
    """A self-contained HTML page: the runs, their checkpoint tables, the notes and every page as a base64 PNG."""
    runs = document["runs"]
    title = str(document["title"])
    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{html.escape(title)}</title>",
        f"<style>{_CSS}</style></head><body><main>",
        f"<h1>{html.escape(title)}</h1>",
        '<p class="sub">Training history read back from each stage directory\'s ledgers '
        "(eval_episodes.csv, train_episodes.csv, stage_config.json, progress.json, plant_identity.json, models/). "
        f"Report only: nothing reads this page back. Generated {html.escape(document['generated_at'])}; the numbers "
        f"are in <code>{html.escape(json_name)}</code> ({html.escape(SCHEMA)}).</p>",
        "<nav>" + "".join(f'<a href="#{p.name}">{html.escape(p.title)}</a>' for p in pages) + "</nav>",
        "<h2>Runs</h2>",
    ]
    rows = []
    for run in runs:
        sat = run.get("saturation") or {}
        alarm = sat.get("first_alarm_at")
        rows.append(
            [
                html.escape(str(run["label"])),
                html.escape(str(run.get("group") or "")),
                html.escape(str((run.get("model") or {}).get("display") or "–")),
                _fmt(run.get("seed")),
                html.escape(str(run.get("status") or "unknown")),
                _fmt((run["evaluations"].get("timesteps") or [None])[-1]),
                _fmt(run.get("final_step")),
                html.escape(", ".join(str(n) for n in run.get("episodes_per_evaluation") or []) or "–"),
                _fmt(sat.get("first_saturated_episode_at")),
                _fmt(sat.get("first_flagged_at")),
                f'<span class="alarm">{_fmt(alarm)}</span>' if alarm is not None else "none",
                "yes" if run.get("band_active") else "no",
            ]
        )
    parts.append(
        _table(
            [
                "run",
                "group",
                "model",
                "seed",
                "status",
                "last evaluation",
                "final",
                "episodes per evaluation",
                "first saturated episode",
                "first flagged evaluation",
                "saturation alarm",
                "CoP band paid",
            ],
            rows,
        )
    )
    parts.append(
        "<p>Stage directories: "
        + "; ".join(
            f"<code>{html.escape(str(run['label']))}</code> = <code>{html.escape(run['stage_dir'])}</code>"
            for run in runs
        )
        + f". Marked checkpoints (J): {', '.join(f'{s:,}' for s in document['marks']['resolved']) or 'none'}.</p>"
    )
    parts.append("<h2>At the marked, best and last checkpoints</h2>")
    parts.append(
        "<p>Each row reads the evaluation at exactly that step (the final checkpoint, saved after the last "
        "evaluation, reads the last one). The counts are episodes of that evaluation: <i>over pad bar</i>, "
        "<i>over tilt bar</i> and <i>over lift bar</i> count full-horizon episodes whose own rail value exceeds the "
        "stage's bar; <i>outside band</i> and <i>outside pad band</i> count full-horizon episodes with either foot's "
        "window-mean CoP outside the band's edges. The rail columns are the mean and the worst full-horizon episode. "
        "CoP in half-lengths (+ toe), tilt in degrees, lift in mm, left minus right in degrees, std in action units "
        "at the checkpoint saved at that step.</p>"
    )
    for run in runs:
        table = run.get("checkpoint_table") or []
        if not table:
            continue
        feet = run.get("feet") or []
        present = {key for row in table for key in row}
        cols: list[tuple[str, str, float]] = [
            (key, header, 1.0) for key, header in _COUNT_HEADERS.items() if key in present
        ]
        cols += [(key, header, scale) for key, header, scale in _TABLE_VALUES if key in present]
        cols += [(f"{f}_foot_cop_fore_aft", f"foot CoP {f}", 1.0) for f in feet]
        cols += [(f"{f}_sole_cop_fore_aft", f"pad CoP {f}", 1.0) for f in feet]
        cols += [(f"l_minus_r_{j}_deg", f"L-R {j.replace('_', ' ')}", 1.0) for j in LR_JOINTS]
        cols += [("actuator_force_frac_max", "force frac", 1.0)]
        std_cols = sorted(
            {k for row in table for k in row if k.startswith("std_") and k != "std_checkpoint"}, key=_std_order
        )
        headers = ["at", "evaluation"] + [c[1] for c in cols] + [k[4:] + " std" for k in std_cols]
        body = []
        for row in table:
            cells = [html.escape(row["at"]), _fmt(row["evaluation_timesteps"])]
            for key, _, scale in cols:
                value = row.get(key)
                cells.append(_fmt(value * scale if isinstance(value, (int, float)) else value))
            cells += [_fmt(row.get(k)) for k in std_cols]
            body.append(cells)
        parts.append(f"<h3>{html.escape(str(run['label']))}</h3>")
        parts.append(_table(headers, body))
    parts.append("<h2>What these pages do not show</h2><ul>")
    parts += [f"<li>{html.escape(note)}</li>" for note in document["not_recorded"]]
    parts.append("</ul><h2>How the numbers are reduced</h2><ul>")
    parts += [
        f"<li><b>{html.escape(key)}</b>: {html.escape(str(value))}</li>" for key, value in document["rules"].items()
    ]
    parts.append("</ul>")
    bars_rows = []
    for run in runs:
        bars = run.get("bars") or {}
        band = run.get("band") or {}
        bars_rows.append(
            [html.escape(str(run["label"]))]
            + [_fmt(bars.get(k)) for k in BAR_KEYS]
            + [_fmt(band.get(f"{BAND_PREFIX}{k}")) for k in ("weight", "lower", "upper", "pad_lower", "pad_upper")]
            + [html.escape(str(run.get("band_edges_source") or "none"))]
        )
    parts.append("<h3>Declared bars (stage_config.json curriculum) and the CoP band (reward_weights)</h3>")
    parts.append(
        _table(
            [
                "run",
                *BAR_KEYS,
                "band weight",
                "band lower",
                "band upper",
                "pad band lower",
                "pad band upper",
                "edges counted",
            ],
            bars_rows,
        )
    )
    plant_rows = []
    for run in runs:
        plant = run.get("plant") or {}
        names = (run.get("policy_std") or {}).get("actuator_names_source") or "–"
        plant_rows.append(
            f"<li><code>{html.escape(str(run['label']))}</code>: plant {html.escape(str(plant.get('status')))}"
            + (f" ({html.escape(str(plant['why']))})" if plant.get("why") else "")
            + f"; actuator names: {html.escape(str(names))}</li>"
        )
    parts.append("<h3>Plant and actuator names</h3><ul>" + "".join(plant_rows) + "</ul>")
    absent_rows = [
        f"<li><code>{html.escape(str(run['label']))}</code>: {html.escape('; '.join(run.get('absent') or []) or 'nothing')}"
        + (f' <span class="alarm">errors: {html.escape("; ".join(run["errors"]))}</span>' if run.get("errors") else "")
        + "</li>"
        for run in runs
    ]
    parts.append("<h3>Absent from these files</h3><ul>" + "".join(absent_rows) + "</ul>")
    parts.append("<h2>Pages</h2>")
    for page in pages:
        encoded = base64.b64encode(page.png).decode("ascii")
        parts.append(
            f'<figure id="{page.name}"><img alt="{html.escape(page.title)}" src="data:image/png;base64,{encoded}">'
            f"<figcaption><b>{html.escape(page.title)}.</b> {html.escape(page.caption)}</figcaption></figure>"
        )
    parts.append("</main></body></html>")
    return "\n".join(parts)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write *data* to *path* through a temp file in its directory and ``os.replace``."""
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            mask = os.umask(0)
            os.umask(mask)
            os.fchmod(handle.fileno(), 0o666 & ~mask)
        os.replace(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_report(
    runs: Sequence[RunHistory],
    out_dir: "str | Path",
    *,
    title: str,
    marks: Sequence["int | str"] = (),
    smooth: int = 5,
    dpi: int = DEFAULT_DPI,
) -> dict[str, Any]:
    """Write *out_dir*'s PNG pages, the HTML page embedding them and the JSON summary; returns what was written.

    The caller has checked that *out_dir* is outside every run directory.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    document = summary(runs, title=title, marks=marks, smooth=smooth)
    pages, errors = render_pages(runs, title=title, marks=document["marks"]["resolved"], smooth=smooth, dpi=dpi)
    document["errors"] = errors
    document["pages"] = [f"{page.name}.png" for page in pages]
    written = []
    for page in pages:
        path = out / f"{page.name}.png"
        _atomic_write_bytes(path, page.png)
        written.append(path)
    atomic_write_json(out / JSON_FILENAME, document, indent=None, allow_nan=False)
    atomic_write_text(out / HTML_FILENAME, render_html(document, pages))
    return {
        "html": out / HTML_FILENAME,
        "json": out / JSON_FILENAME,
        "pngs": written,
        "errors": errors,
        "document": document,
    }
