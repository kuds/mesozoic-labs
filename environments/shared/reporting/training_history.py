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
- the saturation-flagged episodes by the actuator each names
  (``saturated_actuator``) and its sign (``saturated_actuator_positive_share``),
  and the saturation fraction itself (``saturated_actuator_fraction``);
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
  (``leg_q_<left>_minus_<right>_<joint>_deg``) and the hip roll's, each
  episode's left leg's ``<left>_hip_roll_mean_deg`` minus its right's
  (both + abduction, so a mirror-symmetric posture reads 0);
- the head posture (``head_tip_height_mean_m``, ``head_tip_height_min_m``,
  ``neck_pitch_mean_deg``, ``head_pitch_mean_deg``) and, per foot, the hip
  roll (``<foot>_hip_roll_mean_deg``, ``<foot>_hip_roll_max_abs_deg``):
  report-only columns no gate reads (``gait.posture_telemetry``);
- the training falls per window of ``train_episodes.csv``;
- the per-actuator policy standard deviation ``exp(log_std)`` at the
  checkpoint zips in ``models/``;
- per training rollout, from ``policy_probe.npz``'s ``rollout_*`` arrays,
  each actuator's share of sampled actions clipped at a bound and of the
  policy's mean actions beyond one: the training-side twin of the
  deterministic evaluation's saturation.

Every per-episode value is reduced over the evaluation's full-horizon episodes,
the set ``gate_progress.npz`` averages over; the outcome counts are over all its
episodes.  A missing file or column is an absent series, listed in the JSON
summary (``mesozoic.training-history/v1``), never an error.

An evaluation is *flagged* when at least :data:`SATURATION_FLAG_EPISODES` of
its episodes list ``max_actuator_saturation_fraction`` among their failed
screen bars, and :data:`SATURATION_ALARM_IN_A_ROW` flagged evaluations in a
row are an *alarm* (the stance4 and stance5 scorers' report-only rule, set on
evaluations of :data:`SATURATION_RULE_EPISODES` episodes).  Where the ledger
records ``saturated_actuator``, each flagged episode names its actuator, and an
alarm is labelled with the actuator its flagged episodes name most (counted
over its signs, with the sign most of them carry).

Counts are never smoothed, nor are the per-rollout shares.  A comparison
draws the counts as one strip per run (a cell per evaluation, coloured by the
count) and smooths only the continuous evaluation series, with each
evaluation's value drawn faintly behind.

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
from ..gait.constants import SATURATION_ABS
from ..gait.posture_telemetry import HEAD_POSTURE_COLUMNS, HIP_ROLL_FIELDS
from ..gait.stance_metrics import SATURATED_ACTUATOR_COLUMNS
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
#: The policy probe's file (``policy_probe.py``); only its per-rollout ``rollout_*`` arrays are read.
POLICY_PROBE_FILENAME = "policy_probe.npz"
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
#: The ledger's saturating-actuator columns read here (``gait.stance_metrics.SATURATED_ACTUATOR_COLUMNS``): the
#: actuator behind each episode's max_actuator_saturation_fraction, that fraction, and the share of its saturated
#: window steps commanded positive.
ACTUATOR_COLUMN, ACTUATOR_FRACTION_COLUMN, ACTUATOR_SHARE_COLUMN = SATURATED_ACTUATOR_COLUMNS[:3]
#: A flagged episode whose row records no actuator: its window was unmeasured (a NaN fraction fails the bar), or the
#: row was written before the column existed (an empty fraction cell: a resumed run's earlier rows).
UNMEASURED_ACTUATOR = "(unmeasured)"
NOT_RECORDED_ACTUATOR = "(not recorded)"
#: The training-fall window: ``progress.json``'s ``training_rollouts.window_timesteps``.
TRAINING_WINDOW_TIMESTEPS = 262_144
#: The per-actuator std reads: at most this many periodic checkpoints (evenly spaced by step), within this budget.
POLICY_STD_MAX_PERIODIC = 12
POLICY_STD_BUDGET_S = 10.0
#: Two periodic checkpoints are joined by a solid line when their gap is at most this many times the run's spacing.
POLICY_STD_JOIN_GAP = 1.5

#: The probe's per-rollout, per-actuator arrays read (``policy_probe.rollout_action_shares``), each (rollouts,
#: actuators); ``rollout_rows`` (the buffer rows each was read over, 0 for none) beside them.
ROLLOUT_ARRAYS: tuple[str, ...] = (
    "rollout_action_clip_high_share",
    "rollout_action_clip_low_share",
    "rollout_action_saturated_share",
    "rollout_mu_beyond_high_share",
    "rollout_mu_beyond_low_share",
    "rollout_mu_mean",
    "rollout_mu_q50",
    "rollout_mu_q95",
)
#: The two shares drawn, each at the high bound (``+``) and the low one (``-``).
ROLLOUT_SHARES: dict[str, tuple[str, str]] = {
    "rollout_action_clip": ("rollout_action_clip_high_share", "rollout_action_clip_low_share"),
    "rollout_mu_beyond": ("rollout_mu_beyond_high_share", "rollout_mu_beyond_low_share"),
}
#: An actuator's share at one bound is drawn by name once it reaches this at any rollout; the rest are summarised.
ROLLOUT_SHARE_LEVEL = 0.01
#: At most this many (actuator, bound) series are named per panel, the highest peak first.
ROLLOUT_NAMED_SERIES = 6

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
#: The saturating-actuator columns reduced per evaluation like the body columns (the fraction is the screen's
#: ``max_actuator_saturation_fraction``; ``actuators_saturated`` counts the actuators with any saturated step).
SATURATION_COLUMNS: tuple[str, ...] = (ACTUATOR_FRACTION_COLUMN, SATURATED_ACTUATOR_COLUMNS[4])
#: The sagittal joints whose left-minus-right achieved offset ``eval_episodes.csv`` records.
LR_JOINTS: tuple[str, ...] = ("hip_pitch", "knee", "ankle")
_LR_COLUMN = re.compile(
    r"^leg_q_(?P<left>[A-Za-z0-9]+)_minus_(?P<right>[A-Za-z0-9]+)_(?P<joint>hip_pitch|knee|ankle)_deg$"
)
_FOOT_COLUMN = re.compile(r"^(?P<foot>[A-Za-z0-9]+)_(?:foot|sole)_cop_fore_aft$")
#: The head posture columns reduced per evaluation (``gait.posture_telemetry``: ledger-only, written last).
POSTURE_COLUMNS: tuple[str, ...] = HEAD_POSTURE_COLUMNS
#: A foot's hip roll columns: ``<foot>_<field>`` for each of these.
HIP_ROLL_SUFFIXES: tuple[str, ...] = HIP_ROLL_FIELDS
_HIP_ROLL_COLUMN = re.compile(r"^(?P<foot>[A-Za-z0-9]+)_hip_roll_mean_deg$")
#: The asymmetry page's joints: the ledger's left-minus-right offsets, then the hip roll's, derived per episode
#: from its two legs' window means (:func:`hip_roll_pairs`).
LR_PAGE_JOINTS: tuple[str, ...] = (*LR_JOINTS, "hip_roll")
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
    "Hip roll and head posture: recorded per episode (<foot>_hip_roll_mean_deg, <foot>_hip_roll_max_abs_deg, "
    "head_tip_height_mean_m, head_tip_height_min_m, neck_pitch_mean_deg, head_pitch_mean_deg) only in ledgers "
    "written since those columns were added; an older run's are absent series. The left-minus-right hip roll is "
    "each episode's left leg's mean minus its right leg's, both + abduction. A model without a head-tip site or a "
    "neck or head pitch hinge reads nan there, and a model without a box-soled foot records none (its evaluation "
    "runs no recorder). No gate reads them.",
    "The evaluation episodes are the deterministic policy's, at yaw 0 on the evaluation env's own resets. The "
    "judge's perturbation panels (its joint-noise falls, K3) and its verdicts are not in these files: the outcomes "
    "page draws the stance5 replication judge's K3 inputs only from the judge files a caller names (--judged), as a "
    "readout of that criterion, never a verdict. The training falls are the stochastic training policy's.",
    "The saturating actuator and the saturation fraction itself are recorded per episode (saturated_actuator, "
    "saturated_actuator_fraction) only in ledgers written since those columns were added; an older run's are absent "
    "series, and its screen_v2_reasons lists max_actuator_saturation_fraction without naming the actuator. An episode "
    "names one actuator, the lowest index on a tie (saturated_actuator_ties counts the others); the other actuators "
    "that saturated are counted (actuators_saturated), not named.",
    "The CoP band is paid per window step on each loaded foot's CoP; these files keep each episode's window mean per "
    "foot only, so the outside-band count reads that mean against the band's edges.",
    "The commanded leg offsets (leg_dc_*) and the per-foot heel moment margin (<foot>_heel_margin_nm) are kept in "
    "gate_progress.npz only, not in eval_episodes.csv.",
    "Per-actuator std is read at the checkpoints kept in models/ only; between them nothing is recorded here "
    "(policy_probe.npz holds it per update where the run kept that file).",
    "The per-rollout clip shares are the training rollouts' (sampled actions, and the policy's means at the same "
    "observations), from policy_probe.npz only where the run kept it and wrote its rollout_* arrays; an older run's "
    "are absent series. The JSON keeps each actuator's peak and last values; the per-rollout series are in the file.",
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

#: The stance5 replication judge's K3 criterion, as its criteria fix it before training (section "CRITERIA, PER
#: RUN", unchanged from stance3 and stance4), drawn as a readout of the judge's files, never as a verdict (the
#: replication's scorer, ``score_stance5r.py``, applies it).
K3_CRITERIA_FILE = "docs/investigations/stance_2026_10/stance5/replication/criteria.txt"
K3_CRITERION = (
    "K3, robustness at 6,291,456 and at final: joint-space noise falls at most 2 of 20, and own-sigma falls at most "
    "1 of 20"
)
#: K3's bars: the joint-space noise falls (``noise_matched_control_joint``) and the own-sigma falls
#: (``noise_matched_own``) allowed of :data:`K3_EPISODES` (spawns 3042-3061; a fall is an episode shorter than the
#: 1000-step horizon), and the checkpoints K3 reads (6,291,456 and the final).
K3_JOINT_MAX_FALLS = 2
K3_OWN_MAX_FALLS = 1
K3_EPISODES = 20
K3_SLOT_STEPS = 6_291_456
#: The judge files read: one per judged checkpoint, ``judge_<steps>.json`` or ``judge_final.json``.
_JUDGE_FILE = re.compile(r"^judge_(?P<slot>\d+|final)\.json$")


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


def hip_roll_labels(header: Sequence[str]) -> list[str]:
    """The foot labels the header records a hip roll for (``<foot>_hip_roll_mean_deg``), in header order."""
    feet: list[str] = []
    for column in header:
        match = _HIP_ROLL_COLUMN.match(column)
        if match is not None and match.group("foot") not in feet:
            feet.append(match.group("foot"))
    return feet


def hip_roll_pairs(labels: Sequence[str]) -> list[tuple[str, str]]:
    """``(left, right)`` foot pairs among *labels*: labels that differ only in a last ``l`` / ``r`` (``l``/``r``,
    ``fl``/``fr``, ``rl``/``rr``), in the left labels' order, as ``eval_diagnostics.sagittal_leg_dc_pairs`` pairs
    the feet."""
    return [(left, left[:-1] + "r") for left in labels if left.endswith("l") and left[:-1] + "r" in labels]


def hip_roll_lr_column(left: str, right: str) -> str:
    """The derived per-episode column of a pair's left-minus-right hip roll (never a ledger column)."""
    return f"hip_roll_{left}_minus_{right}_deg"


def lr_columns(header: Sequence[str]) -> dict[str, str]:
    """``{joint: column}`` of the left-minus-right offsets the header holds, hip roll's derived (the first pair)."""
    found: dict[str, str] = {}
    for column in header:
        match = _LR_COLUMN.match(column)
        if match is not None and match.group("joint") not in found:
            found[match.group("joint")] = column
    pairs = hip_roll_pairs(hip_roll_labels(header))
    if pairs:
        found["hip_roll"] = hip_roll_lr_column(*pairs[0])
    return {joint: found[joint] for joint in LR_PAGE_JOINTS if joint in found}


def reduced_columns(header: Sequence[str]) -> list[str]:
    """The per-episode columns of the header that are reduced per evaluation, in a fixed order.

    The derived left-minus-right hip roll columns (:func:`hip_roll_lr_column`) are not header columns; they are
    reduced beside these (:func:`reduce_evaluations`).
    """
    present = set(header)
    columns = [f"{foot}_{suffix}" for foot in foot_labels(header) for suffix in FOOT_SUFFIXES]
    columns += list(BODY_COLUMNS) + list(SATURATION_COLUMNS)
    columns += [column for joint, column in lr_columns(header).items() if joint != "hip_roll"]
    columns += list(POSTURE_COLUMNS)
    columns += [f"{foot}_{suffix}" for foot in hip_roll_labels(header) for suffix in HIP_ROLL_SUFFIXES]
    return [column for column in columns if column in present]


def expected_columns(feet: Sequence[str]) -> list[str]:
    """The columns this report reads from ``eval_episodes.csv`` (per foot for *feet*)."""
    columns = ["timesteps", "return", "full_horizon", "cause", "screen_v2_clean", "screen_v2_reasons"]
    columns += [f"{foot}_{suffix}" for foot in feet for suffix in FOOT_SUFFIXES]
    columns += list(BODY_COLUMNS) + [ACTUATOR_COLUMN, ACTUATOR_FRACTION_COLUMN, ACTUATOR_SHARE_COLUMN]
    columns += list(POSTURE_COLUMNS) + [f"{foot}_{suffix}" for foot in feet for suffix in HIP_ROLL_SUFFIXES]
    return columns


def saturation_sign(share: float) -> str:
    """``+`` when every saturated window step of the named actuator was commanded positive, ``-`` when none was, else
    ``+/-`` (both; also an unreadable share)."""
    return "+" if share == 1.0 else "-" if share == 0.0 else "+/-"


def named_actuator(row: Mapping[str, str]) -> str:
    """The actuator a saturation-flagged row names, with its sign (``tail_3_pitch_act +``), or why it names none.

    :data:`NOT_RECORDED_ACTUATOR` when the row's fraction cell is empty (the
    row predates the column), :data:`UNMEASURED_ACTUATOR` when it names no
    actuator (an unmeasured window reads NaN, which fails the bar).
    """
    if not (row.get(ACTUATOR_FRACTION_COLUMN) or "").strip():
        return NOT_RECORDED_ACTUATOR
    name = (row.get(ACTUATOR_COLUMN) or "").strip()
    if not name:
        return UNMEASURED_ACTUATOR
    return f"{name} {saturation_sign(_number(row.get(ACTUATOR_SHARE_COLUMN)))}"


def tally(counts: Iterable[Mapping[str, int] | None]) -> dict[str, int]:
    """The per-label sums of *counts*, most first (then by label); ``None`` entries are skipped."""
    total: dict[str, int] = {}
    for entry in counts:
        for label, count in (entry or {}).items():
            total[label] = total.get(label, 0) + int(count)
    return dict(sorted(total.items(), key=lambda item: (-item[1], item[0])))


def _label_parts(label: str) -> tuple[str, str]:
    """A :func:`named_actuator` label's actuator and sign (``("tail_3_pitch_act", "+")``); no sign when it names none."""
    if label in (UNMEASURED_ACTUATOR, NOT_RECORDED_ACTUATOR):
        return label, ""
    name, _, sign = label.rpartition(" ")
    return name, sign


def most_named(counts: Mapping[str, int] | None) -> str | None:
    """The actuator most of *counts*' episodes name, with the sign most of those carry, or ``None`` when there is none.

    Counted by actuator over its signs, so an actuator whose episodes split
    between ``+`` and ``+/-`` is not outnumbered by a rival of one sign.  A tie
    goes to a named actuator before :data:`UNMEASURED_ACTUATOR` and
    :data:`NOT_RECORDED_ACTUATOR`, then to the first by name; the sign is the
    class most of its episodes carry, ``+/-`` on a tie.
    """
    by_name: dict[str, dict[str, int]] = {}
    for label, count in (counts or {}).items():
        if int(count) > 0:
            name, sign = _label_parts(label)
            by_name.setdefault(name, {})[sign] = by_name.get(name, {}).get(sign, 0) + int(count)
    if not by_name:
        return None
    unnamed = (UNMEASURED_ACTUATOR, NOT_RECORDED_ACTUATOR)
    name = min(by_name, key=lambda key: (-sum(by_name[key].values()), key in unnamed, key))
    if name in unnamed:
        return name
    signs = by_name[name]
    leaders = [sign for sign, count in signs.items() if count == max(signs.values())]
    return f"{name} {leaders[0] if len(leaders) == 1 else '+/-'}"


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
    arrays, NaN where the evaluation could not supply them; with the ledger's
    ``saturated_actuator`` beside ``screen_v2_reasons``, ``flagged_actuators``:
    per evaluation those flagged episodes counted by :func:`named_actuator`
    (``None`` where the screen judged none); per rail of
    :data:`RAIL_COLUMNS` whose bar *bars* declares, ``<rail>_over_bar`` (the
    full-horizon episodes whose own value exceeds the bar) and
    ``<rail>_over_bar_fraction`` (over those that measured it); and
    ``columns``: per reduced column its ``mean``, ``std``, ``min``, ``max``
    and ``n`` over the evaluation's full-horizon episodes, with, per foot
    pair recording a hip roll, the left-minus-right hip roll
    (:func:`hip_roll_lr_column`: each episode's left leg's
    ``<foot>_hip_roll_mean_deg`` minus its right leg's).  A count whose
    column is absent is left out.
    """
    present = set(header)
    steps, groups, fh_rows, falls, fh_known = _full_horizon_rows(header, rows)
    out: dict[str, Any] = {"timesteps": np.asarray(steps, dtype=np.int64), "columns": {}}
    if not steps:
        return out
    episodes, clean, saturated, mean_return = [], [], [], []
    named: list[dict[str, int] | None] = []
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
            if ACTUATOR_COLUMN in present:
                flagged = [row for row in judged if SATURATION_REASON in reason_keys(row.get("screen_v2_reasons"))]
                named.append(tally([{named_actuator(row): 1} for row in flagged]) if judged else None)
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
        if ACTUATOR_COLUMN in present:
            out["flagged_actuators"] = named
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
    for left, right in hip_roll_pairs(hip_roll_labels(header)):
        stats = [
            _stats(
                [
                    _number(row.get(f"{left}_hip_roll_mean_deg")) - _number(row.get(f"{right}_hip_roll_mean_deg"))
                    for row in group
                ]
            )
            for group in fh_rows
        ]
        derived = hip_roll_lr_column(left, right)
        out["columns"][derived] = {key: np.asarray([s[key] for s in stats], dtype=float) for key in stats[0]}
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


def read_rollout_shares(path: Path) -> dict[str, np.ndarray] | None:
    """``policy_probe.npz``'s per-rollout arrays (``timesteps``, ``rollout_rows`` and :data:`ROLLOUT_ARRAYS`).

    ``None`` when the file holds no ``rollout_*`` arrays (written before
    them).  A rollout read over no buffer row (a resumed session's rows from
    before the arrays) is NaN.  Only those keys are read; nothing is
    unpickled.
    """
    with np.load(path, allow_pickle=False) as data:
        if not all(key in data.files for key in ("update_timesteps", "rollout_rows", *ROLLOUT_ARRAYS)):
            return None
        shares = {key: np.asarray(data[key], dtype=float) for key in ROLLOUT_ARRAYS}
        shares["timesteps"] = np.asarray(data["update_timesteps"], dtype=np.int64)
        shares["rollout_rows"] = np.asarray(data["rollout_rows"], dtype=np.int64)
    return shares


def _row_max(values: np.ndarray) -> np.ndarray:
    """(U, k) -> (U,): each row's largest finite value, NaN where none is finite."""
    finite = np.isfinite(values)
    largest: np.ndarray = np.where(finite.any(axis=1), np.where(finite, values, -np.inf).max(axis=1), np.nan)
    return largest


def rollout_series(
    shares: Mapping[str, np.ndarray], names: Sequence[str], key: str, *, level: float = ROLLOUT_SHARE_LEVEL
) -> tuple[list[tuple[str, np.ndarray]], np.ndarray | None, int]:
    """One drawn share (:data:`ROLLOUT_SHARES` *key*) split into named series and a summary of the rest.

    Each actuator at each bound (``<name> +`` the high bound, ``<name> -``
    the low one) whose share reaches *level* at some rollout is named, the
    highest peak first (then by actuator and bound), at most
    :data:`ROLLOUT_NAMED_SERIES`.  Returns the named ``(label, series)``,
    the per-rollout maximum over every other series (``None`` when none is
    left) and how many actuators those others cover.
    """
    high, low = (np.asarray(shares[k], dtype=float) for k in ROLLOUT_SHARES[key])
    candidates: list[tuple[float, int, int, str, np.ndarray]] = []
    for index, name in enumerate(names):
        for side, (sign, values) in enumerate((("+", high[:, index]), ("-", low[:, index]))):
            finite = values[np.isfinite(values)]
            peak = float(finite.max()) if finite.size else math.nan
            candidates.append((peak, index, side, f"{name} {sign}", values))
    over = sorted((c for c in candidates if math.isfinite(c[0]) and c[0] >= level), key=lambda c: (-c[0], c[1], c[2]))
    named = over[:ROLLOUT_NAMED_SERIES]
    rest = [c for c in candidates if not any(c is n for n in named)]
    if not rest:
        return [(c[3], c[4]) for c in named], None, 0
    others = _row_max(np.column_stack([c[4] for c in rest]))
    return [(c[3], c[4]) for c in named], others, len({c[1] for c in rest})


def rollout_summary(run: "RunHistory", *, level: float = ROLLOUT_SHARE_LEVEL) -> dict[str, Any]:
    """The JSON's ``rollout_shares``: per actuator, each array's peak and last measured value, and the named series."""
    shares = run.rollout_shares
    if shares is None:
        return {"recorded": False, "why": run.rollout_why}
    rows = shares["rollout_rows"]
    measured = np.flatnonzero(rows > 0)
    steps = shares["timesteps"]
    names = run.rollout_actuators
    last = int(measured[-1]) if measured.size else None
    per_actuator: dict[str, Any] = {}
    for index, name in enumerate(names):
        entry: dict[str, Any] = {"max": {}, "last": {}, "first_at_level": {}}
        for key in ROLLOUT_ARRAYS:
            short = key[len("rollout_") :]
            values = shares[key][:, index]
            finite = values[np.isfinite(values)]
            if key.endswith("_share"):
                entry["max"][short] = float(finite.max()) if finite.size else None
            entry["last"][short] = float(values[last]) if last is not None else None
        for drawn, (high, low) in ROLLOUT_SHARES.items():
            either = np.maximum(shares[high][:, index], shares[low][:, index])
            hits = np.flatnonzero(np.isfinite(either) & (either >= level))
            entry["first_at_level"][drawn[len("rollout_") :]] = int(steps[hits[0]]) if hits.size else None
        per_actuator[name] = entry
    return {
        "recorded": True,
        "source": POLICY_PROBE_FILENAME,
        "rollouts": int(len(steps)),
        "rollouts_measured": int(measured.size),
        "rows_per_rollout": sorted({int(r) for r in rows[measured]}),
        "first_timesteps": int(steps[measured[0]]) if measured.size else None,
        "last_timesteps": int(steps[last]) if last is not None else None,
        "level": level,
        "actuator_names_source": run.rollout_actuator_source,
        "named": {
            key[len("rollout_") :]: [label for label, _ in rollout_series(shares, names, key, level=level)[0]]
            for key in ROLLOUT_SHARES
        },
        "per_actuator": per_actuator,
    }


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


def _head_target(record: Mapping[str, Any]) -> float | None:
    """The head_clearance reward's target height (m) the stage records (``reward_weights``), else ``None``."""
    weights = record.get("reward_weights")
    value = _number(weights.get("head_clearance_target")) if isinstance(weights, Mapping) else math.nan
    return value if math.isfinite(value) else None


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
    #: The feet the ledger records a hip roll for (``<foot>_hip_roll_mean_deg``).
    hip_roll_feet: list[str] = field(default_factory=list)
    lr: dict[str, str] = field(default_factory=dict)
    evaluations: dict[str, Any] = field(
        default_factory=lambda: {"timesteps": np.zeros(0, dtype=np.int64), "columns": {}}
    )
    training: dict[str, Any] | None = None
    saturation: dict[str, Any] = field(default_factory=dict)
    bars: dict[str, float | None] = field(default_factory=dict)
    band: dict[str, float] | None = None
    #: The head_clearance reward's target height (``reward_weights.head_clearance_target``), when recorded.
    head_target: float | None = None
    #: The judge's K3 inputs read from the judge files a caller named (:func:`read_judge_k3`); ``None``: none named.
    k3: dict[str, Any] | None = None
    #: The edges the outside-band counts read: the run's own band, else a reference (``band_edges_source`` says).
    band_edges: dict[str, float] = field(default_factory=dict)
    band_edges_source: str | None = None
    checkpoint_steps: dict[str, int | None] = field(default_factory=dict)
    saved_checkpoints: list[dict[str, Any]] = field(default_factory=list)
    policy_std: dict[str, Any] = field(default_factory=dict)
    plant: dict[str, Any] = field(default_factory=dict)
    actuators: list[str] | None = None
    actuator_source: str = ""
    #: ``policy_probe.npz``'s per-rollout arrays (:func:`read_rollout_shares`; never serialised whole), why they are
    #: absent, and the actuators they are labelled with (names, else ``actuator[<index>]``) and where those came from.
    rollout_shares: dict[str, np.ndarray] | None = field(default=None, repr=False)
    rollout_why: str | None = None
    rollout_actuators: list[str] = field(default_factory=list)
    rollout_actuator_source: str = ""
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
        """The latest step this run's files show: an evaluation, a training episode, a checkpoint, progress.json or a
        judged checkpoint."""
        candidates = [int(self.steps[-1])] if self.steps.size else []
        if self.training is not None:
            candidates.append(int(self.training["window_end"][-1]))
        candidates += [step for step in self.checkpoint_steps.values() if isinstance(step, int)]
        candidates += [c["timesteps"] for c in self.saved_checkpoints if isinstance(c.get("timesteps"), int)]
        if self.progress_timesteps is not None:
            candidates.append(self.progress_timesteps)
        if self.rollout_shares is not None and self.rollout_shares["timesteps"].size:
            candidates.append(int(self.rollout_shares["timesteps"].max()))
        candidates += [p["timesteps"] for p in (self.k3 or {}).get("checkpoints", []) if p.get("timesteps") is not None]
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
    judged: "str | Path | None" = None,
) -> RunHistory:
    """Read and reduce one stage directory's files; a missing file or column is recorded as absent, never raised.

    *judged*: a directory of the run's judge files (``judge_<steps>.json``, ``judge_final.json``), read for the
    stance5 replication judge's K3 inputs (:func:`read_judge_k3`), drawn as a readout on the outcomes page.
    """
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
    run.head_target = _head_target(record)
    if not record:
        run.absent.append("stage_config.json: no bars or band recorded")

    table = _read_csv(stage / EVAL_EPISODES_FILENAME)
    if table is not None:
        header, rows = table
        run.feet = foot_labels(header)
        run.hip_roll_feet = hip_roll_labels(header)
        run.lr = lr_columns(header)
        run.evaluations = reduce_evaluations(header, rows, bars=run.bars)
        run.fh_rows = _full_horizon_rows(header, rows)[2]
        run.absent += [
            f"{EVAL_EPISODES_FILENAME}: column {column}"
            for column in expected_columns(run.feet or run.hip_roll_feet or ["r", "l"])
            if column not in header
        ]
        run.absent += [
            f"{EVAL_EPISODES_FILENAME}: left-minus-right {joint} (leg_q_*_minus_*_{joint}_deg)"
            for joint in LR_JOINTS
            if joint not in run.lr
        ]
        if "hip_roll" not in run.lr:
            run.absent.append(
                f"{EVAL_EPISODES_FILENAME}: left-minus-right hip roll (no <left>/<right>_hip_roll_mean_deg)"
            )
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
        name_alarms(run)
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
    probe = stage / POLICY_PROBE_FILENAME
    if not probe.is_file():
        run.rollout_why = f"no {POLICY_PROBE_FILENAME}"
    else:
        try:
            run.rollout_shares = read_rollout_shares(probe)
        except Exception as exc:  # noqa: BLE001 - an unreadable file is an absent series
            run.errors.append(f"{POLICY_PROBE_FILENAME}: {type(exc).__name__}: {exc}")
            run.rollout_why = f"{POLICY_PROBE_FILENAME} could not be read"
        else:
            if run.rollout_shares is None:
                run.rollout_why = f"{POLICY_PROBE_FILENAME} has no rollout_* arrays (written before them)"
    if run.rollout_shares is not None:
        width = int(run.rollout_shares[ROLLOUT_ARRAYS[0]].shape[1])
        if run.actuators is not None and len(run.actuators) == width:
            labels, run.rollout_actuator_source = list(run.actuators), run.actuator_source
        else:
            found, run.rollout_actuator_source = actuator_names(run.model_id, width, plant=run.plant)
            labels = found or []
        run.rollout_actuators = labels if labels else [f"actuator[{i}]" for i in range(width)]
    else:
        run.absent.append(f"rollout clip shares: {run.rollout_why}")
    checkpoints: Mapping[str, Any] = progress["checkpoints"] if isinstance(progress.get("checkpoints"), Mapping) else {}
    for kind in ("robust_best", "best"):
        step = checkpoints.get(f"{kind}_model_timesteps")
        if run.checkpoint_steps.get(kind) is None and isinstance(step, int) and not isinstance(step, bool):
            run.checkpoint_steps[kind] = step
    if judged is not None:
        run.k3 = read_judge_k3(Path(judged), seed=run.seed, final_step=run.final_step)
        if run.k3.get("why"):
            run.errors.append(f"judge K3: {run.k3['why']}")
    return run


def name_alarms(run: RunHistory) -> None:
    """Label each of *run*'s alarms with the actuators its flagged evaluations' flagged episodes name.

    Each ``run.saturation["runs"]`` entry gets ``actuators`` (the
    :func:`named_actuator` counts over its evaluations, most first) and
    ``named`` (:func:`most_named`: the actuator, over its signs); nothing when
    the ledger records no ``saturated_actuator``.
    """
    named = run.evaluations.get("flagged_actuators")
    if named is None:
        return
    for entry in run.saturation.get("runs", []):
        inside = [counts for step, counts in zip(run.steps.tolist(), named) if entry["from"] <= step <= entry["to"]]
        entry["actuators"] = tally(inside)
        entry["named"] = most_named(entry["actuators"])


# ---------------------------------------------------------------------------------------------- judge K3


def _whole(value: Any) -> int | None:
    """*value* as a whole number (an int, not a bool), else ``None``."""
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _judge_point(path: Path, *, seed: int | None, final_step: int | None) -> tuple[dict[str, Any] | None, str]:
    """One judge file's K3 inputs, or ``(None, why it is not read)``."""
    match = _JUDGE_FILE.match(path.name)
    assert match is not None
    record = read_json_object(path)
    if record is None:
        return None, "unreadable (not a JSON object)"
    if record.get("mode", "full") != "full" or record.get("smoke") is True or record.get("noise_run") is False:
        return None, f"not a full noise run (mode {record.get('mode')!r}, smoke {record.get('smoke')!r})"
    judged_seed = _whole(record.get("seed"))
    if seed is not None and judged_seed is not None and judged_seed != seed:
        return None, f"judged seed {judged_seed}, not the run's {seed}"
    blocks = {}
    for name, key in (("joint", "noise_matched_control_joint"), ("own", "noise_matched_own")):
        block = record.get(key)
        falls = _whole(block.get("falls")) if isinstance(block, Mapping) else None
        episodes = _whole(block.get("n")) if isinstance(block, Mapping) else None
        if falls is None or episodes is None:
            return None, f"no {key} falls and n"
        blocks[name] = (falls, episodes)
    slot = match.group("slot")
    steps: int | None
    if slot == "final":
        checks = record.get("checks")
        recorded = _whole(checks.get("checkpoint_num_timesteps")) if isinstance(checks, Mapping) else None
        steps = recorded if recorded is not None else final_step
    else:
        steps = int(slot)
    k3_slot = slot == "final" or steps == K3_SLOT_STEPS
    (joint, joint_n), (own, own_n) = blocks["joint"], blocks["own"]
    point: dict[str, Any] = {
        "file": path.name,
        "schema": record.get("schema"),
        "slot": slot,
        "timesteps": steps,
        "k3_slot": k3_slot,
        "joint_falls": joint,
        "joint_episodes": joint_n,
        "own_falls": own,
        "own_episodes": own_n,
    }
    if k3_slot:
        point["joint_within_bar"] = joint <= K3_JOINT_MAX_FALLS
        point["own_within_bar"] = own <= K3_OWN_MAX_FALLS
    return point, ""


def read_judge_k3(directory: Path, *, seed: int | None = None, final_step: int | None = None) -> dict[str, Any]:
    """The stance5 replication judge's K3 inputs from *directory*'s judge files: a readout, never a verdict.

    One point per ``judge_<steps>.json`` and ``judge_final.json`` (the judge's
    files per judged checkpoint; its other panels, ``judge_best.json``,
    ``judge_quick_<steps>.json`` and the like, are listed as not read): the
    joint-space noise falls (``noise_matched_control_joint``: ``falls`` of
    ``n``) and the own-sigma falls (``noise_matched_own``).  The final slot
    sits at the judge's recorded ``checks.checkpoint_num_timesteps``, else at
    *final_step*.  A file that is not a full noise run, records another seed
    than *seed*, or lacks the two blocks is skipped with the reason.  At
    K3's checkpoints (:data:`K3_SLOT_STEPS` and the final) each test says
    whether it is within its bar (:data:`K3_JOINT_MAX_FALLS`,
    :data:`K3_OWN_MAX_FALLS`): a readout of the criterion on the judge's own
    numbers; whether a run holds is the scorer's to say (its missing-file,
    refusal and run-handling rules are not applied here).
    """
    out: dict[str, Any] = {
        "source": str(directory),
        "criterion": K3_CRITERION,
        "criteria_file": K3_CRITERIA_FILE,
        "bars": {"joint_space_falls_max": K3_JOINT_MAX_FALLS, "own_sigma_falls_max": K3_OWN_MAX_FALLS},
        "episodes": K3_EPISODES,
        "k3_checkpoints": [K3_SLOT_STEPS, "final"],
        "readout": "the judge's recorded falls against K3's bars, never a verdict",
        "checkpoints": [],
        "skipped": [],
        "not_read": [],
    }
    if not directory.is_dir():
        out["why"] = f"no such directory: {directory}"
        return out
    points: list[dict[str, Any]] = []
    for path in sorted(directory.glob("judge_*.json")):
        if _JUDGE_FILE.match(path.name) is None:
            out["not_read"].append(path.name)
            continue
        try:
            point, why = _judge_point(path, seed=seed, final_step=final_step)
        except Exception as exc:  # noqa: BLE001 - a bad file costs its own point only
            point, why = None, f"{type(exc).__name__}: {exc}"
        if point is None:
            out["skipped"].append({"file": path.name, "why": why})
        else:
            points.append(point)
    points.sort(key=lambda p: (p["timesteps"] is None, p["timesteps"] or 0, p["file"]))
    out["checkpoints"] = points
    sizes = sorted({p[key] for p in points for key in ("joint_episodes", "own_episodes")})
    if any(size != K3_EPISODES for size in sizes):
        out["note"] = f"K3's bars are set on {K3_EPISODES} episodes; these files hold {sizes}, read unchanged"
    if not points:
        out["why"] = f"no judge_<steps>.json or judge_final.json with both noise panels in {directory}"
    return out


def k3_point(run: "RunHistory", step: int) -> dict[str, Any] | None:
    """The judge's K3 point at exactly *step*, or ``None``."""
    return next((p for p in (run.k3 or {}).get("checkpoints", []) if p.get("timesteps") == step), None)


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
    checkpoint saved at that step when one was read.  ``flagged_actuators`` is
    the evaluation's flagged episodes by the actuator they name (where the
    ledger records it), so the finished models (``final``, ``best``,
    ``robust_best``) are read from their own evaluation's rows.
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
        named = run.evaluations.get("flagged_actuators")
        if named is not None:
            row["flagged_actuators"] = named[index]
        for column in [
            f"{foot}_{suffix}" for foot in run.feet for suffix in ("foot_cop_fore_aft", "sole_cop_fore_aft")
        ]:
            values = run.column(column)
            if values is not None:
                row[column] = values[index]
        for column in (
            "max_sole_cop_fore_aft",
            "max_sole_tilt_deg",
            "max_sole_corner_lift_m",
            ACTUATOR_FRACTION_COLUMN,
        ):
            for stat, suffix in (("mean", ""), ("max", "_worst")):
                values = run.column(column, stat)
                if values is not None:
                    row[f"{column}{suffix}"] = values[index]
        for column in ("actuator_force_frac_max", "com_fore_aft_mean_m", *POSTURE_COLUMNS):
            values = run.column(column)
            if values is not None:
                row[column] = values[index]
        lowest = run.column("head_tip_height_min_m", "min")
        if lowest is not None:
            row["head_tip_height_min_m_worst"] = lowest[index]
        for joint, column in run.lr.items():
            values = run.column(column)
            if values is not None:
                row[f"l_minus_r_{joint}_deg"] = values[index]
        judged = k3_point(run, step)
        if judged is not None:
            row["judge_joint_falls"] = judged["joint_falls"]
            row["judge_own_falls"] = judged["own_falls"]
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
            "hip_roll_feet": run.hip_roll_feet,
            "head_clearance_target": run.head_target,
            "left_minus_right_columns": run.lr,
            "evaluations": evaluations,
            "training": run.training,
            "saturation": run.saturation,
            "checkpoint_steps": run.checkpoint_steps,
            "saved_checkpoints": run.saved_checkpoints,
            "plant": run.plant,
            "policy_std": std,
            "rollout_shares": rollout_summary(run),
            "checkpoint_table": checkpoint_rows(run, marks),
            "judge_k3": run.k3,
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
            "flagged_actuators": "per evaluation, the saturation-flagged episodes (as counted for saturated) by the "
            "actuator each names: eval_episodes.csv's saturated_actuator, the actuator with the largest fraction of "
            f"post-settle window steps at |action| >= {SATURATION_ABS:g} (the lowest index on a tie), with its sign: + "
            "when every one of its saturated steps was commanded positive, - when none was, +/- when both; "
            f"{UNMEASURED_ACTUATOR} an unmeasured window, {NOT_RECORDED_ACTUATOR} a row written before the column. "
            "Each alarm lists the counts over its flagged evaluations and names the actuator most of them name, counted "
            "over its signs (a named actuator before (unmeasured) and (not recorded), then the first by name, on a "
            "tie), with the sign most of its episodes carry (+/- on a tie)",
            "training_falls": f"train_episodes.csv episodes whose cause is not truncated, per window of "
            f"{TRAINING_WINDOW_TIMESTEPS:,} steps (k w, (k + 1) w]; the last window ends at its last episode",
            "rollout_shares": f"from {POLICY_PROBE_FILENAME}, per training rollout (the buffer the next PPO update "
            "trains on, read before it) and per actuator: action_clip_high_share and action_clip_low_share, the "
            "sampled actions above the action space's high bound and below its low one (env.step clips them; the "
            "buffer keeps them unclipped); action_saturated_share, the clipped sample at |action| >= "
            f"{SATURATION_ABS:g}; mu_beyond_high_share and mu_beyond_low_share, the policy's mean action at the same "
            "observations beyond each bound; mu_mean, mu_q50 and mu_q95, that mean's average, median and 95th "
            "percentile. Each actuator's peak share and its values at the last measured rollout; first_at_level, the "
            "first rollout whose share at either bound reaches level; named, the (actuator, bound) series the "
            f"pages draw by name (at most {ROLLOUT_NAMED_SERIES}, the highest peak first). Never smoothed",
            "posture": "head_tip_height_mean_m and head_tip_height_min_m: each episode's post-settle window mean and "
            "minimum of the head tip's height (m above z = 0); neck_pitch_mean_deg and head_pitch_mean_deg: the window "
            "means of the neck's pitch (its pitch hinges summed) and the head's, from home, + nose down; "
            "<foot>_hip_roll_mean_deg: the window mean of the leg's hip roll from home, + abduction on either side, and "
            "<foot>_hip_roll_max_abs_deg its window maximum of |angle from home|; left-minus-right hip roll: per "
            "episode, the left leg's mean minus the right's. Reduced like every other column; no gate reads them",
            "judge_k3": f"from the judge files named per run (--judged): per judged checkpoint, the joint-space noise "
            f"falls (noise_matched_control_joint) and the own-sigma falls (noise_matched_own) of the judge's "
            f"{K3_EPISODES} episodes, against the stance5 replication judge's K3 ({K3_CRITERIA_FILE}: "
            f"{K3_CRITERION}); within_bar at K3's checkpoints only. A readout of the judge's records, never a "
            "verdict: the scorer decides whether a run holds",
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

    ``kind``: ``column`` (a reduced column, ``{foot}`` for each foot: the
    feet with a centre of pressure, or with a hip roll for a hip roll key),
    ``series`` (a continuous evaluation series), ``count`` (a per-evaluation
    count: a line on one run's page, a strip per run on a comparison, never
    smoothed), ``training`` (the training-fall fraction per window, likewise),
    ``actuators`` (``flagged_actuators``: stacked counts per actuator on one
    run's page, a strip per run coloured by the actuator most named on a
    comparison, never smoothed) or ``rollout`` (a :data:`ROLLOUT_SHARES`
    share per training rollout: the named actuators' lines and the rest's
    maximum on one run's page, each run's largest actuator on a comparison,
    never smoothed), or ``k3`` (the judge's K3 inputs at each judged
    checkpoint, from the files a caller named: drawn only when some run has
    them).
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
    (the band edges counted), ``{saturation_note}`` and ``{named_note}`` (the
    runs whose ledger names no actuator).
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
        "smoothed: a comparison draws one strip per run, a cell per evaluation (per window for training falls)."
        "{k3_note}",
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
            Quantity(
                "k3",
                "Judge K3 readout, never a verdict: the stance5 replication judge's joint-space noise falls and "
                f"own-sigma falls of its {K3_EPISODES} episodes at each judged checkpoint, against K3's bars "
                f"({K3_CRITERION})",
                f"falls of {K3_EPISODES}",
                "k3",
                None,
            ),
        ),
    ),
    PageSpec(
        "02_saturation",
        "Actuator saturation and force",
        "A saturation flag is an episode whose screen_v2_reasons lists max_actuator_saturation_fraction: some actuator "
        "was commanded at |action| >= 0.99 on more than the stage's bar fraction of its post-settle window steps (the "
        "metric is the largest per-actuator fraction). An evaluation with >= 5 such episodes is flagged (the scorers' "
        "rule, set on 10-episode evaluations{saturation_note}); two flagged evaluations in a row are an alarm (dotted "
        "line; on a comparison's strip, a triangle). Each flagged episode names its actuator, eval_episodes.csv's "
        "saturated_actuator: the one with the largest fraction (the lowest index on a tie), with its sign (+: every "
        "saturated step commanded +1; -: every one -1; +/-: both){named_note}; an alarm is labelled with the actuator "
        "its flagged episodes name most (counted over its signs, with the sign most of them carry; a comparison's cell "
        "likewise per evaluation). saturated_actuator_fraction is that largest fraction, the metric the bar reads. "
        "actuator_force_frac_max is about force, not the action command. The last two panels are the training "
        "rollouts' (policy_probe.npz): the stochastic policy's samples, which env.step clips to the action bounds "
        "(the rollout buffer keeps them unclipped), and the policy's mean action at the same observations, each "
        "actuator's share per rollout at the high bound (+, solid) and the low one (-, dashed); the deterministic "
        "evaluation acts on that mean, clipped. An actuator at a bound is named once its share reaches "
        f"{ROLLOUT_SHARE_LEVEL:g} at some rollout (at most {ROLLOUT_NAMED_SERIES}, the highest peak first, in its "
        "flagged episodes' colour above, else one no flagged actuator holds); grey: the largest share of every other "
        "actuator and bound. A comparison draws each run's largest actuator share per rollout, its two bounds summed "
        "(as policy_probe/rollout_*_share_max in TensorBoard), with the actuator it traces at its peak beside the "
        "run's name. Never smoothed{rollout_note}.",
        (
            Quantity(
                "saturated",
                "Saturation-flagged episodes per evaluation (>= 5 flags the evaluation)",
                "episodes",
                "count",
            ),
            Quantity(
                "flagged_actuators",
                "Saturation-flagged episodes per evaluation by the actuator each names, with its sign "
                "(saturated_actuator)",
                "episodes",
                "actuators",
            ),
            Quantity(
                ACTUATOR_FRACTION_COLUMN,
                "saturated_actuator_fraction: each episode's largest per-actuator fraction of window steps at "
                f"|action| >= {SATURATION_ABS:g} (max_actuator_saturation_fraction); {_FH}",
                "fraction of window steps",
                spread="max",
            ),
            Quantity(
                "actuator_force_frac_max",
                "actuator_force_frac_max: the highest window mean of |F| / limit over the force-limited actuators "
                f"(limit 1.0; the saturation flags are about the action command, not force); {_FH}",
                "fraction of limit",
                spread="max",
            ),
            Quantity(
                "rollout_action_clip",
                "Training rollouts: share of each actuator's sampled actions clipped at a bound (+ the high bound, "
                "- the low one), per rollout",
                "share of rollout steps",
                "rollout",
                None,
            ),
            Quantity(
                "rollout_mu_beyond",
                "Training rollouts: share of each actuator's policy mean actions beyond a bound (+ above the high "
                "bound, - below the low one), per rollout",
                "share of rollout steps",
                "rollout",
                None,
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
        "Left-right asymmetry of the legs: hip pitch, knee, ankle and hip roll",
        "The achieved joint angle offset from the home pose, left minus right, each side signed about the animal's "
        "leftward axis (leg_q_<left>_minus_<right>_<joint>_deg); a mirror-symmetric posture reads 0. Hip roll: each "
        "episode's left leg's <left>_hip_roll_mean_deg minus its right leg's, both signed + abduction (the foot away "
        "from the midline), so a mirror-symmetric posture reads 0 too and + means the left leg is the more abducted "
        "(a lean to the right, say); the first foot pair on a quadruped. Hip roll is recorded only in ledgers "
        "written since its columns were added{posture_note}.",
        tuple(
            Quantity(
                f"lr:{joint}",
                f"{joint.replace('_', ' ').capitalize()}: left minus right"
                + (" (each leg + abduction)" if joint == "hip_roll" else "")
                + f"; {_FH}",
                "degrees",
            )
            for joint in LR_PAGE_JOINTS
        ),
    ),
)
#: The pages drawn after the policy std page (``10_policy_std``), so the earlier pages keep their names: the head
#: posture and the hip roll per leg (``gait.posture_telemetry``), report-only columns no gate reads.
POSTURE_PAGES: tuple[PageSpec, ...] = (
    PageSpec(
        "11_head_posture",
        "Head posture: head-tip height, neck and head pitch",
        "Report only: no gate criterion reads where the head is (gait.posture_telemetry). Over each episode's "
        "post-settle window: the head tip's height (the model's head_tip site, else snout_tip, in m above the floor "
        "plane, the env's own head_tip_z), its mean and its minimum; the neck's pitch (its pitch hinges summed) and "
        "the head's on the neck, each from its home angle, + nose down. Dotted: the head_clearance reward's target "
        "height, where the stage records one ({head_target}). Recorded only in ledgers written since these columns "
        "were added{posture_note}.",
        (
            Quantity("head_tip_height_mean_m", f"Head-tip height, window mean; {_FH}", "m"),
            Quantity(
                "head_tip_height_min_m",
                f"Head-tip height, window minimum; {_FH}",
                "m",
                spread="min",
            ),
            Quantity("neck_pitch_mean_deg", f"Neck pitch (its hinges summed), window mean; {_FH}", "deg (+ nose down)"),
            Quantity("head_pitch_mean_deg", f"Head pitch on the neck, window mean; {_FH}", "deg (+ nose down)"),
        ),
    ),
    PageSpec(
        "12_hip_roll",
        "Hip roll, per leg",
        "Report only: no gate criterion reads the hip roll (gait.posture_telemetry). Each leg's hip roll hinge "
        "from its home angle over the post-settle window, signed + abduction (the foot away from the midline) on "
        "either side, so a mirror-symmetric posture reads the same on both legs: the window mean, and the window "
        "maximum of the absolute angle from home (unsigned). The left-minus-right difference is on the asymmetry "
        "page. Recorded only in ledgers written since these columns were added{posture_note}.",
        (
            Quantity(
                "{foot}_hip_roll_mean_deg",
                f"Hip roll, window mean; {_FH}",
                "degrees (+ abduction)",
                per_foot=True,
            ),
            Quantity(
                "{foot}_hip_roll_max_abs_deg",
                f"Hip roll, window maximum of |angle from home|; {_FH}",
                "degrees",
                spread="max",
                per_foot=True,
            ),
        ),
    ),
)


def page_names() -> list[str]:
    """Every page's name in the order a report draws them: :data:`PAGES`, ``10_policy_std``, :data:`POSTURE_PAGES`."""
    return [spec.name for spec in PAGES] + ["10_policy_std"] + [spec.name for spec in POSTURE_PAGES]


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


def _quantity_feet(run: RunHistory, quantity: Quantity) -> list[str]:
    """The feet a per-foot quantity is drawn for: those with a hip roll for a hip roll key, else a CoP."""
    return run.hip_roll_feet if "hip_roll" in quantity.key else run.feet


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
    elif key in ("saturated", "flagged_actuators"):
        lines.append((float(SATURATION_FLAG_EPISODES), f"flag: >= {SATURATION_FLAG_EPISODES} episodes", "--"))
    elif key == ACTUATOR_FRACTION_COLUMN:
        for value in _common(run.bars.get(SATURATION_REASON) for run in runs):
            lines.append((value, f"{SATURATION_REASON} bar {value:g}", "--"))
    elif key == "actuator_force_frac_max":
        lines.append((1.0, "limit 1.0", "--"))
    elif key in ROLLOUT_SHARES:
        lines.append((ROLLOUT_SHARE_LEVEL, f"named from {ROLLOUT_SHARE_LEVEL:g}", ":"))
    elif key in ("head_tip_height_mean_m", "head_tip_height_min_m"):
        for value in _common(run.head_target for run in runs):
            lines.append((value, f"head_clearance reward target {value:g} m", ":"))
    elif key == "k3":
        lines.append((float(K3_JOINT_MAX_FALLS), f"K3 bar: joint-space falls <= {K3_JOINT_MAX_FALLS}", "--"))
        lines.append((float(K3_OWN_MAX_FALLS), f"K3 bar: own-sigma falls <= {K3_OWN_MAX_FALLS}", ":"))
    if key in ("neck_pitch_mean_deg", "head_pitch_mean_deg", "{foot}_hip_roll_mean_deg"):
        lines.append((0.0, "", "-"))
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
    figure._history_layout = {  # type: ignore[attr-defined]
        "width": width,
        "height": total,
        "left": left,
        "right": right,
        "bottom": bottom_in,
    }
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
    #: The colour of each label ``flagged_actuators`` names, over every run (:func:`_named_colours`).
    named_colours: dict[str, str] = field(default_factory=dict)


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


def _colourbar(axis: Any, cmap: Any, norm: Any, *, training: bool, top: int, over: bool = False) -> None:
    """A colour key beside *axis*; *over*: the top count is a cap (labelled ``>= top``)."""
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
        bar.set_ticklabels([f">= {t}" if over and t == top else str(t) for t in ticks])
    cax.set_title("fraction" if training else "episodes", fontsize=8, color=_INK_2, loc="left")


def _draw_line_single(axis: Any, context: _Context, quantity: Quantity) -> bool:
    run = context.runs[0]
    feet: list[str | None] = list(_quantity_feet(run, quantity)) if quantity.per_foot else [None]
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


def _named_label(context: _Context, label: str) -> str:
    """The legend label a named actuator is drawn under: its own, or :data:`_OTHER_ACTUATORS` past the coloured ones."""
    return _OTHER_ACTUATORS if context.named_colours.get(label) == _MUTED else label


def _draw_actuators_single(axis: Any, context: _Context, quantity: Quantity) -> bool:
    """Per evaluation, its flagged episodes stacked by the actuator each names (raw counts, a bar per evaluation)."""
    run = context.runs[0]
    named = run.evaluations.get("flagged_actuators")
    if named is None:
        return False
    x = run.steps / 1e6
    width = 0.8 * _spacing(x, 0.1)
    bottom = np.zeros(len(named))
    totals = tally(named)
    drawn: set[str] = set()
    for label in context.named_colours:
        heights = np.asarray([float((counts or {}).get(label, 0)) for counts in named])
        some = heights > 0
        if not some.any():
            continue
        shown = _named_label(context, label)
        legend = f"{shown} ({totals[label]})" if shown == label else shown
        axis.bar(
            x[some],
            heights[some],
            width=width,
            bottom=bottom[some],
            color=context.named_colours[label],
            linewidth=0,
            label=legend if shown not in drawn else "_nolegend_",
        )
        drawn.add(shown)
        bottom += heights
    episodes = max(run.episodes_per_evaluation or [10])
    axis.set_ylim(-0.4, episodes + 0.4)
    if not drawn:
        axis.text(
            0.5,
            0.5,
            "no saturation-flagged episode in these evaluations",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=9.5,
            color=_MUTED,
        )
    return True


def _draw_actuator_strips(axis: Any, context: _Context, quantity: Quantity) -> bool:
    """One row per run, a cell per evaluation with flagged episodes, coloured by the actuator most of them name."""
    from matplotlib.transforms import blended_transform_factory

    drawn = False
    for row, run in enumerate(context.runs):
        named = run.evaluations.get("flagged_actuators")
        if named is None:
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
        x = run.steps / 1e6
        width = _spacing(x, 0.1)
        for xi, counts in zip(x, named):
            label = most_named(counts)
            if label is None:
                continue
            axis.bar(xi, 0.8, width=width, bottom=row - 0.4, color=context.named_colours[label], linewidth=0)
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
    return drawn


def _k3_points(run: RunHistory) -> list[dict[str, Any]]:
    """The judge's K3 points of *run* that sit at a step (the final one needs the judge's or the run's step)."""
    return [p for p in (run.k3 or {}).get("checkpoints", []) if p.get("timesteps") is not None]


def _draw_k3_single(axis: Any, context: _Context, quantity: Quantity) -> bool:
    """The judge's joint-space (circles) and own-sigma (squares) falls per judged checkpoint, K3's filled."""
    points = _k3_points(context.runs[0])
    if not points:
        return False
    x = np.asarray([p["timesteps"] for p in points], dtype=float) / 1e6
    slot = np.asarray([p["k3_slot"] for p in points], dtype=bool)
    top = 0.0
    for key, marker, colour, size, name in (
        ("joint_falls", "o", _SERIES[0], 7.5, "joint-space noise falls"),
        ("own_falls", "s", _SERIES[1], 4.5, "own-sigma falls"),
    ):
        y = np.asarray([p[key] for p in points], dtype=float)
        top = max(top, float(y.max()))
        axis.plot(x, y, color=colour, linewidth=0.8, alpha=0.5, label="_nolegend_")
        axis.plot(
            x[slot], y[slot], linestyle="none", marker=marker, markersize=size, color=colour, label=f"{name} (K3)"
        )
        axis.plot(
            x[~slot],
            y[~slot],
            linestyle="none",
            marker=marker,
            markersize=size,
            markerfacecolor="white",
            markeredgecolor=colour,
            label=f"{name} (report only)",
        )
    for xi, point in zip(x[slot], [p for p, k in zip(points, slot) if k]):
        axis.annotate(
            f"{point['joint_falls']}/{point['own_falls']}",
            (xi, max(point["joint_falls"], point["own_falls"])),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color=_INK,
        )
    axis.set_ylim(-0.6, max(float(K3_JOINT_MAX_FALLS) + 2.0, top + 2.0))
    return True


def _draw_k3_strips(axis: Any, context: _Context, quantity: Quantity) -> bool:
    """One row per run, a cell per judged checkpoint (``joint/own`` written in it), K3's checkpoints outlined."""
    from matplotlib.patches import Rectangle
    from matplotlib.transforms import blended_transform_factory

    cap = K3_JOINT_MAX_FALLS + 3  # the colour tops out a little over K3's bar; the counts are written in the cells
    cmap, norm = _strip_cmap("orange", cap)
    every = np.concatenate([np.asarray([p["timesteps"] for p in _k3_points(run)], dtype=float) for run in context.runs])
    width = 0.45 * _spacing(every / 1e6, 0.5) if every.size else 0.5
    drawn = False
    transform = blended_transform_factory(axis.transAxes, axis.transData)
    for row, run in enumerate(context.runs):
        points = _k3_points(run)
        if not points:
            axis.text(0.005, row, "no judge files", transform=transform, va="center", fontsize=8.5, color=_MUTED)
            continue
        drawn = True
        for point in points:
            xi = point["timesteps"] / 1e6
            shade = cmap(norm(min(point["joint_falls"], cap)))
            axis.bar(xi, 0.8, width=width, bottom=row - 0.4, color=shade, linewidth=0)
            if point["k3_slot"]:
                axis.add_patch(
                    Rectangle(
                        (xi - width / 2, row - 0.4), width, 0.8, fill=False, edgecolor=_INK, linewidth=1.0, zorder=3
                    )
                )
            colour = "white" if point["joint_falls"] >= 0.6 * cap else _INK
            axis.text(
                xi,
                row,
                f"{point['joint_falls']}/{point['own_falls']}",
                ha="center",
                va="center",
                fontsize=7.5,
                color=colour,
            )
    axis.set_ylim(len(context.runs) - 0.5, -0.75)
    axis.set_yticks(range(len(context.runs)))
    axis.set_yticklabels([])
    axis.tick_params(axis="y", length=0)
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
    _colourbar(axis, cmap, norm, training=False, top=cap, over=True)
    return drawn


def _rollout_gap(run: RunHistory) -> str:
    """The rollouts of *run* its ``rollout_*`` arrays did not measure (0 buffer rows), in words; ``""`` for none.

    The leading ones are a resumed session's rows from a file written before the arrays; a later one is a rollout
    whose buffer pass failed.
    """
    shares = run.rollout_shares
    if shares is None:
        return ""
    measured = np.flatnonzero(shares["rollout_rows"] > 0)
    if not measured.size:
        return "no rollout measured"
    leading = int(measured[0])
    later = len(shares["rollout_rows"]) - leading - measured.size
    gaps = [f"not measured before {_fmt_steps(int(shares['timesteps'][leading]))}"] if leading else []
    if later:
        gaps.append(f"{later} later rollout{'s' if later > 1 else ''} not measured")
    return "; ".join(gaps)


def _rollout_axis(axis: Any, peak: float) -> None:
    """A share axis from 0 to a little above *peak*: at least 0.05, at most 1."""
    axis.set_ylim(0.0, min(1.0, max(0.05, 1.15 * peak)) if math.isfinite(peak) else 0.05)


def _rollout_colours(context: _Context, labels: Sequence[str]) -> dict[str, str]:
    """A colour per actuator the named *labels* draw: its flagged-episode colour where it has one, else a free one.

    A free colour is one no flagged label on the page holds, so a colour names one actuator across the page; when
    the flagged labels hold them all, one this panel does not use (which some flagged label above also holds),
    then the mid grey.
    """
    colours: dict[str, str] = {}
    for label in labels:
        name = label.rsplit(" ", 1)[0]
        flagged = [context.named_colours.get(f"{name} {sign}") for sign in ("+", "-", "+/-")]
        colours.setdefault(name, next((c for c in flagged if c is not None and c != _MUTED), ""))
    taken = set(context.named_colours.values())
    free = [colour for colour in _NAMED_SERIES if colour not in taken]
    free += [colour for colour in _NAMED_SERIES if colour in taken and colour not in colours.values()]
    spare = iter(free)
    return {name: colour or next(spare, _MUTED) for name, colour in colours.items()}


def _draw_rollout_single(axis: Any, context: _Context, quantity: Quantity) -> bool:
    """One run's named (actuator, bound) shares per rollout (+ solid, - dashed) and the rest's maximum in grey."""
    run = context.runs[0]
    shares = run.rollout_shares
    if shares is None:
        return False
    x = shares["timesteps"] / 1e6
    named, others, count = rollout_series(shares, run.rollout_actuators, quantity.key)
    colours = _rollout_colours(context, [label for label, _ in named])
    peak = 0.0
    for label, values in named:
        name, sign = label.rsplit(" ", 1)
        axis.plot(x, values, color=colours[name], linestyle="-" if sign == "+" else "--", linewidth=1.4, label=label)
        peak = max(peak, float(np.nanmax(values)))
    if others is not None:
        what = "the rest, largest" if named else f"all {count} actuators, largest"
        axis.plot(x, others, color=_MUTED, linewidth=1.0, label=what)
        finite = others[np.isfinite(others)]
        peak = max(peak, float(finite.max()) if finite.size else 0.0)
    _rollout_axis(axis, peak)
    gap = _rollout_gap(run)
    if gap:
        axis.text(0.006, 0.94, gap, transform=axis.transAxes, fontsize=8.5, va="top", color=_MUTED)
    if not named:
        axis.text(
            0.5,
            0.6,
            f"no actuator reaches {ROLLOUT_SHARE_LEVEL:g} at either bound in any rollout",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=9.5,
            color=_MUTED,
        )
    return True


def _draw_rollout_compare(axis: Any, context: _Context, quantity: Quantity) -> list[tuple]:
    """Each run's largest actuator share per rollout, its two bounds summed (never smoothed), with its peak's actuator."""
    ends: list[tuple] = []
    peak = 0.0
    unrecorded = [run.label for run in context.runs if run.rollout_shares is None]
    notes = [f"not recorded: {', '.join(unrecorded)}"] if unrecorded else []
    notes += [f"{run.label}: {gap}" for run in context.runs if (gap := _rollout_gap(run))]
    for row, note in enumerate(notes):
        axis.text(0.006, 0.94 - 0.09 * row, note, transform=axis.transAxes, fontsize=8.5, va="top", color=_MUTED)
    for index, run in enumerate(context.runs):
        shares = run.rollout_shares
        if shares is None:
            continue
        high, low = (np.asarray(shares[key], dtype=float) for key in ROLLOUT_SHARES[quantity.key])
        values = _row_max(high + low)
        x = shares["timesteps"] / 1e6
        colour, style = context.colours[index], context.styles[index]
        axis.plot(x, values, color=colour, linestyle=style, linewidth=1.3, label="_nolegend_")
        finite = np.flatnonzero(np.isfinite(values))
        if not finite.size:
            continue
        top = int(finite[np.argmax(values[finite])])
        peak = max(peak, float(values[top]))
        # The actuator the line traces at its peak rollout (the lowest index on a tie), once it reaches the level.
        actuator = run.rollout_actuators[int(np.nanargmax(high[top] + low[top]))]
        named = f" ({actuator})" if values[top] >= ROLLOUT_SHARE_LEVEL else ""
        ends.append((float(x[finite[-1]]), float(values[finite[-1]]), f"{run.label}{named}", colour, style))
    if ends:
        _rollout_axis(axis, peak)
    return ends


def _named_handles(context: _Context) -> tuple[list[Any], list[str]]:
    """The key of the actuator colours a comparison's strips draw, for the figure legend (the strip has no room)."""
    from matplotlib.patches import Patch

    drawn = {most_named(counts) for run in context.runs for counts in run.evaluations.get("flagged_actuators") or []}
    shown: dict[str, str] = {}
    for label, colour in context.named_colours.items():
        if label in drawn:
            shown.setdefault(_named_label(context, label), colour)
    return [Patch(color=colour) for colour in shown.values()], [f"flagged: {label}" for label in shown]


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
    # The lines and reference lines only request an autoscale, applied lazily: settle the view limits before the
    # transform reads them, or a label sits where the line would be under the limits before them.
    axis.autoscale_view()
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
                named = f"\n({entry['named']})" if entry.get("named") else ""
                axis.text(
                    x,
                    0.97,
                    f"{text} {_fmt_steps(entry['alarm_at'])}{named} ",
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
        named = (run.saturation.get("runs") or [{}])[0].get("named") if run.saturation.get("alarm") else None
        labels.append(
            f"{run.label}" + (f" ({run.group})" if run.group else "") + (f"; first alarm: {named}" if named else "")
        )
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
        named = run.saturation["runs"][0].get("named")
        labels.append(
            f"saturation alarm (first {run.saturation['first_alarm_at']:,}"
            + (f", {named}" if named else "")
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
        if quantity.kind == "k3" and not any(_k3_points(run) for run in context.runs):
            continue  # drawn only from the judge files a caller named
        title = _quantity_title(quantity, context.runs)
        strip = context.comparison and quantity.kind in ("count", "training", "actuators", "k3")
        if strip:
            height = _STRIP_ROW_IN * len(context.runs) + 0.3
        elif quantity.kind in ("count", "training", "actuators", "k3"):
            height = _COUNT_IN
        else:
            height = _LINE_IN
        if context.comparison and quantity.kind == "rollout":
            title += " [each run's largest actuator share per rollout, its two bounds summed, never smoothed]"
        elif context.comparison and quantity.kind in ("column", "series"):
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
        if strip and quantity.kind == "k3":
            title += (
                " [one cell per judged checkpoint, joint-space/own-sigma falls written in it and coloured by the "
                "joint-space falls; outlined: the checkpoints K3 reads]"
            )
        elif strip and quantity.kind == "training":
            title += f" [one cell per window; colour capped at {TRAINING_FALL_CAP:g}]"
        elif strip and quantity.kind == "actuators":
            title += " [one cell per evaluation with a flagged episode, coloured by the actuator most of them name]"
        elif strip:
            title += " [one cell per evaluation, never smoothed]"
        if quantity.per_foot and context.comparison:
            feet: list[str] = []
            for run in context.runs:
                feet += [foot for foot in _quantity_feet(run, quantity) if foot not in feet]
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
    if context.comparison and any(quantity.kind == "actuators" for quantity in spec.quantities):
        named_handles, named_labels = _named_handles(context)
        handles, labels = handles + named_handles, labels + named_labels
        ncol = min(3, len(labels))  # the actuator names are long: three columns fit the page's width
    figure, axes = _new_figure(
        panels, width=width, dpi=dpi, title=f"{title}\n{spec.title}", legend_rows=math.ceil(len(labels) / ncol)
    )
    for index, (axis, panel) in enumerate(zip(axes, panels)):
        quantity = panel.quantity
        assert quantity is not None
        _style_axis(axis, panel, top=index == 0)
        axis.set_xlim(0.0, context.xmax * 1.01 if context.xmax > 0 else 1.0)
        ends: list[tuple] = []
        if panel.strip and quantity.kind == "actuators":
            drawn = _draw_actuator_strips(axis, context, quantity)
        elif panel.strip and quantity.kind == "k3":
            drawn = _draw_k3_strips(axis, context, quantity)
        elif quantity.kind == "k3":
            drawn = _draw_k3_single(axis, context, quantity)
        elif panel.strip:
            drawn = _draw_strips(axis, context, quantity)
        elif quantity.kind == "rollout" and context.comparison:
            ends = _draw_rollout_compare(axis, context, quantity)
            drawn = bool(ends)
        elif quantity.kind == "rollout":
            drawn = _draw_rollout_single(axis, context, quantity)
        elif context.comparison:
            ends = _draw_line_compare(axis, context, quantity, panel.foot)
            drawn = bool(ends)
        elif quantity.kind == "count":
            drawn = _draw_count_single(axis, context, quantity)
        elif quantity.kind == "actuators":
            drawn = _draw_actuators_single(axis, context, quantity)
        elif quantity.kind == "training":
            drawn = _draw_training_single(axis, context, quantity)
        else:
            drawn = _draw_line_single(axis, context, quantity)
        if not drawn:
            column = (
                ACTUATOR_COLUMN
                if quantity.kind == "actuators"
                else f"{POLICY_PROBE_FILENAME} {quantity.key}_*"
                if quantity.kind == "rollout"
                else "the left and right <foot>_hip_roll_mean_deg"
                if quantity.key == "lr:hip_roll"
                else quantity.key.replace("{foot}", panel.foot or "<foot>")
            )
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


#: The actuators drawn in their own colour (the most-named first); the rest share :data:`_OTHER_ACTUATORS`' grey.
NAMED_COLOURS = 6
_OTHER_ACTUATORS = "other actuators"
#: The named actuators' colours: the palette less the handoff green and the alarm red, from its far end, so the
#: most-named actuator is not the first runs' blue or orange.
_NAMED_SERIES = tuple(reversed(_OTHER_SERIES[:-1]))


def _named_colours(runs: Sequence[RunHistory]) -> dict[str, str]:
    """A fixed colour per label the runs' flagged episodes name, by how many name it over every run (then by label).

    The first :data:`NAMED_COLOURS` take :data:`_NAMED_SERIES` in order (never
    cycled) and the rest share a mid grey; :data:`UNMEASURED_ACTUATOR` is a
    dark grey and :data:`NOT_RECORDED_ACTUATOR` a light one.  A label only
    :func:`most_named` gives (``+/-`` for an actuator's ``+`` and ``-``
    episodes tied in one evaluation) comes after the counted ones.
    """
    named = [entry for run in runs for entry in run.evaluations.get("flagged_actuators") or []]
    counts = tally(named)
    unnamed = {UNMEASURED_ACTUATOR: _INK_2, NOT_RECORDED_ACTUATOR: _AXIS}
    ranked = [label for label in counts if label not in unnamed]
    ranked += sorted({label for entry in named if (label := most_named(entry)) is not None} - set(counts))
    colours = {label: _NAMED_SERIES[i] if i < NAMED_COLOURS else _MUTED for i, label in enumerate(ranked)}
    colours.update({label: colour for label, colour in unnamed.items() if label in counts})
    return colours


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
        named_colours=_named_colours(runs),
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
    unnamed = [run.label for run in runs if run.evaluations.get("flagged_actuators") is None]
    named_note = (
        ""
        if not unnamed
        else f" ({'this ledger' if len(runs) == 1 else ', '.join(unnamed)} records no {ACTUATOR_COLUMN} beside "
        "screen_v2_reasons: an older ledger, or no screen, so no actuator is named)"
    )
    unrecorded = [run.label for run in runs if run.rollout_shares is None]
    rollout_note = (
        ""
        if not unrecorded
        else f" ({'this run' if len(runs) == 1 else ', '.join(unrecorded)}: no {POLICY_PROBE_FILENAME} rollout_* "
        "arrays, an older run or no probe, so those panels are absent)"
    )
    gaps = [run.label for run in runs if _rollout_gap(run)]
    if gaps:
        rollout_note += (
            f" ({'this run' if len(runs) == 1 else ', '.join(gaps)}: some rollouts not measured, said on the panel: "
            f"a resume kept rows from a {POLICY_PROBE_FILENAME} written before the arrays, or a buffer pass failed)"
        )
    unposed = [run.label for run in runs if "head_tip_height_mean_m" not in run.evaluations.get("columns", {})]
    posture_note = (
        ""
        if not unposed
        else f" ({'this ledger' if len(runs) == 1 else ', '.join(unposed)} records none: an older ledger, or a model "
        "whose evaluation runs no recorder, so those panels are absent)"
    )
    judged = [run for run in runs if _k3_points(run)]
    k3_note = (
        ""
        if not judged
        else " The last panel is a readout of the stance5 replication judge's K3 criterion from the judge files named "
        f"for {'this run' if len(runs) == 1 else ', '.join(run.label for run in judged)} ({K3_CRITERIA_FILE}: "
        f"{K3_CRITERION}): the judge's own perturbation panels on spawns 3042-3061 at each judged checkpoint, not "
        "these deterministic evaluations, and never a verdict (the scorer decides whether a run holds)."
    )
    targets = _common(run.head_target for run in runs)
    fields = {
        "episodes": episodes,
        "rollout_note": rollout_note,
        "band": _band_text(runs, "lower", "upper"),
        "pad_band": _band_text(runs, "pad_lower", "pad_upper"),
        "saturation_note": saturation_note,
        "named_note": named_note,
        "posture_note": posture_note,
        "k3_note": k3_note,
        "head_target": (
            "none recorded"
            if not targets
            else ", ".join(f"{value:g} m" for value in targets) + " (reward_weights.head_clearance_target)"
        ),
    }
    pages: list[Page] = []
    errors: list[str] = []

    def draw(spec: PageSpec) -> None:
        try:
            figure = _page_figure(spec, context, title=title, dpi=dpi, width=width)
            caption = spec.caption.format(**fields)
            if comparison:
                caption += f" Runs: {colour_note}; each run's name sits at its line's end."
            pages.append(Page(spec.name, spec.title, caption, _png(figure, dpi)))
        except Exception as exc:  # noqa: BLE001 - a failure costs this page only
            logger.warning("training history page %s failed: %s", spec.name, exc)
            errors.append(f"{spec.name}: {type(exc).__name__}: {exc}")

    with matplotlib.rc_context({"font.family": "DejaVu Sans", "figure.facecolor": "white"}):
        for spec in PAGES:
            draw(spec)
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
        for spec in POSTURE_PAGES:
            draw(spec)
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
    (ACTUATOR_FRACTION_COLUMN, "sat. fraction", 1.0),
    (f"{ACTUATOR_FRACTION_COLUMN}_worst", "sat. fraction worst", 1.0),
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


#: The checkpoint table's head posture and judge columns: ``(row key, header)``, shown where a row has them.
_POSTURE_HEADERS: tuple[tuple[str, str], ...] = (
    ("head_tip_height_mean_m", "head tip m"),
    ("head_tip_height_min_m_worst", "head tip lowest m"),
    ("neck_pitch_mean_deg", "neck pitch"),
    ("head_pitch_mean_deg", "head pitch"),
)
_JUDGE_HEADERS: tuple[tuple[str, str], ...] = (("judge_joint_falls", "judge joint"), ("judge_own_falls", "judge own"))


def _k3_table(run: Mapping[str, Any]) -> str:
    """The run's judge K3 readout as HTML: the source, each judged checkpoint's falls, K3's checkpoints marked."""
    k3 = run.get("judge_k3")
    if not isinstance(k3, Mapping):
        return ""
    rows = []
    for point in k3.get("checkpoints") or []:
        within = ""
        if point.get("k3_slot"):
            within = "; ".join(
                f"{name} {'within' if point.get(key) else 'over'}"
                for name, key in (("joint", "joint_within_bar"), ("own", "own_within_bar"))
            )
        rows.append(
            [
                html.escape(str(point.get("file"))),
                _fmt(point.get("timesteps")),
                "K3" if point.get("k3_slot") else "report only",
                f"{point.get('joint_falls')} / {point.get('joint_episodes')}",
                f"{point.get('own_falls')} / {point.get('own_episodes')}",
                html.escape(within or "–"),
            ]
        )
    skipped = "; ".join(f"{entry['file']}: {entry['why']}" for entry in k3.get("skipped") or [])
    parts = [
        f"<h3>{html.escape(str(run['label']))}: judge K3 readout</h3>",
        f"<p>From <code>{html.escape(str(k3.get('source')))}</code>. K3 ({html.escape(str(k3.get('criteria_file')))}): "
        f"{html.escape(str(k3.get('criterion')))}. A readout of the judge's records against K3's bars, never a "
        "verdict: the scorer decides whether a run holds, under the criteria's missing-file and run-handling rules, "
        "which are not applied here."
        + (f" Skipped: {html.escape(skipped)}." if skipped else "")
        + (f" {html.escape(str(k3['why']))}." if k3.get("why") else "")
        + "</p>",
    ]
    if rows:
        parts.append(
            _table(["judge file", "checkpoint", "read by", "joint-space falls", "own-sigma falls", "K3 bars"], rows)
        )
    return "".join(parts)


def _fmt_named(counts: Any) -> str:
    """A ``flagged_actuators`` entry as table text: ``tail_3_pitch_act + 5; r_ankle_act - 1``, ``none`` or ``–``."""
    if not isinstance(counts, Mapping):
        return "–"
    if not counts:
        return "none"
    return html.escape("; ".join(f"{label} {count}" for label, count in counts.items()))


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
        "(eval_episodes.csv, train_episodes.csv, stage_config.json, progress.json, plant_identity.json, models/)"
        + (", and the judge files named per run (--judged)" if any(run.get("judge_k3") for run in runs) else "")
        + ". "
        f"Report only: nothing reads this page back. Generated {html.escape(document['generated_at'])}; the numbers "
        f"are in <code>{html.escape(json_name)}</code> ({html.escape(SCHEMA)}).</p>",
        "<nav>" + "".join(f'<a href="#{p.name}">{html.escape(p.title)}</a>' for p in pages) + "</nav>",
        "<h2>Runs</h2>",
    ]
    rows = []
    for run in runs:
        sat = run.get("saturation") or {}
        alarm = sat.get("first_alarm_at")
        named = ((sat.get("runs") or [{}])[0] or {}).get("named")
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
                (
                    f'<span class="alarm">{_fmt(alarm)}{f" ({html.escape(named)})" if named else ""}</span>'
                    if alarm is not None
                    else "none"
                ),
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
        "window-mean CoP outside the band's edges; <i>flagged by actuator</i> counts the saturation-flagged episodes by "
        "the actuator each names and its sign (+ every saturated step at +1, - at -1, +/- both), so the final, best "
        "and robust_best checkpoints are read from their own evaluation's rows. The rail and saturation-fraction "
        "columns are the mean and the worst full-horizon episode. "
        "CoP in half-lengths (+ toe), tilt in degrees, lift in mm, left minus right in degrees (hip roll: each leg "
        "+ abduction), the head tip in m (its window mean, and the lowest episode's window minimum), the neck and "
        "head pitch in degrees (+ nose down), std in action units at the checkpoint saved at that step. <i>judge "
        "joint</i> and <i>judge own</i> are the judge's joint-space and own-sigma noise falls (of 20) at a "
        "checkpoint judged at exactly that step, from the judge files named for the run: a readout, never a "
        "verdict.</p>"
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
        cols += [("flagged_actuators", "flagged by actuator", 1.0)] if "flagged_actuators" in present else []
        cols += [(key, header, scale) for key, header, scale in _TABLE_VALUES if key in present]
        cols += [(f"{f}_foot_cop_fore_aft", f"foot CoP {f}", 1.0) for f in feet]
        cols += [(f"{f}_sole_cop_fore_aft", f"pad CoP {f}", 1.0) for f in feet]
        cols += [(f"l_minus_r_{j}_deg", f"L-R {j.replace('_', ' ')}", 1.0) for j in LR_PAGE_JOINTS if j != "hip_roll"]
        cols += [("l_minus_r_hip_roll_deg", "L-R hip roll", 1.0)] if "l_minus_r_hip_roll_deg" in present else []
        cols += [("actuator_force_frac_max", "force frac", 1.0)]
        cols += [(key, header, 1.0) for key, header in _POSTURE_HEADERS if key in present]
        cols += [(key, header, 1.0) for key, header in _JUDGE_HEADERS if key in present]
        std_cols = sorted(
            {k for row in table for k in row if k.startswith("std_") and k != "std_checkpoint"}, key=_std_order
        )
        headers = ["at", "evaluation"] + [c[1] for c in cols] + [k[4:] + " std" for k in std_cols]
        body = []
        for row in table:
            cells = [html.escape(row["at"]), _fmt(row["evaluation_timesteps"])]
            for key, _, scale in cols:
                value = row.get(key)
                if key == "flagged_actuators":
                    cells.append(_fmt_named(value))
                    continue
                cells.append(_fmt(value * scale if isinstance(value, (int, float)) else value))
            cells += [_fmt(row.get(k)) for k in std_cols]
            body.append(cells)
        parts.append(f"<h3>{html.escape(str(run['label']))}</h3>")
        parts.append(_table(headers, body))
    parts += [_k3_table(run) for run in runs if run.get("judge_k3")]
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
