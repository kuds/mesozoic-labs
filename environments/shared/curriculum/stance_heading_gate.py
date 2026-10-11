"""The six-heading stance certificate statistic: ``stance_quality_heading/v1`` (decision D-D32).

What it adds, and why
---------------------
``stance_quality/v2`` (D-D23) classifies every episode of the 40-episode
certification panel on floor truth and certifies a one-sided 95% lower
bound on the clean fraction -- at the training heading only.  The stance
stage always spawns facing +x, so a stance that reads its world heading
passes it: ``20261009_155723`` (T. rex seed 52) passed v2 at 39/40 and
completes 0/8 episodes when the whole scene is turned -90 or -45 degrees
(docs/HEADING_INVARIANCE_DESIGN_2026_10.md).  The report-only heading probe
(D-D27) saw it, and blocked nothing.

This kind judges the same v2 episode classification on a fixed grid of
whole-scene turns (protocol :data:`HEADING_PROTOCOL`, grid
:data:`HEADING_GRID_DEG`): every reset seed of the certification panel is
rolled at every heading, with the prey and the episode's reference
direction turned about the root with the animal, so a heading-invariant
controller faces the same task at each one (every plant declares the
elliptic friction cone since D-D29 and D-D30, so the physics is identical
too).  The grid is fixed here, not in the TOML: a list-valued threshold
would defeat ``gate_schema.same_threshold``, and another grid is another
protocol revision.

The statistic
-------------
The certificate **passes** when both hold:

* the full v2 decision (:func:`~.stance_gate_v2.evaluate_stance_v2_gate`)
  passes at EVERY heading, each against the statue panel rolled at that
  heading (:func:`statue_references`) -- the bound on that heading's
  clean episodes, its full-horizon rail, its reward rails and T. rex's
  hop-or-fall rail, none of which a grouped count implies;
* a seed is **clean** only when its episode is clean at every heading, and
  :func:`~.recovery_gate.binomial_lcb` of ``k`` clean seed groups of ``n``
  clears ``min_clean_stance_lcb`` at ``n >= min_eval_episodes``.

The threshold keys are v2's own (:data:`STANCE_HEADING_THRESHOLD_KEYS`), so
two of them carry two meanings, deliberately: ``min_eval_episodes`` is the
number of seeds per heading AND the number of seed groups, and
``min_clean_stance_lcb`` is each heading's v2 bar AND the grouped bar.  At
40 seeds and 0.80, 37/40 clean groups pass (LCB 0.8174) and 36/40 fail
(0.7856).

The rotations of one seed are not independent trials, so the 240 episodes
are never pooled into the statistic: one different failing seed at each
of the six headings pools to 234/240 (an apparent LCB of 0.9513) while
only 34/40 seeds stand at every heading (0.7253), and each heading alone
still passes v2 at 39/40.  The pooled count is reported, labelled as not
the statistic.  The bound inherits v2's fixed measurement and reference
convention; it is a claim about this tested grid, not every continuous
heading.

Evidence
--------
The certificate records one row per (seed, heading) for the policy and for
the statue, and the grid must be complete and unique
(:func:`grid_refusals`): a missing heading or a duplicated pair fails the
certificate, and a failed rollout can never fall back to the yaw-zero
result.  :data:`STANCE_HEADING_PANEL_FIELDNAMES` is the per-row CSV
(``stance_heading_panel_selected.csv``) publication re-derives from; it is
a superset of v2's columns, so each reader refuses the other's file
(:func:`read_stance_heading_panel` on the missing ``role`` / ``yaw_deg``
columns, ``stance_gate_v2.read_stance_v2_panel`` on their presence).

This module is pure, like :mod:`~.stance_gate_v2`: the standard library,
v2's statistic, :func:`~.recovery_gate.binomial_lcb` and the pure row type
:class:`~environments.shared.gait.stance_metrics.StanceEpisodeMetrics`.
The judge (``reporting.gates``), the certificate builder
(``reporting.stance_report``), publication (``result_bundle.evidence``),
the backfill tool and ancestor reuse share this one implementation.
"""

from __future__ import annotations

import csv
import math
import numbers
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..gait.stance_metrics import STANCE_METRIC_FIELDS, STANCE_METRIC_LATER_FIELDS, StanceEpisodeMetrics
from .recovery_gate import binomial_lcb
from .stance_gate_v2 import (
    STANCE_V2_PANEL_FIELDNAMES,
    STANCE_V2_PANEL_HANDOFF_COLUMNS,
    STANCE_V2_PANEL_MEASUREMENT_COLUMNS,
    STANCE_V2_PANEL_STATUE_COLUMNS,
    STANCE_V2_PANEL_TASK_COLUMNS,
    STANCE_V2_REQUIRED_KEYS,
    STANCE_V2_THRESHOLD_KEYS,
    StanceV2Result,
    StanceV2Thresholds,
    StatueReference,
    evaluate_stance_v2_gate,
    statue_reference,
)

#: The gate kind this module evaluates.  Imported by ``gate_schema`` so the
#: registry and the implementation cannot drift apart.  Not
#: ``stance_quality/v3``: the v2 kind already writes reports with schema
#: ``mesozoic.stance-gate-report/v3``, and a v3 kind would invite reading a
#: v3 report as a v3 certificate.
STANCE_HEADING_GATE_KIND = "stance_quality_heading/v1"

#: The protocol the grid is rolled under: the whole scene (the animal, every
#: mocap body and the episode's reference direction) turned about vertical
#: through the root right after reset, on a stage with no command, no push
#: and no terrain -- the inputs ``reporting.stance_report._apply_spawn_yaw``
#: does not turn.  Another grid or another turn is another protocol.
HEADING_PROTOCOL = "whole_scene_yaw/v1"

#: The certificate's report schema (``reporting.stance_report``), defined here
#: beside the panel CSV contract so the judge, publication and the backfill
#: tool check it without importing the report module.
STANCE_HEADING_REPORT_SCHEMA = "mesozoic.stance-heading-report/v1"

#: The files a heading-kind stage writes in place of v2's
#: ``stance_gate_report.{txt,json}`` and ``stance_panel_selected.csv``.
STANCE_HEADING_REPORT_JSON = "stance_heading_report.json"
STANCE_HEADING_REPORT_TXT = "stance_heading_report.txt"
STANCE_HEADING_PANEL_CSV = "stance_heading_panel_selected.csv"

#: The fixed heading grid, degrees, canonical and in order (the design's
#: grid; the maintainer kept all six on 2026-10-11).
HEADING_GRID_DEG: tuple[float, ...] = (-90.0, -45.0, 0.0, 45.0, 90.0, 180.0)

#: The threshold keys the kind consumes and requires: exactly v2's, so a
#: stance block changes kind by changing ``gate_kind`` alone.
STANCE_HEADING_THRESHOLD_KEYS: frozenset[str] = STANCE_V2_THRESHOLD_KEYS
STANCE_HEADING_REQUIRED_KEYS: frozenset[str] = STANCE_V2_REQUIRED_KEYS

#: What a certificate records as its ``policy_deterministic`` and the one
#: normalization mode the judge admits: the checkpoint's own VecNormalize
#: statistics applied frozen (never re-estimated), with SB3's deterministic
#: ``predict`` -- the pair ``reporting.stance_report._load_policy`` builds.
FROZEN_VECNORMALIZE_MODE = "vecnormalize-frozen"


def canonical_heading_deg(value: Any) -> float:
    """*value* in degrees, wrapped into ``(-180, 180]``: -180 is 180, 360 is 0.

    A real number only: a bool, a string, ``None`` or a non-finite value
    raises ``ValueError`` -- a heading that cannot be read is not 0.
    """
    if value is None or isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"a heading must be a finite number of degrees, not {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"a heading must be a finite number of degrees, not {value!r}")
    wrapped = (number + 180.0) % 360.0 - 180.0
    # ``+ 0.0`` folds -0.0 into 0.0, so a turn of -360 reads as the training heading.
    return 180.0 if wrapped == -180.0 else wrapped + 0.0


def validate_heading_grid(angles: Sequence[Any]) -> tuple[float, ...]:
    """*angles* canonicalised, or ``ValueError`` naming why they are not :data:`HEADING_GRID_DEG`.

    Refuses a non-numeric or non-finite heading, two entries that are one
    heading (``-180`` and ``180``, ``0`` and ``360``, a repeat), and any grid
    other than the protocol's in its order -- a missing or an extra heading
    or the same six reordered: the grid is part of the protocol, never a
    parameter a report may choose.
    """
    if isinstance(angles, (str, bytes)) or not isinstance(angles, Sequence):
        raise ValueError(f"the heading grid must be a list of degrees, not {angles!r}")
    canonical = tuple(canonical_heading_deg(angle) for angle in angles)
    seen: dict[float, Any] = {}
    for raw, heading in zip(angles, canonical):
        if heading in seen:
            raise ValueError(
                f"the heading grid lists {seen[heading]!r} and {raw!r}, which are one heading ({heading:+g} deg); "
                "a duplicated heading would be counted twice"
            )
        seen[heading] = raw
    if canonical != HEADING_GRID_DEG:
        missing = [heading for heading in HEADING_GRID_DEG if heading not in canonical]
        extra = [heading for heading in canonical if heading not in HEADING_GRID_DEG]
        detail = (
            f"missing {', '.join(f'{h:+g}' for h in missing)}" if missing else "",
            f"extra {', '.join(f'{h:+g}' for h in extra)}" if extra else "",
            "" if missing or extra else "reordered",
        )
        raise ValueError(
            f"the heading grid {list(canonical)} is not the {HEADING_PROTOCOL} grid {list(HEADING_GRID_DEG)} "
            f"({'; '.join(part for part in detail if part)}); another grid is another protocol"
        )
    return canonical


@dataclass(frozen=True)
class HeadingEpisode:
    """One episode of the grid: the reset seed, the whole-scene turn it was rolled at, and its floor-truth metrics."""

    seed: int
    yaw_deg: float
    metrics: StanceEpisodeMetrics


def grid_refusals(rows: Iterable[HeadingEpisode], *, seeds: Sequence[int], grid: Sequence[float]) -> list[str]:
    """Every way *rows* fail to be the complete, unique Cartesian grid of *seeds* x *grid*; empty when they are.

    Each (seed, heading) pair must appear exactly once, and no row may sit
    off the grid (another heading, a seed outside *seeds*).  The seeds must
    be distinct whole numbers.  Headings are compared canonically, so a row
    at -180 is the 180 cell.
    """
    refusals: list[str] = []
    seed_list = list(seeds)
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seed_list):
        return [f"the seed list {seed_list!r} is not a list of whole numbers"]
    if len(set(seed_list)) != len(seed_list):
        refusals.append(f"the seed list repeats a seed ({seed_list}), so a seed group would be counted twice")
    try:
        headings = [canonical_heading_deg(heading) for heading in grid]
    except ValueError as exc:
        return [f"the heading grid is unreadable: {exc}"]
    cells: dict[tuple[int, float], int] = {}
    off_grid: list[str] = []
    for row in rows:
        try:
            heading = canonical_heading_deg(row.yaw_deg)
        except ValueError as exc:
            off_grid.append(f"seed {row.seed!r}: {exc}")
            continue
        if row.seed not in seed_list or heading not in headings:
            off_grid.append(f"seed {row.seed!r} at {heading:+g} deg")
            continue
        cells[(row.seed, heading)] = cells.get((row.seed, heading), 0) + 1
    if off_grid:
        shown = "; ".join(off_grid[:3])
        block = f"seeds {seed_list[0]}-{seed_list[-1]}" if seed_list else "no seeds"
        refusals.append(
            f"{len(off_grid)} row(s) sit off the grid of {block} x headings {headings} "
            f"({shown}{'; ...' if len(off_grid) > 3 else ''})"
        )
    duplicated = sorted(cell for cell, count in cells.items() if count > 1)
    if duplicated:
        shown = ", ".join(f"seed {seed} at {heading:+g} deg" for seed, heading in duplicated[:3])
        refusals.append(
            f"{len(duplicated)} (seed, heading) pair(s) appear more than once ({shown}"
            f"{', ...' if len(duplicated) > 3 else ''}); a duplicated pair would be counted twice"
        )
    missing = [(seed, heading) for heading in headings for seed in seed_list if (seed, heading) not in cells]
    if missing:
        by_heading: dict[float, int] = {}
        for _, heading in missing:
            by_heading[heading] = by_heading.get(heading, 0) + 1
        counts = ", ".join(f"{heading:+g} deg: {count}" for heading, count in by_heading.items())
        refusals.append(
            f"{len(missing)} of {len(seed_list) * len(headings)} (seed, heading) cells are missing ({counts}); "
            "a heading that was not measured fails the certificate and never falls back to another heading"
        )
    return refusals


def statue_references(
    rows: Iterable[HeadingEpisode], *, horizon: int, grid: Sequence[float] = HEADING_GRID_DEG
) -> dict[float, StatueReference]:
    """Each heading's statue reference: v2's reducer (:func:`~.stance_gate_v2.statue_reference`) over that heading's rows.

    Never a pooled reference: each heading's policy panel is judged against
    the statue rolled at the same heading on the same seeds.
    """
    by_heading: dict[float, list[StanceEpisodeMetrics]] = {canonical_heading_deg(h): [] for h in grid}
    for row in rows:
        heading = canonical_heading_deg(row.yaw_deg)
        if heading in by_heading:
            by_heading[heading].append(row.metrics)
    return {heading: statue_reference(metrics, horizon=horizon) for heading, metrics in by_heading.items()}


@dataclass(frozen=True)
class StanceHeadingResult:
    """The certificate's verdict and the numbers it was reached on.

    ``per_heading`` holds each heading's full v2 result in grid order (empty
    when the grid itself was refused); ``seed_failures`` names, per seed
    that is not clean at every heading, the headings it failed at.
    ``pooled_clean`` / ``pooled_lcb`` pool the episodes as though the turns
    were independent: reported, never the statistic.
    """

    passed: bool
    failures: tuple[str, ...]
    protocol: str
    grid_deg: tuple[float, ...]
    seeds: tuple[int, ...]
    n_seeds: int
    n_clean_seeds: int
    clean_seed_fraction: float
    clean_seed_lcb: float
    per_heading: tuple[tuple[float, StanceV2Result], ...]
    seed_failures: tuple[tuple[int, tuple[float, ...]], ...]
    pooled_clean: int
    pooled_lcb: float

    def heading_result(self, yaw_deg: Any) -> StanceV2Result | None:
        """The v2 result at *yaw_deg* (canonicalised), or ``None`` when the grid has no such heading."""
        heading = canonical_heading_deg(yaw_deg)
        return next((result for yaw, result in self.per_heading if yaw == heading), None)

    def as_dict(self) -> dict[str, Any]:
        """The certificate-level numbers, JSON-ready (each heading's episode reasons travel with its rows)."""
        return {
            "passed": self.passed,
            "failures": list(self.failures),
            "protocol": self.protocol,
            "grid_deg": list(self.grid_deg),
            "seeds": list(self.seeds),
            "n_seeds": self.n_seeds,
            "n_clean_seeds": self.n_clean_seeds,
            "clean_seed_fraction": self.clean_seed_fraction,
            "clean_seed_lcb": self.clean_seed_lcb,
            "per_heading": [{"yaw_deg": yaw, **result.as_dict()} for yaw, result in self.per_heading],
            "seed_failures": [{"seed": seed, "headings_deg": list(headings)} for seed, headings in self.seed_failures],
            "pooled_episodes": {
                "note": "every (seed, heading) episode counted as independent: NOT the statistic",
                "n_clean": self.pooled_clean,
                "n_episodes": self.n_seeds * len(self.per_heading),
                "clean_lcb": self.pooled_lcb,
            },
        }


def _heading_label(yaw: float) -> str:
    return f"heading {yaw:+g}"


def evaluate_stance_heading_gate(
    policy: Iterable[HeadingEpisode],
    thresholds: StanceV2Thresholds,
    *,
    horizon: int,
    seeds: Sequence[int],
    statue_by_heading: Mapping[float, StatueReference | None],
    control_dt: float | None = None,
    grid: Sequence[float] = HEADING_GRID_DEG,
) -> StanceHeadingResult:
    """Judge the grid's per-episode metrics against the certificate, naming every failure.

    *policy* holds one :class:`HeadingEpisode` per (seed, heading) of
    *seeds* x *grid*; *statue_by_heading* maps each canonical heading to
    the statue reference rolled at it (``None`` with a statue-relative
    criterion declared fails that heading, by name, as v2 does).

    Fails closed throughout: a grid other than the protocol's, an
    incomplete or duplicated grid, fewer seeds than ``min_eval_episodes``,
    a heading whose v2 decision fails, and a grouped bound below the bar
    are failures, never passes.  Each heading's failures are prefixed
    ``"heading <yaw>: "``.
    """
    rows = list(policy)
    seed_list = tuple(seeds)
    n = len(seed_list)
    try:
        headings = validate_heading_grid(list(grid))
    except ValueError as exc:
        return _refused(str(exc), grid=tuple(grid), seeds=seed_list)
    refusals = grid_refusals(rows, seeds=seed_list, grid=headings)
    if refusals:
        return _refused("; ".join(refusals), grid=headings, seeds=seed_list)

    cell = {(row.seed, canonical_heading_deg(row.yaw_deg)): row.metrics for row in rows}
    failures: list[str] = []
    per_heading: list[tuple[float, StanceV2Result]] = []
    clean: dict[tuple[int, float], bool] = {}
    for yaw in headings:
        result = evaluate_stance_v2_gate(
            [cell[(seed, yaw)] for seed in seed_list],
            thresholds,
            horizon=horizon,
            statue=statue_by_heading.get(yaw),
            control_dt=control_dt,
        )
        per_heading.append((yaw, result))
        for seed, reasons in zip(seed_list, result.episode_reasons):
            clean[(seed, yaw)] = not reasons
        failures.extend(f"{_heading_label(yaw)}: {failure}" for failure in result.failures)

    seed_failures = tuple(
        (seed, tuple(yaw for yaw in headings if not clean[(seed, yaw)]))
        for seed in seed_list
        if not all(clean[(seed, yaw)] for yaw in headings)
    )
    k = n - len(seed_failures)
    lcb = binomial_lcb(k, n) if n else math.nan
    grouped: list[str] = []
    if n < thresholds.min_eval_episodes:
        grouped.append(
            f"n_seeds {n} < min_eval_episodes {thresholds.min_eval_episodes} (the grouped bound's power is "
            "specified at this many seed groups)"
        )
    if not math.isfinite(lcb) or lcb < thresholds.min_clean_stance_lcb:
        grouped.append(
            f"clean_seed_lcb {lcb:.4f} < {thresholds.min_clean_stance_lcb:.4f} ({k}/{n} seeds clean at every "
            f"heading of {', '.join(f'{yaw:+g}' for yaw in headings)})"
        )
    pooled = sum(1 for value in clean.values() if value)
    total = n * len(headings)
    return StanceHeadingResult(
        passed=not (grouped or failures),
        failures=tuple(grouped + failures),
        protocol=HEADING_PROTOCOL,
        grid_deg=headings,
        seeds=seed_list,
        n_seeds=n,
        n_clean_seeds=k,
        clean_seed_fraction=k / n if n else math.nan,
        clean_seed_lcb=lcb,
        per_heading=tuple(per_heading),
        seed_failures=seed_failures,
        pooled_clean=pooled,
        pooled_lcb=binomial_lcb(pooled, total) if total else math.nan,
    )


def _refused(reason: str, *, grid: tuple[Any, ...], seeds: tuple[int, ...]) -> StanceHeadingResult:
    """The fail-closed result of a grid that cannot be judged: no heading is scored, no seed is clean."""
    return StanceHeadingResult(
        passed=False,
        failures=(f"grid: {reason}",),
        protocol=HEADING_PROTOCOL,
        grid_deg=tuple(grid),
        seeds=seeds,
        n_seeds=len(seeds),
        n_clean_seeds=0,
        clean_seed_fraction=0.0 if seeds else math.nan,
        clean_seed_lcb=math.nan,
        per_heading=(),
        seed_failures=(),
        pooled_clean=0,
        pooled_lcb=math.nan,
    )


# ── the per-episode grid evidence (``stance_heading_panel_selected.csv``) ──

#: The two roles a row plays: the scored policy, or the zero-action statue
#: rolled at the same (seed, heading) as its reference.
POLICY_ROLE = "policy"
STATUE_ROLE = "statue"

#: Columns every row stamps with one value for the whole certificate: the
#: gate it was scored under (``gate_sha256``), the protocol and its grid,
#: the seed block, the horizon and the control step, and whether the policy
#: was rolled deterministically.  v2's handoff, task and measurement stamps
#: (single-valued too) sit beside them, and the settle window is every row's
#: own ``settle_steps`` metric, which the v2 decision checks per episode.
STANCE_HEADING_PANEL_STAMP_COLUMNS: tuple[str, ...] = (
    "gate_sha256",
    "protocol",
    "heading_grid_deg",
    "seed_start",
    "n_seeds",
    "horizon",
    "control_dt",
    "policy_deterministic",
)

#: Column order of ``stance_heading_panel_selected.csv``: the row's role and
#: heading, then every v2 panel column (each episode's floor-truth metrics,
#: its classification -- blank on a statue row, which is a reference -- and
#: the stamps), then the certificate's own stamps.  The statue columns stamp
#: the reference of the row's OWN heading, single-valued within a heading.
STANCE_HEADING_PANEL_FIELDNAMES: tuple[str, ...] = (
    "role",
    "yaw_deg",
    *STANCE_V2_PANEL_FIELDNAMES,
    *STANCE_HEADING_PANEL_STAMP_COLUMNS,
)


def heading_grid_cell(grid: Sequence[float]) -> str:
    """The ``heading_grid_deg`` stamp: the canonical headings, ``repr`` floats, as ``[a, b, ...]``."""
    return "[" + ", ".join(repr(canonical_heading_deg(heading)) for heading in grid) + "]"


@dataclass(frozen=True)
class StanceHeadingPanelEvidence:
    """One heading panel CSV read back: the policy and statue rows, the recorded flags and the stamps.

    ``policy_clean`` and ``policy_reached_horizon`` are aligned with
    ``policy``; ``statue_reached_horizon`` with ``statue``.
    ``statue_by_heading`` is the reference each heading's rows STAMP (the
    reader does not re-reduce it; :func:`statue_references` over ``statue``
    does, and a consumer compares the two); a heading whose rows stamp none
    maps to ``None``.  ``absent_metrics`` names the later metric columns the
    file lacks, as :func:`~.stance_gate_v2.read_stance_v2_panel` does.
    """

    policy: tuple[HeadingEpisode, ...]
    policy_clean: tuple[bool, ...]
    policy_reached_horizon: tuple[bool, ...]
    statue: tuple[HeadingEpisode, ...]
    statue_reached_horizon: tuple[bool, ...]
    statue_by_heading: dict[float, StatueReference | None]
    checkpoint_sha256: str | None
    normalization_sha256: str | None
    task_sha256: str | None
    measurement_version: str | None
    measurement_sha256: str | None
    measurement_definition_sha256: str | None
    gate_sha256: str | None
    protocol: str | None
    heading_grid_deg: tuple[float, ...]
    seed_start: int
    n_seeds: int
    horizon: int
    control_dt: float
    policy_deterministic: bool
    absent_metrics: tuple[str, ...] = ()


def _cell_bool(value: str | None, *, column: str, index: int, path: Path) -> bool:
    text = (value or "").strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise ValueError(f"{path} row {index} has an unparsable {column} cell {value!r}")


def _single_value(rows: Sequence[Mapping[str, str]], column: str, *, path: Path, scope: str = "") -> str | None:
    values = {(row.get(column) or "").strip() for row in rows}
    if len(values) != 1:
        raise ValueError(f"{path} mixes {column} values across its rows{scope}: {sorted(values)}")
    value = values.pop()
    return value or None


def _whole(text: str | None, *, column: str, path: Path) -> int:
    if text is None or not text.lstrip("-").isdigit():
        raise ValueError(f"{path} stamps an unparsable {column} {text!r}")
    return int(text)


def _finite(text: str | None, *, column: str, path: Path) -> float:
    try:
        value = float(text or "")
    except ValueError as exc:
        raise ValueError(f"{path} stamps an unparsable {column} {text!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{path} stamps a non-finite {column} {text!r}")
    return value


def _statue_stamp(rows: Sequence[Mapping[str, str]], *, path: Path, heading: float) -> StatueReference | None:
    """The statue reference *rows* (one heading's) stamp, single-valued; ``None`` when every cell is blank."""
    scope = f" at heading {heading:+g}"
    cells = [_single_value(rows, column, path=path, scope=scope) for column in STANCE_V2_PANEL_STATUE_COLUMNS]
    if all(cell is None for cell in cells):
        return None
    if any(cell is None for cell in cells[:2]):
        raise ValueError(
            f"{path} records a partial statue reference{scope} ({dict(zip(STANCE_V2_PANEL_STATUE_COLUMNS, cells))})"
        )
    shares_text = (cells[3] or "").strip()
    if shares_text[:1] in "[(" and shares_text[-1:] in "])":
        shares_text = shares_text[1:-1]
    return StatueReference.from_dict(
        {
            "n_episodes": cells[0],
            "n_full_horizon": cells[1],
            "mean_reward": cells[2] or None,
            "foot_load_share": [item for item in shares_text.split(",") if item.strip()],
        }
    )


def read_stance_heading_panel(csv_path: "str | Path") -> StanceHeadingPanelEvidence:
    """Read a ``stance_heading_panel_selected.csv`` back into certificate inputs.

    Strict, like :func:`~.stance_gate_v2.read_stance_v2_panel`, because
    publication re-derives the certificate from what this returns: a file
    with no rows, a column of :data:`STANCE_HEADING_PANEL_FIELDNAMES` missing
    (a v2 panel file among them: it has no ``role`` or ``yaw_deg``), a row
    whose role is neither ``policy`` nor ``statue``, an unparsable metric,
    seed, heading or flag cell, a statue row that records a classification,
    a certificate-wide stamp that differs between rows, or a statue stamp
    that differs within one heading is a ``ValueError``.  An empty metric
    cell is NaN (unmeasured); a missing later-metric column reads as
    unmeasured and is named in ``absent_metrics``.  Grid completeness is the
    consumer's check (:func:`grid_refusals`).
    """
    path = Path(csv_path)
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    missing = [
        column
        for column in STANCE_HEADING_PANEL_FIELDNAMES
        if column not in columns and column not in STANCE_METRIC_LATER_FIELDS
    ]
    if missing:
        raise ValueError(f"{path} is not a {STANCE_HEADING_GATE_KIND} panel: it lacks the column(s) {missing}")
    if not rows:
        raise ValueError(f"{path} holds no episode rows")
    policy: list[HeadingEpisode] = []
    policy_clean: list[bool] = []
    policy_reached: list[bool] = []
    statue: list[HeadingEpisode] = []
    statue_reached: list[bool] = []
    rows_by_heading: dict[float, list[dict[str, str]]] = {}
    for index, row in enumerate(rows):
        role = (row.get("role") or "").strip()
        if role not in {POLICY_ROLE, STATUE_ROLE}:
            raise ValueError(f"{path} row {index} has role {role!r}, not {POLICY_ROLE!r} or {STATUE_ROLE!r}")
        try:
            yaw = canonical_heading_deg(float((row.get("yaw_deg") or "").strip()))
        except ValueError as exc:
            raise ValueError(f"{path} row {index} has an unparsable yaw_deg cell {row.get('yaw_deg')!r}") from exc
        seed_text = (row.get("panel_seed") or "").strip()
        if not seed_text.isdigit():
            raise ValueError(f"{path} row {index} has an unparsable panel_seed cell {seed_text!r}")
        try:
            metrics = StanceEpisodeMetrics.from_row(row)
        except ValueError as exc:
            raise ValueError(f"{path} row {index}: {exc}") from exc
        episode = HeadingEpisode(seed=int(seed_text), yaw_deg=yaw, metrics=metrics)
        reached = _cell_bool(row.get("reached_horizon"), column="reached_horizon", index=index, path=path)
        if role == POLICY_ROLE:
            policy.append(episode)
            policy_clean.append(_cell_bool(row.get("clean"), column="clean", index=index, path=path))
            policy_reached.append(reached)
        else:
            if (row.get("clean") or "").strip() or (row.get("reasons") or "").strip():
                raise ValueError(
                    f"{path} row {index} is a statue row with a classification; the statue is a reference, "
                    "never classified"
                )
            statue.append(episode)
            statue_reached.append(reached)
        rows_by_heading.setdefault(yaw, []).append(row)
    stamps = {
        column: _single_value(rows, column, path=path)
        for column in (
            *STANCE_V2_PANEL_HANDOFF_COLUMNS,
            *STANCE_V2_PANEL_TASK_COLUMNS,
            *STANCE_V2_PANEL_MEASUREMENT_COLUMNS,
            *STANCE_HEADING_PANEL_STAMP_COLUMNS,
        )
    }
    grid_text = (stamps["heading_grid_deg"] or "").strip()
    if grid_text[:1] in "[(" and grid_text[-1:] in "])":
        grid_text = grid_text[1:-1]
    try:
        grid = tuple(canonical_heading_deg(float(item)) for item in grid_text.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"{path} stamps an unparsable heading_grid_deg {stamps['heading_grid_deg']!r}") from exc
    deterministic = _cell_bool(stamps["policy_deterministic"], column="policy_deterministic", index=0, path=path)
    return StanceHeadingPanelEvidence(
        policy=tuple(policy),
        policy_clean=tuple(policy_clean),
        policy_reached_horizon=tuple(policy_reached),
        statue=tuple(statue),
        statue_reached_horizon=tuple(statue_reached),
        statue_by_heading={
            yaw: _statue_stamp(heading_rows, path=path, heading=yaw) for yaw, heading_rows in rows_by_heading.items()
        },
        checkpoint_sha256=stamps["checkpoint_sha256"],
        normalization_sha256=stamps["normalization_sha256"],
        task_sha256=stamps["task_sha256"],
        measurement_version=stamps["measurement_version"],
        measurement_sha256=stamps["measurement_sha256"],
        measurement_definition_sha256=stamps["measurement_definition_sha256"],
        gate_sha256=stamps["gate_sha256"],
        protocol=stamps["protocol"],
        heading_grid_deg=grid,
        seed_start=_whole(stamps["seed_start"], column="seed_start", path=path),
        n_seeds=_whole(stamps["n_seeds"], column="n_seeds", path=path),
        horizon=_whole(stamps["horizon"], column="horizon", path=path),
        control_dt=_finite(stamps["control_dt"], column="control_dt", path=path),
        policy_deterministic=deterministic,
        absent_metrics=tuple(name for name in STANCE_METRIC_FIELDS if name not in columns),
    )
