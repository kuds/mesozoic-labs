"""The hunting gate statistic: ``task_success/v1`` (BEHAVIOR_RECIPES_PLAN §4.4).

What it certifies: that the selected checkpoint's per-episode TASK event
(the bite / strike / food-reached flag the environment reports as
``info["is_success"]``, written as ``task_success`` in the evidence CSV) has
a success probability whose one-sided 95% lower confidence bound clears a
declared bar at a declared panel size.  It replaces the raw
``min_success_rate`` floor for the hunting deliverable: a raw fraction over
a 30-episode panel says nothing about its own power, and the same 50%
reading can come from 15/30 or from a lucky 5/10.  The bound is
:func:`~environments.shared.curriculum.recovery_gate.binomial_lcb` — the
exact one-sided Clopper-Pearson bound the recovery gate already certifies
with — so the two binary-event gates cannot drift on the statistic.

Sizing (plan §4.4): a bound ≥ 0.5 needs 20/30 or 26/40 successes; at n=40 a
true 70% policy passes 81% of the time, a true 80% policy 99%, a true 50%
policy 4%.  The one committed hunting result (29/30) bounds at 0.851.

Three consumers share this one implementation: the post-stage judge
(``reporting.gates._task_success_stage_gate``, also what the backfill tool
runs), publication (``result_bundle.evidence``, bound to the certified
checkpoint's hash) and the sweep's offline row verdict (from the recorded
count).  ``min_avg_reward`` is a collapse RAIL on the selected checkpoint's
mean reward, never the gate; ``min_avg_episode_length`` is optional.

Everything fails closed: a panel smaller than ``min_eval_episodes``, a NaN
bound, a blank or unparsable success cell, and evidence whose rows hash to
more than one checkpoint are all refusals, never passes.  This module
imports only the standard library and ``recovery_gate.binomial_lcb``:
``reporting`` and ``result_bundle`` both import it, so it must not reach
back into either.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .recovery_gate import binomial_lcb

TASK_SUCCESS_GATE_KIND = "task_success/v1"

#: Evidence columns the per-episode CSV may name its success event under:
#: ``task_success`` since the 2026-08-15 rename, ``success`` before it (the
#: rule ``result_bundle.evidence`` applies; both carry the same quantity).
_SUCCESS_COLUMNS = ("task_success", "success")

#: Digest columns the evidence writer (``csv_output.save_evaluation_episodes``)
#: stamps on every row; each must be single-valued across a file, because a
#: file whose rows hash to two checkpoints is evidence for neither.
_HASH_COLUMNS = ("checkpoint_sha256", "normalization_sha256")


def _finite_number(value: Any, *, key: str) -> float:
    """*value* as a finite float, or ``ValueError`` naming *key*."""
    if value is None or isinstance(value, bool):
        raise ValueError(f"task_success/v1 threshold {key} is missing or not numeric ({value!r})")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"task_success/v1 threshold {key} is not numeric ({value!r})") from exc
    if not math.isfinite(number):
        raise ValueError(f"task_success/v1 threshold {key} must be finite, not {value!r}")
    return number


@dataclass(frozen=True)
class TaskSuccessGateThresholds:
    """The ``task_success/v1`` criteria.

    ``min_success_lcb`` is the load-bearing bound and ``min_eval_episodes``
    the panel size its power is specified at; both are required (the schema
    requires them of the kind, and this reader does not invent them).
    ``min_avg_reward`` is a collapse rail checked by the judge, never the
    gate, and ``min_avg_episode_length`` is optional.
    """

    min_success_lcb: float
    min_eval_episodes: int
    min_avg_reward: "float | None" = None
    min_avg_episode_length: "float | None" = None

    @classmethod
    def from_curriculum(cls, curriculum: Mapping[str, Any]) -> TaskSuccessGateThresholds:
        """Read the thresholds a stage's ``[curriculum]`` table declares.

        Raises ``ValueError`` naming the key when ``min_success_lcb`` or
        ``min_eval_episodes`` is missing, non-numeric or non-finite: a gate
        without its bar or its panel size is not a gate, and a default for
        either would be a pass nobody declared.
        """
        min_success_lcb = _finite_number(curriculum.get("min_success_lcb"), key="min_success_lcb")
        min_eval_episodes = _finite_number(curriculum.get("min_eval_episodes"), key="min_eval_episodes")
        if min_eval_episodes < 1 or min_eval_episodes != int(min_eval_episodes):
            raise ValueError(
                f"task_success/v1 threshold min_eval_episodes must be a positive integer, not {min_eval_episodes!r}"
            )
        rail = curriculum.get("min_avg_reward")
        length = curriculum.get("min_avg_episode_length")
        return cls(
            min_success_lcb=min_success_lcb,
            min_eval_episodes=int(min_eval_episodes),
            min_avg_reward=None if rail is None else _finite_number(rail, key="min_avg_reward"),
            min_avg_episode_length=None if length is None else _finite_number(length, key="min_avg_episode_length"),
        )


@dataclass(frozen=True)
class TaskSuccessGateResult:
    passed: bool
    failures: tuple[str, ...]
    success_count: int
    n_episodes: int
    success_fraction: float
    success_lcb: float


def evaluate_task_success_gate(
    successes: Sequence[bool],
    thresholds: TaskSuccessGateThresholds,
) -> TaskSuccessGateResult:
    """Judge a panel's per-episode successes against the bound, naming every failure.

    ``n`` is the panel size, ``k`` the successes; the bound is
    :func:`~environments.shared.curriculum.recovery_gate.binomial_lcb` (exact
    one-sided 95% Clopper-Pearson), NaN for an empty panel so it fails
    closed.  The reward rail and the optional length floor are not judged
    here — they are conjuncts the judge applies to the selected checkpoint's
    aggregates — so this function is the one statistic every consumer
    shares.
    """
    failures: list[str] = []
    n = len(successes)
    k = sum(1 for success in successes if success)
    if n < thresholds.min_eval_episodes:
        failures.append(f"n_episodes {n} < min_eval_episodes {thresholds.min_eval_episodes}")
    success_fraction = k / n if n else math.nan
    success_lcb = binomial_lcb(k, n) if n else math.nan
    if not math.isfinite(success_lcb) or success_lcb < thresholds.min_success_lcb:
        failures.append(
            f"task_success_lcb {success_lcb:.4f} < {thresholds.min_success_lcb:.4f} ({k}/{n} episodes succeeded)"
        )
    return TaskSuccessGateResult(
        passed=not failures,
        failures=tuple(failures),
        success_count=k,
        n_episodes=n,
        success_fraction=success_fraction,
        success_lcb=success_lcb,
    )


#: Per-episode aggregate columns the judge rails on: the selected panel's
#: mean reward (``min_avg_reward``) and mean length (``min_avg_episode_length``).
_AGGREGATE_COLUMNS = ("reward", "length")


@dataclass(frozen=True)
class TaskSuccessEvidence:
    """The per-episode successes of one evidence CSV and the digests its rows carry.

    ``mean_reward`` / ``mean_length`` are the same rows' reward and length
    means — the numbers the reward rail and the optional length floor are
    judged on, so the verdict describes ONE panel rather than the bound
    from this file and a rail from some other checkpoint's evaluation.
    ``None`` when the column is absent.
    """

    successes: list[bool]
    checkpoint_sha256: "str | None"
    normalization_sha256: "str | None"
    mean_reward: "float | None" = None
    mean_length: "float | None" = None


def _parse_success(value: "str | None", *, index: int, path: Path) -> bool:
    if value is None or not value.strip():
        raise ValueError(f"{path} episode {index} has a blank task_success cell")
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"{path} episode {index} has an unparsable task_success cell {value!r}")


def read_task_successes(csv_path: "str | Path") -> TaskSuccessEvidence:
    """Read the per-episode successes off an ``evaluation_*.csv`` evidence file.

    The success column is ``task_success``, or the legacy ``success`` when
    only that is present; a file with neither, a blank or unparsable cell,
    or no rows at all is a ``ValueError``.  The ``checkpoint_sha256`` /
    ``normalization_sha256`` columns must each be single-valued across the
    rows (a mixed file is a ``ValueError``) and read as ``None`` when the
    column is absent — the consumers then refuse to bind the evidence,
    never assume the binding.  The ``reward`` / ``length`` columns, when
    present, must parse as finite numbers on every row (a blank or
    non-finite cell is a ``ValueError``: a rail judged on a partial mean
    is a rail nobody measured) and give ``mean_reward`` / ``mean_length``.
    """
    path = Path(csv_path)
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} holds no episode rows")
    success_column = next((name for name in _SUCCESS_COLUMNS if name in fieldnames), None)
    if success_column is None:
        raise ValueError(f"{path} has no task_success column (columns: {fieldnames})")
    successes = [_parse_success(row.get(success_column), index=index, path=path) for index, row in enumerate(rows, 1)]
    digests: dict[str, str | None] = {}
    for column in _HASH_COLUMNS:
        if column not in fieldnames:
            digests[column] = None
            continue
        values = {(row.get(column) or "").strip() or None for row in rows}
        if len(values) != 1:
            raise ValueError(f"{path} mixes {column} values across its rows: {sorted(str(v) for v in values)}")
        digests[column] = values.pop()
    means: dict[str, float | None] = {}
    for column in _AGGREGATE_COLUMNS:
        if column not in fieldnames:
            means[column] = None
            continue
        values_for_column: list[float] = []
        for index, row in enumerate(rows, 1):
            cell = (row.get(column) or "").strip()
            try:
                number = float(cell)
            except ValueError as exc:
                raise ValueError(f"{path} episode {index} has an unparsable {column} cell {cell!r}") from exc
            if not math.isfinite(number):
                raise ValueError(f"{path} episode {index} has a non-finite {column} cell {cell!r}")
            values_for_column.append(number)
        means[column] = math.fsum(values_for_column) / len(values_for_column)
    return TaskSuccessEvidence(
        successes=successes,
        checkpoint_sha256=digests["checkpoint_sha256"],
        normalization_sha256=digests["normalization_sha256"],
        mean_reward=means["reward"],
        mean_length=means["length"],
    )
