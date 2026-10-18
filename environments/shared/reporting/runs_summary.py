"""The cross-run results table, ``runs_summary.csv``, and the offline-judgement ledger it takes rows from.

Every run writes its own records to its Drive folder; the Drive summary notebook
(``notebooks/google_drive_summary.ipynb``) reads them all into one table, one row per
stage of each training run (``source`` ``training``) and per trial of each legacy sweep
(``sweep`` or ``sweep-analysis``).  A judgement made outside training -- a gate
certificate scored on a saved checkpoint after its run, or a pilot's fixed criteria applied
by a script -- has no Drive folder to land in when whoever made it cannot write to Drive,
so the repository keeps it: ``results/offline_judgements.csv`` (:data:`OFFLINE_JUDGEMENTS_PATH`),
one row per judged checkpoint, in the columns of :data:`OFFLINE_JUDGEMENT_COLUMNS`
(documented in ``results/README.md``).  :func:`load_offline_judgements` reads and
validates it, :func:`merge_runs_summary` adds its rows to the notebook's table as
``source`` ``offline``, and :func:`write_runs_summary` writes the result, the union of every
row's columns, atomically: a temporary file beside the destination, then ``os.replace``
(as ``progress.json`` is written), so a Colab runtime reclaimed mid-write leaves the
previous file on Drive, never half of one.

No pandas: the notebook passes ``DataFrame.to_dict(orient="records")``, whose missing
cells (``None`` or NaN) are written empty.
"""

from __future__ import annotations

import csv
import math
import re
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..file_io import atomic_write_csv
from ..paths import REPOSITORY_ROOT
from ..record_fields import is_sha256_digest
from ..result_bundle import ResultBundleError, canonical_algorithm
from ..species_names import require_model_id
from ..stage_manifest import StageManifestError, resolve_stage_key
from .csv_output import _compute_fieldnames

#: The ledger, tracked in the repository.  Rows are added by pull request; nothing writes it.
OFFLINE_JUDGEMENTS_PATH = REPOSITORY_ROOT / "results" / "offline_judgements.csv"
#: The ``source`` of every ledger row, and of no other row of the table.
OFFLINE_SOURCE = "offline"

#: The ledger's header, exactly, in this order.
OFFLINE_JUDGEMENT_COLUMNS: tuple[str, ...] = (
    "judged_on",
    "species",
    "embodiment",
    "algorithm",
    "run_id",
    "run_path",
    "stage",
    "checkpoint_step",
    "checkpoint_sha256",
    "gate_kind",
    "verdict",
    "passing_count",
    "panel_size",
    "lower_bound",
    "lower_bound_bar",
    "binding_bar",
    "panel_seeds",
    "judging_commit",
    "judged_by",
    "evidence",
    "note",
    "source",
)
#: Columns a row may leave empty.  ``run_id`` and ``run_path`` may each be empty, not both.
_OPTIONAL_COLUMNS = frozenset(
    {
        "run_id",
        "run_path",
        "passing_count",
        "panel_size",
        "lower_bound",
        "lower_bound_bar",
        "binding_bar",
        "note",
    }
)

#: The summary table's leading columns: the notebook's identity columns, as its export
#: has always ordered them.  The ledger's own columns follow them in a block of their own
#: (:data:`_OFFLINE_BLOCK`) in a file that holds an offline row.
RUNS_SUMMARY_FIXED_COLUMNS: tuple[str, ...] = (
    "species",
    "embodiment",
    "algorithm",
    "source",
    "library_version",
    "run_date",
    "run_dir",
    "trial_id",
    "stage",
    "stage_name",
    "seed",
    "n_envs",
)
#: The ledger columns that are not already summary columns.  ``run_id``, ``run_path``,
#: ``gate_kind`` and ``stage`` are shared with the training rows, and ``source`` and the
#: identity columns lead every file.
_OFFLINE_BLOCK: tuple[str, ...] = (
    "judged_on",
    "checkpoint_step",
    "checkpoint_sha256",
    "verdict",
    "passing_count",
    "panel_size",
    "lower_bound",
    "lower_bound_bar",
    "binding_bar",
    "panel_seeds",
    "judging_commit",
    "judged_by",
    "evidence",
    "note",
)

_COMMIT = re.compile(r"[0-9a-f]{7,40}")
_VERDICT = re.compile(r"[A-Z][A-Z0-9_]*")
_TOKEN = re.compile(r"\S+")
_NON_NEGATIVE_INTEGER = re.compile(r"[0-9]+")
#: The gate verdicts, and what each says in the table's ``stage_passed`` column.  Any other
#: verdict (a pilot's WALKS or MIXED) is not a gate's and leaves ``stage_passed`` empty.
_GATE_VERDICTS = {"PASS": True, "FAIL": False}


class OfflineJudgementError(ValueError):
    """The ledger is malformed; the message lists every problem found, by line."""


def _date(text: str) -> str:
    try:
        valid = len(text) == 10 and date.fromisoformat(text).isoformat() == text
    except ValueError:
        valid = False
    if not valid:
        raise ValueError(f"must be a date written YYYY-MM-DD, got {text!r}")
    return text


def _count(text: str) -> int:
    if not _NON_NEGATIVE_INTEGER.fullmatch(text):
        raise ValueError(f"must be a non-negative integer, got {text!r}")
    return int(text)


def _panel_size(text: str) -> int:
    if not _NON_NEGATIVE_INTEGER.fullmatch(text) or int(text) == 0:
        raise ValueError(f"must be a positive integer, got {text!r}")
    return int(text)


def _digest(text: str) -> str:
    if not is_sha256_digest(text):
        raise ValueError(
            f"must be the checkpoint file's digest as sha256_file records it, sha256:<64 lowercase hex>, got {text!r}"
        )
    return text


def _rate(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        value = math.nan
    if not 0.0 <= value <= 1.0:  # NaN fails too
        raise ValueError(f"must be a number from 0 to 1, got {text!r}")
    return value


def _matching(pattern: re.Pattern[str], what: str) -> Callable[[str], str]:
    def parse(text: str) -> str:
        if not pattern.fullmatch(text):
            raise ValueError(f"must be {what}, got {text!r}")
        return text

    return parse


def _text(text: str) -> str:
    return text


def _source(text: str) -> str:
    if text != OFFLINE_SOURCE:
        raise ValueError(f"must be {OFFLINE_SOURCE!r}, got {text!r}")
    return text


def _algorithm(text: str) -> str:
    try:
        public = canonical_algorithm(text)
    except ResultBundleError:
        public = None
    if public != text:
        raise ValueError(f"must be the public algorithm label, PPO or SAC, got {text!r}")
    return text


#: Each column's reader; a column not listed is free text.  Each raises ``ValueError("must ...")``.
_PARSERS: dict[str, Callable[[str], Any]] = {
    "judged_on": _date,
    "algorithm": _algorithm,
    "checkpoint_step": _count,
    "checkpoint_sha256": _digest,
    "gate_kind": _matching(_TOKEN, "one word, the gate kind or the criteria's name"),
    "verdict": _matching(_VERDICT, "an upper-case word (PASS, FAIL, or the criteria's own outcome)"),
    "passing_count": _count,
    "panel_size": _panel_size,
    "lower_bound": _rate,
    "lower_bound_bar": _rate,
    "judging_commit": _matching(_COMMIT, "a commit, 7 to 40 lowercase hexadecimal digits"),
    "source": _source,
}


def _parse_row(cells: Mapping[str, str]) -> tuple[dict[str, Any], list[str]]:
    """The typed row and its problems; a column whose cell is refused reads ``None``."""
    row: dict[str, Any] = {}
    problems: list[str] = []
    for column in OFFLINE_JUDGEMENT_COLUMNS:
        text = cells[column].strip()
        row[column] = None
        if not text:
            if column not in _OPTIONAL_COLUMNS:
                problems.append(f"{column} is required")
            continue
        try:
            row[column] = _PARSERS.get(column, _text)(text)
        except ValueError as exc:
            problems.append(f"{column} {exc}")
    if row["run_id"] is None and row["run_path"] is None:
        problems.append("run_id or run_path is required (the run the checkpoint came from)")
    if row["passing_count"] is not None and row["panel_size"] is not None and row["passing_count"] > row["panel_size"]:
        problems.append(f"passing_count {row['passing_count']} exceeds panel_size {row['panel_size']}")
    if row["species"] is None or row["embodiment"] is None:
        return row, problems
    try:
        model_id = require_model_id(row["species"], row["embodiment"])
    except ValueError as exc:
        problems.append(f"species and embodiment name no model of the species manifest: {exc}")
        return row, problems
    if row["stage"] is not None:
        try:
            entry = resolve_stage_key(model_id, row["stage"])
        except StageManifestError as exc:
            problems.append(f"stage {row['stage']!r} is not a stage of {model_id}: {exc}")
        else:
            if entry.key != row["stage"]:
                problems.append(
                    f"stage must be written as its key, as collected_results.csv writes it: "
                    f"{entry.key!r}, not {row['stage']!r}"
                )
            row["stage"] = entry.reference
    return row, problems


def load_offline_judgements(path: str | Path = OFFLINE_JUDGEMENTS_PATH) -> list[dict[str, Any]]:
    """The ledger's rows, typed, in file order; ``[]`` for an empty file or a header alone.

    The header must be :data:`OFFLINE_JUDGEMENT_COLUMNS` exactly: an unknown, missing,
    repeated or reordered column is refused, as a misspelt one would otherwise be read as
    absent.  Each cell is stripped of surrounding blanks.  ``judged_on`` stays the
    ``YYYY-MM-DD`` text; ``checkpoint_step``, ``passing_count`` and ``panel_size`` read as
    ``int`` and ``lower_bound`` and ``lower_bound_bar`` as ``float``; ``stage`` reads as the
    stage's reference (``1``, ``"recovery"``), as the notebook's rows hold it; an empty
    optional cell reads ``None``.  Every problem in the file is collected and raised
    together as :class:`OfflineJudgementError`, each with its line.  A missing file raises
    ``FileNotFoundError``.
    """
    ledger = Path(path)
    # utf-8-sig: a spreadsheet's byte-order mark is not part of the first column's name.
    with ledger.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        # The first row with a cell is the header; a file with none (empty, or blank lines) has no rows.
        header = next((cells for cells in reader if any(cell.strip() for cell in cells)), None)
        if header is None:
            return []
        header = [name.strip() for name in header]
        if header != list(OFFLINE_JUDGEMENT_COLUMNS):
            unknown = [name for name in header if name not in OFFLINE_JUDGEMENT_COLUMNS]
            missing = [name for name in OFFLINE_JUDGEMENT_COLUMNS if name not in header]
            details = [f"unknown {unknown}" if unknown else "", f"missing {missing}" if missing else ""]
            details = [detail for detail in details if detail] or ["repeated or reordered columns"]
            raise OfflineJudgementError(
                f"{ledger}: the header must be exactly {', '.join(OFFLINE_JUDGEMENT_COLUMNS)} ({'; '.join(details)})"
            )
        rows: list[dict[str, Any]] = []
        problems: list[str] = []
        seen: dict[tuple[str, ...], int] = {}
        for cells in reader:
            line = reader.line_num
            if not any(cell.strip() for cell in cells):
                continue
            if len(cells) != len(header):
                problems.append(f"line {line}: {len(cells)} cells, the header has {len(header)}")
                continue
            key = tuple(cell.strip() for cell in cells)
            if key in seen:
                problems.append(f"line {line}: repeats line {seen[key]}")
                continue
            seen[key] = line
            row, row_problems = _parse_row(dict(zip(header, cells, strict=True)))
            problems += [f"line {line}: {problem}" for problem in row_problems]
            rows.append(row)
    if problems:
        raise OfflineJudgementError(f"{ledger} is not a valid offline-judgement ledger:\n" + "\n".join(problems))
    return rows


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _plain(value: Any) -> Any:
    """*value* as a plain Python value: a NumPy scalar's ``item()``, ``None`` for NaN."""
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        value = value.item()
    return None if _missing(value) else value


def _summary_row(judgement: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(judgement)
    row["run_dir"] = judgement.get("run_path")
    row["stage_passed"] = _GATE_VERDICTS.get(str(judgement.get("verdict")))
    return row


def merge_runs_summary(
    table_rows: Iterable[Mapping[str, Any]],
    offline_judgements: Iterable[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """The notebook's rows, then the ledger's, as plain dicts with missing cells ``None``.

    A ledger row keeps its columns, gains ``run_dir`` (its ``run_path``, as the notebook
    gives its own rows) and ``stage_passed`` (``True`` for PASS, ``False`` for FAIL, else
    ``None``).  A table row may not claim ``source`` ``offline`` (the label is the ledger's),
    and a ledger row must carry it.
    """
    merged: list[dict[str, Any]] = []
    for row in table_rows:
        plain = {key: _plain(value) for key, value in row.items()}
        if plain.get("source") == OFFLINE_SOURCE:
            raise ValueError(f"a table row claims source {OFFLINE_SOURCE!r}, which only ledger rows carry: {plain}")
        merged.append(plain)
    for judgement in offline_judgements:
        if judgement.get("source") != OFFLINE_SOURCE:
            raise ValueError(f"an offline judgement must carry source {OFFLINE_SOURCE!r}: {dict(judgement)}")
        merged.append({key: _plain(value) for key, value in _summary_row(judgement).items()})
    return merged


def source_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """How many rows each ``source`` has, in order of first appearance."""
    counts: dict[str, int] = {}
    for row in rows:
        source = str(row.get("source") or "")
        counts[source] = counts.get(source, 0) + 1
    return counts


def write_runs_summary(rows: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """Write *rows* to *path* atomically, in the union of their columns; returns *path*.

    The columns: :data:`RUNS_SUMMARY_FIXED_COLUMNS`, then the ledger's own (when a row is
    ``offline``), then every other column sorted, then the canonical metric columns and any
    ``eval_*`` (``reporting.csv_output``'s order, which ``collected_results.csv`` uses).  A
    cell a row does not have, or holds as ``None`` or NaN, is written empty.  The file is
    staged beside *path* and moved over it with ``os.replace``, creating *path*'s directory
    if needed; a failure leaves the previous file whole and no temporary beside it.  An
    empty *rows* writes the header alone.
    """
    fixed = list(RUNS_SUMMARY_FIXED_COLUMNS)
    if any(row.get("source") == OFFLINE_SOURCE for row in rows):
        fixed += _OFFLINE_BLOCK
    cells = [{key: ("" if (value := _plain(raw)) is None else value) for key, raw in row.items()} for row in rows]
    fieldnames = _compute_fieldnames(cells, fixed)
    return atomic_write_csv(path, fieldnames, ({column: row.get(column, "") for column in fieldnames} for row in cells))
