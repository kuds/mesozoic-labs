"""One CSV row per episode, held by column and rewritten atomically by its owner: report-only.

``gate_progress.npz`` keeps one row per evaluation, the mean over its
full-horizon episodes; the per-episode values behind those means were
thrown away.  A ledger keeps them: one row per episode, with the episode's
outcome (return, length, how it ended) beside its telemetry, so a reader
can ask which episodes fell, what each one's posture was, and how the two
relate, from one CSV a person can open on Drive.

The rows are stored by column (one list per column and a row count): a
dict per row would hold tens of thousands of boxed values in as many dicts
over a long run.  :meth:`EpisodeLedger.write` assembles the rows only while
writing, with ``file_io.atomic_write_csv``, so the file on disk is always
whole.  Floats are written with ``repr`` (``nan`` for an unmeasured value,
as the stance CSVs write it), integers as integers and strings verbatim.
A column first seen in a later row reads ``""`` on the earlier rows, and a
row that lacks a column reads ``""`` there.

No wall clock decides anything: the owner calls :meth:`EpisodeLedger.write`
at points fixed by the run (``StageAwareEvalCallback``: after every
evaluation), and a write with no row added since the last one writes
nothing.  A write that fails is logged once and costs the file, never the
run.  On a same-stage resume, :meth:`EpisodeLedger.seed_prior` keeps the
prior session's rows recorded at or before the resumed step, as
``gate_progress.npz`` keeps its rows.
"""

from __future__ import annotations

import csv
import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def format_cell(value: Any) -> Any:
    """A ledger value as the CSV cell it is written as: ``repr`` of a float, an int as an int, text verbatim."""
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return int(bool(value))
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return "nan" if math.isnan(number) else repr(number)
    return value


class EpisodeLedger:
    """The rows of one CSV, stored by column, written atomically whenever its owner calls :meth:`write`.

    Args:
        path: The CSV file; ``None`` keeps the rows in memory only.
        fixed_columns: The columns every row has, first and in this order;
            any other column follows in the order it was first seen.
    """

    def __init__(self, path: "str | Path | None", *, fixed_columns: Sequence[str]) -> None:
        self.path = None if path is None else Path(path)
        self._columns: dict[str, list[Any]] = {str(name): [] for name in fixed_columns}
        self._rows = 0
        self._dirty = False
        self._warned = False

    def __len__(self) -> int:
        """The rows held."""
        return self._rows

    @property
    def columns(self) -> tuple[str, ...]:
        """Every column, the fixed ones first."""
        return tuple(self._columns)

    def column(self, name: str) -> list[Any]:
        """One column's values, a copy (``KeyError`` for a column no row has)."""
        return list(self._columns[name])

    def extend(self, rows: Iterable[Mapping[str, Any]]) -> None:
        """Append *rows*; a column first seen here reads ``""`` on the earlier rows."""
        for row in rows:
            for key in row:
                if key not in self._columns:
                    self._columns[str(key)] = [""] * self._rows
            for key, values in self._columns.items():
                values.append(row[key] if key in row else "")
            self._rows += 1
            self._dirty = True

    def rows(self) -> list[dict[str, Any]]:
        """The rows as written: one dict per row, every column, each value a CSV cell (:func:`format_cell`)."""
        names = list(self._columns)
        cells = [[format_cell(value) for value in self._columns[name]] for name in names]
        return [{name: cells[column][row] for column, name in enumerate(names)} for row in range(self._rows)]

    def write(self) -> bool:
        """Rewrite the file atomically when rows were added since the last write; ``True`` when it wrote.

        A failure is logged once and leaves the previous file whole (the
        write is atomic); the rows stay held, so the next write retries.
        """
        if self.path is None or not self._dirty:
            return False
        try:
            from .file_io import atomic_write_csv

            atomic_write_csv(self.path, list(self._columns), self.rows())
        except Exception:  # noqa: BLE001 - a report must not sink a run
            if not self._warned:
                self._warned = True
                logger.warning("Could not write %s; the run continues without it.", self.path, exc_info=True)
            return False
        self._dirty = False
        return True

    def seed_prior(self, path: "str | Path", *, max_timesteps: int | None, step_column: str = "timesteps") -> int:
        """Start from a prior session's file at *path*: keep its rows recorded at or before *max_timesteps*.

        Called on a same-stage resume, before the session adds a row.  The
        kept cells are held as the text the file holds, so they are written
        back unchanged.  A row whose *step_column* is not a number is
        dropped.  Returns the number of rows kept: 0 when there is no
        readable file, and the file then starts over with this session's rows.
        """
        path = Path(path)
        if not path.exists():
            logger.info("Resume: no prior %s; it starts with this session's episodes.", path)
            return 0
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                header = list(reader.fieldnames or [])
                prior = list(reader)
        except Exception:  # noqa: BLE001 - a report must not sink a run
            logger.warning("Resume: could not read %s; it starts with this session's episodes.", path, exc_info=True)
            return 0
        kept = []
        for row in prior:
            try:
                step = float(row.get(step_column, ""))
            except (TypeError, ValueError):
                continue
            if max_timesteps is None or step <= max_timesteps:
                kept.append({name: row.get(name, "") or "" for name in header})
        for name in header:
            if name not in self._columns:
                self._columns[name] = [""] * self._rows
        self.extend(kept)
        self._dirty = False
        return len(kept)
