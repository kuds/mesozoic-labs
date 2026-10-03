"""The run-level record of the trunk run a notebook session resolved (cleanup ROW-4/6, decision 4 (a)).

``<run_dir>/trunk_run.json`` names the trunk run (``TRUNK_DIR``) a session of
the SB3 notebook resolved for the run, as the ``TRUNK_FROM`` value that
reproduces it: ``""`` for none, the run id when the trunk lies (or
resolves) directly under the species/algorithm log directory a
``TRUNK_FROM`` id resolves under (``LOG_BASE/<species>/<algo>``), else its
absolute directory (and for a run named ``auto``, which ``TRUNK_FROM``
cannot name by id; :func:`trunk_from_value`).  It holds no timestamp and no
session token, so a session that resolved the same trunk, however spelt,
writes nothing, and it is a plain run-tree file: the next bundle
write declares and hashes it like any other
(``manifest.build_artifact_manifest``), and it is not one of the artifacts
the bundle writer regenerates, so a ``complete`` bundle certifies it.

:func:`record_trunk_run` (the SB3 notebook's resolve cell calls it once the
trunk is known, and prints the line it returns) writes it

* never into a run whose bundle is ``complete`` (immutable);
* while the run holds no ``ancestors/`` record, for this session's trunk:
  only a reuse from a trunk ties a run to that trunk, and every such reuse
  writes a record, so a run without one is consistent with any trunk;
* never once the run holds a record: the file then names the trunk the
  records came through (when the resolve cell ran before the chain loop
  that recorded them; ``docs/KNOWN_ISSUES.md`` documents the other order).
  A run that holds records but no file (opened before ROW-4/6) is left to
  the resume recipe's manual route, never recorded with a guess.

``reentry.refuse_trunk_other_than_recorded`` (the notebook's RESUME cell
calls it before it trains) reads it through :func:`read_trunk_record`.
"""

from __future__ import annotations

import json
from pathlib import Path

from .constants import ANCESTOR_RECORD_NAME, ANCESTORS_DIRNAME, TRUNK_RECORD_NAME, TRUNK_RECORD_SCHEMA
from .errors import ResultBundleError
from .hashing import _write_json
from .manifest import read_bundle_status

#: Directory names a ``TRUNK_FROM`` id never resolves to: ``"auto"`` is
#: ``ancestors.AUTO_TRUNK`` (D-A25), which the storage cell takes as "select a
#: trunk", never as the run of that name (not imported: ``result_bundle`` does
#: not import ``ancestors``).
_NOT_RUN_IDS = ("", ".", "..", "auto")


def trunk_from_value(trunk_dir: "str | Path | None", *, log_dir: "str | Path") -> str:
    """The ``TRUNK_FROM`` value that resolves to *trunk_dir* in the SB3 notebook's storage cell.

    ``""`` for no trunk; the directory name when *trunk_dir* lies directly
    under *log_dir* (``LOG_BASE/<species>/<algo>``, where the storage cell
    resolves a run id and ``ancestors.select_trunk`` scans), and the run id
    too when *trunk_dir* resolves directly under it (a link to an in-tree
    run, or a ``..`` spelling of one), so every spelling of one trunk gives
    one value and the value of a value is itself; else the absolute
    directory, which also covers a *trunk_dir* whose name is no run id
    (``..`` resolving elsewhere, a filesystem root) and a run named
    ``auto``, which ``TRUNK_FROM = "auto"`` would not name but select a
    trunk for.  Nothing is read: the trunk directory need not exist.
    """
    if trunk_dir is None:
        return ""
    trunk = Path(trunk_dir)
    log_root = Path(log_dir).resolve()
    if trunk.name not in _NOT_RUN_IDS and trunk.parent.resolve() == log_root:
        return trunk.name
    resolved = trunk.resolve()
    if resolved.name not in _NOT_RUN_IDS and resolved.parent == log_root:
        return resolved.name
    return str(resolved)


def _names_trunk(recorded: str, session: str, *, log_dir: "str | Path") -> bool:
    """Whether the recorded ``TRUNK_FROM`` value names this session's trunk (*session*, a :func:`trunk_from_value`).

    An absolute value is compared as the directory it names, so one the
    writer would have recorded otherwise (a run id, or another spelling of
    the path: a hand edit, or a run copied to another ``LOG_BASE``) still
    names that trunk, and the ``TRUNK_FROM`` a refusal or a warning names
    for it is one a session can match.
    """
    if Path(recorded).is_absolute():
        return trunk_from_value(recorded, log_dir=log_dir) == session
    return recorded == session


def ancestor_record_ids(run_dir: "str | Path") -> tuple[str, ...]:
    """The node ids *run_dir* holds an ``ancestors/<id>/ancestor.json`` record for, sorted."""
    root = Path(run_dir) / ANCESTORS_DIRNAME
    if not root.is_dir():
        return ()
    return tuple(sorted(path.parent.name for path in root.glob(f"*/{ANCESTOR_RECORD_NAME}") if path.is_file()))


def read_trunk_record(run_dir: "str | Path") -> "str | None":
    """The ``TRUNK_FROM`` value *run_dir*'s ``trunk_run.json`` records; None when the file is absent.

    A file that exists but cannot be read, or is not exactly
    ``{"schema": TRUNK_RECORD_SCHEMA, "trunk_from": <str>}`` with a value
    of a form :func:`trunk_from_value` writes (``""``, a run id other than
    ``auto``, or an absolute directory), raises :class:`ResultBundleError`:
    the trunk it recorded is unknown.  An absolute value is returned as
    recorded; the writer and ``reentry.refuse_trunk_other_than_recorded``
    compare it as the directory it names.  Writes nothing.
    """
    path = Path(run_dir) / TRUNK_RECORD_NAME
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ResultBundleError(f"cannot read the trunk record {path}: {exc}") from exc
    value = record.get("trunk_from") if isinstance(record, dict) else None
    if (
        not isinstance(record, dict)
        or set(record) != {"schema", "trunk_from"}
        or record["schema"] != TRUNK_RECORD_SCHEMA
        or not isinstance(value, str)
        or not (value == "" or Path(value).is_absolute() or (Path(value).name == value and value not in _NOT_RUN_IDS))
    ):
        raise ResultBundleError(
            f"invalid trunk record {path}: expected {{'schema': {TRUNK_RECORD_SCHEMA!r}, 'trunk_from': <a run id, an "
            "absolute run directory or ''>}"
        )
    return value


def record_trunk_run(run_dir: "str | Path", *, trunk_dir: "str | Path | None", log_dir: "str | Path") -> str:
    """Record in *run_dir*'s ``trunk_run.json`` the trunk this session resolved; return the line to print.

    *trunk_dir* is the session's ``TRUNK_DIR`` (None for no trunk) and
    *log_dir* the directory a ``TRUNK_FROM`` id resolves under
    (:func:`trunk_from_value`).  Writes (atomically, through
    :func:`.hashing._write_json`) only when the bundle is not ``complete``
    and the run holds no ``ancestors/`` record, and then only when the file
    is absent, unreadable or names another trunk (module docstring).
    Otherwise nothing is written and the line says why; a run holding
    records whose file names another trunk, or cannot be read, gets a
    ``WARNING`` (naming the recorded ``TRUNK_FROM``, or that it is unknown).
    Unless the file already names this session's trunk (nothing else is read
    then), an unreadable artifact manifest raises
    (:func:`.manifest.read_bundle_status`).
    """
    run_path = Path(run_dir)
    path = run_path / TRUNK_RECORD_NAME
    session = trunk_from_value(trunk_dir, log_dir=log_dir)
    try:
        recorded, unreadable = read_trunk_record(run_path), None
    except ResultBundleError as exc:
        recorded, unreadable = None, exc
    if unreadable is None and recorded is not None and _names_trunk(recorded, session, log_dir=log_dir):
        return f'Trunk record:  {path} names TRUNK_FROM = "{recorded}", this session\'s trunk.'
    if read_bundle_status(run_path) == "complete":
        if unreadable is not None:
            held = f"is unreadable ({unreadable})"
        elif recorded is not None:
            held = f'names TRUNK_FROM = "{recorded}"'
        else:
            held = "is absent"
        return f"Trunk record:  {path} {held}; nothing is written into a complete bundle."
    records = ancestor_record_ids(run_path)
    if records:
        held_ids = ", ".join(repr(node) for node in records)
        if unreadable is not None:
            return (
                f"WARNING: {unreadable}. This run holds ancestor records ({held_ids}), so the trunk they were reused "
                "through is unknown: the RESUME cell refuses a resume until the file is removed, and a resume then "
                "pins TRUNK_FROM by section 5's manual route."
            )
        if recorded is None:
            return (
                f"Trunk record:  none; this run holds ancestor records ({held_ids}) but no record of the trunk they "
                "were reused through (a run opened before ROW-4/6 or by the command-line curriculum, or one whose "
                f"{TRUNK_RECORD_NAME} was removed): a resume pins TRUNK_FROM by section 5's manual route."
            )
        return (
            f"WARNING: this run's ancestor records ({held_ids}) were reused through "
            f'TRUNK_FROM = "{recorded}" ({path}), but this session resolved TRUNK_FROM = "{session}": the chain loop '
            "refuses a second parent for a recorded node this session's trunk certifies, trains a recorded node it "
            'does not certify again here (every one, under TRUNK_FROM = ""), and the RESUME cell refuses a resume. '
            f'Set TRUNK_FROM = "{recorded}" in the configuration cell and re-run sections 2-3.'
        )
    _write_json(path, {"schema": TRUNK_RECORD_SCHEMA, "trunk_from": session})
    if unreadable is not None:
        return f'Trunk record:  replaced the unreadable {path} ({unreadable}) with TRUNK_FROM = "{session}".'
    if recorded is not None:
        return (
            f'Trunk record:  TRUNK_FROM = "{recorded}" -> "{session}" in {path} (this run holds no ancestor record, '
            "so any trunk is consistent with it)."
        )
    return f'Trunk record:  wrote TRUNK_FROM = "{session}" to {path}.'
