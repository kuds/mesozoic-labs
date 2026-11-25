"""The SB3 notebook's run trees: which tree under a model's log directory a run lives in, and what that makes it.

A notebook run lives at ``LOG_BASE/<species>/<embodiment>/<tree>/<run_id>/``, and its
tree is its kind (:func:`run_tree_name` picks it from ``ALGORITHM``, ``RUN_KIND`` and
``QUICK_TEST``):

* ``<algo>/``, the certification tree (``RUN_KIND = "certify"``, the default).
  ``ancestors.select_trunk`` scans it for ``TRUNK_FROM = "auto"``, a ``TRUNK_FROM``
  run id resolves in it, and seed-replicate discovery
  (``replication.discover_replicates``, which scans a run's own siblings) counts a
  certification run's replicates there.
* ``<algo>_screen/`` (``RUN_KIND = "screen"``): branch screens, pilots and other
  experiments.  Neither scan reaches it, so a screen whose task, gate, plant and
  recipe digests equal a landed recipe's is never a seed replicate of a
  certification run nor its auto trunk.  Screens may count one another as
  replicates, as quick tests do, and a screen still builds on certified ancestors:
  ``"auto"`` scans ``<algo>/`` whatever the session's kind.  A screen pins a screen
  trunk as ``../<algo>_screen/<id>`` or by its absolute path, since a bare id
  resolves in ``<algo>/``.
* ``<algo>_quick_test/`` (``QUICK_TEST = True``, whatever ``RUN_KIND`` says): a
  50,000-step pipeline check, never a trunk or a replicate of a real run (decision
  D-D16 (b)).

The folder is the run's only record of its kind, as it is for quick tests: nothing
in ``provenance.json``, a stage record or a digest names it.  :func:`run_kind_of`
reads it back from a run's path (the Drive summary notebook's ``run_kind`` column,
and its strict promotion cell, which refuses a screen or a quick test), and the
storage cell refuses, before anything is written, a ``RUN_ID`` that names a run in
another tree of the model (:func:`refuse_run_id_of_another_tree`: a fresh runtime
resets ``RUN_KIND`` and ``QUICK_TEST``) and a certification run's ``TRUNK_FROM``
that pins a screen or a quick test (:func:`refuse_uncertified_trunk`).

The tree guards those two scans and the pinned trunk only.  Nothing here refuses a
certification run that loads, resumes or widens a screen's checkpoint (the manual
node's ``MANUAL_LOAD_PATH``, ``widen_checkpoint``, a command-line ``--load``), so a
certification run never trains from a screen's weights.  The screens trained before
``RUN_KIND`` existed sit in ``<algo>/``; :mod:`environments.shared.run_exclusions`
keeps the nine r10 stance screens among them out of every count.  Standard library
only.
"""

from __future__ import annotations

import os
from pathlib import Path

#: ``RUN_KIND`` for a run that may certify: the ``<algo>/`` tree.
CERTIFY = "certify"
#: ``RUN_KIND`` for a branch screen, a pilot or another experiment: the ``<algo>_screen/`` tree.
SCREEN = "screen"
#: The ``RUN_KIND`` knob's values, the default first.
RUN_KINDS = (CERTIFY, SCREEN)
#: The kind of a run in a ``<algo>_quick_test/`` tree (set by ``QUICK_TEST``, not a ``RUN_KIND`` value).
QUICK_TEST = "quick_test"
#: Each kind's tree: the lowercase algorithm, then this suffix.
TREE_SUFFIXES = {CERTIFY: "", SCREEN: "_screen", QUICK_TEST: "_quick_test"}
#: The configuration-cell knobs that select each kind's tree, as the refusals name them.
TREE_KNOBS = {
    CERTIFY: f'RUN_KIND = "{CERTIFY}" and QUICK_TEST = False',
    SCREEN: f'RUN_KIND = "{SCREEN}" and QUICK_TEST = False',
    QUICK_TEST: "QUICK_TEST = True",
}


def run_tree_name(algorithm: str, *, run_kind: str, quick_test: bool) -> str:
    """The tree a notebook run lives in: ``<algo>``, ``<algo>_screen`` or ``<algo>_quick_test``.

    *quick_test* wins over *run_kind*: a quick test of a screen is a quick test,
    under ``<algo>_quick_test``.  A *run_kind* that is not one of :data:`RUN_KINDS`
    raises ``ValueError``, under *quick_test* too, so a misspelt knob never trains.
    """
    if run_kind not in RUN_KINDS:
        raise ValueError(
            f'RUN_KIND={run_kind!r} is not a run kind: "{CERTIFY}" for a run that may certify (under '
            f'{algorithm.lower()}/) or "{SCREEN}" for a branch screen, a pilot or another experiment (under '
            f"{algorithm.lower()}{TREE_SUFFIXES[SCREEN]}/). A quick test is QUICK_TEST = True, whatever RUN_KIND says "
            f"(under {algorithm.lower()}{TREE_SUFFIXES[QUICK_TEST]}/)"
        )
    return algorithm.lower() + TREE_SUFFIXES[QUICK_TEST if quick_test else run_kind]


def run_tree_names(algorithm: str) -> tuple[str, ...]:
    """Every tree a notebook run of *algorithm* can live in: ``<algo>``, ``<algo>_screen`` and ``<algo>_quick_test``."""
    return tuple(algorithm.lower() + suffix for suffix in TREE_SUFFIXES.values())


def run_kind_of(run_dir: "str | Path") -> str:
    """The kind of the run at *run_dir*, read off the tree it sits in (its parent directory's name).

    :data:`QUICK_TEST` under a ``*_quick_test`` directory, :data:`SCREEN` under a
    ``*_screen`` one, else :data:`CERTIFY`: the ``<algo>/`` tree, and a run outside
    the notebook's trees (a flat ``<algo>_<timestamp>`` run from before 2026-03-31,
    a copy elsewhere).  Nothing is read and the path need not exist; a relative path
    such as ``ppo_screen/<run_id>`` works.  Resolve the path first to read the tree a
    link or a ``..`` lands in.
    """
    tree = Path(run_dir).parent.name
    for kind in (QUICK_TEST, SCREEN):
        if tree.endswith(TREE_SUFFIXES[kind]):
            return kind
    return CERTIFY


def refuse_run_id_of_another_tree(run_id: str, *, algorithm: str, run_dir: "str | Path") -> None:
    """Refuse an explicit ``RUN_ID`` that names no run in this session's tree but names one in another tree.

    *run_dir* is this session's run directory, ``<model>/<tree>/<run_id>``, and the
    other trees are the model's :func:`run_tree_names` for *algorithm*.  A fresh
    runtime resets ``RUN_KIND`` and ``QUICK_TEST`` to their defaults, so without this
    a screen resumed or re-entered with ``RUN_KIND`` left at ``"certify"`` would open
    a new certification run under the screen's id, and the chain loop would train the
    screen's recipe into the certification tree.  A run id therefore names one run of
    a model, whatever its tree: re-entering a run takes the knobs that reach its tree,
    and a new run takes an id no tree of the model uses.  ``""`` (the memo, or a fresh
    timestamp) and a run that exists in this session's tree pass.  Raises
    ``ValueError``; it only checks which directories exist and writes nothing, so the
    storage cell calls it before it creates the run directory.
    """
    run_dir = Path(run_dir)
    if not run_id or run_dir.exists():
        return
    model_dir = run_dir.parent.parent
    found = [
        model_dir / tree / run_id
        for tree in run_tree_names(algorithm)
        if tree != run_dir.parent.name and (model_dir / tree / run_id).is_dir()
    ]
    if not found:
        return
    named = "; ".join(f"{path} (reached with {TREE_KNOBS[run_kind_of(path)]})" for path in found)
    raise ValueError(
        f"RUN_ID={run_id!r} names no run under {run_dir.parent}, the tree RUN_KIND and QUICK_TEST select, but it "
        f"names {named}. A fresh runtime resets RUN_KIND and QUICK_TEST to their defaults: to re-enter or resume "
        "that run, set them as its session had them and run the storage cell again; to start a new run, choose a "
        "RUN_ID that no tree of this model uses. Nothing has been written."
    )


def refuse_uncertified_trunk(trunk_from: str, *, log_dir: "str | Path", run_dir: "str | Path") -> None:
    """Refuse a certification run's ``TRUNK_FROM`` when it pins a screen or a quick test.

    *trunk_from* is the notebook's ``TRUNK_FROM``, resolved as the storage cell
    resolves it: a run id (or a relative path, normalised lexically, so
    ``../<algo>_screen/x`` needs no ``<algo>/`` directory) under *log_dir*, the
    certification tree a ``TRUNK_FROM`` id resolves under, and an absolute path as
    itself; the resolved directory's tree decides (:func:`run_kind_of`), so
    ``../<algo>_screen/x`` and a link into a screen tree are refused too.
    *run_dir* is this session's run directory: only a run in the certification
    tree is held to this, and a screen or a quick test may pin any trunk.  ``""``
    (no trunk; any false value, as the storage cell reads it) and ``"auto"``
    (which scans *log_dir* only) pass.  An existing certification run that
    recorded such a trunk before this refusal existed is refused too: it is never
    continued as a certification run, so its node is trained again in a new
    ``RUN_ID``.  Raises ``ValueError``; it opens no file and writes nothing, so
    the storage cell calls it before it creates the run directory.
    """
    if not trunk_from or trunk_from == "auto" or run_kind_of(run_dir) != CERTIFY:
        return
    trunk_dir = Path(os.path.normpath(Path(log_dir) / trunk_from)).resolve()
    kind = run_kind_of(trunk_dir)
    if kind == CERTIFY:
        return
    label = kind.replace("_", " ")
    existing = (
        f" This run already exists: a certification run is never continued on a {label} trunk, so train the node "
        "again in a new RUN_ID."
        if Path(run_dir).exists()
        else ""
    )
    raise ValueError(
        f"TRUNK_FROM={trunk_from!r} resolves to {trunk_dir}, a {label} run (under {trunk_dir.parent.name}/), and "
        f'this session\'s run {Path(run_dir).name!r} is a certification run (RUN_KIND = "{CERTIFY}", under '
        f"{Path(run_dir).parent}): a certification run builds only on certification runs, never on a {label}. Pin "
        f'a run under {Path(log_dir)} or set TRUNK_FROM = "auto", which scans only that tree, or train this '
        f'session as a screen (RUN_KIND = "{SCREEN}").{existing} Nothing has been written.'
    )


def trunk_id_hint(trunk_from: str, *, log_dir: "str | Path", run_dir: "str | Path") -> str:
    """What the storage cell adds to its refusal of a ``TRUNK_FROM`` that is no run under *log_dir*.

    A ``TRUNK_FROM`` run id resolves under *log_dir*, the certification tree, whatever
    ``RUN_KIND`` says.  When *trunk_from* is a bare id and a run of that id sits in
    the model's screen or quick-test tree, the returned text (starting ``". "``)
    names it and, for a screen or a quick test (*run_dir*'s tree), how to pin it:
    ``../<tree>/<id>`` or its absolute path; a certification run never pins it.
    Otherwise ``""``.  Only checks which directories exist.
    """
    if not trunk_from or trunk_from == "auto" or Path(trunk_from).name != trunk_from:
        return ""
    log_dir = Path(log_dir)
    found = [log_dir.parent / f"{log_dir.name}{TREE_SUFFIXES[kind]}" / trunk_from for kind in (SCREEN, QUICK_TEST)]
    found = [path for path in found if path.is_dir()]
    if not found:
        return ""
    certifies = run_kind_of(run_dir) == CERTIFY
    named = "; ".join(
        f"{path} has that id, a {run_kind_of(path).replace('_', ' ')} run"
        + (
            ", which a certification run never pins"
            if certifies
            else f": pin it as TRUNK_FROM = {f'../{path.parent.name}/{path.name}'!r} or by its absolute path"
        )
        for path in found
    )
    return f". A TRUNK_FROM run id resolves under {log_dir} whatever RUN_KIND says; {named}"
