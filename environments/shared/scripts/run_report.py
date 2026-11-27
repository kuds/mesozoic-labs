"""Write a run's report and its model's run index, outside every run (the maintainer's decision of 2026-11-26).

Report only, and read-only on runs.  A run directory holds dozens of files
in three layout generations, and until now the only way to learn what a run
holds (its state, each node's verdict, its certificate, which probes ran,
where its figures and replays are) was to open them one by one on Drive.
This tool reads a run's small records and its artifact manifest and writes
one page for it, and one index of every run of the model::

    python -m environments.shared.scripts.run_report RUN_DIR [--model-dir MODEL_DIR]
    python -m environments.shared.scripts.run_report --model-dir LOG_BASE/<species>/<embodiment>

The first form writes RUN_DIR's report folder and rebuilds its model's index;
the model directory is RUN_DIR's by the layout (``<model>/<tree>/<run_id>``,
or ``<model>/<run>`` for a flat run from before 2026-03-31 or a command-line
curriculum run; ``reporting.run_index.model_dir_of``) unless ``--model-dir``
names it.  Both paths are resolved first, so ``.`` and a bare run id typed
inside a tree work.  The second form writes the report folder of every run of
every tree of the model (``run_index.discover_runs``; runs that reused no
trunk first, so a reused node's link to its trunk's report resolves), then
the index: the backfill for runs already on Drive, which the Drive summary
notebook's last cell runs for every model.

Layout: the tree sits beside the model's run trees, never inside a run, and
is never hashed::

    LOG_BASE/<species>/<embodiment>/reports/
        INDEX.md                         one row per run, grouped by state, newest first
        index.json                       the same rows ("mesozoic.model-index/v1")
        <algo>[_screen|_quick_test]/<run_id>/
            report.md                    the run's page: its state, a row per node, the certificate and
                                         probe lines, its figures, training history and replays, warnings
            index.json                   every file of the run, with its stage, role, size and the digest
                                         artifact_manifest.json records ("mesozoic.run-index/v1")

The run states are ``complete``, ``partial`` and ``failed`` (the status the
run's manifest records), ``bundle_unreadable``, and without a manifest
``judged_no_bundle``, ``post_training_cut``, ``training_cut`` and ``empty``;
the index adds ``unreadable`` for a run whose records raised while it read
them.  ``reporting/run_index.py`` defines each from the files a run holds.
A run whose recorded plant is not the model's current one is *stranded*
(``run_index.plant_drift``): its page and its index row say so, and its page
names no step that re-enters or re-rolls it.  Every run written before
decision D-D59 is (``configs/plant_versions.toml``, note 24), and keeps the
stage directory of the hunt it trained, which still reads as its stage.  A
run whose ``provenance.json`` records another plant identity only in fields
no checkpoint depends on (``run_index.reentry_drift``) is not stranded, but
the storage cell refuses to re-enter it, so its page names no such step
either.  Where the plant contract cannot be imported (MuJoCo is not
installed), every page is still written and says the current plant cannot
be read.

What it writes: only under ``<model_dir>/reports/``, and only what changed.
The tool owns ``report.md`` and ``index.json`` in each run's report folder
and the two index files.  A run's report folder is replaced whole: built in
a hidden temporary folder beside it, then renamed over it, so a build that
fails keeps the old folder; anything else a report folder holds (another
tool's output) is copied into the new folder, and the tool prints its name.
Report-only output of your own is best kept in a folder of ``reports/`` that
is no run's report folder (``reports/<tree>/<run_id>.probes/``, say), which
the tool never touches; nor does it touch a report folder whose run is gone,
which is left as it is and no longer indexed.  A temporary folder a lost
runtime left beside a report folder is removed on the next build (the old
folder put back first when the report folder itself is missing).  The two
index files are each replaced atomically (``file_io``).  A file whose new
text equals the old is not rewritten, so an unchanged run costs Drive no
write and no trash.

Refusals, before anything is written: a RUN_DIR that is no run
(``run_index.is_run_dir``), a run that is not inside the model directory, a
model directory that is no ``<species>/<embodiment>`` directory, that is a
run tree or ``reports/``, that is or lies inside a run of any generation,
and an output inside any run directory
(``result_bundle.refuse_output_inside_run``: a run's manifest hashes every
file under it, and verification rejects a file it did not list).  It prints
what it wrote, and what was already up to date.

Cost: it opens only small records (``provenance.json``, ``trunk_run.json``,
``stage_config.json``, ``gate_verdict.json``, ``ancestor.json``, the
certificate and probe ``.txt`` reports, never their JSON, and a
``plant_identity.json`` for a run whose ``provenance.json`` records no plant),
the committed generated plant manifest once, and the artifact manifest, and
lists the run's files.  Nothing is hashed: sizes and digests
are copied from ``artifact_manifest.json``, and a run without one gets its
sizes from the listing and no digest.  Each index rebuild (every call of the
first form: the SB3 notebook's every ``save_run_bundle`` and its verify
cell) also reads, for every other run of the model, its manifest's status,
its ``provenance.json`` and each stage's ``gate_verdict.json``, with a
directory listing per run and stage (and, for a run of the certification
tree, each stage's ``stage_config.json``, for ``run_exclusions``): a few
small reads per run, which grows with the model's run count.  A partial,
cut-off, legacy or empty run is a state the page names, with the next step
for an unfinished one; a file that cannot be read is a warning on the run's
page, never a failure.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from environments.shared.file_io import atomic_write_text
from environments.shared.model_id import ModelId
from environments.shared.reporting import run_index
from environments.shared.result_bundle import ResultBundleError, refuse_output_inside_run
from environments.shared.result_bundle.constants import ANCESTORS_DIRNAME

#: The files the tool writes in a run's report folder; anything else there is carried into each rebuild.
TOOL_FILES = (run_index.REPORT_FILENAME, run_index.INDEX_JSON_FILENAME)
_NOTHING_WRITTEN = "Nothing has been written."
_PASS_MODEL = "pass --model-dir LOG_BASE/<species>/<embodiment>"


class RunReportError(RuntimeError):
    """A refusal: nothing has been written."""


class Written(dict[str, Path]):
    """The paths a call wrote or found up to date (name -> path), with ``unchanged`` naming the ones not rewritten.

    ``kept`` lists the entries of a run's report folder the tool did not
    write, carried into the rebuilt folder.
    """

    def __init__(self, **paths: Path) -> None:
        super().__init__(paths)
        self.unchanged: set[str] = set()
        self.kept: list[str] = []


def _refuse_model_dir(model_dir: Path) -> None:
    if not model_dir.is_dir():
        raise RunReportError(f"{model_dir} is no directory: {_PASS_MODEL}. {_NOTHING_WRITTEN}")
    if run_index.is_run_dir(model_dir):
        raise RunReportError(
            f"{model_dir} is a run, not a model directory: pass the run as RUN_DIR, or {_PASS_MODEL}. "
            f"{_NOTHING_WRITTEN}"
        )
    if model_dir.name in run_index.RUN_TREES or model_dir.name == run_index.REPORTS_DIRNAME:
        raise RunReportError(
            f"{model_dir} is a run tree, not a model directory: pass its parent, LOG_BASE/<species>/<embodiment>. "
            f"{_NOTHING_WRITTEN}"
        )
    try:
        ModelId(model_dir.parent.name, model_dir.name)
    except ValueError as exc:
        raise RunReportError(
            f"{model_dir} is not a model directory LOG_BASE/<species>/<embodiment> ({exc}): {_PASS_MODEL}. "
            f"{_NOTHING_WRITTEN}"
        ) from exc
    # A run of any generation: refuse_output_inside_run below knows a run only by its provenance.json.
    for parent in model_dir.parents:
        if run_index.is_run_dir(parent):
            raise RunReportError(
                f"{model_dir} lies inside the run directory {parent}: its reports/ would be written inside a run. "
                f"{_PASS_MODEL} outside every run. {_NOTHING_WRITTEN}"
            )


def _refuse(run_dir: "Path | None", model_dir: Path) -> None:
    """Every refusal of the module docstring, before anything is written.  Both paths are resolved."""
    _refuse_model_dir(model_dir)
    if run_dir is not None:
        if not run_index.is_run_dir(run_dir):
            hint = "pass LOG_BASE/<species>/<embodiment>/<tree>/<run_id>"
            if run_index.is_run_dir(run_dir.parent):
                hint = f"this looks like a directory inside a run: pass the run, {run_dir.parent}"
            raise RunReportError(
                f"{run_dir} is no run directory: it holds no provenance.json, artifact_manifest.json or "
                f"collected_results.csv and no stage directory; {hint}. {_NOTHING_WRITTEN}"
            )
        try:
            run_index.report_dir_for(run_dir, model_dir)
        except ValueError as exc:
            raise RunReportError(
                f"{run_dir} is not a run inside the model directory {model_dir} ({exc}); pass --model-dir "
                f"LOG_BASE/<species>/<embodiment> for it. {_NOTHING_WRITTEN}"
            ) from exc
    reports = model_dir / run_index.REPORTS_DIRNAME
    try:
        refuse_output_inside_run(reports)
    except ResultBundleError as exc:
        raise RunReportError(f"{exc} {_NOTHING_WRITTEN}") from exc


def _default_dir_mode() -> int:
    """The mode a plain ``mkdir`` creates a directory with under the current umask (``mkdtemp`` makes 0700)."""
    mask = os.umask(0)
    os.umask(mask)
    return 0o777 & ~mask


def _same_text(path: Path, text: str) -> bool:
    try:
        return path.is_file() and path.read_text(encoding="utf-8") == text
    except (OSError, ValueError):
        return False


def _write_text_if_changed(path: Path, text: str) -> bool:
    """Write *text* to *path* atomically unless the file holds it already; whether it was written."""
    if _same_text(path, text):
        return False
    atomic_write_text(path, text)
    return True


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _clear_stale(target: Path) -> None:
    """Remove what a build whose runtime was lost left beside *target*, putting the old folder back if *target* is gone.

    A staging folder (``.<name>.<random>.tmp``) holds only what a build
    wrote or copied, so it goes; an old folder (``.<name>.old-<hex>``) is
    the previous report folder, renamed aside just before the new one was
    renamed in: it goes too, unless *target* is missing, when the newest is
    renamed back first.
    """
    name = re.escape(target.name)
    staging = re.compile(rf"\.{name}\.[a-z0-9_]{{8}}\.tmp")
    aside = re.compile(rf"\.{name}\.old-[0-9a-f]{{8}}")
    try:
        siblings = list(target.parent.iterdir())
    except OSError:
        return
    olds = sorted((path for path in siblings if aside.fullmatch(path.name)), key=_mtime)
    if olds and not target.exists():
        olds.pop().rename(target)
    for path in olds + [path for path in siblings if staging.fullmatch(path.name)]:
        shutil.rmtree(path, ignore_errors=True)


def _clear_stale_temporaries(path: Path) -> None:
    """Remove the ``.<name>.<random>.tmp`` files ``file_io.atomic_write_text`` left beside *path* on a lost runtime."""
    stale = re.compile(rf"\.{re.escape(path.name)}\.[a-z0-9_]{{8}}\.tmp")
    try:
        siblings = list(path.parent.iterdir())
    except OSError:
        return
    for sibling in siblings:
        if stale.fullmatch(sibling.name) and sibling.is_file():
            try:
                sibling.unlink()
            except OSError:
                pass


def _replace_folder(target: Path, files: dict[str, str]) -> tuple[bool, list[str]]:
    """Replace the folder *target* with one holding *files* (name -> text), whole; keep the old one on a failure.

    Returns whether it was written (``False``: every file already held its
    text, so nothing was touched) and the names of the entries the folder
    held that are not in *files*, which are copied into the new folder.
    """
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    _clear_stale(target)
    carried = sorted(path.name for path in target.iterdir() if path.name not in files) if target.is_dir() else []
    if target.is_dir() and all(_same_text(target / name, text) for name, text in files.items()):
        return False, carried
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".tmp", dir=parent))
    try:
        os.chmod(staging, _default_dir_mode())
        for name, text in files.items():
            (staging / name).write_text(text, encoding="utf-8")
        for name in carried:
            source = target / name
            if source.is_dir() and not source.is_symlink():
                shutil.copytree(source, staging / name, symlinks=True)
            else:
                shutil.copy2(source, staging / name, follow_symlinks=False)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    old = parent / f".{target.name}.old-{uuid.uuid4().hex[:8]}" if target.exists() else None
    if old is not None:
        target.rename(old)
    try:
        staging.rename(target)
    except BaseException:
        if old is not None:
            old.rename(target)
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if old is not None:
        shutil.rmtree(old, ignore_errors=True)
    return True, carried


def _write_model_index(model: Path, summaries: "dict[str, dict] | None" = None) -> Written:
    rows = run_index.build_model_index(model, summaries=summaries)
    reports = model / run_index.REPORTS_DIRNAME
    written = Written(
        model_index=reports / run_index.INDEX_MD_FILENAME, model_index_json=reports / run_index.INDEX_JSON_FILENAME
    )
    texts = {
        "model_index": run_index.render_model_index(model, rows),
        "model_index_json": json.dumps(run_index.model_index_document(model, rows), indent=2) + "\n",
    }
    for name, text in texts.items():
        _clear_stale_temporaries(written[name])
        if not _write_text_if_changed(written[name], text):
            written.unchanged.add(name)
    return written


def write_model_index(model_dir: "str | Path") -> Written:
    """Rebuild ``<model_dir>/reports/INDEX.md`` and ``index.json`` from the model's runs, each atomically."""
    model = Path(model_dir).resolve()
    _refuse(None, model)
    return _write_model_index(model)


def write_run_report(
    run_dir: "str | Path", *, model_dir: "str | Path | None" = None, rebuild_index: bool = True
) -> Written:
    """Write *run_dir*'s report folder (``report.md`` and ``index.json``) whole, then rebuild its model's index.

    *model_dir* defaults to ``run_index.model_dir_of``; both are resolved.
    Returns the paths, ``report`` and ``index`` (and ``model_index`` and
    ``model_index_json`` with *rebuild_index*), and which were up to date.
    Raises :class:`RunReportError` for a refusal (nothing written).
    """
    run = Path(run_dir).resolve()
    model = Path(model_dir).resolve() if model_dir is not None else run_index.model_dir_of(run)
    _refuse(run, model)
    folder = run_index.report_dir_for(run, model)
    built = run_index.build_run_index(run, model_dir=model)
    page = run_index.render_run_report(run, model_dir=model, report_dir=folder, index=built)
    document = {key: value for key, value in built.items() if key != "summary"}
    changed, kept = _replace_folder(
        folder,
        {
            run_index.REPORT_FILENAME: page,
            run_index.INDEX_JSON_FILENAME: json.dumps(document, indent=2) + "\n",
        },
    )
    written = Written(report=folder / run_index.REPORT_FILENAME, index=folder / run_index.INDEX_JSON_FILENAME)
    written.kept = kept
    if not changed:
        written.unchanged.update(("report", "index"))
    elif kept:
        print(f"kept in {folder} (not written by run_report, carried into the rebuilt folder): {', '.join(kept)}")
    if rebuild_index:
        index = _write_model_index(model, {built["summary"]["path"]: built["summary"]})
        written.update(index)
        written.unchanged |= index.unchanged
    return written


@dataclass
class ModelReports:
    """What :func:`write_model_reports` did: the reports rewritten, those up to date, the runs that failed, the index."""

    reports: list[Path] = field(default_factory=list)
    unchanged: list[Path] = field(default_factory=list)
    failed: list[tuple[Path, str]] = field(default_factory=list)
    index: Written = field(default_factory=Written)


def write_model_reports(model_dir: "str | Path") -> ModelReports:
    """Write the report folder of every run of the model at *model_dir*, then its index.

    Runs that reused no trunk node (no ``ancestors/``) go first, so a
    reused node's link to its trunk run's report resolves.  A run whose
    report cannot be built keeps its old folder and is listed in
    ``failed``; the others and the index are written all the same.
    """
    model = Path(model_dir).resolve()
    _refuse(None, model)
    result = ModelReports()
    runs = sorted(run_index.discover_runs(model), key=lambda run: (run / ANCESTORS_DIRNAME).is_dir())
    for run in runs:
        try:
            written = write_run_report(run, model_dir=model, rebuild_index=False)
        except Exception as exc:  # noqa: BLE001 - one run must never cost the model the others' reports
            result.failed.append((run, f"{type(exc).__name__}: {exc}"))
            continue
        (result.unchanged if "report" in written.unchanged else result.reports).append(written["report"])
    result.index = _write_model_index(model)
    return result


def _print(written: Written) -> None:
    for name, path in written.items():
        print(f"{'unchanged' if name in written.unchanged else 'written'}: {path}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", nargs="?", type=Path, help="the run to report (omit with --model-dir: every run)")
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="the model directory, LOG_BASE/<species>/<embodiment> (default with RUN_DIR: by its layout)",
    )
    return parser


def main(argv: "Sequence[str] | None" = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.run_dir is None and args.model_dir is None:
        parser.error("name a RUN_DIR, or --model-dir for every run of a model")
    try:
        if args.run_dir is not None:
            _print(write_run_report(args.run_dir, model_dir=args.model_dir))
            return 0
        result = write_model_reports(args.model_dir)
    except RunReportError as exc:
        # The CLI boundary is where a refusal becomes an exit status.
        raise SystemExit(str(exc)) from exc
    for path in result.reports:
        print(f"written: {path}")
    for path in result.unchanged:
        print(f"unchanged: {path}")
    _print(result.index)
    for run, error in result.failed:
        print(
            f"not written (its previous report, if any, is kept): {run}: {error}; rerun "
            f"`python -m environments.shared.scripts.run_report {run}` to see it again"
        )
    return 1 if result.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
