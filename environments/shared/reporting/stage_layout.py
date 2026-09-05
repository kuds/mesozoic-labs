"""Where a stage's generated artifacts live inside its stage directory.

One module owns the stage-directory layout so writers and readers cannot
drift apart.  Before this existed the figures, the replay videos, and the
per-frame stance CSVs were written straight into the stage root by three
unrelated call sites, and the only reader (``config.upload_curriculum_artifacts``)
found them with a hand-written ``glob("*.mp4")`` — a pairing that breaks
silently the moment either side moves.

The layout
----------
Generated, renderable artifacts are grouped by kind::

    stage1/
    ├── figures/    training_curves.png  foot_contacts.png  stance_diagnostics.png
    │               locomotion_health.png  behavioral_metrics.png
    ├── replays/    <species>_<algo>_stage<N>_<label>.mp4  (+ _front/_side views)
    │               <species>_<algo>_stage<N>_<label>_stance.csv
    ├── models/     (unchanged)
    ├── tensorboard/ (unchanged)
    └── stage_summary.txt  stage_config.json  plant_identity.json
        evaluations.npz  diagnostics.npz  evaluation_*.csv   (unchanged)

Only the *generated* artifacts move.  ``evaluations.npz``, ``diagnostics.npz``
and ``evaluation_{selected,final}.csv`` deliberately stay in the stage root:
the first two are written during training by SB3 callbacks on the hot path
and are read by a dozen call sites including the sweep tooling, and the
evaluation CSVs are the publication evidence contract that
``result_bundle.audit`` names by fixed relative path.  Moving those is a
larger change with a different risk profile and is not what this module is
for.

Filenames inside ``replays/`` are unchanged.  They still carry the
redundant ``<species>_<algo>_stage<N>`` prefix that the path already
states; dropping it is a separate rename, and doing both at once would
make a mechanical move indistinguishable from a semantic one in review.

Reading old runs
----------------
Every accessor here falls back to the legacy flat location, so the ~35
existing runs on Drive keep resolving.  New runs write only the nested
form; nothing rewrites history.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

#: Subdirectory holding generated plots.
FIGURES_DIRNAME = "figures"

#: Subdirectory holding replay videos and their per-frame stance CSVs.
REPLAYS_DIRNAME = "replays"

#: Suffixes that belong under ``replays/``.  The stance CSV is a per-frame
#: trace of the very episode the video shows, so it travels with it rather
#: than with the evaluation evidence.
_REPLAY_SUFFIXES: tuple[str, ...] = (".mp4", ".csv")

#: Figures this layout knows by name.  Used only to recognise legacy
#: stage roots, where a figure is not distinguishable from any other PNG.
FIGURE_NAMES: tuple[str, ...] = (
    "training_curves.png",
    "foot_contacts.png",
    "stance_diagnostics.png",
    "locomotion_health.png",
    "behavioral_metrics.png",
)


def figures_dir(stage_dir: "str | Path", *, create: bool = False) -> Path:
    """Return the directory generated plots are written to."""
    path = Path(stage_dir) / FIGURES_DIRNAME
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def replays_dir(stage_dir: "str | Path", *, create: bool = False) -> Path:
    """Return the directory replay videos and stance CSVs are written to."""
    path = Path(stage_dir) / REPLAYS_DIRNAME
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def iter_figures(stage_dir: "str | Path") -> Iterator[Path]:
    """Yield every figure in a stage directory, nested or legacy.

    A nested figure shadows a legacy one of the same name: a run written
    by a newer trainer into an older stage directory would otherwise be
    reported twice.
    """
    stage_path = Path(stage_dir)
    seen: set[str] = set()
    nested = stage_path / FIGURES_DIRNAME
    if nested.is_dir():
        for path in sorted(nested.glob("*.png")):
            seen.add(path.name)
            yield path
    for name in FIGURE_NAMES:
        if name in seen:
            continue
        legacy = stage_path / name
        if legacy.is_file():
            yield legacy


def iter_replay_files(stage_dir: "str | Path") -> Iterator[Path]:
    """Yield every replay video and stance CSV, nested or legacy.

    Legacy stage roots hold the videos beside ``evaluation_*.csv`` and
    ``stage_config.json``, so the flat branch matches ``*.mp4`` plus only
    those CSVs whose name ends in ``_stance`` — matching every CSV there
    would sweep up the publication evidence files.
    """
    stage_path = Path(stage_dir)
    seen: set[str] = set()
    nested = stage_path / REPLAYS_DIRNAME
    if nested.is_dir():
        for path in sorted(nested.iterdir()):
            if path.is_file() and path.suffix in _REPLAY_SUFFIXES:
                seen.add(path.name)
                yield path
    # `is_file()` on the legacy branch too: the nested branch guards it, and an
    # accessor that yields a directory for one layout but not the other hands
    # its caller -- the GCS upload -- something it cannot read.
    for pattern in ("*.mp4", "*_stance.csv"):
        for path in sorted(stage_path.glob(pattern)):
            if path.name not in seen and path.is_file():
                yield path


def iter_generated_artifacts(stage_dir: "str | Path") -> Iterator[Path]:
    """Yield every figure and replay artifact for a stage, in that order."""
    yield from iter_figures(stage_dir)
    yield from iter_replay_files(stage_dir)


@contextmanager
def staged_artifacts(stage_dir: "str | Path", *, enabled: bool = True) -> Iterator[Path]:
    """Render generated artifacts locally, then publish them in one pass.

    Yields a directory to write into that mirrors the stage layout.  On
    exit every file below it is copied into *stage_dir* at the same
    relative path, via :func:`~environments.shared.file_io.atomic_copy`.

    Why this exists: a stage directory is normally a Drive or GCS-FUSE
    mount, and both matplotlib's ``savefig`` and mediapy's encoder write
    incrementally.  Each flush of a 700 KB mp4 or a 300 KB png is a
    separate round trip to remote storage, interleaved with the encode
    itself.  Rendering to local scratch turns that into one sequential
    copy per file, and means a runtime that dies mid-encode leaves the
    stage directory holding the previous complete artifact set rather
    than a half-written video.

    Sized on run ``20260801_021545`` stage 1 — the 13 generated artifacts
    there total 8.15 MB, which at a 64 KB flush is **130 writes against
    the mount, reduced to 13**.  The absolute time saved is modest (a
    fraction of a second at a plausible per-flush latency); the durability
    and the untouched mount during encode are the point.  The dominant
    Drive cost in a stage directory remains the ~30 model and vecnorm
    files under ``models/``, which this does not address.

    Publication happens in a ``finally``, so the artifacts that *did*
    render still land when a later one raises.  That matters because
    ``generate_stage_artifacts`` records the best and final replays in
    sequence: a plant-identity mismatch on the second must not discard
    the first.

    Pass ``enabled=False`` to write straight through to *stage_dir* —
    used by callers that are already writing to local disk, where the
    staging copy is pure overhead.
    """
    destination = Path(stage_dir)
    if not enabled:
        destination.mkdir(parents=True, exist_ok=True)
        yield destination
        return

    staging = Path(tempfile.mkdtemp(prefix="mesozoic-stage-artifacts-"))
    try:
        yield staging
    finally:
        try:
            published, total_bytes = publish_tree(staging, destination)
            if published:
                logger.info(
                    "Published %d generated artifact(s) (%.1f MB) to %s",
                    published,
                    total_bytes / 1e6,
                    destination,
                )
        except Exception:  # noqa: BLE001 - publication must not sink the run
            logger.warning("Publishing staged artifacts to %s failed", destination, exc_info=True)
        finally:
            shutil.rmtree(staging, ignore_errors=True)


def publish_tree(source: "str | Path", destination: "str | Path") -> tuple[int, int]:
    """Atomically copy every file under *source* into *destination*.

    Relative paths are preserved, so ``source/figures/x.png`` lands at
    ``destination/figures/x.png``.  Returns ``(file_count, total_bytes)``.

    A file that fails to copy is logged and skipped rather than aborting
    the rest: these are diagnostics, and losing one plot must not cost a
    completed run the other twelve artifacts.
    """
    from ..file_io import atomic_copy

    source_path = Path(source)
    destination_path = Path(destination)
    published = 0
    total_bytes = 0
    if not source_path.is_dir():
        return 0, 0

    for path in sorted(source_path.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source_path)
        try:
            size = path.stat().st_size
            atomic_copy(path, destination_path / relative)
        except Exception:  # noqa: BLE001 - one artifact must not sink the rest
            logger.warning("Could not publish staged artifact %s", relative, exc_info=True)
            continue
        published += 1
        total_bytes += size
    return published, total_bytes
