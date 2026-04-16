"""TensorBoard local-buffer helpers for GCS FUSE mounts.

Writing TensorBoard events directly to ``/gcs/...`` is slow and produces
partial-write corruption under concurrent Vertex AI HPT trials.  These
helpers let the SB3 trainer buffer events to local disk and sync to GCS
at the end of a stage.

Extracted from :mod:`environments.shared.train_base` for reuse and
testability.  That module re-exports these names so existing
``from environments.shared.train_base import _sync_tb_to_gcs`` imports
keep working.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def _is_gcs_path(path: str | Path) -> bool:
    """Return True if *path* is on a GCS FUSE mount (``/gcs/...``)."""
    return str(path).startswith("/gcs/")


def _make_local_tb_dir(gcs_tb_path: str | Path) -> Path:
    """Create a local temp directory for TensorBoard event buffering.

    Returns a ``Path`` under ``/tmp`` that mirrors the GCS structure so
    concurrent trials don't collide.
    """
    # Stable suffix derived from the GCS path so restarts reuse the dir.
    suffix = str(gcs_tb_path).replace("/", "_")
    local_dir = Path(tempfile.gettempdir()) / "tb_buffer" / suffix
    local_dir.mkdir(parents=True, exist_ok=True)
    return local_dir


def _sync_tb_to_gcs(local_tb_dir: Path, gcs_tb_path: str | Path) -> None:
    """Copy locally-buffered TensorBoard events to the GCS FUSE mount.

    Uses :func:`shutil.copy` (not ``copy2``) because GCS FUSE does not
    support the ``os.utime`` / ``os.chmod`` calls that ``copy2`` makes
    to preserve file metadata, which causes ``OSError`` on FUSE mounts.
    """
    gcs_dest = Path(gcs_tb_path)
    if not local_tb_dir.exists():
        return
    gcs_dest.mkdir(parents=True, exist_ok=True)
    n_copied = 0
    for src_file in local_tb_dir.rglob("*"):
        if src_file.is_file():
            rel = src_file.relative_to(local_tb_dir)
            dest = gcs_dest / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_file, dest)
            n_copied += 1
    logger.info("Synced %d TensorBoard files to %s", n_copied, gcs_dest)
    shutil.rmtree(local_tb_dir, ignore_errors=True)
