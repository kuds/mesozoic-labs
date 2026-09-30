"""TensorBoard local-buffer helpers for FUSE-mounted remote stores.

Writing TensorBoard events straight to a FUSE mount (Google Drive on Colab,
or ``/gcs/...``) is slow, and an event file held open all stage loses its
un-uploaded tail when the runtime is reclaimed (review CO5).  These helpers
let the SB3 trainer buffer events on local disk and sync them to the mount
on the checkpoint cadence and at the end of a stage.  The ``gcs`` in
``_sync_tb_to_gcs`` and ``gcs_tb_path`` is historical: both serve a Drive
mount the same way.

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

    Returns a ``Path`` under ``/tmp`` named after the remote path, so
    concurrent runs don't collide.
    """
    # Stable suffix derived from the remote path so restarts reuse the dir.
    suffix = str(gcs_tb_path).replace("/", "_")
    local_dir = Path(tempfile.gettempdir()) / "tb_buffer" / suffix
    local_dir.mkdir(parents=True, exist_ok=True)
    return local_dir


def _mirror_remote_run_dirs(local_tb_dir: Path, remote_tb_path: str | Path) -> list[str]:
    """Recreate the remote TensorBoard run-directory NAMES inside the local buffer.

    SB3 numbers each ``learn()``'s run directory from what it finds under
    ``tensorboard_log`` (``get_latest_run_id``: the greatest
    ``<tb_log_name>_<N>``), and a resume (``reset_num_timesteps=False``)
    continues the latest one.  Buffering hides the mount from that scan: on
    a fresh runtime the buffer is empty, so a resumed session landed in
    ``PPO_0`` — and every later resume too — while the pre-crash session sat
    in ``PPO_1``, merging sessions.  Empty directories are all the scan
    needs, and :func:`_sync_tb_to_gcs` copies only files, so mirroring writes
    nothing back to the mount; the new event file then syncs into the same
    remote run directory the interrupted session used.

    Returns the mirrored names (sorted); ``[]`` when the remote directory
    does not exist yet.
    """
    remote = Path(remote_tb_path)
    if not remote.is_dir():
        return []
    mirrored = []
    for entry in remote.iterdir():
        name, sep, run_id = entry.name.rpartition("_")
        if sep and name and run_id.isdigit() and entry.is_dir():
            (Path(local_tb_dir) / entry.name).mkdir(parents=True, exist_ok=True)
            mirrored.append(entry.name)
    return sorted(mirrored)


def _sync_tb_to_gcs(local_tb_dir: Path, gcs_tb_path: str | Path, cleanup: bool = True) -> None:
    """Copy locally-buffered TensorBoard events to the remote FUSE mount.

    Uses :func:`shutil.copy` (not ``copy2``) because GCS FUSE does not
    support the ``os.utime`` / ``os.chmod`` calls that ``copy2`` makes
    to preserve file metadata, which causes ``OSError`` on FUSE mounts.

    Args:
        local_tb_dir: The local buffer directory.
        gcs_tb_path: Destination directory on the remote FUSE mount.
        cleanup: Remove the local buffer after copying.  Pass ``False``
            for mid-training syncs where the writer still holds the files.
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
    if cleanup:
        shutil.rmtree(local_tb_dir, ignore_errors=True)


try:
    from stable_baselines3.common.callbacks import BaseCallback as _BaseCallback
except ImportError:  # SB3 is an optional dependency
    _BaseCallback = object  # type: ignore[misc,assignment]


class PeriodicTbSyncCallback(_BaseCallback):
    """Periodically flush the local TensorBoard buffer to the remote mount.

    Without this, buffered events reach the mount only at stage end
    (:func:`_sync_tb_to_gcs` from ``_save_final_and_sync_tb``), so a
    reclaimed Colab runtime or a crashed worker loses the entire stage's
    TensorBoard logs.

    Args:
        local_tb_dir: Local buffer directory (from ``_make_local_tb_dir``).
        gcs_tb_path: Destination on the remote FUSE mount.
        sync_freq: Sync every N timesteps (compared against
            ``num_timesteps``, which advances by ``n_envs`` per step).
    """

    def __init__(
        self,
        local_tb_dir: Path,
        gcs_tb_path: str | Path,
        sync_freq: int = 500_000,
        verbose: int = 0,
    ):
        if _BaseCallback is not object:
            super().__init__(verbose)
        self.local_tb_dir = Path(local_tb_dir)
        self.gcs_tb_path = gcs_tb_path
        self.sync_freq = sync_freq
        self._last_sync_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_sync_step >= self.sync_freq:
            self._last_sync_step = self.num_timesteps
            try:
                _sync_tb_to_gcs(self.local_tb_dir, self.gcs_tb_path, cleanup=False)
            except Exception:
                logger.warning("Periodic TensorBoard sync to the remote mount failed.", exc_info=True)
        return True
