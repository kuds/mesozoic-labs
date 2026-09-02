"""Atomic file-write helpers for training artifacts.

Training artifacts frequently live on FUSE-mounted remote storage
(Google Drive on Colab, GCS on Vertex AI). An in-place streaming write
there can be observed — or permanently left — truncated if the runtime
dies mid-flush; ``np.savez`` in particular rewrites the whole zip on
every call, so the file spends real time in a half-written state.
These helpers stage writes locally and publish with copy-to-temp +
``os.replace`` so the destination path always holds a complete file.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path


def _default_file_mode() -> int:
    """The mode a plain ``open(..., "w")`` creates a file with under the current umask.

    ``mkstemp`` creates its temporary 0600 regardless of the umask, so a file
    published from one with ``os.replace`` would otherwise land owner-only
    beside the 0644 files ``Path.write_text`` and ``result_bundle.hashing``
    write next to it -- a bundle whose ``summary.json`` a second account or
    a group share cannot read.  ``os.umask`` can only read the mask by
    setting it, hence the set-and-restore.
    """
    mask = os.umask(0)
    os.umask(mask)
    return 0o666 & ~mask


def atomic_copy(src: "str | Path", dst: "str | Path") -> None:
    """Copy *src* over *dst* without exposing a partially-written file.

    The bytes are first copied to a temp file in *dst*'s directory, then
    moved into place with ``os.replace`` so readers (and a runtime
    teardown) only ever see either the old complete file or the new one.
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{dst.name}.", suffix=".tmp", dir=str(dst.parent))
    os.close(fd)
    try:
        shutil.copyfile(str(src), tmp)
        os.chmod(tmp, _default_file_mode())
        os.replace(tmp, str(dst))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_text(path: "str | Path", text: str, *, encoding: str = "utf-8") -> None:
    """Write *text* to *path* without exposing a partially-written file.

    Staged to a temp file in *path*'s directory and published with
    ``os.replace``, so a runtime killed mid-write leaves the previous file
    (or no file) rather than a truncated one -- a half-written JSON record
    on a FUSE mount otherwise wedges every later reader with a decode error.
    """
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{dst.name}.", suffix=".tmp", dir=str(dst.parent))
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(text)
            os.fchmod(handle.fileno(), _default_file_mode())
        os.replace(tmp, str(dst))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_savez(path: "str | Path", **arrays) -> None:
    """``np.savez`` to *path* atomically.

    The archive is written to local scratch first (fast, reliable local
    disk), then published to *path* via :func:`atomic_copy`.
    """
    import numpy as np

    fd, tmp_local = tempfile.mkstemp(suffix=".npz")
    os.close(fd)
    try:
        np.savez(tmp_local, **arrays)
        atomic_copy(tmp_local, path)
    finally:
        try:
            os.unlink(tmp_local)
        except OSError:
            pass
