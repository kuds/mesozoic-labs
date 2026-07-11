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
