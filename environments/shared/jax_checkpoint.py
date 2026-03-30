"""JAX/MJX checkpoint and training-state persistence.

Provides utilities for saving and loading JAX training checkpoints
(parameters, optimizer state, observation statistics, training history)
using pickle.  Includes rotating checkpoint management to limit disk usage.

Usage::

    from environments.shared.jax_checkpoint import save_checkpoint, load_checkpoint

    save_checkpoint(path, params, obs_rms, update=100, history=history)
    ckpt = load_checkpoint(path)
    params = ckpt["params"]
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)


def save_checkpoint(
    path: str | Path,
    params: Any,
    obs_rms: Any | None = None,
    opt_state: Any | None = None,
    update: int = 0,
    history: dict[str, list] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save a JAX training checkpoint.

    Args:
        path: File path for the checkpoint (``.pkl``).
        params: Network parameters (will be transferred to host via
            ``jax.device_get`` if needed).
        obs_rms: Optional observation normalisation statistics.
        opt_state: Optional optimizer state.
        update: Current update number.
        history: Optional dict of training history lists (e.g.
            ``{"reward": [...], "loss": [...]}``.
        extra: Optional dict of additional metadata.

    Returns:
        Path to the saved checkpoint file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import jax

        params = jax.device_get(params)
    except ImportError:
        pass

    data: dict[str, Any] = {
        "params": params,
        "obs_rms": obs_rms,
        "update": update,
    }
    if opt_state is not None:
        data["opt_state"] = opt_state
    if history is not None:
        data["history"] = history
    if extra is not None:
        data.update(extra)

    with open(path, "wb") as f:
        pickle.dump(data, f)

    _logger.info("Checkpoint saved: %s (update %d)", path, update)
    return path


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a JAX training checkpoint.

    Args:
        path: Path to a ``.pkl`` checkpoint file.

    Returns:
        Dictionary with keys ``"params"``, ``"obs_rms"``, ``"update"``,
        and optionally ``"opt_state"``, ``"history"``, etc.
    """
    path = Path(path)
    with open(path, "rb") as f:
        data: dict[str, Any] = pickle.load(f)  # noqa: S301
    _logger.info("Checkpoint loaded: %s (update %d)", path, data.get("update", -1))
    return data


class CheckpointManager:
    """Rotating checkpoint manager that keeps only the N most recent saves.

    Args:
        directory: Directory to store checkpoints in.
        prefix: Filename prefix (e.g. ``"trex_jax"``).
        max_keep: Maximum number of checkpoints to retain.
    """

    def __init__(
        self,
        directory: str | Path,
        prefix: str = "checkpoint",
        max_keep: int = 5,
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.max_keep = max_keep
        self._recent: list[Path] = []

    def save(
        self,
        params: Any,
        update: int,
        obs_rms: Any | None = None,
        **kwargs: Any,
    ) -> Path:
        """Save a checkpoint and rotate old ones.

        Returns:
            Path to the saved checkpoint.
        """
        path = self.directory / f"{self.prefix}_{update}.pkl"
        save_checkpoint(path, params, obs_rms=obs_rms, update=update, **kwargs)

        self._recent.append(path)
        while len(self._recent) > self.max_keep:
            old = self._recent.pop(0)
            old.unlink(missing_ok=True)
            _logger.debug("Removed old checkpoint: %s", old)

        return path

    @property
    def latest(self) -> Path | None:
        """Return the most recently saved checkpoint path, or None."""
        return self._recent[-1] if self._recent else None
