"""JAX/MJX checkpoint and training-state persistence.

Provides utilities for saving and loading JAX training checkpoints
(parameters, optimizer state, observation statistics, training history)
using pickle.  Includes rotating checkpoint management to limit disk usage.

Usage::

    from environments.shared.jax_checkpoint import save_checkpoint, load_checkpoint
    from environments.shared.plant_contract import current_plant_identity

    plant = current_plant_identity("trex")
    save_checkpoint(path, params, obs_rms, update=100, history=history, plant_identity=plant)
    ckpt = load_checkpoint(path, current_plant=plant)
    params = ckpt["params"]
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Mapping

from .plant_contract import (
    MODEL_IDENTITY_ATTRIBUTE,
    PlantCompatibilityError,
    PlantIdentity,
    validate_recorded_identity,
)

_logger = logging.getLogger(__name__)


def save_checkpoint(
    path: str | Path,
    params: Any,
    obs_rms: Any | None = None,
    opt_state: Any | None = None,
    update: int = 0,
    history: dict[str, list] | None = None,
    extra: dict[str, Any] | None = None,
    plant_identity: PlantIdentity | Mapping[str, Any] | None = None,
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
        plant_identity: Plant identity to persist with the parameters and
            normalization state.

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
        if MODEL_IDENTITY_ATTRIBUTE in extra:
            raise ValueError(f"extra may not override reserved key {MODEL_IDENTITY_ATTRIBUTE!r}")
        data.update(extra)
    if plant_identity is not None:
        data[MODEL_IDENTITY_ATTRIBUTE] = (
            plant_identity.to_dict() if isinstance(plant_identity, PlantIdentity) else dict(plant_identity)
        )

    with open(path, "wb") as f:
        pickle.dump(data, f)

    _logger.info("Checkpoint saved: %s (update %d)", path, update)
    return path


def load_checkpoint(
    path: str | Path,
    *,
    current_plant: PlantIdentity | None = None,
    allow_legacy_plant: bool = False,
    unsafe_skip_plant_validation: bool = False,
) -> dict[str, Any]:
    """Load a JAX training checkpoint.

    Args:
        path: Path to a ``.pkl`` checkpoint file.
        current_plant: Expected plant identity.  Required unless
            ``unsafe_skip_plant_validation`` is explicitly enabled.
        allow_legacy_plant: Explicitly allow an untagged pre-contract
            checkpoint.  Tagged incompatible checkpoints are always rejected.
        unsafe_skip_plant_validation: Deliberately load without checking which
            plant produced the checkpoint.  This unsafe escape hatch is for
            low-level inspection only and cannot be combined with
            ``current_plant`` or ``allow_legacy_plant``.

    Returns:
        Dictionary with keys ``"params"``, ``"obs_rms"``, ``"update"``,
        and optionally ``"opt_state"``, ``"history"``, etc.
    """
    path = Path(path)
    if unsafe_skip_plant_validation:
        if current_plant is not None or allow_legacy_plant:
            raise ValueError("unsafe_skip_plant_validation cannot be combined with current_plant or allow_legacy_plant")
        _logger.warning(
            "UNSAFE: loading %s without plant compatibility validation; "
            "do not use its parameters for training or evaluation until verified",
            path,
        )
    elif current_plant is None:
        raise PlantCompatibilityError(
            f"refusing to load {path} without current_plant; pass the current PlantIdentity "
            "for compatibility validation. For low-level inspection only, explicitly pass "
            "unsafe_skip_plant_validation=True."
        )

    with open(path, "rb") as f:
        data: dict[str, Any] = pickle.load(f)  # noqa: S301
    if not unsafe_skip_plant_validation:
        # ``current_plant`` is guaranteed non-None by the guard above.
        assert current_plant is not None
        raw_identity = data.get(MODEL_IDENTITY_ATTRIBUTE)
        if raw_identity is not None and not isinstance(raw_identity, Mapping):
            raise PlantCompatibilityError(
                f"{path} contains invalid plant identity metadata of type {type(raw_identity).__name__}"
            )
        validate_recorded_identity(
            raw_identity,
            current_plant,
            artifact=str(path),
            allow_legacy=allow_legacy_plant,
        )
    _logger.info("Checkpoint loaded: %s (update %d)", path, data.get("update", -1))
    return data


def restore_train_state(
    path: str | Path,
    optimizer: Any | None = None,
    *,
    current_plant: PlantIdentity | None = None,
    allow_legacy_plant: bool = False,
    unsafe_skip_plant_validation: bool = False,
) -> tuple[Any, Any, Any, int]:
    """Load a checkpoint and return ``(params, opt_state, obs_rms, update)``.

    When the checkpoint predates opt_state saving (or *optimizer* is given
    and the checkpoint has no opt_state), a fresh ``optimizer.init(params)``
    state is returned with a warning — resuming then restarts Adam moments,
    which is what the old checkpoints silently did.

    Args:
        path: Path to a ``.pkl`` checkpoint file.
        optimizer: Optional optax optimizer used to initialise a fresh
            opt_state when the checkpoint lacks one.
        current_plant: Expected plant identity.  Required unless
            ``unsafe_skip_plant_validation`` is explicitly enabled.
        allow_legacy_plant: Explicitly allow an untagged pre-contract
            checkpoint.
        unsafe_skip_plant_validation: Deliberately restore without plant
            compatibility validation.  This unsafe inspection-only escape
            hatch cannot be combined with ``current_plant`` or
            ``allow_legacy_plant``.

    Returns:
        ``(params, opt_state, obs_rms, update)`` — ``opt_state`` is ``None``
        if missing and no *optimizer* was provided.
    """
    data = load_checkpoint(
        path,
        current_plant=current_plant,
        allow_legacy_plant=allow_legacy_plant,
        unsafe_skip_plant_validation=unsafe_skip_plant_validation,
    )
    params = data["params"]
    opt_state = data.get("opt_state")
    if opt_state is None:
        if optimizer is not None:
            opt_state = optimizer.init(params)
        _logger.warning(
            "Checkpoint %s has no opt_state — optimizer moments restart from zero on resume.",
            path,
        )
    return params, opt_state, data.get("obs_rms"), int(data.get("update", 0))


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
        plant_identity: PlantIdentity | Mapping[str, Any] | None = None,
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.max_keep = max_keep
        self.plant_identity = plant_identity
        # Discover any pre-existing checkpoints (sorted oldest -> newest by
        # update number) so a resumed run continues to enforce ``max_keep``
        # instead of silently letting old files accumulate on disk.
        existing = list(self.directory.glob(f"{prefix}_*.pkl"))

        def _update_idx(p: Path) -> int:
            stem = p.stem  # e.g. "checkpoint_42"
            suffix = stem[len(prefix) + 1 :]
            try:
                return int(suffix)
            except ValueError:
                return -1

        self._recent: list[Path] = sorted(
            (p for p in existing if _update_idx(p) >= 0),
            key=_update_idx,
        )

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
        plant_identity = kwargs.pop("plant_identity", self.plant_identity)
        save_checkpoint(
            path,
            params,
            obs_rms=obs_rms,
            update=update,
            plant_identity=plant_identity,
            **kwargs,
        )

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
