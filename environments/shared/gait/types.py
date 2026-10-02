"""Versioned measurement options for the reporting-only gait audit.

These defaults are provisional detector settings, not biological acceptance
limits. Certification thresholds belong to the separately calibrated gate.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
MEASUREMENT_SCHEMA = "mesozoic.gait-measurement/v1"


@dataclass(frozen=True)
class GaitProtocol:
    """Time- and morphology-scaled event detector; defaults are provisional."""

    contact_on_bw: float = 0.02
    contact_off_bw: float = 0.01
    min_stance_s: float = 0.02
    min_swing_s: float = 0.02
    phase_tolerance: float = 0.15
    simultaneous_s: float = 0.02
    min_clearance_over_leg: float = 0.01
    min_reposition_over_leg: float = 0.02

    def __post_init__(self) -> None:
        values = asdict(self)
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError("gait protocol options must be finite")
        if not 0.0 <= self.contact_off_bw < self.contact_on_bw:
            raise ValueError("contact thresholds require 0 <= off < on")
        if not 0.0 < self.phase_tolerance < 0.5:
            raise ValueError("phase tolerance must lie strictly between 0 and 0.5")
        for name in (
            "min_stance_s",
            "min_swing_s",
            "simultaneous_s",
            "min_clearance_over_leg",
            "min_reposition_over_leg",
        ):
            if values[name] < 0.0:
                raise ValueError(f"{name} must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": MEASUREMENT_SCHEMA,
            "default_status": "provisional; requires calibration",
            **asdict(self),
        }

    @property
    def sha256(self) -> str:
        return measurement_protocol_sha256(self)


def measurement_protocol_sha256(protocol: GaitProtocol) -> str:
    """Hash the versioned options; caller also pins implementation/morphology."""
    payload = json.dumps(protocol.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ContactEvents:
    """Debounced load state and observed event indices for one foot.

    Initial load is a boundary condition, never a synthetic touchdown. A change
    at the final boundary is emitted only if its minimum dwell was observed.
    """

    loaded: NDArray[np.bool_]
    touchdowns: tuple[int, ...]
    liftoffs: tuple[int, ...]
    complete_cycles: tuple[tuple[int, int, int], ...]
