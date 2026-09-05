"""Shared constants and exceptions for the sweep tool."""

from __future__ import annotations

from typing import Any


class SweepStageError(Exception):
    """Raised when a sweep stage fails and cannot proceed."""


class _SweepJobFailed(SweepStageError):
    """A submitted HPT job failed but may contain partial trial results.

    Attributes:
        hpt_job: The Vertex AI ``HyperparameterTuningJob`` object, which may
            still expose ``.trials`` for completed trial data even when the
            overall job failed.
    """

    def __init__(self, message: str, hpt_job: Any = None):
        super().__init__(message)
        self.hpt_job = hpt_job


# ── Net-arch presets ─────────────────────────────────────────────────────────
# Categorical values for the ``ppo_net_arch`` / ``sac_net_arch`` sweep param.
# Each preset maps to a ``policy_kwargs.net_arch`` list for SB3.
NET_ARCH_PRESETS: dict[str, list[int]] = {
    "small": [64, 64],
    "medium": [256, 256],
    "large": [512, 512],
    "deep": [256, 256, 256],
    "tapered": [512, 256],
    "deep_tapered": [512, 512, 256],
}
