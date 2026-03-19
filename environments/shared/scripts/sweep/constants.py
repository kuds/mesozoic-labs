"""Shared constants, exceptions, and default search spaces for the sweep tool."""

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


# ── Default search spaces ────────────────────────────────────────────────────
# Each entry: parameter_id -> {"type": ..., ...}
# parameter_id uses underscore notation to match Vertex AI HPT arg injection.
# The trial entry point converts them to dot notation for --override.

_DEFAULT_PPO_SEARCH_SPACE = {
    "ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 3e-4, "scale": "log"},
    "ppo_ent_coef": {"type": "double", "min": 1e-4, "max": 0.05, "scale": "log"},
    "ppo_batch_size": {"type": "discrete", "values": [64, 128, 256, 512]},
    "ppo_gamma": {"type": "double", "min": 0.97, "max": 0.999, "scale": "linear"},
    "ppo_n_steps": {"type": "discrete", "values": [1024, 2048, 4096]},
}

_DEFAULT_SAC_SEARCH_SPACE = {
    "sac_learning_rate": {"type": "double", "min": 1e-5, "max": 3e-4, "scale": "log"},
    "sac_batch_size": {"type": "discrete", "values": [128, 256, 512]},
    "sac_gamma": {"type": "double", "min": 0.97, "max": 0.995, "scale": "linear"},
    "sac_tau": {"type": "double", "min": 0.001, "max": 0.01, "scale": "log"},
    "sac_buffer_size": {"type": "discrete", "values": [100000, 300000, 1000000]},
}

_DEFAULT_SEARCH_SPACES = {
    "ppo": _DEFAULT_PPO_SEARCH_SPACE,
    "sac": _DEFAULT_SAC_SEARCH_SPACE,
}

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
