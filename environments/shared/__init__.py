"""Shared utilities for Mesozoic Labs dinosaur environments."""

import logging as _logging

from .config import load_all_stages, load_stage_config
from .curriculum import CurriculumManager
from .metrics import LocomotionMetrics

_logger = _logging.getLogger(__name__)

try:
    from .base_env import BaseDinoEnv
except ImportError as _exc:
    BaseDinoEnv = None  # type: ignore[assignment,misc]
    _logger.debug(
        "BaseDinoEnv not available (gymnasium/mujoco may not be installed): %s",
        _exc,
    )

__all__ = [
    "BaseDinoEnv",
    "CurriculumManager",
    "LocomotionMetrics",
    "load_all_stages",
    "load_stage_config",
]
