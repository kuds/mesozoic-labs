"""Shared utilities for Mesozoic Labs dinosaur environments."""

import logging as _logging

from .config import load_all_stages, load_stage_config
from .curriculum import CurriculumManager
from .metrics import LocomotionMetrics
from .reporting import (
    format_duration,
    format_duration_hms,
    save_results_json,
    write_stage_summary,
    write_training_summary,
)

_logger = _logging.getLogger(__name__)

try:
    from .base_env import BaseDinoEnv
except ImportError as _exc:
    BaseDinoEnv = None  # type: ignore[assignment,misc]
    _logger.debug(
        "BaseDinoEnv not available (gymnasium/mujoco may not be installed): %s",
        _exc,
    )

try:
    from .diagnostics import DiagnosticsCallback
    from .evaluation import eval_policy, record_stage_video
except ImportError as _exc:
    DiagnosticsCallback = None  # type: ignore[assignment,misc]
    eval_policy = None  # type: ignore[assignment]
    record_stage_video = None  # type: ignore[assignment]
    _logger.debug(
        "Training utilities not available (SB3 may not be installed): %s",
        _exc,
    )

__all__ = [
    "BaseDinoEnv",
    "CurriculumManager",
    "DiagnosticsCallback",
    "LocomotionMetrics",
    "eval_policy",
    "format_duration",
    "format_duration_hms",
    "load_all_stages",
    "load_stage_config",
    "record_stage_video",
    "save_results_json",
    "write_stage_summary",
    "write_training_summary",
]
