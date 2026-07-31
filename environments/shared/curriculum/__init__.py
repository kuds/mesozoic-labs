"""
Automated curriculum manager for multi-stage training.

Monitors evaluation metrics and automatically advances through curriculum
stages when performance thresholds are met. Supports both PPO and SAC
algorithms with per-stage hyperparameter loading from TOML configs.

Usage with CurriculumManager directly::

    from environments.shared.curriculum import CurriculumManager

    manager = CurriculumManager(species="velociraptor")

    if manager.should_advance(eval_rewards, eval_lengths):
        manager.advance()
        new_config = manager.current_config()

Usage with CurriculumCallback (SB3 integration)::

    from environments.shared.curriculum import CurriculumManager, CurriculumCallback

    manager = CurriculumManager(species="velociraptor")
    curriculum_cb = CurriculumCallback(manager, eval_env, eval_freq=50000)

    model.learn(total_timesteps=500_000, callback=curriculum_cb)

    if curriculum_cb.ready_to_advance:
        manager.advance()

The manager is separated from the SB3 callbacks so that a JAX or notebook
training loop can drive the curriculum without importing SB3 at all:

* :mod:`~environments.shared.curriculum.sb3_compat` — the optional-SB3 import
  shim every callback subclasses through, and the single patch point for
  ``_SB3_AVAILABLE``
* :mod:`~environments.shared.curriculum.manager` — ``StageThreshold`` and the
  backend-independent ``CurriculumManager``
* :mod:`~environments.shared.curriculum.schedules` — constant schedules and
  entropy-coefficient decay
* :mod:`~environments.shared.curriculum.advancement` — stage entry, reward
  ramping, and gate evaluation
* :mod:`~environments.shared.curriculum.gate_schema` — the versioned,
  fail-closed declaration every stage config's gate is validated against
* :mod:`~environments.shared.curriculum.early_stopping` — abort a stage whose
  evaluation return has collapsed
* :mod:`~environments.shared.curriculum.checkpoints` — checkpoint selection,
  publication, and VecNormalize statistics
"""

from __future__ import annotations

from . import sb3_compat
from .advancement import CurriculumCallback, RewardRampCallback, StageWarmupCallback
from .checkpoints import (
    PublishEvalArtifactsCallback,
    RobustBestModelCallback,
    SaveVecNormalizeCallback,
    load_vecnorm_stats,
)
from .early_stopping import (
    EvalCollapseEarlyStopCallback,
    build_eval_collapse_early_stop_callback,
)
from .gate_schema import (
    GATE_KINDS,
    GATE_SCHEMA_VERSION,
    GateSchemaError,
    validate_gate_config,
    validate_gate_configs,
)
from .manager import CurriculumManager, StageThreshold, thresholds_from_configs
from .schedules import EntCoefDecayCallback, _ConstantSchedule

__all__ = [
    "GATE_KINDS",
    "GATE_SCHEMA_VERSION",
    "CurriculumCallback",
    "CurriculumManager",
    "EntCoefDecayCallback",
    "EvalCollapseEarlyStopCallback",
    "GateSchemaError",
    "PublishEvalArtifactsCallback",
    "RewardRampCallback",
    "RobustBestModelCallback",
    "SaveVecNormalizeCallback",
    "StageThreshold",
    "StageWarmupCallback",
    "_ConstantSchedule",
    "build_eval_collapse_early_stop_callback",
    "load_vecnorm_stats",
    "sb3_compat",
    "thresholds_from_configs",
    "validate_gate_config",
    "validate_gate_configs",
]
