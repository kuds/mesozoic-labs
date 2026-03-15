"""Hyperparameter sweep tool for Mesozoic Labs.

This package provides:

  **Vertex AI** (CLI):
    ``launch``      — Submit a Vertex AI HPT job for one stage.
    ``launch-all``  — Submit Stage 1, 2, and 3 HPT jobs sequentially.
    ``trial``       — Entry point used by each Vertex AI HPT trial worker.

  **Ray Tune** (notebook / programmatic):
    ``ray_search_space`` — Per-species, per-stage search spaces with
                           a ``to_ray_tune()`` converter.
    ``ray_tune``         — Callbacks, trainable, and result helpers.

See ``__main__.py`` for CLI usage or run::

    python -m environments.shared.scripts.sweep --help
"""

from .constants import (
    NET_ARCH_PRESETS,
    SweepStageError,
    _SweepJobFailed,
)
from .orchestration import _eager_refresh, launch_all_stages, launch_sweep
from .ray_search_space import build_search_space, detect_gpu_model, save_search_space, to_ray_tune
from .ray_tune import (
    DriveProgressLogCallback,
    ExperimentStateSyncCallback,
    RayTuneReportCallback,
    TrialTerminationCallback,
    apply_sampled_config,
    collect_ray_results,
    train_trial,
)
from .results import (
    _best_trial_model_path,
    _best_trial_model_path_any,
    _collect_trial_results,
    _evaluate_curriculum_gate,
    _extract_thresholds,
    collect_results_from_disk,
    plot_sweep_results,
    write_results_csv,
)
from .scoring import compute_quality_scores, load_scoring_config
from .search_space import (
    _is_per_stage,
    _resolve_search_space,
    _search_space_for_stage,
    _settings_for_stage,
    _split_stage_block,
)
from .state import (
    _load_sweep_state,
    _save_sweep_state,
    _sweep_state_local_path,
)
from .submit import _is_retryable_gcp_error, _normalize_accelerator_type, _submit_stage_sweep, _validate_machine_type
from .trial import _hpt_arg_to_override, _parse_hpt_extra_args, run_trial

__all__ = [
    "DriveProgressLogCallback",
    "ExperimentStateSyncCallback",
    "NET_ARCH_PRESETS",
    "RayTuneReportCallback",
    "SweepStageError",
    "TrialTerminationCallback",
    "_SweepJobFailed",
    "_best_trial_model_path",
    "_best_trial_model_path_any",
    "_collect_trial_results",
    "_eager_refresh",
    "_evaluate_curriculum_gate",
    "_extract_thresholds",
    "apply_sampled_config",
    "build_search_space",
    "collect_ray_results",
    "collect_results_from_disk",
    "_hpt_arg_to_override",
    "_parse_hpt_extra_args",
    "_is_per_stage",
    "_is_retryable_gcp_error",
    "_load_sweep_state",
    "_normalize_accelerator_type",
    "_resolve_search_space",
    "_save_sweep_state",
    "_search_space_for_stage",
    "_settings_for_stage",
    "_split_stage_block",
    "_submit_stage_sweep",
    "_sweep_state_local_path",
    "_validate_machine_type",
    "launch_all_stages",
    "launch_sweep",
    "plot_sweep_results",
    "run_trial",
    "save_search_space",
    "to_ray_tune",
    "train_trial",
    "write_results_csv",
    "compute_quality_scores",
    "detect_gpu_model",
    "load_scoring_config",
]
