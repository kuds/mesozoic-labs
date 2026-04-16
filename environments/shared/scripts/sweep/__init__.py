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

from .constants import NET_ARCH_PRESETS, SweepStageError
from .constants import _SweepJobFailed as _SweepJobFailed
from .orchestration import _eager_refresh as _eager_refresh
from .orchestration import launch_all_stages, launch_sweep
from .ray_orchestration import (
    create_ray_tuner,
    discover_and_rank_trials,
    evaluate_trials_parallel,
    export_best_trial,
    run_ray_sweep,
)
from .ray_search_space import (
    build_search_space,
    detect_gpu_info,
    detect_gpu_model,
    load_resume_settings,
    resolve_config_path,
    save_search_space,
    to_ray_tune,
)
from .ray_tune import (
    DriveProgressLogCallback,
    ExperimentStateSyncCallback,
    RayTuneReportCallback,
    TrialTerminationCallback,
    apply_sampled_config,
    collect_ray_results,
    train_trial,
)
from .results import _best_trial_model_path as _best_trial_model_path
from .results import _best_trial_model_path_any as _best_trial_model_path_any
from .results import _collect_trial_results as _collect_trial_results
from .results import _evaluate_curriculum_gate as _evaluate_curriculum_gate
from .results import _extract_thresholds as _extract_thresholds
from .results import collect_results_from_disk, plot_sweep_results, write_results_csv
from .scoring import compute_quality_scores, load_scoring_config
from .search_space import _is_per_stage as _is_per_stage
from .search_space import _resolve_search_space as _resolve_search_space
from .search_space import _search_space_for_stage as _search_space_for_stage
from .search_space import _settings_for_stage as _settings_for_stage
from .search_space import _split_stage_block as _split_stage_block
from .state import _load_sweep_state as _load_sweep_state
from .state import _save_sweep_state as _save_sweep_state
from .state import _sweep_state_local_path as _sweep_state_local_path
from .submit import _is_retryable_gcp_error as _is_retryable_gcp_error
from .submit import _normalize_accelerator_type as _normalize_accelerator_type
from .submit import _submit_stage_sweep as _submit_stage_sweep
from .submit import _validate_machine_type as _validate_machine_type
from .trial import _hpt_arg_to_override as _hpt_arg_to_override
from .trial import _parse_hpt_extra_args as _parse_hpt_extra_args
from .trial import run_trial

# Public API. Underscore-prefixed helpers (``_load_sweep_state``,
# ``_submit_stage_sweep``, etc.) are intentionally imported above so existing
# test modules keep working but are NOT re-exported here — consumers should
# import them from their defining submodule if genuinely needed.
__all__ = [
    "DriveProgressLogCallback",
    "ExperimentStateSyncCallback",
    "NET_ARCH_PRESETS",
    "RayTuneReportCallback",
    "SweepStageError",
    "TrialTerminationCallback",
    "apply_sampled_config",
    "build_search_space",
    "collect_ray_results",
    "collect_results_from_disk",
    "compute_quality_scores",
    "create_ray_tuner",
    "detect_gpu_info",
    "detect_gpu_model",
    "discover_and_rank_trials",
    "evaluate_trials_parallel",
    "export_best_trial",
    "launch_all_stages",
    "launch_sweep",
    "load_resume_settings",
    "load_scoring_config",
    "plot_sweep_results",
    "resolve_config_path",
    "run_ray_sweep",
    "run_trial",
    "save_search_space",
    "to_ray_tune",
    "train_trial",
    "write_results_csv",
]
