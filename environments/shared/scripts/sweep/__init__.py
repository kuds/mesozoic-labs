"""Hyperparameter sweep tool for Mesozoic Labs.

This package provides three modes:

  **launch**      — Submit a Vertex AI Hyperparameter Tuning job for one stage
                    and poll until completion (supports ``--resume``).
  **launch-all**  — Submit Stage 1, 2, and 3 HPT jobs sequentially.
  **trial**       — Entry point used by each Vertex AI HPT trial worker.

See ``__main__.py`` for CLI usage or run::

    python -m environments.shared.scripts.sweep --help
"""

from .constants import (
    NET_ARCH_PRESETS,
    SweepStageError,
    _SweepJobFailed,
)
from .orchestration import launch_all_stages, launch_sweep
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
from .submit import _is_retryable_gcp_error, _normalize_accelerator_type, _submit_stage_sweep
from .trial import _hpt_arg_to_override, run_trial

__all__ = [
    "NET_ARCH_PRESETS",
    "SweepStageError",
    "_SweepJobFailed",
    "_best_trial_model_path",
    "_best_trial_model_path_any",
    "_collect_trial_results",
    "_evaluate_curriculum_gate",
    "_extract_thresholds",
    "collect_results_from_disk",
    "_hpt_arg_to_override",
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
    "launch_all_stages",
    "launch_sweep",
    "plot_sweep_results",
    "run_trial",
    "write_results_csv",
]
