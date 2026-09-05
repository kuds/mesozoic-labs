"""Training result reporting utilities.

Provides functions to write human-readable stage and training summaries as
well as machine-readable JSON result files.  These were originally defined
inline in the Colab training notebook and are now shared so that both the
notebook and CLI training scripts can produce consistent output.

The implementation is split across submodules that layer bottom-up; import
from this package rather than from a submodule so callers stay insulated
from that layout:

* :mod:`~environments.shared.reporting.formatting` — value normalisation and
  duration formatting (no intra-package dependencies)
* :mod:`~environments.shared.reporting.gates` — curriculum-gate evaluation
  against recorded evaluation history
* :mod:`~environments.shared.reporting.csv_output` — the canonical
  ``collected_results.csv`` schema, the shared CSV writer, and per-episode
  evaluation evidence
* :mod:`~environments.shared.reporting.text_summaries` — human-readable
  ``stage_summary.txt`` / ``training_summary.txt``
* :mod:`~environments.shared.reporting.summaries` — schema-v3
  ``summary.json`` construction and its provenance block
* :mod:`~environments.shared.reporting.bundles` — idempotent, Drive-portable
  result bundle publication
* :mod:`~environments.shared.reporting.stage_layout` — where a stage's
  generated figures and replays live, and the local-staging publish
* :mod:`~environments.shared.reporting.stage_artifacts` — post-training
  artifact generation for the SB3 and JAX/MJX backends
* :mod:`~environments.shared.reporting.stance_report` — the
  ``stance_quality/v1`` checkpoint report and its probes, behind
  ``scripts/stance_gate_report.py``; deliberately not re-exported here
  (it imports the species registry at module level), so import it by path
"""

from __future__ import annotations

from . import stage_layout
from .bundles import save_result_bundle
from .csv_output import (
    CSV_METRIC_COLUMNS,
    build_results_csv_rows,
    save_evaluation_episodes,
    save_results_csv,
    write_results_csv,
)
from .formatting import format_duration, format_duration_hms, parse_optional_bool
from .gates import evaluate_recorded_gate, evaluate_stage_gate
from .stage_artifacts import (
    build_stage_results_from_eval_data,
    generate_stage_artifacts,
    save_jax_stage_artifacts,
)
from .summaries import build_result_summary, save_results_json
from .text_summaries import write_stage_summary, write_training_summary

__all__ = [
    "CSV_METRIC_COLUMNS",
    "build_result_summary",
    "build_results_csv_rows",
    "build_stage_results_from_eval_data",
    "evaluate_recorded_gate",
    "evaluate_stage_gate",
    "format_duration",
    "format_duration_hms",
    "generate_stage_artifacts",
    "parse_optional_bool",
    "save_evaluation_episodes",
    "save_jax_stage_artifacts",
    "save_result_bundle",
    "save_results_csv",
    "save_results_json",
    "stage_layout",
    "write_results_csv",
    "write_stage_summary",
    "write_training_summary",
]
