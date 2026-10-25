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
* :mod:`~environments.shared.reporting.runs_summary` — the Drive summary
  notebook's cross-run table (``runs_summary.csv``), written atomically, and the
  repository's offline-judgement ledger (``results/offline_judgements.csv``) it
  takes rows from
* :mod:`~environments.shared.reporting.stage_layout` — where a stage's
  generated figures and replays live, and the local-staging publish
* :mod:`~environments.shared.reporting.stage_artifacts` — post-training
  stage artifact generation
* :mod:`~environments.shared.reporting.stance_report` — the
  ``stance_quality/v1`` checkpoint report and its probes, behind
  ``scripts/stance_gate_report.py``; deliberately not re-exported here
  (it imports the species registry at module level), so import it by path
* :mod:`~environments.shared.reporting.episode_trace` — the per-step,
  per-actuator trace of single episodes of a checkpoint, behind
  ``scripts/episode_trace_report.py``; report-only, and not re-exported
  here for the same reason
* :mod:`~environments.shared.reporting.training_history` — a stage's training
  history read back from its ledgers and drawn as pages of graphs, behind
  ``scripts/plot_training_history.py`` and drawn into each stage's
  ``figures/training_history/`` by ``generate_stage_artifacts``; report only,
  and not re-exported here either (it draws with matplotlib), so import it by
  path
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
from .runs_summary import (
    OFFLINE_JUDGEMENT_COLUMNS,
    OFFLINE_JUDGEMENTS_PATH,
    OfflineJudgementError,
    load_offline_judgements,
    merge_runs_summary,
    source_counts,
    write_runs_summary,
)
from .stage_artifacts import (
    build_stage_results_from_eval_data,
    evaluate_stage_checkpoints,
    generate_stage_artifacts,
)
from .summaries import build_result_summary, save_results_json
from .text_summaries import write_stage_summary, write_training_summary

__all__ = [
    "CSV_METRIC_COLUMNS",
    "OFFLINE_JUDGEMENTS_PATH",
    "OFFLINE_JUDGEMENT_COLUMNS",
    "OfflineJudgementError",
    "build_result_summary",
    "build_results_csv_rows",
    "build_stage_results_from_eval_data",
    "evaluate_recorded_gate",
    "evaluate_stage_checkpoints",
    "evaluate_stage_gate",
    "format_duration",
    "format_duration_hms",
    "generate_stage_artifacts",
    "load_offline_judgements",
    "merge_runs_summary",
    "parse_optional_bool",
    "save_evaluation_episodes",
    "save_result_bundle",
    "save_results_csv",
    "save_results_json",
    "source_counts",
    "stage_layout",
    "write_results_csv",
    "write_runs_summary",
    "write_stage_summary",
    "write_training_summary",
]
