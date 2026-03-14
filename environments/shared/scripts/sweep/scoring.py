"""Post-analysis quality scoring for sweep trial ranking.

Computes a weighted composite score from multiple locomotion and performance
metrics, enabling trial comparison beyond raw reward.  The scoring is applied
**after** training completes — it does not affect the training objective
(which remains ``best_mean_reward`` for ASHA / Vertex AI HPT).

Scoring configuration is loaded from ``configs/quality_scoring.toml``.
Each stage has its own set of weighted metrics with direction indicators
(maximize, minimize, minimize_abs).

Normalization is relative (min-max within the current trial set) so no
manual reference values are needed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default config path relative to repo root.
_DEFAULT_CONFIG_NAME = "configs/quality_scoring.toml"


def _find_config_path() -> Path | None:
    """Locate the quality scoring TOML config by walking up from this file."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / _DEFAULT_CONFIG_NAME
        if candidate.exists():
            return candidate
    return None


def load_scoring_config(
    stage: int,
    config_path: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Load the quality scoring config for a given stage.

    Args:
        stage: Curriculum stage number (1, 2, or 3).
        config_path: Optional override path to the TOML file.

    Returns:
        Dict mapping metric names to ``{"weight": float, "direction": str}``.
        Returns an empty dict if no config is found for the stage.
    """
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            logger.warning("tomllib/tomli not available — quality scoring disabled")
            return {}

    if config_path is None:
        config_path = _find_config_path()
    if config_path is None:
        logger.warning("quality_scoring.toml not found — quality scoring disabled")
        return {}

    config_path = Path(config_path)
    if not config_path.exists():
        logger.warning("quality_scoring.toml not found at %s", config_path)
        return {}

    with open(config_path, "rb") as f:
        raw = tomllib.load(f)

    stage_key = f"stage_{stage}"
    stage_section = raw.get(stage_key, {})
    if not stage_section:
        logger.info("No scoring config for stage %d in %s", stage, config_path)
        return {}

    # Each sub-key is a metric name with weight + direction.
    config: dict[str, dict[str, Any]] = {}
    for metric_name, spec in stage_section.items():
        if isinstance(spec, dict) and "weight" in spec and "direction" in spec:
            config[metric_name] = {
                "weight": float(spec["weight"]),
                "direction": str(spec["direction"]),
            }

    return config


def _normalize_metric(
    values: list[float],
    direction: str,
) -> list[float]:
    """Min-max normalize a list of metric values according to direction.

    Args:
        values: Raw metric values (one per trial).
        direction: One of ``"maximize"``, ``"minimize"``, or ``"minimize_abs"``.

    Returns:
        Normalized scores in [0, 1] where 1.0 is best.
    """
    if direction == "minimize_abs":
        values = [abs(v) for v in values]
        direction = "minimize"

    min_val = min(values)
    max_val = max(values)
    spread = max_val - min_val

    if spread < 1e-12:
        # All values are the same — everyone gets full score.
        return [1.0] * len(values)

    if direction == "maximize":
        return [(v - min_val) / spread for v in values]
    else:  # minimize
        return [(max_val - v) / spread for v in values]


def compute_quality_scores(
    rows: list[dict[str, Any]],
    stage: int,
    config: dict[str, dict[str, Any]] | None = None,
    config_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Compute quality scores and ranks for a list of trial result rows.

    Adds ``quality_score`` (float in [0, 1]) and ``quality_rank`` (int,
    1 = best) to each row dict **in place** and returns the rows sorted
    by quality_score descending.

    Args:
        rows: List of trial result dicts.  Each dict should contain metric
            keys matching the scoring config (e.g., ``"fwd_vel_m/s"``,
            ``"ep_length"``).  Missing metrics are skipped (their weight
            is redistributed).
        stage: Curriculum stage number.
        config: Pre-loaded scoring config.  If ``None``, loaded from TOML.
        config_path: Override path to the TOML config file.

    Returns:
        The input rows sorted by quality_score descending, with
        ``quality_score`` and ``quality_rank`` added to each row.
    """
    if not rows:
        return rows

    if config is None:
        config = load_scoring_config(stage, config_path=config_path)

    if not config:
        logger.info("No scoring config for stage %d — skipping quality scoring", stage)
        for row in rows:
            row["quality_score"] = ""
            row["quality_rank"] = ""
        return rows

    # Collect values for each configured metric, tracking which metrics
    # are actually present in the data.
    metric_values: dict[str, list[float]] = {}
    available_metrics: list[str] = []

    for metric_name in config:
        vals: list[float] = []
        all_present = True
        for row in rows:
            raw = row.get(metric_name)
            if raw is None or raw == "" or raw == "N/A":
                all_present = False
                break
            try:
                vals.append(float(raw))
            except (ValueError, TypeError):
                all_present = False
                break

        if all_present and vals:
            metric_values[metric_name] = vals
            available_metrics.append(metric_name)
        else:
            logger.debug(
                "Metric '%s' missing from some trials — excluded from scoring",
                metric_name,
            )

    if not available_metrics:
        logger.warning("No scoreable metrics found for stage %d — skipping", stage)
        for row in rows:
            row["quality_score"] = ""
            row["quality_rank"] = ""
        return rows

    # Normalize weights to sum to 1.0 across available metrics only.
    total_weight = sum(config[m]["weight"] for m in available_metrics)
    if total_weight < 1e-12:
        for row in rows:
            row["quality_score"] = ""
            row["quality_rank"] = ""
        return rows

    normalized_weights = {m: config[m]["weight"] / total_weight for m in available_metrics}

    # Normalize each metric and compute weighted score.
    normalized_scores: dict[str, list[float]] = {}
    for metric_name in available_metrics:
        direction = config[metric_name]["direction"]
        normalized_scores[metric_name] = _normalize_metric(
            metric_values[metric_name],
            direction,
        )

    # Compute composite score per trial.
    for i, row in enumerate(rows):
        score = 0.0
        for metric_name in available_metrics:
            score += normalized_weights[metric_name] * normalized_scores[metric_name][i]
        row["quality_score"] = round(score, 4)

    # Sort by quality_score descending and assign ranks.
    rows.sort(key=lambda r: r.get("quality_score", 0.0), reverse=True)
    for rank, row in enumerate(rows, 1):
        row["quality_rank"] = rank

    logger.info(
        "Quality scoring: %d trials scored on %d/%d metrics (stage %d)",
        len(rows),
        len(available_metrics),
        len(config),
        stage,
    )

    return rows
