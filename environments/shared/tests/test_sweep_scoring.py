"""Tests for sweep quality scoring (scoring.py)."""

from __future__ import annotations

import pytest

from environments.shared.scripts.sweep.scoring import (
    _normalize_metric,
    compute_quality_scores,
)


def _config(**metrics):
    """Build a scoring config dict: name -> (weight, direction)."""
    return {name: {"weight": w, "direction": d} for name, (w, d) in metrics.items()}


class TestNormalizeMetric:
    def test_maximize(self):
        assert _normalize_metric([0.0, 5.0, 10.0], "maximize") == [0.0, 0.5, 1.0]

    def test_minimize(self):
        assert _normalize_metric([0.0, 5.0, 10.0], "minimize") == [1.0, 0.5, 0.0]

    def test_minimize_abs(self):
        scores = _normalize_metric([-10.0, 0.0, 10.0], "minimize_abs")
        assert scores == [0.0, 1.0, 0.0]

    def test_constant_values_all_score_one(self):
        assert _normalize_metric([3.0, 3.0, 3.0], "maximize") == [1.0, 1.0, 1.0]


class TestComputeQualityScores:
    def test_scores_with_toml_native_keys(self):
        rows = [
            {"best_mean_reward": 100.0, "fwd_vel_m/s": 2.0},
            {"best_mean_reward": 200.0, "fwd_vel_m/s": 3.0},
        ]
        config = _config(
            best_mean_reward=(0.5, "maximize"),
            **{"fwd_vel_m/s": (0.5, "maximize")},
        )
        result = compute_quality_scores(rows, stage=2, config=config)
        assert result[0]["quality_score"] == 1.0
        assert result[0]["quality_rank"] == 1
        assert result[1]["quality_score"] == 0.0
        assert result[1]["quality_rank"] == 2

    def test_scores_with_train_base_row_keys(self):
        """Regression: rows from metrics.json / collected_results.csv use
        ``mean_forward_vel`` / ``best_mean_episode_length`` /
        ``mean_distance_traveled`` instead of the TOML's short names.
        Aliases must resolve them — previously every quality_score was ''.
        """
        rows = [
            {
                "best_mean_reward": 1960.0,
                "best_mean_episode_length": 969.0,
                "mean_forward_vel": 0.11,
                "mean_distance_traveled": 1.3,
            },
            {
                "best_mean_reward": 2600.0,
                "best_mean_episode_length": 1000.0,
                "mean_forward_vel": 3.47,
                "mean_distance_traveled": 34.0,
            },
        ]
        config = _config(
            best_mean_reward=(0.25, "maximize"),
            ep_length=(0.25, "maximize"),
            **{
                "fwd_vel_m/s": (0.25, "maximize"),
                "distance_m": (0.25, "maximize"),
            },
        )
        result = compute_quality_scores(rows, stage=2, config=config)
        scores = [r["quality_score"] for r in result]
        assert all(isinstance(s, float) for s in scores), scores
        assert result[0]["best_mean_reward"] == 2600.0  # better trial ranks first
        assert result[0]["quality_rank"] == 1

    def test_success_rate_alias(self):
        rows = [
            {"best_mean_reward": 1.0, "best_mean_success_rate": 0.9},
            {"best_mean_reward": 1.0, "best_mean_success_rate": 0.3},
        ]
        config = _config(
            mean_success_rate=(1.0, "maximize"),
            best_mean_reward=(0.0, "maximize"),
        )
        result = compute_quality_scores(rows, stage=3, config=config)
        assert result[0]["best_mean_success_rate"] == 0.9
        assert result[0]["quality_score"] == pytest.approx(1.0)

    def test_missing_metric_excluded_and_weight_redistributed(self):
        rows = [
            {"best_mean_reward": 10.0},
            {"best_mean_reward": 20.0},
        ]
        config = _config(
            best_mean_reward=(0.5, "maximize"),
            **{"fwd_vel_m/s": (0.5, "maximize")},  # absent from rows
        )
        result = compute_quality_scores(rows, stage=1, config=config)
        assert result[0]["quality_score"] == 1.0  # reward fully determines score

    def test_no_scoreable_metrics_leaves_empty_scores(self):
        rows = [{"foo": 1.0}, {"foo": 2.0}]
        config = _config(**{"fwd_vel_m/s": (1.0, "maximize")})
        result = compute_quality_scores(rows, stage=1, config=config)
        assert all(r["quality_score"] == "" for r in result)

    def test_loads_real_toml_config_and_scores_shipped_style_rows(self):
        """End-to-end with the repo's actual quality_scoring.toml."""
        rows = [
            {
                "best_mean_reward": 1964.43,
                "best_mean_episode_length": 969.5,
                "mean_forward_vel": 0.11,
                "mean_distance_traveled": 1.34,
            },
            {
                "best_mean_reward": 1500.0,
                "best_mean_episode_length": 800.0,
                "mean_forward_vel": 0.5,
                "mean_distance_traveled": 5.0,
            },
        ]
        result = compute_quality_scores(rows, stage=1)
        scores = [r["quality_score"] for r in result]
        assert all(isinstance(s, float) for s in scores), (
            "quality_score empty — alias resolution against quality_scoring.toml failed"
        )
