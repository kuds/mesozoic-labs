"""Tests for sweep orchestration.py — credential refresh, dedup, dry-run."""

from unittest.mock import MagicMock, patch

import pytest

from environments.shared.scripts.sweep import _eager_refresh
from environments.shared.scripts.sweep.orchestration import _dedup_trial_rows

# ── _eager_refresh ───────────────────────────────────────────────────────────


class TestEagerRefresh:
    """Credential refresh retries on transient metadata-server errors."""

    def _call(self, creds, **kwargs):
        """Call _eager_refresh with a dummy request to avoid google.auth import."""
        return _eager_refresh(creds, _request=MagicMock(), **kwargs)

    def test_success_on_first_attempt(self):
        creds = MagicMock()
        self._call(creds, max_retries=3)
        creds.refresh.assert_called_once()

    @patch("time.sleep")
    def test_retries_on_type_error(self, mock_sleep):
        creds = MagicMock()
        creds.refresh.side_effect = [TypeError("string indices"), None]
        self._call(creds, max_retries=3)
        assert creds.refresh.call_count == 2
        mock_sleep.assert_called_once_with(1)

    @patch("time.sleep")
    def test_raises_after_max_retries(self, mock_sleep):
        creds = MagicMock()
        creds.refresh.side_effect = TypeError("string indices")
        with pytest.raises(TypeError):
            self._call(creds, max_retries=3)
        assert creds.refresh.call_count == 3

    @patch("time.sleep")
    def test_exponential_backoff(self, mock_sleep):
        creds = MagicMock()
        creds.refresh.side_effect = [
            TypeError("string indices"),
            TypeError("string indices"),
            TypeError("string indices"),
            None,
        ]
        self._call(creds, max_retries=4)
        assert mock_sleep.call_args_list == [
            ((1,),),
            ((2,),),
            ((4,),),
        ]

    @patch("time.sleep")
    def test_retries_on_refresh_error(self, mock_sleep):
        pytest.importorskip("google.auth.exceptions")
        from google.auth.exceptions import RefreshError

        creds = MagicMock()
        creds.refresh.side_effect = [RefreshError("token expired"), None]
        self._call(creds, max_retries=3)
        assert creds.refresh.call_count == 2
        mock_sleep.assert_called_once_with(1)

    @patch("time.sleep")
    def test_retries_on_transport_error(self, mock_sleep):
        pytest.importorskip("google.auth.exceptions")
        from google.auth.exceptions import TransportError

        creds = MagicMock()
        creds.refresh.side_effect = [TransportError("connection reset"), None]
        self._call(creds, max_retries=3)
        assert creds.refresh.call_count == 2
        mock_sleep.assert_called_once_with(1)

    def test_non_type_error_propagates_immediately(self):
        creds = MagicMock()
        creds.refresh.side_effect = ValueError("unexpected")
        with pytest.raises(ValueError, match="unexpected"):
            self._call(creds, max_retries=3)
        creds.refresh.assert_called_once()


# ── _dedup_trial_rows ───────────────────────────────────────────────────


class TestDedupTrialRows:
    """Deduplication of trial result rows by trial_id."""

    def test_no_duplicates_unchanged(self):
        rows = [
            {"trial_id": "1", "best_mean_reward": 100.0},
            {"trial_id": "2", "best_mean_reward": 200.0},
        ]
        result = _dedup_trial_rows(rows)
        assert len(result) == 2

    def test_duplicate_keeps_last(self):
        rows = [
            {"trial_id": "1", "best_mean_reward": 100.0},
            {"trial_id": "1", "best_mean_reward": 150.0},
        ]
        result = _dedup_trial_rows(rows)
        assert len(result) == 1
        assert result[0]["best_mean_reward"] == 150.0

    def test_rows_without_trial_id_kept(self):
        rows = [
            {"best_mean_reward": 100.0},
            {"best_mean_reward": 200.0},
        ]
        result = _dedup_trial_rows(rows)
        assert len(result) == 2

    def test_mixed_with_and_without_ids(self):
        rows = [
            {"trial_id": "1", "best_mean_reward": 100.0},
            {"best_mean_reward": 50.0},  # no trial_id
            {"trial_id": "1", "best_mean_reward": 150.0},  # duplicate
        ]
        result = _dedup_trial_rows(rows)
        assert len(result) == 2

    def test_empty_list(self):
        assert _dedup_trial_rows([]) == []

    def test_multiple_duplicates(self):
        rows = [
            {"trial_id": "1", "best_mean_reward": 100.0},
            {"trial_id": "2", "best_mean_reward": 200.0},
            {"trial_id": "1", "best_mean_reward": 120.0},
            {"trial_id": "2", "best_mean_reward": 250.0},
            {"trial_id": "3", "best_mean_reward": 300.0},
        ]
        result = _dedup_trial_rows(rows)
        assert len(result) == 3
        by_id = {r["trial_id"]: r for r in result}
        assert by_id["1"]["best_mean_reward"] == 120.0
        assert by_id["2"]["best_mean_reward"] == 250.0
        assert by_id["3"]["best_mean_reward"] == 300.0
