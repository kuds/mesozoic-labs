"""Tests for sweep orchestration.py — credential refresh, dedup, stage chaining."""

from unittest.mock import MagicMock, patch

import pytest

from environments.shared.scripts.sweep import SweepStageError, _eager_refresh
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


# ── launch-all stage chaining ───────────────────────────────────────────────


def _model_path(stage):
    return f"/gcs/b/sweeps/trex/stage{stage}/t{stage}/models/best_model.zip"


def _run_launch_all(rows_for_stage, extra_cli=()):
    """Drive launch_all_stages with Vertex/GCS mocked; return the _submit_stage_sweep kwargs per stage."""
    from environments.shared.scripts.sweep import orchestration as orch
    from environments.shared.scripts.sweep.__main__ import _build_parser

    args = _build_parser().parse_args(
        [
            "launch-all",
            "--species",
            "trex",
            "--project",
            "p",
            "--bucket",
            "b",
            "--image",
            "img",
            "--no-resume",
            "--search-space",
            '{"ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 3e-4}}',
            *extra_cli,
        ]
    )
    submitted = []

    def fake_submit(**kwargs):
        submitted.append(kwargs)
        job = MagicMock()
        job.resource_name = f"jobs/stage{kwargs['stage']}"
        return job

    def reraise(exc, *_args, **_kwargs):
        # The real handler saves state and os._exit()s, which would kill pytest.
        raise exc

    gcloud = MagicMock()
    modules = {
        "google": MagicMock(),
        "google.cloud": gcloud,
        "google.cloud.aiplatform": gcloud.aiplatform,
        "google.cloud.aiplatform.hyperparameter_tuning": gcloud.aiplatform.hyperparameter_tuning,
    }
    with (
        patch.dict("sys.modules", modules),
        patch.object(orch, "_resolve_credentials", return_value=(MagicMock(), "p")),
        patch("environments.shared.config._detect_gpu_info", return_value={}),
        patch.object(orch, "_submit_stage_sweep", side_effect=fake_submit),
        patch.object(orch, "_wait_for_job", side_effect=lambda job, *a, **k: job),
        patch.object(orch, "_collect_and_tag_rows", side_effect=lambda job, stage, *a, **k: rows_for_stage(stage)),
        patch.object(orch, "_save_sweep_state"),
        patch.object(orch, "_handle_stage_failure", side_effect=reraise),
        patch.object(orch, "write_results_csv"),
        patch.object(orch, "plot_sweep_results"),
        patch.object(orch, "_upload_results_to_gcs"),
    ):
        orch.launch_all_stages(args)
    return submitted


def _unevaluable_row(trial_id, stage, reward):
    return {
        "trial_id": trial_id,
        "stage": stage,
        "best_mean_reward": reward,
        "stage_passed": None,
        "gate_evaluable": False,
        "gate_kind": "stance_quality/v1",
        "model_path": f"/gcs/b/sweeps/trex/stage{stage}/{trial_id}/models/best_model.zip",
    }


class TestLaunchAllChaining:
    """Stage N's winner is warm-started into stage N+1 as a task-boundary crossing."""

    def test_chained_load_is_initialize_next_stage(self):
        def rows(stage):
            return [
                {
                    "trial_id": f"t{stage}",
                    "stage": stage,
                    "best_mean_reward": 100.0 * stage,
                    "stage_passed": True,
                    "gate_evaluable": True,
                    "model_path": _model_path(stage),
                }
            ]

        submitted = _run_launch_all(rows)
        assert [s["stage"] for s in submitted] == [1, 2, 3]
        assert submitted[0]["load_path"] is None
        assert submitted[1]["load_path"] == _model_path(1)
        assert submitted[2]["load_path"] == _model_path(2)
        assert [s["load_mode"] for s in submitted[1:]] == ["initialize_next_stage", "initialize_next_stage"]

    def test_not_evaluable_gate_blocks_chaining_without_force_continue(self):
        """A stance-gated stage has no gate-passed trial; the statue's reward must not chain it."""
        with pytest.raises(SweepStageError, match="cannot evaluate offline"):
            _run_launch_all(lambda stage: [_unevaluable_row("statue", stage, 3271.8)])

    def test_force_continue_chains_reward_ranked_trial(self):
        def rows(stage):
            return [_unevaluable_row("chatterer", stage, 2133.4), _unevaluable_row("statue", stage, 3271.8)]

        submitted = _run_launch_all(rows, extra_cli=("--force-continue",))
        assert submitted[1]["load_path"] == _unevaluable_row("statue", 1, 3271.8)["model_path"]
        assert submitted[1]["load_mode"] == "initialize_next_stage"
