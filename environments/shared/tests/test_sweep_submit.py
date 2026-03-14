"""Tests for sweep submit.py — job submission, machine validation, accelerator normalization."""

from unittest.mock import MagicMock, patch

import pytest

from environments.shared.scripts.sweep import (
    _is_retryable_gcp_error,
    _normalize_accelerator_type,
    _submit_stage_sweep,
    _SweepJobFailed,
    _validate_machine_type,
)

# ── _normalize_accelerator_type ──────────────────────────────────────────────


class TestNormalizeAcceleratorType:
    """Short aliases like 'T4' are expanded to full Vertex AI enum labels."""

    def test_short_alias_t4(self):
        assert _normalize_accelerator_type("T4") == "NVIDIA_TESLA_T4"

    def test_short_alias_case_insensitive(self):
        assert _normalize_accelerator_type("t4") == "NVIDIA_TESLA_T4"
        assert _normalize_accelerator_type("v100") == "NVIDIA_TESLA_V100"

    def test_full_name_passthrough(self):
        assert _normalize_accelerator_type("NVIDIA_TESLA_T4") == "NVIDIA_TESLA_T4"

    def test_none_string_returns_none(self):
        assert _normalize_accelerator_type("None") is None
        assert _normalize_accelerator_type("none") is None
        assert _normalize_accelerator_type("NONE") is None

    def test_unknown_type_passthrough(self):
        assert _normalize_accelerator_type("TPU_V3") == "TPU_V3"

    def test_l4_alias(self):
        assert _normalize_accelerator_type("L4") == "NVIDIA_L4"

    def test_a100_alias(self):
        assert _normalize_accelerator_type("A100") == "NVIDIA_TESLA_A100"


# ── _validate_machine_type ───────────────────────────────────────────────────


class TestValidateMachineType:
    """Unsupported machine families are rejected before job submission."""

    @pytest.mark.parametrize(
        "mt",
        [
            "n1-standard-8",
            "n2-standard-4",
            "e2-standard-16",
            "c2-standard-8",
            "a2-highgpu-1g",
            "g2-standard-4",
        ],
    )
    def test_supported_types_pass(self, mt):
        _validate_machine_type(mt)  # should not raise

    @pytest.mark.parametrize("mt", ["c3-standard-4", "c4-standard-8", "c4-highcpu-16", "z1-standard-4"])
    def test_unsupported_types_raise(self, mt):
        with pytest.raises(ValueError, match="not supported by Vertex AI"):
            _validate_machine_type(mt)

    @pytest.mark.parametrize(
        "mt",
        ["n1-standard-8", "a2-highgpu-1g", "g2-standard-4"],
    )
    def test_gpu_compatible_types_with_accelerator(self, mt):
        _validate_machine_type(mt, "NVIDIA_TESLA_T4")  # should not raise

    @pytest.mark.parametrize(
        "mt",
        ["c2-standard-8", "e2-standard-16", "n2-standard-4", "m1-ultramem-40"],
    )
    def test_gpu_incompatible_types_with_accelerator_raise(self, mt):
        with pytest.raises(ValueError, match="not supported for machine type"):
            _validate_machine_type(mt, "NVIDIA_TESLA_T4")

    def test_cpu_only_allows_any_supported_machine(self):
        # accelerator_type=None means CPU-only — no GPU check needed
        _validate_machine_type("c2-standard-8", None)
        _validate_machine_type("e2-standard-16", None)


# ── _is_retryable_gcp_error ────────────────────────────────────────────


class TestIsRetryableGcpError:
    def test_retryable_by_name(self):
        for name in ("ResourceExhausted", "ServiceUnavailable", "GoogleAPICallError", "TooManyRequests"):
            # Create a dynamically-named exception class
            exc_cls = type(name, (Exception,), {})
            assert _is_retryable_gcp_error(exc_cls("quota exceeded")) is True

    def test_non_retryable(self):
        assert _is_retryable_gcp_error(ValueError("bad")) is False
        assert _is_retryable_gcp_error(RuntimeError("crash")) is False


# ── Retry logic in _submit_stage_sweep ──────────────────────────────────


def _make_submit_kwargs(search_space=None, **overrides):
    """Build default kwargs for _submit_stage_sweep to reduce test boilerplate."""
    defaults = dict(
        aiplatform=MagicMock(),
        hpt_module=MagicMock(),
        species="velociraptor",
        stage=1,
        algorithm="ppo",
        timesteps=100000,
        n_envs=4,
        trials=5,
        parallel=2,
        bucket="test-bucket",
        image="test-image",
        machine_type="n1-standard-8",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        search_space=search_space
        or {"ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 3e-4, "scale": "log"}},
    )
    defaults.update(overrides)
    return defaults


class TestSubmitStageRetry:
    def test_retries_on_resource_exhausted(self):
        """hpt_job.run() is retried on retryable errors."""
        ResourceExhausted = type("ResourceExhausted", (Exception,), {})
        mock_job = MagicMock()
        mock_job.run.side_effect = [
            ResourceExhausted("quota"),
            ResourceExhausted("quota"),
            None,  # success on 3rd attempt
        ]

        kwargs = _make_submit_kwargs()
        kwargs["aiplatform"].HyperparameterTuningJob.return_value = mock_job
        kwargs["aiplatform"].CustomJob.return_value = MagicMock()

        with patch("time.sleep"):
            result = _submit_stage_sweep(**kwargs)

        assert mock_job.run.call_count == 3
        assert result is mock_job

    def test_raises_after_all_retries_exhausted(self):
        """After all retry attempts are exhausted, _SweepJobFailed is raised with the hpt_job."""
        ResourceExhausted = type("ResourceExhausted", (Exception,), {})
        mock_job = MagicMock()
        mock_job.run.side_effect = ResourceExhausted("quota")

        kwargs = _make_submit_kwargs()
        kwargs["aiplatform"].HyperparameterTuningJob.return_value = mock_job
        kwargs["aiplatform"].CustomJob.return_value = MagicMock()

        with patch("time.sleep"), pytest.raises(_SweepJobFailed) as exc_info:
            _submit_stage_sweep(**kwargs)

        # 1 initial + 3 retries = 4 total attempts
        assert mock_job.run.call_count == 4
        assert exc_info.value.hpt_job is mock_job
        assert isinstance(exc_info.value.__cause__, ResourceExhausted)

    def test_non_retryable_error_raised_immediately(self):
        """Non-retryable errors are wrapped in _SweepJobFailed without retry."""
        mock_job = MagicMock()
        mock_job.run.side_effect = ValueError("bad parameter")

        kwargs = _make_submit_kwargs()
        kwargs["aiplatform"].HyperparameterTuningJob.return_value = mock_job
        kwargs["aiplatform"].CustomJob.return_value = MagicMock()

        with patch("time.sleep"), pytest.raises(_SweepJobFailed) as exc_info:
            _submit_stage_sweep(**kwargs)

        assert mock_job.run.call_count == 1
        assert exc_info.value.hpt_job is mock_job
        assert isinstance(exc_info.value.__cause__, ValueError)


# ── resume_run output directory isolation ─────────────────────────────


class TestResumeRunOutputDir:
    def test_resume_run_changes_output_base(self):
        """resume_run > 0 appends _r{N} to the output directory."""
        kwargs = _make_submit_kwargs(stage=2, resume_run=1)
        mock_job = MagicMock()
        kwargs["aiplatform"].HyperparameterTuningJob.return_value = mock_job
        kwargs["aiplatform"].CustomJob.return_value = MagicMock()

        _submit_stage_sweep(**kwargs)

        call_args = kwargs["aiplatform"].CustomJob.call_args
        worker_specs = call_args[1]["worker_pool_specs"]
        trial_args = worker_specs[0]["container_spec"]["args"]
        output_idx = trial_args.index("--output-dir")
        output_dir = trial_args[output_idx + 1]
        assert output_dir == "/gcs/test-bucket/sweeps/velociraptor/stage2_r1"

    def test_resume_run_zero_uses_default_path(self):
        """resume_run=0 (default) uses the standard output directory."""
        kwargs = _make_submit_kwargs(stage=2)
        mock_job = MagicMock()
        kwargs["aiplatform"].HyperparameterTuningJob.return_value = mock_job
        kwargs["aiplatform"].CustomJob.return_value = MagicMock()

        _submit_stage_sweep(**kwargs)

        call_args = kwargs["aiplatform"].CustomJob.call_args
        worker_specs = call_args[1]["worker_pool_specs"]
        trial_args = worker_specs[0]["container_spec"]["args"]
        output_idx = trial_args.index("--output-dir")
        output_dir = trial_args[output_idx + 1]
        assert output_dir == "/gcs/test-bucket/sweeps/velociraptor/stage2"


# ── restart_job_on_worker_restart ──────────────────────────────────


class TestRestartJobOnWorkerRestart:
    def test_restart_passed_to_run(self):
        """restart_job_on_worker_restart=True is forwarded to hpt_job.run()."""
        kwargs = _make_submit_kwargs(restart_job_on_worker_restart=True)
        mock_job = MagicMock()
        kwargs["aiplatform"].HyperparameterTuningJob.return_value = mock_job
        kwargs["aiplatform"].CustomJob.return_value = MagicMock()

        _submit_stage_sweep(**kwargs)
        mock_job.run.assert_called_once_with(sync=True, restart_job_on_worker_restart=True)

    def test_restart_not_passed_when_false(self):
        """restart_job_on_worker_restart=False (default) does not add the kwarg."""
        kwargs = _make_submit_kwargs()
        mock_job = MagicMock()
        kwargs["aiplatform"].HyperparameterTuningJob.return_value = mock_job
        kwargs["aiplatform"].CustomJob.return_value = MagicMock()

        _submit_stage_sweep(**kwargs)
        mock_job.run.assert_called_once_with(sync=True)

    def test_cli_arg_parsed(self):
        """--restart-job-on-worker-restart is parsed correctly."""
        from environments.shared.scripts.sweep.__main__ import _build_parser

        parser = _build_parser()

        # launch mode
        args = parser.parse_args(
            [
                "launch",
                "--species",
                "velociraptor",
                "--project",
                "test",
                "--bucket",
                "b",
                "--image",
                "img",
                "--restart-job-on-worker-restart",
            ]
        )
        assert args.restart_job_on_worker_restart is True

        # launch mode with --no- prefix
        args = parser.parse_args(
            [
                "launch",
                "--species",
                "velociraptor",
                "--project",
                "test",
                "--bucket",
                "b",
                "--image",
                "img",
                "--no-restart-job-on-worker-restart",
            ]
        )
        assert args.restart_job_on_worker_restart is False

        # launch-all mode
        args = parser.parse_args(
            [
                "launch-all",
                "--species",
                "velociraptor",
                "--project",
                "test",
                "--bucket",
                "b",
                "--image",
                "img",
                "--restart-job-on-worker-restart",
            ]
        )
        assert args.restart_job_on_worker_restart is True

    def test_cli_default_is_false(self):
        """Default value for --restart-job-on-worker-restart is False."""
        from environments.shared.scripts.sweep.__main__ import _build_parser

        parser = _build_parser()
        args = parser.parse_args(
            [
                "launch",
                "--species",
                "velociraptor",
                "--project",
                "test",
                "--bucket",
                "b",
                "--image",
                "img",
            ]
        )
        assert args.restart_job_on_worker_restart is False
