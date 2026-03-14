"""Tests for sweep constants.py — presets, exceptions."""

from unittest.mock import MagicMock

from environments.shared.scripts.sweep import (
    NET_ARCH_PRESETS,
    SweepStageError,
    _SweepJobFailed,
)

# ── NET_ARCH_PRESETS ─────────────────────────────────────────────────────


class TestNetArchPresets:
    def test_all_presets_are_lists_of_ints(self):
        for name, arch in NET_ARCH_PRESETS.items():
            assert isinstance(arch, list), f"Preset {name} should be a list"
            assert all(isinstance(x, int) for x in arch), f"Preset {name} should contain ints"

    def test_expected_presets_exist(self):
        expected = {"small", "medium", "large", "deep", "tapered", "deep_tapered"}
        assert set(NET_ARCH_PRESETS.keys()) == expected


# ── SweepStageError ─────────────────────────────────────────────────────


class TestSweepStageError:
    def test_is_exception(self):
        assert issubclass(SweepStageError, Exception)

    def test_message(self):
        err = SweepStageError("stage 2 failed")
        assert "stage 2 failed" in str(err)


# ── _SweepJobFailed ────────────────────────────────────────────────────


class TestSweepJobFailed:
    def test_is_sweep_stage_error(self):
        assert issubclass(_SweepJobFailed, SweepStageError)

    def test_carries_hpt_job(self):
        mock_job = MagicMock()
        exc = _SweepJobFailed("job failed", hpt_job=mock_job)
        assert exc.hpt_job is mock_job
        assert "job failed" in str(exc)

    def test_hpt_job_defaults_to_none(self):
        exc = _SweepJobFailed("no job")
        assert exc.hpt_job is None
