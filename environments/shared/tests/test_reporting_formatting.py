"""Tests for environments.shared.reporting.formatting."""

import pytest

from environments.shared.reporting import format_duration, format_duration_hms, parse_optional_bool


class TestFormatDuration:
    """Tests for format_duration (human-readable duration)."""

    def test_seconds_only(self):
        assert format_duration(45) == "45s"

    def test_minutes_and_seconds(self):
        assert format_duration(130) == "2m 10s"

    def test_hours_minutes_seconds(self):
        assert format_duration(3661) == "1h 1m 1s"

    def test_zero(self):
        assert format_duration(0) == "0s"

    def test_exact_hour(self):
        assert format_duration(3600) == "1h 0m 0s"

    def test_exact_minute(self):
        assert format_duration(60) == "1m 0s"

    def test_fractional_seconds(self):
        # Fractional seconds should be truncated (int conversion)
        assert format_duration(59.9) == "59s"


class TestFormatDurationHms:
    """Tests for format_duration_hms (H:MM:SS format)."""

    def test_zero(self):
        assert format_duration_hms(0) == "0:00:00"

    def test_seconds_only(self):
        assert format_duration_hms(45) == "0:00:45"

    def test_minutes_and_seconds(self):
        assert format_duration_hms(130) == "0:02:10"

    def test_hours_minutes_seconds(self):
        assert format_duration_hms(3661) == "1:01:01"

    def test_large_value(self):
        assert format_duration_hms(36000) == "10:00:00"


class TestParseOptionalBool:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(True, True), (False, False), ("true", True), ("False", False), (1, True), (0, False), ("", None)],
    )
    def test_parses_serialized_booleans_strictly(self, value, expected):
        assert parse_optional_bool(value) is expected
