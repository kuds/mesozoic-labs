"""Value normalisation and duration formatting for result reporting.

Leaf helpers with no dependencies on the rest of the reporting package."""

from __future__ import annotations

import math
from typing import Any


def parse_optional_bool(value: Any) -> bool | None:
    """Parse a serialized boolean without treating non-empty strings as true."""
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and math.isnan(value):
            return None
        if value == 1:
            return True
        if value == 0:
            return False
    scalar_item = getattr(value, "item", None)
    if callable(scalar_item):
        try:
            scalar = scalar_item()
        except (TypeError, ValueError):
            return None
        if scalar is not value:
            return parse_optional_bool(scalar)
    return None

def _optional_metric(value: Any, *, digits: int | None = None) -> int | float | None:
    """Normalize an optional numeric metric for JSON output."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"result metric must be numeric or null, got {value!r}")
    if not isinstance(value, (int, float)):
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"result metric must be numeric or null, got {value!r}") from exc
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    if digits is None and isinstance(value, int):
        return int(value)
    return round(numeric, digits) if digits is not None else numeric

def format_duration(seconds: float) -> str:
    """Format seconds into a human-readable string (e.g. ``2h 15m 30s``)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"

def format_duration_hms(seconds: float) -> str:
    """Format seconds as ``H:MM:SS``."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"
