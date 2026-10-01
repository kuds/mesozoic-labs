"""The one sha256 digest pattern and the field validators recorded JSON is checked with.

Standard library only, so the result schema, the species catalog and the
result-bundle readers share it without an import cost or an import cycle.
Each validator raises the caller's own error class (``error=``) with one
message per rule; ``result_schema`` and ``species_catalog`` bind their six
private validator names (``_require_*`` and ``_optional_*``) to these with
:func:`functools.partial`.  The bundle readers (evidence, gate verdict,
audit, ancestor records) keep their own messages and ask only
:func:`is_sha256_digest`.

``plant_contract/digests.py`` keeps its own character-loop check: it is a
hasher file the cleanup plan holds at a zero-line diff (§4.2).  It accepts
exactly the strings :func:`is_sha256_digest` accepts.
"""

from __future__ import annotations

import math
import re
from typing import Any, TypeGuard, cast

#: A recorded content digest: ``sha256:`` and 64 lowercase hex digits.  Use
#: :func:`is_sha256_digest`, which matches it whole: ``match`` or ``search``
#: would accept a trailing newline or trailing junk.
SHA256_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")


def is_sha256_digest(value: object) -> TypeGuard[str]:
    """Whether *value* is a string that is exactly ``sha256:<64 lowercase hex>``.

    False for anything that is not a string, rather than a ``TypeError``.
    """
    return isinstance(value, str) and SHA256_DIGEST_PATTERN.fullmatch(value) is not None


def require_mapping(value: Any, *, field: str, error: type[Exception]) -> dict[str, Any]:
    """*value* when it is a JSON object (a ``dict``), else ``error("{field} must be an object")``."""
    if not isinstance(value, dict):
        raise error(f"{field} must be an object")
    return cast(dict[str, Any], value)


def require_nonempty_string(value: Any, *, field: str, error: type[Exception]) -> str:
    """*value* stripped, when it is a string with a non-blank character; else *error*."""
    if not isinstance(value, str) or not value.strip():
        raise error(f"{field} must be a non-empty string")
    return value.strip()


def optional_nonempty_string(value: Any, *, field: str, error: type[Exception]) -> str | None:
    """``None`` for ``None``, else :func:`require_nonempty_string`."""
    if value is None:
        return None
    return require_nonempty_string(value, field=field, error=error)


def require_positive_int(value: Any, *, field: str, error: type[Exception]) -> int:
    """*value* when it is an ``int`` above zero (a ``bool`` is not one); else *error*."""
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise error(f"{field} must be a positive integer")
    return value


def optional_number(value: Any, *, field: str, error: type[Exception]) -> int | float | None:
    """``None`` for ``None``; *value* when it is a finite ``int`` or ``float`` (not a ``bool``); else *error*."""
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise error(f"{field} must be a number or null")
    if not math.isfinite(float(value)):
        raise error(f"{field} must be finite or null")
    return value


def require_sha256(value: Any, *, field: str, error: type[Exception]) -> str:
    """*value* when :func:`is_sha256_digest` accepts it; else *error*."""
    if not is_sha256_digest(value):
        raise error(f"{field} must be sha256:<64 lowercase hex>")
    return value
