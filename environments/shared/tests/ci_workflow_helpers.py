"""Read python-ci.yml as text for the tests that pin its steps and path filters."""

from __future__ import annotations

import re
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CI_WORKFLOW = REPOSITORY_ROOT / ".github" / "workflows" / "python-ci.yml"


def ci_text() -> str:
    return CI_WORKFLOW.read_text(encoding="utf-8")


def path_filters(text: str | None = None) -> list[list[str]]:
    """The ``paths:`` lists of python-ci.yml's triggers (or of *text*), one list per trigger."""
    filters: list[list[str]] = []
    current: list[str] | None = None
    for line in (ci_text() if text is None else text).splitlines():
        stripped = line.strip()
        if stripped == "paths:":
            current = []
            filters.append(current)
        elif current is not None and stripped.startswith("- "):
            # Either YAML quote style: a `!` pattern must be quoted, and GitHub's docs single-quote it.
            current.append(stripped[2:].strip().strip("\"'"))
        elif current is not None and stripped and not stripped.startswith("#"):
            current = None
    # The trigger pins treat a filter as any-match; GitHub lets a later `!` pattern exclude a path.
    negated = [pattern for patterns in filters for pattern in patterns if pattern.startswith("!")]
    assert not negated, f"python-ci.yml's path filters exclude {negated}; a pinned path could stop triggering"
    return filters


def glob_matches(pattern: str, path: str) -> bool:
    """GitHub's path-filter glob: ``**`` crosses directories, ``*`` does not."""
    regex = re.escape(pattern).replace(r"\*\*", ".*").replace(r"\*", "[^/]*")
    return re.fullmatch(regex, path) is not None
