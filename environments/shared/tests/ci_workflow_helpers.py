"""Read python-ci.yml as text for the tests that pin its jobs, steps and path filters.

As text, not YAML: the test matrix installs the ``test`` extra, which brings no PyYAML.
"""

from __future__ import annotations

import re
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CI_WORKFLOW = REPOSITORY_ROOT / ".github" / "workflows" / "python-ci.yml"

#: ``paths:`` opening a list, optionally anchoring it (``paths: &name``).
_PATHS = re.compile(r"paths:(?:\s+&([\w-]+))?(?:\s+#.*)?")
#: ``paths: *name``: a trigger that reuses an anchored list (GitHub takes aliases, not merge keys).
_PATHS_ALIAS = re.compile(r"paths:\s+\*([\w-]+)(?:\s+#.*)?")
#: The first line of the next job (two spaces in) or of a top-level key after ``jobs:`` (none in).
_NEXT_JOB = re.compile(r"^(?:  )?\S", re.MULTILINE)


def ci_text() -> str:
    return CI_WORKFLOW.read_text(encoding="utf-8")


def ci_code() -> str:
    """python-ci.yml without its full-line comments, so a comment line can neither satisfy nor trip a pin.

    A trailing comment on a code line stays (the file has none today).
    """
    return "\n".join(line for line in ci_text().splitlines() if not line.lstrip().startswith("#")) + "\n"


def job_block(name: str) -> str:
    """Job *name* of :func:`ci_code`, from its ``  name:`` line up to the next job."""
    text = ci_code()
    start = text.index(f"\n  {name}:\n")
    end = _NEXT_JOB.search(text, start + 2)
    return text[start : end.start() if end else len(text)]


def job_steps(name: str) -> list[str]:
    """The steps of :func:`job_block` *name*, each from its ``- `` to the next."""
    job = job_block(name)
    starts = [match.start() for match in re.finditer(r"^      - ", job, re.MULTILINE)]
    return [job[a:b] for a, b in zip(starts, [*starts[1:], len(job)])]


def path_filters(text: str | None = None) -> list[list[str]]:
    """The ``paths:`` lists of python-ci.yml's triggers (or of *text*), one list per trigger.

    A trigger that names an earlier trigger's anchored list by its alias
    (``paths: *name`` after ``paths: &name``) gets that same list object.
    """
    filters: list[list[str]] = []
    anchors: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in (ci_text() if text is None else text).splitlines():
        stripped = line.strip()
        start, alias = _PATHS.fullmatch(stripped), _PATHS_ALIAS.fullmatch(stripped)
        if alias:
            name = alias.group(1)
            assert name in anchors, f"paths: *{name} names no earlier paths: &{name}"
            filters.append(anchors[name])
            current = None
        elif start:
            current = []
            filters.append(current)
            if start.group(1):
                anchors[start.group(1)] = current
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
