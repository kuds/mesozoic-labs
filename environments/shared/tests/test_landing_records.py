"""The one-landing-record rule (docs/README.md, Conventions; adopted by cleanup CU-17).

A PR's landing (its number, merge date and CI result) is recorded in two places only: the status table of
docs/CONSOLIDATION_PLAN_2026_09.md and the PR's CHANGELOG entry.  The rule governs landings from CU-17 on, so the
records of PRs up to LAST_PR_RECORDED_ELSEWHERE (#591, ROW-4/6, the last landing the earlier practice recorded across
the docs) are history and are not checked.  This file fails a landing record of a later PR anywhere else in the
living docs.  Dated investigations and reviews (append-only evidence) and the site are not scanned.
"""

from __future__ import annotations

import re
from pathlib import Path

from environments.shared.paths import REPOSITORY_ROOT
from environments.shared.tests.ci_workflow_helpers import glob_matches, path_filters

LAST_PR_RECORDED_ELSEWHERE = 591
STATUS_HOME = "docs/CONSOLIDATION_PLAN_2026_09.md"
DATED_DIRECTORIES = ("investigations", "reviews")

#: The forms a landing record takes in these docs: "landed as #600", "merged as #600", "#600 landed",
#: "#600's CI", and a row of a landed-PR table ("| #600 | 2026-10-10 |", "| #598–#600 | 2026-10-10 |").
#: Group 1 holds the PR reference(s); the record names the highest number in it.
_LANDING_PHRASES = (
    re.compile(
        r"\b(?:landed|merged)\b[^.;|\n]{0,50}?(#\d+(?:(?:,\s*|,?\s+and\s+|\s*/\s*)#\d+|\s*[–-]\s*#?\d+\b)*)",
        re.IGNORECASE,
    ),
    re.compile(r"((?:#\d+\b[^.;|\n]{0,40}?)+)\b(?:landed|merged)\b", re.IGNORECASE),
    re.compile(r"(#\d+)'s CI\b"),
    re.compile(r"^\|\s*(#\d+(?:\s*[–-]\s*#?\d+)?)\s*\|\s*20\d\d-\d\d-\d\d\s*\|", re.MULTILINE),
)
#: The PR numbers in a match's group 1: each "#NNN", and the end of a "#598–600" range.
_PR_NUMBER = re.compile(r"#(\d+)|[–-]\s*#?(\d+)$")
#: A line that a soft-wrapped line is not joined to or from: a heading, a table row or a fence.
_OWN_LINE = re.compile(r"[ \t]*(?:#{1,6}[ \t]|\||```|~~~)")
#: A line that starts a block of its own: one of those, a list item or a quote.
_BLOCK_START = re.compile(r"[ \t]*(?:#{1,6}[ \t]|\||```|~~~|[-*+>][ \t]|\d+[.)][ \t])")


def living_docs(root: Path = REPOSITORY_ROOT) -> list[Path]:
    """The documents the rule covers: the root and species READMEs, CONTRIBUTING and every living doc."""
    docs = [
        path
        for path in (root / "docs").rglob("*.md")
        if path.relative_to(root / "docs").parts[0] not in DATED_DIRECTORIES
    ]
    return sorted(
        {
            root / "README.md",
            root / "CONTRIBUTING.md",
            root / "results" / "README.md",
            *docs,
            *root.glob("environments/*/README.md"),
        }
    )


def _without_status_section(text: str) -> str:
    head, found, rest = text.partition("\n## Status (")
    assert found, f"{STATUS_HOME} has no '## Status (' section"
    _, _, after = rest.partition("\n## ")
    return head + "\n## " + after


def _unwrapped(text: str) -> str:
    """``text`` with each wrapped line joined to the line before it: a phrase the docs wrap is on one line."""
    lines: list[str] = []
    for line in text.splitlines():
        if (
            lines
            and lines[-1].strip()
            and line.strip()
            and not _BLOCK_START.match(line)
            and not _OWN_LINE.match(lines[-1])
        ):
            lines[-1] += " " + line.strip()
        else:
            lines.append(line)
    return "\n".join(lines)


def landing_records(text: str, *, after: int) -> list[str]:
    """Each landing phrase in ``text`` (its wrapped lines joined) that names a PR numbered above ``after``."""
    found = []
    text = _unwrapped(text)
    for pattern in _LANDING_PHRASES:
        for match in pattern.finditer(text):
            numbers = [int(a or b) for a, b in _PR_NUMBER.findall(match.group(1))]
            if max(numbers) > after:
                found.append(match.group(0).strip())
    return found


def test_later_landings_are_recorded_only_in_the_status_table_and_the_changelog() -> None:
    covered = {path.relative_to(REPOSITORY_ROOT).as_posix() for path in living_docs()}
    assert {STATUS_HOME, "README.md", "docs/NEXT_STEPS.md", "docs/CLEANUP_PLAN_2026_09.md"} <= covered
    assert not any(path.startswith(("docs/investigations/", "docs/reviews/")) for path in covered)
    offenders = []
    for path in living_docs():
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        if relative == STATUS_HOME:
            text = _without_status_section(text)
        offenders += [f"{relative}: {phrase}" for phrase in landing_records(text, after=LAST_PR_RECORDED_ELSEWHERE)]
    assert not offenders, (
        "a landing is recorded only in the consolidation plan's status table and the CHANGELOG "
        "(docs/README.md, Conventions); point there instead:\n" + "\n".join(offenders)
    )


def test_every_scanned_document_triggers_the_workflow() -> None:
    # A PR that edits only one of these documents must still run the scan (gap review CI7).
    filters = path_filters()
    assert len(filters) == 2, f"expected the push and pull_request path filters, found {len(filters)}"
    for path in living_docs():
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        for patterns in filters:
            assert any(glob_matches(pattern, relative) for pattern in patterns), (
                f"{relative} is missing from a python-ci.yml paths filter, so a PR editing only it skips this check"
            )


def test_the_landing_phrases_are_recognised() -> None:
    """The scan sees each form below, wrapped or not, and nothing at or below the cutoff."""
    examples = {
        "CU-10b landed as #590 on 2026-10-03, which completes CU-10": 590,
        "and ROW-4/6 (the notebook PR for decisions 4 and 6) landed as #591": 591,
        "### PR-7. One behavior env ... — LANDED as #556, 2026-09-25": 556,
        "`main` = `d38c785` (#591 merged 2026-10-03 18:12 UTC)": 591,
        "| #591's CI (ROW-4/6, with the records of #590) | 10 CI jobs |": 591,
        "| #589 | 2026-10-03 | Cleanup CU-15, reduced |": 589,
        "| #528–#531 | 2026-09-12 | Recipes Phase A |": 531,
        "| #598–600 | 2026-10-10 | a range written without the second hash |": 600,
        "#590 and #591 landed the same day": 591,
        "landed as #590 and #591": 591,
        "landed as #590/#591": 591,
        "Phase D landed 2026-09-15 as a separate pilot pipeline (#540/#541)": 541,
        "ROW-4/6 landed\nas #591 on 2026-10-03": 591,
        "CU-14b, the first part of CU-14, landed\n   as #570, 2026-09-29": 570,
        "CU-14a landed as\n#581 on 2026-10-01": 581,
    }
    for text, number in examples.items():
        assert landing_records(text, after=number - 1), text
        assert not landing_records(text, after=number), text
    assert not landing_records("cleanup CU-13 (#588) made `extends` per table", after=0)
    assert not landing_records("the status table records which PRs have landed", after=0)
    assert not landing_records("### PR-8. The next PR (#593)\nLanded (status table above)", after=0)
    assert not landing_records("- CU-17 (#593)\n- landed sections become pointers", after=0)
