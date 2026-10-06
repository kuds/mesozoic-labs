"""No living code names the retired tuning routes as present (cleanup CU-7b).

D-D17 retired Ray Tune, the Vertex AI tuning sweeps and job route, and the HPT
report they read (cleanup PR-A and PR-A2; docs/CLEANUP_PLAN_2026_09.md §4.5).
None of the three names has a live meaning left in the library, the tests, the
stage TOMLs, the SB3 notebook or the build and CI configuration, so a line that
names one must say that it is retired ("retired" or "D-D17") on that line or
the one before or after it, so that rewrapping a history note does not fail
the check. The names are matched with their capitals: MuJoCo's ray casts, the
MJCF ``vertex`` attributes, the harness's blocked ``ray`` module and the kept
``_report_hpt_metrics`` identifier are all lowercase.

Not covered, because each still has a live meaning: JAX and MJX (the frozen MJX
interface core and the three dual declarations), "sweep" (the probe and noise
sweeps, ``--sweep-noise``) and GCS (the ``/gcs/`` mount detection the trainer
keeps). Not scanned: ``docs/`` and ``CHANGELOG.md`` (records), the Drive
summary notebook (§4.9 keeps its reader of legacy sweep directories) and
recorded data files.
"""

from __future__ import annotations

import re
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

RETIRED_TUNING_NAMES = re.compile(r"\b(?:Vertex|Ray|HPTs?)\b|\b(?:VertexAI|RayTune)")
RETIRED_MARKERS = ("retired", "D-D17")


def _living_files() -> list[Path]:
    this_file = Path(__file__).resolve()
    files = sorted(path for path in (REPOSITORY_ROOT / "environments").rglob("*.py") if path.resolve() != this_file)
    files += sorted((REPOSITORY_ROOT / "configs").rglob("*.toml"))
    files += sorted((REPOSITORY_ROOT / "configs").rglob("*.py"))
    files += sorted((REPOSITORY_ROOT / "environments").rglob("requirements*.txt"))
    files += sorted(REPOSITORY_ROOT.glob("*.py"))
    files += [REPOSITORY_ROOT / "notebooks" / "sb3_training.ipynb", REPOSITORY_ROOT / "pyproject.toml"]
    files += [REPOSITORY_ROOT / ".pre-commit-config.yaml"]
    files += sorted(path for path in (REPOSITORY_ROOT / ".github").rglob("*") if path.suffix in (".yml", ".yaml"))
    return files


def _unmarked_mentions(text: str) -> list[tuple[int, str]]:
    """``(line number, line)`` of each mention with no marker on it or on the line before or after it."""
    lines = text.splitlines()
    return [
        (index + 1, line)
        for index, line in enumerate(lines)
        if RETIRED_TUNING_NAMES.search(line)
        and not any(marker in near for near in lines[max(index - 1, 0) : index + 2] for marker in RETIRED_MARKERS)
    ]


def test_the_retired_tuning_routes_are_named_only_as_retired() -> None:
    unmarked = [
        f"{path.relative_to(REPOSITORY_ROOT).as_posix()}:{number}: {line.strip()}"
        for path in _living_files()
        for number, line in _unmarked_mentions(path.read_text(encoding="utf-8"))
    ]
    assert not unmarked, (
        "these lines name Vertex AI, Ray Tune or the HPT report, which D-D17 retired, without saying so on the "
        "same line or the one before or after it: describe what the code does now, or mark a history note "
        "'retired' / 'D-D17' (a live use of the word, such as a MuJoCo ray cast, is written in lowercase)\n"
        + "\n".join(unmarked)
    )


def test_the_scan_reaches_its_files_and_matches_only_the_retired_names() -> None:
    names = {path.relative_to(REPOSITORY_ROOT).as_posix() for path in _living_files() if path.is_file()}
    assert {
        "environments/shared/train_base.py",
        "environments/shared/tests/test_train_base.py",
        "configs/trex/stance.toml",
        "notebooks/sb3_training.ipynb",
        "pyproject.toml",
        ".github/workflows/python-ci.yml",
        ".github/actions/upload-coverage/action.yml",
        ".pre-commit-config.yaml",
    } <= names
    # Wording CU-7b replaced is caught; the lowercase live uses are not.
    for retired in (
        "HPT metric reported: x",
        "sync under concurrent Vertex AI HPT trials",
        "the Ray Tune worker",
        "submit the job to VertexAI",
        "the RayTuneReportCallback reports each eval",
        "the HPTs read metrics.json",
    ):
        assert RETIRED_TUNING_NAMES.search(retired), retired
    for live in (
        "mujoco.mj_rayHfield(model)",
        'mesh.get("vertex")',
        "_report_hpt_metrics(",
        '"ray", "mjlab"',
        "RayCast",
        "Vertices",
        "VertexBuffer",
    ):
        assert not RETIRED_TUNING_NAMES.search(live), live


def test_a_marker_counts_only_on_the_mention_or_a_line_beside_it() -> None:
    # A history note wrapped over two lines is marked, whichever side the marker lands on.
    assert _unmarked_mentions("the name is historical: the Vertex AI report\nit sent was retired.") == []
    assert _unmarked_mentions("since D-D17 there is no\nHPT report") == []
    # Two lines away, or past a blank line, the marker belongs to another sentence.
    assert _unmarked_mentions("a\nlog HPT eval\nb\nretired") == [(2, "log HPT eval")]
    assert _unmarked_mentions("retired\n\nthe Ray Tune worker") == [(3, "the Ray Tune worker")]
    assert _unmarked_mentions("the Ray Tune worker\n\nD-D17") == [(1, "the Ray Tune worker")]
