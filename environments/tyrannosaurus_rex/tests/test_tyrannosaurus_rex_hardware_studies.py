"""Reproduce dated sizing/budget arithmetic after model/report reorganizations."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from environments.shared.paths import REPOSITORY_ROOT
from examples.tyrannosaurus_rex_50cm import PARTS_BUDGET_PATH


def _assert_equal(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_equal(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected, strict=True):
            _assert_equal(a, e)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-12, abs=1e-14)
    else:
        assert actual == expected


def test_all_sizing_and_budget_calculators_reproduce_their_retained_reference(tmp_path):
    root = REPOSITORY_ROOT
    relative = Path("docs/hardware/studies")
    studies = tmp_path / relative
    shutil.copytree(root / relative, studies)
    baseline = Path("examples/mini_tyrannosaurus_rex/baseline_rev10/walking_validation.json")
    (tmp_path / baseline).parent.mkdir(parents=True)
    shutil.copy2(root / baseline, tmp_path / baseline)
    medium = tmp_path / PARTS_BUDGET_PATH.parent.relative_to(root)
    # Study directory names are historical document paths, not runtime model
    # identifiers. Resolve the other size from its declared input metadata.
    large_inputs = [p for p in studies.glob("*/inputs.json") if json.loads(p.read_text())["target_height_m"] == 0.8]
    assert len(large_inputs) == 1
    large = large_inputs[0].parent
    commands = [
        [medium / "calculate.py"],
        [
            medium / "calculate.py",
            "--input",
            large / "inputs.json",
            "--output",
            large / "results.json",
        ],
        [medium / "calculate_parts.py"],
        [studies / "compare_costs.py"],
    ]
    # The current gait report is deliberately absent: these historical scaling
    # screens must not accidentally start using a replacement design's trial.
    for command in commands:
        subprocess.run([sys.executable, *map(str, command)], check=True, capture_output=True, text=True, cwd=tmp_path)
    for path in (
        medium / "results.json",
        large / "results.json",
        medium / "parts_budget_results.json",
        studies / "cost_comparison.json",
    ):
        _assert_equal(json.loads(path.read_text()), json.loads((root / path.relative_to(tmp_path)).read_text()))
    assert (medium / "parts_budget.csv").read_text() == PARTS_BUDGET_PATH.with_suffix(".csv").read_text()
