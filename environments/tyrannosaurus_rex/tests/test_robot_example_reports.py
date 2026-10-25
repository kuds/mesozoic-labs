"""Current engineering evidence must identify the sources actually checked.

Archived studies are deliberately excluded; their results describe old plants.
CI still executes the physical regressions independently of these saved reports.
"""

import hashlib
import json

import pytest

from environments.shared.paths import REPOSITORY_ROOT


@pytest.mark.parametrize(
    "directory,report,source_base",
    [
        ("mini_tyrannosaurus_rex", name, "examples/mini_tyrannosaurus_rex")
        for name in ("validation.json", "balance_validation.json", "character_validation.json", "tail_validation.json")
    ]
    + [
        ("tyrannosaurus_rex_50cm", "validation.json", "examples/tyrannosaurus_rex_50cm"),
        ("tyrannosaurus_rex_50cm", "head_study/results.json", "examples/tyrannosaurus_rex_50cm"),
        ("tyrannosaurus_rex_50cm", "proportion_study/results.json", "."),
    ],
)
def test_current_report_source_hashes(directory, report, source_base):
    payload = json.loads((REPOSITORY_ROOT / "examples" / directory / report).read_text())
    assert payload["source_sha256"]
    for name, expected in payload["source_sha256"].items():
        source = REPOSITORY_ROOT / source_base / name
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected, f"Regenerate {report}: {name} changed"


def test_current_walking_replays_are_bound_to_the_loaded_plant_and_measurement():
    base = REPOSITORY_ROOT / "examples/mini_tyrannosaurus_rex"
    report = json.loads((base / "walking_validation.json").read_text())
    helper = REPOSITORY_ROOT / "environments/shared/robot_validation.py"
    assert len(report["models"]) == 3
    for model in report["models"]:
        assert model["controller_sha256"] == hashlib.sha256((base / "walking_pilot.py").read_bytes()).hexdigest()
        assert model["support_validation_sha256"] == hashlib.sha256(helper.read_bytes()).hexdigest()
        for name, expected in (model["source_sha256"] | model["generator_source_sha256"]).items():
            assert hashlib.sha256((base / name).read_bytes()).hexdigest() == expected, name
        assert {row["cap"] for row in model["results"]} == {0.1, 0.15, 0.2, 0.3}
        assert all(row["requested_duration_s"] == 30 for row in model["results"])
