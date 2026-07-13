"""Schema and provenance checks for curated result summaries."""

import json
from datetime import date
from pathlib import Path
from typing import Any, cast

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RESULT_SUMMARIES = tuple(sorted((REPOSITORY_ROOT / "results").glob("*/*/summary.json")))

ALLOWED_MODEL_REVISION_STATUSES = {"current", "historical"}
ALLOWED_VERIFICATION_STATUSES = {"verified", "unverified"}
REQUIRED_PROVENANCE_FIELDS = {
    "model_revision_status",
    "verification_status",
    "evaluation_episodes",
    "repository_commit",
    "model_hash",
    "config_hash",
}
REQUIRED_IDENTITY_FIELDS = ("repository_commit", "model_hash", "config_hash")


def _load_summary(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as summary_file:
        return cast(dict[str, Any], json.load(summary_file))


def test_result_summaries_are_discovered() -> None:
    """Protect against a vacuous schema test caused by a bad repository path."""
    assert RESULT_SUMMARIES, "no results/*/*/summary.json files were found"


@pytest.mark.parametrize("summary_path", RESULT_SUMMARIES, ids=lambda path: str(path.relative_to(REPOSITORY_ROOT)))
def test_result_summary_schema_and_provenance(summary_path: Path) -> None:
    summary = _load_summary(summary_path)

    assert summary["schema_version"] == 2
    assert {
        "species",
        "algorithm",
        "backend",
        "backend_version",
        "date",
        "stages",
        "provenance",
    } <= summary.keys()
    assert summary["species"] == summary_path.parents[1].name
    assert summary["algorithm"].lower() == summary_path.parent.name
    assert summary["backend"] == "stable-baselines3"
    backend_version = summary["backend_version"]
    assert backend_version is None or (isinstance(backend_version, str) and backend_version.strip())
    date.fromisoformat(summary["date"])

    stages = summary["stages"]
    assert set(stages) == {"1", "2", "3"}
    for stage in stages.values():
        assert isinstance(stage["name"], str) and stage["name"]
        assert isinstance(stage["description"], str) and stage["description"]
        assert isinstance(stage["timesteps"], int) and stage["timesteps"] > 0
        assert isinstance(stage["stage_passed"], bool)
        success_rate = stage.get("mean_success_rate")
        assert success_rate is None or isinstance(success_rate, (int, float))
        assert success_rate is None or 0.0 <= success_rate <= 1.0

    assert summary["total_timesteps"] == sum(stage["timesteps"] for stage in stages.values())

    provenance = summary["provenance"]
    assert REQUIRED_PROVENANCE_FIELDS <= provenance.keys()
    assert provenance["model_revision_status"] in ALLOWED_MODEL_REVISION_STATUSES
    assert provenance["verification_status"] in ALLOWED_VERIFICATION_STATUSES
    evaluation_episodes = provenance["evaluation_episodes"]
    assert evaluation_episodes is None or (
        isinstance(evaluation_episodes, int) and not isinstance(evaluation_episodes, bool) and evaluation_episodes > 0
    )

    for field in REQUIRED_IDENTITY_FIELDS:
        value = provenance[field]
        assert value is None or isinstance(value, str)
        assert value is None or value.strip(), f"{field} must be null or a non-empty string"

    claims_current_or_verified = (
        provenance["model_revision_status"] == "current" or provenance["verification_status"] == "verified"
    )
    if claims_current_or_verified:
        missing_fields = [field for field in REQUIRED_IDENTITY_FIELDS if not provenance[field]]
        if evaluation_episodes is None:
            missing_fields.append("evaluation_episodes")
        if backend_version is None:
            missing_fields.append("backend_version")
        assert not missing_fields, f"current or verified results require provenance identifiers: {missing_fields}"
