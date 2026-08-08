"""Cross-check statue-derived gate constants against the plant manifest.

``min_avg_reward`` and ``collapse_peak_floor_reference`` are DERIVED from the
zero-action statue and neither updates itself -- their TOML comments have
said so since issue #491, and forgetting is silent: training proceeds against
a collapse floor and rail calibrated on a plant that no longer exists.  A
reward-weight change at least leaves the statue's trajectory intact, but a
plant revision moves the trajectory itself, so only a re-run of
``zero_action_baseline.py`` / ``stance_quality_baseline.py`` is evidence.

The cross-check is a repo test rather than a runtime hook because plants can
only change through a PR: the manifest generator refuses a changed physics
fingerprint without a revision increase, and CI re-checks monotonicity
against the PR base.  Any plant bump therefore lands exactly where this test
runs, and a bump that forgets the statue re-measurement fails the PR instead
of the training run it would have silently mis-calibrated.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = REPO_ROOT / "configs" / "plant_manifest.generated.json"
CONFIGS_ROOT = REPO_ROOT / "configs"


def _stage_curriculum_tables() -> list[tuple[str, Path, dict]]:
    """Every per-species stage TOML's [curriculum] table, with its species."""
    tables = []
    for species_dir in sorted(p for p in CONFIGS_ROOT.iterdir() if p.is_dir()):
        for toml_path in sorted(species_dir.glob("*.toml")):
            config = tomllib.loads(toml_path.read_text())
            if "curriculum" in config:
                tables.append((species_dir.name, toml_path, config["curriculum"]))
    return tables


def test_statue_derived_constants_record_the_plant_they_were_measured_on() -> None:
    """Every stage that sets the statue-derived collapse reference must pin
    the physics revision it was measured on, and that pin must match the
    manifest -- so a plant revision bump cannot land without either
    re-measuring the statue or consciously updating the provenance pin in
    the same diff (where a reviewer sees the constants did NOT move)."""
    manifest = json.loads(MANIFEST_PATH.read_text())
    plants = manifest["plants"]

    checked = 0
    for species, toml_path, curriculum in _stage_curriculum_tables():
        if "collapse_peak_floor_reference" not in curriculum:
            assert "statue_constants_physics_revision" not in curriculum, (
                f"{toml_path}: statue_constants_physics_revision without "
                "collapse_peak_floor_reference records provenance for a "
                "constant the stage does not set"
            )
            continue
        assert "statue_constants_physics_revision" in curriculum, (
            f"{toml_path}: collapse_peak_floor_reference is statue-derived "
            "and must record the plant it was measured on -- set "
            "statue_constants_physics_revision alongside it (and re-measure "
            "with zero_action_baseline.py if the plant moved)"
        )
        recorded = curriculum["statue_constants_physics_revision"]
        current = plants[species]["physics"]["revision"]
        assert recorded == current, (
            f"{toml_path}: statue constants were measured on physics "
            f"revision {recorded}, but the manifest is at revision {current}. "
            "Re-measure the statue on the current plant "
            "(zero_action_baseline.py / stance_quality_baseline.py), "
            "re-derive min_avg_reward and collapse_peak_floor_reference, and "
            "update this pin in the same commit."
        )
        checked += 1

    # The check must never pass vacuously: trex stage 1 carries the reference
    # today, and losing it without a conscious edit here means the constants
    # lost their only freshness guard.
    assert checked >= 1, "no stage TOML carries collapse_peak_floor_reference; update this test's premise"
