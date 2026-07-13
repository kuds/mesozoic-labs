"""Tests for the generated public species catalog."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest

from environments.shared.species_catalog import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_README_PATH,
    REPOSITORY_ROOT,
    CatalogError,
    _max_reported_velocity,
    _validate_result_summary,
    build_catalog,
    check_catalog,
)


def test_catalog_derives_current_model_and_stage_facts() -> None:
    catalog = build_catalog()
    species = {entry["id"]: entry for entry in catalog["species"]}

    assert {
        species_id: (
            entry["environment"]["observation_dim"],
            entry["environment"]["action_dim"],
            entry["model"]["nq"],
            entry["model"]["nv"],
            entry["model"]["nu"],
            entry["model"]["dynamic_mass_kg"],
        )
        for species_id, entry in species.items()
    } == {
        "velociraptor": (67, 22, 31, 30, 22, 13.5),
        "trex": (83, 21, 40, 37, 21, 85.4),
        "brachiosaurus": (83, 30, 38, 37, 30, 175.3),
    }

    assert [stage["timesteps"] for stage in species["velociraptor"]["stages"]] == [6_000_000, 8_000_000, 8_000_000]
    assert [stage["timesteps"] for stage in species["trex"]["stages"]] == [6_000_000, 8_000_000, 8_000_000]
    assert [stage["timesteps"] for stage in species["brachiosaurus"]["stages"]] == [
        6_000_000,
        16_000_000,
        12_000_000,
    ]


def test_catalog_keeps_current_configs_separate_from_historical_results() -> None:
    catalog = build_catalog()
    brachiosaurus = next(entry for entry in catalog["species"] if entry["id"] == "brachiosaurus")
    current_stage_three = brachiosaurus["stages"][2]
    published_stage_three = brachiosaurus["historical_results"][0]["stages"][2]

    assert current_stage_three["timesteps"] == 12_000_000
    assert published_stage_three["timesteps"] == 8_000_000
    assert brachiosaurus["historical_results"][0]["provenance"]["model_revision_status"] == "historical"
    assert brachiosaurus["historical_results"][0]["provenance"]["verification_status"] == "unverified"


def test_catalog_uses_nullable_video_for_unpublished_stage() -> None:
    catalog = build_catalog()
    brachiosaurus = next(entry for entry in catalog["species"] if entry["id"] == "brachiosaurus")
    assert brachiosaurus["stages"][2]["video"] is None


def test_catalog_labels_published_videos_with_artifact_provenance() -> None:
    catalog = build_catalog()
    videos = [
        stage["video"] for species in catalog["species"] for stage in species["stages"] if stage["video"] is not None
    ]

    assert len(videos) == 8
    assert all(video["algorithm"] == "PPO" for video in videos)
    assert all(video["backend"] == "stable-baselines3" for video in videos)
    assert all(video["backend_version"] is None for video in videos)
    assert all(video["model_revision_status"] == "historical" for video in videos)
    assert all(video["verification_status"] == "unverified" for video in videos)


def test_catalog_exports_effective_early_advancement_gates() -> None:
    catalog = build_catalog()
    species = {entry["id"]: entry for entry in catalog["species"]}

    for entry in species.values():
        stage_one, stage_two, stage_three = [stage["advancement_gate"] for stage in entry["stages"]]
        assert stage_one == {
            "min_avg_reward": 100.0,
            "min_avg_episode_length": 750,
            "min_avg_forward_velocity": None,
            "min_success_rate": None,
            "min_eval_episodes": 10,
            "required_consecutive": 3,
        }
        assert stage_two | {"min_avg_forward_velocity": None} == {
            "min_avg_reward": 100.0,
            "min_avg_episode_length": 750,
            "min_avg_forward_velocity": None,
            "min_success_rate": None,
            "min_eval_episodes": 10,
            "required_consecutive": 3,
        }
        assert stage_three == {
            "min_avg_reward": 100.0,
            "min_avg_episode_length": None,
            "min_avg_forward_velocity": None,
            "min_success_rate": 0.5,
            "min_eval_episodes": 10,
            "required_consecutive": 3,
        }

    assert species["velociraptor"]["stages"][1]["advancement_gate"]["min_avg_forward_velocity"] == 2.0
    assert species["trex"]["stages"][1]["advancement_gate"]["min_avg_forward_velocity"] == 2.0
    assert species["brachiosaurus"]["stages"][1]["advancement_gate"]["min_avg_forward_velocity"] == 0.75


def test_catalog_scopes_success_semantics_to_training_backends() -> None:
    catalog = build_catalog()
    species = {entry["id"]: entry for entry in catalog["species"]}

    velociraptor_metrics = species["velociraptor"]["success_metrics"]
    assert velociraptor_metrics[0]["backends"] == ["stable-baselines3"]
    assert "contacts the prey geom" in velociraptor_metrics[0]["definition"]
    assert velociraptor_metrics[1]["backends"] == ["jax-mjx"]
    assert "physical geom contact is not required" in velociraptor_metrics[1]["definition"]

    trex_metrics = species["trex"]["success_metrics"]
    assert trex_metrics[0]["backends"] == ["stable-baselines3"]
    assert trex_metrics[1]["backends"] == ["jax-mjx"]


def test_manifest_covers_all_curated_results_and_implemented_species() -> None:
    catalog = build_catalog()
    manifested_results = {
        result["summary_path"] for species in catalog["species"] for result in species["historical_results"]
    }
    curated_results = {
        path.relative_to(REPOSITORY_ROOT).as_posix() for path in (REPOSITORY_ROOT / "results").glob("*/*/summary.json")
    }
    assert manifested_results == curated_results

    manifested_species = {species["id"] for species in catalog["species"]}
    implemented_species = {
        path.name
        for path in (REPOSITORY_ROOT / "environments").iterdir()
        if path.is_dir() and (path / "envs").is_dir() and (path / "assets").is_dir()
    }
    assert manifested_species == implemented_species


def _published_summary() -> dict[str, Any]:
    summary_path = REPOSITORY_ROOT / "results" / "velociraptor" / "ppo" / "summary.json"
    return cast(dict[str, Any], json.loads(summary_path.read_text(encoding="utf-8")))


def test_result_validator_rejects_missing_public_metric() -> None:
    summary = deepcopy(_published_summary())
    del summary["stages"]["1"]["best_eval_reward"]

    with pytest.raises(CatalogError, match="best_eval_reward is required"):
        _validate_result_summary(
            summary,
            species_id="velociraptor",
            relative_path="results/velociraptor/ppo/summary.json",
        )


def test_result_validator_rejects_invalid_success_rate_and_step_totals() -> None:
    summary = deepcopy(_published_summary())
    summary["stages"]["3"]["mean_success_rate"] = 1.1
    with pytest.raises(CatalogError, match="must be between 0 and 1"):
        _validate_result_summary(
            summary,
            species_id="velociraptor",
            relative_path="results/velociraptor/ppo/summary.json",
        )

    summary = deepcopy(_published_summary())
    summary["total_timesteps"] += 1
    with pytest.raises(CatalogError, match="stage totals sum"):
        _validate_result_summary(
            summary,
            species_id="velociraptor",
            relative_path="results/velociraptor/ppo/summary.json",
        )


def test_max_reported_velocity_preserves_negative_values_and_missing_data() -> None:
    assert _max_reported_velocity([{"avg_forward_vel": -2.0}, {"avg_forward_vel": -0.5}]) == -0.5
    assert _max_reported_velocity([{"avg_forward_vel": None}]) is None


def test_committed_catalog_is_current() -> None:
    check_catalog()


def test_missing_manifest_path_is_rejected(tmp_path: Path) -> None:
    manifest_text = DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8")
    broken_manifest = tmp_path / "species_manifest.toml"
    broken_manifest.write_text(
        manifest_text.replace("notebooks/sb3_training.ipynb", "notebooks/missing_training.ipynb", 1),
        encoding="utf-8",
    )

    with pytest.raises(CatalogError, match="does not exist"):
        build_catalog(broken_manifest)


def test_public_notebook_references_are_manifested_and_exist() -> None:
    catalog = build_catalog()
    allowed_paths = {notebook["path"] for notebook in catalog["notebooks"]}
    public_documents = [
        REPOSITORY_ROOT / "README.md",
        *sorted((REPOSITORY_ROOT / "environments").glob("*/README.md")),
        *sorted((REPOSITORY_ROOT / "website" / "blog").rglob("*.md")),
        *sorted((REPOSITORY_ROOT / "website" / "docs").rglob("*.md")),
        *sorted((REPOSITORY_ROOT / "website" / "docs").rglob("*.mdx")),
        *sorted((REPOSITORY_ROOT / "docs").glob("*.md")),
    ]
    notebook_pattern = re.compile(r"notebooks/[A-Za-z0-9_./-]+\.ipynb")

    referenced_paths: set[str] = set()
    for document in public_documents:
        referenced_paths.update(notebook_pattern.findall(document.read_text(encoding="utf-8")))

    assert referenced_paths <= allowed_paths
    assert all((REPOSITORY_ROOT / path).is_file() for path in referenced_paths)


def test_training_notebooks_do_not_restore_stale_public_defaults() -> None:
    def notebook_text(name: str) -> str:
        notebook = json.loads((REPOSITORY_ROOT / "notebooks" / name).read_text(encoding="utf-8"))
        chunks: list[str] = []
        for cell in notebook["cells"]:
            source = cell.get("source", "")
            chunks.append("".join(source) if isinstance(source, list) else source)
        return "\n".join(chunks)

    sb3 = notebook_text("sb3_training.ipynb")
    assert '.get("timesteps",' not in sb3
    assert "GPU-Specific Recommended Settings" not in sb3

    ray = notebook_text("ray_tune_sweep.ipynb")
    assert "_settings_for_stage" in ray
    assert "TIMESTEPS_PER_TRIAL_OVERRIDE" in ray
    assert "configs/sweep_ppo.json" not in ray
    assert "configs/sweep_sac.json" not in ray

    jax = notebook_text("jax_training.ipynb")
    assert "10-100x" not in jax
    assert "A100 recommended" not in jax


@pytest.mark.parametrize(
    ("species_id", "website_constant"),
    [
        ("velociraptor", "VELOCIRAPTOR"),
        ("trex", "TREX"),
        ("brachiosaurus", "BRACHIOSAURUS"),
    ],
)
def test_public_model_pages_render_generated_catalog(species_id: str, website_constant: str) -> None:
    page = (REPOSITORY_ROOT / "website" / "docs" / "models" / f"{species_id}.mdx").read_text(encoding="utf-8")

    assert "import SpeciesCatalog from '@site/src/components/SpeciesCatalog';" in page
    assert f"import {{{website_constant}}} from '@site/src/data/species';" in page
    assert f"<SpeciesCatalog species={{{website_constant}}} />" in page


def test_default_paths_are_inside_repository() -> None:
    assert DEFAULT_MANIFEST_PATH.is_relative_to(REPOSITORY_ROOT)
    assert DEFAULT_OUTPUT_PATH.is_relative_to(REPOSITORY_ROOT)
    assert DEFAULT_README_PATH.is_relative_to(REPOSITORY_ROOT)
