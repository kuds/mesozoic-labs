"""Schema and provenance checks for curated result summaries."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest

from environments.shared.result_schema import (
    ResultSchemaError,
    expected_result_directory,
    validate_result_summary,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RESULT_SUMMARIES = tuple(sorted((REPOSITORY_ROOT / "results").glob("*/*/summary.json")))


def _load_summary(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as summary_file:
        return cast(dict[str, Any], json.load(summary_file))


def _published_summary() -> dict[str, Any]:
    return _load_summary(REPOSITORY_ROOT / "results" / "velociraptor" / "ppo" / "summary.json")


def _canonical_plant_identity() -> dict[str, Any]:
    return {
        "schema": "mesozoic.plant-identity/v1",
        "species": "velociraptor",
        "model_path": "environments/velociraptor/assets/raptor.xml",
        "physics_revision": 1,
        "policy_interface_revision": 1,
        "visual_revision": 1,
        "source_closure_sha256": "sha256:" + "1" * 64,
        "policy_interface_sha256": "sha256:" + "2" * 64,
        "physics_sha256": "sha256:" + "3" * 64,
        "visual_sha256": "sha256:" + "4" * 64,
        "nq": 31,
        "nv": 30,
        "nu": 22,
        "observation_dim": 67,
        "action_dim": 22,
    }


def _canonical_summary() -> dict[str, Any]:
    summary = deepcopy(_published_summary())
    plant_identity = _canonical_plant_identity()
    summary.update(
        {
            "bundle_status": "complete",
            "run_id": "velociraptor-stable-baselines3-ppo-test",
            "backend_version": "2.7.0",
            "plant_identity": plant_identity,
        }
    )
    summary["provenance"].update(
        {
            "evaluation_episodes": 30,
            "repository_commit": "a" * 40,
            "model_hash": "sha256:" + "b" * 64,
            "config_hash": "sha256:" + "c" * 64,
            "run_id": summary["run_id"],
            "species": summary["species"],
            "algorithm": summary["algorithm"],
            "backend": summary["backend"],
            "backend_version": summary["backend_version"],
            "captured_at": f"{summary['date']}T12:00:00+00:00",
            "repository_dirty": False,
            "repository_patch_sha256": None,
            "training_seed": summary["seed"],
            "seed_roles": {
                "training": summary["seed"],
                "publication_evaluation": 3042,
            },
            "evaluation_protocols": {
                "publication_evaluation": {
                    "seed": 3042,
                    "episodes": 30,
                    "deterministic": True,
                }
            },
            "evaluation_seeds": [3042],
            "parallel_envs": summary["parallel_envs"],
            "hardware": summary["hardware"],
            "python_version": "3.12.11",
            "platform": "Linux-6.8.0-x86_64",
            "dependency_versions": {
                "mujoco": "3.10.0",
                "stable_baselines3": "2.7.0",
            },
            "plant_identity": plant_identity,
            "selected_checkpoints": {
                str(stage): {
                    "model_path": f"stage{stage}/models/best_model.zip",
                    "model_hash": "sha256:" + ("b" if stage == 3 else "e") * 64,
                    "normalization_path": f"stage{stage}/models/best_model_vecnorm.pkl",
                    "normalization_hash": "sha256:" + "f" * 64,
                }
                for stage in (1, 2, 3)
            },
            "selected_model_path": "stage3/models/best_model.zip",
        }
    )
    for stage in summary["stages"].values():
        stage.setdefault("mean_distance_traveled", 0.0)
        stage.setdefault("mean_success_rate", 0.0)
        stage["publication_gate_passed"] = stage["stage_passed"]
        stage.update(
            {
                "selected_model_reward": stage["final_eval_reward"],
                "selected_model_reward_std": stage.get("final_eval_std", 0.0),
                "selected_model_episode_length": stage["avg_episode_length"],
                "selected_model_episode_length_std": stage.get("avg_episode_length_std", 0.0),
                "selected_model_forward_vel": stage["avg_forward_vel"],
                "selected_model_forward_vel_std": stage.get("avg_forward_vel_std", 0.0),
                "selected_model_distance": stage.get("mean_distance_traveled", 0.0),
                "selected_model_success_rate": stage.get("mean_success_rate", 0.0),
            }
        )
    return summary


def test_result_summaries_are_discovered() -> None:
    """Protect against a vacuous schema test caused by a bad repository path."""
    assert RESULT_SUMMARIES, "no results/*/*/summary.json files were found"


@pytest.mark.parametrize("summary_path", RESULT_SUMMARIES, ids=lambda path: str(path.relative_to(REPOSITORY_ROOT)))
def test_result_summary_schema_and_provenance(summary_path: Path) -> None:
    summary = _load_summary(summary_path)
    relative_path = summary_path.relative_to(REPOSITORY_ROOT)

    validated = validate_result_summary(
        summary,
        expected_species=summary_path.parents[1].name,
        relative_path=relative_path,
        require_complete=True,
    )

    assert validated is summary


def test_partial_summary_accepts_a_nonempty_stage_subset() -> None:
    summary = deepcopy(_published_summary())
    del summary["stages"]["3"]
    summary["total_timesteps"] = sum(stage["timesteps"] for stage in summary["stages"].values())
    summary["final_avg_reward"] = summary["stages"]["2"]["final_eval_reward"]

    validate_result_summary(
        summary,
        expected_species="velociraptor",
        relative_path="results/velociraptor/ppo/summary.json",
        require_complete=False,
    )
    with pytest.raises(ResultSchemaError, match="must contain every advancing stage"):
        validate_result_summary(
            summary,
            expected_species="velociraptor",
            relative_path="results/velociraptor/ppo/summary.json",
            require_complete=True,
        )


def test_partial_summary_rejects_empty_or_unknown_stages() -> None:
    summary = deepcopy(_published_summary())
    summary["stages"] = {}
    summary["total_timesteps"] = 1
    with pytest.raises(ResultSchemaError, match="at least one stage"):
        validate_result_summary(summary, require_complete=False)

    summary = deepcopy(_published_summary())
    summary["stages"]["4"] = summary["stages"].pop("3")
    with pytest.raises(ResultSchemaError, match="invalid stage reference '4'"):
        validate_result_summary(summary, require_complete=False)


def _trex_summary_with_recovery() -> dict[str, Any]:
    """The committed trex summary plus a synthetic recovery stage (schema v3)."""
    summary = _load_summary(REPOSITORY_ROOT / "results" / "trex" / "ppo" / "summary.json")
    summary = deepcopy(summary)
    summary["schema_version"] = 3
    recovery_stage = deepcopy(summary["stages"]["1"])
    recovery_stage["name"] = "recovery"
    recovery_stage["description"] = "Hold the stance against scheduled pushes"
    recovery_stage["timesteps"] = 3_000_000
    # The pilot's honest verdict: gate_kind none/v1 refuses to pass.
    recovery_stage["stage_passed"] = False
    summary["stages"]["recovery"] = recovery_stage
    summary["total_timesteps"] += 3_000_000
    return summary


def test_complete_summary_accepts_the_semantic_recovery_stage() -> None:
    """Recovery joins the summary without weakening the advancing-stage rule.

    The v2 ``exactly {1, 2, 3}`` equality enforced "a complete curriculum
    recorded every advancing stage".  v3 keeps that rule bit-for-bit for the
    advancing trio while the non-advancing recovery stage rides along — and
    its recorded verdict may be False, because a non-advancing pilot's
    failure must not be laundered into (or block) the curriculum's own
    completeness.
    """
    summary = _trex_summary_with_recovery()
    validate_result_summary(
        summary,
        expected_species="trex",
        relative_path="results/trex/ppo/summary.json",
        require_complete=True,
    )

    # Still fail-closed on the advancing trio: recovery cannot substitute
    # for a missing numbered stage.
    summary = _trex_summary_with_recovery()
    summary["total_timesteps"] -= summary["stages"]["2"]["timesteps"]
    del summary["stages"]["2"]
    with pytest.raises(ResultSchemaError, match=r"must contain every advancing stage; missing \['2'\]"):
        validate_result_summary(summary, expected_species="trex", require_complete=True)


def test_summary_rejects_stage_references_outside_the_species_vocabulary() -> None:
    # An id no manifest declares fails closed.
    summary = _trex_summary_with_recovery()
    summary["stages"]["warp"] = deepcopy(summary["stages"]["recovery"])
    summary["total_timesteps"] += 3_000_000
    with pytest.raises(ResultSchemaError, match="invalid stage reference 'warp'"):
        validate_result_summary(summary, expected_species="trex", require_complete=True)

    # A valid id the SPECIES does not declare fails closed too: only trex
    # carries a recovery stage today.
    summary = deepcopy(_published_summary())
    summary["stages"]["recovery"] = deepcopy(summary["stages"]["1"])
    with pytest.raises(ResultSchemaError, match="invalid stage reference 'recovery'"):
        validate_result_summary(summary, expected_species="velociraptor", require_complete=True)


def test_summary_rejects_two_spellings_of_one_stage() -> None:
    """ "2" and "locomotion" are the same trex stage; recording both would
    double-count it in every aggregate."""
    summary = _trex_summary_with_recovery()
    summary["stages"]["locomotion"] = deepcopy(summary["stages"]["2"])
    summary["total_timesteps"] += summary["stages"]["2"]["timesteps"]
    with pytest.raises(ResultSchemaError, match="references stage 'locomotion' twice"):
        validate_result_summary(summary, expected_species="trex", require_complete=True)


def test_summary_accepts_both_supported_schema_versions() -> None:
    summary = deepcopy(_published_summary())
    for version in (2, 3):
        summary["schema_version"] = version
        validate_result_summary(summary, expected_species="velociraptor", require_complete=True)
    summary["schema_version"] = 1
    with pytest.raises(ResultSchemaError, match=r"schema_version must be one of \[2, 3\]"):
        validate_result_summary(summary, expected_species="velociraptor", require_complete=True)


def test_canonical_selected_checkpoints_require_every_advancing_stage() -> None:
    summary = _canonical_summary()
    del summary["provenance"]["selected_checkpoints"]["2"]
    with pytest.raises(ResultSchemaError, match=r"checkpoint for every advancing stage; missing \['2'\]"):
        validate_result_summary(summary, canonical_provenance=True)

    # And the keys stay inside the species' vocabulary: velociraptor
    # declares no recovery stage, so the key fails closed.
    summary = _canonical_summary()
    summary["provenance"]["selected_checkpoints"]["recovery"] = deepcopy(
        summary["provenance"]["selected_checkpoints"]["1"]
    )
    with pytest.raises(ResultSchemaError, match="invalid stage reference 'recovery'"):
        validate_result_summary(summary, canonical_provenance=True)


def test_canonical_provenance_requires_complete_well_formed_identity() -> None:
    summary = _canonical_summary()
    summary["provenance"]["model_hash"] = None

    with pytest.raises(ResultSchemaError, match="canonical result.*missing identifiers"):
        validate_result_summary(summary, canonical_provenance=True)

    summary = _canonical_summary()
    validate_result_summary(summary, canonical_provenance=True)

    summary["provenance"]["model_hash"] = "sha256:short"
    with pytest.raises(ResultSchemaError, match=r"model_hash.*sha256:<64 lowercase hex>"):
        validate_result_summary(summary, canonical_provenance=True)


def test_canonical_provenance_requires_backend_version() -> None:
    summary = _canonical_summary()
    summary["backend_version"] = None

    with pytest.raises(ResultSchemaError, match="canonical result.*missing backend_version"):
        validate_result_summary(summary, canonical_provenance=True)


@pytest.mark.parametrize(
    "field",
    [
        "run_id",
        "captured_at",
        "species",
        "algorithm",
        "backend",
        "backend_version",
        "repository_dirty",
        "repository_patch_sha256",
        "training_seed",
        "seed_roles",
        "evaluation_protocols",
        "evaluation_seeds",
        "parallel_envs",
        "hardware",
        "python_version",
        "platform",
        "dependency_versions",
        "plant_identity",
        "selected_checkpoints",
        "selected_model_path",
    ],
)
def test_canonical_provenance_requires_runtime_fields(field: str) -> None:
    summary = _canonical_summary()
    del summary["provenance"][field]

    with pytest.raises(ResultSchemaError, match=rf"canonical provenance.*missing fields.*{field}"):
        validate_result_summary(summary, canonical_provenance=True)


def test_canonical_provenance_requires_matching_top_level_run_identity() -> None:
    summary = _canonical_summary()
    summary["provenance"]["run_id"] = ""
    with pytest.raises(ResultSchemaError, match=r"provenance\.run_id.*non-empty string"):
        validate_result_summary(summary, canonical_provenance=True)

    summary = _canonical_summary()
    del summary["run_id"]
    with pytest.raises(ResultSchemaError, match="run_id.*non-empty string"):
        validate_result_summary(summary, canonical_provenance=True)

    summary = _canonical_summary()
    summary["run_id"] = "different-run"
    with pytest.raises(ResultSchemaError, match=r"does not match provenance\.run_id"):
        validate_result_summary(summary, canonical_provenance=True)

    summary = _canonical_summary()
    summary["seed"] = 43
    with pytest.raises(ResultSchemaError, match=r"does not match provenance\.training_seed"):
        validate_result_summary(summary, canonical_provenance=True)

    summary = _canonical_summary()
    summary["provenance"]["training_seed"] = -1
    with pytest.raises(ResultSchemaError, match=r"training_seed.*non-negative integer"):
        validate_result_summary(summary, canonical_provenance=True)


def test_canonical_provenance_requires_repository_state() -> None:
    summary = _canonical_summary()
    summary["provenance"]["repository_dirty"] = "false"
    with pytest.raises(ResultSchemaError, match=r"repository_dirty.*must be a boolean"):
        validate_result_summary(summary, canonical_provenance=True)

    summary = _canonical_summary()
    summary["provenance"]["repository_dirty"] = True
    summary["provenance"]["repository_patch_sha256"] = "sha256:" + "d" * 64
    with pytest.raises(ResultSchemaError, match=r"requires a clean repository"):
        validate_result_summary(summary, canonical_provenance=True)

    summary = _canonical_summary()
    summary["provenance"]["repository_patch_sha256"] = "sha256:" + "d" * 64
    with pytest.raises(ResultSchemaError, match=r"must be null for a clean repository"):
        validate_result_summary(summary, canonical_provenance=True)


@pytest.mark.parametrize(
    "captured_at",
    ["not-a-timestamp", "2026-07-18T12:00:00"],
)
def test_canonical_provenance_requires_an_offset_capture_time(captured_at: str) -> None:
    summary = _canonical_summary()
    summary["provenance"]["captured_at"] = captured_at

    with pytest.raises(ResultSchemaError, match=r"captured_at"):
        validate_result_summary(summary, canonical_provenance=True)


def test_canonical_summary_date_matches_capture_time() -> None:
    summary = _canonical_summary()
    summary["date"] = "2000-01-01"

    with pytest.raises(ResultSchemaError, match=r"does not match provenance\.captured_at"):
        validate_result_summary(summary, canonical_provenance=True)


@pytest.mark.parametrize("evaluation_seeds", [[], [-1], [True], ["3042"], [3042, 3042]])
def test_canonical_provenance_requires_valid_evaluation_seeds(evaluation_seeds: list[Any]) -> None:
    summary = _canonical_summary()
    summary["provenance"]["evaluation_seeds"] = evaluation_seeds

    with pytest.raises(ResultSchemaError, match=r"evaluation_seeds.*(?:non-empty|non-negative|unique)"):
        validate_result_summary(summary, canonical_provenance=True)


def test_canonical_provenance_rejects_an_unlisted_evaluation_role_seed() -> None:
    summary = _canonical_summary()
    summary["provenance"]["seed_roles"]["checkpoint_selection_evaluation"] = 999
    summary["provenance"]["evaluation_protocols"]["checkpoint_selection_evaluation"] = {
        "seed": 999,
        "episodes": 30,
        "deterministic": True,
    }

    with pytest.raises(ResultSchemaError, match=r"evaluation_seeds.*exactly match"):
        validate_result_summary(summary, canonical_provenance=True)


@pytest.mark.parametrize(
    "seed_roles",
    [
        {},
        {"training": 43},
        {"training": 42},
        {"training": 42, "publication_evaluation": 999},
        {"training": 42, "network": -1},
    ],
)
def test_canonical_provenance_requires_consistent_seed_roles(
    seed_roles: dict[str, Any],
) -> None:
    summary = _canonical_summary()
    summary["provenance"]["seed_roles"] = seed_roles

    with pytest.raises(ResultSchemaError, match=r"seed_roles"):
        validate_result_summary(summary, canonical_provenance=True)


def test_canonical_summary_requires_every_curriculum_gate_to_pass() -> None:
    summary = _canonical_summary()
    summary["stages"]["2"]["stage_passed"] = False
    summary["stages"]["2"]["publication_gate_passed"] = False

    with pytest.raises(ResultSchemaError, match=r"stage_passed.*must be true"):
        validate_result_summary(summary, canonical_provenance=True)


@pytest.mark.parametrize(
    "field",
    [
        "selected_model_reward",
        "selected_model_reward_std",
        "selected_model_episode_length",
        "selected_model_episode_length_std",
        "selected_model_forward_vel",
        "selected_model_forward_vel_std",
        "selected_model_distance",
        "selected_model_success_rate",
    ],
)
def test_canonical_summary_requires_selected_checkpoint_metrics(field: str) -> None:
    summary = _canonical_summary()
    del summary["stages"]["1"][field]

    with pytest.raises(ResultSchemaError, match=rf"{field} is required"):
        validate_result_summary(summary, canonical_provenance=True)


def test_canonical_summary_requires_a_relative_selected_model_path() -> None:
    summary = _canonical_summary()
    summary["provenance"]["selected_model_path"] = "/content/drive/best_model.zip"

    with pytest.raises(ResultSchemaError, match=r"selected_model_path.*relative POSIX path"):
        validate_result_summary(summary, canonical_provenance=True)


def test_canonical_provenance_requires_runtime_versions() -> None:
    summary = _canonical_summary()
    summary["provenance"]["python_version"] = ""
    with pytest.raises(ResultSchemaError, match=r"python_version.*non-empty string"):
        validate_result_summary(summary, canonical_provenance=True)

    summary = _canonical_summary()
    summary["provenance"]["dependency_versions"] = []
    with pytest.raises(ResultSchemaError, match=r"dependency_versions.*must be an object"):
        validate_result_summary(summary, canonical_provenance=True)


def test_canonical_provenance_requires_top_level_plant_identity() -> None:
    summary = _canonical_summary()
    del summary["plant_identity"]

    with pytest.raises(ResultSchemaError, match=r"plant_identity.*must be an object"):
        validate_result_summary(summary, canonical_provenance=True)


def test_result_paths_are_backend_aware() -> None:
    summary = deepcopy(_published_summary())
    assert expected_result_directory("PPO", "stable-baselines3") == "ppo"
    assert expected_result_directory("PPO", "jax-mjx") == "jax_ppo"
    assert expected_result_directory("JAX_PPO", "jax-mjx") == "jax_ppo"

    summary["backend"] = "jax-mjx"
    validate_result_summary(
        summary,
        expected_species="velociraptor",
        relative_path="results/velociraptor/jax_ppo/summary.json",
    )

    with pytest.raises(ResultSchemaError, match=r"expected jax_ppo for jax-mjx"):
        validate_result_summary(
            summary,
            expected_species="velociraptor",
            relative_path="results/velociraptor/ppo/summary.json",
        )


def test_result_summary_rejects_an_unknown_backend() -> None:
    summary = deepcopy(_published_summary())
    summary["backend"] = "jax"

    with pytest.raises(ResultSchemaError, match="invalid backend"):
        validate_result_summary(summary)
