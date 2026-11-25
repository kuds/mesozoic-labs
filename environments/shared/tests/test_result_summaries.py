"""Schema and provenance checks for curated result summaries.

``results/`` holds no summary since decision D-D59 deleted the four historical ladder summaries (three from March
2026 and the Brachiosaurus run of 2026-07-18) with the hunt nodes their last stage trained; the committed-file checks
below run over whatever ``results/`` holds (none today), and
the schema checks run on ``result_bundle_helpers.HISTORICAL_SUMMARY``, a complete schema-5 historical summary kept
as a fixture.  The reader accepts schema 5 alone: decision D-D63 retired schema 2, the integer-stage form those
summaries had.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from environments.shared.model_id import ModelId
from environments.shared.result_bundle import compare_summary_to_csv
from environments.shared.result_schema import (
    CANONICAL_RUNTIME_PROVENANCE_FIELDS,
    RESULT_SCHEMA_VERSION,
    ResultSchemaError,
    expected_result_directory,
    validate_result_path,
    validate_result_summary,
)

from .ci_workflow_helpers import glob_matches, path_filters
from .result_bundle_helpers import (
    REPOSITORY_ROOT,
    _canonical_summary,
    _load_summary,
    _plant_identity,
    _published_summary,
)

#: results/<species>/<embodiment>/<algorithm>/summary.json (decision D-D33).
RESULT_SUMMARIES = tuple(sorted((REPOSITORY_ROOT / "results").glob("*/*/*/summary.json")))
#: Every curated ``collected_results.csv``, wherever it sits under results/.
RESULT_CSVS = tuple(sorted((REPOSITORY_ROOT / "results").rglob("collected_results.csv")))


def test_result_summaries_are_discovered() -> None:
    """Protect against a vacuous schema test caused by a bad repository path: the glob runs over the real
    ``results/`` directory, and it finds exactly the summaries the species manifest lists (none since decision
    D-D59), so a summary filed anywhere the catalog reads is checked here."""
    import tomllib

    assert (REPOSITORY_ROOT / "results" / "README.md").is_file()
    with (REPOSITORY_ROOT / "configs" / "species_manifest.toml").open("rb") as handle:
        manifest = tomllib.load(handle)
    listed = sorted(
        REPOSITORY_ROOT / path
        for species in manifest["species"].values()
        for embodiment in species["embodiments"].values()
        for path in embodiment["result_summary_paths"]
    )
    assert list(RESULT_SUMMARIES) == listed


@pytest.mark.parametrize("summary_path", RESULT_SUMMARIES, ids=lambda path: str(path.relative_to(REPOSITORY_ROOT)))
def test_result_summary_schema_and_provenance(summary_path: Path) -> None:
    summary = _load_summary(summary_path)
    relative_path = summary_path.relative_to(REPOSITORY_ROOT)

    validated = validate_result_summary(
        summary,
        expected_model_id=ModelId(summary_path.parents[2].name, summary_path.parents[1].name),
        relative_path=relative_path,
        require_complete=True,
    )

    assert validated is summary


def test_every_collected_results_csv_has_a_sibling_summary() -> None:
    """A curated CSV is a derived view of the ``summary.json`` beside it; one without it has nothing to agree with."""
    orphans = [
        str(path.relative_to(REPOSITORY_ROOT)) for path in RESULT_CSVS if not path.with_name("summary.json").is_file()
    ]
    assert orphans == []


def test_every_summary_and_csv_triggers_python_ci() -> None:
    """A PR that edits only a summary or only a CSV must still run the agreement check below."""
    for path in (*RESULT_SUMMARIES, *RESULT_CSVS):
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        for patterns in path_filters():
            assert any(glob_matches(pattern, relative) for pattern in patterns), (
                f"{relative} is missing from a python-ci.yml paths filter, so a PR editing only it skips this check"
            )


@pytest.mark.parametrize(
    "summary_path",
    [path for path in RESULT_SUMMARIES if path.with_name("collected_results.csv").is_file()],
    ids=lambda path: str(path.relative_to(REPOSITORY_ROOT)),
)
def test_collected_results_csv_agrees_with_its_summary(summary_path: Path) -> None:
    """``summary.json`` is canonical (the catalog and website read only it), so the CSV beside it must not
    contradict it: same stages, same metrics, same verdicts and identity (results/README.md)."""
    summary = _load_summary(summary_path)
    assert compare_summary_to_csv(summary, summary_path.with_name("collected_results.csv")) == []


def test_a_summary_of_one_embodiment_is_refused_for_the_other() -> None:
    """Decision D-D33 (review H3): a summary, its path and its provenance name one model by both fields.

    The anatomical Compsognathus's summary is refused for the robot, filed under the robot's directory, and
    beside a provenance naming the other embodiment; a summary or provenance without its embodiment is refused,
    never matched on its species alone.  (Only the identity fields of the committed fixture are relabelled: every
    check below runs before the stages are read.)
    """
    anatomical = ModelId("compsognathus_longipes", "anatomical")
    robot = ModelId("compsognathus_longipes", "robot")
    summary = deepcopy(_published_summary())
    summary.update(anatomical.record())
    with pytest.raises(
        ResultSchemaError,
        match="result model mismatch in result summary: expected compsognathus_longipes/robot, "
        "found compsognathus_longipes/anatomical",
    ):
        validate_result_summary(summary, expected_model_id=robot, require_complete=False)
    with pytest.raises(
        ResultSchemaError,
        match="result model/path mismatch .*: expected compsognathus_longipes/robot, found compsognathus_longipes/anatomical",
    ):
        validate_result_path(
            "results/compsognathus_longipes/anatomical/ppo/summary.json",
            model_id=robot,
            algorithm="PPO",
            backend="stable-baselines3",
        )
    del summary["embodiment"]
    with pytest.raises(ResultSchemaError, match="result summary does not name a model.*embodiment"):
        validate_result_summary(summary, require_complete=False)

    # The canonical fixture recut as the robot's summary beside the anatomical model's provenance (each with
    # its own plant identity, so the summary-to-provenance comparison is what refuses it).
    canonical = _canonical_summary()
    canonical.update(robot.record())
    canonical["plant_identity"] = _plant_identity(robot)
    canonical["provenance"].update(anatomical.record())
    canonical["provenance"]["plant_identity"] = _plant_identity(anatomical)
    with pytest.raises(
        ResultSchemaError,
        match="does not match the provenance's: compsognathus_longipes/robot != compsognathus_longipes/anatomical",
    ):
        validate_result_summary(canonical, canonical_provenance=True)
    del canonical["provenance"]["embodiment"]
    with pytest.raises(ResultSchemaError, match="provenance .*(does not name a model|missing fields).*embodiment"):
        validate_result_summary(canonical, canonical_provenance=True)


def test_the_historical_fixture_is_a_complete_schema_5_summary() -> None:
    """``HISTORICAL_SUMMARY`` validates complete and publishable over its two certified deliverables."""
    summary = _published_summary()
    assert summary["schema_version"] == RESULT_SCHEMA_VERSION and sorted(summary["stages"]) == ["1", "2"]
    assert summary["total_timesteps"] == sum(stage["timesteps"] for stage in summary["stages"].values())
    for require_complete, require_publishable in ((True, False), (False, True)):
        validated = validate_result_summary(
            summary,
            expected_model_id=ModelId("velociraptor_mongoliensis", "anatomical"),
            relative_path="results/velociraptor_mongoliensis/anatomical/ppo/summary.json",
            require_complete=require_complete,
            require_publishable=require_publishable,
        )
        assert validated is summary


def test_a_summary_with_the_removed_hunt_stage_is_refused() -> None:
    """Legacy stage 3, the hunt, resolves to no stage since decision D-D59, so a summary recording it (the four
    deleted historical ladder summaries did) is refused."""
    summary = deepcopy(_published_summary())
    summary["stages"]["3"] = deepcopy(summary["stages"]["2"])
    summary["total_timesteps"] += summary["stages"]["3"]["timesteps"]
    with pytest.raises(ResultSchemaError, match="no stage with legacy number 3"):
        validate_result_summary(
            summary, expected_model_id=ModelId("velociraptor_mongoliensis", "anatomical"), require_complete=False
        )


def test_partial_summary_accepts_a_nonempty_stage_subset() -> None:
    summary = deepcopy(_published_summary())
    del summary["stages"]["2"]
    summary["total_timesteps"] = sum(stage["timesteps"] for stage in summary["stages"].values())
    summary["final_avg_reward"] = summary["stages"]["1"]["final_eval_reward"]

    validate_result_summary(
        summary,
        expected_model_id=ModelId("velociraptor_mongoliensis", "anatomical"),
        relative_path="results/velociraptor_mongoliensis/anatomical/ppo/summary.json",
        require_complete=False,
    )
    # Complete names the walk it certified, which the subset no longer records.
    with pytest.raises(ResultSchemaError, match="names stage '2', which the summary does not record"):
        validate_result_summary(
            summary,
            expected_model_id=ModelId("velociraptor_mongoliensis", "anatomical"),
            relative_path="results/velociraptor_mongoliensis/anatomical/ppo/summary.json",
            require_complete=True,
        )


def test_partial_summary_rejects_empty_or_unknown_stages() -> None:
    summary = deepcopy(_published_summary())
    summary["stages"] = {}
    summary["total_timesteps"] = 1
    with pytest.raises(ResultSchemaError, match="at least one stage"):
        validate_result_summary(summary, require_complete=False)

    summary = deepcopy(_published_summary())
    summary["stages"]["4"] = summary["stages"].pop("2")
    with pytest.raises(ResultSchemaError, match="invalid stage reference '4'"):
        validate_result_summary(summary, require_complete=False)


def _tyrannosaurus_rex_summary_with_recovery() -> dict[str, Any]:
    """The historical fixture as a T-Rex summary plus a synthetic recovery stage, at the writers' schema (v5; was v3
    before decision D-D33 retired it). The committed T-Rex summary it was built on until decision D-D59 is gone.
    Recovery has no deliverable record yet: the stage subset is all it is checked as until one is added."""
    summary = _published_summary()
    tyrannosaurus_rex = ModelId("tyrannosaurus_rex", "anatomical").record()
    summary.update(tyrannosaurus_rex)
    summary["provenance"].update(tyrannosaurus_rex)
    recovery_stage = deepcopy(summary["stages"]["1"])
    recovery_stage["name"] = "recovery"
    recovery_stage["description"] = "Hold the stance against scheduled pushes"
    recovery_stage["timesteps"] = 3_000_000
    # The pilot's honest verdict: gate_kind none/v1 refuses to pass.
    recovery_stage["stage_passed"] = False
    summary["stages"]["recovery"] = recovery_stage
    summary["total_timesteps"] += 3_000_000
    return summary


def test_the_semantic_recovery_stage_is_published_by_its_own_deliverable() -> None:
    """Recovery joins the summary as a stage subset, and publishing it takes its own deliverable record.

    Its recorded verdict may be False (the pilot's gate, none/v1, never passes): the summary is then publishable
    as ``partial``, its walk and stance certified, and never ``complete``, whose bundle certifies every deliverable it
    records.  (Schema 2's rule, every advancing stage present and passed with recovery riding along, left with the
    schema: decision D-D63.)
    """
    tyrannosaurus_rex = ModelId("tyrannosaurus_rex", "anatomical")
    relative_path = "results/tyrannosaurus_rex/anatomical/ppo/summary.json"
    summary = _tyrannosaurus_rex_summary_with_recovery()
    validate_result_summary(
        summary, expected_model_id=tyrannosaurus_rex, relative_path=relative_path, require_complete=False
    )
    with pytest.raises(ResultSchemaError, match=r"must record every deliverable stage .*missing \['recovery'\]"):
        validate_result_summary(summary, expected_model_id=tyrannosaurus_rex, require_complete=True)

    recovery = deepcopy(summary["provenance"]["deliverables"]["1"])
    recovery.update(model_path="recovery/models/best_model.zip", gate_kind="none/v1", certified=False)
    summary["provenance"]["deliverables"]["recovery"] = recovery
    with pytest.raises(ResultSchemaError, match="bundle_status in .* must be 'partial'"):
        validate_result_summary(summary, expected_model_id=tyrannosaurus_rex, require_complete=True)
    summary["bundle_status"] = "partial"
    validate_result_summary(
        summary,
        expected_model_id=tyrannosaurus_rex,
        relative_path=relative_path,
        require_complete=False,
        require_publishable=True,
    )
    with pytest.raises(ResultSchemaError, match="has bundle_status 'partial'"):
        validate_result_summary(summary, expected_model_id=tyrannosaurus_rex, require_complete=True)


def test_summary_rejects_stage_references_outside_the_species_vocabulary() -> None:
    # An id no manifest declares fails closed.
    summary = _tyrannosaurus_rex_summary_with_recovery()
    summary["stages"]["warp"] = deepcopy(summary["stages"]["recovery"])
    summary["total_timesteps"] += 3_000_000
    with pytest.raises(ResultSchemaError, match="invalid stage reference 'warp'"):
        validate_result_summary(
            summary, expected_model_id=ModelId("tyrannosaurus_rex", "anatomical"), require_complete=True
        )

    # A valid id the SPECIES does not declare fails closed too: only trex
    # carries a recovery stage today.
    summary = deepcopy(_published_summary())
    summary["stages"]["recovery"] = deepcopy(summary["stages"]["1"])
    with pytest.raises(ResultSchemaError, match="invalid stage reference 'recovery'"):
        validate_result_summary(
            summary, expected_model_id=ModelId("velociraptor_mongoliensis", "anatomical"), require_complete=True
        )


def test_summary_rejects_two_spellings_of_one_stage() -> None:
    """ "2" and "locomotion" are the same trex stage; recording both would
    double-count it in every aggregate."""
    summary = _tyrannosaurus_rex_summary_with_recovery()
    summary["stages"]["locomotion"] = deepcopy(summary["stages"]["2"])
    summary["total_timesteps"] += summary["stages"]["2"]["timesteps"]
    with pytest.raises(ResultSchemaError, match="references stage 'locomotion' twice"):
        validate_result_summary(
            summary, expected_model_id=ModelId("tyrannosaurus_rex", "anatomical"), require_complete=True
        )


def test_summary_accepts_schema_5_alone() -> None:
    """Re-pinned {2, 5} -> {5} (decision D-D63): every retired version is refused by name."""
    assert RESULT_SCHEMA_VERSION == 5
    velociraptor_mongoliensis = ModelId("velociraptor_mongoliensis", "anatomical")
    summary = deepcopy(_published_summary())
    validate_result_summary(summary, expected_model_id=velociraptor_mongoliensis, require_complete=True)
    for version in (3, 4):
        summary["schema_version"] = version
        with pytest.raises(
            ResultSchemaError,
            match=rf"must be one of \[5\], not {version}; schema 3 and 4 summaries name species IDs decision "
            "D-D33 retired",
        ):
            validate_result_summary(summary, expected_model_id=velociraptor_mongoliensis, require_complete=False)
    unsupported: tuple[Any, ...] = (1, 6, "5", 5.0, True, None)
    for version in unsupported:
        summary["schema_version"] = version
        with pytest.raises(ResultSchemaError, match=rf"schema_version must be one of \[5\], not {version!r}: "):
            validate_result_summary(summary, expected_model_id=velociraptor_mongoliensis, require_complete=False)


def test_a_schema_2_summary_is_refused_with_where_to_read_it() -> None:
    """Decision D-D63 retired schema 2, the integer-stage form of the summaries committed until decision D-D59: one
    is refused, whatever else it records, with the commit that holds them and still reads them (a later tree that
    still declares 0.4.0.dev2 refuses their stage 3, so the message names the commit, not the version)."""
    summary = deepcopy(_published_summary())
    summary["schema_version"] = 2
    for require_complete, canonical_provenance in ((True, False), (False, False), (False, True)):
        with pytest.raises(ResultSchemaError) as refused:
            validate_result_summary(
                summary, require_complete=require_complete, canonical_provenance=canonical_provenance
            )
        message = str(refused.value)
        assert message.startswith("result summary schema_version must be one of [5], not 2; schema 2 is the ")
        for part in (
            "integer-stage form of the result summaries committed under results/ until decision D-D59",
            "git show 6c9982d81c8a:results/<path>",
            "retired by decision D-D63",
            "read it with the tree at commit 6c9982d81c8a (version 0.4.0.dev2, before decision D-D59 removed the "
            "stage 3 those summaries record)",
        ):
            assert part in message, part
    # The species catalog refuses it with the same message, through its own error type.
    from environments.shared.species_catalog import CatalogError, _validate_result_summary

    with pytest.raises(CatalogError, match="not 2; schema 2 is the integer-stage form"):
        _validate_result_summary(
            summary,
            model_id=ModelId("velociraptor_mongoliensis", "anatomical"),
            relative_path="results/velociraptor_mongoliensis/anatomical/ppo/summary.json",
        )


def test_canonical_selected_checkpoints_carry_every_deliverable() -> None:
    summary = _canonical_summary()
    del summary["provenance"]["selected_checkpoints"]["2"]
    with pytest.raises(ResultSchemaError, match="deliverable 2 has no selected checkpoint"):
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


@pytest.mark.parametrize("field", CANONICAL_RUNTIME_PROVENANCE_FIELDS)
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


def test_canonical_provenance_rejects_a_publication_seed_shared_with_training() -> None:
    summary = _canonical_summary()
    training_seed = summary["seed"]
    provenance = summary["provenance"]
    provenance["seed_roles"] = {"training": training_seed, "publication_evaluation": training_seed}
    provenance["evaluation_seeds"] = [training_seed]
    provenance["evaluation_protocols"] = {
        "publication_evaluation": {"seed": training_seed, "episodes": 30, "deterministic": True}
    }

    with pytest.raises(ResultSchemaError, match=rf"seed_roles.*reuses the training seed {training_seed}"):
        validate_result_summary(summary, canonical_provenance=True)


def test_canonical_provenance_rejects_a_publication_seed_shared_with_selection() -> None:
    summary = _canonical_summary()
    provenance = summary["provenance"]
    provenance["seed_roles"]["checkpoint_selection_evaluation"] = 3042
    provenance["evaluation_protocols"]["checkpoint_selection_evaluation"] = {
        "seed": 3042,
        "episodes": 30,
        "deterministic": True,
    }

    with pytest.raises(ResultSchemaError, match=r"reuses the checkpoint_selection_evaluation seed 3042"):
        validate_result_summary(summary, canonical_provenance=True)


def test_canonical_provenance_accepts_the_certification_panel_role_without_a_protocol() -> None:
    """Decision D-B17: the panel role may equal the publication seed and gets no evaluation protocol —
    it is bound per evidence file by the audit; a protocol entry for it is refused as before."""
    summary = _canonical_summary()
    provenance = summary["provenance"]
    provenance["seed_roles"]["certification_panel"] = 3042
    assert provenance["seed_roles"]["publication_evaluation"] == 3042

    validate_result_summary(summary, canonical_provenance=True)

    provenance["evaluation_protocols"]["certification_panel"] = {"seed": 3042, "episodes": 40, "deterministic": True}
    with pytest.raises(ResultSchemaError, match=r"evaluation_protocols .* must exactly match evaluation seed roles"):
        validate_result_summary(summary, canonical_provenance=True)


def test_canonical_provenance_accepts_distinct_seed_roles() -> None:
    summary = _canonical_summary()
    provenance = summary["provenance"]
    provenance["seed_roles"]["checkpoint_selection_evaluation"] = 1042
    provenance["evaluation_seeds"] = [3042, 1042]
    provenance["evaluation_protocols"]["checkpoint_selection_evaluation"] = {
        "seed": 1042,
        "episodes": 30,
        "deterministic": True,
    }

    validate_result_summary(summary, canonical_provenance=True)


def test_canonical_summary_cannot_publish_a_failed_stage_as_certified() -> None:
    summary = _canonical_summary()
    summary["stages"]["2"]["stage_passed"] = False
    summary["stages"]["2"]["publication_gate_passed"] = False

    with pytest.raises(ResultSchemaError, match="deliverables.2 .* recorded as certified, but its recorded verdicts"):
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
        expected_model_id=ModelId("velociraptor_mongoliensis", "anatomical"),
        relative_path="results/velociraptor_mongoliensis/anatomical/jax_ppo/summary.json",
    )

    with pytest.raises(ResultSchemaError, match=r"expected jax_ppo for jax-mjx"):
        validate_result_summary(
            summary,
            expected_model_id=ModelId("velociraptor_mongoliensis", "anatomical"),
            relative_path="results/velociraptor_mongoliensis/anatomical/ppo/summary.json",
        )


def test_result_summary_rejects_an_unknown_backend() -> None:
    summary = deepcopy(_published_summary())
    summary["backend"] = "jax"

    with pytest.raises(ResultSchemaError, match="invalid backend"):
        validate_result_summary(summary)
