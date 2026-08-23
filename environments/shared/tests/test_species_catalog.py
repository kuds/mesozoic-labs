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
    DEFAULT_PLANT_MANIFEST_PATH,
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
    assert catalog["schema_version"] == 2
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
        "trex": (61, 15, 28, 27, 15, 85.72),
        "brachiosaurus": (83, 30, 38, 37, 30, 175.3),
        "dibothrosuchus": (77, 27, 35, 34, 27, 8.65),
    }

    assert [stage["timesteps"] for stage in species["velociraptor"]["stages"]] == [6_000_000, 8_000_000, 12_000_000]
    # Stage 1 is 11M, not the 6M every other species uses: trex 1a is the
    # stance-gated stage, and its 6M budget ran out mid-improvement -- run
    # 20260802_203215's best evaluation was its last, at 6.0M of 6M. Raised
    # 10M -> 11M on main (08a66b3) for the seeded replicate campaign.
    assert [stage["timesteps"] for stage in species["trex"]["stages"]] == [11_000_000, 8_000_000, 8_000_000]
    assert [stage["timesteps"] for stage in species["brachiosaurus"]["stages"]] == [
        6_000_000,
        16_000_000,
        12_000_000,
    ]
    assert [stage["timesteps"] for stage in species["dibothrosuchus"]["stages"]] == [
        6_000_000,
        12_000_000,
        8_000_000,
    ]


def test_catalog_publishes_layered_plant_contract() -> None:
    catalog = build_catalog()

    assert catalog["plant_manifest"] == {
        "path": "configs/plant_manifest.generated.json",
        "schema": "mesozoic.plant-manifest/v1",
        "fingerprint_tool_version": 2,
        "generated_with": {"mujoco": "3.10.0", "float_significant_digits": 12},
    }
    expected_observation_schemas = {
        "velociraptor": "bipedal-target/v1",
        "trex": "bipedal-target/v1",
        "brachiosaurus": "quadrupedal-target/v1",
        "dibothrosuchus": "quadrupedal-target/v1",
    }
    # The T-Rex is at physics r5 for the theropod stance correction: the home
    # keyframe moved off a near-straight knee onto a 135 deg one and the leg
    # ctrlranges were re-centred on it, so both the compiled dynamics and the
    # control interface changed. Its visual revision is deliberately NOT bumped
    # -- the visual layer fingerprints geom/site/material/camera definitions,
    # all of which are body-local and unchanged by a pose edit.
    #
    # The three home-keyframe-residual species then took one more policy bump
    # for the bounded reset height (plant_versions note 5); brachiosaurus does
    # not carry home_reset and so stayed at r3 for that one.
    # Ground settling at reset (note 6) then bumped ALL FOUR, brachiosaurus
    # included: settling applies to every species regardless of keyframe style.
    # Brachiosaurus then took a physics bump (leg servo kp doubled so the
    # animal can statically carry its own weight) and a policy bump (migrated
    # to the home-keyframe-residual mapping the other species use) for the
    # stance repair in plant_versions note 7, and one more bump of each for
    # the foot-sensor repair in note 8 (pad sites enlarged, meta sensors
    # appended, pad + meta summed per leg on both backends; the physics layer
    # fingerprints nsite/nsensor, so new sites move it even though dynamics
    # are unchanged).
    # The perturbation engine (note 11) bumped every policy revision once
    # more: reset/step learned to derive and apply push schedules, and the
    # interface hash covers reset's source through home_reset on all four
    # species. Physics and visual are untouched — no MJCF edit, and with
    # perturbation off (every stage but recovery) the episode is
    # bit-identical to the previous plant.
    expected_policy_revisions = {"velociraptor": 9, "trex": 12, "brachiosaurus": 7, "dibothrosuchus": 6}
    expected_physics_revisions = {"velociraptor": 2, "trex": 7, "brachiosaurus": 4, "dibothrosuchus": 1}
    expected_visual_revisions = {"velociraptor": 3, "trex": 4, "brachiosaurus": 2, "dibothrosuchus": 1}
    digest_pattern = re.compile(r"sha256:[0-9a-f]{64}")
    for species in catalog["species"]:
        plant = species["model"]["plant_contract"]
        assert digest_pattern.fullmatch(plant["bundle_sha256"])
        assert digest_pattern.fullmatch(plant["source_closure_sha256"])
        assert plant["policy_interface"]["revision"] == expected_policy_revisions[species["id"]]
        assert plant["policy_interface"]["observation_schema"] == expected_observation_schemas[species["id"]]
        assert plant["physics"]["revision"] == expected_physics_revisions[species["id"]]
        assert plant["visual"]["revision"] == expected_visual_revisions[species["id"]]
        for layer in ("policy_interface", "physics", "visual"):
            assert digest_pattern.fullmatch(plant[layer]["sha256"])


def test_catalog_keeps_current_configs_separate_from_historical_results() -> None:
    catalog = build_catalog()
    brachiosaurus = next(entry for entry in catalog["species"] if entry["id"] == "brachiosaurus")
    current_stage_three = brachiosaurus["stages"][2]
    published_stage_three = brachiosaurus["historical_results"][0]["stages"][2]

    assert current_stage_three["timesteps"] == 12_000_000
    assert published_stage_three["timesteps"] == 12_009_472
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

    # Stage-1 reward gates are COLLAPSE RAILS: 0.60 x each species' zero-action
    # statue standing reward at the 1a operating point (reset noise 0.05),
    # measured over 40 episodes with
    # environments/shared/scripts/stance_quality_baseline.py -- trex 3495.2
    # (physics r7 with the 20260810 shaping pack: tail_home_pose 0.25 at the
    # settled-droop targets, action_saturation 0.5, leg broad fraction 0.25;
    # 3241.3 before the pack, 3270.3 at tolerance 0.20, 3271.8 on the r6
    # plant), velociraptor 1745.8, brachiosaurus 1739.1 (on
    # the plant repaired by plant_versions notes 7-8), dibothrosuchus 2598.3.
    #
    # A rail sits BELOW its statue deliberately. Section 9 showed the statue
    # is the reward optimum -- it collects 97.0% of the theoretical maximum
    # while paying zero energy and smoothness cost -- so a threshold above it
    # is unreachable and one below it is clearable by a statue. The rail's
    # only job is to reject a policy that has discarded most of the available
    # return; what separates competent from passive is the episode-level
    # stance_success gate, which is not built yet.
    #
    # 0.60 rather than section 12's 0.89: the measured collapse bottomed at
    # 0.27 x statue, so 0.60 clears it by better than 2x, while 0.89 sat within
    # ~2.4% of a competent policy's estimated ceiling and risked rejecting the
    # very policy it was meant to admit. The previous values (trex 1840,
    # everyone else the 100.0 placeholder) were all cleared by their own
    # species' statue and certified nothing.
    stage_one_min_avg_reward = {
        # 0.60 x the statue's standing reward at leg_home_pose_weight 0.5.
        # Was briefly 2550 while that weight was 1.5 (statue 4250.4), reverted
        # with it in issue #491. The FRACTION is the invariant, not the reward.
        # Re-derived at tolerance 0.10 (statue 3241.3 -> rail 1940), then
        # again for the 20260810 shaping pack: statue 3495.2, x0.60 = 2097.1
        # -> 2100 nearest-10.
        "trex": 2100.0,
        "velociraptor": 1050.0,
        "brachiosaurus": 1040.0,
        "dibothrosuchus": 1560.0,
    }

    # T-Rex 1a has moved to stance_quality/v1, which does not consume
    # min_avg_episode_length: min_full_horizon_fraction states section 12's
    # >= 95% requirement directly instead of encoding it as a step count. Its
    # min_avg_reward stays, demoted to a rail. The other three species are
    # still on reward_and_length/v1 pending their own stance calibration.
    stage_one_length = {"trex": None, "velociraptor": 950, "brachiosaurus": 950, "dibothrosuchus": 950}
    # The stance bound's power is specified at n=40; the other species keep
    # the historical default.
    stage_one_eval_episodes = {"trex": 40, "velociraptor": 10, "brachiosaurus": 10, "dibothrosuchus": 10}
    # Only T-Rex 1a declares stance criteria; the rest export nulls.
    stance_null: dict[str, float | None] = {
        "min_full_horizon_fraction": None,
        "max_unsupported_duty": None,
        "max_unsupported_duty_ucb": None,
    }
    stage_one_stance: dict[str, dict[str, float | None]] = {
        "trex": {
            "min_full_horizon_fraction": 0.95,
            "max_unsupported_duty": 0.02,
            "max_unsupported_duty_ucb": 0.02,
        },
        "velociraptor": stance_null,
        "brachiosaurus": stance_null,
        "dibothrosuchus": stance_null,
    }

    for species_id, entry in species.items():
        stage_one, stage_two, stage_three = [stage["advancement_gate"] for stage in entry["stages"]]
        assert stage_one == {
            "min_avg_reward": stage_one_min_avg_reward[species_id],
            "min_avg_episode_length": stage_one_length[species_id],
            "min_avg_forward_velocity": None,
            "min_success_rate": None,
            "min_eval_episodes": stage_one_eval_episodes[species_id],
            "required_consecutive": 3,
            **stage_one_stance[species_id],
        }
        # Stages 2 and 3 stay on reward_and_length/v1, so their stance fields
        # export as nulls.
        assert stage_two | {"min_avg_forward_velocity": None} == {
            "min_avg_reward": 100.0,
            "min_avg_episode_length": 750,
            "min_avg_forward_velocity": None,
            "min_success_rate": None,
            "min_eval_episodes": 10,
            "required_consecutive": 3,
            **stance_null,
        }
        assert stage_three | {"min_avg_forward_velocity": None} == {
            "min_avg_reward": 100.0,
            "min_avg_episode_length": None,
            "min_avg_forward_velocity": None,
            "min_success_rate": 0.5,
            "min_eval_episodes": 10,
            "required_consecutive": 3,
            **stance_null,
        }

    # Trex stage 2 gates a 1.0 m/s walk, re-derived from the plant (Froude
    # 0.14-0.16; the copied raptor 2.0 was this plant's walk-run boundary and
    # passed 0/109 evals on run 20260821_142144 — 2026-08 review §5.3).
    assert species["velociraptor"]["stages"][1]["advancement_gate"]["min_avg_forward_velocity"] == 2.0
    assert species["trex"]["stages"][1]["advancement_gate"]["min_avg_forward_velocity"] == 1.0
    assert species["brachiosaurus"]["stages"][1]["advancement_gate"]["min_avg_forward_velocity"] == 0.75
    # The 2.0 m/s capability target relocated to trex's behavior-stage gate
    # (review §5.3 decision (b)); no other species gates stage 3 on speed.
    assert species["trex"]["stages"][2]["advancement_gate"]["min_avg_forward_velocity"] == 2.0
    for other in ("velociraptor", "brachiosaurus", "dibothrosuchus"):
        assert species[other]["stages"][2]["advancement_gate"]["min_avg_forward_velocity"] is None


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


def test_result_validator_uses_backend_aware_paths() -> None:
    summary = deepcopy(_published_summary())
    summary["backend"] = "jax-mjx"

    validated = _validate_result_summary(
        summary,
        species_id="velociraptor",
        relative_path="results/velociraptor/jax_ppo/summary.json",
    )
    assert validated is summary

    with pytest.raises(CatalogError, match=r"expected jax_ppo for jax-mjx"):
        _validate_result_summary(
            summary,
            species_id="velociraptor",
            relative_path="results/velociraptor/ppo/summary.json",
        )


def test_shared_result_errors_remain_catalog_errors() -> None:
    summary = deepcopy(_published_summary())
    del summary["provenance"]["config_hash"]

    with pytest.raises(CatalogError, match="provenance.*missing fields"):
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


def test_catalog_rejects_plant_contract_that_disagrees_with_environment(tmp_path: Path) -> None:
    plant_manifest = json.loads(DEFAULT_PLANT_MANIFEST_PATH.read_text(encoding="utf-8"))
    plant_manifest["plants"]["velociraptor"]["policy_interface"]["observation_dim"] += 1
    stale_manifest = tmp_path / "plant_manifest.generated.json"
    stale_manifest.write_text(json.dumps(plant_manifest), encoding="utf-8")

    with pytest.raises(CatalogError, match="plant observation_dim mismatch for velociraptor"):
        build_catalog(plant_manifest_path=stale_manifest)


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


def test_jax_notebook_resolves_env_before_binding_reward_functions() -> None:
    notebook = json.loads((REPOSITORY_ROOT / "notebooks" / "jax_training.ipynb").read_text(encoding="utf-8"))
    code = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code")

    create_index = code.index("env = create_env(ctx")
    reward_index = code.index("compute_reward, compute_reward_detailed")

    assert create_index < reward_index, "JAX rewards must bind after create_env populates ctx.env_config"


def test_sb3_notebook_refuses_a_hybrid_model_and_vecnormalize_checkpoint() -> None:
    notebook = json.loads((REPOSITORY_ROOT / "notebooks" / "sb3_training.ipynb").read_text(encoding="utf-8"))
    chunks: list[str] = []
    for cell in notebook["cells"]:
        source = cell.get("source", "")
        chunks.append("".join(source) if isinstance(source, list) else source)
    source_text = "\n".join(chunks)

    assert "refusing to evaluate or export a hybrid checkpoint" in source_text
    assert "using final VecNormalize for best model" not in source_text


def test_sb3_notebook_enforces_the_gate_it_no_longer_evaluates() -> None:
    """Every stage cell must halt on its own gate verdict.

    The notebook used to carry an inline checklist over min_avg_reward /
    min_avg_episode_length / min_avg_forward_vel / min_success_rate. That was
    deleted in favour of the shared `reporting.gates.evaluate_stage_gate`,
    which `generate_stage_artifacts` runs and records onto the results dict --
    so the notebook's remaining job is purely to ENFORCE the recorded verdict.

    Nothing pinned that it still does. Deleting the three enforcement blocks
    leaves the whole `environments/shared/tests` suite green while every run
    silently advances on a failed gate, which is section 12.1's lesson ("a
    gate the trainer never calls is not a gate") reappearing one level up.
    """
    notebook = json.loads((REPOSITORY_ROOT / "notebooks" / "sb3_training.ipynb").read_text(encoding="utf-8"))
    code_cells = ["".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code"]

    for stage in (1, 2, 3):
        results = f"results_{stage}"
        artifact_cells = [cell for cell in code_cells if f"{results} = generate_stage_artifacts(" in cell]
        assert len(artifact_cells) == 1, (
            f"stage {stage} must capture generate_stage_artifacts' return value into {results}; "
            "the gate verdict is recorded onto the dict it returns"
        )
        cell = artifact_cells[0]
        assert f'if not {results}["publication_gate_passed"]:' in cell, (
            f"stage {stage} does not halt on its recorded gate verdict"
        )
        assert f'"; ".join({results}["gate_failures"])' in cell, f"stage {stage} does not report which criteria failed"
        assert "raise RuntimeError(_gate_msg)" in cell, f"stage {stage} warns about gate failure without halting"

    # The deleted checklist must not creep back: a second implementation that
    # knows nothing about `gate_kind` is the exact defect that let a stance-
    # gated stage advance on its reward rail.
    #
    # Scoped to the cells the checklist actually lived in -- `train_stage` and
    # the three artifact cells. The zero-action baseline cell legitimately
    # reads `min_avg_reward` to report whether the statue clears the rail,
    # which is a diagnostic about the gate rather than a second copy of it.
    #
    # Comment lines are excluded because the cell that replaced the checklist
    # explains what it deleted, and naming the retired keys is the point.
    gate_cells = [cell for cell in code_cells if "def train_stage(" in cell or "generate_stage_artifacts(" in cell]
    assert gate_cells, "expected to find the training and artifact cells"
    executable = "\n".join(
        line for cell in gate_cells for line in cell.splitlines() if not line.lstrip().startswith("#")
    )
    for retired in (
        'get("min_avg_reward"',
        'get("min_avg_episode_length"',
        'get("min_avg_forward_vel"',
        'get("min_success_rate"',
        "gate_failures.append(",
        "gate_failures = []",
    ):
        assert retired not in executable, (
            f"{retired!r} is back in the notebook — the curriculum gate belongs in "
            "reporting.gates.evaluate_stage_gate, not in a private per-caller checklist"
        )


def test_sb3_notebook_finalizes_complete_bundle_once() -> None:
    notebook = json.loads((REPOSITORY_ROOT / "notebooks" / "sb3_training.ipynb").read_text(encoding="utf-8"))
    code_cells = ["".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    stage_three_results = "[results_1, results_2, results_3]"

    assert sum(f"write_training_summary(RUN_DIR, {stage_three_results})" in cell for cell in code_cells) == 1
    assert sum(f"save_run_bundle({stage_three_results}, species=SPECIES)" in cell for cell in code_cells) == 1

    completion_cells = [cell for cell in code_cells if 'print("Training complete!")' in cell]
    assert len(completion_cells) == 1
    assert "validate_result_bundle(RUN_DIR, require_complete=True)" in completion_cells[0]


@pytest.mark.parametrize(
    ("species_id", "website_constant"),
    [
        ("velociraptor", "VELOCIRAPTOR"),
        ("trex", "TREX"),
        ("brachiosaurus", "BRACHIOSAURUS"),
        ("dibothrosuchus", "DIBOTHROSUCHUS"),
    ],
)
def test_public_model_pages_render_generated_catalog(species_id: str, website_constant: str) -> None:
    page = (REPOSITORY_ROOT / "website" / "docs" / "models" / f"{species_id}.mdx").read_text(encoding="utf-8")

    assert "import SpeciesCatalog from '@site/src/components/SpeciesCatalog';" in page
    assert f"import {{{website_constant}}} from '@site/src/data/species';" in page
    assert f"<SpeciesCatalog species={{{website_constant}}} />" in page


def test_every_species_model_page_is_tracked_by_git() -> None:
    """A page present on disk but untracked builds locally and breaks CI.

    ``.gitignore`` carried an unanchored ``models/`` rule for training
    artifacts, which also matched ``website/docs/models/``.  Pages added before
    that rule stayed tracked (git never ignores tracked files), so the
    directory looked fine while every *new* page was silently dropped by
    ``git add -A``.  The Docusaurus build then failed on a sidebar entry
    pointing at a document that did not exist in the checkout.  Checking
    presence on disk cannot catch this; only tracking can.
    """
    import subprocess

    try:
        tracked = subprocess.run(
            ["git", "ls-files", "website/docs/models"],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - git absent
        pytest.skip("git is not available")
    if tracked.returncode != 0:  # pragma: no cover - not a work tree
        pytest.skip("not a git work tree")

    tracked_pages = set(tracked.stdout.split())
    for species in build_catalog()["species"]:
        page = f"website/docs/models/{species['id']}.mdx"
        assert page in tracked_pages, (
            f"{page} is not tracked by git. It may exist locally while being excluded by a "
            f".gitignore rule, which builds fine here and fails the Docusaurus job in CI."
        )


def test_default_paths_are_inside_repository() -> None:
    assert DEFAULT_MANIFEST_PATH.is_relative_to(REPOSITORY_ROOT)
    assert DEFAULT_PLANT_MANIFEST_PATH.is_relative_to(REPOSITORY_ROOT)
    assert DEFAULT_OUTPUT_PATH.is_relative_to(REPOSITORY_ROOT)
    assert DEFAULT_README_PATH.is_relative_to(REPOSITORY_ROOT)
