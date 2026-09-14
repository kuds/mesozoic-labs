"""Tests for the generated public species catalog."""

from __future__ import annotations

import ast
import json
import re
import shutil
import tomllib
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest

from environments.shared.curriculum.gate_schema import GATE_KINDS
from environments.shared.result_schema import certified_deliverables
from environments.shared.species_catalog import (
    _HEADLINE_BY_GATE_KIND,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_PLANT_MANIFEST_PATH,
    DEFAULT_README_PATH,
    REPOSITORY_ROOT,
    CatalogError,
    _build_deliverable_metrics,
    _build_result,
    _build_stages,
    _deliverable_headline,
    _format_deliverable,
    _format_verdict,
    _max_reported_velocity,
    _validate_result_summary,
    build_catalog,
    check_catalog,
    current_certification_seeds,
    current_gate_kinds,
    render_readme_results,
    render_readme_species,
)
from environments.shared.stage_manifest import load_stage_manifest, resolve_stage_key

GOLDEN_RESULTS_BLOCK = Path(__file__).parent / "fixtures" / "readme_results_block_2026_09_06.md"


def test_catalog_derives_current_model_and_stage_facts() -> None:
    """Pins the catalog schema and the per-species model/stage facts.

    Re-pinned 3 -> 4 on 2026-09-12 (decision D-A8): stage rows carry the
    recipe DAG and result rows publish per-deliverable certification, and
    the website adapter guards on this number, so the bump is the contract.
    """
    catalog = build_catalog()
    assert catalog["schema_version"] == 4
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
        # Observation widths carry the Phase C 3-dim command segment
        # (BEHAVIOR_RECIPES_PLAN §4.6): 67/61/83/77/53/43 -> 70/64/86/80/56/46.
        "velociraptor": (70, 22, 31, 30, 22, 13.5),
        "trex": (64, 15, 28, 27, 15, 85.72),
        "brachiosaurus": (86, 30, 38, 37, 30, 175.3),
        "dibothrosuchus": (80, 27, 35, 34, 27, 8.65),
        "compsognathus": (56, 14, 24, 23, 14, 1.0),
        "compsognathus_robot": (46, 12, 19, 18, 12, 1.5856),
    }

    assert [stage["timesteps"] for stage in species["velociraptor"]["stages"]] == [6_000_000, 8_000_000, 12_000_000]
    # Stage 1 is 11M, not the 6M every other species uses: trex 1a is the
    # stance-gated stage, and its 6M budget ran out mid-improvement -- run
    # 20260802_203215's best evaluation was its last, at 6.0M of 6M. Raised
    # 10M -> 11M on main (08a66b3) for the seeded replicate campaign.
    #
    # Four rows in MANIFEST order: the recovery stage (3M, recovery.toml)
    # sits between stance and locomotion, labelled by its semantic id — it
    # has no legacy number and none is invented for it.
    assert [stage["timesteps"] for stage in species["trex"]["stages"]] == [
        11_000_000,
        3_000_000,
        8_000_000,
        8_000_000,
    ]
    assert [
        (stage["id"], stage["number"], stage["label"], stage["position"]) for stage in species["trex"]["stages"]
    ] == [
        ("stance", 1, "1", 1),
        ("recovery", None, "recovery", 2),
        ("locomotion", 2, "2", 3),
        ("behavior", 3, "3", 4),
    ]
    # The recipe DAG the committed v2 manifest declares (plan §4.1): every
    # node a deliverable, recovery and locomotion both rooted on stance,
    # "stand" spanning stance and recovery.
    assert [
        (stage["id"], stage["deliverable"], stage["warm_start_from"], stage["recipe"])
        for stage in species["trex"]["stages"]
    ] == [
        ("stance", True, None, "stand"),
        ("recovery", True, "stance", "stand"),
        ("locomotion", True, "stance", "walk"),
        ("behavior", True, "locomotion", "hunt"),
    ]
    for species_id in ("velociraptor", "brachiosaurus", "dibothrosuchus"):
        assert [stage["deliverable"] for stage in species[species_id]["stages"]] == [True, True, True]
        assert [stage["warm_start_from"] for stage in species[species_id]["stages"]] == [None, "stance", "locomotion"]
        assert [stage["recipe"] for stage in species[species_id]["stages"]] == ["stand", "walk", "hunt"]
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
        "compsognathus": "bipedal-target/v1",
        "compsognathus_robot": "bipedal-target/v1",
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
    # Phase C (BEHAVIOR_RECIPES_PLAN §4.6, plant_versions note 12) bumped
    # every policy revision once more: the 3-dim command segment is appended
    # to all six observations. Physics and visual are again untouched.
    expected_policy_revisions = {
        "velociraptor": 10,
        "trex": 13,
        "brachiosaurus": 8,
        "dibothrosuchus": 7,
        "compsognathus": 2,
        "compsognathus_robot": 2,
    }
    expected_physics_revisions = {"velociraptor": 2, "trex": 7, "brachiosaurus": 4, "dibothrosuchus": 1}
    expected_visual_revisions = {"velociraptor": 3, "trex": 4, "brachiosaurus": 2, "dibothrosuchus": 1}
    for revisions in (expected_physics_revisions, expected_visual_revisions):
        revisions.update(compsognathus=1, compsognathus_robot=1)
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

    # Recovery stages declare recovery_quality/v1 criteria; every numbered
    # stage exports nulls for them.
    recovery_null: dict[str, float | None] = {
        "min_recovery_success_lcb": None,
        "min_paired_success_delta_lcb": None,
        "recovery_t_recover_steps": None,
        "recovery_dwell_steps": None,
    }
    # task_success/v1's bar (plan §4.4) is exported on every stage row and
    # is null wherever the kind is not declared — every committed stage
    # except the trex hunt, which adopted the kind (WS-B2; D-B14 keeps the
    # other species' hunts on reward_and_length/v1).
    task_success_null: dict[str, float | None] = {"min_success_lcb": None}
    # The trex hunt under task_success/v1: the LCB95 bar (D-B2, provisional),
    # the declared panel n (D-B1), the statue-derived collapse rail 361 =
    # round(0.6 x 602.13) (D-B4) and neither retired criterion (CF2 / D1
    # retired the velocity target, SS2 the raw-mean success rate).
    trex_stage_three = {
        "gate_kind": "task_success/v1",
        "pending_gate_kind": None,
        "min_avg_reward": 361.0,
        "min_avg_episode_length": None,
        "min_avg_forward_velocity": None,
        "min_success_rate": None,
        "min_eval_episodes": 30,
        "required_consecutive": 3,
        "min_success_lcb": 0.5,
    }

    stage_one_gate_kind = {
        "trex": "stance_quality/v1",
        "velociraptor": "reward_and_length/v1",
        "brachiosaurus": "reward_and_length/v1",
        "dibothrosuchus": "reward_and_length/v1",
    }

    for species_id, entry in species.items():
        if species_id in ("compsognathus", "compsognathus_robot"):
            stages_by_id = {stage["id"]: stage for stage in entry["stages"]}
            first, second, third = (
                stages_by_id[stage_id]["advancement_gate"] for stage_id in ("stance", "locomotion", "behavior")
            )
            assert first["gate_kind"] == "stance_quality/v1"
            assert first["min_avg_reward"] == 1800
            assert first["min_full_horizon_fraction"] == 0.95
            assert first["max_unsupported_duty"] == 0.02
            assert first["max_unsupported_duty_ucb"] == 0.02
            assert second["min_avg_forward_velocity"] == (0.04 if species_id.endswith("_robot") else 0.08)
            assert second["min_avg_episode_length"] == 900
            assert third["min_success_rate"] == 0.7
            assert third["min_avg_episode_length"] is None
            assert first["min_eval_episodes"] == 40
            assert all(gate["min_eval_episodes"] == 20 for gate in (second, third))
            # The semantic recovery row is published without renumbering the
            # advancing curriculum or turning its pilot verdict into a handoff.
            assert [stage.id for stage in load_stage_manifest(species_id).advancing_stages] == [
                "stance",
                "locomotion",
                "behavior",
            ]
            assert [stages_by_id[stage_id]["number"] for stage_id in ("stance", "locomotion", "behavior")] == [1, 2, 3]
            recovery = stages_by_id["recovery"]
            assert recovery["number"] is None
            assert recovery["label"] == "recovery"
            assert recovery["timesteps"] == 3_000_000
            assert recovery["advancement_gate"] == {
                "gate_kind": "recovery_quality/v1",
                "pending_gate_kind": None,
                "min_avg_reward": None,
                "min_avg_episode_length": None,
                "min_avg_forward_velocity": None,
                "min_success_rate": None,
                "min_eval_episodes": 40,
                "required_consecutive": 3,
                "min_recovery_success_lcb": 0.5,
                "min_paired_success_delta_lcb": 0.1,
                "recovery_t_recover_steps": 40,
                "recovery_dwell_steps": 20,
                **stance_null,
                **task_success_null,
            }
            continue
        # Gates are addressed by LEGACY number, not by list position: the
        # trex list has four rows because the recovery stage sits at
        # position 2, and the numbered stages must be unaffected by it.
        gates_by_number = {stage["number"]: stage["advancement_gate"] for stage in entry["stages"]}
        stage_one, stage_two, stage_three = (gates_by_number[number] for number in (1, 2, 3))
        assert stage_one == {
            "gate_kind": stage_one_gate_kind[species_id],
            "pending_gate_kind": None,
            "min_avg_reward": stage_one_min_avg_reward[species_id],
            "min_avg_episode_length": stage_one_length[species_id],
            "min_avg_forward_velocity": None,
            "min_success_rate": None,
            "min_eval_episodes": stage_one_eval_episodes[species_id],
            "required_consecutive": 3,
            **stage_one_stance[species_id],
            **recovery_null,
            **task_success_null,
        }
        # Stages 2 and 3 stay on reward_and_length/v1, so their stance fields
        # export as nulls.
        assert stage_two | {"min_avg_forward_velocity": None} == {
            "gate_kind": "reward_and_length/v1",
            "pending_gate_kind": None,
            "min_avg_reward": 100.0,
            "min_avg_episode_length": 750,
            "min_avg_forward_velocity": None,
            "min_success_rate": None,
            "min_eval_episodes": 10,
            "required_consecutive": 3,
            **stance_null,
            **recovery_null,
            **task_success_null,
        }
        if species_id == "trex":
            assert stage_three == {**trex_stage_three, **stance_null, **recovery_null}
            continue
        assert stage_three | {"min_avg_forward_velocity": None} == {
            "gate_kind": "reward_and_length/v1",
            "pending_gate_kind": None,
            "min_avg_reward": 100.0,
            "min_avg_episode_length": None,
            "min_avg_forward_velocity": None,
            "min_success_rate": 0.5,
            "min_eval_episodes": 10,
            "required_consecutive": 3,
            **stance_null,
            **recovery_null,
            **task_success_null,
        }

    # The recovery stage exports its frozen recovery_quality/v1 gate (P5,
    # 2026-08-28): the certifying criteria the resolution was frozen at, no
    # pending placeholder, and no invented number. Full-dict equality so a
    # silently added or dropped export key fails here, as for the numbered
    # stages above.
    trex_recovery = next(stage for stage in species["trex"]["stages"] if stage["id"] == "recovery")
    assert trex_recovery["number"] is None
    assert trex_recovery["label"] == "recovery"
    assert trex_recovery["timesteps"] == 3_000_000
    assert trex_recovery["advancement_gate"] == {
        "gate_kind": "recovery_quality/v1",
        "pending_gate_kind": None,
        "min_avg_reward": None,
        "min_avg_episode_length": None,
        "min_avg_forward_velocity": None,
        "min_success_rate": None,
        "min_eval_episodes": 40,
        "required_consecutive": 3,
        "min_recovery_success_lcb": 0.30,
        "min_paired_success_delta_lcb": 0.20,
        "recovery_t_recover_steps": 100,
        "recovery_dwell_steps": 50,
        **stance_null,
        **task_success_null,
    }
    assert trex_recovery["video"] is None

    # Trex stage 2 gates a 1.0 m/s walk, re-derived from the plant (Froude
    # 0.14-0.16; the copied raptor 2.0 was this plant's walk-run boundary and
    # passed 0/109 evals on run 20260821_142144 — 2026-08 review §5.3).
    # Stages addressed by legacy NUMBER, not list index: trex's list gained
    # a recovery row at position 2, and "stage 2" must keep meaning
    # locomotion (the no-silent-renumbering invariant).
    def _gate(species_id: str, number: int) -> dict:
        return next(stage["advancement_gate"] for stage in species[species_id]["stages"] if stage["number"] == number)

    assert _gate("velociraptor", 2)["min_avg_forward_velocity"] == 2.0
    assert _gate("trex", 2)["min_avg_forward_velocity"] == 1.0
    assert _gate("brachiosaurus", 2)["min_avg_forward_velocity"] == 0.75
    # No species gates stage 3 on speed. The 2.0 m/s capability target that
    # review §5.3 decision (b) relocated to trex's behavior stage was retired
    # by plan D1 (review CF2: a bite terminates the episode after ~0.5 m, so
    # a bite episode cannot AVERAGE 2.0 m/s); closing speed is measured, not
    # gated. The trex hunt instead certifies the binomial LCB95 on task
    # success (task_success/v1, plan §4.4; D-B2's provisional 0.5 bar) and
    # consumes no raw-mean success rate (review SS2).
    for stage_three_species in ("trex", "velociraptor", "brachiosaurus", "dibothrosuchus"):
        assert _gate(stage_three_species, 3)["min_avg_forward_velocity"] is None
    assert _gate("trex", 3)["min_success_lcb"] == 0.5
    assert _gate("trex", 3)["min_success_rate"] is None


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
    assert manifested_species - {"compsognathus_robot"} == implemented_species
    robot = next(entry for entry in catalog["species"] if entry["id"] == "compsognathus_robot")
    assert robot["environment"]["entrypoint"].startswith("environments.compsognathus.envs.")


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
    """The chain loop must halt on every node's recorded gate verdict.

    The notebook used to carry an inline checklist over min_avg_reward /
    min_avg_episode_length / min_avg_forward_vel / min_success_rate. That was
    deleted in favour of the shared `reporting.gates.evaluate_stage_gate`,
    which `generate_stage_artifacts` runs and records onto the results dict --
    so the notebook's remaining job is purely to ENFORCE the recorded verdict.

    Nothing pinned that it still does. Deleting the enforcement block leaves
    the whole `environments/shared/tests` suite green while every run silently
    advances on a failed gate, which is section 12.1's lesson ("a gate the
    trainer never calls is not a gate") reappearing one level up.

    Re-pinned for the behavior-chain loop (Phase A WS5): the three per-stage
    artifact cells collapsed into ONE chain cell whose enforcement runs for
    every node (recovery included), writes the bundle BEFORE releasing the
    runtime BEFORE raising (invariant 5), and ONE manual single-node cell that
    records a verdict without ever enforcing it.
    """
    notebook = json.loads((REPOSITORY_ROOT / "notebooks" / "sb3_training.ipynb").read_text(encoding="utf-8"))
    code_cells = ["".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code"]

    chain_cells = [cell for cell in code_cells if "# ===== BEHAVIOR CHAIN LOOP =====" in cell]
    manual_cells = [cell for cell in code_cells if "# ===== MANUAL SINGLE NODE" in cell]
    assert len(chain_cells) == 1, "expected exactly one behavior chain loop cell"
    assert len(manual_cells) == 1, "expected exactly one manual single-node cell"
    chain = chain_cells[0]
    assert "results = generate_stage_artifacts(" in chain, (
        "the chain loop must capture generate_stage_artifacts' return value into results; "
        "the gate verdict is recorded onto the dict it returns"
    )
    assert 'if not results["publication_gate_passed"]:' in chain, "the chain loop does not halt on the recorded verdict"
    assert '"; ".join(results["gate_failures"])' in chain, "the chain loop does not report which criteria failed"
    assert "raise RuntimeError(_gate_msg)" in chain, "the chain loop warns about gate failure without halting"
    assert (
        chain.index("save_run_bundle(chain_results()")
        < chain.index("disconnect_runtime(")
        < chain.index("raise RuntimeError(_gate_msg)")
    ), "on a failed gate the bundle is written, then the runtime released, then the loop raises"
    manual = manual_cells[0]
    assert "generate_stage_artifacts(" in manual, "the manual cell must still judge and record the node's verdict"
    assert "raise RuntimeError(_gate_msg)" not in manual and "disconnect_runtime(" not in manual, (
        "the manual single-node cell records its verdict but never enforces it"
    )

    # The deleted checklist must not creep back: a second implementation that
    # knows nothing about `gate_kind` is the exact defect that let a stance-
    # gated stage advance on its reward rail.
    #
    # Scoped to the cells the checklist actually lived in -- `train_stage`,
    # the chain loop and the manual cell. The zero-action baseline cell
    # legitimately reads `min_avg_reward` to report whether the statue clears
    # the rail, which is a diagnostic about the gate rather than a second copy.
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


def test_sb3_notebook_routes_every_bundle_write_through_chain_results() -> None:
    """Every summary/bundle write passes chain_results(); no hand-threaded stage list survives.

    Re-pinned from ``test_sb3_notebook_finalizes_complete_bundle_once`` for the
    behavior-chain loop (Phase A WS5): ``curriculum_results(results_1, ...)`` —
    which spliced the opt-in recovery pilot into a hand-threaded list — is gone
    with the per-stage cells. ``chain_results()`` is the one source of the
    stage-results list (NODE_RESULTS in manifest order), so a save can never
    drop a node this run holds results for, and the ``results_N`` / ``path_N``
    / ``dir_N`` / ``results_r`` variables the old cells threaded forward are
    pinned absent from every code cell.
    """
    notebook = json.loads((REPOSITORY_ROOT / "notebooks" / "sb3_training.ipynb").read_text(encoding="utf-8"))
    code_cells = ["".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code"]

    bundle_calls = [
        line.strip()
        for cell in code_cells
        for line in cell.splitlines()
        if "save_run_bundle(" in line and not line.lstrip().startswith(("def ", "#"))
    ]
    summary_calls = [
        line.strip()
        for cell in code_cells
        for line in cell.splitlines()
        if "write_training_summary(RUN_DIR," in line and not line.lstrip().startswith("#")
    ]
    assert bundle_calls and summary_calls
    assert all("save_run_bundle(chain_results()" in call for call in bundle_calls), bundle_calls
    assert all("write_training_summary(RUN_DIR, chain_results())" in call for call in summary_calls), summary_calls
    joined = "\n".join(code_cells)
    assert "curriculum_results(" not in joined
    assert "save_run_bundle([" not in joined
    for stale in (
        "results_1",
        "results_2",
        "results_3",
        "path_1",
        "path_2",
        "path_3",
        "dir_1",
        "dir_2",
        "dir_3",
        "results_r",
    ):
        assert re.search(rf"\b{stale}\b", joined) is None, (
            f"{stale} is a hand-threaded stage variable; read NODE_HANDOFF"
        )

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


# Review SS5: a published stage_passed is rendered with the gate it was earned
# under.  The committed summaries predate gate provenance and were certified
# by the retired reward gate, while trex's stance stage now declares
# stance_quality/v1 — a bare "Yes" beneath that description misstates what
# was measured.


def test_current_gate_kinds_follow_the_manifest() -> None:
    # The trex hunt adopted task_success/v1 (plan §4.4, WS-B2); every other
    # species' hunt stays on reward_and_length/v1 (D-B14).
    assert current_gate_kinds("trex") == {
        "stance": "stance_quality/v1",
        "recovery": "recovery_quality/v1",
        "locomotion": "reward_and_length/v1",
        "behavior": "task_success/v1",
    }
    assert current_gate_kinds("velociraptor") == {
        "stance": "reward_and_length/v1",
        "locomotion": "reward_and_length/v1",
        "behavior": "reward_and_length/v1",
    }


def test_published_verdicts_carry_gate_provenance() -> None:
    catalog = build_catalog()
    trex = next(entry for entry in catalog["species"] if entry["id"] == "trex")
    stance = next(stage for stage in trex["historical_results"][0]["stages"] if stage["id"] == "stance")
    assert stance["stage_passed"] is True
    assert stance["gate_kind"] is None
    assert stance["current_gate_kind"] == "stance_quality/v1"
    assert stance["gate_retired"] is True

    rendered = render_readme_results(catalog)
    assert "| 1 — Balance | 3008.66 | 0.02 m/s | — | 6M | passed retired gate (reward gate) |" in rendered
    assert "| Yes |" not in rendered


def test_a_verdict_under_the_current_gate_renders_as_a_bare_pass(tmp_path: Path, monkeypatch: Any) -> None:
    from environments.shared import species_catalog

    summary = deepcopy(json.loads((REPOSITORY_ROOT / "results/trex/ppo/summary.json").read_text(encoding="utf-8")))
    summary["stages"]["1"]["gate_kind"] = "stance_quality/v1"
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    monkeypatch.setattr(species_catalog, "_repo_path", lambda relative_path, *, field: summary_path)

    result = species_catalog._build_result("trex", "results/trex/ppo/summary.json")

    stages = {stage["id"]: stage for stage in result["stages"]}
    assert stages["stance"]["gate_retired"] is False
    assert _format_verdict(stages["stance"]) == "Yes"
    assert stages["locomotion"]["gate_retired"] is True
    assert _format_verdict(stages["locomotion"]) == "passed retired gate (reward gate)"
    # Extended 2026-09-12: a current-gate pass on a schema-2 ladder summary
    # is still a ladder pass — it publishes NO deliverable and no primary.
    # Relabelling it certified would mint a certification nothing measured.
    assert result["deliverables"] == []
    assert result["primary_deliverable"] is None
    assert result["target_deliverable"] is None


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        ({"stage_passed": None, "gate_kind": None, "gate_retired": True}, "—"),
        ({"stage_passed": True, "gate_kind": "stance_quality/v1", "gate_retired": False}, "Yes"),
        ({"stage_passed": False, "gate_kind": "stance_quality/v1", "gate_retired": False}, "No"),
        ({"stage_passed": True, "gate_kind": None, "gate_retired": True}, "passed retired gate (reward gate)"),
        (
            {"stage_passed": True, "gate_kind": "stance_quality/v0", "gate_retired": True},
            "passed retired gate (stance_quality/v0)",
        ),
        (
            {"stage_passed": False, "gate_kind": "reward_and_length/v1", "gate_retired": True},
            "failed retired gate (reward_and_length/v1)",
        ),
    ],
)
def test_format_verdict_labels_retired_gates(stage: dict[str, Any], expected: str) -> None:
    assert _format_verdict(stage) == expected


# ── Catalog schema 4: the recipe DAG, per-deliverable publication, and the
# generated README (BEHAVIOR_RECIPES_PLAN §4.3, decisions D-A8 / D-A9 / D-A10).


def _build_catalog_stages(species_id: str) -> list[dict[str, Any]]:
    species = next(entry for entry in build_catalog()["species"] if entry["id"] == species_id)
    return cast(list[dict[str, Any]], species["stages"])


def test_stage_rows_carry_the_recipe_dag() -> None:
    """Every stage row exports deliverable / warm_start_from / recipe as top-level keys.

    Top-level, not inside advancement_gate: the full-dict gate pins above
    must keep holding, and the edge is a property of the node, not of its
    gate.  The values are the committed v2 manifest's, not a derivation.
    """
    for stage in _build_catalog_stages("trex"):
        assert {"deliverable", "warm_start_from", "recipe"} <= set(stage)
        assert not {"deliverable", "warm_start_from", "recipe"} & set(stage["advancement_gate"])
    manifest = load_stage_manifest("trex")
    for stage in _build_catalog_stages("trex"):
        entry = manifest.by_id(stage["id"])
        assert (stage["deliverable"], stage["warm_start_from"], stage["recipe"]) == (
            entry.deliverable,
            entry.warm_start_from,
            entry.recipe,
        )


def test_synthesized_manifest_stage_rows_derive_legacy_edges(tmp_path: Path) -> None:
    """A manifest-less species publishes plan §8.1 as the catalog sees it.

    Only the committed v2 files make stand and walk deliverables: with the
    velociraptor's three stage TOMLs and no stages.toml the reader
    synthesizes the legacy manifest, and the catalog must publish exactly
    what it derives — edges to the previous advancing entry, the last
    advancing entry the only deliverable, no recipe labels — never invent
    a recipe of its own.  The configs_dir kwarg points the builder at the
    temporary tree without monkeypatching the loaders' private constants.
    """
    species_dir = tmp_path / "configs" / "velociraptor"
    species_dir.mkdir(parents=True)
    for config in sorted((REPOSITORY_ROOT / "configs" / "velociraptor").glob("stage*_*.toml")):
        shutil.copy(config, species_dir / config.name)
    assert not (species_dir / "stages.toml").exists()

    stages = _build_stages("velociraptor", [], configs_dir=tmp_path / "configs")

    assert [stage["id"] for stage in stages] == ["stance", "locomotion", "behavior"]
    assert [stage["deliverable"] for stage in stages] == [False, False, True]
    assert [stage["warm_start_from"] for stage in stages] == [None, "stance", "locomotion"]
    assert [stage["recipe"] for stage in stages] == [None, None, None]
    assert stages[0]["config_path"] == "configs/velociraptor/stage1_balance.toml"
    # The gates come from the temporary tree too, through the same kwarg.
    assert current_gate_kinds("velociraptor", tmp_path / "configs") == {
        "stance": "reward_and_length/v1",
        "locomotion": "reward_and_length/v1",
        "behavior": "reward_and_length/v1",
    }


def test_stage_rows_carry_recipe_edges_for_every_species() -> None:
    """Every non-root row names an EARLIER row id (list order is topological)."""
    for species in build_catalog()["species"]:
        seen: list[str] = []
        roots = 0
        for stage in species["stages"]:
            parent = stage["warm_start_from"]
            if parent is None:
                roots += 1
            else:
                assert parent in seen, f"{species['id']} {stage['id']} warm-starts from a later or unknown {parent!r}"
            seen.append(stage["id"])
        assert roots >= 1
        assert any(stage["deliverable"] for stage in species["stages"])


def test_ladder_summaries_keep_stage3_headline_and_publish_no_deliverables() -> None:
    """The four committed schema-2 rows keep their ladder headline and publish nothing new.

    `stage3_success_rate` is the historical headline (plan §4.3 keeps it);
    `deliverables` is [] and both deliverable keys are null because a
    reward-gate pass is not a certification and is never synthesized into
    one.
    """
    catalog = build_catalog()
    rows = {
        (species["id"], result["algorithm"].lower()): result
        for species in catalog["species"]
        for result in species["historical_results"]
    }
    assert {key: row["stage3_success_rate"] for key, row in rows.items()} == {
        ("velociraptor", "ppo"): 0.9333,
        ("velociraptor", "sac"): 0.9,
        ("trex", "ppo"): 0.9667,
        ("brachiosaurus", "ppo"): 1.0,
    }
    for row in rows.values():
        assert row["deliverables"] == []
        assert row["primary_deliverable"] is None
        assert row["target_deliverable"] is None
        assert all({"recipe", "deliverable"} <= set(stage) for stage in row["stages"])


_SHA = "sha256:" + "a" * 64


def _trex_v4_summary(
    verdicts: dict[str, bool],
    *,
    target: str | None,
    primary: str | None,
    bundle_status: str,
) -> dict[str, Any]:
    """The committed trex ladder summary re-cut as a schema-4 result.

    Non-canonical provenance (historical, identifiers null), which is the
    shape the catalog validates: `provenance.species` / `backend` plus the
    deliverables map, primary and target.  Certification is recomputed
    from the recorded verdicts through the shared rule so a fixture can
    never claim more than its chain supports.
    """
    summary = deepcopy(json.loads((REPOSITORY_ROOT / "results/trex/ppo/summary.json").read_text(encoding="utf-8")))
    summary["schema_version"] = 4
    summary["bundle_status"] = bundle_status
    summary["stages"] = {key: stage for key, stage in summary["stages"].items() if key in verdicts}
    gates = current_gate_kinds("trex")
    manifest = load_stage_manifest("trex")
    for key, passed in verdicts.items():
        summary["stages"][key]["stage_passed"] = passed
        summary["stages"][key]["gate_kind"] = gates[resolve_stage_key("trex", key).id]
    summary["total_timesteps"] = sum(int(stage["timesteps"]) for stage in summary["stages"].values())
    entries = [(entry.key, entry) for entry in manifest.stages if entry.key in verdicts]
    certified = certified_deliverables(entries, verdicts, None, species="trex")
    summary["provenance"].update(
        {
            "species": "trex",
            "backend": "stable-baselines3",
            "deliverables": {
                key: {
                    "model_path": f"{key}/models/best_model.zip",
                    "model_hash": _SHA,
                    "normalization_hash": _SHA,
                    "gate_kind": summary["stages"][key]["gate_kind"],
                    "certified": certified[key],
                    "replication": {"count": 1, "runs": [{"run_id": "trex-test", "training_seed": 42}]},
                }
                for key in verdicts
            },
            "primary_deliverable": primary,
            "target_deliverable": target,
        }
    )
    return cast(dict[str, Any], summary)


def _build_result_from(tmp_path: Path, monkeypatch: Any, summary: dict[str, Any]) -> dict[str, Any]:
    from environments.shared import species_catalog

    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    monkeypatch.setattr(species_catalog, "_repo_path", lambda relative_path, *, field: summary_path)
    return _build_result("trex", "results/trex/ppo/summary.json")


def test_v4_partial_summary_publishes_certified_deliverables(tmp_path: Path, monkeypatch: Any) -> None:
    """A failed-hunt run publishes its certified trunk (plan §8.5) with gate-kind headlines.

    Stance and walk certified, hunt failed: the primary is walk ("2"), the
    target hunt stays recorded, every deliverable row carries its gate
    kind, certification, hash, replication and headline, and the stance
    headline names its statistics with null values (D-A9).
    """
    summary = _trex_v4_summary({"1": True, "2": True, "3": False}, target="3", primary="2", bundle_status="partial")

    result = _build_result_from(tmp_path, monkeypatch, summary)

    assert result["primary_deliverable"] == "2"
    assert result["target_deliverable"] == "3"
    assert [
        (row["id"], row["stage_key"], row["label"], row["recipe"], row["certified"]) for row in result["deliverables"]
    ] == [
        ("stance", "1", "1", "stand", True),
        ("locomotion", "2", "2", "walk", True),
        ("behavior", "3", "3", "hunt", False),
    ]
    by_id = {row["id"]: row for row in result["deliverables"]}
    assert by_id["stance"]["gate_kind"] == "stance_quality/v1"
    assert by_id["stance"]["model_hash"] == _SHA
    assert by_id["stance"]["replication_count"] == 1
    # Seed replication (plan §4.5, D-B10/D-B11): the bar is the CURRENT
    # config's — trex stance declares 2, every other stage the default 1 —
    # so a one-run stance is provisional and a one-run walk or hunt is not.
    assert (by_id["stance"]["certification_seeds"], by_id["stance"]["provisional"]) == (2, True)
    assert (by_id["locomotion"]["certification_seeds"], by_id["locomotion"]["provisional"]) == (1, False)
    assert (by_id["behavior"]["certification_seeds"], by_id["behavior"]["provisional"]) == (1, False)
    assert by_id["stance"]["headline"] == [
        {"key": "unsupported_duty_ucb", "label": "unsupported duty 95% UCB", "value": None, "unit": "ratio"},
        {"key": "full_horizon_fraction", "label": "full-horizon episodes", "value": None, "unit": "percent"},
    ]
    assert by_id["locomotion"]["headline"] == [
        {"key": "avg_forward_vel", "label": "avg. forward velocity", "value": 3.47, "unit": "m/s"}
    ]
    # Trex's hunt is task_success/v1 (plan §4.4, WS-B2): the headline names
    # the certified bound and the rate it bounds, and this ladder-shaped
    # fixture records neither selected_model_success_lcb nor
    # selected_model_success_rate, so both carry null values like the stance
    # row above -- never the retired velocity target (plan D1) nor the raw
    # ladder mean.
    assert by_id["behavior"]["gate_kind"] == "task_success/v1"
    assert by_id["behavior"]["headline"] == [
        {"key": "selected_model_success_lcb", "label": "task success LCB95", "value": None, "unit": "ratio"},
        {"key": "selected_model_success_rate", "label": "task success", "value": None, "unit": "percent"},
    ]
    # The ladder headline is untouched by the per-deliverable rows.
    assert result["stage3_success_rate"] == 0.9667
    assert result["max_average_forward_velocity"] == 3.47
    # The exported provenance is the same six-key identity/status surface a
    # v2 row exports: the validated block's deliverables map, primary and
    # target are published only through the rows above, never duplicated
    # under a second shape the CATALOG ROWS contract does not name.
    assert sorted(result["provenance"]) == [
        "config_hash",
        "evaluation_episodes",
        "model_hash",
        "model_revision_status",
        "repository_commit",
        "verification_status",
    ]


def test_provisional_is_rederived_from_the_current_config(tmp_path: Path, monkeypatch: Any) -> None:
    """Decision D-B10: the catalog re-derives ``provisional`` from the CURRENT ``certification_seeds``.

    A stance bundle recorded at n = 1 under a bar of 1 (``provisional: false``
    when written) is relabelled provisional once the config declares 2,
    without republishing; the same bundle with two runs recorded is not.
    The recorded pair is validated (the schema refuses an inconsistent one)
    but never re-served over the current bar.
    """
    summary = _trex_v4_summary({"1": True, "2": True, "3": False}, target="3", primary="2", bundle_status="partial")
    stance = summary["provenance"]["deliverables"]["1"]
    stance.update({"certification_seeds": 1, "provisional": False})

    row = next(r for r in _build_result_from(tmp_path, monkeypatch, summary)["deliverables"] if r["id"] == "stance")
    assert (row["replication_count"], row["certification_seeds"], row["provisional"]) == (1, 2, True)

    stance["replication"] = {
        "count": 2,
        "runs": [{"run_id": "trex-test", "training_seed": 42}, {"run_id": "trex-seed-44", "training_seed": 44}],
    }
    stance.update({"certification_seeds": 2, "provisional": False})
    row = next(r for r in _build_result_from(tmp_path, monkeypatch, summary)["deliverables"] if r["id"] == "stance")
    assert (row["replication_count"], row["certification_seeds"], row["provisional"]) == (2, 2, False)

    # A recorded pair that contradicts itself is a schema failure, never a
    # silently reduced count.
    stance.update({"certification_seeds": 2, "provisional": True})
    with pytest.raises(CatalogError, match="provisional must equal replication.count < certification_seeds"):
        _build_result_from(tmp_path, monkeypatch, summary)


def test_stage_rows_carry_the_current_certification_seeds() -> None:
    """Every stage row exports ``certification_seeds`` top-level (never inside the gate): trex stance 2, else 1."""
    for species in build_catalog()["species"]:
        expected = current_certification_seeds(species["id"])
        for stage in species["stages"]:
            assert stage["certification_seeds"] == expected[stage["id"]]
            assert "certification_seeds" not in stage["advancement_gate"]
            assert stage["certification_seeds"] == (2 if (species["id"], stage["id"]) == ("trex", "stance") else 1)


def test_v4_headline_reads_the_statistic_the_summary_records(tmp_path: Path, monkeypatch: Any) -> None:
    """A stance statistic recorded in the summary stage row surfaces in the headline.

    Phase A summaries record none (so the value is null, D-A9), but the
    headline is valued from the summary stage AS WRITTEN, not from the
    ladder projection that carries only the fixed ladder columns: Phase
    B's export must surface without a second catalog edit.
    """
    summary = _trex_v4_summary({"1": True, "2": True, "3": False}, target="3", primary="2", bundle_status="partial")
    summary["stages"]["1"]["unsupported_duty_ucb"] = 0.01
    summary["stages"]["1"]["full_horizon_fraction"] = 0.99

    result = _build_result_from(tmp_path, monkeypatch, summary)

    stance = next(row for row in result["deliverables"] if row["id"] == "stance")
    assert [(metric["key"], metric["value"]) for metric in stance["headline"]] == [
        ("unsupported_duty_ucb", 0.01),
        ("full_horizon_fraction", 0.99),
    ]
    # The ladder row keeps its fixed columns: the statistic is not projected there.
    assert "unsupported_duty_ucb" not in result["stages"][0]


def test_v4_headline_reads_the_task_success_lcb_the_summary_records(tmp_path: Path, monkeypatch: Any) -> None:
    """A task_success/v1 hunt deliverable headlines selected_model_success_lcb from the stage row as written."""
    summary = _trex_v4_summary({"1": True, "2": True, "3": True}, target="3", primary="3", bundle_status="complete")
    summary["stages"]["3"]["gate_kind"] = "task_success/v1"
    summary["provenance"]["deliverables"]["3"]["gate_kind"] = "task_success/v1"
    summary["stages"]["3"]["selected_model_success_lcb"] = 0.851
    summary["stages"]["3"]["selected_model_success_count"] = 29
    summary["stages"]["3"]["selected_model_n_episodes"] = 30
    summary["stages"]["3"]["selected_model_success_rate"] = 0.9667

    result = _build_result_from(tmp_path, monkeypatch, summary)

    hunt = next(row for row in result["deliverables"] if row["id"] == "behavior")
    assert hunt["gate_kind"] == "task_success/v1"
    assert [(metric["key"], metric["value"]) for metric in hunt["headline"]] == [
        ("selected_model_success_lcb", 0.851),
        ("selected_model_success_rate", 0.9667),
    ]
    assert "selected_model_success_lcb" not in result["stages"][-1]


def test_v4_summary_whose_primary_is_spelled_by_id_over_numeric_keys_is_rejected(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A primary that names a stage by id while the deliverables are keyed by number fails closed.

    The primary is a KEY of provenance.deliverables; "locomotion" over a map
    keyed "1", "2", "3" resolves to a stage but matches no key.  The shared
    validator rejects it with a message naming the keys (a CatalogError
    here, never a bare StopIteration escaping ``python -m ...species_catalog``).
    """
    summary = _trex_v4_summary(
        {"1": True, "2": True, "3": False}, target="3", primary="locomotion", bundle_status="partial"
    )
    with pytest.raises(
        CatalogError, match=r"'locomotion', which is not a key of provenance.deliverables \['1', '2', '3'\]"
    ):
        _build_result_from(tmp_path, monkeypatch, summary)


def test_v4_summary_whose_primary_is_uncertified_is_rejected(tmp_path: Path, monkeypatch: Any) -> None:
    """The catalog never headlines a failed leaf: an uncertified primary is fatal."""
    summary = _trex_v4_summary({"1": True, "2": True, "3": False}, target="3", primary="3", bundle_status="partial")
    with pytest.raises(CatalogError, match="not certified"):
        _build_result_from(tmp_path, monkeypatch, summary)

    # A certified primary that is not the one the deliverables and target
    # imply (walk is the deepest certified, so stance cannot be primary).
    summary = _trex_v4_summary({"1": True, "2": True, "3": False}, target="3", primary="1", bundle_status="partial")
    with pytest.raises(CatalogError, match="is not the one its deliverables and target"):
        _build_result_from(tmp_path, monkeypatch, summary)

    summary = _trex_v4_summary({"1": True, "2": True, "3": False}, target="3", primary=None, bundle_status="partial")
    with pytest.raises(CatalogError, match="no primary deliverable"):
        _build_result_from(tmp_path, monkeypatch, summary)


def test_v4_summary_with_unknown_deliverable_stage_is_rejected(tmp_path: Path, monkeypatch: Any) -> None:
    """A deliverable key outside the species' manifest vocabulary fails closed."""
    summary = _trex_v4_summary({"1": True, "2": True, "3": True}, target="3", primary="3", bundle_status="complete")
    summary["provenance"]["deliverables"]["follow_direction"] = deepcopy(summary["provenance"]["deliverables"]["3"])
    with pytest.raises(CatalogError, match="follow_direction"):
        _build_result_from(tmp_path, monkeypatch, summary)


def test_walk_only_v4_summary_is_publishable(tmp_path: Path, monkeypatch: Any) -> None:
    """A walk-targeted run with no behavior stage publishes (plan §8.5) and has no ladder headline."""
    summary = _trex_v4_summary({"1": True, "2": True}, target="2", primary="2", bundle_status="complete")

    result = _build_result_from(tmp_path, monkeypatch, summary)

    assert [row["id"] for row in result["deliverables"]] == ["stance", "locomotion"]
    assert all(row["certified"] for row in result["deliverables"])
    assert result["primary_deliverable"] == "2"
    assert result["target_deliverable"] == "2"
    assert result["stage3_success_rate"] is None
    assert [stage["id"] for stage in result["stages"]] == ["stance", "locomotion"]


@pytest.mark.parametrize(
    ("gate_kind", "stage_row", "current_gate", "expected"),
    [
        (
            "stance_quality/v1",
            {"avg_forward_vel": 0.02},
            {"min_full_horizon_fraction": 0.95},
            [("unsupported_duty_ucb", None, "ratio"), ("full_horizon_fraction", None, "percent")],
        ),
        # The statistic recorded (Phase B's export): the value passes through.
        (
            "stance_quality/v1",
            {"unsupported_duty_ucb": 0.01, "full_horizon_fraction": 0.99},
            {"min_full_horizon_fraction": 0.95},
            [("unsupported_duty_ucb", 0.01, "ratio"), ("full_horizon_fraction", 0.99, "percent")],
        ),
        (
            "recovery_quality/v1",
            {},
            {"min_recovery_success_lcb": 0.3},
            [("recovery_success_lcb", None, "ratio")],
        ),
        (
            "recovery_quality/v1",
            {"recovery_success_lcb": 0.42},
            {"min_recovery_success_lcb": 0.3},
            [("recovery_success_lcb", 0.42, "ratio")],
        ),
        (
            "reward_and_length/v1",
            {"avg_forward_vel": 3.47, "mean_success_rate": None},
            {"min_avg_forward_velocity": 1.0, "min_success_rate": None},
            [("avg_forward_vel", 3.47, "m/s")],
        ),
        (
            "reward_and_length/v1",
            {"avg_forward_vel": 0.71, "mean_success_rate": 0.9333},
            {"min_avg_forward_velocity": None, "min_success_rate": 0.5},
            [("mean_success_rate", 0.9333, "percent")],
        ),
        (
            "reward_and_length/v1",
            {"avg_forward_vel": 1.68, "mean_success_rate": 0.9667},
            {"min_avg_forward_velocity": 2.0, "min_success_rate": 0.5},
            [("avg_forward_vel", 1.68, "m/s"), ("mean_success_rate", 0.9667, "percent")],
        ),
        # A reward-only rail headlines nothing: neither floor is declared.
        ("reward_and_length/v1", {"avg_forward_vel": 0.02}, {"min_avg_reward": 1050.0}, []),
        # task_success/v1 (plan §4.4): the bound the gate certifies, then
        # the raw selected-checkpoint rate it bounds; nulls until recorded.
        (
            "task_success/v1",
            {"selected_model_success_lcb": 0.5006, "selected_model_success_rate": 0.6667},
            {"min_success_lcb": 0.5},
            [("selected_model_success_lcb", 0.5006, "ratio"), ("selected_model_success_rate", 0.6667, "percent")],
        ),
        (
            "task_success/v1",
            {"mean_success_rate": 0.9667},
            {"min_success_lcb": 0.5},
            [("selected_model_success_lcb", None, "ratio"), ("selected_model_success_rate", None, "percent")],
        ),
        ("none/v1", {"mean_success_rate": 1.0}, {"min_success_rate": 0.5}, []),
        # Unrecorded gate kind: nothing measured, nothing headlined.
        (None, {"mean_success_rate": 1.0}, {"min_success_rate": 0.5}, []),
    ],
)
def test_headline_metric_by_gate_kind(
    gate_kind: str | None,
    stage_row: dict[str, Any],
    current_gate: dict[str, Any],
    expected: list[tuple[str, Any, str]],
) -> None:
    """The headline is chosen by the certifying gate kind and valued from the stage row.

    Stance and recovery name their statistics with null values in Phase A
    (D-A9) and carry the value through as soon as the summary stage records
    it; reward_and_length headlines velocity and/or success according
    to which floors the current gate declares; none/v1 and an unrecorded
    kind headline nothing.
    """
    headline = _deliverable_headline(gate_kind, stage_row, current_gate)
    assert [(metric["key"], metric["value"], metric["unit"]) for metric in headline] == expected
    assert all(set(metric) == {"key", "label", "value", "unit"} for metric in headline)


def test_unknown_gate_kind_has_no_headline_and_is_fatal() -> None:
    """A kind the registry does not know is a catalog failure that names the registry."""
    with pytest.raises(CatalogError, match="_HEADLINE_BY_GATE_KIND"):
        _deliverable_headline("tracking_quality/v1", {}, {})


def test_every_registered_gate_kind_has_a_headline() -> None:
    """The headline registry is keyed by exactly the schema's gate kinds.

    A gate kind added to curriculum.gate_schema.GATE_KINDS without a
    headline entry would publish metric-less deliverables; a stale entry
    would describe a gate nothing declares.
    """
    assert set(_HEADLINE_BY_GATE_KIND) == set(GATE_KINDS)


def test_deliverable_metrics_cover_every_declared_deliverable() -> None:
    """Every manifest deliverable of every species has a stable-baselines3 definition.

    The rows resolve exactly as the notebook's BEHAVIOR knob does, so the
    published semantics attach to the node a chain would actually target.
    """
    catalog = build_catalog()
    for species in catalog["species"]:
        manifest = load_stage_manifest(species["id"])
        metrics = species["deliverable_metrics"]
        assert metrics, species["id"]
        for metric in metrics:
            assert set(metric) == {"deliverable", "stage_id", "backends", "key", "label", "definition"}
            assert manifest.resolve_behavior(metric["deliverable"]).id == metric["stage_id"]
        sb3_covered = {metric["stage_id"] for metric in metrics if "stable-baselines3" in metric["backends"]}
        assert {entry.id for entry in manifest.deliverables} <= sb3_covered, species["id"]
        # The per-backend success metrics are untouched beside them.
        assert species["success_metrics"]
        assert all(set(metric) == {"backends", "key", "label", "definition"} for metric in species["success_metrics"])

    with pytest.raises(CatalogError, match=r"missing \['behavior'\]"):
        _build_deliverable_metrics(
            "velociraptor",
            [
                {
                    "deliverable": "stand",
                    "backends": ["stable-baselines3"],
                    "key": "k",
                    "label": "l",
                    "definition": "d",
                },
                {"deliverable": "walk", "backends": ["stable-baselines3"], "key": "k", "label": "l", "definition": "d"},
            ],
            {"stable-baselines3", "jax-mjx"},
        )


def _metric(deliverable: str, *backends: str) -> dict[str, Any]:
    # No backend named means the evidence backend, stable-baselines3.
    scoped = list(backends) or ["stable-baselines3"]
    return {"deliverable": deliverable, "backends": scoped, "key": "k", "label": "l", "definition": "d"}


def test_deliverable_metrics_reject_unknown_and_duplicate_scopes() -> None:
    """Label resolution follows the manifest; unknown names, backends and duplicate scopes fail closed."""
    # "stand" is the recovery node on trex (the deepest deliverable carrying
    # the label) and the stance node everywhere else.
    trex = _build_deliverable_metrics(
        "trex",
        [_metric("stance"), _metric("stand"), _metric("walk"), _metric("hunt")],
        {"stable-baselines3", "jax-mjx"},
    )
    assert [(metric["deliverable"], metric["stage_id"]) for metric in trex] == [
        ("stance", "stance"),
        ("stand", "recovery"),
        ("walk", "locomotion"),
        ("hunt", "behavior"),
    ]
    velociraptor = _build_deliverable_metrics(
        "velociraptor", [_metric("stand"), _metric("walk"), _metric("hunt")], {"stable-baselines3", "jax-mjx"}
    )
    assert [metric["stage_id"] for metric in velociraptor] == ["stance", "locomotion", "behavior"]

    with pytest.raises(CatalogError, match="unknown deliverable 'follow_direction'"):
        _build_deliverable_metrics("trex", [_metric("follow_direction")], {"stable-baselines3"})
    with pytest.raises(CatalogError, match="unknown backends: \\['torch'\\]"):
        _build_deliverable_metrics("trex", [_metric("stance", "torch")], {"stable-baselines3"})
    with pytest.raises(CatalogError, match="does not train: \\['jax-mjx'\\]"):
        _build_deliverable_metrics("compsognathus", [_metric("stance", "jax-mjx")], {"stable-baselines3"})
    # Two spellings of one (stage, backend) scope are a duplicate.
    with pytest.raises(CatalogError, match="more than one deliverable metric for stage 'recovery'"):
        _build_deliverable_metrics("trex", [_metric("stand"), _metric("recovery")], {"stable-baselines3"})


def test_deliverable_metrics_for_reward_gated_stance_say_so() -> None:
    """Reward-cleared stance is labelled by gate kind, never claimed as stance quality (plan §4.8)."""
    species = {entry["id"]: entry for entry in build_catalog()["species"]}

    def stance_definition(species_id: str) -> str:
        return next(
            str(metric["definition"])
            for metric in species[species_id]["deliverable_metrics"]
            if metric["stage_id"] == "stance" and "stable-baselines3" in metric["backends"]
        )

    for species_id in ("velociraptor", "brachiosaurus", "dibothrosuchus"):
        definition = stance_definition(species_id)
        assert "reward_and_length/v1" in definition, species_id
        assert "statue" in definition, species_id
        assert "certified stance quality" in definition, species_id
    for species_id in ("trex", "compsognathus", "compsognathus_robot"):
        assert "stance_quality/v1" in stance_definition(species_id), species_id


def test_species_manifest_schema_version_is_2(tmp_path: Path) -> None:
    """The committed manifest is schema 2 (D-A8) and the reader refuses schema 1."""
    manifest_text = DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8")
    with DEFAULT_MANIFEST_PATH.open("rb") as handle:
        assert tomllib.load(handle)["schema_version"] == 2
    stale = tmp_path / "species_manifest.toml"
    stale.write_text(manifest_text.replace("schema_version = 2", "schema_version = 1", 1), encoding="utf-8")
    with pytest.raises(CatalogError, match="schema_version must be 2"):
        build_catalog(stale)


def _video(stage: "int | str") -> dict[str, Any]:
    return {
        "stage": stage,
        "path": "website/static/videos/trex_ppo_stage1_best.mp4",
        "algorithm": "PPO",
        "backend": "stable-baselines3",
        "model_revision_status": "historical",
        "verification_status": "unverified",
    }


def test_stage_videos_accept_ids_and_integer_aliases_interchangeably() -> None:
    """Videos are keyed by stage id; the legacy integer is an alias, and both together are a duplicate."""
    by_id = _build_stages("trex", [_video("stance")])
    by_number = _build_stages("trex", [_video(1)])
    assert by_id == by_number
    assert by_id[0]["video"] is not None
    assert by_id[0]["video"]["path"] == "/videos/trex_ppo_stage1_best.mp4"

    with pytest.raises(CatalogError, match="duplicate stage video"):
        _build_stages("trex", [_video("stance"), _video(1)])
    # A semantic-only stage is reachable by its id.
    recovery = next(stage for stage in _build_stages("trex", [_video("recovery")]) if stage["id"] == "recovery")
    assert recovery["video"] is not None
    with pytest.raises(CatalogError, match="unknown stage"):
        _build_stages("trex", [_video("follow_direction")])


def test_committed_stage_videos_are_keyed_by_id() -> None:
    """The committed manifest spells every stage video by id, and the generated rows are unchanged."""
    with DEFAULT_MANIFEST_PATH.open("rb") as handle:
        manifest = tomllib.load(handle)
    for species in manifest["species"]:
        declared = {entry.id for entry in load_stage_manifest(species["id"]).stages}
        for video in species.get("stage_videos", []):
            assert isinstance(video["stage"], str), (species["id"], video["stage"])
            assert video["stage"] in declared
    # Migration is presentation-neutral: the committed rows still label the
    # eight historical videos exactly as test_catalog_labels_published_videos
    # pins, keyed onto the stage rows by id.
    videos = {
        (species["id"], stage["id"]): stage["video"]["path"]
        for species in build_catalog()["species"]
        for stage in species["stages"]
        if stage["video"] is not None
    }
    assert videos[("trex", "stance")] == "/videos/trex_ppo_stage1_best.mp4"
    assert videos[("brachiosaurus", "locomotion")] == "/videos/brachiosaurus_ppo_stage2_best.mp4"
    assert ("trex", "recovery") not in videos


def test_readme_species_table_renders_recipe_edges() -> None:
    """The SPECIES table gains Recipe and Warm-start-from columns (D-A10), parent named by its row label."""
    rendered = render_readme_species(build_catalog())
    assert (
        "| Current stage | Recipe | Warm-start from | Objective | SB3 configured budget | SB3 early-advancement gate |"
        in rendered
    )
    assert "| 1 — Balance | stand (deliverable) | — |" in rendered
    assert "| recovery — Recovery | stand (deliverable) | 1 — Balance |" in rendered
    assert "| 2 — Locomotion | walk (deliverable) | 1 — Balance |" in rendered
    assert "| 3 — Bite | hunt (deliverable) | 2 — Locomotion |" in rendered
    assert "**Per-deliverable success semantics:**" in rendered
    assert "- **stance (1 — Balance) · Stable-Baselines3 — Stance quality (stance_quality/v1):**" in rendered
    assert "- **stand (1 — Balance) · Stable-Baselines3 — Reward-gated stance (reward_and_length/v1):**" in rendered


def test_readme_results_block_is_byte_identical_for_ladder_summaries() -> None:
    """The generated RESULTS block is byte-identical to its pre-Phase-A rendering (D-A10).

    The golden fixture is the block between the RESULTS markers of
    README.md at the merge of PR #528 (`git show HEAD:README.md`, captured
    2026-09-12 before any WS4 edit).  A Deliverables line is only emitted
    for a schema-4 result, and the four committed summaries are schema 2.
    """
    golden = GOLDEN_RESULTS_BLOCK.read_text(encoding="utf-8")
    assert golden.count("### ") == 4
    assert render_readme_results(build_catalog()).rstrip() + "\n" == golden
    committed = DEFAULT_README_PATH.read_text(encoding="utf-8")
    begin, end = "<!-- BEGIN GENERATED: RESULTS -->\n", "<!-- END GENERATED: RESULTS -->"
    assert committed.split(begin, 1)[1].split(end, 1)[0] == golden


def test_readme_results_render_deliverables_for_v4_summary(tmp_path: Path, monkeypatch: Any) -> None:
    """A schema-4 result adds one Deliverables line, and every ladder section stays as it was."""
    catalog = build_catalog()
    baseline = render_readme_results(catalog)
    summary = _trex_v4_summary({"1": True, "2": True, "3": False}, target="3", primary="2", bundle_status="partial")
    v4_result = _build_result_from(tmp_path, monkeypatch, summary)
    trex = next(species for species in catalog["species"] if species["id"] == "trex")
    trex["historical_results"] = [v4_result]

    rendered = render_readme_results(catalog)

    assert "**Deliverables:** " in rendered
    assert (
        "1 — stand (certified; gate stance_quality/v1; 1 run of 2 seeds; provisional; "
        "unsupported duty 95% UCB not recorded; full-horizon episodes not recorded) · "
        "2 — walk (certified, primary; gate reward_and_length/v1; 1 run of 1 seed; avg. forward velocity 3.47 m/s) · "
        "3 — hunt (not certified; gate task_success/v1; 1 run of 1 seed; task success LCB95 not recorded; "
        "task success not recorded)"
    ) in rendered
    # Count and bar are always rendered (D-B11); only a count below the bar says provisional.
    replicated = {**v4_result["deliverables"][0], "replication_count": 2, "provisional": False}
    assert "; 2 runs of 2 seeds; " in _format_deliverable(replicated, primary=False)
    assert "provisional" not in _format_deliverable(replicated, primary=False)
    assert rendered.count("**Deliverables:**") == 1
    # The non-trex sections are untouched by the new line.
    for heading in ("### Velociraptor Mongoliensis (PPO", "### Velociraptor Mongoliensis (SAC", "### Brachiosaurus"):
        section = lambda text: text.split(heading, 1)[1].split("\n### ", 1)[0]  # noqa: E731
        assert section(rendered) == section(baseline)


# ── Website adapter pins (WS4, decisions D-A8, D-A10, D-A13) ──────────────
#
# The site's TypeScript is not imported by any Python test runner, and the
# Docusaurus build does not type-check, so the adapter's contract with the
# generated catalog is pinned here as SOURCE TEXT: a catalog key the site
# never declares, a schema bump the site does not guard, or a gate phrase
# the two renderers spell differently fails this file, which runs on every
# website/src/** PR.

WEBSITE_SRC = REPOSITORY_ROOT / "website" / "src"


def _website_source(relative_path: str) -> str:
    return (WEBSITE_SRC / relative_path).read_text(encoding="utf-8")


def _python_function_code(source: str, name: str) -> str:
    """The source of top-level function *name* with its docstring removed — the code the mirror pins read."""
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            body = node.body[1:] if ast.get_docstring(node) is not None else node.body
            return "\n".join(ast.get_source_segment(source, statement) or "" for statement in body)
    raise AssertionError(f"no top-level function {name!r}")


def _ts_block(source: str, header: str) -> str:
    """The text between *header*'s opening brace and its matching close brace."""
    start = source.index(header)
    open_brace = source.index("{", start)
    depth = 0
    for index in range(open_brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[open_brace + 1 : index]
    raise AssertionError(f"unterminated block after {header!r}")


def _ts_declared_keys(block: str) -> set[str]:
    """The property names declared at the top level of one TS object type."""
    depth = 0
    keys: set[str] = set()
    for line in block.splitlines():
        stripped = line.strip()
        if depth == 0:
            match = re.match(r"^([a-z][a-z0-9_]*)\??:", stripped)
            if match:
                keys.add(match.group(1))
        depth += stripped.count("{") + stripped.count("<") - stripped.count("}") - stripped.count(">")
    return keys


def test_website_adapter_pins_the_catalog_schema_version() -> None:
    """species.ts refuses any catalog schema but the one build_catalog emits (D-A8).

    The guard is the site's only defence against a regenerated JSON whose
    rows changed shape under an unchanged adapter, so its literal is pinned
    to the Python constant rather than to a number typed twice.
    """
    source = _website_source("data/species.ts")
    guards = re.findall(r"if \(catalog\.schema_version !== (\d+)\) throw", source)
    assert guards == [str(build_catalog()["schema_version"])]


def test_website_adapter_declares_every_exported_key(tmp_path: Path, monkeypatch: Any) -> None:
    """Every key the catalog exports per row is declared by the matching Raw* TS type, and vice versa.

    Two-sided: an exported key the adapter never declares is a field the
    site silently drops (the seven gate keys were, before WS4 — D-A13), and
    a declared key the catalog no longer exports is a field the site reads
    as undefined.  Deliverable rows are taken from a schema-4 fixture
    because the four committed ladder summaries publish none (D-A10); the
    result and provenance pins take the committed v2 rows AND that fixture,
    so the six-key provenance projection holds for the first schema-4
    summary committed, not only for the rows committed today.
    """
    source = _website_source("data/species.ts")
    catalog = build_catalog()
    species = catalog["species"]
    stage_rows = [stage for entry in species for stage in entry["stages"]]
    results = [result for entry in species for result in entry["historical_results"]]
    v4_result = _build_result_from(
        tmp_path,
        monkeypatch,
        _trex_v4_summary({"1": True, "2": True, "3": False}, target="3", primary="2", bundle_status="partial"),
    )
    videos = [stage["video"] for stage in stage_rows if stage["video"] is not None]
    assert videos, "at least one published video is needed to pin the video row"

    def exported(rows: list[dict[str, Any]]) -> set[str]:
        keys: set[str] = set()
        for row in rows:
            keys |= set(row)
        return keys

    stage_block = _ts_block(source, "interface RawStage ")
    result_block = _ts_block(source, "interface RawResult ")
    species_block = _ts_block(source, "interface RawSpecies ")
    expectations = {
        "RawCatalog": (_ts_block(source, "interface RawCatalog "), exported([catalog])),
        "RawSpecies": (species_block, exported(species)),
        "RawStage": (stage_block, exported(stage_rows)),
        "RawStage.advancement_gate": (
            _ts_block(stage_block, "advancement_gate:"),
            exported([stage["advancement_gate"] for stage in stage_rows]),
        ),
        "RawStage.video": (_ts_block(stage_block, "video:"), exported(videos)),
        "RawResult": (result_block, exported([*results, v4_result])),
        "RawResult.provenance": (
            _ts_block(result_block, "provenance:"),
            exported([result["provenance"] for result in [*results, v4_result]]),
        ),
        "RawResultStage": (
            _ts_block(source, "interface RawResultStage "),
            exported([stage for result in results for stage in result["stages"]]),
        ),
        "RawResultDeliverable": (
            _ts_block(source, "interface RawResultDeliverable "),
            exported(v4_result["deliverables"]),
        ),
        "RawHeadlineMetric": (
            _ts_block(source, "interface RawHeadlineMetric "),
            exported([metric for row in v4_result["deliverables"] for metric in row["headline"]]),
        ),
        "RawSuccessMetric": (
            _ts_block(source, "interface RawSuccessMetric "),
            exported([metric for entry in species for metric in entry["success_metrics"]]),
        ),
        "RawDeliverableMetric": (
            _ts_block(source, "interface RawDeliverableMetric "),
            exported([metric for entry in species for metric in entry["deliverable_metrics"]]),
        ),
    }
    mismatches = {
        name: (sorted(exported_keys - _ts_declared_keys(block)), sorted(_ts_declared_keys(block) - exported_keys))
        for name, (block, exported_keys) in expectations.items()
        if _ts_declared_keys(block) != exported_keys
    }
    assert mismatches == {}, f"(undeclared, stale) keys per Raw type: {mismatches}"


def test_website_gate_formatter_mirrors_python() -> None:
    """formatGate in SpeciesCatalog/index.tsx renders the same phrases, in the same branch order, as _format_advancement_gate.

    Both renderers are pinned against one phrase list, so a criterion added
    to one and not the other fails here whichever side moved.  The branch
    order pin (none/v1, then recovery_quality/v1, then task_success/v1, then
    the generic path) keeps the frozen-verdict sentence on the recovery row,
    the evidence-CSV sentence on the hunting row, and the consecutive-passes
    tail off both, on both sides.  The task_success rail is rendered as
    "reward rail" so the generic path's "reward ≥ " stays unique to it.
    """
    python_source = (REPOSITORY_ROOT / "environments/shared/species_catalog.py").read_text(encoding="utf-8")
    python_body = python_source.split("def _format_advancement_gate(", 1)[1].split("\ndef ", 1)[0]
    tsx_source = _website_source("components/SpeciesCatalog/index.tsx")
    tsx_body = "function formatGate(" + _ts_block(tsx_source, "function formatGate(")

    phrases = [
        "non-advancing pilot (gate_kind none/v1); never advances",
        " pending calibration (P5)",
        "recovery success LCB95 ≥ ",
        "paired Δ vs each required frozen null LCB95 ≥ ",
        "re-entry ≤ ",
        "-step dwell",
        "verdict from the frozen gate_resolution.json (post-stage; fail-closed when absent or stale)",
        "task success LCB95 ≥ ",
        "reward rail ≥ ",
        "verdict from the selected checkpoint's evaluation_selected.csv (post-stage; fail-closed when absent)",
        "reward ≥ ",
        "episode length ≥ ",
        "avg. velocity ≥ ",
        " m/s",
        "task success ≥ ",
        "full-horizon episodes ≥ ",
        "unsupported duty ≤ ",
        "unsupported duty 95% upper bound ≤ ",
        " episodes/evaluation",
        " consecutive passes",
    ]
    missing = {
        phrase: [side for side, body in (("python", python_body), ("tsx", tsx_body)) if phrase not in body]
        for phrase in phrases
    }
    assert {phrase: sides for phrase, sides in missing.items() if sides} == {}

    for body in (python_body, tsx_body):
        none_branch = body.index("none/v1")
        recovery_branch = body.index("recovery_quality/v1")
        task_success_branch = body.index("task_success/v1")
        # Anchored on a phrase ONLY the generic path emits: "reward ≥ " would
        # resolve inside a branch that rendered its rail with the same words.
        generic_path = body.index("avg. velocity ≥ ")
        assert none_branch < recovery_branch < task_success_branch < generic_path
        # The consecutive-passes tail belongs to the generic path only.
        assert body.index(" consecutive passes") > generic_path
        assert body.index("verdict from the frozen gate_resolution.json") < task_success_branch
        assert task_success_branch < body.index("verdict from the selected checkpoint") < generic_path
        # The hunting rail is "reward rail ≥ "; the plain criterion is generic.
        assert task_success_branch < body.index("reward rail ≥ ") < generic_path
        assert body.index("reward ≥ ") > body.index("reward rail ≥ ")


def test_website_replication_formatter_mirrors_python() -> None:
    """formatReplication in SpeciesCatalog/index.tsx renders the same phrases as _format_replication.

    ``{count} run(s) of {N} seed(s)`` plus ``provisional`` (plan §4.5, D-B11) is
    pinned on both sides so the README line and the site's Runs column cannot
    drift; ``headlineFor`` carries the same count-of-N phrase into the landing
    headline of a provisional primary.  The gate-formatter pin above keeps its
    own anchors: this test reads only the replication formatters.
    """
    python_source = (REPOSITORY_ROOT / "environments/shared/species_catalog.py").read_text(encoding="utf-8")
    python_body = _python_function_code(python_source, "_format_replication")
    tsx_source = _website_source("components/SpeciesCatalog/index.tsx")
    tsx_body = "\n".join(
        line
        for line in _ts_block(tsx_source, "function formatReplication(").splitlines()
        if not line.strip().startswith("//")
    )
    # The rendered fragments, read from the CODE of both renderers (the
    # Python docstring and the TSX comments are stripped above, so a phrase
    # only a comment still carries does not satisfy the pin).
    phrases = [" run", " of ", " seed", "; provisional"]
    missing = {
        phrase: [side for side, body in (("python", python_body), ("tsx", tsx_body)) if phrase not in body]
        for phrase in phrases
    }
    assert {phrase: sides for phrase, sides in missing.items() if sides} == {}
    # The singular/plural pair is decided the same way on both sides, once
    # for the runs and once for the seeds.
    assert python_body.count("== 1 else 's'") == 2
    assert tsx_body.count("=== 1 ? '' : 's'") == 2
    assert '"; provisional"' in python_body and "'; provisional'" in tsx_body
    assert "formatReplication(deliverable)" in tsx_source, "the Runs column renders the formatter"

    adapter = _website_source("data/species.ts")
    headline_body = adapter.split("export function headlineFor(", 1)[1].split("\nexport ", 1)[0]
    assert "(provisional, ${primary.replicationCount} of ${primary.certificationSeeds} seeds)" in headline_body


def test_index_page_keys_video_cards_by_stage_id() -> None:
    """The landing page keys and labels video cards by the manifest stage id/label, never the legacy number (D-A13).

    ``stage.number`` is null for a semantic-only stage (recovery), so a card
    keyed by it collided with its siblings and read "STAGE null"; the
    catalog's ``label`` is the canonical reference.  The hero headline comes
    from ``headlineFor`` so a schema-4 result headlines its certified
    primary deliverable and the ladder rows render exactly as before;
    ``stage3SuccessRate`` is read nowhere outside the adapter.
    """
    page = _website_source("pages/index.tsx")
    assert "key={stage.number}" not in page
    assert "STAGE {stage.number}" not in page
    assert "stage ${stage.number}" not in page
    assert "key={stage.id}" in page
    assert "STAGE {stage.label.toUpperCase()}" in page
    assert "stage ${stage.label}: ${stage.title}" in page

    import_block = page.split("from '@site/src/data/species';", 1)[0]
    assert "headlineFor" in import_block
    assert "headlineFor(result)" in page

    readers = sorted(
        path.relative_to(WEBSITE_SRC).as_posix()
        for path in WEBSITE_SRC.rglob("*.ts*")
        if "stage3SuccessRate" in path.read_text(encoding="utf-8")
    )
    assert readers == ["data/species.ts"]
