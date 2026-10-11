"""Tests for the six-heading stance certificate statistic, ``stance_quality_heading/v1`` (pure; no env).

Like ``test_stance_gate_v2.py``, every grid here is built from one real
:class:`StanceEpisodeMetrics` row -- the perfect synthetic statue trace of
``test_gait_stance_metrics.py`` -- with only the field a defect would move
edited, at the (seed, heading) cells a test names.  The grouped-seed
counterexamples are the design's (docs/HEADING_INVARIANCE_DESIGN_2026_10.md,
"Protocol and statistic"); the rolled certificate is in
``test_stance_heading_report.py``.
"""

from __future__ import annotations

import ast
import csv
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from environments.shared.curriculum import gate_schema, stance_heading_gate
from environments.shared.curriculum.gate_schema import (
    FLOOR_TRUTH_STANCE_GATE_KINDS,
    FROZEN_NULL_GATE_KINDS,
    GATE_KINDS,
    GATE_SCHEMA_VERSION,
    STANCE_GATE_KINDS,
    GateSchemaError,
    gate_config_view,
    validate_gate_config,
)
from environments.shared.curriculum.manager import thresholds_from_configs
from environments.shared.curriculum.recovery_gate import binomial_lcb
from environments.shared.curriculum.stance_gate_v2 import (
    STANCE_GATE_V2_KIND,
    STANCE_V2_PANEL_FIELDNAMES,
    STANCE_V2_REQUIRED_KEYS,
    STANCE_V2_THRESHOLD_KEYS,
    StanceV2Thresholds,
    StatueReference,
    read_stance_v2_panel,
)
from environments.shared.curriculum.stance_heading_gate import (
    HEADING_GRID_DEG,
    HEADING_PROTOCOL,
    STANCE_HEADING_GATE_KIND,
    STANCE_HEADING_PANEL_FIELDNAMES,
    STANCE_HEADING_PANEL_STAMP_COLUMNS,
    STANCE_HEADING_REPORT_SCHEMA,
    STANCE_HEADING_REQUIRED_KEYS,
    STANCE_HEADING_THRESHOLD_KEYS,
    HeadingEpisode,
    canonical_heading_deg,
    evaluate_stance_heading_gate,
    grid_refusals,
    heading_grid_cell,
    read_stance_heading_panel,
    statue_references,
    validate_heading_grid,
)
from environments.shared.gait.stance_metrics import (
    STANCE_METRIC_FIELDS,
    STANCE_METRIC_FOOT_FIELDS,
    StanceEpisodeMetrics,
    episode_stance_metrics,
)

from .test_gait_stance_metrics import statue_trace

HORIZON = 1000
SEEDS = tuple(range(3042, 3082))

#: A heading block: v2's keys at the floor map §5.4 candidate bars, the full-horizon rail, and the kind.
BLOCK: dict[str, Any] = {
    "gate_schema_version": GATE_SCHEMA_VERSION,
    "gate_kind": STANCE_HEADING_GATE_KIND,
    "min_eval_episodes": 40,
    "min_clean_stance_lcb": 0.80,
    "settle_steps": 200,
    "min_all_feet_support": 0.98,
    "max_touchdown_rate": 0.25,
    "max_window_displacement_m": 0.10,
    "min_foot_load_share": 0.30,
    "max_actuator_saturation_fraction": 0.10,
    "max_settle_airborne_substeps": 0,
    "max_settle_peak_floor_force_bw": 1.5,
    "max_sole_corner_lift_m": 0.006,
    "min_full_horizon_fraction": 0.95,
}

#: A pad near-miss: fails one non-hop criterion and nothing else.
NEAR_MISS = {"max_sole_corner_lift_m": 0.01}
#: A fall: the episode ends early.
FALL = {"length": 75}


def thresholds(**overrides: Any) -> StanceV2Thresholds:
    block = {**BLOCK, **overrides}
    return StanceV2Thresholds.from_curriculum({key: value for key, value in block.items() if value is not None})


@pytest.fixture(scope="module")
def clean() -> StanceEpisodeMetrics:
    """One statue episode's real metrics: clean on every criterion."""
    return episode_stance_metrics(statue_trace(HORIZON), settle_steps=200)


def grid(
    episode: StanceEpisodeMetrics,
    defects: "dict[tuple[int, float], dict[str, Any]] | None" = None,
    *,
    seeds: tuple[int, ...] = SEEDS,
    headings: tuple[float, ...] = HEADING_GRID_DEG,
) -> list[HeadingEpisode]:
    """Every (seed, heading) cell clean, except the cells *defects* edits (keyed by seed INDEX and heading)."""
    defects = defects or {}
    return [
        HeadingEpisode(seed, yaw, replace(episode, **defects.get((index, yaw), {})))
        for yaw in headings
        for index, seed in enumerate(seeds)
    ]


def judge(rows: list[HeadingEpisode], block: StanceV2Thresholds | None = None, **kwargs: Any):
    kwargs.setdefault("statue_by_heading", {})
    return evaluate_stance_heading_gate(
        rows, block or thresholds(), horizon=HORIZON, seeds=kwargs.pop("seeds", SEEDS), control_dt=0.01, **kwargs
    )


# ── the registry ─────────────────────────────────────────────────────────────


class TestRegistration:
    def test_the_schema_registers_the_kind_with_the_v2_key_sets(self):
        assert STANCE_HEADING_GATE_KIND == "stance_quality_heading/v1"
        assert STANCE_HEADING_REPORT_SCHEMA == "mesozoic.stance-heading-report/v1"
        assert HEADING_PROTOCOL == "whole_scene_yaw/v1"
        assert GATE_KINDS[STANCE_HEADING_GATE_KIND] == STANCE_HEADING_THRESHOLD_KEYS == STANCE_V2_THRESHOLD_KEYS
        assert gate_schema._REQUIRED_THRESHOLD_KEYS[STANCE_HEADING_GATE_KIND] == STANCE_HEADING_REQUIRED_KEYS
        assert STANCE_HEADING_REQUIRED_KEYS == STANCE_V2_REQUIRED_KEYS
        assert STANCE_HEADING_GATE_KIND in STANCE_GATE_KINDS
        assert FLOOR_TRUTH_STANCE_GATE_KINDS == {STANCE_GATE_V2_KIND, STANCE_HEADING_GATE_KIND}
        # The statue is rolled fresh at JUDGE, as for v2: not a freeze-then-roll kind; and no key changes meaning.
        assert STANCE_HEADING_GATE_KIND not in FROZEN_NULL_GATE_KINDS
        assert GATE_SCHEMA_VERSION == 1

    def test_registration_moves_no_digest_of_an_undeclaring_stage(self):
        """The kind adds no key, so every view projected without it -- each committed stage's -- is unchanged."""
        others = frozenset().union(*(keys for kind, keys in GATE_KINDS.items() if kind != STANCE_HEADING_GATE_KIND))
        assert gate_schema._ALL_THRESHOLD_KEYS == others
        from environments.shared.config import load_all_stages
        from environments.shared.stage_manifest import _CONFIGS_DIR

        for species in sorted(path.parent.name for path in _CONFIGS_DIR.glob("*/stages.toml")):
            for stage, cfg in load_all_stages(species).items():
                curriculum = cfg["curriculum_kwargs"]
                if curriculum.get("gate_kind") == STANCE_HEADING_GATE_KIND:
                    continue
                kind = curriculum.get("gate_kind")
                keys = GATE_KINDS[kind] if kind in GATE_KINDS else others
                expected = {key: curriculum[key] for key in sorted(keys) if key in curriculum}
                assert gate_config_view(curriculum)["thresholds"] == expected, (species, stage)

    def test_a_stance_moves_to_the_kind_by_its_gate_kind_alone(self):
        v2_block = {**BLOCK, "gate_kind": STANCE_GATE_V2_KIND}
        assert gate_config_view(BLOCK)["thresholds"] == gate_config_view(v2_block)["thresholds"]
        assert gate_config_view(BLOCK)["gate_kind"] == STANCE_HEADING_GATE_KIND
        assert validate_gate_config(1, BLOCK) == STANCE_HEADING_GATE_KIND

    def test_the_gate_module_is_pure(self):
        """stdlib, v2's statistic, recovery_gate and the pure gait row type only, as for v2."""
        tree = ast.parse(Path(stance_heading_gate.__file__).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported.add(("." * node.level) + (node.module or ""))
        assert imported == {
            "__future__",
            "csv",
            "math",
            "numbers",
            "collections.abc",
            "dataclasses",
            "pathlib",
            "typing",
            "..gait.stance_metrics",
            ".recovery_gate",
            ".stance_gate_v2",
        }

    def test_config_load_applies_the_floor_truth_bar_checks(self):
        with pytest.raises(GateSchemaError, match="max_sole_cop_fore_aft must be a number strictly between"):
            validate_gate_config(1, {**BLOCK, "max_sole_cop_fore_aft": 1.0})
        with pytest.raises(GateSchemaError, match="max_window_peak_floor_force_bw must be a number of at least 1"):
            validate_gate_config(1, {**BLOCK, "max_window_peak_floor_force_bw": 0.5})
        with pytest.raises(GateSchemaError, match=r"missing required threshold field\(s\) \['settle_steps'\]"):
            validate_gate_config(1, {key: value for key, value in BLOCK.items() if key != "settle_steps"})

    def test_config_load_refuses_the_report_only_probe_beside_the_certificate(self):
        """Decision 3 of 2026-10-11: the certificate replaces the probe on the stages that declare it."""
        with pytest.raises(GateSchemaError, match="remove stance_probe_spawn_yaw_deg"):
            validate_gate_config(1, {**BLOCK, "stance_probe_spawn_yaw_deg": [-90.0, -45.0, 45.0, 90.0]})
        # Under v2 the probe stays a diagnostic (the robot and the velociraptor keep it).
        v2_block = {**BLOCK, "gate_kind": STANCE_GATE_V2_KIND, "stance_probe_spawn_yaw_deg": [90.0]}
        assert validate_gate_config(1, v2_block) == STANCE_GATE_V2_KIND

    def test_the_manager_copies_the_v2_keys_and_refuses_the_kind_in_training(self, caplog):
        from environments.shared.curriculum.manager import CurriculumManager, StageThreshold

        fields = thresholds_from_configs({1: {"curriculum_kwargs": {**BLOCK, "required_consecutive": 1}}})
        assert fields[1]["gate_kind"] == STANCE_HEADING_GATE_KIND
        assert fields[1]["min_clean_stance_lcb"] == 0.80 and fields[1]["max_sole_corner_lift_m"] == 0.006
        assert StageThreshold(**fields[1]).stance_v2_thresholds() == thresholds(required_consecutive=1)
        manager = CurriculumManager(species="velociraptor", stage_thresholds=fields)
        with caplog.at_level("ERROR", logger="environments.shared.curriculum.manager"):
            for _ in range(5):
                assert not manager.should_advance([1e9] * 40, [1000.0] * 40)
        assert manager.summary()["consecutive_passes"][1] == 0
        assert f"declares gate_kind {STANCE_HEADING_GATE_KIND!r}" in caplog.text and "cannot evaluate" in caplog.text


# ── the grid ─────────────────────────────────────────────────────────────────


class TestGrid:
    def test_the_protocol_grid_is_the_six_headings_in_order(self):
        assert HEADING_GRID_DEG == (-90.0, -45.0, 0.0, 45.0, 90.0, 180.0)
        assert validate_heading_grid(list(HEADING_GRID_DEG)) == HEADING_GRID_DEG
        assert heading_grid_cell(HEADING_GRID_DEG) == "[-90.0, -45.0, 0.0, 45.0, 90.0, 180.0]"

    def test_canonicalisation_maps_minus_180_to_180_and_wraps(self):
        assert canonical_heading_deg(-180) == 180.0
        assert canonical_heading_deg(180.0) == 180.0
        assert canonical_heading_deg(540) == 180.0
        assert canonical_heading_deg(360) == 0.0
        assert math.copysign(1.0, canonical_heading_deg(-360.0)) == 1.0  # no -0.0
        assert canonical_heading_deg(-45) == -45.0 and canonical_heading_deg(315) == -45.0
        assert canonical_heading_deg(-270) == 90.0
        assert validate_heading_grid([-90, -45, 0, 45, 90, -180]) == HEADING_GRID_DEG

    @pytest.mark.parametrize(
        "angles",
        [
            [-90.0, -45.0, 0.0, 45.0, 90.0, 180.0, -180.0],
            [-90.0, -45.0, 0.0, 360.0, 45.0, 90.0, 180.0],
            [-90.0, -45.0, 0.0, 45.0, 45.0, 90.0, 180.0],
        ],
        ids=["180/-180", "0/360", "45/45"],
    )
    def test_duplicate_or_synonymous_headings_are_refused(self, angles):
        with pytest.raises(ValueError, match="which are one heading"):
            validate_heading_grid(angles)

    @pytest.mark.parametrize(
        ("angles", "message"),
        [
            ([-90.0, -45.0, 0.0, 45.0, 90.0], r"missing \+180"),
            ([-90.0, -45.0, 0.0, 45.0, 90.0, 180.0, 30.0], r"extra \+30"),
            ([-45.0, -90.0, 0.0, 45.0, 90.0, 180.0], "reordered"),
            ([-90.0, 0.0, 90.0, 180.0], r"missing -45, \+45"),
        ],
        ids=["missing", "extra", "reordered", "cheaper-variant"],
    )
    def test_a_grid_other_than_the_protocols_is_refused(self, angles, message):
        with pytest.raises(ValueError, match=message):
            validate_heading_grid(angles)

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, True, "45", None], ids=repr)
    def test_non_finite_or_non_numeric_headings_are_refused(self, value):
        with pytest.raises(ValueError, match="must be a finite number of degrees"):
            canonical_heading_deg(value)
        with pytest.raises(ValueError, match="must be a finite number of degrees"):
            validate_heading_grid([-90.0, -45.0, value, 45.0, 90.0, 180.0])

    def test_grid_refusals_names_missing_duplicated_and_off_grid_cells(self, clean):
        rows = grid(clean)
        assert grid_refusals(rows, seeds=SEEDS, grid=HEADING_GRID_DEG) == []
        (missing,) = grid_refusals(rows[1:], seeds=SEEDS, grid=HEADING_GRID_DEG)
        assert missing.startswith("1 of 240 (seed, heading) cells are missing (-90 deg: 1)")
        (duplicated,) = grid_refusals([*rows, rows[0]], seeds=SEEDS, grid=HEADING_GRID_DEG)
        assert "1 (seed, heading) pair(s) appear more than once (seed 3042 at -90 deg)" in duplicated
        off = [*rows, HeadingEpisode(7042, 0.0, clean), HeadingEpisode(3042, 30.0, clean)]
        (stray,) = grid_refusals(off, seeds=SEEDS, grid=HEADING_GRID_DEG)
        assert "2 row(s) sit off the grid" in stray and "seed 7042 at +0 deg" in stray and "+30 deg" in stray
        # -180 is the 180 cell, so a row recorded at -180 completes the grid rather than sitting off it.
        relabelled = [
            HeadingEpisode(row.seed, -180.0 if row.yaw_deg == 180.0 else row.yaw_deg, row.metrics) for row in rows
        ]
        assert grid_refusals(relabelled, seeds=SEEDS, grid=HEADING_GRID_DEG) == []
        assert "repeats a seed" in grid_refusals(rows, seeds=(*SEEDS, 3042), grid=HEADING_GRID_DEG)[0]


# ── the grouped statistic ────────────────────────────────────────────────────


class TestGroupedStatistic:
    def test_a_clean_grid_passes_at_the_bound_of_its_seed_groups(self, clean):
        result = judge(grid(clean))
        assert result.passed and result.failures == ()
        assert (result.n_seeds, result.n_clean_seeds) == (40, 40)
        assert result.clean_seed_lcb == pytest.approx(0.9278, abs=1e-4)
        assert [yaw for yaw, _ in result.per_heading] == list(HEADING_GRID_DEG)
        assert result.protocol == HEADING_PROTOCOL and result.grid_deg == HEADING_GRID_DEG
        assert result.pooled_clean == 240

    def test_a_seed_is_clean_only_if_clean_at_every_heading(self, clean):
        result = judge(grid(clean, {(5, 45.0): NEAR_MISS, (5, 180.0): NEAR_MISS, (9, -90.0): NEAR_MISS}))
        assert result.n_clean_seeds == 38
        assert result.seed_failures == ((3047, (45.0, 180.0)), (3051, (-90.0,)))
        assert result.heading_result(45.0).n_clean == 39
        assert result.heading_result(-180.0).n_clean == 39  # canonical lookup
        assert result.heading_result(30.0) is None

    def test_37_of_40_seed_groups_pass_and_36_fail(self, clean):
        """The design's sizing: LCB95(37/40) = 0.8174 clears 0.80; LCB95(36/40) = 0.7856 does not."""
        three = {(index, 0.0): NEAR_MISS for index in range(3)}
        passing = judge(grid(clean, three))
        assert passing.passed and passing.n_clean_seeds == 37
        assert passing.clean_seed_lcb == pytest.approx(0.8174, abs=1e-4)
        four = {(index, -45.0): NEAR_MISS for index in range(4)}
        failing = judge(grid(clean, four))
        assert not failing.passed and failing.clean_seed_lcb == pytest.approx(0.7856, abs=1e-4)
        assert failing.failures[0] == (
            "clean_seed_lcb 0.7856 < 0.8000 (36/40 seeds clean at every heading of -90, -45, +0, +45, +90, +180)"
        )
        # The heading that lost the four seeds fails v2 too, by name.
        assert "heading -45: clean_stance_lcb 0.7856 < 0.8000 (36/40 episodes clean)" in failing.failures

    def test_the_pooled_rotations_counterexample_is_refused(self, clean):
        """One different failing seed at each heading: pooled 234/240 reads 0.9513; only 34/40 seeds stand at all six."""
        diagonal = {(index, yaw): NEAR_MISS for index, yaw in enumerate(HEADING_GRID_DEG)}
        result = judge(grid(clean, diagonal))
        assert result.pooled_clean == 234
        assert result.pooled_lcb == pytest.approx(0.9513, abs=1e-4) and result.pooled_lcb > 0.95
        assert result.n_clean_seeds == 34
        assert result.clean_seed_lcb == pytest.approx(0.7253, abs=1e-4)
        assert not result.passed
        assert result.failures == (
            "clean_seed_lcb 0.7253 < 0.8000 (34/40 seeds clean at every heading of -90, -45, +0, +45, +90, +180)",
        )
        pooled = result.as_dict()["pooled_episodes"]
        assert pooled["n_clean"] == 234 and pooled["n_episodes"] == 240 and "NOT the statistic" in pooled["note"]

    def test_per_heading_summaries_alone_would_pass_the_counterexample(self, clean):
        diagonal = {(index, yaw): NEAR_MISS for index, yaw in enumerate(HEADING_GRID_DEG)}
        result = judge(grid(clean, diagonal))
        assert all(heading.passed and heading.n_clean == 39 for _, heading in result.per_heading)
        assert not result.passed

    def test_each_heading_applies_the_full_v2_decision(self, clean):
        """Three falls at +90 clear the grouped bound (37/40) but not that heading's 0.95 full-horizon rail."""
        falls = {(index, 90.0): FALL for index in range(3)}
        result = judge(grid(clean, falls))
        assert result.n_clean_seeds == 37 and result.clean_seed_lcb == pytest.approx(0.8174, abs=1e-4)
        assert not result.passed
        assert result.failures == ("heading +90: full_horizon_fraction 0.9250 < 0.9500",)

    def test_the_hop_or_fall_rail_is_per_heading(self, clean):
        hops = {(0, -45.0): {"all_feet_support": 0.5}, (1, -45.0): {"all_feet_support": 0.5}}
        result = judge(grid(clean, hops), thresholds(max_hop_or_fall_episodes=1))
        assert result.n_clean_seeds == 38 and result.clean_seed_lcb >= 0.80
        assert not result.passed
        (failure,) = result.failures
        assert failure.startswith("heading -45: hop_or_fall_episodes 2 > 1")
        # One hop at each of two headings stays inside each heading's rail.
        spread = {(0, -45.0): {"all_feet_support": 0.5}, (1, 45.0): {"all_feet_support": 0.5}}
        assert judge(grid(clean, spread), thresholds(max_hop_or_fall_episodes=1)).passed

    def test_each_heading_is_judged_against_its_own_statue(self, clean):
        block = thresholds(min_foot_load_share_statue_ratio=0.8)
        statues = statue_references(grid(clean), horizon=HORIZON)
        assert set(statues) == set(HEADING_GRID_DEG)
        assert judge(grid(clean), block, statue_by_heading=statues).passed
        without_45 = {yaw: reference for yaw, reference in statues.items() if yaw != 45.0}
        result = judge(grid(clean), block, statue_by_heading=without_45)
        assert not result.passed and result.n_clean_seeds == 0
        assert all(
            failure.startswith("heading +45: ") or failure.startswith("clean_seed_lcb") for failure in result.failures
        )
        assert result.heading_result(0.0).passed and not result.heading_result(45.0).passed
        # A statue that leaned on one foot at one heading moves that heading's reference only.
        leaning = [
            HeadingEpisode(row.seed, row.yaw_deg, replace(row.metrics, foot_load_share=(0.9, 0.1)))
            if row.yaw_deg == -90.0
            else row
            for row in grid(clean)
        ]
        references = statue_references(leaning, horizon=HORIZON)
        assert references[-90.0].foot_load_share == (0.9, 0.1)
        assert references[0.0] == statues[0.0]

    def test_yaw_zero_passing_does_not_pass_the_gate(self, clean):
        """Seed 52's shape (20261009_155723): 0, +45, +90 and 180 stand; -90 and -45 fall on every seed."""
        falls = {(index, yaw): FALL for index in range(40) for yaw in (-90.0, -45.0)}
        result = judge(grid(clean, falls))
        assert result.heading_result(0.0).passed and result.heading_result(45.0).passed
        assert not result.passed and result.n_clean_seeds == 0
        named = {failure.split(":", 1)[0] for failure in result.failures}
        assert {"heading -90", "heading -45"} <= named
        assert not {"heading +0", "heading +45", "heading +90", "heading +180"} & named

    def test_an_incomplete_or_duplicated_grid_fails_closed(self, clean):
        rows = grid(clean)
        missing_heading = [row for row in rows if row.yaw_deg != 180.0]
        result = judge(missing_heading)
        assert not result.passed and result.per_heading == () and result.n_clean_seeds == 0
        assert result.failures[0].startswith("grid: 40 of 240 (seed, heading) cells are missing (+180 deg: 40)")
        assert not judge([*rows, rows[7]]).passed
        assert not judge(rows, grid=[-90.0, -45.0, 0.0, 45.0, 90.0]).passed
        reordered = judge(rows, grid=[180.0, -90.0, -45.0, 0.0, 45.0, 90.0])
        assert not reordered.passed and "reordered" in reordered.failures[0]

    def test_an_under_powered_seed_block_fails(self, clean):
        seeds = SEEDS[:20]
        result = judge(grid(clean, seeds=seeds), seeds=seeds)
        assert not result.passed and result.n_clean_seeds == 20
        assert result.failures[0].startswith("n_seeds 20 < min_eval_episodes 40")
        assert all("n_episodes 20 < min_eval_episodes 40" in failure for failure in result.failures[1:7])

    def test_what_would_have_to_be_deleted_for_the_grouped_bound_to_stop_being_consulted(self, clean, monkeypatch):
        diagonal = {(index, yaw): NEAR_MISS for index, yaw in enumerate(HEADING_GRID_DEG)}
        calls: list[tuple[int, int]] = []

        def recorded(k: int, n: int, *args: Any, **kwargs: Any) -> float:
            calls.append((k, n))
            return binomial_lcb(k, n, *args, **kwargs)

        monkeypatch.setattr(stance_heading_gate, "binomial_lcb", recorded)
        result = judge(grid(clean, diagonal))
        assert (34, 40) in calls and not result.passed
        monkeypatch.setattr(stance_heading_gate, "binomial_lcb", lambda k, n, *a, **kw: 1.0)
        assert judge(grid(clean, diagonal)).passed  # only the grouped bound refused it

    def test_the_result_serialises_without_nan_surprises(self, clean):
        payload = judge(grid(clean, {(3, 90.0): NEAR_MISS})).as_dict()
        assert payload["n_clean_seeds"] == 39 and payload["seed_failures"] == [{"seed": 3045, "headings_deg": [90.0]}]
        assert [row["yaw_deg"] for row in payload["per_heading"]] == list(HEADING_GRID_DEG)
        assert payload["per_heading"][4]["n_clean"] == 39


# ── the grid CSV contract ────────────────────────────────────────────────────

_STAMPS = {
    "checkpoint_sha256": "sha256:" + "a" * 64,
    "normalization_sha256": "sha256:" + "b" * 64,
    "task_sha256": "sha256:" + "d" * 64,
    "measurement_version": "floor-truth/v1",
    "measurement_sha256": "sha256:" + "c" * 64,
    "measurement_definition_sha256": "sha256:" + "e" * 64,
    "gate_sha256": "sha256:" + "f" * 64,
    "protocol": HEADING_PROTOCOL,
    "heading_grid_deg": heading_grid_cell(HEADING_GRID_DEG),
    "seed_start": 3042,
    "n_seeds": 2,
    "horizon": HORIZON,
    "control_dt": repr(0.01),
    "policy_deterministic": True,
}


def _write_grid(path: Path, episode: StanceEpisodeMetrics, *, seeds: int = 2, **cells: Any) -> Path:
    """A policy row and a statue row per (seed, heading), each heading stamping its own statue reference."""
    rows = []
    for yaw in HEADING_GRID_DEG:
        for role in ("policy", "statue"):
            for index in range(seeds):
                row: dict[str, Any] = {key: "" for key in STANCE_HEADING_PANEL_FIELDNAMES}
                row.update(episode.as_row())
                row.update(_STAMPS)
                row.update(
                    {
                        "role": role,
                        "yaw_deg": repr(yaw),
                        "episode": index,
                        "panel_seed": 3042 + index,
                        "reached_horizon": True,
                        "clean": True if role == "policy" else "",
                        "statue_n_episodes": seeds,
                        "statue_n_full_horizon": seeds,
                        "statue_mean_reward": repr(episode.reward + yaw),
                        "statue_foot_load_share": "[0.5, 0.5]",
                    }
                )
                row.update({key: (value(row) if callable(value) else value) for key, value in cells.items()})
                rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(STANCE_HEADING_PANEL_FIELDNAMES))
        writer.writeheader()
        writer.writerows(rows)
    return path


class TestPanelEvidence:
    def test_the_columns_hold_every_metric_the_grid_and_the_stamps(self):
        assert STANCE_HEADING_PANEL_FIELDNAMES[:2] == ("role", "yaw_deg")
        assert set(STANCE_V2_PANEL_FIELDNAMES) < set(STANCE_HEADING_PANEL_FIELDNAMES)
        assert set(STANCE_METRIC_FIELDS) | set(STANCE_METRIC_FOOT_FIELDS) <= set(STANCE_HEADING_PANEL_FIELDNAMES)
        assert set(STANCE_HEADING_PANEL_STAMP_COLUMNS) <= set(STANCE_HEADING_PANEL_FIELDNAMES)
        assert len(set(STANCE_HEADING_PANEL_FIELDNAMES)) == len(STANCE_HEADING_PANEL_FIELDNAMES)

    def test_a_panel_reads_back_exactly(self, tmp_path, clean):
        evidence = read_stance_heading_panel(_write_grid(tmp_path / "grid.csv", clean))
        assert len(evidence.policy) == len(evidence.statue) == 12
        assert [(row.seed, row.yaw_deg) for row in evidence.policy[:3]] == [(3042, -90.0), (3043, -90.0), (3042, -45.0)]
        assert all(row.metrics.as_row() == clean.as_row() for row in (*evidence.policy, *evidence.statue))
        assert evidence.policy_clean == (True,) * 12 and evidence.policy_reached_horizon == (True,) * 12
        assert evidence.statue_reached_horizon == (True,) * 12
        assert evidence.statue_by_heading[45.0] == StatueReference(2, 2, clean.reward + 45.0, (0.5, 0.5))
        assert evidence.heading_grid_deg == HEADING_GRID_DEG and evidence.protocol == HEADING_PROTOCOL
        assert (evidence.seed_start, evidence.n_seeds, evidence.horizon) == (3042, 2, HORIZON)
        assert evidence.control_dt == 0.01 and evidence.policy_deterministic is True
        assert evidence.gate_sha256 == "sha256:" + "f" * 64 and evidence.task_sha256 == "sha256:" + "d" * 64
        assert evidence.absent_metrics == ()

    def test_a_v2_panel_file_is_refused_on_its_columns(self, tmp_path, clean):
        path = tmp_path / "v2.csv"
        with path.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=list(STANCE_V2_PANEL_FIELDNAMES))
            writer.writeheader()
            writer.writerow({**{key: "" for key in STANCE_V2_PANEL_FIELDNAMES}, **clean.as_row()})
        with pytest.raises(ValueError, match=r"is not a stance_quality_heading/v1 panel.*'role', 'yaw_deg'"):
            read_stance_heading_panel(path)

    def test_a_heading_panel_file_is_refused_by_the_v2_reader(self, tmp_path, clean):
        with pytest.raises(ValueError, match="records a yaw_deg column, so it is a six-heading certificate's grid"):
            read_stance_v2_panel(_write_grid(tmp_path / "grid.csv", clean))

    @pytest.mark.parametrize(
        ("cells", "message"),
        [
            ({"gate_sha256": lambda row: f"sha256:{row['panel_seed']}"}, "mixes gate_sha256"),
            ({"protocol": lambda row: "whole_scene_yaw/v0" if row["yaw_deg"] == "180.0" else HEADING_PROTOCOL}, "mixes protocol"),
            ({"checkpoint_sha256": lambda row: f"sha256:{row['role']}"}, "mixes checkpoint_sha256"),
            ({"statue_mean_reward": lambda row: repr(float(row["panel_seed"]))}, "mixes statue_mean_reward values across its rows at heading"),
            ({"role": lambda row: "teacher" if row["panel_seed"] == 3043 else row["role"]}, "has role 'teacher'"),
            ({"yaw_deg": lambda row: "north" if row["panel_seed"] == 3043 else row["yaw_deg"]}, "unparsable yaw_deg"),
            ({"clean": lambda row: True}, "is a statue row with a classification"),
            ({"clean": lambda row: "maybe" if row["role"] == "policy" else ""}, "unparsable clean"),
            ({"policy_deterministic": "perhaps"}, "unparsable policy_deterministic"),
            ({"control_dt": "nan"}, "non-finite control_dt"),
            ({"n_seeds": "two"}, "unparsable n_seeds"),
            ({"length": ""}, "not a whole number"),
        ],
        ids=[
            "gate", "protocol", "handoff", "statue-within-heading", "role", "yaw", "statue-classified",
            "clean", "deterministic", "control-dt", "n-seeds", "length",
        ],
    )  # fmt: skip
    def test_every_unreadable_panel_is_refused(self, tmp_path, clean, cells, message):
        with pytest.raises(ValueError, match=message):
            read_stance_heading_panel(_write_grid(tmp_path / "grid.csv", clean, **cells))

    def test_mixed_stamps_are_refused(self, tmp_path, clean):
        grid_mixed = {"heading_grid_deg": lambda row: "[0.0]" if row["panel_seed"] == 3042 else row["heading_grid_deg"]}
        with pytest.raises(ValueError, match="mixes heading_grid_deg"):
            read_stance_heading_panel(_write_grid(tmp_path / "grid.csv", clean, **grid_mixed))
        with pytest.raises(ValueError, match="mixes seed_start"):
            read_stance_heading_panel(
                _write_grid(tmp_path / "grid.csv", clean, seed_start=lambda row: 3042 + (row["role"] == "statue"))
            )

    def test_an_empty_panel_is_refused(self, tmp_path, clean):
        with pytest.raises(ValueError, match="holds no episode rows"):
            read_stance_heading_panel(_write_grid(tmp_path / "grid.csv", clean, seeds=0))

    def test_statue_rows_reduce_to_the_recorded_reference_per_heading(self, tmp_path, clean):
        """The reader returns the stamps; the statue rows re-reduce per heading, and a consumer compares them."""
        evidence = read_stance_heading_panel(_write_grid(tmp_path / "grid.csv", clean))
        reduced = statue_references(evidence.statue, horizon=HORIZON)
        assert reduced[0.0] == StatueReference(2, 2, clean.reward, tuple(clean.foot_load_share))
        # The fixture stamps reward + yaw at each heading: only 0 deg's stamp is what its rows reduce to.
        assert evidence.statue_by_heading[0.0].mean_reward == reduced[0.0].mean_reward
        assert evidence.statue_by_heading[90.0].mean_reward != reduced[90.0].mean_reward
