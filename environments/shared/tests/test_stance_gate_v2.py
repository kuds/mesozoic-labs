"""Tests for the floor-truth stance gate statistic, ``stance_quality/v2`` (pure; no env).

Every panel here is built from one real :class:`StanceEpisodeMetrics` row:
the metrics of the perfect synthetic statue trace of
``test_gait_stance_metrics.py`` (every leg loaded, flat soles, no motion,
zero action), measured with a 200-step settle window over a 1000-step
episode.  Each hack test edits only the field the defect would move --
"this defect, and nothing else" -- so the criterion that refuses it is the
one under test.  The real-env statue-versus-scripted-hack panel is in
``test_stance_gate_v2_report.py``.
"""

from __future__ import annotations

import ast
import csv
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from environments.shared.curriculum import gate_schema, stance_gate_v2
from environments.shared.curriculum.gate_schema import (
    FROZEN_NULL_GATE_KINDS,
    GATE_KINDS,
    GATE_SCHEMA_VERSION,
    STANCE_GATE_KINDS,
    gate_config_view,
)
from environments.shared.curriculum.recovery_gate import binomial_lcb
from environments.shared.curriculum.stance_gate_v2 import (
    CRITERION_BAR_RANGES,
    EPISODE_CRITERIA,
    STANCE_GATE_V2_KIND,
    STANCE_V2_PANEL_FIELDNAMES,
    STANCE_V2_REQUIRED_KEYS,
    STANCE_V2_THRESHOLD_KEYS,
    StanceV2Thresholds,
    StatueReference,
    classify_stance_episode,
    evaluate_stance_v2_gate,
    read_stance_v2_panel,
    statue_reference,
    unrecorded_criteria,
)
from environments.shared.gait.stance_metrics import (
    STANCE_METRIC_FIELDS,
    STANCE_METRIC_FOOT_FIELDS,
    STANCE_METRIC_LATER_FIELDS,
    StanceEpisodeMetrics,
    episode_stance_metrics,
)

from .test_gait_stance_metrics import statue_trace

HORIZON = 1000

#: A complete v2 block at the floor map §5.4 candidate bars (trex flavour:
#: the box-pad flatness keys), as a stage's [curriculum] would declare it.
BLOCK: dict[str, Any] = {
    "gate_schema_version": GATE_SCHEMA_VERSION,
    "gate_kind": STANCE_GATE_V2_KIND,
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
    "min_sole_contacts": 1.5,
    "max_sole_tilt_deg": 4.0,
}


def thresholds(**overrides: Any) -> StanceV2Thresholds:
    block = {**BLOCK, **overrides}
    return StanceV2Thresholds.from_curriculum({key: value for key, value in block.items() if value is not None})


@pytest.fixture(scope="module")
def clean() -> StanceEpisodeMetrics:
    """One statue episode's real metrics: clean on every criterion."""
    return episode_stance_metrics(statue_trace(HORIZON), settle_steps=200)


def panel(episode: StanceEpisodeMetrics, n_clean: int, n: int = 40, **defect: Any) -> list[StanceEpisodeMetrics]:
    """``n_clean`` clean episodes, then ``n - n_clean`` carrying *defect*."""
    return [episode] * n_clean + [replace(episode, **defect)] * (n - n_clean)


def reason_keys(reasons: tuple[str, ...]) -> set[str]:
    return {reason.split(":", 1)[0] for reason in reasons}


# ── the registry ─────────────────────────────────────────────────────────────


class TestRegistration:
    def test_the_schema_registers_the_gate_modules_own_key_sets(self):
        assert GATE_KINDS[STANCE_GATE_V2_KIND] is STANCE_V2_THRESHOLD_KEYS
        assert gate_schema._REQUIRED_THRESHOLD_KEYS[STANCE_GATE_V2_KIND] is STANCE_V2_REQUIRED_KEYS
        assert STANCE_V2_REQUIRED_KEYS == {
            "min_eval_episodes",
            "min_clean_stance_lcb",
            "settle_steps",
            "min_all_feet_support",
            "max_touchdown_rate",
            "max_window_displacement_m",
            "min_foot_load_share",
            "max_actuator_saturation_fraction",
            "max_settle_airborne_substeps",
            "max_settle_peak_floor_force_bw",
        }
        assert STANCE_V2_THRESHOLD_KEYS - STANCE_V2_REQUIRED_KEYS == {
            "min_full_horizon_fraction",
            "min_avg_reward",
            "min_avg_reward_statue_ratio",
            "min_foot_load_share_statue_ratio",
            "min_foot_load_share_windowed",
            "max_foot_contact_fraction",
            "max_phantom_support_fraction",
            "max_nonfoot_load_fraction",
            "max_settle_stance_width_change_m",
            "min_support_geom_duty",
            "min_support_geom_coverage",
            "max_sole_tilt_deg",
            "max_sole_tilt_excess_deg",
            "max_sole_corner_lift_m",
            "min_sole_contacts",
            "max_episode_yaw_change_deg",
            "max_settle_touchdowns",
            "max_sole_cop_fore_aft",
            "max_hop_or_fall_episodes",
            "max_window_airborne_substeps",
            "max_window_peak_floor_force_bw",
            "required_consecutive",
        }

    def test_registration_changes_neither_the_schema_version_nor_the_frozen_null_set(self):
        """A new kind is not a key-meaning change (GATE_SCHEMA_VERSION stays 1), and the statue is rolled
        fresh at JUDGE, so v2 must not join the notebook's freeze-then-roll chain."""
        assert GATE_SCHEMA_VERSION == 1
        assert STANCE_GATE_V2_KIND not in FROZEN_NULL_GATE_KINDS
        assert STANCE_GATE_KINDS == {"stance_quality/v1", STANCE_GATE_V2_KIND}

    def test_every_criterion_reads_a_scalar_metric_field_and_every_key_is_numeric_in_the_view(self):
        assert {metric for _, metric, _ in EPISODE_CRITERIA} <= set(STANCE_METRIC_FIELDS)
        assert {key for key, _, _ in EPISODE_CRITERIA} <= STANCE_V2_THRESHOLD_KEYS
        view = gate_config_view(BLOCK)["thresholds"]
        assert set(view) == set(BLOCK) - {"gate_kind", "gate_schema_version"}
        assert all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in view.values())

    def test_the_gate_module_is_pure(self):
        """stdlib, recovery_gate and the pure gait row type only: reporting and result_bundle both import it."""
        tree = ast.parse(Path(stance_gate_v2.__file__).read_text(encoding="utf-8"))
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
        }


# ── reading the thresholds ───────────────────────────────────────────────────


class TestFromCurriculum:
    @pytest.mark.parametrize("key", sorted(STANCE_V2_REQUIRED_KEYS))
    def test_a_missing_required_key_is_named(self, key):
        with pytest.raises(ValueError, match=rf"threshold {key} is missing"):
            StanceV2Thresholds.from_curriculum({k: v for k, v in BLOCK.items() if k != key})

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), "0.98", True, None], ids=repr)
    def test_a_non_finite_or_non_numeric_bar_is_refused(self, value):
        with pytest.raises(ValueError, match="min_all_feet_support"):
            StanceV2Thresholds.from_curriculum({**BLOCK, "min_all_feet_support": value})

    def test_a_declared_optional_key_must_be_numeric_too(self):
        with pytest.raises(ValueError, match="max_foot_contact_fraction"):
            StanceV2Thresholds.from_curriculum({**BLOCK, "max_foot_contact_fraction": None})

    @pytest.mark.parametrize(
        ("key", "value"),
        [("min_eval_episodes", 0), ("min_eval_episodes", 39.5), ("settle_steps", -1), ("settle_steps", 2.5)],
    )
    def test_counts_must_be_whole_numbers(self, key, value):
        with pytest.raises(ValueError, match=rf"{key} must be a whole number"):
            StanceV2Thresholds.from_curriculum({**BLOCK, key: value})

    @pytest.mark.parametrize("value", [0.0, -0.1, 1.0, 1.5])
    def test_the_bar_must_lie_strictly_inside_zero_one(self, value):
        """0 certifies a 0/40 panel (binomial_lcb(0, n) = 0); no 40-panel bounds at 1."""
        assert binomial_lcb(0, 40) == 0.0
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            StanceV2Thresholds.from_curriculum({**BLOCK, "min_clean_stance_lcb": value})

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("max_window_airborne_substeps", -1),
            ("max_window_airborne_substeps", -0.5),
            ("max_window_peak_floor_force_bw", 0.0),
            ("max_window_peak_floor_force_bw", 0.9),
        ],
    )
    def test_the_window_hop_pair_refuses_a_bar_no_episode_can_meet(self, key, value):
        """A negative substep count, or a peak under the body weight a supported window averages."""
        with pytest.raises(ValueError, match=rf"{key} must be at least"):
            StanceV2Thresholds.from_curriculum({**BLOCK, key: value})

    def test_the_window_hop_pair_reads_at_its_floors_and_is_none_undeclared(self):
        read = thresholds(max_window_airborne_substeps=0, max_window_peak_floor_force_bw=1.0)
        assert (read.max_window_airborne_substeps, read.max_window_peak_floor_force_bw) == (0.0, 1.0)
        with pytest.raises(ValueError, match="max_window_airborne_substeps is not numeric"):
            StanceV2Thresholds.from_curriculum({**BLOCK, "max_window_airborne_substeps": "30"})
        undeclared = thresholds()
        assert undeclared.max_window_airborne_substeps is None and undeclared.max_window_peak_floor_force_bw is None
        assert not {"max_window_airborne_substeps", "max_window_peak_floor_force_bw"} & set(undeclared.declared())

    @pytest.mark.parametrize("value", [0.0, -0.2, 1.0, 1.25])
    def test_the_pad_cop_bar_must_lie_strictly_inside_zero_one(self, value):
        """A fraction of the half-length: 0 refuses every loaded pad and a pad on its edge reads 1."""
        assert ("max_sole_cop_fore_aft", 0.0, 1.0) in [entry[:3] for entry in CRITERION_BAR_RANGES]
        with pytest.raises(ValueError, match="max_sole_cop_fore_aft must lie strictly between 0 and 1"):
            StanceV2Thresholds.from_curriculum({**BLOCK, "max_sole_cop_fore_aft": value})
        assert thresholds(max_sole_cop_fore_aft=0.8).max_sole_cop_fore_aft == 0.8
        assert thresholds().max_sole_cop_fore_aft is None and "max_sole_cop_fore_aft" not in thresholds().declared()

    def test_an_undeclared_optional_criterion_is_none_never_a_permissive_default(self):
        read = thresholds()
        assert read.max_foot_contact_fraction is None
        assert read.min_avg_reward is None
        assert not read.declares_statue_criteria()
        assert set(read.declared()) == set(BLOCK) - {"gate_kind", "gate_schema_version"}
        assert thresholds(min_avg_reward_statue_ratio=0.6).declares_statue_criteria()

    @pytest.mark.parametrize(("dt", "grace"), [(0.01, 10), (0.02, 5)])
    def test_a_settle_window_inside_the_spawn_grace_is_refused(self, dt, grace):
        thresholds(settle_steps=grace + 1).validate_settle_window(dt)
        with pytest.raises(ValueError, match=rf"{grace}-step spawn grace"):
            thresholds(settle_steps=grace).validate_settle_window(dt)


# ── classifying an episode ───────────────────────────────────────────────────


class TestClassification:
    def test_the_statue_is_clean(self, clean):
        assert classify_stance_episode(clean, thresholds(), horizon=HORIZON) == ()

    @pytest.mark.parametrize(
        ("defect", "criteria"),
        [
            # The hopper: airborne after the spawn grace, and the landing's impact.
            (
                {"settle_airborne_substeps": 22.0, "settle_peak_floor_force_bw": 2.28},
                {"max_settle_airborne_substeps", "max_settle_peak_floor_force_bw"},
            ),
            # The prop leg: a pad rolled onto its edge (trex 20260930).
            (
                {"max_sole_corner_lift_m": 0.0166, "min_sole_contacts": 1.0, "max_sole_tilt_deg": 5.95},
                {"max_sole_corner_lift_m", "min_sole_contacts", "max_sole_tilt_deg"},
            ),
            # The saturated crouch (velociraptor 20260922: hip and knee pinned).
            ({"max_actuator_saturation_fraction": 1.0}, {"max_actuator_saturation_fraction"}),
            # The slider.
            ({"window_displacement_m": 0.62}, {"max_window_displacement_m"}),
            # The chatterer and the marcher.
            ({"touchdown_rate": 10.6, "all_feet_support": 0.52}, {"max_touchdown_rate", "min_all_feet_support"}),
            # The one-legged stance.
            ({"min_foot_load_share": 0.05}, {"min_foot_load_share"}),
        ],
        ids=["hopper", "prop-leg", "saturated-crouch", "slider", "chatter", "one-leg"],
    )
    def test_each_audit_hack_fails_on_its_named_criterion(self, clean, defect, criteria):
        reasons = classify_stance_episode(replace(clean, **defect), thresholds(), horizon=HORIZON)
        assert reason_keys(reasons) == criteria

    def test_an_unmeasured_metric_fails_its_criterion_by_name(self, clean):
        """NaN means unmeasured, and ``nan > bar`` is False: unfiltered it would pass."""
        reasons = classify_stance_episode(
            replace(clean, max_sole_corner_lift_m=math.nan), thresholds(), horizon=HORIZON
        )
        assert reasons == ("max_sole_corner_lift_m: max_sole_corner_lift_m is unmeasured (nan)",)

    def test_an_episode_that_ended_early_is_not_clean(self, clean):
        reasons = classify_stance_episode(replace(clean, length=312), thresholds(), horizon=HORIZON)
        assert reasons == ("horizon: length 312 < 1000",)

    def test_an_undeclared_criterion_is_not_applied(self, clean):
        hacked = replace(clean, foot_on_foot_fraction=0.95)
        assert classify_stance_episode(hacked, thresholds(), horizon=HORIZON) == ()
        reasons = classify_stance_episode(hacked, thresholds(max_foot_contact_fraction=0.02), horizon=HORIZON)
        assert reason_keys(reasons) == {"max_foot_contact_fraction"}

    @pytest.mark.parametrize(
        ("key", "field_name", "value"),
        [
            ("min_foot_load_share_windowed", "min_foot_load_share_windowed", 0.1),
            ("max_phantom_support_fraction", "phantom_support_fraction", 0.5),
            ("max_nonfoot_load_fraction", "nonfoot_load_fraction", 0.2),
            ("max_settle_stance_width_change_m", "settle_stance_width_change_m", 0.07),
            ("min_support_geom_duty", "min_support_geom_duty", 0.0),
            ("min_support_geom_coverage", "min_support_geom_coverage", 0.4),
            ("max_sole_tilt_excess_deg", "max_sole_tilt_excess_deg", 12.0),
            ("max_episode_yaw_change_deg", "episode_yaw_change_deg", 42.7),
            ("max_settle_touchdowns", "settle_touchdowns", 3.0),
            ("max_sole_cop_fore_aft", "max_sole_cop_fore_aft", 0.85),
            ("max_window_airborne_substeps", "window_airborne_substeps", 472.0),
            ("max_window_peak_floor_force_bw", "window_peak_floor_force_bw", 3.99),
        ],
    )
    def test_every_optional_criterion_refuses_its_defect_once_declared(self, clean, key, field_name, value):
        bar = {"min_foot_load_share_windowed": 0.3, "max_phantom_support_fraction": 0.05,
               "max_nonfoot_load_fraction": 0.01, "max_settle_stance_width_change_m": 0.04,
               "min_support_geom_duty": 0.9, "min_support_geom_coverage": 0.9,
               "max_sole_tilt_excess_deg": 10.0, "max_episode_yaw_change_deg": 25.0,
               "max_settle_touchdowns": 2, "max_sole_cop_fore_aft": 0.8, "max_window_airborne_substeps": 30,
               "max_window_peak_floor_force_bw": 3.0}[key]  # fmt: skip
        declared = thresholds(**{key: bar})
        assert classify_stance_episode(clean, declared, horizon=HORIZON) == ()
        reasons = classify_stance_episode(replace(clean, **{field_name: value}), declared, horizon=HORIZON)
        assert reason_keys(reasons) == {key}

    def test_a_pad_on_its_front_edge_passes_every_other_sole_bar_and_fails_the_cop_bar(self, clean):
        """The r8 seed-44 left foot (D-D28; seed 7081, a v2-clean episode): two loaded corners, flat, CoP 0.85."""
        edge = replace(clean, min_sole_contacts=2.0, max_sole_tilt_deg=0.7, max_sole_corner_lift_m=0.0018,
                       max_sole_cop_fore_aft=0.85)  # fmt: skip
        assert classify_stance_episode(edge, thresholds(), horizon=HORIZON) == ()
        declared = thresholds(max_sole_cop_fore_aft=0.8)
        assert classify_stance_episode(edge, declared, horizon=HORIZON) == (
            "max_sole_cop_fore_aft: max_sole_cop_fore_aft 0.85 > 0.8",
        )
        assert not stance_gate_v2.is_hop_or_fall(classify_stance_episode(edge, declared, horizon=HORIZON))
        # The statue's worst over 238 full-horizon episodes, a fore-aft sway (seed 5048), is clean.
        assert classify_stance_episode(replace(clean, max_sole_cop_fore_aft=0.48), declared, horizon=HORIZON) == ()
        # A padless foot reads it unmeasured: declared there, it would fail every episode.
        reasons = classify_stance_episode(replace(clean, max_sole_cop_fore_aft=math.nan), declared, horizon=HORIZON)
        assert reasons == ("max_sole_cop_fore_aft: max_sole_cop_fore_aft is unmeasured (nan)",)

    def test_a_post_settle_hop_with_sub_half_step_flights_is_refused_only_by_the_window_hop_pair(self, clean):
        """The compsognathus two-foot hop the D-D26 review certified under the step-level bars.

        Its flights last 1-3 of a step's 10 substeps, so each leg is down on
        every step (``leg_down_frac >= DOWN_SUBSTEP_FRACTION``): full support,
        no touchdown, no flight, and a quiet settle, because it starts after
        it.  The two metrics are the landed plant's (seed 3042: 472 airborne
        substeps and a 3.99 BW landing over the 800 window steps).
        """
        hop = replace(clean, window_airborne_substeps=472.0, window_peak_floor_force_bw=3.99)
        assert (hop.all_feet_support, hop.touchdown_rate, hop.flight_fraction) == (1.0, 0.0, 0.0)
        assert classify_stance_episode(hop, thresholds(), horizon=HORIZON) == ()
        pair = {"max_window_airborne_substeps": 30, "max_window_peak_floor_force_bw": 3.0}
        assert classify_stance_episode(clean, thresholds(**pair), horizon=HORIZON) == ()
        reasons = classify_stance_episode(hop, thresholds(**pair), horizon=HORIZON)
        assert reason_keys(reasons) == set(pair)
        # Each refuses it alone, and the pair needs no statue panel.
        for key, bar in pair.items():
            alone = thresholds(**{key: bar})
            assert not alone.declares_statue_criteria()
            assert reason_keys(classify_stance_episode(hop, alone, horizon=HORIZON)) == {key}
        assert not evaluate_stance_v2_gate([hop] * 40, thresholds(**pair), horizon=HORIZON).passed
        assert evaluate_stance_v2_gate([hop] * 40, thresholds(), horizon=HORIZON).passed


class TestStatueRelative:
    def test_a_declared_ratio_with_no_statue_fails_closed_by_name(self, clean):
        reasons = classify_stance_episode(clean, thresholds(min_foot_load_share_statue_ratio=0.8), horizon=HORIZON)
        assert len(reasons) == 1 and "no statue panel was rolled" in reasons[0]

    def test_the_share_ratio_is_per_foot_against_the_statues_mean(self, clean):
        statue = StatueReference(n_episodes=40, n_full_horizon=40, mean_reward=400.0, foot_load_share=(0.5, 0.5))
        declared = thresholds(min_foot_load_share_statue_ratio=0.8)
        assert classify_stance_episode(clean, declared, horizon=HORIZON, statue=statue) == ()
        # A 0.38 / 0.62 split clears the absolute 0.30 floor but is 0.76 of the statue's share.
        lopsided = replace(clean, foot_load_share=(0.38, 0.62), min_foot_load_share=0.38)
        reasons = classify_stance_episode(lopsided, declared, horizon=HORIZON, statue=statue)
        assert reason_keys(reasons) == {"min_foot_load_share_statue_ratio"}

    @pytest.mark.parametrize("statue_share", [(0.0, 1.0), (math.nan, 0.5), (0.5,)], ids=["zero", "nan", "feet"])
    def test_a_statue_reference_that_cannot_be_divided_by_fails_closed(self, clean, statue_share):
        statue = StatueReference(n_episodes=40, n_full_horizon=40, mean_reward=400.0, foot_load_share=statue_share)
        reasons = classify_stance_episode(
            clean, thresholds(min_foot_load_share_statue_ratio=0.8), horizon=HORIZON, statue=statue
        )
        assert reason_keys(reasons) == {"min_foot_load_share_statue_ratio"}

    @pytest.mark.parametrize("shares", [(0.5, math.nan), (math.nan, 0.5)], ids=["nan-last", "nan-first"])
    def test_an_unmeasured_policy_share_fails_the_ratio_wherever_it_sits(self, clean, shares):
        """``min()`` keeps a NaN only in first position, so ``(0.5, nan)`` would read as a ratio of 1."""
        statue = StatueReference(n_episodes=40, n_full_horizon=40, mean_reward=400.0, foot_load_share=(0.5, 0.5))
        row = replace(clean, foot_load_share=shares)
        reasons = classify_stance_episode(
            row, thresholds(min_foot_load_share_statue_ratio=0.8), horizon=HORIZON, statue=statue
        )
        assert reason_keys(reasons) == {"min_foot_load_share_statue_ratio"}
        assert "is unmeasured (nan)" in reasons[0]

    def test_the_statue_reference_is_its_full_horizon_episodes_only(self, clean):
        fallen = replace(clean, length=200, reward=50.0, foot_load_share=(0.9, 0.1))
        reference = statue_reference([clean, clean, fallen], horizon=HORIZON)
        assert (reference.n_episodes, reference.n_full_horizon) == (3, 2)
        assert reference.mean_reward == clean.reward
        assert reference.foot_load_share == clean.foot_load_share
        nothing_stood = statue_reference([fallen], horizon=HORIZON)
        assert math.isnan(nothing_stood.mean_reward) and nothing_stood.foot_load_share == ()

    def test_the_reference_round_trips_through_json_values(self, clean):
        reference = StatueReference(
            n_episodes=40, n_full_horizon=39, mean_reward=math.nan, foot_load_share=(0.5, math.nan)
        )
        as_json = {**reference.as_dict(), "mean_reward": None, "foot_load_share": [0.5, None]}
        back = StatueReference.from_dict(as_json)
        assert (back.n_episodes, back.n_full_horizon) == (40, 39)
        assert math.isnan(back.mean_reward) and back.foot_load_share[0] == 0.5 and math.isnan(back.foot_load_share[1])
        with pytest.raises(ValueError, match="statue reference is unreadable"):
            StatueReference.from_dict({"n_episodes": 40})


# ── judging a panel ──────────────────────────────────────────────────────────


class TestPanel:
    def test_a_clean_statue_panel_of_forty_passes_at_its_bound(self, clean):
        result = evaluate_stance_v2_gate([clean] * 40, thresholds(), horizon=HORIZON, control_dt=0.01)
        assert result.passed and result.failures == ()
        assert (result.n_clean, result.n_episodes, result.n_full_horizon) == (40, 40, 40)
        assert result.clean_lcb == pytest.approx(0.9278, abs=1e-4)
        assert result.clean_fraction == 1.0 and result.full_horizon_fraction == 1.0

    def test_37_of_40_clears_the_0_80_bar_and_36_does_not(self, clean):
        """GQ-7: LCB95(37/40) = 0.817, LCB95(36/40) = 0.786."""
        passing = evaluate_stance_v2_gate(panel(clean, 37, settle_airborne_substeps=4.0), thresholds(), horizon=HORIZON)
        assert passing.passed and passing.clean_lcb == pytest.approx(0.8174, abs=1e-4)
        failing = evaluate_stance_v2_gate(panel(clean, 36, settle_airborne_substeps=4.0), thresholds(), horizon=HORIZON)
        assert not failing.passed
        assert failing.failures == ("clean_stance_lcb 0.7856 < 0.8000 (36/40 episodes clean)",)

    def test_per_criterion_counts_name_what_failed(self, clean):
        result = evaluate_stance_v2_gate(
            panel(clean, 30, settle_airborne_substeps=4.0, max_sole_corner_lift_m=0.01), thresholds(), horizon=HORIZON
        )
        counts = dict(result.criterion_failures)
        assert counts["max_settle_airborne_substeps"] == 10 and counts["max_sole_corner_lift_m"] == 10
        assert counts["horizon"] == 0 and counts["min_all_feet_support"] == 0
        assert "max_foot_contact_fraction" not in counts  # undeclared criteria are not counted
        assert [bool(reasons) for reasons in result.episode_reasons] == [False] * 30 + [True] * 10

    @pytest.mark.parametrize("n", [0, 39])
    def test_an_under_powered_panel_fails_even_when_every_episode_is_clean(self, clean, n):
        result = evaluate_stance_v2_gate([clean] * n, thresholds(), horizon=HORIZON)
        assert not result.passed
        assert any(f"n_episodes {n} < min_eval_episodes 40" in failure for failure in result.failures)

    def test_an_episode_measured_under_another_settle_window_is_refused(self, clean):
        result = evaluate_stance_v2_gate(
            [clean] * 39 + [replace(clean, settle_steps=100)], thresholds(), horizon=HORIZON
        )
        assert not result.passed
        assert any("another settle window" in failure and "episode 39 at 100" in failure for failure in result.failures)

    def test_the_settle_window_is_checked_against_the_control_step_when_known(self, clean):
        rows = [replace(clean, settle_steps=10)] * 40
        assert evaluate_stance_v2_gate(rows, thresholds(settle_steps=10), horizon=HORIZON).passed
        result = evaluate_stance_v2_gate(rows, thresholds(settle_steps=10), horizon=HORIZON, control_dt=0.01)
        assert not result.passed and any("spawn grace" in failure for failure in result.failures)

    def test_the_panel_rails(self, clean):
        rows = [clean] * 40
        assert evaluate_stance_v2_gate(rows, thresholds(min_avg_reward=clean.reward), horizon=HORIZON).passed
        result = evaluate_stance_v2_gate(rows, thresholds(min_avg_reward=clean.reward + 1), horizon=HORIZON)
        assert not result.passed and any("(rail)" in failure for failure in result.failures)
        nan_reward = [replace(clean, reward=math.nan)] * 40
        assert not evaluate_stance_v2_gate(nan_reward, thresholds(min_avg_reward=0.0), horizon=HORIZON).passed
        short = panel(clean, 36, length=999)
        fraction = evaluate_stance_v2_gate(
            short, thresholds(min_full_horizon_fraction=0.95, min_clean_stance_lcb=0.5), horizon=HORIZON
        )
        assert not fraction.passed and any("full_horizon_fraction 0.9000 < 0.9500" in f for f in fraction.failures)

    def test_the_hop_or_fall_rail_counts_only_early_ends_and_the_hop_or_fall_criteria(self, clean):
        """One hop or fall episode passes the rail at a bar of 1, a second refuses; a pad miss never counts."""
        declared = thresholds(max_hop_or_fall_episodes=1, min_clean_stance_lcb=0.5)
        hop = replace(clean, all_feet_support=0.6, window_displacement_m=1.3)
        fall = replace(clean, length=370)
        pad = replace(clean, max_sole_corner_lift_m=0.05)
        settle = replace(clean, settle_peak_floor_force_bw=2.0)
        assert stance_gate_v2.is_hop_or_fall(classify_stance_episode(hop, declared, horizon=HORIZON))
        assert stance_gate_v2.is_hop_or_fall(classify_stance_episode(fall, declared, horizon=HORIZON))
        assert not stance_gate_v2.is_hop_or_fall(classify_stance_episode(pad, declared, horizon=HORIZON))
        assert not stance_gate_v2.is_hop_or_fall(classify_stance_episode(settle, declared, horizon=HORIZON))
        for defect in (
            "all_feet_support",
            "touchdown_rate",
            "window_displacement_m",
            "max_actuator_saturation_fraction",
        ):
            bad = {"all_feet_support": 0.5}.get(defect, 1.0)
            reasons = classify_stance_episode(replace(clean, **{defect: bad}), declared, horizon=HORIZON)
            assert stance_gate_v2.is_hop_or_fall(reasons), defect

        one = evaluate_stance_v2_gate([clean] * 36 + [hop, pad, pad, settle], declared, horizon=HORIZON)
        assert one.passed and one.hop_or_fall_episodes == 1
        assert one.as_dict()["hop_or_fall_episodes"] == 1
        two = evaluate_stance_v2_gate([clean] * 37 + [hop, fall, pad], declared, horizon=HORIZON)
        assert not two.passed and two.hop_or_fall_episodes == 2
        assert [f for f in two.failures if f.startswith("hop_or_fall_episodes 2 > 1")]
        # The bound alone (37/40 clean, LCB 0.817) admits the same panel at the trex bar.
        assert evaluate_stance_v2_gate([clean] * 37 + [hop, fall, pad], thresholds(), horizon=HORIZON).passed

    def test_the_window_hop_pair_counts_toward_the_hop_or_fall_rail_where_declared(self, clean):
        """A window-pair failure is a hop (D-D26): a gate declaring the pair and the rail counts it."""
        declared = thresholds(
            max_hop_or_fall_episodes=0, max_window_airborne_substeps=40, max_window_peak_floor_force_bw=2.0
        )
        assert evaluate_stance_v2_gate([clean] * 40, declared, horizon=HORIZON).passed
        for field_name, value in (("window_airborne_substeps", 41.0), ("window_peak_floor_force_bw", 2.2)):
            reasons = classify_stance_episode(replace(clean, **{field_name: value}), declared, horizon=HORIZON)
            assert stance_gate_v2.is_hop_or_fall(reasons), field_name
        assert set(stance_gate_v2.HOP_OR_FALL_KEYS) <= {key for key, _, _ in stance_gate_v2.EPISODE_CRITERIA}

    def test_a_rail_of_zero_refuses_a_single_hop(self, clean):
        """0 is the zero-tolerance bar (the statue's 1-in-120 rate fails it on 28% of panels): valid, and strict."""
        zero = thresholds(max_hop_or_fall_episodes=0)
        assert zero.max_hop_or_fall_episodes == 0
        assert evaluate_stance_v2_gate([clean] * 40, zero, horizon=HORIZON).passed
        hop = replace(clean, all_feet_support=0.6, window_displacement_m=1.3)
        result = evaluate_stance_v2_gate([clean] * 39 + [hop], zero, horizon=HORIZON)
        # 39/40 clears the 0.80 bound (0.887); the rail alone refuses.
        assert result.clean_lcb >= 0.80
        assert not result.passed and len(result.failures) == 1
        assert result.failures[0].startswith("hop_or_fall_episodes 1 > 0")

    def test_the_rail_failure_names_only_the_hop_keys_the_gate_declares(self, clean):
        """An undeclared window pair fails no episode, so the rail's text does not name it."""
        hop = replace(clean, all_feet_support=0.6, window_displacement_m=1.3)
        panel = [clean] * 39 + [hop]
        step_level = (
            "min_all_feet_support, max_touchdown_rate, max_window_displacement_m, max_actuator_saturation_fraction"
        )
        without_pair = evaluate_stance_v2_gate(panel, thresholds(max_hop_or_fall_episodes=0), horizon=HORIZON)
        assert without_pair.failures == (
            f"hop_or_fall_episodes 1 > 0 (episodes ending early or failing {step_level}; panel rail)",
        )
        declared = thresholds(
            max_hop_or_fall_episodes=0, max_window_airborne_substeps=40, max_window_peak_floor_force_bw=2.0
        )
        with_pair = evaluate_stance_v2_gate(panel, declared, horizon=HORIZON)
        assert with_pair.failures == (
            f"hop_or_fall_episodes 1 > 0 (episodes ending early or failing {step_level}, max_window_airborne_substeps, "
            "max_window_peak_floor_force_bw; panel rail)",
        )

    def test_an_undeclared_rail_leaves_the_result_as_it_was(self, clean):
        result = evaluate_stance_v2_gate([clean] * 40, thresholds(), horizon=HORIZON)
        assert result.hop_or_fall_episodes is None
        assert "hop_or_fall_episodes" not in result.as_dict()

    @pytest.mark.parametrize("value", [-1, 1.5, True, "1"], ids=repr)
    def test_the_rail_is_a_whole_number_of_episodes(self, value):
        with pytest.raises(ValueError, match="max_hop_or_fall_episodes"):
            StanceV2Thresholds.from_curriculum({**BLOCK, "max_hop_or_fall_episodes": value})

    def test_the_statue_relative_reward_rail(self, clean):
        rows = [clean] * 40
        declared = thresholds(min_avg_reward_statue_ratio=0.6)
        statue = StatueReference(
            n_episodes=40, n_full_horizon=40, mean_reward=clean.reward / 0.6, foot_load_share=(0.5, 0.5)
        )
        assert evaluate_stance_v2_gate(rows, declared, horizon=HORIZON, statue=statue).passed
        higher = replace(statue, mean_reward=clean.reward / 0.5)
        failed = evaluate_stance_v2_gate(rows, declared, horizon=HORIZON, statue=higher)
        assert not failed.passed and any("statue-relative rail" in failure for failure in failed.failures)
        missing = evaluate_stance_v2_gate(rows, declared, horizon=HORIZON)
        assert not missing.passed and any("no statue panel was rolled" in failure for failure in missing.failures)
        fell = replace(statue, mean_reward=math.nan, n_full_horizon=0)
        unscaled = evaluate_stance_v2_gate(rows, declared, horizon=HORIZON, statue=fell)
        assert not unscaled.passed and any("no positive reference" in failure for failure in unscaled.failures)

    def test_in_training_skips_the_statue_criteria_and_says_which(self, clean):
        declared = thresholds(min_avg_reward_statue_ratio=0.6, min_foot_load_share_statue_ratio=0.8)
        result = evaluate_stance_v2_gate([clean] * 40, declared, horizon=HORIZON, in_training=True)
        assert result.passed
        assert result.skipped_criteria == ("min_avg_reward_statue_ratio", "min_foot_load_share_statue_ratio")
        assert not evaluate_stance_v2_gate([clean] * 40, declared, horizon=HORIZON).passed

    @pytest.mark.parametrize("horizon", [0, -5, True])
    def test_a_horizon_no_episode_can_reach_fails(self, clean, horizon):
        assert not evaluate_stance_v2_gate([clean] * 40, thresholds(), horizon=horizon).passed

    def test_what_would_have_to_be_deleted_for_the_bound_to_stop_being_consulted(self, clean, monkeypatch):
        def consulted(*args, **kwargs):
            raise RuntimeError("consulted")

        monkeypatch.setattr(stance_gate_v2, "binomial_lcb", consulted)
        with pytest.raises(RuntimeError, match="consulted"):
            evaluate_stance_v2_gate([clean] * 40, thresholds(), horizon=HORIZON)


# ── the panel CSV contract ───────────────────────────────────────────────────


def _write_panel(path: Path, episode: StanceEpisodeMetrics, n: int = 3, **cells: Any) -> Path:
    rows = []
    for index in range(n):
        row: dict[str, Any] = {key: "" for key in STANCE_V2_PANEL_FIELDNAMES}
        row.update(episode.as_row())
        row.update(
            {
                "episode": index,
                "panel_seed": 3042 + index,
                "reached_horizon": True,
                "clean": True,
                "checkpoint_sha256": "sha256:" + "a" * 64,
                "normalization_sha256": "sha256:" + "b" * 64,
                "task_sha256": "sha256:" + "d" * 64,
                "measurement_version": "floor-truth/v1",
                "measurement_sha256": "sha256:" + "c" * 64,
                "measurement_definition_sha256": "sha256:" + "e" * 64,
            }
        )
        row.update({key: (value(index) if callable(value) else value) for key, value in cells.items()})
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(STANCE_V2_PANEL_FIELDNAMES))
        writer.writeheader()
        writer.writerows(rows)
    return path


class TestPanelEvidence:
    def test_the_columns_hold_every_metric_and_the_panel_stamps(self):
        assert set(STANCE_METRIC_FIELDS) | set(STANCE_METRIC_FOOT_FIELDS) <= set(STANCE_V2_PANEL_FIELDNAMES)
        assert len(set(STANCE_V2_PANEL_FIELDNAMES)) == len(STANCE_V2_PANEL_FIELDNAMES)
        from environments.shared.reporting.stance_report import STANCE_PANEL_FIELDNAMES

        assert set(STANCE_PANEL_FIELDNAMES) <= set(STANCE_V2_PANEL_FIELDNAMES)  # v1's columns stay readable

    def test_a_panel_reads_back_exactly(self, tmp_path, clean):
        path = _write_panel(
            tmp_path / "panel.csv",
            clean,
            statue_n_episodes=3,
            statue_n_full_horizon=3,
            statue_mean_reward=repr(clean.reward),
            statue_foot_load_share="[0.5, 0.5]",
        )
        evidence = read_stance_v2_panel(path)
        assert evidence.panel_seeds == [3042, 3043, 3044]
        assert evidence.recorded_clean == [True] * 3 and evidence.reached_horizon == [True] * 3
        assert evidence.checkpoint_sha256 == "sha256:" + "a" * 64
        assert evidence.task_sha256 == "sha256:" + "d" * 64
        assert evidence.measurement_version == "floor-truth/v1"
        assert evidence.measurement_definition_sha256 == "sha256:" + "e" * 64
        assert evidence.statue == StatueReference(3, 3, clean.reward, (0.5, 0.5))
        assert [row.as_row() for row in evidence.episodes] == [clean.as_row()] * 3

    def test_no_statue_stamp_reads_as_no_statue(self, tmp_path, clean):
        assert read_stance_v2_panel(_write_panel(tmp_path / "panel.csv", clean)).statue is None

    def test_a_v1_panel_file_is_refused_on_its_columns(self, tmp_path):
        path = tmp_path / "v1.csv"
        path.write_text("episode,panel_seed,length,reward,reached_horizon,unsupported_duty\n0,3042,1000,1.0,True,0.0\n")
        with pytest.raises(ValueError, match="is not a stance_quality/v2 panel"):
            read_stance_v2_panel(path)

    @pytest.mark.parametrize(
        ("cells", "message"),
        [
            ({"checkpoint_sha256": lambda i: f"sha256:{i}"}, "mixes checkpoint_sha256"),
            ({"panel_seed": "x"}, "unparsable panel_seed"),
            ({"clean": "maybe"}, "unparsable clean"),
            ({"length": ""}, "not a whole number"),
            ({"statue_mean_reward": "1.0"}, "partial statue reference"),
        ],
        ids=["mixed-hash", "seed", "clean", "length", "partial-statue"],
    )
    def test_every_unreadable_panel_is_refused(self, tmp_path, clean, cells, message):
        with pytest.raises(ValueError, match=message):
            read_stance_v2_panel(_write_panel(tmp_path / "panel.csv", clean, **cells))

    def test_an_empty_panel_is_refused(self, tmp_path, clean):
        with pytest.raises(ValueError, match="holds no episode rows"):
            read_stance_v2_panel(_write_panel(tmp_path / "panel.csv", clean, n=0))

    def test_a_panel_recorded_before_a_later_metric_keeps_its_verdict_and_names_what_it_lacks(self, tmp_path, clean):
        """Its CSV lacks the later columns: unmeasured, named in absent_metrics, and refused only where declared."""
        later = sorted(STANCE_METRIC_LATER_FIELDS)
        path = _write_panel(tmp_path / "panel.csv", clean)
        current = read_stance_v2_panel(path)
        assert current.absent_metrics == ()
        with path.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        columns = [column for column in STANCE_V2_PANEL_FIELDNAMES if column not in later]
        with path.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        evidence = read_stance_v2_panel(path)
        assert set(evidence.absent_metrics) == set(later)
        assert all(math.isnan(getattr(episode, name)) for episode in evidence.episodes for name in later)
        restored = [replace(episode, **{name: getattr(clean, name) for name in later}) for episode in evidence.episodes]
        assert [episode.as_row() for episode in restored] == [clean.as_row()] * 3
        recorded = [name for name in STANCE_METRIC_FIELDS if name not in evidence.absent_metrics]
        # A block that does not declare the keys: the same verdict, bit for bit, as the panel measured today.
        loose = {"min_eval_episodes": 3, "min_clean_stance_lcb": 0.1}
        assert unrecorded_criteria(thresholds(**loose), recorded) == ()
        earlier_result = evaluate_stance_v2_gate(evidence.episodes, thresholds(**loose), horizon=HORIZON)
        assert earlier_result == evaluate_stance_v2_gate(current.episodes, thresholds(**loose), horizon=HORIZON)
        # A block that does: the criteria are named, for the consumer to refuse the panel rather than score it.
        declared = thresholds(**loose, max_sole_cop_fore_aft=0.8, max_window_airborne_substeps=30)
        assert unrecorded_criteria(declared, recorded) == (
            ("max_sole_cop_fore_aft", "max_sole_cop_fore_aft"),
            ("max_window_airborne_substeps", "window_airborne_substeps"),
        )
        assert unrecorded_criteria(declared, STANCE_METRIC_FIELDS) == ()
        # Scored anyway, every episode would fail as unmeasured though nothing failed to measure it.
        result = evaluate_stance_v2_gate(evidence.episodes, declared, horizon=HORIZON)
        assert result.n_clean == 0 and not result.passed
        assert result.episode_reasons[0] == (
            "max_sole_cop_fore_aft: max_sole_cop_fore_aft is unmeasured (nan)",
            "max_window_airborne_substeps: window_airborne_substeps is unmeasured (nan)",
        )
        # Any other metric column missing is still a file that is not a v2 panel.
        columns.remove("min_sole_contacts")
        with path.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        with pytest.raises(ValueError, match="min_sole_contacts"):
            read_stance_v2_panel(path)

    def test_an_empty_metric_cell_is_unmeasured_never_zero(self, tmp_path, clean):
        evidence = read_stance_v2_panel(_write_panel(tmp_path / "panel.csv", clean, settle_airborne_substeps=""))
        assert math.isnan(evidence.episodes[0].settle_airborne_substeps)
        result = evaluate_stance_v2_gate(
            evidence.episodes, thresholds(min_eval_episodes=3, min_clean_stance_lcb=0.1), horizon=HORIZON
        )
        assert result.n_clean == 0
