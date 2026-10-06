"""The velociraptor stance stage's ``stance_quality/v2`` declaration (decision D-D25, 2026-10-06).

``configs/velociraptor/stage1_balance.toml`` moved from
``reward_and_length/v1``, which its zero-action statue cleared by design, to
the floor-truth gate with physics r3.  Its bars were set on the r3 statue's
40-episode panel and validated against the two audited checkpoints of the
``20260922_125248`` stance, rolled on the r2 plant they load on
(``docs/investigations/STANCE_HACK_AUDIT_2026_10.md`` §8).  This module pins
the declaration in the ways that each fail differently:

1. **It validates as the kind it declares**, strictly: every v2 key numeric,
   the settle window non-empty after the spawn grace at the stage's own
   control step, no v1 key left behind.
2. **The foot, settle and width bars are declared.**  The raptor has no box
   sole, so its flatness bars are the support-geom duty and coverage of the
   middle toe, the outer toe and the metatarsal head.  They are optional in
   the kind, and so is the settle-window width change; on the raptor each
   family, like actuator saturation and the required settle bars, refuses
   every audited episode on its own, while the window bars admit the
   quietest of them.
3. **A recorded episode of each kind is classified as it was measured**: the
   final checkpoint's seed-3043 episode, which clears every window bar, is
   refused by exactly those families, and the r3 statue's least favourable
   episode is clean.
4. **The foot bars and the settle peak sit mid-gap**: the statue with
   N(0, 0.05) command jitter, whose lightly loaded digit IV unloads on part
   of the window, is clean, though the bars first set at the statue's edge
   refused it.
5. **The bar and the rails agree with the numbers they were derived from**,
   recomputed here rather than copied: the bound admits 37/40 and refuses
   36/40, and both statue-derived constants are their fractions of the
   statue reference the stage records.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from environments.shared.config import load_stage_config
from environments.shared.curriculum.gate_schema import validate_gate_config
from environments.shared.curriculum.recovery_gate import binomial_lcb
from environments.shared.curriculum.stance_gate_v2 import (
    STANCE_GATE_V2_KIND,
    STANCE_V2_THRESHOLD_KEYS,
    StanceV2Thresholds,
    StatueReference,
    classify_stance_episode,
)
from environments.shared.gait.stance_metrics import episode_stance_metrics, spawn_grace_steps

from .test_gait_stance_metrics import statue_trace

#: The flatness bars of a foot without a box sole: every support geom loaded.
FOOT_KEYS = ("min_support_geom_duty", "min_support_geom_coverage")
#: The settle-window bars: the reset hop and its impact.
SETTLE_KEYS = ("max_settle_airborne_substeps", "max_settle_peak_floor_force_bw")
#: The settle window's splay.
WIDTH_KEYS = ("max_settle_stance_width_change_m",)
#: The parked command.
SATURATION_KEYS = ("max_actuator_saturation_fraction",)
FAMILIES = {"foot": FOOT_KEYS, "settle": SETTLE_KEYS, "width": WIDTH_KEYS, "saturation": SATURATION_KEYS}

#: The statue panel's mean per-foot load share (right, left) that the report rolls beside the policy
#: panel, as the r2 statue measured it (the r3 statue's is the same to 1e-4).
STATUE = StatueReference(n_episodes=40, n_full_horizon=40, mean_reward=1745.84, foot_load_share=(0.49997, 0.50003))

#: The 20260922_125248 stance's final checkpoint on panel seed 3043, as the v2 report measured it on
#: the r2 plant: both feet down, no touchdown and 27 mm of travel -- every window bar clears -- under
#: a saturated command, after a reset hop (42 airborne substeps, a 4.46 BW landing) that splays the
#: feet 0.20 m, with one support geom never loaded.
QUIET_WINDOW_HACK: dict[str, Any] = {
    "all_feet_support": 1.0,
    "touchdown_rate": 0.0,
    "window_displacement_m": 0.0265,
    "min_foot_load_share": 0.4531,
    "min_foot_load_share_windowed": 0.4511,
    "foot_load_share": (0.5469, 0.4531),
    "max_actuator_saturation_fraction": 1.0,
    "settle_airborne_substeps": 42.0,
    "settle_peak_floor_force_bw": 4.4583,
    "settle_stance_width_change_m": 0.2013,
    "min_support_geom_duty": 0.0,
    "min_support_geom_coverage": 0.6667,
}

#: The r3 statue's least favourable episode of the panel (seed 3078, the highest settle peak).
STATUE_WORST_SETTLE: dict[str, Any] = {
    "all_feet_support": 1.0,
    "touchdown_rate": 0.0,
    "window_displacement_m": 0.0079,
    "min_foot_load_share": 0.4998,
    "min_foot_load_share_windowed": 0.4976,
    "foot_load_share": (0.4998, 0.5002),
    "max_actuator_saturation_fraction": 0.0,
    "settle_airborne_substeps": 0.0,
    "settle_peak_floor_force_bw": 1.2847,
    "settle_stance_width_change_m": 0.0062,
    "min_support_geom_duty": 1.0,
    "min_support_geom_coverage": 1.0,
}


#: The r3 statue with N(0, 0.05) noise on its zero command at every step (seeds 3042-3081, one noise draw):
#: its lowest support-geom duty (seed 3077, the left digit IV loaded on 55% of the window) and its highest
#: settle peak (seed 3043), each episode as the v2 report measured it.  The panel kept 94.9% of the statue's
#: reward and was clean on 40/40 under the committed bars.
JITTERED_STATUE: dict[str, dict[str, Any]] = {
    "lowest_duty": {
        "all_feet_support": 1.0,
        "touchdown_rate": 0.0,
        "window_displacement_m": 0.0143,
        "min_foot_load_share": 0.4952,
        "min_foot_load_share_windowed": 0.4695,
        "foot_load_share": (0.5048, 0.4952),
        "max_actuator_saturation_fraction": 0.0,
        "settle_airborne_substeps": 0.0,
        "settle_peak_floor_force_bw": 1.7012,
        "settle_stance_width_change_m": 0.0090,
        "min_support_geom_duty": 0.5533,
        "min_support_geom_coverage": 0.8437,
    },
    "highest_settle_peak": {
        "all_feet_support": 1.0,
        "touchdown_rate": 0.0,
        "window_displacement_m": 0.0089,
        "min_foot_load_share": 0.4907,
        "min_foot_load_share_windowed": 0.4326,
        "foot_load_share": (0.5093, 0.4907),
        "max_actuator_saturation_fraction": 0.0,
        "settle_airborne_substeps": 0.0,
        "settle_peak_floor_force_bw": 1.8944,
        "settle_stance_width_change_m": 0.0148,
        "min_support_geom_duty": 0.6744,
        "min_support_geom_coverage": 0.8848,
    },
}

#: The bars as first set, just under the plain statue's worst, which refused the jittered statue.
EDGE_BARS = {
    "min_support_geom_duty": 0.90,
    "min_support_geom_coverage": 0.95,
    "max_settle_peak_floor_force_bw": 1.5,
}


@pytest.fixture(scope="module")
def stage() -> dict[str, Any]:
    return load_stage_config("velociraptor", "stance")


@pytest.fixture(scope="module")
def curriculum(stage: dict[str, Any]) -> dict[str, Any]:
    return dict(stage["curriculum_kwargs"])


def _reasons(curriculum: dict[str, Any], metrics: dict[str, Any]) -> set[str]:
    thresholds = StanceV2Thresholds.from_curriculum(curriculum)
    clean = episode_stance_metrics(statue_trace(1000), settle_steps=thresholds.settle_steps)
    episode = replace(clean, **metrics)
    reasons = classify_stance_episode(episode, thresholds, horizon=1000, statue=STATUE)
    return {reason.split(":", 1)[0] for reason in reasons}


def _without(curriculum: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    """*curriculum* with *keys* declared so loosely that they can never fail (a required key cannot be dropped)."""
    loose = dict(curriculum)
    for key in keys:
        loose[key] = 1e9 if key.startswith("max_") else -1e9
    return loose


def test_the_stance_declares_the_floor_truth_gate_and_validates_as_it(stage, curriculum):
    assert curriculum["gate_kind"] == STANCE_GATE_V2_KIND
    assert validate_gate_config("stance", curriculum, advancement_enabled=True) == STANCE_GATE_V2_KIND
    thresholds = StanceV2Thresholds.from_curriculum(curriculum)
    env = stage["env_kwargs"]
    thresholds.validate_settle_window(float(env.get("timestep", 0.002)) * int(env.get("frame_skip", 5)))
    # reward_and_length/v1's length rail is a superseded record in the file's comments, not a key.
    assert "min_avg_episode_length" not in curriculum
    assert {key for key in curriculum if key in STANCE_V2_THRESHOLD_KEYS} == set(thresholds.declared()) | {
        "required_consecutive"
    }
    # The sole bars belong to box pads; a raptor foot has none, so a sole bar would fail every episode as NaN.
    assert not {key for key in curriculum if "sole" in key}


@pytest.mark.parametrize("key", FOOT_KEYS + SETTLE_KEYS + WIDTH_KEYS + SATURATION_KEYS)
def test_the_foot_settle_width_and_saturation_bars_are_declared(curriculum, key):
    assert key in curriculum
    assert StanceV2Thresholds.from_curriculum(curriculum).declared()[key] == curriculum[key]


def test_the_settle_window_is_scored_after_the_spawn_grace(stage, curriculum):
    env = stage["env_kwargs"]
    control_dt = float(env.get("timestep", 0.002)) * int(env.get("frame_skip", 5))
    grace = spawn_grace_steps(control_dt)
    # 10 grace steps at 0.01 s; the r3 statue has both feet down by step 5 and its floor force within
    # 10% of its weight by step 15, while the audited stance's reset hop lands inside steps 10-99.
    assert (grace, curriculum["settle_steps"]) == (10, 100)
    assert curriculum["max_settle_airborne_substeps"] == 0


def test_a_recorded_hack_with_a_quiet_window_is_refused_by_exactly_the_four_families(curriculum):
    """The quietest audited episode clears every window bar; only saturation, settle, width and foot refuse it."""
    every_family = tuple(key for keys in FAMILIES.values() for key in keys)
    assert _reasons(curriculum, QUIET_WINDOW_HACK) == set(every_family)
    assert _reasons(_without(curriculum, every_family), QUIET_WINDOW_HACK) == set()
    # Each family refuses it alone, so none of the four is redundant on this episode.
    for name, keys in FAMILIES.items():
        others = tuple(key for key in every_family if key not in keys)
        assert _reasons(_without(curriculum, others), QUIET_WINDOW_HACK) == set(keys), name


def test_the_statues_least_favourable_episode_is_clean(curriculum):
    assert _reasons(curriculum, STATUE_WORST_SETTLE) == set()


@pytest.mark.parametrize("case", sorted(JITTERED_STATUE))
def test_the_jittered_statue_is_clean_where_the_edge_bars_refused_it(curriculum, case):
    """No reward term sees a digit unloading, so the foot bars must not refuse a statue that merely jitters."""
    episode = JITTERED_STATUE[case]
    assert _reasons(curriculum, episode) == set()
    refused_at_the_edge = _reasons({**curriculum, **EDGE_BARS}, episode)
    assert refused_at_the_edge and refused_at_the_edge <= set(EDGE_BARS)
    # Mid-gap: each moved bar sits between the jittered statue and the quietest audited hack.
    for key, bar in ((k, curriculum[k]) for k in EDGE_BARS):
        if key.startswith("max_"):
            assert episode[key.removeprefix("max_")] < bar < QUIET_WINDOW_HACK[key.removeprefix("max_")]
        else:
            assert episode[key] > bar > QUIET_WINDOW_HACK[key]


def test_the_bar_admits_37_of_40_and_refuses_36(curriculum):
    n = curriculum["min_eval_episodes"]
    bar = curriculum["min_clean_stance_lcb"]
    assert n == 40
    assert binomial_lcb(37, n) >= bar > binomial_lcb(36, n)
    assert binomial_lcb(40, n) == pytest.approx(0.9278, abs=1e-4)


def test_the_reward_rails_are_their_fractions_of_the_statue(curriculum):
    """min_avg_reward is 0.60 and collapse_peak_floor 0.75 of the statue, rounded to 10; the ratio states 0.60 itself.

    The reference is the r3 statue's standing reward (2842.76, 40 episodes, seeds 3042-3081), pinned to the
    physics revision it was measured on (test_statue_constant_freshness.py checks the pin); the explicit
    floor still wins over it in collapse_settings_from_config.
    """
    reference = curriculum["collapse_peak_floor_reference"]
    assert reference == 2842.8 and curriculum["statue_constants_physics_revision"] == 3
    assert curriculum["min_avg_reward"] == round(0.60 * reference, -1)
    assert curriculum["collapse_peak_floor"] == round(0.75 * reference, -1)
    assert curriculum["min_avg_reward_statue_ratio"] == 0.60
    assert curriculum["min_full_horizon_fraction"] == 0.95
