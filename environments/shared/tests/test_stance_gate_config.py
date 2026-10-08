"""The T-Rex stance stage's ``stance_quality/v2`` declaration (decision D-D24, 2026-10-06).

``configs/trex/stance.toml`` moved from ``stance_quality/v1`` to the
floor-truth gate with physics r8.  Its bars were set on the r8 statue's
40-episode panel and validated against the four audited r7 checkpoints
(``docs/investigations/STANCE_HACK_AUDIT_2026_10.md`` §7).  This module pins
the declaration in the ways that each fail differently:

1. **It validates as the kind it declares**, strictly: every v2 key numeric,
   the settle window non-empty after the spawn grace at the stage's own
   control step, no v1 key left behind.
2. **The sole and settle bars are declared.**  The sole bars are optional
   in the kind, because the right flatness key depends on the foot, but on
   trex the two families are what separate the audit's propped stances from
   the statue: without both, the window bars admit seed 42's stance on 40/40
   episodes, v1's blind spot again, and the settle bars alone already pass
   one measured prop episode, which only the sole bars refuse.
3. **The bar and the rails agree with the numbers they were derived from**,
   recomputed here rather than copied: the bound admits 37/40 and refuses
   36/40, and the reward rail is 0.60 of the statue reference the stage
   records.
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
    classify_stance_episode,
)
from environments.shared.gait.stance_metrics import episode_stance_metrics, spawn_grace_steps

from .test_gait_stance_metrics import statue_trace

#: The flatness bars of the trex box pad: tilt, corner lift and loaded contact points.
SOLE_KEYS = ("max_sole_tilt_deg", "max_sole_corner_lift_m", "min_sole_contacts")
#: The settle-window bars: the reset hop and its impact.
SETTLE_KEYS = ("max_settle_airborne_substeps", "max_settle_peak_floor_force_bw")

#: The window and settle metrics of the r7 audit checkpoint 20260920_010912 on panel seed 3045,
#: as the v2 report measured them: both feet down, no touchdown, no airborne substep and a settle
#: peak of 1.496 BW -- every bar but the sole ones clears -- on a pad tilted 5.2 degrees, its
#: corners 12.2 mm apart, standing on 1.4 loaded contact points.
PROP_WITH_A_QUIET_RESET: dict[str, float] = {
    "all_feet_support": 1.0,
    "touchdown_rate": 0.0,
    "window_displacement_m": 0.0425,
    "min_foot_load_share": 0.468,
    "min_foot_load_share_windowed": 0.432,
    "max_actuator_saturation_fraction": 0.0,
    "settle_airborne_substeps": 0.0,
    "settle_peak_floor_force_bw": 1.4956,
    "max_sole_tilt_deg": 5.217,
    "max_sole_corner_lift_m": 0.0122,
    "min_sole_contacts": 1.406,
}


@pytest.fixture(scope="module")
def stage() -> dict[str, Any]:
    return load_stage_config("trex", "stance")


@pytest.fixture(scope="module")
def curriculum(stage: dict[str, Any]) -> dict[str, Any]:
    return dict(stage["curriculum_kwargs"])


def _reasons(curriculum: dict[str, Any], **metrics: Any) -> set[str]:
    thresholds = StanceV2Thresholds.from_curriculum(curriculum)
    clean = episode_stance_metrics(statue_trace(1000), settle_steps=thresholds.settle_steps)
    episode = replace(clean, **metrics)
    return {reason.split(":", 1)[0] for reason in classify_stance_episode(episode, thresholds, horizon=1000)}


def test_the_stance_declares_the_floor_truth_gate_and_validates_as_it(stage, curriculum):
    assert curriculum["gate_kind"] == STANCE_GATE_V2_KIND
    assert validate_gate_config("stance", curriculum, advancement_enabled=True) == STANCE_GATE_V2_KIND
    thresholds = StanceV2Thresholds.from_curriculum(curriculum)
    env = stage["env_kwargs"]
    thresholds.validate_settle_window(float(env.get("timestep", 0.002)) * int(env.get("frame_skip", 5)))
    # v1's certifying pair is a superseded record in the file's comments, not a key.
    assert not {"max_unsupported_duty", "max_unsupported_duty_ucb"} & set(curriculum)
    assert {key for key in curriculum if key in STANCE_V2_THRESHOLD_KEYS} == set(thresholds.declared()) | {
        "required_consecutive"
    }


@pytest.mark.parametrize("key", SOLE_KEYS + SETTLE_KEYS)
def test_the_sole_and_settle_bars_are_declared(curriculum, key):
    """Each is load-bearing on trex: the propped stances stand on both feet, at the statue's load share."""
    assert key in curriculum
    assert StanceV2Thresholds.from_curriculum(curriculum).declared()[key] == curriculum[key]


def test_the_settle_window_is_scored_after_the_spawn_grace(stage, curriculum):
    env = stage["env_kwargs"]
    control_dt = float(env.get("timestep", 0.002)) * int(env.get("frame_skip", 5))
    grace = spawn_grace_steps(control_dt)
    # 10 grace steps at 0.01 s; the statue's floor force has settled by step 47, the hacks hop at 5-9.
    assert (grace, curriculum["settle_steps"]) == (10, 200)
    assert curriculum["max_settle_airborne_substeps"] == 0


def test_without_the_sole_bars_a_measured_prop_would_be_clean(curriculum):
    """The audit's seed-44 prop on seed 3045 clears every other bar; only the sole bars refuse it."""
    assert _reasons(curriculum, **PROP_WITH_A_QUIET_RESET) == set(SOLE_KEYS)
    without_soles = {key: value for key, value in curriculum.items() if key not in SOLE_KEYS}
    assert _reasons(without_soles, **PROP_WITH_A_QUIET_RESET) == set()
    # Each sole bar refuses it alone, so none of the three is redundant on this episode.
    for key in SOLE_KEYS:
        only = {k: v for k, v in curriculum.items() if k not in SOLE_KEYS or k == key}
        assert _reasons(only, **PROP_WITH_A_QUIET_RESET) == {key}


def test_a_reset_hop_on_flat_feet_is_refused_by_the_settle_bars(curriculum):
    """The hop the v1 settle prefix hid: both feet off for a few substeps, then a landing over 1.5 BW."""
    assert _reasons(curriculum, settle_airborne_substeps=22.0, settle_peak_floor_force_bw=2.86) == set(SETTLE_KEYS)


def test_the_bar_admits_37_of_40_and_refuses_36(curriculum):
    n = curriculum["min_eval_episodes"]
    bar = curriculum["min_clean_stance_lcb"]
    assert n == 40
    assert binomial_lcb(37, n) >= bar > binomial_lcb(36, n)
    assert binomial_lcb(40, n) == pytest.approx(0.9278, abs=1e-4)


def test_the_reward_rails_are_the_same_fraction_of_the_statue(curriculum):
    """min_avg_reward is 0.60 of the recorded statue reference, rounded to 10; the statue ratio states 0.60 itself."""
    reference = curriculum["collapse_peak_floor_reference"]
    assert curriculum["min_avg_reward"] == round(0.60 * reference, -1)
    assert curriculum["min_avg_reward_statue_ratio"] == 0.60
    assert curriculum["min_full_horizon_fraction"] == 0.95


# ── the stance follow-up (decision D-D27, 2026-10-07) ─────────────────────────

#: The bars D-D27 added to the trex block, at their adopted values.
FOLLOWUP_BARS = {
    "max_settle_stance_width_change_m": 0.08,
    "max_settle_touchdowns": 2,
    "max_episode_yaw_change_deg": 25.0,
    "max_hop_or_fall_episodes": 1,
}


@pytest.mark.parametrize(("key", "bar"), sorted(FOLLOWUP_BARS.items()))
def test_the_followup_bars_are_declared(curriculum, key, bar):
    assert curriculum[key] == bar
    assert StanceV2Thresholds.from_curriculum(curriculum).declared()[key] == bar


def test_the_settled_width_is_captured_where_the_settle_window_ends(stage, curriculum):
    """stance_width_settle_steps and the gate's settle_steps name the same step: the reward's settled width is the
    width the settle bars judge the episode to have settled at."""
    env = stage["env_kwargs"]
    assert env["stance_width_reference"] == "settled"
    assert env["stance_width_settle_steps"] == curriculum["settle_steps"]


def test_a_settle_re_seat_and_a_turn_are_refused(curriculum):
    """Seed 44's largest r8 re-seat (8.75 cm), a stepping re-seat and seed 42's median turn (45.3 degrees)."""
    assert _reasons(curriculum, settle_stance_width_change_m=0.0875) == {"max_settle_stance_width_change_m"}
    assert _reasons(curriculum, settle_touchdowns=3.0) == {"max_settle_touchdowns"}
    # The statue's spawn landing (one foot spawning airborne) plus one re-plant is admitted.
    assert _reasons(curriculum, settle_touchdowns=2.0) == set()
    assert _reasons(curriculum, episode_yaw_change_deg=45.3) == {"max_episode_yaw_change_deg"}


def test_the_r8_seed_44_hop_episode_is_a_hop_for_the_rail(curriculum):
    """Seed 7065's whole-episode hop under the r8 checkpoint: support, touchdowns, drift and saturation."""
    from environments.shared.curriculum.stance_gate_v2 import is_hop_or_fall

    thresholds = StanceV2Thresholds.from_curriculum(curriculum)
    clean = episode_stance_metrics(statue_trace(1000), settle_steps=thresholds.settle_steps)
    hop = replace(
        clean,
        all_feet_support=0.70,
        touchdown_rate=3.75,
        window_displacement_m=1.31,
        max_actuator_saturation_fraction=0.31,
    )
    reasons = classify_stance_episode(hop, thresholds, horizon=1000)
    assert is_hop_or_fall(reasons)
    assert not is_hop_or_fall(classify_stance_episode(replace(clean, max_sole_tilt_deg=2.5), thresholds, horizon=1000))


# ── the pad's centre of pressure (decision D-D28, 2026-10-07) ─────────────────

#: The r8 seed-44 robust_best left foot on seed 5058, clean under every other bar: on its pad's front edge on
#: every window step, the pad 0.67 degrees toe-down on two loaded corners and carrying 0.974 of the foot's load.
FRONT_EDGE_FOOT: dict[str, float] = {
    "min_sole_contacts": 2.026,
    "max_sole_tilt_deg": 0.691,
    "max_sole_corner_lift_m": 0.00188,
    "max_sole_cop_fore_aft": 0.999,
}


def test_the_pad_cop_bar_is_declared_beside_the_sole_bars(curriculum):
    assert curriculum["max_sole_cop_fore_aft"] == 0.80
    assert StanceV2Thresholds.from_curriculum(curriculum).declared()["max_sole_cop_fore_aft"] == 0.80
    # It does not replace min_sole_contacts, which stays where the statue's out-of-sample 2.086 put it.
    assert curriculum["min_sole_contacts"] == 1.5


def test_a_pad_on_its_front_edge_is_refused_by_the_cop_bar_alone(curriculum):
    """Every other sole bar admits the seed-44 left foot: the CoP bar is the one that sees it."""
    assert _reasons(curriculum, **FRONT_EDGE_FOOT) == {"max_sole_cop_fore_aft"}
    without = {key: value for key, value in curriculum.items() if key != "max_sole_cop_fore_aft"}
    assert _reasons(without, **FRONT_EDGE_FOOT) == set()
    # Its best episode over 140 on four seed blocks (7081, on three loaded points), and seed 42's heel-to-toe
    # rock, which reads its edges though its signed mean is centred.
    assert _reasons(curriculum, min_sole_contacts=3.0, max_sole_tilt_deg=0.72, max_sole_cop_fore_aft=0.850) == {
        "max_sole_cop_fore_aft"
    }
    assert _reasons(curriculum, max_sole_cop_fore_aft=0.962) == {"max_sole_cop_fore_aft"}


@pytest.mark.parametrize(
    "value",
    [
        0.362,  # the statue's worst on 3042-3081, 7042-7081 and 3162-3201
        0.480,  # its worst out of sample: seed 5048's fore-aft sway
        0.745,  # N(0, 0.05) command jitter on every step of the statue (13042-13081; 0.717 on 3042-3061)
        0.790,  # the D-D27 369k-step study checkpoint's flat pad, loaded forward, on 7042-7081
    ],
)
def test_the_statue_and_a_flat_forward_loaded_pad_clear_the_cop_bar(curriculum, value):
    assert _reasons(curriculum, max_sole_cop_fore_aft=value) == set()


def test_only_the_trex_stance_declares_the_cop_bar():
    """An undeclared key is not in a stage's gate view, so no other stage's gate digest moves (recovery extends the
    stance's [env], not its [curriculum])."""
    from environments.shared.config import SPECIES_NAMES
    from environments.shared.curriculum.gate_schema import gate_config_view
    from environments.shared.stage_manifest import load_stage_manifest

    declaring = []
    for species in SPECIES_NAMES:
        for entry in load_stage_manifest(species).stages:
            block = load_stage_config(species, entry.reference).get("curriculum_kwargs", {})
            if "max_sole_cop_fore_aft" in gate_config_view(block)["thresholds"]:
                declaring.append((species, entry.id))
    assert declaring == [("trex", "stance")]
