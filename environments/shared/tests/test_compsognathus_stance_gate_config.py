"""The anatomical compsognathus stance's ``stance_quality/v2`` declaration (decision D-D26, 2026-10-07).

``configs/compsognathus/stance.toml`` moved from ``stance_quality/v1``, which
both certified r1 stances cleared while marching on one digit tip, to the
floor-truth gate with physics r2 and the soft-cubic leg interface.  Its bars
were set on the r2 statue's 40-episode panel and validated against the two
certified r1 marches (``20260921_203149`` and ``20261001_225856``), rolled on
the r1 plant they load on.  This module pins the declaration in the ways
that each fail differently:

1. **It validates as the kind it declares**, strictly, with its settle window
   non-empty after the spawn grace at the stage's own control step, and the
   [env]'s settled stance-width reference ending where the gate's settle ends.
2. **The sole, foot, settle, width and saturation bars are declared.**
3. **A recorded march episode is refused by four independent families**: the
   window's support bars (one foot down a tenth of the time, three
   touchdowns per foot per second), the sole bars (a pad standing 7-11
   degrees off level on no contact point, which the centre-of-pressure bar
   reads as an edge), the support-geom bars (no geom loaded on half of any
   step) and the window peak (a step lands at 2.48 BW).  Each refuses it
   alone.
4. **The statue's and the jittered statue's least favourable episodes are
   clean**, and the settle width bar sits at about three times the statue's
   worst settle width change (the trex r8 runs re-seated their feet in the
   settle by 3.6 to 4.4 times their statue's).
5. **The bar and the rails agree with the numbers they were derived from**,
   recomputed here rather than copied.
6. **A recorded post-settle two-foot hop is refused by the window hop pair**:
   its flights last 1-3 of a step's ten substeps, so every step-level bar
   reads a statue, and only the substep pair over the window sees it.
7. **The window peak sits between the jittered statue at sigma 0.03 and the
   softest-landing square-wave hop**: the pair admits the first, and the peak
   alone refuses the hop episode that lands inside the sigma-0.05 jitter's
   range, and that jitter's hardest landing with it.
8. **A soft bounce and a one-foot flutter are certified**: the block's
   measured blind spot on this plant (KNOWN_ISSUES), pinned so that a gate
   revision that closes it updates the records with it.
9. **The pad centre-of-pressure bar (D-D28) refuses an edge stance that
   every other bar admits**: pads leaned onto their front or heel edges,
   level, on two loaded corners with the digits loaded, the swept-back
   crouch, and the toe tap of item 8's blind spot, which rocks its pad from
   edge to edge; the statue, the jittered statue and the short PPO runs'
   means read under 0.41.

Each recorded episode is scored on a synthetic statue's row with its own
readings put in, and carries its readings of the metrics added later (the
window hop pair, the pad's centre of pressure): one it lacks reads as
unmeasured, never as the synthetic statue's.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import pytest

from environments.shared.config import load_stage_config
from environments.shared.curriculum.gate_schema import validate_gate_config
from environments.shared.curriculum.recovery_gate import binomial_lcb
from environments.shared.curriculum.stance_gate_v2 import (
    CRITERION_BAR_RANGES,
    STANCE_GATE_V2_KIND,
    STANCE_V2_THRESHOLD_KEYS,
    StanceV2Thresholds,
    StatueReference,
    classify_stance_episode,
)
from environments.shared.gait.stance_metrics import (
    STANCE_METRIC_LATER_FIELDS,
    episode_stance_metrics,
    spawn_grace_steps,
)

from .test_gait_stance_metrics import statue_trace

#: The box pad's flatness bars: tilt, corner lift and loaded contact points.
SOLE_KEYS = ("max_sole_tilt_deg", "max_sole_corner_lift_m", "min_sole_contacts")
#: The support-geom bars: pad and digits II-IV of each foot loaded.
FOOT_KEYS = ("min_support_geom_duty", "min_support_geom_coverage")
#: The window's support bars: both legs down, no touchdowns.
SUPPORT_KEYS = ("min_all_feet_support", "max_touchdown_rate")
#: The settle-window bars: the reset hop and its impact, and the re-seat.
SETTLE_KEYS = ("max_settle_airborne_substeps", "max_settle_peak_floor_force_bw", "max_settle_stance_width_change_m")
#: The window hop pair: both feet off the floor on a substep, and the landing's peak, over the window.
WINDOW_HOP_KEYS = ("max_window_airborne_substeps", "max_window_peak_floor_force_bw")
#: The families that each refuse the recorded march alone: the sole family with the pad's centre of pressure, and
#: the window pair's peak (the march flies 13 window substeps, under the airborne bar).
FAMILIES = {
    "support": SUPPORT_KEYS,
    "sole": SOLE_KEYS + ("max_sole_cop_fore_aft",),
    "foot": FOOT_KEYS,
    "window peak": ("max_window_peak_floor_force_bw",),
}

#: The r2 statue panel's mean per-foot load share (right, left) and its standing reward under the stance [env].
STATUE = StatueReference(n_episodes=40, n_full_horizon=40, mean_reward=4570.4, foot_load_share=(0.5, 0.5))

#: The 20261001_225856 march's quietest panel episode (seed 3042), as the v2 report measured it on the r1
#: plant: it reaches the horizon with an ordinary settle (a 2.05 BW peak, 2.4 mm of width change), standing on
#: one digit-III tip at a time, so no pad carries a loaded contact point (its centre of pressure reads 1 on every
#: window step), and lands at up to 2.48 BW in the window.
QUIET_MARCH: dict[str, Any] = {
    "all_feet_support": 0.1087,
    "touchdown_rate": 3.125,
    "window_displacement_m": 0.0212,
    "min_foot_load_share": 0.4685,
    "min_foot_load_share_windowed": 0.4168,
    "foot_load_share": (0.4685, 0.5315),
    "max_actuator_saturation_fraction": 0.0025,
    "settle_airborne_substeps": 5.0,
    "settle_peak_floor_force_bw": 2.0452,
    "settle_stance_width_change_m": 0.0024,
    "min_support_geom_duty": 0.0,
    "min_support_geom_coverage": 0.1344,
    "max_sole_tilt_deg": 7.2157,
    "max_sole_corner_lift_m": 0.0076,
    "min_sole_contacts": 0.0,
    "max_sole_cop_fore_aft": 1.0,
    "window_airborne_substeps": 13.0,
    "window_peak_floor_force_bw": 2.4765,
}

#: The settle width change of the 20260921_203149 march's least re-seating episode (seed 3058; 11.5-33 mm).
REALIGNING_MARCH_SETTLE_WIDTH_CHANGE_M = 0.0115

#: The r2 statue's least favourable episodes of the certification panel: the largest settle width change
#: (seed 3061) and the worst pad tilt and fewest contact points (seed 3071).
STATUE_WORST: dict[str, dict[str, Any]] = {
    "widest_settle": {
        "all_feet_support": 1.0,
        "touchdown_rate": 0.0,
        "window_displacement_m": 0.0,
        "min_foot_load_share": 0.4999,
        "min_foot_load_share_windowed": 0.4992,
        "foot_load_share": (0.4999, 0.5001),
        "max_actuator_saturation_fraction": 0.0,
        "settle_airborne_substeps": 0.0,
        "settle_peak_floor_force_bw": 1.0047,
        "settle_stance_width_change_m": 0.0036,
        "min_support_geom_duty": 1.0,
        "min_support_geom_coverage": 1.0,
        "max_sole_tilt_deg": 0.0192,
        "max_sole_corner_lift_m": 0.0,
        "min_sole_contacts": 3.8609,
        "max_sole_cop_fore_aft": 0.151,
        "window_airborne_substeps": 0.0,
        "window_peak_floor_force_bw": 1.0001,
    },
    "most_tilted": {
        "all_feet_support": 1.0,
        "touchdown_rate": 0.0,
        "window_displacement_m": 0.0,
        "min_foot_load_share": 0.5,
        "min_foot_load_share_windowed": 0.4999,
        "foot_load_share": (0.5, 0.5),
        "max_actuator_saturation_fraction": 0.0,
        "settle_airborne_substeps": 0.0,
        "settle_peak_floor_force_bw": 1.0085,
        "settle_stance_width_change_m": 0.0035,
        "min_support_geom_duty": 1.0,
        "min_support_geom_coverage": 1.0,
        "max_sole_tilt_deg": 0.0207,
        "max_sole_corner_lift_m": 0.0,
        "min_sole_contacts": 3.8261,
        "max_sole_cop_fore_aft": 0.1429,
        "window_airborne_substeps": 0.0,
        "window_peak_floor_force_bw": 1.0001,
    },
}

#: The r2 statue with N(0, 0.05) noise on its zero command every step (seeds 3042-3081, noise stream
#: default_rng(123456 + 17 i)): its lowest support-geom duty (seed 3044) and its highest settle peak
#: (seed 3074), each as the v2 report measured it.  Every bar but the window peak admits all 40 of the panel's
#: episodes (the peak refuses 22: JITTERED_STATUE_LANDING below); these two land at 1.99 and 1.95 BW, under it.
JITTERED_STATUE: dict[str, dict[str, Any]] = {
    "lowest_duty": {
        "all_feet_support": 1.0,
        "touchdown_rate": 0.0,
        "window_displacement_m": 0.0337,
        "min_foot_load_share": 0.4998,
        "min_foot_load_share_windowed": 0.4815,
        "foot_load_share": (0.5002, 0.4998),
        "max_actuator_saturation_fraction": 0.0,
        "settle_airborne_substeps": 4.0,
        "settle_peak_floor_force_bw": 1.8755,
        "settle_stance_width_change_m": 0.0035,
        "min_support_geom_duty": 0.81,
        "min_support_geom_coverage": 0.9044,
        "max_sole_tilt_deg": 0.1982,
        "max_sole_corner_lift_m": 0.0002,
        "min_sole_contacts": 2.3465,
        "max_sole_cop_fore_aft": 0.3795,
        "window_airborne_substeps": 10.0,
        "window_peak_floor_force_bw": 1.9926,
    },
    "highest_settle_peak": {
        "all_feet_support": 1.0,
        "touchdown_rate": 0.0,
        "window_displacement_m": 0.0326,
        "min_foot_load_share": 0.499,
        "min_foot_load_share_windowed": 0.4807,
        "foot_load_share": (0.501, 0.499),
        "max_actuator_saturation_fraction": 0.0,
        "settle_airborne_substeps": 3.0,
        "settle_peak_floor_force_bw": 2.0843,
        "settle_stance_width_change_m": 0.0014,
        "min_support_geom_duty": 0.8562,
        "min_support_geom_coverage": 0.9303,
        "max_sole_tilt_deg": 0.1728,
        "max_sole_corner_lift_m": 0.0002,
        "min_sole_contacts": 2.4286,
        "max_sole_cop_fore_aft": 0.3717,
        "window_airborne_substeps": 10.0,
        "window_peak_floor_force_bw": 1.9545,
    },
}

#: The statue's largest settle width change, on the certification panel and out of sample (seeds 7042-7081 and
#: 9042-9081).
STATUE_SETTLE_WIDTH_CHANGE_M = (0.0036, 0.0032)

#: A two-foot hop that starts after the settle (both legs in phase: knees +0.15, ankles and hip pitch -0.15, a
#: 10 Hz square wave from step 200; the D-D26 review's), seed 3042, as the v2 report measured it on the landed
#: plant.  Both legs are down on every window step and the settle is the statue's, but both feet leave the floor
#: on 472 window substeps and land at 3.99 BW; on this plant it also drifts 0.12 m over the window.
POST_SETTLE_HOP: dict[str, Any] = {
    "all_feet_support": 1.0,
    "flight_fraction": 0.0,
    "touchdown_rate": 0.0,
    "window_displacement_m": 0.1241,
    "min_foot_load_share": 0.4998,
    "min_foot_load_share_windowed": 0.4964,
    "foot_load_share": (0.5002, 0.4998),
    "max_actuator_saturation_fraction": 0.0,
    "settle_airborne_substeps": 0.0,
    "settle_peak_floor_force_bw": 1.0054,
    "settle_stance_width_change_m": 0.0011,
    "min_support_geom_duty": 0.805,
    "min_support_geom_coverage": 0.9513,
    "max_sole_tilt_deg": 0.1477,
    "max_sole_corner_lift_m": 0.0001,
    "min_sole_contacts": 2.8621,
    "window_airborne_substeps": 472.0,
    "window_peak_floor_force_bw": 3.9877,
    "max_sole_cop_fore_aft": 0.5144,
}

#: The jittered statue's (sigma 0.03, seeds 3042-3081) most airborne window (seed 3069) and highest window peak
#: (seed 3073) on the landed plant: the noise bounces this light plant in the window too.
JITTERED_STATUE_WINDOW: dict[str, dict[str, float]] = {
    "most_airborne": {
        "window_airborne_substeps": 17.0,
        "window_peak_floor_force_bw": 1.5803,
        "max_sole_cop_fore_aft": 0.2689,
    },
    "highest_peak": {
        "window_airborne_substeps": 7.0,
        "window_peak_floor_force_bw": 1.7412,
        "max_sole_cop_fore_aft": 0.282,
    },
}

#: The softest landing of the post-settle hop at amplitude 0.07 (knees +0.07, ankles and hip pitch -0.07, a 10 Hz
#: square wave from step 200), seed 9076, as the v2 report measured it: 9 airborne window substeps and a 2.23 BW
#: landing, and a statue on every other bar (it drifts 45 mm, under the displacement guard).
SOFTEST_POST_SETTLE_HOP: dict[str, Any] = {
    "all_feet_support": 1.0,
    "flight_fraction": 0.0,
    "touchdown_rate": 0.0,
    "window_displacement_m": 0.0449,
    "min_foot_load_share": 0.5,
    "min_foot_load_share_windowed": 0.4975,
    "foot_load_share": (0.5, 0.5),
    "max_actuator_saturation_fraction": 0.0,
    "settle_airborne_substeps": 0.0,
    "settle_peak_floor_force_bw": 1.0049,
    "settle_stance_width_change_m": 0.0018,
    "min_support_geom_duty": 1.0,
    "min_support_geom_coverage": 1.0,
    "max_sole_tilt_deg": 0.0961,
    "max_sole_corner_lift_m": 0.0001,
    "min_sole_contacts": 3.1391,
    "window_airborne_substeps": 9.0,
    "window_peak_floor_force_bw": 2.2312,
    "max_sole_cop_fore_aft": 0.1888,
}

#: The sigma-0.05 jittered statue's (seeds 3042-3081) most airborne window (seed 3055) and hardest landing
#: (seed 3058): the hop above lands inside both.
JITTERED_STATUE_LANDING: dict[str, dict[str, float]] = {
    "most_airborne": {
        "window_airborne_substeps": 28.0,
        "window_peak_floor_force_bw": 2.0081,
        "max_sole_cop_fore_aft": 0.3736,
    },
    "highest_peak": {
        "window_airborne_substeps": 9.0,
        "window_peak_floor_force_bw": 2.2655,
        "max_sole_cop_fore_aft": 0.3653,
    },
}

#: Two scripted stances from step 200 that the block certifies on the landed plant (KNOWN_ISSUES; the audit's §11
#: hack table), each the least favourable episode of its 40-episode report panel (seeds 3042-3081): the post-settle
#: hop's pattern driven by a 5 Hz sine (a = 0.09; seed 3053, its most airborne window: 32 substeps with both feet off
#: the floor, landing at 1.80 BW), and the right leg alone pumped at 6.25 Hz (a = 0.07; seed 3073, its least-covered
#: feet), whose right foot lifts for one substep at a time and never with the left.
ADMITTED_SCRIPTS: dict[str, dict[str, Any]] = {
    "sine_bounce": {
        "all_feet_support": 1.0,
        "touchdown_rate": 0.0,
        "window_displacement_m": 0.0044,
        "min_foot_load_share": 0.5,
        "min_foot_load_share_windowed": 0.5,
        "foot_load_share": (0.5, 0.5),
        "max_actuator_saturation_fraction": 0.0,
        "settle_airborne_substeps": 0.0,
        "settle_peak_floor_force_bw": 1.0065,
        "settle_stance_width_change_m": 0.0006,
        "min_support_geom_duty": 1.0,
        "min_support_geom_coverage": 1.0,
        "max_sole_tilt_deg": 0.0357,
        "max_sole_corner_lift_m": 0.0,
        "min_sole_contacts": 3.6449,
        "window_airborne_substeps": 32.0,
        "window_peak_floor_force_bw": 1.8025,
        "max_sole_cop_fore_aft": 0.1439,
    },
    "one_leg_pump": {
        "all_feet_support": 1.0,
        "touchdown_rate": 0.0,
        "window_displacement_m": 0.0225,
        "min_foot_load_share": 0.4559,
        "min_foot_load_share_windowed": 0.447,
        "foot_load_share": (0.4559, 0.5441),
        "max_actuator_saturation_fraction": 0.0,
        "settle_airborne_substeps": 0.0,
        "settle_peak_floor_force_bw": 1.0134,
        "settle_stance_width_change_m": 0.0019,
        "min_support_geom_duty": 0.5738,
        "min_support_geom_coverage": 0.8331,
        "max_sole_tilt_deg": 0.2087,
        "max_sole_corner_lift_m": 0.0002,
        "min_sole_contacts": 2.0035,
        "window_airborne_substeps": 0.0,
        "window_peak_floor_force_bw": 1.6299,
        "max_sole_cop_fore_aft": 0.4516,
    },
}


#: The most each admitted group reads of ``max_sole_cop_fore_aft`` on the landed plant (D-D28, measured after it
#: landed): the statue on five seed blocks (200 episodes, its pads' centres of pressure on the heel side), the statue
#: with N(0, sigma) command jitter at sigma 0.03 and 0.05 (seed blocks 3042, 7042 and 9042, and 11042 with an
#: independent noise stream), and the two short PPO runs' deterministic means (seeds 3042-3081).
COP_ADMITTED: dict[str, float] = {"statue": 0.155, "jitter_0.03": 0.289, "jitter_0.05": 0.404, "ppo_means": 0.246}

#: Three stances that every other bar of the block admits, each the least favourable episode of a 40-episode panel
#: (seeds 3042-3081) as the v2 report measured it on the landed plant: both MTP servo targets ramped over the first
#: 300 steps to -0.10 rad, which leans both pads onto their front edges (seed 3071; the panel is clean on 40/40
#: without the bar), and to +0.08 rad, onto their heel edges (seed 3076; 40/40), each pad level on about two loaded
#: corners with its digits loaded; and the swept-back crouch (hip pitch +0.3, ankles -0.2 rad; seed 3042, one of its
#: 33 episodes that no other bar refuses, every one of them on its pads' front edges throughout the window).
EDGE_STANCES: dict[str, dict[str, Any]] = {
    "front_edges": {
        "all_feet_support": 1.0,
        "touchdown_rate": 0.0,
        "window_displacement_m": 0.0092,
        "min_foot_load_share": 0.5,
        "min_foot_load_share_windowed": 0.4998,
        "foot_load_share": (0.5, 0.5),
        "max_actuator_saturation_fraction": 0.0,
        "settle_airborne_substeps": 0.0,
        "settle_peak_floor_force_bw": 1.021,
        "settle_stance_width_change_m": 0.0036,
        "min_support_geom_duty": 1.0,
        "min_support_geom_coverage": 1.0,
        "max_sole_tilt_deg": 0.0423,
        "max_sole_corner_lift_m": 0.0,
        "min_sole_contacts": 2.3478,
        "window_airborne_substeps": 0.0,
        "window_peak_floor_force_bw": 1.005,
        "max_sole_cop_fore_aft": 0.8598,
    },
    "heel_edges": {
        "all_feet_support": 1.0,
        "touchdown_rate": 0.0,
        "window_displacement_m": 0.0076,
        "min_foot_load_share": 0.5,
        "min_foot_load_share_windowed": 0.5,
        "foot_load_share": (0.5, 0.5),
        "max_actuator_saturation_fraction": 0.0,
        "settle_airborne_substeps": 0.0,
        "settle_peak_floor_force_bw": 1.0076,
        "settle_stance_width_change_m": 0.0004,
        "min_support_geom_duty": 1.0,
        "min_support_geom_coverage": 1.0,
        "max_sole_tilt_deg": 0.0406,
        "max_sole_corner_lift_m": 0.0,
        "min_sole_contacts": 2.3591,
        "window_airborne_substeps": 0.0,
        "window_peak_floor_force_bw": 1.002,
        "max_sole_cop_fore_aft": 0.8436,
    },
    "swept_crouch": {
        "all_feet_support": 1.0,
        "touchdown_rate": 0.0,
        "window_displacement_m": 0.0,
        "min_foot_load_share": 0.4999,
        "min_foot_load_share_windowed": 0.4986,
        "foot_load_share": (0.4999, 0.5001),
        "max_actuator_saturation_fraction": 0.0,
        "settle_airborne_substeps": 5.0,
        "settle_peak_floor_force_bw": 1.3994,
        "settle_stance_width_change_m": 0.0014,
        "min_support_geom_duty": 1.0,
        "min_support_geom_coverage": 1.0,
        "max_sole_tilt_deg": 0.0432,
        "max_sole_corner_lift_m": 0.0,
        "min_sole_contacts": 2.0,
        "window_airborne_substeps": 0.0,
        "window_peak_floor_force_bw": 1.0001,
        "max_sole_cop_fore_aft": 1.0,
    },
}


#: The right toe alone driven at +-0.25 and 10 Hz from step 200, the least favourable episode of its 40-episode report
#: panel (seed 3044): its foot lifts for one substep at a time and every other bar admits it (D-D26 recorded it among
#: the stances the block certifies, 40/40), but each tap rocks the right pad from edge to edge, 0.826-0.848 over the
#: panel.
TOE_TAP: dict[str, Any] = {
    "all_feet_support": 1.0,
    "touchdown_rate": 0.0,
    "window_displacement_m": 0.0221,
    "min_foot_load_share": 0.4851,
    "min_foot_load_share_windowed": 0.4742,
    "foot_load_share": (0.4851, 0.5149),
    "max_actuator_saturation_fraction": 0.0,
    "settle_airborne_substeps": 0.0,
    "settle_peak_floor_force_bw": 1.0084,
    "settle_stance_width_change_m": 0.0022,
    "min_support_geom_duty": 0.5663,
    "min_support_geom_coverage": 0.8741,
    "max_sole_tilt_deg": 0.2262,
    "max_sole_corner_lift_m": 0.0002,
    "min_sole_contacts": 1.653,
    "window_airborne_substeps": 0.0,
    "window_peak_floor_force_bw": 1.2902,
    "max_sole_cop_fore_aft": 0.8262,
}


@pytest.fixture(scope="module")
def stage() -> dict[str, Any]:
    return load_stage_config("compsognathus", "stance")


@pytest.fixture(scope="module")
def curriculum(stage: dict[str, Any]) -> dict[str, Any]:
    return dict(stage["curriculum_kwargs"])


def _reasons(curriculum: dict[str, Any], metrics: dict[str, Any]) -> set[str]:
    """The criteria *metrics* fails, scored on a synthetic statue's row with them put in.

    A metric added later that *metrics* does not record reads as unmeasured (NaN), as the judge reads a row
    recorded before it: never as the synthetic statue's reading, which no recorded episode measured.
    """
    thresholds = StanceV2Thresholds.from_curriculum(curriculum)
    clean = episode_stance_metrics(statue_trace(1000), settle_steps=thresholds.settle_steps)
    unrecorded: dict[str, Any] = {name: math.nan for name in STANCE_METRIC_LATER_FIELDS if name not in metrics}
    episode = replace(clean, **unrecorded, **metrics)
    reasons = classify_stance_episode(episode, thresholds, horizon=1000, statue=STATUE)
    return {reason.split(":", 1)[0] for reason in reasons}


def _without(curriculum: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    """*curriculum* with *keys* declared so loosely that they can never fail (a required key cannot be dropped).

    A key whose bar must lie inside a range (``CRITERION_BAR_RANGES``: the pad centre-of-pressure bar, an optional
    key) is dropped instead.
    """
    ranged = {key for key, *_ in CRITERION_BAR_RANGES}
    loose = dict(curriculum)
    for key in keys:
        if key in ranged:
            del loose[key]
        else:
            loose[key] = 1e9 if key.startswith("max_") else -1e9
    return loose


def test_the_stance_declares_the_floor_truth_gate_and_validates_as_it(stage, curriculum):
    assert curriculum["gate_kind"] == STANCE_GATE_V2_KIND
    assert validate_gate_config("stance", curriculum, advancement_enabled=True) == STANCE_GATE_V2_KIND
    thresholds = StanceV2Thresholds.from_curriculum(curriculum)
    env = stage["env_kwargs"]
    thresholds.validate_settle_window(0.002 * int(env["frame_skip"]))
    # stance_quality/v1's keys are a superseded record in the file's comments, not keys.
    assert not {"max_unsupported_duty", "max_unsupported_duty_ucb", "min_avg_episode_length"} & set(curriculum)
    assert {key for key in curriculum if key in STANCE_V2_THRESHOLD_KEYS} == set(thresholds.declared()) | {
        "required_consecutive"
    }


def test_the_settle_window_and_the_settled_width_reference_end_together(stage, curriculum):
    env = stage["env_kwargs"]
    grace = spawn_grace_steps(0.002 * int(env["frame_skip"]))
    assert (grace, curriculum["settle_steps"]) == (5, 200)
    # The width term targets the animal's own width where the gate's settle ends, and pays only after it.
    assert env["stance_width_reference"] == "settled"
    assert env["stance_width_settle_steps"] == curriculum["settle_steps"]


@pytest.mark.parametrize(
    "key",
    SOLE_KEYS
    + ("max_sole_cop_fore_aft",)
    + FOOT_KEYS
    + SUPPORT_KEYS
    + SETTLE_KEYS
    + WINDOW_HOP_KEYS
    + ("max_actuator_saturation_fraction",),
)
def test_the_sole_foot_settle_width_and_saturation_bars_are_declared(curriculum, key):
    assert key in curriculum
    assert StanceV2Thresholds.from_curriculum(curriculum).declared()[key] == curriculum[key]


def test_a_recorded_march_is_refused_by_four_independent_families(curriculum):
    every_family = tuple(key for keys in FAMILIES.values() for key in keys)
    assert _reasons(curriculum, QUIET_MARCH) == set(every_family)
    assert _reasons(_without(curriculum, every_family), QUIET_MARCH) == set()
    for name, keys in FAMILIES.items():
        others = tuple(key for key in every_family if key not in keys)
        assert _reasons(_without(curriculum, others), QUIET_MARCH) == set(keys), name


@pytest.mark.parametrize("case", sorted(STATUE_WORST))
def test_the_statues_least_favourable_episodes_are_clean(curriculum, case):
    assert _reasons(curriculum, STATUE_WORST[case]) == set()


@pytest.mark.parametrize("case", sorted(JITTERED_STATUE))
def test_the_jittered_statue_is_clean(curriculum, case):
    """The noise bounces this light plant in the settle (2.08 BW, 3 airborne substeps): the settle bars admit it."""
    assert _reasons(curriculum, JITTERED_STATUE[case]) == set()


def test_a_recorded_post_settle_hop_is_refused_by_the_window_hop_pair(curriculum):
    """The step-level bars read a statue; the substep pair over the window refuses it, each bar alone."""
    reasons = _reasons(curriculum, POST_SETTLE_HOP)
    assert set(WINDOW_HOP_KEYS) <= reasons
    assert not reasons & set(SUPPORT_KEYS + SETTLE_KEYS + SOLE_KEYS + FOOT_KEYS)
    # On this plant the hop also drifts past the displacement guard; without it the pair still refuses alone.
    assert reasons - set(WINDOW_HOP_KEYS) <= {"max_window_displacement_m"}
    quiet = _without(curriculum, ("max_window_displacement_m",))
    assert _reasons(quiet, POST_SETTLE_HOP) == set(WINDOW_HOP_KEYS)
    for key in WINDOW_HOP_KEYS:
        others = tuple(other for other in WINDOW_HOP_KEYS if other != key) + ("max_window_displacement_m",)
        assert _reasons(_without(curriculum, others), POST_SETTLE_HOP) == {key}
    assert _reasons(_without(curriculum, WINDOW_HOP_KEYS + ("max_window_displacement_m",)), POST_SETTLE_HOP) == set()


@pytest.mark.parametrize("case", sorted(JITTERED_STATUE_WINDOW))
def test_the_window_hop_pair_admits_the_jittered_statue(curriculum, case):
    assert _reasons(curriculum, JITTERED_STATUE_WINDOW[case]) == set()


def test_the_window_peak_refuses_the_softest_hop_and_the_noise_that_lands_like_it(curriculum):
    """No pair of window bars admits the sigma-0.05 jitter and refuses the hop: the bar refuses both.

    The hop's softest episode flies fewer substeps and lands lower than the jitter's worst on each, so a bar that
    admitted that jitter would admit the hop; the peak alone refuses it, and every other bar reads a statue.
    """
    hop = SOFTEST_POST_SETTLE_HOP
    for key in ("window_airborne_substeps", "window_peak_floor_force_bw"):
        assert hop[key] <= max(case[key] for case in JITTERED_STATUE_LANDING.values())
    assert _reasons(curriculum, hop) == {"max_window_peak_floor_force_bw"}
    assert _reasons(_without(curriculum, ("max_window_peak_floor_force_bw",)), hop) == set()
    assert _reasons(curriculum, JITTERED_STATUE_LANDING["most_airborne"]) == {"max_window_peak_floor_force_bw"}
    assert _reasons(curriculum, JITTERED_STATUE_LANDING["highest_peak"]) == {"max_window_peak_floor_force_bw"}
    worst_admitted = max(case["window_peak_floor_force_bw"] for case in JITTERED_STATUE_WINDOW.values())
    assert worst_admitted < curriculum["max_window_peak_floor_force_bw"] < hop["window_peak_floor_force_bw"]


@pytest.mark.parametrize("case", sorted(ADMITTED_SCRIPTS))
def test_a_soft_bounce_and_a_one_foot_flutter_read_a_statue(curriculum, case):
    """The block's measured blind spot on this plant, pinned so that closing it updates the records with it.

    A leg is down on a step when it is loaded on half its substeps, and the window pair counts only substeps with
    both feet off the floor and the peak landing: a bounce that lands under the peak bar and flies under the
    airborne bar, and a foot lifted for one substep at a time, read a statue on every bar.  A gate revision that
    refuses them (a periodicity criterion; KNOWN_ISSUES) changes this pin.
    """
    episode = ADMITTED_SCRIPTS[case]
    assert episode["window_airborne_substeps"] <= curriculum["max_window_airborne_substeps"]
    assert episode["window_peak_floor_force_bw"] <= curriculum["max_window_peak_floor_force_bw"]
    assert _reasons(curriculum, episode) == set()


def test_the_settle_width_bar_is_about_three_times_the_statues_worst(curriculum):
    bar = curriculum["max_settle_stance_width_change_m"]
    assert 2.5 * max(STATUE_SETTLE_WIDTH_CHANGE_M) <= bar <= 3.5 * min(STATUE_SETTLE_WIDTH_CHANGE_M)
    # Every 20260921_203149 episode re-seats past it in its settle.
    assert REALIGNING_MARCH_SETTLE_WIDTH_CHANGE_M > bar


def test_the_bar_admits_37_of_40_and_refuses_36(curriculum):
    n = curriculum["min_eval_episodes"]
    bar = curriculum["min_clean_stance_lcb"]
    assert n == 40
    assert binomial_lcb(37, n) >= bar > binomial_lcb(36, n)


def test_the_reward_rails_are_their_fractions_of_the_statue(curriculum):
    """min_avg_reward is 0.60 of the statue rounded to 10; the reference is pinned to the plant it was measured on.

    The reference is the r2 statue's standing reward under the stance [env] (4570.4, 40 episodes, seeds
    3042-3081); test_statue_constant_freshness.py checks the physics pin.  The collapse detector stays
    unarmed: the stage declares neither a floor nor a fraction of the reference.
    """
    reference = curriculum["collapse_peak_floor_reference"]
    assert reference == STATUE.mean_reward and curriculum["statue_constants_physics_revision"] == 2
    assert curriculum["min_avg_reward"] == round(0.60 * reference, -1)
    assert curriculum["min_avg_reward_statue_ratio"] == 0.60
    assert "collapse_peak_floor" not in curriculum and "collapse_peak_floor_fraction" not in curriculum


# ── the pad's centre of pressure (decision D-D28, measured on this plant) ─────


def test_the_pad_cop_bar_is_declared_beside_the_sole_bars(curriculum):
    assert curriculum["max_sole_cop_fore_aft"] == 0.70
    assert StanceV2Thresholds.from_curriculum(curriculum).declared()["max_sole_cop_fore_aft"] == 0.70
    # The statue stands on the heel side of its pads' centres, under a quarter of the bar.
    assert 4 * COP_ADMITTED["statue"] < curriculum["max_sole_cop_fore_aft"]


@pytest.mark.parametrize("case", sorted(COP_ADMITTED))
def test_the_statue_the_jitter_and_the_ppo_means_clear_the_cop_bar(curriculum, case):
    """The statue's most tilted episode with each group's highest reading in place of its own."""
    episode = {**STATUE_WORST["most_tilted"], "max_sole_cop_fore_aft": COP_ADMITTED[case]}
    assert _reasons(curriculum, episode) == set()


@pytest.mark.parametrize("case", sorted(EDGE_STANCES))
def test_an_edge_stance_every_other_bar_admits_is_refused_by_the_cop_bar_alone(curriculum, case):
    """Level pads on two loaded corners with every digit loaded: tilt, corner lift, contacts and the support-geom bars
    all read them as flat, and only where along each pad the floor pushes sets them apart."""
    episode = EDGE_STANCES[case]
    assert _reasons(curriculum, episode) == {"max_sole_cop_fore_aft"}
    assert _reasons(_without(curriculum, ("max_sole_cop_fore_aft",)), episode) == set()


def test_the_toe_tap_the_block_certified_before_the_cop_bar_is_refused_by_it_alone(curriculum):
    """Of the six one-substep flutters and soft bounces the block certified on this plant (KNOWN_ISSUES), the toe tap
    rocks its pad onto its edges; the two bounces, the burst, the one-leg pump and the shuttle read 0.14-0.48."""
    assert _reasons(curriculum, TOE_TAP) == {"max_sole_cop_fore_aft"}
    assert _reasons(_without(curriculum, ("max_sole_cop_fore_aft",)), TOE_TAP) == set()


def test_the_cop_bar_sits_between_the_noise_and_the_edge_stances(curriculum):
    bar = curriculum["max_sole_cop_fore_aft"]
    admitted, edge = max(COP_ADMITTED.values()), min(case["max_sole_cop_fore_aft"] for case in EDGE_STANCES.values())
    assert admitted + 0.25 < bar < edge - 0.10
