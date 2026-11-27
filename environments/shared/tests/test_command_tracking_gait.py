"""The ``command_tracking/v1`` gait clause: its metrics, its bars, and that it tells a walk from a hop or a slide.

* **The metrics** (``gait.locomotion_metrics``) on synthetic floor-truth
  traces, one canonical gait each: a biped walk, an alternating run, a
  two-footed hop (simultaneous, and staggered inside and outside the 20 ms
  merge, side by side and with feet that swap past the lead dead band), a
  slide, a scoot, stacked feet, phantom support; a quadruped
  lateral-sequence walk, trot, pace, pronk, bound, slide and a three-legged
  gait.  GQ-9 (a) says a walk, a trot and a pace meet the quadruped floors and
  asks for the fixtures that prove it; they are here.
* **The bars**: ``gait_limbs`` is required and selects the required bars
  (GQ-8 (a)'s six separators on a biped, GQ-9 (a)'s three floors on a
  quadruped); each bar refuses exactly the episodes its metric puts past it,
  an unmeasured metric fails, the stand family is not judged on gait, and an
  episode measured on another animal is refused.  The provisional values are
  §4.3's.
* **Discrimination on the real plants**: the T. rex (a biped) and the
  brachiosaurus (a quadruped), each through the floor-truth recorder.  A
  scripted hop (the zero-action statue lifted off the floor by a periodic
  vertical force at its centre of mass: both feet leave and land together)
  and a scripted slide (the statue pushed along a low-friction floor: every
  foot down while the body travels) fail the clause, and so does the hop with
  its feet landing 30 ms apart (on its lead swaps, through the dead band);
  the repository's reference walk (``scripts/reference_walk.py``) passes it,
  as do the velociraptor's, the allosaurus' and the dibothrosuchus' reference
  walks.  The known limit is pinned too: a staggered hop whose feet swap
  their lead past the dead band is refused only by the optional flight cap.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from environments.shared.curriculum.command_tracking_gate import (
    BIPED_LIMBS,
    COMMAND_TRACKING_GATE_KIND,
    COMMAND_TRACKING_THRESHOLD_KEYS,
    CRUISE_FAMILY,
    GAIT_ALLOWED_KEYS,
    GAIT_CRITERIA,
    GAIT_CRITERION_KEYS,
    GAIT_LIMBS_KEY,
    GAIT_REQUIRED_KEYS,
    GAIT_THRESHOLD_KEYS,
    PROVISIONAL_BIPED_FLIGHT_FRACTION,
    PROVISIONAL_BIPED_GAIT_THRESHOLDS,
    PROVISIONAL_FOLLOW_THRESHOLDS,
    PROVISIONAL_QUADRUPED_GAIT_THRESHOLDS,
    PROVISIONAL_WALK_THRESHOLDS,
    QUADRUPED_LIMBS,
    STAND_FAMILY,
    CommandTrackingEpisode,
    FollowThresholds,
    GaitClause,
    WalkThresholds,
    classify_cruise_episode,
    classify_follow_episode,
    classify_stand_episode,
    evaluate_follow_profile,
    evaluate_walk_profile,
    gait_clause_from_curriculum,
    gait_failures,
    panel_measurement,
    reason_key,
    summarize_command_panel,
    thresholds_from_curriculum,
)
from environments.shared.curriculum.gate_schema import (
    GATE_SCHEMA_VERSION,
    GateSchemaError,
    gate_config_sha256,
    gate_config_view,
    validate_gate_config,
)
from environments.shared.direction_commands import DirectionCommandConfig
from environments.shared.gait import events
from environments.shared.gait.constants import DEBOUNCE_S, MEASUREMENT_VERSION
from environments.shared.gait.locomotion_metrics import (
    GAIT_MEASUREMENT,
    GAIT_WINDOW_S,
    LEAD_DEAD_BAND_L,
    LOCOMOTION_METRIC_FIELDS,
    LOCOMOTION_METRICS_VERSION,
    LocomotionEpisodeMetrics,
    episode_locomotion_metrics,
)
from environments.shared.gait.recorder import EpisodeTrace
from environments.shared.model_id import ModelId
from environments.shared.paths import REPOSITORY_ROOT

DT = 0.01
PERIOD = 1.0
STEPS = 600
BODY_WEIGHT = 100.0
BIPED_FEET = np.array([[0.0, -0.15, 0.0], [0.0, 0.15, 0.0]])
QUADRUPED_FEET = np.array([[0.3, -0.1, 0.0], [0.3, 0.1, 0.0], [-0.3, -0.1, 0.0], [-0.3, 0.1, 0.0]])

BIPED = gait_clause_from_curriculum(PROVISIONAL_BIPED_GAIT_THRESHOLDS)
QUADRUPED = gait_clause_from_curriculum(PROVISIONAL_QUADRUPED_GAIT_THRESHOLDS)
WALK_CONFIG = DirectionCommandConfig(
    cruise_speed=0.5,
    speed_range=(0.25, 0.5),
    stop_probability=0.15,
    turn_increment_max=0.0,
    switch_interval_s=100.0,
    initial_heading_offset_max=0.26,
)


# ── synthetic floor-truth traces ─────────────────────────────────────────────


def cyclic(phases: tuple[float, ...], stance: float, *, steps: int = STEPS, dt: float = DT) -> np.ndarray:
    """(steps, limbs) down states: limb i is down for the first *stance* of each cycle shifted by ``phases[i]``."""
    cycles = np.arange(steps) * dt / PERIOD
    return np.stack([((cycles - phase) % 1.0) < stance for phase in phases], axis=1)


def swing(phases: tuple[float, ...], *, steps: int = STEPS, dt: float = DT, amplitude: float = 0.2) -> np.ndarray:
    """(steps, limbs) fore-aft foot offsets from the root, one sinusoid per limb at its phase."""
    cycles = np.arange(steps) * dt / PERIOD
    return np.stack([amplitude * np.sin(2.0 * np.pi * (cycles - phase)) for phase in phases], axis=1)


def gait_trace(
    down: np.ndarray,
    *,
    dt: float = DT,
    foot_x: np.ndarray | None = None,
    speed: float = 0.5,
    touch: np.ndarray | None = None,
    interleg: np.ndarray | None = None,
    labels: tuple[str, ...] | None = None,
) -> EpisodeTrace:
    """A floor-truth trace whose limbs are down exactly where *down* says, the root moving along +x at *speed*.

    A down limb carries its share of body weight on every substep, an up one
    nothing; touch reads the floor (no phantom) and no leg touches another
    unless *touch* or *interleg* say otherwise.  The feet ride with the root
    at their stance positions, offset fore-aft by *foot_x*.
    """
    down = np.asarray(down, dtype=bool)
    steps, n = down.shape
    feet = BIPED_FEET if n == 2 else QUADRUPED_FEET
    labels = labels or (("r", "l") if n == 2 else ("fr", "fl", "rr", "rl"))
    load = np.where(down, BODY_WEIGHT / n, 0.0)
    zeros = np.zeros((steps, n))
    root = np.column_stack([speed * dt * np.arange(1, steps + 1), np.zeros(steps), np.full(steps, 0.5)])
    foot_pos = root[:, None, :] * np.array([1.0, 1.0, 0.0]) + feet[None]
    if foot_x is not None:
        foot_pos[:, :, 0] += foot_x
    return EpisodeTrace(
        model_id=ModelId("synthetic_biped" if n == 2 else "synthetic_quadruped", "anatomical"),
        dt=dt,
        frame_skip=5,
        body_weight_n=BODY_WEIGHT,
        labels=labels,
        support_geom_foot=tuple(range(n)),
        support_geom_names=tuple(f"{label}_sole" for label in labels),
        leg_floor_mean=load,
        leg_floor_min=load,
        leg_floor_max=load,
        leg_down_frac=down.astype(float),
        support_floor_mean=load,
        support_loaded_frac=down.astype(float),
        offsupport_floor_mean=zeros,
        nonleg_floor_mean=np.zeros(steps),
        nonleg_floor_max=np.zeros(steps),
        total_floor_mean=load.sum(axis=1),
        total_floor_max=load.sum(axis=1),
        feet_airborne_substeps=np.where(down.any(axis=1), 0, 5),
        interleg_force_mean=zeros,
        interleg_substep_frac=np.zeros(steps) if interleg is None else interleg,
        leg_body_force_mean=zeros,
        leg_body_substep_frac=np.zeros(steps),
        sole_contacts_mean=zeros,
        sole_cop=np.zeros((steps, n, 2)),
        foot_cop=zeros,
        foot_support_span=np.zeros((n, 2)),
        support_load_pos=zeros,
        touch=load if touch is None else touch,
        foot_pos=foot_pos,
        root_pos=root,
        root_quat=np.tile([1.0, 0.0, 0.0, 0.0], (steps, 1)),
        joint_qpos=np.zeros((steps, 1)),
        sole_tilt_deg=zeros,
        sole_roll_deg=zeros,
        sole_pitch_deg=zeros,
        sole_corner_lift=zeros,
        sole_min_height=zeros,
        action=np.zeros((steps, 1)),
        ctrl=np.zeros((steps, 1)),
        reward=np.zeros(steps),
        spawn_foot_pos=feet.copy(),
        spawn_root_pos=np.array([0.0, 0.0, 0.5]),
        spawn_root_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        spawn_sole_tilt_deg=np.zeros(n),
        home_sole_tilt_deg=np.zeros(n),
        truncated=True,
    )


#: Biped labels are (r, l): the right foot's phase first.
def biped_walk(**kwargs: Any) -> EpisodeTrace:
    return gait_trace(cyclic((0.5, 0.0), 0.6), foot_x=swing((0.5, 0.0)), **kwargs)


def biped_hop(stagger: float = 0.0) -> EpisodeTrace:
    """Both feet down together for 0.6 of each cycle, side by side; *stagger* delays the left foot's cycle."""
    return gait_trace(cyclic((0.0, stagger), 0.6), foot_x=swing((0.0, 0.0)))


def biped_slide() -> EpisodeTrace:
    return gait_trace(np.ones((STEPS, 2), dtype=bool))


#: Quadruped labels are (fr, fl, rr, rl); phases in that order.
LATERAL_WALK = (0.75, 0.25, 0.5, 0.0)
TROT = (0.0, 0.5, 0.5, 0.0)
PACE = (0.5, 0.0, 0.5, 0.0)
PRONK = (0.0, 0.0, 0.0, 0.0)
BOUND = (0.5, 0.5, 0.0, 0.0)


def quadruped(phases: tuple[float, ...], stance: float = 0.6, **kwargs: Any) -> EpisodeTrace:
    return gait_trace(cyclic(phases, stance), foot_x=swing(phases), speed=0.3, **kwargs)


def metrics(trace: EpisodeTrace) -> dict[str, float]:
    return episode_locomotion_metrics(trace).as_gait_metrics()


def failing_keys(gait: dict[str, float], clause: GaitClause) -> set[str]:
    episode = cruise(gait)
    return {reason_key(reason) for reason in gait_failures(episode, clause)}


# ── episode records ──────────────────────────────────────────────────────────

WALK_BLOCK = {**PROVISIONAL_WALK_THRESHOLDS, **PROVISIONAL_BIPED_GAIT_THRESHOLDS}
WALK = thresholds_from_curriculum(WALK_BLOCK)
assert isinstance(WALK, WalkThresholds)
QUADRUPED_WALK = thresholds_from_curriculum({**PROVISIONAL_WALK_THRESHOLDS, **PROVISIONAL_QUADRUPED_GAIT_THRESHOLDS})
assert isinstance(QUADRUPED_WALK, WalkThresholds)
MEASUREMENT = panel_measurement(WALK_CONFIG, WALK)


def cruise(gait: dict[str, float] | None, seed: int = 3042, **overrides: Any) -> CommandTrackingEpisode:
    """A cruise episode that clears every non-gait walk bar, with *gait* as its gait metrics."""
    values: dict[str, Any] = dict(
        episode=seed - 3042,
        seed=seed,
        family=CRUISE_FAMILY,
        length=1000,
        horizon=1000,
        dt=0.01,
        full_horizon=True,
        fall=False,
        termination_reason=None,
        reward=120.0,
        desired_speed=0.4,
        planned_event_count=1,
        initial_heading_error_rad=0.12,
        first_event_settled=True,
        tracking_fraction=0.9,
        tracking_fraction_after_settle=0.95,
        mean_speed_along_heading_mps=0.38,
        mean_desired_speed_mps=0.4,
        stand_hold_fraction=0.93,
        eligible_events=1,
        settled_events=1,
        eligible_stops=0,
        settled_stops=0,
        eligible_turns=0,
        settled_turns=0,
        max_course_progress_m=3.6,
        longest_stop_s=0.0,
        measurement=MEASUREMENT,
        gait_metrics=tuple((gait or {}).items()),
    )
    values.update(overrides)
    return CommandTrackingEpisode(**values)


def stand(gait: dict[str, float] | None, seed: int = 3100) -> CommandTrackingEpisode:
    return cruise(
        gait,
        seed,
        family=STAND_FAMILY,
        desired_speed=0.0,
        initial_heading_error_rad=0.0,
        mean_speed_along_heading_mps=math.nan,
        mean_desired_speed_mps=math.nan,
        stand_hold_fraction=1.0,
        eligible_stops=1,
        settled_stops=1,
    )


def walk_panel(
    cruise_gait: dict[str, float], stand_gait: dict[str, float] | None = None
) -> list[CommandTrackingEpisode]:
    standing = stand_gait if stand_gait is not None else cruise_gait
    return [cruise(cruise_gait, 3042 + index) for index in range(20)] + [
        stand(standing, 3100 + index) for index in range(5)
    ]


# ── the events ───────────────────────────────────────────────────────────────


class TestPairTouchdownEvents:
    def test_touchdowns_within_the_window_merge_into_one_both_event(self):
        assert events.pair_touchdown_events(np.array([10, 50]), np.array([12, 80]), 2) == [
            (10, "B"),
            (50, "L"),
            (80, "R"),
        ]
        # Three steps apart is two events at a 2-step window, one at a 3-step window.
        assert events.pair_touchdown_events(np.array([10]), np.array([13]), 2) == [(10, "L"), (13, "R")]
        assert events.pair_touchdown_events(np.array([10]), np.array([13]), 3) == [(10, "B")]

    def test_one_limbs_repeated_touchdowns_never_merge(self):
        assert events.pair_touchdown_events(np.array([10, 11]), np.array([], dtype=int), 2) == [(10, "L"), (11, "L")]

    def test_a_same_step_pair_is_one_event_and_each_touchdown_merges_once(self):
        assert events.pair_touchdown_events(np.array([5, 6]), np.array([5]), 2) == [(5, "B"), (6, "L")]
        assert events.pair_touchdown_events(np.array([], dtype=int), np.array([], dtype=int), 2) == []


# ── the metrics ──────────────────────────────────────────────────────────────


class TestBipedMetrics:
    def test_a_walk(self):
        """Alternating footfalls, double support, the leading foot swapping twice a stride, no flight."""
        walk = metrics(biped_walk())
        assert walk["limbs"] == 2.0 and walk["window_start_step"] == 100 and walk["window_steps"] == 500
        assert walk["alternation_index"] == walk["pair_alternation_index"] == 1.0
        assert walk["simultaneous_fraction"] == 0.0 and walk["touchdown_events"] == 10.0
        assert walk["flight_fraction"] == walk["hop_flight_fraction"] == 0.0
        assert walk["min_limb_duty"] == pytest.approx(0.6)
        assert walk["all_limb_support_fraction"] == pytest.approx(0.2, abs=0.005)
        assert walk["stride_frequency_hz"] == pytest.approx(1.0) and walk["strides"] == pytest.approx(5.0)
        # Two swaps a stride: the window's 5 strides hold 9, since it opens on a crossing (the lead is named only once
        # the offset leaves the dead band, so the opening crossing is no swap).
        assert walk["lead_swaps_per_stride"] == pytest.approx(9 / 5)
        assert walk["foot_contact_fraction"] == walk["phantom_support_fraction"] == 0.0
        assert walk["window_displacement_m"] == pytest.approx(2.5) and walk["displacement_speed_mps"] == pytest.approx(
            0.5
        )
        assert math.isnan(walk["ipsilateral_phase"]) and math.isnan(walk["fore_alternation_index"])
        assert failing_keys(walk, BIPED) == set()

    def test_an_alternating_run_flies_but_not_as_a_hop(self):
        """Each foot down 0.4 of a cycle: 20% flight, every interval opened by one foot and closed by the other."""
        run = metrics(gait_trace(cyclic((0.5, 0.0), 0.4), foot_x=swing((0.5, 0.0)), speed=1.5))
        assert run["flight_fraction"] == pytest.approx(0.2, abs=0.01)
        assert run["hop_flight_fraction"] == 0.0
        assert run["alternation_index"] == 1.0 and run["simultaneous_fraction"] == 0.0
        # The biped clause leaves the flight cap off (a run may fly); a stage below Froude 0.5 declares it.
        assert failing_keys(run, BIPED) == set()
        capped = gait_clause_from_curriculum(
            {**PROVISIONAL_BIPED_GAIT_THRESHOLDS, "max_flight_fraction": PROVISIONAL_BIPED_FLIGHT_FRACTION}
        )
        assert failing_keys(run, capped) == {"max_flight_fraction"}

    @pytest.mark.parametrize("stagger", [0.0, 0.01, 0.02])
    def test_a_two_footed_hop(self, stagger):
        """Both feet land within the 20 ms merge: one B event a hop, and every flight is hop flight."""
        hop = metrics(biped_hop(stagger))
        assert hop["simultaneous_fraction"] == 1.0 and hop["alternation_index"] == 0.0
        assert hop["flight_fraction"] > 0.35 and hop["hop_flight_fraction"] > 0.3
        assert hop["lead_swaps_per_stride"] == 0.0
        assert failing_keys(hop, BIPED) == {
            "min_alternation_index",
            "max_simultaneous_fraction",
            "max_hop_flight_fraction",
            "min_lead_swaps_per_stride",
        }

    @pytest.mark.parametrize("wobble", [0.0, 0.01], ids=["side_by_side", "wobble_inside_the_band"])
    def test_a_hop_staggered_past_the_merge_window_is_refused_by_its_lead_swaps(self, wobble):
        """30 ms apart the footfalls read as alternating and each flight as one foot leaving and the other landing
        (the plan's 20 ms definitions).  The feet land side by side, their fore-aft offset never leaving the dead
        band (0.03 leg lengths, 1.5 cm here), so the leading foot never swaps, even when it wobbles about zero; a
        plain sign change (locomotion-gait/v1) counted every crossing of that wobble."""
        assert LEAD_DEAD_BAND_L == 0.03
        steps = np.arange(STEPS)
        offset = wobble * np.sin(2.0 * np.pi * 3.0 * steps * DT)
        hop = metrics(gait_trace(cyclic((0.0, 0.03), 0.6), foot_x=np.column_stack([np.zeros(STEPS), offset])))
        assert hop["leg_length_m"] == 0.5 and hop["lead_amplitude_m"] == pytest.approx(wobble, abs=1e-3)
        assert hop["alternation_index"] == 1.0 and hop["simultaneous_fraction"] == 0.0
        assert hop["hop_flight_fraction"] == 0.0 and hop["flight_fraction"] > 0.35
        assert hop["lead_swaps_per_stride"] == 0.0
        assert failing_keys(hop, BIPED) == {"min_lead_swaps_per_stride"}
        # Report-only: the right foot lands 30 ms into each left stride's end, a phase near 0.97 (a walk's is 0.5).
        assert hop["left_right_phase"] == pytest.approx(0.97, abs=0.01)

    def test_a_staggered_hop_whose_feet_swap_past_the_band_needs_the_flight_cap(self):
        """The clause's known limit: a staggered hop whose feet swing fore-aft past the dead band (5 cm, 0.1 leg
        lengths, three times a second) swaps its lead often enough and clears every required biped separator.  Only
        the optional flight cap refuses it; the report-only left/right phase records it far from a walk's 0.5."""
        steps = np.arange(STEPS)
        offset = 0.05 * np.sin(2.0 * np.pi * 3.0 * steps * DT)
        hop = metrics(gait_trace(cyclic((0.0, 0.03), 0.6), foot_x=np.column_stack([np.zeros(STEPS), offset])))
        assert hop["lead_swaps_per_stride"] > 1.0 and hop["flight_fraction"] > 0.35
        assert failing_keys(hop, BIPED) == set()
        capped = gait_clause_from_curriculum(
            {**PROVISIONAL_BIPED_GAIT_THRESHOLDS, "max_flight_fraction": PROVISIONAL_BIPED_FLIGHT_FRACTION}
        )
        assert failing_keys(hop, capped) == {"max_flight_fraction"}
        assert abs(hop["left_right_phase"] - 0.5) > 0.4

    def test_the_lead_dead_band_is_in_leg_lengths(self):
        """The offset must pass 0.03 leg lengths either side of zero to name a lead; the leg length is the spawn root's
        height above the spawn foot sites, and without a positive one the lead swaps are unmeasured."""
        walk = biped_walk()
        assert metrics(walk)["leg_length_m"] == 0.5 and metrics(walk)["lead_swaps_per_stride"] == pytest.approx(9 / 5)
        down = cyclic((0.5, 0.0), 0.6)
        inside = metrics(gait_trace(down, foot_x=swing((0.5, 0.0), amplitude=0.007)))
        outside = metrics(gait_trace(down, foot_x=swing((0.5, 0.0), amplitude=0.009)))
        # The offset's amplitude is twice a foot's: 1.4 cm stays inside the 1.5 cm band, 1.8 cm leaves it.
        assert inside["lead_swaps_per_stride"] == 0.0
        assert outside["lead_swaps_per_stride"] == pytest.approx(9 / 5)
        grounded = replace(walk, spawn_root_pos=np.array([0.0, 0.0, 0.0]))
        assert math.isnan(metrics(grounded)["leg_length_m"])
        assert math.isnan(metrics(grounded)["lead_swaps_per_stride"])
        assert failing_keys(metrics(grounded), BIPED) == {"min_lead_swaps_per_stride"}

    def test_the_left_right_phase_and_the_stance_slip(self):
        """Report-only: a walk's right foot lands half a left stride in, a hop's with the left; a planted foot does not
        slide, a skid's slides at the body's speed."""
        assert metrics(biped_walk())["left_right_phase"] == pytest.approx(0.5, abs=0.01)
        assert metrics(biped_hop(0.0))["left_right_phase"] == pytest.approx(0.0, abs=0.01)
        standing = gait_trace(np.ones((STEPS, 2), dtype=bool), speed=0.0)
        assert metrics(standing)["stance_slip_speed_mps"] == 0.0
        assert metrics(biped_slide())["stance_slip_speed_mps"] == pytest.approx(0.5)
        quadruped_walk = metrics(quadruped(LATERAL_WALK, 0.75))
        assert math.isnan(quadruped_walk["left_right_phase"]) and quadruped_walk["lead_amplitude_m"] > 0.0

    def test_a_slide(self):
        """Both feet down while the body travels: no touchdown, so the footfall metrics are unmeasured."""
        slide = metrics(biped_slide())
        assert slide["all_limb_support_fraction"] == 1.0 and slide["touchdown_events"] == 0.0
        assert slide["window_displacement_m"] == pytest.approx(2.5)
        for key in ("alternation_index", "simultaneous_fraction", "lead_swaps_per_stride", "strides"):
            assert math.isnan(slide[key]), key
        assert slide["flight_fraction"] == slide["hop_flight_fraction"] == 0.0
        assert failing_keys(slide, BIPED) == {
            "min_alternation_index",
            "max_simultaneous_fraction",
            "min_lead_swaps_per_stride",
        }

    def test_a_scoot_keeps_one_foot_ahead(self):
        """Alternating footfalls with the left foot always 10 cm ahead: no lead swap."""
        offsets = np.column_stack([np.zeros(STEPS), np.full(STEPS, 0.1)])
        scoot = metrics(gait_trace(cyclic((0.5, 0.0), 0.6), foot_x=offsets))
        assert scoot["alternation_index"] == 1.0 and scoot["lead_swaps_per_stride"] == 0.0
        assert failing_keys(scoot, BIPED) == {"min_lead_swaps_per_stride"}

    def test_a_limp_repeats_a_foot(self):
        """L L R R: half the consecutive footfalls switch foot, with no flight and no simultaneous landing."""
        cycles = np.arange(STEPS) * DT / PERIOD % 1.0
        left = ~(((cycles >= 0.10) & (cycles < 0.15)) | ((cycles >= 0.30) & (cycles < 0.35)))
        right = ~(((cycles >= 0.60) & (cycles < 0.65)) | ((cycles >= 0.80) & (cycles < 0.85)))
        cycles_x = np.arange(STEPS) * DT / PERIOD
        offsets = np.column_stack([-0.1 * np.sin(4 * np.pi * cycles_x), 0.1 * np.sin(4 * np.pi * cycles_x)])
        limp = metrics(gait_trace(np.column_stack([right, left]), foot_x=offsets))
        assert limp["alternation_index"] == pytest.approx(0.5, abs=0.06)
        assert limp["simultaneous_fraction"] == 0.0 and limp["flight_fraction"] == 0.0
        assert failing_keys(limp, BIPED) == {"min_alternation_index"}

    def test_stacked_feet_and_phantom_support(self):
        """A sole resting on the other foot reads as leg-on-leg contact, and its touch as support the floor never gives."""
        down = cyclic((0.5, 0.0), 0.6)
        stacked = metrics(biped_walk(interleg=np.full(STEPS, 0.5)))
        assert stacked["foot_contact_fraction"] == pytest.approx(0.5)
        assert failing_keys(stacked, BIPED) == {"max_foot_contact_fraction"}
        touch = np.where(down, BODY_WEIGHT / 2, 0.0)
        touch[:, 0] = np.where(~down[:, 0], 1.0, touch[:, 0])  # the right swing foot "touches" on every swing step
        phantom = metrics(biped_walk(touch=touch))
        assert phantom["phantom_support_fraction"] == pytest.approx(0.4, abs=0.01)
        assert failing_keys(phantom, BIPED) == {"max_phantom_support_fraction"}

    def test_the_first_second_is_not_judged(self):
        """A hop that ends before GAIT_WINDOW_S and a walk after it read as the walk (a landing ON the window's first
        step would be judged: touchdowns are counted from that step)."""
        assert GAIT_WINDOW_S == 1.0
        down = cyclic((0.5, 0.0), 0.6)
        down[:80] = cyclic((0.0, 0.0), 0.6, steps=80)
        mixed = metrics(gait_trace(down, foot_x=swing((0.5, 0.0))))
        assert mixed["simultaneous_fraction"] == 0.0 and mixed["alternation_index"] == 1.0
        assert failing_keys(mixed, BIPED) == set()
        hopping = metrics(gait_trace(cyclic((0.0, 0.0), 0.6), foot_x=swing((0.0, 0.0))))
        assert hopping["simultaneous_fraction"] == 1.0

    def test_windows_are_in_seconds(self):
        """At dt 0.02 (both compsognathus) the window starts at step 50 and the merge is one step."""
        down = cyclic((0.5, 0.0), 0.6, steps=300, dt=0.02)
        walk = metrics(gait_trace(down, dt=0.02, foot_x=swing((0.5, 0.0), steps=300, dt=0.02)))
        assert walk["window_start_step"] == 50 and walk["window_steps"] == 250
        assert walk["alternation_index"] == 1.0 and walk["lead_swaps_per_stride"] == pytest.approx(9 / 5)
        # One step (20 ms) apart merges; two steps (40 ms) do not.
        near = metrics(gait_trace(cyclic((0.0, 0.02), 0.6, steps=300, dt=0.02), dt=0.02))
        far = metrics(gait_trace(cyclic((0.0, 0.04), 0.6, steps=300, dt=0.02), dt=0.02))
        assert near["simultaneous_fraction"] == 1.0 and far["simultaneous_fraction"] == 0.0

    def test_an_episode_shorter_than_the_window_is_unmeasured(self):
        short = metrics(gait_trace(cyclic((0.5, 0.0), 0.6, steps=80)))
        assert short["window_steps"] == 0.0 and short["limbs"] == 2.0
        gait_values = {key: value for key, value in short.items() if key not in ("limbs", "window_start_step")}
        assert all(math.isnan(value) for key, value in gait_values.items() if key != "window_steps")
        assert failing_keys(short, BIPED) == set(BIPED.declared()) - {GAIT_LIMBS_KEY}

    def test_limbs_must_pair_by_label(self):
        with pytest.raises(ValueError, match="neither a biped's"):
            episode_locomotion_metrics(gait_trace(np.ones((STEPS, 2), dtype=bool), labels=("a", "b")))
        with pytest.raises(ValueError, match="no steps"):
            episode_locomotion_metrics(gait_trace(np.ones((0, 2), dtype=bool)))
        with pytest.raises(ValueError, match="window_start_s"):
            episode_locomotion_metrics(biped_walk(), window_start_s=-1.0)


class TestQuadrupedMetrics:
    @pytest.mark.parametrize(
        "phases, stance, ipsilateral",
        [(LATERAL_WALK, 0.75, 0.25), (TROT, 0.6, 0.5), (PACE, 0.6, 0.0)],
        ids=["lateral_walk", "trot", "pace"],
    )
    def test_a_walk_a_trot_and_a_pace_meet_the_floors(self, phases, stance, ipsilateral):
        """GQ-9 (a): each pair alternates, no flight, every limb carries; the Hildebrand phase tells them apart."""
        gait = metrics(quadruped(phases, stance))
        assert gait["limbs"] == 4.0 and math.isnan(gait["alternation_index"])
        assert gait["fore_alternation_index"] == gait["hind_alternation_index"] == gait["pair_alternation_index"] == 1.0
        assert gait["simultaneous_fraction"] == 0.0 and gait["flight_fraction"] == 0.0
        assert gait["min_limb_duty"] == pytest.approx(stance)
        assert (
            min(abs(gait["ipsilateral_phase"] - ipsilateral), 1 - abs(gait["ipsilateral_phase"] - ipsilateral)) < 0.02
        )
        assert failing_keys(gait, QUADRUPED) == set()

    def test_a_pronk_is_a_four_footed_hop(self):
        pronk = metrics(quadruped(PRONK))
        assert pronk["pair_alternation_index"] == 0.0 and pronk["simultaneous_fraction"] == 1.0
        assert pronk["flight_fraction"] == pytest.approx(0.4) and pronk["hop_flight_fraction"] > 0.3
        assert failing_keys(pronk, QUADRUPED) == {"min_pair_alternation_index", "max_flight_fraction"}

    def test_a_bound_lands_each_pair_together(self):
        """Fore pair together, hind pair together half a cycle later: always a pair down, never a pair alternating."""
        bound = metrics(quadruped(BOUND))
        assert bound["flight_fraction"] == 0.0 and bound["min_limb_duty"] == pytest.approx(0.6)
        assert bound["fore_simultaneous_fraction"] == bound["hind_simultaneous_fraction"] == 1.0
        assert failing_keys(bound, QUADRUPED) == {"min_pair_alternation_index"}

    def test_a_flightless_bound_whose_pairs_land_apart_clears_the_required_floors(self):
        """Known limit, pinned: each pair's feet land 40 ms apart (past the 20 ms merge) with no flight, the feet of a
        pair side by side.  It reads as alternating pairs, so it clears every required GQ-9 (a) floor; the Hildebrand
        phase reads a trot's 0.5.  The optional lead-swap bar refuses it, but only because a pair side by side never
        swaps: the next test swings the same pairs' feet and clears that bar too."""
        staggered = (0.5, 0.54, 0.0, 0.04)
        gait = metrics(gait_trace(cyclic(staggered, 0.6), foot_x=swing(BOUND), speed=0.3))
        assert gait["flight_fraction"] == 0.0 and gait["min_limb_duty"] == pytest.approx(0.6)
        assert gait["pair_alternation_index"] == 1.0 and gait["simultaneous_fraction"] == 0.0
        assert gait["lead_swaps_per_stride"] == 0.0
        assert abs(gait["ipsilateral_phase"] - 0.5) < 0.02
        assert failing_keys(gait, QUADRUPED) == set()
        with_lead_swaps = gait_clause_from_curriculum(
            {**PROVISIONAL_QUADRUPED_GAIT_THRESHOLDS, "min_lead_swaps_per_stride": 1.0}
        )
        assert failing_keys(gait, with_lead_swaps) == {"min_lead_swaps_per_stride"}
        # With flight the same staggered bound fails the required flight cap.
        flying = metrics(gait_trace(cyclic(staggered, 0.4), foot_x=swing(BOUND), speed=0.3))
        assert "max_flight_fraction" in failing_keys(flying, QUADRUPED)

    def test_a_flightless_bound_whose_pair_feet_swing_their_lead_clears_every_quadruped_bar(self):
        """Known limit, pinned (decision D-D62 (16) (f)): the same staggered bound with each foot swinging on its own
        touchdown phase.  A pair's feet 40 ms apart swing their fore-aft offset with a lead amplitude of 0.049 m, past
        the dead band's 0.015 m on this 0.5 m leg, so the lead swaps twice a stride and the optional lead-swap bar is
        cleared as well: the bar narrows the flightless-bound gap and does not close it."""
        staggered = (0.5, 0.54, 0.0, 0.04)
        gait = metrics(gait_trace(cyclic(staggered, 0.6), foot_x=swing(staggered), speed=0.3))
        assert gait["flight_fraction"] == 0.0 and gait["min_limb_duty"] == pytest.approx(0.6)
        assert gait["pair_alternation_index"] == 1.0 and gait["simultaneous_fraction"] == 0.0
        assert gait["lead_amplitude_m"] > LEAD_DEAD_BAND_L * gait["leg_length_m"]
        assert gait["lead_swaps_per_stride"] == pytest.approx(2.0)
        assert gait["fore_lead_swaps_per_stride"] == pytest.approx(2.0)
        assert gait["hind_lead_swaps_per_stride"] == pytest.approx(2.0)
        assert abs(gait["ipsilateral_phase"] - 0.5) < 0.02
        assert failing_keys(gait, QUADRUPED) == set()
        with_lead_swaps = gait_clause_from_curriculum(
            {**PROVISIONAL_QUADRUPED_GAIT_THRESHOLDS, "min_lead_swaps_per_stride": 1.0}
        )
        assert failing_keys(gait, with_lead_swaps) == set()

    def test_a_slide(self):
        slide = metrics(gait_trace(np.ones((STEPS, 4), dtype=bool), speed=0.3))
        assert slide["all_limb_support_fraction"] == 1.0 and slide["touchdown_events"] == 0.0
        assert math.isnan(slide["pair_alternation_index"]) and slide["window_displacement_m"] == pytest.approx(1.5)
        assert failing_keys(slide, QUADRUPED) == {"min_pair_alternation_index"}

    def test_a_three_legged_gait(self):
        """The right forefoot held up (the 2026-09 skid's lifted forefoot): no duty, and its pair cannot alternate."""
        down = cyclic(LATERAL_WALK, 0.75)
        down[:, 0] = False
        three = metrics(gait_trace(down, foot_x=swing(LATERAL_WALK), speed=0.3))
        assert three["min_limb_duty"] == 0.0 and three["fore_alternation_index"] == 0.0
        assert failing_keys(three, QUADRUPED) == {"min_limb_duty", "min_pair_alternation_index"}

    def test_every_metric_is_a_named_float(self):
        values = episode_locomotion_metrics(quadruped(TROT))
        assert isinstance(values, LocomotionEpisodeMetrics)
        assert tuple(values.as_gait_metrics()) == LOCOMOTION_METRIC_FIELDS
        assert all(isinstance(value, float) for value in values.as_gait_metrics().values())


# ── the clause ───────────────────────────────────────────────────────────────


class TestClause:
    def test_the_criteria_read_the_locomotion_metrics(self):
        assert {metric for _, metric, _ in GAIT_CRITERIA} <= set(LOCOMOTION_METRIC_FIELDS)
        assert GAIT_THRESHOLD_KEYS == GAIT_CRITERION_KEYS | {GAIT_LIMBS_KEY}
        assert GAIT_THRESHOLD_KEYS <= COMMAND_TRACKING_THRESHOLD_KEYS
        assert GAIT_REQUIRED_KEYS[BIPED_LIMBS] == {
            "min_alternation_index",
            "max_simultaneous_fraction",
            "max_hop_flight_fraction",
            "min_lead_swaps_per_stride",
            "max_foot_contact_fraction",
            "max_phantom_support_fraction",
        }
        assert GAIT_REQUIRED_KEYS[QUADRUPED_LIMBS] == {
            "max_flight_fraction",
            "min_limb_duty",
            "min_pair_alternation_index",
        }
        assert GAIT_ALLOWED_KEYS[BIPED_LIMBS] == GAIT_CRITERION_KEYS - {"min_pair_alternation_index"}
        assert GAIT_ALLOWED_KEYS[QUADRUPED_LIMBS] == GAIT_CRITERION_KEYS - {"min_alternation_index"}

    def test_the_provisional_values_are_the_plans(self):
        """GAIT_QUALITY_PLAN_2026_09.md §4.3 A (bipeds, GQ-8 (a)) and B (quadrupeds, GQ-9 (a))."""
        assert PROVISIONAL_BIPED_GAIT_THRESHOLDS == {
            "gait_limbs": 2,
            "min_alternation_index": 0.75,
            "max_simultaneous_fraction": 0.15,
            "max_hop_flight_fraction": 0.15,
            "min_lead_swaps_per_stride": 1.0,
            "max_foot_contact_fraction": 0.02,
            "max_phantom_support_fraction": 0.05,
        }
        assert PROVISIONAL_BIPED_FLIGHT_FRACTION == 0.05
        assert PROVISIONAL_QUADRUPED_GAIT_THRESHOLDS == {
            "gait_limbs": 4,
            "max_flight_fraction": 0.05,
            "min_limb_duty": 0.20,
            "min_pair_alternation_index": 0.75,
        }
        plan = (REPOSITORY_ROOT / "docs" / "GAIT_QUALITY_PLAN_2026_09.md").read_text(encoding="utf-8")
        for header in (
            "Hop-flight ≤ 0.15",
            "Alternation ≥ 0.75",
            "Simultaneous ≤ 0.15",
            "Lead swaps ≥ 1.0",
            "Foot-on-foot ≤ 0.02; phantom ≤ 0.05",
            "Flight ≤ 0.05",
            "Min limb duty ≥ 0.20",
            "Fore alternation ≥ 0.75",
            "Hind alternation ≥ 0.75",
        ):
            assert header in plan, header
        source = Path(__file__).resolve().parents[1] / "curriculum" / "command_tracking_gate.py"
        text = source.read_text(encoding="utf-8")
        assert "PROVISIONAL starting values of a biped's gait clause" in text
        assert "PROVISIONAL starting values of a quadruped's gait clause" in text
        assert text.count("§4.3 A:") >= 6 and text.count("§4.3 B:") >= 3

    def test_the_clause_is_required_in_both_profiles(self):
        for profile in (PROVISIONAL_WALK_THRESHOLDS, PROVISIONAL_FOLLOW_THRESHOLDS):
            with pytest.raises(ValueError, match=r"missing required threshold\(s\) \['gait_limbs'\]"):
                thresholds_from_curriculum(profile)
        follow = thresholds_from_curriculum({**PROVISIONAL_FOLLOW_THRESHOLDS, **PROVISIONAL_QUADRUPED_GAIT_THRESHOLDS})
        assert isinstance(follow, FollowThresholds) and follow.gait == QUADRUPED
        assert WALK.gait == BIPED and WALK.declared() == WALK_BLOCK
        assert thresholds_from_curriculum(WALK.declared()) == WALK

    @pytest.mark.parametrize(
        "limbs, key",
        [(BIPED_LIMBS, key) for key in sorted(GAIT_REQUIRED_KEYS[BIPED_LIMBS])]
        + [(QUADRUPED_LIMBS, key) for key in sorted(GAIT_REQUIRED_KEYS[QUADRUPED_LIMBS])],
    )
    def test_every_required_bar_is_required(self, limbs, key):
        block = PROVISIONAL_BIPED_GAIT_THRESHOLDS if limbs == BIPED_LIMBS else PROVISIONAL_QUADRUPED_GAIT_THRESHOLDS
        with pytest.raises(ValueError, match=rf"for {limbs} limbs .* missing required threshold\(s\) \['{key}'\]"):
            gait_clause_from_curriculum({k: v for k, v in block.items() if k != key})

    def test_each_animal_refuses_the_others_alternation_key(self):
        with pytest.raises(
            ValueError, match=r"for 2 limbs \(a biped\) does not judge \['min_pair_alternation_index'\]"
        ):
            gait_clause_from_curriculum({**PROVISIONAL_BIPED_GAIT_THRESHOLDS, "min_pair_alternation_index": 0.75})
        with pytest.raises(ValueError, match=r"for 4 limbs \(a quadruped\) does not judge \['min_alternation_index'\]"):
            gait_clause_from_curriculum({**PROVISIONAL_QUADRUPED_GAIT_THRESHOLDS, "min_alternation_index": 0.75})

    def test_the_optional_bars_are_accepted_and_kept_in_order(self):
        clause = gait_clause_from_curriculum(
            {
                **PROVISIONAL_QUADRUPED_GAIT_THRESHOLDS,
                "max_simultaneous_fraction": 0.2,
                "min_lead_swaps_per_stride": 1.0,
            }
        )
        assert [key for key, _ in clause.bars] == [key for key, _, _ in GAIT_CRITERIA if key in clause.declared()]
        assert clause.bar("max_simultaneous_fraction") == 0.2 and clause.bar("max_hop_flight_fraction") is None

    @pytest.mark.parametrize(
        "key, value, match",
        [
            ("gait_limbs", 3, "must be 2 .* or 4"),
            ("gait_limbs", 2.5, "must be 2 .* or 4"),
            ("gait_limbs", "2", "not a number"),
            ("min_alternation_index", 0.0, r"in \(0, 1\]"),
            ("min_alternation_index", 1.1, r"in \(0, 1\]"),
            ("max_simultaneous_fraction", 1.0, r"in \[0, 1\)"),
            ("max_hop_flight_fraction", -0.01, r"in \[0, 1\)"),
            ("max_foot_contact_fraction", math.nan, "must be finite"),
            ("min_lead_swaps_per_stride", 0.0, "must be positive"),
            ("max_phantom_support_fraction", True, "not a number"),
        ],
    )
    def test_values_outside_their_rule_are_refused(self, key, value, match):
        with pytest.raises(ValueError, match=match):
            gait_clause_from_curriculum({**PROVISIONAL_BIPED_GAIT_THRESHOLDS, key: value})

    def test_a_zero_cap_and_a_float_limb_count_are_accepted(self):
        clause = gait_clause_from_curriculum(
            {**PROVISIONAL_BIPED_GAIT_THRESHOLDS, "gait_limbs": 2.0, "max_foot_contact_fraction": 0.0}
        )
        assert clause.limbs == 2 and isinstance(clause.limbs, int) and clause.bar("max_foot_contact_fraction") == 0.0

    def test_config_load_refuses_a_bad_clause(self):
        good = {
            "gate_schema_version": GATE_SCHEMA_VERSION,
            "gate_kind": COMMAND_TRACKING_GATE_KIND,
            **WALK_BLOCK,
        }
        assert validate_gate_config("locomotion", good) == COMMAND_TRACKING_GATE_KIND
        for edit, match in (
            ({"gait_limbs": None}, r"missing required threshold\(s\) \['gait_limbs'\]"),
            ({"min_lead_swaps_per_stride": None}, r"missing required threshold\(s\) \['min_lead_swaps_per_stride'\]"),
            ({"gait_limbs": 4}, r"for 4 limbs \(a quadruped\) does not judge \['min_alternation_index'\]"),
            ({"max_hop_flight_fraction": 1.5}, r"must lie in \[0, 1\)"),
        ):
            block = dict(good)
            for key, value in edit.items():
                if value is None:
                    block.pop(key)
                else:
                    block[key] = value
            with pytest.raises(GateSchemaError, match=match):
                validate_gate_config("locomotion", block)

    def test_the_gate_digest_covers_the_clause(self):
        block = {"gate_schema_version": GATE_SCHEMA_VERSION, "gate_kind": COMMAND_TRACKING_GATE_KIND, **WALK_BLOCK}
        view = gate_config_view(block)
        assert {
            key: view["thresholds"][key] for key in PROVISIONAL_BIPED_GAIT_THRESHOLDS
        } == PROVISIONAL_BIPED_GAIT_THRESHOLDS
        for edit in ({"min_alternation_index": 0.8}, {"max_flight_fraction": 0.05}):
            assert gate_config_sha256(gate_config_view({**block, **edit})) != gate_config_sha256(view)
        retyped = {**block, "gait_limbs": 2.0}
        assert gate_config_sha256(gate_config_view(retyped)) == gate_config_sha256(view)


# ── the clause in the verdict ────────────────────────────────────────────────

#: A biped walk's metrics and a quadruped walk's, measured on the synthetic fixtures.
BIPED_WALK = metrics(biped_walk())
QUADRUPED_WALK_GAIT = metrics(quadruped(LATERAL_WALK, 0.75))


def past(clause: GaitClause, key: str) -> float:
    """A value of *key*'s metric just past its bar in *clause*."""
    direction = {criterion: way for criterion, _, way in GAIT_CRITERIA}[key]
    bar = clause.bar(key)
    assert bar is not None
    return bar - 0.01 if direction == "min" else bar + 0.01


class TestVerdict:
    @pytest.mark.parametrize(
        "clause, gait", [(BIPED, BIPED_WALK), (QUADRUPED, QUADRUPED_WALK_GAIT)], ids=["biped", "quadruped"]
    )
    def test_each_bar_refuses_only_the_episodes_past_it(self, clause, gait):
        assert gait_failures(cruise(gait), clause) == []
        metric_of = {key: metric for key, metric, _ in GAIT_CRITERIA}
        for key, _ in clause.bars:
            moved = {**gait, metric_of[key]: past(clause, key)}
            reasons = gait_failures(cruise(moved), clause)
            assert [reason_key(reason) for reason in reasons] == [key], key
            unmeasured = gait_failures(cruise({**gait, metric_of[key]: math.nan}), clause)
            assert [reason_key(reason) for reason in unmeasured] == [key] and "unmeasured" in unmeasured[0]

    def test_a_bar_at_its_value_holds(self):
        exact = {**BIPED_WALK, "alternation_index": 0.75, "simultaneous_fraction": 0.15, "lead_swaps_per_stride": 1.0}
        assert gait_failures(cruise(exact), BIPED) == []

    def test_an_episode_without_gait_metrics_fails_every_bar(self):
        reasons = gait_failures(cruise(None), BIPED)
        assert {reason_key(reason) for reason in reasons} == set(BIPED.declared())
        assert "measured on nan limbs" in reasons[0]

    def test_an_episode_measured_on_another_animal_fails_and_is_refused(self):
        assert [reason_key(reason) for reason in gait_failures(cruise(QUADRUPED_WALK_GAIT), BIPED)][0] == GAIT_LIMBS_KEY
        result = evaluate_walk_profile(walk_panel(QUADRUPED_WALK_GAIT), WALK, measurement=MEASUREMENT)
        assert not result.passed
        assert any("another number of limbs" in failure and "gait_limbs = 2" in failure for failure in result.failures)
        # The stand family's records are checked too, though their gait is not judged.
        mixed = evaluate_walk_profile(walk_panel(BIPED_WALK, QUADRUPED_WALK_GAIT), WALK, measurement=MEASUREMENT)
        assert any("another number of limbs" in failure for failure in mixed.failures)

    def test_the_clause_judges_cruise_and_follow_episodes_and_not_stands(self):
        hop = metrics(biped_hop())
        cruise_reasons = {reason_key(reason) for reason in classify_cruise_episode(cruise(hop), WALK)}
        assert cruise_reasons == {
            "min_alternation_index",
            "max_simultaneous_fraction",
            "max_hop_flight_fraction",
            "min_lead_swaps_per_stride",
        }
        assert classify_stand_episode(stand(hop), WALK) == ()
        follow = thresholds_from_curriculum({**PROVISIONAL_FOLLOW_THRESHOLDS, **PROVISIONAL_BIPED_GAIT_THRESHOLDS})
        assert isinstance(follow, FollowThresholds)
        follow_episode = cruise(
            hop, horizon=2500, length=2500, tracking_fraction=0.9, eligible_events=1, settled_events=1
        )
        reasons = classify_follow_episode(follow_episode, follow, require_stop=False, require_turn=False)
        assert {reason_key(reason) for reason in reasons} == cruise_reasons

    def test_a_walk_panel_passes_and_a_hop_or_slide_panel_fails_on_gait(self):
        assert evaluate_walk_profile(walk_panel(BIPED_WALK), WALK, measurement=MEASUREMENT).passed
        for name, gait in (("hop", metrics(biped_hop())), ("slide", metrics(biped_slide()))):
            result = evaluate_walk_profile(walk_panel(gait), WALK, measurement=MEASUREMENT)
            assert not result.passed, name
            cruise_family = result.family(CRUISE_FAMILY)
            assert cruise_family.n_success == 0 and cruise_family.n_survived == 20, name
            assert dict(cruise_family.criterion_failures)["min_alternation_index"] == 20, name
            assert result.family(STAND_FAMILY).n_success == 5, name
            assert [failure.split(":")[0] for failure in result.failures] == ["cruise family"], name

    def test_a_quadruped_panel(self):
        assert evaluate_walk_profile(walk_panel(QUADRUPED_WALK_GAIT), QUADRUPED_WALK, measurement=MEASUREMENT).passed
        pronk = evaluate_walk_profile(walk_panel(metrics(quadruped(PRONK))), QUADRUPED_WALK, measurement=MEASUREMENT)
        assert not pronk.passed
        assert set(dict(pronk.family(CRUISE_FAMILY).criterion_failures)) == {
            "min_pair_alternation_index",
            "max_flight_fraction",
        }

    def test_a_follow_panel(self):
        follow = thresholds_from_curriculum({**PROVISIONAL_FOLLOW_THRESHOLDS, **PROVISIONAL_BIPED_GAIT_THRESHOLDS})
        assert isinstance(follow, FollowThresholds)

        def judge(gait: dict[str, float]):
            panel = [
                cruise(gait, 3042 + index, horizon=2500, length=2500, eligible_events=1, settled_events=1)
                for index in range(20)
            ]
            return evaluate_follow_profile(
                panel,
                follow,
                measurement=panel_measurement(WALK_CONFIG, follow),
                require_stop=False,
                require_turn=False,
            )

        assert judge(BIPED_WALK).passed
        sliding = judge(metrics(biped_slide()))
        assert not sliding.passed and dict(sliding.families[0].criterion_failures)["min_lead_swaps_per_stride"] == 20

    def test_another_gait_measurement_is_refused(self):
        assert MEASUREMENT.gait_measurement == GAIT_MEASUREMENT == f"{MEASUREMENT_VERSION}+{LOCOMOTION_METRICS_VERSION}"
        other = replace(MEASUREMENT, gait_measurement="floor-truth/v1+locomotion-gait/v1")
        panel = [replace(episode, measurement=other) for episode in walk_panel(BIPED_WALK)]
        result = evaluate_walk_profile(panel, WALK, measurement=MEASUREMENT)
        assert not result.passed and any("another measurement" in failure for failure in result.failures)
        assert MEASUREMENT.as_dict()["gait_measurement"] == GAIT_MEASUREMENT

    def test_the_summary_and_the_record_carry_the_gait(self):
        panel = walk_panel(BIPED_WALK) + [cruise(None, 3200)]
        summary = summarize_command_panel(panel)
        gait = summary["families"][CRUISE_FAMILY]["gait"]
        assert set(gait) == {metric for _, metric, _ in GAIT_CRITERIA}
        assert gait["alternation_index"] == {"mean": 1.0, "unmeasured": 1}
        assert summary["families"][STAND_FAMILY]["gait"]["alternation_index"]["unmeasured"] == 0
        record = cruise(metrics(biped_slide())).as_dict()["gait_metrics"]
        assert record["alternation_index"] is None and record["limbs"] == 2.0


# ── discrimination on the real plants ────────────────────────────────────────

TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")
BRACHIOSAURUS_ALTITHORAX = ModelId("brachiosaurus_altithorax", "anatomical")


def _roll(
    model_id: ModelId,
    seconds: float,
    *,
    controller: Callable[[Any], Callable[[Any], np.ndarray]] | None = None,
    hop_lift_bw: float = 0.0,
    push_bw: float = 0.0,
    friction_scale: float = 1.0,
) -> EpisodeTrace:
    """One episode of *model_id* through the floor-truth recorder, without noise, from seed 0.

    With *controller* (``make(env) -> action(env)``, built after reset) the
    plant is driven by it; otherwise it is the zero-action statue, which a
    force on the root's whole-body centre of mass can move: from 1 s on, an
    upward *hop_lift_bw* body weights for 0.12 s of every 0.6 s (a two-footed
    hop: both feet leave and land together), and from 0.5 s on a forward
    *push_bw* body weights, on a floor whose sliding friction is scaled by
    *friction_scale* (a slide).  The force is applied through ``qfrc_applied``
    at the subtree centre of mass, so it lifts or pushes without pitching
    the animal over its feet.
    """
    import mujoco

    from environments.shared.gait.morphology import Morphology
    from environments.shared.gait.recorder import SubstepContactRecorder
    from environments.shared.species_registry import env_class

    env = env_class(model_id)(reset_noise_scale=0.0)
    env.max_episode_steps = int(round(seconds / env.dt))
    env.model.geom_friction[:, 0] *= friction_scale
    morphology = Morphology.from_env(env, model_id)
    recorder = SubstepContactRecorder(env, morphology).attach()
    try:
        env.reset(seed=0)
        drive = controller(env) if controller is not None else None
        zero = np.zeros(env.action_space.shape[0])
        step = 0
        while True:
            time_s = step * env.dt
            force = np.zeros(3)
            if hop_lift_bw and time_s >= 1.0 and (time_s - 1.0) % 0.6 < 0.12:
                force[2] = hop_lift_bw * morphology.body_weight_n
            if push_bw and time_s >= 0.5:
                force[0] = push_bw * morphology.body_weight_n
            if drive is None:
                env.data.qfrc_applied[:] = 0.0
                if force.any():
                    root = morphology.root_body
                    point = env.data.subtree_com[root].copy()
                    mujoco.mj_applyFT(env.model, env.data, force, np.zeros(3), point, root, env.data.qfrc_applied)
            action = zero if drive is None else drive(env)
            _, reward, terminated, truncated, info = env.step(action)
            recorder.end_step(action, float(reward))
            step += 1
            if terminated or truncated:
                return recorder.end_episode(
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    termination_reason=str(info.get("termination_reason") or ""),
                )
    finally:
        recorder.detach()
        env.close()


def _reference_walk(module: str, *args: Any) -> Callable[[Any], Callable[[Any], np.ndarray]]:
    """A ``make(env) -> action(env)`` for the reference walk of *module* (``environments.<species>.scripts``)."""
    import importlib

    walk = importlib.import_module(f"environments.{module}.scripts.reference_walk").ReferenceWalk

    def make(env: Any) -> Callable[[Any], np.ndarray]:
        controller = walk(env, *args)
        return lambda live: controller.action(live.data, live.dt)

    return make


def _allosaurus_walk(env: Any) -> Callable[[Any], np.ndarray]:
    from environments.allosaurus_fragilis.scripts.reference_walk import ReferenceWalk

    controller = ReferenceWalk(env.model, env.data, walking=True)
    return lambda live: controller.action(live.data)


@pytest.fixture(scope="module")
def tyrannosaurus_rex_traces() -> dict[str, EpisodeTrace]:
    return {
        "walk": _roll(TYRANNOSAURUS_REX, 10.0, controller=_reference_walk("tyrannosaurus_rex")),
        "hop": _roll(TYRANNOSAURUS_REX, 6.0, hop_lift_bw=1.4),
        "slide": _roll(TYRANNOSAURUS_REX, 6.0, push_bw=0.03, friction_scale=0.01),
    }


@pytest.fixture(scope="module")
def brachiosaurus_altithorax_traces() -> dict[str, EpisodeTrace]:
    from environments.brachiosaurus_altithorax.scripts.reference_walk import PERIOD as STRIDE_S
    from environments.brachiosaurus_altithorax.scripts.reference_walk import START

    return {
        "walk": _roll(
            BRACHIOSAURUS_ALTITHORAX, START + 9 * STRIDE_S, controller=_reference_walk("brachiosaurus_altithorax")
        ),
        "hop": _roll(BRACHIOSAURUS_ALTITHORAX, 6.0, hop_lift_bw=1.4),
        "slide": _roll(BRACHIOSAURUS_ALTITHORAX, 6.0, push_bw=0.03, friction_scale=0.01),
    }


def _left_foot_delayed(trace: EpisodeTrace, delay_s: float) -> EpisodeTrace:
    """*trace* with its left foot's floor-truth down state delayed by *delay_s*: the hop's feet land and leave that far
    apart, the right foot first, while every position stays the recorded one (the feet side by side)."""
    left = trace.labels.index("l")
    shift = int(round(delay_s / trace.dt))
    down = np.asarray(trace.leg_down_frac).copy()
    recorded = down[:, left].copy()
    down[shift:, left] = recorded[:-shift]
    down[:shift, left] = recorded[0]
    return replace(trace, leg_down_frac=down)


def _sign_change_lead_swaps(trace: EpisodeTrace) -> int:
    """locomotion-gait/v1's lead swaps over the window: sign changes of the debounced fore-aft offset, no dead band."""
    start = int(round(GAIT_WINDOW_S / trace.dt))
    w, x, y, z = (np.asarray(trace.root_quat[start:])[:, index] for index in range(4))
    heading = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    forward = np.stack([np.cos(heading), np.sin(heading)], axis=1)
    relative = np.asarray(trace.foot_pos[start:, :, :2]) - np.asarray(trace.root_pos[start:, None, :2])
    along = np.einsum("kfi,ki->kf", relative, forward)
    offset = along[:, trace.labels.index("l")] - along[:, trace.labels.index("r")]
    leading = events.debounce(offset > 0.0, events.steps_for(DEBOUNCE_S, trace.dt))
    return int(np.count_nonzero(np.diff(leading.astype(np.int8))))


class TestRealPlants:
    def test_the_tyrannosaurus_rex_reference_walk_passes(self, tyrannosaurus_rex_traces):
        trace = tyrannosaurus_rex_traces["walk"]
        walk = metrics(trace)
        assert not trace.terminated and walk["limbs"] == 2.0
        assert walk["alternation_index"] == 1.0 and walk["simultaneous_fraction"] == 0.0
        assert walk["lead_swaps_per_stride"] > 1.5 and walk["flight_fraction"] == 0.0
        # The lead swings far past the dead band; report-only: the feet alternate half a stride apart, and a stance
        # foot stays planted.
        assert walk["lead_amplitude_m"] > 10 * LEAD_DEAD_BAND_L * walk["leg_length_m"]
        assert walk["left_right_phase"] == pytest.approx(0.5, abs=0.05)
        assert walk["stance_slip_speed_mps"] < 0.1
        assert gait_failures(cruise(walk), BIPED) == []
        assert classify_cruise_episode(cruise(walk), WALK) == ()

    def test_a_tyrannosaurus_rex_hop_staggered_past_the_merge_window_fails_on_its_lead_swaps(
        self, tyrannosaurus_rex_traces
    ):
        """The scripted hop with its feet landing and leaving 30 ms apart, the same foot first each time: the footfalls
        read as alternating (alternation 1, no simultaneous touchdown, almost no hop flight), so of the required
        separators only the lead swaps refuse it.  The feet stay side by side, their offset inside the dead band, so
        no lead swaps; without the band (locomotion-gait/v1) the offset's flicker about zero read as more than one
        swap a stride and this hop passed every required biped separator."""
        trace = _left_foot_delayed(tyrannosaurus_rex_traces["hop"], 0.03)
        hop = metrics(trace)
        assert hop["alternation_index"] == 1.0 and hop["simultaneous_fraction"] == 0.0
        assert hop["hop_flight_fraction"] < 0.02 and hop["flight_fraction"] > 0.25
        assert hop["lead_amplitude_m"] < LEAD_DEAD_BAND_L * hop["leg_length_m"] / 10
        assert hop["lead_swaps_per_stride"] == 0.0
        assert failing_keys(hop, BIPED) == {"min_lead_swaps_per_stride"}
        capped = gait_clause_from_curriculum(
            {**PROVISIONAL_BIPED_GAIT_THRESHOLDS, "max_flight_fraction": PROVISIONAL_BIPED_FLIGHT_FRACTION}
        )
        assert failing_keys(hop, capped) == {"min_lead_swaps_per_stride", "max_flight_fraction"}
        assert _sign_change_lead_swaps(trace) / hop["strides"] > 1.0
        assert abs(hop["left_right_phase"] - 0.5) > 0.4

    def test_a_scripted_tyrannosaurus_rex_hop_fails(self, tyrannosaurus_rex_traces):
        """Both feet leave and land together: no alternation, simultaneous landings, flight that is hop flight."""
        trace = tyrannosaurus_rex_traces["hop"]
        hop = metrics(trace)
        assert not trace.terminated and hop["touchdown_events"] >= 6
        assert hop["simultaneous_fraction"] == 1.0 and hop["alternation_index"] == 0.0
        assert hop["hop_flight_fraction"] > 0.25
        assert {"min_alternation_index", "max_simultaneous_fraction", "max_hop_flight_fraction"} <= failing_keys(
            hop, BIPED
        )

    def test_a_scripted_tyrannosaurus_rex_slide_fails(self, tyrannosaurus_rex_traces):
        """Both feet down for the whole window while the body travels at a walking speed: the clause has no footfall."""
        trace = tyrannosaurus_rex_traces["slide"]
        slide = metrics(trace)
        assert not trace.terminated and slide["all_limb_support_fraction"] == 1.0
        assert slide["touchdown_events"] == 0.0 and slide["displacement_speed_mps"] > 0.4
        # Report-only: the planted feet slide with the body.
        assert slide["stance_slip_speed_mps"] > 0.4
        assert failing_keys(slide, BIPED) >= {"min_alternation_index", "min_lead_swaps_per_stride"}

    def test_the_brachiosaurus_altithorax_reference_walk_passes(self, brachiosaurus_altithorax_traces):
        trace = brachiosaurus_altithorax_traces["walk"]
        walk = metrics(trace)
        assert not trace.terminated and walk["limbs"] == 4.0
        assert walk["flight_fraction"] == 0.0 and walk["min_limb_duty"] > 0.5
        assert walk["pair_alternation_index"] > 0.9
        # Report-only: a lateral-sequence walk, inside §4.3 B's brachiosaurus band [0.10, 0.45].
        assert 0.10 <= walk["ipsilateral_phase"] <= 0.45
        assert gait_failures(cruise(walk), QUADRUPED) == []
        assert classify_cruise_episode(cruise(walk), QUADRUPED_WALK) == ()

    def test_a_scripted_brachiosaurus_altithorax_hop_fails(self, brachiosaurus_altithorax_traces):
        """A pronk: the four feet leave and land together."""
        trace = brachiosaurus_altithorax_traces["hop"]
        pronk = metrics(trace)
        assert not trace.terminated and pronk["simultaneous_fraction"] == 1.0
        assert pronk["pair_alternation_index"] == 0.0 and pronk["flight_fraction"] > 0.3
        assert failing_keys(pronk, QUADRUPED) == {"min_pair_alternation_index", "max_flight_fraction"}

    def test_a_scripted_brachiosaurus_altithorax_slide_fails(self, brachiosaurus_altithorax_traces):
        trace = brachiosaurus_altithorax_traces["slide"]
        slide = metrics(trace)
        assert not trace.terminated and slide["all_limb_support_fraction"] == 1.0
        assert slide["touchdown_events"] == 0.0 and slide["displacement_speed_mps"] > 0.4
        # Flight and duty hold on a slide; the pair alternation, with no footfall, is unmeasured.
        assert failing_keys(slide, QUADRUPED) == {"min_pair_alternation_index"}

    @pytest.mark.parametrize(
        "model_id, make, seconds, clause",
        [
            (
                ModelId("velociraptor_mongoliensis", "anatomical"),
                _reference_walk("velociraptor_mongoliensis"),
                8.0,
                BIPED,
            ),
            (ModelId("allosaurus_fragilis", "anatomical"), _allosaurus_walk, 8.0, BIPED),
            (
                ModelId("dibothrosuchus_elaphros", "anatomical"),
                _reference_walk("dibothrosuchus_elaphros"),
                7.7,
                QUADRUPED,
            ),
        ],
        ids=["velociraptor_mongoliensis", "allosaurus_fragilis", "dibothrosuchus_elaphros"],
    )
    def test_the_other_reference_walks_pass(self, model_id, make, seconds, clause):
        trace = _roll(model_id, seconds, controller=make)
        walk = metrics(trace)
        assert not trace.terminated
        assert gait_failures(cruise(walk), clause) == [], walk
