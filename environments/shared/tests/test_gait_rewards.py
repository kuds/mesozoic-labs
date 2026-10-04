"""The gait reward kit (GAIT_QUALITY_PLAN_2026_09 §5.2-§5.3) on scripted contact sequences.

Every term is pinned without a simulator: :class:`GaitPhaseTracker` is driven
by per-substep floor forces and foot positions of scripted gaits at the T. rex
scale (2 ms substeps, 10 ms control steps, 840 N, L = 0.8766 m, so
T_sw = 0.389 s and the step length l = 0.561 m), and the per-control-step
flight and slip terms by the per-foot support those sequences produce.  A
statue, a two-footed hop, a staggered hop, a one-legged hop with a tapping
foot and tapping in place earn no gait phase; a walk earns ``w`` per control
step; a step-to earns its leading foot's half.  The legacy table, the
carve-out of the task fingerprint and ``stage_config.json`` and the
configuration checks are pinned here too; the T. rex wiring is in
``environments/trex/tests/test_trex_gait_rewards.py``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from environments.shared.gait_rewards import (
    GAIT_REWARD_KIT_LEGACY,
    GaitPhaseTracker,
    GaitRewardConfig,
    drop_inert_kit_keys,
    flight_term,
    slip_term,
)

SUBSTEP = 0.002
FRAME_SKIP = 5
CONTROL_DT = SUBSTEP * FRAME_SKIP
BODY_WEIGHT = 840.0
LEG = 0.8766
GRAVITY = 9.81
SWING_REFERENCE = 1.3 * math.sqrt(LEG / GRAVITY)
STEP_LENGTH = 0.64 * LEG
#: A stance load: well above the step load (half of the 420 N share of body weight).
LOAD = 0.6 * BODY_WEIGHT
#: A tap: above the touch threshold (4.2 N) and below the step load (210 N).
TAP = 50.0
DOWN_N = 0.1 * BODY_WEIGHT / 2


@dataclass(frozen=True)
class Stance:
    """One scripted stance: loaded on ``[on, off)`` seconds at ``x`` (and ``y``), carrying ``load``."""

    on: float
    off: float
    x: float
    y: float = 0.0
    load: float = LOAD


@dataclass
class Episode:
    reward: float
    steps: int
    control_rewards: list[float]
    control_support: list[tuple[float, float]]
    events: list[tuple[float, float]]


def _tracker(weight: float = 1.0, step_over_leg: float = 0.64) -> GaitPhaseTracker:
    return GaitPhaseTracker(
        feet=2,
        substep_s=SUBSTEP,
        control_dt=CONTROL_DT,
        body_weight_n=BODY_WEIGHT,
        leg_length_m=LEG,
        gravity=GRAVITY,
        weight=weight,
        step_over_leg=step_over_leg,
    )


def _sample(stances: Sequence[Stance], index: int) -> tuple[float, float, float]:
    for stance in stances:
        if round(stance.on / SUBSTEP) <= index < round(stance.off / SUBSTEP):
            return stance.load, stance.x, stance.y
    return 0.0, float("nan"), 0.0


def run(
    right: Sequence[Stance],
    left: Sequence[Stance],
    duration: float,
    tracker: GaitPhaseTracker | None = None,
    axis: tuple[float, float] = (1.0, 0.0),
) -> Episode:
    """Drive *tracker* with the scripted stances, sample 0 at the reset and one per substep after it."""
    tracker = tracker or _tracker()
    total, steps = 0.0, 0
    control_rewards: list[float] = []
    control_support: list[tuple[float, float]] = []
    events: list[tuple[float, float]] = []
    samples = round(duration / SUBSTEP)
    step_reward, minima = 0.0, [math.inf, math.inf]
    for index in range(1, samples + 1):
        forces, positions = [], []
        for side in (right, left):
            force, x, y = _sample(side, index)
            forces.append(force)
            positions.append((0.0 if math.isnan(x) else x, y))
        reward, completed, quality = tracker.sample(forces, positions, lambda: axis)
        if completed:
            events.append((index * SUBSTEP, reward))
        total += reward
        steps += completed
        step_reward += reward
        minima = [min(m, f) for m, f in zip(minima, forces, strict=True)]
        if index % FRAME_SKIP == 0:
            control_rewards.append(step_reward)
            control_support.append((minima[0], minima[1]))
            step_reward, minima = 0.0, [math.inf, math.inf]
    return Episode(total, steps, control_rewards, control_support, events)


def walk(
    duration: float = 10.0,
    stride: float = 1.0,
    duty: float = 0.6,
    step: float = 0.6,
    left_step: float | None = None,
    lag: float = 0.5,
    start: float = 0.5,
) -> tuple[list[Stance], list[Stance]]:
    """An alternating gait after a quiet start: right touchdowns at ``start + k * stride``, left ``lag`` later.

    Each foot lands ``step`` (the left foot ``left_step``) ahead of the other foot's latest footprint.
    """
    left_step = step if left_step is None else left_step
    right: list[Stance] = [Stance(0.0, start + stride * duty - stride, 0.0)]
    left: list[Stance] = [Stance(0.0, start + lag * stride + stride * duty - stride, 0.0, y=0.3)]
    right_x, left_x = 0.0, 0.0
    k = 0
    while start + k * stride < duration:
        on = start + k * stride
        right_x = left_x + step
        right.append(Stance(on, on + duty * stride, right_x))
        on_left = on + lag * stride
        left_x = right_x + left_step
        left.append(Stance(on_left, on_left + duty * stride, left_x, y=0.3))
        k += 1
    return right, left


def hop(
    duration: float = 6.0, period: float = 0.13, duty: float = 0.65, lag: float = 0.0, advance: float = 0.15
) -> tuple[list[Stance], list[Stance]]:
    """Two feet landing together every ``period`` (the left foot ``lag`` later), both moving ``advance`` per hop.

    The left foot rides 0.147 m ahead of the right (the seed-42 walker's split stance).
    """
    right, left = [], []
    k = 0
    while k * period < duration:
        on = 0.1 + k * period
        right.append(Stance(on, on + duty * period, k * advance))
        left.append(Stance(on + lag, on + lag + duty * period, k * advance + 0.147, y=0.3))
        k += 1
    return right, left


# ---------------------------------------------------------------------------
# gait phase: the exploits earn nothing
# ---------------------------------------------------------------------------


def test_statue_takes_no_step_and_earns_nothing() -> None:
    episode = run([Stance(0.0, 20.0, 0.0)], [Stance(0.0, 20.0, 0.0, y=0.3)], 10.0)
    assert episode.reward == 0.0
    assert episode.steps == 0


def test_two_footed_hop_earns_nothing() -> None:
    # The audited seed-42 hop: 7.7 Hz, 35% airborne, simultaneous touchdowns.
    right, left = hop()
    episode = run(right, left, 6.0)
    assert episode.steps > 80  # every landing is a loaded step ...
    assert episode.reward == 0.0  # ... and none alternates


def test_staggered_hop_within_a_tenth_of_a_stride_earns_nothing() -> None:
    # The left foot lands 6 ms (under 0.1 * 130 ms) after the right, and ahead of it.
    right, left = hop(lag=0.006)
    episode = run(right, left, 6.0)
    assert episode.steps > 80
    assert episode.reward == 0.0


def test_staggered_hop_past_a_tenth_of_a_stride_earns_almost_nothing() -> None:
    # 20 ms apart (0.15 of a stride): P = exp(-(0.35 / 0.15)^2), and short hop swings
    # cut A, so the rate stays under 1% of a walk's ``w`` per control step.
    right, left = hop(lag=0.02)
    episode = run(right, left, 6.0)
    assert 0.0 < episode.reward / len(episode.control_rewards) < 0.01


def test_one_legged_hop_with_a_tapping_foot_earns_nothing() -> None:
    # The right foot hops forward at 3 Hz; the left foot taps lightly (50 N) mid-flight, alternating with it.
    right = [Stance(0.1 + k / 3, 0.1 + k / 3 + 0.2, 0.3 * k) for k in range(30)]
    left = [Stance(0.1 + k / 3 + 0.25, 0.1 + k / 3 + 0.28, 0.3 * k + 0.4, y=0.3, load=TAP) for k in range(30)]
    episode = run(right, left, 10.0)
    assert episode.steps > 20  # the hopping foot steps; the taps are no steps
    assert episode.reward == 0.0


def test_tapping_beside_a_standing_foot_earns_nothing() -> None:
    right = [Stance(0.0, 20.0, 0.0)]
    left = [Stance(0.1 + 0.4 * k, 0.1 + 0.4 * k + 0.1, 0.5 * (k % 2), y=0.3) for k in range(25)]
    episode = run(right, left, 10.0)
    assert episode.steps > 20  # loaded taps are steps of the left foot, the right foot never steps
    assert episode.reward == 0.0


def test_fast_tapping_in_place_earns_nothing() -> None:
    # Marching in place at 5 Hz (80 ms swings): alternation is perfect, but no foot steps through.
    right, left = walk(stride=0.2, duty=0.6, step=0.0)
    episode = run(right, left, 10.0)
    assert episode.steps > 90
    assert episode.reward == 0.0


def test_fast_tapping_that_creeps_forward_earns_little() -> None:
    # 5 cm steps at 5 Hz: S = 0.09, A = 0.08 / 0.389: under 3% of a walk's rate.
    right, left = walk(stride=0.2, duty=0.6, step=0.05)
    episode = run(right, left, 10.0)
    rate = episode.reward / len(episode.control_rewards)
    assert 0.0 < rate < 0.03


# ---------------------------------------------------------------------------
# gait phase: the walk, the step-to and the run
# ---------------------------------------------------------------------------


def test_walk_earns_the_weight_per_control_step() -> None:
    weight = 0.3
    right, left = walk(duration=10.0, stride=1.0, duty=0.6, step=0.6)
    episode = run(right, left, 10.0, _tracker(weight))
    # Every paid step credits its 1 s stride: w * 1.0 / (2 * 0.01) with P = A = S = 1.
    paid = [reward for _, reward in episode.events if reward > 0.0]
    assert paid == pytest.approx([weight * 1.0 / (2 * CONTROL_DT)] * len(paid), rel=1e-12)
    # The first step of each foot only starts its stride clock; every later one pays.
    assert len(paid) == episode.steps - 2
    # Over whole strides the rate is w per control step, whatever the cadence.
    assert sum(paid) / (len(paid) / 2 * 1.0 / CONTROL_DT) == pytest.approx(weight, rel=1e-12)
    fast_right, fast_left = walk(duration=10.0, stride=0.8, duty=0.5, step=0.6)
    fast = run(fast_right, fast_left, 10.0, _tracker(weight))
    fast_paid = [reward for _, reward in fast.events if reward > 0.0]
    assert sum(fast_paid) / (len(fast_paid) / 2 * 0.8 / CONTROL_DT) == pytest.approx(weight, rel=1e-12)


def test_walk_steps_are_paid_when_their_stance_takes_load() -> None:
    right, left = walk(duration=4.0)
    episode = run(right, left, 4.0)
    # Touchdowns at 0.5 + k (right) and 1.0 + k (left): each is a step at its third loaded sample.
    times = [round(time, 3) for time, _ in episode.events]
    assert times[:4] == [0.504, 1.004, 1.504, 2.004]


def test_step_to_earns_the_leading_foots_half() -> None:
    through = run(*walk(step=0.6), 10.0)
    step_to = run(*walk(step=0.6, left_step=0.0), 10.0)  # the left foot lands level with the right
    assert [time for time, _ in step_to.events] == [time for time, _ in through.events]
    for (time, reward), (_, walked) in zip(step_to.events, through.events, strict=True):
        # Right touchdowns at 0.5 + k pay as in the walk; left ones (at whole seconds) pay nothing.
        left_foot = round(time - 0.004, 6) % 1.0 == 0.0
        assert reward == (0.0 if left_foot else walked)
    assert step_to.reward > 0.0


def test_short_steps_earn_in_proportion_to_their_length() -> None:
    long = run(*walk(step=0.6), 10.0)
    short = run(*walk(step=0.25 * STEP_LENGTH), 10.0)
    assert short.reward == pytest.approx(0.25 * long.reward, rel=1e-9)


def test_aerial_run_earns_gait_phase_and_pays_flight_on_its_flight_steps() -> None:
    # Duty 0.35 at 1.67 Hz: 0.21 s stances, 0.39 s swings, two flights per stride.
    right, left = walk(stride=0.6, duty=0.35, step=0.8)
    episode = run(right, left, 10.0)
    assert episode.reward > 0.0
    flights = [flight_term(support, DOWN_N, 1, 0.5) for support in episode.control_support]
    flying = [index for index, (_, _, flight) in enumerate(flights) if flight]
    # Each 0.6 s stride holds two 0.09 s flights: about 30% of control steps after the quiet start.
    assert 0.25 < len(flying) / len(flights) < 0.4
    assert all(reward == -0.5 for reward, _, flight in flights if flight)
    assert all(reward == 0.0 for reward, _, flight in flights if not flight)


def test_walk_and_statue_have_no_flight_step_and_hops_do() -> None:
    for right, left in (walk(), ([Stance(0.0, 20.0, 0.0)], [Stance(0.0, 20.0, 0.0)])):
        episode = run(right, left, 10.0)
        assert not any(flight_term(s, DOWN_N, 1, 0.5)[2] for s in episode.control_support)
    episode = run(*hop(), 6.0)
    flying = sum(flight_term(s, DOWN_N, 1, 0.5)[2] for s in episode.control_support)
    assert 0.3 < flying / len(episode.control_support) < 0.8


# ---------------------------------------------------------------------------
# the factors, one at a time
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lag", [0.3, 0.4, 0.5, 0.65])
def test_p_is_a_gaussian_of_the_relative_phase(lag: float) -> None:
    # The left foot lands ``lag`` of a stride after the right: phase lag for the right foot's
    # steps, 1 - lag for the left foot's, the same distance from anti-phase.
    episode = run(*walk(lag=lag, step=0.6), 10.0)
    paid = [reward for _, reward in episode.events][2:]
    expected = math.exp(-(((lag - 0.5) / 0.15) ** 2)) / (2 * CONTROL_DT)
    assert paid == pytest.approx([expected] * len(paid), rel=1e-9)


@pytest.mark.parametrize("lag", [0.05, 0.09, 0.91, 0.97])
def test_landings_within_a_tenth_of_a_stride_are_a_hop(lag: float) -> None:
    episode = run(*walk(lag=lag), 10.0)
    assert episode.steps > 15
    assert episode.reward == 0.0


def test_a_is_the_swing_over_the_reference_swing() -> None:
    # A 0.2 s swing: A = 0.2 / 0.389; S and P are 1.
    episode = run(*walk(stride=1.0, duty=0.8, step=0.6), 10.0)
    paid = [reward for _, reward in episode.events if reward > 0.0]
    assert paid[-1] == pytest.approx((0.2 / SWING_REFERENCE) * 1.0 / (2 * CONTROL_DT), rel=1e-9)


def test_s_is_measured_along_the_trunk_heading() -> None:
    right, left = walk(step=0.6)
    straight = run(right, left, 10.0)
    backwards = run(right, left, 10.0, axis=(-1.0, 0.0))
    assert straight.reward > 0.0
    assert backwards.reward == 0.0
    # The trunk faces +y while the feet advance along x: only the left foot, which sits 0.3 m to
    # the trunk's left (+y), lands "ahead" of the right one, by that 0.3 m.
    sideways = run(right, left, 10.0, axis=(0.0, 1.0))
    paid = sorted({round(reward, 9) for _, reward in sideways.events})
    assert paid == [0.0, round(0.3 / STEP_LENGTH / (2 * CONTROL_DT), 9)]


def test_a_long_pause_is_credited_five_reference_swings() -> None:
    # Strides of 3 s, alternating at half a stride: each step credits 5 * 0.389 s, not 3 s.
    episode = run(*walk(duration=20.0, stride=3.0, duty=0.6, step=0.6), 20.0)
    paid = [reward for _, reward in episode.events if reward > 0.0]
    assert paid == pytest.approx([5 * SWING_REFERENCE / (2 * CONTROL_DT)] * len(paid), rel=1e-12)


def test_a_blip_is_no_touchdown_and_chatter_does_not_end_a_stance() -> None:
    tracker = _tracker()
    base_right, base_left = walk(duration=6.0)
    clean = run(base_right, base_left, 6.0, tracker)
    # A 4 ms blip of the left foot mid-swing is no touchdown: no step, no swing restart.
    blip_left = [*base_left, Stance(1.7, 1.704, 5.0, y=0.3)]
    blipped = run(base_right, blip_left, 6.0, _tracker())
    assert blipped.events == clean.events
    # An 8 ms unload inside a stance (chatter) does not split it: same steps, same pay.
    split = [Stance(s.on, s.on + 0.2, s.x, s.y) if 1.0 < s.on < 5.0 else s for s in base_left] + [
        Stance(s.on + 0.208, s.off, s.x, s.y) for s in base_left if 1.0 < s.on < 5.0
    ]
    chattered = run(base_right, split, 6.0, _tracker())
    assert chattered.events == clean.events


def test_an_unload_past_the_chatter_fill_is_a_new_touchdown_with_a_short_swing() -> None:
    base_right, base_left = walk(duration=6.0)
    # The left stance landing at 2.0 s unloads for 12 ms (six samples) 0.2 s after its touchdown.
    split = [Stance(s.on, s.on + 0.2, s.x, s.y) if 1.5 < s.on < 2.5 else s for s in base_left] + [
        Stance(s.on + 0.212, s.off, s.x, s.y) for s in base_left if 1.5 < s.on < 2.5
    ]
    episode = run(base_right, split, 6.0)
    clean = run(base_right, base_left, 6.0)
    assert episode.steps == clean.steps + 1
    assert episode.reward < clean.reward


# ---------------------------------------------------------------------------
# reset and determinism
# ---------------------------------------------------------------------------


def test_reset_starts_a_new_episode_and_the_reset_stance_is_no_step() -> None:
    right, left = walk(duration=6.0)
    tracker = _tracker()
    first = run(right, left, 6.0, tracker)
    tracker.reset()
    second = run(right, left, 6.0, tracker)
    fresh = run(right, left, 6.0)
    assert first.events == second.events == fresh.events
    assert first.reward == second.reward  # bit for bit


def test_a_reset_mid_walk_rearms_every_stride_clock() -> None:
    right, left = walk(duration=6.0)
    tracker = _tracker()
    run(right, left, 3.3, tracker)
    tracker.reset()
    # The walk goes on from 3.3 s, the feet now where they were: the first step of each foot pays nothing.
    shifted_right = [Stance(s.on - 3.3, s.off - 3.3, s.x) for s in right if s.off > 3.3]
    shifted_left = [Stance(s.on - 3.3, s.off - 3.3, s.x, s.y) for s in left if s.off > 3.3]
    resumed = run(
        [Stance(max(s.on, 0.0), s.off, s.x, s.y) for s in shifted_right],
        [Stance(max(s.on, 0.0), s.off, s.x, s.y) for s in shifted_left],
        2.7,
        tracker,
    )
    rewards = [reward for _, reward in resumed.events]
    assert rewards[:2] == [0.0, 0.0]
    assert all(reward > 0.0 for reward in rewards[2:])


def test_identical_sequences_give_identical_rewards() -> None:
    right, left = walk(duration=8.0, stride=0.9, duty=0.62, step=0.55, lag=0.47)
    assert run(right, left, 8.0) == run(right, left, 8.0)


def test_quadruped_phase_templates_are_not_implemented() -> None:
    with pytest.raises(ValueError, match="bipeds only"):
        GaitPhaseTracker(
            feet=4,
            substep_s=SUBSTEP,
            control_dt=CONTROL_DT,
            body_weight_n=BODY_WEIGHT,
            leg_length_m=LEG,
            gravity=GRAVITY,
            weight=1.0,
            step_over_leg=0.64,
        )


# ---------------------------------------------------------------------------
# flight and slip
# ---------------------------------------------------------------------------


def test_flight_counts_the_feet_that_carry_a_tenth_of_their_share() -> None:
    assert flight_term((DOWN_N, 0.0), DOWN_N, 1, 0.5) == (0.0, 1, False)
    assert flight_term((DOWN_N - 1e-9, 0.0), DOWN_N, 1, 0.5) == (-0.5, 0, True)
    assert flight_term((400.0, 0.0), DOWN_N, 2, 0.5) == (-0.5, 1, True)
    assert flight_term((400.0, 400.0), DOWN_N, 2, 0.5) == (0.0, 2, False)
    assert flight_term((0.0, 0.0), DOWN_N, 1, 0.0) == (0.0, 0, True)  # no -0.0 at a zero weight


def test_slip_scales_with_speed_and_counts_only_the_feet_down() -> None:
    scale = math.sqrt(GRAVITY * LEG)
    reward, slip = slip_term((0.3, 0.0), (400.0, 400.0), DOWN_N, scale, 0.2)
    assert slip == pytest.approx(0.3 / scale)
    assert reward == pytest.approx(-0.2 * 0.3 / scale)
    assert slip_term((0.6, 0.0), (400.0, 400.0), DOWN_N, scale, 0.2)[0] == pytest.approx(2 * reward)
    assert slip_term((0.3, 0.3), (400.0, 400.0), DOWN_N, scale, 0.2)[0] == pytest.approx(2 * reward)
    assert slip_term((0.3, 0.3), (400.0, 0.0), DOWN_N, scale, 0.2)[0] == pytest.approx(reward)
    assert slip_term((0.3, 0.3), (0.0, 0.0), DOWN_N, scale, 0.2) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# the legacy table, the carve-out and the configuration checks
# ---------------------------------------------------------------------------


def test_the_legacy_table_is_pinned() -> None:
    assert dict(GAIT_REWARD_KIT_LEGACY) == {
        "support_source": "touch",
        "gait_phase_weight": 0.0,
        "gait_phase_step_over_leg": 0.64,
        "flight_penalty_weight": 0.0,
        "flight_min_feet": 1,
        "foot_slip_penalty_weight": 0.0,
        "foot_collision_penalty_weight": 0.0,
        "leg_contact_penalty_weight": 0.0,
        "terminate_on_leg_contact": False,
    }


def test_carve_out_drops_only_unset_knobs_at_their_legacy_value() -> None:
    effective = {**GAIT_REWARD_KIT_LEGACY, "alive_bonus": 0.5}
    drop_inert_kit_keys(effective, {"alive_bonus": 0.5})
    assert effective == {"alive_bonus": 0.5}
    # A knob the stage sets stays, even at its legacy value.
    effective = {**GAIT_REWARD_KIT_LEGACY}
    drop_inert_kit_keys(effective, {"gait_phase_weight": 0.0, "support_source": "floor"})
    assert set(effective) == {"gait_phase_weight", "support_source"}
    # A default retuned away from the legacy value stays (it moves the digest).
    effective = {**GAIT_REWARD_KIT_LEGACY, "gait_phase_step_over_leg": 0.7, "flight_min_feet": 2}
    drop_inert_kit_keys(effective, {})
    assert effective == {"gait_phase_step_over_leg": 0.7, "flight_min_feet": 2}
    # Equal but not the same type (a bool for an int) is not the legacy value.
    effective = {"flight_min_feet": True, "terminate_on_leg_contact": 0}
    drop_inert_kit_keys(effective, {})
    assert effective == {"flight_min_feet": True, "terminate_on_leg_contact": 0}


def _stage_configs() -> list[tuple[str, int | str, dict]]:
    from environments.shared.config import SPECIES_NAMES, load_stage_config
    from environments.shared.stage_manifest import load_stage_manifest

    return [
        (species, entry.reference, load_stage_config(species, entry.reference))
        for species in SPECIES_NAMES
        for entry in load_stage_manifest(species).stages
    ]


#: The stages whose TOML sets kit knobs: each is a named task revision ``gait-r1`` (plan §5.2 item 5),
#: recorded in a decision row (trex locomotion: D-D23).  A knob set anywhere else fails below.
GAIT_R1_STAGES = {("trex", "locomotion")}


def test_a_stage_records_only_the_kit_knobs_it_sets(tmp_path) -> None:
    # Every species' every stage: the task fingerprint's effective config and the stage_config.json
    # a run records carry exactly the kit knobs the stage's TOML sets, so a stage that sets none keeps
    # its pre-kit digests, and only the named gait-r1 revisions set any.
    from environments.shared.config import save_stage_config
    from environments.shared.species_registry import get_species_config
    from environments.shared.stage_manifest import load_stage_manifest
    from environments.shared.task_fingerprint import _effective_env_kwargs

    configs = _stage_configs()
    assert len(configs) >= 20
    revised = set()
    for species, stage, config in configs:
        stage_id = load_stage_manifest(species).resolve(stage).id
        env_kwargs = config.get("env_kwargs", {})
        explicit = set(GAIT_REWARD_KIT_LEGACY) & set(env_kwargs)
        if explicit:
            revised.add((species, stage_id))
        assert set(GAIT_REWARD_KIT_LEGACY) & set(_effective_env_kwargs(species, env_kwargs)) == explicit, stage_id
        path = save_stage_config(
            tmp_path / species / str(stage),
            stage,
            config,
            "PPO",
            env_class=get_species_config(species).env_class,
            species=species,
        )
        recorded = json.loads(path.read_text())["reward_weights"]
        assert set(GAIT_REWARD_KIT_LEGACY) & set(recorded) == explicit, (species, stage_id)
    assert revised == GAIT_R1_STAGES


def test_a_knob_a_stage_sets_is_recorded(tmp_path) -> None:
    from environments.shared.config import load_stage_config, save_stage_config
    from environments.shared.task_fingerprint import _effective_env_kwargs
    from environments.trex.envs.trex_env import TRexEnv

    # The stance sets no kit knob (the locomotion stage is the gait-r1 revision and sets four).
    config = load_stage_config("trex", "stance")
    config["env_kwargs"] = {**config["env_kwargs"], "gait_phase_weight": 0.3, "flight_min_feet": 1}
    effective = _effective_env_kwargs("trex", config["env_kwargs"])
    path = save_stage_config(tmp_path, 1, config, "PPO", env_class=TRexEnv, species="trex")
    recorded = json.loads(path.read_text())
    for env in (effective, recorded["reward_weights"]):
        # Set knobs stay, the legacy value of flight_min_feet included; unset ones stay carved out.
        assert env["gait_phase_weight"] == 0.3 and env["flight_min_feet"] == 1
        assert "support_source" not in env and "terminate_on_leg_contact" not in env


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"support_source": "sensor"}, "support_source"),
        ({"gait_phase_weight": -0.1}, "gait_phase_weight"),
        ({"flight_penalty_weight": float("nan")}, "flight_penalty_weight"),
        ({"foot_slip_penalty_weight": True}, "foot_slip_penalty_weight"),
        ({"gait_phase_step_over_leg": 0.0}, "gait_phase_step_over_leg"),
        ({"flight_min_feet": 0}, "at least 1"),
        ({"flight_min_feet": 1.0}, "integer"),
        ({"terminate_on_leg_contact": 1}, "bool"),
    ],
)
def test_invalid_configurations_fail_fast(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        GaitRewardConfig(**kwargs)
