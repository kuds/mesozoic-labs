"""End-to-end walk-first gait verdicts: measured synthetic footfall patterns -> stored metrics -> gate.

Each case renders a footfall pattern (``gait_trace`` / ``scheduled_trace``), measures it at 2 ms
(or 10 ms), round-trips the metrics through JSON exactly as a report stores them, and judges them
under the calibrated ``provisional_gait_criteria`` of the walk-first profiles: ``biped_walk`` and
``biped_alternating`` on two feet, ``quadruped_walk`` on four. The cases are the defects a gait
certificate must not have (the 2026-10 review's D1-D14, the verification holes H1-H8 and the round-2
reward hacks) plus the genuine gaits it must not reject, judged by the owner's walk-first decisions:
a step-to gait is refused, timing has a wide tolerance, one off-gait budget of 15 % of the window
absorbs pauses and short bouts, and any symmetrical quadruped walking gait is a walk.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from environments.shared.curriculum.gait_gate import (
    GaitGateThresholds,
    classify_gait_episode,
    provisional_gait_criteria,
    rail_id,
)
from environments.shared.gait.events import boolean_runs
from environments.shared.gait.metrics import episode_gait_metrics
from environments.shared.gait.types import GaitProtocol

from .test_gait_metrics import BW, gait_trace, mirrored, rotated, scheduled_trace, yaw_quaternion

BIPED = ("r", "l")
QUAD = ("fr", "fl", "rr", "rl")
SETTLE = 1.0
LS_WALK = (0.25, 0.75, 0.0, 0.5)  # fore lags the ipsilateral hind by 1/4 cycle
DS_WALK = (0.75, 0.25, 0.0, 0.5)  # ... by 3/4 cycle
TROT = (0.0, 0.5, 0.5, 0.0)
PACE = (0.0, 0.5, 0.0, 0.5)
BIPED_PROFILES = ("biped_walk", "biped_alternating")
WALK = ((0.0, 0.5), 0.5, 0.6, 1.0)  # phases, period, duty, speed of one stride


def judge(trace, feet=BIPED, *, profiles=None, speed=0.4, duration=None):
    """Profile -> (qualified, reasons) from the JSON-stored metrics of ``trace`` (task direction +x)."""
    measured = episode_gait_metrics(
        trace, body_weight_n=BW, leg_length_m=1.0, foot_names=feet, protocol=GaitProtocol(), settle_s=SETTLE
    )
    stored = json.loads(json.dumps(measured, allow_nan=False))
    stored["completed_horizon"] = True
    if duration is None:
        duration = round(float(trace["time_s"][-1] - trace["time_s"][0]), 6) - SETTLE
    profiles = profiles or (BIPED_PROFILES if len(feet) == 2 else ("quadruped_walk",))
    out = {}
    for profile in profiles:
        thresholds = GaitGateThresholds.from_curriculum(
            {
                **provisional_gait_criteria(profile),
                "gait_profile": profile,
                "measurement_protocol_sha256": "sha256:" + "0" * 64,
                "min_eval_episodes": 40,
                "gait_panel_seed_start": 0,
                "min_gait_success_lcb": 0.8,
                "min_episode_forward_vel": speed,
                "min_episode_duration_s": duration,
            }
        )
        out[profile] = classify_gait_episode(stored, thresholds, foot_names=feet)
    return out


def passed(verdicts):
    return {profile: result[0] for profile, result in verdicts.items()}


def rails(verdicts, profile="biped_walk"):
    return {rail_id(reason) for reason in verdicts[profile][1]}


def fore_hind_relabelled(trace):
    """Swap the fore and hind pair labels (limb phase becomes 1 - limb phase)."""
    out = {key: np.array(value, copy=True) for key, value in trace.items()}
    for key in ("floor_force_n", "foot_position_m", "foot_clearance_m", "slip_speed_mps", "touch_force_n"):
        out[key] = out[key][:, [2, 3, 0, 1]]
    return out


def crabbed(trace, degrees):
    """The trunk travels ``degrees`` off its body axis; every foot keeps its body-frame placement."""
    out = {key: np.array(value, copy=True) for key, value in trace.items()}
    shear = float(np.tan(np.radians(degrees)))
    for key in ("foot_position_m", "root_position_m"):
        out[key][..., 1] += shear * out[key][..., 0]
    return out


def _bouts(base, *bouts):
    """Stride plan: ``(every_s, from_s, to_s, stride)`` bouts replace the ``base`` stride."""

    def plan(start):
        for every, begin, end, stride in bouts:
            if begin <= start % every < end:
                return stride
        return base

    return plan


def yawed(trace, degrees):
    """The trunk held ``degrees`` off its travel while the footfalls stay where they are."""
    out = {key: np.array(value, copy=True) for key, value in trace.items()}
    out["root_quat_wxyz"] = yaw_quaternion(np.full(len(out["time_s"]), np.radians(degrees)))
    return out


def paused(trace, at_s, stop_s):
    """A genuine stop: every signal freezes for ``stop_s`` at ``at_s`` (feet planted), then the gait resumes."""
    time = trace["time_s"]
    dt = float(time[1] - time[0])
    k = int(np.searchsorted(time, at_s))
    hold = int(round(stop_s / dt))
    out = {}
    for key, value in trace.items():
        if key == "time_s":
            continue
        frozen = np.repeat(value[k : k + 1], hold, axis=0)
        if key == "slip_speed_mps":
            frozen = np.zeros_like(frozen)
        out[key] = np.concatenate((value[:k], frozen, value[k:]))[: len(time)]
    out["time_s"] = time.copy()
    return out


def merged_girdles(fore, hind):
    """A quadruped whose fore pair is the biped trace ``fore`` and hind pair ``hind`` (each girdle half the load)."""
    out = {}
    for key in ("floor_force_n", "touch_force_n"):
        out[key] = 0.5 * np.concatenate((fore[key], hind[key]), axis=1)
    for key in ("foot_position_m", "foot_clearance_m", "slip_speed_mps"):
        out[key] = np.concatenate((fore[key], hind[key]), axis=1)
    out["foot_position_m"][:, :2, 0] += 0.6
    for key in ("time_s", "body_floor_force_n", "foot_foot_force_n", "root_position_m", "root_quat_wxyz"):
        out[key] = np.array(hind[key], copy=True)
    return out


# -- genuine gaits (D1, D11, D12, D14) ---------------------------------------------------------


@pytest.mark.parametrize("jitter", [0.0, 0.01, 0.03])
@pytest.mark.parametrize("seed", [0, 1])
def test_jittered_biped_walks_are_walks_and_alternating_gaits(jitter, seed):
    trace = gait_trace((0.0, 0.5), duty=0.62, period=0.5, speed=1.0, jitter=jitter, stride_cv=0.03, seed=seed)
    assert passed(judge(trace)) == {"biped_walk": True, "biped_alternating": True}


@pytest.mark.parametrize("seed", [0, 1])
def test_a_velociraptor_like_aerial_run_qualifies_as_alternating_only(seed):
    """A run (duty 0.33, flight ~0.33, Froude ~2) is the velociraptor stage's gait: never a walk."""
    run = gait_trace((0.0, 0.5), duty=0.33, period=0.21, speed=3.3, jitter=0.02, stride_cv=0.03, seed=seed)
    verdict = judge(run, speed=2.0)
    assert passed(verdict) == {"biped_walk": False, "biped_alternating": True}
    assert {"support/flight_fraction", "support/walking_duty"} <= rails(verdict)


def test_a_grounded_run_is_alternating_and_mostly_grounded():
    """Duty 0.48 without an aerial phase: the walk/grounded-run boundary is not drawn (flight is small)."""
    verdict = judge(gait_trace((0.0, 0.5), duty=0.48, period=0.45, speed=1.6, jitter=0.01, seed=2))
    assert verdict["biped_alternating"][0]
    assert "support/walking_duty" not in rails(verdict)


def test_a_genuine_left_heavy_walker_with_a_short_right_stance_is_a_walk():
    """compsognathus 0921-like: right duty 0.494, left 0.60, lag 0.44 (each foot down about half the stride)."""
    walk = gait_trace((0.0, 0.44), duty=(0.494, 0.6), period=0.8, speed=0.8, jitter=0.03, stride_cv=0.05, seed=4)
    verdict = judge(walk)
    assert passed(verdict) == {"biped_walk": True, "biped_alternating": True}, verdict


@pytest.mark.parametrize(
    "phases,duty,period",
    [
        (LS_WALK, 0.72, 1.2),
        (DS_WALK, 0.65, 1.0),
        ((0.15, 0.65, 0.0, 0.5), 0.7, 1.0),  # lateral couplets
        ((0.35, 0.85, 0.0, 0.5), 0.7, 1.0),  # diagonal couplets ("trotting walk")
        (TROT, 0.55, 0.6),  # walking trot
        (PACE, 0.55, 0.6),  # walking pace
    ],
)
@pytest.mark.parametrize("jitter", [0.01, 0.02])
def test_every_symmetrical_quadruped_walking_gait_is_a_quadruped_walk(phases, duty, period, jitter):
    trace = gait_trace(phases, duty=duty, period=period, speed=1.0, jitter=jitter, stride_cv=0.03, seed=4)
    verdict = judge(trace, QUAD)
    assert passed(verdict) == {"quadruped_walk": True}, verdict


@pytest.mark.parametrize(
    "phases,duty,rail",
    [
        (TROT, 0.3, "support/flight_fraction"),  # flying trot: suspension
        ((0.5, 0.5, 0.0, 0.0), 0.4, "persistence/off_gait_fraction"),  # bound
        ((0.0, 0.0, 0.0, 0.0), 0.4, "persistence/off_gait_fraction"),  # pronk
        ((0.6, 0.5, 0.1, 0.0), 0.4, "persistence/off_gait_fraction"),  # transverse gallop
        ((0.55, 0.45, 0.0, 0.0), 0.5, "persistence/off_gait_fraction"),  # half-bound
    ],
)
def test_suspension_and_asymmetric_quadruped_gaits_are_not_walks(phases, duty, rail):
    trace = gait_trace(phases, duty=duty, period=0.5, speed=1.5, jitter=0.01, seed=5)
    verdict = judge(trace, QUAD)
    assert not verdict["quadruped_walk"][0]
    assert rail in rails(verdict, "quadruped_walk")


def test_touchdown_skid_is_tolerated_but_a_skidding_scramble_is_not():
    assert judge(gait_trace((0.0, 0.5), duty=0.35, period=0.3, speed=2.0, slip=0.6))["biped_alternating"][0]
    verdict = judge(gait_trace((0.0, 0.5), duty=0.35, period=0.3, speed=2.0, slip=1.6))
    assert "support/skid_fraction_max" in rails(verdict, "biped_alternating")


@pytest.mark.parametrize("dt", [0.002, 0.01, 0.02])
def test_sampling_at_10_and_20_ms_keeps_the_verdict(dt):
    biped = gait_trace((0.0, 0.5), duty=0.6, period=0.5, dt=dt, jitter=0.01, stride_cv=0.03, seed=5)
    assert passed(judge(biped)) == {"biped_walk": True, "biped_alternating": True}
    quad = gait_trace(LS_WALK, duty=0.72, period=1.2, dt=dt, jitter=0.01, stride_cv=0.03, seed=5)
    assert passed(judge(quad, QUAD)) == {"quadruped_walk": True}


def test_float_accumulated_clock_meets_the_declared_duration():
    trace = gait_trace((0.0, 0.5), duty=0.6, period=0.5, duration=20.0, accumulate_time=True)
    assert trace["time_s"][-1] < 20.0  # 10000 additions of 0.002 end short of the horizon
    verdict = judge(trace, duration=19.0)
    assert verdict["biped_walk"][0], verdict


def test_a_trace_without_the_trunk_orientation_fails_closed():
    trace = gait_trace((0.0, 0.5), duty=0.6, period=0.5, speed=1.0)
    del trace["root_quat_wxyz"]
    for profile, (qualified, reasons) in judge(trace).items():
        assert not qualified and [rail_id(reason) for reason in reasons] == ["episode/telemetry_valid"], profile


# -- limb participation (D4) -------------------------------------------------------------------


def test_one_leg_hop_with_a_token_tapping_foot_fails():
    trace = gait_trace((0.0, 0.5), duty=(0.5, 0.08), period=0.5, speed=1.0, load=(1.0, 0.03))
    verdict = judge(trace)
    assert not any(passed(verdict).values())
    assert {"participation/limb_duty_min", "participation/relative_limb_load_share_min"} <= rails(verdict)


def test_three_legged_walk_with_a_tapping_limb_fails():
    trace = gait_trace(LS_WALK, duty=(0.72, 0.08, 0.72, 0.72), period=1.2, speed=1.0, load=(1.0, 0.03, 1.0, 1.0))
    verdict = judge(trace, QUAD)
    assert not verdict["quadruped_walk"][0]
    assert "participation/relative_limb_load_share_min" in rails(verdict, "quadruped_walk")


# -- persistence: one off-gait budget (D5) ------------------------------------------------------


@pytest.mark.parametrize("flip", [False, True])
def test_walk_then_hop_fails_and_so_does_its_mirror(flip):
    trace = gait_trace((0.0, 0.5), duty=0.6, period=0.5, speed=1.0, jitter=0.01, switch=(7.3, (0.0, 0.0)))
    verdict = judge(mirrored(trace) if flip else trace)
    assert not any(passed(verdict).values())
    assert "persistence/off_gait_fraction" in rails(verdict)


def test_walk_then_stand_fails_persistence():
    trace = gait_trace((0.0, 0.5), duty=0.6, period=0.5, speed=1.0, jitter=0.01, seed=6)
    still = trace["time_s"] >= 8.0
    trace["floor_force_n"][still] = BW / 2.0
    trace["foot_clearance_m"][still] = 0.0
    trace["foot_position_m"][still] = trace["foot_position_m"][np.argmax(still)]
    trace["root_position_m"][still, 0] = trace["root_position_m"][np.argmax(still), 0]
    verdict = judge(trace, speed=0.1)
    assert not any(passed(verdict).values())
    assert "persistence/off_gait_fraction" in rails(verdict)


@pytest.mark.parametrize("stop_s,qualifies", [(0.6, True), (2.4, False)])
def test_a_genuine_pause_costs_budget_time_but_only_a_long_one_fails(stop_s, qualifies):
    """A stop of 0.6 s in a 9 s window (7 %) is inside the 15 % budget; 2.4 s (27 %) is not."""
    stop = ((0.0, 0.5), stop_s, 1.0, 0.0)
    trace = scheduled_trace(_bouts(WALK, (20.0, 4.0, 4.0 + 0.5 * stop_s, stop)), jitter=0.01, seed=3)
    verdict = judge(trace, speed=0.1)
    assert verdict["biped_walk"][0] is qualifies, verdict
    if not qualifies:
        assert "persistence/off_gait_fraction" in rails(verdict)


@pytest.mark.parametrize("bout_s,qualifies", [(0.5, True), (1.0, False)])
def test_hop_bouts_inside_the_budget_pass_and_longer_ones_fail(bout_s, qualifies):
    """Two-foot hop bouts filling 10 % of the window fit the 15 % budget; 20 % do not."""
    hop = ((0.0, 0.0), 0.25, 0.4, 1.0)
    trace = scheduled_trace(_bouts(WALK, (5.0, 5.0 - bout_s, 5.0, hop)), jitter=0.01, seed=3)
    verdict = judge(trace)
    assert verdict["biped_walk"][0] is qualifies, verdict
    if not qualifies:
        assert "persistence/off_gait_fraction" in rails(verdict)


@pytest.mark.parametrize(
    "bout",
    [
        ((0.0, 0.5), 0.5, 0.6, 0.03),  # marking time: taps in place in an unbroken rhythm
        ((0.0, 0.5), 0.9, 0.85, 0.05),  # freezing: slow alternating strides, barely moving
    ],
)
def test_marking_time_and_freeze_bouts_are_off_gait_time(bout):
    """A quarter of the window spent in place fails (round-2 h06/h07: strides in place or standing)."""
    trace = scheduled_trace(_bouts(WALK, (5.0, 3.6, 5.0, bout)), jitter=0.01, seed=1)
    verdict = judge(trace, speed=0.2)
    assert not any(passed(verdict).values())
    assert "persistence/off_gait_fraction" in rails(verdict)


@pytest.mark.parametrize("switch_s", [3.0, 4.5])
def test_walk_to_run_transition_is_neither_a_pause_nor_a_stall(switch_s):
    def plan(start):
        return ((0.0, 0.5), 0.95, 0.62, 1.3) if start < switch_s else ((0.0, 0.5), 0.42, 0.35, 3.2)

    verdict = judge(scheduled_trace(plan, jitter=0.01, seed=2), speed=1.0)
    assert verdict["biped_alternating"][0], verdict


def test_a_slow_start_and_a_speed_change_are_not_penalised():
    """The removed stall rail called a genuine slow segment a stall; a creep at 0.3 of the cruise speed is a walk."""

    def plan(start):
        return ((0.0, 0.5), 0.5, 0.6, 0.3) if 3.0 <= start < 6.0 else WALK

    verdict = judge(scheduled_trace(plan, jitter=0.01, seed=4), speed=0.2)
    assert passed(verdict) == {"biped_walk": True, "biped_alternating": True}, verdict


# -- stepping: swing validity, step-through, stacking, timing (D6, D7, D9, D10) -----------------


@pytest.mark.parametrize("clearance", [0.004, 0.015])
def test_low_clearance_shuffle_fails(clearance):
    trace = gait_trace((0.0, 0.5), duty=0.6, period=0.5, speed=0.3, clearance=clearance)
    verdict = judge(trace, speed=0.1)
    assert not any(passed(verdict).values())
    assert rails(verdict) & {"stepping/median_swing_clearance_over_leg_min", "stepping/valid_swing_fraction_min"}


@pytest.mark.parametrize("degrees", [0.0, 3.0, 7.0, -12.0])
def test_a_step_to_gait_fails_step_through_even_when_its_trunk_crabs(degrees):
    """Round-2 crab attacks: the stance width projects onto the travel heading; the trunk axis does not."""
    trace = crabbed(gait_trace((0.0, 0.5), duty=0.6, period=0.5, speed=1.0, step_to=True, jitter=0.01), degrees)
    verdict = judge(trace, speed=0.4 * float(np.cos(np.radians(degrees))))
    assert not any(passed(verdict).values())
    assert "stepping/step_length_over_leg_min" in rails(verdict)


@pytest.mark.parametrize("degrees", [10.0, -20.0])
def test_a_genuine_walk_crabbing_with_body_frame_placement_passes(degrees):
    walk = crabbed(gait_trace((0.0, 0.5), duty=0.6, period=0.5, speed=1.0, jitter=0.01, seed=3), degrees)
    verdict = judge(walk, speed=0.4)
    assert passed(verdict) == {"biped_walk": True, "biped_alternating": True}, verdict


def _step_to_behind(trace, behind, foot=1):
    """Move every stance of ``foot`` so it lands ``behind`` (L) behind the other foot's footprint."""
    out = {key: np.array(value, copy=True) for key, value in trace.items()}
    other = 1 - foot
    for start, end in boolean_runs(out["floor_force_n"][:, foot] > 0.0):
        if start == 0:
            continue
        loaded = np.flatnonzero(out["floor_force_n"][:start, other] > 0.0)
        if not len(loaded):
            continue
        target = out["foot_position_m"][loaded[-1], other, 0] - behind
        out["foot_position_m"][start:end, foot, 0] += target - out["foot_position_m"][start, foot, 0]
    return out


def test_a_compsognathus_1001_like_step_to_is_refused():
    """Owner decision: the left foot landing a median 0.06-0.10 L behind the right is a step-to, not a walk."""
    walk = gait_trace((0.0, 0.5), duty=0.6, period=0.5, speed=1.0, jitter=0.02, stride_cv=0.04, seed=7)
    step_to = _step_to_behind(walk, 0.08)
    verdict = judge(step_to)
    assert not any(passed(verdict).values())
    assert "stepping/step_length_over_leg_min" in rails(verdict)
    assert passed(judge(walk)) == {"biped_walk": True, "biped_alternating": True}


def _switching_step_to(trace, stride, block, behind=0.01):
    """A step-to gait whose leading foot switches every ``block`` strides, one half step-through per switch."""
    out = {key: np.array(value, copy=True) for key, value in trace.items()}
    stances = sorted(
        (start, end, foot) for foot in (0, 1) for start, end in boolean_runs(out["floor_force_n"][:, foot] > 0.0)
    )
    latest: dict[int, float | None] = {0: None, 1: None}
    leader, led = 0, 0
    for start, end, foot in stances:
        reference = latest[1 - foot]
        if reference is None:
            x = float(out["foot_position_m"][start, foot, 0])
        elif led == block:
            x = reference + stride / 2.0  # the switch: each foot steps through by half a stride
            if foot != leader:
                leader, led = foot, 0
        elif foot == leader:
            x = reference + stride + behind
        else:
            x = reference - behind  # the trailing foot lands just behind the leader
            led += 1
        out["foot_position_m"][start:end, foot, 0] = x
        latest[foot] = x
    return out


@pytest.mark.parametrize("block", [2, 4])
def test_a_step_to_that_switches_its_leading_foot_fails_on_step_through_strides(block):
    """Round-2 st_switch4: each foot leads half the time, so each foot's median step is a healthy 0.25 L."""
    walk = gait_trace((0.0, 0.5), duty=0.6, period=0.5, speed=1.0, jitter=0.01, seed=3)
    verdict = judge(_switching_step_to(walk, 0.5, block))
    assert not any(passed(verdict).values())
    assert rails(verdict) == {"stepping/step_through_stride_fraction_min", "stepping/step_symmetry"}
    assert passed(judge(walk)) == {"biped_walk": True, "biped_alternating": True}


def test_a_toe_reach_step_to_fails_on_its_load_weighted_footprint():
    """Round-2 st_toe_reach: a light toe touch 0.055 L ahead, a slide back under light load, then load behind."""
    step_to = gait_trace((0.0, 0.5), duty=0.6, period=0.5, speed=1.0, step_to=True, jitter=0.01)
    for start, end in boolean_runs(step_to["floor_force_n"][:, 1] > 0.0):
        if start == 0:
            continue
        reach = start + int(0.15 * (end - start))
        fraction = (np.arange(start, reach) - start) / max(reach - start, 1)
        step_to["foot_position_m"][start:reach, 1, 0] += 0.055 * (1.0 - fraction)
        step_to["floor_force_n"][start:reach, 1] = 0.12 * BW / 2.0
    assert "stepping/step_length_over_leg_min" in rails(judge(step_to))


def test_stacked_feet_walk_fails():
    trace = gait_trace((0.0, 0.5), duty=0.6, period=0.5, speed=1.0, foot_foot=0.05 * BW)
    verdict = judge(trace)
    assert rails(verdict) == {"support/foot_foot_contact_fraction"}


@pytest.mark.parametrize("lag,qualifies", [(0.28, False), (0.4, True), (0.45, True)])
def test_timing_has_a_wide_tolerance_but_a_skip_still_fails(lag, qualifies):
    trace = gait_trace((0.0, lag), duty=0.6, period=0.5, speed=1.0, jitter=0.01, seed=7)
    assert judge(trace)["biped_walk"][0] is qualifies


# -- invariance (D14) ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "phases,kwargs",
    [
        ((0.0, 0.5), {"duty": 0.6}),
        ((0.0, 0.42), {"duty": (0.5, 0.6)}),
        ((0.0, 0.0), {"duty": 0.4}),
        ((0.0, 0.5), {"duty": (0.5, 0.08), "load": (1.0, 0.03)}),
        ((0.0, 0.5), {"duty": 0.6, "switch": (6.0, (0.0, 0.3))}),
        ((0.0, 0.5), {"duty": 0.6, "step_to": True}),
        (LS_WALK, {"duty": 0.72, "period": 1.2}),
        (DS_WALK, {"duty": 0.65, "period": 1.0}),
        (TROT, {"duty": 0.5, "period": 0.6}),
        (PACE, {"duty": 0.5, "period": 0.6}),
        ((0.0, 0.5, 0.1, 0.6), {"duty": 0.6, "period": 0.6}),
    ],
)
def test_verdicts_are_invariant_to_mirror_and_pair_relabelling(phases, kwargs):
    kwargs = {"period": 0.5, "speed": 1.0, "jitter": 0.01, "stride_cv": 0.03, "seed": 9, **kwargs}
    trace = gait_trace(phases, **kwargs)
    feet = BIPED if len(phases) == 2 else QUAD
    reference = passed(judge(trace, feet))
    assert passed(judge(mirrored(trace), feet)) == reference
    if len(phases) == 4:
        assert passed(judge(fore_hind_relabelled(trace), feet)) == reference


def test_stored_metrics_and_verdicts_are_deterministic():
    trace = gait_trace((0.0, 0.5), duty=0.6, period=0.5, speed=1.0, jitter=0.02, stride_cv=0.03, seed=12)
    first = episode_gait_metrics(
        trace, body_weight_n=BW, leg_length_m=1.0, foot_names=BIPED, protocol=GaitProtocol(), settle_s=SETTLE
    )
    second = episode_gait_metrics(
        {key: value.copy() for key, value in trace.items()},
        body_weight_n=BW,
        leg_length_m=1.0,
        foot_names=BIPED,
        protocol=GaitProtocol(),
        settle_s=SETTLE,
    )
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert judge(trace) == judge(trace)


@pytest.mark.parametrize("degrees", [5.0, -10.0, 20.0, -30.0])
@pytest.mark.parametrize(
    "case,feet,expected",
    [
        ("walk", BIPED, {"biped_walk": True, "biped_alternating": True}),
        ("run", BIPED, {"biped_walk": False, "biped_alternating": True}),
        ("step_to", BIPED, {"biped_walk": False, "biped_alternating": False}),
        ("hop", BIPED, {"biped_walk": False, "biped_alternating": False}),
        ("trot", QUAD, {"quadruped_walk": True}),
        ("ls_walk", QUAD, {"quadruped_walk": True}),
        ("bound", QUAD, {"quadruped_walk": False}),
    ],
)
def test_h1_verdicts_are_invariant_to_heading(case, feet, expected, degrees):
    """Fore-aft rails use the travel and trunk frames, so a rigidly rotated episode keeps its verdict."""
    trace = {
        "walk": lambda: gait_trace((0.0, 0.5), duty=0.6, period=0.5, jitter=0.01, stride_cv=0.03, seed=2),
        "run": lambda: gait_trace((0.0, 0.5), duty=0.35, period=0.4, speed=1.6, jitter=0.01, seed=2),
        "step_to": lambda: gait_trace((0.0, 0.5), duty=0.6, period=0.5, step_to=True),
        "hop": lambda: gait_trace((0.0, 0.0), duty=0.4, period=0.5),
        "trot": lambda: gait_trace(TROT, duty=0.55, period=0.6, jitter=0.01, seed=4),
        "ls_walk": lambda: gait_trace(LS_WALK, duty=0.72, period=1.2, jitter=0.01, seed=4),
        "bound": lambda: gait_trace((0.5, 0.5, 0.0, 0.0), duty=0.45, period=0.6, jitter=0.01, seed=4),
    }[case]()
    # only the progress rail may see the heading: its bar follows the projection on the task axis
    speed = 0.4 * float(np.cos(np.radians(degrees)))
    assert passed(judge(trace, feet, speed=0.4)) == expected
    assert passed(judge(rotated(trace, degrees), feet, speed=speed)) == expected


# -- verification holes (H2-H7) and round-2 reward hacks --------------------------------------


def test_h2_micro_step_shuffle_fails_on_stride_length():
    shuffle = scheduled_trace(lambda t: ((0.0, 0.5), 0.15, 0.6, 0.6), clearance=0.03, seed=1)  # 0.09 L strides
    assert "stepping/stride_length_over_leg_min" in rails(judge(shuffle, speed=0.1))
    walk = scheduled_trace(lambda t: ((0.0, 0.5), 0.25, 0.55, 1.2), clearance=0.05, seed=1)  # 0.3 L strides
    assert judge(walk)["biped_walk"][0]


def test_h3_toe_drag_swing_fails_on_swing_floor_contact():
    drag = scheduled_trace(lambda t: WALK, swing_drag=0.003, flick=0.15)  # 0.3 % BW, under the contact threshold
    assert {"stepping/swing_ground_fraction_max", "stepping/swing_slip_fraction_max"} <= rails(judge(drag))
    assert judge(scheduled_trace(lambda t: WALK))["biped_walk"][0]


@pytest.mark.parametrize("glide,qualifies", [(0.25, True), (0.45, False)])
def test_h4_skating_at_half_trunk_speed_fails_on_skid_and_glide(glide, qualifies):
    verdict = judge(scheduled_trace(lambda t: WALK, slip=glide))
    assert verdict["biped_walk"][0] is qualifies
    if not qualifies:
        assert rails(verdict) == {"support/skid_fraction_max", "support/walk_glide_stance_fraction_max"}
        assert rails(verdict, "biped_alternating") == {"support/skid_fraction_max"}


def test_h5_antalgic_limp_fails_on_pair_load_and_duty_ratios():
    limp = scheduled_trace(lambda t: ((0.0, 0.5), 0.5, (0.64, 0.42), 1.0), load=(1.0, 0.6))
    assert {"participation/pair_load_ratio_min", "participation/pair_duty_ratio_min"} <= rails(judge(limp))


@pytest.mark.parametrize(
    "duty,load",
    [
        ((0.18, 0.18, 0.55, 0.55), (0.3, 0.3, 1.0, 1.0)),  # rearing: the forelimbs only tap
        ((0.55, 0.55, 0.15, 0.15), (1.0, 1.0, 0.3, 0.3)),  # wheelbarrow: the hind limbs only tap
        ((0.13, 0.2, 0.55, 0.55), (0.25, 0.4, 1.0, 1.0)),  # one token forelimb
    ],
)
def test_h5_rearing_wheelbarrow_and_token_limb_quadrupeds_fail_girdle_participation(duty, load):
    verdict = judge(scheduled_trace(lambda t: (TROT, 0.4, duty, 1.0), feet=4, load=load), QUAD)
    assert not verdict["quadruped_walk"][0]
    assert "participation/girdle_duty_ratio" in rails(verdict, "quadruped_walk")


@pytest.mark.parametrize("load", [(1.0, 1.0, 2.5, 2.5), (2.5, 2.5, 1.0, 1.0)])
def test_h5_hind_heavy_and_fore_heavy_genuine_trotting_walks_pass(load):
    trace = scheduled_trace(lambda t: (TROT, 0.5, 0.55, 1.0), feet=4, load=load, jitter=0.01)
    assert passed(judge(trace, QUAD)) == {"quadruped_walk": True}


def test_h7_a_long_stop_between_normal_strides_is_off_gait_time():
    stop = scheduled_trace(_bouts(WALK, (20.0, 4.0, 5.5, ((0.0, 0.5), 1.5, 1.0, 0.0))))
    verdict = judge(stop, speed=0.1)
    assert not verdict["biped_walk"][0]
    assert "persistence/off_gait_fraction" in rails(verdict)


def test_r2_skimming_swing_fails_swing_floor_contact():
    """A foot hovering 0.006 L over the floor with one flick to 0.022 L is a shuffle (round-2 h04)."""
    walk = scheduled_trace(lambda t: WALK)
    skim = {key: np.array(value, copy=True) for key, value in walk.items()}
    for foot in range(2):
        swinging = skim["floor_force_n"][:, foot] == 0.0
        for start, end in boolean_runs(swinging):
            fraction = (np.arange(start, end) - start) / max(end - start, 1)
            height = 0.006 + 0.016 * np.exp(-(((fraction - 0.5) / 0.05) ** 2))
            skim["foot_clearance_m"][start:end, foot] = height
            skim["foot_position_m"][start:end, foot, 2] = height
    verdict = judge(skim)
    assert rails(verdict) == {"stepping/swing_ground_fraction_max"}


def test_r2_skating_bouts_fail_on_glide_while_touchdown_skid_does_not():
    """Skating in a quarter of the stances averages below the whole-window skid bar (round-2 h10)."""
    skating = scheduled_trace(lambda t: WALK + ((0.9,) if t % 2.0 >= 1.5 else (0.0,)), jitter=0.01, seed=2)
    verdict = judge(skating)
    assert rails(verdict) == {"support/glide_stance_fraction_max", "support/walk_glide_stance_fraction_max"}
    assert rails(verdict, "biped_alternating") == {"support/glide_stance_fraction_max"}
    assert judge(scheduled_trace(lambda t: WALK, slip=0.25, jitter=0.01, seed=2))["biped_walk"][0]


def test_walking_feet_skating_at_half_the_trunk_speed_in_a_third_of_the_stances_fail():
    """Walk-first quadruped skate_2of3_047: stances sliding at 0.47 of trunk speed escape the 0.6 glide ratio."""
    skating = scheduled_trace(
        lambda t: (TROT, 0.6, 0.6, 1.0, 0.47 if t % 1.8 < 1.2 else 0.0), feet=4, jitter=0.01, seed=3
    )
    verdict = judge(skating, QUAD)
    assert "support/walk_glide_stance_fraction_max" in rails(verdict, "quadruped_walk")
    assert "support/glide_stance_fraction_max" not in rails(verdict, "quadruped_walk")


def test_r2_a_limp_padded_by_a_hovering_retouch_fails_on_light_stance():
    """The weak foot's stance is padded with 0.4 stance of light touch (round-2 h15)."""
    limp = scheduled_trace(lambda t: ((0.0, 0.5), 0.5, (0.62, 0.62), 1.0), jitter=0.01, seed=3)
    for start, end in boolean_runs(limp["floor_force_n"][:, 1] > 0.0):
        limp["floor_force_n"][start + int(0.6 * (end - start)) : end, 1] = 0.02 * BW
    assert "participation/light_stance_fraction_max" in rails(judge(limp))


def test_r2_light_contacts_cannot_time_a_trot_whose_weight_is_carried_in_pronk_loads():
    """Phantom contacts set the footfall pattern while the load is carried four feet at a time."""
    trot = gait_trace(TROT, duty=0.6, period=0.6, jitter=0.01, seed=4)
    loaded = trot["floor_force_n"] > 0.0
    together = np.all(loaded, axis=1)
    spoof = {key: np.array(value, copy=True) for key, value in trot.items()}
    spoof["floor_force_n"] = np.where(loaded, np.where(together[:, None], BW / 4.0, 0.006 * BW), 0.0)
    verdict = judge(spoof, QUAD)
    assert not verdict["quadruped_walk"][0]
    assert "participation/light_stance_fraction_max" in rails(verdict, "quadruped_walk")
    assert passed(judge(trot, QUAD)) == {"quadruped_walk": True}


def test_r2_rearing_half_the_window_with_phantom_forelimb_taps_fails_girdle_participation():
    trot = gait_trace(TROT, duty=0.55, period=0.6, jitter=0.01, seed=5)
    rearing = {key: np.array(value, copy=True) for key, value in trot.items()}
    up = (rearing["time_s"] % 4.0) < 2.0
    force = rearing["floor_force_n"]
    fore_load = force[:, :2].sum(axis=1)
    hind_loaded = force[:, 2:] > 0.0
    hind_count = np.maximum(hind_loaded.sum(axis=1), 1)
    force[:, 2:] = np.where(up[:, None] & hind_loaded, force[:, 2:] + (fore_load / hind_count)[:, None], force[:, 2:])
    force[:, :2] = np.where(up[:, None] & (force[:, :2] > 0.0), 0.006 * BW, force[:, :2])
    verdict = judge(rearing, QUAD)
    assert not verdict["quadruped_walk"][0]
    assert "participation/girdle_unloaded_fraction" in rails(verdict, "quadruped_walk")


def test_r2_canter_bouts_and_bound_strides_in_a_walk_are_off_gait():
    """Bouts of an asymmetrical gait beyond the budget fail; walk, trot and pace bouts are all one walk."""
    walk = (LS_WALK, 1.0, 0.7, 0.8)
    canter = ((0.5, 0.25, 0.25, 0.0), 1.0, 0.6, 0.8)  # lead fore, then the diagonal pair, then trailing hind
    cantering = scheduled_trace(_bouts(walk, (6.0, 3.5, 5.5, canter)), feet=4, jitter=0.01, seed=6)
    assert "persistence/off_gait_fraction" in rails(judge(cantering, QUAD), "quadruped_walk")
    bound = ((0.0, 0.0, 0.5, 0.5), 1.0, 0.6, 0.8)
    bounding = scheduled_trace(_bouts(walk, (6.0, 3.5, 5.5, bound)), feet=4, jitter=0.01, seed=6)
    assert "persistence/off_gait_fraction" in rails(judge(bounding, QUAD), "quadruped_walk")
    trot = (TROT, 1.0, 0.6, 0.8)
    trotting = scheduled_trace(_bouts(walk, (6.0, 3.0, 6.0, trot)), feet=4, jitter=0.01, seed=6)
    assert passed(judge(trotting, QUAD)) == {"quadruped_walk": True}


def test_r2_a_left_heavy_walk_symmetric_at_mid_stance_passes():
    """Touchdown lag 0.6 with duty 0.75 / 0.55: evenly spaced mid-stances; the duty ratio judges the rest."""
    walk = gait_trace((0.0, 0.6), duty=(0.75, 0.55), period=0.6, jitter=0.01, seed=8, load=(0.85, 1.0))
    measured = episode_gait_metrics(
        walk, body_weight_n=BW, leg_length_m=1.0, foot_names=BIPED, protocol=GaitProtocol(), settle_s=SETTLE
    )
    assert measured["contralateral"]["r|l"]["alternation_phase_offset_touchdown"] > 0.09
    verdict = judge(walk)
    assert passed(verdict) == {"biped_walk": True, "biped_alternating": True}, verdict


def test_a_kneeling_trunk_fails_on_trunk_height():
    walk = gait_trace(LS_WALK, duty=0.7, period=1.0, speed=1.0, jitter=0.01, seed=2)
    kneeling = {key: np.array(value, copy=True) for key, value in walk.items()}
    kneeling["root_position_m"][:, 2] = 0.3
    verdict = judge(kneeling, QUAD)
    assert rails(verdict, "quadruped_walk") == {"support/trunk_height_over_leg_p10"}


def test_trunk_yaw_from_the_quaternion_never_changes_a_walk_verdict():
    """A trunk swinging +-10 deg each stride (yaw wobble) keeps the walk's step-through."""
    walk = gait_trace((0.0, 0.5), duty=0.6, period=0.5, speed=1.0, jitter=0.01, seed=5)
    walk["root_quat_wxyz"] = yaw_quaternion(np.radians(10.0) * np.sin(2.0 * np.pi * walk["time_s"] / 0.5))
    assert passed(judge(walk)) == {"biped_walk": True, "biped_alternating": True}


# -- walk-first round 2: frames, loaded support, girdle coupling, stops ---------------------------


@pytest.mark.parametrize("degrees", [7.0, 12.0, -12.0])
def test_a_step_to_along_the_line_of_travel_fails_whatever_the_trunk_yaw(degrees):
    """Round-2 walk-first attacks: a 0.6 L stance width turned 7-12 deg projects a step onto the trunk axis."""
    step_to = gait_trace((0.0, 0.5), duty=0.6, period=0.5, speed=1.0, step_to=True, jitter=0.01, width=0.6)
    verdict = judge(yawed(step_to, degrees))
    assert not any(passed(verdict).values())
    assert "stepping/step_length_over_leg_min" in rails(verdict)


@pytest.mark.parametrize("degrees", [13.0, -13.0, 20.0])
def test_a_genuine_walk_with_its_trunk_turned_off_its_travel_passes(degrees):
    """The footfalls decide: a symmetric walk along its travel passes whatever its trunk yaw."""
    walk = gait_trace((0.0, 0.5), duty=0.6, period=0.5, speed=1.0, jitter=0.01, seed=2, width=0.6)
    assert passed(judge(yawed(walk, degrees))) == {"biped_walk": True, "biped_alternating": True}


@pytest.mark.parametrize("degrees", [-8.0, -12.0])
def test_a_body_frame_step_to_on_a_crabbing_path_fails_on_its_lopsided_travel_steps(degrees):
    """Round-2 crab attack: the trailing foot lands beside the leader in the body frame; along the crabbing
    path the 0.6 L stance width puts it 0.08-0.12 L through, lopsided against the leader's 0.4 L."""
    step_to = crabbed(
        gait_trace((0.0, 0.5), duty=0.6, period=0.5, speed=1.0, step_to=True, jitter=0.01, width=0.6), degrees
    )
    verdict = judge(step_to, speed=0.3)
    assert not any(passed(verdict).values())
    assert rails(verdict) == {"stepping/body_frame_step_to_symmetry"}
    # a genuine walk on the same crabbing path keeps its even body-frame steps
    walk = crabbed(gait_trace((0.0, 0.5), duty=0.6, period=0.5, speed=1.0, jitter=0.01, width=0.6), degrees)
    assert passed(judge(walk, speed=0.3)) == {"biped_walk": True, "biped_alternating": True}


def _toe_lingering(run, linger, force_bw):
    """After each lift-off the toe stays on the floor for ``linger`` s with ``force_bw`` of body weight."""
    out = {key: np.array(value, copy=True) for key, value in run.items()}
    time = out["time_s"]
    for foot in range(2):
        for _, end in boolean_runs(run["floor_force_n"][:, foot] > 0.0):
            if end >= len(time):
                continue
            tail = slice(end, int(np.searchsorted(time, time[end] + linger)))
            out["floor_force_n"][tail, foot] = force_bw * BW
            out["touch_force_n"][tail, foot] = force_bw * BW
            out["foot_clearance_m"][tail, foot] = 0.0
            out["foot_position_m"][tail, foot] = out["foot_position_m"][end - 1, foot]
    return out


def test_a_run_whose_flight_is_bridged_by_a_light_toe_contact_is_not_a_walk():
    """Walk-first H02: 20 % ballistic flight, a 2.5 % BW toe lingering 6 % of the stride after push-off."""
    run = gait_trace((0.0, 0.5), duty=0.4, period=0.3, speed=1.2, jitter=0.01, seed=3)
    bridged = _toe_lingering(run, 0.06 * 0.3, 0.025)
    verdict = judge(bridged)
    assert passed(verdict) == {"biped_walk": False, "biped_alternating": True}
    assert rails(verdict) == {"support/unloaded_fraction"}


def test_quadruped_girdles_stepping_at_different_cadences_are_not_a_walk():
    """Walk-first quadruped attack: each girdle alternates, the fore pair at twice the hind cadence."""
    fore = gait_trace((0.0, 0.5), duty=0.68, period=0.5, speed=1.0, jitter=0.01, seed=1)
    hind = gait_trace((0.25, 0.75), duty=0.68, period=1.0, speed=1.0, jitter=0.01, seed=2)
    verdict = judge(merged_girdles(fore, hind), QUAD)
    assert not verdict["quadruped_walk"][0]
    assert "persistence/off_gait_fraction" in rails(verdict, "quadruped_walk")
    same = gait_trace((0.25, 0.75), duty=0.68, period=1.0, speed=1.0, jitter=0.01, seed=1)
    assert passed(judge(merged_girdles(same, hind), QUAD)) == {"quadruped_walk": True}


def test_phantom_toe_contacts_cannot_hide_a_flying_trot():
    """Walk-first quadruped attack: 0.03 BW contacts before each touchdown and after each lift-off erase the
    suspensions of a duty-0.3 trot from the contact record (and lift its contact duty to 0.5)."""
    trot = gait_trace(TROT, duty=0.5, period=0.5, speed=1.5, jitter=0.01, seed=4)
    spoof = {key: np.array(value, copy=True) for key, value in trot.items()}
    time = spoof["time_s"]
    for foot in range(4):
        for start, end in boolean_runs(trot["floor_force_n"][:, foot] > 0.0):
            edge = int(np.searchsorted(time, time[start] + 0.05))
            tail = int(np.searchsorted(time, time[min(end, len(time) - 1)] - 0.05))
            spoof["floor_force_n"][start:edge, foot] = 0.03 * BW
            spoof["floor_force_n"][tail:end, foot] = 0.03 * BW
    verdict = judge(spoof, QUAD)
    assert not verdict["quadruped_walk"][0]
    assert "support/unloaded_fraction" in rails(verdict, "quadruped_walk")
    assert passed(judge(trot, QUAD)) == {"quadruped_walk": True}


def test_a_girdle_unloaded_on_alternate_strides_fails_girdle_participation():
    """Walk-first wheelbarrow: every other 1 s stride the hind feet only touch (0.03 BW each)."""
    walk = gait_trace(LS_WALK, duty=0.75, period=1.0, speed=0.8, jitter=0.01, seed=5)
    wheelbarrow = {key: np.array(value, copy=True) for key, value in walk.items()}
    force = wheelbarrow["floor_force_n"]
    odd = (np.floor(wheelbarrow["time_s"] + 2.0) % 2.0) == 1.0
    fore_loaded = force[:, :2] > 0.0
    hind_loaded = force[:, 2:] > 0.0
    lift = odd & np.any(fore_loaded, axis=1)
    hind_total = np.where(hind_loaded, 0.03 * BW, 0.0)
    force[:, 2:] = np.where(lift[:, None], hind_total, force[:, 2:])
    fore_share = (BW - hind_total.sum(axis=1)) / np.maximum(fore_loaded.sum(axis=1), 1)
    force[:, :2] = np.where(lift[:, None] & fore_loaded, fore_share[:, None], force[:, :2])
    verdict = judge(wheelbarrow, QUAD)
    assert not verdict["quadruped_walk"][0]
    assert "participation/girdle_unloaded_fraction" in rails(verdict, "quadruped_walk")


@pytest.mark.parametrize("stop_s,qualifies", [(0.5, True), (1.0, True), (1.6, False)])
def test_a_stop_at_brachiosaurus_cadence_costs_only_its_standing_time(stop_s, qualifies):
    """A stop inside a 2 s lateral-sequence stride: the moving clock stops with the trunk, so the stop is
    charged once (6 % and 11 % of the 9 s window pass) and never stretches its strides into pauses."""
    walk = gait_trace(LS_WALK, duty=0.75, period=2.0, speed=0.8, jitter=0.01, seed=6)
    verdict = judge(paused(walk, 5.0, stop_s), QUAD, speed=0.6)
    assert verdict["quadruped_walk"][0] is qualifies, verdict
    if not qualifies:
        assert "persistence/off_gait_fraction" in rails(verdict, "quadruped_walk")


def test_kneeling_for_two_fifths_of_the_window_fails_on_trunk_height():
    """Walk-first kneel bouts: a median trunk height stays high when the trunk kneels 40 % of the time."""
    walk = gait_trace(LS_WALK, duty=0.7, period=1.0, speed=1.0, jitter=0.01, seed=2)
    kneeling = {key: np.array(value, copy=True) for key, value in walk.items()}
    kneeling["root_position_m"][(kneeling["time_s"] % 5.0) >= 3.0, 2] = 0.3
    assert rails(judge(kneeling, QUAD), "quadruped_walk") == {"support/trunk_height_over_leg_p10"}
