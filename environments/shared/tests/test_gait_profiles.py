"""End-to-end gait profile verdicts: measured synthetic footfall patterns -> stored metrics -> gate.

Each case renders a footfall pattern (``gait_trace``), measures it at 2 ms
(or 10 ms), round-trips the metrics through JSON exactly as a report stores
them, and judges them under the calibrated ``provisional_gait_criteria``. The
cases are the defects a gait certificate must not have (D1-D14 of the 2026-10
gait-checker review and the holes H1-H8 its verification found in the first
proposal) plus the genuine gaits it must not reject.
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
from environments.shared.gait.metrics import episode_gait_metrics
from environments.shared.gait.types import GaitProtocol

from .test_gait_metrics import BW, gait_trace, mirrored, rotated, scheduled_trace

BIPED = ("r", "l")
QUAD = ("fr", "fl", "rr", "rl")
SETTLE = 1.0
LS_WALK = (0.25, 0.75, 0.0, 0.5)  # fore lags the ipsilateral hind by 1/4 cycle
DS_WALK = (0.75, 0.25, 0.0, 0.5)  # ... by 3/4 cycle
TROT = (0.0, 0.5, 0.5, 0.0)
PACE = (0.0, 0.5, 0.0, 0.5)


def judge(trace, feet=BIPED, *, profiles=None, speed=0.4, duration=None):
    """Profile -> (qualified, reasons) from the JSON-stored metrics of ``trace`` (task direction +x)."""
    measured = episode_gait_metrics(
        trace, body_weight_n=BW, leg_length_m=1.0, foot_names=feet, protocol=GaitProtocol(), settle_s=SETTLE
    )
    stored = json.loads(json.dumps(measured, allow_nan=False))
    stored["completed_horizon"] = True
    if duration is None:
        duration = round(float(trace["time_s"][-1] - trace["time_s"][0]), 6) - SETTLE
    profiles = profiles or (
        ("biped_alternating",) if len(feet) == 2 else ("quadruped_walk", "quadruped_trot", "quadruped_pace")
    )
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


def rails(verdicts, profile="biped_alternating"):
    return {rail_id(reason) for reason in verdicts[profile][1]}


def fore_hind_relabelled(trace):
    """Swap the fore and hind pair labels (limb phase becomes 1 - limb phase)."""
    out = {key: np.array(value, copy=True) for key, value in trace.items()}
    for key in ("floor_force_n", "foot_position_m", "foot_clearance_m", "slip_speed_mps", "touch_force_n"):
        out[key] = out[key][:, [2, 3, 0, 1]]
    return out


# -- genuine gaits pass (D1, D11, D12, D14) ------------------------------------------


@pytest.mark.parametrize("jitter", [0.0, 0.01, 0.03])
@pytest.mark.parametrize("duty", [0.62, 0.45, 0.3])
@pytest.mark.parametrize("seed", [0, 1])
def test_jittered_alternating_biped_walks_and_runs_pass(duty, jitter, seed):
    trace = gait_trace((0.0, 0.5), duty=duty, period=0.5, speed=1.0, jitter=jitter, stride_cv=0.03, seed=seed)
    verdict = judge(trace)
    assert verdict["biped_alternating"][0], verdict


@pytest.mark.parametrize(
    "phases,duty,period,profile",
    [
        (LS_WALK, 0.72, 1.2, "quadruped_walk"),
        (DS_WALK, 0.65, 1.0, "quadruped_walk"),
        (TROT, 0.5, 0.6, "quadruped_trot"),
        (TROT, 0.35, 0.5, "quadruped_trot"),  # flying trot is a trot
        (PACE, 0.5, 0.6, "quadruped_pace"),
    ],
)
@pytest.mark.parametrize("jitter", [0.01, 0.02])
def test_jittered_quadruped_gaits_pass_only_their_own_profile(phases, duty, period, profile, jitter):
    trace = gait_trace(phases, duty=duty, period=period, speed=1.0, jitter=jitter, stride_cv=0.03, seed=4)
    verdict = passed(judge(trace, QUAD))
    assert verdict == {name: name == profile for name in verdict}


def test_touchdown_skid_is_tolerated_but_a_skidding_scramble_is_not():
    assert judge(gait_trace((0.0, 0.5), duty=0.35, period=0.3, speed=2.0, slip=0.6))["biped_alternating"][0]
    verdict = judge(gait_trace((0.0, 0.5), duty=0.35, period=0.3, speed=2.0, slip=1.6))
    assert "support/skid_fraction_max" in rails(verdict)


def test_long_stride_duty_half_trot_with_suspension_is_not_vetoed_by_absolute_flight_time():
    trace = gait_trace(TROT, duty=0.45, period=1.2, speed=1.5, jitter=0.01, seed=2)
    assert passed(judge(trace, QUAD))["quadruped_trot"]


@pytest.mark.parametrize("dt", [0.002, 0.01])
def test_sampling_at_10_ms_keeps_the_verdict(dt):
    biped = gait_trace((0.0, 0.5), duty=0.6, period=0.5, dt=dt, jitter=0.01, stride_cv=0.03, seed=5)
    assert judge(biped)["biped_alternating"][0]
    quad = gait_trace(LS_WALK, duty=0.72, period=1.2, dt=dt, jitter=0.01, stride_cv=0.03, seed=5)
    assert passed(judge(quad, QUAD)) == {"quadruped_walk": True, "quadruped_trot": False, "quadruped_pace": False}


def test_float_accumulated_clock_meets_the_declared_duration():
    trace = gait_trace((0.0, 0.5), duty=0.6, period=0.5, duration=20.0, accumulate_time=True)
    assert trace["time_s"][-1] < 20.0  # 10000 additions of 0.002 end short of the horizon
    verdict = judge(trace, duration=19.0)
    assert verdict["biped_alternating"][0], verdict


# -- quadruped profile boundaries (D3) -------------------------------------------------


@pytest.mark.parametrize("dissociation_s", [0.025, 0.04, 0.05])
def test_dissociated_trot_and_pace_are_never_walks(dissociation_s):
    period = 0.6
    lead = dissociation_s / period  # hind feet land this much earlier
    trot = (0.0, 0.5, 0.5 - lead, -lead)
    pace = (0.0, 0.5, -lead, 0.5 - lead)
    for phases, profile in ((trot, "quadruped_trot"), (pace, "quadruped_pace")):
        trace = gait_trace(phases, duty=0.55, period=period, speed=1.0, jitter=0.01, seed=3)
        verdict = passed(judge(trace, QUAD))
        assert verdict == {name: name == profile for name in verdict}, (dissociation_s, profile, verdict)


# -- limb participation (D4) -----------------------------------------------------------


def test_one_leg_hop_with_a_token_tapping_foot_fails():
    trace = gait_trace((0.0, 0.5), duty=(0.5, 0.08), period=0.5, speed=1.0, load=(1.0, 0.03))
    verdict = judge(trace)
    assert not verdict["biped_alternating"][0]
    assert {"participation/limb_duty_min", "participation/relative_limb_load_share_min"} <= rails(verdict)


def test_three_legged_walk_with_a_tapping_limb_fails_every_profile():
    trace = gait_trace(LS_WALK, duty=(0.72, 0.08, 0.72, 0.72), period=1.2, speed=1.0, load=(1.0, 0.03, 1.0, 1.0))
    verdict = judge(trace, QUAD)
    assert not any(passed(verdict).values())
    for profile in verdict:
        assert "participation/relative_limb_load_share_min" in rails(verdict, profile)


# -- persistence (D5) -------------------------------------------------------------------


@pytest.mark.parametrize("flip", [False, True])
def test_walk_then_hop_fails_and_so_does_its_mirror(flip):
    trace = gait_trace((0.0, 0.5), duty=0.6, period=0.5, speed=1.0, jitter=0.01, switch=(7.3, (0.0, 0.0)))
    verdict = judge(mirrored(trace) if flip else trace)
    assert not verdict["biped_alternating"][0]
    assert "persistence/off_gait_fraction" in rails(verdict)


def test_walk_then_stand_fails_persistence():
    trace = gait_trace((0.0, 0.5), duty=0.6, period=0.5, speed=1.0, jitter=0.01, seed=6)
    still = trace["time_s"] >= 8.0
    trace["floor_force_n"][still] = BW / 2.0
    trace["foot_clearance_m"][still] = 0.0
    trace["foot_position_m"][still] = trace["foot_position_m"][np.argmax(still)]
    trace["root_position_m"][still, 0] = trace["root_position_m"][np.argmax(still), 0]
    verdict = judge(trace, speed=0.1)
    assert not verdict["biped_alternating"][0]
    assert "persistence/off_gait_fraction" in rails(verdict)


# -- swing validity, lead exchange, stacking, timing symmetry (D6, D7, D9, D10) ---------


@pytest.mark.parametrize("clearance", [0.004, 0.015])
def test_low_clearance_shuffle_fails(clearance):
    trace = gait_trace((0.0, 0.5), duty=0.6, period=0.5, speed=0.3, clearance=clearance)
    verdict = judge(trace, speed=0.1)
    assert not verdict["biped_alternating"][0]
    assert rails(verdict) & {"stepping/median_swing_clearance_over_leg_min", "stepping/valid_swing_fraction_min"}


def test_step_to_gait_fails_on_lead_exchange():
    trace = gait_trace((0.0, 0.5), duty=0.6, period=0.5, speed=1.0, step_to=True, jitter=0.01)
    verdict = judge(trace)
    assert not verdict["biped_alternating"][0]
    assert "stepping/lead_exchange_fraction_min" in rails(verdict)


def test_stacked_feet_walk_fails():
    trace = gait_trace((0.0, 0.5), duty=0.6, period=0.5, speed=1.0, foot_foot=0.05 * BW)
    verdict = judge(trace)
    assert rails(verdict) == {"support/foot_foot_contact_fraction"}


@pytest.mark.parametrize("lag,qualifies", [(0.35, False), (0.4, False), (0.45, True)])
def test_staggered_timing_beyond_the_symmetry_band_fails(lag, qualifies):
    trace = gait_trace((0.0, lag), duty=0.6, period=0.5, speed=1.0, jitter=0.01, seed=7)
    assert judge(trace)["biped_alternating"][0] is qualifies


# -- invariance (D14) ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "phases,kwargs",
    [
        ((0.0, 0.5), {"duty": 0.6}),
        ((0.0, 0.42), {"duty": (0.5, 0.6)}),
        ((0.0, 0.0), {"duty": 0.4}),
        ((0.0, 0.5), {"duty": (0.5, 0.08), "load": (1.0, 0.03)}),
        ((0.0, 0.5), {"duty": 0.6, "switch": (6.0, (0.0, 0.3))}),
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


# -- verification holes of the first proposal (H1-H8) ------------------------------------
WALK = ((0.0, 0.5), 0.5, 0.6, 1.0)  # phases, period, duty, speed of one stride


def _bouts(base, *bouts):
    """Stride plan: ``(every_s, from_s, to_s, stride)`` bouts replace the ``base`` stride."""

    def plan(start):
        for every, begin, end, stride in bouts:
            if begin <= start % every < end:
                return stride
        return base

    return plan


@pytest.mark.parametrize("degrees", [5.0, -10.0, 20.0, -30.0])
@pytest.mark.parametrize(
    "case,feet,expected",
    [
        ("walk", BIPED, {"biped_alternating": True}),
        ("run", BIPED, {"biped_alternating": True}),
        ("step_to", BIPED, {"biped_alternating": False}),
        ("hop", BIPED, {"biped_alternating": False}),
        ("trot", QUAD, {"quadruped_walk": False, "quadruped_trot": True, "quadruped_pace": False}),
        ("ls_walk", QUAD, {"quadruped_walk": True, "quadruped_trot": False, "quadruped_pace": False}),
    ],
)
def test_h1_verdicts_are_invariant_to_heading(case, feet, expected, degrees):
    """H1: fore-aft rails use the local travel frame, so a rigidly rotated episode keeps its verdict."""
    trace = {
        "walk": lambda: gait_trace((0.0, 0.5), duty=0.6, period=0.5, jitter=0.01, stride_cv=0.03, seed=2),
        "run": lambda: gait_trace((0.0, 0.5), duty=0.35, period=0.4, speed=1.6, jitter=0.01, seed=2),
        "step_to": lambda: gait_trace((0.0, 0.5), duty=0.6, period=0.5, step_to=True),
        "hop": lambda: gait_trace((0.0, 0.0), duty=0.4, period=0.5),
        "trot": lambda: gait_trace(TROT, duty=0.5, period=0.6, jitter=0.01, seed=4),
        "ls_walk": lambda: gait_trace(LS_WALK, duty=0.72, period=1.2, jitter=0.01, seed=4),
    }[case]()
    # only the progress rail may see the heading: its bar follows the projection on the task axis
    speed = 0.4 * float(np.cos(np.radians(degrees)))
    assert passed(judge(trace, feet, speed=0.4)) == expected
    assert passed(judge(rotated(trace, degrees), feet, speed=speed)) == expected


@pytest.mark.parametrize("degrees", [1.5, 2.0, -3.0])
def test_h1_exact_step_to_a_few_degrees_off_the_task_axis_still_fails(degrees):
    trace = rotated(gait_trace((0.0, 0.5), duty=0.6, period=0.5, step_to=True), degrees)
    assert "stepping/lead_exchange_fraction_min" in rails(judge(trace))


def test_h2_micro_step_shuffle_fails_on_stride_length():
    shuffle = scheduled_trace(lambda t: ((0.0, 0.5), 0.15, 0.6, 0.6), clearance=0.03, seed=1)  # 0.09 L strides
    assert rails(judge(shuffle, speed=0.1)) == {"stepping/stride_length_over_leg_min"}
    walk = scheduled_trace(lambda t: ((0.0, 0.5), 0.25, 0.55, 1.2), clearance=0.05, seed=1)  # 0.3 L strides
    assert judge(walk)["biped_alternating"][0]


def test_h3_toe_drag_swing_fails_on_swing_floor_contact():
    drag = scheduled_trace(lambda t: WALK, swing_drag=0.003, flick=0.15)  # 0.3 % BW, under the contact threshold
    assert {"stepping/swing_ground_fraction_max", "stepping/swing_slip_fraction_max"} <= rails(judge(drag))
    assert judge(scheduled_trace(lambda t: WALK))["biped_alternating"][0]


@pytest.mark.parametrize("glide,qualifies", [(0.25, True), (0.45, False)])
def test_h4_skating_at_half_trunk_speed_fails_on_skid(glide, qualifies):
    verdict = judge(scheduled_trace(lambda t: WALK, slip=glide))
    assert verdict["biped_alternating"][0] is qualifies
    if not qualifies:
        assert rails(verdict) == {"support/skid_fraction_max"}


def test_h5_antalgic_limp_fails_on_pair_load_and_duty_ratios():
    limp = scheduled_trace(lambda t: ((0.0, 0.5), 0.5, (0.64, 0.42), 1.0), load=(1.0, 0.6))
    assert {"participation/pair_load_ratio_min", "participation/pair_duty_ratio_min"} <= rails(judge(limp))


@pytest.mark.parametrize(
    "duty,load",
    [
        ((0.18, 0.18, 0.5, 0.5), (0.3, 0.3, 1.0, 1.0)),  # rearing: the forelimbs only tap
        ((0.5, 0.5, 0.15, 0.15), (1.0, 1.0, 0.3, 0.3)),  # wheelbarrow: the hind limbs only tap
        ((0.13, 0.2, 0.5, 0.5), (0.25, 0.4, 1.0, 1.0)),  # one token forelimb
    ],
)
def test_h5_rearing_wheelbarrow_and_token_limb_quadrupeds_fail_girdle_participation(duty, load):
    verdict = judge(scheduled_trace(lambda t: (TROT, 0.4, duty, 1.0), feet=4, load=load), QUAD)
    assert not any(passed(verdict).values())
    assert "participation/girdle_duty_ratio" in rails(verdict, "quadruped_trot")


@pytest.mark.parametrize("load", [(1.0, 1.0, 2.5, 2.5), (2.5, 2.5, 1.0, 1.0)])
def test_h5_hind_heavy_and_fore_heavy_genuine_trots_pass(load):
    trace = scheduled_trace(lambda t: (TROT, 0.5, 0.55, 1.0), feet=4, load=load, jitter=0.01)
    assert passed(judge(trace, QUAD))["quadruped_trot"]


@pytest.mark.parametrize("bout_s", [0.6, 0.5])
def test_h6_hop_bouts_fail_the_off_gait_budget(bout_s):
    hop = ((0.0, 0.0), 0.25, 0.4, 1.0)
    trace = scheduled_trace(_bouts(WALK, (4.0, 4.0 - bout_s, 4.0, hop)), jitter=0.01, seed=3)
    assert "persistence/off_gait_fraction" in rails(judge(trace))


@pytest.mark.parametrize("fraction,most_accepted", [(0.40, 0), (0.28, 4)])
def test_h6_irregular_scramble_bouts_fail(fraction, most_accepted):
    """Uncoordinated bouts (random per-foot timing at twice the cadence) cost coverage, locking and off-gait time.

    At 40 % of the window every episode fails. At 28 % about half do, far from the
    37/40 a panel needs: a scramble that never locks near a competing gait is the
    documented residual of the off-gait budget (docs/GAIT_CERTIFICATION.md).
    """
    accepted = 0
    for seed in range(8):
        rng = np.random.default_rng(seed)
        jumbled = [
            (
                (float(rng.uniform()), float(rng.uniform())),
                float(rng.uniform(0.2, 0.35)),
                float(rng.uniform(0.3, 0.7)),
                1.0,
            )
            for _ in range(64)
        ]

        def scramble(start, jumbled=jumbled):
            return jumbled[int(start * 13.0) % len(jumbled)] if start % 5.0 >= 5.0 * (1.0 - fraction) else WALK

        accepted += judge(scheduled_trace(scramble, seed=seed))["biped_alternating"][0]
    assert accepted <= most_accepted


def test_h6_a_walk_forgives_one_mistimed_footfall_but_a_biped_hop_bout_is_off_gait():
    base = (LS_WALK, 1.5, 0.75, 0.8)
    late_fore = ((0.42, 0.75, 0.0, 0.5), 1.5, 0.75, 0.8)  # one stride's limb phase drifts toward trot
    walk = scheduled_trace(_bouts(base, (20.0, 4.5, 6.0, late_fore)), feet=4)
    verdict = judge(walk, QUAD)
    assert passed(verdict)["quadruped_walk"], verdict["quadruped_walk"]
    hop = ((0.0, 0.0), 0.75, 0.4, 1.0)
    biped = scheduled_trace(_bouts(((0.0, 0.5), 0.75, 0.6, 1.0), (20.0, 4.5, 6.0, hop)))  # a two-stride hop bout
    assert "persistence/off_gait_fraction" in rails(judge(biped))


@pytest.mark.parametrize("switch_s", [3.0, 4.5])
def test_h7_walk_to_run_transition_is_not_a_pause(switch_s):
    def plan(start):
        return ((0.0, 0.5), 0.95, 0.62, 1.3) if start < switch_s else ((0.0, 0.5), 0.42, 0.35, 3.2)

    verdict = judge(scheduled_trace(plan, jitter=0.01, seed=2), speed=1.0)
    assert verdict["biped_alternating"][0], verdict


def test_h7_a_stop_between_normal_strides_is_still_a_pause():
    stop = scheduled_trace(_bouts(WALK, (20.0, 4.0, 4.5, ((0.0, 0.5), 1.5, 1.0, 0.0))))
    verdict = judge(stop, speed=0.1)
    assert not verdict["biped_alternating"][0]
    assert "persistence/off_gait_fraction" in rails(verdict)
