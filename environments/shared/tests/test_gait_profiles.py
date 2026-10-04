"""End-to-end gait profile verdicts: measured synthetic footfall patterns -> stored metrics -> gate.

Each case renders a footfall pattern (``gait_trace``), measures it at 2 ms
(or 10 ms), round-trips the metrics through JSON exactly as a report stores
them, and judges them under the calibrated ``provisional_gait_criteria``. The
cases are the defects a gait certificate must not have (D1-D14 of the 2026-10
gait-checker review) plus the genuine gaits it must not reject.
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

from .test_gait_metrics import BW, gait_trace, mirrored

BIPED = ("r", "l")
QUAD = ("fr", "fl", "rr", "rl")
SETTLE = 1.0
LS_WALK = (0.25, 0.75, 0.0, 0.5)  # fore lags the ipsilateral hind by 1/4 cycle
DS_WALK = (0.75, 0.25, 0.0, 0.5)  # ... by 3/4 cycle
TROT = (0.0, 0.5, 0.5, 0.0)
PACE = (0.0, 0.5, 0.0, 0.5)


def judge(trace, feet=BIPED, *, profiles=None, speed=0.4, duration=None):
    """Profile -> (qualified, reasons) from the JSON-stored metrics of ``trace``."""
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
    assert rails(verdict) & {"persistence/longest_off_template", "persistence/template_coverage"}


def test_walk_then_stand_fails_persistence():
    trace = gait_trace((0.0, 0.5), duty=0.6, period=0.5, speed=1.0, jitter=0.01, seed=6)
    still = trace["time_s"] >= 8.0
    trace["floor_force_n"][still] = BW / 2.0
    trace["foot_clearance_m"][still] = 0.0
    trace["foot_position_m"][still] = trace["foot_position_m"][np.argmax(still)]
    trace["root_position_m"][still, 0] = trace["root_position_m"][np.argmax(still), 0]
    verdict = judge(trace, speed=0.1)
    assert not verdict["biped_alternating"][0]
    assert "persistence/longest_off_template" in rails(verdict)


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
