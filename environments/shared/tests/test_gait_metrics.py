"""Adversarial finite-window fixtures for the pure gait measurement layer."""

import json
from dataclasses import replace

import numpy as np
import pytest

from environments.shared.gait.events import boolean_runs, circular_distance, limb_phase, segment_limb
from environments.shared.gait.metrics import episode_gait_metrics
from environments.shared.gait.types import GaitProtocol, measurement_protocol_sha256

BW = 981.0
PERIOD = 0.8
PROTOCOL = GaitProtocol()
BIPED = ("r", "l")
QUAD = ("fr", "fl", "rr", "rl")


def stepping_trace(phases=(0.0, 0.5), duty=0.625, dt=0.01, duration=4.8, slip=0.0, speed=0.8):
    """Continuous periodic feet evaluated at any requested physics sample rate."""
    time = np.arange(round(duration / dt) + 1, dtype=float) * dt
    feet = len(phases)
    force = np.zeros((len(time), feet))
    position = np.zeros((len(time), feet, 3))
    clearance = np.zeros((len(time), feet))
    for index, phase in enumerate(phases):
        offset = 0.2 + phase * PERIOD
        coordinate = (time - offset) / PERIOD
        cycle = np.floor(coordinate + 1e-10)
        fraction = coordinate - cycle
        fraction[np.abs(fraction) < 1e-10] = 0.0
        loaded = fraction < duty - 1e-10
        force[:, index] = loaded * BW
        swing = np.clip((fraction - duty) / (1.0 - duty), 0.0, 1.0)
        # Each foot lands where its own phase puts it: alternating feet pass each other.
        position[:, index, 0] = (cycle + swing + phase) * speed * PERIOD
        position[:, index, 1] = (-1.0 if index % 2 else 1.0) * 0.1
        clearance[:, index] = np.where(loaded, 0.0, 0.05 * np.sin(np.pi * swing))
        position[:, index, 2] = clearance[:, index]
    loaded_count = np.maximum(1, np.sum(force > 0.0, axis=1))
    force /= loaded_count[:, None]
    root = np.zeros((len(time), 3))
    root[:, 0] = time * speed
    root[:, 2] = 1.0
    return {
        "time_s": time,
        "floor_force_n": force,
        "foot_position_m": position,
        "foot_clearance_m": clearance,
        "slip_speed_mps": (force > 0.0) * slip,
        "body_floor_force_n": np.zeros(len(time)),
        "foot_foot_force_n": np.zeros(len(time)),
        "root_position_m": root,
        "touch_force_n": force.copy(),
    }


def gait_trace(
    phases,
    *,
    duty=0.6,
    period=0.8,
    duration=10.0,
    dt=0.002,
    speed=0.8,
    clearance=0.05,
    jitter=0.0,
    stride_cv=0.0,
    seed=0,
    load=None,
    switch=None,
    step_to=False,
    slip=0.0,
    foot_foot=0.0,
    accumulate_time=False,
):
    """Footfall-pattern puppet: a shared stride clock, per-foot phase/duty/load.

    ``phases`` are touchdown offsets in cycles; ``jitter`` is the touchdown SD
    in cycles and ``stride_cv`` the stride-period coefficient of variation.
    ``load`` weights share body weight among loaded feet; a foot loaded alone
    carries ``min(load, 1)`` body weights (a token tap stays light).
    ``switch=(t, phases)`` changes the pattern for strides starting after
    ``t``. ``step_to`` places the second foot level with the first instead of
    passing it. Feet land half a stance travel ahead of the hip.
    """
    rng = np.random.default_rng(seed)
    feet = len(phases)
    duties = np.broadcast_to(np.asarray(duty, dtype=float), (feet,))
    apex = np.broadcast_to(np.asarray(clearance, dtype=float), (feet,))
    loads = np.ones(feet) if load is None else np.asarray(load, dtype=float)
    starts = [-2.0 * period]
    while starts[-1] < duration + 2.0 * period:
        starts.append(starts[-1] + period * (1.0 + stride_cv * rng.standard_normal()))
    stances: list[list[tuple[float, float]]] = [[] for _ in range(feet)]
    for start, stop in zip(starts[:-1], starts[1:]):
        stride = stop - start
        pattern = phases if switch is None or start < switch[0] else switch[1]
        for foot in range(feet):
            touchdown = start + (pattern[foot] % 1.0) * stride + jitter * stride * rng.standard_normal()
            stances[foot].append((touchdown, touchdown + duties[foot] * stride))
    for foot in range(feet):
        ordered = sorted(stances[foot])
        stances[foot] = [
            (td, min(lo, ordered[k + 1][0] - 0.04) if k + 1 < len(ordered) else lo)
            for k, (td, lo) in enumerate(ordered)
        ]
    if accumulate_time:
        time = np.concatenate(([0.0], np.cumsum(np.full(round(duration / dt), dt))))
    else:
        time = np.arange(round(duration / dt) + 1, dtype=float) * dt
    count = len(time)
    loaded = np.zeros((count, feet), dtype=bool)
    position = np.zeros((count, feet, 3))
    height = np.zeros((count, feet))
    placed: list[list[float]] = []
    for foot in range(feet):
        lateral = (-1.0 if foot % 2 else 1.0) * 0.1
        fore = 0.6 if feet == 4 and foot < 2 else 0.0
        placements = []
        for td, lo in stances[foot]:
            x = speed * td + 0.5 * speed * (lo - td) + fore
            if step_to and foot == 1:
                lead = [p for (t0, _), p in zip(stances[0], placed[0]) if t0 <= td]
                x = lead[-1] if lead else x
            placements.append(x)
        placed.append(placements)
        x = np.full(count, placements[0])
        for k, (td, lo) in enumerate(stances[foot]):
            loaded[(time >= td) & (time < lo), foot] = True
            x[time >= td] = placements[k]
            if k + 1 < len(stances[foot]):
                nxt_td = stances[foot][k + 1][0]
                swing = (time >= lo) & (time < nxt_td)
                fraction = (time[swing] - lo) / (nxt_td - lo)
                x[swing] = placements[k] + (placements[k + 1] - placements[k]) * (3 * fraction**2 - 2 * fraction**3)
                height[swing, foot] = apex[foot] * np.sin(np.pi * fraction)
        position[:, foot, 0] = x
        position[:, foot, 1] = lateral
        position[:, foot, 2] = height[:, foot]
    weighted = loaded * loads
    force = BW * weighted / np.maximum(np.sum(weighted, axis=1, keepdims=True), 1.0)
    root = np.zeros((count, 3))
    root[:, 0] = speed * time
    root[:, 2] = 1.0
    return {
        "time_s": time,
        "floor_force_n": force,
        "foot_position_m": position,
        "foot_clearance_m": np.where(loaded, 0.0, height),
        "slip_speed_mps": loaded * slip,
        "body_floor_force_n": np.zeros(count),
        "foot_foot_force_n": np.full(count, foot_foot),
        "root_position_m": root,
        "touch_force_n": force.copy(),
    }


def mirrored(trace):
    """Left/right mirror image with the canonical foot order kept."""
    feet = trace["floor_force_n"].shape[1]
    order = [1, 0] if feet == 2 else [1, 0, 3, 2]
    out = {key: np.array(value, copy=True) for key, value in trace.items()}
    for key in ("floor_force_n", "foot_position_m", "foot_clearance_m", "slip_speed_mps", "touch_force_n"):
        out[key] = out[key][:, order]
    out["foot_position_m"][..., 1] *= -1.0
    out["root_position_m"][..., 1] *= -1.0
    return out


def metrics(trace, foot_names=BIPED, **kwargs):
    return episode_gait_metrics(
        trace,
        body_weight_n=BW,
        leg_length_m=1.0,
        foot_names=foot_names,
        protocol=kwargs.pop("protocol", PROTOCOL),
        **kwargs,
    )


def test_alternating_walk_has_cycles_support_and_locked_antiphase():
    result = metrics(stepping_trace())
    assert result["telemetry_valid"]
    assert result["duration_s"] == pytest.approx(4.8)
    assert result["mean_speed_mps"] == pytest.approx(0.8)
    assert result["flight_fraction"] == 0.0
    assert result["support_count_fraction"]["2"] == pytest.approx(0.25, abs=1e-3)
    assert result["skid_fraction_max"] == 0.0
    pair = result["pair_phase"]["r>l"]
    assert pair["mean_phase"] == pytest.approx(0.5, abs=1e-6)
    assert pair["locking"] == pytest.approx(1.0)
    template = result["templates"]["biped_alternating"]
    assert template["alternation_phase_offset_max"] == pytest.approx(0.0, abs=1e-6)
    assert template["template_coverage"] == pytest.approx(1.0)
    assert template["alternating_overlap_index_max"] == pytest.approx(0.0, abs=1e-6)
    for foot in result["per_foot"].values():
        assert foot["duty_factor"] == pytest.approx(0.625, abs=0.01)
        assert foot["valid_swing_fraction"] == 1.0
        assert foot["median_swing_clearance_over_leg"] == pytest.approx(0.05, abs=1e-3)
    assert result["contralateral"]["r|l"]["lead_exchange_fraction"] == 1.0
    assert result["gait_label"].startswith("walk")
    # JSON never contains non-standard NaN tokens, including unavailable fields.
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("dt", [0.005, 0.01, 0.02])
def test_elapsed_time_metrics_are_sampling_rate_invariant(dt):
    baseline = metrics(stepping_trace(dt=0.005, slip=0.1))
    result = metrics(stepping_trace(dt=dt, slip=0.1))
    for name in ("duration_s", "mean_speed_mps", "limb_duty_min", "skid_fraction_max"):
        assert result[name] == pytest.approx(baseline[name], abs=1e-6)
    for name in ("alternation_phase_offset_max", "template_coverage", "phase_locking_min"):
        assert result["templates"]["biped_alternating"][name] == pytest.approx(
            baseline["templates"]["biped_alternating"][name], abs=0.02
        )


def test_aerial_alternating_run_is_measured_without_a_blanket_flight_veto():
    result = metrics(stepping_trace(duty=0.375))
    assert result["flight_fraction"] == pytest.approx(0.25, abs=0.01)
    assert result["max_flight_s"] == pytest.approx(0.1, abs=0.011)
    assert result["templates"]["biped_alternating"]["alternation_phase_offset_max"] < 0.01
    assert result["hop_flight_fraction"] == pytest.approx(0.0, abs=1e-6)
    assert result["gait_label"].startswith("aerial_run")


@pytest.mark.parametrize("duty", [0.375, 0.625])
def test_synchronous_hop_is_far_from_alternation_by_independent_statistics(duty):
    result = metrics(stepping_trace(phases=(0.0, 0.0), duty=duty))
    template = result["templates"]["biped_alternating"]
    assert template["alternation_phase_offset_max"] == pytest.approx(0.5, abs=1e-6)
    assert template["alternating_overlap_index_max"] == pytest.approx(1.0)
    assert template["template_coverage"] == 0.0
    assert result["contralateral"]["r|l"]["lead_exchange_fraction"] == 0.0
    assert result["gait_label"].startswith("hop")


def test_staggered_hop_and_circular_phase_wrap_do_not_average_to_alternation():
    result = metrics(stepping_trace(phases=(0.0, 0.025), duty=0.375))
    assert result["pair_phase"]["r>l"]["mean_phase"] == pytest.approx(0.025, abs=1e-3)
    assert result["templates"]["biped_alternating"]["alternation_phase_offset_max"] == pytest.approx(0.475, abs=1e-3)
    np.testing.assert_allclose(circular_distance([0.99, 0.01], 0.0), [0.01, 0.01])
    np.testing.assert_allclose(circular_distance([0.99, 0.01], 0.5), [0.49, 0.49])


def test_phase_zero_jitter_keeps_a_synchronous_pair_synchronous():
    trace = gait_trace((0.0, 0.0), duty=0.4, jitter=0.02, seed=3)
    result = metrics(trace, settle_s=1.0)
    pair = result["pair_phase"]["r>l"]
    assert pair["locking"] > 0.9
    assert min(pair["mean_phase"], 1.0 - pair["mean_phase"]) < 0.02


def test_unaligned_sample_grid_has_only_time_resolution_error():
    result = metrics(stepping_trace(duty=0.375, dt=0.007))
    assert result["flight_fraction"] == pytest.approx(0.25, abs=0.014 / PERIOD)
    assert result["templates"]["biped_alternating"]["alternation_phase_offset_max"] < 0.01
    assert result["per_foot"]["r"]["complete_cycles"] == 5


def test_skid_is_stance_slip_over_trunk_travel_and_unloaded_motion_is_not_slip():
    trace = stepping_trace(slip=0.4)
    result = metrics(trace)
    assert result["skid_fraction_max"] == pytest.approx(0.5, abs=0.02)
    trace["slip_speed_mps"] = np.where(trace["floor_force_n"] > 0.0, 0.0, 100.0)
    assert metrics(trace)["skid_fraction_max"] == 0.0


def test_body_support_is_an_impulse_share_and_foot_contact_a_time_share():
    trace = stepping_trace()
    mask = (trace["time_s"] >= 1.0) & (trace["time_s"] < 1.48)
    trace["body_floor_force_n"][mask] = BW * 0.1
    trace["foot_foot_force_n"][mask] = BW * 0.1
    originals = {name: array.copy() for name, array in trace.items()}
    result = metrics(trace)
    assert result["body_support_fraction"] == pytest.approx(0.01 / 1.01, abs=1e-3)
    assert result["foot_foot_contact_fraction"] == pytest.approx(0.1, abs=1e-3)
    for name in trace:
        np.testing.assert_array_equal(trace[name], originals[name])


def test_standing_has_no_steps_and_undefined_phase():
    trace = stepping_trace()
    trace["floor_force_n"][:] = BW / 2.0
    trace["foot_position_m"][:] = 0.0
    trace["foot_clearance_m"][:] = 0.0
    trace["root_position_m"][:, 0] = 0.0
    result = metrics(trace)
    assert result["flight_fraction"] == 0.0
    assert result["mean_speed_mps"] == 0.0
    assert result["complete_cycles_min"] == 0
    assert result["limb_phase_coverage_min"] == 0.0
    assert result["templates"]["biped_alternating"]["template_coverage"] == 0.0
    assert result["gait_label"].startswith("no_complete_stride")


def test_shuffling_without_clearance_is_not_a_valid_step():
    trace = stepping_trace()
    trace["foot_clearance_m"][:] = 0.002
    result = metrics(trace)
    assert result["valid_swing_fraction_min"] == 0.0
    assert result["median_swing_clearance_over_leg_min"] == pytest.approx(0.002)


def test_contact_chatter_and_impact_bounce_are_merged_into_stance():
    time = np.arange(0.0, 2.001, 0.002)
    force = np.full(len(time), BW / 2.0)
    for onset in (0.2, 0.4, 0.6, 0.8):
        force[(time >= onset) & (time < onset + 0.006)] = 0.0
    zeros = np.zeros(len(time))
    contacts = segment_limb(
        time, force, zeros, np.zeros((len(time), 2)), threshold_n=0.005 * BW, leg_length_m=1.0, protocol=PROTOCOL
    )
    assert contacts.touchdowns == () and contacts.merged_unloads == 4 and np.all(contacts.stance)
    # Real swings (0.2 s, 5 cm) with a 20 ms low bounce right after each touchdown.
    force = np.zeros(len(time))
    clearance = np.zeros(len(time))
    for start in np.arange(0.1, 1.9, 0.5):
        force[(time >= start) & (time < start + 0.3)] = BW / 2.0
        swing = (time >= start + 0.3) & (time < start + 0.5)
        clearance[swing] = 0.05
        bounce = (time >= start + 0.01) & (time < start + 0.03)
        force[bounce] = 0.0
        clearance[bounce] = 0.003
    contacts = segment_limb(
        time, force, clearance, np.zeros((len(time), 2)), threshold_n=0.005 * BW, leg_length_m=1.0, protocol=PROTOCOL
    )
    touchdowns = time[list(contacts.touchdowns)]
    np.testing.assert_allclose(touchdowns, [0.1, 0.6, 1.1, 1.6], atol=1e-9)


def test_continuous_phase_pauses_and_edges():
    time = np.arange(0.0, 6.0, 0.01)
    events = np.array([1.0, 2.0, 3.0, 5.5])
    phase, defined = limb_phase(time, events, 1.0, 2.0)
    assert defined[time < 0.0].sum() == 0
    assert np.all(defined[(time >= 0.0) & (time < 3.0)])
    assert not np.any(defined[(time >= 3.0) & (time < 5.5)])  # a 2.5-stride pause is undefined
    assert phase[np.searchsorted(time, 1.5)] == pytest.approx(0.5)
    assert boolean_runs(np.array([0, 1, 1, 0, 1], dtype=bool)) == [(1, 3), (4, 5)]


def test_settling_is_excluded_from_the_window():
    settled = metrics(stepping_trace(), settle_s=0.7)
    assert settled["duration_s"] == pytest.approx(4.1)
    assert settled["mean_speed_mps"] == pytest.approx(0.8)


@pytest.mark.parametrize(
    "field",
    [
        "floor_force_n",
        "foot_position_m",
        "foot_clearance_m",
        "slip_speed_mps",
        "body_floor_force_n",
        "foot_foot_force_n",
        "root_position_m",
        "touch_force_n",
    ],
)
def test_nonfinite_telemetry_fails_closed_and_remains_json_safe(field):
    trace = stepping_trace()
    trace[field].flat[10] = np.nan
    result = metrics(trace)
    assert not result["telemetry_valid"]
    assert result["templates"] == {}
    assert any(field in error for error in result["telemetry_errors"])
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "problem", ["missing", "reset", "shape", "transposed", "negative", "no_window", "feet", "weight"]
)
def test_missing_or_invalid_telemetry_is_not_a_successful_measurement(problem):
    trace = stepping_trace()
    kwargs = {}
    if problem == "missing":
        del trace["slip_speed_mps"]
    elif problem == "reset":
        trace["time_s"][240:] -= trace["time_s"][240]
    elif problem == "shape":
        trace["floor_force_n"] = trace["floor_force_n"][:, :1]
    elif problem == "transposed":
        trace["floor_force_n"] = np.ascontiguousarray(trace["floor_force_n"].T)
    elif problem == "negative":
        trace["floor_force_n"][100:150, 0] = -50.0
    elif problem == "feet":
        kwargs["foot_names"] = ("left", "right")
    elif problem == "weight":
        result = episode_gait_metrics(trace, body_weight_n=0.0, leg_length_m=1.0, foot_names=BIPED, protocol=PROTOCOL)
        assert not result["telemetry_valid"]
        return
    else:
        kwargs["settle_s"] = 4.8
    result = metrics(trace, **kwargs)
    assert not result["telemetry_valid"]
    assert result["telemetry_errors"]


@pytest.mark.parametrize(
    "phases,label,trot,pace,walk",
    [
        ((0.0, 0.5, 0.5, 0.0), "trot", 0.0, 0.5, None),
        ((0.0, 0.5, 0.0, 0.5), "pace", 0.5, 0.0, None),
        ((0.25, 0.75, 0.0, 0.5), "walk", 0.25, 0.25, 0.25),
    ],
)
def test_quadruped_patterns_keep_leg_pair_distinctions(phases, label, trot, pace, walk):
    result = metrics(gait_trace(phases, duty=0.7, period=1.0), foot_names=QUAD, settle_s=1.0)
    templates = result["templates"]
    for profile in ("quadruped_walk", "quadruped_trot", "quadruped_pace"):
        assert templates[profile]["alternation_phase_offset_max"] < 0.01
    assert templates["quadruped_trot"]["synchrony_phase_offset_max"] == pytest.approx(trot, abs=0.01)
    assert templates["quadruped_pace"]["synchrony_phase_offset_max"] == pytest.approx(pace, abs=0.01)
    if walk is not None:
        assert templates["quadruped_walk"]["walk_limb_phase_min"] == pytest.approx(walk, abs=0.01)
    assert label in result["gait_label"]


def test_metrics_are_order_deterministic_rounded_and_layout_independent():
    trace = gait_trace((0.0, 0.5), jitter=0.02, stride_cv=0.03, seed=11)
    first = metrics(trace, settle_s=1.0)
    again = metrics({key: np.asfortranarray(value) for key, value in trace.items()}, settle_s=1.0)
    assert json.dumps(first, sort_keys=True) == json.dumps(again, sort_keys=True)

    def decimals(value):
        if isinstance(value, dict):
            return all(decimals(item) for item in value.values())
        if isinstance(value, list):
            return all(decimals(item) for item in value)
        if isinstance(value, float):
            return round(value, 6) == value
        return True

    assert decimals(first)


def test_protocol_validation_and_identity_change_with_measurement_options():
    assert PROTOCOL.sha256 == measurement_protocol_sha256(PROTOCOL)
    assert PROTOCOL.sha256.startswith("sha256:")
    assert replace(PROTOCOL, chatter_fill_s=0.02).sha256 != PROTOCOL.sha256
    assert replace(PROTOCOL, template_alternation_tolerance=0.1).sha256 != PROTOCOL.sha256
    with pytest.raises(ValueError):
        GaitProtocol(template_walk_band_low=0.3, template_walk_band_high=0.2)
    with pytest.raises(ValueError):
        GaitProtocol(chatter_fill_s=float("nan"))
    with pytest.raises(ValueError):
        GaitProtocol(pause_factor=1.0)
    with pytest.raises(ValueError):
        GaitProtocol(template_synchrony_tolerance=0.3, template_gross_extra_tolerance=0.2)
