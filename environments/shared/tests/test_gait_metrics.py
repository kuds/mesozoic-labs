"""Adversarial finite-window fixtures for the pure gait measurement layer."""

import json
from dataclasses import replace

import numpy as np
import pytest

from environments.shared.gait.events import circular_distance, contact_events
from environments.shared.gait.metrics import episode_gait_metrics
from environments.shared.gait.types import GaitProtocol, measurement_protocol_sha256

BW = 981.0
PERIOD = 0.8
PROTOCOL = GaitProtocol()


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
        position[:, index, 0] = (cycle + swing) * speed * PERIOD
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


def metrics(trace, foot_names=("r", "l"), **kwargs):
    return episode_gait_metrics(
        trace,
        body_weight_n=BW,
        leg_length_m=1.0,
        foot_names=foot_names,
        protocol=kwargs.pop("protocol", PROTOCOL),
        **kwargs,
    )


def test_alternating_walk_has_real_complete_cycles_and_support():
    result = metrics(stepping_trace())
    assert result["telemetry_valid"]
    assert result["duration_s"] == pytest.approx(4.8)
    assert result["mean_speed_mps"] == pytest.approx(0.8)
    assert result["phase_match_fraction"] == 1.0
    assert result["simultaneous_fraction"] == 0.0
    assert result["flight_fraction"] == 0.0
    assert result["support_count_fraction"]["2"] == pytest.approx(0.25)
    assert result["slip_over_leg"] == 0.0
    for foot in result["per_foot"].values():
        assert foot["complete_cycles"] == foot["valid_cycles"] == 5
        assert foot["duty_factor_mean"] == pytest.approx(0.625)
        assert foot["cadence_hz"] == pytest.approx(1.25)
        assert foot["reposition_over_leg_min"] == pytest.approx(0.64)
    # JSON never contains non-standard NaN tokens, including unavailable fields.
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("dt", [0.005, 0.01, 0.02])
def test_elapsed_time_metrics_are_sampling_rate_invariant(dt):
    baseline = metrics(stepping_trace(dt=0.005, slip=0.1))
    result = metrics(stepping_trace(dt=dt, slip=0.1))
    for name in (
        "flight_fraction",
        "phase_match_fraction",
        "simultaneous_fraction",
        "slip_over_leg",
        "duration_s",
        "mean_speed_mps",
        "loaded_slip_speed_mps",
    ):
        assert result[name] == pytest.approx(baseline[name], abs=1e-12)
    for name in ("r", "l"):
        assert result["per_foot"][name]["complete_cycles"] == baseline["per_foot"][name]["complete_cycles"]


def test_grounded_running_contact_pattern_is_not_mislabelled_as_mechanical_walk():
    trace = stepping_trace()
    trace["root_position_m"][:, 2] += 0.01 * np.cos(2 * np.pi * trace["time_s"] / 0.4)
    result = metrics(trace)
    assert result["phase_match_fraction"] == 1.0
    assert result["flight_fraction"] == 0.0
    assert "mechanical_gait" not in result
    assert result["per_foot"]["r"]["valid_cycles"] == 5


def test_aerial_alternating_run_is_measured_without_a_blanket_flight_veto():
    result = metrics(stepping_trace(duty=0.375))
    assert result["phase_match_fraction"] == 1.0
    assert result["simultaneous_fraction"] == 0.0
    assert result["flight_fraction"] == pytest.approx(0.25)
    assert result["max_flight_s"] == pytest.approx(0.1)
    assert result["per_foot"]["r"]["valid_cycles"] == 5


@pytest.mark.parametrize("duty,expected_flight", [(0.375, 0.625), (0.625, 0.375)])
def test_synchronous_hop_is_not_alternating_even_with_high_duty(duty, expected_flight):
    result = metrics(stepping_trace(phases=(0.0, 0.0), duty=duty))
    assert result["phase_match_fraction"] == 0.0
    assert result["simultaneous_fraction"] == 1.0
    assert result["flight_fraction"] == pytest.approx(expected_flight)
    assert result["pair_phase"]["r|l"]["synchrony_match_fraction"] == 1.0


def test_staggered_hop_and_circular_phase_wrap_do_not_average_to_alternation():
    result = metrics(stepping_trace(phases=(0.0, 0.025), duty=0.375))
    assert result["phase_match_fraction"] == 0.0
    assert result["simultaneous_fraction"] == 1.0
    assert result["pair_phase"]["r|l"]["mean_phase"] == pytest.approx(0.025)
    np.testing.assert_allclose(circular_distance([0.99, 0.01], 0.0), [0.01, 0.01])
    np.testing.assert_allclose(circular_distance([0.99, 0.01], 0.5), [0.49, 0.49])


def test_phase_zero_jitter_preserves_complete_pairs_and_circular_mean():
    trace = stepping_trace(phases=(0.0, 0.0), duty=0.375, dt=0.001)
    time = trace["time_s"]
    left_loaded = trace["floor_force_n"][:, 0] > 0.0
    right_loaded = np.zeros(len(time), dtype=bool)
    for cycle in range(-1, 7):
        touchdown = 0.2 + cycle * PERIOD + (0.008 if cycle % 2 == 0 else -0.008)
        right_loaded |= (time >= touchdown - 1e-12) & (time < touchdown + 0.3 - 1e-12)
    loads = np.stack((left_loaded, right_loaded), axis=1)
    trace["floor_force_n"] = loads * BW / np.maximum(1, np.sum(loads, axis=1))[:, None]
    result = metrics(trace)
    pair = result["pair_phase"]["r|l"]
    assert pair["samples"] == pair["eligible_cycles"] == 5
    assert pair["synchrony_match_fraction"] == 1.0
    assert pair["alternation_match_fraction"] == 0.0
    assert pair["concentration"] > 0.99
    assert circular_distance(pair["mean_phase"], 0.0) < 0.01


def test_unaligned_sample_grid_has_only_time_resolution_error():
    result = metrics(stepping_trace(duty=0.375, dt=0.007))
    assert result["flight_fraction"] == pytest.approx(0.25, abs=0.014 / PERIOD)
    assert result["max_flight_s"] == pytest.approx(0.1, abs=0.014)
    assert result["phase_match_fraction"] == 1.0
    assert result["per_foot"]["r"]["complete_cycles"] == 5


def test_sliding_and_unloaded_foot_motion_are_separate():
    trace = stepping_trace(slip=0.8)
    result = metrics(trace)
    assert result["phase_match_fraction"] == 1.0
    assert result["loaded_slip_speed_mps"] == pytest.approx(0.8)
    assert result["slip_over_leg"] == pytest.approx(0.4)
    trace["slip_speed_mps"] = np.where(trace["floor_force_n"] > 0.0, 0.0, 100.0)
    result = metrics(trace)
    assert result["loaded_slip_speed_mps"] == 0.0
    assert result["slip_over_leg"] == 0.0


def test_subthreshold_loaded_sliding_is_visible_without_synthetic_cycles():
    trace = stepping_trace()
    trace["floor_force_n"][:] = BW * 0.005
    trace["slip_speed_mps"][:] = 0.8
    trace["body_floor_force_n"][:] = BW * 0.005
    result = metrics(trace)
    assert result["slip_over_leg"] == pytest.approx(3.84)
    assert result["loaded_slip_speed_mps"] == pytest.approx(0.8)
    assert result["body_support_fraction"] == pytest.approx(1.0 / 3.0)
    assert result["body_contact_fraction"] == 1.0
    assert result["body_loaded_fraction"] == 0.0
    assert all(foot["complete_cycles"] == 0 for foot in result["per_foot"].values())


def test_standing_has_no_artificial_reset_steps_and_sliding_partial_stance_cannot_escape():
    trace = stepping_trace()
    trace["floor_force_n"][:] = BW / 2.0
    trace["foot_position_m"][:] = 0.0
    trace["foot_clearance_m"][:] = 0.0
    trace["root_position_m"][:, 0] = 0.0
    result = metrics(trace)
    assert result["flight_fraction"] == 0.0
    assert result["mean_speed_mps"] == 0.0
    assert result["phase_match_fraction"] is None
    assert all(foot["complete_cycles"] == 0 for foot in result["per_foot"].values())
    trace["slip_speed_mps"][:] = 0.8
    result = metrics(trace)
    assert result["slip_over_leg"] == pytest.approx(3.84)


def test_shuffling_without_clearance_or_reposition_does_not_make_valid_steps():
    trace = stepping_trace()
    trace["foot_clearance_m"][:] = 0.0
    trace["foot_position_m"][:] = 0.0
    result = metrics(trace)
    assert result["phase_match_fraction"] == 1.0
    assert all(foot["complete_cycles"] == 5 and foot["valid_cycles"] == 0 for foot in result["per_foot"].values())


def test_contact_chatter_and_schmitt_hysteresis_do_not_generate_cycles():
    time = np.arange(0.0, 1.001, 0.001)
    force = np.full(len(time), BW / 2.0)
    for onset in (0.2, 0.4, 0.6, 0.8):
        force[(time >= onset) & (time < onset + 0.005)] = 0.0
    result = contact_events(time, force, body_weight_n=BW, protocol=PROTOCOL)
    assert result.touchdowns == result.liftoffs == result.complete_cycles == ()
    assert np.all(result.loaded)
    force[:] = BW * 0.015  # Between thresholds; initial support is unobserved.
    result = contact_events(time, force, body_weight_n=BW, protocol=PROTOCOL)
    assert not np.any(result.loaded)
    force[0] = BW * 0.03
    result = contact_events(time, force, body_weight_n=BW, protocol=PROTOCOL)
    assert np.all(result.loaded)


def test_unconfirmed_terminal_touchdown_and_settling_do_not_complete_a_stride():
    full = metrics(stepping_trace())
    trimmed = metrics(stepping_trace(duration=4.21))
    assert trimmed["per_foot"]["r"]["complete_cycles"] == full["per_foot"]["r"]["complete_cycles"] - 1
    settled = metrics(stepping_trace(), settle_s=0.7)
    assert settled["duration_s"] == pytest.approx(4.1)
    assert settled["mean_speed_mps"] == pytest.approx(0.8)
    for foot in settled["per_foot"].values():
        assert all(cycle["touchdown_s"] >= 0.7 for cycle in foot["cycles"])


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
    assert result["phase_match_fraction"] is None
    assert any(field in error for error in result["telemetry_errors"])
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("problem", ["missing", "reset", "shape", "negative", "no_window"])
def test_missing_or_invalid_telemetry_is_not_a_successful_measurement(problem):
    trace = stepping_trace()
    kwargs = {}
    if problem == "missing":
        del trace["slip_speed_mps"]
    elif problem == "reset":
        trace["time_s"][240:] -= trace["time_s"][240]
    elif problem == "shape":
        trace["floor_force_n"] = trace["floor_force_n"][:, :1]
    elif problem == "negative":
        trace["slip_speed_mps"][100, 0] = -1.0
    else:
        kwargs["settle_s"] = 4.8
    result = metrics(trace, **kwargs)
    assert not result["telemetry_valid"]
    assert result["telemetry_errors"]


@pytest.mark.parametrize(
    "phases,alternating,diagonal,ipsilateral,four_beat",
    [
        ((0.0, 0.5, 0.5, 0.0), 1.0, 1.0, 0.0, 0.0),  # trot
        ((0.0, 0.5, 0.0, 0.5), 1.0, 0.0, 1.0, 0.0),  # pace
        ((0.0, 0.5, 0.25, 0.75), 1.0, 0.0, 0.0, 1.0),  # four-beat walk
        ((0.0, 0.0, 0.0, 0.0), 0.0, 1.0, 1.0, 0.0),  # pronk
    ],
)
def test_quadruped_patterns_keep_leg_pair_distinctions(phases, alternating, diagonal, ipsilateral, four_beat):
    result = metrics(stepping_trace(phases=phases, duty=0.75), foot_names=("fr", "fl", "rr", "rl"))
    assert result["phase_match_fraction"] == alternating
    assert result["diagonal_phase_match_fraction"] == diagonal
    assert result["ipsilateral_phase_match_fraction"] == ipsilateral
    assert result["four_beat_fraction"] == four_beat
    assert all(foot["valid_cycles"] >= 4 for foot in result["per_foot"].values())


def test_missing_other_foot_events_are_counted_as_phase_mismatches():
    trace = stepping_trace()
    # Remove one complete left stance, producing a two-period left stride.
    trace["floor_force_n"][(trace["time_s"] >= 2.2) & (trace["time_s"] < 2.7), 1] = 0.0
    result = metrics(trace)
    pair = result["pair_phase"]["r|l"]
    assert pair["eligible_cycles"] == 5
    assert pair["samples"] == 4
    assert pair["alternation_match_fraction"] == pytest.approx(0.8)


def test_body_support_collision_duration_and_input_immutability():
    trace = stepping_trace()
    mask = (trace["time_s"] >= 1.0) & (trace["time_s"] < 1.48)
    trace["body_floor_force_n"][mask] = BW * 0.1
    trace["foot_foot_force_n"][mask] = BW * 0.1
    originals = {name: array.copy() for name, array in trace.items()}
    result = metrics(trace)
    assert result["body_support_fraction"] == pytest.approx(0.01 / 1.01)
    assert result["body_contact_fraction"] == pytest.approx(0.1)
    assert result["body_loaded_fraction"] == pytest.approx(0.1)
    assert result["foot_foot_contact_fraction"] == pytest.approx(0.1)
    for name in trace:
        np.testing.assert_array_equal(trace[name], originals[name])


def test_protocol_validation_and_identity_change_with_measurement_options():
    assert PROTOCOL.sha256 == measurement_protocol_sha256(PROTOCOL)
    assert PROTOCOL.sha256.startswith("sha256:")
    assert replace(PROTOCOL, min_stance_s=0.05).sha256 != PROTOCOL.sha256
    with pytest.raises(ValueError):
        GaitProtocol(contact_on_bw=0.01, contact_off_bw=0.02)
    with pytest.raises(ValueError):
        GaitProtocol(min_stance_s=float("nan"))
