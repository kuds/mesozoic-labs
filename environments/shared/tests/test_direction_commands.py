"""Direction-command semantics, deterministic schedules and honest stop metrics."""

from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from environments.shared.direction_commands import (
    DirectionCommandConfig,
    DirectionCommandController,
    body_frame_velocity,
    gaussian_tracking_reward,
    tracking_metrics,
    wrap_angle,
)


def _controller(config=None, seed=42, heading=0.0):
    controller = DirectionCommandController(config)
    controller.reset(np.random.default_rng(seed), heading)
    return controller


def test_wrap_preserves_turn_sign_across_pi_and_defines_half_turn():
    assert wrap_angle(math.radians(-179 - 179)) == pytest.approx(math.radians(2))
    assert wrap_angle(math.radians(179 + 179)) == pytest.approx(math.radians(-2))
    assert wrap_angle(math.pi) == -math.pi
    assert wrap_angle(-math.pi) == -math.pi
    assert wrap_angle(9 * math.pi) == pytest.approx(-math.pi)
    with pytest.raises(ValueError, match="finite"):
        wrap_angle(math.nan)


@pytest.mark.parametrize(
    ("heading", "world_velocity", "expected"),
    [
        (0.0, [1.0, 2.0], [1.0, 2.0]),
        (math.pi / 2, [0.0, 1.0], [1.0, 0.0]),
        (math.pi / 2, [1.0, 0.0, 8.0], [0.0, -1.0]),
        (math.pi, [-2.0, 1.0], [2.0, -1.0]),
    ],
)
def test_body_velocity_is_forward_left_and_ignores_vertical(heading, world_velocity, expected):
    np.testing.assert_allclose(body_frame_velocity(np.array(world_velocity), heading), expected, atol=1e-14)


def test_adapter_turns_left_and_right_without_strafing_or_reinterpreting_units():
    controller = _controller()
    controller.set_target(math.pi / 2, 1.05)
    left = controller.update(0.0, 0.0)
    np.testing.assert_allclose(left.physical, [1.05 * 0.25, 0.0, 0.3])
    np.testing.assert_allclose(left.normalized, [1.05 * 0.25 / 1.5, 0.0, 0.3 / 0.6])
    controller.set_target(-math.pi / 2, 1.05)
    right = controller.update(0.0, 0.0)
    assert right.physical[2] == pytest.approx(-0.3)
    aligned = controller.update(1.0, -math.pi / 2)
    np.testing.assert_allclose(aligned.physical, [1.05, 0.0, 0.0])
    np.testing.assert_allclose(aligned.normalized, [0.7, 0.0, 0.0])
    assert aligned.heading_error == 0.0
    with pytest.raises(ValueError):
        aligned.normalized[:] = 0.0


def test_adapter_crosses_heading_wrap_by_the_short_route():
    controller = _controller(heading=math.radians(179))
    controller.set_target(math.radians(-179), 1.0)
    state = controller.update(0.0, math.radians(179))
    assert state.heading_error == pytest.approx(math.radians(2))
    assert state.physical[2] == pytest.approx(1.5 * math.radians(2))


def test_scaling_is_fixed_across_sampled_speed_ranges_and_physical_zero_stays_zero():
    a = _controller(DirectionCommandConfig(speed_range=(0.5, 0.75)))
    b = _controller(DirectionCommandConfig(speed_range=(0.75, 1.05)))
    for controller in (a, b):
        controller.set_target(0.0, 0.75)
    np.testing.assert_array_equal(a.update(0.0, 0.0).normalized, b.update(0.0, 0.0).normalized)
    np.testing.assert_array_equal(a.update(0.0, 0.0).normalized, [0.5, 0.0, 0.0])
    a.set_target(math.pi, 0.0)
    stop = a.update(0.0, 1.0)
    assert not stop.heading_active
    assert stop.heading_error == 0.0
    np.testing.assert_array_equal(stop.physical, np.zeros(3))
    np.testing.assert_array_equal(stop.normalized, np.zeros(3))


def test_seeded_schedule_is_independent_of_actual_motion_and_update_frequency():
    config = DirectionCommandConfig(
        switch_interval_s=1.0,
        switch_jitter_s=0.2,
        speed_range=(0.5, 1.05),
        stop_probability=0.2,
    )
    frequent = _controller(config, seed=123, heading=1.0)
    sparse = _controller(config, seed=123, heading=1.0)
    for time_s in np.linspace(0.0, 10.0, 101):
        frequent.update(float(time_s), current_heading=float(time_s * 0.2))
    sparse.update(10.0, current_heading=-2.0)
    assert frequent.events == sparse.events
    assert frequent.schedule_seed == sparse.schedule_seed
    different = _controller(config, seed=124, heading=1.0)
    different.update(10.0, current_heading=-2.0)
    assert different.events != sparse.events
    events = sparse.events
    for prior, current in zip(events, events[1:]):
        assert abs(wrap_angle(current["desired_heading"] - prior["desired_heading"])) <= math.pi / 6 + 1e-12
        assert 0.8 <= current["start_time_s"] - prior["start_time_s"] <= 1.2
    assert all(event["desired_speed"] == 0.0 or 0.5 <= event["desired_speed"] <= 1.05 for event in events)


def test_episode_rng_is_drawn_once_and_future_sampling_does_not_consume_environment_rng():
    rng = np.random.default_rng(10)
    reference = np.random.default_rng(10)
    expected_seed = int(reference.integers(0, 2**32, dtype=np.uint64))
    controller = DirectionCommandController()
    controller.reset(rng, 0.0)
    controller.update(100.0, 0.0)
    assert controller.schedule_seed == expected_seed
    assert rng.random() == reference.random()


def test_switch_times_are_seconds_and_include_exact_boundary_without_repeat():
    controller = _controller(DirectionCommandConfig(switch_interval_s=0.03, straight_probability=0.0))
    assert controller.update(0.02, 0.0).event_id == 0
    assert controller.update(0.03, 0.0).event_id == 1
    assert controller.update(0.03, 0.0).event_id == 1
    at_nine = controller.update(0.09, 0.0)
    assert at_nine.event_id == 3
    assert at_nine.event_start_s == pytest.approx(0.09)
    with pytest.raises(ValueError, match="backwards"):
        controller.update(0.08, 0.0)
    reset = controller.reset(np.random.default_rng(42), 0.0)
    assert reset.time_s == 0.0 and reset.event_id == 0


def test_external_target_is_persistent_timestamped_and_reset_restores_schedule():
    controller = _controller()
    controller.update(0.1, 0.0)
    controller.set_target(0.5, 0.75, time_s=0.2)
    state = controller.update(9.0, 0.0)
    assert state.target_source == "external"
    assert state.desired_heading == 0.5
    assert state.desired_speed == 0.75
    assert state.event_start_s == 0.2
    assert state.event_id == 1
    controller.set_target(-0.25, 0.5)
    assert controller.events[-1]["start_time_s"] == 9.0
    events = controller.events
    events[-1]["desired_speed"] = -123
    assert controller.events[-1]["desired_speed"] == 0.5
    with pytest.raises(ValueError, match="backwards"):
        controller.set_target(0.0, 0.0, time_s=8.0)
    controller.reset(np.random.default_rng(42), 0.0)
    assert controller.update(4.0, 0.0).target_source == "schedule"


def test_default_stage_moves_at_fixed_cruise_and_stop_sampling_is_opt_in():
    controller = _controller()
    controller.update(200.0, 0.0)
    assert {event["desired_speed"] for event in controller.events} == {1.05}
    stopped = _controller(DirectionCommandConfig(stop_probability=1.0))
    state = stopped.update(20.0, 2.0)
    assert {event["desired_speed"] for event in stopped.events} == {0.0}
    np.testing.assert_array_equal(state.physical, np.zeros(3))


def test_metrics_distinguish_tracking_the_adapter_from_reaching_the_world_heading():
    controller = _controller()
    controller.set_target(math.pi / 2, 1.05)
    state = controller.update(0.0, 0.0)
    metrics = tracking_metrics(state, np.array([state.physical[0], 0.0]), float(state.physical[2]))
    assert metrics["tracking_command_in_tolerance"]
    assert not metrics["tracking_in_tolerance"]
    assert metrics["tracking_error_heading"] == pytest.approx(math.pi / 2)
    assert metrics["tracking_heading_active"]
    assert metrics["tracking_error_v"] == 0.0
    assert metrics["tracking_error_yaw"] == 0.0
    state = controller.update(1.0, math.pi / 2)
    reached = tracking_metrics(state, np.array([0.0, 1.05]), 0.0)
    assert reached["tracking_in_tolerance"]
    assert reached["actual_v_x"] == pytest.approx(1.05)


def test_stop_metrics_mask_heading_and_do_not_reward_drift_or_spinning_as_success():
    controller = _controller()
    controller.set_target(math.pi, 0.0)
    state = controller.update(0.0, 0.0)
    stopped = tracking_metrics(state, np.zeros(2), 0.0)
    assert stopped["tracking_in_tolerance"]
    assert not stopped["tracking_heading_active"]
    assert stopped["tracking_error_heading"] == 0.0
    assert not tracking_metrics(state, np.array([0.11, 0.0]), 0.0)["tracking_in_tolerance"]
    assert not tracking_metrics(state, np.zeros(2), 0.16)["tracking_in_tolerance"]
    assert gaussian_tracking_reward(stopped["tracking_error_v"], stopped["tracking_error_yaw"]) == 1.0


def test_poststep_metrics_use_measured_heading_but_the_command_that_was_executed():
    controller = _controller(DirectionCommandConfig(turn_slowdown=False))
    controller.set_target(math.pi / 2, 1.05)
    executed = controller.update(0.0, 0.0)
    assert executed.heading_error == math.pi / 2
    # The animal has reached its heading since that observation was emitted.
    poststep = tracking_metrics(
        executed, np.array([0.0, 1.05]), float(executed.physical[2]), current_heading=math.pi / 2
    )
    assert poststep["actual_v_x"] == pytest.approx(1.05)
    assert poststep["tracking_error_yaw"] == 0.0
    assert poststep["tracking_error_heading"] == 0.0
    assert poststep["tracking_in_tolerance"]
    assert executed.current_heading == 0.0  # The executed snapshot stays unchanged.
    # Reaching a heading while still moving at the adapter's earlier reduced
    # speed is command tracking, but not yet the full requested speed goal.
    slow = _controller()
    slow.set_target(math.pi / 2, 1.05)
    executed_slow = slow.update(0.0, 0.0)
    slowing = tracking_metrics(
        executed_slow,
        np.array([0.0, float(executed_slow.physical[0])]),
        float(executed_slow.physical[2]),
        current_heading=math.pi / 2,
    )
    assert slowing["tracking_command_in_tolerance"]
    assert not slowing["tracking_in_tolerance"]


def test_gaussian_reward_uses_physical_errors_and_is_bounded():
    assert gaussian_tracking_reward(0.0, 0.0) == 1.0
    assert gaussian_tracking_reward(0.25, 0.0) == pytest.approx(math.exp(-1))
    assert gaussian_tracking_reward(0.0, 0.15) == pytest.approx(math.exp(-1))
    assert gaussian_tracking_reward(0.25, 0.15) == pytest.approx(math.exp(-2))
    assert gaussian_tracking_reward(1e308, 1e308) == 0.0
    for errors in [(-1.0, 0.0), (0.0, math.nan)]:
        with pytest.raises(ValueError):
            gaussian_tracking_reward(*errors)
    with pytest.raises(ValueError, match="positive"):
        gaussian_tracking_reward(0.0, 0.0, velocity_sigma=0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"speed_scale": 0.0},
        {"lateral_speed_scale": 0.0},
        {"yaw_rate_scale": 0.0},
        {"yaw_rate_max": 0.7},
        {"switch_interval_s": 0.0},
        {"switch_jitter_s": 4.0},
        {"switch_jitter_s": -1.0},
        {"stop_probability": 1.01},
        {"straight_probability": -0.1},
        {"speed_range": (1.05, 0.5)},
        {"speed_range": (0.5, 2.0)},
        {"speed_range": (0.5,)},
        {"cruise_speed": -0.1},
        {"yaw_gain": math.inf},
        {"turn_increment_max": math.pi + 0.01},
        {"minimum_turn_speed_fraction": -0.1},
    ],
)
def test_invalid_contracts_fail_before_an_episode(kwargs):
    with pytest.raises(ValueError):
        DirectionCommandConfig(**kwargs)


def test_config_is_frozen_manifest_is_stable_and_logs_are_serializable():
    config = DirectionCommandConfig(speed_range=[0.5, 1.05])
    assert config.speed_range == (0.5, 1.05)
    with pytest.raises(FrozenInstanceError):
        config.speed_scale = 9.0
    controller = _controller(config)
    manifest = controller.manifest()
    state = controller.update(10.0, 0.0)
    assert controller.manifest() == manifest
    json.dumps({"manifest": manifest, "events": controller.events, "info": state.as_info()}, allow_nan=False)


def test_calls_require_reset_and_invalid_inputs_do_not_change_target():
    controller = DirectionCommandController()
    with pytest.raises(RuntimeError, match="reset"):
        controller.update(0.0, 0.0)
    with pytest.raises(RuntimeError, match="reset"):
        controller.set_target(0.0, 1.0)
    controller.reset(np.random.default_rng(42), 0.0)
    for speed in (-1.0, 2.0, math.inf):
        with pytest.raises(ValueError):
            controller.set_target(1.0, speed)
    assert controller.events[-1]["desired_heading"] == 0.0
    with pytest.raises(ValueError):
        tracking_metrics(controller.update(0.0, 0.0), np.array([math.nan, 0.0]), 0.0)
