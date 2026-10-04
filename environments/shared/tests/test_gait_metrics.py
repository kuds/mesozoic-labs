"""Adversarial finite-window fixtures for the pure gait measurement layer."""

import json
from dataclasses import replace

import numpy as np
import pytest

from environments.shared.gait.events import (
    boolean_runs,
    circular_distance,
    limb_phase,
    segment_limb,
    stride_is_normal,
    stride_references,
)
from environments.shared.gait.metrics import episode_gait_metrics
from environments.shared.gait.types import GaitProtocol, measurement_protocol_sha256

BW = 981.0
PERIOD = 0.8
PROTOCOL = GaitProtocol()
BIPED = ("r", "l")
QUAD = ("fr", "fl", "rr", "rl")


def yaw_quaternion(yaw_rad):
    """Root quaternions (w, x, y, z) of a level trunk whose body x axis points at ``yaw_rad``."""
    yaw = np.asarray(yaw_rad, dtype=float)
    out = np.zeros(yaw.shape + (4,))
    out[..., 0] = np.cos(yaw / 2.0)
    out[..., 3] = np.sin(yaw / 2.0)
    return out


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
        "root_quat_wxyz": yaw_quaternion(np.zeros(len(time))),
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
    width=0.2,
):
    """Footfall-pattern puppet: a shared stride clock, per-foot phase/duty/load.

    ``phases`` are touchdown offsets in cycles; ``jitter`` is the touchdown SD
    in cycles and ``stride_cv`` the stride-period coefficient of variation.
    ``load`` weights share body weight among loaded feet; a foot loaded alone
    carries ``min(load, 1)`` body weights (a token tap stays light).
    ``switch=(t, phases)`` changes the pattern for strides starting after
    ``t``. ``step_to`` places the second foot level with the first instead of
    passing it. Feet land half a stance travel ahead of the hip, ``width``
    apart (the first foot of each pair at +width/2).
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
        lateral = (-1.0 if foot % 2 else 1.0) * 0.5 * width
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
        "root_quat_wxyz": yaw_quaternion(np.zeros(count)),
        "touch_force_n": force.copy(),
    }


def scheduled_trace(
    plan,
    *,
    feet=2,
    duration=10.0,
    dt=0.002,
    clearance=0.05,
    jitter=0.0,
    seed=0,
    load=None,
    slip=0.0,
    swing_drag=0.0,
    flick=0.2,
):
    """Footfall puppet whose every stride follows ``plan(start) -> (phases, period, duty, speed[, glide])``.

    Stride-by-stride schedules give bouts (hop bouts, scrambles) and cadence
    changes (a walk-to-run transition); the trunk integrates the scheduled
    speed and each foot lands half its stance travel ahead of the trunk.
    ``slip`` is the stance glide as a fraction of trunk speed (a plan's
    optional fifth element overrides it for that stride). ``swing_drag``
    > 0 keeps the swing foot on the floor with that force (x BW) and sliding,
    except for a ``flick`` fraction of the swing in mid-swing (a toe drag).
    """
    rng = np.random.default_rng(seed)
    loads = np.ones(feet) if load is None else np.asarray(load, dtype=float)
    time = np.arange(round(duration / dt) + 1, dtype=float) * dt
    count = len(time)
    strides = []
    start = -1.0
    while start < duration + 1.0:
        phases, period, duty, speed, *glide_override = plan(start)
        duties = np.broadcast_to(np.asarray(duty, dtype=float), (feet,))
        strides.append((start, period, phases, duties, speed, glide_override[0] if glide_override else slip))
        start += period
    speed_at = np.zeros(count)
    for start, period, _, _, speed, _ in strides:
        speed_at[(time >= start) & (time < start + period)] = speed
    root_x = np.concatenate(([0.0], np.cumsum(speed_at[:-1] * dt)))
    stances: list[list[tuple[float, float, float, float]]] = [[] for _ in range(feet)]
    for start, period, phases, duties, speed, stride_slip in strides:
        for foot in range(feet):
            touchdown = start + (phases[foot] % 1.0) * period + jitter * period * rng.standard_normal()
            stances[foot].append((touchdown, touchdown + duties[foot] * period, speed, stride_slip))
    loaded = np.zeros((count, feet), dtype=bool)
    position = np.zeros((count, feet, 3))
    height = np.zeros((count, feet))
    drag = np.zeros((count, feet), dtype=bool)
    glide = np.zeros((count, feet))
    for foot in range(feet):
        ordered = sorted(stances[foot])
        ordered = [
            (td, min(lo, ordered[k + 1][0] - 0.04) if k + 1 < len(ordered) else lo, v, g)
            for k, (td, lo, v, g) in enumerate(ordered)
        ]
        lateral = (-1.0 if foot % 2 else 1.0) * 0.1
        fore = 0.6 if feet == 4 and foot < 2 else 0.0
        placements = [float(np.interp(td, time, root_x)) + 0.5 * v * (lo - td) + fore for td, lo, v, _ in ordered]
        x = np.full(count, placements[0])
        for k, (td, lo, v, g) in enumerate(ordered):
            stance = (time >= td) & (time < lo)
            loaded[stance, foot] = True
            x[time >= td] = placements[k]
            x[stance] = placements[k] + g * v * (time[stance] - td)
            glide[stance, foot] = g * v
            if k + 1 < len(ordered):
                next_td = ordered[k + 1][0]
                swing = (time >= lo) & (time < next_td)
                fraction = (time[swing] - lo) / (next_td - lo)
                lift = x[swing][0] if np.any(swing) else placements[k]
                x[swing] = lift + (placements[k + 1] - lift) * (3 * fraction**2 - 2 * fraction**3)
                bell = clearance * np.sin(np.pi * fraction)
                if swing_drag > 0.0:
                    airborne = np.abs(fraction - 0.5) < 0.5 * flick
                    bell = np.where(airborne, clearance * np.sin(np.pi * (fraction - 0.5 + 0.5 * flick) / flick), 0.0)
                    drag[np.flatnonzero(swing)[~airborne], foot] = True
                height[swing, foot] = bell
        position[:, foot, 0] = x
        position[:, foot, 1] = lateral
        position[:, foot, 2] = height[:, foot]
    weighted = loaded * loads
    force = BW * weighted / np.maximum(np.sum(weighted, axis=1, keepdims=True), 1.0)
    force = np.where(drag, swing_drag * BW, force)
    velocity = np.abs(np.gradient(position[:, :, 0], time, axis=0))
    slip_speed = np.where(loaded, glide, np.where(drag, velocity, 0.0))
    root = np.zeros((count, 3))
    root[:, 0] = root_x
    root[:, 2] = 1.0
    return {
        "time_s": time,
        "floor_force_n": force,
        "foot_position_m": position,
        "foot_clearance_m": np.where(loaded | drag, 0.0, height),
        "slip_speed_mps": slip_speed,
        "body_floor_force_n": np.zeros(count),
        "foot_foot_force_n": np.zeros(count),
        "root_position_m": root,
        "root_quat_wxyz": yaw_quaternion(np.zeros(count)),
        "touch_force_n": force.copy(),
    }


def rotated(trace, degrees, about=(0.0, 0.0)):
    """The whole episode (positions and trunk) rotated about +z; the declared task direction is not touched."""
    angle = np.radians(degrees)
    cos, sin = float(np.cos(angle)), float(np.sin(angle))
    out = {key: np.array(value, copy=True) for key, value in trace.items()}
    for key in ("foot_position_m", "root_position_m"):
        x = out[key][..., 0] - about[0]
        y = out[key][..., 1] - about[1]
        out[key][..., 0] = about[0] + cos * x - sin * y
        out[key][..., 1] = about[1] + sin * x + cos * y
    # q -> q_z(angle) * q
    w, x, y, z = (out["root_quat_wxyz"][:, k].copy() for k in range(4))
    c, s = float(np.cos(angle / 2.0)), float(np.sin(angle / 2.0))
    out["root_quat_wxyz"] = np.stack([c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w], axis=1)
    return out


def mirrored(trace):
    """Left/right mirror image (across the x-z plane) with the canonical foot order kept."""
    feet = trace["floor_force_n"].shape[1]
    order = [1, 0] if feet == 2 else [1, 0, 3, 2]
    out = {key: np.array(value, copy=True) for key, value in trace.items()}
    for key in ("floor_force_n", "foot_position_m", "foot_clearance_m", "slip_speed_mps", "touch_force_n"):
        out[key] = out[key][:, order]
    out["foot_position_m"][..., 1] *= -1.0
    out["root_position_m"][..., 1] *= -1.0
    out["root_quat_wxyz"][:, 1] *= -1.0  # a mirrored body: (w, x, y, z) -> (w, -x, y, -z)
    out["root_quat_wxyz"][:, 3] *= -1.0
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
    template = result["templates"]["alternation"]
    assert template["alternation_phase_offset_max"] == pytest.approx(0.0, abs=1e-6)
    assert template["template_coverage"] == pytest.approx(1.0)
    assert template["alternating_overlap_index_max"] == pytest.approx(0.0, abs=1e-6)
    for foot in result["per_foot"].values():
        assert foot["duty_factor"] == pytest.approx(0.625, abs=0.01)
        assert foot["valid_swing_fraction"] == 1.0
        assert foot["median_swing_clearance_over_leg"] == pytest.approx(0.05, abs=1e-3)
    for foot in result["per_foot"].values():
        # each footprint lands half a stride (0.32 L) past the other one, every step
        assert foot["step_length_over_leg_median"] == pytest.approx(0.32, abs=0.01)
        assert foot["step_through_fraction"] == 1.0
    assert result["step_length_over_leg_min"] == pytest.approx(0.32, abs=0.01)
    # every stride (a step of each foot) steps through with both feet
    assert result["contralateral"]["r|l"]["step_through_stride_fraction"] == 1.0
    assert result["step_through_stride_fraction_min"] == 1.0
    assert result["trunk_height_over_leg_median"] == 1.0
    assert result["trunk_crab_angle_deg_median"] == pytest.approx(0.0, abs=1e-6)
    assert template["off_gait_fraction"] == 0.0
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
        assert result["templates"]["alternation"][name] == pytest.approx(
            baseline["templates"]["alternation"][name], abs=0.02
        )


def test_aerial_alternating_run_is_measured_without_a_blanket_flight_veto():
    result = metrics(stepping_trace(duty=0.375))
    assert result["flight_fraction"] == pytest.approx(0.25, abs=0.01)
    assert result["max_flight_s"] == pytest.approx(0.1, abs=0.011)
    assert result["templates"]["alternation"]["alternation_phase_offset_max"] < 0.01
    assert result["hop_flight_fraction"] == pytest.approx(0.0, abs=1e-6)
    assert result["gait_label"].startswith("aerial_run")


@pytest.mark.parametrize("duty", [0.375, 0.625])
def test_synchronous_hop_is_far_from_alternation_by_independent_statistics(duty):
    result = metrics(stepping_trace(phases=(0.0, 0.0), duty=duty))
    template = result["templates"]["alternation"]
    assert template["alternation_phase_offset_max"] == pytest.approx(0.5, abs=1e-6)
    assert template["alternating_overlap_index_max"] == pytest.approx(1.0)
    assert template["template_coverage"] == 0.0
    assert template["off_gait_fraction"] == pytest.approx(1.0, abs=0.01)
    assert result["step_length_over_leg_min"] == pytest.approx(0.0, abs=1e-6)
    assert result["gait_label"].startswith("hop")


def test_staggered_hop_and_circular_phase_wrap_do_not_average_to_alternation():
    result = metrics(stepping_trace(phases=(0.0, 0.025), duty=0.375))
    assert result["pair_phase"]["r>l"]["mean_phase"] == pytest.approx(0.025, abs=1e-3)
    assert result["templates"]["alternation"]["alternation_phase_offset_max"] == pytest.approx(0.475, abs=1e-3)
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
    assert result["templates"]["alternation"]["alternation_phase_offset_max"] < 0.01
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
    assert result["templates"]["alternation"]["template_coverage"] == 0.0
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


def test_pause_rule_uses_the_cadence_on_both_sides_not_the_episode_median():
    def pauses(periods):
        periods = np.asarray(periods, dtype=float)
        return (~stride_is_normal(periods, stride_references(periods, 3), 2.0)).nonzero()[0].tolist()

    assert pauses([1.0, 1.0, 1.0, 3.0, 1.0, 1.0]) == [3]  # a stop between normal strides
    assert pauses([1.0, 1.0, 1.0, 3.0, 3.0, 1.0, 1.0, 1.0]) == [3, 4]  # a second stop next to the first
    assert pauses([3.0, 1.0, 1.0, 1.0]) == [0]  # a stop before the first normal stride
    # walk-to-run: the stride halves; every stride matches the cadence of its own side
    assert pauses([0.95, 0.95, 0.95, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42]) == []
    assert pauses([0.946, 0.976, 0.436, 0.456, 0.47, 0.436]) == []  # only one walk stride before the change
    assert pauses([1.0]) == []
    time = np.arange(0.0, 8.0, 0.01)
    events = np.array([1.0, 1.95, 2.9, 3.32, 3.74, 4.16, 4.58, 5.0])
    _, defined = limb_phase(time, events, stride_references(np.diff(events), 3), 2.0)
    assert np.all(defined[(time >= 1.0) & (time < 5.0)])


def test_fore_aft_metrics_are_measured_in_the_travel_and_trunk_frames():
    walk = gait_trace((0.0, 0.5), duty=0.6, period=0.8, speed=0.8, jitter=0.01, seed=1)
    reference = metrics(walk, settle_s=1.0)
    for degrees in (5.0, -12.0, 30.0):
        turned = metrics(rotated(walk, degrees), settle_s=1.0)
        for name in ("step_length_over_leg_min", "stride_length_over_leg_min", "skid_fraction_max"):
            assert turned[name] == pytest.approx(reference[name], abs=2e-6), (degrees, name)
        for foot in ("r", "l"):
            assert turned["per_foot"][foot]["step_length_over_leg_median"] == pytest.approx(
                reference["per_foot"][foot]["step_length_over_leg_median"], abs=2e-6
            )
        # progress is still measured along the declared task direction
        assert turned["mean_speed_mps"] == pytest.approx(
            reference["mean_speed_mps"] * np.cos(np.radians(degrees)), 1e-3
        )
    assert reference["stride_length_over_leg_min"] == pytest.approx(0.64, abs=0.02)
    assert reference["step_length_over_leg_min"] == pytest.approx(0.32, abs=0.02)
    # an exact step-to gait walking a few degrees off the task axis never steps through
    step_to = gait_trace((0.0, 0.5), duty=0.6, period=0.8, speed=0.8, step_to=True)
    for degrees in (0.0, 1.5, -3.0):
        assert metrics(rotated(step_to, degrees), settle_s=1.0)["step_length_over_leg_min"] == pytest.approx(
            0.0, abs=1e-6
        )


def crabbed(trace, degrees):
    """The trunk travels ``degrees`` off its body axis: the path is sheared sideways while every
    foot keeps its body-frame placement (the trunk orientation stays on +x)."""
    out = {key: np.array(value, copy=True) for key, value in trace.items()}
    shear = float(np.tan(np.radians(degrees)))
    for key in ("foot_position_m", "root_position_m"):
        out[key][..., 1] += shear * out[key][..., 0]
    return out


@pytest.mark.parametrize("degrees", [3.0, 7.0, -10.0])
def test_a_crab_or_a_trunk_yaw_moves_one_frame_but_never_both(degrees):
    """Step length runs along the line of progression; step symmetry is judged in the travel or the trunk frame.

    A step-to crabbing a few degrees (the round-2 crab attacks: feet placed in the body frame while the
    trunk travels off its axis) projects its 0.6 L stance width onto the travel heading, so it seems to
    step through there, lopsidedly; along its trunk it is still a step-to. A step-to along the line of
    travel whose trunk is yawed instead (the walk-first round-1 attacks) is the mirror case. A genuine
    walk keeps symmetric steps in the frame it walks in, whichever way its trunk points.
    """
    # the trailing foot lands beside the leader: its step is the 0.6 L stance width projected on a frame
    # turned ``degrees`` from the body's lateral line
    shift = -0.6 * float(np.sin(np.radians(degrees)))
    sheared = metrics(
        crabbed(gait_trace((0.0, 0.5), duty=0.6, period=0.8, speed=0.8, step_to=True, width=0.6), degrees),
        settle_s=1.0,
    )
    assert sheared["trunk_crab_angle_deg_median"] == pytest.approx(-degrees, abs=0.05)
    assert sheared["step_length_trunk_over_leg_min"] == pytest.approx(0.0, abs=0.002)
    assert sheared["step_length_over_leg_min"] == pytest.approx(shift, abs=0.01)
    assert sheared["step_symmetry_trunk_min"] == 0.0
    assert sheared["step_symmetry"] == sheared["step_symmetry_travel_min"] < 0.25
    yawed = gait_trace((0.0, 0.5), duty=0.6, period=0.8, speed=0.8, step_to=True, width=0.6)
    yawed["root_quat_wxyz"] = yaw_quaternion(np.full(len(yawed["time_s"]), np.radians(degrees)))
    turned = metrics(yawed, settle_s=1.0)
    assert turned["step_length_over_leg_min"] == pytest.approx(0.0, abs=0.002)
    assert turned["step_length_trunk_over_leg_min"] == pytest.approx(shift, abs=0.01)
    # a genuine walk keeps symmetric steps along its travel whichever way its trunk points
    walk = gait_trace((0.0, 0.5), duty=0.6, period=0.8, speed=0.8, width=0.6)
    walk["root_quat_wxyz"] = yaw_quaternion(np.full(len(walk["time_s"]), np.radians(degrees)))
    walking = metrics(walk, settle_s=1.0)
    assert walking["step_length_over_leg_min"] == pytest.approx(0.32, abs=0.02)
    assert walking["step_symmetry_travel_min"] > 0.9
    assert walking["step_symmetry"] == walking["step_symmetry_travel_min"]
    # ... and along its trunk when the feet keep their body-frame placement on a crabbing path
    crab_walk = metrics(crabbed(gait_trace((0.0, 0.5), duty=0.6, period=0.8, speed=0.8, width=0.6), degrees))
    assert crab_walk["step_symmetry_trunk_min"] > 0.9
    assert crab_walk["step_symmetry"] == crab_walk["step_symmetry_trunk_min"]


def test_trunk_yaw_wobble_within_a_stride_is_averaged_out():
    walk = gait_trace((0.0, 0.5), duty=0.6, period=0.8, speed=0.8)
    wobbling = {key: np.array(value, copy=True) for key, value in walk.items()}
    wobbling["root_quat_wxyz"] = yaw_quaternion(np.radians(10.0) * np.sin(2.0 * np.pi * walk["time_s"] / 0.8))
    steady = metrics(walk, settle_s=1.0)
    wobble = metrics(wobbling, settle_s=1.0)
    assert wobble["step_length_over_leg_min"] == pytest.approx(steady["step_length_over_leg_min"], abs=0.01)


def test_swing_floor_contact_stride_length_and_girdle_shares_are_measured():
    clean = metrics(stepping_trace())
    assert clean["swing_slip_fraction_max"] == 0.0
    # only the lift-off and landing ends of the bell are below the 0.01 L scuff height
    assert 0.05 < clean["swing_ground_fraction_max"] < 0.15
    assert clean["stride_length_over_leg_min"] == pytest.approx(0.64, abs=0.01)
    assert clean["girdle_load_share_min"] is None
    assert clean["girdle_unloaded_fraction"] is None
    assert clean["glide_stance_fraction_max"] == 0.0
    assert clean["light_stance_fraction_max"] == 0.0
    walk = (((0.0, 0.5), 0.8, 0.6, 0.8),)
    dragged = metrics(scheduled_trace(lambda t: walk[0], swing_drag=0.002), settle_s=1.0)
    assert dragged["swing_ground_fraction_max"] > 0.75
    assert dragged["swing_slip_fraction_max"] > 0.5
    rearing = scheduled_trace(
        lambda t: ((0.0, 0.5, 0.5, 0.0), 0.6, (0.18, 0.18, 0.5, 0.5), 1.0), feet=4, load=(0.3, 0.3, 1.0, 1.0)
    )
    quad = metrics(rearing, foot_names=QUAD, settle_s=1.0)
    assert quad["girdle_duty_ratio"] == pytest.approx(0.36, abs=0.02)
    assert quad["girdle_load_share_min"] < 0.15
    assert quad["fore_load_share"] == quad["girdle_load_share_min"]
    assert quad["girdle_unloaded_fraction"] > 0.5  # the forelimbs carry under 12 % stride after stride
    trot = scheduled_trace(lambda t: ((0.0, 0.5, 0.5, 0.0), 0.6, 0.55, 1.0), feet=4)
    assert metrics(trot, foot_names=QUAD, settle_s=1.0)["girdle_unloaded_fraction"] == 0.0


def test_footprints_are_load_weighted_so_swing_retraction_toe_reach_and_slides_do_not_count():
    """A step-to gait whose trailing foot (a) swings 0.03 L past the planted foot and retracts to land
    level, (b) touches down lightly 0.06 L ahead and slides back before loading, or (c) lands 0.01 L
    behind and slides 0.03 L forward under load: its weight-bearing footprint never steps through."""
    step_to = gait_trace((0.0, 0.5), duty=0.6, period=0.8, speed=0.8, step_to=True)
    retracting = {key: np.array(value, copy=True) for key, value in step_to.items()}
    toe_reach = {key: np.array(value, copy=True) for key, value in step_to.items()}
    swinging = retracting["floor_force_n"][:, 1] == 0.0
    for start, end in boolean_runs(swinging):
        if start == 0 or end == len(swinging):
            continue
        fraction = (np.arange(start, end) - start) / (end - start)
        retracting["foot_position_m"][start:end, 1, 0] += 0.06 * np.exp(-(((fraction - 0.85) / 0.06) ** 2))
    ahead = retracting["foot_position_m"][:, 1, 0] - retracting["foot_position_m"][:, 0, 0]
    assert np.max(ahead) > 0.02  # the swinging foot does pass the planted one
    for start, end in boolean_runs(toe_reach["floor_force_n"][:, 1] > 0.0):
        if start == 0:
            continue
        reach = start + int(0.15 * (end - start))  # light toe contact ahead, sliding back
        fraction = (np.arange(start, reach) - start) / max(reach - start, 1)
        toe_reach["foot_position_m"][start:reach, 1, 0] += 0.06 * (1.0 - fraction)
        toe_reach["floor_force_n"][start:reach, 1] = 0.02 * BW
    for trace in (retracting, toe_reach):
        measured = metrics(trace, settle_s=1.0)
        assert measured["per_foot"]["l"]["step_length_over_leg_median"] == pytest.approx(0.0, abs=0.003)
        assert measured["step_length_over_leg_min"] == pytest.approx(0.0, abs=0.003)
    # a trailing foot that lands 0.01 L behind and slides 0.03 L forward under load is centred on its load
    sliding = {key: np.array(value, copy=True) for key, value in step_to.items()}
    for start, end in boolean_runs(sliding["floor_force_n"][:, 1] > 0.0):
        if start == 0:
            continue
        sliding["foot_position_m"][start:end, 1, 0] += -0.01 + 0.03 * np.linspace(0.0, 1.0, end - start)
    slid = metrics(sliding, settle_s=1.0)
    assert slid["per_foot"]["l"]["step_length_over_leg_median"] == pytest.approx(0.005, abs=0.003)


def test_standing_in_place_strides_glide_and_light_stance_are_measured():
    walk = ((0.0, 0.5), 0.5, 0.6, 1.0)
    # marking time: 1.5 s of in-place strides out of every 5 s, the rhythm unbroken
    marking = scheduled_trace(lambda t: ((0.0, 0.5), 0.5, 0.6, 0.02) if t % 5.0 >= 3.5 else walk)
    template = metrics(marking, settle_s=1.0)["templates"]["alternation"]
    assert template["in_place_stride_fraction"] == pytest.approx(0.3, abs=0.06)
    assert template["off_gait_fraction"] >= template["in_place_stride_fraction"]
    clean = metrics(scheduled_trace(lambda t: walk), settle_s=1.0)["templates"]["alternation"]
    assert clean["in_place_stride_fraction"] == clean["standing_fraction"] == clean["off_gait_fraction"] == 0.0
    # standing: the trunk stops for 0.25 s twice a second while the feet keep their clock
    frozen = scheduled_trace(lambda t: walk)
    stops = (frozen["time_s"] % 1.0) >= 0.75
    for start, end in boolean_runs(stops):
        frozen["root_position_m"][start:end] = frozen["root_position_m"][start]
        frozen["root_position_m"][end:, 0] -= frozen["root_position_m"][end, 0] - frozen["root_position_m"][start, 0]
    standing = metrics(frozen, settle_s=1.0)["templates"]["alternation"]["standing_fraction"]
    assert 0.15 <= standing <= 0.26
    # skating in one stride of four, at 0.9 of trunk speed; touchdown skid alone never glides
    skating = scheduled_trace(lambda t: walk + ((0.9,) if t % 2.0 >= 1.5 else (0.0,)))
    skated = metrics(skating, settle_s=1.0)
    assert skated["glide_stance_fraction_max"] == pytest.approx(0.25, abs=0.05)
    assert skated["skid_fraction_max"] < 0.35
    # a stance whose last 40 % only touches the floor (2 % of body weight)
    light = scheduled_trace(lambda t: walk)
    for start, end in boolean_runs(light["floor_force_n"][:, 1] > 0.0):
        light["floor_force_n"][start + int(0.6 * (end - start)) : end, 1] = 0.02 * BW
    assert metrics(light, settle_s=1.0)["per_foot"]["l"]["light_stance_fraction"] == pytest.approx(0.4, abs=0.03)


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
        "root_quat_wxyz",
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
    "problem",
    [
        "missing",
        "no_quaternion",
        "zero_quaternion",
        "reset",
        "shape",
        "transposed",
        "negative",
        "no_window",
        "feet",
        "weight",
    ],
)
def test_missing_or_invalid_telemetry_is_not_a_successful_measurement(problem):
    trace = stepping_trace()
    kwargs = {}
    if problem == "missing":
        del trace["slip_speed_mps"]
    elif problem == "no_quaternion":
        # the trunk-frame step-through test fails closed; there is no travel-heading fallback
        del trace["root_quat_wxyz"]
    elif problem == "zero_quaternion":
        trace["root_quat_wxyz"][50] = 0.0
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
    "phases,label,limb_phase",
    [
        ((0.0, 0.5, 0.5, 0.0), "trot", 0.5),
        ((0.0, 0.5, 0.0, 0.5), "pace", 0.0),
        ((0.25, 0.75, 0.0, 0.5), "walk", 0.25),
    ],
)
def test_quadruped_symmetrical_gaits_share_one_alternation_template_and_keep_their_labels(phases, label, limb_phase):
    result = metrics(gait_trace(phases, duty=0.7, period=1.0), foot_names=QUAD, settle_s=1.0)
    template = result["templates"]["alternation"]
    assert set(template["pairs"]) == {"fr>fl", "rr>rl"}
    assert template["alternation_phase_offset_max"] < 0.01
    assert template["off_gait_fraction"] == 0.0
    assert template["asymmetric_stride_fraction"] == 0.0
    # Hildebrand's limb phase and label stay report-only diagnostics
    assert min(result["limb_phase_mean"], 1.0 - result["limb_phase_mean"]) == pytest.approx(
        min(limb_phase, 1.0 - limb_phase), abs=0.01
    )
    assert label in result["gait_label"]


def test_quadruped_strides_with_both_pairs_off_anti_phase_are_asymmetric_off_gait_time():
    """A canter-like gait: both contralateral pairs land 0.3 cycle apart in every stride."""
    canter = gait_trace((0.3, 0.0, 0.3, 0.0), duty=0.6, period=0.8, jitter=0.005, seed=3)
    template = metrics(canter, foot_names=QUAD, settle_s=1.0)["templates"]["alternation"]
    assert template["asymmetric_stride_fraction"] > 0.9
    assert template["off_gait_fraction"] > 0.9
    # one pair off by itself is a stumble, not an asymmetrical stride
    limp = gait_trace((0.3, 0.0, 0.5, 0.0), duty=0.6, period=0.8, jitter=0.005, seed=3)
    assert metrics(limp, foot_names=QUAD, settle_s=1.0)["templates"]["alternation"]["asymmetric_stride_fraction"] == 0.0


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
        GaitProtocol(template_alternation_tolerance=0.4, off_gait_extra_tolerance=0.1)
    with pytest.raises(ValueError):
        GaitProtocol(chatter_fill_s=float("nan"))
    with pytest.raises(ValueError):
        GaitProtocol(pause_factor=1.0)
    with pytest.raises(ValueError):
        GaitProtocol(pause_neighbour_strides=0)
    with pytest.raises(ValueError):
        GaitProtocol(step_to_bout_steps=0)
    with pytest.raises(ValueError):
        GaitProtocol(heading_window_strides=0.0)
    with pytest.raises(ValueError):
        GaitProtocol(trunk_axis_window_strides=0.0)
    with pytest.raises(ValueError):
        GaitProtocol(standing_speed_fraction=1.0)
    with pytest.raises(ValueError):
        GaitProtocol(glide_skid_ratio=0.0)
    with pytest.raises(ValueError):
        GaitProtocol(light_load_bw_per_limb=0.005)
    with pytest.raises(ValueError):
        GaitProtocol(girdle_local_min_share=0.6)
    for option, step in (
        ("pause_neighbour_strides", 1),
        ("heading_window_strides", 1),
        ("trunk_axis_window_strides", 1),
        ("off_gait_bout_strides", 1),
        ("off_gait_extra_tolerance", 0.01),
        ("in_place_stride_over_leg", 0.01),
        ("standing_window_strides", 0.1),
        ("standing_speed_fraction", 0.01),
        ("step_through_min_over_leg", 0.01),
        ("glide_skid_ratio", 0.1),
        ("girdle_local_min_share", 0.01),
    ):
        changed = replace(PROTOCOL, **{option: getattr(PROTOCOL, option) + step})
        assert changed.sha256 != PROTOCOL.sha256


def test_a_stop_is_charged_once_on_the_moving_clock():
    """Walk-first round 2: the phase clock stops with the trunk, so a 1.2 s stop between 0.8 s strides is
    standing time only (no undefined, off-template or in-place time around it) and coverage stays whole."""
    walk = gait_trace((0.0, 0.5), duty=0.6, period=0.8, speed=0.8, jitter=0.01, seed=4)
    time = walk["time_s"]
    k = int(np.searchsorted(time, 5.0))
    hold = int(round(1.2 / float(time[1] - time[0])))
    stopped = {"time_s": time.copy()}
    for key, value in walk.items():
        if key != "time_s":
            frozen = np.repeat(value[k : k + 1], hold, axis=0)
            if key == "slip_speed_mps":
                frozen = np.zeros_like(frozen)
            stopped[key] = np.concatenate((value[:k], frozen, value[k:]))[: len(time)]
    template = metrics(stopped, settle_s=1.0)["templates"]["alternation"]
    assert template["standing_fraction"] == pytest.approx(1.2 / 9.0, abs=0.02)
    assert template["off_gait_fraction"] == pytest.approx(template["standing_fraction"], abs=0.005)
    assert template["undefined_phase_fraction"] == 0.0
    assert metrics(stopped, settle_s=1.0)["limb_phase_coverage_min"] == 1.0


def test_unloaded_time_is_flight_measured_on_load():
    """Feet in light contact (2 % of body weight) through a ballistic phase still leave the body unloaded."""
    run = gait_trace((0.0, 0.5), duty=0.35, period=0.4, speed=2.0)
    bridged = {key: np.array(value, copy=True) for key, value in run.items()}
    airborne = np.all(run["floor_force_n"] == 0.0, axis=1)
    bridged["floor_force_n"][airborne] = 0.01 * BW
    measured, reference = metrics(bridged, settle_s=1.0), metrics(run, settle_s=1.0)
    assert measured["flight_fraction"] == 0.0 < reference["flight_fraction"]
    assert measured["unloaded_fraction"] == pytest.approx(reference["flight_fraction"], abs=0.01)
