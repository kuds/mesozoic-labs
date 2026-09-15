"""World-direction requests mapped onto the existing three command inputs.

This module has no simulator or training dependencies.  The policy receives
forward/lateral speed in the gravity-aligned body heading frame and a turn
rate, never an angle in a velocity slot.  All public angles are radians,
speeds are metres/second, rates are radians/second and times are seconds.

Episode schedules depend on their own seeded random generator and the
previous *requested* heading, not on the animal's actions.  Two controllers
with the same reset seed therefore see the same requests even if they move
differently or are updated at different frequencies.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


def _finite(value: float, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _positive(value: float, name: str) -> float:
    number = _finite(value, name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def wrap_angle(angle: float) -> float:
    """Wrap radians to [-pi, pi); an exact half-turn goes clockwise."""
    return (_finite(angle, "angle") + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class DirectionCommandConfig:
    """A fixed task contract; sampled ranges do not redefine policy scaling.

    The default task holds 1.05 m/s and changes requested world heading by
    at most 30 degrees every four seconds.  Stops and variable speeds are
    opt-in.  ``straight_probability`` holds the previous requested heading
    at a switch; it does not look at the actual heading.
    """

    cruise_speed: float = 1.05
    speed_range: tuple[float, float] | None = None
    stop_probability: float = 0.0
    turn_increment_max: float = math.pi / 6.0
    straight_probability: float = 0.25
    switch_interval_s: float = 4.0
    switch_jitter_s: float = 0.0
    speed_scale: float = 1.5
    lateral_speed_scale: float = 1.0
    yaw_rate_scale: float = 0.6
    yaw_gain: float = 1.5
    yaw_rate_max: float = 0.3
    turn_slowdown: bool = True
    minimum_turn_speed_fraction: float = 0.25

    def __post_init__(self) -> None:
        for name in (
            "cruise_speed",
            "stop_probability",
            "turn_increment_max",
            "straight_probability",
            "switch_jitter_s",
            "minimum_turn_speed_fraction",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        for name in (
            "switch_interval_s",
            "speed_scale",
            "lateral_speed_scale",
            "yaw_rate_scale",
            "yaw_gain",
            "yaw_rate_max",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        for name in ("stop_probability", "straight_probability", "minimum_turn_speed_fraction"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if not 0.0 <= self.cruise_speed <= self.speed_scale:
            raise ValueError("cruise_speed must lie in [0, speed_scale]")
        if not 0.0 <= self.turn_increment_max <= math.pi:
            raise ValueError("turn_increment_max must lie in [0, pi]")
        if not 0.0 <= self.switch_jitter_s < self.switch_interval_s:
            raise ValueError("switch_jitter_s must lie in [0, switch_interval_s)")
        if self.yaw_rate_max > self.yaw_rate_scale:
            raise ValueError("yaw_rate_max must not exceed yaw_rate_scale")
        if not isinstance(self.turn_slowdown, bool):
            raise ValueError("turn_slowdown must be a bool")
        if self.speed_range is not None:
            try:
                if len(self.speed_range) != 2:
                    raise ValueError("speed_range must contain exactly two speeds")
                bounds = tuple(_finite(value, "speed_range") for value in self.speed_range)
            except TypeError as exc:
                raise ValueError("speed_range must contain exactly two speeds") from exc
            if not 0.0 <= bounds[0] <= bounds[1] <= self.speed_scale:
                raise ValueError("speed_range must satisfy 0 <= low <= high <= speed_scale")
            object.__setattr__(self, "speed_range", bounds)


@dataclass(frozen=True)
class DirectionCommandState:
    """One command snapshot.  Array values are immutable float32 copies.

    A stop has ``heading_active=False`` and a masked zero ``heading_error``;
    consumers must preserve that mask when aggregating heading metrics.
    """

    time_s: float
    desired_heading: float
    desired_speed: float
    current_heading: float
    heading_error: float
    heading_active: bool
    physical: np.ndarray
    normalized: np.ndarray
    event_id: int
    event_start_s: float
    target_source: str

    def __post_init__(self) -> None:
        for name in ("physical", "normalized"):
            values = np.array(getattr(self, name), dtype=np.float32, copy=True)
            if values.shape != (3,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain three finite command components")
            values.setflags(write=False)
            object.__setattr__(self, name, values)

    def as_info(self) -> dict[str, float | int | bool | str]:
        """Scalar fields suitable for simulator ``info`` or event logging."""
        return {
            "desired_heading": self.desired_heading,
            "desired_speed": self.desired_speed,
            "actual_heading": self.current_heading,
            "heading_error_rad": self.heading_error,
            "heading_active": self.heading_active,
            "command_v_x": float(self.physical[0]),
            "command_v_y": float(self.physical[1]),
            "command_yaw_rate": float(self.physical[2]),
            "command_v_x_normalized": float(self.normalized[0]),
            "command_v_y_normalized": float(self.normalized[1]),
            "command_yaw_rate_normalized": float(self.normalized[2]),
            "command_time_s": self.time_s,
            "command_event_id": self.event_id,
            "command_event_start_s": self.event_start_s,
            "command_target_source": self.target_source,
        }


class DirectionCommandController:
    """Seeded training requests with a persistent external-control override.

    Call ``reset(rng, initial_heading)`` once after the environment's existing
    reset draws.  Call ``update(time_s, current_heading)`` at each control
    boundary.  ``set_target`` overrides automatic switching until reset;
    pass its optional ``time_s`` to timestamp a user action exactly, or omit
    it immediately after ``update`` to use that update's timestamp.
    """

    def __init__(self, config: DirectionCommandConfig | None = None) -> None:
        self.config = config if config is not None else DirectionCommandConfig()
        if not isinstance(self.config, DirectionCommandConfig):
            raise TypeError("config must be a DirectionCommandConfig")
        self._rng: np.random.Generator | None = None
        self._schedule_seed: int | None = None
        self._events: list[dict[str, Any]] = []
        self._last_time_s = 0.0
        self._next_switch_s = 0.0
        self._desired_heading = 0.0
        self._desired_speed = 0.0
        self._external_override = False

    @property
    def schedule_seed(self) -> int | None:
        return self._schedule_seed

    @property
    def events(self) -> list[dict[str, Any]]:
        """Copies of activated events, never future policy-dependent samples."""
        return [dict(event) for event in self._events]

    def manifest(self) -> dict[str, Any]:
        """Stable task description; episode-specific data lives in ``events``."""
        return {
            "schema": "mesozoic.direction-commands/v1",
            "config": asdict(self.config),
            "request_frame": "world_heading_radians_and_speed_metres_per_second",
            "policy_frame": "gravity_aligned_body_heading_vx_vy_yaw_rate",
            "policy_components": ["v_x_cmd", "v_y_cmd", "yaw_rate_cmd"],
            "policy_units": ["m/s", "m/s", "rad/s"],
            "time_unit": "seconds",
            "heading_wrap": "[-pi, pi); exact half-turn clockwise",
            "schedule": "PCG64; increments relative to previous requested world heading",
            "adapter": "bounded_proportional_yaw_with_optional_cosine_speed_reduction/v1",
            "stop": "zero forward/lateral/yaw commands; heading masked",
            "external_target": "persists until reset",
        }

    def reset(self, rng: np.random.Generator, initial_heading: float) -> DirectionCommandState:
        """Start an episode with a straight request and a dedicated RNG seed."""
        heading = wrap_angle(initial_heading)
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        self._schedule_seed = int(rng.integers(0, 2**32, dtype=np.uint64))
        self._rng = np.random.Generator(np.random.PCG64(self._schedule_seed))
        self._events = []
        self._last_time_s = 0.0
        self._desired_heading = heading
        self._desired_speed = self._sample_speed()
        self._external_override = False
        self._append_event(0.0, "schedule")
        self._next_switch_s = self._sample_interval()
        return self._state(0.0, heading)

    def update(self, time_s: float, current_heading: float) -> DirectionCommandState:
        """Advance all elapsed schedule events and adapt to the actual heading."""
        self._require_reset()
        time_s = self._validate_time(time_s)
        heading = wrap_angle(current_heading)
        if not self._external_override:
            # Catching up produces the same requests as visiting every boundary.
            while time_s + 1e-12 >= self._next_switch_s:
                self._desired_heading = wrap_angle(self._desired_heading + self._sample_turn())
                self._desired_speed = self._sample_speed()
                self._append_event(self._next_switch_s, "schedule")
                self._next_switch_s += self._sample_interval()
        self._last_time_s = time_s
        return self._state(time_s, heading)

    def set_target(self, heading: float, speed: float, *, time_s: float | None = None) -> None:
        """Hold an external world request until reset, without clipping its units.

        A heading remains stored during a stop but has no control objective.
        The provided timestamp must not precede the latest control update.
        Call ``update`` next to obtain the adapted policy command.
        """
        self._require_reset()
        heading = wrap_angle(heading)
        speed = _finite(speed, "speed")
        if not 0.0 <= speed <= self.config.speed_scale:
            raise ValueError("speed must lie in [0, speed_scale]")
        event_time = self._last_time_s if time_s is None else self._validate_time(time_s)
        self._desired_heading = heading
        self._desired_speed = speed
        self._last_time_s = event_time
        self._external_override = True
        self._append_event(event_time, "external")

    def _require_reset(self) -> None:
        if self._rng is None:
            raise RuntimeError("reset the command controller before using it")

    def _validate_time(self, time_s: float) -> float:
        time_s = _finite(time_s, "time_s")
        if time_s < self._last_time_s:
            raise ValueError("time_s must not go backwards; reset before a new episode")
        return time_s

    def _sample_speed(self) -> float:
        assert self._rng is not None
        probability = self.config.stop_probability
        if probability >= 1.0 or (probability > 0.0 and self._rng.random() < probability):
            return 0.0
        bounds = self.config.speed_range
        if bounds is None:
            return self.config.cruise_speed
        if bounds[0] == bounds[1]:
            return bounds[0]
        return float(self._rng.uniform(*bounds))

    def _sample_turn(self) -> float:
        assert self._rng is not None
        probability = self.config.straight_probability
        if self.config.turn_increment_max == 0.0 or probability >= 1.0:
            return 0.0
        if probability > 0.0 and self._rng.random() < probability:
            return 0.0
        return float(self._rng.uniform(-self.config.turn_increment_max, self.config.turn_increment_max))

    def _sample_interval(self) -> float:
        assert self._rng is not None
        jitter = self.config.switch_jitter_s
        offset = float(self._rng.uniform(-jitter, jitter)) if jitter > 0.0 else 0.0
        return self.config.switch_interval_s + offset

    def _append_event(self, time_s: float, source: str) -> None:
        self._events.append(
            {
                "event_id": len(self._events),
                "start_time_s": float(time_s),
                "desired_heading": self._desired_heading,
                "desired_speed": self._desired_speed,
                "heading_active": self._desired_speed > 0.0,
                "source": source,
                "schedule_seed": self._schedule_seed,
            }
        )

    def _state(self, time_s: float, heading: float) -> DirectionCommandState:
        active = self._desired_speed > 0.0
        error = wrap_angle(self._desired_heading - heading) if active else 0.0
        forward_speed = self._desired_speed
        yaw_rate = 0.0
        if active:
            yaw_rate = float(np.clip(self.config.yaw_gain * error, -self.config.yaw_rate_max, self.config.yaw_rate_max))
            if self.config.turn_slowdown:
                forward_speed *= max(self.config.minimum_turn_speed_fraction, math.cos(error))
        physical = np.array([forward_speed, 0.0, yaw_rate], dtype=np.float32)
        scales = np.array(
            [self.config.speed_scale, self.config.lateral_speed_scale, self.config.yaw_rate_scale], dtype=np.float64
        )
        event = self._events[-1]
        return DirectionCommandState(
            time_s=time_s,
            desired_heading=self._desired_heading,
            desired_speed=self._desired_speed,
            current_heading=heading,
            heading_error=error,
            heading_active=active,
            physical=physical,
            normalized=np.clip(physical / scales, -1.0, 1.0),
            event_id=event["event_id"],
            event_start_s=event["start_time_s"],
            target_source=event["source"],
        )


def body_frame_velocity(world_velocity: np.ndarray, heading: float) -> np.ndarray:
    """Project 2-D/3-D world velocity into the planar body heading frame.

    The returned two components are forward and leftward speed.  Vertical
    velocity is ignored intentionally; this is a gravity-aligned frame,
    not a frame that tilts with the pelvis on a slope.
    """
    velocity = np.asarray(world_velocity, dtype=np.float64)
    if velocity.shape not in ((2,), (3,)) or not np.all(np.isfinite(velocity)):
        raise ValueError("world_velocity must contain two or three finite components")
    heading = wrap_angle(heading)
    cosine, sine = math.cos(heading), math.sin(heading)
    return np.array(
        [cosine * velocity[0] + sine * velocity[1], -sine * velocity[0] + cosine * velocity[1]], dtype=np.float64
    )


def tracking_metrics(
    state: DirectionCommandState,
    world_velocity: np.ndarray,
    actual_yaw_rate: float,
    *,
    current_heading: float | None = None,
    velocity_tolerance: float = 0.2,
    yaw_rate_tolerance: float = 0.15,
    heading_tolerance: float = math.pi / 12.0,
    stop_speed_tolerance: float = 0.1,
    stop_yaw_rate_tolerance: float = 0.15,
) -> dict[str, float | bool]:
    """Measure both emitted-command tracking and the user's requested goal.

    ``actual_yaw_rate`` must use the same heading-rate definition as the
    controller, not an untransformed tilted sensor's z gyro component.
    Pass ``current_heading`` when measuring after a physics step: the
    state's physical command is the command that was executed, while
    velocity projection and heading error use the new measured heading.
    This function evaluates one sample, not settle/dwell or episode success.
    At a stop, heading is masked and translational speed plus actual turn
    rate determine whether the animal is stopped.
    """
    actual_yaw_rate = _finite(actual_yaw_rate, "actual_yaw_rate")
    tolerances = {
        "velocity_tolerance": velocity_tolerance,
        "yaw_rate_tolerance": yaw_rate_tolerance,
        "heading_tolerance": heading_tolerance,
        "stop_speed_tolerance": stop_speed_tolerance,
        "stop_yaw_rate_tolerance": stop_yaw_rate_tolerance,
    }
    for name, value in tolerances.items():
        _positive(value, name)
    actual_heading = state.current_heading if current_heading is None else wrap_angle(current_heading)
    velocity = body_frame_velocity(world_velocity, actual_heading)
    error_v = float(np.linalg.norm(velocity - state.physical[:2]))
    error_yaw = abs(actual_yaw_rate - float(state.physical[2]))
    speed = float(np.linalg.norm(velocity))
    error_heading = abs(wrap_angle(state.desired_heading - actual_heading)) if state.heading_active else 0.0
    requested_speed_error = abs(speed - state.desired_speed)
    command_ok = error_v <= velocity_tolerance and error_yaw <= yaw_rate_tolerance
    if state.heading_active:
        goal_ok = command_ok and error_heading <= heading_tolerance and requested_speed_error <= velocity_tolerance
    else:
        goal_ok = speed <= stop_speed_tolerance and abs(actual_yaw_rate) <= stop_yaw_rate_tolerance
    return {
        "actual_v_x": float(velocity[0]),
        "actual_v_y": float(velocity[1]),
        "actual_speed": speed,
        "actual_heading": actual_heading,
        "actual_yaw_rate": actual_yaw_rate,
        "tracking_error_v": error_v,
        "tracking_error_yaw": error_yaw,
        "tracking_error_heading": error_heading,
        "tracking_heading_active": state.heading_active,
        "tracking_error_requested_speed": requested_speed_error,
        "tracking_command_in_tolerance": bool(command_ok),
        "tracking_in_tolerance": bool(goal_ok),
    }


def gaussian_tracking_reward(
    velocity_error: float,
    yaw_rate_error: float,
    *,
    velocity_sigma: float = 0.25,
    yaw_rate_sigma: float = 0.15,
) -> float:
    """Gaussian command-tracking quality in [0, 1], including zero-speed stops.

    Errors are in m/s and rad/s, respectively.  The caller combines this
    dimensionless quality with balance, effort and optional heading rewards.
    It is not itself a direction-following success criterion.
    """
    velocity_error = _finite(velocity_error, "velocity_error")
    yaw_rate_error = _finite(yaw_rate_error, "yaw_rate_error")
    if velocity_error < 0.0 or yaw_rate_error < 0.0:
        raise ValueError("tracking errors must be nonnegative magnitudes")
    velocity_sigma = _positive(velocity_sigma, "velocity_sigma")
    yaw_rate_sigma = _positive(yaw_rate_sigma, "yaw_rate_sigma")
    scaled_v = velocity_error / velocity_sigma
    scaled_yaw = yaw_rate_error / yaw_rate_sigma
    # Multiplication saturates to inf for enormous finite errors; exp(-inf)
    # then correctly returns zero rather than raising a power overflow.
    return math.exp(-(scaled_v * scaled_v + scaled_yaw * scaled_yaw))
