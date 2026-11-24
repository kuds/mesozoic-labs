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
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any, ClassVar

import numpy as np

from environments.shared.command_frame import COMMAND_COMPONENTS, COMMAND_RANGE, COMMAND_WIDTH


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


def _stand_weight_pairs(value: Any) -> tuple[tuple[str, float], ...]:
    """*value* (a mapping or ``(name, weight)`` pairs) as pairs sorted by name; every name once, every weight >= 0."""
    if isinstance(value, Mapping):
        items = list(value.items())
    else:
        try:
            items = [tuple(item) for item in value]
        except TypeError as exc:
            raise ValueError("stand_weights must be a table of reward weights by name") from exc
    pairs: dict[str, float] = {}
    for item in items:
        if len(item) != 2:
            raise ValueError("stand_weights must be a table of reward weights by name")
        name, weight = item
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f"a stand_weights name must be a reward kwarg name, got {name!r}")
        if name in pairs:
            raise ValueError(f"stand_weights names {name!r} twice")
        if isinstance(weight, bool):
            raise ValueError(f"stand_weights[{name!r}] must be a number, got {weight!r}")
        number = _finite(weight, f"stand_weights[{name!r}]")
        if number < 0.0:
            raise ValueError(f"stand_weights[{name!r}] must be at least 0, got {number!r}")
        pairs[name] = number
    return tuple(sorted(pairs.items()))


def stand_weights_of(command_config: Any) -> dict[str, float]:
    """The stand block of a command config or of its record (``{name: weight}``); empty for ``None``."""
    if command_config is None:
        return {}
    if isinstance(command_config, DirectionCommandConfig):
        return command_config.stand_weight_map()
    if isinstance(command_config, Mapping):
        return dict(_stand_weight_pairs(command_config.get("stand_weights", ())))
    raise TypeError(f"a command config is a DirectionCommandConfig or its record, not {type(command_config).__name__}")


#: Speeds the tracking defaults scale with: a ``speed_scale`` of 1.5 m/s keeps them as stated.
TRACKING_REFERENCE_SPEED_SCALE = 1.5


@dataclass(frozen=True)
class DirectionCommandConfig:
    """A fixed task contract; sampled ranges do not redefine policy scaling.

    The default task holds 1.05 m/s and changes requested world heading by
    at most 30 degrees every four seconds.  Stops and variable speeds are
    opt-in.  ``straight_probability`` holds the previous requested heading
    at a switch; it does not look at the actual heading.

    ``initial_heading_offset_max`` (radians) turns the episode's first
    requested heading away from the spawn heading by U(-max, +max), drawn
    from the controller's own stream at reset; 0 draws nothing.  With
    ``turn_increment_max = 0`` and a switch interval longer than the
    episode, ``stop_probability`` acts only at reset, which gives
    whole-episode stand commands.

    The ``tracking_*`` fields parameterize the command-tracking reward and
    metrics ``BaseDinoEnv`` scores while the command is live:
    ``tracking_weight`` times :func:`gaussian_tracking_reward` with the two
    sigmas, and the :func:`tracking_metrics` tolerances.  A ``None`` speed
    sigma or tolerance scales with ``speed_scale``
    (:meth:`tracking_parameters`), so a small animal cannot pass a walk
    command while standing.

    ``stand_weights`` is the stand block (decision D-D62): reward weights,
    by constructor kwarg name, that ``BaseDinoEnv`` pays in place of the
    constructor's on a step whose command is exactly zero, so that a zero
    command means "stand".  Given as a mapping (a TOML
    ``[commands.stand_weights]`` table) or as ``(name, weight)`` pairs, it
    is held as pairs sorted by name; each weight is finite and at least 0.
    Which names an env accepts is the env's to say
    (``BaseDinoEnv`` refuses a name that is not one of its reward weights).

    ``cruise_ramp_start_fraction`` and ``cruise_ramp_timesteps`` are the
    cruise ramp (decision D-D62): training starts the commanded speeds
    (``cruise_speed`` and ``speed_range``) at the start fraction of these
    values and raises them linearly to 1 over the ramp's timesteps
    (``curriculum.cruise_ramp.CruiseRampCallback``).  The config holds the
    ramp's end state, which is the task the stage is certified on; the
    environment never reads the two fields.  Both off (1.0 and 0) or both
    on (a start fraction in (0, 1) and a positive budget).

    The fields after ``minimum_turn_speed_fraction`` came after the command
    manifest's first schema.  Each is recorded (:meth:`task_record`, which
    the command manifest and the task fingerprint read) only while it
    differs from :attr:`UNRECORDED_VALUES`, the value every task before it
    ran with, so adding one moved no recorded task.  The values are
    declared there, not read off the field defaults: a changed default
    moves the record.
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
    initial_heading_offset_max: float = 0.0
    tracking_weight: float = 2.5
    tracking_velocity_sigma: float | None = None
    tracking_yaw_rate_sigma: float = 0.15
    tracking_velocity_tolerance: float | None = None
    tracking_yaw_rate_tolerance: float = 0.15
    tracking_heading_tolerance: float = math.pi / 12.0
    tracking_stop_speed_tolerance: float | None = None
    tracking_stop_yaw_rate_tolerance: float = 0.15
    stand_weights: Any = ()
    cruise_ramp_start_fraction: float = 1.0
    cruise_ramp_timesteps: int = 0

    UNRECORDED_VALUES: ClassVar[dict[str, Any]] = {
        "initial_heading_offset_max": 0.0,
        "tracking_weight": 2.5,
        "tracking_velocity_sigma": None,
        "tracking_yaw_rate_sigma": 0.15,
        "tracking_velocity_tolerance": None,
        "tracking_yaw_rate_tolerance": 0.15,
        "tracking_heading_tolerance": math.pi / 12.0,
        "tracking_stop_speed_tolerance": None,
        "tracking_stop_yaw_rate_tolerance": 0.15,
        "stand_weights": (),
        "cruise_ramp_start_fraction": 1.0,
        "cruise_ramp_timesteps": 0,
    }

    def __post_init__(self) -> None:
        for name in (
            "cruise_speed",
            "stop_probability",
            "turn_increment_max",
            "straight_probability",
            "switch_jitter_s",
            "minimum_turn_speed_fraction",
            "initial_heading_offset_max",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        for name in (
            "switch_interval_s",
            "speed_scale",
            "lateral_speed_scale",
            "yaw_rate_scale",
            "yaw_gain",
            "yaw_rate_max",
            "tracking_weight",
            "tracking_yaw_rate_sigma",
            "tracking_yaw_rate_tolerance",
            "tracking_heading_tolerance",
            "tracking_stop_yaw_rate_tolerance",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        for name in ("tracking_velocity_sigma", "tracking_velocity_tolerance", "tracking_stop_speed_tolerance"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, _positive(getattr(self, name), name))
        if not 0.0 <= self.initial_heading_offset_max <= math.pi:
            raise ValueError("initial_heading_offset_max must lie in [0, pi]")
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
        object.__setattr__(self, "stand_weights", _stand_weight_pairs(self.stand_weights))
        fraction = _finite(self.cruise_ramp_start_fraction, "cruise_ramp_start_fraction")
        object.__setattr__(self, "cruise_ramp_start_fraction", fraction)
        timesteps = self.cruise_ramp_timesteps
        if isinstance(timesteps, bool) or not isinstance(timesteps, (int, np.integer)) or timesteps < 0:
            raise ValueError("cruise_ramp_timesteps must be a whole number of timesteps, at least 0")
        object.__setattr__(self, "cruise_ramp_timesteps", int(timesteps))
        if not 0.0 < fraction <= 1.0:
            raise ValueError("cruise_ramp_start_fraction must lie in (0, 1]")
        if (fraction < 1.0) != (self.cruise_ramp_timesteps > 0):
            raise ValueError(
                "a cruise ramp needs both a cruise_ramp_start_fraction below 1 and a positive "
                "cruise_ramp_timesteps; with neither the commanded speeds start at their end state"
            )

    def task_record(self) -> dict[str, Any]:
        """Every field, except one that holds its :attr:`UNRECORDED_VALUES` value (see the class docstring).

        A recorded ``stand_weights`` is an object, ``{name: weight}``.
        """
        record = asdict(self)
        for name, value in self.UNRECORDED_VALUES.items():
            if record[name] == value:
                del record[name]
        if "stand_weights" in record:
            record["stand_weights"] = dict(self.stand_weights)
        return record

    def field_record(self) -> dict[str, Any]:
        """Every field, as a JSON record holds it: a range as a list and the stand weights as an object.

        ``DirectionCommandConfig(**config.field_record())`` rebuilds the
        config exactly (``config.env_kwargs_record`` records a stage's
        config this way).
        """
        record = asdict(self)
        record["speed_range"] = None if self.speed_range is None else list(self.speed_range)
        record["stand_weights"] = dict(self.stand_weights)
        return record

    def stand_weight_map(self) -> dict[str, float]:
        """The stand block as ``{name: weight}``; empty when the config declares none."""
        return dict(self.stand_weights)

    @property
    def has_cruise_ramp(self) -> bool:
        """Whether training ramps the commanded speeds in (``cruise_ramp_timesteps > 0``)."""
        return self.cruise_ramp_timesteps > 0

    def ramped(self, fraction: float) -> "DirectionCommandConfig":
        """This config with ``cruise_speed`` and ``speed_range`` scaled by *fraction* in (0, 1].

        Every other field is kept, so the stop probability, the heading
        offset, the scales and the tracking parameters (whose speed defaults
        follow ``speed_scale``, not the cruise) are the end state's: a
        ramped config draws its stops on the same seeds as the end state,
        and each cruise speed is *fraction* of the end state's, to rounding.
        """
        fraction = _finite(fraction, "fraction")
        if not 0.0 < fraction <= 1.0:
            raise ValueError("a ramp fraction must lie in (0, 1]")
        speed_range = None
        if self.speed_range is not None:
            speed_range = (fraction * self.speed_range[0], fraction * self.speed_range[1])
        return replace(self, cruise_speed=fraction * self.cruise_speed, speed_range=speed_range)

    def tracking_parameters(self) -> dict[str, Any]:
        """The tracking reward's weight and sigmas and the metrics' tolerances, each ``None`` resolved.

        The speed defaults are fractions of ``speed_scale /``
        :data:`TRACKING_REFERENCE_SPEED_SCALE`: a velocity sigma of 0.25, a
        velocity tolerance of 0.2 and a stop-speed tolerance of 0.1 of it.
        ``tolerances`` is the keyword set :func:`tracking_metrics` takes.
        """
        ratio = self.speed_scale / TRACKING_REFERENCE_SPEED_SCALE

        def scaled(value: float | None, fraction: float) -> float:
            return fraction * ratio if value is None else value

        return {
            "weight": self.tracking_weight,
            "velocity_sigma": scaled(self.tracking_velocity_sigma, 0.25),
            "yaw_rate_sigma": self.tracking_yaw_rate_sigma,
            "tolerances": {
                "velocity_tolerance": scaled(self.tracking_velocity_tolerance, 0.2),
                "yaw_rate_tolerance": self.tracking_yaw_rate_tolerance,
                "heading_tolerance": self.tracking_heading_tolerance,
                "stop_speed_tolerance": scaled(self.tracking_stop_speed_tolerance, 0.1),
                "stop_yaw_rate_tolerance": self.tracking_stop_yaw_rate_tolerance,
            },
        }


def validate_command_task(mode: str, config: Any) -> None:
    """Refuse a live command task an environment could not run: raise ``ValueError`` or ``TypeError``.

    *mode* must be a live mode (``"heading"`` or ``"heading_and_speed"``)
    and *config* a :class:`DirectionCommandConfig`; ``"heading"`` holds
    ``cruise_speed``, so a speed range or stops need ``"heading_and_speed"``.
    """
    if mode not in ("heading", "heading_and_speed"):
        raise ValueError(f"command mode {mode!r} is not a live mode ('heading' or 'heading_and_speed')")
    if not isinstance(config, DirectionCommandConfig):
        raise TypeError(f"command_mode={mode!r} needs a command_config (a DirectionCommandConfig)")
    if mode == "heading" and (config.speed_range is not None or config.stop_probability > 0.0):
        raise ValueError("command_mode='heading' holds cruise_speed: speed_range and stops need 'heading_and_speed'")


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
            if values.shape != (COMMAND_WIDTH,) or not np.all(np.isfinite(values)):
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

    def replace_config(self, config: DirectionCommandConfig) -> None:
        """Draw from *config* from the next draw on (a speed, a turn or an interval); the current request stands.

        The cruise ramp's swap (``BaseDinoEnv.set_command_config``, which
        checks what may differ).  A whole-episode walk draws only at reset,
        so the swap takes effect at the next episode.
        """
        if not isinstance(config, DirectionCommandConfig):
            raise TypeError("config must be a DirectionCommandConfig")
        self.config = config

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
            "config": self.config.task_record(),
            "request_frame": "world_heading_radians_and_speed_metres_per_second",
            "policy_frame": "gravity_aligned_body_heading_vx_vy_yaw_rate",
            "policy_components": list(COMMAND_COMPONENTS),
            "policy_units": ["m/s", "m/s", "rad/s"],
            "time_unit": "seconds",
            "heading_wrap": "[-pi, pi); exact half-turn clockwise",
            "schedule": "PCG64; increments relative to previous requested world heading",
            "adapter": "bounded_proportional_yaw_with_optional_cosine_speed_reduction/v1",
            "stop": "zero forward/lateral/yaw commands; heading masked",
            "external_target": "persists until reset",
        }

    def reset(self, rng: np.random.Generator, initial_heading: float) -> DirectionCommandState:
        """Start an episode with a dedicated RNG seed.

        The first request holds *initial_heading*, turned by the
        ``initial_heading_offset_max`` draw when that is positive.
        """
        heading = wrap_angle(initial_heading)
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        self._schedule_seed = int(rng.integers(0, 2**32, dtype=np.uint64))
        self._rng = np.random.Generator(np.random.PCG64(self._schedule_seed))
        self._events = []
        self._last_time_s = 0.0
        self._desired_heading = heading
        if self.config.initial_heading_offset_max > 0.0:
            offset_max = self.config.initial_heading_offset_max
            self._desired_heading = wrap_angle(heading + float(self._rng.uniform(-offset_max, offset_max)))
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
            normalized=np.clip(physical / scales, *COMMAND_RANGE),
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
