"""The opt-in observation (sensor) noise: zero-mean Gaussian noise per observation segment.

Off by default for every model (design record
``docs/TRICERATOPS_ROBOT_PORT_2026_11.md`` §7).  Two ``[env]`` keys, which a
species takes only by listing them in its constructor (a species that does
not list them raises ``TypeError`` if a TOML sets them):

* ``observation_noise_std`` -- ``None`` (off, the default) or a table of
  segment name to standard deviation, each in its segment's own units (rad,
  rad/s, m/s, N, ...): the switch.  The names are the plant contract's
  observation segments for the model's observation schema
  (``plant_contract.policy_layer.observation_segments``); a segment the table
  leaves out has no noise, and the command, the task's input, never has any.
  A TOML inline table, kept a plain ``dict`` (``SubprocVecEnv`` pickles the
  stage's kwargs).  A stage that ``extends`` another replaces a nested table
  whole, so it restates or inherits the whole table.
* ``observation_noise_eval`` -- ``False`` (the default) or ``True``: whether
  an env a trainer did not mark (every evaluation env) sees the noise too.

**Who sees it.** An episode is noised when the env is a trainer's rollout env
(``training_rollouts``, ``episode_start.mark_training_rollouts``) or
``observation_noise_eval`` is true, latched when the episode begins
(``BaseDinoEnv._begin_episode``), so a change of role takes effect at the next
reset.  The policy probe's state bank forces the eval flag off
(``policy_probe.bank_env_kwargs``).

**Where.** ``BaseDinoEnv.step`` adds the noise to the observation it returns,
after the command recompute (and the behaviour env's ``set_direction`` to the
observation it returns).  Never inside the species' ``_get_obs`` (hashed by
the plant contract, and run by its observation probe on synthetic state) and
never in ``reset`` (hashed as ``home_reset``): the reset observation of every
episode is the clean one, as plant validation and the probe are.  Off, or in
an episode that is not noised, ``step`` returns the very object ``_get_obs``
built.  Noise is added inside the env, before ``VecNormalize``, so its
``obs_rms`` learns the noisy distribution.

**The draw.** Each noised step draws ``std x standard_normal(width)`` over the
whole observation width, in observation order and in float64, so switching
one segment on or off never changes another's realisation.  Each unit
segment (``*_orientation_quaternion``, ``*_projected_gravity``,
``*_direction``, ``*_direction_body``) with a positive std is renormalised to
unit length (``v / max(|v|, 1e-9)``); the result is float32.  The draws come
from the noise's own stream (``episode_start.feature_stream`` with
``OBSERVATION_NOISE_STREAM_KEY``), never from ``np_random``, so every reset
draw and every physics step is the same with the noise on or off.  The draws
are per env, so ``symmetry.mirror_env_state`` refuses an env whose episode is
noised: its step observations would not be the mirror images of its source's.

Not fingerprinted: the task fingerprint records the keys in its ``env``
section and, while the noise is on, :data:`OBSERVATION_NOISE_IMPLEMENTATION`
in its ``observation_noise`` section.  Numpy only at the top level; the
segment table and the plant's observation schema are read where a noise is
resolved.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

import numpy as np

from .command_frame import COMMAND_SEGMENT_NAME

#: The ``[env]`` keys of the noise, in signature order; the first is the switch (``None``: off).
OBSERVATION_NOISE_ENV_KEYS = ("observation_noise_std", "observation_noise_eval")
#: Names what the noise means; the task fingerprint's ``observation_noise``
#: section records it.  Bump it on any change of meaning, not merely of shape,
#: as ``task_fingerprint.SCHEDULE_IMPLEMENTATION`` is bumped.
OBSERVATION_NOISE_IMPLEMENTATION = "obs_noise/v1+gaussian-per-segment+post-step+episode-stream+unit-segment-renormalize"
#: The name endings of the unit segments, renormalised after their noise.
UNIT_SEGMENT_SUFFIXES = ("_orientation_quaternion", "_projected_gravity", "_direction", "_direction_body")
#: The smallest norm a noisy unit segment is divided by.
UNIT_NORM_FLOOR = 1e-9


@dataclass(frozen=True)
class NoiseSegment:
    """One observation segment's slice and standard deviation."""

    name: str
    start: int
    width: int
    std: float
    #: A unit segment with a positive std: renormalised after its noise.
    renormalize: bool


@dataclass(frozen=True)
class ObservationNoise:
    """A validated noise: every segment of one env's observation, in order, with its standard deviation."""

    segments: tuple[NoiseSegment, ...]
    eval_noise: bool
    width: int = field(init=False)
    std: np.ndarray = field(init=False, repr=False, compare=False)
    unit_slices: tuple[slice, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        width = sum(segment.width for segment in self.segments)
        std = np.zeros(width, dtype=np.float64)
        for segment in self.segments:
            std[segment.start : segment.start + segment.width] = segment.std
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "std", std)
        object.__setattr__(
            self,
            "unit_slices",
            tuple(slice(s.start, s.start + s.width) for s in self.segments if s.renormalize),
        )

    @property
    def std_by_segment(self) -> dict[str, float]:
        """Every segment's standard deviation, by name, in observation order (0.0 where the table left it out)."""
        return {segment.name: segment.std for segment in self.segments}

    def apply(self, obs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """*obs* plus one draw of the noise from *rng*, the unit segments renormalised, as float32.

        Draws ``width`` standard normals whatever the table holds, so every
        segment's realisation is the same whichever others are on.
        """
        noisy = np.asarray(obs, dtype=np.float64) + self.std * rng.standard_normal(self.width)
        for block in self.unit_slices:
            noisy[block] /= max(float(np.linalg.norm(noisy[block])), UNIT_NORM_FLOOR)
        return noisy.astype(np.float32)


def observation_schema(env: Any) -> str:
    """The observation schema (``configs/plant_versions.toml``) of the manifest model *env* is an env of.

    The model is the one whose ``env_entrypoint`` class is nearest in
    ``type(env).__mro__``, so a subclass of a model's class (a behaviour env, a
    test subclass) reads that model's schema.
    """
    from .plant_contract.versions import load_plant_versions
    from .species_names import model_entry, model_ids

    entrypoints = {str(model_entry(model_id)["env_entrypoint"]): model_id for model_id in model_ids()}
    for cls in type(env).__mro__:
        model_id = entrypoints.get(f"{cls.__module__}:{cls.__qualname__}")
        if model_id is not None:
            return load_plant_versions()[1][model_id].observation_schema
    raise ValueError(
        f"observation_noise_std: {type(env).__qualname__} is not the env class of a manifest model, nor a subclass "
        "of one, so it has no observation schema to name its observation segments"
    )


def observation_layout(env: Any, schema: str | None = None) -> tuple[tuple[str, int], ...]:
    """Each observation segment of *env*'s observation, ``(name, width)``, in observation order.

    The names are the plant contract's (``policy_layer.observation_segments``,
    for *schema*, the model's own by default), and so are the widths, except
    ``foot_contact``'s, which is the number of the env's foot channels
    (``_foot_observation_groups`` where it declares them, else
    ``_foot_sensor_groups``): the legacy tables list every touch sensor there
    (the Brachiosaurus' eight touch sensors feed four channels).
    """
    from .plant_contract.policy_layer import _segment_width, observation_segments

    if schema is None:
        schema = observation_schema(env)
    declared = getattr(env, "_foot_observation_groups", None)
    channels = declared if declared is not None else getattr(env, "_foot_sensor_groups", ())
    layout = []
    for segment in observation_segments(env.model, env, schema):
        name = str(segment["name"])
        width = len(channels) if name == "foot_contact" else _segment_width(segment)
        layout.append((name, int(width)))
    return tuple(layout)


def resolve_observation_noise(
    env: Any, std_by_segment: Any, eval_noise: Any, *, schema: str | None = None
) -> ObservationNoise | None:
    """The validated noise of *env*'s observation, or ``None`` while it is off (*std_by_segment* ``None``).

    Refuses, with ``ValueError``: an eval flag that is not a bool (even
    while the table is off); the eval flag without a table; a table that is
    not a mapping; a layout whose widths do not sum to the observation width;
    a segment name the observation does not have (listing the valid names); a
    std that is a bool, not a number, negative or not finite, in float64 or
    in float32, the observation's dtype (``1e39`` would make every noisy
    observation ``inf``); a positive std on the command; and a table with no
    positive std (leave the table unset to turn the noise off).
    """
    if not isinstance(eval_noise, (bool, np.bool_)):
        raise ValueError(f"observation_noise_eval must be true or false, got {eval_noise!r}")
    if std_by_segment is None:
        if eval_noise:
            raise ValueError(
                "observation_noise_eval needs observation_noise_std: the table is the noise's switch (None is off)"
            )
        return None
    if not isinstance(std_by_segment, Mapping):
        raise ValueError(f"observation_noise_std must be a table of segment = std, got {std_by_segment!r}")
    layout = observation_layout(env, schema)
    width = int(np.prod(env.observation_space.shape))
    if sum(w for _, w in layout) != width:
        raise ValueError(
            f"observation_noise_std: the observation segments {list(layout)} sum to {sum(w for _, w in layout)}, "
            f"not the observation width {width}"
        )
    names = [name for name, _ in layout]
    if unknown := [key for key in std_by_segment if key not in names]:
        raise ValueError(
            f"observation_noise_std names {unknown!r}, not segments of this observation; the segments are {names}"
        )
    segments = []
    start = 0
    for name, segment_width in layout:
        value = std_by_segment.get(name, 0.0)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"observation_noise_std.{name} must be a number, got {value!r}")
        std = float(value)
        with np.errstate(over="ignore"):
            finite = math.isfinite(std) and bool(np.isfinite(np.float32(std)))
        if not finite or std < 0.0:
            raise ValueError(
                f"observation_noise_std.{name} must be finite and nonnegative, in float32 (the observation's dtype) "
                f"too, got {std!r}"
            )
        if name == COMMAND_SEGMENT_NAME and std > 0.0:
            raise ValueError(
                f"observation_noise_std.{name} = {std!r}: the command is the task's input, not a sensor, and is "
                "never noised"
            )
        renormalize = std > 0.0 and name.endswith(UNIT_SEGMENT_SUFFIXES)
        segments.append(NoiseSegment(name, start, segment_width, std, renormalize))
        start += segment_width
    if not any(segment.std > 0.0 for segment in segments):
        raise ValueError("observation_noise_std sets no positive std: leave it unset (None) to turn the noise off")
    return ObservationNoise(tuple(segments), bool(eval_noise))


__all__ = [
    "OBSERVATION_NOISE_ENV_KEYS",
    "OBSERVATION_NOISE_IMPLEMENTATION",
    "UNIT_NORM_FLOOR",
    "UNIT_SEGMENT_SUFFIXES",
    "NoiseSegment",
    "ObservationNoise",
    "observation_layout",
    "observation_schema",
    "resolve_observation_noise",
]
