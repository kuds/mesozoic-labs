"""The opt-in actuation delay: every control target reaches ``data.ctrl`` a whole number of physics steps late.

Off by default for every model (design record
``docs/TRICERATOPS_ROBOT_PORT_2026_11.md`` §5).  Two ``[env]`` keys, both
seconds, which a species takes only by listing them in its constructor (a
species that does not list them raises ``TypeError`` if a TOML sets them):

* ``actuation_delay_range_s`` -- ``None`` (off, the default) or ``(lo, hi)``:
  the switch, and the distribution a trainer's rollout env draws each
  episode's delay from, uniform over the whole physics steps in ``[lo, hi]``.
* ``actuation_delay_eval_s`` -- ``None`` (the default) or one value: the fixed
  delay of every env that is not a trainer rollout env
  (``episode_start.mark_training_rollouts``).  ``None`` draws from the range
  there too.  It need not lie in the range.

Each value must be shorter than an episode: a delay of at least
``max_episode_steps x frame_skip`` physics steps is refused, naming it in
seconds and in steps, since no target would reach the plant before the
episode ends (the mark of a value typed in milliseconds).

``BaseDinoEnv.step`` keeps a FIFO of the nu-wide ctrl, one entry per physics
step: at the episode's first step it is primed with ``data.ctrl`` as reset left
it (the reset keyframe's ``key_ctrl``), and immediately before each
``mj_step`` the step's target is appended and the oldest entry written to
``data.ctrl``.  A target set at control step n therefore reaches the plant
exactly k physics steps later, and substeps 0 to k-1 of an episode hold the
prime.  k = 0 is bit-identical to the delay off; k may exceed ``frame_skip``.
The FIFO never touches ``env.model``, so plant validation holds at any time.

Not fingerprinted: the task fingerprint records the keys in its ``env``
section and, while the delay is on, :data:`ACTUATION_DELAY_IMPLEMENTATION` in
its ``actuation_delay`` section.  Numpy only.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np

#: The ``[env]`` keys of the delay, in signature order.
ACTUATION_DELAY_ENV_KEYS = ("actuation_delay_range_s", "actuation_delay_eval_s")
#: Names what the delay means; the task fingerprint's ``actuation_delay``
#: section records it.  Bump it on any change of meaning, not merely of shape,
#: as ``task_fingerprint.SCHEDULE_IMPLEMENTATION`` is bumped.
ACTUATION_DELAY_IMPLEMENTATION = "ctrl_fifo/v1+per-physics-step+zero-order-hold+reset-ctrl-prime+episode-stream"

#: The largest gap, in seconds, between a delay and its nearest whole number of physics steps.
_WHOLE_STEP_TOLERANCE_S = 1e-9


@dataclass(frozen=True)
class ActuationDelay:
    """A validated delay configuration, in seconds and in whole physics steps of *timestep*."""

    range_s: tuple[float, float]
    eval_s: float | None
    timestep: float
    range_steps: tuple[int, int]
    eval_steps: int | None


def delay_steps(seconds: Any, timestep: float, *, name: str) -> int:
    """*seconds* as a whole number of physics steps of *timestep*.

    Refuses a bool, a non-number, a non-finite or negative value, and a value
    more than 1e-9 s from a whole number of steps, naming the timestep and the
    nearest whole value (0.010 s is 10 steps at 1 ms and 5 at 2 ms; 0.003 s at
    2 ms is refused).
    """
    if isinstance(seconds, bool) or not isinstance(seconds, Real):
        raise ValueError(f"{name} must be a number of seconds, got {seconds!r}")
    value = float(seconds)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative, got {value!r}")
    steps = round(value / timestep)
    if abs(steps * timestep - value) > _WHOLE_STEP_TOLERANCE_S:
        raise ValueError(
            f"{name} = {value!r} s is not a whole number of physics steps of {timestep!r} s; "
            f"the nearest whole value is {steps} steps = {round(steps * timestep, 12)!r} s"
        )
    return int(steps)


def validate_actuation_delay(
    range_s: Any, eval_s: Any, timestep: float, *, episode_steps: int | None = None
) -> ActuationDelay | None:
    """The validated delay, or ``None`` while it is off (*range_s* ``None``).

    Refuses an eval value without a range, a range that is a string or not a
    pair, ``lo > hi``, every value :func:`delay_steps` refuses, and, given
    *episode_steps* (the physics steps of an episode, ``max_episode_steps x
    frame_skip``), any value of at least that many steps: no target would
    reach the plant within an episode.
    """
    if range_s is None:
        if eval_s is not None:
            raise ValueError(
                "actuation_delay_eval_s needs actuation_delay_range_s: the range is the delay's switch (None is off)"
            )
        return None
    if isinstance(range_s, (str, bytes)) or not hasattr(range_s, "__len__") or len(range_s) != 2:
        raise ValueError(f"actuation_delay_range_s must be a pair (lo, hi) of seconds, got {range_s!r}")
    lo_steps = delay_steps(range_s[0], timestep, name="actuation_delay_range_s[0]")
    hi_steps = delay_steps(range_s[1], timestep, name="actuation_delay_range_s[1]")
    lo, hi = float(range_s[0]), float(range_s[1])
    if lo > hi:
        raise ValueError(f"actuation_delay_range_s must have lo <= hi, got {range_s!r}")
    eval_steps = None if eval_s is None else delay_steps(eval_s, timestep, name="actuation_delay_eval_s")
    if episode_steps is not None:
        for name, seconds, steps in (
            ("actuation_delay_range_s[1]", hi, hi_steps),
            ("actuation_delay_eval_s", eval_s, eval_steps),
        ):
            if steps is not None and steps >= episode_steps:
                raise ValueError(
                    f"{name} = {float(seconds)!r} s is {steps} physics steps of {timestep!r} s, at least the "
                    f"episode's {episode_steps} (max_episode_steps x frame_skip): no control target would reach "
                    "the plant within an episode; the delay keys are in seconds, not milliseconds"
                )
    return ActuationDelay(
        range_s=(lo, hi),
        eval_s=None if eval_s is None else float(eval_s),
        timestep=float(timestep),
        range_steps=(lo_steps, hi_steps),
        eval_steps=eval_steps,
    )


def primed_fifo(hold: Any, steps: int) -> deque[np.ndarray]:
    """A FIFO holding *steps* float64 copies of *hold*, with room for one more (``maxlen = steps + 1``).

    Appending a target and reading entry 0 then gives the value appended
    *steps* appends earlier, or the prime for the first *steps* appends.
    """
    return deque((np.array(hold, dtype=np.float64) for _ in range(steps)), maxlen=steps + 1)
