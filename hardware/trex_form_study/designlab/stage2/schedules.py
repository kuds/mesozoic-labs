"""Learning-rate and clip schedules as instances of module-level classes (``STAGE2_PLAN.md`` S2-14; Track B rule K9).

stable-baselines3 stores a model's ``learning_rate``, ``lr_schedule`` and ``clip_range`` in its archive through
cloudpickle.  A closure (Stage 1's ``policy.linear_schedule``) is pickled BY VALUE, with its code object, and
``PPO.load`` then calls it: bytecode compiled by one Python minor version and run by another can kill the interpreter
with no traceback (the repository's ``environments/shared/policy_loading.py``).  An instance of a class that lives in an
importable module is pickled BY REFERENCE (``designlab.stage2.schedules.LinearSchedule`` and two floats), so an
archive saved with these holds no bytecode, and the loader (``s2load``) passes fresh instances anyway.

The classes must stay at module level in this file, and this file must never be run as ``__main__`` (a class of
``__main__`` is pickled by value): it has no CLI.  Standard library only; no row executes it (it is in no record key).
"""

from __future__ import annotations


class LinearSchedule:
    """SB3 schedule: ``start`` at the beginning (progress remaining 1) to ``end`` at the end (progress remaining 0);
    the formula of Stage 1's ``policy.linear_schedule``."""

    def __init__(self, start: float, end: float) -> None:
        self.start = float(start)
        self.end = float(end)

    def __call__(self, progress_remaining: float) -> float:
        return self.end + (self.start - self.end) * float(progress_remaining)

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return (self.start, self.end) == (other.start, other.end)

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.start, self.end))

    def __repr__(self) -> str:
        return f"LinearSchedule(start={self.start!r}, end={self.end!r})"


class ConstantSchedule:
    """SB3 schedule: ``value`` at every progress."""

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def __call__(self, progress_remaining: float) -> float:
        return self.value

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self.value == other.value

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.value))

    def __repr__(self) -> str:
        return f"ConstantSchedule(value={self.value!r})"


def describe(s) -> dict:
    """What a run's settings record of a schedule: ``{"class": "LinearSchedule", "start", "end"}`` or ``{"class":
    "ConstantSchedule", "value"}``.  A wrapper that holds its schedule as ``value_schedule`` (SB3's ``FloatSchedule``,
    which a model's ``lr_schedule`` and ``clip_range`` are) is described by what it wraps.  Anything else, a function
    above all, is refused: a schedule that cannot be described here is one the archive could not hold by reference."""
    seen = 0
    while not isinstance(s, (LinearSchedule, ConstantSchedule)) and hasattr(s, "value_schedule") and seen < 8:
        s, seen = s.value_schedule, seen + 1
    if isinstance(s, LinearSchedule):
        return {"class": "LinearSchedule", "start": s.start, "end": s.end}
    if isinstance(s, ConstantSchedule):
        return {"class": "ConstantSchedule", "value": s.value}
    raise TypeError(f"{s!r} is not a schedule of designlab.stage2.schedules (a float or a function is not recorded: "
                    "pass LinearSchedule or ConstantSchedule)")


__all__ = ["ConstantSchedule", "LinearSchedule", "describe"]
