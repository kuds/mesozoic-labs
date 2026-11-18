"""Scratch stand-in for designlab/stage2/schedules.py: module-level schedule classes (no closure, no bytecode)."""


class LinearSchedule:
    """SB3 schedule: progress_remaining (1 -> 0) -> value, linear from ``start`` to ``end``."""

    def __init__(self, start: float, end: float):
        self.start, self.end = float(start), float(end)

    def __call__(self, progress_remaining: float) -> float:
        return self.end + (self.start - self.end) * float(progress_remaining)

    def __repr__(self) -> str:
        return f"LinearSchedule({self.start!r}, {self.end!r})"


class ConstantSchedule:
    def __init__(self, value: float):
        self.value = float(value)

    def __call__(self, progress_remaining: float) -> float:
        return self.value

    def __repr__(self) -> str:
        return f"ConstantSchedule({self.value!r})"
