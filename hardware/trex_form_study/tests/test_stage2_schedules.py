"""Schedules as instances of module-level classes (``designlab/stage2/schedules.py``; ``STAGE2_PLAN.md`` S2-14): their
values, that a pickle holds them by reference (no bytecode), and what a run's settings record of them.

That an archive saved with them holds no bytecode member is ``tests/test_stage2_s2load.py``'s.
"""

from __future__ import annotations

import pickle
import pickletools

import pytest

from designlab.stage2 import schedules
from designlab.stage2.schedules import ConstantSchedule, LinearSchedule


def test_linear_schedule_values():
    s = LinearSchedule(1e-4, 3e-5)
    assert s(1.0) == 1e-4                                    # the beginning: all of the progress remains
    assert s(0.0) == 3e-5                                    # the end
    assert s(0.5) == pytest.approx(6.5e-5, rel=1e-12)
    assert s(0.5) == 3e-5 + (1e-4 - 3e-5) * 0.5              # the formula, to the bit
    assert isinstance(s(1), float) and isinstance(s(0.25), float)
    assert LinearSchedule(2, 2)(0.3) == 2.0 and (s.start, s.end) == (1e-4, 3e-5)


def test_linear_schedule_is_stage1s_formula():
    pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    from designlab.stage1 import policy

    for start, end in ((1e-4, 3e-5), (1e-5, 3e-6), (3e-5, 3e-5)):
        mine, theirs = LinearSchedule(start, end), policy.linear_schedule(start, end)
        assert all(mine(p / 64) == theirs(p / 64) for p in range(65))


def test_constant_schedule_values():
    s = ConstantSchedule(0.2)
    assert s(1.0) == s(0.5) == s(0.0) == 0.2 and isinstance(s(1.0), float)
    assert ConstantSchedule(0)(0.7) == 0.0 and isinstance(ConstantSchedule(0).value, float)


def _globals(payload: bytes) -> set:
    """The (module, name) pairs a pickle stream refers to, without running it."""
    found, strings = set(), []
    for op, arg, _ in pickletools.genops(payload):
        if op.name == "GLOBAL":
            found.add(tuple(str(arg).split(" ", 1)))
        elif op.name == "STACK_GLOBAL":
            found.add((strings[-2], strings[-1]))
        elif isinstance(arg, str):
            strings.append(arg)
    return found


@pytest.mark.parametrize("obj", [LinearSchedule(1e-4, 3e-5), ConstantSchedule(0.2)], ids=["linear", "constant"])
def test_pickle_round_trip_is_by_reference(obj):
    for protocol in range(2, pickle.HIGHEST_PROTOCOL + 1):
        payload = pickle.dumps(obj, protocol)
        back = pickle.loads(payload)
        assert back == obj and type(back) is type(obj) and back(0.5) == obj(0.5)
        assert _globals(payload) == {("designlab.stage2.schedules", type(obj).__name__)}
    cloudpickle = pytest.importorskip("cloudpickle")         # what stable-baselines3 writes an archive with
    payload = cloudpickle.dumps(obj)
    assert cloudpickle.loads(payload) == obj
    assert _globals(payload) == {("designlab.stage2.schedules", type(obj).__name__)}   # no reducer of cloudpickle
    assert type(obj).__module__ == "designlab.stage2.schedules" and "<locals>" not in type(obj).__qualname__


def test_equality_hash_and_repr():
    assert LinearSchedule(1e-4, 3e-5) == LinearSchedule(1e-4, 3e-5) != LinearSchedule(1e-4, 3e-6)
    assert ConstantSchedule(0.2) == ConstantSchedule(0.2) != ConstantSchedule(0.0)
    assert LinearSchedule(0.2, 0.2) != ConstantSchedule(0.2) and ConstantSchedule(0.2) != 0.2
    assert len({LinearSchedule(1, 0), LinearSchedule(1, 0), ConstantSchedule(1), ConstantSchedule(1)}) == 2
    assert repr(LinearSchedule(1e-4, 3e-5)) == "LinearSchedule(start=0.0001, end=3e-05)"
    assert repr(ConstantSchedule(0.2)) == "ConstantSchedule(value=0.2)"
    assert eval(repr(LinearSchedule(1e-4, 3e-5)), vars(schedules)) == LinearSchedule(1e-4, 3e-5)


def test_describe():
    assert schedules.describe(LinearSchedule(1e-4, 3e-5)) == {"class": "LinearSchedule", "start": 1e-4, "end": 3e-5}
    assert schedules.describe(ConstantSchedule(0.2)) == {"class": "ConstantSchedule", "value": 0.2}

    class Wrapper:                                           # the shape of SB3's FloatSchedule
        def __init__(self, inner):
            self.value_schedule = inner

    assert schedules.describe(Wrapper(Wrapper(LinearSchedule(1.0, 0.5)))) == {"class": "LinearSchedule", "start": 1.0,
                                                                              "end": 0.5}
    for bad in (lambda p: 1e-4, 1e-4, None, Wrapper(lambda p: 0.0)):
        with pytest.raises(TypeError, match="not a schedule"):
            schedules.describe(bad)


def test_describe_reads_a_models_wrapped_schedule():
    utils = pytest.importorskip("stable_baselines3.common.utils")
    if not hasattr(utils, "FloatSchedule"):
        pytest.skip("this stable-baselines3 has no FloatSchedule")
    wrapped = utils.FloatSchedule(LinearSchedule(1e-4, 3e-5))             # what a model's lr_schedule is
    assert schedules.describe(wrapped) == {"class": "LinearSchedule", "start": 1e-4, "end": 3e-5}
    assert wrapped(0.5) == LinearSchedule(1e-4, 3e-5)(0.5)
