"""The servo delay queue at the add-on's delays (STAGE2_75_ADDON_PLAN.md A-1): k = 6 to 10 physics steps.

The 75 cm panels ran k = 0 to 5 (0 to 10 ms).  The add-on runs today's walk at 12 to 20 ms, so the queue of
``speedfit.servo_delay`` is checked here above k = 5.  No physics: a stub with the attributes the queue touches.
"""
import numpy as np
import pytest

from designlab.stage1 import speedfit


class _Data:
    pass


class _Stub:
    """The four things ``servo_delay`` uses of a plant: ``body``, ``d.ctrl``, ``_v.ctrl`` (the same array), and
    the class's ``reset`` and ``command``."""
    body = "stub"

    def __init__(self):
        self.d, self._v = _Data(), _Data()
        self.d.ctrl = np.zeros(2)
        self._v.ctrl = self.d.ctrl
        self.home = np.array([-1.0, -1.0])

    def reset(self):
        self.d.ctrl[:] = self.home

    def command(self, x):
        self._v.ctrl[:] = x


@pytest.mark.parametrize("k", [5, 6, 7, 8, 9, 10])
def test_a_target_reaches_the_servos_k_steps_after_it_is_set(k):
    p = _Stub()
    with speedfit.servo_delay(p, k) as queue:
        assert queue.maxlen == k + 1
        p.reset()
        applied = []
        for j in range(40):
            p.command(np.array([float(j), float(j)]))
            applied.append(float(p.d.ctrl[0]))
        # before k steps have passed the servos hold the reset's targets; after that, the target set k steps ago
        assert applied == [-1.0 if j < k else float(j - k) for j in range(40)]
        # a reset inside the block refills the queue with the reset's targets
        p.reset()
        p.command(np.array([99.0, 99.0]))
        assert float(p.d.ctrl[0]) == -1.0
    assert "command" not in p.__dict__ and "reset" not in p.__dict__


def test_the_row_names_of_the_add_on_give_k_as_half_the_milliseconds():
    from designlab.stage2 import bar
    for ms in (12, 14, 16, 18, 20):
        assert bar.parse_row(f"R2/42@{ms}ms") == ("R2/42", ms // 2)
        assert bar.parse_row(f"R0@{ms}ms") == ("R0", ms // 2)
    for bad in ("R2/42@22ms", "R2/42@13ms"):   # above the runner's limit; not a whole physics step
        with pytest.raises(ValueError):
            bar.parse_row(bad)
