"""Prevent a supported shuffle or stationary rocking policy from passing a walking gate."""

import json

import numpy as np

from examples.tyrannosaurus_rex_60cm.platform.evaluate_gait import PROTOCOL, command_at, summarize, walking_metrics


def test_weight_shift_counts_alternations_but_requires_sole_clearance():
    t = np.arange(0, 1.0, 0.02)
    loads = np.zeros((50, 2))
    for i in range(5):
        loads[i * 10 : (i + 1) * 10, i % 2] = 1.0
    n, medians, counts = walking_metrics(t, loads, np.zeros_like(loads), 0.02)
    assert n == 4 and counts == [2, 1] and medians == [0.0, 0.0]
    # Both soles support substantial load during the whole step.
    n, medians, _ = walking_metrics(t, np.full_like(loads, 0.5), np.zeros_like(loads), 0.02)
    assert n == 0 and medians == [None, None]


def test_early_fall_before_gait_window_is_result_not_exception():
    protocol = json.loads(PROTOCOL.read_text())
    t = [0.02]
    trace = dict(
        t=t,
        xy=[[0.0, 0.0]],
        yaw=[0.0],
        z=[0.36],
        tilt=[0.0],
        loads=[[0.4, 0.6]],
        clear=[[0.0, 0.0]],
        slip=[[0.0, 0.0]],
        expression=[[0.0, 0.0]],
        unexpected=[0.0],
    )
    result = summarize(trace, 0.02, -1, 15, True, protocol["confirmation"])
    assert not result["passed"] and "early_end" in result["failed_checks"]
    assert result["metrics"]["forward_mps"] is None
    for sign in (-1, 1):
        assert command_at(7.0, sign, protocol["commands"])[:2] == (0.12, 0.30 * sign)
