"""judge_d22.py's choice (D23) on constructed measures, and the 10 ms foot-load window on a known series."""

from __future__ import annotations

import sys

import numpy as np
import pytest
from conftest import RL_DIR  # noqa: F401

sys.path.insert(0, str(RL_DIR / "analysis"))
import judge_d22 as jd  # noqa: E402


def run(load, heat=0.8, falls=20, f1=True, f2=True):
    rules = {k: True for k in ("F1", "F2", "T1", "T2", "T3", "T4", "T5")}
    rules.update(F1=f1, F2=f2)
    return {"rules": rules, "load_n": load, "heat": heat, "falls": falls}


def runs(l128=(140, 120), l160=(100, 102), **kw160):
    r = {(128, 0): run(l128[0]), (128, 1): run(l128[1])}
    r.update({(160, 0): run(l160[0], **kw160), (160, 1): run(l160[1], **kw160)})
    return r


def test_160_wins_only_when_its_worse_seed_is_15_percent_below_128s_better_seed():
    assert jd.choose(runs((140, 120), (100, 101)))["choice"] == 160  # 101 <= 0.85 x 120 = 102
    assert jd.choose(runs((140, 120), (100, 103)))["choice"] == 128  # 103 > 102
    assert jd.choose(runs((140, 120), (90, 125)))["choice"] == 128  # one 160 seed not clearly lower


def test_falls_and_heat_can_keep_the_default():
    assert jd.choose(runs(falls=21))["choice"] == 128  # 42 > 40 falls over both seeds
    assert jd.choose(runs(falls=20))["choice"] == 160  # a tie is not "more"
    assert jd.choose(runs(heat=0.86))["choice"] == 128  # 0.86 > 0.80 + 0.05
    assert jd.choose(runs(heat=0.85))["choice"] == 160


def test_a_length_that_does_not_walk_loses_and_missing_runs_decide_nothing():
    r = runs()
    r[(160, 1)] = run(100, f2=False)
    assert jd.choose(r)["choice"] == 128
    r = runs()
    for s in (0, 1):
        r[(128, s)]["rules"]["F1"] = False
    assert jd.choose(r)["choice"] == 160  # only 160 walks, whatever its loads
    r = runs()
    r[(128, 0)] = None
    assert jd.choose(r)["choice"] is None
    r = runs()
    for k in r:
        r[k]["rules"]["F1"] = False
    assert jd.choose(r)["choice"] is None


def test_the_ten_millisecond_window_of_the_load_recorder():
    from telemetry.loads import LoadRecorder

    class Fake:
        pass

    rec = LoadRecorder.__new__(LoadRecorder)  # only the summary's arithmetic is under test
    rec.names, rec.foot_names, rec.every, rec.ball, rec.shared = [], ["f"], 1, [], {}
    rec.force, rec.moment = [], []
    series = np.zeros(1000)
    series[500:505] = 100.0  # a 5 ms spike of 100 N
    rec.foot = [[v] for v in series]
    s = rec.summary(100.0, physics_dt=0.001)
    assert s["window_s"] == pytest.approx(0.010)
    assert s["feet"]["f"]["normal_n"]["max"] == 100.0
    assert s["feet"]["f"]["normal_window_n"]["max"] == pytest.approx(50.0)  # 5 of 10 samples
