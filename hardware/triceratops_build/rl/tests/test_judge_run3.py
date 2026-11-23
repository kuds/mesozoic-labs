"""judge_run3.py on the stored run-3 outputs and on the cases the pre-registration names (every episode fell, no
target episode analysable): those fail, they do not crash."""

from __future__ import annotations

import copy
import json
import sys

import pytest
from conftest import RL_DIR  # noqa: F401

sys.path.insert(0, str(RL_DIR / "analysis"))
import judge_run3 as j  # noqa: E402

G = RL_DIR / "analytics" / "results" / "run3-s0" / "gait_report.json"
W = RL_DIR / "robustness" / "results" / "run3-s0" / "sweep_run3-s0.json.gz"


@pytest.fixture(scope="module")
def stored():
    return j._load(str(G)), j._load(str(W))


def all_fell(summary):
    """A point the way sweep.py writes it when every episode fell: no speed, drift or drive figures."""
    keep = ("episodes", "falls", "fall_terminations", "fall_times_s", "completed", "mass_kg", "fall_rate")
    out = {k: v for k, v in summary.items() if k in keep}
    out.update(falls=summary["episodes"], completed=0, fall_rate=1.0)
    return out


def test_stored_seed_reproduces_its_verdicts(stored):
    r = j.judge_seed(*stored)
    assert [r[k]["pass"] for k in j.RULES] == [True] * 6 + [False]


@pytest.mark.parametrize("point,rules", [(j.STEP[3], ("T1", "T3")), (j.STEP[2], ("T3",)), (j.RAMP[8], ("T2", "T3")),
                                         (j.FRICTION_03, ("T4",)), (j.NOMINAL, ("T1", "T2", "T4"))])
def test_a_point_where_every_episode_fell_fails_its_rules(stored, point, rules):
    g, w = copy.deepcopy(stored[0]), copy.deepcopy(stored[1])
    w["points"][point]["summary"] = all_fell(w["points"][point]["summary"])
    r = j.judge_seed(g, w)
    assert all(not r[k]["pass"] for k in rules)


def test_no_analysable_target_episode_fails_f1_and_f2(stored):
    g, w = copy.deepcopy(stored[0]), stored[1]
    t = g["cases"]["target_0.25"]["summary"]
    for k in ("speed", "drives", "footfall"):
        t.pop(k, None)
    t["falls"], t["short_episodes"] = 5, [100000, 100001, 100002, 100003, 100004]
    r = j.judge_seed(g, w)
    assert not r["F1"]["pass"] and not r["F2"]["pass"] and r["F2"]["worst"] is None


def test_verdicts_combine_across_seeds():
    p = {k: {"pass": True} for k in j.RULES}
    f = {k: {"pass": False} for k in j.RULES}
    assert set(j.verdict({"s0": p, "s1": p}).values()) == {"PASS"}
    assert set(j.verdict({"s0": p, "s1": f}).values()) == {"MIXED"}
    assert set(j.verdict({"s0": f, "s1": f}).values()) == {"FAIL"}
    assert all(v.startswith("NOT JUDGED") for v in j.verdict({"s0": p, "s1": None}).values())
