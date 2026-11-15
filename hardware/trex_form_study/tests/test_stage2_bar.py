"""Tests for the bar's runner (designlab/stage2/bar.py): conditions and keys, the plan table, delayed rows, the judge's
plumbing, records, resume, the no-overwrite rule, workers, attribution, replication and the budget ledger.

Run from hardware/trex_form_study (the fit's venv, one BLAS thread):
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/test_stage2_bar.py -q

Fast tests use (a) Stage 1's own run directory through ``setio.stage1_view`` (the pseudo-set ``S1``: stock conditions,
R2 rows and the delay wrapper) and (b) a synthetic set ``T0`` in pytest's temporary directory, made of Stage 1's own
arrays of 75-M-H-10 under the bar's names: the brisk walk as ``v0.4``, ``nominal_to_brisk`` as ``nominal_to_v0.4``,
Stage 1's stop and turn entries *declared* causal as ``stop2_v0.4`` and ``turn90_pi4_<dir>``.  ``T0`` is a fixture for
plans, keys, records and the runner's plumbing: its walk item is Stage 1's brisk walk exactly, its "moves" join
nothing and say nothing about a stop or a turn.  Only R0 rows run episodes on it; the tests of the runner's rules
that name R2 rows there use synthetic episode records (``fake_episode``) and simulate nothing.  Both need the Stage 1
arrays and the round-two model files (git-ignored; ``python tools/restore_inputs.py``) and skip without them.

Slow (``STAGE2_SLOW=1``): one full episode per judge on a built and frozen set under ``R0@10ms`` (set
``STAGE2_BAR_SET``, default ``V0``, under ``STAGE2_BAR_SET_ROOT``, default ``stage2/sets``).  On a set of the 75 cm
round these are build checks: they run only when ``STAGE2_DEV_EPISODES`` names the file that lists them.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from designlab.stage1 import control, metrics, panels, plant, speedfit, tune  # noqa: E402
from designlab.stage2 import bar, readings, s2io, setio  # noqa: E402

BODY = "75-M-H-10"
RUN = ROOT / "stage1" / "runs" / "2026-11-08"
ONE_THREAD = all(os.environ.get(v) == "1" for v in s2io.THREAD_VARS)
HAVE = (RUN / BODY / "transitions" / "stop.npz").exists()
try:
    HAVE = HAVE and all(Path(f).exists() for f in plant.model_files(BODY))
except Exception:  # noqa: BLE001  (no model record: the same skip)
    HAVE = False
needs_run = pytest.mark.skipif(not (HAVE and ONE_THREAD), reason=(
    "needs the Stage 1 arrays and the round-two models (tools/restore_inputs.py) and one BLAS thread "
    "(OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1)"))
SLOW = os.environ.get("STAGE2_SLOW") == "1"
SET = "T0"
W = setio.walk_name(setio.BAR_SPEED)

#: T0: (bar name, Stage 1 name) of each file.
T0_REFS = (("nominal", "nominal"), (W, "brisk"))
T0_MOVES = (("start", "start"), (setio.change_name("nominal", W), "nominal_to_brisk"),
            (setio.stop_name(2, W), "stop"), (setio.turn_name("pi4", "left"), "enter_walk_0.35_left"),
            (setio.turn_name("pi4", "right"), "enter_walk_0.35_right"))


def make_t0(root: Path, drop=(), unavailable=(), label=None, extra_entry=False, freeze=True) -> Path:
    """The synthetic set ``T0`` under ``root`` (module docstring); ``drop`` leaves entries out, ``unavailable``
    records them ``available: false``, ``label`` sets the walk entry's label, ``extra_entry`` adds another entry."""
    src = RUN / BODY
    d = setio.set_dir(SET, BODY, root)
    refs1, trs1 = s2io.read_json(src / "refs.json"), s2io.read_json(src / "transitions.json")
    refs, trs, manifest = {}, {}, {}
    for name, old in T0_REFS:
        if name in drop:
            continue
        e = copy.deepcopy(refs1["references"][old])
        e.update(name=name, file=f"refs/{name}.npz", set=SET, label=label if name == W else None, strides=1,
                 window_s=e["period_s"], crouch_frac=0.03, margin_rule="a fixture's label")
        refs[name] = e
    for name, old in T0_MOVES:
        if name in drop:
            continue
        if name in unavailable:
            trs[name] = dict(name=name, kind="turn", available=False, numbers=dict(hip_yaw_range_use=0.93),
                             reason="no reference within the thigh twist and the leg drives")
            continue
        e = copy.deepcopy(trs1["transitions"][old])
        e.update(name=name, file=f"transitions/{name}.npz", a1_over=None)
        kind = setio.parse_name(name)["kind"]
        if kind == "stop":
            e["move"] = dict(kind="stop", builder="fixture", causal=True, command_phase=-0.5)
        elif kind == "turn":
            p = setio.parse_name(name)
            e["move"] = dict(kind="turn", builder="fixture", causal=True, command_phase=0.0 if p["sign"] > 0 else 0.5)
            e["turn"] = dict(angle_rad=p["angle_rad"], rate_rad_s=p["sign"] * p["rate_rad_s"], direction=p["direction"])
        else:
            e["move"] = dict(kind=kind, builder="fixture", causal=False, command_phase=None)
        trs[name] = e
    if extra_entry:
        refs["v0.5"] = dict(name="v0.5", available=False, reason="a fixture's extra entry", numbers=None)
    for doc, sub, kind in ((refs, "refs", "walk"), (trs, "transitions", "move")):
        for name, e in doc.items():
            if not e.get("file"):
                continue
            old = dict(T0_REFS + T0_MOVES)[name]
            (d / sub).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / sub / f"{old}.npz", d / e["file"])
            manifest[name] = dict(kind=kind, sha256=e["sha256"], entry_sha256=setio.entry_sha256(kind, e))
    s2io.write_json(d / "refs.json", dict(body=BODY, set=SET, schema=1, references=refs))
    s2io.write_json(d / "transitions.json", dict(body=BODY, set=SET, schema=1, transitions=trs))
    freezes = [dict(index=0, at="fixture", commit=None, manifest=manifest,
                    manifest_sha256=s2io.sha256_json(manifest))] if freeze else []
    s2io.write_json(d / "set.json", dict(set=SET, body=BODY, role="fixture", crouch_frac=0.03,
                                         rules=dict(fixture="Stage 1's arrays under the bar's names"),
                                         not_available={}, freezes=freezes))
    return d


@pytest.fixture(scope="module")
def t0(tmp_path_factory) -> Path:
    """The root of a frozen ``T0``."""
    root = tmp_path_factory.mktemp("sets")
    make_t0(root)
    return root


@pytest.fixture(scope="module")
def t0_docs(t0) -> dict:
    return setio.load(setio.set_dir(SET, BODY, t0))


def planner_of(root: Path) -> bar.BarPlanner:
    d = setio.set_dir(SET, BODY, root)
    docs = setio.load(d)
    return bar.BarPlanner(d, docs["refs.json"]["references"], docs["transitions.json"]["transitions"], docs["set.json"])


def cond_of(docs: dict, key: str, block: str = bar.HEADLINE_BLOCK) -> bar.BarCondition:
    return next(c for c in bar.conditions(docs, block) if c.key == key)


def quiet(*a, **k):
    pass


def fake_episode(cond, seed: int, fell: bool = False, before=None, tail_over: bool = False, fail=(), relax=(),
                 cands=("slip",)) -> dict:
    """A synthetic episode record of a bar condition (no simulation): survives and succeeds unless told otherwise;
    ``tail_over`` fails the sustained-drive term in reading 2 only; ``fail`` names other failed terms."""
    terms, t3 = bar.terms_of(cond), bar.terms_but_a1(cond)
    s2 = {t: not fell and t not in fail and not (tail_over and t == "drive_sustained") for t in terms}
    s3 = {t: not fell and t not in fail for t in t3}
    return dict(seed=int(seed), fell=bool(fell), fall_time_s=3.0 if fell else None, fall_cause="tilt" if fell else None,
                fall_onset_s=2.5 if fell else None, fall_candidates=list(cands) if fell else None,
                limits_before_onset=None, survival_s=3.0 if fell else 20.0, planned_s=20.0, strides=3.5 if fell else 24.0,
                success=s2, succeeded=all(s2.values()), success_but_A1=s3, succeeded_but_A1=all(s3.values()),
                failed_terms=["survived"] if fell else [t for t, v in s2.items() if not v],
                failed_terms_but_A1=["survived"] if fell else [t for t, v in s3.items() if not v],
                command_s=None if cond.judge == "bar_walk" else 13.0, command_ref_s=None,
                fell_before_command=(None if cond.judge == "bar_walk" else bool(before)) if fell else
                (None if cond.judge == "bar_walk" else False), segment_reached=cond.walk, window_reached=not fell,
                omega=None, crouch_frac=0.03, measures={}, relaxed=list(relax), wall_s=0.25)


def faked(monkeypatch, plan):
    """Replace ``bar.run_episode`` by ``plan(row, cond, seed, relax) -> episode`` (the in-process worker calls it);
    returns the list of calls."""
    calls = []

    def run_episode(row, ctx, cond, seed, relax=(), trace_root=None):
        calls.append((row.name, cond.key, int(seed), tuple(relax)))
        return plan(row, cond, int(seed), tuple(relax))

    monkeypatch.setattr(bar, "run_episode", run_episode)
    return calls


def stock_cut(monkeypatch, seconds: float = 1.0, keys=("walk/brisk",)):
    """Stage 1's conditions ``keys`` cut to ``seconds`` (a smoke check) as the pseudo-set S1's conditions."""
    conds = [dataclasses.replace(c, seconds=float(seconds)) for c in panels.conditions() if c.key in keys]
    monkeypatch.setattr(bar, "stock_conditions", lambda: conds)
    monkeypatch.setattr(bar, "stock_default", lambda: conds)
    return conds


def read(path) -> dict:
    return json.loads(Path(path).read_text())


def rec_path(out, row, key, block=bar.HEADLINE_BLOCK, set_name=SET, control=None) -> Path:
    behavior, name = key.split("/")
    return bar.block_dir(out, set_name, BODY, block, control) / "eval" / row / behavior / f"{name}.json"


# ------------------------------------------------------------------------------------------------
# Conditions, names, keys
# ------------------------------------------------------------------------------------------------
GATED = ["walk/v0.4", "walk/v0.4_mu0.54", "turn/q90_left", "turn/q90_left_mu0.54", "turn/q90_right",
         "turn/q90_right_mu0.54", "nudge/forward", "nudge/forward_mu0.54", "nudge/backward", "nudge/backward_mu0.54",
         "nudge/left", "nudge/left_mu0.54", "nudge/right", "nudge/right_mu0.54", "stop/two_step", "stop/two_step_mu0.54"]


@needs_run
def test_the_16_gated_conditions(t0_docs):
    conds = bar.conditions(t0_docs)
    gated = [c for c in conds if c.gated]
    assert [c.key for c in gated] == GATED
    assert all(isinstance(c, bar.BarCondition) and c.set == SET and c.block == "seeds3042" for c in conds)
    assert {c.judge for c in gated} == set(bar.BAR_JUDGES) and all(c.floor == "hard" for c in conds)
    by = {c.key: c for c in conds}
    assert by["walk/v0.4"].friction is None and by["walk/v0.4_mu0.54"].friction == 0.54
    assert by["turn/q90_right"].move == "turn90_pi4_right" and by["turn/q90_right"].direction == "right"
    assert by["stop/two_step"].move == "stop2_v0.4" and by["nudge/left"].frac == 0.25
    assert (by["nudge/left"].push_at, by["nudge/left"].push_spread) == (2.0, 1.0)
    # every item takes the walk's seed slot (slot 0), whatever its judge
    assert {panels.episode_seeds(c, 3042)["slot"] for c in conds} == {0}
    assert panels.episode_seeds(by["turn/q90_left"], 3042) == panels.episode_seeds(by["walk/v0.4"], 3042)
    # reported: only what the set has entries for (T0: the nominal walk; the larger nudges need no other reference)
    reported = [c.key for c in conds if not c.gated]
    assert reported == ["walk/nominal", "walk/nominal_mu0.54"] + [f"nudge/{d}_x{f:g}" for f in (0.375, 0.5)
                                                                 for d in ("forward", "backward", "left", "right")]
    assert by["nudge/forward_x0.375"].frac == 0.375 and by["nudge/forward_x0.5"].frac == 0.5
    # names hold %g numbers, never a rounded .2f
    assert all("0.40" not in c.key and "0.50" not in c.key for c in conds)
    # the replication block is a field of the condition (and so of its key)
    assert {c.block for c in bar.conditions(t0_docs, "seeds5042")} == {"seeds5042"}
    with pytest.raises(ValueError):
        bar.conditions(t0_docs, "seeds7042")
    assert bar.select(conds, ["walk/v0.4", "two_step"]) == [by["walk/v0.4"], by["stop/two_step"]]
    with pytest.raises(ValueError, match="walk/v0.9"):
        bar.select(conds, ["walk/v0.9"])


@needs_run
def test_reported_conditions_follow_the_sets_entries(tmp_path):
    make_t0(tmp_path, extra_entry=True)
    docs = setio.load(setio.set_dir(SET, BODY, tmp_path))
    docs["transitions.json"]["transitions"]["turn90_0.5_left"] = dict(name="turn90_0.5_left", available=False,
                                                                       reason="over a leg drive", numbers=None)
    docs["transitions.json"]["transitions"]["stop1_v0.4"] = dict(name="stop1_v0.4", file="transitions/stop1_v0.4.npz")
    keys = [c.key for c in bar.conditions(docs) if not c.gated]
    assert "walk/v0.5" in keys and "walk/v0.5_mu0.54" in keys and "walk/v0.45" not in keys
    assert "turn/t90_0.5_left" in keys and "turn/t90_0.5_right" not in keys and "stop/one_step" in keys
    pl = bar.BarPlanner(setio.set_dir(SET, BODY, tmp_path), docs["refs.json"]["references"],
                        docs["transitions.json"]["transitions"], docs["set.json"])
    why = pl.missing(cond_of(docs, "walk/v0.5"))
    assert why["reference"] == "refs/v0.5" and why["reason"] == "a fixture's extra entry"
    assert pl.missing(cond_of(docs, "turn/t90_0.5_left"))["reason"] == "over a leg drive"
    assert pl.missing(cond_of(docs, "walk/v0.4")) is None


def test_row_names_parse():
    assert bar.parse_row("R2/42@4ms") == ("R2/42", 2) and bar.parse_row("R0@10ms") == ("R0", 5)
    assert bar.parse_row("R2/43@0ms") == ("R2/43", 0) and bar.parse_row("R4/42@20ms") == ("R4/42", 10)
    for bad in ("R2/42@3ms", "R2/42@22ms", "R2/42", "R0@ms", "R0@-2ms", "@4ms"):
        with pytest.raises(ValueError):
            bar.parse_row(bad)
    assert bar.controller_of("R2/42@10ms") == "R2" and bar.controller_of("R0@10ms") == "R0"
    assert bar.delay_ms("R2/43@6ms") == 6 and bar.delay_ms("R2/43") is None
    with pytest.raises(ValueError):
        bar.Delayed(panels.Row(), 11)
    with pytest.raises(ValueError):
        bar.Delayed(panels.Row(), 0.5)


@needs_run
def test_rows_are_stage1s_rows_in_the_delay_wrapper():
    rows = bar.rows(BODY, ["R2/42@4ms", "R2/43@10ms", "R0@10ms"], RUN)
    assert list(rows) == ["R2/42@4ms", "R2/43@10ms", "R0@10ms"]
    r = rows["R2/42@4ms"]
    assert isinstance(r, bar.Delayed) and isinstance(r.inner, panels.TrackerRow) and r.k == 2 and r.inner.name == "R2/42"
    gains, info = panels.r2_gains(RUN / BODY, 42)
    assert np.array_equal(r.inner.gains, gains)
    d = r.describe()
    assert d["gains"] == control.gains_dict(gains) and d["gains_info"]["sha256"] == plant.sha256(RUN / BODY / "tune.json")
    assert (d["latency_steps"], d["latency_ms"], d["delayed_row"], d["row"]) == (2, 4, "R2/42@4ms", "R2/42")
    assert d["servo_delay_rule"] == speedfit.LATENCY_RULE
    # R0@10ms is the panels' R0 (no feedback term, no ankle cap) with five steps of delay
    r0 = rows["R0@10ms"]
    assert r0.inner.describe() == panels.tracker_rows(BODY, RUN / BODY, ("R0",))["R0"].describe() and r0.k == 5
    assert r0.inner.terms == () and r0.inner.ankle_cap == ()
    # it runs every condition it is given, and gating is the condition's
    c = panels.Condition("carpet", "low", True, "carpet", floor="low")
    assert r0.runs(c) and r0.gated(c) and not r0.gated(dataclasses.replace(c, gated=False))
    # sources: the inner row's with bar.py and speedfit.py; the spec rebuilds the row in a worker
    src = r.sources()
    assert set(src) == {"designlab/stage2/bar.py", "designlab/stage1/speedfit.py"}
    assert src["designlab/stage2/bar.py"] == s2io.sha256_file(ROOT / "designlab" / "stage2" / "bar.py")
    spec = r.spec()
    assert spec["cls"] == "designlab.stage2.bar.Delayed" and spec["inner"]["cls"].endswith("panels.TrackerRow")
    again = panels.row_from_spec(json.loads(json.dumps(spec)))
    assert type(again) is bar.Delayed and again.describe() == d and again.name == r.name and again.k == 2
    # the key holds a portable description: no absolute path
    assert str(ROOT) not in json.dumps(s2io.portable(d))


@needs_run
def test_delayed_wraps_a_policy_row_and_merges_its_sources():
    policy_rows = pytest.importorskip("designlab.stage1.policy_rows")

    class StubPolicy(panels.Policy):
        ckpt = dict(seed=42, steps=1000, run=dict(env=dict(rate=1.05, tasks=["walk"])))

        def describe(self):
            return dict(policy="stub", sha256="0" * 64)

    inner = policy_rows.PolicyOnlyRow("R4/42", StubPolicy(), {"refs/nominal.npz": "ab"})
    assert isinstance(inner, panels.Row) and not isinstance(inner, panels.TrackerRow)
    row = bar.Delayed(inner, 5)
    assert row.name == "R4/42@10ms"
    d = row.describe()
    assert d["policy"] == dict(policy="stub", sha256="0" * 64) and d["tracker_mode"] is None and d["latency_ms"] == 10
    src = row.sources()
    assert set(inner.sources()) < set(src) and inner.sources().items() <= src.items()
    assert {"designlab/stage2/bar.py", "designlab/stage1/speedfit.py"} <= set(src)
    assert row.spec()["inner"]["cls"].endswith("policy_rows.PolicyOnlyRow") and row.spec()["k"] == 5
    assert row.runs(panels.conditions()[0]) and bar.controller_of(row.name) == "R4"


@needs_run
def test_bar_sources_are_the_four_modules():
    s = bar.bar_sources()
    assert sorted(s["files"]) == ["designlab/stage1/speedfit.py", "designlab/stage2/bar.py",
                                  "designlab/stage2/readings.py", "designlab/stage2/setio.py"]
    for rel, sha in s["files"].items():
        assert sha == s2io.sha256_file(ROOT / rel)
    assert s["sha256"] == s2io.sha256_json(s["files"])


def key_of(root: Path, row, cond_key="walk/v0.4", sources=None, block="seeds3042", edit=None) -> dict:
    d = setio.set_dir(SET, BODY, root)
    docs = setio.load(d)
    pl = bar.BarPlanner(d, docs["refs.json"]["references"], docs["transitions.json"]["transitions"], docs["set.json"])
    cond = cond_of(docs, cond_key, block)
    res = pl.resolve(cond)
    inputs, _ = setio.read_inputs(d, res["files"])
    if edit:
        inputs = edit(copy.deepcopy(inputs))
    ctx = dict(plant_inputs=dict(mjcf="aa"), versions=dict(python="3"), env={})
    return bar.record_key(BODY, SET, block, row, cond, res, bar.BLOCKS[block], inputs, ctx,
                          sources or dict(panels="p", bar="b"), 4.5, 0.03)


@needs_run
def test_the_key_ignores_labels_and_changes_with_what_an_episode_runs(t0, tmp_path, monkeypatch):
    row = bar.rows(BODY, ["R0@10ms"], RUN)["R0@10ms"]
    key = key_of(t0, row)
    assert key == json.loads(json.dumps(key)) and key["schema"] == "bar-1" and key["set"] == SET
    assert sorted(key["inputs"]) == ["refs/nominal.npz", "refs/v0.4.npz", "transitions/nominal_to_v0.4.npz",
                                     "transitions/start.npz"]
    assert all(set(v) == {"sha256", "entry_sha256"} for v in key["inputs"].values())
    assert (key["omega"], key["crouch_frac"], key["command"]) == (4.5, 0.03, None)
    assert key["condition"]["stock"] is False and key["condition"]["block"] == "seeds3042"
    assert key["seeds"] == list(range(3042, 3062)) and key["plan"][-1] == ["ref", "v0.4", 10]
    # a label, a rule text and another entry of the set change no key
    make_t0(tmp_path / "labelled", label="new", extra_entry=True)
    assert key_of(tmp_path / "labelled", row) == key
    # ... a file's sha256, an entry number the planner reads, bar.py's bytes, a row source, the row, the seeds do
    def other_file(inp):
        inp["files"]["refs/v0.4.npz"]["sha256"] = "0" * 64
        return inp

    def other_entry(inp):
        inp["files"]["refs/v0.4.npz"]["entry_sha256"] = "1" * 64
        return inp

    assert key_of(t0, row, edit=other_file) != key and key_of(t0, row, edit=other_entry) != key
    assert key_of(t0, row, sources=dict(panels="p", bar="other")) != key
    assert key_of(t0, row, sources=dict(panels="other", bar="b")) != key
    assert key_of(t0, bar.rows(BODY, ["R0@8ms"], RUN)["R0@8ms"]) != key
    assert key_of(t0, row, block="seeds5042") != key
    monkeypatch.setattr(row, "sources", lambda: {"designlab/stage2/bar.py": "changed"})
    assert key_of(t0, row)["row_sources"] != key["row_sources"]
    # the command of a commanded item is in its key
    turn = key_of(t0, bar.rows(BODY, ["R0@10ms"], RUN)["R0@10ms"], "turn/q90_left")
    assert turn["command"]["name"] == "turn90_pi4_left" and turn["command"]["phase"] == 0.0
    assert turn["command"]["ref_s"] == pytest.approx(14.530576625340654, abs=1e-9)
    assert bar.key_differences(key, dict(key, omega=4.6, extra=1)) == ["key.extra: (absent) -> 1",
                                                                         "key.omega: 4.5 -> 4.6"]


# ------------------------------------------------------------------------------------------------
# Plans
# ------------------------------------------------------------------------------------------------
@needs_run
def test_the_walk_plan_is_stage1s_rule(t0, t0_docs):
    pl = planner_of(t0)
    res = pl.resolve(cond_of(t0_docs, "walk/v0.4"))
    s1 = panels.Planner(RUN / BODY, s2io.read_json(RUN / BODY / "refs.json")["references"],
                        s2io.read_json(RUN / BODY / "transitions.json")["transitions"])
    brisk = s1.resolve(next(c for c in panels.conditions() if c.key == "walk/brisk"))
    rename = {"brisk": "v0.4", "nominal_to_brisk": "nominal_to_v0.4"}
    assert res["plan"] == [tuple(rename.get(x, x) for x in item) for item in brisk["plan"]]
    assert res["seconds"] == brisk["seconds"] and res["steady_ref_s"] == brisk["steady_ref_s"]
    assert res["files"] == ["refs/nominal.npz", "refs/v0.4.npz", "transitions/nominal_to_v0.4.npz",
                            "transitions/start.npz"]
    # the committed Stage 1 record of that walk holds the same plan length
    committed = read(RUN / BODY / "eval" / "R0" / "walk" / "brisk.json")
    assert committed["seconds"] == res["seconds"]
    # walk/nominal: start, then nominal to the end
    nominal = pl.resolve(cond_of(t0_docs, "walk/nominal"))
    assert [x[:2] for x in nominal["plan"]] == [("transition", "start"), ("ref", "nominal")]
    assert nominal["target_speed_m_s"] == t0_docs["refs.json"]["references"]["nominal"]["speed_m_s"]
    assert res["target_speed_m_s"] == 0.4  # the rung's target is the speed its name holds


@needs_run
def test_plans_hold_at_the_slowest_clock_and_tracks_at_the_fastest(t0, t0_docs):
    pl = planner_of(t0)
    slow, fast, rate = bar.SLOW * bar.RATE, bar.FAST * bar.RATE, bar.RATE
    T = pl.period_s("v0.4")

    def ends(plan):
        times = pl.segment_times(plan)
        return [t for t, _ in times], times[-1][0] + times[-1][1]

    lead = [("transition", "start"), ("ref", "nominal", 1), ("transition", "nominal_to_v0.4")]
    # walk: Stage 1's rule; K the fewest strides
    r = pl.resolve(cond_of(t0_docs, "walk/v0.4"))
    assert r["plan"][:3] == lead and r["plan"][3][:2] == ("ref", "v0.4")
    t0s, t_end = ends(r["plan"])
    need = (r["steady_ref_s"] + 0.5 * T) / slow * rate / rate + 10.0 * rate
    need = (r["steady_ref_s"] + 0.5 * T) / (1 - control.PHASE_RATE_LIMIT) + 10.0 * rate
    assert t_end >= need > t_end - T and r["seconds"] == t_end / rate
    # turn: 2 strides (2.5 for a right turn), the turn, K strides; the command is the turn's first frame
    for d, n in (("left", 2.0), ("right", 2.5)):
        r = pl.resolve(cond_of(t0_docs, f"turn/q90_{d}"))
        assert r["plan"][:3] == lead and r["plan"][3] == ("ref", "v0.4", n)
        assert r["plan"][4] == ("transition", f"turn90_pi4_{d}") and r["plan"][5][:2] == ("ref", "v0.4")
        t0s, t_end = ends(r["plan"])
        assert r["command_ref_s"] == t0s[4] and r["at_s"] == 2.0
        assert r["seconds"] == pytest.approx(t0s[4] / slow + 2.0 + 3.0 + 0.2, abs=1e-12)
        assert t_end >= r["seconds"] * fast > t_end - T
        # at the slowest clock the episode still holds the command + at_s + 3 s
        assert r["seconds"] - (t0s[4] / slow + 5.0) >= 0.2 - 1e-12
        assert r["command"]["convention"].startswith("causal") and r["sign"] == (1.0 if d == "left" else -1.0)
    # nudge: the latest onset at the slowest clock + 0.1 + 3.0 + 2.0
    r = pl.resolve(cond_of(t0_docs, "nudge/left"))
    t0s, t_end = ends(r["plan"])
    assert r["plan"][:3] == lead and r["push"]["steady_ref_s"] == t0s[3] == r["steady_ref_s"]
    assert r["seconds"] == pytest.approx((t0s[3] + 3.0 * T) / slow + 0.1 + 3.0 + 2.0, abs=1e-12)
    assert t_end >= r["seconds"] * fast > t_end - T
    assert r["push"] == dict(kind="walking", steady_ref_s=t0s[3], at_strides=2.0, spread_strides=1.0, period_s=T, frac=0.25)
    # stop: 2.5 strides, the stop, a hold; 2.0 s after the planned rest at the slowest clock
    r = pl.resolve(cond_of(t0_docs, "stop/two_step"))
    assert r["plan"][3] == ("ref", "v0.4", 2.5) and r["plan"][4] == ("transition", "stop2_v0.4")
    assert r["plan"][5][0] == "hold" and r["plan"][5][1] > 0
    t0s, t_end = ends(r["plan"])
    assert r["command_ref_s"] == t0s[4] and r["rest_ref_s"] == t0s[5]
    assert r["seconds"] == pytest.approx(t0s[5] / slow + 2.0 + 0.2, abs=1e-12)
    assert t_end == pytest.approx(r["seconds"] * fast, abs=1e-9)
    # a cut is a smoke check: shorter, and it says so
    cut = pl.resolve(dataclasses.replace(cond_of(t0_docs, "stop/two_step"), seconds=3.0))
    assert cut["seconds"] == 3.0 and cut["cut_from_s"] == r["seconds"] and cut["plan"] == r["plan"]


@needs_run
def test_the_planners_times_are_the_tracks(t0, t0_docs):
    d = setio.set_dir(SET, BODY, t0)
    ctx = bar.BarContext(BODY, d, {"refs.json": t0_docs["refs.json"], "transitions.json": t0_docs["transitions.json"]},
                         t0_docs["set.json"])
    for key in ("walk/v0.4", "turn/q90_right", "stop/two_step", "nudge/backward_mu0.54"):
        cond = cond_of(t0_docs, key)
        b = ctx.built(cond)
        assert isinstance(b, bar.BarBuilt) and ctx.built(cond) is b
        times = ctx.bar_planner.segment_times(b.plan)
        assert len(times) == len(b.track.segments)
        for (t, dur), seg in zip(times, b.track.segments, strict=True):
            assert t == pytest.approx(seg.t0, abs=1e-12) and dur == pytest.approx(seg.duration, abs=1e-12)
        if cond.judge == "bar_walk":  # Stage 1's rule: the walk ends in its gait (the periodic segment carries on)
            assert b.track.t_end == pytest.approx(b.seconds * bar.RATE, abs=1e-12)
        else:
            assert b.track.t_end >= b.seconds * bar.FAST * bar.RATE - 1e-9
        assert b.seconds == b.resolved["seconds"] and b.speed_cmd == pytest.approx(0.4 * 1.05)
        assert b.stride_s == pytest.approx(ctx.bar_planner.period_s("v0.4") / 1.05)
        assert b.omega == pytest.approx(bar.omega_of(BODY, d, 0.03), abs=0) and b.crouch_frac == 0.03
        assert b.air_gap_max_s == pytest.approx(0.5 * setio.min_planned_swing_s(b.track) / 1.05)
        assert b.legs == (cond.judge == "bar_turn")
        if cond.judge == "bar_turn":  # decision 14 from the whole track's own margins
            want = np.maximum(np.minimum(0.10, b.spec.ref_margins - 0.02), 1e-3)
            assert np.array_equal(b.required, want) and b.command["ref_s"] == b.track.segments[4].t0
            assert [e["name"] for e in b.track.events] == [cond.move] and b.track.events[0]["kind"] == "turn"
        else:
            assert np.all(b.required == 0.10)
        if cond.friction is not None:
            assert b.p is ctx.plant("hard", 0.54) and b.p is not ctx.plant("hard", None)
    # the track's LIPM frequency follows the set's crouch (and a control's)
    other = bar.BarContext(BODY, d, {"refs.json": t0_docs["refs.json"], "transitions.json": t0_docs["transitions.json"]},
                           dict(t0_docs["set.json"], crouch_frac=0.05))
    b5 = other.built(cond_of(t0_docs, "walk/v0.4"))
    assert b5.crouch_frac == 0.05 and b5.omega == pytest.approx(bar.omega_of(BODY, d, 0.05), abs=0)
    assert b5.omega > ctx.built(cond_of(t0_docs, "walk/v0.4")).omega and b5.track.omega == b5.omega
    ctx.release()
    assert ctx._plants == {} and ctx._built == {}


@needs_run
def test_a_nudges_force(t0, t0_docs):
    d = setio.set_dir(SET, BODY, t0)
    ctx = bar.BarContext(BODY, d, {"refs.json": t0_docs["refs.json"], "transitions.json": t0_docs["transitions.json"]},
                         t0_docs["set.json"])
    from designlab.stage1 import env

    for key, frac, direction in (("nudge/forward", 0.25, (1.0, 0.0)), ("nudge/left", 0.25, (0.0, 1.0)),
                                 ("nudge/backward_x0.5", 0.5, (-1.0, 0.0)), ("nudge/right_x0.375", 0.375, (0.0, -1.0))):
        b = ctx.built(cond_of(t0_docs, key))
        f = b.push_force()
        want = b.mass * frac * env.PUSH_BOUND_M_S[BODY] / 0.1
        assert np.linalg.norm(f) == pytest.approx(want, rel=1e-12) and env.PUSH_BOUND_M_S[BODY] == 0.589
        c, s = math.cos(b.home_yaw), math.sin(b.home_yaw)
        assert f / want == pytest.approx([c * direction[0] - s * direction[1], s * direction[0] + c * direction[1]])
        t_on = [b.push_onset(seed) for seed in (3042, 3043, 3044)]
        pu = b.resolved["push"]
        for seed, t in zip((3042, 3043, 3044), t_on, strict=True):
            u = panels.push_u(b.cond, seed)
            assert t == (pu["steady_ref_s"] + (2.0 + u) * pu["period_s"]) / 1.05 and 0.0 <= u < 1.0
            assert b.push(seed) == (t, 0.1, float(f[0]), float(f[1]))
        assert max(t_on) + 0.1 + 3.0 + 2.0 <= b.seconds
    assert ctx.built(cond_of(t0_docs, "walk/v0.4")).push(3042) is None


# ------------------------------------------------------------------------------------------------
# Rows on real episodes (the S1 view: Stage 1's own conditions)
# ------------------------------------------------------------------------------------------------
def strip(x, drop=("wall_s",)):
    """``x`` without wall times (and other named keys), NaN made comparable."""
    if isinstance(x, dict):
        return {k: strip(v, drop) for k, v in x.items() if k not in drop}
    if isinstance(x, (list, tuple)):
        return [strip(v, drop) for v in x]
    if isinstance(x, float) and math.isnan(x):
        return "NaN"
    return x


def s1_context() -> bar.BarContext:
    _, docs = setio.stage1_view(RUN, BODY)
    return bar.BarContext(BODY, RUN / BODY, docs)


def same_recorders(a: metrics.Recorder, b: metrics.Recorder) -> bool:
    return (a.rows == b.rows and a.steps == b.steps and np.array_equal(a.U, b.U, equal_nan=True)
            and np.array_equal(a.Q, b.Q) and np.array_equal(a.loads, b.loads) and np.array_equal(a.force_sq, b.force_sq))


@needs_run
def test_delayed_zero_is_the_tracker_row_and_five_is_servo_delay_by_hand():
    ctx = s1_context()
    cond = dataclasses.replace(next(c for c in panels.conditions() if c.key == "walk/brisk"), seconds=2.5)
    r2 = panels.tracker_rows(BODY, RUN / BODY, ("R2/42",))["R2/42"]
    # k = 0: the episode record is panels.Row.run's value for value, and the recorder's arrays are equal bit for bit
    stock = r2.run(ctx, cond, 3042)
    mine = bar.run_episode(bar.Delayed(r2, 0), ctx, cond, 3042)
    assert strip(json.loads(json.dumps(panels._jsonable(stock)))) == strip(json.loads(json.dumps(panels._jsonable(mine))))
    b0, o0, rec0 = r2.episode(ctx, cond, 3042)
    b1, o1, rec1 = bar.Delayed(r2, 0).episode(ctx, cond, 3042)
    assert b0 is b1 and same_recorders(rec0, rec1) and strip(panels._jsonable(o0), ("wall_s", "steps_per_s")) == strip(
        panels._jsonable(o1), ("wall_s", "steps_per_s"))
    # k = 5 on R0: speedfit.servo_delay applied by hand
    r0 = panels.tracker_rows(BODY, RUN / BODY, ("R0",))["R0"]
    b = ctx.built(cond)
    with speedfit.servo_delay(b.p, 5) as queue:
        assert queue.maxlen == 6
        _, _, by_hand = r0.episode(ctx, cond, 3042)
    assert "command" not in b.p.__dict__ and "reset" not in b.p.__dict__
    _, _, delayed = bar.Delayed(r0, 5).episode(ctx, cond, 3042)
    assert "command" not in b.p.__dict__ and same_recorders(by_hand, delayed)
    _, _, plain = bar.Delayed(r0, 0).episode(ctx, cond, 3042)
    _, _, stock0 = r0.episode(ctx, cond, 3042)
    assert same_recorders(plain, stock0) and not same_recorders(plain, delayed)


@needs_run
def test_stock_conditions_are_judged_by_panels_judge(tmp_path, monkeypatch):
    (cond,) = stock_cut(monkeypatch, 1.5)
    out = tmp_path / "out"
    doc = bar.evaluate(bar.STOCK_SET, BODY, out, ["R0@0ms"], ["walk/brisk"], episodes=2, procs=1, stage1_run=RUN,
                       log=quiet)
    rec = read(rec_path(out, "R0@0ms", "walk/brisk", set_name="S1"))
    assert rec["schema"] == "bar-1" and rec["set"] == "S1" and rec["status"] == "done" and rec["block"] == "seeds3042"
    assert rec["definition"]["stock"] is True and rec["definition"]["seconds"] == 1.5
    assert rec["terms"] == list(panels.terms_of(cond)) and rec["terms_but_A1"] is None
    assert rec["counts_but_A1"] is None and rec["counts"]["episodes"] == 2 and rec["counts"]["judged"] is False
    assert rec["rules"]["stage1"] == json.loads(json.dumps(panels.rules_of(cond))) and "stock" in rec["rules"]
    assert rec["omega"] == bar.omega_of(BODY, RUN / BODY, 0.03) and rec["crouch_frac"] == 0.03
    assert rec["set_freeze"] is None and rec["key"]["sources"]["bar"] == bar.bar_sources()["sha256"]
    assert rec["key"]["sources"]["panels"] == panels.panel_sources()["sha256"]
    assert rec["key"]["context"] == json.loads(json.dumps(panels.key_context(BODY)))
    assert sorted(rec["key"]["inputs"]) == panels.files_of(cond)
    assert rec["key_sha256"] == s2io.sha256_json(rec["key"]) and len(rec["sessions"]) == 1
    s = rec["sessions"][0]
    assert s["seeds_run"] == [3042, 3043] and s["blas_threads"] == 1 and s["procs"] == 1 and s["wall_s"] > 0
    # each episode is panels.Row.run's on the same condition: Stage 1's judge and Stage 1's trace
    ctx = s1_context()
    inner = panels.tracker_rows(BODY, RUN / BODY, ("R0",))["R0"]
    for e in rec["episodes"]:
        want = inner.run(ctx, cond, e["seed"], trace_root=tmp_path / "stock")
        want = json.loads(json.dumps(panels._jsonable(want)))
        want["trace"]["path"] = e["trace"]["path"]  # the row's directory differs: R0@0ms against R0
        assert strip(e) == strip(want)
        z = np.load(bar.block_dir(out, "S1", BODY, "seeds3042") / "eval" / "traces" / e["trace"]["path"])
        assert sorted(z.files) == ["cols", "q", "rows", "t", "t_ref"]  # panels.write_trace's five, nothing added
    assert doc["grid"]["walk/brisk"]["R0@0ms"]["counts"]["episodes"] == 2
    assert not list(out.rglob("*.tmp")) and not (out / ".lock").exists()
    assert not (bar.block_dir(out, "S1", BODY, "seeds3042") / "partial" / "R0@0ms" / "walk" / "brisk").exists()


@needs_run
def test_two_spawned_workers_run_and_a_worker_refuses_other_sources_or_threads(tmp_path, monkeypatch):
    stock_cut(monkeypatch, 1.0)
    out = tmp_path / "out"
    doc = bar.evaluate(bar.STOCK_SET, BODY, out, ["R0@0ms"], ["walk/brisk"], episodes=2, procs=2, stage1_run=RUN,
                       log=quiet)
    assert doc["sessions"][-1]["episodes_run"] == 2 and doc["sessions"][-1]["incidents"] == []
    rec = read(rec_path(out, "R0@0ms", "walk/brisk", set_name="S1"))
    assert [e["seed"] for e in rec["episodes"]] == [3042, 3043] and not any(e.get("error") for e in rec["episodes"])
    # the same episodes in this process: the worker count changes nothing
    one = tmp_path / "one"
    bar.evaluate(bar.STOCK_SET, BODY, one, ["R0@0ms"], ["walk/brisk"], episodes=2, procs=1, stage1_run=RUN, log=quiet)
    assert strip(read(rec_path(one, "R0@0ms", "walk/brisk", set_name="S1"))["episodes"]) == strip(rec["episodes"])
    # a worker refuses to start on other bar sources, other panel sources, or another thread count
    kw = dict(set_name="S1", rows=[], bar=bar.bar_sources()["sha256"],
              sources=panels.fingerprint(panels.panel_sources()["sha256"], panels.key_context(BODY)))
    assert bar._worker(BODY, str(RUN / BODY), "bar", bar.RATE, None, **kw).describe()["set"] == "S1"
    with pytest.raises(tune.PoolError, match="bar's sources"):
        bar._worker(BODY, str(RUN / BODY), "bar", bar.RATE, None, **dict(kw, bar="0" * 64))
    with pytest.raises(tune.PoolError, match="panels' sources"):
        bar._worker(BODY, str(RUN / BODY), "bar", bar.RATE, None, **dict(kw, sources="0" * 64))
    with pytest.raises(tune.InputError, match="differ"):
        bar._worker(BODY, str(RUN / BODY), "bar", bar.RATE, dict(files={"refs/nominal.npz": "0"}), **kw)
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    with pytest.raises(s2io.S2Error, match="one BLAS thread"):
        bar._worker(BODY, str(RUN / BODY), "bar", bar.RATE, None, **kw)
    with pytest.raises(s2io.S2Error, match="one BLAS thread"):
        bar.evaluate(bar.STOCK_SET, BODY, out, ["R0@0ms"], ["walk/brisk"], episodes=2, procs=1, stage1_run=RUN,
                     log=quiet)


# ------------------------------------------------------------------------------------------------
# The judge on real episodes of T0 (R0 rows only)
# ------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def t0_run(t0, tmp_path_factory) -> Path:
    """One R0@0ms episode of the walk, a turn, a nudge and the stop of T0 (seed 3042), with traces."""
    out = tmp_path_factory.mktemp("t0run")
    bar.evaluate(SET, BODY, out, ["R0@0ms"], ["walk/v0.4", "turn/q90_left", "nudge/forward", "stop/two_step"],
                 episodes=1, procs=1, root=t0, log=quiet)
    return out


EPISODE_KEYS = {"seed", "fell", "fall_time_s", "fall_cause", "survival_s", "planned_s", "strides",
                "worst_sustained_ratio", "worst_speed_ratio", "joint_shortfall_rad", "success", "succeeded",
                "success_but_A1", "succeeded_but_A1", "failed_terms", "failed_terms_but_A1", "command_s",
                "command_ref_s", "fell_before_command", "segment_reached", "window_reached", "omega", "crouch_frac",
                "measures", "trace", "relaxed", "wall_s"}


@needs_run
def test_a_walk_episode_carries_every_reading_and_stage1s_beside(t0, t0_run):
    rec = read(rec_path(t0_run, "R0@0ms", "walk/v0.4"))
    (e,) = rec["episodes"]
    assert EPISODE_KEYS <= set(e) and not e["fell"] and e["window_reached"] and e["command_s"] is None
    assert e["fell_before_command"] is None and e["segment_reached"] == "v0.4" and not e.get("error")
    assert rec["terms"] == list(bar.WALK_TERMS) and set(e["success"]) == set(bar.WALK_TERMS)
    assert set(e["success_but_A1"]) == set(bar.WALK_TERMS) - {"drive_sustained"} | {"drive_sustained_legs"}
    m = e["measures"]
    assert {"bar", "window", "stage1", "look", "convention"} <= set(m)
    b = m["bar"]
    assert {"along_commanded_heading_m", "lateral_from_line_max_m", "lateral_end_m", "heading_drift_deg",
            "lateral_from_episode_start_line_max_m", "along_target_m", "toe", "drives", "grf_peak_bw"} <= set(b)
    assert b["along_target_m"] == 4.0 and b["t_end_s"] - b["t_start_s"] == pytest.approx(10.0)
    # T0's "v0.4" is Stage 1's brisk walk (0.351 m/s): it covers less than 4.0 m, and the term says so
    assert 3.3 < b["along_commanded_heading_m"] < 4.0 and e["success"]["along"] is False
    assert e["success"]["line"] is (b["lateral_from_line_max_m"] <= 0.25)
    toe = b["toe"]
    assert {"toe_clearance_own_swing_m", "own_swings", "ref_swings", "interrupted_swings", "air_gaps",
            "air_gap_longest_s", "air_gap_max_s", "ok", "limit_m", "toe_clearance_own_duration_m", "worst_swing",
            "ref_swings_cut", "shortest_planned_s"} <= set(toe) and "swings" not in toe and "swing_details" not in toe
    # C3's Check names these directly under measures.bar; the reading as first written stands beside the judged one
    for k in bar.TOE_KEYS:
        assert b[k] == toe[k], k
    assert set(bar.TOE_KEYS) >= {"toe_clearance_own_swing_m", "own_swings", "ref_swings", "interrupted_swings",
                                 "air_gaps", "air_gap_max_s"}
    ws = toe["worst_swing"]
    assert set(ws) == {"foot", "t_lo_s", "t_td_s", "own_duration_s", "planned_s", "min_at_frac", "min_from_edge"}
    assert ws["planned_s"] >= toe["shortest_planned_s"] > 0.2 and 0.25 <= ws["min_at_frac"] <= 0.75
    assert toe["toe_clearance_own_duration_m"] <= toe["toe_clearance_own_swing_m"] + 1e-12  # (this walk lands late)
    assert "amended on 15 Nov" in rec["rules"]["toe"] and "STAGE2_75_PLAN.md C3" in rec["rules"]["toe"]
    assert toe["limit_m"] == pytest.approx(0.04 * 0.501854, rel=1e-5) and e["success"]["toe_clearance_own_swing"] is toe["ok"]
    assert b["stage1_toe_clearance_m"] == m["window"]["toe_clearance_m"]
    # the three readings' drive term: on this body the tail servo alone is over its sustained rating
    dr = b["drives"]
    assert dr["judged_span"] == "window" and dr["window"]["worst_a1"][0].startswith("tail") and dr["window"]["worst_a1"][1] > 1.0
    assert dr["window"]["worst_leg"][1] <= 1.0 and dr["window"]["worst"] == m["window"]["drive_sustained_worst"]
    assert e["success"]["drive_sustained"] is False and e["success_but_A1"]["drive_sustained_legs"] is True
    assert "drive_sustained" in e["failed_terms"] and "drive_sustained_legs" not in e["failed_terms_but_A1"]
    # the episode form of the drive ratios is control.run's worst_sustained_ratio
    assert dr["episode"]["worst"][0] == e["worst_sustained_ratio"][0]
    assert dr["episode"]["worst"][1] == pytest.approx(e["worst_sustained_ratio"][1], rel=1e-12)
    assert dr["episode"]["steps"] == [0, int(round(e["survival_s"] / 0.002))]
    # Stage 1's reading of the same episode is Stage 1's own record of its brisk walk (same track, seeds, omega)
    committed = next(x for x in read(RUN / BODY / "eval" / "R0" / "walk" / "brisk.json")["episodes"] if x["seed"] == 3042)
    assert m["stage1"]["success"] == committed["success"] and m["stage1"]["succeeded"] == committed["succeeded"]
    assert m["stage1"]["failed_terms"] == committed["failed_terms"] and m["stage1"]["terms"] == list(committed["success"])
    assert strip(m["window"]) == strip(committed["measures"]["window"])
    assert e["survival_s"] == committed["survival_s"] and e["planned_s"] == committed["planned_s"]
    assert e["worst_sustained_ratio"] == committed["worst_sustained_ratio"]
    # the recorder's one-step convention, read on this episode
    assert m["convention"]["holds"] is True and m["convention"]["touchdowns_on_rows"] >= 3
    # record level
    assert rec["counts"]["succeeded"] == 0 and rec["counts"]["judged"] is False and rec["counts_look"] is None
    assert rec["falls"] == dict(before_command=0, after_command=0, no_command=0)
    assert rec["set_freeze"]["index"] == 0 and len(rec["set_freeze"]["manifest_sha256"]) == 64
    assert rec["omega"] == e["omega"] == rec["key"]["omega"] and rec["crouch_frac"] == 0.03
    for k in ("clocks", "readings", "plans", "rows", "seeds", "pass_", "omega", "error", "key", "walk", "toe",
              "stage1", "delay", "set"):
        assert k in rec["rules"], k
    assert rec["rules"]["set"]["margin_rule"] == "a fixture's label" and "episode" in rec["rules"]["stage1"]


@needs_run
def test_a_bar_trace_holds_stage1s_arrays_and_what_the_readings_need(t0, t0_run):
    rec = read(rec_path(t0_run, "R0@0ms", "walk/v0.4"))
    (e,) = rec["episodes"]
    path = bar.block_dir(t0_run, SET, BODY, "seeds3042") / "eval" / "traces" / e["trace"]["path"]
    assert e["trace"]["path"] == "R0@0ms/walk/v0.4/3042.npz" and s2io.sha256_file(path) == e["trace"]["sha256"]
    z = np.load(path, allow_pickle=False)
    assert list(z.files)[:5] == ["t", "t_ref", "rows", "cols", "q"]
    assert {"contact", "steps", "dt", "weight", "t100", "low100", "ref_swing100", "planned_s100", "swings",
            "swing_cols", "command_s"} <= set(z.files)
    assert list(z["cols"]) == list(metrics.COLS) and z["rows"].dtype == np.float32 and z["t"].dtype == np.float64
    assert len(z["t"]) == e["trace"]["rows"] and len(z["t100"]) == e["trace"]["updates"] and len(z["t"]) == (len(z["t100"]) + 1) // 2
    assert float(z["dt"]) == 0.002 and math.isnan(float(z["command_s"])) and int(z["steps"]) == e["trace"]["steps"]
    # Stage 1's five arrays are those of Stage 1's own trace of this walk: the stock run of the same episode
    ctx = s1_context()
    cond = next(c for c in panels.conditions() if c.key == "walk/brisk")
    r0 = panels.tracker_rows(BODY, RUN / BODY, ("R0",))["R0"]
    _, _, live = r0.episode(ctx, cond, 3042)
    assert np.array_equal(z["rows"], live.U[::2].astype(np.float32), equal_nan=True)
    assert np.array_equal(z["t"], live.t[::2]) and np.array_equal(z["q"], live.Q[::2].astype(np.float32))
    # the contact bits, the low points and the reference's swing are the live recorder's
    contact = bar.read_trace_contact(path)
    assert contact.shape == (int(z["steps"]), 2) and np.array_equal(contact, live.contact())
    assert np.array_equal(z["t100"], live.t) and np.array_equal(z["ref_swing100"], live.col("ref_swing").astype(int))
    assert np.array_equal(z["low100"], live.U[:, [metrics.C["low0"], metrics.C["low1"]]].astype(np.float32))
    # reading (a) and the line come back from the trace alone
    toe = e["measures"]["bar"]["toe"]
    again = readings.toe_clearance_own_swing(z["t100"], z["low100"].astype(float), contact, float(z["dt"]),
                                             z["ref_swing100"], *toe["window_s"], toe["air_gap_max_s"],
                                             planned_s=z["planned_s100"])
    assert again["toe_clearance_own_swing_m"] == pytest.approx(toe["toe_clearance_own_swing_m"], abs=1e-8)
    assert again["toe_clearance_own_duration_m"] == pytest.approx(toe["toe_clearance_own_duration_m"], abs=1e-8)
    assert (again["own_swings"], again["ref_swings"], again["interrupted_swings"], again["air_gaps"]) == (
        toe["own_swings"], toe["ref_swings"], toe["interrupted_swings"], toe["air_gaps"])
    table = z["swings"]
    assert table.shape == (len(again["swings"]), 7) and list(z["swing_classes"]) == list(readings.SWING_CLASSES)
    assert list(z["swing_cols"]) == list(bar.SWING_COLS) and z["planned_s100"].dtype == np.float64
    read_ = table[table[:, 4] == 0]
    assert len(read_) == sum(toe["own_swings"]) and np.nanmin(read_[:, 3]) == pytest.approx(
        toe["toe_clearance_own_swing_m"], abs=1e-12)
    assert np.nanmin(table[:, 6]) <= toe["toe_clearance_own_duration_m"] + 1e-12
    # the rows' planned durations are the track's own planned swings on the simulated clock: the brisk walk's swing
    # in every row of a reference swing, nothing elsewhere
    in_swing = z["ref_swing100"] >= 0
    assert np.all(np.isnan(z["planned_s100"][~in_swing])) and np.all(np.isfinite(z["planned_s100"][in_swing]))
    built = s1_context().built(cond)
    planned = {round((b_ - a_) / 1.05, 9) for _, a_, b_ in setio.planned_swings(built.track)}
    assert {round(float(x), 9) for x in z["planned_s100"][in_swing]} <= planned
    assert np.array_equal(z["planned_s100"], bar.planned_swing_rows(built.track, live.col("t_ref"), live.col("ref_swing")),
                          equal_nan=True)
    assert np.all(read_[:, 5] > 0.2) and toe["shortest_planned_s"] == pytest.approx(min(planned), abs=1e-9)


@needs_run
def test_commanded_items_read_their_command_and_falls_after_it(t0, t0_docs, t0_run):
    """T0's "moves" join nothing, so R0 falls after each command: the episodes still carry the command instant and
    the split of falls, and the stop's event reaches control.run's decision 18 reading."""
    pl = planner_of(t0)
    turn = read(rec_path(t0_run, "R0@0ms", "turn/q90_left"))
    (e,) = turn["episodes"]
    res = pl.resolve(cond_of(t0_docs, "turn/q90_left"))
    assert turn["command"] == json.loads(json.dumps(res["command"])) and e["command_ref_s"] == res["command_ref_s"]
    assert e["command_s"] is not None and 0.90 * res["command_ref_s"] / 1.05 < e["command_s"] < 1.10 * res["command_ref_s"] / 1.05
    assert e["fell"] and e["fall_time_s"] > e["command_s"] and e["fell_before_command"] is False
    assert e["failed_terms"] == ["survived"] == e["failed_terms_but_A1"] and not any(e["success"].values())
    assert turn["terms"] == list(bar.TURN_TERMS) and turn["falls"] == dict(before_command=0, after_command=1, no_command=0)
    assert e["measures"]["bar"]["at_s"] == 2.0 and e["measures"]["bar"]["sign"] == 1.0
    assert e["measures"]["bar"]["trunk_heading_at_command_deg"] is not None and e["measures"]["legs"]["poses"] > 0
    # C5: the joint range and the leg gap are read over the window (here cut short by the fall), the drives' peak
    # speed over the episode; the whole episode's readings are reported beside
    legs = e["measures"]["legs"]
    assert legs["span"] == "window" and legs["window_complete"] is False and legs["window_s"][0] == e["command_s"]
    assert legs["poses"] < legs["poses_episode"] == e["measures"]["legs_episode"]["poses"]
    mb = e["measures"]["bar"]
    assert (mb["drive_speed_span"], mb["joint_range_span"], mb["no_leg_interference_span"]) == ("episode", "window", "window")
    assert mb["joint_range_episode"]["joint_shortfall_rad"] == e["joint_shortfall_rad"]
    assert (turn["drive_speed_span"], turn["joint_range_span"]) == ("episode", "window")
    assert "whole episode (the recorder holds the episode's peak only), which is at least as strict as the window" in " ".join(
        turn["rules"]["turn"].split())
    assert turn["counts"]["falls"] == 1 and turn["counts"]["errors"] == 0
    stop = read(rec_path(t0_run, "R0@0ms", "stop/two_step"))
    (e,) = stop["episodes"]
    b = e["measures"]["bar"]
    assert stop["terms"] == list(bar.STOP_TERMS) and b["reached_command"] and b["command_convention"].startswith("causal")
    assert b["decision_18"]["name"] == "stop2_v0.4" and b["decision_18"]["reached"] is True
    assert b["decision_18"]["command_ref_s"] == e["command_ref_s"] == stop["command"]["ref_s"]
    assert abs(b["decision_18"]["command_s"] - e["command_s"]) <= 0.011  # control.run takes the update after it
    assert isinstance(b["touchdowns_after_command"], int) and b["last_touchdown_before_command"]["before_command_s"] > 0
    assert "toe_reported" in b and b["drives"]["judged_span"] == "episode" and b["drives"]["window"] is None
    nudge = read(rec_path(t0_run, "R0@0ms", "nudge/forward"))
    (e,) = nudge["episodes"]
    push = e["measures"]["push"]
    assert nudge["terms"] == list(bar.NUDGE_TERMS) and push["frac"] == 0.25 and push["direction"] == "forward"
    assert push["dv_m_s"] == pytest.approx(0.25 * 0.589) and e["command_s"] == push["onset_s"]
    assert push["end_s"] == pytest.approx(push["onset_s"] + 0.1) and e["measures"]["recovery"]["limit_s"] == 3.0
    assert e["measures"]["recovery"]["planned_until_s"] == nudge["seconds"]
    assert e["fell_before_command"] is (e["fell"] and e["fall_time_s"] < e["command_s"])


@needs_run
def test_an_incomplete_judged_window_is_an_error_not_a_failed_term(t0, t0_docs, tmp_path, monkeypatch):
    d = setio.set_dir(SET, BODY, t0)
    ctx = bar.BarContext(BODY, d, {"refs.json": t0_docs["refs.json"], "transitions.json": t0_docs["transitions.json"]},
                         t0_docs["set.json"])
    row = bar.rows(BODY, ["R0@0ms"], RUN)["R0@0ms"]
    for key, cut in (("walk/v0.4", 6.0), ("turn/q90_left", 6.0), ("nudge/left", 6.0), ("stop/two_step", 6.0)):
        cond = dataclasses.replace(cond_of(t0_docs, key), seconds=cut)
        with pytest.raises(bar.Unreadable, match="no fall"):
            bar.run_episode(row, ctx, cond, 3042)
    # through the runner: the worker records it as an episode that failed to run; the record is not judged
    cut = dataclasses.replace(cond_of(t0_docs, "walk/v0.4"), seconds=6.0)
    monkeypatch.setattr(bar, "conditions", lambda docs, block="seeds3042", every=False: [cut])
    out = tmp_path / "out"
    doc = bar.evaluate(SET, BODY, out, ["R0@0ms"], None, episodes=1, procs=1, root=t0, log=quiet)
    rec = read(rec_path(out, "R0@0ms", "walk/v0.4"))
    (e,) = rec["episodes"]
    assert "Unreadable" in e["error"] and e["fell"] and e["failed_terms"] == list(bar.WALK_TERMS)
    assert rec["counts"]["errors"] == 1 and rec["counts"]["falls"] == 0 and rec["counts"]["pass_"] is None
    assert rec["counts_but_A1"]["errors"] == 1 and rec["falls"] == dict(before_command=0, after_command=0, no_command=0)
    assert doc["grid"]["walk/v0.4"]["R0@0ms"]["counts"]["errors"] == 1


# ------------------------------------------------------------------------------------------------
# Counts, records and the runner's rules (synthetic episodes: no simulation)
# ------------------------------------------------------------------------------------------------
def test_the_two_readings_count_apart_when_only_the_tail_is_over():
    cond = bar.BarCondition("walk", "v0.4", True, "bar_walk", set=SET)
    seeds = bar.BLOCKS["seeds3042"]
    eps = [fake_episode(cond, s, tail_over=True) for s in seeds]
    c2, c3 = bar.counts(eps, seeds, "success"), bar.counts(eps, seeds, "but_A1")
    assert (c2["succeeded"], c2["pass_"], c2["judged"]) == (0, False, True)
    assert (c3["succeeded"], c3["pass_"], c3["judged"]) == (20, True, True)
    assert c2["survived"] == c3["survived"] == 20 and c2["success_lcb"] == 0.0 and c3["success_lcb"] == pytest.approx(0.861, abs=1e-3)
    f = bar.failure("R2/42@10ms", eps, None, c2["pass_"], "success")
    assert f["label"] == "body-limited: drive_sustained" and f["kind"] == "success" and f["reading"] == "success"
    assert bar.failure("R2/42@10ms", eps, None, c3["pass_"], "but_A1") is None
    # the pass rule: 20 up and at least 17 succeed, on a whole block in order, with no episode that failed to run
    eps = [fake_episode(cond, s, fail=("line",) if i < 3 else ()) for i, s in enumerate(seeds)]
    assert bar.counts(eps, seeds)["pass_"] is True and bar.counts(eps[:19] + [fake_episode(cond, 3061, fell=True)], seeds)["pass_"] is False
    eps4 = [fake_episode(cond, s, fail=("line",) if i < 4 else ()) for i, s in enumerate(seeds)]
    assert bar.counts(eps4, seeds)["pass_"] is False and bar.counts(eps4, seeds)["succeeded"] == 16
    assert bar.counts(eps[:10], seeds[:10])["pass_"] is None and bar.counts(eps, list(range(4042, 4062)))["judged"] is False
    assert bar.counts(eps[::-1], seeds)["judged"] is False
    broken = eps[:19] + [bar.failed_episode(cond, 3061, "boom")]
    c = bar.counts(broken, seeds)
    assert (c["errors"], c["falls"], c["judged"], c["pass_"]) == (1, 0, False, None)
    # the replication block is judged on its own seeds
    rep = bar.BLOCKS["seeds5042"]
    assert bar.counts([fake_episode(cond, s) for s in rep], rep)["pass_"] is True
    assert bar.counts([fake_episode(cond, s) for s in rep], seeds)["judged"] is False
    assert bar.terms_but_a1(cond) == tuple("drive_sustained_legs" if t == "drive_sustained" else t for t in bar.WALK_TERMS)
    assert bar.term_class("drive_sustained_legs") == "hardware" and bar.term_class("heading") == "controller"
    fe = bar.failed_episode(cond, 3042, "why", ("slip",), 1.5)
    assert fe["error"] == "why" and fe["relaxed"] == ["slip"] and fe["wall_s"] == 1.5 and not fe["succeeded_but_A1"]
    assert set(fe["success_but_A1"]) == set(bar.terms_but_a1(cond))
    nominal = dataclasses.replace(cond, name="nominal", walk="nominal", gated=False)
    assert bar.terms_of(nominal) == bar.WALK_TERMS + ("cadence", "stride", "grf")


@needs_run
def test_a_record_holds_both_readings_and_the_verdict_words(t0, tmp_path, monkeypatch):
    out = tmp_path / "out"

    def plan(row, cond, seed, relax):
        if row.name.startswith("R2/43"):
            return fake_episode(cond, seed, tail_over=True, fail=("line",) if seed < 3050 else ())
        return fake_episode(cond, seed, tail_over=True)

    calls = faked(monkeypatch, plan)
    doc = bar.evaluate(SET, BODY, out, ["R2/42@10ms", "R2/43@10ms", "R0@10ms", "R2/42@4ms", "R2/43@4ms"], ["walk/v0.4"],
                       procs=1, root=t0, attribute="none", log=quiet)
    assert len(calls) == 100 and doc["sessions"][-1]["episodes_run"] == 100
    rec = read(rec_path(out, "R2/42@10ms", "walk/v0.4"))
    assert rec["counts"]["pass_"] is False and rec["counts"]["succeeded"] == 0 and rec["counts"]["reading"] == "success"
    assert rec["counts_but_A1"]["pass_"] is True and rec["counts_but_A1"]["succeeded"] == 20
    assert rec["counts_look"] is None and rec["counts_look_reason"] == bar.LOOK_NOT_DEFINED
    assert rec["failure"]["label"] == "body-limited: drive_sustained" and rec["failure_but_A1"] is None
    assert rec["gated"] is True and rec["controller"]["latency_ms"] == 10 and rec["a1_over"] is False
    assert rec["term_classes"]["drive_sustained_legs"] == "hardware" and rec["term_classes"]["survived"] is None
    assert rec["seeds"] == list(range(3042, 3062)) and [e["seed"] for e in rec["episodes"]] == rec["seeds"]
    assert read(rec_path(out, "R2/43@10ms", "walk/v0.4"))["counts_but_A1"]["succeeded"] == 12
    v = doc["verdicts"]["walk/v0.4"]
    assert v["R2@10ms"]["judged_delay"] is True and v["R2@10ms"]["rows"] == ["R2/42@10ms", "R2/43@10ms"]
    assert v["R2@10ms"]["reading_1"] == dict(verdict=None, reason=bar.LOOK_NOT_DEFINED)
    assert v["R2@10ms"]["reading_2"]["verdict"] == "body-limited"
    assert v["R2@10ms"]["reading_3"]["verdict"] == "provisional but for A1"  # one of the two seeds passes
    # away from 10 ms a cell gives "passes at <ms> ms" and never either word
    assert v["R2@4ms"]["judged_delay"] is False and v["R2@4ms"]["reading_3"]["verdict"] == "one of two passes at 4 ms but for A1"
    assert v["R2@4ms"]["reading_2"]["verdict"] == "does not pass at 4 ms"
    assert not any("shown" in x["verdict"] or "provisional" in x["verdict"] for x in (v["R2@4ms"]["reading_2"],
                                                                                    v["R2@4ms"]["reading_3"]))
    assert "R0@10ms" not in v  # an R0 row gives no verdict
    text = bar.summary_text(doc)
    assert "but for A1: 20 succeed pass" in text and "every limit: 0 succeed FAIL" in text
    summary = read(bar.block_dir(out, SET, BODY, "seeds3042") / "eval" / "summary.json")
    assert summary["grid"] == doc["grid"] and len(summary["sessions"]) == 1 and summary["block"] == "seeds3042"
    # both seeds pass both readings: shown
    faked(monkeypatch, lambda row, cond, seed, relax: fake_episode(cond, seed))
    doc = bar.evaluate(SET, BODY, tmp_path / "ok", ["R2/42@10ms", "R2/43@10ms"], ["walk/v0.4"], procs=1, root=t0, log=quiet)
    v = doc["verdicts"]["walk/v0.4"]["R2@10ms"]
    assert v["reading_2"]["verdict"] == "shown" and v["reading_3"]["verdict"] == "shown but for A1"


@needs_run
def test_a_finished_record_of_another_key_is_refused_and_supersede_keeps_it(t0, tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    calls = faked(monkeypatch, lambda row, cond, seed, relax: fake_episode(cond, seed))
    args = (SET, BODY, out, ["R0@10ms"], ["walk/v0.4", "stop/two_step"])
    bar.evaluate(*args, episodes=3, procs=1, root=t0, log=quiet)
    path = rec_path(out, "R0@10ms", "walk/v0.4")
    before, old = path.read_bytes(), read(path)
    assert len(calls) == 6
    # the same key: the finished records are skipped and left as they are
    doc = bar.evaluate(*args, episodes=3, procs=1, root=t0, log=quiet)
    assert len(calls) == 6 and doc["sessions"][-1]["episodes_run"] == 0 and path.read_bytes() == before
    assert doc["sessions"][-1]["skipped"] == ["R0@10ms walk/v0.4", "R0@10ms stop/two_step"]
    # another key (bar.py's bytes changed): refused before any episode, with the differing fields; nothing is touched
    real = bar.bar_sources()
    monkeypatch.setattr(bar, "bar_sources", lambda: dict(real, sha256="f" * 64))
    with pytest.raises(bar.KeyMismatch) as exc:
        bar.evaluate(*args, episodes=3, procs=1, root=t0, log=quiet)
    msg = str(exc.value)
    assert "key.sources.bar" in msg and real["sha256"][:20] in msg and "f" * 20 in msg and "--supersede" in msg
    assert str(path) in msg and str(rec_path(out, "R0@10ms", "stop/two_step")) in msg
    assert len(calls) == 6 and path.read_bytes() == before and not (out / ".lock").exists()
    # other seeds are another key too
    with pytest.raises(bar.KeyMismatch, match="key.seeds"):
        bar.evaluate(*args, episodes=4, procs=1, root=t0, log=quiet)
    assert path.read_bytes() == before
    # the CLI prints the fields and exits 1
    code = bar.main(["run", "--set", SET, "--set-root", str(t0), "--body", BODY, "--rows", "R0@10ms", "--conditions",
                     "walk/v0.4", "--episodes", "3", "--procs", "1", "--out", str(out)])
    assert code == 1 and "key.sources.bar" in capsys.readouterr().err and path.read_bytes() == before
    # --supersede needs its reason; with it the old record is moved aside, kept, and the new one says why
    with pytest.raises(bar.BarError, match="supersede-reason"):
        bar.evaluate(*args, episodes=3, procs=1, root=t0, supersede=True, log=quiet)
    assert path.read_bytes() == before
    doc = bar.evaluate(*args, episodes=3, procs=1, root=t0, supersede=True, supersede_reason="a reading was fixed",
                       log=quiet)
    kept = path.with_name("v0.4.json.superseded") / f"{old['key_sha256'][:12]}.json"
    assert kept.read_bytes() == before and len(calls) == 12
    new = read(path)
    assert new["key"]["sources"]["bar"] == "f" * 64 and new["key_sha256"] != old["key_sha256"]
    assert new["supersedes"]["reason"] == "a reading was fixed" and new["supersedes"]["label"] == "re-run after a result"
    assert new["supersedes"]["key_sha256"] == old["key_sha256"] and new["supersedes"]["registered"]["counts"] == old["counts"]
    assert doc["grid"]["walk/v0.4"]["R0@10ms"]["supersedes"]["reason"] == "a reading was fixed"
    assert list(doc["grid"]) == ["stop/two_step", "walk/v0.4"]  # the superseded copies are not in the grid
    # a "not run" record holds no result: it is replaced without --supersede when the reference is there
    make_t0(tmp_path / "b", unavailable=("turn90_pi4_left",))
    bar.evaluate(SET, BODY, tmp_path / "o2", ["R0@10ms"], ["turn/q90_left"], episodes=2, procs=1, root=tmp_path / "b",
                 log=quiet)
    p2 = rec_path(tmp_path / "o2", "R0@10ms", "turn/q90_left")
    assert read(p2)["status"] == "not run"
    bar.evaluate(SET, BODY, tmp_path / "o2", ["R0@10ms"], ["turn/q90_left"], episodes=2, procs=1, root=t0, log=quiet)
    assert read(p2)["status"] == "done" and not p2.with_name("q90_left.json.superseded").exists()


@needs_run
def test_resume_per_episode(t0, tmp_path, monkeypatch):
    out = tmp_path / "out"
    args = (SET, BODY, out, ["R0@10ms"], ["walk/v0.4"])
    pdir = bar.block_dir(out, SET, BODY, "seeds3042") / "partial" / "R0@10ms" / "walk" / "v0.4"

    def killed_at(seed_):
        def plan(row, cond, seed, relax):
            if seed == seed_:
                raise KeyboardInterrupt  # not an Exception: the run dies as under a kill
            return fake_episode(cond, seed)
        return plan

    calls = faked(monkeypatch, killed_at(3044))
    with pytest.raises(KeyboardInterrupt):
        bar.evaluate(*args, episodes=5, procs=1, root=t0, log=quiet)
    assert [c[2] for c in calls] == [3042, 3043, 3044] and not rec_path(out, "R0@10ms", "walk/v0.4").exists()
    assert not (out / ".lock").exists()
    (kdir,) = list(pdir.iterdir())
    assert sorted(p.name for p in kdir.iterdir()) == ["3042.json", "3043.json"] and len(kdir.name) == 16
    part = read(kdir / "3042.json")
    assert part["episode"]["seed"] == 3042 and part["key_sha256"].startswith(kdir.name) and part["session"]["procs"] == 1
    # a torn partial file is ignored; a partial of another key is not reused; one whose episode failed to run is not
    (kdir / "3043.json").write_text('{"schema": "bar-partial-1", "key_sha2')
    s2io.write_json(kdir / "3045.json", dict(part, seed=3045, key_sha256="0" * 64, episode=dict(part["episode"], seed=3045)))
    s2io.write_json(kdir / "3046.json", dict(part, seed=3046, episode=bar.failed_episode(
        bar.BarCondition("walk", "v0.4", True, "bar_walk"), 3046, "boom")))
    (pdir / ("0" * 16)).mkdir()
    s2io.write_json(pdir / ("0" * 16) / "3044.json", dict(part, seed=3044, key_sha256="0" * 64))
    calls = faked(monkeypatch, lambda row, cond, seed, relax: fake_episode(cond, seed))
    doc = bar.evaluate(*args, episodes=5, procs=1, root=t0, log=quiet)
    assert [c[2] for c in calls] == [3043, 3044, 3045, 3046]
    s = doc["sessions"][-1]
    assert s["reused_partial"] == ["R0@10ms walk/v0.4: seeds [3042]"] and s["episodes_run"] == 4
    rec = read(rec_path(out, "R0@10ms", "walk/v0.4"))
    assert [e["seed"] for e in rec["episodes"]] == [3042, 3043, 3044, 3045, 3046] and rec["counts"]["errors"] == 0
    assert [x["seeds_run"] for x in rec["sessions"]] == [[3042], [3043, 3044, 3045, 3046]]
    assert rec["sessions"][0]["note"] == "episodes kept from a run that did not finish" and rec["sessions"][0]["ended_at"] is None
    assert not pdir.exists()  # the partial files go when the record is written
    # a complete record is skipped; an episode that failed to run is run again and merged in
    assert bar.evaluate(*args, episodes=5, procs=1, root=t0, log=quiet)["sessions"][-1]["episodes_run"] == 0

    def raising(row, cond, seed, relax):
        if seed == 3043:
            raise RuntimeError("the episode broke")
        return fake_episode(cond, seed)

    out2 = tmp_path / "out2"
    faked(monkeypatch, raising)
    doc = bar.evaluate(SET, BODY, out2, ["R0@10ms"], ["walk/v0.4"], episodes=3, procs=1, root=t0, log=quiet)
    rec = read(rec_path(out2, "R0@10ms", "walk/v0.4"))
    bad = rec["episodes"][1]
    assert "RuntimeError: the episode broke" in bad["error"] and bad["fell"] and bad["failed_terms"] == list(bar.WALK_TERMS)
    assert rec["counts"]["errors"] == 1 and rec["counts"]["falls"] == 0 and rec["counts"]["pass_"] is None
    calls = faked(monkeypatch, lambda row, cond, seed, relax: fake_episode(cond, seed))
    doc = bar.evaluate(SET, BODY, out2, ["R0@10ms"], ["walk/v0.4"], episodes=3, procs=1, root=t0, log=quiet)
    assert [c[2] for c in calls] == [3043] and doc["sessions"][-1]["rerun_failed"] == ["R0@10ms walk/v0.4: seeds [3043]"]
    rec = read(rec_path(out2, "R0@10ms", "walk/v0.4"))
    assert rec["counts"]["errors"] == 0 and [x["seeds_run"] for x in rec["sessions"]] == [[3042, 3043, 3044], [3043]]


@needs_run
def test_a_second_runner_on_a_locked_run_directory_is_refused(t0, tmp_path, monkeypatch):
    out = tmp_path / "out"
    calls = faked(monkeypatch, lambda row, cond, seed, relax: fake_episode(cond, seed))
    with s2io.Lock(out / ".lock"):
        with pytest.raises(s2io.S2Error, match="is held by pid"):
            bar.evaluate(SET, BODY, out, ["R0@10ms"], ["walk/v0.4"], episodes=1, procs=1, root=t0, log=quiet)
    assert calls == [] and not list(out.rglob("*.json"))
    bar.evaluate(SET, BODY, out, ["R0@10ms"], ["walk/v0.4"], episodes=1, procs=1, root=t0, log=quiet)
    assert len(calls) == 1


@needs_run
def test_a_condition_without_a_reference_gets_a_not_run_record(tmp_path, monkeypatch):
    root = tmp_path / "sets"
    make_t0(root, unavailable=("turn90_pi4_left",), drop=("stop2_v0.4",))
    out = tmp_path / "out"
    calls = faked(monkeypatch, lambda row, cond, seed, relax: fake_episode(cond, seed))
    doc = bar.evaluate(SET, BODY, out, ["R0@10ms", "R2/42@10ms"], None, episodes=2, procs=1, root=root, log=quiet)
    # all 16 gated conditions have a record under each row
    for row in ("R0@10ms", "R2/42@10ms"):
        recs = {k: read(rec_path(out, row, k)) for k in GATED}
        not_run = sorted(k for k, r in recs.items() if r["status"] == "not run")
        assert not_run == ["stop/two_step", "stop/two_step_mu0.54", "turn/q90_left", "turn/q90_left_mu0.54"]
        t = recs["turn/q90_left"]
        assert t["reason"] == "no reference within the thigh twist and the leg drives" and t["numbers"] == dict(hip_yaw_range_use=0.93)
        assert t["reference"] == "transitions/turn90_pi4_left" and t["episodes"] == [] and t["counts"]["pass_"] is None
        assert t["counts_but_A1"]["pass_"] is None and t["key"]["plan"] is None and t["key"]["inputs"] == {}
        assert recs["stop/two_step"]["reason"] == "the set has no entry 'stop2_v0.4'"
        assert all(r["status"] == "done" for k, r in recs.items() if k not in not_run)
    assert len(calls) == 2 * 12 * 2 and len(doc["sessions"][-1]["not_run"]) == 8
    assert doc["verdicts"]["turn/q90_left"]["R2@10ms"]["reading_2"]["verdict"] == "not run"
    # the second run writes nothing new
    stamp = read(rec_path(out, "R0@10ms", "turn/q90_left"))["ended_at"]
    bar.evaluate(SET, BODY, out, ["R0@10ms", "R2/42@10ms"], None, episodes=2, procs=1, root=root, log=quiet)
    assert len(calls) == 48 and read(rec_path(out, "R0@10ms", "turn/q90_left"))["ended_at"] == stamp
    # a set that is not frozen, or whose file differs from its manifest, is refused
    make_t0(tmp_path / "open", freeze=False)
    with pytest.raises(tune.InputError, match="not frozen"):
        bar.evaluate(SET, BODY, tmp_path / "o", ["R0@10ms"], ["walk/v0.4"], episodes=1, procs=1, root=tmp_path / "open",
                     log=quiet)
    d = make_t0(tmp_path / "changed")
    with open(d / "refs" / "v0.4.npz", "ab") as fh:
        fh.write(b"\0")
    with pytest.raises(tune.InputError, match="sha256"):
        bar.evaluate(SET, BODY, tmp_path / "o", ["R0@10ms"], ["walk/v0.4"], episodes=1, procs=1,
                     root=tmp_path / "changed", log=quiet)


@needs_run
def test_attribution_lifts_limits_on_r2_rows_and_cites_the_walk_record(t0, tmp_path, monkeypatch):
    out = tmp_path / "out"

    def plan(row, cond, seed, relax):
        i = seed - 3042
        if relax:  # a lifted run: the limit lifted, nothing falls
            return fake_episode(cond, seed, relax=relax)
        if cond.key == "walk/v0.4":
            return fake_episode(cond, seed, fell=i < 6)
        if cond.key == "turn/q90_left":  # six falls, four of them after the command
            return fake_episode(cond, seed, fell=i < 6, before=i < 2)
        if cond.key == "turn/q90_right":  # five falls after the command
            return fake_episode(cond, seed, fell=i < 5, before=False)
        return fake_episode(cond, seed, fell=i < 2, before=False)

    calls = faked(monkeypatch, plan)
    conds = ["walk/v0.4", "turn/q90_left", "turn/q90_right", "stop/two_step"]
    rows = ["R2/42@10ms", "R0@10ms", "R2/42@4ms"]
    doc = bar.evaluate(SET, BODY, out, rows, conds, procs=1, root=t0, attribute="judged", log=quiet)
    lifted = [c for c in calls if c[3]]
    assert {(c[0], c[1], c[3]) for c in lifted} == {("R2/42@10ms", "walk/v0.4", ("slip",)),
                                                    ("R2/42@10ms", "turn/q90_right", ("slip",))}
    assert len(lifted) == 40 and doc["sessions"][-1]["attribution_episodes_run"] == 40
    walk = read(rec_path(out, "R2/42@10ms", "walk/v0.4"))
    a = walk["attribution"]
    assert a["label"] == "body-limited: slip" and a["falls"] == 6 and a["episodes"] == 20 and a["candidates"] == ["slip"]
    assert a["lifted"]["slip"]["significant"] is True and a["rule_bar"] == bar.RULES["attribution"]
    assert walk["failure"]["falls_label"] == "body-limited: slip" and walk["failure"]["kind"] == "falls"
    assert walk["failure_but_A1"]["falls_label"] == "body-limited: slip"
    assert walk["falls"] == dict(before_command=0, after_command=0, no_command=6)
    # the lifted runs leave no trace and no partial file behind
    assert not (bar.block_dir(out, SET, BODY, "seeds3042") / "partial").exists()
    left = read(rec_path(out, "R2/42@10ms", "turn/q90_left"))
    assert left["falls"] == dict(before_command=2, after_command=4, no_command=0)
    assert left["attribution"]["label"] == "lead-in falls: see walk/v0.4" and left["attribution"]["cites"] == "walk/v0.4"
    assert left["attribution"]["record"] == "T0/75-M-H-10/seeds3042/eval/R2/42@10ms/walk/v0.4.json"
    assert left["attribution"]["episodes"] == 0 and left["failure"]["falls_label"] == "lead-in falls: see walk/v0.4"
    right = read(rec_path(out, "R2/42@10ms", "turn/q90_right"))
    assert right["attribution"]["label"] == "body-limited: slip" and right["falls"]["after_command"] == 5
    stop = read(rec_path(out, "R2/42@10ms", "stop/two_step"))
    assert stop["attribution"]["label"] == "lead-in falls: see walk/v0.4" and stop["attribution"]["falls_after_command"] == 2
    # R0 rows are not attributed; "judged" leaves the other delays alone
    assert read(rec_path(out, "R0@10ms", "walk/v0.4"))["attribution"] is None
    assert read(rec_path(out, "R0@10ms", "walk/v0.4"))["failure"]["falls_label"].startswith("not attributed (falls; attribution is applied to R2")
    four = read(rec_path(out, "R2/42@4ms", "walk/v0.4"))
    assert four["attribution"] is None and four["failure"]["falls_label"] == "not attributed (falls; run with --attribute)"
    # --attribute all on finished records: only the attribution runs
    n = len(calls)
    doc = bar.evaluate(SET, BODY, out, rows, conds, procs=1, root=t0, attribute="all", log=quiet)
    assert doc["sessions"][-1]["episodes_run"] == 0 and len(calls) == n + 40
    assert read(rec_path(out, "R2/42@4ms", "walk/v0.4"))["attribution"]["label"] == "body-limited: slip"
    assert read(rec_path(out, "R2/42@4ms", "turn/q90_left"))["attribution"]["cites"] == "walk/v0.4"
    # a walk record with fewer than five falls is controller-limited by decision 20, with no limit lifted
    out3 = tmp_path / "few"
    calls = faked(monkeypatch, lambda row, cond, seed, relax: fake_episode(cond, seed, fell=seed < 3045))
    bar.evaluate(SET, BODY, out3, ["R2/42@10ms"], ["walk/v0.4"], procs=1, root=t0, log=quiet)
    a = read(rec_path(out3, "R2/42@10ms", "walk/v0.4"))["attribution"]
    assert len(calls) == 20 and a["episodes"] == 0 and a["label"].startswith("controller-limited (decision 20; 3 falls")


@needs_run
def test_replication_takes_what_passed_and_is_never_pooled(tmp_path, monkeypatch):
    root = tmp_path / "sets"
    d = make_t0(root)
    doc = s2io.read_json(d / "transitions.json")  # the right turn frozen with a1_over (a fixture's flag)
    doc["transitions"]["turn90_pi4_right"]["a1_over"] = True
    manifest = {}
    s2io.write_json(d / "transitions.json", doc)
    sd = s2io.read_json(d / "set.json")
    for name, m in sd["freezes"][0]["manifest"].items():
        kind = m["kind"]
        entry = (doc["transitions"] if kind == "move" else s2io.read_json(d / "refs.json")["references"])[name]
        manifest[name] = dict(m, entry_sha256=setio.entry_sha256(kind, entry))
    sd["freezes"][0].update(manifest=manifest, manifest_sha256=s2io.sha256_json(manifest))
    s2io.write_json(d / "set.json", sd)
    out = tmp_path / "out"

    def first(row, cond, seed, relax):
        if row.name == "R2/43@10ms":
            return fake_episode(cond, seed, fail=("joint_range",))  # fails both readings
        if cond.key == "walk/v0.4_mu0.54":
            return fake_episode(cond, seed, tail_over=True)  # passes reading 3 only
        return fake_episode(cond, seed)

    faked(monkeypatch, first)
    conds = ["walk/v0.4", "walk/v0.4_mu0.54", "turn/q90_right"]
    rows = ["R2/42@10ms", "R2/43@10ms", "R0@10ms"]
    one = bar.evaluate(SET, BODY, out, rows, conds, procs=1, root=root, attribute="none", log=quiet)
    turn = read(rec_path(out, "R2/42@10ms", "turn/q90_right"))
    # a turn frozen with a1_over: both readings recorded, reading 2 not read (no pass), reading 3 can pass
    assert turn["a1_over"] is True and turn["counts"]["pass_"] is None and turn["counts"]["not_read"] == "a1_over"
    assert turn["counts"]["reason"] == "not read: a1_over"
    assert turn["counts"]["succeeded"] == 20 and turn["counts_but_A1"]["pass_"] is True
    assert one["verdicts"]["turn/q90_right"]["R2@10ms"]["reading_2"]["verdict"] == "not read: a1_over"
    assert one["verdicts"]["turn/q90_right"]["R2@10ms"]["reading_3"]["verdict"] == "provisional but for A1"
    frozen = {p: p.read_bytes() for p in bar.block_dir(out, SET, BODY, "seeds3042").rglob("*.json")}
    # the replication block: exactly the records that passed in reading 2 or 3, of rows with a verdict
    calls = faked(monkeypatch, lambda row, cond, seed, relax: fake_episode(cond, seed, fail=("joint_range",)))
    two = bar.evaluate(SET, BODY, out, rows, conds, block="seeds5042", passed_of="seeds3042", procs=1, root=root,
                       attribute="none", log=quiet)
    assert {(c[0], c[1]) for c in calls} == {("R2/42@10ms", k) for k in conds} and len(calls) == 60
    assert {c[2] for c in calls} == set(range(5042, 5062))
    assert sorted(two["sessions"][-1]["not_selected"]) == sorted(f"{r} {k}" for r in ("R2/43@10ms", "R0@10ms") for k in conds)
    rep = {k: read(rec_path(out, "R2/42@10ms", k, block="seeds5042")) for k in conds}
    assert rep["walk/v0.4"]["replication_of"]["readings"] == ["success", "but_A1"]
    assert rep["walk/v0.4_mu0.54"]["replication_of"]["readings"] == ["but_A1"]
    assert rep["turn/q90_right"]["replication_of"]["readings"] == ["but_A1"]  # reading 2 gave it no verdict
    r = rep["walk/v0.4"]
    assert r["block"] == "seeds5042" and r["seeds"] == list(range(5042, 5062)) and r["definition"]["block"] == "seeds5042"
    assert r["replication_of"]["block"] == "seeds3042" and r["replication_of"]["record"] == "T0/75-M-H-10/seeds3042/eval/R2/42@10ms/walk/v0.4.json"
    assert r["replication_of"]["key_sha256"] == read(rec_path(out, "R2/42@10ms", "walk/v0.4"))["key_sha256"]
    # judged on its own 20 seeds: it fails here while the first block's record stays a pass; nothing is pooled
    assert r["counts"]["judged"] is True and r["counts"]["pass_"] is False and r["counts"]["episodes"] == 20
    assert read(rec_path(out, "R2/42@10ms", "walk/v0.4"))["counts"]["pass_"] is True
    assert {p: p.read_bytes() for p in frozen} == frozen
    assert not rec_path(out, "R0@10ms", "walk/v0.4", block="seeds5042").exists()
    assert not rec_path(out, "R2/43@10ms", "walk/v0.4", block="seeds5042").exists()
    assert two["block"] == "seeds5042" and set(two["grid"]) == set(conds)
    assert two["verdicts"] == {} and "no verdict" in two["verdicts_note"] and one["verdicts_note"] is None
    assert all(set(v) == {"R2/42@10ms"} for v in two["grid"].values())
    with pytest.raises(ValueError, match="another block"):
        bar.evaluate(SET, BODY, out, rows, conds, block="seeds3042", passed_of="seeds3042", procs=1, root=root, log=quiet)


@needs_run
def test_the_omega_control_runs_apart_with_stage1s_crouch(tmp_path, monkeypatch):
    root = tmp_path / "sets"
    d = make_t0(root)
    sd = s2io.read_json(d / "set.json")
    s2io.write_json(d / "set.json", dict(sd, crouch_frac=0.05))  # a set that crouches 0.05 H
    out = tmp_path / "out"
    faked(monkeypatch, lambda row, cond, seed, relax: fake_episode(cond, seed))
    bar.evaluate(SET, BODY, out, ["R2/42@10ms"], ["walk/v0.4"], episodes=2, procs=1, root=root, log=quiet)
    bar.evaluate(SET, BODY, out, ["R2/42@10ms"], ["walk/v0.4"], episodes=2, procs=1, root=root, control="omega_s1",
                 log=quiet)
    own = read(rec_path(out, "R2/42@10ms", "walk/v0.4"))
    ctl = read(rec_path(out, "R2/42@10ms", "walk/v0.4", control="omega_s1"))
    assert rec_path(out, "R2/42@10ms", "walk/v0.4", control="omega_s1").is_relative_to(out / "controls" / "omega_s1")
    assert own["crouch_frac"] == own["key"]["crouch_frac"] == 0.05 and "control" not in own
    assert ctl["crouch_frac"] == ctl["key"]["crouch_frac"] == 0.03 and ctl["control"] == "omega at Stage 1's 0.03 H crouch"
    ctl_summary = read(bar.block_dir(out, SET, BODY, "seeds3042", "omega_s1") / "eval" / "summary.json")
    assert ctl_summary["verdicts"] == {} and ctl_summary["verdicts_note"] == "a control gives no verdict"
    assert own["omega"] == bar.omega_of(BODY, d, 0.05) and ctl["omega"] == bar.omega_of(BODY, d, 0.03)
    assert own["omega"] > ctl["omega"] and own["key"]["omega"] == own["omega"] and ctl["key"]["omega"] == ctl["omega"]
    assert bar.key_differences(own["key"], ctl["key"]) and all("omega" in x or "crouch" in x for x in
                                                               bar.key_differences(own["key"], ctl["key"]))
    with pytest.raises(ValueError):
        bar.run_root(out, "other")
    # a set that records no crouch is refused: the track's frequency would be a guess
    s2io.write_json(d / "set.json", {k: v for k, v in sd.items() if k != "crouch_frac"})
    with pytest.raises(bar.BarError, match="crouch_frac"):
        bar.evaluate(SET, BODY, tmp_path / "none", ["R2/42@10ms"], ["walk/v0.4"], episodes=2, procs=1, root=root, log=quiet)


@needs_run
def test_the_ledger_cuts_the_first_unit_that_does_not_fit_and_every_unit_after_it(t0, tmp_path, monkeypatch, capsys):
    b = bar.Budget(tmp_path / "budget.json", 3, cap_core_h=1.0)
    assert b.start("u1", 0.4) and b.spent() == 0.0
    b.add("u1", "rec#s1", 900.0)
    b.add("u1", "rec#s1", 600.0)  # the same item again: the larger value stands, nothing is counted twice
    b.add("u1", "rec#s2", 180.0)
    assert b.spent() == pytest.approx(0.3) and b.unit("u1")["used_core_h"] == pytest.approx(0.3)
    assert not b.start("u2", 0.8)  # 0.3 + 0.8 > 1.0: cut
    assert not b.start("u3", 0.01)  # ... and every unit after it, however cheap
    assert b.start("u1", 5.0)  # a started unit always continues
    doc = read(tmp_path / "budget.json")
    t = doc["tasks"]["3"]
    assert [u["unit"] for u in t["units"]] == ["u1"] and [c["unit"] for c in t["cut"]] == ["u2", "u3"]
    assert t["units"][0]["estimate_core_h"] == 0.4 and t["units"][0]["spent_before_core_h"] == 0.0 and t["units"][0]["started_at"]
    assert t["cut"][0]["spent_before_core_h"] == pytest.approx(0.3) and "exceeds" in t["cut"][0]["reason"]
    assert "an earlier unit" in t["cut"][1]["reason"] and t["cut"][0]["label"] == "cut by the pre-registered order"
    assert bar.Budget(tmp_path / "budget.json", 4).spent() == 0.0 and bar.BUDGET_CORE_H == {3: 16.0, 4: 12.0}
    with pytest.raises(ValueError):
        bar.Budget(tmp_path / "budget.json", 5)
    # through the runner: every episode's wall seconds are counted, per record and session
    out = tmp_path / "out"
    rates = dict(R2=dict(wall_per_sim_s=0.45), R0=dict(wall_per_sim_s=0.09))
    calls = faked(monkeypatch, lambda row, cond, seed, relax: fake_episode(cond, seed, fell=not relax and seed < 3048))
    args = (SET, BODY, out, ["R2/42@10ms", "R0@10ms"], ["walk/v0.4"])
    with pytest.raises(bar.BarError, match="needs the committed rates"):
        bar.evaluate(*args, procs=1, root=t0, task=3, log=quiet)
    doc = bar.evaluate(*args, procs=1, root=t0, task=3, rates=rates, unit="T3.1 T0 @10ms", log=quiet)
    led = read(out / "budget.json")["tasks"]["3"]
    assert led["cap_core_h"] == 16.0 and [u["unit"] for u in led["units"]] == [
        "T3.1 T0 @10ms", "attribution T0/75-M-H-10/seeds3042/eval/R2/42@10ms/walk/v0.4.json"]
    seconds = read(rec_path(out, "R0@10ms", "walk/v0.4"))["seconds"]
    assert led["units"][0]["estimate_core_h"] == pytest.approx(20 * seconds * (0.45 + 0.09) / 3600)
    assert led["units"][0]["used_core_h"] == pytest.approx(40 * 0.25 / 3600) and len(led["units"][0]["items"]) == 2
    assert led["units"][1]["estimate_core_h"] == pytest.approx(20 * seconds * 0.45 / 3600)
    assert led["units"][1]["used_core_h"] == pytest.approx(20 * 0.25 / 3600) and led["cut"] == []
    assert led["units"][1]["spent_before_core_h"] == pytest.approx(40 * 0.25 / 3600)
    assert bar.Budget(out / "budget.json", 3).spent() == pytest.approx(60 * 0.25 / 3600)
    assert doc["sessions"][-1]["task"] == 3 and read(rec_path(out, "R0@10ms", "walk/v0.4"))["sessions"][0]["unit"] == "T3.1 T0 @10ms"
    # a resumed unit counts nothing twice
    bar.evaluate(*args, procs=1, root=t0, task=3, rates=rates, unit="T3.1 T0 @10ms", log=quiet)
    assert bar.Budget(out / "budget.json", 3).spent() == pytest.approx(60 * 0.25 / 3600)
    # a unit that does not fit is refused before any episode (exit 3 from the CLI), and so is the next one
    n = len(calls)
    with pytest.raises(bar.BudgetCut, match="cut by the pre-registered order"):
        bar.evaluate(SET, BODY, out, ["R2/42@4ms"], ["walk/v0.4"], procs=1, root=t0, task=3, rates=rates,
                     unit="T3.4 T0 @4ms", budget_cap_core_h=0.01, log=quiet)
    assert len(calls) == n and not rec_path(out, "R2/42@4ms", "walk/v0.4").exists()
    (out / "COST.md").write_text("# Cost\n\n```json\n" + json.dumps(dict(schema="bar-cost-1", rates=rates)) + "\n```\n")
    code = bar.main(["run", "--set", SET, "--set-root", str(t0), "--body", BODY, "--rows", "R0@6ms", "--conditions",
                     "walk/v0.4", "--procs", "1", "--task", "3", "--unit", "T3.4 T0 @6ms", "--out", str(out)])
    assert code == 3 and "cut by the pre-registered order: T3.4 T0 @6ms" in capsys.readouterr().err and len(calls) == n
    led = read(out / "budget.json")["tasks"]["3"]
    assert [c["unit"] for c in led["cut"]] == ["T3.4 T0 @4ms", "T3.4 T0 @6ms"]
    # attribution is checked record by record: one that does not fit is labelled, and the record keeps its episodes
    out2 = tmp_path / "out2"
    tiny = 20 * seconds * 0.45 / 3600 + 1e-4  # room for the run's estimate, not for a record's attribution
    doc = bar.evaluate(SET, BODY, out2, ["R2/42@10ms"], ["walk/v0.4"], procs=1, root=t0, task=4, rates=rates,
                       unit="T4.1", budget_cap_core_h=tiny, log=quiet)
    rec = read(rec_path(out2, "R2/42@10ms", "walk/v0.4"))
    assert rec["attribution"]["label"] == "not attributed (budget, pre-registered order)" and rec["attribution"]["cut"] is True
    assert rec["attribution"]["reported_as"] == "not attributed: cut by the pre-registered order"
    assert rec["failure"]["falls_label"] == "not attributed (budget, pre-registered order)" and len(rec["episodes"]) == 20
    led = read(out2 / "budget.json")["tasks"]["4"]
    assert [c["unit"] for c in led["cut"]] == ["attribution T0/75-M-H-10/seeds3042/eval/R2/42@10ms/walk/v0.4.json"]
    assert doc["sessions"][-1]["attribution_cut"] == ["R2/42@10ms walk/v0.4"]
    assert bar.read_cost(out / "COST.md")["rates"] == rates and bar.rate_of(rates, "R2/43@6ms") == 0.45
    assert bar.rate_of(rates, "R4/42@10ms") == 0.45  # a row type that was not measured: the largest rate
    with pytest.raises(bar.BarError, match="bar-cost-1"):
        bar.read_cost(rec_path(out, "R0@10ms", "walk/v0.4"))


@needs_run
def test_a_run_on_a_set_of_the_75cm_round_is_a_tasks_unit_or_a_listed_check(t0, tmp_path, monkeypatch):
    ok = ("V0", ["R0@10ms", "R0@0ms"], 1, None, "dev.json", None, None)
    bar._check_rule(*ok)
    for bad, word in ((("V0", ["R0@10ms", "R2/42@10ms"], 1, None, "dev.json", None, None), "not an R0 row"),
                      (("F25F9cl", ["R0@10ms"], 1, None, "dev.json", None, None), "not V0"),
                      (("F25F9", ["R0@10ms"], 1, None, "dev.json", None, None), "not V0"),
                      (("V0", ["R0@10ms"], 2, None, "dev.json", None, None), "2 episodes"),
                      (("V0", ["R0@10ms"], 1, None, "dev.json", None, None, "seeds5042"), "seeds5042"),
                      (("V0", ["R0@10ms"], 1, None, None, None, None), "--dev-episodes"),
                      (("V0", ["R0@10ms"], 1, None, "dev.json", "seeds3042", None), "no check")):
        with pytest.raises(bar.BarError, match=word):
            bar._check_rule(*bad)
    bar._check_rule("F25F9cl", ["R2/42@10ms"], 20, 3, None, None, None)  # a task's unit
    bar._check_rule(SET, ["R2/42@10ms"], 20, None, None, None, None)  # not a set of the round (a fixture)
    bar._check_rule("S1", ["R2/42@10ms"], 20, None, None, None, None)
    with pytest.raises(bar.BarError, match="build check"):
        bar.evaluate("V0", BODY, tmp_path / "x", ["R2/42@10ms"], ["walk/v0.4"], procs=1, root=t0, log=quiet)
    assert not (tmp_path / "x").exists()
    # a check's episodes are listed in the file --dev-episodes names
    faked(monkeypatch, lambda row, cond, seed, relax: fake_episode(cond, seed))
    dev = tmp_path / "dev_episodes.json"
    for _ in (1, 2):
        bar.evaluate(SET, BODY, tmp_path / "out", ["R0@10ms"], ["walk/v0.4", "stop/two_step"], episodes=1, procs=1,
                     root=t0, dev_episodes=dev, log=quiet)
    listed = read(dev)
    assert [(x["set"], x["row"], x["condition"], x["seed"]) for x in listed["episodes"]] == [
        (SET, "R0@10ms", "walk/v0.4", 3042), (SET, "R0@10ms", "stop/two_step", 3042)]
    assert listed["rule"] == bar.RULES["checks"] and listed["episodes"][0]["bar_sha256"] == bar.bar_sources()["sha256"]
    assert listed["episodes"][0]["reason"] == "the bar's build check: smoke, no record"


@needs_run
def test_groups_and_named_ladder_conditions(t0, t0_docs, tmp_path, monkeypatch):
    keys = lambda names: [c.key for c in bar.choose(t0_docs, "seeds3042", names)]  # noqa: E731
    assert keys(None) == GATED == keys(["@gated"])
    assert keys(["@walk"]) == ["walk/v0.4", "walk/v0.4_mu0.54"]
    assert keys(["@rest"]) == [k for k in GATED if not k.startswith("walk/")] and len(keys(["@rest"])) == 14
    assert keys(["@speed-ladder"]) == ["walk/nominal"]  # what T0 has entries for, at mu 0.9
    assert keys(["@ladders"]) == [f"nudge/{d}_x{f:g}" for f in (0.375, 0.5) for d in ("forward", "backward", "left", "right")]
    assert keys(["@walk", "stop/two_step"]) == ["walk/v0.4", "walk/v0.4_mu0.54", "stop/two_step"]
    # a ladder condition named outright exists whether or not the set lists its reference: it gets a "not run" record
    assert keys(["walk/v0.5_mu0.54", "turn/t90_1_left", "stop/one_step"]) == ["walk/v0.5_mu0.54", "turn/t90_1_left",
                                                                             "stop/one_step"]
    every = [c.key for c in bar.conditions(t0_docs, every=True)]
    assert len(every) == 16 + 10 + 6 + 8 + 1 and every[:16] == GATED
    with pytest.raises(ValueError, match="walk/v0.9"):
        keys(["walk/v0.9"])
    with pytest.raises(ValueError, match="@everything"):
        keys(["@everything"])
    calls = faked(monkeypatch, lambda row, cond, seed, relax: fake_episode(cond, seed))
    out = tmp_path / "out"
    doc = bar.evaluate(SET, BODY, out, ["R0@10ms"], ["walk/v0.5", "turn/t90_0.5_right", "stop/one_step", "@walk"],
                       episodes=1, procs=1, root=t0, log=quiet)
    assert [c[1] for c in calls] == ["walk/v0.4", "walk/v0.4_mu0.54"] and len(doc["sessions"][-1]["not_run"]) == 3
    r = read(rec_path(out, "R0@10ms", "walk/v0.5"))
    assert r["status"] == "not run" and r["reason"] == "the set has no entry 'v0.5'" and r["gated"] is False
    assert read(rec_path(out, "R0@10ms", "turn/t90_0.5_right"))["reference"] == "transitions/turn90_0.5_right"
    assert read(rec_path(out, "R0@10ms", "stop/one_step"))["reason"] == "the set has no entry 'stop1_v0.4'"


# ------------------------------------------------------------------------------------------------
# Kill and resume (a real kill of a real run on the S1 view)
# ------------------------------------------------------------------------------------------------
VOLATILE = ("wall_s", "started_at", "ended_at", "sessions", "episode_wall_s", "written_at", "at")


@needs_run
def test_a_killed_run_resumes_to_equal_records(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT), MUJOCO_GL="disable")
    cmd = [sys.executable, "-m", "designlab.stage2.bar", "run", "--set", "S1", "--stage1-run", str(RUN), "--body", BODY,
           "--rows", "R0@4ms", "--conditions", "walk/nominal,walk/brisk", "--episodes", "3", "--procs", "1"]
    killed, whole = tmp_path / "killed", tmp_path / "whole"
    proc = subprocess.Popen(cmd + ["--out", str(killed)], cwd=ROOT, env=env, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    partial = bar.block_dir(killed, "S1", BODY, "seeds3042") / "partial"
    t0_ = time.time()
    while time.time() - t0_ < 120 and proc.poll() is None and len(list(partial.rglob("*.json"))) < 2:
        time.sleep(0.05)
    assert proc.poll() is None, "the run finished before it could be killed"
    proc.send_signal(signal.SIGKILL)
    proc.wait()
    kept = sorted(p.relative_to(partial).as_posix() for p in partial.rglob("*.json"))
    assert len(kept) >= 2 and (killed / ".lock").exists()  # the dead run's lock is still there ...
    records = lambda out: sorted(p for p in bar.block_dir(out, "S1", BODY, "seeds3042").rglob("*.json")  # noqa: E731
                                 if "partial" not in p.parts and p.name != "summary.json")
    n_before = len(records(killed))
    assert n_before < 2
    # ... and is taken over: its holder is gone
    again = subprocess.run(cmd + ["--out", str(killed)], cwd=ROOT, env=env, capture_output=True, text=True)
    assert again.returncode == 0, again.stderr[-2000:]
    assert "taken over" in again.stderr and not (killed / ".lock").exists() and not partial.exists()
    ref = subprocess.run(cmd + ["--out", str(whole)], cwd=ROOT, env=env, capture_output=True, text=True)
    assert ref.returncode == 0, ref.stderr[-2000:]
    a, b = records(killed), records(whole)
    assert [p.relative_to(killed) for p in a] == [p.relative_to(whole) for p in b] and len(a) == 2
    resumed = 0
    for pa, pb in zip(a, b, strict=True):
        ra, rb = read(pa), read(pb)
        assert strip(ra, VOLATILE) == strip(rb, VOLATILE), pa
        assert [e["trace"]["sha256"] for e in ra["episodes"]] == [e["trace"]["sha256"] for e in rb["episodes"]]
        assert sorted(s for x in ra["sessions"] for s in x["seeds_run"]) == [3042, 3043, 3044]
        resumed += len(ra["sessions"]) > 1
        for e in ra["episodes"]:  # every trace is there and is the one its record names
            tr = bar.block_dir(killed, "S1", BODY, "seeds3042") / "eval" / "traces" / e["trace"]["path"]
            assert s2io.sha256_file(tr) == e["trace"]["sha256"]
    assert resumed >= 1  # at least one record was finished from episodes the killed run left
    # a third run finds everything done
    third = subprocess.run(cmd + ["--out", str(killed)], cwd=ROOT, env=env, capture_output=True, text=True)
    assert third.returncode == 0 and "0 episodes run" in third.stdout and "2 records up to date" in third.stdout


# ------------------------------------------------------------------------------------------------
# Dry run, the kinematic player, refcheck, reread
# ------------------------------------------------------------------------------------------------
@needs_run
def test_dry_run_writes_nothing_and_its_text_holds_the_rates(t0, tmp_path, capsys):
    out = tmp_path / "out"
    rates = dict(schema="bar-cost-1", rates=dict(R2=dict(wall_per_sim_s=0.5, row="R2/42@10ms"),
                                                 R0=dict(wall_per_sim_s=0.1, row="R0@10ms")),
                 measured_on=dict(condition="walk/brisk", body=BODY, stage1_run="stage1/runs/2026-11-08", cut_s=8.0,
                                  seeds=[3042, 3043]))
    doc = bar.dry_run(SET, BODY, out, ["R2/42@10ms", "R0@10ms"], ["@gated"], root=t0, rates=rates, log=quiet)
    assert not out.exists() and len(doc["episodes"]) == 32 and doc["totals"]["episodes"] == 640
    walk = next(x for x in doc["episodes"] if x["condition"] == "walk/v0.4" and x["row"] == "R2/42@10ms")
    assert walk["status"] == "to run" and walk["core_h"] == pytest.approx(20 * walk["seconds"] * 0.5 / 3600)
    assert doc["totals"]["core_h"] == pytest.approx(sum(x["core_h"] for x in doc["episodes"]))
    (tmp_path / "COST.md").write_text(doc["text"])
    assert bar.read_cost(tmp_path / "COST.md") == rates and "never on a reference of a Stage 2 set" in doc["text"]
    # the CLI prints the document and nothing else (bar dry-run ... > COST.md)
    cost = tmp_path / "COST.md"
    assert bar.main(["dry-run", "--set", SET, "--set-root", str(t0), "--body", BODY, "--rows", "R0@10ms", "--conditions",
                     "walk/v0.4", "--cost", str(cost), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert printed.startswith("# Cost of the bar's episodes") and "```json" in printed and not out.exists()
    # the rates are measured on Stage 1's own committed condition (two short episodes per row type)
    m = bar.measure_rates(BODY, ["R0@10ms", "R0@0ms"], RUN, seconds=2.0)
    assert set(m["rates"]) == {"R0"} and m["rates"]["R0"]["row"] == "R0@10ms" and m["rates"]["R0"]["simulated_s"] == pytest.approx(4.0, abs=0.01)
    assert 0.0 < m["rates"]["R0"]["wall_per_sim_s"] < 5.0 and m["measured_on"]["set"] == "S1"
    assert m["measured_on"]["condition"] == "walk/brisk" and m["schema"] == "bar-cost-1"


@needs_run
def test_the_kinematic_player_is_stage1s_on_a_straight_walk_and_follows_any_track(t0, t0_docs):
    from designlab.stage1.refs import Reference

    p = plant.build(BODY, "hard", settle=False)
    d = setio.set_dir(SET, BODY, t0)
    ref = Reference.load(d / "refs" / "nominal.npz")
    stock, track = panels.reference_recorder(p, ref, 8.0)
    track.run_dir = d
    mine = bar.reference_recorder(p, track, 8.0)
    assert mine.rows == stock.rows and mine.steps == stock.steps
    cols = [metrics.C[c] for c in ("t", "t_ref", "com_x", "com_y", "foot0_x", "foot0_y", "foot1_x", "foot1_y", "low0",
                                   "low1", "roll", "pitch", "yaw", "ref_swing", "ref_u", "sole0_x", "sole1_y", "root_x")]
    assert np.allclose(mine.U[:, cols], stock.U[:, cols], atol=1e-9, rtol=0) and np.allclose(mine.Q, stock.Q, atol=1e-12)
    assert np.array_equal(mine.contact(), stock.contact())  # contact from the reference's support flags
    T, H = ref.period, float(p.meta["H_m"])
    a, b = (metrics.window(r, T, T + 6.0, 0.0, H) for r in (mine, stock))
    for k in ("cadence_steps_s", "duty", "toe_clearance_m", "stride_H", "speed_m_s"):
        assert a[k] == pytest.approx(b[k], rel=1e-9, abs=1e-9), k
    # a whole item of the bar: the lead-in, the walk; the played CoM advances at the walk's speed times the clock
    pl = planner_of(t0)
    omega = control.omega0(p, ref.q_home)
    item = bar.reference_item(p, d, cond_of(t0_docs, "walk/v0.4"), pl, omega)
    brisk = t0_docs["refs.json"]["references"]["v0.4"]["speed_m_s"]
    assert item["along_commanded_heading_m"] == pytest.approx(10.0 * 1.05 * brisk, abs=6e-3) and item["clock_rate"] == 1.05
    assert item["ok"] == dict(along=False, line=True, toe_clearance_own_swing=True) and item["pass_"] is False
    assert item["toe"]["own_swings"] == item["toe"]["ref_swings"] and item["toe"]["interrupted_swings"] == 0
    assert item["toe"]["air_gaps"] == 0 and item["lateral_from_line_max_m"] < 0.1
    nominal = bar.reference_item(p, d, cond_of(t0_docs, "walk/nominal"), pl, omega)
    assert nominal["pass_"] is True and nominal["along_commanded_heading_m"] == pytest.approx(
        10.0 * 1.05 * t0_docs["refs.json"]["references"]["nominal"]["speed_m_s"], abs=6e-3)  # (the CoM's own surge)
    # segments are joined in one world frame: no jump of the root across a join
    res = pl.resolve(cond_of(t0_docs, "walk/v0.4"))
    rec = bar.reference_recorder(p, setio.build_track(p, d, res["plan"], omega, pl.tr_doc), res["seconds"], d, rate=1.05)
    step = np.linalg.norm(np.diff(rec.U[:, [metrics.C["root_x"], metrics.C["root_y"]]], axis=0), axis=1)
    assert step.max() < 0.01 and rec.rows == int(round(res["seconds"] / 0.002)) // 5 + (1 if int(round(res["seconds"] / 0.002)) % 5 else 0)
    with pytest.raises(ValueError, match="run_dir"):
        bar.reference_recorder(p, control.build_track(p, d, res["plan"], omega), 1.0)


@needs_run
def test_refcheck_reads_every_reference_with_the_bars_readings(t0, tmp_path):
    lines = []
    doc = bar.refcheck(SET, BODY, t0, tmp_path / "out", log=lines.append)
    assert read(tmp_path / "out" / SET / BODY / "refcheck.json")["items"].keys() == doc["items"].keys()
    assert list(doc["items"]) == ["walk/v0.4", "turn/q90_left", "turn/q90_right", "stop/two_step", "walk/nominal"]
    assert doc["set_freeze"]["index"] == 0 and doc["omega"] == bar.omega_of(BODY, setio.set_dir(SET, BODY, t0), 0.03)
    assert doc["items"]["walk/nominal"]["pass_"] is True and doc["items"]["walk/v0.4"]["ok"]["along"] is False
    # T0's "moves" are Stage 1's files under other names: the readings are taken all the same
    stop = doc["items"]["stop/two_step"]
    assert stop["stop"]["reached_command"] and isinstance(stop["stop"]["touchdowns_after_command"], int)
    turn = doc["items"]["turn/q90_left"]
    assert turn["turn"]["at_s"] == 2.0 and turn["file"]["at_s"] == 2.0 and turn["toe"]["air_gap_max_s"] > 0
    assert doc["pass_"] is False and len(lines) == 5 and "walk/nominal: pass" in lines[4]
    assert bar.main(["refcheck", "--set", SET, "--set-root", str(t0), "--body", BODY, "--out", str(tmp_path / "o2")]) == 1


@needs_run
def test_reread_checks_the_stock_trace_before_it_reads_the_own_swing_clearance(tmp_path):
    lines = []
    doc = bar.reread(RUN, BODY, ["R0"], ["walk/brisk"], tmp_path / "out", episodes=1, log=lines.append)
    assert doc == read(tmp_path / "out" / BODY / "reread.json")
    assert (doc["traces_equal"], doc["traces"], doc["pass_"], doc["complete"]) == (1, 1, True, True)
    (e,) = doc["episodes"]
    committed = read(RUN / BODY / "eval" / "R0" / "walk" / "brisk.json")["episodes"][0]
    assert e["trace_sha256"] == e["committed_trace_sha256"] == committed["trace"]["sha256"] and e["trace_equal"]
    assert e["toe_clearance_reference_clock_m"] == committed["measures"]["window"]["toe_clearance_m"]
    assert e["difference_m"] == pytest.approx(e["toe_clearance_own_swing_m"] - e["toe_clearance_reference_clock_m"])
    assert e["interrupted_swings"] == 0 and e["own_swings"] == e["ref_swings"] and e["convention"]["holds"] is True
    assert e["window_s"] == [committed["measures"]["window"]["t_start_s"], committed["measures"]["window"]["t_end_s"]]
    assert "trace equal" in lines[0] and "own swing" in lines[0]
    # a record the trace does not match: the own-swing reading is not taken, and the check fails
    fake = tmp_path / "records" / BODY / "eval" / "R0" / "walk"
    fake.mkdir(parents=True)
    s2io.write_json(fake / "brisk.json", dict(episodes=[dict(seed=3042, trace=dict(sha256="0" * 64))]))
    doc = bar.reread(RUN, BODY, ["R0"], ["walk/brisk"], tmp_path / "out2", records=tmp_path / "records", episodes=1,
                     log=quiet)
    (e,) = doc["episodes"]
    assert doc["pass_"] is False and e["trace_equal"] is False and "toe_clearance_own_swing_m" not in e
    assert e["not_read"] == "the stock trace differs from the committed record's"
    assert s2io.read_json(fake / "brisk.json")["episodes"][0]["trace"]["sha256"] == "0" * 64  # nothing written there


# ------------------------------------------------------------------------------------------------
# The chain
# ------------------------------------------------------------------------------------------------
CHAIN = ROOT / "stage2" / "runs" / "2026-11-15-75cm" / "run_75cm.sh"


def test_the_chain_lists_the_units_in_the_pre_registered_order(tmp_path):
    assert subprocess.run(["bash", "-n", str(CHAIN)]).returncode == 0
    r = subprocess.run(["bash", str(CHAIN), "list"], capture_output=True, text=True, env=dict(os.environ, PY="python"))
    assert r.returncode == 0, r.stderr
    steps = [line.split(" [to run]: ")[0] for line in r.stdout.splitlines()]
    cmds = dict(line.split(" [to run]: ") for line in r.stdout.splitlines())
    assert len(steps) == len(set(steps)) and steps[:3] == ["refcheck-V0", "refcheck-F25F9", "refcheck-F25F9cl"]
    assert steps[-1] == "report" and "designlab.stage2.barreport" in cmds["report"]
    t3 = [s for s in steps if s.startswith("T3.") and not s.startswith("T3.10")]
    assert t3[:13] == ["T3.1-F25F9cl-10ms", "T3.1-F25F9-10ms", "T3.1-V0-10ms", "T3.2-F25F9cl-10ms-seeds5042",
                       "T3.3-F25F9cl-0ms", "T3.3-F25F9-0ms", "T3.3-V0-0ms", "T3.3-F25F9cl-omega_s1-10ms",
                       "T3.3-F25F9cl-omega_s1-0ms", "T3.4-F25F9cl-4ms", "T3.4-F25F9-4ms", "T3.4-V0-4ms", "T3.4-F25F9cl-6ms"]
    tiers = [s.split("-")[0] for s in steps if s.startswith("T")]
    assert tiers == sorted(tiers, key=lambda x: (int(x[1]), int(x.split(".")[1])))  # tier order, task 3 then task 4
    assert [s for s in steps if s.startswith("T3.5")] == [f"T3.5-{s}-{d}ms" for d in (2, 8) for s in ("F25F9cl", "F25F9", "V0")]
    assert [s for s in steps if s.startswith("T3.6")] == [f"T3.6-{s}-{d}ms" for d in (10, 0) for s in ("F25F9cl", "V0")]
    t4 = [s for s in steps if s.startswith("T4.")]
    assert t4 == ["T4.1-F25F9cl-10ms", "T4.1-V0-10ms", "T4.2-F25F9cl-10ms-seeds5042", "T4.3-F25F9cl-0ms", "T4.3-V0-0ms",
                  "T4.4-F25F9cl-4ms", "T4.4-V0-4ms", "T4.5-F25F9cl-6ms", "T4.5-V0-6ms", "T4.6-F25F9cl-10ms", "T4.6-V0-10ms",
                  "T4.7-F25F9cl-0ms-seeds5042", "T4.7-F25F9cl-4ms-seeds5042", "T4.7-F25F9cl-6ms-seeds5042"]
    # every unit names its task, its unit and the committed cost file; each tier's rows, conditions and attribution
    for s in steps:
        if s.startswith("T"):
            task = s[1]
            assert f"--task {task} " in cmds[s] and "--cost stage2/runs/2026-11-15-75cm/COST.md" in cmds[s]
            assert "--out stage2/runs/2026-11-15-75cm" in cmds[s] and "--body 75-M-H-10" in cmds[s]
            unit = cmds[s].split("--unit ")[1].split()[0]
            assert unit == s if not s.startswith(f"T{task}.10") else unit.startswith(f"T{task}.")
    c = cmds["T3.1-F25F9cl-10ms"]
    assert "--rows R2/42@10ms,R2/43@10ms,R0@10ms --conditions @walk" in c and c.endswith("--attribute judged")
    assert "--block seeds5042 --passed-of seeds3042" in cmds["T3.2-F25F9cl-10ms-seeds5042"]
    assert "R0@" not in cmds["T3.2-F25F9cl-10ms-seeds5042"]  # an R0 row is not replicated
    assert "--control omega_s1" in cmds["T3.3-F25F9cl-omega_s1-0ms"] and "--rows R2/42@0ms --conditions walk/v0.4 " in cmds["T3.3-F25F9cl-omega_s1-0ms"]
    assert "--conditions @speed-ladder" in cmds["T3.6-V0-0ms"] and "R0@" not in cmds["T3.6-V0-0ms"]
    assert "--set F25F9 " in cmds["T3.7-F25F9-0ms"] and "--conditions walk/v0.5 " in cmds["T3.7-F25F9-0ms"]
    assert "walk/v0.45_mu0.54,walk/v0.5_mu0.54" in cmds["T3.7-F25F9cl-0ms"]
    assert all(cmds[s].endswith("--attribute none") for s in t3 if not s.startswith("T3.1-"))
    assert "--conditions @rest" in cmds["T4.1-V0-10ms"] and cmds["T4.1-V0-10ms"].endswith("--attribute judged")
    assert "--conditions @ladders" in cmds["T4.6-F25F9cl-10ms"] and "R0@" not in cmds["T4.6-F25F9cl-10ms"]
    assert all("--set F25F9 " not in cmds[s] for s in t4)  # F25F9 is a control for the walk only
    assert all(cmds[s].endswith("--attribute none") for s in t4 if not s.startswith("T4.1-"))
    # tier 10: the remaining records' attribution, in the tiers' order, under their own units; no control, no tier 1
    t10 = [s for s in steps if s.startswith("T3.10")]
    assert all(cmds[s].endswith("--attribute all") for s in t10) and not any("omega" in s for s in t10)
    assert [cmds[s].split("--unit ")[1].split()[0] for s in t10] == [s for s in t3 if not s.startswith("T3.1-")
                                                                  and "omega" not in s]
    text = CHAIN.read_text()
    assert "pgrep -f 'run_75cm.sh all'" in text and "done/ALL" in text and "tools/lowprio.sh" in text
    # with no argument it runs nothing
    assert subprocess.run(["bash", str(CHAIN)], capture_output=True).returncode == 2


def test_a_step_of_the_chain_is_skipped_once_done_and_a_cut_goes_on(tmp_path):
    """The chain's ``step`` on a copy of the script with stub commands: markers, exit codes in the logs, a cut."""
    root = tmp_path / "study"
    run = root / "stage2" / "runs" / "2026-11-15-75cm"
    run.mkdir(parents=True)
    (root / "tools").mkdir()
    (root / "tools" / "lowprio.sh").write_text('#!/bin/bash\nexec "$@"\n')
    text = CHAIN.read_text()
    head = text.split("r2() {")[0]
    (run / "run_75cm.sh").write_text(head + 'step a sh -c "echo A >> ran; exit 0"\nstep b sh -c "echo B >> ran; exit 3"\n'
                                     'step c sh -c "echo C >> ran; exit 0"\nstep d sh -c "echo D >> ran; exit 7"\n'
                                     'step e sh -c "echo E >> ran"\n')
    (run / "done").mkdir()
    (run / "logs").mkdir()
    for _ in (1, 2):
        r = subprocess.run(["bash", str(run / "run_75cm.sh"), "all"], capture_output=True, text=True)
        assert r.returncode == 7
    assert (root / "ran").read_text().split() == ["A", "B", "C", "D", "D"]  # done and cut steps are not run again
    assert sorted(p.name for p in (run / "done").iterdir()) == ["a", "b.cut", "c"]
    assert "Z exit 0\n" in (run / "logs" / "a.log").read_text() and "Z exit 3\n" in (run / "logs" / "b.log").read_text()
    assert (run / "logs" / "d.log").read_text().count("Z exit 7\n") == 2 and "FAILED d (exit 7)" in (run / "logs" / "run_75cm.log").read_text()


# ------------------------------------------------------------------------------------------------
# Slow: one full episode per judge on a built set under R0@10ms
# ------------------------------------------------------------------------------------------------
SLOW_SET = os.environ.get("STAGE2_BAR_SET", "V0")
SLOW_ROOT = Path(os.environ.get("STAGE2_BAR_SET_ROOT", str(setio.ROOT)))
SLOW_DEV = os.environ.get("STAGE2_DEV_EPISODES")


@pytest.mark.skipif(not SLOW, reason="slow: set STAGE2_SLOW=1")
@needs_run
def test_slow_one_episode_per_judge_carries_every_reading(tmp_path):
    d = setio.set_dir(SLOW_SET, BODY, SLOW_ROOT)
    if not (d / "set.json").exists() or setio.frozen(s2io.read_json(d / "set.json")) is None:
        pytest.skip(f"no frozen set {SLOW_SET} under {SLOW_ROOT}")
    if SLOW_SET in bar.PREREGISTERED_SETS and not SLOW_DEV:
        pytest.skip(f"episodes on {SLOW_SET} are build checks: name the file that lists them in STAGE2_DEV_EPISODES")
    conds = ["walk/v0.4", "turn/q90_left", "turn/q90_right", "nudge/left", "stop/two_step"]
    out = tmp_path / "out"
    bar.evaluate(SLOW_SET, BODY, out, ["R0@10ms"], conds, episodes=1, procs=1, root=SLOW_ROOT, dev_episodes=SLOW_DEV,
                 log=quiet)
    for key in conds:
        rec = read(rec_path(out, "R0@10ms", key, set_name=SLOW_SET))
        if rec["status"] == "not run":
            assert rec["reason"]
            continue
        (e,) = rec["episodes"]
        assert not e.get("error"), e.get("error")
        assert EPISODE_KEYS <= set(e) and set(e["success"]) == set(rec["terms"])
        assert set(e["success_but_A1"]) == set(rec["terms_but_A1"]) and e["omega"] == rec["omega"]
        assert e["measures"]["bar"]["drives"]["episode"]["ratios"] and e["measures"]["convention"]["holds"] is not False
        b = e["measures"]["bar"]
        if e["fell"]:
            assert e["failed_terms"] == ["survived"] and (key.startswith("walk") or e["fell_before_command"] is not None)
            continue
        assert e["window_reached"] is True
        if key.startswith("walk"):
            assert b["toe"]["toe_clearance_own_swing_m"] is not None and b["along_target_m"] == 4.0
            assert e["measures"]["stage1"]["terms"] and b["drives"]["window"]["ratios"]
        elif key.startswith("turn"):
            assert b["heading_error_at_deg"] is not None and b["heading_error_hold_max_deg"] is not None
            assert b["toe"]["toe_clearance_own_swing_m"] is not None and e["measures"]["legs"]["poses"] > 0
            assert b["at_s"] == 2.0 and e["command_s"] is not None and b["drives"]["window"]["ratios"]
        elif key.startswith("nudge"):
            assert e["measures"]["recovery"]["after_s"] is not None or not e["measures"]["recovery"]["recovered"]
            assert e["measures"]["push"]["frac"] == 0.25 and e["command_s"] == e["measures"]["push"]["onset_s"]
        else:
            assert isinstance(b["touchdowns_after_command"], int) and b["decision_18"]["reached"] is True
            assert b["stop_acceleration_1hz_m_s2"] is not None and b["rest_s"] is not None
        z = np.load(bar.block_dir(out, SLOW_SET, BODY, "seeds3042") / "eval" / "traces" / e["trace"]["path"])
        assert {"contact", "low100", "t100", "ref_swing100", "swings", "command_s"} <= set(z.files)


@pytest.mark.skipif(not SLOW, reason="slow: set STAGE2_SLOW=1")
@needs_run
def test_slow_every_reference_passes_its_own_bar(tmp_path):
    d = setio.set_dir(SLOW_SET, BODY, SLOW_ROOT)
    if not (d / "set.json").exists():
        pytest.skip(f"no set {SLOW_SET} under {SLOW_ROOT}")
    doc = bar.refcheck(SLOW_SET, BODY, SLOW_ROOT, tmp_path / "out", log=quiet)
    done = {k: v for k, v in doc["items"].items() if v["status"] == "done"}
    assert done
    for key, item in done.items():
        if key.startswith("stop/two_step"):
            assert item["stop"]["touchdowns_after_command"] == 2 and item["ok"]["stopped_by_end"]
        if key.startswith("turn/"):
            assert item["ok"]["heading"] and item["file"]["heading_at_bar_deg"] >= 80.0
        if SLOW_SET in bar.PREREGISTERED_SETS:
            assert item["pass_"], (key, item["ok"])
