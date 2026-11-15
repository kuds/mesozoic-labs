"""Tests for the bar's runner (designlab/stage2/bar.py): conditions and keys, the plan table, delayed rows, the judge's
plumbing, records, resume, the no-overwrite rule, workers, attribution, replication and the budget ledger.

Run from hardware/trex_form_study (the fit's venv, one BLAS thread):
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/test_stage2_bar.py -q

Fast tests use (a) Stage 1's own run directory through ``setio.stage1_view`` (the pseudo-set ``S1``: stock conditions,
R2 rows and the delay wrapper) and (b) a synthetic set ``T0`` in pytest's temporary directory, made of Stage 1's own
arrays of 75-M-H-10 under the bar's names: the brisk walk as ``v0.4``, ``nominal_to_brisk`` as ``nominal_to_v0.4``,
Stage 1's stop and turn entries *declared* causal as ``stop2_v0.4`` and ``turn90_pi4_<dir>``.  ``T0`` is a fixture for
plans, keys, records and the runner's plumbing: its walk item is Stage 1's brisk walk exactly, its "moves" join
nothing and say nothing about a stop or a turn.  Only R0 rows run on it.  Both need the Stage 1 arrays and the
round-two model files (git-ignored; ``python tools/restore_inputs.py``) and skip without them.

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
            "air_gap_longest_s", "air_gap_max_s", "ok", "limit_m"} <= set(toe) and "swings" not in toe
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
    assert {"contact", "steps", "dt", "weight", "t100", "low100", "ref_swing100", "swings", "command_s"} <= set(z.files)
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
                                             z["ref_swing100"], *toe["window_s"], toe["air_gap_max_s"])
    assert again["toe_clearance_own_swing_m"] == pytest.approx(toe["toe_clearance_own_swing_m"], abs=1e-8)
    assert (again["own_swings"], again["ref_swings"], again["interrupted_swings"], again["air_gaps"]) == (
        toe["own_swings"], toe["ref_swings"], toe["interrupted_swings"], toe["air_gaps"])
    table = z["swings"]
    assert table.shape == (len(again["swings"]), 5) and list(z["swing_classes"]) == list(readings.SWING_CLASSES)
    read_ = table[table[:, 4] == 0]
    assert len(read_) == sum(toe["own_swings"]) and np.nanmin(read_[:, 3]) == pytest.approx(
        toe["toe_clearance_own_swing_m"], abs=1e-12)


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
    monkeypatch.setattr(bar, "conditions", lambda docs, block="seeds3042": [cut])
    out = tmp_path / "out"
    doc = bar.evaluate(SET, BODY, out, ["R0@0ms"], None, episodes=1, procs=1, root=t0, log=quiet)
    rec = read(rec_path(out, "R0@0ms", "walk/v0.4"))
    (e,) = rec["episodes"]
    assert "Unreadable" in e["error"] and e["fell"] and e["failed_terms"] == list(bar.WALK_TERMS)
    assert rec["counts"]["errors"] == 1 and rec["counts"]["falls"] == 0 and rec["counts"]["pass_"] is None
    assert rec["counts_but_A1"]["errors"] == 1 and rec["falls"] == dict(before_command=0, after_command=0, no_command=0)
    assert doc["grid"]["walk/v0.4"]["R0@0ms"]["counts"]["errors"] == 1
