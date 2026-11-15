"""Tests for the bar's runner (designlab/stage2/bar.py): conditions and keys, the plan table, delayed rows, the judge's
plumbing, records, resume, the no-overwrite rule, workers, attribution, replication and the budget ledger.

Run from hardware/trex_form_study (the fit's venv, one BLAS thread):
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/test_stage2_bar.py -q

Fast tests use (a) Stage 1's own run directory through ``setio.stage1_view`` (the pseudo-set ``S1``: stock conditions,
R2 rows and the delay wrapper) and (b) a synthetic set ``T0`` in pytest's temporary directory, made of Stage 1's own
arrays of 75-M-H-10 under the bar's names: the brisk walk as ``v0.4``, ``nominal_to_brisk`` as ``nominal_to_v0.4``,
Stage 1's stop and turn entries *declared* causal as ``stop2_v0.4`` and ``turn90_pi4_<dir>``.  ``T0`` is a fixture for
plans, keys, records and the runner's plumbing: its walk item is Stage 1's brisk walk exactly (``set.json`` declares
it a stand-in for the 0.40 m/s rung), its "moves" join nothing and say nothing about a stop or a turn.  Only R0 rows
run episodes on it (the runner refuses another row on a scratch or fixture set: STAGE2_75_PLAN.md, amendment 5 of
15 Nov); the tests of the runner's rules that name R2 rows there use synthetic episode records (``fake_episode``,
``bar.SYNTHETIC_EPISODES``) and simulate nothing.  Both need the Stage 1 arrays and the round-two model files
(git-ignored; ``python tools/restore_inputs.py``) and skip without them.

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


def make_t0(root: Path, drop=(), unavailable=(), label=None, extra_entry=False, freeze=True, crouch=0.03,
            stand_in=True, lead_in=None, more_refs=(), more_moves=()) -> Path:
    """The synthetic set ``T0`` under ``root`` (module docstring); ``drop`` leaves entries out, ``unavailable``
    records them ``available: false``, ``label`` sets the walk entry's label, ``extra_entry`` adds another entry,
    ``crouch`` is the crouch ``set.json`` and every walk entry record, ``stand_in`` declares that its walk stands in
    for the 0.40 m/s rung (it is Stage 1's brisk walk, 0.351 m/s).  ``lead_in`` is ``set.json``'s (None: no such
    key); ``more_refs`` and ``more_moves`` add (bar name, Stage 1 name) files."""
    src = RUN / BODY
    d = setio.set_dir(SET, BODY, root)
    refs1, trs1 = s2io.read_json(src / "refs.json"), s2io.read_json(src / "transitions.json")
    refs, trs, manifest = {}, {}, {}
    all_refs, all_moves = T0_REFS + tuple(more_refs), T0_MOVES + tuple(more_moves)
    for name, old in all_refs:
        if name in drop:
            continue
        if name in unavailable:
            refs[name] = dict(name=name, kind="walk", available=False, numbers=dict(mid_swing_min_frac_H=0.0382),
                              reason="dropped from the ladder: a fixture's reason", crouch_frac=crouch)
            continue
        e = copy.deepcopy(refs1["references"][old])
        e.update(name=name, file=f"refs/{name}.npz", set=SET, label=label if name == W else None, strides=1,
                 window_s=e["period_s"], crouch_frac=crouch, margin_rule="a fixture's label")
        refs[name] = e
    for name, old in all_moves:
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
            old = dict(all_refs + all_moves)[name]
            (d / sub).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / sub / f"{old}.npz", d / e["file"])
            manifest[name] = dict(kind=kind, sha256=e["sha256"], entry_sha256=setio.entry_sha256(kind, e))
    s2io.write_json(d / "refs.json", dict(body=BODY, set=SET, schema=1, references=refs))
    s2io.write_json(d / "transitions.json", dict(body=BODY, set=SET, schema=1, transitions=trs))
    freezes = [dict(index=0, at="fixture", commit=None, manifest=manifest,
                    manifest_sha256=s2io.sha256_json(manifest))] if freeze else []
    s2io.write_json(d / "set.json", dict(set=SET, body=BODY, role="fixture", crouch_frac=crouch,
                                         stand_in_walks=bool(stand_in),
                                         rules=dict(fixture="Stage 1's arrays under the bar's names"),
                                         not_available={}, freezes=freezes,
                                         **({} if lead_in is None else dict(lead_in=lead_in))))
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
    returns the list of calls.  Nothing is simulated, so the rule that only R0 rows run on a fixture set is set
    aside (``bar.SYNTHETIC_EPISODES``)."""
    calls = []

    def run_episode(row, ctx, cond, seed, relax=(), trace_root=None, **kw):
        calls.append((row.name, cond.key, int(seed), tuple(relax)))
        return plan(row, cond, int(seed), tuple(relax))

    monkeypatch.setattr(bar, "run_episode", run_episode)
    monkeypatch.setattr(bar, "SYNTHETIC_EPISODES", True)
    return calls


def stock_cut(monkeypatch, seconds: float = 1.0, keys=("walk/brisk",)):
    """Stage 1's conditions ``keys`` cut to ``seconds`` (a smoke check) as the pseudo-set S1's conditions."""
    conds = [dataclasses.replace(c, seconds=float(seconds)) for c in panels.conditions() if c.key in keys]
    monkeypatch.setattr(bar, "stock_conditions", lambda: conds)
    monkeypatch.setattr(bar, "stock_default", lambda: conds)
    return conds


def read(path) -> dict:
    return json.loads(Path(path).read_text())


def read_bytes_key(data: bytes) -> str:
    return json.loads(data)["key_sha256"]


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
    # the delay's rule is the bar's own: the row's k for every episode, not the speed fit's random draw
    assert d["servo_delay_rule"] == bar.RULES["delay"] != speedfit.LATENCY_RULE
    assert "the same for every episode" in d["servo_delay_rule"] and "is not used" in d["servo_delay_rule"]
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
def test_a_set_that_starts_straight_into_its_bar_walk(t0, t0_docs, tmp_path, monkeypatch):
    """``RULES["lead_in"]`` (STAGE2_75_PLAN.md A8 as amended, DECISION_LOG.md P34): a set whose ``set.json`` says
    ``lead_in: "start_v0.4"`` begins every plan with that start in place of start, one stride of nominal and the
    change; the strides of the 0.40 m/s walk before a command, the judged windows and the planned seconds' rules are
    the other sets'; a rung is reached through its neighbour's change; a condition on the nominal walk is not run;
    its records carry ``lead_in`` and the rule.  A set with ``lead_in: "start"``, or with no such key, is planned
    and recorded exactly as before: no key is added to its records."""
    alt = setio.start_name(W)
    assert alt == "start_v0.4" and bar.LEAD_INS == ("start", alt)
    root = tmp_path / "sets"
    d = make_t0(root, unavailable=("nominal", "start", setio.change_name("nominal", W)), lead_in=alt,
                more_refs=(("v0.45", "fr05"),), more_moves=((alt, "start"), ("v0.4_to_v0.45", "brisk_to_fr05")))
    docs = setio.load(d)
    pl, ref = planner_of(root), planner_of(t0)
    assert (pl.lead, pl.home_walk, ref.lead, ref.home_walk) == (alt, W, "start", "nominal")
    change = ("transition", setio.change_name("nominal", W))
    for key in ("walk/v0.4", "walk/v0.4_mu0.54", "turn/q90_left", "turn/q90_right", "stop/two_step",
                "nudge/forward", "nudge/backward_mu0.54"):
        a, b = pl.resolve(cond_of(docs, key)), ref.resolve(cond_of(t0_docs, key))
        k = b["plan"].index(change) + 1
        assert b["plan"][:k] == [("transition", "start"), ("ref", "nominal", 1), change]
        # the same items after the lead-in (the count of a walk's last strides and a stop's hold follow the lead-in's
        # own length by the same rules)
        assert a["plan"][0] == ("transition", alt)
        assert [x[:1 if x[0] == "hold" else 2] for x in a["plan"][1:]] == [x[:1 if x[0] == "hold" else 2] for x in b["plan"][k:]]
        assert a["files"] == sorted({f"refs/{W}.npz"} | {f"{'refs' if x[0] == 'ref' else 'transitions'}/{x[1]}.npz"
                                                        for x in a["plan"] if x[0] != "hold"})
        assert "refs/nominal.npz" not in a["files"] and f"transitions/{alt}.npz" in a["files"]
        assert a["window"] == b["window"] and a["target_speed_m_s"] == b["target_speed_m_s"] == 0.4
        assert (a["at_s"], a["period_s"]) == (b["at_s"], b["period_s"])
        if key.startswith(("turn", "stop")):  # the same strides of the 0.40 m/s walk before the command
            assert a["plan"][1] == b["plan"][k] and a["plan"][1][2] in (2.0, 2.5)
            assert a["command_ref_s"] - a["steady_ref_s"] == pytest.approx(b["command_ref_s"] - b["steady_ref_s"], abs=1e-9)
            assert a["command"]["phase"] == b["command"]["phase"] and a["command"]["convention"] == b["command"]["convention"]
            # the planned seconds follow the same rule from the command's own time
            assert (a["seconds"] - a["command_ref_s"] / (bar.SLOW * bar.RATE) if key.startswith("turn") else
                    a["seconds"] - a["rest_ref_s"] / (bar.SLOW * bar.RATE)) == pytest.approx(
                b["seconds"] - (b["command_ref_s"] if key.startswith("turn") else b["rest_ref_s"]) / (bar.SLOW * bar.RATE),
                abs=1e-9)
        if key.startswith("nudge"):
            assert a["push"] == dict(b["push"], steady_ref_s=a["steady_ref_s"])
            assert a["seconds"] - a["steady_ref_s"] / (bar.SLOW * bar.RATE) == pytest.approx(
                b["seconds"] - b["steady_ref_s"] / (bar.SLOW * bar.RATE), abs=1e-9)
    # the walk item: Stage 1's rule on the start's own steady point (as walk/nominal after start)
    a = pl.resolve(cond_of(docs, "walk/v0.4"))
    assert a["steady_ref_s"] == pl.phase_time(a["plan"], 0, 1.0) and len(a["plan"]) == 2
    P, t_walk = pl.window_s(W), pl.segment_times(a["plan"])[1][0]
    need = (a["steady_ref_s"] + panels.WINDOW_MARGIN_STRIDES * P) / bar.SLOW + tune.WALK_WINDOW_S * bar.RATE
    assert a["plan"][1] == ("ref", W, max(1, math.ceil((need - t_walk) / P)))
    # a ladder rung: from 0.40 m/s through its neighbour's change, one stride of the rung on the way
    rung = pl.resolve(cond_of(docs, "walk/v0.45"))
    assert rung["plan"][:3] == [("transition", alt), ("ref", W, 1), ("transition", "v0.4_to_v0.45")]
    assert rung["plan"][3][:2] == ("ref", "v0.45") and len(rung["plan"]) == 4
    # what the set does not hold is not run, with the entry's own reason
    every = {c.key: c for c in bar.conditions(docs, every=True)}
    why = pl.missing(every["walk/nominal"])
    assert why["reference"] == "refs/nominal" and why["reason"] == "dropped from the ladder: a fixture's reason"
    assert why["numbers"] == dict(mid_swing_min_frac_H=0.0382)
    assert pl.missing(every["walk/v0.5"])["reference"] == "refs/v0.5"
    assert pl.lead_in("nominal") is None and pl.lead_in("v0.5") is None and pl.lead_in("v0.35") is None
    assert bar.BarPlanner.step_change("v0.5") == ("v0.45", "v0.45_to_v0.5")
    assert bar.BarPlanner.step_change("v0.3") == ("v0.35", "v0.35_to_v0.3") and bar.BarPlanner.step_change(W) is None
    # without its start the set runs nothing: every condition names that reference
    gone = make_t0(tmp_path / "gone", unavailable=("nominal", "start", alt), lead_in=alt, more_moves=((alt, "start"),))
    g = planner_of(tmp_path / "gone")
    assert all(g.missing(c)["reference"] == f"transitions/{alt}" for c in bar.conditions(setio.load(gone))
               if c.walk == W)
    # the home pose is the body's: read from the 0.40 m/s walk's file it gives the same LIPM frequency
    assert bar.omega_of(BODY, d, 0.03, home=W) == bar.omega_of(BODY, setio.set_dir(SET, BODY, t0), 0.03)
    ctx = bar.BarContext(BODY, d, {"refs.json": docs["refs.json"], "transitions.json": docs["transitions.json"]},
                         docs["set.json"])
    built = ctx.built(cond_of(docs, "turn/q90_left"))
    assert [s.name for s in built.track.segments[:3]] == [alt, W, setio.turn_name("pi4", "left")]
    assert built.omega == bar.omega_of(BODY, d, 0.03, home=W) and built.command["ref_s"] == built.track.segments[2].t0
    ctx.release()
    # another value is refused, and an absent or null key is "start"
    with pytest.raises(tune.InputError, match="lead_in 'start_v0.5'"):
        bar.BarPlanner(d, docs["refs.json"]["references"], docs["transitions.json"]["transitions"],
                       dict(docs["set.json"], lead_in="start_v0.5"))
    assert bar.lead_in_of(None) == bar.lead_in_of({}) == bar.lead_in_of(dict(lead_in=None)) == "start"
    # a set.json that disagrees with the set's entries is refused, in both directions: the exception on a set that
    # holds its nominal walk; F25F9's shape (no nominal walk, start_v0.4 and the 0.40 m/s walk) under another lead-in
    both = make_t0(tmp_path / "both", lead_in=alt, more_moves=((alt, "start"),))
    with pytest.raises(tune.InputError, match="records lead_in 'start_v0.4' but the set holds its nominal walk"):
        planner_of(tmp_path / "both")
    assert planner_of(make_t0(tmp_path / "plain", more_moves=((alt, "start"),)).parents[1]).lead == "start"  # an extra file alone: fine
    for i, said in enumerate((None, "start")):
        make_t0(tmp_path / f"shape{i}", unavailable=("nominal", "start", setio.change_name("nominal", W)), lead_in=said,
                more_moves=((alt, "start"),))
        with pytest.raises(tune.InputError, match=f"the set holds start_v0.4 and no nominal walk, but set.json records lead_in {said!r}"):
            planner_of(tmp_path / f"shape{i}")
    assert both.is_dir() and "is refused" in bar.RULES["lead_in"]
    # a rung whose neighbour's change failed its gates: the set records that under the direct change's name, and
    # the "not run" reason and numbers are that entry's (the neighbour's change has no entry of its own)
    make_t0(tmp_path / "step2", unavailable=("nominal", "start", setio.change_name("nominal", W), "nominal_to_v0.45"),
            lead_in=alt, more_refs=(("v0.45", "fr05"),), more_moves=((alt, "start"), ("nominal_to_v0.45", "brisk_to_fr05")))
    why = planner_of(tmp_path / "step2").missing(cond_of(setio.load(setio.set_dir(SET, BODY, tmp_path / "step2")), "walk/v0.45"))
    assert why == dict(reference="transitions/v0.4_to_v0.45", numbers=dict(hip_yaw_range_use=0.93),
                       reason="no transition into v0.45 (no reference within the thigh twist and the leg drives)")
    # --- the records: lead_in and its rule on such a set, on both kinds of record; no key on any other set
    faked(monkeypatch, lambda row, cond, seed, relax: fake_episode(cond, seed))
    out = tmp_path / "out"
    bar.evaluate(SET, BODY, out, ["R0@10ms"], ["walk/v0.4", "walk/nominal", "turn/q90_left"], episodes=1, procs=1,
                 root=root, log=quiet)
    done, not_run = read(rec_path(out, "R0@10ms", "walk/v0.4")), read(rec_path(out, "R0@10ms", "walk/nominal"))
    assert done["status"] == "done" and done["lead_in"] == alt and done["plan"][0] == ["transition", alt]
    assert done["rules"]["lead_in"] == bar.RULES["lead_in"]
    assert sorted(done["key"]["inputs"]) == [f"refs/{W}.npz", f"transitions/{alt}.npz"]
    assert not_run["status"] == "not run" and not_run["lead_in"] == alt and not_run["reference"] == "refs/nominal"
    assert not_run["rules"]["lead_in"] == bar.RULES["lead_in"]
    assert read(rec_path(out, "R0@10ms", "turn/q90_left"))["lead_in"] == alt
    for words in ("STAGE2_75_PLAN.md A8 as amended", "DECISION_LOG.md P34", "changed after a result",
                  "Every record of such a set carries lead_in"):
        assert words in bar.RULES["lead_in"], words
    plain = tmp_path / "plain_out"
    bar.evaluate(SET, BODY, plain, ["R0@10ms"], ["walk/v0.4"], episodes=1, procs=1, root=t0, log=quiet)
    rec = read(rec_path(plain, "R0@10ms", "walk/v0.4"))
    assert "lead_in" not in rec and "lead_in" not in rec["rules"] and rec["plan"][0] == ["transition", "start"]
    # the tables: the summary's cells and lines label such a set's rows, done and not run, and no other set's; the
    # kinematic check's record says it too
    doc = bar.summarize(out, SET, BODY, bar.HEADLINE_BLOCK)
    cells = [doc["grid"][k]["R0@10ms"] for k in ("walk/v0.4", "walk/nominal", "turn/q90_left")]
    assert [c["lead_in"] for c in cells] == [alt] * 3 and [c["status"] for c in cells] == ["done", "not run", "done"]
    label = f" [lead-in {alt}: an exception, A8 as amended (P34)]"
    lines = [x for x in bar.summary_text(doc).splitlines() if "R0@10ms" in x]
    assert len(lines) == 3 and all(x.endswith(label) for x in lines)
    plain_doc = bar.summarize(plain, SET, BODY, bar.HEADLINE_BLOCK)
    assert "lead_in" not in plain_doc["grid"]["walk/v0.4"]["R0@10ms"] and "lead-in" not in bar.summary_text(plain_doc)
    rc = bar.refcheck(SET, BODY, root, log=quiet)
    assert rc["lead_in"] == alt and rc["lead_in_rule"] == bar.RULES["lead_in"]
    assert "lead_in" not in bar.refcheck(SET, BODY, t0, log=quiet)
    said = make_t0(tmp_path / "said", lead_in="start")  # the key with the value every other set has
    assert json.dumps(planner_of(tmp_path / "said").resolve(cond_of(setio.load(said), "turn/q90_right")), default=str) \
        == json.dumps(ref.resolve(cond_of(t0_docs, "turn/q90_right")), default=str)


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
    assert rec["rules"]["delay"] == bar.RULES["delay"] and rec["set_info"] is None and rec["change_after_result"] is None
    assert s["traces"] is True and s["no_traces_for"] == [] and s["dirty_paths"] is not None
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
              "stage1", "delay", "set", "freeze", "checks", "no_overwrite"):
        assert k in rec["rules"], k
    assert rec["rules"]["delay"] == bar.RULES["delay"] and "its last walk segment repeats" in rec["rules"]["plans"]
    # what the record copies from set.json: a fixture whose walk stands in for the rung; the first freeze of each file
    info = rec["set_info"]
    assert info["role"] == "fixture" and info["stand_in_walks"] is True and rec["stand_in_walks"] is True
    assert info["freeze"]["index"] == 0 and set(info["first_freeze"]) == set(rec["key"]["inputs"])
    assert set(info["first_freeze"].values()) == {0} and info["change_after_result"] is None
    assert rec["change_after_result"] is None and e["trace"].get("kept") is None
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
    # a "not run" record is a registered outcome (H3): when the set later has the reference the key differs and the
    # run stops; --supersede keeps the "not run" record beside the new one, which says it is a re-run
    make_t0(tmp_path / "b", unavailable=("turn90_pi4_left",))
    bar.evaluate(SET, BODY, tmp_path / "o2", ["R0@10ms"], ["turn/q90_left"], episodes=2, procs=1, root=tmp_path / "b",
                 log=quiet)
    p2 = rec_path(tmp_path / "o2", "R0@10ms", "turn/q90_left")
    first = p2.read_bytes()
    assert read(p2)["status"] == "not run"
    with pytest.raises(bar.KeyMismatch, match="not run"):
        bar.evaluate(SET, BODY, tmp_path / "o2", ["R0@10ms"], ["turn/q90_left"], episodes=2, procs=1, root=t0, log=quiet)
    assert p2.read_bytes() == first and not p2.with_name("q90_left.json.superseded").exists()
    with pytest.raises(bar.KeyMismatch, match="key.seeds"):  # a "not run" record of another key is not rewritten either
        bar.evaluate(SET, BODY, tmp_path / "o2", ["R0@10ms"], ["turn/q90_left"], episodes=3, procs=1,
                     root=tmp_path / "b", log=quiet)
    assert p2.read_bytes() == first
    bar.evaluate(SET, BODY, tmp_path / "o2", ["R0@10ms"], ["turn/q90_left"], episodes=2, procs=1, root=t0, log=quiet,
                 supersede=True, supersede_reason="the set has the turn now")
    done = read(p2)
    kept2 = p2.with_name("q90_left.json.superseded") / f"{read_bytes_key(first)[:12]}.json"
    assert done["status"] == "done" and kept2.read_bytes() == first
    assert done["supersedes"]["reason"] == "the set has the turn now" and done["supersedes"]["registered"]["status"] == "not run"


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
    # every episode of the call finishes before any record is attributed; then record by record, in the order made
    order = ["lift" if c[3] else "episode" for c in calls]
    assert order == ["episode"] * 240 + ["lift"] * 40
    assert [c[1] for c in lifted[::20]] == ["walk/v0.4", "turn/q90_right"]
    assert len(lifted) == 40 and doc["sessions"][-1]["attribution_episodes_run"] == 40
    walk = read(rec_path(out, "R2/42@10ms", "walk/v0.4"))
    a = walk["attribution"]
    assert a["label"] == "body-limited: slip" and a["falls"] == 6 and a["episodes"] == 20 and a["candidates"] == ["slip"]
    assert a["lifted"]["slip"]["significant"] is True and a["rule_bar"] == bar.RULES["attribution"]
    assert walk["failure"]["falls_label"] == "body-limited: slip" and walk["failure"]["kind"] == "falls"
    assert walk["failure_but_A1"]["falls_label"] == "body-limited: slip"
    assert walk["falls"] == dict(before_command=0, after_command=0, no_command=6)
    # the record names the session that attributed it (here the one that ran its episodes: no second entry)
    a_sess = a["session"]
    assert a_sess["attribution"] is True and a_sess["lifted"] == ["slip"] and a_sess["seeds_run"] == []
    assert a_sess["id"] == walk["sessions"][-1]["id"] and len(walk["sessions"]) == 1 and a_sess["commit"] == walk["sessions"][0]["commit"]
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
    four = read(rec_path(out, "R2/42@4ms", "walk/v0.4"))
    assert four["attribution"]["label"] == "body-limited: slip"
    # ... by a later session: the record gets an entry for it, with no seed run, after the one that ran the episodes
    assert [x["seeds_run"] for x in four["sessions"]] == [list(range(3042, 3062)), []]
    assert four["sessions"][1]["attribution"] is True and four["sessions"][1]["id"] == doc["sessions"][-1]["id"]
    assert four["sessions"][1]["id"] != four["sessions"][0]["id"] and four["attribution"]["session"] == four["sessions"][1]
    cited = read(rec_path(out, "R2/42@4ms", "turn/q90_left"))
    assert cited["attribution"]["session"]["lifted"] == [] and len(cited["sessions"]) == 2
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
    frozen = {p: p.read_bytes() for p in bar.block_dir(out, SET, BODY, "seeds3042").rglob("*.json")
              if p.name != "summary.json"}
    # a shown or provisional verdict never stands alone: before the replication ran, it says so
    v = one["verdicts"]["walk/v0.4"]["R2@10ms"]
    assert v["reading_2"]["verdict"] == "provisional" and v["reading_2"]["replication"] == {"R2/42@10ms": "replication not run yet"}
    assert "provisional [R2/42@10ms: replication not run yet]" in bar.summary_text(one)

    def second(row, cond, seed, relax):  # the fresh seeds: the low-grip walk repeats its reading 3 pass, nothing else
        if cond.key == "walk/v0.4_mu0.54":
            return fake_episode(cond, seed, tail_over=True)
        return fake_episode(cond, seed, fail=("joint_range",))

    # the replication block: exactly the records that passed in reading 2 or 3, of rows with a verdict
    calls = faked(monkeypatch, second)
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
    # the headline block's summary was made again: beside each verdict, the replication word of each record that passed
    head = read(bar.block_dir(out, SET, BODY, "seeds3042") / "eval" / "summary.json")
    assert len(head["sessions"]) == 1 and head["grid"] == one["grid"]
    hv = head["verdicts"]
    assert hv["walk/v0.4"]["R2@10ms"]["reading_2"]["replication"] == {"R2/42@10ms": "not replicated"}
    assert hv["walk/v0.4"]["R2@10ms"]["reading_3"]["replication"] == {"R2/42@10ms": "not replicated"}
    assert hv["walk/v0.4_mu0.54"]["R2@10ms"]["reading_3"]["replication"] == {"R2/42@10ms": "replicated"}
    assert "replication" not in hv["walk/v0.4_mu0.54"]["R2@10ms"]["reading_2"]  # body-limited: no word beside it
    assert "replication" not in hv["turn/q90_right"]["R2@10ms"]["reading_2"]  # not read: a1_over
    text = bar.summary_text(head)
    assert "provisional but for A1 [R2/42@10ms: replicated]" in text and "provisional [R2/42@10ms: not replicated]" in text
    assert set(bar.REPLICATION_WORDS) == {"replicated", "not replicated", "replication cut by the pre-registered order"}
    # a replication the ledger cut is printed as cut
    cut_rec = "T0/75-M-H-10/seeds5042/eval/R2/43@10ms/walk/v0.4.json"
    s2io.write_json(out / "budget.json", dict(tasks={"3": dict(units=[], cut=[dict(unit="T3.2", detail=dict(paths=[cut_rec]))])}))
    assert bar.cut_paths(out) == {cut_rec}
    assert bar.replication_word(out, SET, BODY, "R2/43@10ms", "walk/v0.4", "success") == "replication cut by the pre-registered order"
    assert bar.replication_word(out, SET, BODY, "R2/43@10ms", "walk/v0.4_mu0.54", "success") == "replication not run yet"
    assert bar.replication_word(out, SET, BODY, "R2/42@10ms", "walk/v0.4", "success") == "not replicated"
    (out / "budget.json").unlink()
    # H4: only the candidate is replicated; the controls of the round are not run on the fresh seeds
    for control_set in ("V0", "F25F9", "V0.2"):
        with pytest.raises(bar.BarError, match="is a control of the 75 cm round"):
            bar.evaluate(control_set, BODY, tmp_path / "c", ["R2/42@10ms"], conds, block="seeds5042", passed_of="seeds3042",
                         procs=1, root=root, task=3, rates={}, log=quiet)
    with pytest.raises(tune.InputError, match="set.json is missing"):  # the candidate passes that rule (no such set here)
        bar.evaluate("F25F9cl", BODY, tmp_path / "c", ["R2/42@10ms"], conds, block="seeds5042", passed_of="seeds3042",
                     procs=1, root=root, task=3, rates={}, log=quiet)
    assert "only F25F9cl" in bar.RULES["replication"] and "controls are not run on seeds5042" in bar.RULES["replication"]
    assert bar.verdict_text(dict(verdict="shown", replication=None, replication_note="a control is not replicated")) == (
        "shown [a control is not replicated]")


@needs_run
def test_the_omega_control_runs_apart_with_stage1s_crouch(tmp_path, monkeypatch):
    root = tmp_path / "sets"
    d = make_t0(root, crouch=0.05)  # a set that crouches 0.05 H: set.json and every walk entry say so
    sd = s2io.read_json(d / "set.json")
    out = tmp_path / "out"
    faked(monkeypatch, lambda row, cond, seed, relax: fake_episode(cond, seed))
    bar.evaluate(SET, BODY, out, ["R2/42@10ms"], ["walk/v0.4"], episodes=2, procs=1, root=root, log=quiet)
    # a control gives no verdict and is neither replicated nor attributed: the runner refuses to be asked
    for kw in (dict(), dict(attribute="all"), dict(attribute="none", block="seeds5042", passed_of="seeds3042")):
        with pytest.raises(bar.BarError, match="neither replicated nor attributed"):
            bar.evaluate(SET, BODY, out, ["R2/42@10ms"], ["walk/v0.4"], episodes=2, procs=1, root=root,
                         control="omega_s1", log=quiet, **kw)
    bar.evaluate(SET, BODY, out, ["R2/42@10ms"], ["walk/v0.4"], episodes=2, procs=1, root=root, control="omega_s1",
                 attribute="none", log=quiet)
    own = read(rec_path(out, "R2/42@10ms", "walk/v0.4"))
    ctl = read(rec_path(out, "R2/42@10ms", "walk/v0.4", control="omega_s1"))
    # each record states the rule it ran by: the set's own crouch, or the control's
    assert own["rules"]["omega"] == bar.RULES["omega"] and ctl["rules"]["omega"] == bar.CONTROL_RULES["omega_s1"]
    assert "Stage 1's 0.03 H crouch instead of the set's own" in ctl["rules"]["omega"]
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
    # ... and so is one whose set.json disagrees with its frozen walk entries (no freeze covers set.json's own value):
    # under the same manifest hash the track would run at another frequency
    s2io.write_json(d / "set.json", dict(sd, crouch_frac=0.03))
    for kw in (dict(), dict(control="omega_s1", attribute="none")):
        with pytest.raises(bar.BarError, match=r"crouch_frac 0.03, its walk entry 'nominal' records 0.05"):
            bar.evaluate(SET, BODY, tmp_path / "none", ["R2/42@10ms"], ["walk/v0.4"], episodes=2, procs=1, root=root,
                         log=quiet, **kw)
    with pytest.raises(bar.BarError, match="crouch_frac"):
        bar.refcheck(SET, BODY, root, log=quiet)
    with pytest.raises(bar.BarError, match="crouch_frac"):  # a worker checks the set it reads itself
        bar._worker(BODY, str(d), "bar", bar.RATE, None, set_name=SET, files=["refs/nominal.npz", "refs/v0.4.npz"])
    assert not (tmp_path / "none").exists() or not list((tmp_path / "none").rglob("*.json"))
    assert bar.set_crouch(dict(set="X", crouch_frac=0.05), dict(a=dict(crouch_frac=0.05)), ["a"]) == 0.05
    with pytest.raises(bar.BarError, match="records None"):
        bar.set_crouch(dict(set="X", crouch_frac=0.05), dict(a=dict()), ["a"])


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
    assert led["units"][0]["detail"]["paths"] == [f"T0/75-M-H-10/seeds3042/eval/{r}/walk/v0.4.json"
                                                  for r in ("R2/42@10ms", "R0@10ms")]
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
    assert rec["failure_but_A1"]["falls_label"] == "not attributed (budget, pre-registered order)"
    assert rec["attribution"]["session"]["lifted"] == [] and rec == json.loads(json.dumps(rec))
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
    bar._check_rule("S1", ["R2/42@10ms"], 20, None, None, None, None)  # Stage 1's own run directory: R2 rows may run
    # a version of a registered set is under the same rule: no R2 row on it without a task, and no check either
    assert [bar.registered_name(x) for x in ("V0", "F25F9cl.2", "V0.2", SET, "S1", "D0")] == ["V0", "F25F9cl", "V0", None,
                                                                                         None, None]
    for name in ("F25F9cl.2", "V0.2"):
        with pytest.raises(bar.BarError, match="build check"):
            bar._check_rule(name, ["R0@10ms"], 1, None, "dev.json", None, None)
        bar._check_rule(name, ["R2/42@10ms"], 20, 3, None, None, None)
    # a scratch or fixture set: R0 rows only, never an R2 row, with a task or without (amendment 5 of 15 Nov)
    bar._check_rule(SET, ["R0@10ms", "R0@0ms"], 20, None, None, None, None)
    for task in (None, 3):
        with pytest.raises(bar.BarError, match="never an R2 row"):
            bar._check_rule(SET, ["R0@10ms", "R2/42@10ms"], 20, task, None, None, None)
    with pytest.raises(bar.BarError, match="never an R2 row"):
        bar.evaluate(SET, BODY, tmp_path / "x", ["R2/42@10ms"], ["walk/v0.4"], procs=1, root=t0, log=quiet)
    # where a run may write: never under stage1/ or the sets; a run without a task not into a task's directory, and a
    # check on a set of the round not under stage2/runs/
    for bad, word in ((ROOT / "stage1" / "runs" / "x", "stage1/"), (ROOT / "stage2" / "sets" / "x", "stage2/sets/"),
                      (t0 / "x", "the sets' root")):
        with pytest.raises(bar.BarError, match=word):
            bar._check_out(bad, SET, 3, t0)
    with pytest.raises(bar.BarError, match="stage2/runs/"):
        bar._check_out(ROOT / "stage2" / "runs" / "2026-11-15-75cm", "V0", None, t0)
    bar._check_out(ROOT / "stage2" / "runs" / "2026-11-15-75cm", "V0", 3, t0)
    bar._check_out(ROOT / "stage2" / "runs" / "some-later-run", SET, None, t0)
    led = tmp_path / "ledger"
    s2io.write_json(led / "budget.json", dict(tasks={}))
    with pytest.raises(bar.BarError, match="holds a budget.json"):
        bar._check_out(led, SET, None, t0)
    bar._check_out(led, SET, 4, t0)
    with pytest.raises(bar.BarError, match="holds a budget.json"):
        bar.evaluate(SET, BODY, led, ["R0@10ms"], ["walk/v0.4"], episodes=1, procs=1, root=t0, log=quiet)
    # the sweep of a killed run's temporary files touches the bar's own directories only
    for q in ("COST.md.tmp", "budget.json.tmp", f"{SET}/{BODY}/a.json.tmp", "controls/omega_s1/b.tmp", "other/c.tmp"):
        (led / q).parent.mkdir(parents=True, exist_ok=True)
        (led / q).write_text("x")
    assert sorted(bar._sweep(led, SET)) == ["T0/75-M-H-10/a.json.tmp", "budget.json.tmp", "controls/omega_s1/b.tmp"]
    assert (led / "COST.md.tmp").exists() and (led / "other" / "c.tmp").exists() and not (led / "budget.json.tmp").exists()
    with pytest.raises(bar.BarError, match="build check"):
        bar.evaluate("V0", BODY, tmp_path / "x", ["R2/42@10ms"], ["walk/v0.4"], procs=1, root=t0, log=quiet)
    assert not (tmp_path / "x").exists()
    # a check's episodes are listed in the file --dev-episodes names: when they start, and again when recorded.  A
    # check that is killed is on the list all the same, and its resume lists nothing twice
    dev = tmp_path / "dev_episodes.json"
    s2io.write_json(dev, dict(scratch_sets=[dict(set="D0", note="listed by hand")]))  # what else the file holds stays

    def killed(row, cond, seed, relax):
        if cond.key == "stop/two_step":
            raise KeyboardInterrupt
        return fake_episode(cond, seed)

    faked(monkeypatch, killed)
    args = (SET, BODY, tmp_path / "out", ["R0@10ms"], ["walk/v0.4", "stop/two_step"])
    with pytest.raises(KeyboardInterrupt):
        bar.evaluate(*args, episodes=1, procs=1, root=t0, dev_episodes=dev, log=quiet)
    listed = read(dev)
    assert [(x["condition"], x["state"]) for x in listed["episodes"]] == [("walk/v0.4", "done"), ("stop/two_step", "started")]
    assert listed["scratch_sets"] == [dict(set="D0", note="listed by hand")]
    faked(monkeypatch, lambda row, cond, seed, relax: fake_episode(cond, seed))
    for _ in (1, 2):
        bar.evaluate(*args, episodes=1, procs=1, root=t0, dev_episodes=dev, log=quiet)
    listed = read(dev)
    assert [(x["set"], x["row"], x["condition"], x["seed"], x["state"]) for x in listed["episodes"]] == [
        (SET, "R0@10ms", "walk/v0.4", 3042, "done"), (SET, "R0@10ms", "stop/two_step", 3042, "done")]
    assert listed["episodes"][0]["bar_sha256"] == bar.bar_sources()["sha256"] and listed["scratch_sets"]
    assert listed["episodes"][0]["reason"] == "the bar's build check: smoke, no record"
    assert all(x["started_at"] and x["recorded_at"] and x["error"] is False for x in listed["episodes"])
    # an episode a killed check finished (kept as a partial file) is listed when the resume records it
    dev2 = tmp_path / "dev2.json"

    def killed_at_second_row(row, cond, seed, relax):
        if row.name == "R0@0ms":
            raise KeyboardInterrupt
        return fake_episode(cond, seed)

    faked(monkeypatch, killed_at_second_row)
    args2 = (SET, BODY, tmp_path / "out2", ["R0@10ms", "R0@0ms"], ["walk/v0.4"])
    with pytest.raises(KeyboardInterrupt):
        bar.evaluate(*args2, episodes=1, procs=1, root=t0, dev_episodes=dev2, log=quiet)
    assert [(x["row"], x["state"]) for x in read(dev2)["episodes"]] == [("R0@10ms", "started"), ("R0@0ms", "started")]
    faked(monkeypatch, lambda row, cond, seed, relax: fake_episode(cond, seed))
    doc = bar.evaluate(*args2, episodes=1, procs=1, root=t0, dev_episodes=dev2, log=quiet)
    assert doc["sessions"][-1]["reused_partial"] == ["R0@10ms walk/v0.4: seeds [3042]"]
    assert [(x["row"], x["state"]) for x in read(dev2)["episodes"]] == [("R0@10ms", "done"), ("R0@0ms", "done")]


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
# The review's fixes of 15 Nov 2026 (one test per rule the review found the runner short of)
# ------------------------------------------------------------------------------------------------
@needs_run
def test_the_turns_joint_range_and_leg_gap_are_read_over_its_window(t0, t0_docs):
    """C5: over [t_c, t_c + at_s + 3 s] the joint range by decision 14 and no leg interference.  The turn's track is
    played kinematically (a recorder that does not fall) and judged; an excursion in the lead-in, 10 s before the
    command, or after the window fails nothing, the same excursion inside the window fails the term."""
    d = setio.set_dir(SET, BODY, t0)
    ctx = bar.BarContext(BODY, d, {"refs.json": t0_docs["refs.json"], "transitions.json": t0_docs["transitions.json"]},
                         t0_docs["set.json"])
    b = ctx.built(cond_of(t0_docs, "turn/q90_left"))
    rec = bar.reference_recorder(b.p, b.track, b.seconds, d, rate=bar.RATE)
    rec.legs_every = tune.LEGS_EVERY
    rec.poses = [np.array([float(k)]) for k in range(0, rec.rows, tune.LEGS_EVERY)]  # a pose is its update row here
    out = dict(fell=False, fall_time_s=None, fall_cause=None, seconds=b.seconds, strides=20.0,
               worst_sustained_ratio=["hip_pitch_l", 0.5], worst_speed_ratio=["hip_pitch_l", 0.5], segment_reached="v0.4")

    class Gap:  # legs.LegGap's trace on those poses: interference at the rows named
        def __init__(self, bad=()):
            self.bad = set(bad)

        def trace(self, poses):
            hit = [int(q[0]) for q in poses if int(q[0]) in self.bad]
            return dict(poses=len(poses), interference_free=not hit, interference_poses=len(hit), tight_poses=0,
                        min_gap_all_mm=5.0, min_gap_all_pair=None, min_gap_moving_mm=5.0, min_gap_moving_pair=None)

    def judged(Q=None, bad=()):
        r = copy.copy(rec)
        r.Q = rec.Q if Q is None else Q
        return bar._judge(b, out, r, 3042, Gap(bad))[0]

    e = judged()
    t_c = e["command_s"]
    t_end = t_c + 2.0 + 3.0
    assert e["window_reached"] and e["success"]["joint_range"] and e["success"]["no_leg_interference"]
    assert 10.0 < t_c < 16.0 and float(rec.t[-1]) > t_end + 0.5
    mb = e["measures"]["bar"]
    assert mb["joint_range"]["span"] == "window" and mb["joint_range"]["window_s"] == [t_c, t_end]
    assert mb["joint_range"]["rows"] == int(np.sum((rec.t >= t_c - 1e-9) & (rec.t <= t_end + 1e-9)))
    assert (mb["drive_speed_span"], mb["joint_range_span"], mb["no_leg_interference_span"]) == ("episode", "window", "window")
    legs = e["measures"]["legs"]
    assert legs["span"] == "window" and legs["window_complete"] is True and legs["poses"] < legs["poses_episode"]
    assert legs["poses"] == sum(1 for k in range(0, rec.rows, 2) if t_c - 1e-9 <= rec.t[k] <= t_end + 1e-9)
    assert e["measures"]["legs_episode"]["poses"] == len(rec.poses)
    j = int(np.flatnonzero(plant.rest_ends(b.p.joints) == 0)[0])
    row = lambda t: int(np.searchsorted(rec.t, t - 1e-9))  # noqa: E731

    def excursion(ta, tb):  # joint j held 0.05 rad beyond its upper limit over [ta, tb)
        Q = rec.Q.copy()
        Q[row(ta) : row(tb), j] = b.hi[j] + 0.05
        return judged(Q)

    lead = excursion(3.0, 3.5)
    assert lead["success"]["joint_range"] is True and lead["measures"]["bar"]["joint_range"] == mb["joint_range"]
    assert lead["joint_shortfall_rad"] > 0.05 and lead["measures"]["bar"]["joint_range_episode"]["ok"] is False
    assert lead["succeeded"] == e["succeeded"]
    inside = excursion(t_c + 1.0, t_c + 1.2)
    assert inside["success"]["joint_range"] is False and "joint_range" in inside["failed_terms"]
    assert inside["measures"]["bar"]["joint_range"]["joint_worst"] == b.p.joints[j]
    assert inside["measures"]["bar"]["joint_range"]["joint_shortfall_rad"] == pytest.approx(b.required[j] + 0.05)
    after = excursion(t_end + 0.05, t_end + 0.5)
    assert after["success"]["joint_range"] is True and after["joint_shortfall_rad"] > 0.05
    # the window's own edges are inside it
    assert excursion(t_c, t_c + 0.0105)["success"]["joint_range"] is False  # the first update at or after t_c
    assert excursion(t_end - 0.0105, t_end)["success"]["joint_range"] is False  # the last update at or before the end
    assert excursion(t_c - 0.0105, t_c - 1e-6)["success"]["joint_range"] is True  # the update just before the command
    # the leg gap: a pose that interferes in the lead-in or after the window fails nothing; one inside it does
    pose = lambda t: (row(t) // 2) * 2  # noqa: E731
    for t_bad, ok in ((3.0, True), (t_c + 1.0, False), (t_end + 0.3, True)):
        e2 = judged(bad={pose(t_bad)})
        assert e2["success"]["no_leg_interference"] is ok, t_bad
        assert e2["measures"]["legs_episode"]["interference_free"] is False
        assert e2["measures"]["legs"]["interference_free"] is ok
    # a walk keeps Stage 1's span for the joint range (the episode): C4 names no window for it
    assert "over the episode the drives' peak speed" in bar.RULES["walk"]


@needs_run
def test_a_rungs_entry_holds_its_names_speed_and_a_causal_move_its_phase(t0, t0_docs, tmp_path):
    d = setio.set_dir(SET, BODY, t0)
    refs, trs = t0_docs["refs.json"]["references"], t0_docs["transitions.json"]["transitions"]
    sd = t0_docs["set.json"]
    walk = cond_of(t0_docs, "walk/v0.4")
    # T0 declares that its walk stands in for the rung; without that the set is refused before any plan is made
    strict = bar.BarPlanner(d, refs, trs, dict(sd, stand_in_walks=False))
    with pytest.raises(tune.InputError, match=r"walk 'v0.4': its entry's speed_m_s is 0.35\d+, its name holds 0.4 m/s"):
        strict.resolve(walk)
    assert planner_of(t0).stand_in_walks is True and planner_of(t0).resolve(walk)["target_speed_m_s"] == 0.4
    strict.resolve(cond_of(t0_docs, "walk/nominal"))  # the nominal walk's name holds no speed
    for factor, ok in ((1 + 5e-7, True), (1 - 5e-7, True), (1 + 2e-6, False), (1 - 2e-6, False)):
        near = copy.deepcopy(refs)
        near["v0.4"]["speed_m_s"] = 0.4 * factor
        pl = bar.BarPlanner(d, near, trs, dict(sd, stand_in_walks=False))
        if ok:
            assert pl.resolve(walk)["target_speed_m_s"] == 0.4
        else:
            with pytest.raises(tune.InputError, match="relative tolerance 1e-06"):
                pl.resolve(cond_of(t0_docs, "nudge/left"))  # every item that plays the walk
    # a set of the round, or a version of one, has no stand-in walk
    for name in ("V0", "F25F9cl.2"):
        with pytest.raises(tune.InputError, match="no stand-in walk"):
            bar.BarPlanner(d, refs, trs, dict(sd, set=name))
    # through the runner: refused before any episode
    root = tmp_path / "sets"
    make_t0(root, stand_in=False)
    with pytest.raises(tune.InputError, match="stand_in_walks"):
        bar.evaluate(SET, BODY, tmp_path / "out", ["R0@10ms"], ["walk/v0.4"], episodes=1, procs=1, root=root, log=quiet)
    assert not list((tmp_path / "out").rglob("*.json"))
    # a causal move is placed at the walk phase its entry was built for: the leads are 2, 2.5 and 2.5 strides
    pl = planner_of(t0)
    for key, name, built_for, other in (("turn/q90_left", "turn90_pi4_left", 0.0, 0.5),
                                        ("turn/q90_right", "turn90_pi4_right", 0.5, 0.0),
                                        ("stop/two_step", "stop2_v0.4", -0.5, 0.0)):
        assert pl.resolve(cond_of(t0_docs, key))["command"]["phase"] == built_for
        for phase in (other, built_for + 0.01, None):
            trs2 = copy.deepcopy(trs)
            trs2[name]["move"]["command_phase"] = phase
            with pytest.raises(tune.InputError, match="was built for the walk phase"):
                bar.BarPlanner(d, refs, trs2, sd).resolve(cond_of(t0_docs, key))
        trs2 = copy.deepcopy(trs)
        trs2[name]["move"]["command_phase"] = built_for + 1.0  # whole strides apart are one phase
        assert bar.BarPlanner(d, refs, trs2, sd).resolve(cond_of(t0_docs, key))["plan"]
    trs2 = copy.deepcopy(trs)
    trs2["turn90_pi4_left"]["move"]["causal"] = False
    with pytest.raises(tune.InputError, match="not a causal move"):
        bar.BarPlanner(d, refs, trs2, sd).resolve(cond_of(t0_docs, "turn/q90_left"))
    trs2 = copy.deepcopy(trs)  # Stage 1's non-causal stop is read from Stage 1's command: no phase to hold
    trs2["stop2_v0.4"]["move"] = dict(kind="stop", causal=False, command_phase=None)
    res = bar.BarPlanner(d, refs, trs2, sd).resolve(cond_of(t0_docs, "stop/two_step"))
    assert res["command"]["convention"].startswith("Stage 1's")


@needs_run
def test_a_record_names_its_freeze_and_a_change_after_a_result(tmp_path, monkeypatch):
    """A13: a reference a later freeze added is labelled in every record that plays it; the freeze's stored
    manifest hash must be its manifest's."""
    root = tmp_path / "sets"
    d = make_t0(root)
    sd = s2io.read_json(d / "set.json")
    full = sd["freezes"][0]["manifest"]
    first = {k: v for k, v in full.items() if k != "turn90_pi4_left"}
    why = "the left turn was built after the walk's records"
    sd["freezes"] = [dict(index=0, at="fixture", commit=None, manifest=first, manifest_sha256=s2io.sha256_json(first)),
                     dict(index=1, at="fixture", commit=None, manifest=full, manifest_sha256=s2io.sha256_json(full),
                          change_after_result=why, after_check_episodes=True, reason="the freeze after the checks")]
    s2io.write_json(d / "set.json", sd)
    faked(monkeypatch, lambda row, cond, seed, relax: fake_episode(cond, seed))
    out = tmp_path / "out"
    conds = ["walk/v0.4", "turn/q90_left", "turn/q90_right"]
    doc = bar.evaluate(SET, BODY, out, ["R0@10ms"], conds, episodes=2, procs=1, root=root, log=quiet)
    walk, left, right = (read(rec_path(out, "R0@10ms", k)) for k in conds)
    assert walk["change_after_result"] is None and right["change_after_result"] is None
    assert left["change_after_result"] == dict(label="change after a result", freezes=[
        dict(freeze=1, reason=why, files=["transitions/turn90_pi4_left.npz"])])
    info = left["set_info"]
    assert info["first_freeze"]["transitions/turn90_pi4_left.npz"] == 1 and info["first_freeze"]["refs/v0.4.npz"] == 0
    assert info["after_check_episodes"] == [dict(freeze=1, reason="the freeze after the checks")]
    assert info["freeze"]["change_after_result"] == why and left["set_freeze"]["index"] == 1
    assert walk["set_info"]["freeze"]["index"] == 1 and set(walk["set_info"]["first_freeze"].values()) == {0}
    assert left["set_freeze"]["manifest_sha256"] == s2io.sha256_json(full)
    assert doc["grid"]["turn/q90_left"]["R0@10ms"]["change_after_result"]["label"] == "change after a result"
    assert doc["grid"]["walk/v0.4"]["R0@10ms"]["change_after_result"] is None
    assert bar.summary_text(doc).count("[change after a result]") == 1
    assert "change_after_result" in bar.RULES["freeze"] and left["rules"]["freeze"] == bar.RULES["freeze"]
    # a manifest edited under its old hash (with the file's entry edited to match) is refused, by the runner, by a
    # worker and by refcheck: the record's set_freeze.manifest_sha256 names the manifest the files were checked against
    sd["freezes"][1]["manifest"]["v0.4"] = dict(full["v0.4"], entry_sha256="0" * 64)
    s2io.write_json(d / "set.json", sd)
    with pytest.raises(tune.InputError, match="freeze 1 stores manifest_sha256"):
        bar.evaluate(SET, BODY, tmp_path / "o2", ["R0@10ms"], ["walk/v0.4"], episodes=2, procs=1, root=root, log=quiet)
    with pytest.raises(tune.InputError, match="manifest_sha256"):
        bar._worker(BODY, str(d), "bar", bar.RATE, None, set_name=SET, files=["refs/nominal.npz"])
    with pytest.raises(tune.InputError, match="manifest_sha256"):
        bar.refcheck(SET, BODY, root, log=quiet)
    assert not list((tmp_path / "o2").rglob("*.json"))
    bar.check_manifest(dict(set="X"))  # a set that was never frozen has nothing to compare (the runner refuses it)


@needs_run
def test_a_trace_that_is_not_kept_is_hashed_all_the_same(t0, t0_run, tmp_path, monkeypatch):
    """R3: every episode's trace sha256 is in its record.  With --no-traces-for R0 or --no-traces the trace is
    written, hashed and removed; the record and its session say so."""
    kept = read(rec_path(t0_run, "R0@0ms", "stop/two_step"))["episodes"][0]["trace"]
    assert "kept" not in kept and kept["path"] == "R0@0ms/stop/two_step/3042.npz"
    out = tmp_path / "out"
    doc = bar.evaluate(SET, BODY, out, ["R0@0ms"], ["stop/two_step"], episodes=1, procs=1, root=t0, log=quiet,
                       no_traces_for=("R0",))
    rec = read(rec_path(out, "R0@0ms", "stop/two_step"))
    tr = rec["episodes"][0]["trace"]
    assert tr == dict({k: v for k, v in kept.items() if k != "path"}, kept=False, reason="--no-traces-for R0")
    assert tr["sha256"] == kept["sha256"] and tr["rows"] == kept["rows"]
    assert not list(out.rglob("*.npz")) and not (bar.block_dir(out, SET, BODY, "seeds3042") / "eval" / "traces").exists()
    assert not list(out.rglob("*.tmp"))
    assert rec["sessions"][0]["no_traces_for"] == ["R0"] and rec["sessions"][0]["traces"] is True
    assert doc["sessions"][-1]["no_traces_for"] == ["R0"] and "kept: false" in rec["rules"]["trace"]
    # a stock episode (Stage 1's own trace) under --no-traces: the same
    stock_cut(monkeypatch, 1.0)
    a, b = tmp_path / "a", tmp_path / "b"
    bar.evaluate(bar.STOCK_SET, BODY, a, ["R0@0ms"], ["walk/brisk"], episodes=1, procs=1, stage1_run=RUN, log=quiet)
    bar.evaluate(bar.STOCK_SET, BODY, b, ["R0@0ms"], ["walk/brisk"], episodes=1, procs=1, stage1_run=RUN, log=quiet,
                 traces=False)
    ta = read(rec_path(a, "R0@0ms", "walk/brisk", set_name="S1"))["episodes"][0]["trace"]
    rb = read(rec_path(b, "R0@0ms", "walk/brisk", set_name="S1"))
    tb = rb["episodes"][0]["trace"]
    assert tb["sha256"] == ta["sha256"] and tb["kept"] is False and tb["reason"] == "--no-traces" and "path" not in tb
    assert rb["sessions"][0]["traces"] is False and not list(b.rglob("*.npz")) and len(list(a.rglob("*.npz"))) == 1
    # a row whose controller is not named keeps its trace
    assert bar.BarWorker.not_kept(type("W", (), dict(keep_traces=True, no_traces_for=("R0",)))(), "R2/42@10ms") is None


@needs_run
def test_the_dirty_flag_leaves_the_runs_own_outputs_out(t0, tmp_path, monkeypatch):
    """The registered run writes into the repository: its own records must not turn every later session dirty."""
    out = tmp_path / "out"
    faked(monkeypatch, lambda row, cond, seed, relax: fake_episode(cond, seed))
    status = {"lines": []}
    seen = []

    def git_state(own_outputs=()):  # s2io.git_state on a status this test sets (paths as git prints them)
        seen.append([str(x) for x in own_outputs])
        own = [Path(o) for o in own_outputs]
        dirty = [x for x in status["lines"] if not any(Path(x[3:]) == o or o in Path(x[3:]).parents for o in own)]
        return dict(commit="c" * 40, dirty=bool(dirty), dirty_paths=dirty)

    monkeypatch.setattr(bar.s2io, "git_state", git_state)
    status["lines"] = [f"?? {out.resolve()}/T0/75-M-H-10/seeds3042/eval/R0@10ms/walk/v0.4.json", f"?? {out.resolve()}/budget.json"]
    bar.evaluate(SET, BODY, out, ["R0@10ms"], ["walk/v0.4"], episodes=1, procs=1, root=t0, log=quiet)
    rec = read(rec_path(out, "R0@10ms", "walk/v0.4"))
    assert seen[-1] == [str(out.resolve())]
    assert rec["provenance"]["git"] == dict(commit="c" * 40, dirty=False, dirty_paths=[])
    assert rec["sessions"][0]["dirty"] is False and rec["sessions"][0]["dirty_paths"] == []
    assert {"versions", "stage1_sources", "panel_sources", "bar_sources"} <= set(rec["provenance"])
    # an edited source elsewhere in the study still shows, in the record's provenance and in its session
    status["lines"].append(" M hardware/trex_form_study/designlab/stage1/control.py")
    bar.evaluate(SET, BODY, out, ["R0@10ms"], ["stop/two_step"], episodes=1, procs=1, root=t0, log=quiet)
    rec = read(rec_path(out, "R0@10ms", "stop/two_step"))
    assert rec["provenance"]["git"]["dirty_paths"] == [" M hardware/trex_form_study/designlab/stage1/control.py"]
    assert rec["sessions"][0]["dirty"] is True and rec["sessions"][0]["dirty_paths"] == rec["provenance"]["git"]["dirty_paths"]
    # refcheck and reread write their own records too
    bar.refcheck(SET, BODY, t0, tmp_path / "rc", log=quiet)
    assert seen[-1] == [str((tmp_path / "rc").resolve())]


@needs_run
def test_bar_run_exits_4_while_an_episode_failed_to_run(t0, tmp_path, monkeypatch, capsys):
    """C8: the seed is run again on the next run of the same key; until then the step is not done."""
    out = tmp_path / "out"

    def raising(row, cond, seed, relax):
        if seed == 3043 and cond.key == "stop/two_step":
            raise RuntimeError("the episode broke")
        return fake_episode(cond, seed)

    argv = ["run", "--set", SET, "--set-root", str(t0), "--body", BODY, "--rows", "R0@10ms", "--conditions",
            "walk/v0.4,stop/two_step", "--episodes", "3", "--procs", "1", "--out", str(out)]
    faked(monkeypatch, raising)
    assert bar.main(argv) == 4
    err = capsys.readouterr().err
    assert "R0@10ms stop/two_step: 1 episodes failed to run" in err and "walk/v0.4" not in err and "exit 4" in err
    rec = read(rec_path(out, "R0@10ms", "stop/two_step"))
    assert rec["counts"]["errors"] == 1 and rec["counts"]["pass_"] is None
    summary = read(bar.block_dir(out, SET, BODY, "seeds3042") / "eval" / "summary.json")
    assert summary["sessions"][-1]["incomplete"] == ["R0@10ms stop/two_step: 1 episodes failed to run"]
    assert bar.main(argv) == 4  # the same failure again: still not done
    calls = faked(monkeypatch, lambda row, cond, seed, relax: fake_episode(cond, seed))
    assert bar.main(argv) == 0 and [c[1:3] for c in calls] == [("stop/two_step", 3043)]
    assert read(rec_path(out, "R0@10ms", "stop/two_step"))["counts"]["errors"] == 0
    assert bar.main(argv) == 0 and len(calls) == 1 and "exits 4" in bar.RULES["error"]


@needs_run
def test_a_reruns_label_survives_a_kill_a_failed_episode_and_a_second_rerun(t0, tmp_path, monkeypatch):
    """R2: a record made after a result says so (supersedes.reason) however its run went; the registered record's
    counts stay with it; the moved record's traces move with it; a record file that cannot be read is no missing
    record."""
    out = tmp_path / "out"
    state = dict(fail=(), text=b"first", kill=None, error=None)

    def run_episode(row, ctx, cond, seed, relax=(), trace_root=None, **kw):  # a synthetic episode with a trace file
        if seed == state["kill"]:
            raise KeyboardInterrupt
        if seed == state["error"]:
            raise RuntimeError("the episode broke")
        e = fake_episode(cond, seed, fail=state["fail"])
        if trace_root is not None:
            rel = panels.trace_path(row.name, cond, seed)
            (Path(trace_root) / rel).parent.mkdir(parents=True, exist_ok=True)
            (Path(trace_root) / rel).write_bytes(state["text"] + str(seed).encode())
            e["trace"] = dict(path=rel.as_posix(), sha256=s2io.sha256_file(Path(trace_root) / rel), rows=1)
        return e

    monkeypatch.setattr(bar, "run_episode", run_episode)
    monkeypatch.setattr(bar, "SYNTHETIC_EPISODES", True)
    args = (SET, BODY, out, ["R0@10ms"], ["walk/v0.4"])
    path = rec_path(out, "R0@10ms", "walk/v0.4")
    sdir = path.with_name("v0.4.json.superseded")
    traces = bar.block_dir(out, SET, BODY, "seeds3042") / "eval" / "traces"
    bar.evaluate(*args, episodes=3, procs=1, root=t0, log=quiet)
    first, first_bytes = read(path), path.read_bytes()
    real = bar.bar_sources()
    # -- a re-run that is killed at its second episode, after the registered record was moved aside
    monkeypatch.setattr(bar, "bar_sources", lambda: dict(real, sha256="a" * 64))
    state.update(fail=("line",), text=b"second", kill=3043)
    with pytest.raises(KeyboardInterrupt):
        bar.evaluate(*args, episodes=3, procs=1, root=t0, log=quiet, supersede=True, supersede_reason="a reading was fixed")
    k1 = first["key_sha256"][:12]
    assert not path.exists() and (sdir / f"{k1}.json").read_bytes() == first_bytes
    reason = read(sdir / f"{k1}.reason.json")
    assert reason["reason"] == "a reading was fixed" and reason["label"] == "re-run after a result" and reason["sequence"] == 1
    assert reason["traces"] == dict(moved_to=f"traces.superseded/{k1}", files=3)
    # the registered record's traces moved with it: the kept record's hashes still name files that exist
    moved = traces.with_name("traces.superseded") / k1
    for e in first["episodes"]:
        assert s2io.sha256_file(moved / e["trace"]["path"]) == e["trace"]["sha256"]
    # the same command again (with or without --supersede): the label is carried into the record
    state.update(kill=None)
    doc = bar.evaluate(*args, episodes=3, procs=1, root=t0, log=quiet)
    second = read(path)
    assert second["supersedes"] == reason and second["counts"]["succeeded"] == 0 and first["counts"]["succeeded"] == 3
    assert doc["sessions"][-1]["superseded"] == [] and doc["sessions"][-1]["reused_partial"] == ["R0@10ms walk/v0.4: seeds [3042]"]
    assert doc["grid"]["walk/v0.4"]["R0@10ms"]["supersedes"]["reason"] == "a reading was fixed"
    assert "[re-run after a result: a reading was fixed]" in bar.summary_text(doc)
    assert s2io.sha256_file(traces / second["episodes"][0]["trace"]["path"]) == second["episodes"][0]["trace"]["sha256"]
    assert second["episodes"][0]["trace"]["sha256"] != first["episodes"][0]["trace"]["sha256"]
    # -- a second re-run, one of whose episodes fails to run: registered stays the first record, replaces is the second
    monkeypatch.setattr(bar, "bar_sources", lambda: dict(real, sha256="b" * 64))
    state.update(fail=(), text=b"third", error=3044)
    bar.evaluate(*args, episodes=3, procs=1, root=t0, log=quiet, supersede=True, supersede_reason="another fix")
    third = read(path)
    k2 = second["key_sha256"][:12]
    sup = third["supersedes"]
    assert third["counts"]["errors"] == 1 and sup["reason"] == "another fix" and sup["sequence"] == 2
    assert sup["registered"]["key_sha256"] == first["key_sha256"] and sup["registered"]["counts"] == first["counts"]
    assert sup["replaces"]["key_sha256"] == second["key_sha256"] and sup["replaces"]["counts"] == second["counts"]
    assert sup["key_sha256"] == second["key_sha256"] and sup["record"] == f"{k2}.json"
    assert sorted(q.name for q in sdir.iterdir()) == sorted([f"{k1}.json", f"{k1}.reason.json", f"{k2}.json",
                                                             f"{k2}.reason.json"])
    # the next run of the same key runs the failed seed again and keeps the label
    state.update(error=None)
    bar.evaluate(*args, episodes=3, procs=1, root=t0, log=quiet)
    again = read(path)
    assert again["counts"]["errors"] == 0 and again["supersedes"] == sup and again["key_sha256"] == third["key_sha256"]
    assert [x["seeds_run"] for x in again["sessions"]] == [[3042, 3043, 3044], [3044]]
    assert bar.carried_supersedes(path) == sup and bar.carried_supersedes(rec_path(out, "R0@10ms", "stop/two_step")) is None
    # -- a record file that cannot be read (a merge conflict, a hand edit) stops the run; --supersede keeps it
    garbled = b"<<<<<<< HEAD\n" + path.read_bytes()
    path.write_bytes(garbled)
    with pytest.raises(bar.BarError, match="cannot be read"):
        bar.evaluate(*args, episodes=3, procs=1, root=t0, log=quiet)
    assert path.read_bytes() == garbled
    bar.evaluate(*args, episodes=3, procs=1, root=t0, log=quiet, supersede=True, supersede_reason="the file was garbled")
    last = read(path)
    assert (sdir / "unreadable-3.json").read_bytes() == garbled and last["supersedes"]["replaces"]["unreadable"] is True
    assert last["supersedes"]["registered"]["key_sha256"] == first["key_sha256"] and last["counts"]["errors"] == 0


@needs_run
def test_a_replication_record_that_is_run_again_keeps_what_it_replicates(t0, tmp_path, monkeypatch):
    out = tmp_path / "out"
    args = (SET, BODY, out, ["R2/42@10ms"], ["walk/v0.4"])
    faked(monkeypatch, lambda row, cond, seed, relax: fake_episode(cond, seed))
    bar.evaluate(*args, procs=1, root=t0, attribute="none", log=quiet)

    def broken(row, cond, seed, relax):
        if seed == 5043:
            raise RuntimeError("the episode broke")
        return fake_episode(cond, seed)

    faked(monkeypatch, broken)
    bar.evaluate(*args, block="seeds5042", passed_of="seeds3042", procs=1, root=t0, attribute="none", log=quiet)
    path = rec_path(out, "R2/42@10ms", "walk/v0.4", block="seeds5042")
    rep = read(path)["replication_of"]
    assert read(path)["counts"]["errors"] == 1 and rep["readings"] == ["success", "but_A1"]
    # the head block's verdict says the replication is not there yet in either sense
    assert bar.replication_word(out, SET, BODY, "R2/42@10ms", "walk/v0.4", "success") == "replication incomplete"
    calls = faked(monkeypatch, lambda row, cond, seed, relax: fake_episode(cond, seed))
    bar.evaluate(*args, block="seeds5042", procs=1, root=t0, attribute="none", log=quiet)  # no --passed-of this time
    again = read(path)
    assert [c[2] for c in calls] == [5043] and again["counts"]["errors"] == 0 and again["replication_of"] == rep
    assert bar.replication_word(out, SET, BODY, "R2/42@10ms", "walk/v0.4", "success") == "replicated"


@needs_run
def test_the_workers_inputs_are_compared_with_the_sessions(t0, tmp_path, monkeypatch):
    """The workers read the set themselves; what they read must be what the records' keys name."""
    closed = []
    d = setio.set_dir(SET, BODY, t0)
    real_inputs, _ = setio.read_inputs(d, ["refs/nominal.npz", "refs/v0.4.npz", "transitions/nominal_to_v0.4.npz",
                                           "transitions/start.npz"])

    class StalePool:
        def __init__(self, *a, **k):
            self.inputs = copy.deepcopy(real_inputs)
            self.inputs["files"]["refs/v0.4.npz"]["sha256"] = "0" * 64  # the set changed after the runner's check
            self.incidents = []

        def close(self):
            closed.append(1)

        def map(self, jobs, failed=None):
            raise AssertionError("no episode may run")

    monkeypatch.setattr(bar.tune, "Pool", StalePool)
    with pytest.raises(tune.InputError, match="other inputs than the session's"):
        bar.evaluate(SET, BODY, tmp_path / "out", ["R0@10ms"], ["walk/v0.4"], episodes=1, procs=1, root=t0, log=quiet)
    assert closed == [1] and not list((tmp_path / "out").rglob("*.json"))

    class OtherFreeze(StalePool):
        def __init__(self, *a, **k):
            self.inputs = dict(copy.deepcopy(real_inputs), freeze=dict(index=7, manifest_sha256="0" * 64))
            self.incidents = []

    monkeypatch.setattr(bar.tune, "Pool", OtherFreeze)
    with pytest.raises(tune.InputError, match="freeze"):
        bar.evaluate(SET, BODY, tmp_path / "out", ["R0@10ms"], ["walk/v0.4"], episodes=1, procs=1, root=t0, log=quiet)


def _children(pid: int) -> list[int]:
    out = []
    for q in Path("/proc").iterdir():
        if q.name.isdigit():
            try:
                stat = (q / "stat").read_text()
            except OSError:
                continue
            fields = stat[stat.rindex(")") + 2:].split()
            if int(fields[1]) == pid and fields[0] != "Z":
                out.append(int(q.name))
    return out


def _gone(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return True
    return stat[stat.rindex(")") + 2:].split()[0] == "Z"


@needs_run
@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="needs /proc")
def test_a_killed_runners_workers_end_with_it(tmp_path):
    """A SIGKILL of the runner leaves no worker that still writes into the run directory (the lock is stale the
    moment the runner dies, and a resume takes it over)."""
    env = dict(os.environ, PYTHONPATH=str(ROOT), MUJOCO_GL="disable")
    out = tmp_path / "out"
    cmd = [sys.executable, "-m", "designlab.stage2.bar", "run", "--set", "S1", "--stage1-run", str(RUN), "--body", BODY,
           "--rows", "R0@4ms", "--conditions", "walk/nominal", "--episodes", "8", "--procs", "2", "--out", str(out)]
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    partial = bar.block_dir(out, "S1", BODY, "seeds3042") / "partial"
    t0_ = time.time()
    while time.time() - t0_ < 120 and proc.poll() is None and len(list(partial.rglob("*.json"))) < 1:
        time.sleep(0.05)
    assert proc.poll() is None, "the run finished before it could be killed"
    kids = _children(proc.pid)
    assert len(kids) >= 2  # the two workers (and multiprocessing's resource tracker)
    proc.send_signal(signal.SIGKILL)
    proc.wait()
    files = lambda: sorted((str(q), q.stat().st_size) for q in out.rglob("*") if q.is_file())  # noqa: E731
    t1 = time.time()
    while time.time() - t1 < 10 and not all(_gone(k) for k in kids):
        time.sleep(0.02)
    assert all(_gone(k) for k in kids), [k for k in kids if not _gone(k)]
    assert time.time() - t1 < 1.0  # at once: no worker finishes its episode first (an episode here takes seconds)
    before = files()
    assert time.time() - t1 < 1.2  # (the snapshot is taken before an orphan could have finished an episode)
    time.sleep(3.5)
    assert files() == before  # nothing was written after the runner died
    again = subprocess.run(cmd[:-6] + ["--episodes", "2", "--procs", "2", "--out", str(tmp_path / "other")], cwd=ROOT,
                           env=env, capture_output=True, text=True)
    assert again.returncode == 0, again.stderr[-2000:]  # workers end with a runner that finishes, too: nothing hangs


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
    # the rates are measured on Stage 1's own committed condition, under the row type's 0 ms row, after a warm-up
    # episode that is not counted, with every episode's times and the machine's load kept (here cut short: a test)
    m = bar.measure_rates(BODY, ["R0@10ms", "R0@4ms"], RUN, seconds=2.0, seeds=(3042, 3043))
    r0 = m["rates"]["R0"]
    assert set(m["rates"]) == {"R0"} and r0["row"] == "R0@0ms" and r0["simulated_s"] == pytest.approx(4.0, abs=0.01)
    assert 0.0 < r0["wall_per_sim_s"] < 5.0 and m["measured_on"]["set"] == "S1" and m["measured_on"]["delay_ms"] == 0
    assert m["measured_on"]["condition"] == "walk/brisk" and m["schema"] == "bar-cost-1" and m["measured_on"]["cut_s"] == 2.0
    assert [e["seed"] for e in r0["episodes"]] == [3042, 3043] and r0["warm_up"]["counted"] is False
    assert r0["wall_s"] == pytest.approx(sum(e["wall_s"] for e in r0["episodes"])) and r0["full_length"] is True
    assert r0["wall_per_sim_s"] == pytest.approx(r0["wall_s"] / r0["simulated_s"])
    assert len(m["load"]["before"]["load_average"]) == 3 and "after" in m["load"] and m["load"]["cpu_count"] >= 1
    assert bar.COST_SEEDS == (3042, 3043, 3044)
    text = bar.dry_run(SET, BODY, out, ["R0@10ms"], ["walk/v0.4"], root=t0, rates=m, log=quiet)["text"]
    assert "after a warm-up episode that is not counted" in " ".join(text.split()) and "upper estimate" in text
    assert "cut to 2.0 s" in text and bar.read_cost_text(text)["rates"]["R0"]["row"] == "R0@0ms"


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
    # the judged reading (the planned swing from the own lift-off), the reading as first written and Stage 1's
    assert e["toe_clearance_own_duration_m"] < e["toe_clearance_own_swing_m"] <= e["toe_clearance_reference_clock_m"] + 0.003
    assert e["toe_clearance_own_swing_H"] == pytest.approx(e["toe_clearance_own_swing_m"] / e["H_m"])
    assert e["own_duration_ok"] is False and e["own_swing_ok"] is True and e["reference_clock_ok"] is True
    assert e["worst_swing"]["own_duration_s"] > e["worst_swing"]["planned_s"] and "as first written" in lines[0]
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
    # tier (1) is "with attribution": its attribution steps stand directly after its episode steps, set by set, and
    # before tier (2) (STAGE2_75_PLAN.md, confirmation 5 under H6); no attribution starts inside an episode unit
    assert t3[:16] == ["T3.1-F25F9cl-10ms", "T3.1-F25F9-10ms", "T3.1-V0-10ms", "T3.1-F25F9cl-10ms-attribution",
                       "T3.1-F25F9-10ms-attribution", "T3.1-V0-10ms-attribution", "T3.2-F25F9cl-10ms-seeds5042",
                       "T3.3-F25F9cl-0ms", "T3.3-F25F9-0ms", "T3.3-V0-0ms", "T3.3-F25F9cl-omega_s1-10ms",
                       "T3.3-F25F9cl-omega_s1-0ms", "T3.4-F25F9cl-4ms", "T3.4-F25F9-4ms", "T3.4-V0-4ms", "T3.4-F25F9cl-6ms"]
    attr = [s for s in steps if s.endswith("-attribution")]
    assert attr == [f"T3.1-{x}-10ms-attribution" for x in ("F25F9cl", "F25F9", "V0")] + [
        f"T4.1-{x}-10ms-attribution" for x in ("F25F9cl", "V0")]
    for a in attr:  # the unit's own command again, now with its attribution, under the unit's name
        unit_step = a.removesuffix("-attribution")
        assert cmds[a] == cmds[unit_step].replace("--attribute none", "--attribute judged") != cmds[unit_step]
        assert steps.index(unit_step) < steps.index(a) < steps.index(f"T{a[1]}.2-F25F9cl-10ms-seeds5042")
        assert all(steps.index(e) < steps.index(a) for e in steps if e.startswith(f"T{a[1]}.1-") and e not in attr)
    judged = [s for s in steps if s.startswith("T") and "--attribute judged" in cmds[s]]
    assert judged == attr  # an episode unit never attributes: its attribution is the step after the tier's episodes
    tiers = [s.split("-")[0] for s in steps if s.startswith("T")]
    assert tiers == sorted(tiers, key=lambda x: (int(x[1]), int(x.split(".")[1])))  # tier order, task 3 then task 4
    assert [s for s in steps if s.startswith("T3.5")] == [f"T3.5-{s}-{d}ms" for d in (2, 8) for s in ("F25F9cl", "F25F9", "V0")]
    assert [s for s in steps if s.startswith("T3.6")] == [f"T3.6-{s}-{d}ms" for d in (10, 0) for s in ("F25F9cl", "V0")]
    t4 = [s for s in steps if s.startswith("T4.")]
    assert t4 == ["T4.1-F25F9cl-10ms", "T4.1-V0-10ms", "T4.1-F25F9cl-10ms-attribution", "T4.1-V0-10ms-attribution",
                  "T4.2-F25F9cl-10ms-seeds5042", "T4.3-F25F9cl-0ms", "T4.3-V0-0ms",
                  "T4.4-F25F9cl-4ms", "T4.4-V0-4ms", "T4.5-F25F9cl-6ms", "T4.5-V0-6ms", "T4.6-F25F9cl-10ms", "T4.6-V0-10ms",
                  "T4.7-F25F9cl-0ms-seeds5042", "T4.7-F25F9cl-4ms-seeds5042", "T4.7-F25F9cl-6ms-seeds5042"]
    # every unit names its task, its unit and the committed cost file; each tier's rows, conditions and attribution
    for s in steps:
        if s.startswith("T"):
            task = s[1]
            assert f"--task {task} " in cmds[s] and "--cost stage2/runs/2026-11-15-75cm/COST.md" in cmds[s]
            assert "--out stage2/runs/2026-11-15-75cm" in cmds[s] and "--body 75-M-H-10" in cmds[s]
            unit = cmds[s].split("--unit ")[1].split()[0]
            if s.startswith(f"T{task}.10"):
                assert unit.startswith(f"T{task}.")
            else:
                assert unit == s.removesuffix("-attribution")
    c = cmds["T3.1-F25F9cl-10ms"]
    assert "--rows R2/42@10ms,R2/43@10ms,R0@10ms --conditions @walk" in c and c.endswith("--attribute none")
    assert "--block seeds5042 --passed-of seeds3042" in cmds["T3.2-F25F9cl-10ms-seeds5042"]
    assert "R0@" not in cmds["T3.2-F25F9cl-10ms-seeds5042"]  # an R0 row is not replicated
    assert "--control omega_s1" in cmds["T3.3-F25F9cl-omega_s1-0ms"] and "--rows R2/42@0ms --conditions walk/v0.4 " in cmds["T3.3-F25F9cl-omega_s1-0ms"]
    assert "--conditions @speed-ladder" in cmds["T3.6-V0-0ms"] and "R0@" not in cmds["T3.6-V0-0ms"]
    assert "--set F25F9 " in cmds["T3.7-F25F9-0ms"] and "--conditions walk/v0.5 " in cmds["T3.7-F25F9-0ms"]
    assert "walk/v0.45_mu0.54,walk/v0.5_mu0.54" in cmds["T3.7-F25F9cl-0ms"]
    assert all(cmds[s].endswith("--attribute none") for s in t3 if s not in attr)
    assert "--conditions @rest" in cmds["T4.1-V0-10ms"] and cmds["T4.1-V0-10ms"].endswith("--attribute none")
    assert "--conditions @ladders" in cmds["T4.6-F25F9cl-10ms"] and "R0@" not in cmds["T4.6-F25F9cl-10ms"]
    assert all("--set F25F9 " not in cmds[s] for s in t4)  # F25F9 is a control for the walk only
    assert all(cmds[s].endswith("--attribute none") for s in t4 if s not in attr)
    # tier 10: the remaining records' attribution, in the tiers' order, under their own units; no control, no tier 1
    t10 = [s for s in steps if s.startswith("T3.10")]
    assert all(cmds[s].endswith("--attribute all") for s in t10) and not any("omega" in s for s in t10)
    assert [cmds[s].split("--unit ")[1].split()[0] for s in t10] == [s for s in t3 if not s.startswith("T3.1-")
                                                                  and "omega" not in s]
    text = CHAIN.read_text()
    assert "pgrep -f 'run_75cm.sh all'" in text and "done/ALL" in text and "tools/lowprio.sh" in text
    assert "done/STOPPED ] ||" in text  # the watchdog does not restart a chain that stopped on incomplete steps
    # with no argument it runs nothing
    assert subprocess.run(["bash", str(CHAIN)], capture_output=True).returncode == 2


def test_a_step_of_the_chain_is_skipped_once_done_and_a_cut_goes_on(tmp_path):
    """The chain's ``step`` on a copy of the script with stub commands: markers, exit codes in the logs, a cut, and
    exit 4 (episodes failed to run): the command once more, then an ``.incomplete`` marker and on with the chain."""
    root = tmp_path / "study"
    run = root / "stage2" / "runs" / "2026-11-15-75cm"
    run.mkdir(parents=True)
    (root / "tools").mkdir()
    (root / "tools" / "lowprio.sh").write_text('#!/bin/bash\nexec "$@"\n')
    text = CHAIN.read_text()
    head = text.split("r2() {")[0]
    steps = ('step a sh -c "echo A >> ran; exit 0"\nstep b sh -c "echo B >> ran; exit 3"\n'
             'step x sh -c "echo X >> ran; [ -e fixed ] || exit 4"\nstep c sh -c "echo C >> ran; exit 0"\n'
             'step y sh -c "echo Y >> ran; [ -e yy ] && exit 0; touch yy; exit 4"\n')
    (run / "run_75cm.sh").write_text(head + steps + 'step d sh -c "echo D >> ran; exit 7"\nstep e sh -c "echo E >> ran"\n')
    (run / "done").mkdir()
    (run / "logs").mkdir()
    for _ in (1, 2):
        r = subprocess.run(["bash", str(run / "run_75cm.sh"), "all"], capture_output=True, text=True)
        assert r.returncode == 7
    # done and cut steps are not run again; a step whose episodes failed to run is run twice in a chain run, the
    # chain goes on after it, and the next start runs it again
    assert (root / "ran").read_text().split() == ["A", "B", "X", "X", "C", "Y", "Y", "D", "X", "X", "D"]
    assert sorted(q.name for q in (run / "done").iterdir()) == ["a", "b.cut", "c", "x.incomplete", "y"]
    assert "Z exit 0\n" in (run / "logs" / "a.log").read_text() and "Z exit 3\n" in (run / "logs" / "b.log").read_text()
    assert (run / "logs" / "d.log").read_text().count("Z exit 7\n") == 2
    assert (run / "logs" / "x.log").read_text().count("Z exit 4\n") == 4
    ylog = (run / "logs" / "y.log").read_text()
    assert ylog.count("Z exit 4\n") == 1 and ylog.count("Z exit 0\n") == 1
    log = (run / "logs" / "run_75cm.log").read_text()
    assert "FAILED d (exit 7)" in log and log.count("INCOMPLETE x (episodes failed to run, twice)") == 2
    assert log.count("again x (episodes failed to run)") == 2 and log.count("again y (episodes failed to run)") == 1
    # the chain's end: done/ALL only when no step is incomplete; else done/STOPPED and exit 4
    (run / "run_75cm.sh").write_text(head + steps + "finish\n")
    r = subprocess.run(["bash", str(run / "run_75cm.sh"), "all"], capture_output=True, text=True)
    assert r.returncode == 4 and (run / "done" / "STOPPED").exists() and not (run / "done" / "ALL").exists()
    assert "NOT all done: incomplete steps (episodes failed to run): x" in (run / "logs" / "run_75cm.log").read_text()
    (root / "fixed").write_text("")
    r = subprocess.run(["bash", str(run / "run_75cm.sh"), "all"], capture_output=True, text=True)
    assert r.returncode == 0 and (run / "done" / "ALL").exists() and (run / "done" / "x").exists()
    assert not (run / "done" / "x.incomplete").exists()
    r = subprocess.run(["bash", str(run / "run_75cm.sh"), "list"], capture_output=True, text=True)
    assert "x [done]" in r.stdout and "b [cut]" in r.stdout
    # `all` itself refuses to start while done/STOPPED stands
    assert 'if [ -z "$LIST" ] && [ -e "$RUN/done/STOPPED" ]' in text and "[ -n \"$LIST\" ] || finish" in text


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
