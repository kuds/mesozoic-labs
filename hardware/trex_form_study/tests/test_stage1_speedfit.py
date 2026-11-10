"""Tests for the speed round's controller pieces (designlab/stage1/speedfit.py; STAGE1B_PLAN.md sections 3 and 4): the
speed mix and its servo delay, the fit's checkpoint and resume, the selection (speed G5) and the validation rows.

Run from hardware/trex_form_study (the fit's venv):  python -m pytest tests/test_stage1_speedfit.py -q
Every test is fast: the fit and the selection on a stub pool, real episodes cut to a few simulated seconds, and one
recorded G5 episode (a walk that fell at 4.9 s).  The tests that run episodes need the Stage 1 run directory's
references and transitions (git-ignored; ``python -m designlab.stage1 prepare``) and skip without them.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import designlab  # noqa: E402,F401  (selects the GL backend before mujoco renders)
from designlab.stage1 import control, env, ledger, panels, plant, speedfit, tune  # noqa: E402
from designlab.stage1.__main__ import main as cli  # noqa: E402

BODY = "60-N-T-10"
RUN = ROOT / "stage1" / "runs" / "2026-11-08"
needs_run = pytest.mark.skipif(not (RUN / BODY / "refs" / "nominal.npz").exists(),
                               reason="the run directory's references are not generated (designlab.stage1 prepare)")
COND = {c.key: c for c in panels.conditions()}


# ------------------------------------------------------------------------------------------------
# The mix and the delay draw
# ------------------------------------------------------------------------------------------------
def test_the_speed_mix_and_its_terms():
    """Stage 1's three episodes first (their slots and directions as tune's), then the brisk and Fr 0.05 walks, judged
    as the panels judge walk/brisk and walk/fr05: cadence, stride and peak GRF dropped, no start or stop terms."""
    assert speedfit.EPISODES == ("walk", "walking_turn", "in_place_turn", "brisk", "fr05")
    assert speedfit.EPISODES[:3] == tune.EPISODES and speedfit.OBJECTIVE["mix"]["slots"]["brisk"] == 3
    for seed in (1000, 1001, 2000):
        d = speedfit.directions(seed)
        assert {k: d[k] for k in tune.EPISODES} == tune.directions(seed) and d["brisk"] is d["fr05"] is None
    for kind in speedfit.SPEED_WALKS:
        cond = speedfit.panel_condition(kind)
        assert cond.key == f"walk/{kind}" and cond.gated and cond.friction is None and cond.floor == "hard"
        assert speedfit.SUCCESS[kind] == panels.terms_of(cond)  # the panels' success terms (WALK_RULE)
        assert not {"cadence", "stride", "grf"} & set(speedfit.LIMITS[kind]) and "speed" in speedfit.LIMITS[kind]
        assert speedfit.TRACKING[kind] == tune.TRACKING["walk"]
        assert set(panels.files_of(cond)) <= set(speedfit.mix_files())
    assert {k: speedfit.LIMITS[k] for k in tune.EPISODES} == tune.LIMITS
    assert {k: speedfit.SUCCESS[k] for k in tune.EPISODES} == tune.SUCCESS
    assert set(tune.mix_files()) < set(speedfit.mix_files())
    assert {"refs/brisk.npz", "refs/fr05.npz", "transitions/nominal_to_brisk.npz",
            "transitions/brisk_to_fr05.npz"} == set(speedfit.mix_files()) - set(tune.mix_files())
    # the term vector names every term once; the speed walks carry none of the start and stop terms
    assert len(set(speedfit.TERM_ORDER)) == len(speedfit.TERM_ORDER)
    assert not [t for t in speedfit.TERM_ORDER if t.split(".")[0] in speedfit.SPEED_WALKS
                and t.split(".")[1] in ("start", "stop")]
    assert [t for t in speedfit.TERM_ORDER if t.split(".")[0] in tune.EPISODES] == list(tune.TERM_ORDER)
    # the worst case: every episode a fall at its start
    bad = speedfit.failed_evaluation(1000, "boom")
    assert bad["J"] == bad["S"] + bad["P"] == 2 * len(speedfit.EPISODES) + 1 and bad["falls"] == 5
    assert bad["episodes"]["brisk"]["start_terms"] == {} and set(bad["episodes"]["fr05"]["success"]) == set(
        speedfit.SUCCESS["fr05"])
    assert bad["terms"] == [1.0] * len(speedfit.TERM_ORDER) and "J = 11" in speedfit.FAILURE_RULE
    # the records carry the rules as strings, and the objective is new, on stage1-tune-v2
    assert speedfit.OBJECTIVE["name"] == "stage1-tune-speed-v1" and speedfit.OBJECTIVE["base"] == "stage1-tune-v2"
    assert all(isinstance(v, str) and v for v in speedfit.RULES.values())
    json.dumps(speedfit.OBJECTIVE)
    assert speedfit.CMA_SETTINGS["popsize"] == 12 and speedfit.BUDGET == 360 and speedfit.BUDGET % 12 == 0
    assert speedfit.CMA_SETTINGS["sigma0"] == speedfit.SIGMA0 == tune.SIGMA0 == 0.1  # Stage 1's (plan section 3)
    assert speedfit.CMA_SETTINGS["train_episodes"] == [1000, 2000]
    assert speedfit.CMA_SETTINGS["validation_episodes"] == [2000, 2020] and speedfit.SEEDS == (42,)


def test_each_episode_draws_its_servo_delay_from_its_seed():
    """LATENCY_RULE: 0-5 physics steps, uniform, from the episode seed and the episode's slot (purpose 3, apart from
    the start and sensor-noise seeds), the same for every member of a generation."""
    for seed in (1000, 1500, 2019):
        lat = speedfit.latencies(seed)
        assert list(lat) == list(speedfit.EPISODES) and lat == speedfit.latencies(seed)
        for j, kind in enumerate(speedfit.EPISODES):
            want = int(np.random.default_rng(tune.derived_seed(seed, j, 3)).integers(0, 6))
            assert lat[kind] == speedfit.latency_steps(seed, j) == want
    draws = [speedfit.latency_steps(s, j) for s in range(*tune.TRAIN_EPISODES) for j in range(5)]
    c = Counter(draws)
    assert set(c) == set(range(6)) and all(abs(n / len(draws) - 1 / 6) < 0.02 for n in c.values())
    # the five episodes draw independently: not one delay per evaluation
    assert sum(len(set(speedfit.latencies(s).values())) > 1 for s in range(1000, 1100)) > 90
    # the delay's seed is not the start's or the noise's
    assert tune.derived_seed(1000, 0, 3) not in {tune.derived_seed(1000, j, p) for j in range(5) for p in (0, 1, 2)}
    assert "0-10 ms" in speedfit.LATENCY_RULE and speedfit.LATENCY_STEPS == (0, 5)


# ------------------------------------------------------------------------------------------------
# Real episodes (cut to a few simulated seconds)
# ------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cut_ev():
    if not (RUN / BODY / "refs" / "nominal.npz").exists():
        pytest.skip("the run directory's references are not generated")
    return speedfit.Evaluator(BODY, RUN / BODY, cut_s=2.0)


@pytest.fixture(scope="module")
def start():
    return speedfit.start_record(BODY)


@needs_run
def test_a_tiny_real_evaluation(cut_ev, start):
    """One evaluation of the five episodes cut to 2 s: each episode's delay is its draw, the speed walks are the
    panels' plans and are judged over their last 10 s without start and stop terms, the score is tune.score's, and a
    second evaluation repeats the first exactly."""
    g = speedfit.gains_vector(start["gains"])
    r = cut_ev.evaluate(g, 1000)
    assert "error" not in r and set(r["episodes"]) == set(speedfit.EPISODES)
    assert r["latency_steps"] == speedfit.latencies(1000) and r["directions"] == speedfit.directions(1000)
    assert r["simulated_s"] == pytest.approx(10.0) and r["cpu_s"] > 0
    for kind, e in r["episodes"].items():
        assert e["latency_steps"] == r["latency_steps"][kind] and e["seconds"] == pytest.approx(2.0)
        assert set(e["success"]) == set(speedfit.SUCCESS[kind]) and set(e["limits"]) == set(speedfit.LIMITS[kind])
        assert set(e["tracking"]) == set(speedfit.TRACKING[kind])
    for kind in speedfit.SPEED_WALKS:  # cut short of their planned length: no window, every term 1 (SPEED_WALK_RULE)
        e = r["episodes"][kind]
        assert e["kind"] == kind and e["start_terms"] == {} == e["stop_terms"] and e["start"] is None
        assert e["judged_window_s"] is None and not e["window"]["complete"] and not e["fell"]
        assert set(e["tracking"].values()) == set(e["limits"].values()) == {1.0} and not e["succeeded"]
        assert e["success"] == {n: n == "survived" for n in speedfit.SUCCESS[kind]}
    assert r["episodes"]["walk"]["start_terms"] and not r["episodes"]["walk"]["window"]["complete"]  # cut: no stop
    sc = tune.score(r["episodes"])
    assert r["J"] == sc["J"] and set(r["groups"]) == {*speedfit.EPISODES, "start", "stop"}
    assert len(r["terms"]) == len(speedfit.TERM_ORDER)
    again = cut_ev.evaluate(g, 1000)
    assert again["J"] == r["J"] and again["terms"] == r["terms"]
    assert [e["fall_time_s"] for e in again["episodes"].values()] == [e["fall_time_s"] for e in r["episodes"].values()]
    # the speed walks are built as the panels build walk/brisk and walk/fr05 (their whole planned lengths)
    _, docs = tune.read_inputs(RUN / BODY, panels.panel_files())
    ctx = panels.Context(BODY, RUN / BODY, docs)
    for kind in speedfit.SPEED_WALKS:
        b = ctx.built(speedfit.panel_condition(kind))
        s = cut_ev.specs[(kind, None)]
        assert [tuple(x) for x in s.plan] == [tuple(x) for x in b.plan] and s.seconds == b.seconds == b.spec.seconds
        assert s.speed_target == b.spec.speed_target and s.period_s == b.spec.period_s
        assert 15.0 < s.seconds < 30.0
    mix = cut_ev.describe()
    assert mix["cut_s"] == 2.0 and mix["simulated_s_per_evaluation"] == pytest.approx(10.0)
    assert 95.0 < sum(cut_ev.specs[(k, d)].seconds for k, d in speedfit.directions(0).items()) < 105.0  # the full mix


@needs_run
def test_a_speed_walk_is_judged_as_the_panels_judge_it(cut_ev, start, monkeypatch):
    """In slot 0 and without delay, the speed mix's brisk walk is the panels' walk/brisk episode: on an episode that runs
    its planned length (here a spec planned at the 2 s the panels' cut condition runs), the same success terms and
    window measures over its last 10 s (panels.judge's window, clipped to the episode)."""
    g = speedfit.gains_vector(start["gains"])
    _, docs = tune.read_inputs(RUN / BODY, panels.panel_files())
    ctx = panels.Context(BODY, RUN / BODY, docs)
    row = panels.TrackerRow("R2/start", "R2", g, dict(source="the speed fit's start"))
    for kind in speedfit.SPEED_WALKS:
        monkeypatch.setitem(cut_ev.specs, (kind, None), dataclasses.replace(cut_ev.specs[(kind, None)], seconds=2.0))
        mine = cut_ev.episode(kind, None, g, 3042, 0, latency=0)
        theirs = row.run(ctx, dataclasses.replace(speedfit.panel_condition(kind), seconds=2.0), 3042)
        ta, tb = mine["judged_window_s"]  # the last 10 s up to the last feedback update
        assert tb - ta == pytest.approx(10.0) and 1.98 < tb < 2.0 and mine["window"]["complete"]
        assert mine["success"] == theirs["success"] and mine["succeeded"] == theirs["succeeded"]
        assert mine["fell"] == theirs["fell"] and mine["fall_time_s"] == theirs["fall_time_s"]
        assert {k: mine["window"].get(k) for k in panels.WINDOW_KEYS} == theirs["measures"]["window"]


# ------------------------------------------------------------------------------------------------
# The fit: checkpoint and resume (stub pool)
# ------------------------------------------------------------------------------------------------
STUB_INPUTS = dict(dir="stub", rule=tune.INPUTS_RULE, files={"refs/nominal.npz": "0" * 64},
                   records={"refs.json": "1" * 64, "transitions.json": "2" * 64})


def _stub_eval(gains, seed):
    x = tune.to_unit(gains)
    J = float(np.sum((x - 0.3) ** 2)) + 1e-3 * (int(seed) % 7)
    lat = speedfit.latencies(seed)
    eps = {k: dict(direction=None, latency_steps=lat[k], fell=False, fall_time_s=None, fall_cause=None,
                   survived_frac=1.0, succeeded=True, success={"survived": True}, window={}) for k in speedfit.EPISODES}
    return dict(J=J, S=0.0, P=J, T=J, L=0.0, groups={}, falls=0, episodes=eps, terms=[J], simulated_s=1.0, wall_s=0.0,
                cpu_s=0.0, episode_seed=int(seed), latency_steps=lat)


class _StubPool:
    """``tune.Pool`` on an analytic objective: the checkpoints, the log and the selection without simulation."""

    inputs = STUB_INPUTS

    def __init__(self, body, src, workers=1, *args, **kw):
        self.src, self.workers, self.incidents = src, workers, []
        self.inputs = json.loads(json.dumps(type(self).inputs))
        self.factory = kw.get("factory")

    def describe(self):
        return {"stub": str(self.src)}

    def map(self, jobs, failed=None):
        return [_stub_eval(g, s) for g, s in jobs]

    def close(self):
        pass


def _log(path):
    return [{k: v for k, v in json.loads(line).items() if k not in ("wall_s", "cpu_s")}
            for line in path.read_text().splitlines()]


def test_the_fit_checkpoints_every_generation_and_resumes_exactly(tmp_path, monkeypatch):
    """A fit stopped after a generation (and one that died with evaluations logged past its checkpoint) resumes to
    exactly the uninterrupted fit; it starts from the body's Stage 1 G5 selection with Stage 1's sigma0 0.1, records the
    start once, and is not restarted over, nor resumed on other sources, inputs or start."""
    monkeypatch.setattr(speedfit, "Pool", _StubPool)
    quiet = dict(log=lambda m: None)
    a = speedfit.fit(BODY, 42, evals=36, out=tmp_path / "a", **quiet)
    assert a["evals"] == 36 and len(a["history"]) == 3 and a["status"] == "partial"
    want, info = env.tuned_gains(RUN / BODY, BODY)
    assert a["start"]["candidate"] == info["candidate"] == "seed42_gen20" and a["start"]["sha256"] == info["sha256"]
    assert speedfit.gains_vector(a["start"]["gains"]).tolist() == want.tolist()
    first = tune.CMA(tune.to_unit(want), speedfit.SIGMA0, speedfit.POPSIZE, 42)
    rows = _log(speedfit.run_dir_of(tmp_path / "a", BODY, 42) / "evals.jsonl")
    assert [r["x"] for r in rows[:12]] == first.ask().tolist()  # generation 0 sampled about the start, sigma0 0.1
    assert a["settings"]["cma"]["sigma0"] == 0.1 and a["objective"]["name"] == "stage1-tune-speed-v1"
    st = json.loads(speedfit.start_path(tmp_path / "a", BODY).read_text())
    assert speedfit.start_key(st) == speedfit.start_key(a["start"]) and st["rule"] == speedfit.START_RULE
    speedfit.fit(BODY, 42, evals=12, out=tmp_path / "b", **quiet)
    d = speedfit.run_dir_of(tmp_path / "b", BODY, 42)
    with (d / "evals.jsonl").open("a") as fh:  # a generation half logged when the process died
        fh.write(json.dumps({"gen": 1, "i": 0}) + "\n{\"gen\": 1, \"i\"")
    b = speedfit.fit(BODY, 42, evals=36, resume=True, out=tmp_path / "b", **quiet)
    assert _log(d / "evals.jsonl") == rows
    assert b["state"] == a["state"] and [h["mean"] for h in b["history"]] == [h["mean"] for h in a["history"]]
    assert len(b["sessions"]) == 2 and b["sessions"][1]["from_generation"] == 1
    eps = [h["episode"] for h in b["history"]]
    assert eps == [int(x) for x in tune.training_order(42)[:3]]  # Stage 1's training order
    assert [h["latency_steps"] for h in b["history"]] == [speedfit.latencies(e) for e in eps]
    assert all(r["episode"] == eps[r["gen"]] for r in _log(d / "evals.jsonl"))  # common random numbers
    assert all(r["episodes"]["fr05"]["lat"] == speedfit.latencies(r["episode"])["fr05"]
               for r in _log(d / "evals.jsonl"))
    for s in b["sessions"]:
        assert s["speed_sources"] == speedfit.speed_sources() and s["inputs"] == STUB_INPUTS
        assert s["speed_sources_unchanged_during_session"]
    assert all(h["mean_gains"][13] == 0.0 for h in b["history"])  # capture_margin held at 0.0
    with pytest.raises(SystemExit, match="exists"):
        speedfit.fit(BODY, 42, evals=48, out=tmp_path / "b", **quiet)
    before = (d / "cma.json").read_text()
    real = speedfit.speed_sources
    monkeypatch.setattr(speedfit, "speed_sources", lambda: dict(real(), sha256="0" * 64))
    with pytest.raises(SystemExit, match="sources changed"):
        speedfit.fit(BODY, 42, evals=48, resume=True, out=tmp_path / "b", **quiet)
    monkeypatch.setattr(speedfit, "speed_sources", real)
    monkeypatch.setattr(_StubPool, "inputs", dict(STUB_INPUTS, files={"refs/nominal.npz": "9" * 64}))
    with pytest.raises(SystemExit, match=r"inputs changed .*refs/nominal.npz"):
        speedfit.fit(BODY, 42, evals=48, resume=True, out=tmp_path / "b", **quiet)
    monkeypatch.setattr(_StubPool, "inputs", STUB_INPUTS)
    with pytest.raises(SystemExit, match="own Stage 1 fit"):  # START_RULE: a pick starts from its own selection
        speedfit.fit(BODY, 42, evals=48, resume=True, out=tmp_path / "b", start_from="60-N-H-10", **quiet)
    real_start = speedfit.start_record
    moved = dict(real_start(BODY), candidate="seed43_gen60", sha256="f" * 64)  # as if the Stage 1 record changed
    monkeypatch.setattr(speedfit, "start_record", lambda *a, **k: moved)
    with pytest.raises(SystemExit, match="settings differ"):  # another start
        speedfit.fit(BODY, 42, evals=48, resume=True, out=tmp_path / "b", **quiet)
    with pytest.raises(SystemExit, match="recorded start"):  # another seed may not start elsewhere either
        speedfit.fit(BODY, 43, evals=12, out=tmp_path / "b", **quiet)
    monkeypatch.setattr(speedfit, "start_record", real_start)
    assert (d / "cma.json").read_text() == before  # a refused resume changes nothing
    c = speedfit.fit(BODY, 42, evals=48, resume=True, out=tmp_path / "b", **quiet)
    assert c["evals"] == 48 and len(c["sessions"]) == 3


def test_the_start_is_a_passed_stage1_selection(tmp_path):
    """START_RULE: a body with a Stage 1 fit starts from its own G5 selection and --start-from may name only it; a
    body without one has no start; a selection that did not pass G5 is refused."""
    s = speedfit.start_record(BODY)
    assert s["start_from"] == BODY and s["status"] == "passed"
    assert s["file"] == f"stage1/runs/2026-11-08/{BODY}/tune.json"
    assert speedfit.start_record(BODY, BODY) == s and speedfit.pick_of(BODY) == BODY
    for other in ("60-N-H-10", "75-M-H-10"):
        with pytest.raises(SystemExit, match="has its own Stage 1 fit"):
            speedfit.start_record(BODY, other)
    with pytest.raises(SystemExit, match="no Stage 1 G5 record"):
        speedfit.start_record(BODY, stage1=tmp_path)
    doc = json.loads((RUN / BODY / "tune.json").read_text())
    for v in doc["seeds"].values():
        v["pass_"] = False
    (tmp_path / BODY).mkdir()
    (tmp_path / BODY / "tune.json").write_text(json.dumps(doc))
    with pytest.raises(SystemExit, match="passed G5"):
        speedfit.start_record(BODY, stage1=tmp_path)


# ------------------------------------------------------------------------------------------------
# Selection: speed G5 (synthetic results)
# ------------------------------------------------------------------------------------------------
class _ScriptedPool(_StubPool):
    """Each candidate's validation outcome scripted by its gains: mean P and {seed: (kind, share survived)} falls."""

    outcomes: dict = {}

    def map(self, jobs, failed=None):
        out = []
        for g, s in jobs:
            o = self.outcomes.get(tuple(np.asarray(g, float).tolist()), dict(P=0.5, falls={}))
            r = _stub_eval(g, s)
            kind, surv = o["falls"].get(s, (None, 1.0))
            for k, e in r["episodes"].items():
                e.update(fell=k == kind, survived_frac=surv if k == kind else 1.0, succeeded=k != kind,
                         success={"survived": k != kind, "speed": k != kind})
            S = (1.0 + (1.0 - surv)) if kind else 0.0
            r.update(J=S + o["P"], S=S, P=o["P"], T=o["P"], falls=int(kind is not None))
            out.append(r)
        return out


def test_speed_g5_selects_on_the_validation_objective(tmp_path, monkeypatch):
    """The start and the means after each third of the generations, on validation episodes 2000-2019; fewest falls,
    then the lower mean S, then the lower mean P; per kind the survive and success counts (and by delay); judged at
    the last generation of a finished fit or at --at-generation, while --partial on an unfinished fit is a smoke check
    (not judged); a fit on other sources is not judged; a rerun reuses the kept evaluations (G5_RESUME_RULE)."""
    monkeypatch.setattr(speedfit, "Pool", _StubPool)
    speedfit.fit(BODY, 42, evals=36, out=tmp_path, log=lambda m: None)
    hist = json.loads((speedfit.run_dir_of(tmp_path, BODY, 42) / "cma.json").read_text())["history"]
    st = json.loads(speedfit.start_path(tmp_path, BODY).read_text())
    gains = {"start": tuple(speedfit.gains_vector(st["gains"]).tolist())}
    gains.update({f"seed42_gen{g}": tuple(hist[g - 1]["mean_gains"]) for g in tune.candidate_generations(3)})
    script = {"start": dict(P=0.0, falls={2001: ("fr05", 0.5), 2002: ("fr05", 0.5)}),  # two falls, the lowest P
              "seed42_gen1": dict(P=0.30, falls={2004: ("brisk", 0.9)}),  # one late fall: the lower S
              "seed42_gen2": dict(P=0.10, falls={2004: ("brisk", 0.2)}),  # one early fall
              "seed42_gen3": dict(P=0.05, falls={2003: ("fr05", 0.9), 2005: ("walk", 0.9)})}  # lowest J of the fallen
    monkeypatch.setattr(_ScriptedPool, "outcomes", {gains[n]: o for n, o in script.items()})
    monkeypatch.setattr(speedfit, "Pool", _ScriptedPool)
    quiet = dict(log=lambda m: None)
    with pytest.raises(SystemExit, match="no recorded start"):
        speedfit.speed_g5("60-N-H-10", out=tmp_path, **quiet)
    unfinished = speedfit.speed_g5(BODY, out=tmp_path, **quiet)  # 36 of 360 evaluations: nothing to judge
    assert "selected" not in unfinished["seeds"]["42"] and "not done" in unfinished["seeds"]["42"]["reason"]
    doc = speedfit.speed_g5(BODY, out=tmp_path, require_done=False, **quiet)
    assert doc["validation_episodes"] == list(range(2000, 2020)) and doc["selection_rule"] == tune.SELECTION_RULE
    v = doc["seeds"]["42"]
    assert v["selected"] == "seed42_gen1" and v["judged_at_generation"] == 3
    assert v["judged"] is False and v["partial"] and "--partial" in v["reason"]  # 20 episodes, but a smoke check
    k = doc["evaluations_kept"]  # the start's 20 were kept by the call before (the unfinished fit's selection)
    assert k["added"] == 60 and k["reused"] == 20 and k["file"].endswith(speedfit.G5_LOG)
    assert [x["candidate"] for x in sorted(v["selection"], key=lambda x: x["rank"])] == [
        "seed42_gen1", "seed42_gen2", "seed42_gen3", "start"]
    assert v["selected_gains"] == control.gains_dict(gains["seed42_gen1"])
    c = v["conditions"]
    assert set(c) == set(speedfit.EPISODES) and c["brisk"]["survived"] == 19 and c["brisk"]["succeeded"] == 19
    assert c["fr05"]["survived"] == c["fr05"]["runs"] == 20 and c["fr05"]["pass_rule_met"] and not c["brisk"][
        "pass_rule_met"]
    lat = speedfit.latencies(2004)["brisk"]
    assert c["brisk"]["by_latency_steps"][str(lat)]["survived"] == c["brisk"]["by_latency_steps"][str(lat)]["runs"] - 1
    assert sum(b["runs"] for b in c["walk"]["by_latency_steps"].values()) == 20
    assert v["start_conditions"]["fr05"]["survived"] == 18
    runs = doc["candidates"]["seed42_gen3"]["runs"]
    assert len(runs) == 20 and runs[0]["latency_steps"] == speedfit.latencies(2000)
    assert runs[0]["episodes"]["brisk"]["latency_steps"] == speedfit.latencies(2000)["brisk"]
    saved = json.loads((tmp_path / BODY / "speedg5.json").read_text())
    assert saved["seeds"] == json.loads(json.dumps(doc["seeds"])) and saved["gate"] == "speed G5"
    assert saved["speed_sources"]["sha256"] == speedfit.speed_sources()["sha256"] and saved["rules"] == speedfit.RULES
    # a rerun (a restart) runs nothing that was kept, and gives the same record; a line cut short is dropped
    log_path = tmp_path / BODY / speedfit.G5_LOG
    assert len(log_path.read_text().splitlines()) == 80
    with log_path.open("a") as fh:
        fh.write('{"key": "abc", "resu')

    class _NoRun(_ScriptedPool):
        def map(self, jobs, failed=None):
            raise AssertionError(f"{len(jobs)} kept evaluations run again")

    monkeypatch.setattr(speedfit, "Pool", _NoRun)
    again = speedfit.speed_g5(BODY, out=tmp_path, require_done=False, **quiet)
    assert again["seeds"] == doc["seeds"] and again["candidates"] == doc["candidates"]
    assert again["evaluations_kept"]["reused"] == 80 and again["evaluations_kept"]["dropped_lines"] == 1
    assert len(log_path.read_text().splitlines()) == 80
    other = speedfit.G5Log(log_path, dict(again["evaluations_kept"]["context"], speed_sources="0" * 64))
    assert other.get(gains["start"], 2000) is None  # kept on other sources: not reused
    assert speedfit.G5Log(log_path, again["evaluations_kept"]["context"]).get(gains["start"], 2000) == doc[
        "candidates"]["start"]["runs"][0]
    monkeypatch.setattr(speedfit, "Pool", _ScriptedPool)
    at = speedfit.speed_g5(BODY, out=tmp_path, at_generation=2, **quiet)  # STAGE1B_PLAN section 5's budget rule
    assert at["seeds"]["42"]["judged_at_generation"] == 2 and at["runs"]["42"]["candidates"] == [
        "seed42_gen1", "seed42_gen2"]
    assert at["seeds"]["42"]["judged"] and not at["seeds"]["42"]["partial"] and at["seeds"]["42"]["reason"] is None
    beyond = speedfit.speed_g5(BODY, out=tmp_path, at_generation=4, **quiet)
    assert "selected" not in beyond["seeds"]["42"] and "generation 3 of 4" in beyond["seeds"]["42"]["reason"]
    smoke = speedfit.speed_g5(BODY, out=tmp_path, episodes=2, require_done=False, **quiet)
    assert smoke["seeds"]["42"]["judged"] is False
    speedfit.fit(BODY, 43, evals=12, out=tmp_path, cut_s=1.0, **quiet)  # a fit on cut episodes is a smoke check
    cut = speedfit.speed_g5(BODY, out=tmp_path, seeds=(42, 43), require_done=False, **quiet)
    assert "selected" in cut["seeds"]["42"] and "cut to 1 s" in cut["seeds"]["43"]["reason"]
    path = speedfit.run_dir_of(tmp_path, BODY, 42) / "cma.json"
    fit = json.loads(path.read_text())
    fit["sessions"][0]["speed_sources"]["sha256"] = "0" * 64
    path.write_text(json.dumps(fit))
    other = speedfit.speed_g5(BODY, out=tmp_path, require_done=False, **quiet)
    assert other["seeds"]["42"]["refused"] and other["seeds_refused"] == [42] and "selected" not in other["seeds"]["42"]


# ------------------------------------------------------------------------------------------------
# Servo delay and the validation rows
# ------------------------------------------------------------------------------------------------
def _spy(monkeypatch):
    """Record every servo target the controller sets (after plant.Plant.command) and every one the servos get (at
    plant.Plant.step)."""
    calls = dict(set=[], got=[])
    command, step = plant.Plant.command, plant.Plant.step

    def spy_command(self, *a, **kw):
        command(self, *a, **kw)
        calls["set"].append(self._v.ctrl.copy())

    def spy_step(self, n=1):
        calls["got"].append(self.d.ctrl.copy())
        step(self, n)

    monkeypatch.setattr(plant.Plant, "command", spy_command)
    monkeypatch.setattr(plant.Plant, "step", spy_step)
    return calls


@needs_run
def test_the_rows_delay_reaches_the_servos(monkeypatch, start):
    """R2s/<s>@10ms and R2@10ms: every servo target reaches the servos 5 physics steps after the controller sets it
    (the queue starts with the home targets); the plant is restored after the episode; the margin ledger's latency model
    gives the same episode bit for bit; without delay the row is panels.TrackerRow bit for bit."""
    _, docs = tune.read_inputs(RUN / BODY, panels.panel_files())
    ctx = panels.Context(BODY, RUN / BODY, docs)
    cond = dataclasses.replace(COND["walk/nominal"], seconds=0.2)
    g = speedfit.gains_vector(start["gains"])
    p = ctx.built(cond).p  # (built and settled before the spy)
    calls = _spy(monkeypatch)
    row = speedfit.DelayedRow("R2@10ms", g, dict(source="test"), latency_steps=5)
    e = row.run(ctx, cond, 3042)
    set_, got = calls["set"], calls["got"]
    assert len(set_) == len(got) == 100 and "command" not in p.__dict__ and "reset" not in p.__dict__
    for j in range(5):
        assert np.array_equal(got[j], p.ctrl_home)
    for j in range(5, 100):
        assert np.array_equal(got[j], set_[j - 5]), j
    assert not np.array_equal(set_[50], set_[45])  # the targets move, so the check bites
    # the queue itself: k + 1 targets, the oldest applied
    with speedfit.servo_delay(p, 5) as q:
        p.reset()
        assert q.maxlen == 6 and len(q) == 5 and all(np.array_equal(x, p.ctrl_home) for x in q)
        p.command(p.ctrl_home * 0 + 0.1, np.zeros(14), np.zeros(14))
        assert len(q) == 6 and np.array_equal(p.d.ctrl, p.ctrl_home)
    assert "command" not in p.__dict__
    # the ledger's latency margin (ledger._latency) gives the same episode
    monkeypatch.undo()
    base = panels.TrackerRow("R2@10ms", "R2", g, dict(source="test"))
    undo = ledger._latency(p, 5)
    try:
        led = base.run(ctx, cond, 3042)
    finally:
        undo()
    mine = row.run(ctx, cond, 3042)
    for k in ("fell", "fall_time_s", "worst_sustained_ratio", "worst_speed_ratio", "estimator", "ankle_cap_residual",
              "joint_margin_min_rad", "success"):
        assert mine[k] == led[k] == e[k], k
    zero = speedfit.DelayedRow("R2s/42", g, dict(source="test")).run(ctx, cond, 3042)
    plain = panels.TrackerRow("R2s/42", "R2", g, dict(source="test")).run(ctx, cond, 3042)
    assert {k: v for k, v in zero.items() if k != "wall_s"} == {k: v for k, v in plain.items() if k != "wall_s"}
    assert zero["estimator"] != mine["estimator"]  # 10 ms of delay changes the episode
    # decision 20's lifted runs (eval --attribute) keep the delay: the relaxation runs inside servo_delay
    lifted = row.run(ctx, cond, 3042, relax=("slip",))
    undo = ledger._latency(p, 5)
    try:
        led = base.run(ctx, cond, 3042, relax=("slip",))
    finally:
        undo()
    assert lifted["relaxed"] == ["slip"] and "command" not in p.__dict__
    for k in ("fell", "fall_time_s", "worst_sustained_ratio", "estimator", "success"):
        assert lifted[k] == led[k], k


def _speed_run(tmp_path, start, judged=True):
    """A speed run directory holding the body's start and a speed G5 record for fit seed 42."""
    sr = tmp_path / "speed"
    sp = speedfit.start_path(sr, BODY)
    sp.parent.mkdir(parents=True)
    sp.write_text(json.dumps(start))
    sel = dict(control.gains_dict(speedfit.gains_vector(start["gains"]) * 1.01), capture_margin=0.0)
    (sr / BODY / "speedg5.json").write_text(json.dumps(dict(start=start, seeds={"42": dict(
        selected="seed42_gen20", selected_gains=sel, judged=judged, judged_at_generation=30)})))
    return sr, sel


@needs_run
def test_the_rows_are_registered_and_read_their_gains(tmp_path, monkeypatch, start):
    """R2s/42 (no delay), R2s/42@10ms and R2@10ms through panels.tracker_rows (eval --rows): their gains, delays,
    conditions (ROW_CONDITIONS only), keys and specs; a smoke selection is refused."""
    sr, sel = _speed_run(tmp_path, start)
    monkeypatch.setenv(speedfit.SPEED_RUN_ENV, str(sr))
    rows = panels.tracker_rows(BODY, RUN / BODY, ("R2s/42", "R2s/42@10ms", "R2@10ms", "R2@0ms"))
    assert [r.latency_steps for r in rows.values()] == [0, 5, 5, 0]
    assert rows["R2@0ms"].gains.tolist() == rows["R2@10ms"].gains.tolist()
    assert rows["R2s/42"].gains.tolist() == rows["R2s/42@10ms"].gains.tolist() == speedfit.gains_vector(sel).tolist()
    assert rows["R2@10ms"].gains.tolist() == speedfit.gains_vector(start["gains"]).tolist()
    assert rows["R2@10ms"].gains_info["stage1_sha256"] == start["sha256"]
    for r in rows.values():
        assert [c.key for c in panels.conditions() if r.runs(c)] == list(speedfit.ROW_CONDITIONS)
        assert all(r.gated(COND[k]) for k in speedfit.ROW_CONDITIONS)
        d = r.describe()
        assert d["latency_ms"] == 2 * r.latency_steps and d["tracker_mode"] == "R2" and d["rate"] == tune.COMMAND_RATE
        assert list(r.sources()) == ["designlab/stage1/speedfit.py"]
        again = panels.row_from_spec(r.spec())
        assert type(again) is speedfit.DelayedRow and again.describe() == d
    # each row its own records: the description is the key's row (panels.record_key)
    assert len({json.dumps(panels._portable(r.describe()), sort_keys=True) for r in rows.values()}) == 4
    assert speedfit.ms_steps(10, "x") == 5
    for bad in ("R2s/42@3ms", "R2s/42@22ms", "R2s/x", "R2@6ms"):
        with pytest.raises(ValueError):
            panels.tracker_rows(BODY, RUN / BODY, (bad,))
    with pytest.raises(ValueError, match="no selection"):
        panels.tracker_rows(BODY, RUN / BODY, ("R2s/43",))
    smoke, _ = _speed_run(tmp_path / "smoke", start, judged=False)
    monkeypatch.setenv(speedfit.SPEED_RUN_ENV, str(smoke))
    with pytest.raises(ValueError, match="smoke check"):
        panels.tracker_rows(BODY, RUN / BODY, ("R2s/42",))


@needs_run
def test_a_row_runs_through_the_panels(tmp_path, monkeypatch, start):
    """eval's runner with the speed rows: a record per row and condition under <out>/<body>/eval/<row>/, the delay in
    its key; the dry run lists exactly the rows' seven conditions."""
    sr, _ = _speed_run(tmp_path, start)
    monkeypatch.setenv(speedfit.SPEED_RUN_ENV, str(sr))
    out = tmp_path / "out"
    assert cli(["eval", "--body", BODY, "--rows", "R2s/42,R2s/42@10ms,R2@10ms", "--behaviors", "walk,turn",
                "--row-module", "designlab.stage1.speedfit", "--episodes", "1", "--dry-run", "--no-measure",
                "--inputs", str(RUN), "--out", str(out)]) == 0
    cut = [dataclasses.replace(COND["walk/brisk"], seconds=0.4), dataclasses.replace(COND["walk/brisk_mu0.54"],
                                                                                    seconds=0.4)]
    monkeypatch.setattr(panels, "conditions", lambda: cut)
    doc = panels.evaluate(BODY, out, rows=("R2@10ms",), episodes=1, inputs=RUN, traces=False, log=lambda m: None)
    rec = json.loads(panels.record_path(out, BODY, "R2@10ms", cut[0]).read_text())
    assert rec["key"]["row"]["latency_steps"] == 5 and rec["counts"]["episodes"] == 1 and rec["gated"]
    assert rec["key"]["row_sources"] == {"designlab/stage1/speedfit.py": plant.sha256(Path(speedfit.__file__))}
    assert not panels.record_path(out, BODY, "R2@10ms", cut[1]).exists()  # not one of the rows' conditions
    assert doc["grid"]["walk"]["brisk"]["rows"]["R2@10ms"]["episodes"] == 1


def test_a_worker_refuses_other_speed_fit_sources():
    """SOURCES_RULE: a worker (a replacement included) that computes other speed-fit sources than the session's
    refuses to build its evaluator; the sources follow speedfit.py's imports as the tuner's follow tune.py's."""
    with pytest.raises(tune.PoolError, match="changed during the session"):
        speedfit._evaluator(BODY, str(RUN / BODY), "R2", tune.COMMAND_RATE, None, sources="0" * 64)
    files = tune.package_files()
    s = speedfit.speed_sources(files)
    assert s["modules"] == tune.tuner_modules(files, root="speedfit.py") and "panels.py" in s["modules"]
    assert s["sha256"] == plant.sources_sha256({m: files[m] for m in s["modules"]})
    changed = dict(files, **{"panels.py": files["panels.py"] + b"#"})
    assert speedfit.speed_sources(changed)["sha256"] != s["sha256"]
    assert speedfit.speed_sources(dict(files, **{"report.py": b"#"}))["sha256"] == s["sha256"]  # not imported


K_SCREENS = {"60-N-T-10-K": ("60-N-T-10", 0.5842493875125405, 0.584), "60-N-H-10-K": ("60-N-H-10", 0.5737071506575937,
                                                                                    0.574)}


def _fake_k_registry(monkeypatch, tmp_path=None, with_ids=True):
    """Arm K's two bodies registered as the faster-knee branch registers them (``BODIES``, ``candidate_id``), their
    case directory under ``tmp_path`` holding envelope screens named by candidate id with the real screens' values."""
    import designlab.stage1 as stage1

    case = tmp_path / "narrow_knee_60" if tmp_path is not None else None
    for body, (cid, dv, _) in K_SCREENS.items():
        monkeypatch.setitem(stage1.BODIES, body, str(case) if case else stage1.BODIES[cid])
        if case is not None:
            path = case / "review" / "screening" / f"{cid if with_ids else body}_envelope.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(dict(push_recovery=dict(max_push_dv_ms=dv))))
    if with_ids:
        monkeypatch.setattr(stage1, "candidate_id", lambda b: {k: v[0] for k, v in K_SCREENS.items()}.get(b, b),
                            raising=False)
    else:
        monkeypatch.delattr(stage1, "candidate_id", raising=False)


def test_bodies_outside_stage1s_three_have_their_push_bound_and_no_g7(tmp_path, monkeypatch):
    """The hooks a faster-knee body needs in eval: its push bound comes from its own envelope screen, named by its
    candidate id (the faster-knee registry's candidate_id; the body's name in a registry without one), as the listed
    three's came from theirs (each listed value is its screen's to 3 decimals); env's get reads only the list, before
    and after; a body with no plan figures gets 'G7 not run' (speedfit's guard on panels.g7) instead of a KeyError."""
    fresh = env._PushBounds()
    for body, v in env.PUSH_BOUND_M_S.items():
        assert fresh[body] == v
    assert env.PUSH_BOUND_M_S.get("no-such-body", 0.5) == 0.5
    with pytest.raises(KeyError):
        fresh["no-such-body"]
    _fake_k_registry(monkeypatch, tmp_path)
    for body, (_, _, want) in K_SCREENS.items():
        assert env.PUSH_BOUND_M_S.get(body) is None and env.PUSH_BOUND_M_S[body] == want
        assert env.PUSH_BOUND_M_S.get(body, 0.5) == 0.5 and body not in env.PUSH_BOUND_M_S  # (no cache in the list)
    assert dict(env.PUSH_BOUND_M_S) == {"60-N-T-10": 0.487, "60-N-H-10": 0.478, "75-M-H-10": 0.589}
    _fake_k_registry(monkeypatch, tmp_path / "by_body", with_ids=False)  # a registry without candidate ids
    assert env.PUSH_BOUND_M_S["60-N-T-10-K"] == 0.584
    monkeypatch.setattr(panels, "G7_PLAN", {k: v for k, v in panels.G7_PLAN.items() if k != BODY})
    assert panels.g7.speed_round and panels.g7.__wrapped__.__module__ == "designlab.stage1.panels"
    doc = panels.summarize(BODY, tmp_path, inputs=RUN)
    assert "no G7 figures" in doc["G7"]["error"] and "G7 not run" in panels.summary_text(doc)


@pytest.mark.skipif("60-N-T-10-K" not in speedfit.BODIES, reason="arm K's bodies are not registered yet (the merge)")
def test_the_faster_knee_bodies_push_bounds():
    """After the merge: arm K's push bounds are their screens' (0.5842 and 0.5737 m/s to 3 decimals) and the three
    listed bounds are unchanged."""
    for body, (_, _, want) in K_SCREENS.items():
        assert env.PUSH_BOUND_M_S[body] == want
    assert dict(env.PUSH_BOUND_M_S) == {"60-N-T-10": 0.487, "60-N-H-10": 0.478, "75-M-H-10": 0.589}
    assert speedfit.pick_of("60-N-T-10-K") == "60-N-T-10" and speedfit.pick_of("60-N-H-10-K") == "60-N-H-10"


def test_a_faster_knee_body_starts_from_its_pick(monkeypatch):
    """START_RULE on arm K (the faster-knee registry stood in): a body without a Stage 1 fit starts from its pick, the
    body its candidate id names, by default or named; any other start body is refused."""
    _fake_k_registry(monkeypatch)
    for body, (pick, _, _) in K_SCREENS.items():
        s = speedfit.start_record(body)
        assert s["body"] == body and s["start_from"] == pick == speedfit.pick_of(body)
        assert s["sha256"] == speedfit.start_record(pick)["sha256"]
        assert speedfit.start_record(body, pick) == s
        other = "60-N-H-10" if pick == "60-N-T-10" else "60-N-T-10"
        with pytest.raises(SystemExit, match=f"starts from the pick it is built from, {pick}"):
            speedfit.start_record(body, other)
        with pytest.raises(SystemExit, match="starts from the pick"):
            speedfit.start_record(body, body)


def test_eval_finds_a_faster_knee_bodys_models_by_candidate_id(monkeypatch):
    """eval's check for the round-two models (__main__.round_two_models) reads them by candidate id, as the plant does,
    so arm K's rows run in the main checkout without --allow-rebuilt-models."""
    from designlab.stage1.__main__ import round_two_models

    if not round_two_models(BODY):
        pytest.skip("this checkout has no round-two models")
    _fake_k_registry(monkeypatch)  # (each K body in its pick's case directory, which holds the pick's models)
    assert round_two_models("60-N-T-10-K") and round_two_models("60-N-H-10-K")
    import designlab.stage1 as stage1

    monkeypatch.delattr(stage1, "candidate_id")
    assert not round_two_models("60-N-T-10-K")  # by the body's own name there is no such file


def test_the_speed_rows_are_attributed():
    """ATTRIBUTION_RULE: the speed controllers are in panels.ATTRIBUTED, so a speed row's record with 5 or more falls
    gets decision 20's lifted runs (eval --attribute) and its falls read 'not attributed' until then, as R2's do;
    Stage 1's rows are unchanged (R2 and R3 attributed, R0, R1 and R4 not); the lifted runs keep the delay
    (test_the_rows_delay_reaches_the_servos)."""
    assert set(speedfit.SPEED_CONTROLLERS) == {"R2s", "R2@0ms", "R2@10ms"}
    assert set(panels.ATTRIBUTED) >= {"R2", "R3", *speedfit.SPEED_CONTROLLERS}
    assert not {"R0", "R1", "R4"} & set(panels.ATTRIBUTED)

    def ep(i, fell):
        return dict(seed=3042 + i, fell=fell, error=False, succeeded=not fell, strides=3.0 if fell else 30.0,
                    failed_terms=["survived"] if fell else [], fall_cause="tilt" if fell else None,
                    fall_candidates=["slip", "cop_at_pad_edge_with_ankle_saturated"] if fell else None)

    eps = [ep(i, i < 6) for i in range(20)]
    for row in ("R2s/42", "R2s/42@10ms", "R2@10ms", "R2@0ms", "R2/42"):
        rec = dict(row=row, status="done", counts=panels.counts(eps), episodes=eps, attribution=None)
        assert panels._attribution_needed(rec, True) and not panels._attribution_needed(rec, False)
        cands, lift = panels.attribution_plan(rec)
        assert cands == ["slip", "cop_at_pad_edge_with_ankle_saturated"] and lift == ["slip"]
        assert panels.failure(row, eps, None, False)["falls_label"] == "not attributed (falls; run eval --attribute)"
    rec = dict(row="R0", status="done", counts=panels.counts(eps), episodes=eps, attribution=None)
    assert not panels._attribution_needed(rec, True)
    assert "ATTRIBUTION_RULE" in speedfit.ROWS_RULE and speedfit.RULES["attribution"] == speedfit.ATTRIBUTION_RULE


def test_the_speed_run_is_the_rounds_unless_named(monkeypatch, tmp_path):
    """ROWS_RULE: the rows read their gains from STAGE1B_SPEED_RUN if set, else the round's run directory, whatever run
    directory the panels read their references from (arm S reads Stage 1's)."""
    monkeypatch.delenv(speedfit.SPEED_RUN_ENV, raising=False)
    assert speedfit.speed_run_of(RUN / BODY) == speedfit.RUN == ROOT / "stage1" / "runs" / "2026-11-10-speed"
    monkeypatch.setenv(speedfit.SPEED_RUN_ENV, str(tmp_path))
    assert speedfit.speed_run_of(RUN / BODY) == tmp_path


# ------------------------------------------------------------------------------------------------
# Stage 1 unchanged
# ------------------------------------------------------------------------------------------------
@needs_run
def test_stage1_tune_objective_and_records_are_unchanged():
    """Stage 1's objective is the one its fits recorded; this module is in neither the tuner's nor the panels'
    sources, so their records' keys do not see it; and a recorded G5 validation episode (60-N-H-10, the tuning start,
    episode 2010: its walk fell at 4.896 s) reproduces through tune's evaluator and through the speed evaluator at
    zero delay, every recorded value equal."""
    for body in ("60-N-T-10", "60-N-H-10", "75-M-H-10"):
        doc = json.loads((RUN / body / "tune.json").read_text())
        assert doc["objective"] == json.loads(json.dumps(tune.OBJECTIVE))
        assert doc["objective"]["name"] == "stage1-tune-v2"
        assert doc["selection_rule"] == tune.SELECTION_RULE and doc["success_rule"] == tune.SUCCESS_RULE
    assert "speedfit.py" not in tune.tuner_modules() and "speedfit.py" not in panels.key_modules()
    assert "speedfit.py" in speedfit.speed_sources()["modules"]
    body = "60-N-H-10"
    if not (RUN / body / "refs" / "nominal.npz").exists():
        pytest.skip("no references for 60-N-H-10")
    doc = json.loads((RUN / body / "tune.json").read_text())
    want = next(r for r in doc["candidates"]["start"]["runs"] if r["episode"] == 2010)["episodes"]["walk"]
    got = {}
    got["tune"] = tune.Evaluator(body, RUN / body).episode("walk", None, control.START_GAINS, 2010, 0)
    sev = speedfit.Evaluator(body, RUN / body)
    got["speed"] = sev.episode("walk", None, control.START_GAINS, 2010, 0, latency=0)
    assert want["fell"] and want["fall_time_s"] == pytest.approx(4.896)
    for name, e in got.items():
        rec = tune._validation_record(dict(episode_seed=2010, J=0, S=0, P=0, T=0, L=0, groups={}, falls=0, terms=[],
                                           episodes={"walk": e}, wall_s=0.0))["episodes"]["walk"]
        assert json.loads(json.dumps(rec)) == want, name
    # the speed mix draws its own delay for that episode (which changes it unless the draw is 0)
    assert got["speed"]["latency_steps"] == 0 and speedfit.latency_steps(2010, 0) in range(6)


# ------------------------------------------------------------------------------------------------
# The command line
# ------------------------------------------------------------------------------------------------
def test_speedfit_and_speedg5_command_line(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(speedfit, "Pool", _StubPool)
    base = ["--body", BODY, "--out", str(tmp_path)]
    assert cli(["speedfit", *base, "--evals", "24"]) == 0
    assert cli(["speedfit", *base, "--evals", "36", "--resume"]) == 0
    doc = json.loads((speedfit.run_dir_of(tmp_path, BODY, 42) / "cma.json").read_text())
    assert doc["evals"] == 36 and len(doc["sessions"]) == 2 and doc["seed"] == 42
    with pytest.raises(SystemExit):
        cli(["speedfit", *base, "--evals", "48"])
    assert cli(["speedg5", *base, "--partial"]) == 0
    out = capsys.readouterr().out
    assert "speed G5 60-N-T-10" in out and "selected" in out
    assert cli(["speedg5", *base, "--at-generation", "3"]) == 0
    doc["sessions"][1]["speed_sources"]["sha256"] = "0" * 64  # a resumed session on other sources
    (speedfit.run_dir_of(tmp_path, BODY, 42) / "cma.json").write_text(json.dumps(doc))
    assert cli(["speedg5", *base, "--partial"]) == 1  # refused: no silent selection


# ------------------------------------------------------------------------------------------------
# Plan section 4's reading of the rows (speedreport) and the cost record
# ------------------------------------------------------------------------------------------------
def _record(row, key, survived, succeeded, failed=("slip_worst",), candidates=("slip",), falls_label=None):
    """A synthetic panel record of ``row`` on condition ``key``: the first ``succeeded`` episodes succeed, the next ones
    up to ``survived`` stay up but fail ``failed``, the rest fall with ``candidates``."""
    eps = []
    for i in range(panels.RUNS):
        fell, ok = i >= survived, i < succeeded
        eps.append(dict(seed=panels.SEEDS[i], fell=fell, error=False, succeeded=ok, strides=3.0 if fell else 30.0,
                        failed_terms=["survived"] if fell else ([] if ok else list(failed)),
                        fall_cause="tilt" if fell else None, fall_candidates=list(candidates) if fell else None))
    c = panels.counts(eps)
    f = panels.failure(row, eps, None, c["pass_"])
    if f and falls_label:
        f["falls_label"] = falls_label
    return dict(schema=panels.SCHEMA, row=row, key_name=key, status="done", counts=c, episodes=eps, failure=f,
                attribution=None, key=dict(sources="synthetic"), controller=dict(controller=row))


def _write_records(out, body, records):
    for (row, key), rec in records.items():
        path = panels.record_path(out, body, row, COND[key])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec))


def test_the_speed_report_reads_plan_section_4(tmp_path, monkeypatch, capsys):
    """speedreport: each row and condition shown / survives only / fails / not run, the failing limits with counts and
    the falls' candidates and label; Q1 on the speed walks at 0 ms beside Stage 1's R2 row of the start's seed, Q3 at
    10 ms beside R2@10ms; a trade where R2s loses the nominal walk or a turn the baseline shows; arm K beside R2@0ms;
    it is in no source hash."""
    from designlab.stage1 import speedreport

    assert "speedreport.py" not in speedfit.speed_sources()["modules"] and "speedreport.py" not in panels.key_modules()
    out, stage1 = tmp_path / "speed", tmp_path / "stage1"
    turns = [k for k in speedfit.ROW_CONDITIONS if k.startswith("turn/")]
    start = dict(body=BODY, start_from=BODY, candidate="seed42_gen20", cma_seed=42, sha256="a" * 64, file="x")
    sp = speedfit.start_path(out, BODY)
    sp.parent.mkdir(parents=True)
    sp.write_text(json.dumps(start))
    base = {("R2/42", "walk/nominal"): _record("R2/42", "walk/nominal", 20, 18),
            ("R2/42", "walk/brisk"): _record("R2/42", "walk/brisk", 17, 17),
            ("R2/42", "walk/fr05"): _record("R2/42", "walk/fr05", 1, 0)}
    base.update({("R2/42", k): _record("R2/42", k, 20, 20) for k in turns})
    _write_records(stage1, BODY, base)
    rows = {("R2s/42", "walk/nominal"): _record("R2s/42", "walk/nominal", 20, 15),
            ("R2s/42", "walk/brisk"): _record("R2s/42", "walk/brisk", 20, 19),
            ("R2s/42", "walk/fr05"): _record("R2s/42", "walk/fr05", 20, 10, failed=("speed",)),
            ("R2s/42", turns[0]): _record("R2s/42", turns[0], 18, 18, falls_label="controller-limited")}
    rows.update({("R2s/42", k): _record("R2s/42", k, 20, 20) for k in turns[1:]})
    rows.update({("R2s/42@10ms", "walk/brisk"): _record("R2s/42@10ms", "walk/brisk", 12, 12,
                                                         falls_label="body-limited: slip"),
                 ("R2s/42@10ms", "walk/fr05"): _record("R2s/42@10ms", "walk/fr05", 0, 0),
                 ("R2@10ms", "walk/brisk"): _record("R2@10ms", "walk/brisk", 0, 0)})
    _write_records(out, BODY, rows)
    (out / BODY / "speedg5.json").write_text(json.dumps(dict(seeds_selected=[42], seeds={"42": dict(
        selected="seed42_gen20", judged=True, partial=False, judged_at_generation=30, reason=None)})))
    doc = speedreport.write(out, tmp_path / "dest", stage1)
    r = doc["reports"][BODY]
    assert doc["bodies"] == [BODY] and r["arm"] == "S" and r["rows"] == ["R2s/42", "R2s/42@10ms", "R2@10ms"]
    q1, q3 = r["answers"]["42"]["Q1"], r["answers"]["42"]["Q3"]
    assert q1["walk/brisk"]["speed"]["cls"] == "shown" and q1["walk/brisk"]["baseline"]["cls"] == "fails"
    assert q1["walk/brisk"]["baseline_row"] == "Stage 1 R2/42" and q1["walk/brisk"]["baseline"]["survived"] == 17
    fr = q1["walk/fr05"]["speed"]
    assert fr["cls"] == "survives only" and fr["failed_terms"] == {"speed": 10} and fr["falls"] == 0
    b = q3["walk/brisk"]["speed"]
    assert b["cls"] == "fails" and b["falls"] == 8 and b["fall_candidates"] == {"slip": 8} and q3["walk/brisk"]["answer"]
    assert b["falls_label"] == "body-limited: slip" and q3["walk/brisk"]["baseline"]["cls"] == "fails"
    assert q3["walk/nominal"]["speed"]["cls"] == "not run" and not q3["walk/nominal"]["answer"]
    t = r["trades"]["42"]
    assert t["walk/nominal"]["reading"] == "trade" and t[turns[0]]["reading"] == "trade"
    assert all(t[k]["reading"] == "kept" for k in turns[1:])
    assert r["trade_conditions"] == sorted(["walk/nominal", turns[0]])
    assert speedreport.compare("fails", "shown") == "gained" and speedreport.compare("fails", "fails") == "not lost"
    assert speedreport.compare("shown", "not run") == "unknown"
    md = (tmp_path / "dest" / speedreport.MARKDOWN).read_text()
    assert f"Trades at 0 ms, seed 42: walk/nominal, {turns[0]}" in md and "Q3, seed 42, walk/brisk at 10 ms" in md
    assert "falls 8 (slip 8; body-limited: slip)" in md
    assert json.loads((tmp_path / "dest" / speedreport.JSON_NAME).read_text())["reports"][BODY]["arm"] == "S"
    # arm K: the baseline at 0 ms is R2@0ms on the body; the pick's Stage 1 row is shown beside
    _fake_k_registry(monkeypatch)
    kb = "60-N-T-10-K"
    sp = speedfit.start_path(out, kb)
    sp.parent.mkdir(parents=True)
    sp.write_text(json.dumps(dict(start, body=kb)))
    _write_records(out, kb, {("R2s/42", "walk/brisk"): _record("R2s/42", "walk/brisk", 20, 20),
                             ("R2@0ms", "walk/brisk"): _record("R2@0ms", "walk/brisk", 20, 17),
                             ("R2@0ms", "walk/nominal"): _record("R2@0ms", "walk/nominal", 20, 20)})
    k = speedreport.collect(out, stage1)["reports"][kb]
    q2 = k["answers"]["42"]["Q2"]["walk/brisk"]
    assert k["arm"] == "K" and k["pick"] == BODY and q2["baseline_row"] == "R2@0ms" and q2["speed"]["cls"] == "shown"
    assert q2["baseline"]["cls"] == "shown" and q2["pick_stage1"]["cls"] == "fails"
    assert k["trades"]["42"]["walk/nominal"]["reading"] == "unknown"  # R2s/42's nominal walk not run yet
    assert cli(["speedreport", "--out", str(out), "--stage1", str(stage1), "--body", BODY]) == 0
    assert "STAGE1B_TABLES.md" in capsys.readouterr().out and (out / speedreport.MARKDOWN).exists()


@needs_run
def test_the_cost_record_projects_the_round(monkeypatch, start):
    """COST_RULE on a cut evaluation (the code path only): the episode measured is training_order(42)[0] (1865); the
    projections give the fit, speed G5, the rows, R2@0ms and the attribution as CPU hours and as wall hours with one
    idle core per job and with four jobs at once; arm K's body costs more than arm S's (R2@0ms and its attribution)."""
    monkeypatch.setattr(speedfit, "Evaluator", functools.partial(speedfit.Evaluator, cut_s=0.2))
    doc = speedfit.measure_cost(BODY, RUN / BODY, speedfit.gains_vector(start["gains"]), log=lambda m: None)
    assert doc["episode_seed"] == int(tune.training_order(42)[0]) == 1865 and "(1865)" in speedfit.COST_RULE
    p = doc["projections"]
    assert set(p) == {"fit_360", "fit_360_at_this_evaluation", "speed_g5", "rows", "r2_at_0ms", "attribution_per_row",
                      "per_body"}
    lo, hi = speedfit.PARALLEL_SLOWDOWN
    for k in ("fit_360", "speed_g5", "rows", "r2_at_0ms"):
        assert p[k]["wall_h_idle_core"] == p[k]["cpu_h"] and p[k]["wall_h_four_at_once"] == [p[k]["cpu_h"] * lo,
                                                                                               p[k]["cpu_h"] * hi]
    assert p["rows"]["episodes"] == 3 * 20 * 7 and p["r2_at_0ms"]["cpu_h"] == pytest.approx(p["rows"]["cpu_h"] / 3)
    s, k = p["per_body"]["arm_S"], p["per_body"]["arm_K"]
    assert k["cpu_h"][0] > s["cpu_h"][0] and s["cpu_h"][1] > s["cpu_h"][0] and k["rows"] == 4 and s["rows"] == 3
    json.dumps(doc)
