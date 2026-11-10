"""Tests for the speed round's controller pieces (designlab/stage1/speedfit.py; STAGE1B_PLAN.md sections 3 and 4): the
speed mix and its servo delay, the fit's checkpoint and resume, the selection (speed G5) and the validation rows.

Run from hardware/trex_form_study (the fit's venv):  python -m pytest tests/test_stage1_speedfit.py -q
Every test is fast: the fit and the selection on a stub pool, real episodes cut to a few simulated seconds, and one
recorded G5 episode (a walk that fell at 4.9 s).  The tests that run episodes need the Stage 1 run directory's
references and transitions (git-ignored; ``python -m designlab.stage1 prepare``) and skip without them.
"""

from __future__ import annotations

import dataclasses
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
    assert speedfit.CMA_SETTINGS["sigma0"] == speedfit.SIGMA0 == 0.05 < tune.SIGMA0
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
    for kind in speedfit.SPEED_WALKS:
        e = r["episodes"][kind]
        assert e["kind"] == kind and e["start_terms"] == {} == e["stop_terms"] and e["start"] is None
        ta, tb = e["judged_window_s"]  # the last 10 s up to the last feedback update (panels.judge's window)
        assert tb - ta == pytest.approx(10.0) and 1.98 < tb < 2.0 and e["window"]["complete"]
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
def test_a_speed_walk_is_judged_as_the_panels_judge_it(cut_ev, start):
    """In slot 0 and without delay, the speed mix's brisk walk is the panels' walk/brisk episode (cut alike): the same
    success terms and window measures."""
    g = speedfit.gains_vector(start["gains"])
    _, docs = tune.read_inputs(RUN / BODY, panels.panel_files())
    ctx = panels.Context(BODY, RUN / BODY, docs)
    row = panels.TrackerRow("R2/start", "R2", g, dict(source="the speed fit's start"))
    for kind in speedfit.SPEED_WALKS:
        mine = cut_ev.episode(kind, None, g, 3042, 0, latency=0)
        theirs = row.run(ctx, dataclasses.replace(speedfit.panel_condition(kind), seconds=2.0), 3042)
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
    exactly the uninterrupted fit; it starts from the body's Stage 1 G5 selection with sigma0 0.05, records the start
    once, and is not restarted over, nor resumed on other sources, inputs or start."""
    monkeypatch.setattr(speedfit, "Pool", _StubPool)
    quiet = dict(log=lambda m: None)
    a = speedfit.fit(BODY, 42, evals=36, out=tmp_path / "a", **quiet)
    assert a["evals"] == 36 and len(a["history"]) == 3 and a["status"] == "partial"
    want, info = env.tuned_gains(RUN / BODY, BODY)
    assert a["start"]["candidate"] == info["candidate"] == "seed42_gen20" and a["start"]["sha256"] == info["sha256"]
    assert speedfit.gains_vector(a["start"]["gains"]).tolist() == want.tolist()
    first = tune.CMA(tune.to_unit(want), speedfit.SIGMA0, speedfit.POPSIZE, 42)
    rows = _log(speedfit.run_dir_of(tmp_path / "a", BODY, 42) / "evals.jsonl")
    assert [r["x"] for r in rows[:12]] == first.ask().tolist()  # generation 0 sampled about the start, sigma0 0.05
    assert a["settings"]["cma"]["sigma0"] == 0.05 and a["objective"]["name"] == "stage1-tune-speed-v1"
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
    with pytest.raises(SystemExit, match="settings differ"):  # another start
        speedfit.fit(BODY, 42, evals=48, resume=True, out=tmp_path / "b", start_from="60-N-H-10", **quiet)
    with pytest.raises(SystemExit, match="recorded start"):  # another seed may not start elsewhere either
        speedfit.fit(BODY, 43, evals=12, out=tmp_path / "b", start_from="60-N-H-10", **quiet)
    assert (d / "cma.json").read_text() == before  # a refused resume changes nothing
    c = speedfit.fit(BODY, 42, evals=48, resume=True, out=tmp_path / "b", **quiet)
    assert c["evals"] == 48 and len(c["sessions"]) == 3


def test_the_start_is_a_passed_stage1_selection(tmp_path):
    """START_RULE: the body's own Stage 1 G5 selection, or a named body's; a body with no Stage 1 fit is told to name
    its pick; a selection that did not pass G5 is refused."""
    s = speedfit.start_record(BODY)
    assert s["start_from"] == BODY and s["status"] == "passed"
    assert s["file"] == f"stage1/runs/2026-11-08/{BODY}/tune.json"
    other = speedfit.start_record(BODY, "60-N-H-10")
    assert other["start_from"] == "60-N-H-10" and other["candidate"] == "seed43_gen60" and other["body"] == BODY
    with pytest.raises(SystemExit, match="--start-from"):
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
    the last generation of a finished fit or at --at-generation; a fit on other sources is not judged."""
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
    assert v["selected"] == "seed42_gen1" and v["judged"] and v["judged_at_generation"] == 3
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
    at = speedfit.speed_g5(BODY, out=tmp_path, at_generation=2, **quiet)  # STAGE1B_PLAN section 5's budget rule
    assert at["seeds"]["42"]["judged_at_generation"] == 2 and at["runs"]["42"]["candidates"] == [
        "seed42_gen1", "seed42_gen2"]
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


def test_bodies_outside_stage1s_three_have_their_push_bound_and_no_g7(tmp_path, monkeypatch):
    """The hooks a faster-knee body needs in eval: its push bound comes from its own envelope screen, as the listed
    three's came from theirs (each listed value is its screen's to 3 decimals); env's get reads only the list; a body
    with no plan figures gets 'G7 not run' instead of a KeyError."""
    fresh = env._PushBounds()
    for body, v in env.PUSH_BOUND_M_S.items():
        assert fresh[body] == v
    assert env.PUSH_BOUND_M_S.get("no-such-body", 0.5) == 0.5
    with pytest.raises(KeyError):
        fresh["no-such-body"]
    monkeypatch.setattr(panels, "G7_PLAN", {k: v for k, v in panels.G7_PLAN.items() if k != BODY})
    doc = panels.summarize(BODY, tmp_path, inputs=RUN)
    assert "no G7 figures" in doc["G7"]["error"] and "G7 not run" in panels.summary_text(doc)


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
