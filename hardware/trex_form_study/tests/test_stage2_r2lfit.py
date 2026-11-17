"""Tests for R2L's fit, selection, records and rows (``designlab/stage2/r2lfit.py``; round C design D21, sections 3.8
and 4; Track B rules Q1-Q6).

Fast: the rotation table per (kind, direction); the fit, its resume, its budget blocks, the selection's cells and
the rows, all on a stub evaluator that scores from a formula of the gains (no physics); the objective's item terms
on synthetic bar records; the conditions on the committed JSON of set V0 (no array is read).  Every output goes to
``tmp_path``.  Slow (``STAGE2_SLOW=1``, needs the restored set arrays and models): slot 0's episode against the bar's
R2 row bit for bit on set V0 (two development episodes).
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from designlab.stage1 import panels, plant, speedfit, tune  # noqa: E402
from designlab.stage2 import bar, lookbar, r2lfit, s2io, setio  # noqa: E402

BODY = "75-M-H-10"
SLOW = os.environ.get("STAGE2_SLOW") == "1"
ONE_THREAD = all(os.environ.get(v) == "1" for v in s2io.THREAD_VARS)
V0 = setio.set_dir("V0", BODY)
GATES2 = "designlab.stage2.gates2"  # the gates package (work package CG)


def quiet(*a, **k):
    pass


class Killed(Exception):
    """What the stub raises to stand for a killed process."""


class Stub:
    """A ``tune.Pool`` evaluator that scores from a formula: every term grows with the squared distance of the
    candidate (in the [0, 1] scaling) from a target, and with the delay; ``cost(spec)`` is the evaluation's CPU
    seconds; ``kill_after`` calls raise ``Killed``.  ``calls`` logs (gains, spec) in the order evaluated."""

    def __init__(self, cost=None, kill_after=None, watch=None):
        self.inputs = dict(dir="stub", files={"refs/v0.4.npz": dict(sha256="a" * 64, entry_sha256="b" * 64)},
                           freeze=dict(index=0, manifest_sha256="c" * 64))
        self.cost = cost or (lambda spec: 1.0)
        self.kill_after = kill_after
        self.watch = watch  # a path that must exist at the first call (rotation.json)
        self.calls = []
        self.watched = None
        x0 = tune.to_unit(speedfit.gains_vector(speedfit.start_record(BODY)["gains"]))
        self.target = np.clip(x0 + 0.08, 0.0, 1.0)

    def describe(self):
        return {"stub": True}

    def evaluate(self, gains, spec):
        if self.watch is not None and self.watched is None:
            self.watched = Path(self.watch).exists()
        if self.kill_after is not None and len(self.calls) >= self.kill_after:
            raise Killed()
        self.calls.append((np.asarray(gains, float).copy(), spec))
        d = float(np.sum((tune.to_unit(gains) - self.target) ** 2))
        eps = {}
        for s in spec["slots"]:
            e = r2lfit.blank_episode(s)
            v = float(min(1.0, d * (1.0 + 0.1 * s["k"])))
            e["tracking"] = {n: v for n in e["tracking"]}
            e["limits"] = {n: 0.5 * v for n in e["limits"]}
            e["seconds"] = 20.0
            eps[s["kind"]] = e
        return r2lfit.scored(spec, eps, 0.0, float(self.cost(spec)))


def kind_of(spec) -> str:
    """"fit", "decisive" or "reported" for an evaluation's spec."""
    if spec["seed"] < 2000:
        return "fit"
    return "decisive" if all(s["k"] == r2lfit.DECISIVE_K for s in spec["slots"]) else "reported"


def costs(fit=1.0, decisive=1.0, reported=1.0):
    return lambda spec: dict(fit=fit, decisive=decisive, reported=reported)[kind_of(spec)]


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Outputs, the gates and the index under ``tmp_path``; the evaluator a stub (``env.use(stub)``)."""
    monkeypatch.setattr(r2lfit, "GATES", tmp_path / "gates")
    monkeypatch.setattr(r2lfit, "INDEX", tmp_path / "r2l" / "index.json")
    monkeypatch.setattr(panels, "ROW_FACTORIES", dict(panels.ROW_FACTORIES))
    ns = types.SimpleNamespace(out=tmp_path / "work", tmp=tmp_path, stub=None, mp=monkeypatch, gates2=None)

    def use(stub):
        ns.stub = stub
        monkeypatch.setattr(r2lfit, "_evaluator", lambda *a, **k: stub)
        return stub

    ns.use = use
    return ns


class GateError(RuntimeError):
    pass


class Refused(GateError):
    pass


def gates2_stub():
    """The gates package as r2lfit calls it (``gates2.require_g9_ref(body, set_name, path=, root=)``, its
    ``GateError`` and ``Refused``), on a record in its form (``pass_``, its set and body); ``calls`` logs the calls."""
    calls = []

    def require_g9_ref(body, set_name, path=None, root=None):
        calls.append(dict(body=body, set=set_name, path=Path(path), root=root))
        doc = s2io.read_json(path) if Path(path).is_file() else None
        if not isinstance(doc, dict):
            raise Refused("there is no G9-ref record")
        if (doc.get("body"), doc.get("set")) != (body, set_name):
            raise Refused(f"the G9-ref record is of {doc.get('set')}/{doc.get('body')}")
        if doc.get("pass_") is not True:
            raise Refused("G9-ref did not pass")
        return dict(path=s2io.portable(str(Path(path).resolve())), sha256=s2io.sha256_file(path), pass_=True,
                    at=doc.get("at"), set_freeze=doc.get("set_freeze"))

    return types.SimpleNamespace(GateError=GateError, Refused=Refused, require_g9_ref=require_g9_ref, calls=calls)


def g9_record(set_name="V6noF5", body=BODY, pass_=True, **extra):
    """A G9-ref record in the gates package's form (schema, set, body, ``pass_``, the freeze, its own sha256)."""
    doc = dict(schema="s2-gate-g9-ref-1", gate="G9-ref", body=body, set=set_name, pass_=pass_,
               at="2026-11-17T00:00:00.000000Z", set_freeze=dict(index=1, manifest_sha256="d" * 64), **extra)
    doc["sha256"] = s2io.sha256_json(doc)
    return doc


def passed_gate(env, set_name="V6noF5", body=BODY, passed=True, doc=None):
    """A G9-ref record of ``set_name`` and ``body`` (``doc``, or one in the gates package's form that passed or
    not) and the gates package's stub."""
    s2io.write_json(env.tmp / "gates" / set_name / body / "g9_ref.json",
                    doc if doc is not None else g9_record(set_name, body, pass_=passed))
    if env.gates2 is None:
        env.gates2 = gates2_stub()
        env.mp.setattr(r2lfit, "_gates2", lambda: env.gates2)
    return env.gates2


def read_lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


# ------------------------------------------------------------------------------------------------
# The rotation (rule Q2)
# ------------------------------------------------------------------------------------------------
def test_rotation_follows_rule_q2():
    doc = r2lfit.rotation(42)
    t = doc["table"]
    order = tune.training_order(42)
    assert len(t) == 30 and doc["block"] == 6
    for g, row in enumerate(t):
        b = g // 6
        assert row["block"] == b and row["episode"] == int(order[g])
        ks = [s["k"] for s in row["slots"]]
        assert ks == [(g + j) % 6 for j in range(5)] and len(set(ks)) == 5  # five different delays per generation
        assert [s["condition"] for s in row["slots"]] == [
            "walk/v0.4", f"turn/q90_{('left', 'right')[b % 2]}", "stop/two_step",
            f"nudge/{('forward', 'left', 'backward', 'right')[b % 4]}", "walk/nominal"]
        for s in row["slots"]:
            assert s["start_seed"] == tune.derived_seed(row["episode"], s["j"], 0)
            assert s["noise_seed"] == tune.derived_seed(row["episode"], s["j"], 1)
        spec = r2lfit.spec_of(row)
        assert spec["seed"] == row["episode"] and [s["k"] for s in spec["slots"]] == ks


def test_rotation_every_kind_and_direction_meets_every_delay():
    t = r2lfit.rotation(42)["table"]
    for b in range(5):  # within every block every slot meets all six delays
        rows = t[6 * b : 6 * b + 6]
        for j in range(5):
            assert sorted(r["slots"][j]["k"] for r in rows) == list(range(6))
    for b in range(4):  # left and right turns: every delay, 10 ms included, in any two consecutive blocks
        cov = r2lfit.coverage(t[6 * b : 6 * b + 12])
        for side in r2lfit.SIDES:
            assert cov[f"turn/{side}"]["met"] == list(range(6))
    for b in range(2):  # the four nudge directions in any four consecutive blocks
        cov = r2lfit.coverage(t[6 * b : 6 * b + 24])
        for d in r2lfit.NUDGE_DIRECTIONS:
            assert cov[f"nudge/{d}"]["met"] == list(range(6))
    one = r2lfit.coverage(t, 6)  # a fit cut after one block names what it did not meet
    assert one["turn/right"]["not_met"] == list(range(6)) and one["nudge/left"]["not_met"] == list(range(6))
    assert one["turn/left"]["not_met"] == [] and one["walk"]["not_met"] == []


def test_validation_spec_directions():
    for s, side, d in ((2000, "left", "forward"), (2001, "right", "left"), (2002, "left", "backward"),
                       (2003, "right", "right")):
        spec = r2lfit.validation_spec(s, 5)
        assert spec["seed"] == s and {x["k"] for x in spec["slots"]} == {5}
        assert spec["slots"][1]["condition"] == f"turn/q90_{side}" and spec["slots"][3]["condition"] == f"nudge/{d}"


def test_rotation_cli_prints_the_table_and_writes_nothing(tmp_path, capsys):
    assert r2lfit.main(["rotation", "--body", BODY, "--set", "V6noF5", "--seed", "42", "--out", str(tmp_path)]) == 0
    text = capsys.readouterr().out
    assert sum(1 for x in text.splitlines() if x.startswith("g")) == 30
    assert "walk/v0.4@0ms" in text and list(tmp_path.iterdir()) == []


def test_conditions_are_the_bars_gated_ones_at_mu_0_9():
    docs = setio.load(V0)  # committed JSON only
    known = {c.key: c for c in bar.conditions(docs, bar.HEADLINE_BLOCK, every=True)}
    for key in r2lfit.CONDITIONS:
        c = known[key]
        assert c.friction is None and c.floor == "hard"
        assert c.gated is (key != "walk/nominal")
        if c.judge == "bar_nudge":
            assert c.frac == 0.25


def test_constants_match_the_rules_and_the_modules():
    assert (r2lfit.POPSIZE, r2lfit.SIGMA0) == (tune.POPSIZE, tune.SIGMA0) == (12, 0.1)
    assert r2lfit.LOOK_SETS == lookbar.LOOK_SETS
    assert r2lfit.share_core_h(r2lfit.FIT_CORE_H, r2lfit.REPORTED_CORE_H) == r2lfit.SHARE_CORE_H == 4.0
    assert json.loads((ROOT / "stage2" / "tb" / "r2l" / "index.json").read_text()) == {
        "schema": "s2-r2l-index-1", "fits": []}
    assert r2lfit.read_index(ROOT / "stage2" / "tb" / "r2l" / "index.json")["fits"] == []


# ------------------------------------------------------------------------------------------------
# The objective (rule Q3)
# ------------------------------------------------------------------------------------------------
H = 1.0  # so the own-swing toe limit is 0.04 m


def walk_rec(**bar_kw):
    b = dict(along_commanded_heading_m=3.0, along_target_m=4.0, lateral_from_line_max_m=0.125,
             toe=dict(toe_clearance_own_swing_m=0.03, interrupted_swings=0, count_ok=True))
    b.update(bar_kw)
    return dict(fell=False, measures=dict(bar=b))


def walk_measure():
    return dict(tracking=dict(dcm=0.1, speed=0.2, heading=0.3), limits={n: 0.0 for n in tune.LIMITS["walk"]},
                dcm_error_rms_m=0.002)


def test_walk_terms():
    tr, lim, rd = r2lfit.episode_terms("walk", walk_rec(), walk_measure(), H)
    assert tr == dict(dcm=0.1, speed=0.2, heading=0.3)
    assert list(lim) == list(r2lfit.LIMITS["walk"])
    assert lim["along"] == pytest.approx(0.25) and lim["line"] == pytest.approx(0.5)
    assert lim["toe_clearance_own_swing"] == pytest.approx(0.25)
    assert not set(panels.NOMINAL_ONLY_TERMS) & set(lim) and "toe_clearance" not in lim
    assert rd["along_m"] == 3.0
    _, lim2, _ = r2lfit.episode_terms("nominal", walk_rec(toe=dict(toe_clearance_own_swing_m=0.05,
                                                                       interrupted_swings=1, count_ok=True)),
                                      walk_measure(), H)
    assert lim2["toe_clearance_own_swing"] == 1.0  # the own-swing term failed on an interrupted swing


def test_turn_stop_nudge_terms():
    e = dict(fell=False, measures=dict(bar=dict(heading_error_at_deg=-10.0, heading_error_hold_max_deg=30.0,
                                                joint_range=dict(joint_shortfall_rad=0.05),
                                                toe=dict(toe_clearance_own_swing_m=0.05, interrupted_swings=0,
                                                         count_ok=True))))
    m = dict(tracking=dict(dcm=1.0, speed=1.0, heading=1.0), limits={n: 0.0 for n in tune.LIMITS["walk"]},
             dcm_error_rms_m=0.01)
    tr, lim, _ = r2lfit.episode_terms("turn", e, m, H)
    assert tr == dict(dcm=0.5)  # the straight walk's speed and heading are not the turn's
    assert lim["heading_at"] == 0.5 and lim["heading_hold"] == 1.0 and lim["joint_range"] == pytest.approx(0.5)
    assert lim["toe_clearance_own_swing"] == 0.0 and "speed" not in lim
    stop = dict(fell=False, worst_speed_ratio=["x", 0.9], joint_shortfall_rad=-0.1,
                measures=dict(bar=dict(touchdowns_after_command=3, forward_stopped_strides=0.75,
                                       drives=dict(episode=dict(worst=["x", 1.5])))))
    tr, lim, _ = r2lfit.episode_terms("stop", stop, dict(dcm_error_rms_m=0.0), H)
    assert tr == dict(dcm=0.0)
    assert lim == dict(drive_sustained=0.5, drive_speed=0.0, joint_range=0.0, touchdowns=0.5, stop_time=0.5)
    stop["measures"]["bar"].update(touchdowns_after_command=2, forward_stopped_strides=None)
    _, lim, _ = r2lfit.episode_terms("stop", stop, dict(dcm_error_rms_m=0.0), H)
    assert lim["touchdowns"] == 0.0 and lim["stop_time"] == 1.0  # never stopped
    nudge = dict(fell=False, worst_speed_ratio=["x", 1.0], joint_shortfall_rad=0.0,
                 measures=dict(bar=dict(drives=dict(episode=dict(worst=["x", 0.5]))),
                               recovery=dict(recovered=True, after_s=1.5)))
    _, lim, _ = r2lfit.episode_terms("nudge", nudge, dict(dcm_error_rms_m=0.0), H)
    assert lim == dict(drive_sustained=0.0, drive_speed=0.0, joint_range=0.0, recovery=0.5)
    nudge["measures"]["recovery"] = dict(recovered=False, after_s=None)
    assert r2lfit.episode_terms("nudge", nudge, dict(dcm_error_rms_m=0.0), H)[1]["recovery"] == 1.0


def test_a_fall_scores_one_and_the_worst_case():
    for kind in r2lfit.KINDS:
        tr, lim, _ = r2lfit.episode_terms(kind, dict(fell=True), None, H)
        assert set(tr.values()) == {1.0} and set(lim.values()) == {1.0}
    spec = r2lfit.validation_spec(2000, 5)
    assert r2lfit.failed_evaluation(spec, "boom")["J"] == r2lfit.WORST_J == 11.0
    ok = r2lfit.scored(spec, {s["kind"]: r2lfit.blank_episode(s) for s in spec["slots"]})
    assert ok["J"] == 0.0 and ok["falls"] == 0 and len(ok["terms"]) == len(r2lfit.TERM_ORDER)
    assert set(ok["groups"]) == set(r2lfit.KINDS)


# ------------------------------------------------------------------------------------------------
# The fit
# ------------------------------------------------------------------------------------------------
def test_fit_writes_the_rotation_first_and_gives_every_member_the_same_episodes(env):
    out = env.out
    d = r2lfit.fit_dir(out, "F25F9cl", BODY, 42)
    stub = env.use(Stub(costs(fit=1.0), watch=d / "rotation.json"))
    doc = r2lfit.fit(BODY, "F25F9cl", out=out, log=quiet)
    assert stub.watched is True  # rotation.json was there at the first evaluation
    rot = s2io.read_json(d / "rotation.json")
    assert rot["written_at"] < doc["sessions"][0]["started_at"]
    assert rot["table"] == r2lfit.rotation(42)["table"]
    assert doc["status"] == "done" and len(doc["history"]) == 30 and doc["settings"]["role"]["role"] == "fit"
    assert doc["rotation_met"]["not_met"] == {}
    table = r2lfit.rotation(42)["table"]
    for g in range(30):
        specs = [s for _, s in stub.calls[12 * g : 12 * g + 12]]
        assert all(s == r2lfit.spec_of(table[g]) for s in specs)  # every member, the same five episodes
    lines = read_lines(d / "evals.jsonl")
    assert len(lines) == 360
    for x in lines:
        slots = table[x["gen"]]["slots"]
        assert [(e["cond"], e["k"]) for e in x["episodes"].values()] == [(s["condition"], s["k"]) for s in slots]
    for h, row in zip(doc["history"], table, strict=True):
        assert [(s["condition"], s["k"], s["direction"]) for s in h["slots"]] == [
            (s["condition"], s["k"], s["direction"]) for s in row["slots"]]
    led = s2io.read_json(d / "ledger.json")
    assert led["fit_cpu_s"] == pytest.approx(360.0) and len(led["blocks"]) == 5
    assert [x["start"] for x in led["block_decisions"]] == [True] * 4
    assert (out / "F25F9cl" / BODY / "start.json").exists()


def test_fit_cut_by_its_budget_ends_at_a_whole_block(env):
    for cost, gens, excess in ((50.0, 12, 0.0), (100.0, 6, 0.0), (150.0, 6, 150.0 * 72 - 8640.0)):
        out = env.out / f"c{int(cost)}"
        env.use(Stub(costs(fit=cost)))
        doc = r2lfit.fit(BODY, "F25F9cl", out=out, log=quiet)
        assert len(doc["history"]) == gens and gens % 6 == 0 and doc["status"] == "ended_by_budget"
        assert doc["ended"]["decision"]["start"] is False
        led = s2io.read_json(r2lfit.fit_dir(out, "F25F9cl", BODY, 42) / "ledger.json")
        assert led["fit_cpu_s"] == pytest.approx(cost * 12 * gens)
        assert led["fit_excess_s"] == pytest.approx(excess)  # a started block finishes; its excess is recorded
        assert [b["complete"] for b in led["blocks"]] == [True] * (gens // 6)
        if gens == 6:
            assert doc["rotation_met"]["not_met"]["turn/right"] == list(range(6))


def test_a_killed_fit_resumes_at_its_next_generation_with_the_same_state(env):
    a_out, b_out = env.out / "a", env.out / "b"
    env.use(Stub(costs(fit=1.0)))
    a = r2lfit.fit(BODY, "F25F9cl", out=a_out, log=quiet)
    env.use(Stub(costs(fit=1.0), kill_after=12 * 3 + 5))  # killed in generation 3
    with pytest.raises(Killed):
        r2lfit.fit(BODY, "F25F9cl", out=b_out, log=quiet)
    d = r2lfit.fit_dir(b_out, "F25F9cl", BODY, 42)
    mid = s2io.read_json(d / "cma.json")
    assert mid["status"] == "running" and len(mid["history"]) == 3 and mid["state"]["generation"] == 3
    with open(d / "evals.jsonl", "a") as fh:  # an evaluation logged after the checkpoint, as a kill can leave one
        fh.write(json.dumps(dict(gen=3, i=0)) + "\n")
    with pytest.raises(r2lfit.R2LError, match="exists"):
        r2lfit.fit(BODY, "F25F9cl", out=b_out, log=quiet)  # never restarted over
    env.use(Stub(costs(fit=1.0)))
    b = r2lfit.fit(BODY, "F25F9cl", out=b_out, resume=True, log=quiet)
    assert b["sessions"][1]["from_generation"] == 3 and len(env.stub.calls) == 12 * 27
    assert b["state"] == a["state"]
    strip = [{k: v for k, v in h.items() if k != "wall_s"} for h in a["history"]]
    assert [{k: v for k, v in h.items() if k != "wall_s"} for h in b["history"]] == strip
    assert (d / "evals.jsonl").read_text() == (r2lfit.fit_dir(a_out, "F25F9cl", BODY, 42) / "evals.jsonl").read_text()


def test_a_resume_refuses_another_rotation_table(env):
    out = env.out
    env.use(Stub(kill_after=12 * 2))
    with pytest.raises(Killed):
        r2lfit.fit(BODY, "F25F9cl", out=out, log=quiet)
    p = r2lfit.fit_dir(out, "F25F9cl", BODY, 42) / "rotation.json"
    good = s2io.read_json(p)
    bad = json.loads(json.dumps(good))
    bad["table"][7]["slots"][1]["k"] = (bad["table"][7]["slots"][1]["k"] + 1) % 6
    s2io.write_json(p, bad)
    env.use(Stub())
    with pytest.raises(r2lfit.R2LError, match="another table"):
        r2lfit.fit(BODY, "F25F9cl", out=out, resume=True, log=quiet)
    s2io.write_json(p, good)
    assert r2lfit.fit(BODY, "F25F9cl", out=out, resume=True, log=quiet)["status"] == "done"


def test_a_look_set_waits_for_g9_ref_as_the_gates_package_reads_it(env):
    env.use(Stub())
    path = env.tmp / "gates" / "V6noF5" / BODY / "g9_ref.json"
    # the gates package is not here: no fit on a look set starts
    assert r2lfit.GATES2 == GATES2
    env.mp.setitem(sys.modules, GATES2, None)
    assert r2lfit._gates2() is None
    with pytest.raises(r2lfit.R2LError, match="gates package"):
        r2lfit.fit(BODY, "V6noF5", out=env.out, log=quiet)
    env.mp.delitem(sys.modules, GATES2)
    g2 = passed_gate(env, passed=False)
    with pytest.raises(r2lfit.R2LError, match="G9-ref has not passed.*did not pass"):
        r2lfit.fit(BODY, "V6noF5", out=env.out, log=quiet)
    path.unlink()
    with pytest.raises(r2lfit.R2LError, match="no G9-ref record"):
        r2lfit.fit(BODY, "V6noF5", out=env.out, log=quiet)
    # a record that says "passed" in another form than the gates package's is not a pass
    passed_gate(env, doc=dict(passed=True, status="passed", outcome="passed", body=BODY, set="V6noF5"))
    with pytest.raises(r2lfit.R2LError, match="did not pass"):
        r2lfit.fit(BODY, "V6noF5", out=env.out, log=quiet)
    passed_gate(env, set_name="V6noF5", doc=g9_record("V6", BODY))
    with pytest.raises(r2lfit.R2LError, match="of V6/"):
        r2lfit.fit(BODY, "V6noF5", out=env.out, log=quiet)
    assert not (env.out / "V6noF5").exists()  # nothing was written by a refused fit
    # the gates package's record that passed: the fit starts and cites it
    passed_gate(env)
    doc = r2lfit.fit(BODY, "V6noF5", out=env.out, log=quiet)
    gate = doc["sessions"][0]["gate"]
    assert gate["sha256"] == s2io.sha256_file(path) and gate["pass_"] is True
    assert gate["at"] == s2io.read_json(path)["at"] and gate["set_freeze"] == s2io.read_json(path)["set_freeze"]
    assert gate["checked_by"] == "designlab.stage2.gates2.require_g9_ref" and gate["modified_at"]
    assert g2.calls[-1] == dict(body=BODY, set="V6noF5", path=path, root=r2lfit.SET_ROOT)
    assert len(g2.calls) == 5
    # not on a set that is no look set
    assert r2lfit.gate("F25F9cl", BODY) is None and len(g2.calls) == 5


@pytest.mark.skipif(importlib.util.find_spec(GATES2) is None,
                    reason="the gates package (work package CG) is not on this branch")
def test_the_gate_calls_the_real_gates_package(env):
    """With the real gates package: its signature is the one r2lfit calls, and a set without a record is refused."""
    with pytest.raises(r2lfit.R2LError, match="G9-ref has not passed as the gates package reads it"):
        r2lfit.gate("V6noF5", BODY)


def test_a_development_fit_is_not_selected_or_published(env):
    env.use(Stub())
    doc = r2lfit.fit(BODY, "V0", out=env.out, max_generations=2, popsize=4, log=quiet)
    assert doc["settings"]["role"]["role"] == "development" and len(doc["history"]) == 2
    assert len(env.stub.calls) == 8 and doc["status"] == "done"
    with pytest.raises(r2lfit.R2LError, match="development"):
        r2lfit.select(BODY, "V0", out=env.out, log=quiet)


# ------------------------------------------------------------------------------------------------
# The selection (rules Q4, Q5)
# ------------------------------------------------------------------------------------------------
def test_the_decisive_cell_runs_first_and_whole_past_its_reserve(env):
    env.use(Stub(costs(fit=150.0, decisive=100.0, reported=1.0)))
    r2lfit.fit(BODY, "F25F9cl", out=env.out, log=quiet)  # block 0 alone: 10,800 s
    env.stub.calls.clear()
    sel = r2lfit.select(BODY, "F25F9cl", out=env.out, log=quiet)
    assert sel["judged_at_generation"] == 6 and [c for c in sel["candidates"]] == ["start", "gen2", "gen4", "gen6"]
    assert len(env.stub.calls) == 80 and all(kind_of(s) == "decisive" for _, s in env.stub.calls)
    assert all(len(sel["decisive"]["runs"][n]) == 20 for n in sel["candidates"])
    assert sel["decisive"]["cpu_s"] == pytest.approx(8000.0)
    assert sel["overrun"]["over_reserve_s"] == pytest.approx(8000.0 - 4320.0)
    assert sel["overrun"]["over_share_s"] == pytest.approx(10800.0 + 8000.0 - 14400.0)
    assert [c["status"] for c in sel["by_delay"]] == ["not run: budget"] * 10
    assert [(c["k"], c["candidate"]) for c in sel["by_delay"]] == [
        (k, n) for k in (0, 2, 4, 1, 3) for n in (sel["selected"], "start")]
    ranked = sorted(sel["selection"], key=lambda x: x["rank"])
    assert ranked[0]["candidate"] == sel["selected"]
    led = s2io.read_json(r2lfit.fit_dir(env.out, "F25F9cl", BODY, 42) / "ledger.json")
    assert [x["item"] for x in led["order"]][:2] == ["block 0", "decisive cell"]
    assert led["spent_s"] == pytest.approx(18800.0)


def test_reported_cells_run_in_order_while_the_share_holds_their_estimate(env):
    env.use(Stub(costs(fit=20.0, decisive=50.0, reported=50.0)))
    doc = r2lfit.fit(BODY, "F25F9cl", out=env.out, log=quiet)
    assert len(doc["history"]) == 30  # 7,200 s: every block fits
    env.stub.calls.clear()
    sel = r2lfit.select(BODY, "F25F9cl", out=env.out, reported_core_h=0.2, log=quiet)
    assert sel["selected"] != "start"  # (else the start's cells would reuse the selected one's evaluations)
    kinds = [kind_of(s) for _, s in env.stub.calls]
    assert kinds[:80] == ["decisive"] * 80 and "decisive" not in kinds[80:]  # the decisive cell first
    est = 50.0 * 80 / 400 * 25 * 1.25
    cells = sel["by_delay"]
    assert [(c["k"], c["candidate"]) for c in cells] == [(k, n) for k in (0, 2, 4, 1, 3) for n in (sel["selected"],
                                                                                                    "start")]
    assert all(c["estimate_s"] == pytest.approx(est) for c in cells)
    left = 3.8 * 3600 - 7200.0 - 4000.0
    for i, c in enumerate(cells):
        assert c["share_left_s"] == pytest.approx(left - 250.0 * min(i, 9))
        assert c["started"] is (i < 9) and c["status"] == ("run" if i < 9 else "not run: budget")
    assert sel["overrun_reported"] is None
    gains = {n: np.asarray(c["gains"]) for n, c in sel["candidates"].items()}
    ran = [(s["slots"][0]["k"], s["seed"], next(n for n, g in gains.items() if np.array_equal(g, x)))
           for x, s in env.stub.calls[80:]]
    assert ran == [(c["k"], s, c["candidate"]) for c in cells if c["started"] for s in r2lfit.REPORTED_SEEDS]
    led = s2io.read_json(r2lfit.fit_dir(env.out, "F25F9cl", BODY, 42) / "ledger.json")
    assert [x["started"] for x in led["reported"]] == [True] * 9 + [False]


def test_a_reported_cell_past_the_share_stops_the_others(env):
    env.use(Stub(costs(fit=20.0, decisive=50.0, reported=200.0)))
    r2lfit.fit(BODY, "F25F9cl", out=env.out, log=quiet)
    sel = r2lfit.select(BODY, "F25F9cl", out=env.out, reported_core_h=0.2, log=quiet)
    assert sel["selected"] != "start"
    started = [c["started"] for c in sel["by_delay"]]
    assert started == [True, True, True] + [False] * 7
    assert sel["overrun_reported"]["over_share_s"] == pytest.approx(3000.0 - 2480.0)
    assert all(c["status"] == "not run: budget" for c in sel["by_delay"][3:])


def test_a_rerun_of_the_selection_reuses_every_kept_evaluation(env):
    env.use(Stub(costs(fit=20.0, decisive=50.0, reported=50.0)))
    r2lfit.fit(BODY, "F25F9cl", out=env.out, log=quiet)
    a = r2lfit.select(BODY, "F25F9cl", out=env.out, log=quiet)
    env.stub.calls.clear()
    b = r2lfit.select(BODY, "F25F9cl", out=env.out, log=quiet)
    assert env.stub.calls == [] and b["evaluations_kept"]["added"] == 0
    assert b["selection"] == a["selection"] and b["by_delay"] == a["by_delay"] and b["spent_s"] == a["spent_s"]


# ------------------------------------------------------------------------------------------------
# Publishing and the rows (rule Q6)
# ------------------------------------------------------------------------------------------------
def fitted(env, set_name="V6noF5"):
    passed_gate(env, set_name)
    env.use(Stub(costs(fit=20.0, decisive=50.0, reported=50.0)))
    r2lfit.fit(BODY, set_name, out=env.out, log=quiet)
    return r2lfit.select(BODY, set_name, out=env.out, log=quiet)


class FakeCtx:
    def built(self, cond):
        return types.SimpleNamespace()


def test_publish_then_the_row_resolves_and_refuses(env, capsys):
    sel = fitted(env)
    idx = r2lfit.INDEX
    e = r2lfit.publish(BODY, "V6noF5", out=env.out, index=idx)
    dst = idx.parent / "V6noF5" / BODY
    assert e == dict(set="V6noF5", body=BODY, seed=42,
                     r2lg5=dict(path=str((dst / "r2lg5.json").resolve()), sha256=s2io.sha256_file(dst / "r2lg5.json")))
    assert s2io.read_json(idx) == dict(schema="s2-r2l-index-1", fits=[e])
    src = env.out / "V6noF5" / BODY
    for rel in ("start.json", "r2lg5.json", "r2lg5_evals.jsonl", "r2lfit/42/rotation.json", "r2lfit/42/cma.json",
                "r2lfit/42/evals.jsonl", "r2lfit/42/ledger.json"):
        assert s2io.sha256_file(dst / rel) == s2io.sha256_file(src / rel)
    assert (dst / "COLLECTED.json").exists()
    before = {p.name: s2io.sha256_file(p) for p in (dst / "COLLECTED.json", idx)}
    assert r2lfit.publish(BODY, "V6noF5", out=env.out, index=idx) == e  # again: nothing changes
    assert {p.name: s2io.sha256_file(p) for p in (dst / "COLLECTED.json", idx)} == before
    assert len(s2io.read_json(idx)["fits"]) == 1
    capsys.readouterr()

    row = r2lfit.r2l_row(BODY, None, "R2L-V6noF5/42")
    assert isinstance(row, lookbar.LookRow) and isinstance(row.inner, r2lfit.R2LRow)
    inner = row.inner
    assert inner.mode == "R2" and inner.fit_set == "V6noF5" and inner.name == "R2L-V6noF5/42"
    assert np.array_equal(inner.gains, speedfit.gains_vector(sel["selected_gains"]))
    desc = row.describe()
    assert desc["gains_info"]["file"] == str((dst / "r2lg5.json").resolve())
    assert desc["gains_file"] == dict(path=e["r2lg5"]["path"], sha256=e["r2lg5"]["sha256"])
    names = {Path(p).name for p in row.sources()}
    assert names == {"r2lfit.py", "lookbar.py", "lookread.py"}
    again = panels.row_from_spec(row.spec())
    assert again.describe() == desc
    # another set's condition, and a stock one, are refused before any physics
    for cond in (types.SimpleNamespace(set="V6", key="walk/v0.4", judge="bar_walk"),
                 types.SimpleNamespace(key="walk/nominal", judge="walk")):
        with pytest.raises(r2lfit.R2LError, match="another set"):
            row.episode(FakeCtx(), cond, 3042)
    # a missing entry
    for body, name in ((BODY, "R2L-V6noF5/43"), ("60-N-T-10", "R2L-V6noF5/42"), (BODY, "R2L-F25F9cl/42")):
        with pytest.raises(r2lfit.R2LError, match="entries"):
            r2lfit.r2l_row(body, None, name)
    with pytest.raises(ValueError):
        r2lfit.r2l_row(BODY, None, "R2/42")
    # registered by set, wrapped by the bar for a delay
    assert r2lfit.register(idx) == ["R2L-V6noF5"]
    rows = bar.rows(BODY, ["R2L-V6noF5/42@10ms"])
    d = rows["R2L-V6noF5/42@10ms"]
    assert isinstance(d, bar.Delayed) and d.k == 5 and isinstance(d.inner, lookbar.LookRow)
    # a changed gains file
    text = (dst / "r2lg5.json").read_text()
    (dst / "r2lg5.json").write_text(text + " ")
    with pytest.raises(r2lfit.R2LError, match="changed"):
        r2lfit.r2l_row(BODY, None, "R2L-V6noF5/42")
    with pytest.raises(r2lfit.R2LError, match="changed"):
        panels.row_from_spec(row.spec())
    (dst / "r2lg5.json").write_text(text)
    # a published fit is never replaced by another selection
    s = s2io.read_json(src / "r2lg5.json")
    s["written_at"] = "later"
    s2io.write_json(src / "r2lg5.json", s)
    with pytest.raises(r2lfit.R2LError, match="never replaced"):
        r2lfit.publish(BODY, "V6noF5", out=env.out, index=idx)


def test_importing_r2lfit_changes_no_file_or_table():
    code = (
        "import hashlib, json\n"
        "from pathlib import Path\n"
        "f = Path('designlab/stage1/speedfit.py')\n"
        "h0 = hashlib.sha256(f.read_bytes()).hexdigest()\n"
        "from designlab.stage1 import panels\n"
        "from designlab.stage2 import bar\n"
        "def state():\n"
        "    return dict(verdict=repr(getattr(panels, 'VERDICT_ROWS', None)),\n"
        "                attributed=repr(getattr(panels, 'ATTRIBUTED', None)), rows=sorted(panels.ROW_FACTORIES),\n"
        "                bar=bar.bar_sources()['sha256'], panels=panels.panel_sources()['sha256'])\n"
        "before = state()\n"
        "import designlab.stage2.r2lfit\n"
        "print(json.dumps(dict(h0=h0, h1=hashlib.sha256(f.read_bytes()).hexdigest(), before=before, after=state())))\n")
    envv = dict(os.environ, PYTHONPATH=".", MUJOCO_GL="disable", PYTHONDONTWRITEBYTECODE="1",
                **{v: "1" for v in s2io.THREAD_VARS})
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=envv, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    x = json.loads(r.stdout.strip().splitlines()[-1])
    assert x["h0"] == x["h1"] and x["before"] == x["after"]


# ------------------------------------------------------------------------------------------------
# Slow: slot 0 is the bar's R2 row, bit for bit (two development episodes on V0)
# ------------------------------------------------------------------------------------------------
def _have_v0() -> bool:
    try:
        return (V0 / "refs" / "v0.4.npz").exists() and all(Path(f).exists() for f in plant.model_files(BODY))
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not (SLOW and ONE_THREAD and _have_v0()), reason=(
    "slow (STAGE2_SLOW=1); needs set V0's arrays and the round-two models (tools/restore_inputs.py) and one BLAS "
    "thread"))
def test_slot_0_is_the_bars_r2_row_bit_for_bit():
    ev = r2lfit.Evaluator(BODY, "V0", V0)
    cond = ev.conds["walk/v0.4"]
    gains = speedfit.gains_vector(speedfit.start_record(BODY)["gains"])
    b1, out1, rec1 = r2lfit.run_slot(ev.ctx, cond, gains, 2000, 0, 5)
    U1, Q1 = rec1.U.copy(), rec1.Q.copy()
    b2, out2, rec2 = bar.Delayed(panels.TrackerRow("R2/start", "R2", gains, None), 5).episode(ev.ctx, cond, 2000)
    assert b1 is b2 and np.array_equal(U1, rec2.U) and np.array_equal(Q1, rec2.Q)
    assert (out1["fell"], out1["seconds"]) == (out2["fell"], out2["seconds"])
