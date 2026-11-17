"""Gate 1's runner of the served panels (``designlab/stage2/gate1.py``; ``stage2/design/servo_model/
SERVO_MODEL_DESIGN.md`` section 4, 5.1 and package S3; the owner's condition 2, ``DECISION_LOG.md`` P41).

The closed-loop tests run cut episodes (0.1 s, a smoke check that no rule judges) of the served R0 row on a
synthetic set ``T0`` that ``tests/test_stage2_g8a.py``'s fixture builds in a temporary directory from Stage 1's
arrays: no Stage 2 reference set is touched (rule X1), and every record is written under a temporary directory.
Simulation only: the servo model's values are a model, none is measured, and no record here is a result.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from designlab.stage1 import drive, panels, plant  # noqa: E402
from designlab.stage2 import bar, gate1, regress, s2io, servo, setio  # noqa: E402

BODY = servo.BODY
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
BAR_SOURCES = "45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d"
PANEL_SOURCES = "c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1"
RATES = dict(schema=bar.COST_SCHEMA, rates=dict(R0=dict(wall_per_sim_s=0.1), R2=dict(wall_per_sim_s=0.2)),
             measured_on=dict(condition="walk/brisk", body=BODY))


def quiet(*a, **k):
    pass


@pytest.fixture
def cut_episodes(monkeypatch):
    """``bar.run_episode`` with the row's real (served) episode and a stand-in record in place of the bar's judge,
    which reads no cut walk (``bar.Unreadable``: a cut condition is a smoke check, as the bar's tests treat it)."""
    import time

    def run_episode(row, ctx, cond, seed, relax=(), trace_root=None, keep_trace=True, not_kept_reason=None):
        t0 = time.perf_counter()
        b, out, rec = row.episode(ctx, cond, int(seed), tuple(relax))
        fell = bool(out.get("fell"))
        return dict(seed=int(seed), fell=fell, succeeded=not fell, succeeded_but_A1=not fell, failed_terms=[],
                    failed_terms_but_A1=[], fall_cause=out.get("fall_cause"), fell_before_command=None,
                    fall_candidates=[], survival_s=float(rec.t[-1]) if rec.rows else 0.0, synthetic_judge=True,
                    wall_s=time.perf_counter() - t0)

    monkeypatch.setattr(bar, "run_episode", run_episode)


def _t0(tmp_path) -> Path:
    from tests import test_stage2_g8a as G

    G.make_t0(tmp_path)
    return tmp_path


def _files(d: Path) -> list[str]:
    return sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file())


# ------------------------------------------------------------------------------------------------
# The route, the names, the rules
# ------------------------------------------------------------------------------------------------
def test_the_route_keeps_the_bars_hash_and_reads_its_machinery_by_import():
    """Route (c), ruled 16 Nov 2026: a runner outside bar.py; the registered pair of hashes holds with it in the
    code, it is in neither, and the regress check passes."""
    assert (bar.bar_sources()["sha256"], panels.panel_sources()["sha256"]) == (BAR_SOURCES, PANEL_SOURCES)
    names = {Path(f).name for f in bar.bar_sources()["files"]} | set(panels.panel_sources()["modules"])
    assert "gate1.py" not in names and "servo.py" not in names
    assert regress.check() == []
    assert issubclass(gate1.GateWorker, bar.BarWorker) and gate1.HEADLINE == "seeds3042"
    assert gate1.SECOND == "seeds5042" and bar.BLOCKS[gate1.SECOND] == tuple(range(5042, 5062))
    src = (ROOT / "designlab" / "stage2" / "gate1.py").read_text()
    assert "bar.run_episode" in src and "bar.record_key" in src and "bar.counts" in src
    assert set(gate1.RULES) >= {"route", "rows", "condition", "seeds", "pass_", "repeat", "verdict", "readings", "key",
                                "development", "cost"}
    assert gate1.RULES["readings"] == servo.READINGS_RULE and gate1.RULES_REGISTERED is False


def test_condition_2s_walk_delays_settings_and_rows():
    assert (gate1.CONDITION, gate1.SLIPPERY, gate1.DELAYS_MS) == ("walk/v0.3", "walk/v0.3_mu0.54", (4, 10))
    assert gate1.CORNERS == ("F-", "F+", "P0", "P+", "M-", "M+") and gate1.DRAWN == "D"
    assert gate1.settings_of(["@judged"]) == list(servo.JUDGED)
    assert gate1.settings_of(["@named", "D", "N"]) == [s for s in servo.NAMED if s != "D"] + ["D"]
    assert gate1.settings_of(["@all"]) == list(servo.NAMED)
    with pytest.raises(ValueError):
        gate1.settings_of(["nominal"])
    rows = gate1.row_names(["SV0", "SV2/42"], ["F+", "D"], (4, 10))
    assert [r[0] for r in rows] == ["SV0/F+@4ms", "SV0/F+@10ms", "SV0/D@4ms", "SV0/D@10ms", "SV2/42/F+@4ms",
                                    "SV2/42/F+@10ms", "SV2/42/D@4ms", "SV2/42/D@10ms"]
    assert [gate1.base_of(r[0]) for r in rows[::2]] == ["SV0", "SV0", "SV2/42", "SV2/42"]
    with pytest.raises(ValueError, match="s2rows"):
        gate1.row_names(["SV5/R5/42"], ["N"], (4,))
    with pytest.raises(ValueError):
        gate1.row_names(["SV0"], ["N"], (3,))
    # the bar builds them, the delay outside the served row
    r = bar.rows(BODY, ["SV0/D@10ms", "SV2/42/F-@4ms"])
    assert isinstance(r["SV0/D@10ms"], bar.Delayed) and r["SV0/D@10ms"].k == 5
    assert isinstance(r["SV0/D@10ms"].inner, servo.Served) and r["SV2/42/F-@4ms"].inner.inner.name == "R2/42"
    # the bar's own lists know R0 and R2 only: a served row is neither, so this runner carries its own
    assert bar.controller_of("SV0/D@10ms") == "SV0" and "SV0" not in bar.NO_VERDICT_CONTROLLERS
    assert "SV2" not in bar.ATTRIBUTED_CONTROLLERS


def _ep(seed: int, fell: bool = False, ok: bool = True, error: bool = False) -> dict:
    return dict(seed=seed, fell=fell or error, succeeded=ok and not fell and not error,
                succeeded_but_A1=ok and not fell and not error, error="x" if error else None)


def _block(fall=(), fail=(), errors=(), seeds=bar.BLOCKS[gate1.HEADLINE]) -> list[dict]:
    return [_ep(s, s in fall, s not in fail, s in errors) for s in seeds]


def test_the_pass_rule_is_condition_2s_as_approved():
    """P41: a corner passes when all 20 stay up whatever the limits; the drawn panel by the bar's whole rule (20 up,
    17 within every limit); every other setting is read by no rule; a record is judged only when it holds a block's
    20 seeds in order with no failed episode, on gate 1's walk at 4 or 10 ms."""
    seeds = list(bar.BLOCKS[gate1.HEADLINE])
    many = seeds[:10]  # (10 of 20 outside a limit)
    r = gate1.reading("F+", _block(fail=many), seeds)
    assert (r["rule"], r["judged"], r["pass_"], r["survived"], r["succeeded"]) == ("corner", True, True, 20, 10)
    assert r["limits"] == "recorded, not judged"
    assert gate1.reading("M-", _block(fall=seeds[:1]), seeds)["pass_"] is False
    assert gate1.reading("D", _block(fail=seeds[:3]), seeds)["pass_"] is True  # 17 of 20
    assert gate1.reading("D", _block(fail=seeds[:4]), seeds)["pass_"] is False  # 16 of 20
    assert gate1.reading("D", _block(fall=seeds[:1]), seeds)["pass_"] is False
    for st in ("N", "today", "V-", "J+", "G5", "T+", "X6L", "W1", "W2", "D/3042"):
        r = gate1.reading(st, _block(), seeds)
        assert (r["rule"], r["judged"], r["pass_"]) == ("reported", False, None), st
    assert gate1.reading("F-", _block(), seeds, applies=False)["pass_"] is None
    assert gate1.reading("F-", _block(errors=seeds[:1]), seeds)["pass_"] is None  # (C8: run again, never a count)
    assert gate1.reading("F-", _block()[:19], seeds[:19])["pass_"] is None
    second = list(bar.BLOCKS[gate1.SECOND])
    assert gate1.reading("D", _block(seeds=second), second)["pass_"] is True
    assert gate1.gate1_reading_applies("walk/v0.3", 4, False) and gate1.gate1_reading_applies("walk/v0.3", 10, False)
    assert not gate1.gate1_reading_applies("walk/v0.3", 6, False)
    assert not gate1.gate1_reading_applies("walk/v0.3_mu0.54", 4, False)
    assert not gate1.gate1_reading_applies("walk/v0.3", 4, True)
    # no verdict from a served stand-in row; a row of another controller gives its reading's
    ok = gate1.reading("D", _block(), seeds)
    assert gate1.verdict("SV0/D@4ms", ok)["value"] is None and gate1.verdict("SV2/42/D@4ms", ok)["value"] is None
    assert gate1.verdict("SV5/R5/42/D@4ms", ok)["value"] is True


def test_the_out_directory_rules():
    """RULES["development"]: never under stage1/, the sets' root or (rules not registered) stage2/runs/."""
    for bad in (s2io.STUDY / "stage1" / "x", setio.ROOT / "x", s2io.STUDY / "stage2" / "runs" / "gate1"):
        with pytest.raises(gate1.Gate1Error, match="writes no record there"):
            gate1.check_out(bad)
    gate1.check_out(Path("/tmp/claude-x/out"))
    assert gate1.look_set("V6") == "V6" and gate1.look_set("V6noF5.2") == "V6noF5" and gate1.look_set("V0") is None


@pytest.mark.skipif(not ONE_THREAD, reason="a run needs one BLAS thread (OMP_NUM_THREADS=1 ...)")
def test_a_run_on_a_look_set_is_refused_before_anything(tmp_path):
    with pytest.raises(gate1.Gate1Error, match="look set"):
        gate1.run("V6", BODY, tmp_path / "out", ["SV0"], log=quiet)
    assert not (tmp_path / "out").exists()


# ------------------------------------------------------------------------------------------------
# The dry run: the plan and its cost, nothing run, nothing written
# ------------------------------------------------------------------------------------------------
@needs_run
def test_the_dry_run_prints_the_plan_and_cost_and_writes_nothing(tmp_path):
    root = _t0(tmp_path / "sets")
    before = _files(tmp_path)
    logged = []
    doc = gate1.dry_run("T0", BODY, ["SV0", "SV2/42"], ["F+", "D"], (4, 10), ["walk/v0.4", "walk/v0.3"],
                        root=root, rates=RATES, log=logged.append)
    assert _files(tmp_path) == before and logged == [doc["text"]]
    head = [x for x in doc["records"] if x["block"] == gate1.HEADLINE]
    assert len(head) == 2 * 2 * 2 * 2  # bases x settings x delays x conditions
    v4 = [x for x in head if x["condition"] == "walk/v0.4"]
    assert all(x["status"] == "to run" and x["episodes"] == 20 for x in v4)
    assert all(x["status"].startswith("not run") and x["episodes"] == 0 for x in head if x["condition"] == "walk/v0.3")
    sec = v4[0]["seconds"]
    one = next(x for x in v4 if x["row"] == "SV0/F+@4ms")
    two = next(x for x in v4 if x["row"] == "SV2/42/F+@4ms")
    assert one["core_h"] == pytest.approx(20 * sec * 0.1 / 3600) and one["rate"] == "R0"
    assert two["core_h"] == pytest.approx(20 * sec * 0.2 / 3600) and two["rate"] == "R2"
    assert one["core_h_with_layer"] == pytest.approx(one["core_h"] * gate1.LAYER_COST)
    assert {x["rule"] for x in v4} == {"corner", "whole"}
    # the second block: the drawn panel only, each if its headline record passes
    second = [x for x in doc["records"] if x["block"] == gate1.SECOND]
    assert sorted(x["row"] for x in second if x["condition"] == "walk/v0.4") == [
        "SV0/D@10ms", "SV0/D@4ms", "SV2/42/D@10ms", "SV2/42/D@4ms"]
    assert all(x["status"].startswith("if selected") for x in second if x["condition"] == "walk/v0.4")
    t = doc["totals"]
    assert (t["planned_records"], t["planned_episodes"], t["conditional_records"]) == (8, 160, 4)
    assert "runs no episode and writes nothing" in doc["text"] and "development record" in doc["text"]


# ------------------------------------------------------------------------------------------------
# A run: records of its own, the readings, resume, the repeat
# ------------------------------------------------------------------------------------------------
@needs_run
def test_a_run_writes_its_own_records_with_the_servo_readings(tmp_path, cut_episodes):
    root = _t0(tmp_path / "sets")
    out = tmp_path / "out"
    doc = gate1.run("T0", BODY, out, ["SV0"], ["F+", "D", "M-"], (4,), ["walk/v0.4"], episodes=2, cut_s=0.1,
                    root=root, log=quiet)
    recs = {r["row"]: r for r in gate1.records(out, "T0", BODY)}
    assert sorted(recs) == ["SV0/D@4ms", "SV0/F+@4ms", "SV0/M-@4ms"]
    for name, r in recs.items():
        assert r["schema"] == gate1.SCHEMA and r["development"] is True and r["status"] == "done"
        assert r["path"] == f"T0/{BODY}/seeds3042/gate1/{name}/walk/v0.4.json"
        k = r["key"]
        assert k["schema"] == gate1.SCHEMA and k["sources"] == dict(panels=PANEL_SOURCES, bar=BAR_SOURCES)
        assert k["table_sha256"] == servo.TABLE_SHA256 and list(k["runner_sources"]) == ["designlab/stage2/gate1.py"]
        assert "designlab/stage2/servo.py" in k["row_sources"] and k["row"]["served_row"] == name.split("@")[0]
        assert r["key_sha256"] == s2io.sha256_json(k) and k["seeds"] == [3042, 3043] and r["cut"] is True
        assert r["verdict"]["value"] is None and "stand-in" in r["verdict"]["reason"]
        assert r["reading"]["pass_"] is None and r["attribution"]["episodes"] == 0
        for e in r["episodes"]:
            assert e["servo"]["readings"]["physics_steps"] == 50 and not e.get("error")
        sr = r["servo_readings"]
        assert (sr["runs"], sr["runs_read"], sr["reported_only"]) == (2, 2, True)
        assert set(sr["inside_play_share"]) == set(servo.LEG_DRIVES)
    assert recs["SV0/D@4ms"]["servo_settings"] == ["D/3042", "D/3043"]  # (20 drawn sets by seed: one per run)
    assert recs["SV0/F+@4ms"]["servo_settings"] == ["F+"] and recs["SV0/M-@4ms"]["setting"] == "M-"
    assert doc["session"]["episodes_run"] == 6 and doc["condition2"]["holds"] is None
    s = s2io.read_json(out / "T0" / BODY / "gate1_summary.json")
    assert [x["row"] for x in s["records"]] == ["SV0/D@4ms", "SV0/F+@4ms", "SV0/M-@4ms"]
    # the plant the worker used is given back (the context's plant: no step hook, its own envelope)
    ctx = bar._CONTEXT["ctx"]
    for p in ctx._plants.values():
        assert "step" not in p.__dict__ and type(p.envelope) is drive.Envelope
    # a second run of the same key keeps the records; another key is refused, never overwritten
    stamp = {n: (out / r["path"]).stat().st_mtime_ns for n, r in recs.items()}
    again = gate1.run("T0", BODY, out, ["SV0"], ["F+", "D", "M-"], (4,), ["walk/v0.4"], episodes=2, cut_s=0.1,
                      root=root, log=quiet)
    assert again["session"]["episodes_run"] == 0 and len(again["session"]["skipped"]) == 3
    assert {n: (out / r["path"]).stat().st_mtime_ns for n, r in recs.items()} == stamp
    with pytest.raises(gate1.Gate1Error, match="another key"):
        gate1.run("T0", BODY, out, ["SV0"], ["F+"], (4,), ["walk/v0.4"], episodes=2, cut_s=0.12, root=root, log=quiet)
    # a condition the set has no reference for gets a "not run" record
    gate1.run("T0", BODY, out, ["SV0"], ["F+"], (4,), ["walk/v0.3"], episodes=2, cut_s=0.1, root=root, log=quiet)
    nr = s2io.read_json(out / "T0" / BODY / "seeds3042" / "gate1" / "SV0" / "F+@4ms" / "walk" / "v0.3.json")
    assert nr["status"] == "not run" and nr["why"]["reason"]


@needs_run
def test_the_second_block_runs_the_drawn_panel_whose_headline_reading_passed(tmp_path, cut_episodes):
    root = _t0(tmp_path / "sets")
    out = tmp_path / "out"
    kw = dict(bases=["SV0"], settings=["F+", "D"], delays_ms=(4,), conditions=["walk/v0.4"], episodes=1, cut_s=0.1,
              root=root, log=quiet)
    gate1.run("T0", BODY, out, **kw)
    # a cut record is not judged, so nothing is selected for the second block
    doc = gate1.run("T0", BODY, out, block=gate1.SECOND, **kw)
    assert doc["session"]["episodes_run"] == 0 and len(doc["session"]["not_selected"]) == 1
    assert "not judged" in doc["session"]["not_selected"][0]
    # a headline reading that passed selects the drawn panel's record (written by hand here: no episode passes a
    # cut condition); F+ never has a second block
    head = gate1.record_path(out, "T0", BODY, gate1.HEADLINE, "SV0/D@4ms", _cond(root))
    rec = s2io.read_json(head)
    rec["reading"]["pass_"] = True
    s2io.write_json(head, rec)
    doc = gate1.run("T0", BODY, out, block=gate1.SECOND, **kw)
    assert doc["session"]["episodes_run"] == 1 and doc["session"]["not_selected"] == []
    sec = s2io.read_json(gate1.record_path(out, "T0", BODY, gate1.SECOND, "SV0/D@4ms", _cond(root)))
    assert sec["block"] == gate1.SECOND and sec["seeds"] == [5042] and sec["servo_settings"] == ["D/5042"]
    assert sec["repeat_of"]["record"] == f"T0/{BODY}/seeds3042/gate1/SV0/D@4ms/walk/v0.4.json"
    assert not gate1.record_path(out, "T0", BODY, gate1.SECOND, "SV0/F+@4ms", _cond(root)).exists()
    rows = {(x["block"], x["row"]): x for x in doc["records"]}
    assert rows[(gate1.HEADLINE, "SV0/D@4ms")]["repeat"] == "repeat not judged"


@needs_run
def test_the_bars_own_judge_reads_a_served_episode(tmp_path):
    """No stand-in: one whole episode of the served drawn panel through ``bar.run_episode`` (the row's episode, then
    the bar's judge), and a cut one, which the bar's judge cannot read: recorded as the bar records an episode that
    failed to run, never a count, and run again on the next run of the same key (C8)."""
    root = _t0(tmp_path / "sets")
    out = tmp_path / "out"
    gate1.run("T0", BODY, out, ["SV0"], ["D"], (4,), ["walk/v0.4"], episodes=1, root=root, log=quiet)
    r = s2io.read_json(gate1.record_path(out, "T0", BODY, gate1.HEADLINE, "SV0/D@4ms", _cond(root)))
    e = r["episodes"][0]
    assert not e.get("error") and "synthetic_judge" not in e and e["seed"] == 3042 and r["cut"] is False
    n = e["servo"]["readings"]["physics_steps"]  # (the layer's steps: the episode's, to its end or its fall)
    assert e["servo"]["setting"] == "D/3042" and n > 0 and abs(n * 0.002 - e["survival_s"]) < 0.01
    assert set(e) >= {"succeeded", "failed_terms", "measures", "survival_s", "trace"} and e["trace"]["kept"] is False
    assert r["counts"]["episodes"] == 1 and r["reading"]["pass_"] is None  # (one seed: no block, not judged)
    assert not list((out / "T0" / BODY / gate1.HEADLINE / "gate1" / "traces").rglob("*.npz"))  # (hashed, not kept)
    cut = tmp_path / "cut"
    gate1.run("T0", BODY, cut, ["SV0"], ["F+"], (4,), ["walk/v0.4"], episodes=1, cut_s=0.1, root=root, log=quiet)
    r = s2io.read_json(gate1.record_path(cut, "T0", BODY, gate1.HEADLINE, "SV0/F+@4ms", _cond(root)))
    assert "Unreadable" in r["episodes"][0]["error"] and r["episodes"][0]["servo"] is None
    assert r["counts"]["errors"] == 1 and r["servo_readings"]["runs_without_readings"] == 1
    again = gate1.run("T0", BODY, cut, ["SV0"], ["F+"], (4,), ["walk/v0.4"], episodes=1, cut_s=0.1, root=root,
                      log=quiet)
    assert again["session"]["rerun_failed"] == ["SV0/F+@4ms walk/v0.4"] and again["session"]["episodes_run"] == 1


def _cond(root):
    docs = setio.load(setio.set_dir("T0", BODY, root))
    return bar.choose(docs, gate1.HEADLINE, ["walk/v0.4"])[0]


# ------------------------------------------------------------------------------------------------
# The summary: the repeat and condition 2
# ------------------------------------------------------------------------------------------------
def _rec(out, block, base, setting, ms, passed, cond="walk/v0.3"):
    row = f"{base}/{setting}@{ms}ms"
    ctrl = base.split("/")[0]
    gives = ctrl not in servo.NO_VERDICT_CONTROLLERS
    doc = dict(schema=gate1.SCHEMA, development=True, status="done", block=block, row=row, base=base, setting=setting,
               delay_ms=ms, key_name=cond, counts=dict(survived=20, succeeded=20, episodes=20),
               reading=dict(rule=gate1.rule_of(setting), pass_=passed),
               verdict=dict(value=passed if gives else None, reason=None),
               servo_readings=dict(runs_left_supply_window=0, inside_play_share_leg_mean=0.1))
    s2io.write_json(Path(out) / "T0" / BODY / block / "gate1" / row / "walk" / f"{cond.split('/')[1]}.json", doc)


def test_the_repeat_must_pass_too_and_condition_2_needs_two_policies(tmp_path, monkeypatch):
    out = tmp_path / "out"
    for base in ("SV5/R5/42", "SV5/R5/43"):
        for s in servo.JUDGED:
            for ms in gate1.DELAYS_MS:
                _rec(out, gate1.HEADLINE, base, s, ms, True)
                if s == "D":
                    _rec(out, gate1.SECOND, base, s, ms, True)
    _rec(out, gate1.HEADLINE, "SV0", "D", 4, True)
    doc = gate1.summarize(out, "T0", BODY, write=False)
    rows = {(x["block"], x["row"]): x for x in doc["records"]}
    assert rows[(gate1.HEADLINE, "SV5/R5/42/D@4ms")]["repeat"] == "replicated"
    assert rows[(gate1.HEADLINE, "SV0/D@4ms")]["repeat"] == "repeat not run"
    assert rows[(gate1.HEADLINE, "SV0/D@4ms")]["verdict"] is None
    # while gate 1's servo rules are not registered no record is a gate 1 record: no verdict on condition 2
    c = doc["condition2"]
    assert c["holds"] is None and c["holds_if_registered"] is True and "not registered" in c["reason"]
    monkeypatch.setattr(gate1, "RULES_REGISTERED", True)
    assert gate1.summarize(out, "T0", BODY, write=False)["condition2"]["holds"] is True
    _rec(out, gate1.SECOND, "SV5/R5/43", "D", 10, False)  # P41: a failed repeat stops it
    doc = gate1.summarize(out, "T0", BODY, write=False)
    assert {(x["block"], x["row"]): x for x in doc["records"]}[(gate1.HEADLINE, "SV5/R5/43/D@10ms")]["repeat"] == (
        "not replicated")
    assert doc["condition2"]["holds"] is False
    _rec(out, gate1.SECOND, "SV5/R5/43", "D", 10, True)
    _rec(out, gate1.HEADLINE, "SV5/R5/43", "P+", 4, False)  # a corner that falls
    assert gate1.summarize(out, "T0", BODY, write=False)["condition2"]["holds"] is False
    _rec(out, gate1.HEADLINE, "SV5/R5/43", "P+", 4, True)
    for s in servo.JUDGED:  # one policy alone: not condition 2 (P40's condition 5)
        for ms in gate1.DELAYS_MS:
            p = out / "T0" / BODY / gate1.HEADLINE / "gate1" / f"SV5/R5/43/{s}@{ms}ms"
            for f in p.rglob("*.json"):
                f.unlink()
    assert gate1.summarize(out, "T0", BODY, write=False)["condition2"]["holds"] is None
    text = gate1.summary_text(gate1.summarize(out, "T0", BODY, write=False))
    assert "SV5/R5/42/D@4ms" in text and "replicated" in text
    assert json.loads(json.dumps(doc))["schema"] == gate1.SCHEMA + "-summary"
