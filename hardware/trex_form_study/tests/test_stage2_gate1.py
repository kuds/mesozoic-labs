"""Gate 1's runner of the served panels (``designlab/stage2/gate1.py``; ``stage2/design/servo_model/
SERVO_MODEL_DESIGN.md`` section 4, 5.1 and package S3; the owner's condition 2, ``DECISION_LOG.md`` P41).

The closed-loop tests run cut episodes (0.1 s, ``--cut-s``: a smoke check that is never judged) of the served R0 row
on a synthetic set ``T0`` that ``tests/test_stage2_g8a.py``'s fixture builds in a temporary directory from Stage 1's
arrays: no Stage 2 reference set is touched (rule X1), and every record and listing is written under a temporary
directory.  Simulation only: the servo model's values are a model, none is measured, and no record here is a result.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from designlab.stage1 import drive, panels, plant, tune  # noqa: E402
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


#: A cap no test run here comes near (``RULES["cap"]``: a run must name one).
CAP = 100


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
                                "development", "cost", "placement", "cap", "cut", "supply"}
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


def _left(e: dict, left: bool) -> dict:
    return dict(e, servo=dict(setting="x", readings=dict(left_supply_window=left)))


def test_a_v_minus_or_w1_run_that_leaves_the_supply_window_is_not_counted_as_up():
    """The lead's fifth list (the check's finding 3; design section 4): a V- or W1 run that did not fall and left the
    supply window is "left the supply window" in the counts, in both readings, not stayed up; a run of any other
    setting that left it is counted as the bar counts it (reported only).  Both settings are reported, so no verdict
    moves."""
    seeds = list(bar.BLOCKS[gate1.HEADLINE])
    eps = _block(fall=seeds[:1])
    eps = [_left(e, e["seed"] in seeds[:4]) for e in eps]  # (3042 fell and left; 3043-3045 stayed on their feet)
    assert gate1.SUPPLY_SETTINGS == ("V-", "W1")
    for st in ("V-", "W1"):
        for which in ("success", "but_A1"):
            c = gate1.counts(st, eps, seeds, which)
            assert (c["survived"], c["succeeded"], c["left_supply_window"], c["falls"]) == (16, 16, 3, 1), (st, which)
            assert c["by_the_bar"] == dict(survived=19, succeeded=19) and c["pass_"] is False
        r = gate1.reading(st, eps, seeds)
        assert (r["survived"], r["left_supply_window"], r["pass_"], r["rule"]) == (16, 3, None, "reported")
    for st in ("D", "F+", "N", "J+"):
        c = gate1.counts(st, eps, seeds)
        assert (c["survived"], c["left_supply_window"]) == (19, 0) and "by_the_bar" not in c, st
    assert gate1.reading("F+", eps, seeds)["pass_"] is False  # (a fall; leaving the window fails no corner)
    assert gate1.reading("F+", [_left(e, True) for e in _block()], seeds)["pass_"] is True
    assert gate1.counts("V-", _block(), seeds)["pass_"] is True  # (the bar's pass when nothing left the window)


def test_the_summary_names_runs_that_left_the_supply_window(tmp_path):
    seeds = list(bar.BLOCKS[gate1.HEADLINE])
    eps = [_left(e, e["seed"] in seeds[:2]) for e in _block()]
    for st in ("V-", "N"):
        _rec(tmp_path, gate1.HEADLINE, "SV0", st, 4, None, counts=gate1.counts(st, eps, seeds), left=2)
    doc = gate1.summarize(tmp_path, "T0", BODY, write=False)
    rows = {x["setting"]: x for x in doc["records"]}
    assert (rows["V-"]["survived"], rows["V-"]["not_up_left_supply_window"], rows["V-"]["left_supply_window"]) == (
        18, 2, 2)
    assert (rows["N"]["survived"], rows["N"]["not_up_left_supply_window"], rows["N"]["left_supply_window"]) == (
        20, 0, 2)
    text = gate1.summary_text(doc)
    assert "| `SV0/V-@4ms` | reported | 18/20 |" in text and "2, of them 2 not up" in text
    assert "| `SV0/N@4ms` | reported | 20/20 |" in text and "2 (reported)" in text


def test_sv2_rows_are_refused_on_a_scratch_or_fixture_set(tmp_path):
    """The lead's fifth list (the check's finding 2): no served R2 row on a set that is not registered, as amendment 5
    says of R2 rows; refused by the dry run and the run before anything is read or written."""
    for name in ("V0", "F25F9", "F25F9cl", "F25F9cl.2"):
        gate1.check_rows(name, ["SV0", "SV2/42"])
    gate1.check_rows("T0", ["SV0"])
    for name in ("T0", "scratch", "S1"):
        with pytest.raises(gate1.Gate1Error, match="amendment 5"):
            gate1.check_rows(name, ["SV0", "SV2/42"])
    with pytest.raises(gate1.Gate1Error, match="never runs there"):
        gate1.dry_run("T0", BODY, ["SV2/42"], ["F+"], (4,), root=tmp_path / "nowhere", rates=RATES, log=quiet)
    if ONE_THREAD:
        with pytest.raises(gate1.Gate1Error, match="never runs there"):
            gate1.run("T0", BODY, tmp_path / "out", ["SV0", "SV2/42"], ["F+"], (4,), root=tmp_path / "nowhere",
                      log=quiet, max_episodes=CAP, dev_runs=tmp_path / "dev.json")
    assert _files(tmp_path) == []


def test_the_runs_options_cut_s_for_tests_only_and_a_required_cap(capsys):
    """The lead's fifth list (the check's findings 2 and 4): ``--cut-s`` is for tests only and its help says so;
    ``run`` requires ``--max-episodes``; the listing's default file sits beside rule X1's
    ``stage2/tb/dev_episodes.json`` and is not it."""
    helps = {}
    for cmd in ("run", "dry-run"):
        with pytest.raises(SystemExit):
            gate1.main([cmd, "--help"])
        helps[cmd] = " ".join(capsys.readouterr().out.split())
        assert "--cut-s CUT_S for tests only" in helps[cmd] and "is never judged" in helps[cmd], cmd
        assert "--max-episodes" in helps[cmd]
    assert "stage2/tb/gate1_dev_runs.json" in helps["run"]
    with pytest.raises(SystemExit) as ei:
        gate1.main(["run", "--set", "T0", "--out", "/tmp/claude-x/never"])
    assert ei.value.code == 2 and "--max-episodes" in capsys.readouterr().err
    assert gate1.DEV_RUNS == s2io.STUDY / "stage2" / "tb" / "gate1_dev_runs.json"
    assert gate1.DEV_RUNS.name != "dev_episodes.json" and "stage2/tb/dev_episodes.json" in gate1.RULES["development"]


def test_the_worker_refuses_to_start_when_servo_or_gate1_changed_during_the_session():
    """The lead's fifth list (the check's finding 6): ``GateWorker`` checks that ``servo.py`` and ``gate1.py`` are
    the session's, beside the bar's and the panels' sources ``bar.BarWorker`` checks."""
    mine = gate1.runner_sources()
    assert sorted(mine) == ["designlab/stage2/gate1.py", "designlab/stage2/servo.py"]
    for f in mine:
        changed = dict(mine, **{f: "0" * 64})
        with pytest.raises(tune.PoolError, match="changed during the session"):
            gate1.GateWorker(BODY, "/nowhere", set_name="T0", runner=changed)


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
        gate1.run("V6", BODY, tmp_path / "out", ["SV0"], log=quiet, max_episodes=CAP, dev_runs=tmp_path / "dev.json")
    assert not (tmp_path / "out").exists()


# ------------------------------------------------------------------------------------------------
# The dry run: the plan and its cost, nothing run, nothing written
# ------------------------------------------------------------------------------------------------
@needs_run
def test_the_dry_run_prints_the_plan_and_cost_and_writes_nothing(tmp_path, monkeypatch):
    root = _t0(tmp_path / "sets")
    before = _files(tmp_path)
    logged = []
    with pytest.raises(gate1.Gate1Error, match="amendment 5"):  # (T0 is a fixture: no SV2 row there)
        gate1.dry_run("T0", BODY, ["SV0", "SV2/42"], ["F+", "D"], (4,), root=root, rates=RATES, log=logged.append)
    assert logged == []
    # (T0 taken as a registered set, so that the plan holds an SV2 row and its cost; the planner reads T0 as before)
    monkeypatch.setattr(gate1, "registered_set", lambda name: "V0" if name == "T0" else None)
    doc = gate1.dry_run("T0", BODY, ["SV0", "SV2/42"], ["F+", "D"], (4, 10), ["walk/v0.4", "walk/v0.3"],
                        root=root, rates=RATES, max_episodes=CAP, log=logged.append)
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
    # the second block: the drawn panel only, and never a row that gives no verdict (the lead's fifth list: the
    # bar's form; its D record is reported)
    second = [x for x in doc["records"] if x["block"] == gate1.SECOND]
    assert sorted(x["row"] for x in second if x["condition"] == "walk/v0.4") == [
        "SV0/D@10ms", "SV0/D@4ms", "SV2/42/D@10ms", "SV2/42/D@4ms"]
    assert all(x["status"] == f"not selected: {gate1.NO_REPEAT}" and x["episodes"] == 0
               for x in second if x["condition"] == "walk/v0.4")
    t = doc["totals"]
    assert (t["planned_records"], t["planned_episodes"], t["conditional_records"]) == (8, 160, 0)
    assert "runs no episode and writes nothing" in doc["text"] and "development record" in doc["text"]
    # the cap: 160 episodes to run on the headline block are over a cap of 100, which run refuses
    assert doc["cap"][gate1.HEADLINE] == dict(to_run=160, up_to=160, max_episodes=CAP, within=False)
    assert doc["cap"][gate1.SECOND] == dict(to_run=0, up_to=0, max_episodes=CAP, within=True)
    assert "cap 100: over it, so `run` refuses it" in doc["text"]
    # a verdict-giving row's D record would be selected by its headline reading (the SV5 rows, once they exist)
    monkeypatch.setattr(servo, "NO_VERDICT_CONTROLLERS", ())
    doc = gate1.dry_run("T0", BODY, ["SV0"], ["D"], (4,), ["walk/v0.4"], root=root, rates=RATES, log=quiet)
    second = [x for x in doc["records"] if x["block"] == gate1.SECOND]
    assert [x["status"] for x in second] == ["if selected: runs only if the headline record's reading passes (P41)"]
    assert doc["totals"]["conditional_records"] == 1 and _files(tmp_path) == before


# ------------------------------------------------------------------------------------------------
# A run: records of its own, the readings, resume, the repeat
# ------------------------------------------------------------------------------------------------
@needs_run
def test_a_run_writes_its_own_records_with_the_servo_readings(tmp_path):
    root = _t0(tmp_path / "sets")
    out = tmp_path / "out"
    kw = dict(root=root, log=quiet, max_episodes=CAP, dev_runs=tmp_path / "dev.json")
    doc = gate1.run("T0", BODY, out, ["SV0"], ["F+", "D", "M-"], (4,), ["walk/v0.4"], episodes=2, cut_s=0.1, **kw)
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
        assert r["counts"]["left_supply_window"] == 0 and r["counts"]["errors"] == 0
        for e in r["episodes"]:
            assert e["servo"]["readings"]["physics_steps"] == 50 and not e.get("error")
            assert e["cut"] is True and e["judged"] is False and e["succeeded"] is None
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
    again = gate1.run("T0", BODY, out, ["SV0"], ["F+", "D", "M-"], (4,), ["walk/v0.4"], episodes=2, cut_s=0.1, **kw)
    assert again["session"]["episodes_run"] == 0 and len(again["session"]["skipped"]) == 3
    assert {n: (out / r["path"]).stat().st_mtime_ns for n, r in recs.items()} == stamp
    with pytest.raises(gate1.Gate1Error, match="another key"):
        gate1.run("T0", BODY, out, ["SV0"], ["F+"], (4,), ["walk/v0.4"], episodes=2, cut_s=0.12, **kw)
    # a condition the set has no reference for gets a "not run" record
    gate1.run("T0", BODY, out, ["SV0"], ["F+"], (4,), ["walk/v0.3"], episodes=2, cut_s=0.1, **kw)
    nr = s2io.read_json(out / "T0" / BODY / "seeds3042" / "gate1" / "SV0" / "F+@4ms" / "walk" / "v0.3.json")
    assert nr["status"] == "not run" and nr["why"]["reason"]
    assert not (tmp_path / "dev.json").exists()  # (T0 is no registered set: no development run is listed)


@needs_run
def test_a_cut_walk_episode_is_never_judged_and_never_run_again(tmp_path):
    """The lead's fifth list (the check's finding 4): a cut condition's episode is the served row's episode alone,
    never judged (the bar's judge reads no cut walk), never an episode that failed to run, so a later run of the
    same key keeps it; its record is read by no rule."""
    root = _t0(tmp_path / "sets")
    cut = tmp_path / "cut"
    kw = dict(root=root, log=quiet, max_episodes=CAP, dev_runs=tmp_path / "dev.json")
    gate1.run("T0", BODY, cut, ["SV0"], ["F+"], (4,), ["walk/v0.4"], episodes=1, cut_s=0.1, **kw)
    r = s2io.read_json(gate1.record_path(cut, "T0", BODY, gate1.HEADLINE, "SV0/F+@4ms", _cond(root)))
    e = r["episodes"][0]
    assert not e.get("error") and e["judged"] is False and "never judged" in e["not_judged"]
    assert e["succeeded"] is None and "measures" not in e and "trace" not in e and e["survival_s"] > 0.0
    assert e["servo"]["setting"] == "F+" and e["servo"]["readings"]["physics_steps"] == 50
    assert r["counts"]["errors"] == 0 and r["servo_readings"]["runs_without_readings"] == 0
    assert r["cut"] is True and r["reading"]["pass_"] is None and r["reading"]["judged"] is False
    again = gate1.run("T0", BODY, cut, ["SV0"], ["F+"], (4,), ["walk/v0.4"], episodes=1, cut_s=0.1, **kw)
    assert again["session"]["rerun_failed"] == [] and again["session"]["episodes_run"] == 0
    assert again["session"]["skipped"] == ["SV0/F+@4ms walk/v0.4"]


@needs_run
def test_the_second_block_takes_a_verdict_rows_drawn_panel_whose_headline_reading_passed(tmp_path, monkeypatch):
    """P41's repeat, in the bar's form (the lead's fifth list): a served stand-in row gives no verdict, so its D
    record is reported and never sent to the second block, even with a headline reading that passed; a row that
    gives a verdict (an SV5 row, once one exists: here SV0 taken as one) is sent when its headline reading passed."""
    root = _t0(tmp_path / "sets")
    out = tmp_path / "out"
    kw = dict(bases=["SV0"], settings=["F+", "D"], delays_ms=(4,), conditions=["walk/v0.4"], episodes=1, cut_s=0.1,
              root=root, log=quiet, max_episodes=CAP, dev_runs=tmp_path / "dev.json")
    gate1.run("T0", BODY, out, **kw)
    # a headline reading that passed (written by hand here: no cut condition is judged)
    head = gate1.record_path(out, "T0", BODY, gate1.HEADLINE, "SV0/D@4ms", _cond(root))
    rec = s2io.read_json(head)
    rec["reading"]["pass_"] = True
    s2io.write_json(head, rec)
    doc = gate1.run("T0", BODY, out, block=gate1.SECOND, **kw)
    assert doc["session"]["episodes_run"] == 0 and doc["session"]["not_selected"] == [
        f"SV0/D@4ms walk/v0.4: {gate1.NO_REPEAT}"]
    assert not gate1.record_path(out, "T0", BODY, gate1.SECOND, "SV0/D@4ms", _cond(root)).exists()
    rows = {(x["block"], x["row"]): x for x in doc["records"]}
    assert rows[(gate1.HEADLINE, "SV0/D@4ms")]["repeat"] == f"no repeat ({gate1.NO_REPEAT})"
    assert rows[(gate1.HEADLINE, "SV0/D@4ms")]["verdict"] is None
    # a row that gives a verdict: selected by its headline reading; F+ never has a second block
    monkeypatch.setattr(servo, "NO_VERDICT_CONTROLLERS", ())
    rec["reading"]["pass_"] = None
    s2io.write_json(head, rec)
    doc = gate1.run("T0", BODY, out, block=gate1.SECOND, **kw)
    assert doc["session"]["episodes_run"] == 0 and "not judged" in doc["session"]["not_selected"][0]
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
    """One whole episode of the served drawn panel through ``bar.run_episode`` (the row's episode, then the bar's
    judge)."""
    root = _t0(tmp_path / "sets")
    out = tmp_path / "out"
    gate1.run("T0", BODY, out, ["SV0"], ["D"], (4,), ["walk/v0.4"], episodes=1, root=root, log=quiet, max_episodes=1,
              dev_runs=tmp_path / "dev.json")
    r = s2io.read_json(gate1.record_path(out, "T0", BODY, gate1.HEADLINE, "SV0/D@4ms", _cond(root)))
    e = r["episodes"][0]
    assert not e.get("error") and "judged" not in e and e["seed"] == 3042 and r["cut"] is False
    n = e["servo"]["readings"]["physics_steps"]  # (the layer's steps: the episode's, to its end or its fall)
    assert e["servo"]["setting"] == "D/3042" and n > 0 and abs(n * 0.002 - e["survival_s"]) < 0.01
    assert set(e) >= {"succeeded", "failed_terms", "measures", "survival_s", "trace"} and e["trace"]["kept"] is False
    assert r["counts"]["episodes"] == 1 and r["reading"]["pass_"] is None  # (one seed: no block, not judged)
    assert not list((out / "T0" / BODY / gate1.HEADLINE / "gate1" / "traces").rglob("*.npz"))  # (hashed, not kept)


@needs_run
def test_the_cap_and_the_development_listing(tmp_path, monkeypatch):
    """The lead's fifth list (the check's finding 2): a run refuses a plan over its episode cap before anything is
    written or run; a run on V0, F25F9 or F25F9cl is a development run, listed with its command, set, rows, settings,
    seeds and count in this runner's own listing, before its episodes run and again when its records are written."""
    root = _t0(tmp_path / "sets")
    out, dev = tmp_path / "out", tmp_path / "dev.json"
    kw = dict(root=root, log=quiet, dev_runs=dev)
    args = ("T0", BODY, out, ["SV0"], ["F+", "M-"], (4,), ["walk/v0.4"])
    with pytest.raises(gate1.Gate1Error, match="over the cap of 3"):
        gate1.run(*args, episodes=2, cut_s=0.1, max_episodes=3, **kw)
    assert [f for f in _files(tmp_path) if not f.startswith("sets/")] == []
    # (T0 taken as a version of V0: the run is a development run, listed; the planner reads T0 as before)
    monkeypatch.setattr(gate1, "registered_set", lambda name: "V0" if name == "T0" else None)
    with pytest.raises(gate1.Gate1Error, match="over the cap"):
        gate1.run(*args, episodes=2, cut_s=0.1, max_episodes=0, **kw)
    assert not dev.exists()  # (a refused run runs nothing and is not listed)
    seen = []
    real = gate1.list_dev_run

    def spy(path, entry):
        seen.append((entry["state"], (out / "T0").exists() and bool(gate1.records(out, "T0", BODY))))
        return real(path, entry)

    monkeypatch.setattr(gate1, "list_dev_run", spy)
    doc = gate1.run(*args, episodes=2, cut_s=0.1, max_episodes=4, **kw)
    assert doc["session"]["episodes_run"] == 4
    assert seen == [("started", False), ("done", True)]  # (listed before any episode ran)
    d = s2io.read_json(dev)
    assert d["schema"] == gate1.DEV_RUNS_SCHEMA and d["rule"] == gate1.RULES["development"] and len(d["runs"]) == 1
    e = d["runs"][0]
    assert e["command"].startswith("python -m designlab.stage2.gate1 run --set T0 --body 75-M-H-10 --rows SV0 ")
    assert "--settings F+,M- --delays 4 --conditions walk/v0.4 --episodes 2" in e["command"]
    assert "--max-episodes 4" in e["command"] and "--cut-s 0.1" in e["command"]
    assert (e["set"], e["registered_set"], e["body"], e["block"]) == ("T0", "V0", BODY, gate1.HEADLINE)
    assert e["rows"] == ["SV0/F+@4ms", "SV0/M-@4ms"] and e["settings"] == ["F+", "M-"] and e["delays_ms"] == [4]
    assert e["seeds"] == [3042, 3043] and e["count"] == 4 and e["episodes_run"] == 4 and e["max_episodes"] == 4
    assert e["state"] == "done" and e["session"] == doc["session"]["id"] and e["ended_at"]
    # a second run of the same plan starts nothing and is listed all the same, as its own session
    again = gate1.run(*args, episodes=2, cut_s=0.1, max_episodes=0, command="by hand", **kw)
    d = s2io.read_json(dev)
    assert again["session"]["episodes_run"] == 0 and len(d["runs"]) == 2
    assert (d["runs"][1]["command"], d["runs"][1]["count"], d["runs"][1]["state"]) == ("by hand", 0, "done")


@needs_run
def test_a_change_to_servo_or_gate1_during_a_session_stops_the_run(tmp_path, monkeypatch):
    """The lead's fifth list (the check's finding 6): the worker compares servo.py and gate1.py with the session's
    hashes, taken when the session starts, and refuses to start on others."""
    root = _t0(tmp_path / "sets")
    real, calls = gate1.runner_sources, []

    def changing():
        calls.append(1)
        got = real()
        return got if len(calls) == 1 else dict(got, **{"designlab/stage2/servo.py": "0" * 64})

    monkeypatch.setattr(gate1, "runner_sources", changing)
    with pytest.raises(tune.PoolError, match="changed during the session"):
        gate1.run("T0", BODY, tmp_path / "out", ["SV0"], ["F+"], (4,), ["walk/v0.4"], episodes=1, cut_s=0.1,
                  root=root, log=quiet, max_episodes=CAP, dev_runs=tmp_path / "dev.json")
    assert len(calls) == 2 and gate1.records(tmp_path / "out", "T0", BODY) == []


def _cond(root):
    docs = setio.load(setio.set_dir("T0", BODY, root))
    return bar.choose(docs, gate1.HEADLINE, ["walk/v0.4"])[0]


# ------------------------------------------------------------------------------------------------
# The summary: the repeat and condition 2
# ------------------------------------------------------------------------------------------------
def _rec(out, block, base, setting, ms, passed, cond="walk/v0.3", counts=None, left=0):
    row = f"{base}/{setting}@{ms}ms"
    ctrl = base.split("/")[0]
    gives = ctrl not in servo.NO_VERDICT_CONTROLLERS
    doc = dict(schema=gate1.SCHEMA, development=True, status="done", block=block, row=row, base=base, setting=setting,
               delay_ms=ms, key_name=cond, counts=counts or dict(survived=20, succeeded=20, episodes=20),
               reading=dict(rule=gate1.rule_of(setting), pass_=passed),
               verdict=dict(value=passed if gives else None, reason=None),
               servo_readings=dict(runs_left_supply_window=left, inside_play_share_leg_mean=0.1))
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
    assert rows[(gate1.HEADLINE, "SV0/D@4ms")]["repeat"] == f"no repeat ({gate1.NO_REPEAT})"
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


def test_a_repeat_not_run_or_not_judged_is_pending(tmp_path, monkeypatch):
    """The lead's fifth list (the check's finding 1): a passing D record's verdict is True when replicated, False
    when not replicated, and pending (None) for anything else, a repeat that has not run or could not be judged
    among them, so that it never counts (C8); condition 2 then waits."""
    monkeypatch.setattr(gate1, "RULES_REGISTERED", True)
    out = tmp_path / "out"
    for base in ("SV5/R5/42", "SV5/R5/43"):
        for s in servo.JUDGED:
            for ms in gate1.DELAYS_MS:
                _rec(out, gate1.HEADLINE, base, s, ms, True)
                if s == "D" and (base, ms) != ("SV5/R5/43", 10):
                    _rec(out, gate1.SECOND, base, s, ms, True)

    def one():
        doc = gate1.summarize(out, "T0", BODY, write=False)
        x = {(r["block"], r["row"]): r for r in doc["records"]}[(gate1.HEADLINE, "SV5/R5/43/D@10ms")]
        return x["repeat"], x["verdict"], doc["condition2"]["holds"]

    assert gate1.REPEAT_VERDICT == {"replicated": True, "not replicated": False}
    assert one() == ("repeat not run", None, None)
    _rec(out, gate1.SECOND, "SV5/R5/43", "D", 10, None)  # (run, and not judged: a failed episode, C8)
    assert one() == ("repeat not judged", None, None)
    _rec(out, gate1.SECOND, "SV5/R5/43", "D", 10, False)
    assert one() == ("not replicated", False, False)
    _rec(out, gate1.SECOND, "SV5/R5/43", "D", 10, True)
    assert one() == ("replicated", True, True)
    # the headline reading of a D record that did not pass: no repeat, and the record's own verdict
    _rec(out, gate1.HEADLINE, "SV5/R5/43", "D", 10, False)
    assert one() == ("no repeat (the headline reading did not pass)", False, False)
