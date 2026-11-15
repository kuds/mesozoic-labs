"""Tests for the bar's tables (designlab/stage2/barreport.py): what is read, the words of every cell, the checks at
the top, the comparison with JPS section 2.3, and that the two files regenerate byte for byte.

Run from hardware/trex_form_study (the fit's venv):  python -m pytest -p no:cacheprovider tests/test_stage2_barreport.py -q
All fast.  Every record is synthetic: made with the runner's own functions (``bar.record_key``,
``bar.condition_record``, ``bar.not_run_record``, ``bar.record_path``) from episode dicts written here, in pytest's
temporary directory.  Nothing is simulated, no array is needed, and no number in this file is a result.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from designlab.stage1 import panels  # noqa: E402
from designlab.stage2 import bar, barreport, s2io, setio  # noqa: E402

BODY = "75-M-H-10"
H, R = bar.HEADLINE_BLOCK, bar.REPLICATION_BLOCK
TODAY = barreport._today_sources()
PROV = dict(git=dict(commit="0" * 40, dirty=False, dirty_paths=[]))
SESSION = dict(id="2026-11-15T08:00:00Z/1", started_at="2026-11-15T08:00:00Z", commit="0" * 40, dirty=False, wall_s=5.0)
MARGIN_RULE = "duty and lift set after results on seeds 3042-3061 (JPS 2.3)"
STAGE1_COMMAND = ("Stage 1's (decision 18): where the reference's path speed begins to ramp down; the reference "
                  "anticipates this command")
HEADING_RULE = "no reference within the heading rule"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ------------------------------------------------------------------------------------------------
# Synthetic sets, episodes and records
# ------------------------------------------------------------------------------------------------
class Row:
    """What the runner's record functions read of a row."""

    def __init__(self, name: str):
        self.name = name

    def describe(self) -> dict:
        return dict(row=self.name.rsplit("@", 1)[0], delayed_row=self.name, latency_ms=bar.delay_ms(self.name))

    def sources(self) -> dict:
        return {}

    def gated(self, cond) -> bool:
        return bool(cond.gated)


def set_docs(name: str, a1_over=()) -> dict:
    """A set's documents with one freeze that lists every entry."""
    walks = [setio.NOMINAL] + [setio.walk_name(v) for v in setio.SPEEDS]
    moves = [setio.START, setio.change_name(setio.NOMINAL, bar.BAR_WALK), setio.stop_name(2, bar.BAR_WALK),
             setio.stop_name(1, bar.BAR_WALK)] + [setio.turn_name(r, d) for r in setio.TURN_RATES for d in setio.DIRECTIONS]
    margin = MARGIN_RULE if name.startswith("F25F9cl") else None
    refs = {w: dict(file=f"refs/{w}.npz", sha256=sha(name + w), margin_rule=margin, label=None) for w in walks}
    trs = {m: dict(file=f"transitions/{m}.npz", sha256=sha(name + m), a1_over=m in a1_over, move=dict(causal=True))
           for m in moves}
    manifest = {n: dict(kind="walk" if n in refs else "move", sha256=e["sha256"], entry_sha256=sha("entry" + name + n))
                for n, e in {**refs, **trs}.items()}
    sd = dict(set=name, body=BODY, role="synthetic", rules={},
              freezes=[dict(index=0, at="2026-11-15T07:00:00Z", commit="a" * 40, manifest=manifest,
                            manifest_sha256=s2io.sha256_json(manifest))])
    return {"set.json": sd, "refs.json": {"references": refs}, "transitions.json": {"transitions": trs},
            "screen.json": None}


def episode(cond, seed: int, fell: bool = False, before=None, tail_over: bool = False, fail=(), error=None,
            speed=None, stage1: bool = True, touchdowns: int = 2, convention: str = "causal: the move's first frame") -> dict:
    """A synthetic episode record of a bar condition: it stays up and succeeds unless told otherwise; ``tail_over``
    fails the sustained-drive term in reading 2 only; ``speed`` is a walk's measured speed (default 3 % over the
    rung's target, so that no measured value equals a target)."""
    if error:
        return bar.failed_episode(cond, seed, error)
    terms, t3 = bar.terms_of(cond), bar.terms_but_a1(cond)
    s2 = {t: not fell and t not in fail and not (tail_over and t == "drive_sustained") for t in terms}
    s3 = {t: not fell and t not in fail for t in t3}
    tail = 1.21 if tail_over else 0.71
    span = dict(worst=["tail_yaw_servo", tail] if tail_over else ["left_knee_servo", 0.83],
                worst_leg=["left_knee_servo", 0.83], worst_a1=["tail_yaw_servo", tail],
                drive_sustained=not tail_over, drive_sustained_legs=True)
    j = cond.judge
    windowed = j in ("bar_walk", "bar_turn")
    m_bar: dict = dict(drives=dict(judged_span="window" if windowed else "episode",
                                   window=span if windowed and not fell else None, episode=span))
    measures: dict = dict(bar=m_bar, window=None, stage1=None, legs=None, push=None, recovery=None)
    toe = dict(toe=dict(limit_m=0.0200741), toe_clearance_own_swing_m=0.0236 + 1e-5 * (seed % 7),
               toe_clearance_own_duration_m=0.0171, stage1_toe_clearance_m=0.0228)
    if j == "bar_walk":
        v = setio.parse_name(cond.walk)["speed_m_s"] or 0.2431
        sp = 1.03 * v if speed is None else float(speed)
        measures["stage1"] = dict(succeeded=bool(stage1) and not fell, speed_target_m_s=v)
        if not fell:
            m_bar.update(toe, along_commanded_heading_m=10.0 * sp, lateral_from_line_max_m=0.051 + 0.001 * (seed % 7),
                         along_target_m=10.0 * v, line_max_m=0.25)
            measures["window"] = dict(speed_m_s=sp)
    elif j == "bar_turn" and not fell:
        m_bar.update(toe, heading_error_at_deg=-4.3 - 0.1 * (seed % 5), heading_error_hold_max_deg=6.2, tol_deg=10.0,
                     at_s=2.0, hold_s=3.0, time_to_within_10deg_s=1.71, turned_own_deg=86.4)
    elif j == "bar_nudge":
        measures["push"] = dict(force_N=[0.0, 21.87, 0.0], frac=float(cond.frac))
        measures["recovery"] = dict(after_s=None if fell else 1.23, limit_s=3.0, recovered=not fell)
    elif j == "bar_stop" and not fell:
        m_bar.update(touchdowns_after_command=int(touchdowns), forward_stopped_s=1.11, forward_stopped_strides=1.31,
                     stop_acceleration_1hz_m_s2=0.61, deceleration_raw_peak_m_s2=1.34, command_convention=convention)
    walk = j == "bar_walk"
    return dict(seed=int(seed), fell=bool(fell), fall_time_s=3.0 if fell else None, fall_cause="tilt" if fell else None,
                fall_onset_s=2.5 if fell else None, fall_candidates=["slip"] if fell else None,
                limits_before_onset=None, survival_s=3.0 if fell else 20.0, planned_s=20.0, strides=3.5 if fell else 24.0,
                success=s2, succeeded=all(s2.values()), success_but_A1=s3, succeeded_but_A1=all(s3.values()),
                failed_terms=["survived"] if fell else [t for t, v in s2.items() if not v],
                failed_terms_but_A1=["survived"] if fell else [t for t, v in s3.items() if not v],
                command_s=None if walk else 13.0, command_ref_s=None,
                fell_before_command=None if walk else (bool(before) if fell else False), segment_reached=cond.walk,
                window_reached=not fell, omega=None, crouch_frac=None, measures=measures, relaxed=[], wall_s=0.25)


def put(study: Path, out: Path, set_name: str, row: str, key: str, block: str = H, control=None, not_run=None,
        sources=None, a1_over=(), extra=None, per_seed=None, bad_input: bool = False, **kw) -> dict:
    """Write one synthetic record where the runner would (``bar.record_path``) and the set's ``set.json`` where the
    record's ``inputs.dir`` says; returns the record.  ``per_seed(seed)`` gives an episode's own arguments."""
    docs = set_docs(set_name, a1_over)
    s2io.write_json(study / "stage2" / "sets" / set_name / BODY / "set.json", docs["set.json"])
    cond = next(c for c in bar.conditions(docs, block, every=True) if c.key == key)
    seeds = list(bar.BLOCKS[block])
    row_ = Row(row)
    files = sorted({f"refs/{setio.NOMINAL}.npz", f"refs/{cond.walk}.npz"}
                   | ({f"transitions/{cond.move}.npz"} if cond.move else set()))
    fz = docs["set.json"]["freezes"][0]
    listed = {f: {k: fz["manifest"][Path(f).stem][k] for k in ("sha256", "entry_sha256")} for f in files}
    if bad_input:
        listed[files[-1]] = dict(listed[files[-1]], sha256=sha("another file"))
    inputs = dict(dir=f"stage2/sets/{set_name}/{BODY}", files=listed,
                  freeze=dict(index=0, manifest_sha256=fz["manifest_sha256"]))
    src = dict(TODAY, **(sources or {}))
    more = dict(extra or {})
    if control:
        more.update(control=bar.CONTROLS[control][1], control_name=control)
    if not_run:
        k = bar.record_key(BODY, set_name, block, row_, cond, None, seeds, inputs, {}, src, None, None)
        rec = bar.not_run_record(BODY, set_name, block, row_, cond, k, not_run, PROV, inputs, docs, more, control)
    else:
        command = None
        if cond.move:
            conv = STAGE1_COMMAND if cond.name == "one_step" else "causal: the move's first frame"
            command = dict(name=cond.move, ref_s=14.5, phase=0.0, convention=conv)
            if cond.judge == "bar_stop":
                kw.setdefault("convention", conv)
        resolved = dict(plan=[["transition", setio.START], ["ref", cond.walk, 20]], seconds=20.0, files=files,
                        command=command)
        crouch = bar.CONTROLS[control][0] if control else 0.05
        k = bar.record_key(BODY, set_name, block, row_, cond, resolved, seeds, inputs, {}, src,
                           4.8281 if control else 4.8864, crouch)
        eps = [episode(cond, s, **{**kw, **(per_seed(s) if per_seed else {})}) for s in seeds]
        rec = bar.condition_record(BODY, set_name, block, row_, cond, eps, k, seeds, PROV, inputs, [SESSION], docs,
                                   extra=more, control=control)
    s2io.write_json(bar.record_path(out, set_name, BODY, block, row, cond, control), rec)
    return rec


def run_path(set_name: str, row: str, key: str, block: str = H, control=None) -> str:
    """A record's path relative to the run directory, as ``budget.json`` lists it."""
    return (bar.block_dir(Path(), set_name, BODY, block, control) / "eval" / row / f"{key}.json").as_posix()


def build(study: Path, out: Path) -> None:
    """The synthetic run the tests read (comments say what each record is there for)."""
    cl, v0, f25 = "F25F9cl", "V0", "F25F9"
    few = lambda n: (lambda s: dict(fail=("duty",)) if s % 1000 < 42 + n else {})  # noqa: E731  (n episodes fail a term)
    # -- the candidate's walk at 10 ms: one row passes both readings, the other only but for A1 (the tail is over)
    put(study, out, cl, "R2/42@10ms", "walk/v0.4")
    put(study, out, cl, "R2/43@10ms", "walk/v0.4", tail_over=True)
    put(study, out, cl, "R0@10ms", "walk/v0.4", per_seed=lambda s: dict(fell=s < 3045))
    # ... replicated: R2/42 passes again, R2/43 does not (reading 3 triggered it)
    rep = lambda row, key: dict(replication_of=dict(  # noqa: E731
        block=H, record=run_path(cl, row, key), readings=["success", "but_A1"] if row == "R2/42@10ms" else ["but_A1"]))
    put(study, out, cl, "R2/42@10ms", "walk/v0.4", block=R, extra=rep("R2/42@10ms", "walk/v0.4"))
    put(study, out, cl, "R2/43@10ms", "walk/v0.4", block=R, extra=rep("R2/43@10ms", "walk/v0.4"), tail_over=True,
        per_seed=few(10))
    # -- on the low grip both rows pass; the ledger cut their replication
    put(study, out, cl, "R2/42@10ms", "walk/v0.4_mu0.54")
    put(study, out, cl, "R2/43@10ms", "walk/v0.4_mu0.54", per_seed=few(3))
    # -- 4 ms: both pass (no verdict word at a reported delay)
    put(study, out, cl, "R2/42@4ms", "walk/v0.4")
    put(study, out, cl, "R2/43@4ms", "walk/v0.4")
    # -- the ladder and the pull request's rows at 0 ms
    for row in ("R2/42@0ms", "R2/43@0ms"):
        put(study, out, cl, row, "walk/v0.4", speed=0.4151)
        put(study, out, cl, row, "walk/v0.45", speed=0.4662)
        put(study, out, cl, row, "walk/v0.5", speed=0.5139)
        put(study, out, v0, row, "walk/v0.35", speed=0.3661, tail_over=True, stage1=False)
    put(study, out, cl, "R2/42@10ms", "walk/v0.45", per_seed=lambda s: dict(fell=s < 3050))
    put(study, out, cl, "R2/43@10ms", "walk/v0.45", per_seed=lambda s: dict(fell=s < 3050))
    # -- the rest of the bar at 10 ms
    turn_l = setio.turn_name(setio.QUICK, "left")
    for row in ("R2/42@10ms", "R2/43@10ms"):  # a turn frozen with a1_over: reading 2 is not read
        put(study, out, cl, row, "turn/q90_left", a1_over=(turn_l,), tail_over=True)
        put(study, out, cl, row, "turn/q90_right", not_run=dict(  # no reference at all
            reason="no reference within the thigh twist and the leg drives", reference="transitions/turn90_pi4_right",
            numbers=dict(range_use=0.91, limit=0.87, f=0.6)))
        put(study, out, cl, row, "turn/t90_1_left", not_run=dict(  # a ladder rate outside the heading rule
            reason=HEADING_RULE, reference="transitions/turn90_1_left",
            numbers=dict(heading_at_slowest_clock_deg=77.7, limit_deg=80.0)))
        put(study, out, cl, row, "stop/one_step", per_seed=lambda s: dict(touchdowns=1))
        put(study, out, cl, row, "nudge/forward", per_seed=lambda s: dict(fell=s < 3044, before=s < 3043))
    put(study, out, cl, "R2/42@10ms", "nudge/left")  # its other row has no record
    # -- a re-run after a result: the registered record (it fails) is kept beside the later one (it passes)
    put(study, out, cl, "R2/43@10ms", "stop/two_step")
    put(study, out, cl, "R2/42@10ms", "stop/two_step", per_seed=lambda s: dict(touchdowns=3,
                                                                              fail=("touchdowns_after_command",)))
    path = bar.record_path(out, cl, BODY, H, "R2/42@10ms", SimpleNamespace(behavior="stop", name="two_step"))
    field = bar.supersede_record(path, json.loads(path.read_text()), "synthetic: run again to test the label")
    put(study, out, cl, "R2/42@10ms", "stop/two_step", extra=dict(supersedes=field))
    # -- the omega control
    put(study, out, cl, "R2/42@10ms", "walk/v0.4", control="omega_s1")
    # -- the controls of the round
    put(study, out, v0, "R2/42@10ms", "walk/v0.4", per_seed=lambda s: dict(fell=True))
    put(study, out, v0, "R2/43@10ms", "walk/v0.4", per_seed=lambda s: dict(fell=True),
        sources=dict(bar=sha("another bar.py")))  # made by other sources: listed, flagged, tabulated
    put(study, out, v0, "R0@10ms", "walk/v0.4", bad_input=True)  # an input that the latest freeze does not list
    put(study, out, f25, "R0@10ms", "walk/v0.4", per_seed=lambda s: dict(error="synthetic failure") if s == 3050 else {})
    # -- the ledger: task 3 started two units and cut a third; task 4 has none
    started = [run_path(cl, r, k) for r in ("R2/42@10ms", "R2/43@10ms", "R0@10ms") for k in ("walk/v0.4", "walk/v0.4_mu0.54")]
    started += [run_path(f25, r, "walk/v0.4") for r in ("R2/42@10ms", "R2/43@10ms", "R0@10ms")]
    unit = lambda name, paths, used: dict(  # noqa: E731
        unit=name, estimate_core_h=0.4, spent_before_core_h=0.0, started_at="2026-11-15T08:00:00Z",
        used_core_h=used / 3600.0, items={"episodes": float(used)}, detail=dict(paths=paths))
    cut = dict(unit="T3.2b", estimate_core_h=0.3, spent_before_core_h=15.9, at="2026-11-15T20:00:00Z",
               reason="spent 15.900 + estimate 0.300 core-h exceeds task 3's 16 core-h",
               label="cut by the pre-registered order",
               detail=dict(paths=[run_path(cl, r, "walk/v0.4_mu0.54", R) for r in ("R2/42@10ms", "R2/43@10ms")]))
    s2io.write_json(out / "budget.json", dict(schema="bar-budget-1", rule=bar.RULES["budget"], tasks={"3": dict(
        cap_core_h=16.0, cut=[cut], units=[
            unit("T3.1", started, 1800.0),
            unit("T3.2a", [run_path(cl, r, "walk/v0.4", R) for r in ("R2/42@10ms", "R2/43@10ms")], 5400.0)])}))


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    study = tmp_path_factory.mktemp("study")
    out = study / "stage2" / "runs" / "synthetic"
    build(study, out)
    doc = barreport.load(out, study=study)
    tabs = barreport.tables(doc)
    return SimpleNamespace(study=study, out=out, doc=doc, tabs=tabs, md=barreport.markdown(tabs))


def part(tabs: dict, table: str, key: str) -> dict:
    return next(p for p in tabs[table]["parts"] if p["key"] == key)


def lines(tabs: dict, table: str, key: str, **where) -> list[dict]:
    return [x for x in part(tabs, table, key)["lines"] if all(x.get(k) == v for k, v in where.items())]


def one(tabs: dict, table: str, key: str, **where) -> dict:
    found = lines(tabs, table, key, **where)
    assert len(found) == 1, (where, found)
    return found[0]


def every_line(tabs: dict):
    """(table, part, line) of every line of every table."""
    for name in barreport.ORDER:
        for p in tabs[name]["parts"]:
            for x in p["lines"]:
                yield name, p, x


def strings(x):
    if isinstance(x, dict):
        for v in x.values():
            yield from strings(v)
    elif isinstance(x, list):
        for v in x:
            yield from strings(v)
    elif isinstance(x, str):
        yield x


J, REP4 = "10 ms (judged)", "4 ms (reported)"


# ------------------------------------------------------------------------------------------------
# Reading
# ------------------------------------------------------------------------------------------------
def test_load_indexes_every_record_by_set_body_block_row_condition(run):
    doc = run.doc
    rec = doc["F25F9cl"][BODY][H]["R2/43@10ms"]["walk/v0.4"]
    assert rec["counts"]["succeeded"] == 0 and rec["counts_but_A1"]["succeeded"] == 20
    assert rec["_path"] == "stage2/runs/synthetic/F25F9cl/75-M-H-10/seeds3042/eval/R2/43@10ms/walk/v0.4.json"
    assert set(doc) == {"F25F9cl", "F25F9", "V0", barreport.META}
    files = [q for q in run.out.rglob("*.json") if "eval" in q.parts and not any(x.endswith(".superseded") for x in q.parts)]
    main = [q for q in files if "controls" not in q.parts]
    assert len(list(barreport._records(doc))) == len(main) == run.tabs["checks"]["records"]["found"]
    meta = doc[barreport.META]
    assert list(meta["controls"]) == ["omega_s1"] and len(files) - len(main) == run.tabs["checks"]["records"]["controls"] == 1
    assert meta["out"] == "stage2/runs/synthetic" and meta["unreadable"] == [] and meta["misplaced"] == []
    # the record kept beside the re-run is read with it, and is no record of the index
    kept = doc["F25F9cl"][BODY][H]["R2/42@10ms"]["stop/two_step"]["_superseded"]
    assert len(kept) == 1 and kept[0]["record"]["counts"]["succeeded"] == 0 and kept[0]["reason"]["sequence"] == 1


def test_load_keeps_the_sets_and_the_body_asked_for(run):
    doc = barreport.load(run.out, sets=["V0"], study=run.study)
    assert set(doc) == {"V0", barreport.META} and doc[barreport.META]["controls"] == {"omega_s1": {}}
    assert set(barreport.load(run.out, body="60-N-T-10", study=run.study)) == {barreport.META}
    tabs = barreport.tables(doc)
    assert {x["set"] for _, _, x in every_line(tabs) if "set" in x} == {"V0"}
    with pytest.raises(s2io.S2Error, match="no such run directory"):
        barreport.load(run.out / "absent", study=run.study)


def test_files_that_are_no_record_are_listed_and_a_summary_is_skipped(run, tmp_path):
    out = tmp_path / "study" / "runs" / "x"
    put(tmp_path / "study", out, "V0", "R0@10ms", "walk/v0.4")
    edir = bar.block_dir(out, "V0", BODY, H) / "eval"
    bar.summarize(out, "V0", BODY, H)  # the runner's summary.json is no record
    (edir / "R0@10ms" / "walk" / "torn.json").write_text("{")
    s2io.write_json(edir / "R0@10ms" / "walk" / "other.json", dict(schema="panels-1"))
    (edir / "traces" / "R0@10ms").mkdir(parents=True)
    s2io.write_json(edir / "traces" / "R0@10ms" / "x.json", dict(schema=bar.SCHEMA))
    doc = barreport.load(out, study=tmp_path / "study")
    assert (edir / "summary.json").is_file() and len(list(barreport._records(doc))) == 1
    assert sorted(Path(p).name for p in doc[barreport.META]["unreadable"]) == ["other.json", "torn.json"]
    assert "torn.json" in barreport.markdown(barreport.tables(doc))


# ------------------------------------------------------------------------------------------------
# Byte for byte
# ------------------------------------------------------------------------------------------------
def test_the_tables_regenerate_byte_for_byte(run, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    tabs = barreport.write(run.out, a, study=run.study)
    barreport.write(run.out, b, study=run.study)
    for name in (barreport.TABLES_JSON, barreport.TABLES_MD):
        assert (a / name).read_bytes() == (b / name).read_bytes()
    text = (a / barreport.TABLES_JSON).read_text()
    assert json.loads(text) == tabs == run.tabs
    # BAR_TABLES.md is a function of bar_tables.json alone
    assert barreport.markdown(json.loads(text)) == (a / barreport.TABLES_MD).read_text() == run.md
    assert not list(a.glob("*.tmp"))


def test_the_tables_hold_no_time_and_no_host(run, tmp_path):
    """The same records under another directory, with other times, hosts and commits in them, give the same bytes."""
    study = tmp_path / "elsewhere"
    shutil.copytree(run.study, study)
    for q in (study / "stage2" / "runs").rglob("*.json"):
        doc = json.loads(q.read_text())
        if doc.get("schema") == bar.SCHEMA:
            doc["ended_at"] = "2031-01-01T00:00:00Z"
            doc["provenance"] = dict(git=dict(commit="f" * 40, dirty=True), host="another")
            doc["sessions"] = [dict(s, started_at="2031-01-01T00:00:00Z", commit="f" * 40) for s in doc.get("sessions") or []]
            q.write_text(json.dumps(doc))
    tabs = barreport.tables(barreport.load(study / "stage2" / "runs" / "synthetic", study=study))
    assert barreport.markdown(tabs) == run.md
    assert str(run.study) not in run.md and str(tmp_path) not in run.md
    assert "2031" not in run.md and "provenance" not in json.dumps(run.tabs)


def test_cli_writes_both_files(run, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(s2io, "STUDY", run.study)  # the study directory the paths are relative to
    assert barreport.main(["--out", str(run.out), "--dest", str(tmp_path / "d"), "--set", "V0", "--set", "F25F9"]) == 0
    printed = capsys.readouterr().out
    assert "records found" in printed and str(tmp_path / "d" / barreport.TABLES_MD) in printed
    tabs = json.loads((tmp_path / "d" / barreport.TABLES_JSON).read_text())
    assert tabs["checks"]["sets_asked"] == ["V0", "F25F9"] and tabs["schema"] == barreport.SCHEMA
    assert barreport.markdown(tabs) == (tmp_path / "d" / barreport.TABLES_MD).read_text()
    assert barreport.main(["--out", str(tmp_path / "none")]) == 1


# ------------------------------------------------------------------------------------------------
# Readings, verdicts, delays
# ------------------------------------------------------------------------------------------------
def test_the_two_readings_are_never_pooled_and_differ_where_only_the_tail_is_over(run):
    x = one(run.tabs, "walk", "counts", set="F25F9cl", condition="walk/v0.4", delay=J, row="R2/43@10ms")
    assert x["r2"] == dict(k=0, n=20) and x["r3"] == dict(k=20, n=20) and x["up"] == dict(k=20, n=20)
    assert x["r2_pass"] == "does not pass" and x["r3_pass"] == "passes"
    assert x["r2_lcb"] == 0.0 and x["r3_lcb"] == pytest.approx(panels.lcb(20, 20))  # the record's own bounds
    assert x["reading_1"] == barreport.READING_1 and x["stage1"] == dict(k=20, n=20)
    f = one(run.tabs, "walk", "fails", set="F25F9cl", condition="walk/v0.4", delay=J, row="R2/43@10ms")
    assert f["failed_2"] == {"drive_sustained": 20} and f["failed_3"] == {}
    assert f["worst_drive"] == "tail_yaw_servo" and f["worst_ratio"] == 1.21 and f["worst_leg_ratio"] == 0.83
    # no count of any table pools readings or blocks: every "k of n" has n <= 20
    for _, p, line in every_line(run.tabs):
        for c in p["columns"]:
            v = line.get(c["key"])
            if c["kind"] == "of" and v is not None:
                assert v["n"] <= bar.RUNS and v["k"] <= v["n"], (p["key"], c["key"], v)
    # the readings stand in the plan's order in every table that has them
    for name in barreport.ORDER:
        for p in run.tabs[name]["parts"]:
            titles = [c["title"] for c in p["columns"]]
            order = [i for t in ("Reading 1", "Reading 2", "Reading 3") for i, x_ in enumerate(titles) if x_.startswith(t)][:3]
            assert order == sorted(order), (name, p["key"])


def test_shown_needs_both_rows_and_provisional_one(run):
    v = one(run.tabs, "walk", "verdicts", set="F25F9cl", condition="walk/v0.4", delay=J)
    assert v["reading_2"].startswith("provisional [") and v["reading_3"].startswith("shown but for A1 [")
    low = one(run.tabs, "walk", "verdicts", set="F25F9cl", condition="walk/v0.4_mu0.54", delay=J)
    assert low["reading_2"].startswith("shown [") and low["reading_3"].startswith("shown but for A1 [")
    none = one(run.tabs, "walk", "verdicts", set="V0", condition="walk/v0.4", delay=J)
    assert not none["reading_2"].startswith(("shown", "provisional")) and none["reading_2"] == "not attributed"
    # the words are the runner's own (bar.verdict), cell by cell
    doc = run.doc
    for ckey, ms, want2, want3 in (("walk/v0.4", 10, "provisional", "shown but for A1"),
                                   ("walk/v0.4_mu0.54", 10, "shown", "shown but for A1"),
                                   ("walk/v0.4", 4, "passes at 4 ms", "passes at 4 ms but for A1")):
        got = barreport.verdict_of(doc, "F25F9cl", BODY, ckey, ms)
        cells = [bar._cell(doc["F25F9cl"][BODY][H][f"{r}@{ms}ms"][ckey]) for r in barreport.R2_ROWS]
        assert got["reading_2"]["verdict"] == bar.verdict(cells, ms, "success")["verdict"] == want2
        assert got["reading_3"]["verdict"] == bar.verdict(cells, ms, "but_A1")["verdict"] == want3


def test_an_r0_row_gives_no_verdict_and_one_row_alone_is_incomplete(run):
    assert all(x["controller"].startswith("R2") for x in part(run.tabs, "walk", "verdicts")["lines"])
    r0 = one(run.tabs, "walk", "counts", set="F25F9cl", condition="walk/v0.4", delay=J, row="R0@10ms")
    assert r0["up"] == dict(k=17, n=20) and r0["r2_pass"] == "does not pass"
    g = one(run.tabs, "glance", "glance", set="F25F9cl", condition="walk/v0.4")
    assert g["r0"] == barreport.NO_VERDICT_R0 and g["r0_up"] == dict(k=17, n=20)
    assert not any(x["row"].startswith("R0") for x in part(run.tabs, "replication", "replication")["lines"])
    alone = one(run.tabs, "rest", "verdicts", set="F25F9cl", condition="nudge/left", delay=J)
    assert alone["reading_2"] == "incomplete (R2/43@10ms: not run)"


def test_the_judged_delay_is_marked_and_another_delay_gives_no_verdict_word(run):
    delays = {x["delay"] for _, p, x in every_line(run.tabs) if "delay" in x}
    assert J in delays and REP4 in delays and all(d.endswith(("(judged)", "(reported)")) for d in delays)
    assert all(d.startswith("10 ms") == d.endswith("(judged)") for d in delays)
    for name in ("walk", "ladder", "rest"):  # the judged delay first within each set
        seen: dict = {}
        for x in part(run.tabs, name, "counts")["lines"]:
            seen.setdefault(x["set"], []).append(x["delay"])
        for ds in seen.values():
            if J in ds:
                assert ds[0] == J and J not in ds[ds.count(J):]
    v = one(run.tabs, "walk", "verdicts", set="F25F9cl", condition="walk/v0.4", delay=REP4)
    assert v["reading_2"] == "passes at 4 ms" and v["reading_3"] == "passes at 4 ms but for A1"
    x = one(run.tabs, "walk", "counts", set="F25F9cl", condition="walk/v0.4", delay=REP4, row="R2/42@4ms")
    assert x["r2_pass"] == "passes at 4 ms"
    for _, p, line in every_line(run.tabs):  # "shown" and "provisional" stand at the judged delay only
        if p["key"] == "verdicts" and line["delay"] != J:
            assert not line["reading_2"].startswith(("shown", "provisional")), line
            assert not line["reading_3"].startswith(("shown", "provisional")), line


def test_a_turn_frozen_with_a1_over_is_not_read_in_reading_2(run):
    x = one(run.tabs, "rest", "counts", set="F25F9cl", condition="turn/q90_left", delay=J, row="R2/42@10ms")
    assert x["r2_pass"] == "not read: a1_over" and x["r2"] == dict(k=0, n=20) and x["r3_pass"] == "passes"
    v = one(run.tabs, "rest", "verdicts", set="F25F9cl", condition="turn/q90_left", delay=J)
    assert v["reading_2"] == "not read: a1_over" and v["reading_3"].startswith("shown but for A1 [")
    g = one(run.tabs, "glance", "glance", set="F25F9cl", condition="turn/q90_left")
    assert g["reading_2"] == "not read: a1_over"
    # it is no pass: reading 3 alone triggers a replication line
    r = one(run.tabs, "replication", "replication", set="F25F9cl", condition="turn/q90_left", row="R2/42@10ms")
    assert r["triggered_by"] == ["reading 3 (every limit but for A1)"]


# ------------------------------------------------------------------------------------------------
# Not run, cut, re-run
# ------------------------------------------------------------------------------------------------
def test_a_missing_record_reads_not_run_and_a_unit_the_ledger_never_started_reads_cut(run):
    meta = run.doc[barreport.META]
    word = lambda s, row, key, block=H: barreport.absent_word(meta, s, BODY, block, row, key)  # noqa: E731
    # task 4 has no cut unit: a missing record is "not run"
    assert word("F25F9cl", "R2/43@10ms", "nudge/left") == barreport.NOT_RUN
    assert word("V0", "R2/42@10ms", "stop/two_step") == barreport.NOT_RUN
    # task 3 cut a unit: its record paths read "cut" ...
    assert word("F25F9cl", "R2/42@10ms", "walk/v0.4_mu0.54", R) == barreport.CUT
    # ... and so does every cell of the registered order that no started unit lists (the units after the cut one)
    assert word("V0", "R2/42@2ms", "walk/v0.4") == word("F25F9cl", "R2/42@6ms", "walk/v0.5") == barreport.CUT
    # a cell of a started unit whose record is not there, and a cell outside the registered order, are "not run"
    assert word("F25F9", "R2/42@10ms", "walk/v0.4") == barreport.NOT_RUN
    assert word("F25F9cl", "R2/42@8ms", "walk/v0.5") == barreport.NOT_RUN
    assert word("D0", "R2/42@10ms", "walk/v0.4") == barreport.NOT_RUN
    # in the tables: a whole set and delay without a record is one line with its word
    v2 = one(run.tabs, "walk", "counts", set="V0", delay="2 ms (reported)")
    assert v2["status"] == barreport.CUT and v2["condition"] == "walk/v0.4, walk/v0.4_mu0.54"
    assert v2["row"] == "R2/42@2ms, R2/43@2ms, R0@2ms" and v2.get("up") is None
    assert one(run.tabs, "walk", "verdicts", set="V0", delay="2 ms (reported)")["reading_2"] == barreport.CUT
    r4 = one(run.tabs, "rest", "counts", set="V0", delay=REP4)
    assert r4["status"] == barreport.NOT_RUN and r4["item"] == "14 conditions, 42 records: none found"
    # a single missing record among others has its own line and path
    f = one(run.tabs, "walk", "counts", set="F25F9", condition="walk/v0.4", delay=J, row="R2/42@10ms")
    assert f["status"] == barreport.NOT_RUN and f["path"].endswith("F25F9/75-M-H-10/seeds3042/eval/R2/42@10ms/walk/v0.4.json")
    assert one(run.tabs, "walk", "verdicts", set="F25F9", condition="walk/v0.4", delay=J)["reading_2"] == barreport.NOT_RUN
    assert f"| {barreport.CUT} |" in run.md and f"| {barreport.NOT_RUN} |" in run.md
    # without a ledger nothing is "cut"
    doc = copy.deepcopy(run.doc)
    doc[barreport.META]["budget"] = None
    assert not [s for _, _, x in every_line(barreport.tables(doc)) for s in strings(x) if barreport.CUT in s]


def test_a_condition_without_a_reference_prints_its_reason_and_the_references_numbers(run):
    x = one(run.tabs, "rest", "counts", set="F25F9cl", condition="turn/t90_1_left", row="R2/42@10ms")
    assert x["status"] == (f"not run: {HEADING_RULE} [transitions/turn90_1_left] "
                           "(heading_at_slowest_clock_deg=77.7, limit_deg=80)")
    assert x["up"] is None and x["r2"] is None and x["r2_pass"] is None  # never a blank, never a fail of the robot
    v = one(run.tabs, "rest", "verdicts", set="F25F9cl", condition="turn/t90_1_left")
    assert v["reading_2"] == v["reading_3"] == x["status"]
    assert not lines(run.tabs, "rest", "fails", condition="turn/t90_1_left")
    g = one(run.tabs, "glance", "glance", set="F25F9cl", condition="turn/q90_right")
    assert g["reading_2"].startswith("not run: no reference within the thigh twist and the leg drives")
    assert "range_use=0.91" in g["reading_2"]
    tally = one(run.tabs, "glance", "tally", set="F25F9cl")
    assert sum(tally["reading_2"].values()) == sum(tally["reading_3"].values()) == 16
    assert tally["reading_2"] == {"incomplete": 1, "not attributed": 1, "not read: a1_over": 1, "not run": 10,
                                  "provisional": 2, "shown": 1}
    assert tally["reading_3"]["shown but for A1"] == 3 and tally["reading_3"]["provisional but for A1"] == 1
    # the control for the walk only: its 14 other gated conditions say why they have no record
    f = one(run.tabs, "glance", "glance", set="F25F9", condition="stop/two_step")
    assert f["reading_2"] == barreport.NOT_RUN and f["note"] == barreport.WALK_ONLY_NOTE
    assert one(run.tabs, "glance", "glance", set="V0", condition="stop/two_step")["note"] is None
    assert len(lines(run.tabs, "glance", "glance", set="V0")) == 16


def test_a_rerun_after_a_result_stands_beside_the_registered_record(run):
    two = lines(run.tabs, "rest", "counts", set="F25F9cl", condition="stop/two_step", row="R2/42@10ms")
    assert len(two) == 2
    later, first = two
    assert any(x.startswith(barreport.RERUN + ": synthetic: run again") for x in later["labels"])
    assert later["r2"] == dict(k=20, n=20) and ".superseded" not in later["path"]
    assert any(x.startswith(barreport.REGISTERED) for x in first["labels"]) and first["r2"] == dict(k=0, n=20)
    assert ".superseded/" in first["path"] and barreport.MARGIN_LABEL in first["labels"]
    # the verdict is the registered records'; the re-run's is printed beside it, labelled
    reg, re_ = lines(run.tabs, "rest", "verdicts", set="F25F9cl", condition="stop/two_step")
    assert reg["which"] == "registered records" and reg["reading_2"].startswith("provisional [")
    assert one(run.tabs, "walk", "verdicts", set="F25F9", condition="walk/v0.4", delay=J)["which"] == "no record"
    assert re_["which"] == barreport.RERUN and re_["reading_2"].startswith("shown [") and barreport.RERUN in re_["labels"]
    g = one(run.tabs, "glance", "glance", set="F25F9cl", condition="stop/two_step")
    assert g["reading_2"].startswith("provisional [")
    assert run.tabs["checks"]["records"]["kept_beside_a_rerun"] == 1
    # where the kept file cannot be read, the counts copied into the later record stand in
    rec = copy.deepcopy(run.doc["F25F9cl"][BODY][H]["R2/42@10ms"]["stop/two_step"])
    kept, rec["_superseded"] = rec["_superseded"], []
    brief = barreport.registered(rec)
    assert brief["_brief"] and brief["counts"]["succeeded"] == 0 and brief["counts"]["pass_"] is False
    # a later record that names no registered one (its reason file was lost): the oldest record kept beside it
    rec["_superseded"], rec["supersedes"] = kept, dict(reason=None, label=barreport.RERUN)
    assert barreport.registered(rec)["counts"]["succeeded"] == 0 and ".superseded/" in barreport.registered(rec)["_path"]
    assert barreport.registered(None) is None and barreport.registered(kept[0]["record"]) is kept[0]["record"]


def test_a_rerun_of_the_second_block_does_not_change_the_replication_word(run, tmp_path):
    study = tmp_path / "study"
    out = study / "runs" / "x"
    put(study, out, "F25F9cl", "R2/42@10ms", "walk/v0.4")
    put(study, out, "F25F9cl", "R2/42@10ms", "walk/v0.4", block=R, per_seed=lambda s: dict(fell=True))
    path = bar.record_path(out, "F25F9cl", BODY, R, "R2/42@10ms", SimpleNamespace(behavior="walk", name="v0.4"))
    field = bar.supersede_record(path, json.loads(path.read_text()), "synthetic: the second block run again")
    put(study, out, "F25F9cl", "R2/42@10ms", "walk/v0.4", block=R, extra=dict(supersedes=field))
    doc = barreport.load(out, study=study)
    assert barreport.replication_word(doc, "F25F9cl", BODY, "R2/42@10ms", "walk/v0.4", "success") == "not replicated"
    assert bar.replication_word(out, "F25F9cl", BODY, "R2/42@10ms", "walk/v0.4", "success") == "replicated"  # the later one
    first, later = part(barreport.tables(doc), "replication", "replication")["lines"]
    assert first["b_up"] == dict(k=0, n=20) and first["words"][0].endswith(": not replicated")
    assert any(x.startswith(barreport.REGISTERED) for x in first["labels"]) and ".superseded/" in first["b_path"]
    assert later["b_up"] == dict(k=20, n=20) and later["words"] == [barreport.RERUN + ": the registered record's word stands"]
    assert any(x.startswith(barreport.RERUN + ": synthetic") for x in later["labels"])


def test_a_rerun_that_has_not_finished_leaves_the_registered_record(run, tmp_path):
    study = tmp_path / "study"
    out = study / "runs" / "x"
    put(study, out, "F25F9cl", "R2/42@10ms", "walk/v0.4")
    path = bar.record_path(out, "F25F9cl", BODY, H, "R2/42@10ms", SimpleNamespace(behavior="walk", name="v0.4"))
    bar.supersede_record(path, json.loads(path.read_text()), "synthetic: killed before it finished")
    assert not path.exists()
    doc = barreport.load(out, study=study)
    rec = doc["F25F9cl"][BODY][H]["R2/42@10ms"]["walk/v0.4"]
    assert rec["_rerun_pending"] and ".superseded/" in rec["_path"]
    x = one(barreport.tables(doc), "walk", "counts", set="F25F9cl", condition="walk/v0.4", row="R2/42@10ms")
    assert x["r2"] == dict(k=20, n=20) and any("has no record yet" in s for s in x["labels"])


# ------------------------------------------------------------------------------------------------
# Replication
# ------------------------------------------------------------------------------------------------
def test_replication_sits_beside_its_original_and_is_never_pooled(run):
    a = one(run.tabs, "replication", "replication", set="F25F9cl", condition="walk/v0.4", row="R2/42@10ms")
    assert a["a_r2"] == a["b_r2"] == dict(k=20, n=20)
    assert a["a_path"].endswith("seeds3042/eval/R2/42@10ms/walk/v0.4.json")
    assert a["b_path"].endswith("seeds5042/eval/R2/42@10ms/walk/v0.4.json")
    assert a["triggered_by"] == ["reading 2 (every limit)", "reading 3 (every limit but for A1)"]
    assert a["words"] == ["reading 2 (every limit): replicated", "reading 3 (every limit but for A1): replicated"]
    b = one(run.tabs, "replication", "replication", set="F25F9cl", condition="walk/v0.4", row="R2/43@10ms")
    assert b["a_r3"] == dict(k=20, n=20) and b["b_r3"] == dict(k=10, n=20) and b["b_r2"] == dict(k=0, n=20)
    assert b["triggered_by"] == ["reading 3 (every limit but for A1)"]
    assert b["words"] == ["reading 3 (every limit but for A1): not replicated"]
    # a passing record whose replication the ledger cut is listed with that word and no second block
    c = one(run.tabs, "replication", "replication", set="F25F9cl", condition="walk/v0.4_mu0.54", row="R2/42@10ms")
    assert c["b_up"] is None and c["b_r2"] is None and all(w.endswith(bar.REPLICATION_WORDS[2]) for w in c["words"])
    assert c["b_path"].endswith("seeds5042/eval/R2/42@10ms/walk/v0.4_mu0.54.json")
    # the candidate's verdict carries the word of each row's record that passed
    v = one(run.tabs, "walk", "verdicts", set="F25F9cl", condition="walk/v0.4", delay=J)
    assert v["reading_2"] == "provisional [R2/42@10ms: replicated]"
    assert v["reading_3"] == "shown but for A1 [R2/42@10ms: replicated; R2/43@10ms: not replicated]"
    low = one(run.tabs, "walk", "verdicts", set="F25F9cl", condition="walk/v0.4_mu0.54", delay=J)
    assert low["reading_2"] == ("shown [R2/42@10ms: replication cut by the pre-registered order; "
                                "R2/43@10ms: replication cut by the pre-registered order]")
    # only the candidate is replicated; a control's passing record gets no line; a block is never summed
    reps = part(run.tabs, "replication", "replication")["lines"]
    assert {x["set"] for x in reps} == {"F25F9cl"}
    assert not any("40" in str(v) for x in reps for k, v in x.items() if k[:2] in ("a_", "b_") and "path" not in k)
    for name in ("walk", "ladder", "rest"):  # the headline tables read the headline block only
        assert not any("seeds5042" in str(x.get("path")) for x in part(run.tabs, name, "counts")["lines"])


def test_replication_words_are_the_runners(run):
    for row in ("R2/42@10ms", "R2/43@10ms"):
        for key in ("walk/v0.4", "walk/v0.4_mu0.54"):
            for which in ("success", "but_A1"):
                assert (barreport.replication_word(run.doc, "F25F9cl", BODY, row, key, which)
                        == bar.replication_word(run.out, "F25F9cl", BODY, row, key, which)), (row, key, which)
    # a pass that no unit of the ledger covers, in a task without a cut: not run yet
    assert barreport.replication_word(run.doc, "F25F9cl", BODY, "R2/42@10ms", "turn/q90_left", "but_A1") \
        == bar.REPLICATION_PENDING[0]


def test_verdict_words_equal_the_runners_summary(run, tmp_path):
    out = tmp_path / "run"
    shutil.copytree(run.out, out)
    summary = bar.summarize(out, "F25F9cl", BODY, H)
    checked = 0
    for ckey, by_ctrl in summary["verdicts"].items():
        if ckey == "stop/two_step":  # the summary reads the later record, the tables the registered one
            continue
        for name, cell in by_ctrl.items():
            ms = bar.delay_ms("R2/42" + name[len("R2"):])
            mine = barreport.verdict_of(run.doc, "F25F9cl", BODY, ckey, ms)
            for rkey in ("reading_2", "reading_3"):
                assert mine[rkey]["verdict"] == cell[rkey]["verdict"], (ckey, name, rkey)
                if cell[rkey].get("replication") and ckey.startswith("walk/v0.4"):
                    assert mine[rkey]["replication"] == cell[rkey]["replication"]
                checked += 1
    assert checked >= 20


# ------------------------------------------------------------------------------------------------
# Labels
# ------------------------------------------------------------------------------------------------
def test_every_line_of_the_candidate_carries_its_margin_label(run):
    n = 0
    for name, p, x in every_line(run.tabs):
        if x.get("set") == "F25F9cl":
            assert barreport.MARGIN_LABEL in x["labels"], (name, p["key"], x)
            n += 1
        elif "set" in x:
            assert barreport.MARGIN_LABEL not in x["labels"]
    assert n > 60
    seeds = [p for p in run.tabs["jps"]["parts"] if p["key"].startswith("seeds F25F9cl")]
    assert seeds and all(p["notes"] == [barreport.MARGIN_LABEL] for p in seeds)
    assert barreport.labels_of("F25F9cl.2") == [barreport.MARGIN_LABEL, "a version of set F25F9cl"]
    assert barreport.labels_of("V0") == []


def test_a_change_after_a_result_a_control_and_stage_1s_command_are_labelled(run):
    rec = copy.deepcopy(run.doc["V0"][BODY][H]["R0@10ms"]["walk/v0.4"])
    rec["change_after_result"] = dict(label="change after a result", freezes=[dict(freeze=1, reason="a new turn", files=[])])
    assert f"{barreport.CHANGE}: freeze 1: a new turn" in barreport.labels_of("V0", rec)
    stop = one(run.tabs, "rest", "counts", set="F25F9cl", condition="stop/one_step", row="R2/42@10ms")
    assert f"command read by {STAGE1_COMMAND}" in stop["labels"] and stop["item"] == "stop, Stage 1's one-step stop"
    m = one(run.tabs, "rest", "bar_stop", set="F25F9cl", condition="stop/one_step", row="R2/42@10ms")
    assert m["convention"] == STAGE1_COMMAND and m["touchdowns"] == {"1": 20} and m["touchdowns_limit"] == 2
    c = one(run.tabs, "controls", "controls", row="R2/42@10ms")
    assert c["control"] == "omega at Stage 1's 0.03 H crouch" and c["verdict"] == "no verdict (a control)"
    assert (c["crouch"], c["ref_crouch"]) == (0.03, 0.05) and c["omega"] != c["ref_omega"]
    assert c["up"] == c["ref_up"] == dict(k=20, n=20) and "controls/omega_s1/" in c["path"] and "controls" not in c["ref_path"]
    assert "control: omega at Stage 1's 0.03 H crouch" in c["labels"]
    absent = one(run.tabs, "controls", "controls", row="R2/42@0ms")  # task 3 cut a unit; the control was not started
    assert absent["status"] == barreport.CUT and absent["up"] is None


def test_plain_words_for_items_and_grips(run):
    conds = barreport._conds()
    assert len(conds) == 41 and barreport._group("@gated") == [c.key for c in bar.conditions(set_docs("V0")) if c.gated]
    words = {k: (barreport.item_of(d), barreport.grip_of(d)) for k, d in conds.items()}
    assert words["walk/v0.4"] == ("walk at 0.4 m/s", "mu 0.9")
    assert words["walk/nominal_mu0.54"] == ("walk at the nominal speed", "mu 0.54")
    assert words["turn/q90_left"] == ("turn left, 90 deg at pi/4 rad/s (the quick turn)", "mu 0.9")
    assert words["turn/t90_0.35_right"][0] == "turn right, 90 deg at 0.35 rad/s"
    assert words["nudge/backward_x0.375"][0] == "nudge backward, 0.375 x the push bound"
    assert words["stop/two_step_mu0.54"] == ("stop, two touchdowns", "mu 0.54")


# ------------------------------------------------------------------------------------------------
# Checks at the top
# ------------------------------------------------------------------------------------------------
def test_a_record_of_other_sources_is_listed_at_the_top_and_still_tabulated_flagged(run):
    c = run.tabs["checks"]
    assert c["today_sources"] == TODAY and len(c["other_sources"]) == 1
    o = c["other_sources"][0]
    assert o["path"].endswith("V0/75-M-H-10/seeds3042/eval/R2/43@10ms/walk/v0.4.json") and o["differs"] == ["bar"]
    x = one(run.tabs, "walk", "counts", set="V0", condition="walk/v0.4", delay=J, row="R2/43@10ms")
    assert x["up"] == dict(k=0, n=20) and f"{barreport.OTHER_SOURCES} (bar)" in x["labels"]
    clean = one(run.tabs, "walk", "counts", set="V0", condition="walk/v0.4", delay=J, row="R2/42@10ms")
    assert not any(s.startswith(barreport.OTHER_SOURCES) for s in clean["labels"])
    top = run.md.split("## How to read the tables")[0]
    assert "Records whose `key.sources` differ from today's: 1" in top and o["path"] in top


def test_inputs_are_checked_against_the_sets_latest_freeze(run, tmp_path):
    c = run.tabs["checks"]
    assert len(c["not_in_latest_freeze"]) == 1 and c["not_checked"] == []
    bad = c["not_in_latest_freeze"][0]
    assert bad["path"].endswith("V0/75-M-H-10/seeds3042/eval/R0@10ms/walk/v0.4.json")
    assert bad["differences"] == ["refs/v0.4.npz: another sha256"] and bad["set_dir"] == "stage2/sets/V0/75-M-H-10"
    x = one(run.tabs, "walk", "counts", set="V0", condition="walk/v0.4", delay=J, row="R0@10ms")
    assert barreport.NOT_IN_MANIFEST in x["labels"]
    assert c["freezes"]["stage2/sets/V0/75-M-H-10"]["index"] == 0 and "entries" not in c["freezes"]["stage2/sets/V0/75-M-H-10"]
    # a later freeze that lists the file under another hash, an entry it dropped, a set that is gone
    study = tmp_path / "study"
    shutil.copytree(run.study, study)
    sj = study / "stage2" / "sets" / "F25F9cl" / BODY / "set.json"
    sd = json.loads(sj.read_text())
    manifest = copy.deepcopy(sd["freezes"][0]["manifest"])
    manifest["v0.45"]["entry_sha256"] = sha("another entry")
    del manifest["v0.5"]
    sd["freezes"].append(dict(index=1, manifest=manifest, manifest_sha256=s2io.sha256_json(manifest)))
    sj.write_text(json.dumps(sd))
    (study / "stage2" / "sets" / "F25F9" / BODY / "set.json").unlink()
    c2 = barreport.tables(barreport.load(study / "stage2" / "runs" / "synthetic", study=study))["checks"]
    why = {tuple(x["differences"]) for x in c2["not_in_latest_freeze"]}
    assert ("refs/v0.45.npz: another entry_sha256",) in why and ("refs/v0.5.npz: not listed",) in why
    assert [Path(x["path"]).parts[3] for x in c2["not_checked"]] == ["F25F9"]
    assert c2["freezes"]["stage2/sets/F25F9cl/75-M-H-10"]["index"] == 1
    assert c2["freezes"]["stage2/sets/F25F9/75-M-H-10"] == dict(found=False)


def test_errors_unjudged_counts_and_core_hours_are_listed(run):
    c = run.tabs["checks"]
    assert [Path(x["path"]).parts[3] for x in c["errors_or_unjudged"]] == ["F25F9"]
    assert c["errors_or_unjudged"][0]["errors"] == 1 and c["errors_or_unjudged"][0]["judged"] is False
    x = one(run.tabs, "walk", "counts", set="F25F9", condition="walk/v0.4", delay=J, row="R0@10ms")
    assert x["r2_pass"] == "not judged (1 episode failed to run)" and x["episodes"] == 20 and x["up"] == dict(k=19, n=20)
    assert c["records"]["by_set"]["F25F9cl"]["not_run"] == 4 and c["records"]["not_run"] == 4
    b = c["budget"]["3"]
    assert b["cap_core_h"] == 16.0 == b["registered_cap_core_h"] == bar.BUDGET_CORE_H[3]
    assert b["spent_core_h"] == pytest.approx(2.0) and [u["unit"] for u in b["units"]] == ["T3.1", "T3.2a"]
    assert [u["unit"] for u in b["cut"]] == ["T3.2b"] and "4" not in c["budget"]
    top = run.md.split("## How to read the tables")[0]
    assert "Task 3: 2.000 core-hours spent of 16" in top and "| T3.2b | 0.300 | 15.900 |" in top
    assert "2 units started, 1 cut; units started after the first cut (2026-11-15T20:00:00Z): none" in top
    assert "`COST.md`: none" in top and b["started_after_the_first_cut"] == []
    # a unit that started after the first cut is named: the cut units would not be the order's tail
    doc = copy.deepcopy(run.doc)
    doc[barreport.META]["budget"]["tasks"]["3"]["units"].append(dict(
        unit="T3.9", estimate_core_h=0.1, spent_before_core_h=2.0, started_at="2026-11-15T21:00:00Z", items={}, detail={}))
    assert barreport.checks(doc)["budget"]["3"]["started_after_the_first_cut"] == ["T3.9"]
    assert "units started after the first cut (2026-11-15T20:00:00Z): T3.9." in barreport.markdown(barreport.tables(doc))


def test_counts_are_the_records_own_and_recompute_from_their_episodes(run, tmp_path):
    assert run.tabs["checks"]["counts_not_recomputed"] == []
    assert "do not recompute from their own episodes (`bar.counts`): 0" in run.md
    for rec in barreport._records(run.doc):  # every count a table prints is the record's, equal to bar.counts
        assert barreport.counts_differences(rec) == []
        d = barreport.digest(rec)
        if rec["status"] == "done":
            c2, c3 = bar.counts(rec["episodes"], rec["seeds"]), bar.counts(rec["episodes"], rec["seeds"], "but_A1")
            assert (d["up"]["k"], d["r2"]["k"], d["r3"]["k"]) == (c2["survived"], c2["succeeded"], c3["succeeded"])
            assert (d["up_lcb"], d["r2_lcb"], d["r3_lcb"]) == (c2["survival_lcb"], c2["success_lcb"], c3["success_lcb"])
    assert run.tabs["checks"]["second_block_outside_the_rule"] == []
    # a record whose counts were changed by hand is named at the top
    study = tmp_path / "study"
    out = study / "runs" / "x"
    put(study, out, "V0", "R2/42@10ms", "walk/v0.4", per_seed=lambda s: dict(fell=s < 3046))
    path = bar.record_path(out, "V0", BODY, H, "R2/42@10ms", SimpleNamespace(behavior="walk", name="v0.4"))
    rec = json.loads(path.read_text())
    rec["counts"].update(survived=20, succeeded=20, pass_=True)
    path.write_text(json.dumps(rec))
    put(study, out, "V0", "R0@10ms", "walk/v0.4", block=R)  # no second block for a control set or an R0 row
    c = barreport.tables(barreport.load(out, study=study))["checks"]
    assert [x["why"] for x in c["second_block_outside_the_rule"]] == [
        ["an R0 row", "a control set of the round", "no replication_of field"]]
    assert len(c["counts_not_recomputed"]) == 1 and c["counts_not_recomputed"][0]["differences"] == [
        "reading 2 (every limit): survived is 20 in the record, 16 from its episodes",
        "reading 2 (every limit): succeeded is 20 in the record, 16 from its episodes",
        "reading 2 (every limit): pass_ is True in the record, False from its episodes"]
    # reading 2 of a turn frozen with a1_over keeps its pass under pass_if_read: that is no difference
    a1 = run.doc["F25F9cl"][BODY][H]["R2/42@10ms"]["turn/q90_left"]
    assert a1["counts"]["pass_"] is None and a1["counts"]["pass_if_read"] is False and barreport.counts_differences(a1) == []


def test_check_episodes_are_counted_and_held_to_rule_r4(run, tmp_path):
    assert run.tabs["checks"]["dev_episodes"] is None and "no `dev_episodes.json`" in run.md
    out = tmp_path / "study" / "runs" / "x"
    put(tmp_path / "study", out, "V0", "R0@10ms", "walk/v0.4")
    entry = lambda s, row, seed=3042: dict(set=s, body=BODY, block=H, row=row, condition="walk/v0.4", seed=seed)  # noqa: E731
    s2io.write_json(out / "dev_episodes.json", dict(
        episodes=[entry("V0", "R0@10ms"), entry("V0", "R0@0ms"), entry("V0", "R2/42@10ms"), entry("F25F9cl", "R0@10ms"),
                  entry("V0", "R0@10ms", 3043), entry("D0", "R2/42@10ms")],
        scratch_sets=dict(sets=dict(T0="a fixture", D0="a scratch set"))))
    tabs = barreport.tables(barreport.load(out, study=tmp_path / "study"))
    dev = tabs["checks"]["dev_episodes"]
    assert dev["listed"] == 6 and dev["scratch_sets"] == ["D0", "T0"]
    assert [(x["set"], x["row"], x["seed"]) for x in dev["outside_the_rule"]] == [
        ("V0", "R2/42@10ms", 3042), ("F25F9cl", "R0@10ms", 3042), ("V0", "R0@10ms", 3043)]
    md = barreport.markdown(tabs)
    assert "(smoke; none is a record): 6; of these outside rule R4" in md and "set `F25F9cl`, row `R0@10ms`" in md


def test_cost_rates_are_printed_with_how_they_were_measured(run, tmp_path):
    out = tmp_path / "study" / "runs" / "x"
    put(tmp_path / "study", out, "V0", "R0@10ms", "walk/v0.4")
    cost = dict(schema=bar.COST_SCHEMA, rates=dict(R2=dict(
        wall_per_sim_s=0.4567, row="R2/42@0ms", full_length=True, episodes=[dict(seed=3042), dict(seed=3043)],
        warm_up=dict(seed=3042, counted=False))),
        measured_on=dict(condition="walk/brisk", body=BODY, seeds=[3042, 3043]),
        load=dict(before=dict(load_average=[1.5, 2.0, 2.5])))
    (out / "COST.md").write_text("# Cost\n\n```json\n" + json.dumps(cost, indent=1) + "\n```\n")
    md = barreport.markdown(barreport.tables(barreport.load(out, study=tmp_path / "study")))
    assert ("R2 0.457 (row `R2/42@0ms`, 2 episodes, run their full length, after a warm-up episode that is not "
            "counted)") in md
    assert "Stage 1's own condition `walk/brisk`" in md and "load averages before the measurement: 1.5, 2, 2.5" in md
    assert "Core-hours: no `budget.json`" in md


# ------------------------------------------------------------------------------------------------
# Measured values and limits
# ------------------------------------------------------------------------------------------------
def test_a_measured_column_never_holds_a_target(run):
    # a record whose measured speed differs from its target shows the measured one
    x = one(run.tabs, "ladder", "bar_walk", set="V0", condition="walk/v0.35", row="R2/42@0ms")
    assert x["speed"] == dict(median=0.3661, n=20) and x["along"]["median"] == pytest.approx(3.661)
    assert x["along_limit"] == 3.5 and x["lateral_limit"] == 0.25 and x["toe_limit"] == pytest.approx(20.0741)
    assert x["toe"]["median"] == pytest.approx(23.63) and x["toe_first"]["median"] == pytest.approx(17.1)
    assert x["toe_stage1"]["median"] == pytest.approx(22.8)
    assert "| 0.366 (20) | 3.661 (20) | 3.5 |" in run.md
    targets = {10.0 * v for v in setio.SPEEDS} | set(setio.SPEEDS)
    kinds = set()
    for name, p, line in every_line(run.tabs):
        for c in p["columns"]:
            kinds.add(c["kind"])
            v = line.get(c["key"])
            if c["kind"] == "measured":  # by construction of the synthetic records no measurement equals a target
                assert "measured" in c["title"] and "imit" not in c["title"].split(":")[0], c["title"]
                value = v["median"] if isinstance(v, dict) else v
                assert value is None or value not in targets, (name, p["key"], c["key"], v)
                assert v is None or isinstance(v, (float, dict)), (c["key"], v)
            elif c["kind"] in ("limit", "defined"):
                assert "measured" not in c["title"] and not isinstance(v, dict), (c["title"], v)
                assert c["title"].startswith("Limit") or "as defined" in c["title"] or "as run" in c["title"] \
                    or "crouch" in c["title"], c["title"]
    assert {"measured", "limit", "defined", "of", "claim"} <= kinds
    # the judged toe clearance stands beside the reading as first written and Stage 1's, both reported
    titles = [c["title"] for c in part(run.tabs, "walk", "bar_walk")["columns"]]
    i = next(i for i, t in enumerate(titles) if t.startswith("Toe clearance, the judged term"))
    assert titles[i + 1].startswith("Limit: at least (mm")
    assert "as first written" in titles[i + 2] and "never judged" in titles[i + 2]
    assert "Stage 1's reading" in titles[i + 3] and "never judged" in titles[i + 3]
    assert [c["title"] for c in part(run.tabs, "rest", "bar_turn")["columns"]][-5:-1] == titles[i:i + 4]


def test_the_items_own_columns(run):
    t = one(run.tabs, "rest", "bar_turn", set="F25F9cl", condition="turn/q90_left", row="R2/42@10ms")
    assert t["heading_at"] == dict(median=-4.5, n=20) and t["within"] == dict(median=1.71, n=20)
    assert (t["heading_limit"], t["at_defined"]) == (10.0, 2.0) and t["heading_hold"]["median"] == 6.2
    n = one(run.tabs, "rest", "bar_nudge", set="F25F9cl", condition="nudge/forward", row="R2/42@10ms")
    assert n["recovered"] == dict(k=18, n=20) and n["back"] == dict(median=1.23, n=18) and n["back_limit"] == 3.0
    assert n["force_defined"] == pytest.approx(21.87)
    f = one(run.tabs, "rest", "fails", set="F25F9cl", condition="nudge/forward", row="R2/42@10ms")
    assert (f["falls_before"], f["falls_after"], f["falls_no_command"]) == (1, 1, 0) and f["failed_2"] == {"survived": 2}
    s = lines(run.tabs, "rest", "bar_stop", set="F25F9cl", condition="stop/two_step", row="R2/42@10ms")
    assert [x["touchdowns"] for x in s] == [{"2": 20}, {"3": 20}] and s[0]["stopped_s"] == dict(median=1.11, n=20)
    assert s[0]["accel_limit"] == 1.5 and s[0]["stopped"] == dict(k=20, n=20)
    w = one(run.tabs, "walk", "fails", set="V0", condition="walk/v0.4", delay=J, row="R2/42@10ms")
    assert w["falls_no_command"] == 20 and w["failed_2"] == {"survived": 20} and w["worst_drive"] is None
    assert w["attribution"] == "not attributed (falls; run with --attribute)" == w["label_2"]


def test_numbers_are_printed_with_a_fixed_precision_or_g():
    assert barreport._num(3) == "3" and barreport._num(4.0) == "4" and barreport._num(0.25) == "0.25"
    assert barreport._num(0.36614, "m/s") == "0.366" and barreport._num(23.6349, "mm") == "23.6"
    assert barreport._num(2.0, "core-h") == "2.000" and barreport._num(1e-9) == "1e-09"
    col = dict(kind="measured", unit="deg")
    assert barreport._cell_text(dict(median=-4.25, n=20), col) == "-4.2 (20)" and barreport._cell_text(None, col) == "-"
    assert barreport._cell_text(dict(median=None, n=0), col) == "-"
    assert barreport._cell_text({}, dict(kind="terms")) == "none"
    assert barreport._cell_text({"a": 2, "b": 1}, dict(kind="terms")) == "a: 2; b: 1"
    assert barreport._cell_text(dict(k=17, n=20), dict(kind="of")) == "17 of 20"
    assert barreport._cell_text("a/b.json", dict(kind="path")) == "`a/b.json`"
    assert barreport._cell_text(True, dict(kind="flag")) == "yes"
    assert barreport._cell_text("not defined", dict(kind="flag")) == "not defined"


# ------------------------------------------------------------------------------------------------
# The speed ladder
# ------------------------------------------------------------------------------------------------
def test_the_highest_rung_that_passes_under_both_rows(run):
    hi = one(run.tabs, "ladder", "highest", set="F25F9cl", grip="mu 0.9", delay=J)
    assert hi["reading_2_highest"] == "none"  # 0.40 m/s is provisional, 0.45 m/s falls
    assert hi["reading_3_highest"] == "walk/v0.4 (walk at 0.4 m/s)"
    assert hi["reading_2_rungs"][0].startswith("walk/nominal: cut by the pre-registered order")
    assert "walk/v0.4: provisional [R2/42@10ms: replicated]" in hi["reading_2_rungs"]
    assert [r.split(":")[0] for r in hi["reading_2_rungs"]] == ["walk/nominal", "walk/v0.3", "walk/v0.35", "walk/v0.4",
                                                                "walk/v0.45", "walk/v0.5"]
    zero = one(run.tabs, "ladder", "highest", set="F25F9cl", grip="mu 0.9", delay="0 ms (reported)")
    assert zero["reading_2_highest"] == zero["reading_3_highest"] == "walk/v0.5 (walk at 0.5 m/s)"
    assert "walk/v0.5: passes at 0 ms" in zero["reading_2_rungs"]
    v0 = one(run.tabs, "ladder", "highest", set="V0", grip="mu 0.9", delay="0 ms (reported)")
    assert v0["reading_2_highest"] == "none" and v0["reading_3_highest"] == "walk/v0.35 (walk at 0.35 m/s)"
    assert not lines(run.tabs, "ladder", "counts", condition="walk/v0.4")  # the 0.40 m/s rung's records are table 1's
    assert one(run.tabs, "ladder", "counts", set="V0", condition="walk/v0.35", row="R2/42@0ms")["stage1"] == dict(k=0, n=20)


# ------------------------------------------------------------------------------------------------
# The comparison with JPS section 2.3
# ------------------------------------------------------------------------------------------------
def strip(cell: str) -> str:
    return re.sub(r"\*\*|`|\*", "", cell).strip()


def test_jps_claims_are_the_reports_own_lines():
    text = (ROOT / barreport.JPS_FILE).read_text(encoding="utf-8").split("\n")
    assert [c["line"] for c in barreport.JPS_ROWS] == list(range(92, 102)) and set(barreport.JPS_MAP) == set(range(92, 102))
    for claim in (barreport.JPS_HEADER,) + barreport.JPS_ROWS:
        cells = [strip(c) for c in text[claim["line"] - 1].strip().strip("|").split("|")]
        assert cells == [claim[k] for k in barreport.JPS_CLAIM_KEYS], claim["line"]
    assert text[85].startswith("### 2.3 ")
    for cells in barreport.JPS_MAP.values():  # every mapped cell is a condition the bar defines
        for c in cells:
            assert isinstance(c, str) or (c[0] in barreport.SETS and c[1] in barreport._conds() and c[2] % 2 == 0)


def test_jps_numbers_appear_only_under_jps_claim(run):
    claims = {c[k] for c in barreport.JPS_ROWS for k in barreport.JPS_CLAIM_KEYS if k not in ("delay", "floor_plant")}
    assert len(claims) > 30
    jps = part(run.tabs, "jps", "jps")
    assert len(jps["lines"]) == len(barreport.JPS_ROWS) == 10
    for line in jps["lines"]:
        for k, v in line.items():
            for s in strings(v):
                assert k.startswith("jps_claim_") or not any(c in s for c in claims), (k, s)
    for c in jps["columns"]:
        assert c["title"].startswith("JPS's claim: ") == (c["kind"] == "claim") == c["key"].startswith("jps_claim_")
    # nowhere else in the JSON
    rest = {k: v for k, v in run.tabs.items() if k != "jps"}
    rest["jps_other"] = [dict(p, lines=None) if p["key"] == "jps" else p for p in run.tabs["jps"]["parts"]]
    assert not [s for s in strings(rest) if any(c in s for c in claims)]
    # in the Markdown: a claim stands in a cell of the JPS table under a header that says so, and nowhere else
    md = run.md.split("\n")
    head = next(i for i, x in enumerate(md) if x.startswith("| JPS line |"))
    titles = md[head].split(" | ")
    n = 0
    for i, row in enumerate(md):
        if not any(c in row for c in claims):
            continue
        assert head + 2 <= i < head + 2 + 10, row
        cells = row.split(" | ")
        assert len(cells) == len(titles)
        for title, cell in zip(titles, cells, strict=True):
            if any(c in cell for c in claims):
                assert title.startswith("JPS's claim: "), (title, cell)
                n += 1
    assert n >= 40
    assert "(JPS's claim) is set beside this round's 0.50 m/s rung" in run.md
    assert "commanded causally" in run.md and "JPS's were not" in run.md


def test_jps_rows_stand_beside_this_rounds_records_seed_by_seed(run):
    rows = {x["line"]: x for x in part(run.tabs, "jps", "jps")["lines"]}
    # 0.496 m/s beside the 0.50 m/s rung, with the measured speed and the three readings of each row
    r95 = rows[95]
    assert "F25F9cl walk/v0.5 at 0 ms (walk at 0.5 m/s, mu 0.9; rows R2/42@0ms and R2/43@0ms); the rung's target " \
           "speed 0.5 m/s" in r95["matched"]
    assert ("F25F9cl walk/v0.5 at 0 ms, R2/42@0ms: stayed up 20 of 20; Stage 1's reading 20 of 20; reading 2 20 of 20 "
            "(passes at 0 ms); reading 3 20 of 20 (passes at 0 ms); median speed measured 0.514 (20) m/s "
            f"[{barreport.MARGIN_LABEL}]") in r95["record"]
    assert "F25F9cl walk/v0.45_mu0.54 at 0 ms, R2/42@0ms: cut by the pre-registered order" in r95["record"]
    assert ("F25F9cl walk/v0.5 at 0 ms, the controller R2 from its two rows: reading 2: passes at 0 ms; reading 3: "
            "passes at 0 ms but for A1") in r95["record"]
    assert ("F25F9cl walk/v0.4 at 10 ms, the controller R2 from its two rows: reading 2: provisional [R2/42@10ms: "
            "replicated]; reading 3: shown but for A1 [R2/42@10ms: replicated; R2/43@10ms: not replicated]") \
        in rows[99]["record"]
    assert any(p.endswith("seeds3042/eval/R2/43@0ms/walk/v0.5.json") for p in r95["paths"])
    r92 = rows[92]
    assert any("V0 walk/v0.35 at 0 ms, R2/43@0ms: stayed up 20 of 20; Stage 1's reading 0 of 20; reading 2 0 of 20"
               in s and "median speed measured 0.366 (20) m/s" in s for s in r92["record"])
    # cells this round has no condition for, and cells outside the registered order
    assert rows[96]["record"] == ["the walk at 0.55 m/s: this round's speed ladder ends at 0.50 m/s: not run"]
    assert rows[97]["record"][0].endswith(": not run") and rows[96]["paths"] == rows[97]["paths"] == []
    assert "F25F9cl walk/v0.5 at 8 ms, R2/42@8ms: not run" in rows[99]["record"]
    assert "F25F9cl walk/v0.5 at 6 ms, R2/42@6ms: cut by the pre-registered order" in rows[99]["record"]
    assert "F25F9cl stop/two_step at 2 ms, R2/42@2ms: not run" in rows[100]["record"]
    assert rows[100]["record"][-1] == "the stop from 0.50 m/s: this round's stop is from the 0.40 m/s walk: not run"
    assert "F25F9cl nudge/left at 2 ms, R2/43@2ms: not run" in rows[101]["record"]
    # seed by seed: one part per matched cell with a record, a line per seed, the two rows side by side
    sp = part(run.tabs, "jps", "seeds V0 walk/v0.35 at 0 ms")
    assert [x["seed"] for x in sp["lines"][:-1]] == list(bar.BLOCKS[H])
    assert [c["title"] for c in sp["columns"]] == ["Seed"] + [
        f"{r}@0ms: {t}" for r in ("R2/42", "R2/43") for t in ("stayed up", "Stage 1's reading", "reading 2", "reading 3")]
    assert sp["lines"][0] == dict(seed=3042, up_0=True, stage1_0=False, r2_0=False, r3_0=True,
                                  up_1=True, stage1_1=False, r2_1=False, r3_1=True)
    assert sp["lines"][-1]["r3_0"] == "20 of 20" and sp["lines"][-1]["r2_0"] == "0 of 20"
    assert len(sp["paths"]) == 2 and all("walk/v0.35.json" in p for p in sp["paths"])
    keys = [p["key"] for p in run.tabs["jps"]["parts"][1:]]
    assert keys == ["seeds V0 walk/v0.35 at 0 ms", "seeds F25F9cl walk/v0.4 at 0 ms", "seeds F25F9cl walk/v0.45 at 0 ms",
                    "seeds F25F9cl walk/v0.5 at 0 ms", "seeds F25F9cl walk/v0.4 at 10 ms"]
    judged = part(run.tabs, "jps", "seeds F25F9cl walk/v0.4 at 10 ms")
    assert judged["lines"][3]["r2_1"] is False and judged["lines"][3]["r3_1"] is True  # the tail is over in reading 2
    assert "| 3042 | yes | no | no | yes | yes | no | no | yes |" in run.md


def test_jps_table_is_part_of_the_tables_and_keeps_to_the_sets_asked_for(run):
    assert barreport.jps_table(run.doc)["title"] == run.tabs["jps"]["title"] == "6. The comparison with JPS section 2.3"
    assert json.loads(json.dumps(barreport.jps_table(run.doc))) == run.tabs["jps"]
    only = barreport.jps_table(barreport.load(run.out, sets=["V0"], study=run.study))
    rows = {x["line"]: x for x in only["parts"][0]["lines"]}
    assert rows[95]["record"][0] == "F25F9cl walk/v0.4 at 0 ms: the set is not among the sets asked for"
    assert [p["key"] for p in only["parts"][1:]] == ["seeds V0 walk/v0.35 at 0 ms"]


# ------------------------------------------------------------------------------------------------
# The plan's text and the chain
# ------------------------------------------------------------------------------------------------
def test_every_verdict_table_cites_the_rules_amended_after_a_result(run):
    plan = " ".join((ROOT / barreport.PLAN_FILE).read_text(encoding="utf-8").split())
    assert [a["rule"] for a in barreport.AMENDED_AFTER_RESULT] == ["A7", "C3"]
    for a in barreport.AMENDED_AFTER_RESULT:  # the plan's own labels and sentences, word for word
        assert a["label"] in plan and a["label"] + " " + a["text"] in plan, a["rule"]
    assert plan.count("[changed 15 Nov after a result:") == len(barreport.AMENDED_AFTER_RESULT)
    n = 0
    for name in barreport.ORDER:
        for p in run.tabs[name]["parts"]:
            verdict = any(c["title"].startswith(("Reading 2 (every limit)", "Highest rung", "This round's record"))
                          and c["kind"] in ("word", "list") for c in p["columns"]) or p["key"] == "replication"
            if verdict:
                n += 1
                notes = " ".join(p.get("notes") or [])
                for a in barreport.AMENDED_AFTER_RESULT:
                    assert f"{a['rule']} {a['label']}" in notes, (name, p["key"])
    assert n >= 7
    assert run.md.count(barreport.AMENDED_AFTER_RESULT[1]["label"]) == n + 1  # and once under "How to read"


def test_the_registered_order_is_the_chains(run):
    """``planned()`` holds the cells of the chain's units on the headline block (``run_75cm.sh list`` runs nothing)."""
    script = ROOT / "stage2" / "runs" / "2026-11-15-75cm" / "run_75cm.sh"
    listed = subprocess.run(["bash", str(script), "list"], capture_output=True, text=True, check=True).stdout
    cells, controls, units = {}, set(), 0
    for line in listed.splitlines():
        if "designlab.stage2.bar run " not in line:
            continue
        args = line.split(": ", 1)[1].split()
        opt = {a: args[i + 1] for i, a in enumerate(args[:-1]) if a.startswith("--")}
        units += 1
        if opt.get("--block", H) != H:
            continue
        keys = [k for c in opt["--conditions"].split(",") for k in (barreport._group(c) if c.startswith("@") else [c])]
        for row in opt["--rows"].split(","):
            base, ms = barreport._split_row(row)
            for k in keys:
                if "--control" in opt:
                    controls.add((opt["--control"], opt["--set"], k, base, ms))
                else:
                    assert cells.setdefault((opt["--set"], k, ms, base), int(opt["--task"])) == int(opt["--task"])
    assert units > 80 and cells == barreport.planned()
    c, s, k, base, delays = barreport.OMEGA_CONTROL
    assert controls == {(c, s, k, base, ms) for ms in delays}
    assert sum(1 for t in cells.values() if t == 3) == 3 * 2 * 6 * 3 + 2 * 5 * 4 * 2 + 4 * 2
    assert sum(1 for t in cells.values() if t == 4) == 2 * (14 * 4 * 3 + 15 * 2)
    assert set(barreport.TASK_OF) == set(bar.BEHAVIORS) and set(barreport.TASK_OF.values()) == set(bar.BUDGET_CORE_H)


def test_a_run_directory_without_records_renders_every_table(tmp_path):
    out = tmp_path / "study" / "stage2" / "runs" / "empty"
    out.mkdir(parents=True)
    tabs = barreport.tables(barreport.load(out, study=tmp_path / "study"))
    md = barreport.markdown(tabs)
    assert tabs["checks"]["records"]["found"] == 0 and "Records found: 0 (0 done, 0 \"not run\")" in md
    for name in barreport.ORDER:
        assert f"## {tabs[name]['title']}" in md
    # every cell of the registered order is there and reads "not run"
    w = part(tabs, "walk", "counts")["lines"]
    assert len(w) == 3 * 6 and {x["status"] for x in w} == {barreport.NOT_RUN}
    r = part(tabs, "rest", "counts")["lines"]
    assert len(r) == 2 * 4 and r[0]["item"] == "29 conditions, 72 records: none found"  # 14 x 3 rows + 15 x 2 rows
    g = part(tabs, "glance", "glance")["lines"]
    assert len(g) == 3 * 16 and {x["reading_2"] for x in g} == {barreport.NOT_RUN}
    jps = part(tabs, "jps", "jps")["lines"]
    assert all(s.endswith((": not run", "not run")) for x in jps for s in x["record"])
    assert len(tabs["jps"]["parts"]) == 1 and part(tabs, "controls", "controls")["lines"][0]["status"] == barreport.NOT_RUN
    assert part(tabs, "other", "other")["lines"] == [] and barreport.CUT not in md.split("## 1.")[1]


def test_a_stock_condition_and_a_row_without_a_delay_are_listed_in_the_last_table(run, tmp_path):
    study = tmp_path / "study"
    out = study / "runs" / "x"
    rec = put(study, out, "V0", "R0@10ms", "walk/v0.4")
    stock = dict(rec, set=bar.STOCK_SET, row="R2/42", behavior="stand", condition="still", key_name="stand/still",
                 counts_but_A1=None, definition=dict(behavior="stand", name="still", judge="stand", stock=True))
    s2io.write_json(bar.block_dir(out, bar.STOCK_SET, BODY, H) / "eval" / "R2/42" / "stand" / "still.json", stock)
    tabs = barreport.tables(barreport.load(out, study=study))
    x = one(tabs, "other", "other", set=bar.STOCK_SET)
    assert x["item"] == "Stage 1's condition stand/still" and x["r3_pass"] == "no such reading (Stage 1's one reading)"
    assert x["r2"] == dict(k=20, n=20) and tabs["checks"]["not_checked"] == [] and tabs["checks"]["records"]["found"] == 2
    assert not lines(tabs, "other", "other", set="V0")  # a record a table cites is not repeated


def test_output_paths_are_not_ignored_by_git():
    """The two files are committed with the run: git must not ignore them (logs/ is where a smoke goes)."""
    run_dir = "stage2/runs/2026-11-15-75cm"
    for name in (barreport.TABLES_JSON, barreport.TABLES_MD):
        r = subprocess.run(["git", "check-ignore", "-q", f"{run_dir}/{name}"], cwd=ROOT)
        assert r.returncode == 1, name
    smoke = "logs/barreport-smoke/" + barreport.TABLES_MD
    assert subprocess.run(["git", "check-ignore", "-q", smoke], cwd=ROOT).returncode == 0
