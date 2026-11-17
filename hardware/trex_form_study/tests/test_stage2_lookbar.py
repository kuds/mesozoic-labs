"""Tests for the rows of the bar runner with the look reading (``designlab/stage2/lookbar.py``; round C design
D17, sections 3.5 and 4; Track B rules U4-U6, M1).

The rules on records (``companion_episode``, ``collect``) are tested on tables and on synthetic bar records and
sidecars in ``tmp_path``: nothing is simulated there.  The closed-loop tests run on the S1 view (Stage 1's own run
directory) and on a synthetic set ``T0`` that a fixture builds in a temporary directory from Stage 1's arrays (rule
X1: no Stage 2 reference set is touched); they need the Stage 1 arrays and the round-two models
(``tools/restore_inputs.py``) and one BLAS thread, and skip without them.  Slow (``STAGE2_SLOW=1``): whole episodes
of ``R0L`` against ``R0`` on the S1 view.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from designlab.stage1 import metrics, panels, plant, tune  # noqa: E402
from designlab.stage2 import bar, g8a, lookbar, lookread, s2io, setio  # noqa: E402

BODY = "75-M-H-10"
RUN = ROOT / "stage1" / "runs" / "2026-11-08"
ONE_THREAD = all(os.environ.get(v) == "1" for v in s2io.THREAD_VARS)
SLOW = os.environ.get("STAGE2_SLOW") == "1"
HAVE = (RUN / BODY / "transitions" / "stop.npz").exists()
try:
    HAVE = HAVE and all(Path(f).exists() for f in plant.model_files(BODY))
except Exception:  # noqa: BLE001  (no model record: the same skip)
    HAVE = False
needs_run = pytest.mark.skipif(not (HAVE and ONE_THREAD), reason=(
    "needs the Stage 1 arrays and the round-two models (tools/restore_inputs.py) and one BLAS thread "
    "(OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1)"))
SET = "T0"
W = setio.walk_name(setio.BAR_SPEED)
SEEDS = list(bar.BLOCKS[bar.HEADLINE_BLOCK])


def quiet(*a, **k):
    pass


def plain(x):
    return json.loads(json.dumps(s2io.plain(x)))


# ------------------------------------------------------------------------------------------------
# A synthetic frozen set of Stage 1's arrays under the bar's names (as the bar's own tests build it)
# ------------------------------------------------------------------------------------------------
T0_REFS = (("nominal", "nominal"), (W, "brisk"))
T0_MOVES = (("start", "start"), (setio.change_name("nominal", W), "nominal_to_brisk"),
            (setio.stop_name(2, W), "stop"), (setio.turn_name("pi4", "left"), "enter_walk_0.35_left"))
ABSENT = setio.turn_name("pi4", "right")


def make_t0(root: Path) -> Path:
    """``T0``: two walks, the start, the change, a stop and a left "turn" declared causal (Stage 1's entries: they
    join nothing, and the open loop falls in them); the right turn is recorded as not available."""
    src = RUN / BODY
    d = setio.set_dir(SET, BODY, root)
    refs1, trs1 = s2io.read_json(src / "refs.json"), s2io.read_json(src / "transitions.json")
    refs, trs, manifest = {}, {}, {}
    for name, old in T0_REFS:
        e = copy.deepcopy(refs1["references"][old])
        e.update(name=name, file=f"refs/{name}.npz", set=SET, label=None, strides=1, window_s=e["period_s"],
                 crouch_frac=0.03, margin_rule="a fixture's")
        refs[name] = e
    for name, old in T0_MOVES:
        e = copy.deepcopy(trs1["transitions"][old])
        e.update(name=name, file=f"transitions/{name}.npz", a1_over=None)
        p = setio.parse_name(name)
        if p["kind"] == "stop":
            e["move"] = dict(kind="stop", builder="fixture", causal=True, command_phase=-0.5)
        elif p["kind"] == "turn":
            e["move"] = dict(kind="turn", builder="fixture", causal=True, command_phase=0.0)
            e["turn"] = dict(angle_rad=p["angle_rad"], rate_rad_s=p["sign"] * p["rate_rad_s"], direction=p["direction"])
        else:
            e["move"] = dict(kind=p["kind"], builder="fixture", causal=False, command_phase=None)
        trs[name] = e
    trs[ABSENT] = dict(name=ABSENT, kind="turn", available=False, reason="a fixture's reason", numbers=None)
    for doc, sub, kind in ((refs, "refs", "walk"), (trs, "transitions", "move")):
        for name, e in doc.items():
            if not e.get("file"):
                continue
            (d / sub).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / sub / f"{dict(T0_REFS + T0_MOVES)[name]}.npz", d / e["file"])
            manifest[name] = dict(kind=kind, sha256=e["sha256"], entry_sha256=setio.entry_sha256(kind, e))
    s2io.write_json(d / "refs.json", dict(body=BODY, set=SET, schema=1, references=refs))
    s2io.write_json(d / "transitions.json", dict(body=BODY, set=SET, schema=1, transitions=trs))
    freeze = dict(index=0, at="fixture", commit=None, manifest=manifest, manifest_sha256=s2io.sha256_json(manifest))
    s2io.write_json(d / "set.json", dict(set=SET, body=BODY, role="fixture", crouch_frac=0.03, not_available={},
                                         stand_in_walks=True, rules=dict(fixture="Stage 1's arrays"),
                                         freezes=[freeze]))
    return d


@pytest.fixture(scope="module")
def t0(tmp_path_factory) -> Path:
    """The root of a frozen ``T0``."""
    root = tmp_path_factory.mktemp("sets")
    make_t0(root)
    return root


@pytest.fixture(scope="module")
def t0_set(t0) -> dict:
    """``T0`` opened as the bar's workers read it, with the conditions the tests play."""
    keys = ["walk/v0.4", "walk/nominal", "turn/q90_left", "stop/two_step", "nudge/forward"]
    return lookbar.open_set(BODY, SET, keys, root=t0)


@pytest.fixture(scope="module", autouse=True)
def _release():
    yield
    g8a.release()


def stock_cut(monkeypatch, seconds: float = 1.0, keys=("walk/brisk",)):
    """Stage 1's conditions ``keys`` cut to ``seconds`` as the pseudo-set S1's conditions (a smoke check)."""
    conds = [dataclasses.replace(c, seconds=float(seconds)) for c in panels.conditions() if c.key in keys]
    monkeypatch.setattr(bar, "stock_conditions", lambda: conds)
    monkeypatch.setattr(bar, "stock_default", lambda: conds)
    return conds


def r0l(k: int = 5, sidecar=None):
    name = f"R0L@{bar.STEP_MS * k:d}ms"
    return lookbar.rows(BODY, [name], sidecar=sidecar)[name]


# ------------------------------------------------------------------------------------------------
# The module, the rows by name, what is relied on from the bar
# ------------------------------------------------------------------------------------------------
def test_the_module_registers_r0l_and_names_the_look_sets():
    assert os.environ["MUJOCO_GL"] == "disable"
    assert lookbar.LOOK_SETS == ("V6", "V6noF5") and "R0L" in panels.ROW_FACTORIES
    assert lookbar.SIDECAR_SCHEMA == "s2-look-ep-1" and lookbar.COMPANION_SCHEMA == "s2-look-1"
    assert lookbar.ON_COMMAND_FRAC == 0.9 and lookbar.RATE == 1.05
    row = panels.tracker_rows(BODY, RUN / BODY, ("R0L",))["R0L"]
    r0 = panels.tracker_rows(BODY, RUN / BODY, ("R0",))["R0"]
    assert isinstance(row, lookbar.LookRow) and row.name == "R0L" and row.inner.name == "R0"
    assert row.hard_only is r0.hard_only and row.last is None and row.sidecar is None
    for c in panels.conditions():
        assert row.runs(c) == r0.runs(c) and row.gated(c) == r0.gated(c)
    # the description is the inner row's with the look reading's bands, rules and sets; the sources gain two files
    d = row.describe()
    assert {k: v for k, v in d.items() if k != "look_reading"} == r0.describe()
    assert sorted(d["look_reading"]) == ["bands", "look_sets", "rule"]
    assert d["look_reading"]["bands"] == dict(lookread.BANDS) and d["look_reading"]["look_sets"] == ["V6", "V6noF5"]
    assert set(lookread.RULES) | set(lookbar.RULES) == set(d["look_reading"]["rule"])
    # the two modules' tables share no key, so every rule text of both is in the description (it is in the key):
    # the look reading's own rule U5 text under "reference", this module's use of it under "reference_use"
    rule = d["look_reading"]["rule"]
    assert not set(lookread.RULES) & set(lookbar.RULES) and len(rule) == len(lookread.RULES) + len(lookbar.RULES)
    assert all(rule[k] == v for k, v in lookread.RULES.items()) and all(rule[k] == v for k, v in lookbar.RULES.items())
    assert rule["reference"] == lookread.RULES["reference"] and lookread.MODULE_READING in rule["reference"]
    assert rule["reference_use"].startswith("STAGE2_TRACKB_PLAN rule U5: the reference's values are lookread.playback")
    assert rule == lookbar.rule_texts()
    assert sorted(row.sources()) == ["designlab/stage2/lookbar.py", "designlab/stage2/lookread.py"]
    assert row.sources()["designlab/stage2/lookread.py"] == next(iter(lookread.sources().values()))
    with pytest.raises(ValueError, match="no seed"):
        panels.tracker_rows(BODY, RUN / BODY, ("R0L/3",))
    with pytest.raises(ValueError, match="LookRow already"):
        lookbar.LookRow(row)


def test_a_rule_key_of_both_modules_is_refused_and_never_drops_a_text(monkeypatch):
    """A key that ``lookread.RULES`` and ``lookbar.RULES`` both hold would leave one text out of every row's
    description and every companion record: the one table is not made then."""
    monkeypatch.setitem(lookbar.RULES, "window", "a second text under the look reading's key")
    with pytest.raises(lookbar.LookBarError, match="both hold the key.*window"):
        lookbar.rule_texts()
    with pytest.raises(lookbar.LookBarError, match="both hold the key"):
        panels.tracker_rows(BODY, RUN / BODY, ("R0L",))["R0L"].describe()


#: Code state A (``STAGE2_75_PLAN.md``: "Code state A is `c377ba1`", the commit that records the third freeze of the
#: 75 cm sets), and the two hashes of the modules in every existing record's key as they are there (the registered
#: pair: ``bar.bar_sources()`` over bar.py, readings.py, setio.py and speedfit.py; ``panels.panel_sources()``).
STATE_A = "c377ba1"
BAR_SOURCES_A = "45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d"
PANEL_SOURCES_A = "c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1"
#: Design D18's list of what round C edits nothing of, less ``barreport.py``: the working branch changed that file
#: after state A for the 75 cm round's bar tables (it writes reports and is in no record's key).
ROUND_A_PATHS = ("designlab/stage1",) + tuple(f"designlab/stage2/{m}.py" for m in (
    "bar", "readings", "setio", "s2io", "moves", "sets", "setscreen"))


def test_the_modules_of_every_existing_record_key_are_state_as():
    """Design D18: code state B changes no module of any existing record key.  The two hashes a bar record's key
    holds of its sources are the registered ones of state A (this runs in an export without git too)."""
    assert bar.bar_sources()["sha256"] == BAR_SOURCES_A
    assert panels.panel_sources()["sha256"] == PANEL_SOURCES_A
    assert sorted(bar.bar_sources()["files"]) == ["designlab/stage1/speedfit.py"] + [
        f"designlab/stage2/{m}.py" for m in ("bar", "readings", "setio")]


def test_no_round_a_or_stage_1_file_differs_from_state_a():
    """Design D18 and section 4: ``git diff <state A> -- designlab/stage1 designlab/stage2/{bar,readings,setio,
    s2io,moves,sets,setscreen}.py`` names no file (the files as they are in this checkout, committed or not).
    Skipped where git cannot tell: an export without a repository, or a clone whose history does not reach state
    A; the hashes of the test above hold there all the same."""
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")  # (a read: no index.lock is left behind)

    def git(*args):
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, env=env)

    try:
        tracked = git("ls-files", "--error-unmatch", "designlab/stage2/bar.py")  # (this tree is the repository's own)
        known = git("cat-file", "-e", STATE_A + "^{commit}")
    except OSError:
        pytest.skip("no git here")
    if tracked.returncode != 0 or known.returncode != 0:
        pytest.skip(f"git cannot tell: no repository here, or its history does not reach state A ({STATE_A})")
    for path in ROUND_A_PATHS:
        assert (ROOT / path).exists(), path
    diff = git("diff", "--no-ext-diff", "--name-only", STATE_A, "--", *ROUND_A_PATHS)
    assert diff.returncode == 0, diff.stderr[-2000:]
    assert diff.stdout.split() == [], "differ from state A: " + diff.stdout


def test_rows_are_the_bars_names_with_every_inner_row_a_look_row(tmp_path):
    rows = lookbar.rows(BODY, ["R0L@10ms", "R0@0ms", "R0L@6ms"], sidecar=tmp_path / "side")
    assert list(rows) == ["R0L@10ms", "R0@0ms", "R0L@6ms"]
    for name, row in rows.items():
        look = lookbar.look_row_of(row)
        assert isinstance(row, bar.Delayed) and row.name == name and isinstance(row.inner, lookbar.LookRow)
        assert look is row.inner and look.sidecar == str(tmp_path / "side") and look.sidecar_row == name
        assert row.k == bar.parse_row(name)[1]
    # a row that is no LookRow is wrapped under its own name; R0L keeps its own
    assert rows["R0@0ms"].inner.name == "R0" and rows["R0L@10ms"].inner.name == "R0L"
    assert rows["R0@0ms"].inner.describe() == rows["R0L@10ms"].inner.describe()  # (they execute the same)
    # what the delay wrapper adds to a description is what ``collect`` takes off again
    row = rows["R0L@10ms"]
    assert set(row.describe()) - set(row.inner.describe()) == set(lookbar.DELAYED_KEYS)
    rec = dict(controller=plain(s2io.portable(row.describe())))
    assert lookbar.record_row_sha256(rec) == row.inner.row_sha256()
    assert sorted(row.sources()) == ["designlab/stage1/speedfit.py", "designlab/stage2/bar.py",
                                     "designlab/stage2/lookbar.py", "designlab/stage2/lookread.py"]
    assert lookbar.rows(BODY, ["R0L@10ms"])["R0L@10ms"].inner.sidecar is None
    assert lookbar.look_row_of(bar.rows(BODY, ["R0@10ms"])["R0@10ms"]) is None
    with pytest.raises(ValueError, match="<row>@<ms>ms"):
        lookbar.rows(BODY, ["R0L"])
    with pytest.raises(ValueError, match="registered"):
        lookbar.rows(BODY, ["R9/1@10ms"])
    assert lookbar.sidecar_path("s", "R5/42@10ms", "walk/v0.4", 3042) == Path("s/R5/42@10ms/walk/v0.4/3042.json")


def test_a_spec_round_trip_rebuilds_the_row(tmp_path):
    row = r0l(3, sidecar=tmp_path / "side")
    again = panels.row_from_spec(plain(row.spec()))
    assert type(again) is bar.Delayed and type(again.inner) is lookbar.LookRow and again.name == row.name
    assert again.describe() == row.describe() and again.sources() == row.sources() and again.k == 3
    assert (again.inner.name, again.inner.sidecar, again.inner.sidecar_row) == ("R0L", str(tmp_path / "side"), "R0L@6ms")
    assert again.inner.inner.describe() == row.inner.inner.describe() and again.inner.row_sha256() == row.inner.row_sha256()
    assert row.spec()["inner"]["cls"] == "designlab.stage2.lookbar.LookRow"


def test_what_is_relied_on_from_the_bar():
    """Design 2.3's last paragraph: a later change to any of these is seen here."""
    assert callable(bar._Played.q) and callable(bar._Played.loads)
    assert bar.BarBuilt.recorder is not panels.Built.recorder and "recorder" in vars(bar.BarBuilt)
    for name in ("rows", "evaluate", "run_episode", "context_for", "counts", "conditions", "stock_conditions",
                 "block_dir", "check_manifest", "set_crouch", "walks_of", "parse_row", "is_bar"):
        assert callable(getattr(bar, name)), name
    assert issubclass(bar.Unreadable, RuntimeError) and bar.STOCK_SET == "S1"
    assert bar.BAR_JUDGES == ("bar_walk", "bar_turn", "bar_nudge", "bar_stop")
    assert set(lookbar.READ_JUDGES) | {lookbar.STOP_JUDGE} < set(bar.BAR_JUDGES)
    # the bar's own names of an episode's fields that the companion reads
    assert bar.READINGS == {"success": ("succeeded", "failed_terms"), "but_A1": ("succeeded_but_A1", "failed_terms_but_A1")}
    # as built, the bar's runner lets only R0 rows run on a set that is none of the 75 cm round's three
    assert bar.controller_of("R0L@10ms") == "R0L" and bar.registered_name("V6") is None
    with pytest.raises(bar.BarError, match="only R0 rows"):
        bar._check_rule("V6", ["R0L@10ms"], 20, None, None, None, None)
    bar._check_rule("V6", ["R0@10ms"], 20, None, None, None, None)
    bar._check_rule("S1", ["R0L@10ms", "R5/42@10ms"], 20, None, None, None, None)


# ------------------------------------------------------------------------------------------------
# Pure readings of a recorder
# ------------------------------------------------------------------------------------------------
def fake_rec(t, t_ref, com=None):
    U = np.full((len(t), len(metrics.COLS)), np.nan)
    U[:, metrics.C["t"]], U[:, metrics.C["t_ref"]] = t, t_ref
    if com is not None:
        U[:, metrics.C["com_x"]], U[:, metrics.C["com_y"]] = com[:, 0], com[:, 1]
    rec = types.SimpleNamespace(U=U, rows=len(t), t=U[:, metrics.C["t"]], col=lambda n: U[:, metrics.C[n]])
    rec.sim_time = lambda x: None if (x > t_ref[-1] + 1e-12 or x < t_ref[0] - 1e-12) else float(np.interp(x, t_ref, t))
    return rec


def test_the_clock_block():
    t = np.arange(0, 10.0, 0.01)
    c = lookbar.clock_block(dict(reference_time_s=10.5), fake_rec(t, 1.05 * t), 12.0)
    assert c["mean_factor"] == pytest.approx(1.0, abs=1e-12) and c["min_factor"] == pytest.approx(1.0, abs=1e-9)
    assert c["t_ref_reached"] == 10.5 and c["t_ref_planned"] == pytest.approx(12.6) and c["t_ref_start"] == 0.0
    assert c["t_ref_last_update"] == pytest.approx(1.05 * 9.99) and c["rate"] == 1.05 and c["updates"] == 1000
    # a clock held 20 % slow in the second half
    tr = np.where(t < 5.0, 1.05 * t, 1.05 * 5.0 + 0.84 * (t - 5.0))
    c = lookbar.clock_block({}, fake_rec(t, tr), 12.0)
    assert c["mean_factor"] == pytest.approx((5.0 + 0.8 * 4.99) / 9.99, abs=1e-9)
    assert c["min_factor"] == pytest.approx(0.8, abs=1e-9) and c["t_ref_reached"] == pytest.approx(tr[-1])
    one = lookbar.clock_block({}, fake_rec(t[:1], tr[:1]), 1.0)
    assert one["mean_factor"] is None and one["min_factor"] is None and one["t_ref_reached"] == 0.0
    none = lookbar.clock_block({}, types.SimpleNamespace(rows=0), 1.0)
    assert none["t_ref_reached"] is None and none["t_ref_planned"] == pytest.approx(1.05)


def test_on_command_at_stop_is_the_stride_mean_before_the_command():
    t = np.arange(0, 12.0, 0.01)
    b = types.SimpleNamespace(speed_cmd=0.42, stride_s=0.8, home_yaw=0.0, command=dict(ref_s=1.05 * 8.0))

    def read(speed_before, yaw=0.0):
        x = np.where(t < 7.2, 0.42 * t, 0.42 * 7.2 + speed_before * (t - 7.2))
        com = np.column_stack([np.cos(yaw) * x, np.sin(yaw) * x])
        return lookbar.on_command_at_stop(types.SimpleNamespace(**dict(vars(b), home_yaw=yaw)), fake_rec(t, 1.05 * t, com))

    on = read(0.42)
    assert on["term"] is True and on["speed_m_s"] == pytest.approx(0.42) and on["needed_m_s"] == pytest.approx(0.378)
    assert on["command_s"] == pytest.approx(8.0) and on["stride_s"] == 0.8 and on["frac"] == 0.9 and on["reason"] is None
    assert read(0.3781)["term"] is True and read(0.3779)["term"] is False  # the limit, without slack
    assert read(0.0)["speed_m_s"] == pytest.approx(0.0, abs=1e-12)  # a row that stands before its command
    assert read(0.42, yaw=1.0)["speed_m_s"] == pytest.approx(0.42)  # along the commanded heading
    # sideways motion is no forward speed
    side = lookbar.on_command_at_stop(b, fake_rec(t, 1.05 * t, np.column_stack([0 * t, 0.42 * t])))
    assert side["term"] is False and side["speed_m_s"] == pytest.approx(0.0, abs=1e-12)
    # a trace that never reaches the command, or begins less than a stride before it: the term fails, with its reason
    short = lookbar.on_command_at_stop(b, fake_rec(t[:700], 1.05 * t[:700], np.column_stack([0.42 * t[:700], 0 * t[:700]])))
    assert short["term"] is False and short["speed_m_s"] is None and "never reached" in short["reason"]
    early = types.SimpleNamespace(**dict(vars(b), command=dict(ref_s=1.05 * 0.5)))
    late = lookbar.on_command_at_stop(early, fake_rec(t, 1.05 * t, np.column_stack([0.42 * t, 0 * t])))
    assert late["term"] is False and late["command_s"] == pytest.approx(0.5) and "one stride" in late["reason"]


def test_the_short_air_runs_of_a_window():
    dt, n = 0.002, 5000
    contact = np.ones((n, 2), bool)
    contact[1000:1200, 0] = False  # a swing of 0.4 s
    contact[2000:2030, 0] = False  # an air run of 60 ms
    contact[2500:2520, 1] = False  # 40 ms
    contact[4990:, 1] = False  # cut by the trace: not complete
    contact[400:450, 1] = False  # before the window
    ser = dict(contact=contact, dt=dt, t50=np.arange(0, n * dt, 0.02))
    a = lookbar.air_runs(ser, 1.5, 9.0, 0.18)
    assert (a["in_window"], a["shorter_than_air_gap_max"], a["air_gap_max_s"]) == (3, 2, 0.18)
    assert a["short"] == [[0, pytest.approx(4.0), pytest.approx(0.06)], [1, pytest.approx(5.0), pytest.approx(0.04)]]
    assert lookbar.air_runs(ser, 4.5, 9.0, 0.18)["in_window"] == 1  # the first swing lies before this window
    assert lookbar.air_runs(ser, 1.5, 9.0, 0.05)["shorter_than_air_gap_max"] == 1


# ------------------------------------------------------------------------------------------------
# One companion episode, on tables
# ------------------------------------------------------------------------------------------------
TERMS = dict(bob=True, head=True, tail=True, toe_off=True)
TRACE = "Traceback (most recent call last):\n  File \"bar.py\", line 1, in run_episode\n"
UNREADABLE = TRACE + "designlab.stage2.bar.Unreadable: walk/v0.4 seed 3042: no fall, and the trace ends at 9.000 s"
POLICY = dict(versions_trained=dict(torch="2.15.0+cu126"), versions_judged=dict(torch="2.15.0+cpu"), cross_version=False,
              fingerprint_max_abs_diff=2e-7, fingerprint_tol=1e-5, checkpoint=dict(steps=500000), source_change=False)


def bar_ep(seed=3042, fell=False, ok=True, a1=None, failed=(), survival=20.0, steps=10000, error=None) -> dict:
    """A bar episode record as far as the companion reads it."""
    if error is not None:
        return dict(seed=seed, fell=True, survival_s=0.0, planned_s=None, succeeded=False, succeeded_but_A1=False,
                    failed_terms=["survived", "along"], failed_terms_but_A1=["survived", "along"], error=error)
    a1 = ok if a1 is None else a1
    return dict(seed=seed, fell=fell, survival_s=survival, planned_s=20.0, succeeded=bool(ok and not fell),
                succeeded_but_A1=bool(a1 and not fell), failed_terms=["survived"] if fell else list(failed),
                failed_terms_but_A1=["survived"] if fell else [t for t in failed if t != "drive_sustained"],
                trace=dict(steps=steps, sha256="ab" * 32), command_s=None, window_reached=not fell)


def block(seed=3042, key="walk/v0.4", fell=False, survival=20.0, steps=10000, terms=None, read=True, factor=1.0,
          policy=None, on_command=None, fault=None, reason=None, row_sha256="r" * 64) -> dict:
    """A row's block of an episode (the sidecar's document) as far as the companion reads it."""
    terms = dict(TERMS, **(terms or {})) if read else None
    return dict(schema=lookbar.SIDECAR_SCHEMA, row="R0L@10ms", condition=key, seed=seed, steps=steps, fell=fell,
                survival_s=survival, planned_s=20.0, fall_cause="tilt" if fell else None,
                clock=dict(mean_factor=factor, min_factor=factor, t_ref_reached=1.05 * factor * survival,
                           t_ref_planned=21.0, t_ref_start=0.0),
                read=read, reason=reason, fault=fault, window=[10.0, 20.0] if read else None, window_ref=[10.5, 21.0],
                own=dict(steps=23) if read else None, reference=dict(steps=23), terms=terms,
                not_read=[] if read else None, too_few_steps=False if read else None,
                look_ok=all(terms.values()) if read else None, compare=None,
                air_runs=dict(in_window=23, shorter_than_air_gap_max=1) if read else None,
                on_command_at_stop=on_command, policy_check=policy, state_digest="d" * 64, row_sha256=row_sha256)


def test_a_read_walk_episode_on_a_look_set_and_on_another():
    e, b = bar_ep(), block()
    on = lookbar.companion_episode("walk/v0.4", "bar_walk", 3042, e, b, True)
    assert (on["outcome"], on["succeeded"], on["succeeded_but_A1"], on["succeeded_look"], on["succeeded_but_A1_look"]) \
        == ("read", True, True, True, True)
    assert on["failed_terms"] == [] and on["look"]["look_ok"] is True and on["reading_1"] == lookbar.READING_1["walk_turn"]
    assert on["clock"]["t_ref_reached"] == pytest.approx(21.0) and on["planned_s"] == 20.0 and on["survival_s"] == 20.0
    assert on["state_digest"] == "d" * 64 and on["error"] is None and on["reason"] is None and on["fell"] is False
    assert sorted(on["look"]) == sorted(lookbar.LOOK_KEYS) and on["bar"]["trace"]["steps"] == 10000
    # a failed look term: the bar's readings stand, reading 1 and "but A1 with the look" fail, the term is named
    b2 = block(terms=dict(bob=False, tail=False))
    on = lookbar.companion_episode("walk/v0.4", "bar_walk", 3042, e, b2, True)
    assert (on["succeeded"], on["succeeded_but_A1"], on["succeeded_look"], on["succeeded_but_A1_look"]) == (True, True, False, False)
    assert on["failed_terms"] == ["look_bob", "look_tail"] == on["failed_terms_but_A1"] and on["look_failed_terms"] == ["bob", "tail"]
    # the bar's reading 2 fails on the tail drive alone (A1): reading 3 with the look holds
    e3 = bar_ep(ok=False, a1=True, failed=("drive_sustained",))
    on = lookbar.companion_episode("turn/q90_left", "bar_turn", 3042, e3, b, True)
    assert (on["succeeded"], on["succeeded_but_A1"], on["succeeded_look"], on["succeeded_but_A1_look"]) == (False, True, False, True)
    assert on["failed_terms"] == ["drive_sustained"] and on["failed_terms_but_A1"] == []
    # on any other set the numbers are recorded and decide nothing: reading 1 is null, no look term is a failed term
    off = lookbar.companion_episode("walk/v0.4", "bar_walk", 3042, e, b2, False)
    assert (off["outcome"], off["succeeded"], off["succeeded_but_A1"], off["succeeded_look"], off["succeeded_but_A1_look"]) \
        == ("read", True, True, None, None)
    assert off["failed_terms"] == [] and off["look_failed_terms"] == ["bob", "tail"] and off["look_ok"] is False
    assert off["reading_1"] == lookbar.READING_1["not_counted"] and off["look"]["terms"]["bob"] is False


def test_a_fall_an_error_and_an_unread_episode():
    fell = lookbar.companion_episode("walk/v0.4", "bar_walk", 3042, bar_ep(fell=True, survival=7.5),
                                     block(fell=True, survival=7.5, read=False, reason="fell"), True)
    assert (fell["outcome"], fell["fell"], fell["survival_s"], fell["failed_terms"]) == ("fell", True, 7.5, ["survived"])
    assert not any(fell[k] for k in ("succeeded", "succeeded_but_A1", "succeeded_look", "succeeded_but_A1_look"))
    assert fell["look"] is None and fell["look_reason"] == "fell" and fell["clock"]["t_ref_reached"] == pytest.approx(7.875)
    # the bar could not read it, it did not fall, the row has a policy: unread, in no reading succeeding
    slow = block(factor=0.9, policy=POLICY)
    for e, error in ((bar_ep(error=UNREADABLE), None), (None, "designlab.stage2.bar.Unreadable: walk/v0.4 seed 3042: x")):
        un = lookbar.companion_episode("walk/v0.4", "bar_walk", 3042, e, slow, True, error)
        assert (un["outcome"], un["reason"], un["fell"], un["error"]) == ("unread", "clock_behind", False, None)
        assert un["unreadable"].startswith("designlab.stage2.bar.Unreadable: walk/v0.4 seed 3042")
        assert not any(un[k] for k in ("succeeded", "succeeded_but_A1", "succeeded_look", "succeeded_but_A1_look"))
        assert un["survival_s"] == 20.0 and un["planned_s"] == 20.0 and un["failed_terms"] == []
        assert un["clock"]["t_ref_reached"] == pytest.approx(18.9) and un["clock"]["t_ref_planned"] == 21.0
        assert un["policy_check"] == POLICY
    kept = lookbar.companion_episode("walk/v0.4", "bar_walk", 3042, bar_ep(error=UNREADABLE), block(policy=POLICY), True)
    assert (kept["outcome"], kept["reason"]) == ("unread", "unreadable")  # its clock kept the commanded rate
    edge = block(factor=1.0 - 1e-12, policy=POLICY)
    assert lookbar.companion_episode("w", "bar_walk", 1, bar_ep(error=UNREADABLE), edge, True)["reason"] == "unreadable"
    # the same of a row without a policy is an error (C8 stands), and so is any other fault, with or without a policy
    err = lookbar.companion_episode("walk/v0.4", "bar_walk", 3042, bar_ep(error=UNREADABLE), block(factor=0.9), True)
    assert (err["outcome"], err["reason"]) == ("error", None) and "Unreadable" in err["error"]
    boom = TRACE + "RuntimeError: boom"
    for b in (block(policy=POLICY), None):
        err = lookbar.companion_episode("walk/v0.4", "bar_walk", 3042, bar_ep(error=boom), b, True)
        assert err["outcome"] == "error" and err["error"].endswith("RuntimeError: boom") and err["succeeded_look"] is False
    assert err["fell"] is None and err["survival_s"] == 0.0 and err["clock"] is None and err["bar"] is None
    # an unreadable episode that fell is no "unread" (the bar raises Unreadable only without a fall; a block that says
    # otherwise is an error)
    fell_block = block(fell=True, survival=3.0, policy=POLICY)
    assert lookbar.companion_episode("w", "bar_walk", 1, bar_ep(error=UNREADABLE), fell_block, True)["outcome"] == "error"
    assert lookbar.is_unreadable(UNREADABLE) and lookbar.is_unreadable("Unreadable: x") and not lookbar.is_unreadable(boom)
    assert not lookbar.is_unreadable(None) and not lookbar.is_unreadable("this was not Unreadable: at all")
    # the last line decides: an Unreadable that another fault followed is that fault
    assert not lookbar.is_unreadable(UNREADABLE + "\n\nDuring handling of the above exception, another exception "
                                     "occurred:\n\n" + boom)


def test_a_look_reading_that_cannot_be_given_is_an_error_on_a_look_set_only():
    fault = block(read=False, fault="look reading: LookError: the reference of walk/v0.4: not played: x")
    on = lookbar.companion_episode("walk/v0.4", "bar_walk", 3042, bar_ep(), fault, True)
    assert on["outcome"] == "error" and "rule U5" in on["error"] and "not played" in on["error"]
    assert not on["succeeded"] and not on["succeeded_look"] and on["look"] is None and on["look_fault"] == fault["fault"]
    # on another set the bar's reading stands and the fault is recorded
    off = lookbar.companion_episode("walk/v0.4", "bar_walk", 3042, bar_ep(), fault, False)
    assert (off["outcome"], off["succeeded"], off["error"]) == ("read", True, None) and off["look_fault"] == fault["fault"]
    # a fall stays a fall, whatever the reading
    fell = lookbar.companion_episode("turn/q90_left", "bar_turn", 3042, bar_ep(fell=True), dict(fault, fell=True), True)
    assert fell["outcome"] == "fell"
    # a nudge or a stop item has no look reading: none is missed
    nudge = lookbar.companion_episode("nudge/left", "bar_nudge", 3042, bar_ep(), block(read=False, reason="not read"), True)
    assert (nudge["outcome"], nudge["succeeded"], nudge["succeeded_look"], nudge["succeeded_but_A1_look"]) == ("read", True, True, True)
    assert nudge["reading_1"] == lookbar.READING_1["other"] and nudge["look"] is None


def test_the_stop_items_term_decides_on_a_look_set_only():
    met = dict(term=True, speed_m_s=0.41, needed_m_s=0.378)
    missed = dict(term=False, speed_m_s=0.05, needed_m_s=0.378)
    for oc, good in ((met, True), (missed, False)):
        b = block(key="stop/two_step", read=False, reason="not read", on_command=oc)
        on = lookbar.companion_episode("stop/two_step", "bar_stop", 3042, bar_ep(), b, True)
        assert (on["succeeded"], on["succeeded_but_A1"], on["succeeded_look"], on["succeeded_but_A1_look"]) == (good,) * 4
        assert on["failed_terms"] == ([] if good else ["on_command_at_stop"]) == on["failed_terms_but_A1"]
        assert on["on_command_at_stop"] == oc and on["bar"]["succeeded"] is True and on["outcome"] == "read"
        off = lookbar.companion_episode("stop/two_step", "bar_stop", 3042, bar_ep(), b, False)
        assert (off["succeeded"], off["succeeded_but_A1"], off["succeeded_look"]) == (True, True, None)
        assert off["failed_terms"] == [] and off["on_command_at_stop"] == oc  # recorded, deciding nothing
    # a stop item whose block holds no term (the reading raised) does not meet it
    none = lookbar.companion_episode("stop/two_step", "bar_stop", 3042, bar_ep(), block(read=False), True)
    assert none["succeeded"] is False and none["failed_terms"] == ["on_command_at_stop"]


def test_a_stock_condition_has_stage_1s_one_reading():
    e = dict(seed=3042, fell=False, survival_s=18.4, planned_s=18.4, succeeded=True, failed_terms=[])
    b = block(key="walk/nominal", survival=18.4, read=False, reason="a stock condition has no look reading")
    ep = lookbar.companion_episode("walk/nominal", "walk", 3042, e, b, False)
    assert (ep["outcome"], ep["succeeded"], ep["succeeded_but_A1"], ep["succeeded_look"]) == ("read", True, None, None)
    assert ep["failed_terms_but_A1"] is None and ep["look"] is None


WARNED = dict(BADQACC=dict(number=1, lastinfo=0))
WARNED_CAUSE = "MuJoCo warning BADQACC (x1)"  # (tune.warned_fall's cause of WARNED)


def warned_pair(seed=3042, key="walk/v0.4", survival=20.0, policy=POLICY) -> tuple[dict, dict]:
    """A bar episode and its block of an episode during which MuJoCo warned, as ``tune.warned_fall`` leaves them:
    fell, at 0 s, with the warnings; the time simulated is the bar's ``survival_s``."""
    e = dict(bar_ep(seed, fell=True, survival=survival), fall_time_s=0.0, fall_cause=WARNED_CAUSE,
             mujoco_warnings=WARNED)
    b = dict(block(seed, key=key, fell=True, survival=survival, read=False, reason="fell", policy=policy),
             fall_time_s=0.0, fall_cause=WARNED_CAUSE, mujoco_warnings=WARNED)
    return e, b


def test_an_episode_during_which_mujoco_warned_is_a_fall_at_its_start_credited_no_survival():
    """The lead's ruling 9 of 17 Nov 2026 (``tune.warned_fall``): such an episode is a fall at its start.  Its
    companion episode holds ``warned`` and ``fall_time_s`` 0 and survived nothing (the time simulated is beside it),
    so the monitor's rule M3 credits it no survival.  An ordinary fall keeps its fall time and its survival."""
    m2 = pytest.importorskip("designlab.stage2.monitor2")
    e, b = warned_pair()
    for look_set in (True, False):
        ep = lookbar.companion_episode("walk/v0.4", "bar_walk", 3042, e, b, look_set)
        assert (ep["outcome"], ep["fell"], ep["warned"], ep["fall_time_s"], ep["survival_s"], ep["simulated_s"]) == (
            "fell", True, True, 0.0, 0.0, 20.0)
        assert ep["failed_terms"] == ["survived"] and ep["fall_cause"] == WARNED_CAUSE
        assert not any(ep[k] for k in ("succeeded", "succeeded_but_A1", "succeeded_look", "succeeded_but_A1_look"))
        assert m2.has_clock(ep) and m2.survival_credit(ep) == 0.0
    s = m2.summarize([ep])
    assert (s["S"], s["mean_survival_s"], s["outcomes"]["fell"]) == (0.0, 0.0, 1)
    # either document says so: the block, the bar's episode record, or a fall cause of tune.warned_fall's alone
    for ee, bb in ((bar_ep(fell=True), b), (e, block(fell=True, read=False)),
                   (bar_ep(fell=True), dict(block(fell=True, read=False), fall_cause=WARNED_CAUSE))):
        ep = lookbar.companion_episode("walk/v0.4", "bar_walk", 3042, ee, bb, True)
        assert (ep["outcome"], ep["warned"], ep["fall_time_s"], ep["survival_s"], ep["simulated_s"]) == (
            "fell", True, 0.0, 0.0, 20.0)
    # an ordinary fall: its own fall time (the block's, else the bar's) and its survival, credited as its clock ran
    for ee, bb in ((bar_ep(fell=True, survival=7.5), dict(block(fell=True, survival=7.5, read=False), fall_time_s=7.4)),
                   (dict(bar_ep(fell=True, survival=7.5), fall_time_s=7.4), block(fell=True, survival=7.5, read=False))):
        fell = lookbar.companion_episode("walk/v0.4", "bar_walk", 3042, ee, bb, True)
        assert (fell["outcome"], fell["warned"], fell["fall_time_s"], fell["survival_s"], fell["simulated_s"]) == (
            "fell", False, 7.4, 7.5, 7.5)
        assert m2.survival_credit(fell) == pytest.approx(7.5)
    read = lookbar.companion_episode("walk/v0.4", "bar_walk", 3042, bar_ep(), block(), True)
    assert (read["warned"], read["fall_time_s"], read["survival_s"], read["simulated_s"]) == (False, None, 20.0, 20.0)
    assert m2.survival_credit(read) == pytest.approx(20.0)
    assert not lookbar.is_warned(None, {}, dict(fall_cause="tilt", mujoco_warnings={})) and lookbar.is_warned(None, e)
    # the rule is in every row's description (and so in its key) and in every companion record
    assert "MuJoCo warned is a fall at its start" in lookbar.RULES["outcome"]
    assert "survival_s 0" in lookbar.rule_texts()["outcome"]


def test_a_fault_names_no_path_of_this_machine():
    """The lead's ruling 8 of 17 Nov 2026: a fault is ``<what>: <Type>: <message>`` without this machine's paths: a
    file under the study is named relative to it, any other one by its last three parts; a condition key, a speed in
    m/s or a share is no path."""
    here = s2io.STUDY / "stage2" / "sets" / "V6" / BODY / "refs" / "v0.4.npz"
    f = lookbar._fault("look reading", FileNotFoundError(2, "No such file or directory", str(here)))
    assert f == f"look reading: FileNotFoundError: [Errno 2] No such file or directory: 'stage2/sets/V6/{BODY}/refs/v0.4.npz'"
    there = OSError(f"cannot read /var/tmp/pytest-of-x/test0/sets0/T0/{BODY}/refs/v0.4.npz: gone")
    assert lookbar._fault("clock", there) == f"clock: OSError: cannot read .../{BODY}/refs/v0.4.npz: gone"
    keep = "walk/v0.4 at 0.40 m/s over 1/2 a stride, R5/42@10ms (rule K4), the ratio 3 / 4"
    assert lookbar._no_paths(keep) == keep and lookbar._fault("x", ValueError(keep)) == f"x: ValueError: {keep}"
    once = lookbar._no_paths(f"{s2io.STUDY}/a/b.npz and '/x/y/z/w.npz'")
    assert once == "a/b.npz and '.../y/z/w.npz'" and lookbar._no_paths(once) == once
    # the lead's ruling 12 of 17 Nov 2026 (the check of the rows package, low point 2): no path keeps its first part,
    # however short (as built a path of three parts or fewer was kept whole); one with a space is cut up to it
    for text, cut in (("/home/user/a", ".../user/a"), ("at /content: gone", "at ...: gone"),
                      ("/home/user/b c/d", ".../user/b c/d"), ("/a/b/c/d/e", ".../c/d/e"),
                      ("/content/drive/My Drive/mesozoic/stage2/sets/V6/x.npz",
                       ".../drive/My Drive/mesozoic/stage2/sets/V6/x.npz")):
        assert lookbar._no_paths(text) == cut, text
        assert text.split("/")[1].split()[0] not in cut and lookbar._no_paths(cut) == cut, text
    # a path under an output root given to it is named relative to that root (a summary's own paths are)
    assert lookbar._no_paths("/r/out/V6/B/eval/x.json: bad", roots=("/r/out",)) == "V6/B/eval/x.json: bad"
    assert lookbar._no_paths("/r/out and /r/outer/x", roots=("/r/out",)) == ".../out and .../outer/x"


# ------------------------------------------------------------------------------------------------
# collect, on synthetic bar records and sidecars
# ------------------------------------------------------------------------------------------------
def bar_record(row, set_name, key, judge, episodes, a1_over=False, status="done", stock=False) -> dict:
    behavior, name = key.split("/")
    rec = dict(schema=bar.SCHEMA, body=BODY, set=set_name, block=bar.HEADLINE_BLOCK, row=row.name, behavior=behavior,
               condition=name, key_name=key, gated=True, status=status, a1_over=a1_over,
               controller=plain(s2io.portable(row.describe())), definition=dict(judge=judge, stock=stock),
               seeds=SEEDS[: len(episodes)] if status == "done" else SEEDS, episodes=episodes, key_sha256="k" * 64)
    if status == "done":
        rec.update(counts=bar.counts(episodes, rec["seeds"], "success"),
                   counts_but_A1=None if stock else bar.counts(episodes, rec["seeds"], "but_A1"))
        if a1_over:
            rec["counts"] = dict(rec["counts"], pass_=None, not_read="a1_over", reason="not read: a1_over",
                                 pass_if_read=rec["counts"]["pass_"])
    else:
        rec.update(counts=dict(reading="success", episodes=0, errors=0, judged=False, pass_=None), counts_but_A1=None,
                   reason="no reference", episodes=[])
    return plain(rec)


def write_run(out: Path, set_name: str, row, key: str, judge: str, pairs, **kw) -> tuple[Path, Path]:
    """A synthetic bar record of ``(bar episode, block or None)`` pairs with its sidecars under ``out``."""
    behavior, name = key.split("/")
    bdir = bar.block_dir(out, set_name, BODY, bar.HEADLINE_BLOCK)
    sha = lookbar.look_row_of(row).row_sha256()
    for e, b in pairs:
        if b is not None:
            s2io.write_json(lookbar.sidecar_path(bdir / "look_partial", row.name, key, e["seed"]),
                            dict(b, condition=key, seed=e["seed"], row=row.name, row_sha256=sha))
    path = bdir / "eval" / row.name / behavior / f"{name}.json"
    s2io.write_json(path, bar_record(row, set_name, key, judge, [e for e, _ in pairs], **kw))
    return path, lookbar.companion_path(out, set_name, BODY, bar.HEADLINE_BLOCK, row.name, behavior, name)


def twenty(special: dict | None = None, key="walk/v0.4", policy=None, **each) -> list:
    """20 episodes that survive, succeed and pass the look, each with its block; ``special``: seed -> (e, b)."""
    out = []
    for s in SEEDS:
        if special and s in special:
            out.append(special[s])
        else:
            out.append((bar_ep(s), block(s, key=key, policy=policy, **each)))
    return out


def test_collect_counts_the_three_readings_with_the_bars_arithmetic(tmp_path):
    row = r0l()
    special = {
        3043: (bar_ep(3043), block(3043, terms=dict(head=False))),  # read, the look fails
        3044: (bar_ep(3044, ok=False, a1=True, failed=("drive_sustained",)), block(3044)),  # A1 alone
        3045: (bar_ep(3045, fell=True, survival=4.0, steps=2000), block(3045, fell=True, survival=4.0, steps=2000, read=False)),
    }
    path, cpath = write_run(tmp_path, "V6", row, "walk/v0.4", "bar_walk", twenty(special))
    summary = lookbar.collect(tmp_path, "V6", BODY)
    doc = s2io.read_json(cpath)
    assert doc["schema"] == "s2-look-1" and doc["look_set"] is True and doc["row"] == "R0L@10ms" and doc["condition"] == "walk/v0.4"
    assert doc["bar_record"] == dict(path="V6/75-M-H-10/seeds3042/eval/R0L@10ms/walk/v0.4.json",
                                     sha256=s2io.sha256_file(path), key_sha256="k" * 64, status="done")
    assert [e["outcome"] for e in doc["episodes"]].count("read") == 19 and doc["outcomes"] == dict(read=19, fell=1, unread=0, error=0)
    eps = doc["episodes"]
    # each count is bar.counts on the companion's episodes in that reading
    for key, field, name in (("counts", "succeeded", "success"), ("counts_but_A1", "succeeded_but_A1", "but_A1"),
                             ("counts_look", "succeeded_look", "look"),
                             ("counts_but_A1_look", "succeeded_but_A1_look", "but_A1_look")):
        want = bar.counts([dict(seed=e["seed"], fell=e["outcome"] == "fell", succeeded=e[field]) for e in eps], SEEDS)
        assert doc[key] == plain(dict(want, reading=name)), key
    assert [doc[k]["succeeded"] for k in ("counts", "counts_but_A1", "counts_look", "counts_but_A1_look")] == [18, 19, 17, 18]
    assert all(doc[k]["survived"] == 19 and doc[k]["falls"] == 1 and doc[k]["judged"] and doc[k]["pass_"] is False
               for k in ("counts", "counts_look"))
    assert doc["failures_by_term"] == dict(drive_sustained=1, look_head=1, survived=1)
    assert doc["look_failures_by_term"] == dict(head=1) and doc["policy"] is None and doc["counts_equal_bar"] is None
    assert doc["short_air_runs"] == dict(episodes_read=19, episodes_with_any=19, total=19)
    assert doc["bands"] == dict(lookread.BANDS) and set(doc["rules"]) == set(lookread.RULES) | set(lookbar.RULES)
    # every rule text of both modules, none dropped: the look reading's rule U5 text is the one under "reference"
    assert doc["rules"] == plain(lookbar.rule_texts()) and len(doc["rules"]) == len(lookread.RULES) + len(lookbar.RULES)
    assert doc["rules"]["reference"] == lookread.RULES["reference"] != doc["rules"]["reference_use"]
    assert sorted(doc["sidecars"]["sha256"]) == sorted(str(s) for s in SEEDS) and "commit" in doc["provenance"]
    assert sorted(doc["sources"]) == ["bar", "designlab/stage2/lookbar.py", "designlab/stage2/lookread.py"]
    assert doc["sha256"] == s2io.sha256_json({k: v for k, v in doc.items() if k != "sha256"})
    assert doc["versions"] == panels.library_versions() and doc["blas_threads"] == 1 and doc["at"]
    assert summary["sha256"] == s2io.sha256_json({k: v for k, v in summary.items() if k != "sha256"})
    assert summary["records"][0]["counts_look"] == doc["counts_look"] and summary["look_set"] is True and summary["refused"] == []
    assert s2io.read_json(cpath.parents[2] / "summary.json")["records"][0]["record"].endswith("look/R0L@10ms/walk/v0.4.json")
    assert "with the look 17/20" in lookbar.summary_text(summary)
    # collecting again leaves the companion as it is; a changed sidecar makes it anew
    before = cpath.read_bytes()
    lookbar.collect(tmp_path, "V6", BODY)
    assert cpath.read_bytes() == before
    sp = lookbar.sidecar_path(cpath.parents[3] / "look_partial", row.name, "walk/v0.4", 3046)
    s2io.write_json(sp, dict(s2io.read_json(sp), terms=dict(TERMS, bob=False), look_ok=False))
    lookbar.collect(tmp_path, "V6", BODY)
    assert s2io.read_json(cpath)["counts_look"]["succeeded"] == 16
    assert lookbar.main(["collect", "--set", "V6", "--body", BODY, "--out", str(tmp_path)]) == 0


def test_an_unread_episode_survived_and_succeeded_in_no_reading(tmp_path):
    row = r0l()  # (its description stands in for a policy row's: the blocks carry the policy check)
    special = {3050: (bar_ep(3050, error=UNREADABLE), block(3050, factor=0.8, policy=POLICY)),
               3051: (bar_ep(3051, error=TRACE + "RuntimeError: worker lost"), None)}
    _, cpath = write_run(tmp_path, "V6", row, "turn/q90_left", "bar_turn", twenty(special, key="turn/q90_left", policy=POLICY))
    lookbar.collect(tmp_path, "V6", BODY)  # (a lost worker's episode has no sidecar and needs none)
    doc = s2io.read_json(cpath)
    by = {e["seed"]: e for e in doc["episodes"]}
    assert (by[3050]["outcome"], by[3050]["reason"]) == ("unread", "clock_behind") and by[3051]["outcome"] == "error"
    assert doc["outcomes"] == dict(read=18, fell=0, unread=1, error=1) and doc["unread_reasons"] == dict(clock_behind=1)
    for key in ("counts", "counts_but_A1", "counts_look", "counts_but_A1_look"):
        c = doc[key]
        # the unread episode survived and did not succeed; the error is counted as the bar counts one
        assert (c["episodes"], c["survived"], c["succeeded"], c["falls"], c["errors"]) == (20, 19, 18, 0, 1), key
        assert c["judged"] is False and c["pass_"] is None
    # without the error the record is judged, and the unread episode is in it as a survivor that did not succeed
    out2 = tmp_path / "two"
    _, cpath2 = write_run(out2, "V6", row, "turn/q90_left", "bar_turn",
                          twenty({3050: special[3050]}, key="turn/q90_left", policy=POLICY))
    lookbar.collect(out2, "V6", BODY)
    doc2 = s2io.read_json(cpath2)
    for key in ("counts", "counts_but_A1", "counts_look"):
        c = doc2[key]
        assert (c["survived"], c["succeeded"], c["falls"], c["errors"], c["judged"], c["pass_"]) == (20, 19, 0, 0, True, True)
    pol = doc2["policy"]
    assert pol["versions_trained"] == POLICY["versions_trained"] and pol["versions_judged"] == POLICY["versions_judged"]
    assert (pol["cross_version"], pol["fingerprint_max_abs_diff"], pol["fingerprint_tol"]) == (False, 2e-7, 1e-5)
    assert pol["checkpoint"] == dict(steps=500000) and pol["episodes_checked"] == 20 and pol["same_checkpoint"] is True


def test_collect_holds_an_episode_during_which_mujoco_warned_as_a_fall_at_its_start(tmp_path):
    """Ruling 9 in the companion record: the bar record and the sidecar keep the time simulated (so the sidecar is
    that episode's), the companion episode is a fall at 0 s that survived nothing, and the counts hold one fall."""
    row = r0l()
    path, cpath = write_run(tmp_path, "V6", row, "walk/v0.4", "bar_walk", twenty({3045: warned_pair(3045, policy=None)}))
    lookbar.collect(tmp_path, "V6", BODY)
    doc = s2io.read_json(cpath)
    by = {e["seed"]: e for e in doc["episodes"]}
    assert (by[3045]["outcome"], by[3045]["warned"], by[3045]["fall_time_s"], by[3045]["survival_s"],
            by[3045]["simulated_s"], by[3045]["fall_cause"]) == ("fell", True, 0.0, 0.0, 20.0, WARNED_CAUSE)
    assert all((e["warned"], e["fall_time_s"], e["survival_s"]) == (False, None, 20.0) for s, e in by.items() if s != 3045)
    assert doc["outcomes"] == dict(read=19, fell=1, unread=0, error=0)
    for key in ("counts", "counts_but_A1", "counts_look", "counts_but_A1_look"):
        assert (doc[key]["survived"], doc[key]["falls"], doc[key]["errors"]) == (19, 1, 0), key
    sp = lookbar.sidecar_path(bar.block_dir(tmp_path, "V6", BODY, bar.HEADLINE_BLOCK) / "look_partial", row.name,
                              "walk/v0.4", 3045)
    assert s2io.read_json(sp)["survival_s"] == 20.0 == s2io.read_json(path)["episodes"][3]["survival_s"]
    assert doc["rules"]["outcome"] == lookbar.RULES["outcome"]


def test_a_missing_or_foreign_sidecar_is_refused(tmp_path):
    row = r0l()

    def refused(name, mutate, match):
        out = tmp_path / name
        _, cpath = write_run(out, "V6", row, "walk/v0.4", "bar_walk", twenty())
        other = write_run(out, "V6", row, "walk/nominal", "bar_walk", twenty(key="walk/nominal"))[1]
        sp = lookbar.sidecar_path(bar.block_dir(out, "V6", BODY, bar.HEADLINE_BLOCK) / "look_partial", row.name,
                                  "walk/v0.4", 3047)
        mutate(sp)
        with pytest.raises(lookbar.LookBarError, match=match):
            lookbar.collect(out, "V6", BODY)
        # the faulty record gets no companion; every other record's is written, and the summary names the refusal
        assert not cpath.exists() and other.exists()
        summary = s2io.read_json(other.parents[2] / "summary.json")
        assert len(summary["records"]) == 1 and len(summary["refused"]) == 1 and "seed 3047" in summary["refused"][0]
        # ... and the record itself, which has no companion
        cells = summary["refused_records"]
        assert [(c["row"], c["condition"], c["status"], c["companion"], c["record"], c["moved_to"]) for c in cells] == [
            ("R0L@10ms", "walk/v0.4", "done", "none", None, None)]
        assert cells[0]["reason"] == summary["refused"][0] and cells[0]["bar_record"].endswith("eval/R0L@10ms/walk/v0.4.json")
        # the lead's ruling 12 of 17 Nov 2026 (the check of the rows package, low point 3): a refusal's text in the
        # summary names no absolute path (as built it held the output root's): its paths are relative to the root
        for text in summary["refused"] + [c["reason"] for c in cells]:
            assert str(out) not in text and str(tmp_path) not in text and not lookbar._ABS_PATH.search(text), text
        assert summary["refused"][0].startswith(cells[0]["bar_record"] + ": seed 3047: ")
        assert lookbar.main(["collect", "--set", "V6", "--body", BODY, "--out", str(out)]) == 1

    def change(**kw):
        return lambda sp: s2io.write_json(sp, dict(s2io.read_json(sp), **kw))

    refused("missing", lambda sp: sp.unlink(), "no sidecar 3047.json")
    refused("steps", change(steps=9999), "steps differ")
    refused("fell", change(fell=True), "fell differ")
    refused("survival", change(survival_s=19.998), "survival_s differ")
    refused("row", change(row_sha256="0" * 64), "another row's description")
    refused("seed", change(seed=3048), "schema, condition or seed")
    refused("torn", lambda sp: sp.write_text("{"), "no sidecar")
    # an unreadable episode needs its sidecar too (its clock decides between unread and error); a lost worker's none
    out = tmp_path / "unreadable"
    write_run(out, "V6", row, "walk/v0.4", "bar_walk", twenty({3050: (bar_ep(3050, error=UNREADABLE), None)}))
    with pytest.raises(lookbar.LookBarError, match="seed 3050: no sidecar"):
        lookbar.collect(out, "V6", BODY)


def test_a_refusal_leaves_no_companion_of_another_bar_record_and_the_summary_names_the_record(tmp_path):
    """After a refusal the path of the record's companion holds no companion of *another* bar record: one made
    before the bar record changed is moved aside.  One made of the bar record as it is stays (a tree copied
    without its sidecars).  Either way the summary holds the refused record, with what became of its companion."""
    row = r0l()
    bdir = bar.block_dir(tmp_path, "V6", BODY, bar.HEADLINE_BLOCK)
    path, cpath = write_run(tmp_path, "V6", row, "walk/v0.4", "bar_walk", twenty())
    _, other = write_run(tmp_path, "V6", row, "walk/nominal", "bar_walk", twenty(key="walk/nominal"))
    lookbar.collect(tmp_path, "V6", BODY)
    old, old_file = s2io.read_json(cpath), s2io.sha256_file(cpath)
    assert old["counts_look"]["succeeded"] == 20 and old["bar_record"]["sha256"] == s2io.sha256_file(path)
    # the bar record changes (an episode of it now fell), and that episode's sidecar is not its episode's any more
    rec = s2io.read_json(path)
    rec["episodes"][3] = bar_ep(rec["episodes"][3]["seed"], fell=True, survival=4.0, steps=2000)
    rec.update(counts=bar.counts(rec["episodes"], rec["seeds"], "success"),
               counts_but_A1=bar.counts(rec["episodes"], rec["seeds"], "but_A1"))
    s2io.write_json(path, plain(rec))
    with pytest.raises(lookbar.LookBarError, match="seed 3045: the sidecar is not this episode's"):
        lookbar.collect(tmp_path, "V6", BODY)
    moved = lookbar.stale_path(cpath, old_file)  # (named by the first 12 hex of the file's own sha256)
    assert moved == cpath.with_name("v0.4.json.superseded") / f"{old_file[:12]}.json"
    assert not cpath.exists() and s2io.read_json(moved) == old and other.exists()
    # beside it under the bar's own name for what was moved aside, so a reader that skips those (as the bar's
    # summary and collect do) finds no companion at all for this record: no file under look/ but the other one's
    look = bdir / "look"
    current = [q for q in sorted(look.glob("**/*.json")) if q.name != "summary.json"
               and not any(x.endswith(".superseded") for x in q.relative_to(look).parts)]
    assert current == [other] and moved.parent.name.endswith(".superseded")
    summary = s2io.read_json(bdir / "look" / "summary.json")
    assert [c["condition"] for c in summary["records"]] == ["walk/nominal"] and len(summary["refused"]) == 1
    cell, = summary["refused_records"]
    assert (cell["row"], cell["condition"], cell["status"], cell["companion"], cell["record"]) == (
        "R0L@10ms", "walk/v0.4", "done", "moved_aside", None)
    assert cell["moved_to"] == "V6/75-M-H-10/seeds3042/look/R0L@10ms/walk/v0.4.json.superseded/" + moved.name
    assert cell["bar_record"] == "V6/75-M-H-10/seeds3042/eval/R0L@10ms/walk/v0.4.json"
    assert cell["bar_record_sha256"] == s2io.sha256_file(path) != old["bar_record"]["sha256"]
    assert "seed 3045" in cell["reason"] and summary["sha256"] == s2io.sha256_json(
        {k: v for k, v in summary.items() if k != "sha256"})
    assert "R0L@10ms walk/v0.4: refused; the companion of another bar record was moved aside to" in lookbar.summary_text(summary)
    # again: the same refusal, and nothing is left to move
    with pytest.raises(lookbar.LookBarError, match="seed 3045"):
        lookbar.collect(tmp_path, "V6", BODY)
    cell, = s2io.read_json(bdir / "look" / "summary.json")["refused_records"]
    assert (cell["companion"], cell["moved_to"]) == ("none", None) and sorted(p.name for p in moved.parent.iterdir()) == [moved.name]
    # with that episode's sidecar its own again the companion is made anew; what was moved aside stays there
    sp = lookbar.sidecar_path(bdir / "look_partial", row.name, "walk/v0.4", 3045)
    s2io.write_json(sp, dict(s2io.read_json(sp), fell=True, survival_s=4.0, steps=2000, read=False, terms=None,
                             look_ok=None, reason="fell"))
    summary = lookbar.collect(tmp_path, "V6", BODY)
    new = s2io.read_json(cpath)
    assert new["bar_record"]["sha256"] == s2io.sha256_file(path) and new["counts_look"]["succeeded"] == 19
    assert summary["refused_records"] == [] and len(summary["records"]) == 2 and s2io.read_json(moved) == old
    # the sidecars are gone and the bar record is as it was (a copied tree of finished records): the companion is
    # this bar record's and stays as it is; the summary says that it was kept and not made again
    before = cpath.read_bytes()
    shutil.rmtree(bdir / "look_partial" / row.name / "walk" / "v0.4")
    with pytest.raises(lookbar.LookBarError, match="no sidecar"):
        lookbar.collect(tmp_path, "V6", BODY)
    assert cpath.read_bytes() == before and not lookbar.stale_path(cpath, s2io.sha256_file(cpath)).exists()
    summary = s2io.read_json(bdir / "look" / "summary.json")
    cell, = summary["refused_records"]
    assert (cell["condition"], cell["companion"], cell["moved_to"]) == ("walk/v0.4", "kept", None)
    assert cell["record"] == "V6/75-M-H-10/seeds3042/look/R0L@10ms/walk/v0.4.json"
    assert [c["condition"] for c in summary["records"]] == ["walk/nominal"]
    assert "R0L@10ms walk/v0.4: refused; the companion made earlier of this bar record is kept" in lookbar.summary_text(summary)
    # a file at the companion's path that cannot be read is no companion of this bar record either
    cpath.write_text("{")
    with pytest.raises(lookbar.LookBarError, match="no sidecar"):
        lookbar.collect(tmp_path, "V6", BODY)
    cell, = s2io.read_json(bdir / "look" / "summary.json")["refused_records"]
    assert cell["companion"] == "moved_aside" and not cpath.exists()
    assert (tmp_path / cell["moved_to"]).read_text() == "{"


def test_on_a_set_that_is_no_look_set_reading_1_is_null_and_the_counts_are_the_bars(tmp_path):
    row = r0l()
    special = {3043: (bar_ep(3043), block(3043, terms=dict(bob=False, head=False))),
               3044: (bar_ep(3044, ok=False, a1=True, failed=("drive_sustained",)), block(3044)),
               3045: (bar_ep(3045, error=UNREADABLE), block(3045, factor=0.8, policy=POLICY))}
    path, cpath = write_run(tmp_path, "F25F9cl", row, "walk/v0.4", "bar_walk", twenty(special, terms=dict(bob=False)))
    stop = {3046: (bar_ep(3046), block(3046, key="stop/two_step", read=False, on_command=dict(term=False, speed_m_s=0.1)))}
    spath, scpath = write_run(tmp_path, "F25F9cl", row, "stop/two_step", "bar_stop",
                              twenty(stop, key="stop/two_step", read=False, on_command=dict(term=True, speed_m_s=0.41)))
    summary = lookbar.collect(tmp_path, "F25F9cl", BODY)
    doc, rec = s2io.read_json(cpath), s2io.read_json(path)
    assert doc["look_set"] is False and doc["counts_look"] is None and doc["counts_but_A1_look"] is None
    assert doc["counts_look_reason"] == lookbar.READING_1["not_counted"] == doc["reading_1"]
    # the companion's counts are the bar record's own, the bar's count of an unreadable episode included
    assert doc["counts"] == rec["counts"] and doc["counts_but_A1"] == rec["counts_but_A1"] and doc["counts_equal_bar"] is True
    assert (doc["counts"]["succeeded"], doc["counts"]["errors"], doc["counts"]["survived"]) == (18, 1, 19)
    assert all(e["succeeded_look"] is None and e["succeeded_but_A1_look"] is None for e in doc["episodes"])
    by = {e["seed"]: e for e in doc["episodes"]}
    assert by[3045]["outcome"] == "unread" and by[3043]["succeeded"] is True  # the look decides nothing here
    assert doc["look_failures_by_term"] == dict(bob=18, head=1) and doc["failures_by_term"] == dict(drive_sustained=1)
    # the stop item's term is recorded and decides nothing on this set
    sdoc = s2io.read_json(scpath)
    assert sdoc["counts"] == s2io.read_json(spath)["counts"] and sdoc["counts"]["succeeded"] == 20
    assert {e["seed"]: e for e in sdoc["episodes"]}[3046]["on_command_at_stop"]["term"] is False
    assert summary["look_set"] is False and [c["counts_look"] for c in summary["records"]] == [None, None]
    # on a look set the same stop record counts the term in every reading
    out2 = tmp_path / "look"
    _, c2 = write_run(out2, "V6noF5", row, "stop/two_step", "bar_stop",
                      twenty(stop, key="stop/two_step", read=False, on_command=dict(term=True, speed_m_s=0.41)))
    lookbar.collect(out2, "V6noF5", BODY)
    d2 = s2io.read_json(c2)
    assert [d2[k]["succeeded"] for k in ("counts", "counts_but_A1", "counts_look", "counts_but_A1_look")] == [19] * 4
    assert d2["failures_by_term"] == dict(on_command_at_stop=1) and d2["reading_1"] == lookbar.READING_1["other"]
    # a bar record whose counts are not bar.counts of its episodes is refused
    bad = s2io.read_json(path)
    bad["counts"]["succeeded"] = 20
    s2io.write_json(path, bad)
    with pytest.raises(lookbar.LookBarError, match="not bar.counts"):
        lookbar.collect(tmp_path, "F25F9cl", BODY)


def test_a_turn_frozen_with_a1_over_a_record_not_run_and_records_of_other_rows(tmp_path):
    row = r0l()
    special = {3043: (bar_ep(3043, ok=False, a1=True, failed=("drive_sustained",)), block(3043, key="turn/q90_left"))}
    _, cpath = write_run(tmp_path, "V6", row, "turn/q90_left", "bar_turn", twenty(special, key="turn/q90_left"),
                         a1_over=True)
    _, npath = write_run(tmp_path, "V6", row, "turn/q90_right", "bar_turn", [], status="not run")
    plain_row = bar.rows(BODY, ["R0@10ms"])["R0@10ms"]  # no LookRow: its record gets no companion
    bdir = bar.block_dir(tmp_path, "V6", BODY, bar.HEADLINE_BLOCK)
    s2io.write_json(bdir / "eval" / "R0@10ms" / "walk" / "v0.4.json",
                    dict(schema=bar.SCHEMA, key_name="walk/v0.4", row="R0@10ms", status="done",
                         controller=plain(s2io.portable(plain_row.describe())), episodes=[]))
    s2io.write_json(bdir / "eval" / "summary.json", dict(schema=bar.SCHEMA))
    summary = lookbar.collect(tmp_path, "V6", BODY)
    assert [(c["row"], c["condition"], c["status"]) for c in summary["records"]] == [
        ("R0L@10ms", "turn/q90_left", "done"), ("R0L@10ms", "turn/q90_right", "not run")]
    doc = s2io.read_json(cpath)
    # readings 1 and 2 give a turn frozen with a1_over no verdict; reading 3 with the four bands kept is its headline
    for key in ("counts", "counts_look"):
        assert doc[key]["pass_"] is None and doc[key]["not_read"] == "a1_over" and doc[key]["reason"] == "not read: a1_over"
        assert doc[key]["pass_if_read"] is True and doc[key]["succeeded"] == 19
    assert doc["counts_but_A1_look"]["pass_"] is True and doc["counts_but_A1_look"]["succeeded"] == 20 and doc["a1_over"] is True
    nr = s2io.read_json(npath)
    assert nr["bar_record"]["status"] == "not run" and nr["episodes"] == [] and nr["counts_look"] is None
    assert nr["counts"] is None and nr["counts_look_reason"] == "no reference"
    assert not (bdir / "look" / "R0@10ms").exists()
    assert lookbar.collect(tmp_path / "nothing", "V6", BODY)["records"] == []
    assert not (tmp_path / "nothing").exists()  # nothing to read: nothing is written, not even a lock


def test_the_command_line_assigns_mujoco_gl_itself_and_one_thread(tmp_path):
    """``python -m designlab.stage2.lookbar``: the module's own copy runs ``main``; whatever backend and thread
    counts the caller's environment names, the process has no OpenGL and one BLAS thread."""
    row = r0l()
    write_run(tmp_path, "V6", row, "walk/v0.4", "bar_walk", twenty())
    code = ("import os, runpy, sys\n"
            f"sys.argv = ['lookbar', 'collect', '--set', 'V6', '--body', {BODY!r}, '--out', {str(tmp_path)!r}]\n"
            "try:\n    runpy.run_module('designlab.stage2.lookbar', run_name='__main__')\n"
            "except SystemExit as e:\n    code = e.code\n"
            "print(code, os.environ['MUJOCO_GL'], os.environ['OMP_NUM_THREADS'], 'PYOPENGL_PLATFORM' in os.environ)\n")
    env = {k: v for k, v in os.environ.items() if k not in s2io.THREAD_VARS}
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                         env=dict(env, PYTHONPATH=".", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl"))
    assert out.returncode == 0, out.stderr[-2000:]
    lines = out.stdout.strip().splitlines()
    assert lines[-1] == "0 disable 1 False" and "with the look 20/20" in out.stdout
    assert (bar.block_dir(tmp_path, "V6", BODY, bar.HEADLINE_BLOCK) / "look" / "summary.json").is_file()


# ------------------------------------------------------------------------------------------------
# Closed loop: the wrapper changes nothing an episode executes
# ------------------------------------------------------------------------------------------------
def same_record(a: dict, b: dict) -> bool:
    return plain({k: v for k, v in a.items() if k != "wall_s"}) == plain({k: v for k, v in b.items() if k != "wall_s"})


@needs_run
@pytest.mark.parametrize("k", [0, 3])
def test_r0l_is_r0_digest_for_digest_and_record_for_record_on_the_s1_view(k):
    """``Delayed(LookRow(R0), k)`` against ``Delayed(R0, k)``: the digest after every physics step and
    ``panels.judge``'s episode record (no record is written; Stage 1's own run directory)."""
    opened = lookbar.open_set(BODY, "S1", ["walk/brisk"], stage1_run=RUN)
    ctx, cond = opened["ctx"], dataclasses.replace(opened["conds"]["walk/brisk"], seconds=2.0)
    name = f"@{bar.STEP_MS * k:d}ms"
    look = lookbar.rows(BODY, ["R0L" + name, "R0" + name])
    ref = bar.rows(BODY, ["R0" + name])["R0" + name]
    d0 = g8a.row_episode(ref, ctx, cond, 3042)
    assert len(d0["digests"]) == 1000 and d0["fell_step"] is None
    e0 = bar.run_episode(ref, ctx, cond, 3042)
    for row in look.values():
        d1 = g8a.row_episode(row, ctx, cond, 3042)
        assert d1["digests"] == d0["digests"] and d1["fell_step"] is None
        assert "recorder" not in vars(ctx.built(cond))
        e1 = bar.run_episode(row, ctx, cond, 3042)
        assert same_record(e1, e0) and "success" in e1 and "succeeded_but_A1" not in e1  # (panels.judge's record)
        last = lookbar.look_row_of(row).last
        assert (last["read"], last["reason"], last["set"]) == (False, "a stock condition has no look reading", "S1")
        assert (last["steps"], last["fell"], last["survival_s"]) == (1000, False, e1["survival_s"])
        assert last["clock"]["mean_factor"] == pytest.approx(1.0, abs=1e-9) and last["fault"] is None
        assert last["clock"]["t_ref_reached"] == pytest.approx(2.1, abs=1e-9) and len(last["state_digest"]) == 64
    assert g8a.row_episode(ref, ctx, cond, 3043)["digests"] != d0["digests"]  # (another seed is another episode)


@needs_run
def test_on_a_bar_walk_item_the_wrapper_changes_neither_digests_nor_record_nor_trace(t0, t0_set, tmp_path):
    """A whole walk item of the fixture set at 6 ms: the look reading is read (50 Hz poses are lent to a walk's
    recorder, which the bar's own has none of), and the digests, the bar's episode record and the trace's bytes are
    those of the unwrapped row."""
    ctx, cond = t0_set["ctx"], t0_set["conds"]["walk/nominal"]
    look, ref = r0l(3), bar.rows(BODY, ["R0@6ms"])["R0@6ms"]
    d0, d1 = g8a.row_episode(ref, ctx, cond, 3042), g8a.row_episode(look, ctx, cond, 3042)
    assert d0["digests"] == d1["digests"] and len(d0["digests"]) == round(ctx.built(cond).seconds / 0.002)
    last = lookbar.look_row_of(look).last  # (read beside a gate's lent digest recorder, whose wrapper is looked through)
    assert last["read"] is True and last["fault"] is None and last["steps"] == len(d0["digests"])
    assert last["clock"]["mean_factor"] == pytest.approx(1.0, abs=1e-9)
    e0 = bar.run_episode(ref, ctx, cond, 3042, trace_root=tmp_path / "a")
    e1 = bar.run_episode(look, ctx, cond, 3042, trace_root=tmp_path / "b")
    assert e0["trace"]["sha256"] == e1["trace"]["sha256"] and e0["trace"]["steps"] == len(d0["digests"])
    trace = dict(e1["trace"], path=e0["trace"]["path"])  # (the path holds the row's name)
    assert same_record(dict(e1, trace=trace), e0) and e1["succeeded"] is True
    assert "recorder" not in vars(ctx.built(cond)) and type(ctx.built(cond).recorder()).__name__ == "Recorder"
    assert ctx.built(cond).recorder().legs_every == 0  # the class's own again: a walk's recorder holds no poses


@needs_run
def test_the_look_block_of_a_walk_item(t0_set, tmp_path):
    ctx, cond = t0_set["ctx"], t0_set["conds"]["walk/nominal"]
    row = r0l(5, sidecar=tmp_path / "side")
    look = lookbar.look_row_of(row)
    seen = {}
    real = bar._judge

    def judge(b, out, rec, seed, lg=None):  # the sidecar is there before the bar's judge sees the episode
        seen["sidecar"] = s2io.read_json(lookbar.sidecar_path(tmp_path / "side", "R0L@10ms", cond.key, seed))
        return real(b, out, rec, seed, lg)

    bar._judge = judge
    try:
        e = bar.run_episode(row, ctx, cond, 6001)
    finally:
        bar._judge = real
    blk = look.last
    assert seen["sidecar"] == plain(blk) and blk["schema"] == "s2-look-ep-1"
    assert (blk["row"], blk["look_row"], blk["condition"], blk["seed"], blk["set"], blk["judge"]) == (
        "R0L@10ms", "R0L", "walk/nominal", 6001, "T0", "bar_walk")
    assert (blk["fell"], blk["survival_s"], blk["planned_s"]) == (False, e["survival_s"], e["planned_s"])
    assert blk["steps"] == round(e["survival_s"] / 0.002)
    assert blk["row_sha256"] == look.row_sha256() and blk["policy_check"] is None and blk["fault"] is None
    # rule U2's window is the bar's judged last 10 s, and the reference is played over its reference times
    b = ctx.built(cond)
    w = e["measures"]["window"]
    assert blk["window"] == pytest.approx([w["t_start_s"], w["t_end_s"]], abs=1e-9) and blk["read"] is True
    assert blk["window_ref"] == pytest.approx(list(lookread.window_ref(b)), abs=1e-12)
    assert blk["reference"]["window_ref"] == blk["window_ref"] and blk["reference"]["steps"] >= lookread.MIN_STEPS
    assert blk["own"]["steps"] == blk["reference"]["steps"] and blk["too_few_steps"] is False
    # Stage 1's stock walk has a level hip and flat feet: no toe-off to read, no bob to be within 30 % of
    assert blk["terms"] == dict(bob=False, head=False, tail=True, toe_off=True) and blk["not_read"] == ["toe_off"]
    assert blk["look_ok"] is False and blk["compare"]["reference"]["toe_off_side"] == "under 0.02 rad"
    assert abs(blk["compare"]["tail_phase_diff_cycles"]) < 0.02 and blk["compare"]["bands"] == dict(lookread.BANDS)
    assert sorted(blk["compare"]) == sorted(lookbar.COMPARE_KEYS) and blk["compare"]["tail_amp_rad"] > 0.1
    assert blk["compare"]["tail_amp_side"] == "at or over 0.02 rad" and blk["compare"]["tail_amp_min_rad"] == 0.02
    assert blk["air_runs"]["in_window"] == blk["own"]["lift_offs"] and blk["air_runs"]["shorter_than_air_gap_max"] == 0
    assert blk["air_runs"]["air_gap_max_s"] == b.air_gap_max_s and blk["on_command_at_stop"] is None
    assert blk["clock"]["mean_factor"] == pytest.approx(1.0, abs=1e-9)
    assert blk["clock"]["t_ref_reached"] == pytest.approx(1.05 * blk["survival_s"], abs=1e-9)
    assert blk["clock"]["t_ref_planned"] == pytest.approx(1.05 * blk["planned_s"], abs=1e-12)
    # the reference is played once per built condition
    assert look._refs[cond][0] is b and look._refs[cond][1]["reading"] == blk["reference"]
    kept = look._refs[cond][1]
    bar.run_episode(row, ctx, cond, 6002)
    assert look._refs[cond][1] is kept
    # an attribution's lifted run writes no sidecar (it is not the record's episode)
    row.inner.episode(ctx, cond, 6003, ("slip",))
    assert not lookbar.sidecar_path(tmp_path / "side", "R0L@10ms", cond.key, 6003).exists() and look.last["relaxed"] == ["slip"]


@needs_run
def test_the_recorder_is_the_classs_method_again_after_an_episode_that_raised(t0_set):
    ctx, cond = t0_set["ctx"], t0_set["conds"]["walk/v0.4"]
    b = ctx.built(cond)

    class Boom(RuntimeError):
        pass

    class Raising(panels.Row):
        name, hard_only, lent = "R0", True, None

        def episode(self, ctx_, c, seed, relax=()):
            rec = ctx_.built(c).recorder()  # looked up on the instance: the lent recorder, with 50 Hz poses
            Raising.lent = ("recorder" in vars(ctx_.built(c)), type(rec).__name__, rec.legs_every)
            raise Boom

    row = lookbar.LookRow(Raising())
    with pytest.raises(Boom):
        row.episode(ctx, cond, 3042)
    assert Raising.lent == (True, "Recorder", tune.LEGS_EVERY) and row.last is None
    assert "recorder" not in vars(b) and b.recorder.__func__ is bar.BarBuilt.recorder and b.recorder().legs_every == 0
    # a recorder lent from outside for the episode (a gate's digests) is kept, made to hold poses, and given back
    rows: list = []
    with g8a.lent_recorder(b, rows):
        outer = vars(b)["recorder"]
        with pytest.raises(Boom):
            row.episode(ctx, cond, 3042)
        assert Raising.lent == (True, "_Digests", tune.LEGS_EVERY) and vars(b)["recorder"] is outer
    assert "recorder" not in vars(b)
    # a nudge or a stop item is lent nothing
    nudge = t0_set["conds"]["nudge/forward"]
    with pytest.raises(Boom):
        row.episode(ctx, nudge, 3042)
    assert Raising.lent == (False, "Recorder", 0)


@needs_run
def test_a_fault_in_the_reading_is_recorded_and_never_raised(t0_set, monkeypatch):
    ctx, cond = t0_set["ctx"], t0_set["conds"]["walk/nominal"]
    row = r0l(0)
    look = lookbar.look_row_of(row)

    def not_played(*a, **k):
        raise lookread.LookError("not played: a periodic segment must be a straight walk")

    kept = {}
    real_block, real_playback, real_compare = lookbar.LookRow._block, lookread.playback, lookread.compare

    def block_of(self, *args):  # (what the row read the block from, for a second reading without a second episode)
        kept["args"] = args
        return real_block(self, *args)

    monkeypatch.setattr(lookbar.LookRow, "_block", block_of)
    monkeypatch.setattr(lookread, "playback", not_played)
    e = bar.run_episode(row, ctx, cond, 6001)  # the bar's episode is the unwrapped row's
    blk = look.last
    assert e["succeeded"] is True and blk["read"] is False and blk["look_ok"] is None and blk["terms"] is None
    assert "LookError" in blk["fault"] and "not played" in blk["fault"] and blk["reason"] is None
    assert blk["clock"]["mean_factor"] is not None and blk["steps"] > 0
    # the fault is recorded as "<what>: <type>: <message>", as every other fault of a block is: no traceback, and so
    # no path of this machine, goes into a sidecar or a companion record
    assert blk["fault"] == ("look reading: LookError: the reference of walk/nominal: LookError: not played: a periodic "
                            "segment must be a straight walk")

    def no_terms(*a, **k):
        raise ValueError("terms that cannot be read")

    monkeypatch.setattr(lookread, "playback", real_playback)
    monkeypatch.setattr(lookread, "compare", no_terms)
    blk2 = lookbar.LookRow(look.inner)._block(*kept["args"])  # a fault of any type inside the reading itself
    assert blk2["fault"] == "look reading: ValueError: terms that cannot be read"
    assert blk2["read"] is False and blk2["look_ok"] is None and blk2["terms"] is None
    assert blk2["reference"]["steps"] >= lookread.MIN_STEPS and blk2["own"]["steps"] == blk2["reference"]["steps"]
    for fault in (blk["fault"], blk2["fault"]):
        assert "Traceback" not in fault and "\n" not in fault and str(ROOT) not in fault and ".py" not in fault
    # a fault whose message names files (the lead's ruling 8 of 17 Nov 2026: no path of this machine): a file under
    # the study is named relative to it, one elsewhere (here the fixture's set under pytest's temporary directory)
    # by its last three parts; the same through the reference (``_reference``) and through the reading itself
    here = ROOT / "stage2" / "sets" / SET / BODY / "refs" / "nominal.npz"
    there = t0_set["run_dir"] / "refs" / "nominal.npz"
    assert str(t0_set["run_dir"]).startswith("/") and not str(there).startswith(str(ROOT))

    def lost(*a, **k):
        raise FileNotFoundError(2, "No such file or directory", str(here))

    def elsewhere(*a, **k):
        raise OSError(f"cannot read {there} (nor {here})")

    monkeypatch.setattr(lookread, "compare", real_compare)
    monkeypatch.setattr(lookread, "playback", lost)
    blk3 = lookbar.LookRow(look.inner)._block(*kept["args"])  # (a new row: the reference is read again)
    assert blk3["fault"] == ("look reading: LookError: the reference of walk/nominal: FileNotFoundError: [Errno 2] No "
                             f"such file or directory: 'stage2/sets/{SET}/{BODY}/refs/nominal.npz'")
    monkeypatch.setattr(lookread, "playback", real_playback)
    monkeypatch.setattr(lookread, "compare", elsewhere)
    blk4 = lookbar.LookRow(look.inner)._block(*kept["args"])
    assert blk4["fault"] == (f"look reading: OSError: cannot read .../{BODY}/refs/nominal.npz (nor stage2/sets/{SET}/"
                             f"{BODY}/refs/nominal.npz)")
    for fault in (blk3["fault"], blk4["fault"]):
        assert str(ROOT) not in fault and str(t0_set["run_dir"]) not in fault and "/tmp" not in fault
    # the two other fault sites cut the same way (the check of the rows package, low point 1: as built only the look
    # reading's site was tested end to end): the clock, state digest and policy check, and a stop item's term
    monkeypatch.setattr(lookread, "compare", real_compare)
    real_clock, real_stop = lookbar.clock_block, lookbar.on_command_at_stop

    def clock_lost(*a, **k):
        raise OSError(f"cannot read {there}")

    def stop_lost(*a, **k):
        raise FileNotFoundError(2, "No such file or directory", str(here))

    monkeypatch.setattr(lookbar, "clock_block", clock_lost)
    blk5 = lookbar.LookRow(look.inner)._block(*kept["args"])
    assert blk5["fault"].split("; ")[0] == f"clock: OSError: cannot read .../{BODY}/refs/nominal.npz"
    assert blk5["clock"] is None
    monkeypatch.setattr(lookbar, "clock_block", real_clock)
    monkeypatch.setattr(lookbar, "on_command_at_stop", stop_lost)
    b, out, rec, _, seed, relax = kept["args"]
    stop = t0_set["conds"]["stop/two_step"]
    assert stop.judge == lookbar.STOP_JUDGE
    blk6 = lookbar.LookRow(look.inner)._block(b, out, rec, stop, seed, relax)
    assert blk6["fault"] == (f"{lookbar.ON_COMMAND_TERM}: FileNotFoundError: [Errno 2] No such file or directory: "
                             f"'stage2/sets/{SET}/{BODY}/refs/nominal.npz'")
    monkeypatch.setattr(lookbar, "on_command_at_stop", real_stop)
    for fault in (blk5["fault"], blk6["fault"]):
        assert str(ROOT) not in fault and str(t0_set["run_dir"]) not in fault and not lookbar._ABS_PATH.search(fault)
    # a walk or turn item that cannot be played is a fault of the definition: an error on a look set, never "no look"
    on = lookbar.companion_episode(cond.key, cond.judge, 6001, e, blk, True)
    off = lookbar.companion_episode(cond.key, cond.judge, 6001, e, blk, False)
    assert (on["outcome"], on["succeeded_look"]) == ("error", False) and "not played" in on["error"]
    assert (off["outcome"], off["succeeded"], off["succeeded_look"]) == ("read", True, None) and "not played" in off["look_fault"]


# ------------------------------------------------------------------------------------------------
# Episodes in this process (the monitor's entry)
# ------------------------------------------------------------------------------------------------
PAIRS = [("walk/nominal", 6001), ("walk/v0.4", 6004), ("turn/q90_left", 6007), ("stop/two_step", 6010),
         ("nudge/forward", 6013), ("turn/q90_right", 6016)]


@needs_run
def test_episodes_on_the_fixture_set(t0, tmp_path, monkeypatch):
    rows = lookbar.rows(BODY, ["R0L@10ms"])
    eps = lookbar.episodes(BODY, SET, rows, PAIRS, traces=tmp_path / "traces", root=t0)
    assert [(e["condition"], e["seed"], e["row"], e["set"]) for e in eps] == [(k, s, "R0L@10ms", SET) for k, s in PAIRS]
    by = {e["condition"]: e for e in eps}
    assert [e["outcome"] for e in eps] == ["read", "read", "fell", "fell", "fell", "not_run"]
    # a condition whose reference the set does not hold is not run
    nr = by["turn/q90_right"]
    assert nr["reason"] == "a fixture's reason" and nr["survival_s"] == 0.0 and nr["planned_s"] is None and nr["clock"] is None
    # every episode that ran holds its clock times and its planned length; this set is no look set
    ran = eps[:-1]
    for e in ran:
        assert e["clock"]["t_ref_reached"] == pytest.approx(1.05 * e["survival_s"], abs=1e-6)
        assert e["clock"]["t_ref_planned"] == pytest.approx(1.05 * e["planned_s"], abs=1e-9)
        assert e["succeeded_look"] is None and e["succeeded_but_A1_look"] is None and e["error"] is None
        assert e["start_seed"] == panels.episode_seeds(next(c for c in bar.conditions(setio.load(setio.set_dir(SET, BODY, t0)), every=True)
                                                            if c.key == e["condition"]), e["seed"])["start"]
        assert len(e["state_digest"]) == 64 and e["policy_check"] is None and e["wall_s"] > 0
    assert len({e["start_seed"] for e in ran}) == 5
    walk = by["walk/nominal"]
    assert walk["succeeded"] is True and walk["look"]["terms"]["tail"] is True and walk["look_failed_terms"] == ["bob", "head"]
    assert walk["bar"]["trace"]["path"] == "R0L@10ms/walk/nominal/6001.npz"
    assert (tmp_path / "traces" / "R0L@10ms" / "walk" / "nominal" / "6001.npz").is_file()
    # the fixture's "moves" join nothing: the open loop falls in them; a stop item carries its term all the same
    stop = by["stop/two_step"]
    assert stop["on_command_at_stop"]["command_s"] is not None and stop["on_command_at_stop"]["speed_m_s"] > 0.3
    assert stop["look"] is None and stop["look_reason"].startswith("nudge and stop items")
    assert by["turn/q90_left"]["look_reason"] == "fell before its window" and by["turn/q90_left"]["failed_terms"] == ["survived"]
    # the same episodes on a look set (the fixture named one for this test): the look terms decide reading 1
    monkeypatch.setattr(lookbar, "LOOK_SETS", ("V6", SET))
    on = lookbar.episodes(BODY, SET, rows, PAIRS[:2], root=t0)
    assert [e["state_digest"] for e in on] == [e["state_digest"] for e in eps[:2]]  # the same episodes
    assert on[0]["succeeded"] is True and on[0]["succeeded_look"] is False and on[0]["failed_terms"] == ["look_bob", "look_head"]
    assert on[0]["reading_1"] == lookbar.READING_1["walk_turn"] and on[1]["succeeded_look"] is False


@needs_run
def test_episodes_catches_the_bars_unreadable_apart_from_any_other_fault(t0, monkeypatch):
    rows = lookbar.rows(BODY, ["R0L@0ms"])
    real = bar.run_episode
    calls = []

    def run_episode(row, ctx, cond, seed, **kw):
        calls.append((cond.key, seed))
        e = real(row, ctx, cond, seed, **kw)
        if seed == 6001:
            raise bar.Unreadable(f"{cond.key} seed {seed}: no fall, and the trace ends early")
        if seed == 6002:
            raise RuntimeError("the disk is full")
        return e

    monkeypatch.setattr(bar, "run_episode", run_episode)
    monkeypatch.setattr(lookbar, "LOOK_SETS", ("V6", SET))
    eps = lookbar.episodes(BODY, SET, rows, [("walk/nominal", 6001), ("walk/nominal", 6002), ("walk/nominal", 6003)], root=t0)
    assert calls == [("walk/nominal", s) for s in (6001, 6002, 6003)]  # the others go on, none is run twice
    # R0L has no policy: its unreadable episode is an error, for which C8 stands
    assert [e["outcome"] for e in eps] == ["error", "error", "read"]
    assert eps[0]["error"].startswith("designlab.stage2.bar.Unreadable: walk/nominal seed 6001") and lookbar.is_unreadable(eps[0]["error"])
    assert "RuntimeError: the disk is full" in eps[1]["error"] and not lookbar.is_unreadable(eps[1]["error"])
    assert eps[0]["clock"]["t_ref_reached"] > 14 and eps[0]["succeeded"] is False and eps[2]["succeeded"] is True
    # the same unreadable episode of a row with a policy is unread (the block says which row it is)
    look = lookbar.look_row_of(rows["R0L@0ms"])
    look.inner.provenance = lambda: dict(POLICY)
    un = lookbar.episodes(BODY, SET, rows, [("walk/nominal", 6001)], root=t0)[0]
    assert (un["outcome"], un["reason"], un["fell"], un["error"]) == ("unread", "unreadable", False, None)
    assert un["policy_check"] == POLICY and un["clock"]["t_ref_planned"] == pytest.approx(1.05 * un["planned_s"])
    # what episodes refuses
    with pytest.raises(lookbar.LookBarError, match="hold a LookRow"):
        lookbar.episodes(BODY, SET, bar.rows(BODY, ["R0@0ms"]), [("walk/nominal", 6001)], root=t0)
    with pytest.raises(ValueError, match="not among the conditions"):
        lookbar.episodes(BODY, SET, rows, [("walk/sideways", 6001)], root=t0)
    with pytest.raises(tune.InputError, match="set.json is missing"):
        lookbar.episodes(BODY, "T1", rows, [("walk/nominal", 6001)], root=t0)


@needs_run
def test_the_monitor_reads_these_episodes(tmp_path):
    """``monitor2._episodes`` goes through ``lookbar.rows`` and ``lookbar.episodes``; what comes back is a table
    its rules take (the baseline's record on the S1 view, tiny: each stock condition's first seed)."""
    m2 = pytest.importorskip("designlab.stage2.monitor2")
    pytest.importorskip("torch")  # (the baseline's record names the trainer's sources: train2 imports it)
    doc = m2.baseline(BODY, "S1", tmp_path / "monitor", tiny=True)
    assert doc["episodes_by"] == "lookbar.episodes" and doc["valid"] is True and doc["row"] == "R0L@10ms"
    assert [(e["condition"], e["seed"], e["outcome"]) for e in doc["episodes"]] == [
        ("walk/nominal", 6001, "read"), ("walk/brisk", 6004, "read")]
    for e in doc["episodes"]:
        assert m2.has_clock(e) and e["survival_credit"] == pytest.approx(e["survival_s"], abs=1e-9)
        assert e["survival_credit"] <= e["survival_s"] and len(e["state_digest"]) == 64
    assert doc["N"] == 0 and doc["S"] == pytest.approx(np.mean([e["survival_s"] for e in doc["episodes"]]), abs=1e-9)
    assert doc["N_reading2"] == 1 and doc["mean_clock_factor"] == pytest.approx(1.0, abs=1e-9)
    assert doc["distinct_start_seeds"] == 2 and doc["traces"] == "traces/r0l"
    assert (tmp_path / "monitor" / "traces" / "r0l" / "R0L@10ms" / "walk" / "brisk" / "6004.npz").is_file()
    again = m2.baseline(BODY, "S1", tmp_path / "again", tiny=True)
    assert m2.baseline_equal(again, doc) == dict(equal=True, episodes=2, differ=[], digests_compared=True)


@needs_run
def test_an_episode_during_which_mujoco_warned_through_the_monitors_entry(monkeypatch):
    """Ruling 9 end to end on the S1 view (one stock condition cut to 1 s): the same episode, once as it runs and once
    with MuJoCo's warnings read as raised (``tune.sim_warnings``, which the row's episode hands to
    ``tune.warned_fall``).  The plant ran the same; the warned one is a fall at 0 s, credited no survival."""
    m2 = pytest.importorskip("designlab.stage2.monitor2")
    stock_cut(monkeypatch, 1.0)
    rows = lookbar.rows(BODY, ["R0L@10ms"])
    look = lookbar.look_row_of(rows["R0L@10ms"])
    calm, = lookbar.episodes(BODY, "S1", rows, [("walk/brisk", 6004)])
    assert (calm["outcome"], calm["warned"], calm["fall_time_s"]) == ("read", False, None)
    assert calm["survival_s"] == calm["simulated_s"] > 0.9 and m2.survival_credit(calm) == pytest.approx(calm["survival_s"])
    warnings = dict(BADQACC=dict(number=2, lastinfo=7))
    monkeypatch.setattr(tune, "sim_warnings", lambda d: warnings)
    warned, = lookbar.episodes(BODY, "S1", rows, [("walk/brisk", 6004)])
    blk = look.last
    assert (blk["fell"], blk["fall_time_s"], blk["mujoco_warnings"]) == (True, 0.0, warnings)
    assert blk["survival_s"] == calm["simulated_s"] and blk["fall_cause"].startswith(lookbar.WARNED_CAUSE)
    assert warned["state_digest"] == calm["state_digest"] and warned["steps"] == calm["steps"]
    assert (warned["outcome"], warned["fell"], warned["warned"], warned["fall_time_s"], warned["survival_s"]) == (
        "fell", True, True, 0.0, 0.0)
    assert warned["simulated_s"] == calm["simulated_s"] and warned["failed_terms"] == ["survived"]
    assert m2.has_clock(warned) and m2.survival_credit(warned) == 0.0


# ------------------------------------------------------------------------------------------------
# run: the bar's runner with these rows, then collect
# ------------------------------------------------------------------------------------------------
@needs_run
def test_run_on_the_s1_view_writes_bar_records_sidecars_and_companions(tmp_path, monkeypatch, capsys):
    stock_cut(monkeypatch, 1.0)
    out = tmp_path / "out"
    code = lookbar.main(["run", "--set", "S1", "--body", BODY, "--rows", "R0L@10ms,R0@10ms", "--conditions", "walk/brisk",
                         "--episodes", "2", "--procs", "1", "--stage1-run", str(RUN), "--out", str(out)])
    text = capsys.readouterr().out
    assert code == 0 and "R0L@10ms walk/brisk" in text and "reading 1 does not count on this set" in text
    bdir = bar.block_dir(out, "S1", BODY, bar.HEADLINE_BLOCK)
    for row in ("R0L@10ms", "R0@10ms"):
        rec = s2io.read_json(bdir / "eval" / row / "walk" / "brisk.json")
        doc = s2io.read_json(bdir / "look" / row / "walk" / "brisk.json")
        assert rec["status"] == "done" and rec["counts_look"] is None and "look_reading" in rec["controller"]
        assert rec["attribution"] is None and rec["controller"]["delayed_row"] == row
        assert doc["bar_record"]["key_sha256"] == rec["key_sha256"] and doc["counts"] == rec["counts"]
        assert doc["counts_but_A1"] is None and doc["counts_look"] is None and doc["counts_equal_bar"] is True
        assert [e["outcome"] for e in doc["episodes"]] == ["read", "read"] and doc["look_set"] is False
        for e, b in zip(doc["episodes"], rec["episodes"], strict=True):
            side = s2io.read_json(lookbar.sidecar_path(bdir / "look_partial", row, "walk/brisk", e["seed"]))
            assert (side["row"], side["steps"], side["fell"], side["survival_s"]) == (row, 500, False, b["survival_s"])
            assert e["state_digest"] == side["state_digest"] and e["clock"] == side["clock"] and e["look"] is None
            assert e["succeeded"] == b["succeeded"] and e["succeeded_but_A1"] is None
        assert sorted(rec["key"]["row_sources"]) == ["designlab/stage1/speedfit.py", "designlab/stage2/bar.py",
                                                     "designlab/stage2/lookbar.py", "designlab/stage2/lookread.py"]
    # R0L and the wrapped R0 ran the same episodes
    a, b = (s2io.read_json(bdir / "look" / r / "walk" / "brisk.json")["episodes"] for r in ("R0L@10ms", "R0@10ms"))
    assert [e["state_digest"] for e in a] == [e["state_digest"] for e in b]
    summary = s2io.read_json(bdir / "look" / "summary.json")
    assert [c["row"] for c in summary["records"]] == ["R0@10ms", "R0L@10ms"] and summary["refused"] == []
    # a second run finds the records done and leaves the companions as they are
    before = (bdir / "look" / "R0L@10ms" / "walk" / "brisk.json").read_bytes()
    doc = lookbar.run("S1", BODY, out, ["R0L@10ms", "R0@10ms"], ["walk/brisk"], episodes=2, procs=1, stage1_run=RUN, log=quiet)
    assert doc["bar"]["sessions"][-1]["episodes_run"] == 0 and len(doc["records"]) == 2
    assert (bdir / "look" / "R0L@10ms" / "walk" / "brisk.json").read_bytes() == before
    assert lookbar.main(["collect", "--set", "S1", "--body", BODY, "--out", str(out)]) == 0


@needs_run
def test_run_on_a_fixture_set_and_what_the_bar_refuses_there(t0, tmp_path):
    out = tmp_path / "out"
    doc = lookbar.run(SET, BODY, out, ["R0@10ms"], ["walk/nominal", "stop/two_step"], episodes=1, procs=1, root=t0, log=quiet)
    cells = {c["condition"]: c for c in doc["records"]}
    assert sorted(cells) == ["stop/two_step", "walk/nominal"] and doc["look_set"] is False
    bdir = bar.block_dir(out, SET, BODY, bar.HEADLINE_BLOCK)
    walk = s2io.read_json(bdir / "look" / "R0@10ms" / "walk" / "nominal.json")
    rec = s2io.read_json(bdir / "eval" / "R0@10ms" / "walk" / "nominal.json")
    ep = walk["episodes"][0]
    assert ep["outcome"] == "read" and ep["look"]["terms"] == dict(bob=False, head=False, tail=True, toe_off=True)
    assert ep["steps"] == rec["episodes"][0]["trace"]["steps"] and walk["counts"] == rec["counts"]
    assert walk["counts_but_A1"] == rec["counts_but_A1"] and walk["look_failures_by_term"] == dict(bob=1, head=1)
    assert walk["short_air_runs"] == dict(episodes_read=1, episodes_with_any=0, total=0)
    stop = s2io.read_json(bdir / "look" / "R0@10ms" / "stop" / "two_step.json")
    assert stop["episodes"][0]["outcome"] == "fell" and stop["episodes"][0]["on_command_at_stop"]["speed_m_s"] > 0.3
    assert rec["set_freeze"]["index"] == 0 and walk["set"] == SET
    # as built, the bar's runner refuses a row that is no R0 row on a set outside the 75 cm round, R0L included
    with pytest.raises(bar.BarError, match="only R0 rows run on it"):
        lookbar.run(SET, BODY, tmp_path / "refused", ["R0L@10ms"], ["walk/nominal"], episodes=1, procs=1, root=t0, log=quiet)
    assert lookbar.main(["run", "--set", SET, "--body", BODY, "--rows", "R0L@10ms", "--set-root", str(t0),
                         "--out", str(tmp_path / "refused")]) == 1


# ------------------------------------------------------------------------------------------------
# Slow: whole episodes on the S1 view
# ------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not SLOW, reason="whole closed-loop episodes on Stage 1's own run directory (STAGE2_SLOW=1)")
@needs_run
@pytest.mark.parametrize("key,seed,k", [("walk/nominal", 3042, 5), ("walk/brisk", 3043, 5), ("turn/walking_0.35_left", 3042, 0),
                                        ("start_stop/twice", 3044, 5), ("push/walking_left", 3042, 5),
                                        ("carpet/robot_floor_carpet", 3042, 5)])
def test_whole_stock_episodes_of_r0l_are_r0s(key, seed, k):
    opened = lookbar.open_set(BODY, "S1", [key], stage1_run=RUN)
    ctx, cond = opened["ctx"], opened["conds"][key]
    name = f"@{bar.STEP_MS * k:d}ms"
    look, ref = lookbar.rows(BODY, ["R0L" + name])["R0L" + name], bar.rows(BODY, ["R0" + name])["R0" + name]
    d0, d1 = g8a.row_episode(ref, ctx, cond, seed), g8a.row_episode(look, ctx, cond, seed)
    assert d0["digests"] == d1["digests"] and d0["fell_step"] == d1["fell_step"] and len(d0["digests"]) > 1000
    e0, e1 = bar.run_episode(ref, ctx, cond, seed), bar.run_episode(look, ctx, cond, seed)
    assert same_record(e1, e0)
    blk = lookbar.look_row_of(look).last
    assert (blk["fell"], blk["survival_s"], blk["steps"]) == (e0["fell"], e0["survival_s"], len(d0["digests"]))
