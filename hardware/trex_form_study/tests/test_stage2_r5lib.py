"""R5's reference library (``designlab/stage2/r5lib.py``; round C design D3, section 3.3; rules E4, T1-T3).

Kinematic only: no test here steps a plant.  The tests read Stage 1's own run directory (the S1 view) and a
synthetic frozen set ``T0`` that a fixture builds in a temporary directory from Stage 1's arrays under the bar's
names (``nominal``, ``v0.4`` from the brisk walk, ``start``, ``nominal_to_v0.4``, ``stop2_v0.4`` from Stage 1's
stop, two quick turns from Stage 1's turn entries, one turn recorded as not available).  ``T0``'s moves do not
join its walks; it carries the names and the entries, which is all a table needs.
"""

from __future__ import annotations

import ast
import copy
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from designlab.stage1 import control, env, panels, plant, tune  # noqa: E402
from designlab.stage2 import r5lib, s2io, setio  # noqa: E402

BODY = "75-M-H-10"
RUN = ROOT / "stage1" / "runs" / "2026-11-08"
RATE = 1.05
ONE_THREAD = all(os.environ.get(v) == "1" for v in s2io.THREAD_VARS)
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
T0_REFS = (("nominal", "nominal"), (W, "brisk"))
T0_MOVES = (("start", "start"), (setio.change_name("nominal", W), "nominal_to_brisk"),
            (setio.stop_name(2, W), "stop"), (setio.turn_name("pi4", "left"), "enter_walk_0.35_left"),
            (setio.turn_name("pi4", "right"), "enter_walk_0.35_right"))
ABSENT_TURN = setio.turn_name("0.35", "left")
ABSENT_REASON = "no reference within the thigh twist and the leg drives"
NOTE_V03 = "over the knee drive's Stage 0 limit"
#: ``make_t0(more=True)`` adds the 0.30 m/s rung (Stage 1's nominal walk under the name ``v0.3``, its entry's speed
#: one ulp below 0.30) with its change, and the turn in place under design D5's names, from Stage 1's.
V03_BELOW = float(np.nextafter(0.3, 0.0))
MORE_REFS = (("v0.3", "nominal"),) + tuple((f"inplace_0.5_{d}", f"in_place_0.50_{d}") for d in ("left", "right"))
MORE_MOVES = ((setio.change_name("nominal", "v0.3"), "nominal_to_brisk"),) + tuple(
    (f"{k}_inplace_0.5_{d}", f"{k}_in_place_0.50_{d}") for d in ("left", "right") for k in ("start", "stop"))


def make_t0(root: Path, freeze: bool = True, more: bool = False, set_name: str = SET) -> Path:
    """The synthetic set ``T0`` under ``root`` (module docstring); ``more``: with ``MORE_REFS`` and ``MORE_MOVES``;
    ``set_name``: the name it is made under (``make_look``)."""
    src = RUN / BODY
    d = setio.set_dir(set_name, BODY, root)
    refs1, trs1 = s2io.read_json(src / "refs.json"), s2io.read_json(src / "transitions.json")
    refs, trs, manifest = {}, {}, {}
    t0_refs, t0_moves = T0_REFS + (MORE_REFS if more else ()), T0_MOVES + (MORE_MOVES if more else ())
    for name, old in t0_refs:
        e = copy.deepcopy(refs1["references"][old])
        e.update(name=name, file=f"refs/{name}.npz", set=set_name, strides=1, window_s=e["period_s"], crouch_frac=0.03)
        if name == "v0.3":
            e["speed_m_s"] = V03_BELOW
        refs[name] = e
    for name, old in t0_moves:
        e = copy.deepcopy(trs1["transitions"][old])
        e.update(name=name, file=f"transitions/{name}.npz", a1_over=None)
        if "inplace" in name:  # (no name ``setio`` makes: design D5's)
            e["move"] = dict(kind=name.split("_")[0], builder="fixture", causal=False, command_phase=None)
            trs[name] = e
            continue
        if name == "nominal_to_v0.3":
            e["gaits"]["b"].update(name="v0.3", speed=V03_BELOW)
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
    trs[ABSENT_TURN] = dict(name=ABSENT_TURN, kind="turn", available=False, reason=ABSENT_REASON,
                            numbers=dict(hip_yaw_range_use=0.93))
    for doc, sub, kind in ((refs, "refs", "walk"), (trs, "transitions", "move")):
        for name, e in doc.items():
            if not e.get("file"):
                continue
            (d / sub).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / sub / f"{dict(t0_refs + t0_moves)[name]}.npz", d / e["file"])
            manifest[name] = dict(kind=kind, sha256=e["sha256"], entry_sha256=setio.entry_sha256(kind, e))
    s2io.write_json(d / "refs.json", dict(body=BODY, set=set_name, schema=1, references=refs))
    s2io.write_json(d / "transitions.json", dict(body=BODY, set=set_name, schema=1, transitions=trs))
    freezes = [dict(index=0, at="fixture", commit=None, manifest=manifest,
                    manifest_sha256=s2io.sha256_json(manifest))] if freeze else []
    s2io.write_json(d / "set.json", dict(set=set_name, body=BODY, role="fixture", crouch_frac=0.03,
                                         not_available={"v0.3": dict(reason=NOTE_V03, numbers=None)},
                                         freezes=freezes))
    return d

#: The synthetic look set ``make_look`` builds: the walk ``v0.4`` a window of ``LOOK_K`` strides that varies from
#: stride to stride, and the causal moves that leave it at join rule J1's leave point (move, Stage 1's file,
#: command phase, a_phase strides into the window: ``moves_look.LEAVE_PHASE``).
LOOK_SET = "T6"
LOOK_K = 3
LOOK_AMP_RAD = 0.02
LOOK_MOVES = ((setio.turn_name("pi4", "left"), "enter_walk_0.35_left", 0.0, 2.0),
              (setio.turn_name("pi4", "right"), "enter_walk_0.35_right", 0.5, 2.5),
              (setio.stop_name(2, W), "stop", -0.5, 2.5))
LOOK_FADE_FRAMES = 60


def make_look(root: Path, set_name: str = LOOK_SET, freeze: bool = True, stand_in_walks: bool = False) -> Path:
    """A synthetic look set under ``root``: ``T0`` (``make_t0``) with ``v0.4`` a window of ``LOOK_K`` strides of
    Stage 1's brisk walk (each stride advanced by the stride motion, the window's own ``phase`` 0..K) whose joints
    carry a variation over the window, ``LOOK_AMP_RAD x sin(2 pi t / W)`` (periodic in the window as a whole, as
    F14's is, so no two strides of it are alike), and the quick turns and the two-step stop moved to the frame the
    window holds at their leave point (a_phase strides into it; the window is uniform in time, so a_time_s is
    a_phase x the stride period and lies on a frame), with ``look.window_a``, ``a_phase``, ``a_time_s`` and
    ``a_windows`` in their entries as ``moves_look.Bound.look`` writes them.  Each move is Stage 1's file with its
    joints shifted onto that frame, the shift fading over its first ``LOOK_FADE_FRAMES`` frames: it starts at the
    frame it was built from and joins nothing else.  ``stand_in_walks``: ``set.json`` says its walks stand in for
    the rungs they are named after (the bar's planner then reads the brisk walk as ``v0.4``)."""
    d = make_t0(root, freeze=False, set_name=set_name)
    one = dict(np.load(RUN / BODY / "refs" / "brisk.npz"))
    n, dt = len(one["t"]), float(one["dt"])
    T = n * dt
    adv = np.asarray(one["stride_motion"], float)
    win = {}
    for key, x in one.items():
        if x.ndim and len(x) == n and key not in ("q_home", "joints", "stride_motion"):
            parts = []
            for k in range(LOOK_K):
                y = np.array(x, copy=True)
                if key == "t":
                    y = y + k * T
                elif key in ("q", "com", "com_ref", "zmp_ref", "zmp_mb", "feet"):
                    y[..., 0:2] += k * adv[3:5]
                parts.append(y)
            win[key] = np.concatenate(parts)
        else:
            win[key] = x
    Wn = LOOK_K * T
    t = win["t"]
    win["q"][:, 7:] += LOOK_AMP_RAD * np.sin(2.0 * np.pi * t / Wn)[:, None]
    win["qd"][:, 6:] += (LOOK_AMP_RAD * 2.0 * np.pi / Wn * np.cos(2.0 * np.pi * t / Wn))[:, None]
    win["stride_motion"] = np.r_[adv[:3], LOOK_K * adv[3:5]]
    win["phase"] = np.arange(LOOK_K * n) / n
    win["window_strides"] = np.array(LOOK_K)
    refs, trs = s2io.read_json(d / "refs.json"), s2io.read_json(d / "transitions.json")
    e = refs["references"][W]
    e.update(sha256=s2io.write_npz(d / "refs" / f"{W}.npz", win), frames=LOOK_K * n, strides=LOOK_K, window_s=Wn)
    for name, src, cmd, a_phase in LOOK_MOVES:
        arr = dict(np.load(RUN / BODY / "transitions" / f"{src}.npz"))
        ka = int(round(a_phase * n))
        m = min(LOOK_FADE_FRAMES, len(arr["q"]))
        fade = np.zeros(len(arr["q"]))
        fade[:m] = 0.5 * (1.0 + np.cos(np.pi * np.arange(m) / m))
        arr["q"][:, 7:] += np.outer(fade, win["q"][ka, 7:] - arr["q"][0, 7:])
        a_time = a_phase * T
        te = trs["transitions"][name]
        window_a = dict(reference=W, strides=LOOK_K, window_s=Wn, period_s=T, offset_strides=a_phase - cmd,
                        a_phase=a_phase, a_time_s=a_time, a_windows=a_time / Wn, varies=True)
        te.update(sha256=s2io.write_npz(d / te["file"], arr),
                  look=dict(laws="look", window_a=window_a, window_b=None, a_phase=a_phase, a_time_s=a_time,
                            a_windows=a_time / Wn))
        assert te["move"]["causal"] and te["move"]["command_phase"] == cmd
    s2io.write_json(d / "refs.json", refs)
    s2io.write_json(d / "transitions.json", trs)
    sd = s2io.read_json(d / "set.json")
    if stand_in_walks:
        sd["stand_in_walks"] = True
    if freeze:
        manifest = {}
        for doc, kind in ((refs["references"], "walk"), (trs["transitions"], "move")):
            for name, x in doc.items():
                if x.get("file"):
                    manifest[name] = dict(kind=kind, sha256=x["sha256"], entry_sha256=setio.entry_sha256(kind, x))
        sd["freezes"] = [dict(index=0, at="fixture", commit=None, manifest=manifest,
                              manifest_sha256=s2io.sha256_json(manifest))]
    s2io.write_json(d / "set.json", sd)
    return d


@pytest.fixture(scope="module")
def p():
    return plant.build(BODY, "hard")


@pytest.fixture(scope="module")
def s1(p):
    return r5lib.Library(p, BODY, "S1", RATE)


@pytest.fixture(scope="module")
def t0_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("sets")
    make_t0(root)
    return root


@pytest.fixture(scope="module")
def t0(p, t0_root):
    return r5lib.Library(p, BODY, SET, RATE, root=t0_root)


# ------------------------------------------------------------------------------------------------
# The task table (rule T1) and what the modules import
# ------------------------------------------------------------------------------------------------
L = [("transition", "start"), ("ref", "nominal", 1), ("transition", "nominal_to_v0.4")]
E = r5lib.END
#: Rule T1's table, written out: name -> (plan, weight).
T1 = {
    "stand": ([("hold", 12.0), ("transition", "start")], 1),
    "walk_nominal": ([("transition", "start"), ("ref", "nominal", E)], 2),
    "walk_v0.4": (L + [("ref", "v0.4", E)], 3),
    **{f"walk_v{v}": ([("transition", "start"), ("ref", "nominal", 1), ("transition", f"nominal_to_v{v}"),
                       ("ref", f"v{v}", E)], 0.5) for v in ("0.3", "0.35", "0.45", "0.5")},
    "turn_q90_left": (L + [("ref", "v0.4", 2.0), ("transition", "turn90_pi4_left"), ("ref", "v0.4", E)], 1.5),
    "turn_q90_right": (L + [("ref", "v0.4", 2.5), ("transition", "turn90_pi4_right"), ("ref", "v0.4", E)], 1.5),
    **{f"turn_{r}_{d}": (L + [("ref", "v0.4", 2.0 if d == "left" else 2.5), ("transition", f"turn90_{r}_{d}"),
                              ("ref", "v0.4", E)], 0.25) for r in ("0.35", "0.5", "1") for d in ("left", "right")},
    "stop": (L + [("ref", "v0.4", 2.5), ("transition", "stop2_v0.4"), ("hold", r5lib.STOP_HOLD_S)], 2),
    "start_stop": (L + [("ref", "v0.4", 2.5), ("transition", "stop2_v0.4"), ("hold", 2.0), ("transition", "start"),
                        ("ref", "nominal", 2)], 1),
    **{f"inplace_{d}": ([("transition", f"start_inplace_0.5_{d}"), ("ref", f"inplace_0.5_{d}", 3),
                         ("transition", f"stop_inplace_0.5_{d}")], 0.5) for d in ("left", "right")},
}
#: Rule M4's table: monitor condition -> the tasks whose weights its failure doubles.
M4 = {"walk/nominal": ["walk_nominal"], "walk/v0.4": ["walk_v0.4"], "nudge/forward": ["walk_v0.4"],
      "nudge/left": ["walk_v0.4"], "turn/q90_left": ["turn_q90_left", "turn_q90_right"],
      "stop/two_step": ["stop", "start_stop"]}


def test_task_table_is_rule_t1():
    got = {t.name: ([tuple(x) for x in t.plan], t.weight) for t in r5lib.TASKS}
    assert got == {n: ([tuple(x) for x in plan], float(w)) for n, (plan, w) in T1.items()}
    assert len(r5lib.TASKS) == 19 and r5lib.MAX_SECONDS == 30.0
    stand = next(t for t in r5lib.TASKS if t.name == "stand")
    assert stand.seconds == 10.0 and all(t.seconds is None for t in r5lib.TASKS if t.name != "stand")
    # the commanded heading is rule E4's integral in every task but the two turns in place (the lead's ruling)
    assert (r5lib.HEADING_COMMAND, r5lib.HEADING_REFERENCE) == ("command", "reference")
    assert {t.name: t.heading for t in r5lib.TASKS if t.heading != "command"} == {
        "inplace_left": "reference", "inplace_right": "reference"}
    assert all(t.heading == "command" for t in r5lib.S1_TASKS)
    by_condition: dict = {}
    for t in r5lib.TASKS:
        for c in t.monitor:
            by_condition.setdefault(c, []).append(t.name)
    assert by_condition == M4


def test_lead_strides_are_the_bars():
    bar = pytest.importorskip("designlab.stage2.bar")
    assert r5lib.TURN_LEAD == bar.TURN_LEAD_STRIDES and r5lib.STOP_LEAD == bar.STOP_LEAD_STRIDES[2]
    assert r5lib.BAR_WALK == bar.BAR_WALK and r5lib.MID_SWING == bar.readings.MID_SWING


def _imports(path: Path) -> set:
    out = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            out |= {f"{'.' * node.level}{node.module or ''}:{a.name}" for a in node.names}
        elif isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
    return out


def test_no_set_is_iterated_and_the_modules_import_as_designed():
    """An episode may not depend on PYTHONHASHSEED: the library, the core and the env hold no set at all.  The
    library imports neither the bar nor the core nor the env; the core does not import the env."""
    stage2 = ROOT / "designlab" / "stage2"
    for name in ("r5lib.py", "r5core.py", "r5env.py"):
        tree = ast.parse((stage2 / name).read_text())
        for node in ast.walk(tree):
            assert not isinstance(node, (ast.Set, ast.SetComp)), f"{name}:{node.lineno} builds a set"
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in ("set", "frozenset"), f"{name}:{node.lineno} builds a set"
    lib, core = _imports(stage2 / "r5lib.py"), _imports(stage2 / "r5core.py")
    assert not any(x.split(":")[-1] in ("bar", "r5core", "r5env") for x in lib)
    assert not any(x.split(":")[-1] == "r5env" for x in core)


# ------------------------------------------------------------------------------------------------
# The S1 view
# ------------------------------------------------------------------------------------------------
@needs_run
def test_s1_table_and_inputs(s1):
    assert list(s1.tasks) == ["stand", "walk", "brisk", "stop"] and all(t["available"] for t in s1.tasks.values())
    for name in ("stand", "walk", "brisk"):
        assert [tuple(x) for x in s1.tasks[name]["plan"]] == [tuple(x) for x in env.TASKS[name][0]]
    assert s1.tasks["stop"]["plan"][-1] == ["hold", r5lib.STOP_HOLD_S]
    assert s1.tasks["stand"]["seconds"] == 10.0
    for name in ("walk", "brisk", "stop"):
        assert s1.tasks[name]["seconds"] == pytest.approx(s1.script(name).t_end / RATE)
    inp = s1.inputs
    assert inp["freeze"] is None and inp["set"] == "S1" and inp["body"] == BODY
    assert inp["plant_inputs"] == panels.key_context(BODY)["plant_inputs"]
    refs = s2io.read_json(RUN / BODY / "refs.json")["references"]
    assert inp["files"]["refs/nominal.npz"] == dict(sha256=refs["nominal"]["sha256"],
                                                    entry_sha256=setio.entry_sha256("walk", refs["nominal"]))
    assert sorted(inp["files"]) == ["refs/brisk.npz", "refs/nominal.npz", "transitions/brisk_to_nominal.npz",
                                    "transitions/nominal_to_brisk.npz", "transitions/start.npz", "transitions/stop.npz"]
    q_home = np.load(RUN / BODY / "refs" / "nominal.npz")["q_home"]
    assert s1.omega == control.omega0(s1.p, q_home) and s1.crouch_frac == 0.03
    doc = s1.describe()
    text = json.dumps(s2io.plain(doc))  # (a plain document)
    assert str(ROOT) not in text and "/home" not in text and "/tmp" not in text
    assert doc["absent"] == {} and doc["command_rule"] == r5lib.COMMAND_RULE and list(doc["tasks"]) == list(s1.tasks)
    assert doc["crouch_rule"] == r5lib.CROUCH_RULE
    # the names of the tasks the view holds, from its records alone (no plant): what the library resolves
    assert r5lib.task_names(BODY, "S1") == list(s1.tasks) == ["stand", "walk", "brisk", "stop"]


@needs_run
def test_s1_roots():
    assert r5lib.s1_root(BODY) == RUN
    assert r5lib.s1_root("75-M-H-10-X") == ROOT / "stage2" / "x"
    assert r5lib.s1_root("75-M-H-10-J") == ROOT / "stage2" / "j"


# ------------------------------------------------------------------------------------------------
# A frozen set: the resolved table, absent tasks, refusals
# ------------------------------------------------------------------------------------------------
@needs_run
def test_resolved_table_and_absent_tasks_on_a_synthetic_set(t0, t0_root):
    have = [n for n, t in t0.tasks.items() if t["available"]]
    assert have == ["stand", "walk_nominal", "walk_v0.4", "turn_q90_left", "turn_q90_right", "stop", "start_stop"]
    assert list(t0.tasks) == [t.name for t in r5lib.TASKS]  # the table's order, absent tasks in place
    absent = t0.describe()["absent"]
    assert sorted(absent) == sorted(set(T1) - set(have))
    assert absent["turn_0.35_left"] == f"transitions/{ABSENT_TURN}: {ABSENT_REASON}"  # the entry's own reason
    assert absent["walk_v0.3"] == "transitions/nominal_to_v0.3: the set has no entry transitions/nominal_to_v0.3"
    assert "the set has no entry" in absent["inplace_left"]
    assert absent["turn_1_right"].startswith("transitions/turn90_1")
    assert t0.why_not("refs", "v0.3") == NOTE_V03  # set.json's note where there is no entry
    # the same table from the set's records alone (no plant, no array): the names an env config stores (rule K4)
    assert r5lib.task_names(BODY, SET, t0_root) == have
    docs = setio.load(setio.set_dir(SET, BODY, t0_root))
    records = (docs["refs.json"]["references"], docs["transitions.json"]["transitions"], docs["set.json"])
    for td in r5lib.TASKS:
        assert r5lib.absent_reason(td, *records) == t0.tasks[td.name]["reason"], td.name
    assert r5lib.why_not(*records, "refs", "v0.3") == NOTE_V03 and r5lib.why_not(*records, "refs", "v0.4") is None
    for name in have:
        t = t0.tasks[name]
        assert t["reason"] is None and all(isinstance(x[-1], (int, float, str)) for x in t["plan"])
        assert not any(x[-1] == r5lib.END for x in t["plan"] if x[0] == "ref")
    # "to the end": whole windows until the plan reaches 30 s at the commanded clock, and no window more
    for name in ("walk_nominal", "walk_v0.4", "turn_q90_left"):
        plan = [tuple(x) for x in t0.tasks[name]["plan"]]
        (t_last, _), window = t0.segment_times(plan)[-1], t0._load(plan[-1]).period
        t_end = t_last + plan[-1][2] * window
        assert t_end >= 30.0 * RATE > t_end - window and t0.tasks[name]["seconds"] == 30.0
        assert t0.script(name).t_end == pytest.approx(t_end, abs=1e-9)
    assert t0.tasks["stand"]["seconds"] == 10.0 and t0.tasks["stop"]["seconds"] < 30.0
    inp = t0.inputs
    doc = s2io.read_json(setio.set_dir(SET, BODY, t0_root) / "set.json")
    assert inp["freeze"] == dict(index=0, manifest_sha256=doc["freezes"][0]["manifest_sha256"])
    assert "transitions/turn90_pi4_left.npz" in inp["files"] and f"transitions/{ABSENT_TURN}.npz" not in inp["files"]
    with pytest.raises(r5lib.Absent, match="thigh twist"):
        t0.script("turn_0.35_left")
    with pytest.raises(tune.InputError, match=ABSENT_TURN):
        t0.from_plan([("transition", "start"), ("transition", ABSENT_TURN)])


@needs_run
def test_a_set_must_be_frozen_and_unchanged(p, tmp_path):
    make_t0(tmp_path / "a", freeze=False)
    with pytest.raises(tune.InputError, match="not frozen"):
        r5lib.Library(p, BODY, SET, RATE, root=tmp_path / "a")
    assert r5lib.Library(p, BODY, SET, RATE, root=tmp_path / "a", require_frozen=False).inputs["freeze"] is None
    d = make_t0(tmp_path / "b")
    with open(d / "refs" / "v0.4.npz", "ab") as f:
        f.write(b"x")
    with pytest.raises(tune.InputError, match="v0.4.npz has sha256"):
        r5lib.Library(p, BODY, SET, RATE, root=tmp_path / "b")
    d = make_t0(tmp_path / "c")
    doc = s2io.read_json(d / "set.json")
    del doc["crouch_frac"]
    s2io.write_json(d / "set.json", doc)
    with pytest.raises(tune.InputError, match="crouch_frac"):
        r5lib.Library(p, BODY, SET, RATE, root=tmp_path / "c")


def _edit(path: Path, change) -> None:
    doc = s2io.read_json(path)
    change(doc)
    s2io.write_json(path, doc)


@needs_run
def test_a_set_whose_crouch_differs_from_a_walk_entrys_is_refused(p, tmp_path):
    """No freeze covers ``set.json``'s ``crouch_frac``, with which the track's LIPM frequency is computed; a freeze
    covers each walk entry's own.  A set where they differ is refused with both values, as ``bar.set_crouch``
    refuses it: for every walk entry the library reads, also one that only a plan given as a plan reads."""
    bar = pytest.importorskip("designlab.stage2.bar")
    # set.json edited after the freeze: the frozen set still reads as frozen and unchanged
    d = make_t0(tmp_path / "a")
    ok = r5lib.Library(p, BODY, SET, RATE, root=tmp_path / "a")
    assert ok.crouch_frac == 0.03 and ok.describe()["crouch_rule"] == r5lib.CROUCH_RULE
    assert "refused with both values" in r5lib.CROUCH_RULE and "bar.set_crouch" in r5lib.CROUCH_RULE
    _edit(d / "set.json", lambda doc: doc.update(crouch_frac=0.08))
    with pytest.raises(tune.InputError) as err:
        r5lib.Library(p, BODY, SET, RATE, root=tmp_path / "a")
    text = str(err.value)
    assert f"{SET}/{BODY}: set.json records crouch_frac 0.08" in text
    assert "its walk entry 'nominal' records 0.03" in text and "its walk entry 'v0.4' records 0.03" in text
    assert "transitions" not in text  # (a move's entry records no crouch)
    docs = setio.load(d)
    with pytest.raises(bar.BarError, match="records crouch_frac 0.08, its walk entry 'nominal' records 0.03"):
        bar.set_crouch(docs["set.json"], docs["refs.json"]["references"], ["nominal", "v0.4"])  # the bar's refusal
    # a walk entry with a crouch of its own, or with none
    for i, (own, said) in enumerate(((0.05, "0.05"), (None, "None"))):
        d = make_t0(tmp_path / f"b{i}", freeze=False)

        def other(doc, own=own):
            if own is None:
                del doc["references"]["v0.4"]["crouch_frac"]
            else:
                doc["references"]["v0.4"]["crouch_frac"] = own

        _edit(d / "refs.json", other)
        with pytest.raises(tune.InputError, match=f"records crouch_frac 0.03, its walk entry 'v0.4' records {said}:"):
            r5lib.Library(p, BODY, SET, RATE, root=tmp_path / f"b{i}", require_frozen=False)
    # a walk that no task of the table reads is not read, and is refused when a plan given as a plan reads it
    d = make_t0(tmp_path / "c", freeze=False)

    def one_more(doc):
        e = copy.deepcopy(doc["references"]["v0.4"])
        e.update(name="v0.45", file="refs/v0.45.npz", crouch_frac=0.06)
        doc["references"]["v0.45"] = e

    _edit(d / "refs.json", one_more)
    shutil.copyfile(d / "refs" / "v0.4.npz", d / "refs" / "v0.45.npz")
    lib = r5lib.Library(p, BODY, SET, RATE, root=tmp_path / "c", require_frozen=False)
    assert "refs/v0.45.npz" not in lib.inputs["files"] and not lib.tasks["walk_v0.45"]["available"]
    with pytest.raises(tune.InputError, match="records crouch_frac 0.03, its walk entry 'v0.45' records 0.06"):
        lib.from_plan([("transition", "start"), ("ref", "v0.45", 2)])
    assert "refs/v0.45.npz" not in lib.inputs["files"]  # (a refused file is not noted as read)
    assert lib.from_plan([("transition", "start"), ("ref", "nominal", 2)]).t_end > 0.0  # the others still build


# ------------------------------------------------------------------------------------------------
# Command, heading, phase
# ------------------------------------------------------------------------------------------------
def _segments(script):
    return [(s.t0, s.t0 + s.duration) for s in script.track.segments]


@needs_run
def test_command_is_constant_inside_a_segment_and_changes_at_first_frames_only(s1, t0):
    refs = s2io.read_json(RUN / BODY / "refs.json")["references"]
    v_nom, v_brisk = refs["nominal"]["speed_m_s"], refs["brisk"]["speed_m_s"]
    s = s1.script("brisk")  # start, nominal, nominal_to_brisk, brisk, brisk_to_nominal, nominal, stop
    want = [v_nom, v_nom, v_brisk, v_brisk, v_nom, v_nom, 0.0]
    for (a, b), v in zip(_segments(s), want, strict=True):
        for t in np.linspace(a, b, 23, endpoint=False):
            assert s.command(float(t)) == (v, 0.0)
        assert s.command(b - 1e-9) == (v, 0.0)  # nothing before the next segment's first frame depends on it
    assert s.command(s.t_end + 5.0) == (0.0, 0.0)  # the last segment's command continues
    stand = s1.script("stand")
    assert stand.command(3.0) == (0.0, 0.0) and stand.command(12.0) == (v_nom, 0.0)  # a start: gait b from frame 0
    hold = s1.script("stop")
    assert hold.command(hold.t_end - 1.0) == (0.0, 0.0) and hold.kind(hold.t_end - 1.0) == "hold"
    # a stop commands zero from its first frame, while the reference still walks (the command instant of a stop)
    i = next(k for k, x in enumerate(hold.plan) if x == ("transition", "stop"))
    t_stop = hold.track.segments[i].t0
    assert hold.command(t_stop) == (0.0, 0.0) and hold.command(t_stop - 1e-9) == (v_nom, 0.0)
    assert hold.walking(t_stop + 1.0)
    # every change of the command over a whole script falls on a segment's first frame
    ts = np.arange(0.0, s.t_end, 0.004)
    cmd = np.array([s.command(float(t)) for t in ts])
    starts = np.array([a for a, _ in _segments(s)])
    for k in np.flatnonzero(np.any(cmd[1:] != cmd[:-1], axis=1)):
        assert np.any((starts > ts[k]) & (starts <= ts[k + 1]))


@needs_run
def test_a_turns_pulse_integrates_to_its_angle(t0):
    refs = s2io.read_json(RUN / BODY / "refs.json")["references"]
    for d, sign in (("left", 1.0), ("right", -1.0)):
        s = t0.script(f"turn_q90_{d}")
        i = next(k for k, x in enumerate(s.plan) if x[0] == "transition" and x[1].startswith("turn90"))
        a, b = _segments(s)[i]
        w0, pulse = math.pi / 4, (math.pi / 2) / (math.pi / 4)
        assert s.has_turn and b - a > pulse
        v_a = refs["nominal"]["speed_m_s"]  # the walk the fixture's turn leaves (gaits.a)
        assert s.command(a - 1e-9) == (refs["brisk"]["speed_m_s"], 0.0) and s.heading(a - 1e-9) == 0.0
        assert s.command(a) == (v_a, sign * w0) and s.command(a + pulse - 1e-9) == (v_a, sign * w0)
        assert s.command(a + pulse) == (v_a, 0.0) and s.command(b - 1e-9) == (v_a, 0.0)
        ts = np.linspace(a, b, 20001)
        w = np.array([s.command(float(t))[1] for t in ts])
        assert np.trapezoid(w, ts) == pytest.approx(sign * math.pi / 2, abs=1e-3)
        assert s.heading(a + 0.5 * pulse) == pytest.approx(sign * math.pi / 4, abs=1e-12)
        for t in (a + pulse, b - 1e-9, b + 3.0, s.t_end, 1.25 * s.t_end + 2.0):
            assert s.heading(t) == pytest.approx(sign * math.pi / 2, abs=1e-12)
        # the event of the causal move sits at its first frame
        ev = [e for e in s.events if e["kind"] == "turn"]
        assert len(ev) == 1 and ev[0]["t_ref"] == a


@pytest.fixture(scope="module")
def t0_more(p, tmp_path_factory):
    root = tmp_path_factory.mktemp("sets_more")
    make_t0(root, more=True)
    return r5lib.Library(p, BODY, SET, RATE, root=root)


@needs_run
def test_in_a_turn_in_place_the_commanded_heading_is_the_references_own_yaw(t0_more):
    """The lead's ruling of 16 Nov 2026 on rule E4: in ``inplace_left`` and ``inplace_right``, and in no other task,
    the commanded heading follows the reference's own yaw.  The same plan read by rule E4's integral (a plan given
    as a plan: ``from_plan``) runs far ahead of its reference through the whole turn; the command is the same."""
    lib = t0_more
    have = [n for n, t in lib.tasks.items() if t["available"]]
    assert have == ["stand", "walk_nominal", "walk_v0.4", "walk_v0.3", "turn_q90_left", "turn_q90_right", "stop",
                    "start_stop", "inplace_left", "inplace_right"]
    assert {n: t["heading"] for n, t in lib.tasks.items() if t["heading"] != "command"} == {
        "inplace_left": "reference", "inplace_right": "reference"}  # (absent tasks carry theirs too)
    assert lib.describe()["tasks"]["inplace_left"]["heading"] == "reference"
    assert "inplace_left and inplace_right" in r5lib.COMMAND_RULE
    for d, sign in (("left", 1.0), ("right", -1.0)):
        s = lib.script(f"inplace_{d}")
        e4 = lib.from_plan(s.plan)
        assert s.heading_rule == s.describe()["heading"] == "reference" and e4.heading_rule == "command"
        assert e4 is not s and e4.plan == s.plan and not s.has_turn
        start, turn, stop = _segments(s)
        ts = np.concatenate([np.linspace(0.0, 1.25 * s.t_end + 2.0, 1201), [turn[0], turn[1] - 1e-9, s.t_end]])
        for t in ts:
            t = float(t)
            assert s.heading(t) == s.ref_yaw(t) == e4.ref_yaw(t) and s.command(t) == e4.command(t)
        # rule E4, followed to the letter: 0.5 rad/s from the start's first frame, zero from the stop's
        assert s.command(0.0) == s.command(turn[0]) == (0.0, sign * 0.5) and s.command(stop[0]) == (0.0, 0.0)
        gap = np.array([e4.ref_yaw(float(t)) - e4.heading(float(t)) for t in np.linspace(turn[0], turn[1], 301)])
        assert np.all(sign * gap < -math.radians(60.0))  # the integral: over 60 degrees ahead, the whole turn long
        assert sign * (e4.ref_yaw(turn[0]) - e4.heading(turn[0])) == pytest.approx(-math.radians(71.57), abs=2e-3)
        assert abs(e4.ref_yaw(s.t_end) - e4.heading(s.t_end)) < 1e-6  # ... and level again at the plan's end only
        assert e4.heading(s.t_end) == pytest.approx(sign * 0.5 * stop[0], abs=1e-12)  # 0.5 rad/s until the stop
        assert s.heading(s.t_end) == pytest.approx(sign * 0.5 * stop[0], abs=1e-6)
    # walking turns keep rule E4 as coded, on the same set
    s = lib.script("turn_q90_left")
    i = next(k for k, x in enumerate(s.plan) if x[0] == "transition" and x[1].startswith("turn90"))
    a, _ = _segments(s)[i]
    assert s.heading_rule == "command" and s.heading(a + 1.0) == pytest.approx(math.pi / 4, abs=1e-12)
    with pytest.raises(ValueError, match="heading"):
        r5lib.Script(lib, s.plan, heading="gyro")


@needs_run
def test_a_plan_given_as_a_plan_can_name_the_references_yaw(t0_more):
    """A row that plays a turn-in-place plan as a plan (``from_plan``) names the heading the task's definition
    names: with ``heading="reference"`` the script's commanded heading is the reference's own yaw, as the task's
    is; without it, rule E4's integral, as before.  The command, the track and every other table are the same."""
    lib = t0_more
    for d in ("left", "right"):
        task = lib.script(f"inplace_{d}")
        e4 = lib.from_plan(task.plan)
        ref = lib.from_plan(task.plan, heading=r5lib.HEADING_REFERENCE)
        assert (e4.heading_rule, ref.heading_rule, task.heading_rule) == ("command", "reference", "reference")
        assert ref is not e4 and ref is not task and ref.describe()["heading"] == "reference"
        # kept by plan and heading: the default is the script it always was
        assert lib.from_plan(task.plan) is e4 is lib.from_plan(task.plan, heading="command")
        assert lib.from_plan([list(x) for x in task.plan], heading="reference") is ref
        _, turn, _ = _segments(task)
        ts = np.concatenate([np.linspace(0.0, 1.25 * task.t_end + 2.0, 601), [turn[0], turn[1] - 1e-9, task.t_end]])
        for t in ts:
            t = float(t)
            assert ref.heading(t) == task.heading(t) == ref.ref_yaw(t) and ref.command(t) == e4.command(t)
            assert _tables(ref, t) == _tables(e4, t) == _tables(task, t)
        assert max(abs(e4.heading(float(t)) - ref.heading(float(t))) for t in ts) > math.radians(60.0)
        # with a track already built from the plan (a row's built condition), as without one
        track = setio.build_track(lib.p, lib.dir, task.plan, lib.omega, lib.tr_doc)
        built = lib.from_plan(task.plan, track=track, heading="reference")
        assert built.track is track and built.heading_rule == "reference" and built is not ref
        assert lib.from_plan(task.plan, track=track).heading_rule == "command"
        assert built.heading(turn[0] + 0.7) == ref.heading(turn[0] + 0.7)
    with pytest.raises(ValueError, match="heading 'gyro': 'command' or 'reference'"):
        lib.from_plan(task.plan, heading="gyro")


@needs_run
def test_the_duty_bands_target_speed_is_the_speed_a_rungs_name_holds(t0_more):
    """Rule W3's band switches below a target speed of 0.30 m/s and one rung is 0.30 m/s: the target is the speed
    the rung's name holds, as the bar takes it, so an entry whose computed speed is one ulp below 0.30 is no slow
    walk.  The command stays the entry's own speed."""
    bar = pytest.importorskip("designlab.stage2.bar")
    lib = t0_more
    assert V03_BELOW < 0.3 == setio.SPEEDS[0] and lib.refs_doc["v0.3"]["speed_m_s"] == V03_BELOW
    v_nom = lib.refs_doc["nominal"]["speed_m_s"]
    s = lib.script("walk_v0.3")  # start, one window of nominal, the change, v0.3 to the end
    assert [x[1] for x in s.plan] == ["start", "nominal", "nominal_to_v0.3", "v0.3"]
    want = [(v_nom, v_nom), (v_nom, v_nom), (V03_BELOW, 0.3), (V03_BELOW, 0.3)]  # (the command, the target)
    for (a, b), (v_cmd, v_target) in zip(_segments(s), want, strict=True):
        for t in (a, 0.5 * (a + b), b - 1e-9):
            assert s.command(t) == (v_cmd, 0.0) and s.target_speed(t) == v_target
    assert s.target_speed(s.t_end + 3.0) == 0.3 and not s.target_speed(s.t_end - 1.0) < 0.30
    planner = bar.BarPlanner(lib.dir, lib.refs_doc, lib.tr_doc, dict(lib.set_doc, stand_in_walks=True))
    for name in ("v0.3", "v0.4", "nominal"):  # the bar's own reading of a walk's target
        assert r5lib.target_speed_of(name, lib.refs_doc[name]["speed_m_s"]) == planner.speed(name)
    assert planner.speed("v0.3") == 0.3 and planner.speed("nominal") == v_nom
    # a name that holds no speed leaves the entry's own: a stand, a Stage 1 name, a turn in place, no name
    for name in ("stand", "brisk", "nominal", "inplace_0.5_left", "start", None):
        assert r5lib.target_speed_of(name, 0.123) == 0.123
    stop = lib.script("stop")
    assert stop.target_speed(stop.t_end - 1.0) == 0.0 and lib.script("inplace_left").target_speed(1.0) == 0.0
    turn = lib.script("turn_q90_left")  # a finite turn keeps the walk it leaves: gaits.a
    i = next(k for k, x in enumerate(turn.plan) if x[0] == "transition" and x[1].startswith("turn90"))
    assert turn.target_speed(_segments(turn)[i][0] + 0.5) == turn.command(_segments(turn)[i][0] + 0.5)[0]


@needs_run
def test_phase_is_zero_at_left_lift_offs_and_follows_the_stride(s1):
    s = s1.script("brisk")
    lifts = [t0 for foot, t0, _ in setio.planned_swings(s.track) if foot == 0]
    assert len(lifts) > 15
    for t in lifts:
        ph = s.phase(t)
        assert min(ph, 1.0 - ph) < 1e-6, (t, ph)
    for t in np.linspace(0.0, s.t_end * 1.2, 997):
        assert 0.0 <= s.phase(float(t)) < 1.0
    # inside a periodic segment the phase is the local time over the stride, and half a stride on it is one half
    seg = s.track.segments[3]
    assert seg.name == "brisk" and s.stride_s(seg.t0 + 0.1) == seg.period
    assert s.phase(seg.t0 + 0.5 * seg.period) == pytest.approx(0.5, abs=1e-9)
    assert s1.script("stop").phase(s1.script("stop").t_end - 0.5) == 0.0  # a hold


@needs_run
def test_support_and_mid_swing_are_the_tracks(s1):
    s = s1.script("brisk")
    for t in np.linspace(0.0, 1.3 * s.t_end, 3001):
        ref = s.track.info(float(t))[1]
        sw, u = int(ref["swing"]), float(ref["u"])
        assert s.support(float(t)).tolist() == [sw != 0, sw != 1]
        mid = 0.25 <= u <= 0.75
        assert s.mid_swing(float(t)).tolist() == [mid and sw == 0, mid and sw == 1]
        assert s.walking(float(t)) == (not bool(ref["stand"]))


@needs_run
def test_swing_s_is_the_planned_swing_in_progress(s1):
    s = s1.script("brisk")
    swings = sorted((a, b) for _, a, b in setio.planned_swings(s.track))
    lengths = sorted({round(b - a, 6) for a, b in swings})
    assert len(lengths) > 2  # the nominal walk's, the brisk walk's and the changes' own
    near = dict(abs=1e-9)
    for k, (a, b) in enumerate(swings):
        assert s.swing_s(0.5 * (a + b)) == pytest.approx(b - a, **near) and s.swing_s(a) == pytest.approx(b - a, **near)
        if k + 1 < len(swings) and swings[k + 1][0] > b:  # double support: the next swing's
            t = 0.5 * (b + swings[k + 1][0])
            nxt = swings[k + 1][1] - swings[k + 1][0]
            if s.kind(t) == "periodic":  # ... of the segment's own file, which loops (the track's next may be a move's)
                seg = s.track.segments[s.segment_index(t)]
                sw = np.asarray(seg.info["swing"])
                own = [len(r) * seg.step for j in (0, 1) for r in control._swing_runs(sw == j, True)]
                assert any(s.swing_s(t) == pytest.approx(x, **near) for x in own)
                assert abs(s.swing_s(t) - nxt) < 0.02
            else:
                assert s.swing_s(t) == pytest.approx(nxt, **near)
    assert s.swing_s(0.0) == pytest.approx(swings[0][1] - swings[0][0], **near)
    assert s.swing_s(s.t_end + 9.0) == pytest.approx(swings[-1][1] - swings[-1][0], **near)  # a move holds its last


@needs_run
@pytest.mark.parametrize("wrapped", [False, True])
def test_a_k_stride_window_is_read_by_its_own_phase(p, tmp_path, wrapped):
    """Design D5: a periodic file may hold a window of K strides with its own ``phase``; an item then counts
    windows, the stride is the entry's ``period_s`` and the phase is the file's, modulo 1.  Here: Stage 1's brisk
    stride twice over, as ``v0.4``, its phase written 0..2 or wrapped at 1."""
    d = make_t0(tmp_path, freeze=False)
    one = dict(np.load(RUN / BODY / "refs" / "brisk.npz"))
    n, stride = len(one["t"]), float(one["dt"]) * len(one["t"])
    adv = np.asarray(one["stride_motion"], float)
    two = {}
    for key, x in one.items():
        if x.ndim and len(x) == n and key not in ("q_home", "joints", "stride_motion"):
            second = np.array(x, copy=True)
            if key == "t":
                second = second + stride
            elif key == "q":
                second[:, 0:2] += adv[3:5]
            elif key in ("com", "com_ref", "zmp_ref", "zmp_mb") or key == "feet":
                second[..., 0:2] += adv[3:5]
            two[key] = np.concatenate([x, second])
        else:
            two[key] = x
    two["stride_motion"] = np.r_[adv[:3], 2.0 * adv[3:5]]
    ph = np.arange(2 * n) / n
    two["phase"] = ph % 1.0 if wrapped else ph
    two["window_strides"] = np.array(2)
    refs = s2io.read_json(d / "refs.json")
    e = refs["references"]["v0.4"]
    e.update(sha256=s2io.write_npz(d / "refs" / "v0.4.npz", two), frames=2 * n, strides=2, window_s=2.0 * stride)
    s2io.write_json(d / "refs.json", refs)
    lib = r5lib.Library(p, BODY, SET, RATE, root=tmp_path, require_frozen=False)
    s = lib.script("walk_v0.4")  # start, one window of nominal, the change, v0.4 to the end
    seg = s.track.segments[-1]
    windows = s.plan[-1][2]
    assert seg.period == pytest.approx(2.0 * stride) and seg.duration == pytest.approx(windows * 2.0 * stride)
    assert s.t_end >= 30.0 * RATE > s.t_end - 2.0 * stride
    for x in (0.0, 0.3, 0.999, 1.0, 1.5, 2.0, 2.25, 3.7, 2.0 * windows + 0.6):  # strides into the segment
        t = seg.t0 + x * stride
        off = abs(s.phase(t) - x % 1.0)
        assert off == pytest.approx(0.0, abs=1e-9) or off == pytest.approx(1.0, abs=1e-9)
        assert s.stride_s(t) == stride == e["period_s"] and s.command(t) == (e["speed_m_s"], 0.0)
        assert s.steady(t) == (x >= 1.0)
        assert s.swing_s(t) == pytest.approx((1.0 - 0.575) * stride, abs=1e-9)  # each swing's own length
    assert s.walk_progress(seg.t0, seg.t0 + 2.0 * stride) == pytest.approx(2.0 * float(np.hypot(*adv[3:5])), abs=1e-9)
    assert s.walk_progress(seg.t0 + 0.2, seg.t0 + 0.2 + stride) == pytest.approx(float(np.hypot(*adv[3:5])), abs=1e-6)
    # the second stride of the window repeats the first: the same tables one stride on
    for x in (0.1, 0.37, 0.8):
        a, b = seg.t0 + x * stride, seg.t0 + (x + 1.0) * stride
        np.testing.assert_allclose(s.foot_rel(a), s.foot_rel(b), atol=1e-9)
        assert s.support(a).tolist() == s.support(b).tolist() and s.mid_swing(a).tolist() == s.mid_swing(b).tolist()
    t, q, _ = s.start_state(seg.t0 + 1.25 * stride)
    assert np.array_equal(q, two["q"][n + n // 4]) and t == pytest.approx(seg.t0 + 1.25 * stride)


# ------------------------------------------------------------------------------------------------
# Join rule J1's leave point (the lead's ruling 21 of 17 Nov 2026)
# ------------------------------------------------------------------------------------------------
LEAVE_TASKS = ("turn_q90_left", "turn_q90_right", "stop", "start_stop")


def _the_join_into(track, move: str, p) -> dict:
    """``gates2.joins``' join of ``track`` into the move ``move`` (ruling 15's limit: each joint's jump within the
    largest frame-to-frame change of the walk it leaves)."""
    from designlab.stage2 import gates2

    js = [j for j in gates2.joins(track, len(p.joints), p.joints) if j["after"] == f"finite {move}"]
    assert len(js) == 1, js
    return js[0]


@needs_run
def test_a_look_sets_turns_and_stop_leave_the_walk_at_j1s_leave_point(p, tmp_path):
    """On a set whose moves carry ``look.window_a`` (here ``make_look``'s: a window of 3 strides that varies from
    stride to stride) the env plays the walk before a turn or the stop for ``m + a_windows`` windows (T1's 2 or 2.5
    strides are join rule J1's leave point, m = 0 whole windows) and the move starts at the frame it was built from:
    each such task's join into its move is within the walk's own largest frame-to-frame change at every joint (the
    lead's ruling 15's limit).  The plan as before (2 or 2.5 whole windows of the file) jumps far past that limit."""
    from designlab.stage2 import gates2

    make_look(tmp_path)
    lib = r5lib.Library(p, BODY, LOOK_SET, RATE, root=tmp_path)
    refs = lib.refs_doc
    T, Wn = refs[W]["period_s"], refs[W]["window_s"]
    assert refs[W]["strides"] == LOOK_K and Wn == pytest.approx(LOOK_K * T)
    names = [n for n, *_ in LOOK_MOVES]
    for task in LEAVE_TASKS:
        t = lib.tasks[task]
        assert t["available"], task
        td = next(x for x in r5lib.TASKS if x.name == task)
        i = next(k for k, it in enumerate(td.plan) if it[0] == "transition" and it[1] in names) - 1
        move = td.plan[i + 1][1]
        e = lib.tr_doc[move]["look"]
        lead = td.plan[i][2]
        assert lead == (r5lib.STOP_LEAD if "stop" in move else r5lib.TURN_LEAD[setio.parse_name(move)["direction"]])
        assert t["plan"][i] == ["ref", W, e["a_windows"]] and e["a_phase"] == lead  # (m = 0: the lead is j strides)
        assert [tuple(x) for x in t["plan"][:i]] == [tuple(x) for x in td.plan[:i]]
        (leave,) = t["leave"]
        assert leave == dict(item=i, windows=e["a_windows"], whole_windows=0, strides=LOOK_K, window_s=Wn,
                             lead_strides=lead, a_phase=e["a_phase"], a_time_s=e["a_time_s"], a_windows=e["a_windows"],
                             command_phase=lib.tr_doc[move]["move"]["command_phase"])
        s = lib.script(task)
        walk, seg = s.track.segments[i], s.track.segments[i + 1]
        assert (walk.name, seg.name) == (W, move) and seg.duration > 0
        assert seg.t0 - walk.t0 == pytest.approx(e["a_time_s"], abs=1e-9)  # the walk ends at the move's own time
        j = _the_join_into(s.track, move, p)
        assert j["ok"] and j["over"] == [] and j["walk"] == f"periodic {W}", (task, j)
        assert j["jump_rad"] < 1e-9  # (the window is uniform in time: the leave point lies on a frame)
        assert all(x["ok"] for x in gates2.joins(s.track, len(p.joints), p.joints)), task
        # the plan as it was before the ruling: the move built for a_phase strides into the window starts at the
        # window's boundary (left) or 1.5 strides into it (right): a jump past the walk's own step at most joints
        old = [tuple(x) for x in t["plan"]]
        old[i] = ("ref", W, lead)
        j_old = _the_join_into(lib.from_plan(old).track, move, p)
        assert not j_old["ok"] and j_old["jump_rad"] > 0.8 * LOOK_AMP_RAD and len(j_old["over"]) >= 6, (task, j_old)
    d = lib.describe()
    assert d["leave_rule"] == r5lib.LEAVE_RULE and d["tasks"]["stop"]["leave"] == lib.tasks["stop"]["leave"]
    assert all("leave" not in d["tasks"][k] for k in ("stand", "walk_nominal", "walk_v0.4"))


@needs_run
def test_on_the_75cm_rounds_sets_and_the_fixture_every_plan_is_as_before(t0):
    """No move of the 75 cm round's three sets carries a leave point: ``place`` returns every task's plan item for
    item there, from the committed records alone (no array is read), and on the synthetic set ``T0`` the library's
    resolved plans are the table's with "to the end" sized, its description holding no leave rule."""
    for set_name in ("V0", "F25F9", "F25F9cl"):
        d = setio.set_dir(set_name, BODY)
        refs = s2io.read_json(d / "refs.json")["references"]
        trs = s2io.read_json(d / "transitions.json")["transitions"]
        assert not any(isinstance(x, dict) and x.get("look") for x in trs.values()), set_name
        for td in r5lib.TASKS:
            plan, leaves = r5lib.place(td.plan, refs, trs)
            assert plan == [tuple(x) for x in td.plan] and leaves == {}, (set_name, td.name)
    for name, t in t0.tasks.items():
        assert "leave" not in t, name
        if t["available"]:
            td = next(x for x in r5lib.TASKS if x.name == name)
            assert t["plan"] == [list(x) for x in t0._resolve(td.plan)], name
    assert "leave_rule" not in t0.describe()
    assert t0.tasks["turn_q90_left"]["plan"][3] == ["ref", W, 2.0] and t0.tasks["stop"]["plan"][3] == ["ref", W, 2.5]


def test_a_leave_point_that_does_not_fit_is_refused():
    """``leave_point`` (``LEAVE_RULE``): None where the move carries no leave point; the windows the walk plays when
    it does; ``tune.InputError`` where the window is another walk's or of other strides or length, where a_windows is
    not a_time_s over the window, where a_phase is not at the command's foot, where the lead in strides does not
    reach it by whole windows, or where the move is not causal."""
    K, T = 12, 0.86
    Wn = K * T

    def entry(a_phase=2.0, cmd=0.0, ref=W, k=K, w=Wn, a_time=None, causal=True):
        a_time = a_phase * T if a_time is None else a_time
        return dict(move=dict(causal=causal, command_phase=cmd),
                    look=dict(window_a=dict(reference=ref, strides=k, window_s=w), a_phase=a_phase, a_time_s=a_time,
                              a_windows=a_phase * T / Wn))

    walk = dict(strides=K, window_s=Wn, period_s=T)
    assert r5lib.leave_point("x", None, W, walk, 2.0) is None
    assert r5lib.leave_point("x", dict(move=dict(causal=True, command_phase=0.0)), W, walk, 2.0) is None
    no_phase = entry()
    no_phase["look"]["a_phase"] = None
    assert r5lib.leave_point("x", no_phase, W, walk, 2.0) is None  # (a move of Stage 1's construction)
    lp = r5lib.leave_point("x", entry(), W, walk, 2.0)
    assert lp["windows"] == pytest.approx(2.0 / K) and lp["whole_windows"] == 0 and lp["strides"] == K
    assert r5lib.leave_point("x", entry(2.5, 0.5), W, walk, 2.5)["windows"] == pytest.approx(2.5 / K)
    assert r5lib.leave_point("x", entry(2.5, -0.5), W, walk, 2.5)["windows"] == pytest.approx(2.5 / K)
    lp = r5lib.leave_point("x", entry(2.0), W, walk, 2.0 + 2 * K)  # (a lead of two windows and two strides)
    assert lp["whole_windows"] == 2 and lp["windows"] == pytest.approx(2.0 + 2.0 / K)
    bad = ((entry(ref="nominal"), 2.0, "leaves the window of 'nominal'"),
           (entry(k=9), 2.0, "built from a window of 9 strides"),
           (entry(w=Wn * 1.01), 2.0, "built from a window of 12 strides"),
           (entry(a_time=2.0 * T + 0.01), 2.0, "is not a_time_s"),
           (entry(2.0, 0.5), 2.0, "is not at the command's foot"),
           (entry(2.5, 0.0), 2.5, "is not at the command's foot"),
           (entry(2.5, 0.5), 2.0, "does not reach it"),
           (entry(2.0, 0.0), 2.0 + 0.5 * K, "does not reach it"),
           (entry(causal=False), 2.0, "the move is not causal"))
    for e, lead, why in bad:
        with pytest.raises(tune.InputError, match=why):
            r5lib.leave_point("x", e, W, walk, lead)
    with pytest.raises(tune.InputError, match="missing or not numbers"):
        r5lib.leave_point("x", entry(), W, dict(strides=K), 2.0)
    assert "ruling 21 of 17 Nov 2026" in r5lib.LEAVE_RULE and "m + a_windows" in r5lib.LEAVE_RULE


#: A set root that holds the look work package's arrays of the committed look sets (read only; the records are the
#: committed ones of this checkout): without it the committed look sets are read under ``setio.ROOT``.
LOOK_ARRAYS = os.environ.get("STAGE2_LOOK_SET_ROOT")


def _committed_look_sets(tmp_path) -> Path | None:
    """A set root in ``tmp_path`` with the committed records of the look sets whose moves carry a leave point and,
    linked (never copied), the arrays of those whose walk ``v0.4`` is found with the records' sha256 (under
    ``LOOK_ARRAYS`` or ``setio.ROOT``); None when no array is found.  (The Library checks every file it reads
    against its entry's sha256.)"""
    root, found = tmp_path / "look_sets", 0
    for d in sorted(setio.ROOT.glob("V6*/*/")):
        if not (d / "transitions.json").is_file():
            continue
        trs = s2io.read_json(d / "transitions.json")["transitions"]
        if not any(isinstance(x, dict) and (x.get("look") or {}).get("a_phase") is not None for x in trs.values()):
            continue
        rel = d.relative_to(setio.ROOT)
        bases = [Path(LOOK_ARRAYS)] if LOOK_ARRAYS else []
        src = next((b / rel for b in bases + [setio.ROOT] if (b / rel / "refs" / f"{W}.npz").is_file()), None)
        want = s2io.read_json(d / "refs.json")["references"][W].get("sha256")
        if src is None or s2io.sha256_file(src / "refs" / f"{W}.npz") != want:
            continue
        (root / rel).mkdir(parents=True)
        for f in ("set.json", "refs.json", "transitions.json"):
            shutil.copyfile(d / f, root / rel / f)
        for sub in ("refs", "transitions"):
            (root / rel / sub).symlink_to((src / sub).resolve(), target_is_directory=True)
        found += 1
    return root if found else None


@needs_run
def test_the_committed_look_sets_turns_and_stop_join_within_their_walks_step(p, tmp_path):
    """Each turn and the stop that a committed look set holds as available (today V6noF5/75-M-H-10's four turns and
    its stop, V6/75-M-H-10's stop), played through the env's plan (``Library.place``, then ``_resolve``) on the set's
    own arrays: the join into the move is within the walk's own largest frame-to-frame change at every joint
    (ruling 15's limit).  The look sets' ``start`` is not available today, so the task's plan is played from its
    first walk (its start left out); the walk it leaves and the move are the task's.  Needs the look sets' arrays
    (under ``setio.ROOT``, or a set root holding them in ``STAGE2_LOOK_SET_ROOT``)."""
    root = _committed_look_sets(tmp_path)
    if root is None:
        pytest.skip("the look sets' arrays are not here (STAGE2_LOOK_SET_ROOT: a set root that holds them)")
    played = []
    for d in sorted(root.glob("*/*/")):
        set_name = d.parent.name
        lib = r5lib.Library(p, BODY, set_name, RATE, root=root, require_frozen=False)
        for td in r5lib.TASKS:
            if td.name == "start_stop":
                continue  # (its stop is the stop task's; its second start is not available)
            moves = [it[1] for it in td.plan if it[0] == "transition" and lib.why_not("transitions", it[1]) is None
                     and (lib.tr_doc[it[1]].get("look") or {}).get("a_phase") is not None]
            usable = all(lib.why_not("transitions" if it[0] == "transition" else "refs", it[1]) is None
                         for it in td.plan[1:] if it[0] != "hold")
            if not moves or not usable:
                continue
            assert td.plan[0] == ("transition", setio.START)
            plan, leaves = lib.place(td.plan[1:])
            ((i, lp),) = leaves.items()
            e = lib.tr_doc[moves[0]]["look"]
            assert plan[i] == ("ref", W, e["a_windows"]) and lp["whole_windows"] == 0
            assert lp["a_phase"] == td.plan[i + 1][2] == e["a_phase"]
            s = lib.from_plan(lib._resolve(plan))
            j = _the_join_into(s.track, moves[0], p)
            assert j["ok"], (set_name, td.name, j["over"], j["jump_rad"])
            old = list(plan)
            old[i] = ("ref", W, td.plan[i + 1][2])
            j_old = _the_join_into(lib.from_plan(lib._resolve(old)).track, moves[0], p)
            played.append((set_name, td.name, f"{j['jump_rad']:.3g}", f"before: {j_old['jump_rad']:.3g}"))
    assert played, "no turn or stop of a committed look set was played"
    print("played:", played)


# ------------------------------------------------------------------------------------------------
# Tables past the plan's end, and against the played poses
# ------------------------------------------------------------------------------------------------
def _tables(s, t: float) -> list:
    return [s.command(t), s.phase(t), s.stride_s(t), s.swing_s(t), s.walking(t), s.steady(t), s.support(t).tolist(),
            s.mid_swing(t).tolist(), s.foot_rel(t).tolist(), s.head_rp(t).tolist()]


@needs_run
def test_every_table_continues_past_the_end_as_the_track_does(s1):
    """At clock factor 1.25 an episode sized for 1.05 passes its plan's end.  A periodic last segment loops, a hold
    holds, a finite last segment holds its last frame: the tables' continuation is ``Track.joint``'s."""
    step = 0.02 * RATE * 1.25
    # (a) a plan ending in a periodic segment
    s = s1.from_plan([("transition", "start"), ("ref", "nominal", 2)])
    last = s.track.segments[-1]
    assert last.kind == "periodic" and s.steady(s.t_end + 1.0)
    for t in np.arange(s.t_end, 1.25 * s.t_end + 2.0, step):
        back = float(t) - math.ceil((t - s.t_end + 1e-9) / last.period) * last.period
        assert last.t0 <= back < s.t_end
        np.testing.assert_allclose(s.track.joint(float(t)), s.track.joint(back), atol=1e-9)
        a, b = _tables(s, float(t)), _tables(s, back)
        assert a[0] == b[0] and a[2:8] == b[2:8]
        assert a[1] == pytest.approx(b[1], abs=1e-9) or abs(a[1] - b[1]) == pytest.approx(1.0, abs=1e-9)
        np.testing.assert_allclose(a[8], b[8], atol=1e-9)
        np.testing.assert_allclose(a[9], b[9], atol=1e-9)
        assert s.heading(float(t)) == 0.0
    # (b) a plan ending in a hold, (c) one ending in a move
    for s in (s1.script("stop"), s1.script("walk")):
        kind = s.track.segments[-1].kind
        assert kind == ("hold" if s is s1.script("stop") else "finite")
        end = _tables(s, s.t_end + 1e-6)
        for t in np.arange(s.t_end + step, 1.25 * s.t_end + 2.0, step):
            assert np.array_equal(s.track.joint(float(t)), s.track.joint(s.t_end + 1e-6))
            assert _tables(s, float(t)) == end
        assert not end[4] or kind == "finite"  # the hold stands


@needs_run
def test_foot_and_head_tables_are_the_played_poses(s1):
    """Inside the plan the reward's references are the forward kinematics of the track's own poses, joins included
    (the bar's placement of the files in one world frame; the tables do not depend on the placement)."""
    bar = pytest.importorskip("designlab.stage2.bar")
    import mujoco

    p = s1.p
    s = s1.script("brisk")
    play = bar._Played(s.track, s1.dir)
    kd = mujoco.MjData(p.m)
    Rh = r5lib.head_home(p)
    joins = [a for a, _ in _segments(s)][1:]
    ts = np.concatenate([np.linspace(0.0, s.t_end - 1e-6, 801), np.array(joins) - 2e-3, np.array(joins) + 2e-3])
    for t in ts:
        kd.qpos[:] = play.q(float(t))
        mujoco.mj_kinematics(p.m, kd)
        foot = r5lib.foot_rel_of(kd.xpos, kd.xmat, p.trunk, p.feet)
        np.testing.assert_allclose(s.foot_rel(float(t)), foot, atol=2e-5)
        np.testing.assert_allclose(s.head_rp(float(t)), r5lib.head_rp_of(kd.xmat, r5lib.head_body(p), Rh), atol=2e-5)
        np.testing.assert_allclose(kd.qpos[p.qadr], s.track.joint(float(t))[: len(p.joints)], atol=1e-9)


# ------------------------------------------------------------------------------------------------
# Starts
# ------------------------------------------------------------------------------------------------
@needs_run
def test_start_states_are_the_files_frames(s1):
    s = s1.script("brisk")
    nom = np.load(RUN / BODY / "refs" / "nominal.npz")
    seg = s.track.segments[1]
    t, q, qd = s.start_state(seg.t0 + 37.4 * seg.step)
    assert t == seg.t0 + 37 * seg.step and np.array_equal(q, nom["q"][37]) and np.array_equal(qd, nom["qd"][37])
    t, q, _ = s.start_state(seg.t0)
    assert t == seg.t0 and np.array_equal(q, nom["q"][0])
    tr = np.load(RUN / BODY / "transitions" / "nominal_to_brisk.npz")
    seg = s.track.segments[2]
    t, q, qd = s.start_state(seg.t0 + 0.5 * (tr["t"][200] + tr["t"][201]))
    assert t == seg.t0 + tr["t"][200] and np.array_equal(q, tr["q"][200]) and np.array_equal(qd, tr["qd"][200])
    q[:] = 0.0  # a copy: the file's frame is untouched
    assert np.array_equal(s.start_state(t)[1], tr["q"][200])
    stop = s1.script("stop")
    with pytest.raises(ValueError, match="hold"):
        stop.start_state(stop.t_end - 1.0)


@needs_run
def test_mid_spans_leave_out_holds_and_the_stand(s1, t0):
    assert s1.script("stand").mid_spans() == [] and t0.script("stand").mid_spans() == []  # rule T3: every task but it
    for lib, name in ((s1, "walk"), (s1, "brisk"), (s1, "stop"), (t0, "start_stop"), (t0, "stop")):
        s = lib.script(name)
        spans = s.mid_spans()
        first = next(seg for seg in s.track.segments if seg.kind != "hold")
        assert spans[0][0] == first.t0 + 0.2 and spans[-1][1] <= s.t_end - 2.0 * RATE + 1e-12
        assert all(b > a for a, b in spans) and all(spans[k][1] <= spans[k + 1][0] for k in range(len(spans) - 1))
        for a, b in spans:
            for t in (a, 0.5 * (a + b), b - 1e-9):
                assert s.kind(t) != "hold"
                assert s.start_state(t)[0] <= t + 1e-9  # (a frame time is a sum: one ulp may sit either side)
    # the start-stop task's 2 s hold lies between two spans and inside none
    s = t0.script("start_stop")
    hold = next(seg for seg in s.track.segments if seg.kind == "hold")
    assert not any(a < hold.t0 + 0.5 * hold.duration < b for a, b in s.mid_spans())
    assert any(a >= hold.t0 + hold.duration for a, _ in s.mid_spans())


# ------------------------------------------------------------------------------------------------
# The required joint-range margin (decision 14), scripts from plans
# ------------------------------------------------------------------------------------------------
@needs_run
def test_required_margin_never_exceeds_what_the_reference_keeps(s1, t0):
    walk = s1.script("brisk")
    assert not walk.has_turn
    np.testing.assert_array_equal(walk.required_margin, np.minimum(tune.WALK_RANGE_RAD, walk.ref_margin))
    turn = t0.script("turn_q90_left")
    want = np.maximum(np.minimum(tune.WALK_RANGE_RAD, turn.ref_margin - tune.TURN_RANGE_SLACK_RAD), tune.AT_LIMIT_RAD)
    np.testing.assert_array_equal(turn.required_margin, want)
    for s in (walk, turn):
        assert np.all(s.required_margin <= np.maximum(s.ref_margin, tune.AT_LIMIT_RAD))
        assert np.all(s.required_margin <= tune.WALK_RANGE_RAD)


@needs_run
def test_from_plan_is_the_same_script_and_takes_a_built_track(s1):
    cond = next(c for c in panels.conditions() if c.key == "walk/brisk")
    _, docs = setio.stage1_view(RUN, BODY)
    plan = panels.Planner(RUN / BODY, docs["refs.json"]["references"],
                          docs["transitions.json"]["transitions"]).resolve(cond)["plan"]
    a = s1.from_plan(plan)
    assert s1.from_plan([list(x) for x in plan]) is a and a.plan == [tuple(x) for x in plan]
    track = control.build_track(s1.p, RUN / BODY, plan, s1.omega)
    b = s1.from_plan(plan, track=track)
    assert b.track is track and b is s1.from_plan(plan, track=track) and b is not a
    for t in np.linspace(0.0, 1.2 * a.t_end, 301):
        assert _tables(a, float(t)) == _tables(b, float(t))
    np.testing.assert_array_equal(a.track.joint(7.7), track.joint(7.7))
    with pytest.raises(ValueError, match="segments"):
        r5lib.Script(s1, plan[:-1], track=track)
    with pytest.raises(tune.InputError, match="no entry"):
        s1.from_plan([("transition", "start"), ("ref", "v0.4", 2)])


@needs_run
def test_walk_progress_is_the_stride_over_a_window(s1):
    s = s1.script("walk")
    seg = s.track.segments[1]
    ref = np.load(RUN / BODY / "refs" / "nominal.npz")
    stride = float(np.hypot(*ref["stride_motion"][3:5]))
    assert s.walk_progress(seg.t0, seg.t0 + seg.period) == pytest.approx(stride, abs=1e-12)
    assert s.walk_progress(seg.t0 + 0.3, seg.t0 + 0.3 + 3 * seg.period) == pytest.approx(3 * stride, abs=1e-9)
    assert s.walk_progress(seg.t0 - 1.0, seg.t0 + 0.4) == s.walk_progress(seg.t0, seg.t0 + 0.4) > 0.0
    assert s.walk_progress(0.0, 1.0) == 0.0 and s.walk_start(1.0) is None and s.walk_start(seg.t0 + 0.4) == seg.t0
    assert not s.steady(seg.t0 + 0.99 * seg.period) and s.steady(seg.t0 + seg.period)
