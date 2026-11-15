"""Tests for reading a Stage 2 reference set (designlab/stage2/setio.py): names, the entry subsets a key hashes,
``read_inputs`` against a freeze, the Stage 1 view, command events, the shortest planned swing, and the ignore rules
of the set and run directories.

Run from hardware/trex_form_study (the fit's venv):  python -m pytest tests/test_stage2_setio.py -q
All fast.  Synthetic sets live in pytest's temporary directory; the tests on Stage 1's own run directory need its
arrays (git-ignored; ``python tools/restore_inputs.py``) and skip without them.
"""

from __future__ import annotations

import ast
import copy
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from designlab.stage1 import control, panels, plant, tune  # noqa: E402
from designlab.stage1.refs import Reference  # noqa: E402
from designlab.stage2 import s2io, setio  # noqa: E402

BODY = "75-M-H-10"
RUN = ROOT / "stage1" / "runs" / "2026-11-08"
needs_run = pytest.mark.skipif(not (RUN / BODY / "transitions" / "stop.npz").exists(),
                               reason="the Stage 1 arrays are not restored (tools/restore_inputs.py)")


# ------------------------------------------------------------------------------------------------
# Names
# ------------------------------------------------------------------------------------------------
def all_names() -> dict:
    """Every name round A makes, with what it must parse back to."""
    walks = {setio.walk_name(None): None} | {setio.walk_name(v): v for v in setio.SPEEDS}
    out = {n: dict(kind="walk", speed_m_s=v) for n, v in walks.items()}
    out[setio.START] = dict(kind="start")
    for a in walks:
        for b in walks:
            if a != b:
                out[setio.change_name(a, b)] = dict(kind="change", a=a, b=b)
    for steps in (1, 2):
        for w in walks:
            out[setio.stop_name(steps, w)] = dict(kind="stop", steps=steps, walk=w)
    for rate, value in setio.TURN_RATES.items():
        for direction in ("left", "right"):
            out[setio.turn_name(rate, direction)] = dict(kind="turn", rate=rate, rate_rad_s=value, direction=direction,
                                                         angle_deg=90, angle_rad=math.pi / 2,
                                                         sign=1.0 if direction == "left" else -1.0)
    return out


def test_names_round_trip_exactly():
    names = all_names()
    assert len(names) == 6 + 1 + 30 + 12 + 8
    for name, want in names.items():
        got = setio.parse_name(name)
        assert got == dict(want, name=name), name
    # the values come back as the very floats that went in
    assert setio.parse_name(setio.walk_name(0.35))["speed_m_s"] == 0.35
    for v in setio.SPEEDS:
        assert setio.parse_name(setio.walk_name(v))["speed_m_s"] == v
    assert setio.parse_name(setio.walk_name(None))["speed_m_s"] is None
    for rate in setio.TURN_RATES:
        for direction in ("left", "right"):
            assert setio.parse_name(setio.turn_name(rate, direction))["rate_rad_s"] == setio.TURN_RATES[rate]
    assert setio.parse_name(setio.turn_name("pi4", "left"))["rate_rad_s"] == math.pi / 4
    assert setio.turn_name(setio.QUICK, "left") == "turn90_pi4_left" and setio.TURN_RATES[setio.QUICK] == math.pi / 4
    # the names the design writes out
    assert [setio.walk_name(v) for v in (None, *setio.SPEEDS)] == ["nominal", "v0.3", "v0.35", "v0.4", "v0.45", "v0.5"]
    assert setio.walk_name(setio.BAR_SPEED) == "v0.4" and setio.BAR_SPEED in setio.SPEEDS
    assert setio.change_name("nominal", "v0.4") == "nominal_to_v0.4"
    assert (setio.stop_name(2, "v0.4"), setio.stop_name(1, "v0.4")) == ("stop2_v0.4", "stop1_v0.4")
    assert setio.turn_name("pi4", "right") == "turn90_pi4_right" and setio.turn_name("0.35", "left") == "turn90_0.35_left"
    # a rate's label other than pi4 is its own %g text
    assert all(f"{v:g}" == k and float(k) == v for k, v in setio.TURN_RATES.items() if k != "pi4")
    assert setio.RAMP_S == 0.375 and setio.ROOT == ROOT / "stage2" / "sets"


def test_no_name_holds_a_rounded_value():
    for name in all_names():
        assert not re.search(r"\d\.\d0\b", name), name  # no fixed-decimal text such as 0.40
        assert "0.785" not in name, name  # no three-digit rounding of pi/4
    # a value its name would round has no name
    for v in (0.1234567, math.pi / 4, 1 / 3, 1e-5, 0.0, -0.4, math.nan, math.inf):
        with pytest.raises(ValueError):
            setio.walk_name(v)
    # and a name the makers did not make is not parsed: fixed decimals, Stage 1's names, near misses
    for name in ("v0.40", "v.4", "v00.4", "v4e-1", "V0.4", "v0.4 ", "brisk", "stop", "stop_v0.4", "stop3_v0.4",
                 "stop2_brisk", "stop_in_place_0.50_left", "enter_walk_0.35_left", "walk_0.50_left", "turn90_0.785_left",
                 "turn90_0.50_left", "turn90_pi4_up", "turn45_pi4_left", "turn90_pi4", "nominal_to_nominal",
                 "nominal_to_brisk", "v0.4_to_", "start_", "", None, 0.4):
        with pytest.raises(ValueError):
            setio.parse_name(name)
    for args in (("pi/4", "left"), ("0.785", "left"), (0.5, "left"), ("pi4", "up")):
        with pytest.raises(ValueError):
            setio.turn_name(*args)
    for args in ((3, "v0.4"), (True, "v0.4"), (2, "brisk")):
        with pytest.raises(ValueError):
            setio.stop_name(*args)
    with pytest.raises(ValueError):
        setio.change_name("v0.4", "v0.40")
    # no name is one by which Stage 1 registers its own stop command
    assert not [n for n in all_names() if control.TRANSITIONS.get(n, (None, None))[1] == "stand"]


# ------------------------------------------------------------------------------------------------
# Entry subsets
# ------------------------------------------------------------------------------------------------
def walk_entry(**over) -> dict:
    e = dict(name="v0.4", file="refs/v0.4.npz", sha256="a" * 64, frames=160, dt_s=0.0053788, period_s=0.860621,
             speed_m_s=0.4, stride_m=0.344248, lift_m=0.0407, cadence_steps_s=2.3239, yaw_rate_rad_s=0.0,
             stride_motion=[0.0, 0.0, 0.0, 0.344248, 0.0], ik_ok=True, set="V0", label=None,
             margin_rule="duty and lift set after results on seeds 3042-3061 (JPS 2.3)", crouch_frac=0.03, strides=1,
             window_s=0.860621, look_file="look/v0.4.npz", look_sha256="b" * 64, info=dict(accuracy="r2"),
             definition=dict(kind="walk", spec=dict(duty=0.5732)), built_by=dict(commit="c" * 40, at="2026-11-15T00:00:00Z"))
    return e | over


def move_entry(**over) -> dict:
    stand = dict(name="stand", period=math.nan, duty=math.nan, lift=0.0, speed=0.0, yaw_rate=0.0)
    e = dict(name="stop2_v0.4", kind="stop", from_="v0.4", to="stand", file="transitions/stop2_v0.4.npz",
             sha256="d" * 64, frames=560, duration_s=3.0121,
             gaits=dict(a=dict(name="v0.4", period=0.860621, duty=0.5732, lift=0.0407, speed=0.4, yaw_rate=0.0), b=stand),
             move=dict(kind="stop", builder="stop_two_step", causal=True, command_phase=-0.5, command_t_s=0.0,
                       rules=["S2A-causal: the reference is the walk up to the command"], laws="today"),
             turn=None, a1_over=False, pass_=True, ik_ok=True, zmp_iters=6,
             reference=dict(touchdowns_after_command=2), built_by=dict(commit="c" * 40), fallback=None)
    return e | over


def test_entry_sha256_ignores_labels_and_provenance():
    w = walk_entry()
    assert setio.entry_subset("walk", w) == {k: w[k] for k in setio.WALK_SUBSET}
    base = setio.entry_sha256("walk", w)
    assert base == s2io.sha256_json({k: w[k] for k in setio.WALK_SUBSET})
    for change in (dict(label="new"), dict(margin_rule="another text"), dict(built_by=dict(commit="e" * 40)),
                   dict(look_sha256="f" * 64), dict(info=dict(accuracy="r1")), dict(lift_m=0.05), dict(extra=1)):
        assert setio.entry_sha256("walk", walk_entry(**change)) == base, change
    assert setio.entry_sha256("walk", dict(reversed(list(w.items())))) == base  # nor the order of its keys
    for change in (dict(period_s=0.860622), dict(sha256="0" * 64), dict(speed_m_s=0.45), dict(strides=2),
                   dict(crouch_frac=0.05), dict(frames=161), dict(file="refs/v0.45.npz"), dict(window_s=1.7)):
        assert setio.entry_sha256("walk", walk_entry(**change)) != base, change
    # a Stage 1 entry has no strides, window_s or crouch_frac: the subset is the keys it has
    old = {k: v for k, v in w.items() if k not in ("strides", "window_s", "crouch_frac")}
    assert sorted(setio.entry_subset("walk", old)) == sorted(k for k in setio.WALK_SUBSET if k in old)
    assert setio.entry_sha256("walk", old) != base

    m = move_entry()
    sub = setio.entry_subset("move", m)
    assert sub == dict(file=m["file"], sha256=m["sha256"], frames=560, duration_s=3.0121, kind="stop",
                       gaits=dict(a=m["gaits"]["a"], b=pytest.approx(m["gaits"]["b"], nan_ok=True)),
                       move=dict(causal=True, command_phase=-0.5, kind="stop"), a1_over=False)
    base = setio.entry_sha256("move", m)  # (a stand's period is NaN: hashed all the same)
    assert all(setio.entry_sha256(k, m) == base for k in setio.MOVE_KINDS)
    for change in (dict(pass_=False), dict(built_by=None), dict(reference=None), dict(fallback=dict(rule="D21")),
                   dict(move=dict(m["move"], rules=["another rule text"], builder="x", laws="look", command_t_s=1.0))):
        assert setio.entry_sha256("move", move_entry(**change)) == base, change
    for change in (dict(move=dict(m["move"], command_phase=0.0)), dict(move=dict(m["move"], causal=False)),
                   dict(move=dict(m["move"], kind="turn")), dict(a1_over=True), dict(duration_s=3.0),
                   dict(gaits=dict(a=dict(m["gaits"]["a"], period=0.9), b=m["gaits"]["b"])),
                   dict(turn=dict(angle_rad=math.pi / 2, rate_rad_s=math.pi / 4, direction="left", stride_factor=0.8))):
        assert setio.entry_sha256("move", move_entry(**change)) != base, change
    # of a turn's block only the angle, the rate and the direction are hashed
    t1 = move_entry(turn=dict(angle_rad=math.pi / 2, rate_rad_s=math.pi / 4, direction="left", stride_factor=0.8))
    t2 = move_entry(turn=dict(angle_rad=math.pi / 2, rate_rad_s=math.pi / 4, direction="left", stride_factor=0.75,
                              kappa=1.0000001))
    assert setio.entry_sha256("move", t1) == setio.entry_sha256("move", t2)
    assert setio.entry_sha256("move", t1) != setio.entry_sha256("move", move_entry(
        turn=dict(angle_rad=math.pi / 2, rate_rad_s=math.pi / 4, direction="right", stride_factor=0.8)))
    assert setio.entry_subset("turn", t1)["turn"] == dict(angle_rad=math.pi / 2, rate_rad_s=math.pi / 4, direction="left")
    # the subset is a copy
    sub["gaits"]["a"]["period"] = 0.0
    assert m["gaits"]["a"]["period"] == 0.860621
    with pytest.raises(ValueError):
        setio.entry_subset("wlak", w)


# ------------------------------------------------------------------------------------------------
# A tiny frozen set
# ------------------------------------------------------------------------------------------------
def freeze(doc: dict, entries: dict, index=None) -> None:
    """Append a freeze of ``entries`` {name: (kind, entry)} to a set.json document, as ``sets freeze`` records it."""
    manifest = {n: dict(kind=k, sha256=e["sha256"], entry_sha256=setio.entry_sha256(k, e)) for n, (k, e) in entries.items()}
    doc["freezes"].append(dict(index=len(doc["freezes"]) if index is None else index, at="2026-11-15T00:00:00Z",
                               commit="c" * 40, manifest=manifest, manifest_sha256=s2io.sha256_json(manifest),
                               mixed_sources=False, dirty=False))


def write_set(d: Path, frozen: bool = True) -> dict:
    """A set directory with one walk and two moves (three small arrays), frozen or not; returns its documents."""
    rng = np.random.default_rng(1)
    refs, trs = {}, {}
    for sub, name, store, kind in (("refs", "v0.4", refs, "walk"), ("transitions", "start", trs, "move"),
                                   ("transitions", "stop2_v0.4", trs, "move")):
        sha = s2io.write_npz(d / sub / f"{name}.npz", dict(t=np.arange(4.0), q=rng.normal(size=(4, 3))))
        store[name] = (walk_entry if kind == "walk" else move_entry)(name=name, file=f"{sub}/{name}.npz", sha256=sha)
    set_doc = dict(set="T", body=BODY, crouch_frac=0.03, freezes=[])
    if frozen:
        freeze(set_doc, {"v0.4": ("walk", refs["v0.4"]), "start": ("start", trs["start"]),
                         "stop2_v0.4": ("stop_two_step", trs["stop2_v0.4"])})
    docs = {"set.json": set_doc, "refs.json": dict(body=BODY, set="T", references=refs),
            "transitions.json": dict(body=BODY, set="T", transitions=trs)}
    for name, doc in docs.items():
        s2io.write_json(d / name, doc)
    return docs


FILES = ["refs/v0.4.npz", "transitions/start.npz", "transitions/stop2_v0.4.npz"]


def test_read_inputs_on_a_frozen_set(tmp_path):
    d = setio.set_dir("T", BODY, root=tmp_path)
    assert d == tmp_path / "T" / BODY
    docs = write_set(d)
    inp, got = setio.read_inputs(d, FILES)
    assert list(inp) == ["dir", "files", "freeze"] and inp["dir"] == str(d)
    assert inp["freeze"] == dict(index=0, manifest_sha256=docs["set.json"]["freezes"][0]["manifest_sha256"])
    assert inp["files"] == {
        f: dict(sha256=s2io.sha256_file(d / f), entry_sha256=docs["set.json"]["freezes"][0]["manifest"][n]["entry_sha256"])
        for f, n in zip(FILES, ("v0.4", "start", "stop2_v0.4"), strict=True)}
    # the documents are the files' own, in tune.read_inputs' shape: Stage 1's Planner and Context take them
    assert list(got) == ["refs.json", "transitions.json"]
    assert got["refs.json"] == json.loads((d / "refs.json").read_text())
    assert got["refs.json"]["references"]["v0.4"]["period_s"] == 0.860621
    assert got["transitions.json"]["transitions"]["stop2_v0.4"]["move"]["command_phase"] == -0.5
    planner = panels.Planner(d, got["refs.json"]["references"], got["transitions.json"]["transitions"])
    assert planner.tr_doc["start"]["gaits"]["a"]["speed"] == 0.4
    assert setio.read_inputs(d, FILES[:1])[0]["files"].keys() == {"refs/v0.4.npz"}  # only what a run loads
    assert setio.load(d)["screen.json"] is None and setio.frozen(setio.load(d)["set.json"])["index"] == 0
    # a label, a rule text or provenance may change after the freeze: no key reads them
    edited = copy.deepcopy(docs)
    edited["refs.json"]["references"]["v0.4"].update(label="new", built_by=dict(commit="0" * 40), look_sha256="1" * 64)
    edited["transitions.json"]["transitions"]["start"]["move"]["rules"] = ["reworded"]
    for name in ("refs.json", "transitions.json"):
        s2io.write_json(d / name, edited[name])
    assert setio.read_inputs(d, FILES)[0] == inp


def test_read_inputs_refuses_a_changed_file_a_changed_subset_and_an_unfrozen_set(tmp_path):
    d = tmp_path / "T" / BODY
    docs = write_set(d)
    good = (d / "refs" / "v0.4.npz").read_bytes()

    # a changed file: its sha256 is no longer its entry's
    s2io.write_npz(d / "refs" / "v0.4.npz", dict(t=np.arange(5.0)))
    with pytest.raises(tune.InputError, match=r"refs/v0.4.npz has sha256 .* refs.json records"):
        setio.read_inputs(d, FILES)
    # ... and with the entry rewritten to match it, it is no longer the frozen one
    edited = copy.deepcopy(docs)
    edited["refs.json"]["references"]["v0.4"]["sha256"] = s2io.sha256_file(d / "refs" / "v0.4.npz")
    s2io.write_json(d / "refs.json", edited["refs.json"])
    with pytest.raises(tune.InputError) as err:
        setio.read_inputs(d, FILES)
    assert "the freeze (index 0) holds" in str(err.value) and "hashed subset" in str(err.value)
    with pytest.raises(tune.InputError):  # a frozen entry never changes, whether or not a freeze is required
        setio.read_inputs(d, FILES, require_frozen=False)
    (d / "refs" / "v0.4.npz").write_bytes(good)
    s2io.write_json(d / "refs.json", docs["refs.json"])
    assert setio.read_inputs(d, FILES)[0]["freeze"]["index"] == 0

    # a changed subset: the file is the same, a number the episode reads is not
    for record, key, name, change in (("refs.json", "references", "v0.4", dict(period_s=0.87)),
                                      ("transitions.json", "transitions", "stop2_v0.4",
                                       dict(move=dict(kind="stop", causal=True, command_phase=0.0)))):
        edited = copy.deepcopy(docs)
        edited[record][key][name].update(change)
        s2io.write_json(d / record, edited[record])
        with pytest.raises(tune.InputError, match="its entry's hashed subset is"):
            setio.read_inputs(d, FILES)
        s2io.write_json(d / record, docs[record])

    # a missing file, a file no record lists, an entry recorded as not available: all named in one error
    (d / "transitions" / "start.npz").rename(d / "transitions" / "gone.npz")
    with pytest.raises(tune.InputError) as err:
        setio.read_inputs(d, FILES + ["transitions/gone.npz"])
    assert "transitions/start.npz is missing" in str(err.value) and "transitions/gone.npz is not listed" in str(err.value)
    (d / "transitions" / "gone.npz").rename(d / "transitions" / "start.npz")
    edited = copy.deepcopy(docs)
    edited["transitions.json"]["transitions"]["stop2_v0.4"]["available"] = False
    s2io.write_json(d / "transitions.json", edited["transitions.json"])
    with pytest.raises(tune.InputError, match="not available"):
        setio.read_inputs(d, FILES)
    s2io.write_json(d / "transitions.json", docs["transitions.json"])

    # an entry added after the freeze is not frozen
    sha = s2io.write_npz(d / "refs" / "v0.45.npz", dict(t=np.arange(3.0)))
    edited = copy.deepcopy(docs)
    edited["refs.json"]["references"]["v0.45"] = walk_entry(name="v0.45", file="refs/v0.45.npz", sha256=sha, speed_m_s=0.45)
    s2io.write_json(d / "refs.json", edited["refs.json"])
    with pytest.raises(tune.InputError, match=r"refs/v0.45.npz is not in the latest freeze \(index 0\)"):
        setio.read_inputs(d, FILES + ["refs/v0.45.npz"])
    inp, _ = setio.read_inputs(d, FILES + ["refs/v0.45.npz"], require_frozen=False)
    assert inp["freeze"] is None and len(inp["files"]) == 4  # no freeze covers what was read
    assert setio.read_inputs(d, FILES)[0]["freeze"]["index"] == 0
    # a second freeze that lists it: the latest freeze is the one that counts
    set_doc = copy.deepcopy(docs["set.json"])
    freeze(set_doc, {"v0.4": ("walk", docs["refs.json"]["references"]["v0.4"]),
                     "start": ("start", docs["transitions.json"]["transitions"]["start"]),
                     "stop2_v0.4": ("stop_two_step", docs["transitions.json"]["transitions"]["stop2_v0.4"]),
                     "v0.45": ("walk", edited["refs.json"]["references"]["v0.45"])})
    s2io.write_json(d / "set.json", set_doc)
    inp, _ = setio.read_inputs(d, FILES + ["refs/v0.45.npz"])
    assert inp["freeze"] == dict(index=1, manifest_sha256=set_doc["freezes"][1]["manifest_sha256"])

    # an unfrozen set
    u = tmp_path / "U" / BODY
    write_set(u, frozen=False)
    assert setio.frozen(setio.load(u)["set.json"]) is None
    with pytest.raises(tune.InputError, match="not frozen"):
        setio.read_inputs(u, FILES)
    inp, _ = setio.read_inputs(u, FILES, require_frozen=False)
    assert inp["freeze"] is None and sorted(inp["files"]) == FILES
    (u / "set.json").unlink()
    with pytest.raises(tune.InputError, match="set.json is missing"):
        setio.read_inputs(u, FILES, require_frozen=False)


# ------------------------------------------------------------------------------------------------
# Stage 1's run directory as the pseudo-set S1
# ------------------------------------------------------------------------------------------------
@needs_run
def test_stage1_view_gives_documents_the_planner_accepts(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # a relative run directory is the study's, wherever the process stands
    inp, docs = setio.stage1_view("stage1/runs/2026-11-08", BODY)
    assert inp["dir"] == str(RUN / BODY) and inp["freeze"] is None
    assert setio.stage1_view(RUN, BODY) == (inp, docs)
    want, want_docs = tune.read_inputs(RUN / BODY, panels.panel_files())
    assert docs == want_docs and list(docs) == ["refs.json", "transitions.json"]
    assert len(inp["files"]) == 28 and set(want["files"]) <= set(inp["files"])
    assert all(inp["files"][f]["sha256"] == sha for f, sha in want["files"].items())
    refs_doc, tr_doc = docs["refs.json"]["references"], docs["transitions.json"]["transitions"]
    assert inp["files"]["refs/nominal.npz"]["entry_sha256"] == setio.entry_sha256("walk", refs_doc["nominal"])
    assert inp["files"]["transitions/stop.npz"]["entry_sha256"] == setio.entry_sha256("move", tr_doc["stop"])
    assert len({v["entry_sha256"] for v in inp["files"].values()}) == 28
    # Stage 1's planner and context take the documents as they are
    planner = panels.Planner(RUN / BODY, refs_doc, tr_doc)
    conds = {c.key: c for c in panels.conditions()}
    res = planner.resolve(conds["walk/brisk"])
    assert res["plan"][0] == ("transition", "start") and res["seconds"] > 10.0
    assert planner.resolve(conds["start_stop/twice"])["starts_ref_s"]
    ctx = panels.Context(BODY, RUN / BODY, docs)
    assert ctx.refs_doc is refs_doc and ctx.planner.period("nominal") == refs_doc["nominal"]["period_s"]
    # a file that differs from its entry is refused
    with pytest.raises(tune.InputError, match="refs.json is missing"):
        setio.stage1_view(tmp_path, BODY)


@needs_run
def test_stage1_view_refuses_a_changed_file(tmp_path):
    import shutil

    d = tmp_path / "run" / BODY
    for sub in ("refs", "transitions"):
        shutil.copytree(RUN / BODY / sub, d / sub)
        shutil.copy(RUN / BODY / f"{sub}.json", d / f"{sub}.json")
    assert setio.stage1_view(tmp_path / "run", BODY)[0]["files"] == setio.stage1_view(RUN, BODY)[0]["files"]
    s2io.write_npz(d / "refs" / "brisk.npz", dict(t=np.arange(3.0)))
    (d / "transitions" / "stop.npz").unlink()
    with pytest.raises(tune.InputError) as err:
        setio.stage1_view(tmp_path / "run", BODY)
    assert "refs/brisk.npz has sha256" in str(err.value) and "transitions/stop.npz is missing" in str(err.value)


# ------------------------------------------------------------------------------------------------
# Commands
# ------------------------------------------------------------------------------------------------
PLAN = [("transition", "start"), ("ref", "nominal", 1), ("transition", "nominal_to_v0.4"), ("ref", "v0.4", 2.5),
        ("transition", "turn90_pi4_right"), ("ref", "v0.4", 3), ("transition", "stop2_v0.4"), ("hold", 2.0)]


def plan_docs() -> dict:
    walk = dict(name="v0.4", period=0.860621, duty=0.5732, speed=0.4)
    return {"start": move_entry(name="start", kind="start", move=dict(kind="start", causal=False, command_phase=None)),
            "nominal_to_v0.4": {k: v for k, v in move_entry(name="nominal_to_v0.4", kind="walk").items() if k != "move"},
            "turn90_pi4_right": move_entry(name="turn90_pi4_right", kind="walk", gaits=dict(a=walk, b=walk),
                                           move=dict(kind="turn", causal=True, command_phase=0.5)),
            "stop2_v0.4": move_entry()}


def fake_track(plan, starts):
    segs = [SimpleNamespace(kind={"ref": "periodic", "transition": "finite", "hold": "hold"}[item[0]],
                            name=item[1] if item[0] != "hold" else "stand", t0=t0)
            for item, t0 in zip(plan, starts, strict=True)]
    return SimpleNamespace(segments=segs, events=[])


def test_add_events_appends_one_event_per_causal_move():
    starts = [0.0, 5.678, 6.814, 12.7, 14.851525, 19.2, 21.781863, 24.8]
    tr_doc = plan_docs()
    cmds = setio.commands(PLAN, tr_doc)
    assert cmds == [dict(index=4, name="turn90_pi4_right", kind="turn", command_phase=0.5, speed_m_s=0.4, period_s=0.860621),
                    dict(index=6, name="stop2_v0.4", kind="stop", command_phase=-0.5, speed_m_s=0.4, period_s=0.860621)]
    track = fake_track(PLAN, starts)
    events = setio.add_events(track, PLAN, tr_doc)
    assert events == track.events == [
        dict(kind="turn", name="turn90_pi4_right", t_ref=14.851525, command_phase=0.5, speed_m_s=0.4, period_s=0.860621),
        dict(kind="stop", name="stop2_v0.4", t_ref=21.781863, command_phase=-0.5, speed_m_s=0.4, period_s=0.860621)]
    # the command is the move's first frame: its segment's t0, exactly
    assert [e["t_ref"] for e in events] == [track.segments[4].t0, track.segments[6].t0]
    # control.run reads the events of kind "stop" (decision 18's reading): one, the bar's own
    assert [e["name"] for e in track.events if e["kind"] == "stop"] == ["stop2_v0.4"]
    # a second call on the same track is refused, and so is a track that is not the plan's
    with pytest.raises(ValueError, match="already holds"):
        setio.add_events(track, PLAN, tr_doc)
    assert len(track.events) == 2
    with pytest.raises(ValueError, match="segments"):
        setio.add_events(fake_track(PLAN[:-1], starts[:-1]), PLAN, tr_doc)
    with pytest.raises(ValueError, match="the plan has the transition"):
        setio.add_events(fake_track(PLAN[:4] + [("transition", "other")] + PLAN[5:], starts), PLAN, tr_doc)
    # a plan with no causal move adds nothing; a move the record does not hold is an input error
    quiet = fake_track(PLAN[:4], starts[:4])
    assert setio.add_events(quiet, PLAN[:4], tr_doc) == [] and quiet.events == []
    with pytest.raises(tune.InputError, match="not in transitions.json"):
        setio.commands(PLAN, {k: v for k, v in tr_doc.items() if k != "stop2_v0.4"})
    # a causal move may not carry a name by which Stage 1 registers its own stop command
    for name in ("stop", "stop_in_place_0.50_left"):
        with pytest.raises(tune.InputError, match="Stage 1 registers"):
            setio.commands([("transition", name)], {name: move_entry(name=name)})


@needs_run
def test_events_and_planned_swings_on_a_real_track():
    """On a track of Stage 1's own files: the event has Stage 1's shape and the segment's t0; a Stage 1 plan gets
    no event of ours; the shortest planned swing is the brisk walk's, as ``Track.info`` plays it."""
    run = RUN / BODY
    _, docs = setio.stage1_view(RUN, BODY)
    tr_doc = docs["transitions.json"]["transitions"]
    p = plant.build(BODY, "hard")
    omega = control.omega0(p, Reference.load(run / "refs" / "nominal.npz").q_home)
    plan = [("transition", "start"), ("ref", "nominal", 2), ("transition", "nominal_to_brisk"), ("ref", "brisk", 2.5),
            ("transition", "brisk_to_nominal"), ("ref", "nominal", 1), ("transition", "stop"), ("hold", 1.0)]
    track = setio.build_track(p, run, plan, omega, tr_doc)
    assert [e["name"] for e in track.events] == ["stop"]  # Stage 1's own stop event only: its entries have no move block
    stage1_event = track.events[0]
    causal = copy.deepcopy(tr_doc)
    causal["nominal_to_brisk"]["move"] = dict(kind="turn", causal=True, command_phase=0.5)
    track = setio.build_track(p, "stage1/runs/2026-11-08/" + BODY, plan, omega, causal)
    ours = track.events[1]
    assert list(ours) == list(stage1_event)  # the same keys in the same order
    assert ours == dict(kind="turn", name="nominal_to_brisk", t_ref=track.segments[2].t0, command_phase=0.5,
                        speed_m_s=tr_doc["nominal_to_brisk"]["gaits"]["a"]["speed"],
                        period_s=tr_doc["nominal_to_brisk"]["gaits"]["a"]["period"])
    assert ours["t_ref"] == pytest.approx(5.678 + 2 * docs["refs.json"]["references"]["nominal"]["period_s"], abs=1e-3)

    swings = setio.planned_swings(track)
    brisk = Reference.load(run / "refs" / "brisk.npz")
    shortest = int(np.sum(~brisk.support[:, 0])) * brisk.step  # the brisk walk's swing frames at its frame interval
    assert setio.min_planned_swing_s(track) == pytest.approx(shortest, abs=1e-9)
    assert min(b - a for _, a, b in swings) == setio.min_planned_swing_s(track)
    assert {f for f, _, _ in swings} == {0, 1} and all(b > a for _, a, b in swings)
    # the same runs as the track gives a recorder: its swing foot sampled every half millisecond
    ts = np.arange(0.0, track.t_end, 0.0005)
    foot = np.array([track.info(t)[1]["swing"] for t in ts])
    change = np.flatnonzero(foot[1:] != foot[:-1]) + 1
    runs = [(int(foot[a]), ts[a], ts[b]) for a, b in zip(np.r_[0, change[:-1]], change, strict=True) if foot[a] >= 0]
    assert len(runs) == len(swings)
    for (f1, a1, b1), (f2, a2, b2) in zip(runs, swings, strict=True):
        assert f1 == f2 and abs(a1 - a2) <= 0.0005 + 1e-9 and abs(b1 - b2) <= 0.0005 + 1e-9


def test_min_planned_swing_s_on_synthetic_segments():
    """Frames last from their own time to the next frame's; runs join across segments; a run the track's end cuts is
    left out; a track with no swing has no shortest one."""
    step = 0.1
    walk = np.array([0, 0, 0, -1, -1, 1, 1, 1, 1, -1])  # left 0.3 s, right 0.4 s per stride of 1 s
    periodic = SimpleNamespace(kind="periodic", name="w", t0=0.0, duration=2.5, step=step, n=10, info=dict(swing=walk))
    finite = SimpleNamespace(kind="finite", name="m", t0=2.5, duration=1.0, t=np.array([0.0, 0.25, 0.5, 0.6, 1.0]),
                             info=dict(swing=np.array([1, 1, -1, 0])))  # the right swing goes on for 0.5 s; left cut
    track = SimpleNamespace(segments=[periodic, finite], t_end=3.5)
    assert setio.planned_swings(track) == [
        (0, pytest.approx(0.0), pytest.approx(0.3)), (1, pytest.approx(0.5), pytest.approx(0.9)),
        (0, pytest.approx(1.0), pytest.approx(1.3)), (1, pytest.approx(1.5), pytest.approx(1.9)),
        (0, pytest.approx(2.0), pytest.approx(2.3)), (1, pytest.approx(2.5), pytest.approx(3.0))]
    assert setio.min_planned_swing_s(track) == pytest.approx(0.3)
    hold = SimpleNamespace(kind="hold", name="stand", t0=3.5, duration=2.0, info=dict(swing=np.array([-1])))
    track = SimpleNamespace(segments=[periodic, finite, hold], t_end=5.5)
    assert setio.planned_swings(track)[-1] == (0, pytest.approx(3.1), pytest.approx(3.5))  # now it ends: complete
    # a periodic segment that stops inside a swing, followed by a move that finishes it: one swing of 0.2 + 0.05 s
    short = SimpleNamespace(kind="periodic", name="w", t0=0.0, duration=0.2, step=step, n=10, info=dict(swing=walk))
    tail = SimpleNamespace(kind="finite", name="m", t0=0.2, duration=0.4, t=np.array([0.0, 0.05, 0.4]),
                           info=dict(swing=np.array([0, -1])))
    assert setio.min_planned_swing_s(SimpleNamespace(segments=[short, tail])) == pytest.approx(0.25)
    with pytest.raises(s2io.S2Error, match="no complete swing"):
        setio.min_planned_swing_s(SimpleNamespace(segments=[hold]))


# ------------------------------------------------------------------------------------------------
# The module's imports and the directories' ignore rules
# ------------------------------------------------------------------------------------------------
def test_setio_imports_stage1_control_tune_and_s2io_only():
    tree = ast.parse((ROOT / "designlab" / "stage2" / "setio.py").read_text())
    local = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level:
            local += [(node.level, node.module, a.name) for a in node.names]
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module]
            assert not [n for n in names if n and n.split(".")[0] == "designlab"], names
    assert sorted(local) == [(1, None, "s2io"), (2, "stage1", "control"), (2, "stage1", "tune")]


IGNORE_LINES = ["*.npz", "*.tmp", ".lock", "partial/", "traces/", "logs/", "done/"]


def test_set_and_run_directories_ignore_what_is_not_a_record():
    for sub in ("sets", "runs"):
        assert (ROOT / "stage2" / sub / ".gitignore").read_text().splitlines() == IGNORE_LINES
    if subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--git-dir"], capture_output=True).returncode:
        pytest.skip("not a git checkout")

    def ignored(rel: str) -> bool:
        done = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "-q", rel], capture_output=True)
        assert done.returncode in (0, 1), done.stderr
        return done.returncode == 0

    for rel in ("stage2/sets/x/y/refs/a.npz", "stage2/sets/x/y/refs/a.npz.tmp", "stage2/sets/x/y/refs.json.tmp",
                "stage2/sets/x/y/partial/a.json", "stage2/sets/x/y/.lock", "stage2/sets/x/y/transitions/a.npz",
                "stage2/sets/x/y/look/a.npz", "stage2/sets/x/y/logs/a.log", "stage2/sets/x/y/done/a",
                "stage2/runs/r/x/traces/a.npz", "stage2/runs/r/x/eval/traces/R0/walk/a/3042.npz",
                "stage2/runs/r/x/logs/a.log", "stage2/runs/r/x/done/a", "stage2/runs/r/x/.lock",
                "stage2/runs/r/x/partial/k/3042.json", "stage2/runs/r/budget.json.tmp"):
        assert ignored(rel), rel
    for rel in ("stage2/sets/x/y/refs.json", "stage2/sets/x/y/transitions.json", "stage2/sets/x/y/set.json",
                "stage2/sets/x/y/screen.json", "stage2/sets/x/y/SCREEN.md", "stage2/sets/legacy_check.json",
                "stage2/runs/r/x/seeds3042/eval/R0@10ms/walk/v0.4.json", "stage2/runs/r/budget.json",
                "stage2/runs/r/run_75cm.sh", "stage2/sets/.gitignore", "stage2/runs/.gitignore"):
        assert not ignored(rel), rel
