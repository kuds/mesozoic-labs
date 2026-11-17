"""Tests of ``designlab/stage2/sets.py``: the sets' definitions, and building, verifying and freezing a set.

Three kinds (the round-A design's section 4.3):

1. **Definitions** (fast, no model): the three sets' specs and labels, the pull request's function field by field,
   the items of a set, their names and their identities.
2. **The machinery on a tiny synthetic set** (fast): ``sets.produce`` is replaced by a stand-in that writes small
   arrays, so that ``build``, the partials, the merge, the lock, the kill points, ``verify`` and ``freeze`` run in
   seconds on set ``F25F9`` (which has no twin in Stage 1's run or in the look ladder) under a temporary root.  The
   fallback rule and the turn rule are run on ``sets._Job`` with its ``move`` scripted.
3. **The built V0 set** (marked ``built``: skipped without ``stage2/sets/V0/75-M-H-10``), and the process-state
   test (``STAGE2_SLOW=1``: walks and a move generated in subprocesses).

Run from hardware/trex_form_study:  python -m pytest -p no:cacheprovider tests/test_stage2_sets.py
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, fields, replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import designlab  # noqa: E402,F401  (selects the GL backend before mujoco is imported)
from designlab import envelope, gait, sizing, stage1  # noqa: E402
from designlab.stage1 import plant, tune  # noqa: E402
from designlab.stage2 import look, moves, s2io, setio, sets, setscreen  # noqa: E402

BODY = "75-M-H-10"
BODY60 = "60-N-T-10"
SET = "F25F9"  # the synthetic tests' set: no walk of it is Stage 1's or the look ladder's
V0_SET = setio.ROOT / "V0" / BODY
RUN = ROOT / "stage1" / "runs" / "2026-11-08" / BODY
PY = sys.executable

slow = pytest.mark.skipif(not os.environ.get("STAGE2_SLOW"), reason="walks and a move generated; set STAGE2_SLOW=1")
needs_run = pytest.mark.skipif(not (RUN / "transitions" / "start.npz").exists(),
                               reason="the Stage 1 arrays are not here (tools/restore_inputs.py)")
built = pytest.mark.skipif(not (V0_SET / "set.json").exists(), reason="the V0 set is not built (stage2/sets/V0)")


def arrays_here(d: Path, docs: dict) -> bool:
    """Whether a built set's git-ignored arrays are in place: a fresh clone holds its committed records only, and
    ``sets verify`` reads the arrays."""
    entries = list(docs["refs.json"]["references"].values()) + list(docs["transitions.json"]["transitions"].values())
    return all((d / e["file"]).is_file() for e in entries if e.get("available") is not False)
has_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


@pytest.fixture(autouse=True)
def one_thread(monkeypatch):
    for v in s2io.THREAD_VARS:
        monkeypatch.setenv(v, "1")


def _keys(x, found=None) -> set:
    found = set() if found is None else found
    if isinstance(x, dict):
        for k, v in x.items():
            found.add(k)
            _keys(v, found)
    elif isinstance(x, list):
        for v in x:
            _keys(v, found)
    return found


def no_wall_time(x) -> bool:
    return not {k for k in _keys(x) if k == "seconds" or k.endswith("_seconds") or k in ("wall_s", "elapsed_s")}


# ------------------------------------------------------------------------------------------------
# 1. Definitions
# ------------------------------------------------------------------------------------------------
def test_the_three_sets():
    assert list(sets.SETS) == ["V0", "F25F9", "F25F9cl"]
    assert {n: s.role for n, s in sets.SETS.items()} == dict(V0="control", F25F9="control", F25F9cl="candidate")
    assert all(s.name == n for n, s in sets.SETS.items())
    assert sets.SETS["F25F9cl"].margin_rule == "duty and lift set after results on seeds 3042-3061 (JPS 2.3)"
    assert sets.SETS["V0"].margin_rule is None and sets.SETS["F25F9"].margin_rule is None
    assert (sets.CL_DUTY, sets.CL_LIFT, sets.CL_RUNGS) == (0.59, 0.095, (0.40, 0.45, 0.50))
    assert sets.walk_names() == ["nominal", "v0.3", "v0.35", "v0.4", "v0.45", "v0.5"]
    with pytest.raises(sets.SetError, match="not a reference set"):
        sets.items("V7", BODY)
    with pytest.raises(sets.SetError, match="not a registered body"):
        sets.items("V0", "90-X")


@pytest.mark.parametrize("body", [BODY, BODY60])
def test_v0_is_todays_walk_at_every_speed(body):
    """``RULES["S2S-speeds"]``: NOMINAL at the nominal speed; ``look.speed_spec`` between two of Stage 1's walks;
    above FAST's speed FAST's stride, duty and lift at the higher Froude number, labelled new."""
    H, _ = sets.body_facts(body)
    brisk, fast = sizing.BRISK.describe(H)["speed"], envelope.FAST.describe(H)["speed"]
    assert sets.v0_spec(body, H) is sizing.NOMINAL and sets.v0_spec(body, H, None) is sizing.NOMINAL
    assert sets.nominal_speed(H) == pytest.approx(math.sqrt(0.012 * gait.G * H))
    assert sets.v0_spec(body, H, 0.40) == look.speed_spec(H)  # the look ladder's 0.40 m/s rung
    st = sets.SETS["V0"]
    for v in setio.SPEEDS:
        spec = st.spec_of(body, H, v)
        assert spec.describe(H)["speed"] == pytest.approx(v, rel=1e-12)
        if v < brisk:
            assert spec == look.speed_spec(H, v, sizing.NOMINAL, sizing.BRISK) and st.label_of(body, H, v) is None
        elif v <= fast:
            assert spec == look.speed_spec(H, v, sizing.BRISK, envelope.FAST) and st.label_of(body, H, v) is None
        else:
            assert spec == replace(envelope.FAST, froude=v * v / (gait.G * H)) and st.label_of(body, H, v) == "new"
            assert (spec.stride_rel, spec.duty, spec.lift_frac) == (envelope.FAST.stride_rel, envelope.FAST.duty,
                                                                    envelope.FAST.lift_frac)
    assert st.label_of(body, H, None) is None
    # on the 75 cm pick only 0.50 m/s is above FAST's speed (0.4961 m/s)
    if body == BODY:
        assert [st.label_of(body, H, v) for v in setio.SPEEDS] == [None, None, None, None, "new"]


def _pr_function():
    spec = importlib.util.spec_from_file_location("jetson_plem_study_under_test", sets.PR_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._tuned_f9_spec


def test_tuned_f9_spec_is_the_pull_requests_function():
    """The copy against the pull request's script, imported by path: field by field, at every rung, on both picks;
    the candidate is that spec with duty 0.59 and lift 0.095 H."""
    pr = _pr_function()
    names = [f.name for f in fields(gait.GaitSpec)]
    assert len(names) == 38
    for body in (BODY, BODY60):
        H, _ = sets.body_facts(body)
        for v in (sets.nominal_speed(H), *setio.SPEEDS):
            ours, theirs = sets.tuned_f9_spec(body, H, v), pr(body, H, v)
            for k in names:
                assert getattr(ours, k) == getattr(theirs, k), (body, v, k)
            assert type(ours) is gait.GaitSpec
        assert sets.pr_differences(body, H) == []
        for v in (None, *setio.SPEEDS):
            f9 = sets.SETS["F25F9"].spec_of(body, H, v)
            assert f9 == pr(body, H, sets.nominal_speed(H) if v is None else v)
            assert sets.SETS["F25F9cl"].spec_of(body, H, v) == replace(f9, duty=0.59, lift_frac=0.095)
        assert sets.SETS["F25F9"].label_of(body, H, 0.3) is None
    # the crouch is the spec's, per body: 0.05 H at 75 cm and 0.07 H at 60 cm
    assert sets.tuned_f9_spec(BODY, 0.5, 0.4).crouch == 0.05 and sets.tuned_f9_spec(BODY60, 0.4, 0.4).crouch == 0.07
    # the candidate's rungs other than 0.40, 0.45 and 0.50 m/s are labelled new
    lab = sets.SETS["F25F9cl"].label_of
    assert [lab(BODY, 0.5, v) for v in (None, *setio.SPEEDS)] == ["new", "new", "new", None, None, None]


def test_pr_differences_names_a_field_that_moved(monkeypatch):
    real = sets.tuned_f9_spec
    monkeypatch.setattr(sets, "tuned_f9_spec", lambda body, H, v: replace(real(body, H, v), lift_frac=0.082))
    diff = sets.pr_differences(BODY, 0.5)
    assert len(diff) == 6 and all("lift_frac: 0.082 here, 0.081 in the pull request's function" in x for x in diff)


def test_importing_the_pull_requests_script_leaves_the_environment(monkeypatch):
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    assert sets.pr_function()(BODY, 0.5, 0.4).froude == pytest.approx(0.16 / (gait.G * 0.5))
    assert "MUJOCO_GL" not in os.environ and os.environ["OMP_NUM_THREADS"] == "1"


def test_one_crouch_per_set_and_body(monkeypatch):
    for name in sets.SETS:
        for body in (BODY, BODY60):
            specs = sets.walk_specs(name, body)
            assert len({s.crouch for s in specs.values()}) == 1
    two = sets.SetDef("two", "control", lambda body, H, v: replace(sizing.NOMINAL, crouch=0.03 if v is None else 0.05),
                      lambda body, H, v: None, "two crouches")
    monkeypatch.setitem(sets.SETS, "two", two)
    with pytest.raises(sets.SetError, match="2 crouches"):
        sets.items("two", BODY)


@pytest.mark.parametrize("set_name", list(sets.SETS))
@pytest.mark.parametrize("body", list(stage1.BODIES))
def test_items_of_every_set_on_every_registered_body(set_name, body):
    """22 items in build order, the same for every set and body: nothing is hard-wired to a body or a set."""
    its = sets.items(set_name, body)
    H, model = sets.body_facts(body)
    assert model == plant.record(body)["xml_sha256"] and H == float(plant.record(body)["H"])
    assert [i.name for i in its] == [
        "nominal", "v0.3", "v0.35", "v0.4", "v0.45", "v0.5", "start", "nominal_to_v0.3", "nominal_to_v0.35",
        "nominal_to_v0.4", "nominal_to_v0.45", "nominal_to_v0.5", "stop2_v0.4", "stop1_v0.4",
        "turn90_0.35_left", "turn90_0.35_right", "turn90_0.5_left", "turn90_0.5_right", "turn90_pi4_left",
        "turn90_pi4_right", "turn90_1_left", "turn90_1_right"]
    assert [i.kind for i in its] == ["walk"] * 6 + ["start"] + ["change"] * 5 + ["stop_two_step", "stop_one_step"] + ["turn"] * 8
    assert set(i.kind for i in its) == set(sets.KINDS)
    by = {i.name: i for i in its}
    # the start reads the nominal walk, or the 0.40 m/s walk where the nominal walk is no rung (S2S-lead-in)
    assert all(i.depends == () for i in its[:6]) and by["start"].depends == ("nominal", "v0.4")
    lead = by["start"].definition["lead_in"]
    assert (lead["rule"], lead["name"]) == ("S2S-lead-in", "start_v0.4") == ("S2S-lead-in", setio.start_name("v0.4"))
    assert lead["start"]["builder"] == "start" and lead["start"]["specs"]["b"]["name"] == "v0.4"
    assert lead["start"]["specs"]["a"]["spec"] is None and by["start"].definition["specs"]["b"]["name"] == "nominal"
    assert "start_v0.4" not in by  # no item of its own: the start item makes it where the set needs it
    assert by["nominal_to_v0.4"].depends == ("nominal", "v0.4")
    assert by["nominal_to_v0.5"].depends == ("nominal", "v0.5", "v0.45")  # its fallback's neighbour, toward 0.40 m/s
    assert by["nominal_to_v0.3"].depends == ("nominal", "v0.3", "v0.35")
    assert by["stop2_v0.4"].depends == by["stop1_v0.4"].depends == by["turn90_1_right"].depends == ("v0.4",)
    groups = sets.groups(its)
    assert len(groups) == 18 and [len(v) for v in groups.values()] == [1] * 14 + [2] * 4
    assert list(groups)[14:] == ["turn90_0.35", "turn90_0.5", "turn90_pi4", "turn90_1"]
    specs = sets.walk_specs(set_name, body)
    for i in its:
        json.dumps(i.definition, allow_nan=False)  # JSON-plain, no function object, no NaN
        assert i.definition["kind"] == i.kind and i.definition["body_model_sha256"] == model
        if i.kind == "walk":
            assert i.definition == dict(kind="walk", spec=json.loads(json.dumps(look._plain(asdict(specs[i.name])))),
                                        body_model_sha256=model, generator=sets.WALK_GENERATOR)
            assert len(i.definition["spec"]) == 38
        else:
            assert i.definition["laws"] == "today" and "builder" in i.definition and "specs" in i.definition
    assert by["start"].definition["builder"] == "start" and by["start"].definition["causal"] is False
    assert by["stop2_v0.4"].definition["causal"] is True and by["stop1_v0.4"].definition["causal"] is False
    assert "comparison" in by["stop1_v0.4"].definition["label"]
    assert by["nominal_to_v0.4"].definition["fallback"] == dict(rule="S2S-fallback", steps=[dict(step=1, ramp_strides=2.0)])
    assert by["nominal_to_v0.45"].definition["fallback"]["steps"][1] == dict(step=2, change="v0.4_to_v0.45", ramp_strides=1.0)
    assert by["nominal_to_v0.3"].definition["fallback"]["steps"][1]["change"] == "v0.35_to_v0.3"
    t = by["turn90_pi4_right"].definition
    assert t["args"] == dict(angle=math.pi / 2, rate=-math.pi / 4, ramp_s=0.375) and t["causal"] is True
    assert t["stride_rule"] == dict(grid=[1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6], limit_kinematic=0.85,
                                    limit_final=0.87, limit_other_joints_rad=0.05, steps_down_max=2)
    assert t["stride_rule"]["limit_other_joints_rad"] == setscreen.RANGE_RULE_MOVE_RAD  # A12's limit of a move
    assert "S2A-appendage" in t["rules"] and "S2A-stride" in t["rules"]
    assert by["turn90_pi4_left"].definition["args"]["rate"] == math.pi / 4


def test_neighbours_lead_toward_the_bars_speed():
    assert [sets.neighbour(v) for v in setio.SPEEDS] == [0.35, 0.40, None, 0.40, 0.45]


def test_every_name_parses_back_to_its_definition_exactly():
    """Names are ``setio``'s, numbers with ``%g``: no name holds a rounded value or a fixed-decimal format."""
    for set_name in sets.SETS:
        H, _ = sets.body_facts(BODY)
        for i in sets.items(set_name, BODY):
            p = setio.parse_name(i.name)
            assert not re.search(r"\d\.\d*0(_|$)", i.name), i.name  # no trailing zero: %g, never %.2f
            if i.kind == "walk":
                v = p["speed_m_s"]
                assert setio.walk_name(v) == i.name
                speed = math.sqrt(i.definition["spec"]["froude"] * gait.G * H)
                assert speed == pytest.approx(sets.nominal_speed(H) if v is None else v, rel=1e-12)
            elif i.kind == "turn":
                assert i.definition["args"]["rate"] == p["sign"] * p["rate_rad_s"] == p["sign"] * setio.TURN_RATES[p["rate"]]
                assert i.definition["args"]["angle"] == p["angle_rad"] == math.pi / 2
                assert setio.turn_name(p["rate"], p["direction"]) == i.name
            elif i.kind == "change":
                assert (i.definition["specs"]["a"]["name"], i.definition["specs"]["b"]["name"]) == (p["a"], p["b"])
            elif i.kind in ("stop_two_step", "stop_one_step"):
                assert p["steps"] == (2 if i.kind == "stop_two_step" else 1) and p["walk"] == "v0.4"
    assert setio.parse_name("turn90_pi4_left")["rate_rad_s"] == math.pi / 4
    assert f"{0.40:g}" == "0.4" and setio.walk_name(0.35) == "v0.35" and f"{sets.CL_DUTY:g}" == "0.59"


def test_the_rules_say_what_the_plan_says():
    r = sets.RULES
    assert set(r) == {"S2S-sets", "S2S-speeds", "S2S-margin", "S2S-crouch", "S2S-process", "S2S-formats", "S2S-twins",
                      "S2S-stage0", "S2S-fallback", "S2S-lead-in", "S2S-turns", "S2S-not-available", "S2S-freeze",
                      "S2S-omega"}
    for text in ("lead_in 'start'", "start_v0.4, built and gated like start, with no fallback", "lead_in 'start_v0.4'",
                 "a change from the neighbouring rung toward 0.40 m/s", "A start_v0.4 that fails a gate stops the set",
                 "lead_in null", "STAGE2_75_PLAN.md A8 as amended", "DECISION_LOG.md P34", "changed after a result"):
        assert text in r["S2S-lead-in"], text
    assert "a failed start_v0.4" in r["S2S-fallback"]
    for text in ("(1) a change with a ramp of 2 strides instead of 1", "v0.4_to_v0.45, v0.45_to_v0.5, v0.4_to_v0.35, "
                 "v0.35_to_v0.3", "no transition within the gates", "A start and the stops have no fallback",
                 "nominal_to_v0.4 has step (1) only", "stops the set and goes to the lead"):
        assert text in r["S2S-fallback"], text
    assert sets.NO_HEADING == "no reference within the heading rule" and sets.NO_HEADING in r["S2S-turns"]
    assert sets.NO_TWIST == "no reference within the thigh twist and the leg drives" and sets.NO_TWIST in r["S2S-turns"]
    assert sets.NO_GATES == "no transition within the gates"
    assert "never drops or changes one" in r["S2S-freeze"] and "No ramp, grid, gate or constant is changed" in r["S2S-not-available"]
    for text in ("every entry a build decided built by that moves.py (A4; no flag waives it)", "a tree git cannot read counts as dirty",
                 "change_after_result with its reason", "after_check_episodes true with reason", "before_any_episode",
                 "refcheck.json", "no rule here makes it block a freeze"):
        assert text in r["S2S-freeze"], text
    assert "SetDef.stage1_twins, ladder_twins" in r["S2S-twins"] and "also names the entry gates" in r["S2S-turns"]
    assert sets.SETS["V0"].stage1_twins == ("nominal", "start") and sets.SETS["V0"].ladder_twins == ("nominal", "v0.4")
    assert all(sets.SETS[n].stage1_twins == sets.SETS[n].ladder_twins == () for n in ("F25F9", "F25F9cl"))
    assert sets.HEADING_GATES == ("heading_total", "heading_at_bar", "heading_at_slowest_clock")
    src = (ROOT / "designlab" / "stage2" / "sets.py").read_text()
    code = src.split('"""', 2)[2]  # past the module docstring, whose examples name a body
    assert BODY not in code and BODY60 not in code  # nothing is hard-wired to a body
    assert ".2f" not in code and ".3f" not in code  # numbers in names and records with %g


def test_cli_skeleton_sets_one_thread_before_numpy_and_runs_the_named_module():
    src = (ROOT / "designlab" / "stage2" / "sets.py").read_text()
    head = src.index('if __name__ == "__main__":')
    assert head < src.index("import numpy") and 'os.environ.setdefault(_v, "1")' in src[head:src.index("import numpy")]
    assert src.rstrip().endswith("sys.exit(_named.main())") and "from designlab.stage2 import sets as _named" in src
    assert "np.savez(" not in src  # arrays go through s2io.write_npz (atomic, a file object)


# ------------------------------------------------------------------------------------------------
# 2. A tiny synthetic set: sets.produce replaced by a stand-in
# ------------------------------------------------------------------------------------------------
#: The stand-in's module (also written to a file for the subprocesses of the kill-point test): ``produce`` makes
#: small arrays that depend on the item's name and ``MODE["salt"]`` only, with entries of the builder's shape.
FAKE_CODE = r'''
import math, shutil, zlib
import numpy as np
from designlab.stage2 import moves, s2io, setio, sets

MODE = dict(salt="", fail={}, blocks_set=None, calls=[], sources=None)


def built_by():
    return dict(commit="0" * 40, dirty=False, dirty_paths=[], panel_sources_sha256="p" * 64,
                stage2_modules=MODE["sources"] or sets.modules_now(), versions={}, blas_threads=1, accuracy="r2",
                at="2026-11-15T00:00:00Z")


def arrays(name):
    rng = np.random.default_rng(zlib.crc32((name + MODE["salt"]).encode()))
    q = rng.standard_normal((8, 5))
    return dict(t=0.01 * np.arange(8), q=q, qd=2.0 * q, qdd=3.0 * q)


def stride_record(names, ok=True, reason=None):
    row = dict(f=0.7, left=0.71, right=0.71, ik_ok=True, pass_=True, ik_fail_frames=dict(left=0, right=0),
               tightest_other=dict(left=dict(joint="left_hip_roll", end="min", margin_rad=0.101),
                                   right=dict(joint="right_hip_roll", end="max", margin_rad=0.101)))
    return dict(rule="S2A-stride", text="the stride rule", grid=[1.0, 0.7, 0.6], limit_kinematic=0.85, limit_final=0.87,
                limit_other_joints_rad=0.05, steps_down_max=2, tried=[row], chosen=0.7, steps_down=0,
                builds=[dict(f=0.7, ik_ok=True, hip_yaw_range_use=0.72, tightest_other_rad=0.1005)],
                outcome="frozen", a1_over=False, reason=None, references={n: ok for n in names},
                no_reference=None if ok else reason)


def produce(set_name, body, name, root=setio.ROOT):
    its = {i.name: i for i in sets.items(set_name, body)}
    group = its[name].group if name in its else name
    MODE["calls"].append(group)
    d = setio.set_dir(set_name, body, root)
    scratch = sets.scratch_dir(d, group)
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)
    out = dict(item=group, set=set_name, body=body, refs={}, transitions={}, look_entries={}, stride_rules={}, files={},
               header=dict(model=dict(file_sha256=its["nominal"].definition["body_model_sha256"]), H_m=0.5,
                           q_home=[0.0, 0.0], joints=["a", "b"]),
               blocks_set=None, log=[dict(what=group, seconds=1.5, text="a stand-in")])
    refs_now = s2io.read_json(d / "refs.json")["references"] if (d / "refs.json").is_file() else {}
    mine = [i for i in its.values() if i.group == group]

    def put(sub, nm, arrs):
        sha = s2io.write_npz(scratch / sub / f"{nm}.npz", arrs)
        out["files"][f"{sub}/{nm}.npz"] = sha
        return sha

    def inputs(walks):
        return {w: (None if refs_now[w].get("available") is False else refs_now[w]["sha256"]) for w in walks}

    for it in mine:
        why = MODE["fail"].get(it.name) or MODE["fail"].get(group)
        if it.kind == "walk":
            v = setio.parse_name(it.name)["speed_m_s"]
            spec = it.definition["spec"]
            if why:
                out["refs"][it.name] = sets._unavailable(it.name, "walk", why, dict(over=1.02), set=set_name,
                                                         crouch_frac=spec["crouch"], spec=spec, dropped=True,
                                                         group=group, definition=it.definition, built_by=built_by())
                continue
            a = arrays(it.name)
            sha, sha_look = put("refs", it.name, a), put("look", it.name, dict(a, phase=np.arange(8.0)))
            out["refs"][it.name] = dict(
                name=it.name, kind="walk", file=f"refs/{it.name}.npz", sha256=sha, frames=8, dt_s=0.01, period_s=0.08,
                speed_m_s=math.sqrt(spec["froude"] * 9.81 * 0.5) if v is None else v, stride_m=0.3,
                yaw_rate_rad_s=0.0, strides=1, window_s=0.08, crouch_frac=spec["crouch"], spec=spec, set=set_name,
                label=None, margin_rule=None, look_file=f"look/{it.name}.npz", look_sha256=sha_look,
                stage1_sha256=None, equals_stage1=None, ladder_sha256=None, equals_ladder=None, ladder_twin=None,
                stage0=dict(blocks_freeze=[]), available=True, group=group, definition=it.definition,
                built_by=built_by())
            out["look_entries"][it.name] = dict(file=f"look/{it.name}.npz", sha256=sha_look, frames=8)
            continue
        walks = it.depends[:2] if it.kind == "change" else it.depends[:1]
        if why:
            out["transitions"][it.name] = sets._unavailable(it.name, it.kind, why, dict(gate="x", value=1.0),
                                                            item=it.definition, group=group, inputs=inputs(walks),
                                                            set=set_name, pass_=False, built_by=built_by())
            continue
        name, d_ = it.name, it.definition
        if it.kind == "start" and refs_now[walks[0]].get("available") is False:  # S2S-lead-in: no nominal walk
            walks, bar = it.depends, it.depends[1]
            lead = d_["lead_in"]
            has = refs_now[bar].get("available") is not False
            out["transitions"][it.name] = sets._unavailable(
                it.name, it.kind, "the walk nominal is not a rung of the set" + (
                    f"; the set starts from standing straight into {bar} ({lead['name']})" if has else ""),
                dict(walks={}), item=d_, group=group, inputs=inputs(walks), set=set_name, pass_=None)
            if not has:
                continue
            name, d_ = lead["name"], dict(d_, **lead["start"])
        sha = put("transitions", name, arrays(name))
        e = dict(name=name, kind="walk" if it.kind in ("change", "turn") else it.kind.split("_")[0],
                 from_=d_["specs"]["a"]["name"], to=d_["specs"]["b"]["name"],
                 gaits=dict(a=dict(name=d_["specs"]["a"]["name"], period=0.08, speed=0.4, duty=0.58),
                            b=dict(name=d_["specs"]["b"]["name"], period=0.08, speed=0.4, duty=0.58)),
                 frames=8, duration_s=0.07, split="statics", ik_ok=True, ik_fail_frames=0,
                 file=f"transitions/{name}.npz", sha256=sha, available=True, set=set_name, group=group,
                 item=it.definition,
                 definition=dict({k: d_[k] for k in ("builder", "args", "specs", "causal", "laws", "rules")},
                                 body_model_sha256=d_["body_model_sha256"], ref_a_sha256=None, ref_b_sha256=None),
                 inputs=inputs(walks), a1_over=False, fallback=None, hip_yaw_range_use=0.1,
                 tightest_other=dict(joint="left_hip_roll", end="min", margin_rad=0.2),
                 stage1_sha256=None, equals_stage1=None, stage1_twin=None, stage0=dict(blocks_freeze=[], a1_over=False),
                 move=dict(kind="turn" if it.kind == "turn" else d_["builder"].split("_")[0], builder=d_["builder"],
                           causal=d_["causal"], command_phase=-0.5 if d_["causal"] else None, rules=d_["rules"],
                           laws="today"),
                 gates=dict(ik_ok=True, join_b=True), pass_=True, reference=dict(accel_forward_lp1hz_peak_m_s2=0.4),
                 built_by=built_by())
        if it.kind == "stop_two_step":
            e["reference"].update(touchdowns_after_command=2, forward_stopped_strides=1.2)
        if it.kind == "turn":
            p = setio.parse_name(it.name)
            e["turn"] = dict(angle_rad=p["angle_rad"], rate_rad_s=p["sign"] * p["rate_rad_s"], direction=p["direction"],
                             ramp_s=0.375, stride_factor=0.7, heading_total_rad=p["sign"] * p["angle_rad"])
            e["reference"].update(heading_at_bar_deg=85.0, heading_at_slowest_clock_deg=80.6)
            e["hip_yaw_range_use"] = 0.72
            e["tightest_other"] = dict(joint="left_hip_roll", end="min", margin_rad=0.1005)
        out["transitions"][name] = e
    if mine[0].kind == "turn":
        label = setio.parse_name(mine[0].name)["rate"]
        why = MODE["fail"].get(group)
        rec = stride_record([i.name for i in mine], ok=not why, reason=why)
        out["stride_rules"][label] = rec
        for i in mine:
            if not why:
                out["transitions"][i.name]["turn"]["stride_rule"] = rec
    if MODE["blocks_set"] == group:
        out["blocks_set"] = f"{group}: no transition within the gates"
    return out


def main(argv):
    """The kill-point test's child: build the set with the stand-in, in this process."""
    import os
    from pathlib import Path
    root, set_name, body = argv[:3]
    sets.produce = produce
    sets._require_models = lambda body: None
    r = sets.build(set_name, body, root=root, procs=0, log=lambda line: None)
    print(len(r["ran"]), len(r["recovered"]), len(r["kept"]))
    return 0
'''
fake: dict = {}
exec(compile(FAKE_CODE, "fake_sets", "exec"), fake)
MODE = fake["MODE"]


@pytest.fixture
def standin(monkeypatch):
    """``sets.produce`` replaced by the stand-in; no model file is needed; a clean ``MODE``."""
    MODE.update(salt="", fail={}, blocks_set=None, calls=[], sources=None)
    monkeypatch.setattr(sets, "produce", fake["produce"])
    monkeypatch.setattr(sets, "_require_models", lambda body: None)
    return MODE


def build(root, only=None, log=None) -> dict:
    return sets.build(SET, BODY, root=root, only=only, procs=0, log=(log.append if log is not None else lambda line: None))


def tree(d: Path) -> dict:
    """relative path -> bytes of every file below ``d`` (the logs aside)."""
    return {p.relative_to(d).as_posix(): p.read_bytes() for p in sorted(d.rglob("*"))
            if p.is_file() and "logs" not in p.relative_to(d).parts}


def mtimes(d: Path) -> dict:
    return {p.relative_to(d).as_posix(): p.stat().st_mtime_ns for p in sorted(d.rglob("*"))
            if p.is_file() and "logs" not in p.relative_to(d).parts}


def docs_of(d: Path) -> dict:
    return {n: json.loads((d / n).read_text()) for n in ("refs.json", "transitions.json", "set.json")}


def edit_json(path: Path, fn) -> None:
    doc = json.loads(path.read_text())
    fn(doc)
    s2io.write_json(path, doc)


def write_screen(d: Path, **over) -> None:
    """A ``screen.json`` with the keys ``sets verify`` reads, one block per available entry of the set as it is."""
    docs = docs_of(d)
    blocks = {}
    for kind, entries in (("walk", docs["refs.json"]["references"]), ("move", docs["transitions.json"]["transitions"])):
        for name, e in entries.items():
            if e.get("available") is False:
                continue
            b = dict(name=name, kind=kind, arrays=dict(sha256=e["look_sha256"] if kind == "walk" else e["sha256"]),
                     verdict=dict(blocks_freeze=[], a1_over=False),
                     drives={g: dict(cont_ratio=0.5, peak_ratio=0.6, speed_ratio=0.7) for g in ("neck", "tail", "knee")},
                     range=dict(hip_yaw_range_use=0.72))
            if kind == "walk":
                b["reference"] = dict(file=e["file"], sha256=e["sha256"])
            blocks[name] = b
    doc = dict(set=SET, body=BODY, schema=2, blocks=blocks, refused={})
    for path, value in over.items():
        node = doc
        *head, last = path.split("__")
        for key in head:
            node = node[key]
        node[last] = value
    s2io.write_json(d / "screen.json", doc)


@pytest.fixture
def legacy_ok(monkeypatch, tmp_path):
    """A legacy record of today's ``moves.py`` (the committed one is the lead's to make at the freeze), and a clean
    tree as ``s2io.git_state`` reports it."""
    rec = tmp_path / "legacy_check.json"
    names = {n: dict(sha256="a", recorded="a", equal=True) for n in ("start", "stop")}
    s2io.write_json(rec, dict(moves_sha256=s2io.sha256_file(s2io.module_path("moves.py")),
                              bodies={b: dict(complete=True, names=names) for b in look.BODIES}))
    monkeypatch.setattr(sets, "LEGACY_CHECK", rec)
    state = dict(commit="c" * 40, dirty=False, dirty_paths=[])
    monkeypatch.setattr(s2io, "git_state", lambda own_outputs=(): dict(state))
    return state


def test_build_makes_the_set_and_a_second_build_does_nothing(tmp_path, standin):
    log: list = []
    r = build(tmp_path, log=log)
    d = Path(r["dir"])
    assert d == tmp_path / SET / BODY
    names = [i.name for i in sets.items(SET, BODY)]
    assert sorted(r["ran"]) == sorted(sets.groups(sets.items(SET, BODY))) and r["kept"] == r["recovered"] == r["mismatch"] == []
    assert r["blocked"] is None and r["not_available"] == {}
    # the walks first, then the start and the change into the bar's walk, then the rest in the items' order
    assert standin["calls"][:8] == ["nominal", "v0.3", "v0.35", "v0.4", "v0.45", "v0.5", "start", "nominal_to_v0.4"]
    docs = docs_of(d)
    assert list(docs["refs.json"]["references"]) == names[:6]
    assert sorted(docs["transitions.json"]["transitions"]) == sorted(names[6:])
    files = sorted(tree(d))
    assert files == sorted(["refs.json", "transitions.json", "set.json"] + [f"refs/{n}.npz" for n in names[:6]]
                           + [f"look/{n}.npz" for n in names[:6]] + [f"transitions/{n}.npz" for n in names[6:]])
    for n in names[:6]:
        e = docs["refs.json"]["references"][n]
        assert e["sha256"] == s2io.sha256_file(d / "refs" / f"{n}.npz") and e["look_sha256"] == s2io.sha256_file(d / "look" / f"{n}.npz")
        assert docs["set.json"]["look_entries"][n]["sha256"] == e["look_sha256"]
    # Stage 1's top level and the set's own
    r_doc, t_doc, s_doc = docs["refs.json"], docs["transitions.json"], docs["set.json"]
    assert {"body", "accuracy", "model", "H_m", "q_home", "joints", "feedforward", "references", "set", "schema", "rules"} == set(r_doc)
    assert {"body", "transitions", "method", "preview", "shift_strides", "settle_strides", "path_half_width",
            "walk_ramp_strides", "windows", "g3_com_mm", "g3_zmp_mm", "causal_limits", "set", "schema", "rules"} == set(t_doc)
    assert {"set", "body", "schema", "role", "note", "spec_rule", "margin_rule", "speeds", "crouch_frac", "omega_rule",
            "rules", "lead_in", "look_entries", "stride_rules", "not_available", "blocked", "freezes"} == set(s_doc)
    assert s_doc["lead_in"] == "start" and r["lead_in"] == "start" and "S2S-lead-in" in t_doc["rules"]
    # the limits of a commanded move as the owner's decision P34 set them (STAGE2_75_PLAN.md A7, A12 as amended)
    assert t_doc["causal_limits"] == dict(warmup_com_mm=1.0, warmup_zmp_mm=4.0, joint_at_command_rad=1e-3,
                                          consistency_first_stride_mm=4.0, zmp_after_first_stride_mm=4.0,
                                          cop_margin_single_mm=10.0, joint_range_rad=0.05)
    assert (t_doc["g3_com_mm"], t_doc["g3_zmp_mm"]) == (1.0, 2.0)  # Stage 1's, which a move's first and last stride keep
    assert (r_doc["set"], r_doc["body"], r_doc["schema"], r_doc["accuracy"]) == (SET, BODY, 1, "r2")
    assert s_doc["role"] == "control" and s_doc["crouch_frac"] == 0.05 and s_doc["freezes"] == [] and s_doc["blocked"] is None
    assert list(s_doc["speeds"]) == names[:6] and s_doc["speeds"]["v0.45"] == 0.45
    assert set(s_doc["stride_rules"]) == set(setio.TURN_RATES) and s_doc["rules"] == sets.RULES
    assert set(moves.RULES) < set(t_doc["rules"]) and "S2S-fallback" in t_doc["rules"]
    # no wall time in a record: it is in the log, which git ignores
    assert all(no_wall_time(x) for x in docs.values())
    assert "seconds=1.5" in (d / "logs" / "sets.log").read_text() and not (d / "partial").exists() or not list((d / "partial").iterdir())
    assert not (d / ".lock").exists() and not list(d.rglob("*.tmp"))
    # a second build runs nothing and writes nothing
    before, calls = mtimes(d), len(standin["calls"])
    again = build(tmp_path)
    assert again["ran"] == [] and sorted(again["kept"]) == sorted(r["ran"]) and len(standin["calls"]) == calls
    assert mtimes(d) == before
    st = sets.status(SET, BODY, tmp_path)
    assert st["built"] and st["missing"] == [] and set(st["items"].values()) == {"built"} and st["frozen"] is None


def test_status_of_a_set_that_is_not_built(tmp_path):
    st = sets.status(SET, BODY, tmp_path)
    assert not st["built"] and len(st["missing"]) == 22 and set(st["items"].values()) == {"missing"}
    v = sets.verify(SET, BODY, tmp_path)
    assert not v["ok"] and "cannot be read" in v["failures"][0]


def test_only_builds_the_named_items_and_their_group(tmp_path, standin):
    r = build(tmp_path, only=["v0.4", "nominal"])
    assert r["ran"] == ["nominal", "v0.4"]
    r = build(tmp_path, only=["turn90_pi4_left"])  # a turn rate is built for both directions
    assert r["ran"] == ["turn90_pi4"]
    d = Path(r["dir"])
    assert sorted(docs_of(d)["transitions.json"]["transitions"]) == ["turn90_pi4_left", "turn90_pi4_right"]
    assert sets.status(SET, BODY, tmp_path)["items"]["turn90_pi4_right"] == "built"
    with pytest.raises(sets.SetError, match="no item or group"):
        build(tmp_path, only=["turn90_2_left"])


def test_what_changed_is_built_again_and_only_that(tmp_path, standin, monkeypatch):
    r = build(tmp_path)
    d = Path(r["dir"])
    # a walk's array is gone: the walk is built again; its array is the same, so its entry and its moves stand
    (d / "refs" / "v0.45.npz").unlink()
    before = mtimes(d)
    r = build(tmp_path)
    assert r["ran"] == ["v0.45"] and mtimes(d) == dict(before, **{"refs/v0.45.npz": mtimes(d)["refs/v0.45.npz"],
                                                                  "look/v0.45.npz": mtimes(d)["look/v0.45.npz"]})
    # the same walk rebuilt to other arrays (other code): its entry is written, and what was built from it follows
    standin["salt"] = "x"
    (d / "refs" / "v0.45.npz").unlink()
    was = docs_of(d)
    r = build(tmp_path)
    assert r["ran"] == ["v0.45", "nominal_to_v0.45"]  # (v0.5's change read v0.45 only if its fallback needed it)
    now = docs_of(d)
    assert now["refs.json"]["references"]["v0.45"]["sha256"] != was["refs.json"]["references"]["v0.45"]["sha256"]
    assert now["refs.json"]["references"]["v0.4"] == was["refs.json"]["references"]["v0.4"]
    # a definition that moved: the walk at that speed and every move from or to it
    standin["salt"] = ""
    st = sets.SETS[SET]
    monkeypatch.setitem(sets.SETS, SET, replace(st, spec_of=lambda body, H, v: replace(
        st.spec_of(body, H, v), lift_frac=0.09 if v == 0.3 else st.spec_of(body, H, v).lift_frac)))
    assert sets.status(SET, BODY, tmp_path)["items"]["v0.3"] == "stale"
    r = build(tmp_path)
    assert r["ran"] == ["v0.3", "nominal_to_v0.3"]
    assert docs_of(d)["refs.json"]["references"]["v0.3"]["spec"]["lift_frac"] == 0.09


def test_an_item_with_no_reference_is_an_entry_too(tmp_path, standin):
    """A turn rate outside the heading rule, a dropped rung and the change into it: ``available: false`` with the
    reason and the numbers, no file, listed in ``set.json``; never built again; ``verify`` notes them."""
    standin["fail"] = {"turn90_1": sets.NO_HEADING, "v0.5": "dropped from the ladder: over knee speed 1.02",
                       "nominal_to_v0.5": "the walk v0.5 is not a rung of the set"}
    r = build(tmp_path)
    d = Path(r["dir"])
    gone = ["v0.5", "nominal_to_v0.5", "turn90_1_left", "turn90_1_right"]
    assert sorted(r["not_available"]) == sorted(gone) and r["blocked"] is None
    docs = docs_of(d)
    assert docs["refs.json"]["references"]["v0.5"]["available"] is False and "file" not in docs["refs.json"]["references"]["v0.5"]
    assert "v0.5" not in docs["set.json"]["look_entries"]
    for n in gone[1:]:
        e = docs["transitions.json"]["transitions"][n]
        assert e["available"] is False and e["reason"] and "numbers" in e and "file" not in e
        assert docs["set.json"]["not_available"][n] == dict(reason=e["reason"], numbers=e["numbers"])
    assert not any((d / sub / f"{n}.npz").exists() for n in gone for sub in ("refs", "look", "transitions"))
    assert docs["set.json"]["stride_rules"]["1"]["references"] == {"turn90_1_left": False, "turn90_1_right": False}
    assert docs["set.json"]["stride_rules"]["1"]["no_reference"] == sets.NO_HEADING
    calls = len(standin["calls"])
    assert build(tmp_path)["ran"] == [] and len(standin["calls"]) == calls  # a recorded result is not built again
    st = sets.status(SET, BODY, tmp_path)
    assert st["items"]["turn90_1_left"] == f"not available: {sets.NO_HEADING}" and st["missing"] == []
    write_screen(d)
    v = sets.verify(SET, BODY, tmp_path)
    assert v["ok"], v["failures"]
    assert sorted(v["not_available"]) == sorted(gone) and len(v["available"]) == 18
    assert f"turn90_1_left: not available: {sets.NO_HEADING}" in v["notes"] and f"turn rate 1: {sets.NO_HEADING}" in v["notes"]
    # the bar's planner reads such an entry as "no reference" with its reason (bar.BarPlanner.why_not's test)
    e = docs["transitions.json"]["transitions"]["turn90_1_left"]
    assert not (e.get("available") is not False and e.get("file"))
    with pytest.raises(tune.InputError, match="not listed|not available"):
        setio.read_inputs(d, ["transitions/turn90_1_left.npz"], require_frozen=False)


def test_an_entry_that_loses_its_reference_leaves_no_array_behind(tmp_path, standin):
    """An item built again that ends with no reference: its entry is replaced and the array an earlier build put in
    place goes (a walk's two files); ``verify`` notes any array no entry lists."""
    r = build(tmp_path)
    d = Path(r["dir"])
    assert (d / "transitions" / "stop1_v0.4.npz").is_file() and (d / "look" / "v0.3.npz").is_file()
    standin["fail"] = {"stop1_v0.4": sets.NO_GATES, "v0.3": "dropped from the ladder: over knee speed 1.02",
                       "nominal_to_v0.3": "the walk v0.3 is not a rung of the set"}
    edit_json(d / "transitions.json", lambda doc: doc["transitions"]["stop1_v0.4"].update(item={}))
    edit_json(d / "refs.json", lambda doc: doc["references"]["v0.3"].update(definition={}))
    log: list = []
    r = build(tmp_path, log=log)
    assert r["ran"] == ["v0.3", "nominal_to_v0.3", "stop1_v0.4"]
    assert not any((d / rel).exists() for rel in ("transitions/stop1_v0.4.npz", "refs/v0.3.npz", "look/v0.3.npz",
                                                  "transitions/nominal_to_v0.3.npz"))
    assert f"{SET} {BODY}: removed look/v0.3.npz: no entry of the set names it any more" in log
    write_screen(d)
    v = verify(d)
    assert v["ok"] and not any("an array that no entry of the set lists" in x for x in v["notes"])
    (d / "transitions" / "left_over.npz").write_bytes(b"x")
    v = verify(d)
    assert v["ok"] and "transitions/left_over.npz: an array that no entry of the set lists (nothing reads it; remove it)" in v["notes"]


def test_every_move_entry_carries_the_sets_margin_rule(tmp_path, monkeypatch):
    """The plan's A3: every entry of the candidate carries ``margin_rule``, a move's (with or without a reference)
    as a walk's; the other sets' entries carry None."""
    monkeypatch.setattr(sets, "_produce_straight", lambda job, item: job.out["transitions"].update(
        {item.name: dict(name=item.name), "other": sets._unavailable("other", "change", "x", {})}))
    out = sets.produce("F25F9cl", BODY, "start", tmp_path)
    rule = "duty and lift set after results on seeds 3042-3061 (JPS 2.3)"
    assert out["transitions"]["start"]["margin_rule"] == out["transitions"]["other"]["margin_rule"] == rule == sets.RULES["S2S-margin"]
    assert sets.SETS["F25F9cl"].margin_rule == rule
    for name in ("V0", "F25F9"):
        assert sets.produce(name, BODY, "start", tmp_path)["transitions"]["start"]["margin_rule"] is None


def test_a_failed_start_stops_the_set(tmp_path, standin, capsys):
    standin["fail"] = {"start": sets.NO_GATES}
    standin["blocks_set"] = "start"
    r = build(tmp_path)
    assert r["blocked"] == [dict(by="start", reason="start: no transition within the gates")]
    assert r["ran"] == ["nominal", "v0.3", "v0.35", "v0.4", "v0.45", "v0.5", "start"] and len(r["skipped"]) == 11
    d = Path(r["dir"])
    assert [b["by"] for b in docs_of(d)["set.json"]["blocked"]] == ["start"]
    write_screen(d)
    v = sets.verify(SET, BODY, tmp_path)
    assert not v["ok"] and any("the set is stopped" in x for x in v["failures"])
    monkey_argv = ["build", "--set", SET, "--body", BODY, "--root", str(tmp_path), "--procs", "0"]
    assert sets.main(monkey_argv) == 1 and "the set is stopped" in capsys.readouterr().out
    # the start built again within its gates lifts the stop
    standin["fail"], standin["blocks_set"] = {}, None
    edit_json(d / "transitions.json", lambda doc: doc["transitions"].pop("start"))
    r = build(tmp_path)
    assert r["blocked"] is None and "start" in r["ran"] and docs_of(d)["set.json"]["blocked"] is None


def test_blocked_names_every_failing_group_and_a_passing_merge_lifts_only_its_own(tmp_path, standin, monkeypatch,
                                                                                   capsys):
    """Ruling 23 of 17 Nov 2026 (the look sets' check, finding 3): ``set.json`` ``blocked`` is a list that names
    every failing group that stops the set, the start and the change into the bar's walk both when both fail; a
    group merged again within its gates lifts its own stop and no other.  An older record's dict (one group: the
    last blocking merge of older code overwrote the other) or null still reads, and a stop that the records show
    (``blocks_in_records``) is kept when another group's merge passes."""
    both = ("start", "nominal_to_v0.4")
    standin["fail"] = {g: sets.NO_GATES for g in both}
    inner = fake["produce"]

    def produce(*args, **kw):
        out = inner(*args, **kw)
        if out["item"] in MODE["fail"] and out["item"] in both:
            out["blocks_set"] = f"{out['item']}: no transition within the gates"
        return out

    monkeypatch.setattr(sets, "produce", produce)
    r = build(tmp_path)  # one process: the failed start stops this run before the change is built
    assert [b["by"] for b in r["blocked"]] == ["start"] and "nominal_to_v0.4" in r["skipped"]
    d = Path(r["dir"])
    r = build(tmp_path, only=["nominal_to_v0.4"])
    want = [dict(by=g, reason=f"{g}: no transition within the gates") for g in both]
    assert r["blocked"] == want == docs_of(d)["set.json"]["blocked"] and r["ran"] == ["nominal_to_v0.4"]
    write_screen(d)
    v = sets.verify(SET, BODY, tmp_path)
    assert [x for x in v["failures"] if x.startswith("the set is stopped")] == [
        f"the set is stopped: {g}: no transition within the gates (it goes to the lead)" for g in both]
    assert sets.status(SET, BODY, tmp_path)["blocked"] == want
    argv = ["build", "--set", SET, "--body", BODY, "--root", str(tmp_path), "--procs", "0"]
    assert sets.main(argv) == 1 and ("the set is stopped: start: no transition within the gates; nominal_to_v0.4: no "
                                     "transition within the gates") in capsys.readouterr().out
    # the change built again within its gates: the start still stops the set
    del standin["fail"]["nominal_to_v0.4"]
    edit_json(d / "transitions.json", lambda doc: doc["transitions"].pop("nominal_to_v0.4"))
    r = build(tmp_path, only=["nominal_to_v0.4"])
    assert r["blocked"] == want[:1] == docs_of(d)["set.json"]["blocked"]
    write_screen(d)
    v = sets.verify(SET, BODY, tmp_path)
    assert [x for x in v["failures"] if x.startswith("the set is stopped")] == [
        "the set is stopped: start: no transition within the gates (it goes to the lead)"]
    # an older record: blocked a dict naming the change alone, though the start's record fails its gates
    standin["fail"]["nominal_to_v0.4"] = sets.NO_GATES
    edit_json(d / "transitions.json", lambda doc: doc["transitions"].pop("nominal_to_v0.4"))
    build(tmp_path, only=["nominal_to_v0.4"])
    old = dict(by="nominal_to_v0.4", reason="nominal_to_v0.4: no transition within the gates")
    edit_json(d / "set.json", lambda doc: doc.update(blocked=old))
    assert sets.status(SET, BODY, tmp_path)["blocked"] == [old] and sets.blocked_list(old) == [old]
    assert sets.blocked_list(None) == [] and sets.blocked_list([old]) == [old]
    assert sets.blocks_in_records(docs_of(d)) == {g: f"{g}: {sets.NO_GATES} ({sets.RULES['S2S-fallback']})"
                                                  for g in both}
    # verify names the dict's stop and the start's, which the records show and the dict leaves out (ruling 36)
    assert [x for x in sets.verify(SET, BODY, tmp_path)["failures"] if x.startswith("the set is stopped")] == [
        f"the set is stopped: {old['reason']} (it goes to the lead)",
        f"the set is stopped: start: {sets.NO_GATES} ({sets.RULES['S2S-fallback']}) (it goes to the lead; the "
        "records show it and set.json blocked does not name it)"]
    del standin["fail"]["nominal_to_v0.4"]
    edit_json(d / "transitions.json", lambda doc: doc["transitions"].pop("nominal_to_v0.4"))
    r = build(tmp_path, only=["nominal_to_v0.4"])
    assert r["blocked"] == [dict(by="start", reason=f"start: {sets.NO_GATES} ({sets.RULES['S2S-fallback']})")]
    assert docs_of(d)["set.json"]["blocked"] == r["blocked"]
    # the start within its gates: nothing stops the set, and the record says null as a set never stopped
    del standin["fail"]["start"]
    edit_json(d / "transitions.json", lambda doc: doc["transitions"].pop("start"))
    r = build(tmp_path)
    assert r["blocked"] is None and docs_of(d)["set.json"]["blocked"] is None and r["ran"] == ["start"]


#: A build with two spawned processes, the stand-in installed in each: the script's top level runs again in every
#: spawned child (as ``__mp_main__``), so the children build with the stand-in too.
TWO_PROCS = r"""
import json, sys
import fake_sets
from designlab.stage2 import sets, setio

root, set_name, body = sys.argv[1:4]
both = ("start", "nominal_to_v0.4")
fake_sets.MODE["fail"] = {g: sets.NO_GATES for g in both}
_inner = fake_sets.produce


def produce(*args, **kw):
    out = _inner(*args, **kw)
    if out["item"] in both:
        out["blocks_set"] = f"{out['item']}: no transition within the gates"
    return out


sets.produce = produce
sets._require_models = lambda body: None

if __name__ == "__main__":
    sets.build(set_name, body, root=root, procs=0, only=sets.walk_names(), log=lambda line: None)  # the walks here
    r = sets.build(set_name, body, root=root, procs=2, log=lambda line: None)
    doc = json.load(open(setio.set_dir(set_name, body, root) / "set.json"))
    print(json.dumps(dict(blocked=r["blocked"], recorded=doc["blocked"], ran=r["ran"], skipped=r["skipped"])))
"""


def test_a_build_with_two_processes_names_both_stops(tmp_path):
    """Ruling 36 of 17 Nov 2026 (the start law's check, low point 2): with two processes the start and the change
    into the bar's walk run together; when both fail, the build and ``set.json`` ``blocked`` name both, in that
    order, and the groups not yet started are not built."""
    mod = tmp_path / "fake_sets.py"
    mod.write_text(FAKE_CODE)
    script = tmp_path / "two_procs.py"
    script.write_text(TWO_PROCS)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(tmp_path), str(ROOT)]), MUJOCO_GL="disable")
    env.pop(sets.FAIL_AT, None)
    r = subprocess.run([PY, str(script), str(tmp_path / "root"), SET, BODY], capture_output=True, text=True, env=env,
                       cwd=ROOT, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    got = json.loads(r.stdout.strip().splitlines()[-1])
    want = [dict(by=g, reason=f"{g}: no transition within the gates") for g in ("start", "nominal_to_v0.4")]
    assert got["blocked"] == want == got["recorded"]
    assert {"start", "nominal_to_v0.4"} <= set(got["ran"]) and got["skipped"]
    assert not set(got["ran"]) & set(got["skipped"])


@pytest.mark.parametrize("set_name", list(sets.SETS))
def test_the_frozen_sets_records_name_no_stop(set_name):
    """The round's three sets on 75-M-H-10 record ``blocked`` null and their records show no stop, so a merge into
    them writes null again (ruling 23 leaves their records as they are)."""
    d = setio.ROOT / set_name / BODY
    if not (d / "set.json").exists():
        pytest.skip(f"the {set_name} set is not built (stage2/sets/{set_name})")
    docs = docs_of(d)
    assert docs["set.json"]["blocked"] is None and sets.blocks_in_records(docs) == {}
    for group in sets.STOP_GROUPS + ("nominal",):
        assert sets._blocked_after(docs["set.json"], docs, group, None) is None


def test_a_fallbacks_change_is_merged_and_later_removed(tmp_path, standin):
    """An item may make an entry under another name (the change from the neighbouring rung, fallback step 2); when
    the item is built again without it, the entry goes."""
    build(tmp_path, only=sets.walk_names())
    d = tmp_path / SET / BODY
    real = fake["produce"]

    def with_fallback(set_name, body, name, root=setio.ROOT):
        out = real(set_name, body, name, root)
        if out["item"] == "nominal_to_v0.5":
            e = out["transitions"].pop("nominal_to_v0.5")
            other = dict(e, name="v0.45_to_v0.5", file="transitions/v0.45_to_v0.5.npz",
                         fallback=dict(rule="S2S-fallback", step=2, of="nominal_to_v0.5"),
                         inputs={"v0.45": e["inputs"]["nominal"], "v0.5": e["inputs"]["v0.5"]})
            scratch = sets.scratch_dir(d, "nominal_to_v0.5") / "transitions"
            os.replace(scratch / "nominal_to_v0.5.npz", scratch / "v0.45_to_v0.5.npz")
            out["files"] = {"transitions/v0.45_to_v0.5.npz": out["files"].pop("transitions/nominal_to_v0.5.npz")}
            other["definition"] = dict(e["definition"], specs=dict(a=dict(name="v0.45"), b=dict(name="v0.5")))
            other["inputs"] = {"v0.45": s2io.read_json(d / "refs.json")["references"]["v0.45"]["sha256"],
                               "v0.5": e["inputs"]["v0.5"]}
            out["transitions"]["v0.45_to_v0.5"] = other
            out["transitions"]["nominal_to_v0.5"] = sets._unavailable(
                "nominal_to_v0.5", "change", "no direct transition within the gates; the rung is reached from v0.45",
                dict(tried=[]), item=e["item"], group="nominal_to_v0.5", inputs=e["inputs"], set=set_name, pass_=False)
        return out

    sets.produce = with_fallback
    try:
        r = build(tmp_path, only=["nominal_to_v0.5"])
    finally:
        sets.produce = real
    tr = docs_of(d)["transitions.json"]["transitions"]
    assert r["ran"] == ["nominal_to_v0.5"] and tr["nominal_to_v0.5"]["available"] is False
    assert tr["v0.45_to_v0.5"]["fallback"]["step"] == 2 and (d / "transitions" / "v0.45_to_v0.5.npz").is_file()
    assert sets.status(SET, BODY, tmp_path)["items"]["v0.45_to_v0.5"] == "built (made by the item nominal_to_v0.5)"
    assert _check_names(d) == []
    # the direct change built again (as after a definition moved): the fallback's entry is no longer made, and goes
    edit_json(d / "transitions.json", lambda doc: doc["transitions"]["nominal_to_v0.5"].update(item={}))
    r = build(tmp_path, only=["nominal_to_v0.5"])
    tr = docs_of(d)["transitions.json"]["transitions"]
    assert "v0.45_to_v0.5" not in tr and tr["nominal_to_v0.5"]["available"] is True
    assert not (d / "transitions" / "v0.45_to_v0.5.npz").exists()  # and its array with it


def test_a_set_that_starts_into_its_bar_walk_records_its_lead_in(tmp_path, standin, legacy_ok):
    """``set.json`` ``lead_in`` follows the walks (``RULES["S2S-lead-in"]``): ``start`` with the nominal walk,
    ``start_v0.4`` without it, with that start an entry made by the ``start`` item; ``verify`` notes the exception;
    a missing array of that start is built again, on a frozen set too; with the nominal walk back the set starts into
    it again and ``start_v0.4`` goes."""
    standin["fail"] = {"nominal": "dropped from the ladder: over clearance swing_clearance 0.0382"}
    r = build(tmp_path)
    d = tmp_path / SET / BODY
    docs = docs_of(d)
    tr = docs["transitions.json"]["transitions"]
    assert r["lead_in"] == docs["set.json"]["lead_in"] == "start_v0.4" and r["blocked"] is None
    assert tr["start"]["available"] is False and tr["start_v0.4"]["available"] is True and tr["start_v0.4"]["group"] == "start"
    assert tr["start_v0.4"]["inputs"] == dict(nominal=None, **{"v0.4": docs["refs.json"]["references"]["v0.4"]["sha256"]})
    assert (d / "transitions" / "start_v0.4.npz").is_file() and not (d / "transitions" / "start.npz").exists()
    assert set(docs["set.json"]["not_available"]) >= {"nominal", "start"} and "start_v0.4" not in docs["set.json"]["not_available"]
    st = sets.status(SET, BODY, tmp_path)
    assert st["lead_in"] == "start_v0.4" and st["items"]["start_v0.4"] == "built (made by the item start)"
    assert st["items"]["start"].startswith("not available: the walk nominal is not a rung of the set")
    assert _check_names(d) == []
    write_screen(d)
    v = verify(d)
    assert v["ok"], v["failures"]
    assert any(n.startswith("lead-in: start_v0.4 (the nominal walk is not a rung") for n in v["notes"])
    assert "start_v0.4" in v["available"] and "start" in v["not_available"]
    # the records must agree with the walks, and the start must be there and be a start into that walk
    edit_json(d / "set.json", lambda doc: doc.update(lead_in="start"))
    f = verify(d)["failures"]
    assert "set.json records lead_in 'start', its walks give 'start_v0.4'" in f
    assert "start_v0.4: the set holds it and its lead-in is 'start'" in f
    edit_json(d / "set.json", lambda doc: doc.update(lead_in="start_v0.4"))
    whole_ = tree(d)
    edit_json(d / "transitions.json", lambda doc: doc["transitions"]["start_v0.4"]["definition"]["specs"]["b"].update(name="v0.45"))
    assert "start_v0.4: its definition starts into 'v0.45'" in verify(d)["failures"]
    edit_json(d / "transitions.json", lambda doc: doc["transitions"]["start_v0.4"]["definition"]["specs"]["b"].update(name="v0.4"))
    edit_json(d / "transitions.json", lambda doc: doc["transitions"]["start_v0.4"]["item"].update(causal=True))
    assert any(x.startswith("start_v0.4: made by the item start, whose definition is not the one made now") for x in verify(d)["failures"])
    edit_json(d / "transitions.json", lambda doc: doc["transitions"]["start_v0.4"]["item"].update(causal=False))
    assert tree(d) == whole_ and verify(d)["ok"]
    # a second build does nothing; a missing array of the start is built again (the start item's group)
    assert build(tmp_path)["ran"] == []
    (d / "transitions" / "start_v0.4.npz").unlink()
    r = build(tmp_path)
    assert r["ran"] == ["start"] and tree(d) == whole_
    # frozen: the manifest lists the start the set has; its array is put back and no record is written
    fz = sets.freeze(SET, BODY, tmp_path, log=lambda line: None)
    assert "start_v0.4" in fz["manifest"] and "start" not in fz["manifest"] and "nominal" not in fz["manifest"]
    assert fz["manifest"]["start_v0.4"]["kind"] == "start" and docs_of(d)["set.json"]["lead_in"] == "start_v0.4"
    whole_ = tree(d)
    json_before = {n: (d / n).stat().st_mtime_ns for n in ("refs.json", "transitions.json", "set.json")}
    (d / "transitions" / "start_v0.4.npz").unlink()
    r = build(tmp_path)
    assert r["ran"] == ["start"] and r["restored"] == ["start_v0.4"] and r["mismatch"] == [] and tree(d) == whole_
    assert {n: (d / n).stat().st_mtime_ns for n in json_before} == json_before
    # --- another set directory: with its nominal walk back, the set starts into it and start_v0.4 goes
    other = tmp_path / "again"
    standin["fail"] = {"nominal": "dropped from the ladder: over clearance swing_clearance 0.0382"}
    build(other)
    standin["fail"] = {}
    edit_json(other / SET / BODY / "refs.json", lambda doc: doc["references"]["nominal"].update(definition={}))  # (as after its definition moved)
    r = build(other)
    docs = docs_of(other / SET / BODY)
    tr = docs["transitions.json"]["transitions"]
    assert "nominal" in r["ran"] and "start" in r["ran"] and docs["set.json"]["lead_in"] == "start"
    assert "start_v0.4" not in tr and tr["start"]["available"] is True and "start" not in docs["set.json"]["not_available"]
    # --- neither walk: no lead-in
    none = tmp_path / "none"
    standin["fail"] = {"nominal": "dropped from the ladder: x", "v0.4": "dropped from the ladder: x"}
    r = build(none)
    assert r["lead_in"] is None and docs_of(none / SET / BODY)["set.json"]["lead_in"] is None
    assert "start_v0.4" not in docs_of(none / SET / BODY)["transitions.json"]["transitions"]
    write_screen(none / SET / BODY)
    assert any("the set has no lead-in" in n for n in sets.verify(SET, BODY, none)["notes"])


def _check_names(d: Path) -> list:
    docs, bad = docs_of(d), []
    its = {i.name: i for i in sets.items(SET, BODY)}
    for kind, entries in (("walk", docs["refs.json"]["references"]), ("move", docs["transitions.json"]["transitions"])):
        for name, e in entries.items():
            sets._check_name(name, kind, e, its.get(name), bad)
    return bad


def test_names_are_checked_against_their_definitions(tmp_path, standin):
    build(tmp_path)
    d = tmp_path / SET / BODY
    assert _check_names(d) == []
    its = {i.name: i for i in sets.items(SET, BODY)}
    tr = docs_of(d)["transitions.json"]["transitions"]
    bad: list = []
    sets._check_name("stop", "move", tr["stop1_v0.4"], None, bad)  # Stage 1's own stop command's name
    sets._check_name("stop_in_place_0.50_left", "move", tr["stop1_v0.4"], None, bad)
    sets._check_name("turn90_pi4_left", "move", tr["turn90_pi4_right"], its["turn90_pi4_right"], bad)  # the other rate sign
    sets._check_name("turn90_0.5_left", "move", tr["turn90_pi4_left"], None, bad)  # another rate
    sets._check_name("nominal_to_v0.45", "move", tr["nominal_to_v0.4"], None, bad)
    sets._check_name("stop2_v0.4", "move", tr["stop1_v0.4"], None, bad)  # two touchdowns, the one-step builder
    sets._check_name("v0.40", "walk", {}, None, bad)  # a fixed-decimal name
    sets._check_name("v0.4", "walk", dict(speed_m_s=0.41), None, bad)
    assert len(bad) == 8, bad
    assert "Stage 1 registers its own stop command" in bad[0] and "Stage 1 registers its own stop command" in bad[1]
    assert "turns" in bad[2] and "turns" in bad[3] and "joins" in bad[4] and "builder is 'stop_one_step'" in bad[5]
    assert "not a name setio makes" in bad[6] and "its name holds 0.4" in bad[7]


# ---------------------------------------------------------------- resume, kill points, lock
def test_a_finished_partial_is_merged_and_a_stale_one_removed(tmp_path, standin):
    build(tmp_path, only=sets.walk_names())
    d = tmp_path / SET / BODY
    # a worker finished the start and the build was killed before the parent merged it
    part = sets.build_item(SET, BODY, "start", tmp_path)
    assert part["item"] == "start" and (d / "partial" / "start.json").is_file() and (d / "transitions" / "start.npz").is_file()
    assert "start" not in docs_of(d)["transitions.json"]["transitions"] and not sets.scratch_dir(d, "start").exists()
    assert sets.scratch_dir(d, "start") == d / "partial" / f"build-start-{os.getpid()}"  # this process's own
    # ... and another item's partial whose array is not in place, a partial of another set, the screen's own partial
    other = sets.build_item(SET, BODY, "stop2_v0.4", tmp_path)
    (d / "transitions" / "stop2_v0.4.npz").unlink()
    s2io.write_json(d / "partial" / "stop1_v0.4.json", dict(other, item="stop1_v0.4", set="V0"))
    s2io.write_json(d / "partial" / "screen__nominal.json", dict(name="nominal"))
    (d / "partial" / "build-v0.3").mkdir()
    (d / "refs.json.tmp").write_text("{")
    calls = len(standin["calls"])
    log: list = []
    r = build(tmp_path, only=["start", "stop2_v0.4"], log=log)
    assert r["recovered"] == ["start"] and r["ran"] == ["stop2_v0.4"] and len(standin["calls"]) == calls + 1
    assert "removed a partial item that is not of the set as it is now: stop1_v0.4.json" in log
    assert "removed a partial item that is not of the set as it is now: stop2_v0.4.json" in log
    assert "removed a stale scratch directory: partial/build-v0.3" in log and "removed a stale temporary file: refs.json.tmp" in log
    assert sorted(p.name for p in (d / "partial").iterdir()) == ["screen__nominal.json"]  # the screen's is left alone
    assert sorted(docs_of(d)["transitions.json"]["transitions"]) == ["start", "stop2_v0.4"]


#: Every write of one item, for a walk (it changes ``refs.json`` and ``set.json``) and for a turn rate (two files;
#: ``transitions.json`` and ``set.json``): a record that an item does not change is not written, so there is no kill
#: point after it.
POINTS = ([(p, "v0.4") for p in sets.FAIL_POINTS if p != "transitions.json"]
          + [(p, "turn90_pi4") for p in sets.FAIL_POINTS if p != "refs.json"]
          + [(p, "nominal") for p in ("refs.json", "transitions.json", "set.json")])  # the first merge of a new set


def _child(tmp: Path, root: Path, fail_at: str | None = None) -> subprocess.CompletedProcess:
    mod = tmp / "fake_sets.py"
    if not mod.exists():
        mod.write_text(FAKE_CODE)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(tmp), str(ROOT)]), MUJOCO_GL="disable")
    env.pop(sets.FAIL_AT, None)
    if fail_at:
        env[sets.FAIL_AT] = fail_at
    code = "import sys, fake_sets; sys.exit(fake_sets.main(sys.argv[1:]))"
    return subprocess.run([PY, "-c", code, str(root), SET, BODY], capture_output=True, text=True, env=env, cwd=ROOT)


@pytest.fixture(scope="module")
def whole(tmp_path_factory):
    """The synthetic set built in one go in a child process: what every interrupted build must end in."""
    tmp = tmp_path_factory.mktemp("whole")
    r = _child(tmp, tmp / "root")
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["18", "0", "0"]
    return tree(tmp / "root" / SET / BODY)


@pytest.mark.parametrize("point,item", POINTS, ids=[f"{p}:{i}" for p, i in POINTS])
def test_a_build_killed_at_any_point_resumes_to_the_same_files(tmp_path, whole, point, item):
    """Killed after the arrays are placed, after the partial is written, after each record is merged and after the
    partial is deleted (a walk, and a turn rate with its two files): the process dies there with no cleanup, the
    lock is left behind; a second build takes the stale lock over and ends in the files of an uninterrupted build,
    byte for byte (the stand-in's provenance has a fixed time)."""
    root = tmp_path / "root"
    r = _child(tmp_path, root, f"{point}:{item}")
    assert r.returncode == 86, (r.returncode, r.stderr[-400:])
    d = root / SET / BODY
    assert (d / ".lock").is_file()  # the killed builder's
    left = sorted(p.name for p in (d / "partial").glob("*.json")) if (d / "partial").is_dir() else []
    assert left == ([f"{item}.json"] if point not in ("npz", "deleted") else [])
    r = _child(tmp_path, root)
    assert r.returncode == 0, r.stderr[-600:]
    ran, recovered, kept = (int(x) for x in r.stdout.split())
    assert recovered == (0 if point in ("npz", "deleted") else 1) and ran + recovered + kept == 18
    assert tree(d) == whole
    assert not (d / ".lock").exists() and not list(d.rglob("*.tmp"))
    assert not (d / "partial").is_dir() or not list((d / "partial").iterdir())
    # and once more: nothing to do, nothing written
    before = mtimes(d)
    r = _child(tmp_path, root)
    assert r.stdout.split() == ["0", "0", "18"] and mtimes(d) == before


def test_an_incomplete_set_of_records_is_refused_unless_a_killed_first_merge_left_it(tmp_path, standin):
    """``refs.json`` without ``set.json`` is made again only where a partial still holds every entry of it (a build
    killed inside the first merge of a new set); any other incomplete set of records is refused as before."""
    build(tmp_path, only=["nominal"])
    d = tmp_path / SET / BODY
    (d / "set.json").unlink()
    with pytest.raises(sets.SetError, match="it holds refs.json, transitions.json and not the other record"):
        build(tmp_path)
    assert (d / "refs.json").is_file() and (d / "transitions.json").is_file()
    # the same records beside the partial of the item they hold: the killed merge is made again
    part = sets.build_item(SET, BODY, "nominal", tmp_path)
    assert set(part["refs"]) == {"nominal"} and (d / "partial" / "nominal.json").is_file()
    log: list = []
    r = build(tmp_path, only=["nominal"], log=log)
    assert r["recovered"] == ["nominal"] and r["ran"] == [] and (d / "set.json").is_file()
    assert sum("the first merge of the set was killed before set.json was written" in x for x in log) == 2
    assert list(docs_of(d)["refs.json"]["references"]) == ["nominal"]
    # an entry beyond what the partials hold: nothing is removed
    build(tmp_path, only=["v0.3"])
    sets.build_item(SET, BODY, "nominal", tmp_path)
    (d / "set.json").unlink()
    with pytest.raises(sets.SetError, match="and not the other record"):
        build(tmp_path)
    assert sorted(docs_of_two(d)) == ["nominal", "v0.3"]


def docs_of_two(d: Path) -> list:
    return list(json.loads((d / "refs.json").read_text())["references"])


def test_a_worker_writes_only_while_its_builder_holds_the_lock(tmp_path, standin):
    """An item started by a builder (``owner``: its pid) puts nothing in place once the set's lock names another
    holder or none: a worker that outlives its builder cannot write beside the builder that took the set over."""
    build(tmp_path, only=sets.walk_names())
    d = tmp_path / SET / BODY
    with pytest.raises(sets.SetError, match=r"the set's lock names no holder, not the builder that started this item \(pid 4242\)"):
        sets.build_item(SET, BODY, "start", tmp_path, owner=4242)
    with s2io.Lock(d / ".lock", log=lambda line: None):  # this process holds the set
        with pytest.raises(sets.SetError, match=rf"the set's lock names pid {os.getpid()}, not the builder .* \(pid 4242\)"):
            sets.build_item(SET, BODY, "start", tmp_path, owner=4242)
        assert not (d / "transitions" / "start.npz").exists() and not (d / "partial" / "start.json").exists()
        assert not sets.scratch_dir(d, "start").exists()
        part = sets.build_item(SET, BODY, "start", tmp_path, owner=os.getpid())
        assert part["item"] == "start" and (d / "transitions" / "start.npz").is_file() and (d / "partial" / "start.json").is_file()
    assert sets.build_item(SET, BODY, "stop2_v0.4", tmp_path)["item"] == "stop2_v0.4"  # no builder to answer to


def test_the_kill_points_are_the_writes_of_one_item():
    src = (ROOT / "designlab" / "stage2" / "sets.py").read_text()
    assert sets.FAIL_POINTS == ("npz", "partial", "refs.json", "transitions.json", "set.json", "deleted")
    assert sets.FAIL_AT == "STAGE2_SETS_FAIL_AT" and src.count("_fail_at(") == 5  # the definition and four call sites
    assert os.environ.get(sets.FAIL_AT) is None


def test_a_second_builder_is_refused_and_a_stale_lock_is_taken_over(tmp_path, standin):
    build(tmp_path, only=["nominal"])
    d = tmp_path / SET / BODY
    with s2io.Lock(d / ".lock", log=lambda line: None):  # a live builder holds the directory
        with pytest.raises(s2io.S2Error, match="is held by pid"):
            build(tmp_path)
        with pytest.raises(s2io.S2Error, match="is held by pid"):
            sets.freeze(SET, BODY, tmp_path)
    assert not (d / ".lock").exists()
    dead = subprocess.Popen([PY, "-c", "pass"])
    dead.wait()
    import socket
    (d / ".lock").write_text(json.dumps(dict(pid=dead.pid, host=socket.gethostname(), started_at="2026-11-15T00:00:00Z")))
    log: list = []
    r = build(tmp_path, only=["v0.4"], log=log)
    assert r["ran"] == ["v0.4"] and any("taken over" in line for line in log) and not (d / ".lock").exists()


def test_build_requires_one_thread_and_the_models(tmp_path, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    with pytest.raises(s2io.S2Error, match="one BLAS thread"):
        sets.build(SET, BODY, root=tmp_path, procs=0)
    with pytest.raises(s2io.S2Error, match="one BLAS thread"):
        sets.build_item(SET, BODY, "nominal", tmp_path)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setattr(stage1, "case_dir", lambda body: tmp_path / "no_models")
    with pytest.raises(sets.SetError, match="round-two model files are not here"):
        sets.build(SET, BODY, root=tmp_path, procs=0)
    assert not (tmp_path / SET).exists() or not list((tmp_path / SET / BODY).iterdir())


# ---------------------------------------------------------------- verify
@pytest.fixture
def made(tmp_path, standin):
    """The synthetic set built and screened (a stand-in ``screen.json``): ``verify`` passes."""
    build(tmp_path)
    d = tmp_path / SET / BODY
    write_screen(d)
    return d


def verify(d: Path) -> dict:
    return sets.verify(SET, BODY, d.parents[1])


def test_verify_passes_on_the_set_as_built(made):
    v = verify(made)
    assert v["ok"] and v["failures"] == [] and len(v["available"]) == 22 and v["not_available"] == {} and v["frozen"] is None
    assert "turn rate pi4: stride factor 0.7" in v["notes"]
    assert sets.main(["verify", "--set", SET, "--body", BODY, "--root", str(made.parents[1])]) == 0


def fails(made, edit, *, screen=None, where="transitions.json") -> list:
    if edit is not None:
        edit_json(made / where, edit)
    if screen is not None:
        write_screen(made, **screen)
    v = verify(made)
    assert not v["ok"]
    return v["failures"]


def test_verify_names_a_file_that_is_not_its_entrys(made):
    (made / "transitions" / "start.npz").write_bytes(b"other")
    (made / "look" / "v0.3.npz").unlink()
    f = verify(made)["failures"]
    assert len(f) == 2 and "v0.3: look/v0.3.npz is missing" in f[0] and "start: transitions/start.npz has sha256" in f[1]
    assert sets.status(SET, BODY, made.parents[1])["items"]["start"] == "array missing"


def test_verify_checks_the_two_formats_of_a_walk(made):
    a = dict(np.load(made / "look" / "v0.4.npz"))
    a["qdd"] = a["qdd"] + 1e-12
    sha = s2io.write_npz(made / "look" / "v0.4.npz", a)

    def edit(doc):
        doc["references"]["v0.4"]["look_sha256"] = sha

    edit_json(made / "set.json", lambda doc: doc["look_entries"]["v0.4"].update(sha256=sha))
    f = fails(made, edit, where="refs.json", screen={})
    assert f == ["v0.4: qdd of the Stage 1-format file is not the look-format file's"]


@pytest.mark.parametrize("edit,text", [
    (lambda t: t["stop2_v0.4"].update(pass_=False, gates=dict(ik_ok=True, join_b=False)), "stop2_v0.4: held as a reference with pass_ False (failing: join_b)"),
    (lambda t: t["stop2_v0.4"]["reference"].update(touchdowns_after_command=3), "stop2_v0.4: 3 touchdowns after the command, not 2"),
    (lambda t: t["stop2_v0.4"]["reference"].update(forward_stopped_strides=1.6), "stopped 1.6 strides after the command (at most 1.5)"),
    (lambda t: t["stop2_v0.4"]["reference"].update(accel_forward_lp1hz_peak_m_s2=1.51), "1 Hz forward acceleration 1.51 m/s^2 (at most 1.5)"),
    (lambda t: t["turn90_pi4_left"]["turn"].update(heading_total_rad=1.5707), "turn90_pi4_left: total heading 1.5707 rad"),
    (lambda t: t["turn90_pi4_left"]["reference"].update(heading_at_slowest_clock_deg=79.9), "heading_at_slowest_clock_deg 79.9 (at least 80)"),
    (lambda t: t["turn90_pi4_left"]["reference"].update(heading_at_bar_deg=79.0), "heading_at_bar_deg 79.0 (at least 80)"),
    (lambda t: t["turn90_pi4_left"].update(hip_yaw_range_use=0.871), "thigh-twist use 0.871 (at most 0.87)"),
    (lambda t: t["turn90_pi4_left"]["tightest_other"].update(margin_rad=0.0499), "(at least 0.05 rad)"),
    (lambda t: t["turn90_pi4_left"]["move"].update(rules=["S2A-stride"]), "S2A-appendage is not among its rules"),
    (lambda t: t["turn90_pi4_left"]["turn"].update(stride_rule=None), "holds no stride rule record with the outcome 'frozen'"),
    (lambda t: t["turn90_pi4_left"]["turn"].update(ramp_s=0.387), "its ramps last 0.387 s, not 0.375"),
    (lambda t: t["start"]["built_by"].update(blas_threads=2), "start: built with blas_threads 2 at accuracy 'r2'"),
    (lambda t: t["start"]["built_by"].update(accuracy="r1"), "start: built with blas_threads 1 at accuracy 'r1'"),
    (lambda t: t["start"].update(a1_over=None), "start: its entry's a1_over is None"),
    (lambda t: t["start"]["stage0"].update(blocks_freeze=[dict(limit="knee speed", value=1.01)]), "a Stage 0 gate blocked it where it was built"),
    (lambda t: t["start"].update(stage1_sha256="f" * 64, equals_stage1=True, stage1_twin="start"), "start: it does not hash to Stage 1's start"),
    (lambda t: t["start"]["item"].update(causal=True), "start: its entry is not of the definition made now"),
    (lambda t: t["start"]["inputs"].update(nominal="0" * 64), "start: its entry is not of the definition made now, or was built from another file"),
    (lambda t: t.pop("stop1_v0.4"), "stop1_v0.4: not built"),
    (lambda t: t.update(stop=dict(t["stop1_v0.4"], name="stop", group="stop1_v0.4")), "stop: a transition may not carry a name by which Stage 1"),
    (lambda t: t["start"].update(available=False, reason="x", numbers={}), "start: recorded as not available and names a file"),
    (lambda t: t["start"].update(available=False), "start: recorded as not available without a reason and its numbers"),
    (lambda t: t["start"].update(margin_rule="another rule"), "start: its entry's margin_rule is 'another rule', the set's is None"),
])
def test_verify_fails_on_a_move(made, edit, text):
    f = fails(made, lambda doc: edit(doc["transitions"]))
    assert any(text in x for x in f), f


@pytest.mark.parametrize("edit,text", [
    (lambda r: r["v0.4"].update(crouch_frac=0.03), "v0.4: its entry's crouch_frac is 0.03, set.json's 0.05"),
    (lambda r: r["v0.4"]["spec"].update(duty=0.56), "v0.4: the spec in its entry is not the set's at this speed"),
    (lambda r: r["v0.4"]["definition"]["spec"].update(duty=0.56), "v0.4: its entry is not of the definition made now"),
    (lambda r: r["v0.4"].update(stage1_sha256="f" * 64), "v0.4: Stage 1's file of the same spec is None, its entry records"),
    (lambda r: r["v0.4"].update(ladder_sha256="f" * 64), "v0.4: the look ladder's rung of the same spec is None"),
    (lambda r: r["v0.4"].update(speed_m_s=0.41), "v0.4: its entry's speed is 0.41 m/s, its name holds 0.4"),
    (lambda r: r["v0.4"].pop("built_by"), "v0.4: no built_by"),
    (lambda r: r["v0.4"].update(label="new"), "v0.4: its entry's label is 'new', the set's at this speed is None (STAGE2_75_PLAN.md A2)"),
    (lambda r: r["v0.3"].update(margin_rule="x"), "v0.3: its entry's margin_rule is 'x', the set's is None (STAGE2_75_PLAN.md A3"),
])
def test_verify_fails_on_a_walk(made, edit, text):
    f = fails(made, lambda doc: edit(doc["references"]), where="refs.json")
    assert any(text in x for x in f), f


@pytest.mark.parametrize("screen,text", [
    ({"blocks__start__verdict__blocks_freeze": [dict(limit="hip_pitch speed", value=1.03)]},
     "start: the screen blocks its freeze and the set holds it as a reference: hip_pitch speed 1.03"),
    ({"blocks__v0.4__verdict__blocks_freeze": [dict(limit="knee speed", value=1.01)]}, "v0.4: the screen blocks its freeze"),
    ({"blocks__start__arrays__sha256": "0" * 64}, "start: its screen block is of another file"),
    ({"blocks__v0.4__reference__sha256": "0" * 64}, "v0.4: its screen block names another Stage 1-format file"),
    ({"blocks__start__verdict__a1_over": True}, "start: its entry's a1_over is False, the screen's True"),
    ({"blocks__start__drives__tail__peak_ratio": 1.2}, "start: a neck or tail ratio is above 1 and it is not a turn with a1_over"),
    ({"blocks__turn90_pi4_left__drives__tail__peak_ratio": 1.2}, "turn90_pi4_left: a neck or tail ratio is above 1"),
    ({"blocks__v0.4__drives__neck__cont_ratio": 1.01}, "v0.4: a neck or tail ratio is above 1 on a walk"),
    ({"blocks__turn90_pi4_left__range__hip_yaw_range_use": 0.88}, "turn90_pi4_left: the screen reads a thigh-twist use of 0.88"),
])
def test_verify_reads_the_screen(made, screen, text):
    f = fails(made, None, screen=screen)
    assert any(text in x for x in f), f


def test_verify_wants_a_block_for_every_reference_and_none_beside(made):
    doc = json.loads((made / "screen.json").read_text())
    doc["blocks"]["ghost"] = doc["blocks"].pop("start")
    s2io.write_json(made / "screen.json", doc)
    f = verify(made)["failures"]
    assert "start: no block in screen.json (python -m designlab.stage2.setscreen run)" in f
    assert "ghost: screen.json holds a block of a reference the set does not hold" in f
    (made / "screen.json").unlink()
    f = verify(made)["failures"]
    assert f == ["screen.json is missing: the set is not screened (python -m designlab.stage2.setscreen run)"]


def test_a_turn_over_a_tail_limit_is_held_only_with_a1_over_and_the_appendage_rule(made):
    def a1(doc):
        doc["transitions"]["turn90_pi4_left"]["a1_over"] = True

    edit_json(made / "transitions.json", a1)
    write_screen(made, blocks__turn90_pi4_left__drives__tail__peak_ratio=1.2, blocks__turn90_pi4_left__verdict__a1_over=True)
    v = verify(made)
    assert v["ok"], v["failures"]
    assert any("turn90_pi4_left: a1_over" in x and "third reading only" in x for x in v["notes"])


def test_verify_reads_set_json(made):
    f = fails(made, lambda doc: doc.update(crouch_frac=0.03), where="set.json")
    assert any("set.json records crouch_frac 0.03, the set's specs [0.05]" in x for x in f) and len(f) == 7  # and six walks
    edit_json(made / "set.json", lambda doc: doc.update(crouch_frac=0.05))
    f = fails(made, lambda doc: doc["stride_rules"].pop("pi4"), where="set.json")
    assert f == ["turn rate pi4: set.json stride_rules holds no complete record"]
    build(made.parents[1])  # (nothing: the records are the builder's truth; put the record back by hand)
    rec = docs_of(made)["transitions.json"]["transitions"]["turn90_pi4_left"]["turn"]["stride_rule"]

    def lose_a_reading(doc):
        doc["stride_rules"]["pi4"] = json.loads(json.dumps(rec))
        doc["stride_rules"]["pi4"]["tried"][0]["tightest_other"]["right"] = None

    f = fails(made, lose_a_reading, where="set.json")
    assert f == ["turn rate pi4: the stride rule's row at f 0.7 lacks the right turn's twist use or tightest other joint"]
    f = fails(made, lambda doc: doc["stride_rules"]["pi4"].update(json.loads(json.dumps(rec)), references={"turn90_pi4_left": True}),
              where="set.json")
    assert any("turn rate pi4: the stride rule's record says" in x for x in f)
    edit_json(made / "set.json", lambda doc: doc["stride_rules"].update(pi4=rec))
    f = fails(made, lambda doc: doc["not_available"].update(start=dict(reason="x", numbers={})), where="set.json")
    assert f == ["start: set.json not_available lists it and its entry does not say so"]


def test_a_no_op_edit_to_moves_changes_nothing_but_a_note(made, monkeypatch, tmp_path):
    """After a comment is appended to (a copy of) ``moves.py``: ``verify`` passes and says "built by other code,
    arrays identical"; no entry, no hashed subset and no manifest moves."""
    manifest = sets.manifest_of(docs_of(made))
    copy = tmp_path / "moves.py"
    copy.write_text(s2io.module_path("moves.py").read_text() + "\n# a comment\n")
    real = sets.modules_now()
    key = "designlab/stage2/moves.py"
    assert key in real and s2io.sha256_file(copy) != real[key]
    monkeypatch.setattr(sets, "modules_now", lambda: dict(real, **{key: s2io.sha256_file(copy)}))
    before = mtimes(made)
    v = verify(made)
    assert v["ok"] and v["failures"] == []
    notes = [x for x in v["notes"] if "built by other code" in x]
    assert len(notes) == 22 and ("start: built by other code (not rebuilt: its arrays are its entry's and were not "
                                 "compared with what today's code builds; moves.py moved since)") in notes
    assert sets.manifest_of(docs_of(made)) == manifest and mtimes(made) == before
    assert build(made.parents[1])["ran"] == [] and mtimes(made) == before  # and nothing is built again for it


def test_the_hashed_subset_of_an_entry_ignores_labels_rules_and_provenance(made, tmp_path, monkeypatch):
    """``setio.entry_sha256`` on the builder's entries: a label, a rule text, ``look_sha256``, ``built_by``, the
    Stage 0 block or the stride rule's record change no hash; ``period_s``, ``move.command_phase`` and ``a1_over``
    do.  The manifest is the same from another working directory."""
    docs = docs_of(made)
    walk, turn = docs["refs.json"]["references"]["v0.4"], docs["transitions.json"]["transitions"]["turn90_pi4_left"]
    w0, t0 = setio.entry_sha256("walk", walk), setio.entry_sha256("turn", turn)
    for key, value in (("label", "new"), ("margin_rule", "x"), ("look_sha256", "0" * 64), ("built_by", {}),
                       ("stage0", {}), ("definition", {}), ("group", "g"), ("set", "other")):
        assert setio.entry_sha256("walk", dict(walk, **{key: value})) == w0, key
    for key, value in (("built_by", {}), ("stage0", {}), ("fallback", dict(step=1)), ("tightest_other", {}),
                       ("item", {}), ("inputs", {}), ("label", "x"), ("gates", {}), ("reference", {})):
        assert setio.entry_sha256("turn", dict(turn, **{key: value})) == t0, key
    assert setio.entry_sha256("turn", dict(turn, turn=dict(turn["turn"], stride_rule=None))) == t0
    assert setio.entry_sha256("walk", dict(walk, period_s=0.081)) != w0
    assert setio.entry_sha256("walk", dict(walk, crouch_frac=0.03)) != w0
    assert setio.entry_sha256("turn", dict(turn, move=dict(turn["move"], command_phase=0.0))) != t0
    assert setio.entry_sha256("turn", dict(turn, a1_over=True)) != t0
    assert setio.entry_sha256("turn", dict(turn, sha256="0" * 64)) != t0
    m = sets.manifest_of(docs)
    assert m["v0.4"] == dict(kind="walk", sha256=walk["sha256"], entry_sha256=w0, look_sha256=walk["look_sha256"])
    assert m["turn90_pi4_left"] == dict(kind="turn", sha256=turn["sha256"], entry_sha256=t0)
    assert {r["kind"] for r in m.values()} == set(sets.KINDS)
    monkeypatch.chdir(tmp_path)
    assert sets.manifest_of(docs_of(made)) == m


# ---------------------------------------------------------------- freeze
def test_freeze_appends_a_manifest_of_every_reference(made, legacy_ok, standin):
    root = made.parents[1]
    rec = sets.freeze(SET, BODY, root, log=lambda line: None)
    doc = docs_of(made)["set.json"]
    assert doc["freezes"] == [rec] and rec["index"] == 0 and rec["commit"] == "c" * 40
    assert set(rec) == {"index", "at", "commit", "manifest", "manifest_sha256", "legacy_check_sha256", "mixed_sources",
                        "dirty", "added"}
    assert rec["manifest"] == sets.manifest_of(docs_of(made)) and len(rec["manifest"]) == 22
    assert rec["manifest_sha256"] == s2io.sha256_json(rec["manifest"]) and rec["mixed_sources"] is False and rec["dirty"] is False
    assert rec["legacy_check_sha256"] == s2io.sha256_file(sets.LEGACY_CHECK) and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", rec["at"])
    fz = setio.frozen(doc)
    assert fz["index"] == 0 and fz["manifest_sha256"] == rec["manifest_sha256"] and fz["entries"] == rec["manifest"]
    # the bar's reader takes the frozen set: every file, its entry's hashed subset, the freeze
    files = [f"refs/{n}.npz" for n in sets.walk_names()] + ["transitions/turn90_pi4_left.npz", "transitions/start.npz"]
    inputs, _ = setio.read_inputs(made, files, require_frozen=True)
    assert inputs["freeze"] == dict(index=0, manifest_sha256=rec["manifest_sha256"])
    v = verify(made)
    assert v["ok"] and v["frozen"] == dict(index=0, manifest_sha256=rec["manifest_sha256"], entries=22)
    assert set(sets.status(SET, BODY, root)["items"].values()) == {"built (frozen)"}
    # a freeze that would add nothing appends nothing
    assert sets.freeze(SET, BODY, root, log=lambda line: None) == rec and len(docs_of(made)["set.json"]["freezes"]) == 1


def test_a_later_freeze_adds_entries_and_keeps_every_earlier_one(tmp_path, standin, legacy_ok):
    standin["fail"] = {"stop1_v0.4": sets.NO_GATES}
    build(tmp_path)
    d = tmp_path / SET / BODY
    write_screen(d)
    first = sets.freeze(SET, BODY, tmp_path, log=lambda line: None)
    assert len(first["manifest"]) == 21 and "stop1_v0.4" not in first["manifest"]
    assert any("stop1_v0.4: not available" in x for x in verify(d)["notes"])
    # the comparison stop is built later (its entry had not been made): the frozen entries are not written again
    standin["fail"] = {}
    frozen_before = {n: e for n, e in docs_of(d)["transitions.json"]["transitions"].items() if n != "stop1_v0.4"}
    edit_json(d / "transitions.json", lambda doc: doc["transitions"].pop("stop1_v0.4"))
    edit_json(d / "set.json", lambda doc: doc["not_available"].pop("stop1_v0.4"))
    refs_mtime = (d / "refs.json").stat().st_mtime_ns
    r = build(tmp_path)
    assert r["ran"] == ["stop1_v0.4"] and (d / "refs.json").stat().st_mtime_ns == refs_mtime
    now = docs_of(d)["transitions.json"]["transitions"]
    assert {n: e for n, e in now.items() if n != "stop1_v0.4"} == frozen_before and now["stop1_v0.4"]["available"] is True
    v = verify(d)
    assert not v["ok"] and v["failures"] == ["stop1_v0.4: no block in screen.json (python -m designlab.stage2.setscreen run)"]
    assert "stop1_v0.4: not in the latest freeze (index 0)" in v["notes"]
    write_screen(d)
    second = sets.freeze(SET, BODY, tmp_path, change_after_result="the comparison stop was added after task 3's first unit",
                         log=lambda line: None)
    assert second["index"] == 1 and second["added"] == ["stop1_v0.4"] and len(second["manifest"]) == 22
    assert {n: second["manifest"][n] for n in first["manifest"]} == first["manifest"]  # every earlier (name, sha256)
    assert second["change_after_result"] == "the comparison stop was added after task 3's first unit"
    assert "change_after_result" not in first and docs_of(d)["set.json"]["freezes"] == [first, second]
    assert verify(d)["ok"]


@pytest.mark.parametrize("label", ["after_check_episodes", "before_any_episode", "none", "two"])
def test_a_later_freeze_says_when_it_was_made(tmp_path, standin, legacy_ok, label):
    """A freeze that adds references to an earlier one is refused without one of the three statements (the plan's
    A13 and its confirmation 8 under rule R4); ``after_check_episodes`` is recorded as the runner reads it (true,
    with ``reason``) and is no change after a result."""
    standin["fail"] = {"stop1_v0.4": sets.NO_GATES}
    build(tmp_path)
    d = tmp_path / SET / BODY
    write_screen(d)
    first = sets.freeze(SET, BODY, tmp_path, log=lambda line: None)
    standin["fail"] = {}
    edit_json(d / "transitions.json", lambda doc: doc["transitions"].pop("stop1_v0.4"))
    edit_json(d / "set.json", lambda doc: doc["not_available"].pop("stop1_v0.4"))
    assert build(tmp_path)["ran"] == ["stop1_v0.4"]
    write_screen(d)
    root = ["freeze", "--set", SET, "--body", BODY, "--root", str(tmp_path)]
    if label == "none":
        with pytest.raises(sets.SetError, match="this freeze would add stop1_v0.4 to freeze 0: say when it is made, with "
                                                "--change-after-result REASON .* --after-check-episodes REASON .* or "
                                                "--before-any-episode"):
            sets.freeze(SET, BODY, tmp_path, log=lambda line: None)
        assert sets.main(root) == 1 and docs_of(d)["set.json"]["freezes"] == [first]
        return
    if label == "two":
        with pytest.raises(sets.SetError, match="--change-after-result and --after-check-episodes are given"):
            sets.freeze(SET, BODY, tmp_path, change_after_result="x", after_check_episodes="y", log=lambda line: None)
        with pytest.raises(SystemExit):  # and the command line takes one of them
            sets.main(root + ["--after-check-episodes", "y", "--before-any-episode"])
        assert docs_of(d)["set.json"]["freezes"] == [first]
        return
    if label == "after_check_episodes":
        assert sets.main(root + ["--after-check-episodes", "the comparison stop, after V0's check episodes"]) == 0
        second = docs_of(d)["set.json"]["freezes"][1]
        assert second["after_check_episodes"] is True and second["reason"] == "the comparison stop, after V0's check episodes"
        assert "change_after_result" not in second and "before_any_episode" not in second
    else:
        second = sets.freeze(SET, BODY, tmp_path, before_any_episode=True, log=lambda line: None)
        assert second["before_any_episode"] is True and not {"change_after_result", "after_check_episodes", "reason"} & set(second)
    assert second["index"] == 1 and second["added"] == ["stop1_v0.4"] and docs_of(d)["set.json"]["freezes"][0] == first
    assert verify(d)["ok"]


def test_verify_and_freeze_cite_the_bars_kinematic_check(made, legacy_ok):
    """``refcheck.json`` beside the set's records (the bar's ``refcheck``: each reference played kinematically
    through the bar's own readings): ``verify`` prints what fails as notes, never as a failure; a freeze cites the
    record and refuses one that is not of the files being frozen."""
    root = made.parents[1]
    assert (f"no refcheck.json beside the set's records: its references were not played through the bar's own "
            f"readings. {sets.REFCHECK_HOW}") in verify(made)["notes"]  # (ruling 40: the placed refcheck on a look set)
    docs = docs_of(made)
    files = {"refs/v0.4.npz": dict(sha256=docs["refs.json"]["references"]["v0.4"]["sha256"]),
             "transitions/turn90_pi4_left.npz": dict(sha256=docs["transitions.json"]["transitions"]["turn90_pi4_left"]["sha256"])}
    doc = dict(schema="bar-refcheck-1", set=SET, body=BODY, inputs=dict(dir="x", files=files), pass_=False, items={
        "walk/v0.4": dict(status="done", pass_=True, ok=dict(along=True, toe_clearance_own_swing=True)),
        "turn/q90_left": dict(status="done", pass_=False, ok=dict(heading=True, toe_clearance_own_swing=False),
                              toe=dict(toe_clearance_own_swing_H=0.0323450669)),
        "turn/t90_1_left": dict(status="not run", pass_=None, reason="no reference")})
    s2io.write_json(made / "refcheck.json", doc)
    state = sets.refcheck_state(made, setio.load(made))
    assert state["current"] and state["pass_"] is False and state["not_run"] == ["turn/t90_1_left"]
    assert state["failing"] == {"turn/q90_left": dict(fails=["toe_clearance_own_swing"], toe_clearance_own_swing_H=0.0323450669)}
    v = verify(made)
    note = [x for x in v["notes"] if x.startswith("refcheck")]
    assert v["ok"] and len(note) == 1 and "turn/q90_left FAILS toe_clearance_own_swing (own-swing toe clearance 0.03235 H)" in note[0]
    rec = sets.freeze(SET, BODY, root, log=lambda line: None)
    assert rec["refcheck"] == dict(file=state["file"], sha256=s2io.sha256_file(made / "refcheck.json"), pass_=False,
                                   failing=state["failing"], not_run=["turn/t90_1_left"])
    assert verify(made)["ok"]
    # a record of other files: a note, and no freeze cites it
    edit_json(made / "set.json", lambda d_: d_.update(freezes=[]))
    files["refs/v0.4.npz"]["sha256"] = "0" * 64
    s2io.write_json(made / "refcheck.json", doc)
    v = verify(made)
    assert v["ok"] and any("is not current: 1 of the files it read are not the set's now (refs/v0.4.npz). "
                           + sets.REFCHECK_HOW in x for x in v["notes"])
    with pytest.raises(sets.SetError, match="the refcheck record .* is not of the files being frozen") as refused:
        sets.freeze(SET, BODY, root, log=lambda line: None)
    assert str(refused.value).endswith(f"(refs/v0.4.npz)): make it again. {sets.REFCHECK_HOW}")
    with pytest.raises(sets.SetError, match="no such refcheck record"):
        sets.freeze(SET, BODY, root, refcheck=made / "none.json", log=lambda line: None)
    # every reference passing: one line says so
    files["refs/v0.4.npz"]["sha256"] = docs["refs.json"]["references"]["v0.4"]["sha256"]
    doc["items"]["turn/q90_left"].update(pass_=True, ok=dict(heading=True, toe_clearance_own_swing=True))
    s2io.write_json(made / "elsewhere.json", dict(doc, pass_=True))
    (made / "refcheck.json").unlink()
    rec = sets.freeze(SET, BODY, root, refcheck=made / "elsewhere.json", log=lambda line: None)
    assert rec["refcheck"]["pass_"] is True and rec["refcheck"]["failing"] == {}
    assert sets.refcheck_lines(sets.refcheck_state(made, setio.load(made), made / "elsewhere.json"))[0].endswith(
        "every reference that was played passes its own bar item")


#: LT3's check's suggested line for the look sets' freeze text (stage2/design/checkpoint/reviews/
#: roundC__check_LT3_17nov.md, finding 3), which ruling 40 of 17 Nov 2026 puts into sets.py word for word.
LT3_REFCHECK_LINE = (
    "On a look set or a version of one (V6, V6noF5, V6.2, ...) the record is made with `python -m "
    "designlab.stage2.gates2 refcheck --set S --body B --out <the sets' root>` (bar refcheck's record made inside "
    "lookbar.look_set_placement, its turns and stop at join rule J1's leave point, the only kind G9-ref part (a) "
    "reads); on any other set with `python -m designlab.stage2.bar refcheck --set S --body B --out <the sets' root>`.")


def test_the_freeze_text_names_the_placed_refcheck_for_a_look_set(capsys):
    """Ruling 40 of 17 Nov 2026 (LT3's check, finding 3): the six places of ``sets.py`` that named ``bar refcheck``
    alone name the placed refcheck for a look set in the check's words: the module docstring (section 3),
    ``REFCHECK``'s comment and ``REFCHECK_HOW`` beside it, ``verify``'s note on a set with no record and
    ``refcheck_lines`` on a record that is not current, ``freeze``'s refusal of such a record (the three read in
    ``test_verify_and_freeze_cite_the_bars_kinematic_check``), and the ``--refcheck`` help.  No place names ``bar
    refcheck`` alone any more."""
    assert sets.REFCHECK_HOW == LT3_REFCHECK_LINE
    check = ROOT / "stage2" / "design" / "checkpoint" / "reviews" / "roundC__check_LT3_17nov.md"
    if check.is_file():
        assert LT3_REFCHECK_LINE in check.read_text()
    flat = lambda s: " ".join(s.replace("``", "`").split())  # noqa: E731
    bare = lambda s: "".join(s.split())  # noqa: E731  (argparse wraps its help: compared without whitespace)
    assert LT3_REFCHECK_LINE[:-1] + " (`REFCHECK_HOW`; ruling 40 of 17 Nov 2026)." in flat(sets.__doc__)
    src = (ROOT / "designlab" / "stage2" / "sets.py").read_text()
    lines = src.splitlines()
    i = lines.index('REFCHECK = "refcheck.json"')
    j = i
    while lines[j - 1].startswith("#:"):
        j -= 1
    comment = flat(" ".join(x[2:] for x in lines[j:i]))
    for words in ("ruling 40 of 17 Nov 2026, in the words of LT3's check", "on a look set or a version of one (V6, "
                  "V6noF5, V6.2, ...) `python -m designlab.stage2.gates2 refcheck --set S --body B --out <the sets' "
                  "root>`", "on any other set `python -m designlab.stage2.bar refcheck --set S --body B --out <the "
                  "sets' root>`"):
        assert words in comment, words
    assert lines[i + 1].startswith("REFCHECK_HOW = (")
    for old in ("made by ``python -m designlab.stage2.bar refcheck --out", "(python -m designlab.stage2.bar refcheck --out",
                "(python -m designlab.stage2.bar refcheck)", "run python -m designlab.stage2.bar refcheck again",
                "the record of bar refcheck to cite"):
        assert old not in src, old
    with pytest.raises(SystemExit):
        sets.main(["freeze", "--help"])
    assert bare("the refcheck record to cite (default: the set directory's refcheck.json, where there is one). "
                + LT3_REFCHECK_LINE) in bare(capsys.readouterr().out)


def test_a_frozen_entry_is_never_dropped_or_changed(made, legacy_ok, standin):
    root = made.parents[1]
    first = sets.freeze(SET, BODY, root, log=lambda line: None)
    # its entry changed (a number an episode reads)
    edit_json(made / "refs.json", lambda doc: doc["references"]["v0.4"].update(period_s=0.081))
    v = verify(made)
    assert any("v0.4: its entry's hashed subset is" in x and "the freeze holds" in x for x in v["failures"])
    with pytest.raises(sets.SetError, match="sets verify: v0.4: its entry's hashed subset") as err:
        sets.freeze(SET, BODY, root, log=lambda line: None)
    assert "v0.4: an earlier freeze holds sha256" in str(err.value) and "takes a new name or a new set version" in str(err.value)
    with pytest.raises(tune.InputError, match="hashed subset"):
        setio.read_inputs(made, ["refs/v0.4.npz"], require_frozen=True)
    edit_json(made / "refs.json", lambda doc: doc["references"]["v0.4"].update(period_s=0.08))
    assert verify(made)["ok"]
    # its entry gone
    edit_json(made / "transitions.json", lambda doc: doc["transitions"].pop("start"))
    with pytest.raises(sets.SetError, match="start: an earlier freeze holds it and the set no longer does"):
        sets.freeze(SET, BODY, root, log=lambda line: None)
    assert any("start: the latest freeze lists it and the set has no such entry" in x for x in verify(made)["failures"])
    assert docs_of(made)["set.json"]["freezes"] == [first]
    # ... and a build says so and builds nothing (it could only put the array back, never the entry)
    calls, before = len(standin["calls"]), mtimes(made)
    with pytest.raises(sets.SetError, match="the latest freeze lists start and transitions.json holds no such entry: "
                                            "restore the committed record"):
        build(root)
    assert len(standin["calls"]) == calls and mtimes(made) == before
    assert sets.main(["build", "--set", SET, "--body", BODY, "--root", str(root), "--procs", "0"]) == 1


def _g9_ref(path: Path, made: Path, refs_named, blocked=(), **over) -> Path:
    """A G9-ref record of the synthetic set's latest freeze (``gates2``'s shape: its own sha256) whose part (a) names
    ``refs_named`` among the failing items' references still available."""
    sd = docs_of(made)["set.json"]
    last = sd["freezes"][-1]
    doc = dict(schema=sets.G9_REF_SCHEMA, gate="G9-ref", body=BODY, set=SET,
               set_freeze=dict(index=last["index"], manifest_sha256=last["manifest_sha256"]),
               parts=dict(a=dict(ok=False, still_available=sorted(refs_named), blocked=list(blocked), failing_references=[
                   dict(item=f"item/{r}", reference=r, failing=["toe_clearance"], entry_available=True,
                        in_latest_freeze=True, still_available=True) for r in refs_named])), pass_=False)
    doc.update(over)
    doc["sha256"] = s2io.sha256_json(doc)
    s2io.write_json(path, doc)
    return path


def test_the_g9_ref_schema_is_the_gates_own():
    """``sets.G9_REF_SCHEMA`` is ``gates2.SCHEMAS["g9_ref"]`` (read in a child: importing ``gates2`` selects MuJoCo's
    GL backend)."""
    code = "from designlab.stage2 import gates2; print(gates2.SCHEMAS['g9_ref'])"
    env = dict(os.environ, PYTHONPATH=str(ROOT), MUJOCO_GL="disable")
    r = subprocess.run([PY, "-c", code], capture_output=True, text=True, env=env, cwd=ROOT, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip().splitlines()[-1] == sets.G9_REF_SCHEMA


def test_a_freeze_drops_an_entry_marked_on_the_g9_ref_record_that_named_it(made, legacy_ok, standin, tmp_path, capsys):
    """Ruling 32 (A) of 17 Nov 2026 (after ruling 14): ``sets mark`` marks a frozen entry that a G9-ref record of the
    latest freeze names as still available not available, with the lead's reason and that record (path and sha256);
    the next freeze drops it and records the drop with the mark and the record.  The earlier freeze is not rewritten,
    the entry keeps its definition and provenance, a build builds nothing for it, and ``verify`` passes before and
    after the drop.  A turn's stride rule record no longer holds the marked direction as a reference."""
    root = made.parents[1]
    first = sets.freeze(SET, BODY, root, log=lambda line: None)
    before = docs_of(made)
    rec = _g9_ref(tmp_path / "g9_ref.json", made, ["transitions/stop1_v0.4.npz", "transitions/turn90_pi4_left.npz"])
    argv = ["mark", "--set", SET, "--body", BODY, "--root", str(root), "--name", "stop1_v0.4", "--reason",
            "the bar's reference check fails its toe clearance", "--g9-ref", str(rec)]
    assert sets.main(argv) == 0 and "the next freeze drops it" in capsys.readouterr().out
    sets.mark(SET, BODY, "turn90_pi4_left", "the same, on the left turn", rec, root, log=lambda line: None)
    docs = docs_of(made)
    cite = dict(path=s2io.portable(str(rec.resolve())), sha256=s2io.sha256_file(rec))
    for name in ("stop1_v0.4", "turn90_pi4_left"):
        e, old = docs["transitions.json"]["transitions"][name], before["transitions.json"]["transitions"][name]
        mk = e["marked"]
        assert e["available"] is False and "file" not in e and "sha256" not in e and mk["g9_ref"] == cite
        assert mk["was"] == dict(file=old["file"], sha256=old["sha256"], entry_sha256=setio.entry_sha256("move", old))
        assert mk["freeze"] == dict(index=0, manifest_sha256=first["manifest_sha256"]) and mk["rule"] == sets.MARK_RULE
        assert e["numbers"] == dict(g9_ref=[dict(item=f"item/transitions/{name}.npz", failing=["toe_clearance"])])
        assert e["definition"] == old["definition"] and e["built_by"] == old["built_by"] and e["item"] == old["item"]
        assert docs["set.json"]["not_available"][name] == dict(reason=e["reason"], numbers=e["numbers"])
        assert cite["path"] in e["reason"]
    assert docs["set.json"]["stride_rules"]["pi4"]["references"] == {"turn90_pi4_left": False, "turn90_pi4_right": True}
    assert docs["set.json"]["freezes"] == [first]  # nothing of the freeze moved
    v = verify(made)
    assert v["ok"], v["failures"]
    assert any(x.startswith("stop1_v0.4: marked not available on the G9-ref record") and "the next freeze drops it"
               in x for x in v["notes"])
    assert sets.status(SET, BODY, root)["items"]["stop1_v0.4"].startswith("not available: the bar's reference check")
    calls = len(standin["calls"])
    build(root)  # a build builds nothing for a marked entry (nor for the rest: all current)
    assert len(standin["calls"]) == calls
    second = sets.freeze(SET, BODY, root, log=lambda line: None)
    doc = docs_of(made)["set.json"]
    assert doc["freezes"] == [first, second] and second["index"] == 1 and second["added"] == []
    assert set(first["manifest"]) - set(second["manifest"]) == {"stop1_v0.4", "turn90_pi4_left"}
    for name in ("stop1_v0.4", "turn90_pi4_left"):
        row = first["manifest"][name]
        assert second["dropped"][name] == dict(kind=row["kind"], sha256=row["sha256"], entry_sha256=row["entry_sha256"],
                                               reason=docs["transitions.json"]["transitions"][name]["marked"]["reason"],
                                               marked_at=docs["transitions.json"]["transitions"][name]["marked"]["at"],
                                               g9_ref=cite, rule=sets.MARK_RULE)
    v = verify(made)
    assert v["ok"] and v["frozen"]["index"] == 1 and "stop1_v0.4" in v["not_available"]
    # a freeze with nothing to add or drop appends nothing
    assert sets.freeze(SET, BODY, root, log=lambda line: None) == second


def test_a_freeze_that_drops_an_entry_nobody_marked_is_refused(made, legacy_ok, standin, tmp_path):
    """Ruling 32 (A) opens one door only: an entry of the latest freeze that the set holds as not available without
    the mark of a G9-ref record is still refused by ``freeze`` (and by ``verify`` and ``build``), and so is a mark that
    rests on a record the set cannot be held to (another set, freeze or entry; a blocking item; a sha256 not its
    own), or on an entry another reference was built from."""
    root = made.parents[1]
    sets.freeze(SET, BODY, root, log=lambda line: None)
    rec = _g9_ref(tmp_path / "g9_ref.json", made, ["transitions/stop1_v0.4.npz", "refs/v0.3.npz"])
    refuse = [  # (the record, the entry, what the refusal says)
        (rec, "stop2_v0.4", "does not name transitions/stop2_v0.4.npz"),
        (rec, "v0.3", "were built from v0.3"),
        (_g9_ref(tmp_path / "other.json", made, ["transitions/stop1_v0.4.npz"], set="V0"), "stop1_v0.4", "of V0/"),
        (_g9_ref(tmp_path / "old.json", made, ["transitions/stop1_v0.4.npz"], set_freeze=dict(index=0, manifest_sha256="0" * 64)),
         "stop1_v0.4", "made on another freeze"),
        (_g9_ref(tmp_path / "blk.json", made, ["transitions/stop1_v0.4.npz"], blocked=["walk/v0.4"]), "stop1_v0.4",
         "failing blocking items"),
        (tmp_path / "none.json", "stop1_v0.4", "no G9-ref record can be read"),
        (rec, "stop9", "has no entry"),
    ]
    bad = _g9_ref(tmp_path / "bad.json", made, ["transitions/stop1_v0.4.npz"])
    edit_json(bad, lambda doc: doc.update(pass_=True))  # its sha256 is no longer its own
    refuse.append((bad, "stop1_v0.4", "not a G9-ref record"))
    before = {n: (made / n).read_bytes() for n in ("refs.json", "transitions.json", "set.json")}
    for path, name, text in refuse:
        with pytest.raises(sets.SetError, match=re.escape(text)):
            sets.mark(SET, BODY, name, "a reason", path, root, log=lambda line: None)
    with pytest.raises(sets.SetError, match="needs the lead's reason"):
        sets.mark(SET, BODY, "stop1_v0.4", " ", rec, root, log=lambda line: None)
    assert before == {n: (made / n).read_bytes() for n in before}  # nothing was written
    # the entry set not available by hand, with no mark: verify, freeze and build refuse it as before
    def unmark(doc):
        e = doc["transitions"]["stop1_v0.4"]
        for k in ("file", "sha256"):
            e.pop(k)
        e.update(available=False, reason="by hand", numbers={})
    edit_json(made / "transitions.json", unmark)
    edit_json(made / "set.json", lambda doc: doc["not_available"].update({"stop1_v0.4": dict(reason="by hand",
                                                                                           numbers={})}))
    assert "stop1_v0.4: the latest freeze lists it and its entry is not available" in verify(made)["failures"]
    with pytest.raises(sets.SetError, match="stop1_v0.4: an earlier freeze holds it and the set no longer does"):
        sets.freeze(SET, BODY, root, log=lambda line: None)
    with pytest.raises(sets.SetError, match="the latest freeze lists stop1_v0.4 and transitions.json holds it as not "
                                            "available"):
        build(root)
    assert len(docs_of(made)["set.json"]["freezes"]) == 1


@pytest.mark.parametrize("set_name", list(sets.SETS))
def test_the_round_sets_records_hold_no_mark_and_no_drop(set_name):
    """Ruling 32 (A) rewrites no record of the round's three frozen sets: none of their entries is marked, none of
    their freezes drops one, and their rules are the ones a new set of theirs gets (no mark rule among them)."""
    d = setio.ROOT / set_name / BODY
    if not (d / "set.json").exists():
        pytest.skip(f"the {set_name} set is not built (stage2/sets/{set_name})")
    docs = docs_of(d)
    entries = {**docs["refs.json"]["references"], **docs["transitions.json"]["transitions"]}
    assert not any("marked" in e for e in entries.values())
    assert docs["set.json"]["freezes"] and not any("dropped" in f for f in docs["set.json"]["freezes"])
    assert "S2S-mark" not in docs["set.json"]["rules"] and sets.MARK_RULE not in json.dumps(docs)


def test_build_on_a_frozen_set_only_puts_matching_arrays_back(made, legacy_ok, standin):
    """A missing array of a frozen entry is rebuilt and placed only when its sha256 is the manifest's; no record is
    written (mtimes unchanged); a rebuild to another sha256 fails with both hashes and places nothing."""
    root = made.parents[1]
    sets.freeze(SET, BODY, root, log=lambda line: None)
    whole_ = tree(made)
    for rel in ("refs/v0.35.npz", "look/v0.4.npz", "transitions/turn90_0.5_right.npz", "transitions/start.npz"):
        (made / rel).unlink()
    (made / "transitions" / "stop2_v0.4.npz").write_bytes(b"not the frozen file")
    assert sets.status(SET, BODY, root)["items"]["start"] == "array missing"
    json_before = {n: (made / n).stat().st_mtime_ns for n in ("refs.json", "transitions.json", "set.json", "screen.json")}
    r = build(root)
    assert r["ran"] == ["v0.35", "v0.4", "start", "stop2_v0.4", "turn90_0.5"] and r["mismatch"] == []
    assert sorted(r["restored"]) == ["start", "stop2_v0.4", "turn90_0.5_left", "turn90_0.5_right", "v0.35", "v0.4"]
    assert tree(made) == whole_ and verify(made)["ok"]
    assert {n: (made / n).stat().st_mtime_ns for n in json_before} == json_before
    assert not (made / "partial").is_dir() or not list((made / "partial").iterdir())
    # other code now makes another array: nothing is placed, nothing is written, the build fails with both hashes
    frozen_sha = docs_of(made)["refs.json"]["references"]["v0.35"]["sha256"]
    (made / "refs" / "v0.35.npz").unlink()
    (made / "transitions" / "start.npz").unlink()
    standin["salt"] = "other code"
    with pytest.raises(sets.SetError) as err:
        build(root)
    text = str(err.value)
    assert "a frozen entry was not reproduced; nothing of it was written" in text
    assert f"v0.35: the rebuilt file is not the frozen one (frozen sha256 {frozen_sha}, built " in text and "start:" in text
    assert not (made / "refs" / "v0.35.npz").exists() and not (made / "transitions" / "start.npz").exists()
    assert {n: (made / n).stat().st_mtime_ns for n in json_before} == json_before
    assert not list((made / "partial").glob("*.json")) and not (made / ".lock").exists()
    # and a frozen entry that today's code would not hold at all
    standin["salt"] = ""
    standin["fail"] = {"start": sets.NO_GATES}
    with pytest.raises(sets.SetError, match="start: built now it has no reference: no transition within the gates"):
        build(root, only=["start"])
    assert docs_of(made)["transitions.json"]["transitions"]["start"]["available"] is True
    standin["fail"] = {}
    assert build(root)["mismatch"] == [] and tree(made) == whole_


@pytest.mark.parametrize("what", ["verify", "legacy", "legacy_moves", "screen_refused"])
def test_freeze_refuses(made, legacy_ok, monkeypatch, what):
    root = made.parents[1]
    if what == "verify":
        (made / "transitions" / "start.npz").write_bytes(b"other")
        match = "sets verify: start: transitions/start.npz has sha256"
    elif what == "legacy":
        edit_json(sets.LEGACY_CHECK, lambda doc: doc["bodies"][BODY60]["names"]["stop"].update(equal=False))
        match = "the legacy check is not current: 60-N-T-10 is not complete with every transition equal"
    elif what == "legacy_moves":
        edit_json(sets.LEGACY_CHECK, lambda doc: doc.update(moves_sha256="0" * 64))
        match = "the legacy check is not current: it was made by another moves.py"
    else:
        edit_json(made / "screen.json", lambda doc: doc.update(refused={"stop1_v0.4": "ends inside a swing"}))
        match = "the screen refused stop1_v0.4: screen.json is not complete"
    with pytest.raises(sets.SetError, match=match):
        sets.freeze(SET, BODY, root, log=lambda line: None)
    assert docs_of(made)["set.json"]["freezes"] == []
    assert sets.main(["freeze", "--set", SET, "--body", BODY, "--root", str(root)]) == 1


def test_freeze_without_a_legacy_record(made, legacy_ok, monkeypatch, tmp_path):
    monkeypatch.setattr(sets, "LEGACY_CHECK", tmp_path / "none.json")
    assert sets.legacy_state() == dict(ok=False, why=f"{tmp_path / 'none.json'} is missing", sha256=None, moves_sha256=None)
    with pytest.raises(sets.SetError, match="the legacy check is not current"):
        sets.freeze(SET, BODY, made.parents[1], log=lambda line: None)


def test_freeze_refuses_mixed_sources_and_a_dirty_tree_unless_told(tmp_path, standin, legacy_ok):
    build(tmp_path, only=sets.walk_names())
    standin["sources"] = dict(sets.modules_now(), **{"designlab/stage2/sets.py": "0" * 64})  # the moves: other code
    build(tmp_path)
    d = tmp_path / SET / BODY
    write_screen(d)
    v = verify(d)
    assert v["ok"] and any("built from 2 differing source states" in x and "--allow-mixed" in x for x in v["notes"])
    with pytest.raises(sets.SetError, match="built from differing source states"):
        sets.freeze(SET, BODY, tmp_path, log=lambda line: None)
    legacy_ok.update(dirty=True, dirty_paths=[" M hardware/trex_form_study/designlab/stage2/moves.py"])
    with pytest.raises(sets.SetError, match="the tree is dirty .* M hardware/trex_form_study/designlab/stage2/moves.py"):
        sets.freeze(SET, BODY, tmp_path, allow_mixed=True, log=lambda line: None)
    rec = sets.freeze(SET, BODY, tmp_path, allow_mixed=True, allow_dirty=True, log=lambda line: None)
    assert rec["mixed_sources"] is True and rec["dirty"] is True
    assert verify(d)["ok"]  # the manifest says so
    # a freeze that hides it is caught
    edit_json(d / "set.json", lambda doc: doc["freezes"][0].update(mixed_sources=False))
    assert any("differing sources and the freeze does not say mixed_sources" in x for x in verify(d)["failures"])
    edit_json(d / "set.json", lambda doc: doc["freezes"][0].update(mixed_sources=True, manifest_sha256="0" * 64))
    assert "the latest freeze's manifest_sha256 is not its manifest's" in verify(d)["failures"]


def test_entries_built_on_a_dirty_tree_need_the_flag(tmp_path, standin, legacy_ok):
    real = fake["built_by"]
    fake["built_by"] = lambda: dict(real(), dirty=True, dirty_paths=[" M x.py"])
    try:
        build(tmp_path)
    finally:
        fake["built_by"] = real
    d = tmp_path / SET / BODY
    write_screen(d)
    assert any("22 entries were built on a dirty tree" in x for x in verify(d)["notes"])
    with pytest.raises(sets.SetError, match="built on a dirty tree: nominal"):
        sets.freeze(SET, BODY, tmp_path, log=lambda line: None)
    assert sets.freeze(SET, BODY, tmp_path, allow_dirty=True, log=lambda line: None)["dirty"] is True


def test_a_tree_git_cannot_read_is_not_a_clean_tree(made, legacy_ok):
    legacy_ok.update(commit=None, dirty=None, dirty_paths=None)
    with pytest.raises(sets.SetError, match="git cannot tell the state of the tree"):
        sets.freeze(SET, BODY, made.parents[1], log=lambda line: None)
    rec = sets.freeze(SET, BODY, made.parents[1], allow_dirty=True, log=lambda line: None)
    assert rec["commit"] is None and rec["dirty"] is True


def test_freeze_wants_every_entry_built_by_the_legacy_checks_moves(tmp_path, standin, legacy_ok, monkeypatch):
    """The plan's A4: ``legacy_check.json``'s ``moves_sha256`` is the sha256 of the ``moves.py`` that built the sets.
    A set built wholly by another ``moves.py`` is one source state, so it is not "mixed": the freeze refuses it all
    the same, with no flag to waive it; an entry with no reference gets the provenance note too, and is built again
    by today's code while the set is not frozen."""
    other = dict(sets.modules_now(), **{"designlab/stage2/moves.py": "0" * 64})
    standin["sources"] = other
    standin["fail"] = {"turn90_1": "no stride factor"}
    build(tmp_path)
    d = tmp_path / SET / BODY
    write_screen(d)
    v = verify(d)
    notes = [x for x in v["notes"] if "built by other code" in x]
    assert v["ok"] and len(notes) == 22 and not any("differing source states" in x for x in v["notes"])
    assert ("turn90_1_left: built by other code (not rebuilt: its verdict is that code's; moves.py moved since)") in notes
    for flags in (dict(), dict(allow_mixed=True, allow_dirty=True)):
        with pytest.raises(sets.SetError, match=r"22 entries were built by another moves.py than the legacy check's "
                                                r"\(nominal, nominal_to_v0.3, .*STAGE2_75_PLAN.md A4"):
            sets.freeze(SET, BODY, tmp_path, log=lambda line: None, **flags)
    assert docs_of(d)["set.json"]["freezes"] == []
    # today's code builds the entry with no reference again, and nothing else (the arrays pin the others)
    standin["sources"], calls = None, len(standin["calls"])
    r = build(tmp_path)
    assert r["ran"] == ["turn90_1"] and standin["calls"][calls:] == ["turn90_1"]
    assert build(tmp_path)["ran"] == []
    by = docs_of(d)["transitions.json"]["transitions"]["turn90_1_left"]["built_by"]["stage2_modules"]
    assert by == sets.modules_now()
    # ... while a frozen set's records are not built again for a change of code
    edit_json(d / "transitions.json", lambda doc: [e["built_by"].update(stage2_modules=sets.modules_now())
                                                   for e in doc["transitions"].values()])
    edit_json(d / "refs.json", lambda doc: [e["built_by"].update(stage2_modules=sets.modules_now())
                                            for e in doc["references"].values()])
    sets.freeze(SET, BODY, tmp_path, log=lambda line: None)
    real = sets.modules_now()
    monkeypatch.setattr(sets, "modules_now", lambda: dict(real, **{"designlab/stage2/sets.py": "1" * 64}))
    assert build(tmp_path)["ran"] == [] and len(standin["calls"]) == calls + 1
    assert any("turn90_1_left: built by other code (not rebuilt: its verdict is that code's; sets.py moved since)" in x
               for x in verify(d)["notes"])


# ---------------------------------------------------------------- the fallback rule and the turn rule, scripted
class Script:
    """``sets._Job.move`` scripted: each call takes the next (pass, Stage 0 blocks, flags) of its move's name."""

    def __init__(self, job, plan: dict):
        self.job, self.plan, self.calls = job, {k: list(v) for k, v in plan.items()}, []

    def __call__(self, item, mv, ref_a, ref_b, e_a, e_b, twin_a, twin_b, depends):
        step = self.plan[mv.name].pop(0)
        self.calls.append((mv.name, mv.builder, dict(mv.args)))
        self.job.out["files"][f"transitions/{mv.name}.npz"] = "s" * 64
        failed = list(step.get("failed", ()))
        blocks = list(step.get("blocks", ()))
        entry = dict(name=mv.name, sha256="s" * 64, file=f"transitions/{mv.name}.npz", available=True,
                     pass_=not failed, gates={g: False for g in failed} | dict(ik_ok=True), item=item.definition,
                     group=item.group, inputs=self.job.inputs(depends), definition=mv.definition(), built_by={},
                     ik_ok=step.get("ik_ok", True), hip_yaw_range_use=step.get("twist", 0.7), a1_over=bool(step.get("a1")),
                     fallback=None, stage0=dict(blocks_freeze=blocks, text="x"), turn=dict(stride_factor=mv.args.get("stride_factor")))
        other = dict(joint="left_hip_roll", end="min", margin_rad=step.get("margin", 0.12))
        return entry, dict(ok=not failed and not blocks, over_leg_drive=bool(step.get("leg")), a1_over=bool(step.get("a1")),
                           tightest_other=other, failed=failed, blocks=blocks)


@pytest.fixture
def job_of(tmp_path, standin, monkeypatch):
    """A ``sets._Job`` on the synthetic set's walks, with ``move`` scripted and no body built."""
    build(tmp_path, only=sets.walk_names())

    def make(name: str, plan: dict):
        its = {i.name: i for i in sets.items(SET, BODY)}
        job = sets._Job(SET, BODY, its[name].group, tmp_path)
        job.scratch.mkdir(parents=True, exist_ok=True)
        script = Script(job, plan)
        job.move = script
        monkeypatch.setattr(sets.refs.Reference, "load", classmethod(lambda cls, path: None))
        return job, script, its

    return make


PASS, FAIL = dict(), dict(failed=["join_b"])


def test_fallback_a_change_within_its_gates_has_none(job_of):
    job, script, its = job_of("nominal_to_v0.45", {"nominal_to_v0.45": [PASS]})
    sets._produce_straight(job, its["nominal_to_v0.45"])
    e = job.out["transitions"]["nominal_to_v0.45"]
    assert e["available"] and e["fallback"] is None and e["ramp_strides"] == 1.0 and list(job.out["transitions"]) == ["nominal_to_v0.45"]
    assert script.calls == [("nominal_to_v0.45", "change", dict(ramp_strides=1.0))] and job.out["blocks_set"] is None


def test_fallback_step_1_is_a_ramp_of_two_strides_under_the_same_name(job_of):
    job, script, its = job_of("nominal_to_v0.45", {"nominal_to_v0.45": [FAIL, PASS]})
    sets._produce_straight(job, its["nominal_to_v0.45"])
    e = job.out["transitions"]["nominal_to_v0.45"]
    assert [c[2] for c in script.calls] == [dict(ramp_strides=1.0), dict(ramp_strides=2.0)]
    assert e["available"] and e["ramp_strides"] == 2.0 and e["fallback"]["step"] == 1 and e["fallback"]["rule"] == "S2S-fallback"
    assert e["fallback"]["text"] == sets.RULES["S2S-fallback"] and len(e["fallback"]["tried"]) == 1
    assert e["fallback"]["tried"][0]["failed_gates"] == ["join_b"] and e["fallback"]["tried"][0]["ramp_strides"] == 1.0
    assert job.out["files"] == {"transitions/nominal_to_v0.45.npz": "s" * 64}


def test_fallback_a_blocking_stage0_gate_is_retried_like_a_failed_gate(job_of):
    block = dict(blocks=[dict(limit="knee speed", value=1.02)])
    job, script, its = job_of("nominal_to_v0.3", {"nominal_to_v0.3": [block, PASS]})
    sets._produce_straight(job, its["nominal_to_v0.3"])
    e = job.out["transitions"]["nominal_to_v0.3"]
    assert e["fallback"]["step"] == 1 and e["fallback"]["tried"][0]["stage0"]["blocks_freeze"] == block["blocks"]


def test_fallback_step_2_is_the_change_from_the_neighbouring_rung(job_of):
    job, script, its = job_of("nominal_to_v0.5", {"nominal_to_v0.5": [FAIL, FAIL], "v0.45_to_v0.5": [PASS]})
    sets._produce_straight(job, its["nominal_to_v0.5"])
    assert [c[0] for c in script.calls] == ["nominal_to_v0.5", "nominal_to_v0.5", "v0.45_to_v0.5"]
    assert script.calls[2][2] == dict(ramp_strides=1.0)  # one stride of ramp: step (1) is not applied again
    direct, other = job.out["transitions"]["nominal_to_v0.5"], job.out["transitions"]["v0.45_to_v0.5"]
    assert direct["available"] is False and "the rung is reached from v0.45 (v0.45_to_v0.5; fallback step 2)" in direct["reason"]
    assert len(direct["numbers"]["tried"]) == 2 and direct["fallback"] == dict(rule="S2S-fallback", step=2, by="v0.45_to_v0.5")
    assert other["available"] and other["fallback"]["step"] == 2 and other["fallback"]["of"] == "nominal_to_v0.5"
    assert other["group"] == "nominal_to_v0.5" and other["inputs"].keys() == {"v0.45", "v0.5"}
    assert job.out["files"] == {"transitions/v0.45_to_v0.5.npz": "s" * 64} and job.out["blocks_set"] is None
    # toward 0.40 m/s from below as well
    job, script, its = job_of("nominal_to_v0.3", {"nominal_to_v0.3": [FAIL, FAIL], "v0.35_to_v0.3": [PASS]})
    sets._produce_straight(job, its["nominal_to_v0.3"])
    assert "v0.35_to_v0.3" in job.out["transitions"]


def test_fallback_step_3_records_no_transition_within_the_gates(job_of):
    job, script, its = job_of("nominal_to_v0.5", {"nominal_to_v0.5": [FAIL, FAIL], "v0.45_to_v0.5": [FAIL]})
    sets._produce_straight(job, its["nominal_to_v0.5"])
    e = job.out["transitions"]
    assert list(e) == ["nominal_to_v0.5"] and e["nominal_to_v0.5"]["available"] is False
    assert e["nominal_to_v0.5"]["reason"] == sets.NO_GATES and len(e["nominal_to_v0.5"]["numbers"]["tried"]) == 3
    assert [t["change"] for t in e["nominal_to_v0.5"]["numbers"]["tried"]] == ["nominal_to_v0.5"] * 2 + ["v0.45_to_v0.5"]
    assert job.out["files"] == {} and job.out["blocks_set"] is None  # a ladder rung does not stop the set


def test_the_change_into_the_bars_walk_has_step_1_only_and_stops_the_set(job_of):
    job, script, its = job_of("nominal_to_v0.4", {"nominal_to_v0.4": [FAIL, FAIL]})
    sets._produce_straight(job, its["nominal_to_v0.4"])
    e = job.out["transitions"]["nominal_to_v0.4"]
    assert len(script.calls) == 2 and e["available"] is False and e["reason"] == sets.NO_GATES
    assert job.out["blocks_set"].startswith("nominal_to_v0.4: no transition within the gates")


@pytest.mark.parametrize("name,stops", [("start", True), ("stop2_v0.4", False), ("stop1_v0.4", False)])
def test_a_start_and_the_stops_have_no_fallback(job_of, name, stops):
    job, script, its = job_of(name, {name: [FAIL]})
    sets._produce_straight(job, its[name])
    e = job.out["transitions"][name]
    assert len(script.calls) == 1 and e["available"] is False and e["reason"] == sets.NO_GATES and "file" not in e
    assert e["numbers"]["failed_gates"] == ["join_b"] and e["pass_"] is False and job.out["files"] == {}
    assert (job.out["blocks_set"] is not None) is stops
    job, script, its = job_of(name, {name: [PASS]})
    sets._produce_straight(job, its[name])
    e = job.out["transitions"][name]
    assert e["available"] and job.out["blocks_set"] is None
    assert e.get("label") == ("comparison: Stage 1's one-step stop on the same walk (not causal)" if name == "stop1_v0.4" else None)


def test_a_move_from_a_dropped_rung_is_not_built(tmp_path, standin, job_of):
    job, script, its = job_of("nominal_to_v0.5", {})
    job.refs["v0.5"] = dict(available=False, reason="dropped from the ladder: over knee speed", numbers=dict(x=1))
    sets._produce_straight(job, its["nominal_to_v0.5"])
    e = job.out["transitions"]["nominal_to_v0.5"]
    assert script.calls == [] and e["available"] is False and e["inputs"] == dict(nominal=job.refs["nominal"]["sha256"], **{"v0.5": None})
    assert e["reason"] == "the walk v0.5 is not a rung of the set (v0.5: dropped from the ladder: over knee speed)"
    # its neighbour dropped: the fallback's second step cannot be built, and says so
    job, script, its = job_of("nominal_to_v0.5", {"nominal_to_v0.5": [FAIL, FAIL]})
    job.refs["v0.45"] = dict(available=False, reason="dropped", numbers={})
    sets._produce_straight(job, its["nominal_to_v0.5"])
    tried = job.out["transitions"]["nominal_to_v0.5"]["numbers"]["tried"]
    assert tried[2] == dict(change="v0.45_to_v0.5", not_built="the walk v0.45 is not a rung of the set")
    # the bar's walk dropped: no turn
    job, script, its = job_of("turn90_pi4_left", {})
    job.refs["v0.4"] = dict(available=False, reason="dropped", numbers={})
    sets._produce_turns(job)
    assert [e["available"] for e in job.out["transitions"].values()] == [False, False]
    assert job.out["stride_rules"]["pi4"]["references"] == {"turn90_pi4_left": False, "turn90_pi4_right": False}


DROPPED = dict(available=False, reason="dropped from the ladder: over clearance swing_clearance 0.0382",
               numbers=dict(mid=0.0382))


def test_lead_in_of_reads_the_walks():
    """``RULES["S2S-lead-in"]``: start where the nominal walk is a rung; start_v0.4 where it is not and the 0.40 m/s
    walk is; none where neither is, or while either is not built."""
    ok, gone = dict(sha256="a"), dict(available=False)
    assert sets.lead_in_of({"nominal": ok, "v0.4": ok}) == "start" == setio.START
    assert sets.lead_in_of({"nominal": ok, "v0.4": gone}) == "start"
    assert sets.lead_in_of({"nominal": gone, "v0.4": ok}) == "start_v0.4" == setio.start_name("v0.4")
    assert sets.lead_in_of({"nominal": gone, "v0.4": gone}) is None
    assert sets.lead_in_of({"nominal": gone}) is None and sets.lead_in_of({"v0.4": ok}) is None and sets.lead_in_of({}) is None
    assert setio.parse_name("start_v0.4") == dict(kind="start", name="start_v0.4", walk="v0.4")
    assert setio.parse_name("start") == dict(kind="start", name="start") and setio.start_name("nominal") == "start"
    for bad in ("start_nominal", "start_v0.40", "start_", "start_x"):
        with pytest.raises(ValueError):
            setio.parse_name(bad)


def test_a_set_without_its_nominal_walk_starts_straight_into_the_bars_walk(job_of):
    """The plan's A8 as amended (DECISION_LOG.md P34): the ``start`` item of a set whose nominal walk is no rung
    builds ``start_v0.4``, a start from standing into the 0.40 m/s walk, gated like ``start`` and with no fallback;
    the item's own entry says why it has no reference and which start the set has."""
    job, script, its = job_of("start", {"start_v0.4": [PASS]})
    job.refs["nominal"] = dict(DROPPED)
    sets._produce_straight(job, its["start"])
    tr = job.out["transitions"]
    assert script.calls == [("start_v0.4", "start", {})] and sorted(tr) == ["start", "start_v0.4"]
    alt, start = tr["start_v0.4"], tr["start"]
    assert alt["available"] and alt["group"] == "start" and alt["item"] == its["start"].definition
    assert alt["definition"]["specs"]["b"]["name"] == "v0.4" and alt["definition"]["specs"]["a"]["spec"] is None
    assert alt["definition"] == its["start"].definition["lead_in"]["start"]  # what the item's definition announced
    assert alt["inputs"] == dict(nominal=None, **{"v0.4": job.refs["v0.4"]["sha256"]}) == start["inputs"]
    assert alt["lead_in"] == dict(rule="S2S-lead-in", text=sets.RULES["S2S-lead-in"], of="start") and alt["fallback"] is None
    assert start["available"] is False and "file" not in start and start["pass_"] is None
    assert start["reason"] == ("the walk nominal is not a rung of the set (nominal: dropped from the ladder: over "
                               "clearance swing_clearance 0.0382); the set starts from standing straight into v0.4 "
                               "(start_v0.4; STAGE2_75_PLAN.md A8 as amended)")
    assert start["lead_in"] == dict(rule="S2S-lead-in", by="start_v0.4") and start["numbers"]["walks"]["nominal"] == dict(mid=0.0382)
    assert job.out["files"] == {"transitions/start_v0.4.npz": "s" * 64} and job.out["blocks_set"] is None
    # a start_v0.4 outside its gates: no retry, no file, and the set is stopped (it is not run)
    job, script, its = job_of("start", {"start_v0.4": [FAIL]})
    job.refs["nominal"] = dict(DROPPED)
    sets._produce_straight(job, its["start"])
    tr = job.out["transitions"]
    assert len(script.calls) == 1 and [tr[n]["available"] for n in ("start", "start_v0.4")] == [False, False]
    assert tr["start_v0.4"]["reason"] == sets.NO_GATES and tr["start_v0.4"]["numbers"]["failed_gates"] == ["join_b"]
    assert tr["start_v0.4"]["group"] == "start" and tr["start"]["reason"].endswith("; start_v0.4: no transition within the gates")
    assert tr["start"]["lead_in"] == dict(rule="S2S-lead-in", by=None) and job.out["files"] == {}
    assert job.out["blocks_set"].startswith("start_v0.4: no transition within the gates")
    # a Stage 0 gate that blocks it stops the set in the same way
    job, script, its = job_of("start", {"start_v0.4": [dict(blocks=[dict(limit="knee speed", value=1.02)])]})
    job.refs["nominal"] = dict(DROPPED)
    sets._produce_straight(job, its["start"])
    assert job.out["transitions"]["start_v0.4"]["available"] is False and job.out["blocks_set"] is not None
    # neither walk is a rung: nothing is built, the set has no lead-in, and nothing more is stopped
    job, script, its = job_of("start", {})
    job.refs["nominal"], job.refs["v0.4"] = dict(DROPPED), dict(DROPPED)
    sets._produce_straight(job, its["start"])
    tr = job.out["transitions"]
    assert script.calls == [] and list(tr) == ["start"] and tr["start"]["available"] is False
    assert tr["start"]["reason"].startswith("the walk nominal, v0.4 is not a rung of the set") and job.out["blocks_set"] is None
    assert tr["start"]["inputs"] == dict(nominal=None, **{"v0.4": None})
    # with its nominal walk the set starts into it, as before: no start_v0.4, the nominal walk its one input
    job, script, its = job_of("start", {"start": [PASS]})
    sets._produce_straight(job, its["start"])
    assert list(job.out["transitions"]) == ["start"] and script.calls == [("start", "start", {})]
    assert job.out["transitions"]["start"]["inputs"] == dict(nominal=job.refs["nominal"]["sha256"])


def test_a_rung_of_a_set_without_its_nominal_walk_is_reached_from_its_neighbour(job_of):
    """The items that need the nominal walk are not available, with that reason; a ladder rung is reached by the
    fallback's step (2), the change from the neighbouring rung toward 0.40 m/s, built at once."""
    job, script, its = job_of("nominal_to_v0.45", {"v0.4_to_v0.45": [PASS]})
    job.refs["nominal"] = dict(DROPPED)
    sets._produce_straight(job, its["nominal_to_v0.45"])
    tr = job.out["transitions"]
    assert script.calls == [("v0.4_to_v0.45", "change", dict(ramp_strides=1.0))] and sorted(tr) == ["nominal_to_v0.45", "v0.4_to_v0.45"]
    direct, step = tr["nominal_to_v0.45"], tr["v0.4_to_v0.45"]
    assert direct["available"] is False and direct["pass_"] is None  # never built, not failed
    assert direct["reason"] == ("the walk nominal is not a rung of the set (nominal: dropped from the ladder: over "
                                "clearance swing_clearance 0.0382); the rung is reached from v0.4 (v0.4_to_v0.45; "
                                "fallback step 2)")
    assert direct["fallback"] == dict(rule="S2S-fallback", step=2, by="v0.4_to_v0.45")
    assert direct["inputs"] == {"nominal": None, "v0.45": job.refs["v0.45"]["sha256"], "v0.4": job.refs["v0.4"]["sha256"]}
    assert direct["numbers"]["tried"] == [dict(change="nominal_to_v0.45", not_built=(
        "the walk nominal is not a rung of the set (nominal: dropped from the ladder: over clearance swing_clearance 0.0382)"))]
    assert step["available"] and step["group"] == "nominal_to_v0.45" and step["ramp_strides"] == 1.0
    assert step["fallback"]["step"] == 2 and step["fallback"]["of"] == "nominal_to_v0.45" and step["inputs"].keys() == {"v0.4", "v0.45"}
    assert job.out["files"] == {"transitions/v0.4_to_v0.45.npz": "s" * 64} and job.out["blocks_set"] is None
    # the next rung up comes from 0.45 m/s
    job, script, its = job_of("nominal_to_v0.5", {"v0.45_to_v0.5": [PASS]})
    job.refs["nominal"] = dict(DROPPED)
    sets._produce_straight(job, its["nominal_to_v0.5"])
    assert [c[0] for c in script.calls] == ["v0.45_to_v0.5"] and job.out["transitions"]["v0.45_to_v0.5"]["available"]
    # that change outside its gates: no reference for the rung, with what was tried; a rung does not stop the set
    job, script, its = job_of("nominal_to_v0.5", {"v0.45_to_v0.5": [FAIL]})
    job.refs["nominal"] = dict(DROPPED)
    sets._produce_straight(job, its["nominal_to_v0.5"])
    e = job.out["transitions"]
    assert list(e) == ["nominal_to_v0.5"] and e["nominal_to_v0.5"]["available"] is False and job.out["files"] == {}
    assert e["nominal_to_v0.5"]["reason"].endswith("; no change from the neighbouring rung within the gates")
    assert [t["change"] for t in e["nominal_to_v0.5"]["numbers"]["tried"]] == ["nominal_to_v0.5", "v0.45_to_v0.5"]
    assert e["nominal_to_v0.5"]["fallback"] == dict(rule="S2S-fallback", step=3, tried=2) and job.out["blocks_set"] is None
    assert e["nominal_to_v0.5"]["inputs"].keys() == {"nominal", "v0.5", "v0.45"}
    # the change into the 0.40 m/s walk needs the nominal walk: not available, nothing built, the set not stopped
    # (the set's lead-in is start_v0.4)
    job, script, its = job_of("nominal_to_v0.4", {})
    job.refs["nominal"] = dict(DROPPED)
    sets._produce_straight(job, its["nominal_to_v0.4"])
    e = job.out["transitions"]["nominal_to_v0.4"]
    assert script.calls == [] and e["available"] is False and job.out["blocks_set"] is None
    assert e["reason"] == "the walk nominal is not a rung of the set (nominal: dropped from the ladder: over clearance swing_clearance 0.0382)"
    # a rung that is itself dropped has no change from anywhere
    job, script, its = job_of("nominal_to_v0.3", {})
    job.refs["nominal"], job.refs["v0.3"] = dict(DROPPED), dict(DROPPED)
    sets._produce_straight(job, its["nominal_to_v0.3"])
    assert script.calls == [] and list(job.out["transitions"]) == ["nominal_to_v0.3"]
    assert job.out["transitions"]["nominal_to_v0.3"]["reason"].startswith("the walk nominal, v0.3 is not a rung of the set")
    # its neighbour dropped: the change cannot be built, and the entry says so
    job, script, its = job_of("nominal_to_v0.5", {})
    job.refs["nominal"], job.refs["v0.45"] = dict(DROPPED), dict(DROPPED)
    sets._produce_straight(job, its["nominal_to_v0.5"])
    tried = job.out["transitions"]["nominal_to_v0.5"]["numbers"]["tried"]
    assert script.calls == [] and tried[1] == dict(change="v0.45_to_v0.5", not_built="the walk v0.45 is not a rung of the set")


def pick_stub(chosen=0.8):
    def pick(src, a, ref_a, angle, rate, grid=moves.STRIDE_GRID, limit=0.85, other_margin=0.05):
        assert tuple(grid) == moves.STRIDE_GRID and (limit, other_margin) == (0.85, 0.05)  # the item's own constants
        return dict(rule="S2A-stride", text="t", grid=[float(f) for f in grid], limit_kinematic=limit, limit_final=0.87,
                    limit_other_joints_rad=other_margin, steps_down_max=2, angle_rad=angle, rate_rad_s=rate,
                    tried=[dict(f=0.6 if chosen is None else chosen, left=0.8, right=0.8, ik_ok=True,
                                pass_=chosen is not None, tightest_other={})],
                    chosen=chosen, steady_turn_kinematic=None, steps_down=0, builds=[],
                    outcome="picked" if chosen is not None else "none",
                    reason=None if chosen is not None else "no stride factor of the grid passes the kinematic rule")
    return pick


@pytest.fixture
def turns(job_of, monkeypatch):
    def run(plan: dict, chosen=0.8, rate="pi4"):
        left, right = f"turn90_{rate}_left", f"turn90_{rate}_right"
        job, script, its = job_of(left, {left: list(plan.get("left", [])), right: list(plan.get("right", plan.get("left", [])))})
        monkeypatch.setattr(moves, "pick_stride_factor", pick_stub(chosen))
        job.source = lambda fresh=True: None
        sets._produce_turns(job)
        return job, script, job.out["stride_rules"][rate], job.out["transitions"][left], job.out["transitions"][right]

    return run


def test_turns_both_directions_are_built_at_the_picked_factor(turns):
    job, script, rec, left, right = turns(dict(left=[PASS]))
    assert [(c[0], c[2]["stride_factor"], c[2]["rate"]) for c in script.calls] == [
        ("turn90_pi4_left", 0.8, math.pi / 4), ("turn90_pi4_right", 0.8, -math.pi / 4)]
    assert all(c[2]["ramp_s"] == 0.375 and c[2]["angle"] == math.pi / 2 for c in script.calls)
    assert rec["outcome"] == "frozen" and rec["chosen"] == 0.8 and rec["steps_down"] == 0 and rec["a1_over"] is False
    assert rec["references"] == {"turn90_pi4_left": True, "turn90_pi4_right": True} and rec["no_reference"] is None
    assert rec["walk"] == "v0.4" and rec["walk_sha256"] == job.refs["v0.4"]["sha256"]
    b = rec["builds"][0]
    assert (b["f"], b["ik_ok"], b["hip_yaw_range_use"], b["tightest_other_rad"]) == (0.8, True, 0.7, 0.12)
    assert set(b["tightest_other"]) == {"left", "right"} and b["pass_"] == dict(left=True, right=True)
    assert left["available"] and right["available"] and left["turn"]["stride_rule"] == rec == right["turn"]["stride_rule"]
    assert sorted(job.out["files"]) == ["transitions/turn90_pi4_left.npz", "transitions/turn90_pi4_right.npz"]


@pytest.mark.parametrize("worse,how", [(dict(twist=0.871), "twist"), (dict(margin=0.0499), "margin"), (dict(ik_ok=False), "ik")])
def test_turns_step_down_when_either_direction_is_outside_the_stride_rule(turns, worse, how):
    """The worse direction decides: a thigh twist over 0.87, another joint under a move's 0.05 rad, or an IK failure
    in one direction steps the factor down for both."""
    job, script, rec, left, right = turns(dict(left=[PASS, PASS], right=[worse, PASS]))
    assert [(c[0], c[2]["stride_factor"]) for c in script.calls] == [
        ("turn90_pi4_left", 0.8), ("turn90_pi4_right", 0.8), ("turn90_pi4_left", 0.75), ("turn90_pi4_right", 0.75)]
    assert rec["outcome"] == "frozen" and rec["chosen"] == 0.75 and rec["steps_down"] == 1 and [b["f"] for b in rec["builds"]] == [0.8, 0.75]
    assert left["available"] and right["available"] and left["turn"]["stride_factor"] == 0.75


def test_turns_step_down_at_most_twice_then_there_is_no_reference(turns):
    bad = dict(margin=0.049)
    job, script, rec, left, right = turns(dict(left=[bad, bad, bad]))
    assert [c[2]["stride_factor"] for c in script.calls] == [0.8, 0.8, 0.75, 0.75, 0.7, 0.7]
    assert rec["outcome"] == "none" and rec["chosen"] is None and rec["steps_down"] == 2
    assert rec["references"] == {"turn90_pi4_left": False, "turn90_pi4_right": False}
    for e in (left, right):
        assert e["available"] is False and e["reason"].startswith(sets.NO_TWIST + ": the built reference is not within")
        assert "file" not in e and e["numbers"]["stride_rule"]["builds"][-1]["f"] == 0.7 and "text" not in e["numbers"]["stride_rule"]
    assert rec["no_reference"] == left["reason"] and job.out["files"] == {}


def test_turns_a_leg_drive_over_its_limit_leaves_no_reference_and_no_retry(turns):
    job, script, rec, left, right = turns(dict(left=[PASS], right=[dict(leg=True, blocks=[dict(limit="knee speed", value=1.04)])]))
    assert len(script.calls) == 2 and rec["outcome"] == "none"
    assert rec["reason"] == "the built reference is over a leg drive's Stage 0 limit"
    assert left["available"] is False and left["reason"] == f"{sets.NO_TWIST}: {rec['reason']}"
    assert right["numbers"]["stage0"]["blocks_freeze"] == [dict(limit="knee speed", value=1.04)]
    # a built pair that is also outside an entry gate: the reason names it, in both directions' entries
    over = dict(leg=True, failed=["heading_at_slowest_clock"], blocks=[dict(limit="knee speed", value=1.04)])
    job, script, rec, left, right = turns(dict(left=[over], right=[dict(leg=True, blocks=[dict(limit="knee speed", value=1.04)])]), rate="1")
    want = f"{sets.NO_TWIST}: the built reference is over a leg drive's Stage 0 limit; also outside heading_at_slowest_clock"
    assert left["reason"] == right["reason"] == rec["no_reference"] == want and rec["outcome"] == "none"


def test_turns_a_tail_limit_alone_keeps_the_reference_with_a1_over(turns):
    job, script, rec, left, right = turns(dict(left=[dict(a1=True)], right=[PASS]))
    assert rec["outcome"] == "frozen" and rec["a1_over"] is True and rec["builds"][0]["a1_over"] is True
    assert left["available"] and left["a1_over"] is True and right["available"] and right["a1_over"] is False


def test_turns_outside_the_heading_rule_have_no_reference_and_no_retry(turns):
    slow_clock = dict(failed=["heading_at_slowest_clock"])
    job, script, rec, left, right = turns(dict(left=[slow_clock]), rate="1")
    assert len(script.calls) == 2  # built once: the stride rule held, the heading rule did not
    assert rec["outcome"] == "frozen" and rec["references"] == {"turn90_1_left": False, "turn90_1_right": False}
    assert rec["no_reference"] == sets.NO_HEADING == left["reason"] == right["reason"]
    assert left["available"] is False and left["pass_"] is False and left["numbers"]["failed_gates"] == ["heading_at_slowest_clock"]
    assert job.out["files"] == {}
    # another gate: no reference, named by its gates; the other direction, within its gates, keeps its own
    job, script, rec, left, right = turns(dict(left=[dict(failed=["zmp_consistency", "heading_at_bar"])], right=[PASS]))
    assert left["available"] is False and left["reason"] == "no reference within the gates (entry gates: zmp_consistency, heading_at_bar)"
    assert right["available"] is True and rec["references"] == {"turn90_pi4_left": False, "turn90_pi4_right": True}
    job, script, rec, left, right = turns(dict(left=[dict(blocks=[dict(limit="clearance swing_clearance", value=0.03)])]))
    assert left["reason"] == "no reference within the gates (Stage 0: x)" and rec["no_reference"] == left["reason"]


def test_turns_without_a_stride_factor_build_nothing(turns):
    job, script, rec, left, right = turns(dict(left=[]), chosen=None)
    assert script.calls == [] and rec["outcome"] == "none" and rec["chosen"] is None and rec["builds"] == []
    assert left["available"] is False and left["reason"] == f"{sets.NO_TWIST}: no stride factor of the grid passes the kinematic rule"
    assert left["numbers"]["stride_rule"]["tried"] and rec["references"] == {"turn90_pi4_left": False, "turn90_pi4_right": False}


@pytest.fixture
def look_turns(job_of, monkeypatch):
    """``turns`` on a look set's job: ``_produce_turns`` takes the look's branch (``SetDef.look``), its stride pick
    stubbed in ``moves_look``, its walk read once beforehand."""
    def run(plan: dict, chosen=0.8, rate="pi4"):
        left, right = f"turn90_{rate}_left", f"turn90_{rate}_right"
        job, script, its = job_of(left, {left: list(plan.get("left", [])), right: list(plan.get("right", plan.get("left", [])))})
        walk = job.walk(job.mine[0].depends[0])
        job.walk = lambda name: walk
        job.st = replace(job.st, look=True)
        monkeypatch.setattr(sets._ml(), "pick_stride_factor", pick_stub(chosen))
        job.source = lambda fresh=True: None
        sets._produce_turns(job)
        return job, script, job.out["stride_rules"][rate], job.out["transitions"][left], job.out["transitions"][right]

    return run


def test_a_look_sets_turn_reasons_name_only_the_gates_each_direction_failed(look_turns, turns):
    """Ruling 26 of 17 Nov 2026 (the look sets' check, findings 5 to 8; review point (f): ``turn90_pi4_right``'s
    reason said "also outside join_command", which its rate group's left turn failed): in a look set each direction's
    reason names the entry gates its own build failed and no other; the rate's ``no_reference`` gives each
    direction's where they differ.  The round's sets keep their words (one reason for both: tested above)."""
    leg = [dict(limit="knee speed", value=1.04)]
    over = dict(leg=True, failed=["join_command"], blocks=leg)
    job, script, rec, left, right = look_turns(dict(left=[over], right=[dict(leg=True, blocks=leg)]))
    base = f"{sets.NO_TWIST}: the built reference is over a leg drive's Stage 0 limit"
    assert rec["outcome"] == "none" and left["reason"] == f"{base}; also outside join_command" and right["reason"] == base
    assert rec["no_reference"] == f"turn90_pi4_left: {base}; also outside join_command; turn90_pi4_right: {base}"
    assert left["numbers"]["failed_gates"] == ["join_command"] and right["numbers"]["failed_gates"] == []
    # both directions failing the same gate: one reason, as before
    job, script, rec, left, right = look_turns(dict(left=[over], right=[over]))
    assert left["reason"] == right["reason"] == rec["no_reference"] == f"{base}; also outside join_command"
    # outside the gates at a frozen stride factor: each direction its own gates
    job, script, rec, left, right = look_turns(dict(left=[dict(failed=["join_command"])], right=[dict(failed=["zmp_error"])]))
    assert left["reason"] == "no reference within the gates (entry gates: join_command)"
    assert right["reason"] == "no reference within the gates (entry gates: zmp_error)"
    assert rec["no_reference"] == f"turn90_pi4_left: {left['reason']}; turn90_pi4_right: {right['reason']}"
    # the same pair in a set of today's laws: one reason for both, the union of the pair's gates (state A)
    job, script, rec, left, right = turns(dict(left=[over], right=[dict(leg=True, blocks=leg)]))
    assert left["reason"] == right["reason"] == rec["no_reference"] == f"{base}; also outside join_command"


# ---------------------------------------------------------------- twins
def test_v0_owes_its_twins_on_a_body_that_has_them(made, job_of, monkeypatch):
    """The plan's A4 asks of V0, on a body with a Stage 1 run record and ladder rungs, that ``nominal`` and ``start``
    are Stage 1's own files and that the look-format ``nominal`` and ``v0.4`` are the ladder's rungs: a twin that is
    not found is a failure of ``verify`` and stops the build, not a silent null."""
    v0 = sets.SETS["V0"]
    assert sets.twin_owed(v0, BODY, "nominal", walk=True) == dict(stage1=True, ladder=True)
    assert sets.twin_owed(v0, BODY, "v0.4", walk=True) == dict(stage1=False, ladder=True)
    assert sets.twin_owed(v0, BODY, "start", walk=False) == dict(stage1=True, ladder=False)
    assert sets.twin_owed(v0, BODY, "stop2_v0.4", walk=False) == dict(stage1=False, ladder=False)
    other = next(b for b in stage1.BODIES if not sets.stage1_has(b, "refs.json") and not sets.ladder_has(b))
    assert sets.twin_owed(v0, other, "nominal", walk=True) == dict(stage1=False, ladder=False)
    assert sets.twin_owed(sets.SETS["F25F9cl"], BODY, "nominal", walk=True) == dict(stage1=False, ladder=False)
    # the synthetic set, held to V0's twins: none of its entries is Stage 1's or the ladder's
    monkeypatch.setitem(sets.SETS, SET, replace(sets.SETS[SET], stage1_twins=v0.stage1_twins, ladder_twins=v0.ladder_twins))
    f = verify(made)["failures"]
    assert sorted(x.split(":")[0] for x in f) == ["nominal", "nominal", "start", "v0.4"]
    assert any("start: the set must hold Stage 1's own file under this name and its entry's equals_stage1 is None" in x for x in f)
    assert any("v0.4: its look-format file must be the look ladder's rung and its entry's equals_ladder is None" in x for x in f)
    # ... and the builder stops at such a walk before anything of it is recorded
    job, script, its = job_of("nominal", {})
    monkeypatch.setattr(sets, "walk_arrays", lambda src, spec: (dict(), dict(), dict()))
    job.source, job.put, job.built_by = (lambda fresh=True: {}), (lambda sub, name, arrays: "s" * 64), (lambda: {})
    with pytest.raises(sets.GateError, match="nominal: the set must hold Stage 1's own walk under this name and none "
                                             "of the same spec is recorded"):
        sets._produce_walk(job, job.mine[0])
    assert job.out["refs"] == {}
    # on a body that has neither record the twins are not owed: a note
    monkeypatch.setattr(sets, "stage1_has", lambda body, record: False)
    monkeypatch.setattr(sets, "ladder_has", lambda body: False)
    v = verify(made)
    assert v["ok"] and any("Stage 1's run holds no record of" in x for x in v["notes"])
    assert any("the look ladder holds no rung of" in x for x in v["notes"])


def test_twins_are_found_by_spec_not_by_name():
    H, _ = sets.body_facts(BODY)
    s1 = json.loads((RUN / "refs.json").read_text())["references"]
    assert sets.stage1_walk_twin(BODY, sizing.NOMINAL) == dict(name="nominal", sha256=s1["nominal"]["sha256"],
                                                               record="stage1/runs/2026-11-08/75-M-H-10/refs.json")
    assert sets.stage1_walk_twin(BODY, envelope.FAST)["name"] == "fr05"
    assert sets.stage1_walk_twin(BODY, look.speed_spec(H)) is None
    assert sets.stage1_walk_twin(BODY, sets.tuned_f9_spec(BODY, H, 0.4)) is None
    assert sets.stage1_walk_twin("75-M-H-10-J", sizing.NOMINAL) is None  # no run of that body under Stage 1's run
    ladder = json.loads(sets.LOOK_REFS.read_text())["bodies"][BODY]
    assert sets.ladder_twin(BODY, sizing.NOMINAL) == dict(speed="nominal", variant="V0", sha256=ladder["nominal"]["V0"]["sha256"],
                                                          record="stage2/look/look_refs.json")
    assert sets.ladder_twin(BODY, look.speed_spec(H))["speed"] == "v040"
    assert sets.ladder_twin(BODY, sets.v0_spec(BODY, H, 0.45)) is None and sets.ladder_twin("75-M-H-10-J", sizing.NOMINAL) is None
    # a move is Stage 1's own transition when its ends are Stage 1's files and its schedule Stage 1's
    nom = moves.GaitX.of("nominal", sizing.NOMINAL, H)
    brisk = moves.GaitX.of("brisk", sizing.BRISK, H)
    tn, tb = sets.stage1_walk_twin(BODY, sizing.NOMINAL), sets.stage1_walk_twin(BODY, sizing.BRISK)
    s1t = json.loads((RUN / "transitions.json").read_text())["transitions"]
    assert sets.stage1_move_twin(BODY, moves.start("start", nom), None, tn)["sha256"] == s1t["start"]["sha256"]
    assert sets.stage1_move_twin(BODY, moves.change("x", nom, brisk), tn, tb)["name"] == "nominal_to_brisk"
    assert sets.stage1_move_twin(BODY, moves.change("x", nom, brisk, ramp_strides=2.0), tn, tb) is None
    assert sets.stage1_move_twin(BODY, moves.stop_one_step("x", nom), tn, None)["name"] == "stop"
    assert sets.stage1_move_twin(BODY, moves.stop_two_step("x", nom), tn, None) is None  # causal: not Stage 1's
    assert sets.stage1_move_twin(BODY, moves.start("start", nom), None, None) is None  # its walk is not Stage 1's file
    v0 = {i.name: i for i in sets.items("V0", BODY)}
    assert v0["nominal"].definition["spec"] == json.loads(json.dumps(look._plain(asdict(sizing.NOMINAL))))


# ---------------------------------------------------------------- ignore rules
@has_git
def test_the_arrays_and_every_writers_scratch_are_ignored_and_the_records_are_not():
    d = "stage2/sets/V0/75-M-H-10"
    ignored = [f"{d}/refs/x.npz", f"{d}/look/x.npz", f"{d}/transitions/x.npz", f"{d}/refs/x.npz.tmp", f"{d}/refs.json.tmp",
               f"{d}/transitions.json.tmp", f"{d}/set.json.tmp", f"{d}/screen.json.tmp", f"{d}/partial/x.json",
               f"{d}/partial/x.json.tmp", f"{d}/partial/build-x/refs/x.npz", f"{d}/partial/build-x/refs/x.npz.tmp",
               f"{d}/partial/screen__x.json", f"{d}/.lock", f"{d}/traces/x.npz", f"{d}/logs/sets.log", f"{d}/logs/setscreen.log",
               f"{d}/done/x", "stage2/sets/legacy_check.json.tmp"]
    kept = [f"{d}/refs.json", f"{d}/transitions.json", f"{d}/set.json", f"{d}/screen.json", f"{d}/SCREEN.md",
            "stage2/sets/legacy_check.json", "stage2/sets/.gitignore"]
    r = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "--no-index", "--stdin"], input="\n".join(ignored + kept),
                       capture_output=True, text=True)
    assert sorted(r.stdout.split()) == sorted(ignored)


@has_git
def test_a_built_set_shows_only_its_records_to_git(tmp_path, standin):
    """In a repository with the sets' ignore rules, a built set (with what a killed build leaves) shows git its
    JSON records and nothing else."""
    repo = tmp_path / "repo"
    (repo / "stage2" / "sets").mkdir(parents=True)
    shutil.copy(setio.ROOT / ".gitignore", repo / "stage2" / "sets" / ".gitignore")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    root = repo / "stage2" / "sets"
    build(root)
    d = root / SET / BODY
    sets.build_item(SET, BODY, "start", root)  # a finished partial not merged
    (d / "partial" / "build-v0.3" / "refs").mkdir(parents=True)
    (d / "partial" / "build-v0.3" / "refs" / "v0.3.npz").write_bytes(b"x")
    (d / "refs.json.tmp").write_text("{")
    (d / ".lock").write_text("{}")
    st = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"], capture_output=True,
                        text=True, check=True).stdout.split("\n")
    assert sorted(x[3:] for x in st if x) == ["stage2/sets/.gitignore", f"stage2/sets/{SET}/{BODY}/refs.json",
                                               f"stage2/sets/{SET}/{BODY}/set.json", f"stage2/sets/{SET}/{BODY}/transitions.json"]


# ------------------------------------------------------------------------------------------------
# 3. Built arrays: the process state, and the V0 set of the 75 cm pick
# ------------------------------------------------------------------------------------------------
STATE_CODE = r"""
import json, sys
from pathlib import Path
from designlab import gait
from designlab.stage2 import s2io, sets
root, order = sys.argv[1], sys.argv[2].split(",")
out = {}
assert gait.ACCURACY == "r1"
for name in order:
    part = sets.build_item("V0", "75-M-H-10", name, root)
    sets.merge(Path(root) / "V0" / "75-M-H-10", part, log=lambda line: None)
    out[name] = part["files"]
    assert gait.ACCURACY == "r1"  # the level the process had: nothing is left behind
print(json.dumps(out))
"""


def _state(root: Path, order: str) -> dict:
    env = dict(os.environ, PYTHONPATH=str(ROOT), MUJOCO_GL="disable")
    r = subprocess.run([PY, "-c", STATE_CODE, str(root), order], capture_output=True, text=True, env=env, cwd=ROOT,
                       timeout=3600)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


@slow
@needs_run
def test_a_walk_has_one_sha256_whatever_the_process_built_before(tmp_path):
    """Process state: the same walk built first in a fresh process, and built after another walk and a move in one
    process, has the same sha256 in both formats; and a build with one worker and with two gives the same files."""
    a = _state(tmp_path / "a", "v0.3")
    b = _state(tmp_path / "b", "nominal,start,v0.3")
    assert a["v0.3"] == b["v0.3"] and set(a["v0.3"]) == {"refs/v0.3.npz", "look/v0.3.npz"}
    stage1 = json.loads((RUN / "refs.json").read_text())["references"]["nominal"]["sha256"]
    assert b["nominal"]["refs/nominal.npz"] == stage1
    start = json.loads((RUN / "transitions.json").read_text())["transitions"]["start"]["sha256"]
    assert b["start"] == {"transitions/start.npz": start}
    got = {}
    for procs in (1, 2):
        root = tmp_path / f"p{procs}"
        r = sets.build("V0", BODY, root=root, only=["nominal", "v0.3"], procs=procs, log=lambda line: None)
        assert sorted(r["ran"]) == ["nominal", "v0.3"]
        got[procs] = {n: e["sha256"] for n, e in docs_of(root / "V0" / BODY)["refs.json"]["references"].items()}
    assert got[1] == got[2] == {"nominal": stage1, "v0.3": a["v0.3"]["refs/v0.3.npz"]}


@built
def test_v0_set_reuses_stage1_files():
    """The built V0 set: ``refs/nominal.npz`` and ``transitions/start.npz`` hash to Stage 1's own files."""
    docs = setio.load(V0_SET)
    s1r = json.loads((RUN / "refs.json").read_text())["references"]
    s1t = json.loads((RUN / "transitions.json").read_text())["transitions"]
    nominal, start = docs["refs.json"]["references"]["nominal"], docs["transitions.json"]["transitions"]["start"]
    assert nominal["sha256"] == nominal["stage1_sha256"] == s1r["nominal"]["sha256"] and nominal["equals_stage1"] is True
    assert start["sha256"] == start["stage1_sha256"] == s1t["start"]["sha256"] and start["equals_stage1"] is True
    assert nominal["stage1_twin"] == "nominal" and start["stage1_twin"] == "start"
    for name, e in docs["refs.json"]["references"].items():
        if name != "nominal":
            assert e.get("stage1_sha256") is None and e.get("equals_stage1") is None, name
    if (V0_SET / "refs" / "nominal.npz").exists():
        assert s2io.sha256_file(V0_SET / "refs" / "nominal.npz") == s1r["nominal"]["sha256"]
        assert s2io.sha256_file(V0_SET / "transitions" / "start.npz") == s1t["start"]["sha256"]


@built
def test_v0_look_files_are_the_ladders():
    """The built V0 set: ``look/nominal.npz`` and ``look/v0.4.npz`` hash to the look ladder's V0 rungs."""
    docs = setio.load(V0_SET)
    ladder = json.loads(sets.LOOK_REFS.read_text())["bodies"][BODY]
    for name, speed in (("nominal", "nominal"), ("v0.4", "v040")):
        e = docs["refs.json"]["references"][name]
        assert e["look_sha256"] == e["ladder_sha256"] == ladder[speed]["V0"]["sha256"] and e["equals_ladder"] is True
        assert e["ladder_twin"] == [speed, "V0"]
        le = docs["set.json"]["look_entries"][name]
        for key in ("spec", "info", "measured", "price", "flags", "frames", "ik_ok", "ik_fail_frames", "rolls"):
            assert le[key] == ladder[speed]["V0"][key], (name, key)
        if (V0_SET / "look" / f"{name}.npz").exists():
            assert s2io.sha256_file(V0_SET / "look" / f"{name}.npz") == ladder[speed]["V0"]["sha256"]


@built
def test_the_built_v0_set_holds_what_the_rules_say():
    """The built V0 set against its own rules: every item is there, no wall time, one crouch, the labels; every turn
    rate's stride rule record is the rule on its own rows; what has no reference says why."""
    docs = setio.load(V0_SET)
    refs_, tr, sd = docs["refs.json"]["references"], docs["transitions.json"]["transitions"], docs["set.json"]
    its = sets.items("V0", BODY)
    # (the records hold their entries in the order the items were finished: two workers at a time)
    assert sorted(i.name for i in its if i.kind == "walk") == sorted(refs_) and {i.name for i in its if i.kind != "walk"} <= set(tr)
    assert sd["lead_in"] == "start" and "start_v0.4" not in tr  # its nominal walk is a rung
    assert all(no_wall_time(docs[k]) for k in ("refs.json", "transitions.json", "set.json"))
    assert sd["crouch_frac"] == 0.03 and sd["role"] == "control" and sd["set"] == "V0" and sd["body"] == BODY
    for name, e in refs_.items():
        if e.get("available") is not False:
            assert e["crouch_frac"] == 0.03 and e["label"] == ("new" if name == "v0.5" else None) and e["margin_rule"] is None
            assert e["built_by"]["accuracy"] == "r2" and e["built_by"]["blas_threads"] == 1
            assert e["strides"] == 1 and e["window_s"] == e["period_s"]
    for label, rec in sd["stride_rules"].items():
        if "tried" not in rec:
            continue
        assert (rec["limit_kinematic"], rec["limit_final"], rec["limit_other_joints_rad"]) == (0.85, 0.87, 0.05)
        for row in rec["tried"]:
            ok = bool(row["ik_ok"] and max(row["left"], row["right"]) <= 0.85
                      and min(row["tightest_other"][x]["margin_rad"] for x in ("left", "right")) >= 0.05)
            assert ok is row["pass_"], (label, row)
        assert [r["pass_"] for r in rec["tried"]][:-1] == [False] * (len(rec["tried"]) - 1)
        assert [r["f"] for r in rec["tried"]] == [f for f in moves.STRIDE_GRID][: len(rec["tried"])]
        for b in rec["builds"]:
            assert set(b["tightest_other"]) == {"left", "right"}
        names = [setio.turn_name(label, x) for x in setio.DIRECTIONS]
        assert rec["references"] == {n: tr[n].get("available") is not False for n in names}
        for n in names:
            if tr[n].get("available") is not False:
                assert tr[n]["turn"]["stride_factor"] == rec["chosen"] and tr[n]["turn"]["stride_rule"] == rec
                assert tr[n]["hip_yaw_range_use"] <= 0.87 and tr[n]["tightest_other"]["margin_rad"] >= 0.05
                assert tr[n]["pass_"] and tr[n]["stage0"]["blocks_freeze"] == []
    for name, e in tr.items():
        if e.get("available") is False:
            assert e["reason"] and "numbers" in e and "file" not in e and sd["not_available"][name]["reason"] == e["reason"]
        else:
            assert e["pass_"] is True and e["a1_over"] in (True, False) and e["built_by"]["accuracy"] == "r2"
    if docs["screen.json"] is not None and arrays_here(V0_SET, docs):
        v = sets.verify("V0", BODY)
        assert v["ok"], v["failures"]


@pytest.mark.parametrize("set_name", list(sets.SETS))
def test_a_built_sets_entries_carry_their_labels_and_its_kinematic_check_is_of_its_files(set_name):
    """A built set on 75-M-H-10 as its committed records hold it: every entry carries the set's ``margin_rule`` (the
    plan's A3: the candidate's is the rule's text, a move's as a walk's, with or without a reference) and every walk
    its label (A2); an entry with no reference that a build decided carries that build's provenance; V0's twins are
    Stage 1's and the ladder's (A4); the record of the bar's kinematic check beside the set is of the set's files,
    and ``sets verify`` gives what fails in it as notes, never as a failure."""
    d = setio.ROOT / set_name / BODY
    if not (d / "set.json").exists():
        pytest.skip(f"the {set_name} set is not built (stage2/sets/{set_name})")
    docs = setio.load(d)
    st = sets.SETS[set_name]
    H, _ = sets.body_facts(BODY)
    refs_, tr = docs["refs.json"]["references"], docs["transitions.json"]["transitions"]
    assert (st.margin_rule == "duty and lift set after results on seeds 3042-3061 (JPS 2.3)") == (set_name == "F25F9cl")
    for name, e in {**refs_, **tr}.items():
        assert "margin_rule" in e and e["margin_rule"] == st.margin_rule, name
    for name, e in refs_.items():
        assert e["label"] == st.label_of(BODY, H, setio.parse_name(name)["speed_m_s"]), name
    for name, e in tr.items():
        if e.get("available") is False and (e.get("numbers") or {}).get("tried") and any("gates" in t for t in e["numbers"]["tried"]):
            assert e["built_by"]["stage2_modules"], name  # a change was built for it: its verdict has a builder
    for name in st.stage1_twins:
        assert {**refs_, **tr}[name]["equals_stage1"] is True, name
    for name in st.ladder_twins:
        assert refs_[name]["equals_ladder"] is True, name
    rc = sets.refcheck_state(d, docs)
    if rc is None:
        pytest.skip(f"no refcheck.json beside the {set_name} set")
    assert rc["current"], rc["why"]
    assert rc["file"] == f"stage2/sets/{set_name}/{BODY}/refcheck.json" and set(rc["files"]) <= {
        f"{'refs' if n in refs_ else 'transitions'}/{n}.npz" for n in {**refs_, **tr}}
    lines = sets.refcheck_lines(rc)
    assert len(lines) == max(1, len(rc["failing"])) and all(any(f"{c} FAILS" in x for x in lines) for c in rc["failing"])
    assert rc["pass_"] is (not rc["failing"])
    if docs["screen.json"] is not None and arrays_here(d, docs):
        v = sets.verify(set_name, BODY)
        assert v["ok"], v["failures"]  # what the kinematic check says is a note, never a failure of verify
        assert [x for x in v["notes"] if x.startswith("refcheck")] == lines


@pytest.mark.parametrize("set_name", list(sets.SETS))
def test_a_built_sets_lead_in_follows_its_walks(set_name):
    """A built set on 75-M-H-10 against ``RULES["S2S-lead-in"]`` (the plan's A8 as amended): ``set.json`` ``lead_in``
    is what its walks give; a set that starts into its 0.40 m/s walk holds ``start_v0.4`` as the ``start`` item's
    reference, says why ``start`` has none, and holds no item that needs the nominal walk; a rung reached from its
    neighbour holds that change by the fallback's step (2).  As the plan says of 15 Nov, this is F25F9 alone."""
    d = setio.ROOT / set_name / BODY
    if not (d / "set.json").exists():
        pytest.skip(f"the {set_name} set is not built (stage2/sets/{set_name})")
    docs = setio.load(d)
    refs_, tr, sd = docs["refs.json"]["references"], docs["transitions.json"]["transitions"], docs["set.json"]
    its = {i.name: i for i in sets.items(set_name, BODY)}
    lead, alt = sets.lead_in_of(refs_), setio.start_name("v0.4")
    assert sd["lead_in"] == lead and (lead == alt) == (set_name == "F25F9")
    if lead == "start":
        assert alt not in tr and (tr["start"].get("available") is not False
                                  or "start" in {b["by"] for b in sets.blocked_list(sd["blocked"])})
        return
    assert lead == alt and refs_["nominal"]["available"] is False and refs_["v0.4"].get("available") is not False
    e, start = tr[alt], tr["start"]
    assert e["group"] == "start" and e["item"] == its["start"].definition
    assert start["available"] is False and start["lead_in"]["rule"] == "S2S-lead-in" and "file" not in start
    assert start["reason"].startswith("the walk nominal is not a rung of the set") and sd["not_available"]["start"]["reason"] == start["reason"]
    if e.get("available") is False:  # the set is stopped: it is not run
        assert "start" in {b["by"] for b in sets.blocked_list(sd["blocked"])} and start["lead_in"]["by"] is None
    else:
        assert e["pass_"] is True and e["stage0"]["blocks_freeze"] == [] and e["fallback"] is None and sd["blocked"] is None
        assert (e["move"]["builder"], e["move"]["causal"], e["from_"], e["to"]) == ("start", False, "stand", "v0.4")
        assert e["definition"]["specs"]["b"]["name"] == "v0.4" and e["lead_in"]["of"] == "start" and start["lead_in"]["by"] == alt
        assert e["inputs"] == {"nominal": None, "v0.4": refs_["v0.4"]["sha256"]} and (d / e["file"]).name == f"{alt}.npz"
        assert e["join_b"]["reference"] == "v0.4" and e["gates"]["zmp_error"] and e["zmp_error_history_m"][-1] <= 4.0e-3
    for v in setio.SPEEDS:  # every change from the nominal walk
        walk = setio.walk_name(v)
        n = tr[setio.change_name("nominal", walk)]
        assert n["available"] is False and n["reason"].startswith("the walk nominal")
        nb = sets.neighbour(v)
        other = None if nb is None else setio.change_name(setio.walk_name(nb), walk)
        if other in tr:  # the rung is reached from its neighbour
            assert tr[other]["pass_"] and tr[other]["fallback"]["step"] == 2 and n["fallback"] == dict(rule="S2S-fallback", step=2, by=other)
        elif nb is not None and all(refs_[w].get("available") is not False for w in (walk, setio.walk_name(nb))):
            tried = n["numbers"]["tried"]  # that change was built and is outside a gate: no reference, with its numbers
            assert [t["change"] for t in tried] == [n["name"], other] and tried[1]["failed_gates"] and n["fallback"]["step"] == 3
    if docs["screen.json"] is not None and arrays_here(d, docs):
        v = sets.verify(set_name, BODY)
        assert v["ok"], v["failures"]
        assert any(x.startswith(f"lead-in: {alt}") for x in v["notes"]) and alt in v["available"]

