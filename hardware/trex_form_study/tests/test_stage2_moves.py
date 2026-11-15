"""Tests for the Stage 2 move builder (``designlab/stage2/moves.py``): starts, changes, stops and finite turns for
any ``gait.GaitSpec``, with Stage 1's 18 transitions as one of its schedules.

Run from hardware/trex_form_study:  python -m pytest tests/test_stage2_moves.py -q

1. Exact equality with Stage 1: the gaits (``GaitX.legacy`` against ``transitions.gait_of``), every plan array of all
   18 transitions of both picks, the generated arrays and summaries at one correction pass (and, with STAGE2_SLOW=1,
   at the full passes with each npz's sha256 against the Stage 1 run's record), the solver functions called, and the
   committed record of the whole check (``stage2/sets/legacy_check.json``).
2. The causal builder and the stride rule: the incremental preview form, causality (nothing before the command frame
   depends on the commanded plan), a null command, committed footprints (rule F), the command at a lift-off, the
   mirror of a turn, its closure and ramps, the appendage rule, the fade of the excused transient, the stride rule's
   logic on a stubbed kinematic pass, a causal move's summary and its gates (each the rule's comparison, limits
   written out); with STAGE2_SLOW=1 a scratch build of the two-touchdown stop and of the quick right turn from V0's
   0.40 m/s walk on 75-M-H-10.
3. The reference's own readings: the acceleration on the frames' own grid with the track's history given to the
   1 Hz filter, touchdowns counted in the file's frames; the gait accuracy put back by every call that sets it; the
   legacy record's provenance per body.

Every test that builds arrays does so in a subprocess with one BLAS thread; tests that need the git-ignored Stage 1
arrays skip without them (``tools/restore_inputs.py``).  The suite writes nothing under the study tree: the numbers
of its builds go to pytest's temporary directory, and to the directory STAGE2_TEST_LOGS names when it is set.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import designlab  # noqa: E402,F401  (selects the GL backend before mujoco is imported)
from designlab import gait  # noqa: E402
from designlab.stage1 import refs, transitions  # noqa: E402
from designlab.stage2 import look, moves  # noqa: E402

RUN = ROOT / "stage1" / "runs" / "2026-11-08"
RECORD = ROOT / "stage2" / "sets" / "legacy_check.json"
BODY, BODY75 = "60-N-T-10", "75-M-H-10"
PY = sys.executable
PLAN_ARRAYS = ("phase", "t", "foot_pos", "foot_yaw", "swing", "swing_u", "zmp", "heading", "path", "root_z", "neck",
               "tail")
slow = pytest.mark.skipif(not os.environ.get("STAGE2_SLOW"), reason="full-pass builds; set STAGE2_SLOW=1")
ARRAYS = all((RUN / b / "refs" / "nominal.npz").exists() and (RUN / b / "transitions" / "stop.npz").exists()
             for b in look.BODIES)
needs_arrays = pytest.mark.skipif(not ARRAYS, reason="the git-ignored Stage 1 arrays are not here "
                                                     "(tools/restore_inputs.py --group picks-arrays)")


def _env() -> dict:
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[var] = "1"
    return env


def _run(code: str, *args: str, timeout: int = 3600) -> dict:
    """``code`` in a subprocess with one BLAS thread; its last line of stdout is JSON."""
    r = subprocess.run([PY, "-c", code, *args], cwd=ROOT, env=_env(), capture_output=True, text=True, timeout=timeout)
    assert r.returncode == 0, r.stderr[-4000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def _log(tmp, name: str, doc: dict) -> None:
    """Keep a subprocess's numbers for a report: under pytest's temporary directory ``tmp`` and, when the variable
    STAGE2_TEST_LOGS names a directory, there as well.  By itself the suite writes nothing under the study tree."""
    for folder in filter(None, (tmp, os.environ.get("STAGE2_TEST_LOGS"))):
        path = Path(folder) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, indent=1))


def _same(x, y) -> bool:
    x, y = np.asarray(x), np.asarray(y)
    return x.shape == y.shape and bool(np.array_equal(x, y, equal_nan=x.dtype.kind == "f"))


_SRC = {}


def _src(body: str) -> dict:
    if body not in _SRC:
        _SRC[body] = refs.source(body)
    return _SRC[body]


@pytest.fixture(scope="module")
def v0() -> dict:
    """V0's 0.40 m/s walk on 75-M-H-10 as a move sees it (no arrays: plan level)."""
    src = _src(BODY75)
    H = src["H"]
    a = moves.GaitX.of("v0.4", look.speed_spec(H), H)
    g = a.legacy()
    return dict(src=src, H=H, a=a, g=g, geo=transitions.Geometry.of(src["M"], src["q_home"], g.crouch))


def _f25f9cl_timing(H: float, v: float = 0.40) -> gait.GaitSpec:
    """The timing of the candidate set's 0.40 m/s walk as the design's prototypes built it (T 0.993 s, duty 0.59 on
    75-M-H-10); plan level only: the set's own spec comes with the set builder."""
    fr = v * v / (gait.G * H)
    sr = round(min(0.94, max(0.62, 0.863 * ((max(fr, 0.025) / 0.05) ** 0.20))), 4)
    return gait.GaitSpec(froude=fr, duty=0.59, lift_frac=0.095, head_amp=0.05, tail_amp=-0.125, sway_gain=1.0,
                         stride_rel=sr, crouch=0.05, zmp_iters=6, zmp_gain=0.5, n=160)


# ------------------------------------------------------------------------------------------------
# 1. Exact equality with Stage 1
# ------------------------------------------------------------------------------------------------
def _same_gait(g1: transitions.Gait, g2: transitions.Gait) -> bool:
    """Equal as dataclasses, field by field with the same types (the stand's NaN period equals itself)."""
    if type(g1) is not type(g2):
        return False
    for f in fields(g1):
        x, y = getattr(g1, f.name), getattr(g2, f.name)
        if type(x) is not type(y) or not (x == y or (x != x and y != y)):
            return False
    return True


@pytest.mark.parametrize("body", look.BODIES)
def test_legacy_gaits_equal_stage1(body):
    """``GaitX.legacy`` is ``transitions.gait_of``'s constructor call for every Stage 1 reference and the stand, the
    four in-place turn gaits included (speed 0.0, exact split)."""
    H = _src(body)["H"]
    names = refs.reference_names(body) + ["stand"]
    assert len(names) == 11
    for name in names:
        mine, theirs = moves.stage1_gait(body, name, H).legacy(), transitions.gait_of(body, name, H)
        assert _same_gait(mine, theirs), name
        if name != "stand":
            assert mine == theirs
    for d in ("left", "right"):
        g = moves.stage1_gait(body, f"in_place_0.50_{d}", H)
        assert g.in_place and g.legacy().speed == 0.0 and g.legacy().split == "exact"
    assert moves.GaitX.stand().legacy().stand and moves.stage1_gait(body, "nominal", H).split == "statics"


def _assert_same_plan(p1: transitions.Plan, p2: transitions.Plan, label: str) -> None:
    for key in PLAN_ARRAYS:
        assert _same(getattr(p1, key), getattr(p2, key)), f"{label}: {key}"
    assert p1.out == p2.out and p1.ramps == p2.ramps and (p1.kind, p1.first, p1.last) == (p2.kind, p2.first, p2.last)
    assert p1.name == p2.name and _same_gait(p1.a, p2.a) and _same_gait(p1.b, p2.b)
    for key in ("initial", "final"):
        assert _same(p1.stand_point[key], p2.stand_point[key]), f"{label}: stand point {key}"
    assert len(p1.swings) == len(p2.swings)
    for s1, s2 in zip(p1.swings, p2.swings, strict=True):
        assert s1[:4] == s2[:4], label
        for i in (4, 5):
            assert _same(s1[i][0], s2[i][0]) and s1[i][1] == s2[i][1], label


@pytest.mark.parametrize("body", look.BODIES)
def test_legacy_plans_equal_stage1(body):
    """Every array of ``moves.plan`` equals ``transitions.plan``'s for all 18 transitions, with the output slice,
    ramps, stand points and swings; the gaits come through ``GaitX.legacy``, not from ``gait_of``."""
    src = _src(body)
    H = src["H"]
    assert len(transitions.TRANSITIONS) == 18
    for name, (a_name, b_name) in transitions.TRANSITIONS.items():
        a, b = transitions.gait_of(body, a_name, H), transitions.gait_of(body, b_name, H)
        geo = transitions.Geometry.of(src["M"], src["q_home"], (b if a.stand else a).crouch)
        mv = moves.stage1(body, name, H)
        assert not mv.causal and mv.schedule.command is None and mv.laws is moves.TODAY and mv.builder == "stage1"
        _assert_same_plan(transitions.plan(name, a, b, geo), moves.plan(mv, geo), f"{body} {name}")


def test_named_builders_are_stage1s_schedules():
    """``start``, ``change`` with one stride of ramp and ``stop_one_step`` plan exactly Stage 1's start, speed change
    and stop; a change refuses two crouches and takes a longer ramp on a longer window."""
    src = _src(BODY)
    H = src["H"]
    nominal, brisk = moves.stage1_gait(BODY, "nominal", H), moves.stage1_gait(BODY, "brisk", H)
    geo = transitions.Geometry.of(src["M"], src["q_home"], nominal.legacy().crouch)
    stand = transitions.Gait("stand")
    for mv, (a, b) in ((moves.start("start", nominal), (stand, nominal.legacy())),
                       (moves.change("nominal_to_brisk", nominal, brisk), (nominal.legacy(), brisk.legacy())),
                       (moves.stop_one_step("stop", nominal), (nominal.legacy(), stand))):
        assert not mv.causal and mv.rules == ("S2A-gates",)
        _assert_same_plan(transitions.plan(mv.name, a, b, geo), moves.plan(mv, geo), mv.name)
    two = moves.change("nominal_to_brisk", nominal, brisk, ramp_strides=2)
    assert two.schedule.window == (-2, 5) and two.schedule.ramps == dict(all=[0.0, 2.0]) and two.args == dict(ramp_strides=2.0)
    pl = moves.plan(two, geo)
    assert pl.period_fn(1.0) == pytest.approx(0.5 * (nominal.legacy().period + brisk.legacy().period))  # halfway
    assert pl.period_fn(2.0) == brisk.legacy().period and pl.period_fn(0.0) == nominal.legacy().period
    deeper = moves.GaitX.of("deep", replace(brisk.spec, crouch=0.05), H)
    with pytest.raises(ValueError, match="one crouch"):
        moves.change("x", nominal, deeper)
    with pytest.raises(ValueError, match="needs a walk"):
        moves.start("x", moves.GaitX.stand())


#: Run in a subprocess (one BLAS thread) on 60-N-T-10 with the Stage 1 run's references:
#: (a) ``nominal_to_brisk`` and ``stop`` through ``transitions.generate`` (``gait_of`` patched to one correction pass)
#:     and through ``moves.generate`` (the same gaits through ``GaitX``), with counters on Stage 1's solver functions;
#: (b) a null command: the nominal walk commanded, causally, to walk on (one solve, no correction pass);
#: (c) the two-touchdown stop from the nominal walk, solved once (no correction), and its summary; the same stop's
#:     kinematic pass; the quick right turn from the nominal walk, solved once, and its summary;
#: (d) the gait accuracy of the process (round one's when it starts) before, inside and after each of those calls.
FAST_CODE = r"""
import json, math, sys
from dataclasses import replace
import numpy as np
from designlab import gait
from designlab.stage1 import refs, transitions
from designlab.stage2 import moves

body, run = sys.argv[1], sys.argv[2]
levels = dict(start=gait.ACCURACY, inside=[], after={})
src = refs.source(body)
H, q_home = src["H"], src["q_home"]
load = lambda name: transitions._load_ref(run, name)
plain = lambda x: json.dumps(refs._plain(x), sort_keys=True)

def same(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return bool(x.shape == y.shape and x.dtype == y.dtype and np.array_equal(x, y, equal_nan=x.dtype.kind == "f"))

def passes(g, n):
    return g if g.spec is None else replace(g, spec=replace(g.spec, zmp_iters=n))

orig = transitions.gait_of
transitions.gait_of = lambda b, name, H: (lambda g: g if g.stand else replace(g, zmp_iters=1))(orig(b, name, H))
out = dict(legacy={})
for name in ("nominal_to_brisk", "stop"):
    a_name, b_name = transitions.TRANSITIONS[name]
    ref_a, ref_b = load(a_name), load(b_name)
    g1 = transitions.generate(src, name, ref_a)
    a1, s1 = transitions.arrays(g1), transitions.summary(g1, ref_a, ref_b, q_home)
    a, b = (passes(moves.stage1_gait(body, n, H), 1) for n in (a_name, b_name))
    mv = moves.Move(name, a, b, moves.legacy_schedule(a.legacy(), b.legacy()), "stage1", dict(body=body, name=name))
    calls, originals = {}, {}
    def counted(mod, fname):
        f = originals[(mod, fname)] = getattr(mod, fname)
        def w(*args, **kw):
            calls[fname] = calls.get(fname, 0) + 1
            levels["inside"].append(gait.ACCURACY)
            return f(*args, **kw)
        setattr(mod, fname, w)
    for fname in ("initial_state", "com_path", "solve_frames", "derivatives", "multibody_zmp", "idyn", "preview_run"):
        counted(transitions, fname)
    for fname in ("com_path_causal", "preview_run_switch", "_causal_setup", "_generate_causal"):
        counted(moves, fname)
    gait.set_accuracy("r1")  # transitions.generate above left round two's level set: Stage 1's behaviour
    g2 = moves.generate(src, mv, ref_a)
    levels["after"]["generate " + name] = gait.ACCURACY
    for (mod, fname), f in originals.items():
        setattr(mod, fname, f)
    a2, s2 = moves.arrays(g2), moves.summary(g2, ref_a, ref_b, q_home)
    out["legacy"][name] = dict(
        keys_stage1=list(a1), keys_moves=list(a2), arrays_equal={k: same(a1[k], a2[k]) for k in a1 if k in a2},
        summary_keys_stage1=list(s1), summary_equal={k: plain(s1[k]) == plain(s2[k]) for k in s1 if k in s2},
        summary_missing=[k for k in s1 if k not in s2], summary_has_seconds="seconds" in s2,
        iters=[g1["zmp_iters"], g2["zmp_iters"]], history=[g1["zmp_error_history_m"], g2["zmp_error_history_m"]],
        calls=calls, causal=g2["causal"], pass_=s2["pass_"], gates=s2["gates"], move=s2["move"])
transitions.gait_of = orig

nominal, ref = moves.stage1_gait(body, "nominal", H), load("nominal")
a = passes(nominal, 0)
null = moves.Move("null_command", a, a, moves.walk_schedule(a.legacy(), (0.0, 2), command=0.0, rule_f=True), "change",
                  {}, causal=True)
g = moves.generate(src, null, ref)
levels["after"]["generate null_command"] = gait.ACCURACY
pl = g["plan"]
ph = pl.phase[pl.out]
kq = np.round(ph * ref.n).astype(int) % ref.n
out["null"] = dict(
    frames=int(pl.n_out), ik_ok=g["ik_ok"],
    com_mm=float(1e3 * np.linalg.norm(g["com"] - moves.walk_frames(ref, ph, pl.first, "com"), axis=1).max()),
    zmp_mm=float(1e3 * np.linalg.norm(g["zmp_mb"] - moves.walk_frames(ref, ph, pl.first, "zmp_mb"), axis=1).max()),
    joint_rad=float(np.abs(g["q"][:, 7:] - ref.q[kq, 7:]).max()), consistency=g["zmp_consistency_history_m"],
    transient=g["causal_transient_m"], join_b=moves.summary(g, ref, ref, q_home)["join_b"])

stop = moves.stop_two_step("stop2_nominal", passes(nominal, 0))
g = moves.generate(src, stop, ref)
pl = g["plan"]
t = pl.t[pl.out] - pl.t[pl.out][0]
bare = moves.forward_acceleration(t, g["com"][:, :2], pl.heading[pl.out])  # at rest before and after, no frame before
m = int(math.ceil(moves.ACCEL_HISTORY_S / ref.step))
full = moves.forward_acceleration(
    t, g["com"][:, :2], pl.heading[pl.out], prev=(float(g["warmup"]["t"][-2]), g["warmup"]["com"][-2, :2]),
    before=(moves.walk_forward_acceleration(ref, pl.first + np.arange(-m, 0) / ref.n), ref.step))
out["stop"] = dict(summary=moves.summary(g, ref, None, q_home), array_keys=list(moves.arrays(g)),
                   frames=int(pl.n_out), solves=len(g["zmp_error_history_m"]),
                   file_touchdowns=moves.touchdowns(moves.arrays(g)["swing"], t), frame_s=float(t[1] - t[0]),
                   warmup_t=[float(g["warmup"]["t"][0]), float(g["warmup"]["t"][-2]), float(g["warmup"]["t"][-1])],
                   bare=dict(lp_peak=float(np.abs(bare[1]).max()), raw_peak=float(np.abs(bare[2]).max()),
                             raw0=float(bare[2][0]), lp0=float(bare[1][0])),
                   full=dict(lp_peak=float(np.abs(full[1]).max()), raw_peak=float(np.abs(full[2]).max()),
                             raw0=float(full[2][0]), lp0=float(full[1][0]), samples=int(len(full[0])),
                             last_sample_s=float(full[0][-1]), duration_s=float(t[-1])),
                   walk_accel_at_command=float(moves.walk_forward_acceleration(ref, [pl.first])[0]))
kin = moves.kinematic(src, stop, ref)
levels["after"]["kinematic stop"] = gait.ACCURACY
out["kinematic"] = dict(ik_ok=kin["ik_ok"], frames=int(len(kin["q"])), same_q=same(kin["q"], g["q"]))
turn = moves.turn("turn90_pi4_right", passes(nominal, 0), math.pi / 2, -math.pi / 4, 0.8)
g = moves.generate(src, turn, ref)
out["turn"] = dict(summary=moves.summary(g, ref, ref, q_home), frames=int(g["plan"].n_out))
out["accuracy"] = dict(levels, inside=sorted(set(levels["inside"])), end=gait.ACCURACY)
print(json.dumps(out))
"""


@pytest.fixture(scope="module")
def fast_builds(tmp_path_factory) -> dict:
    if not ARRAYS:
        pytest.skip("the git-ignored Stage 1 arrays are not here (tools/restore_inputs.py --group picks-arrays)")
    got = _run(FAST_CODE, BODY, str(RUN / BODY), timeout=3600)
    _log(tmp_path_factory.mktemp("moves_fast"), "test_stage2_moves_fast.json", got)
    return got


def test_legacy_generate_equals_stage1_at_one_correction(fast_builds):
    """With each gait's correction run once (its own gain): ``moves.arrays(moves.generate(...))`` has
    ``transitions.arrays(transitions.generate(...))``'s keys in the same order and every key ``array_equal``, for a
    speed change and for the stop (the ``walk.zmp_iters if b.stand`` branch); the summary is equal on every key
    Stage 1 writes but its wall time."""
    for name in ("nominal_to_brisk", "stop"):
        r = fast_builds["legacy"][name]
        assert r["iters"] == [1, 1] and len(r["history"][0]) == 2 and r["history"][0] == r["history"][1]
        assert r["keys_moves"] == r["keys_stage1"] and len(r["keys_stage1"]) == 27
        assert set(r["arrays_equal"]) == set(r["keys_stage1"])
        assert all(r["arrays_equal"].values()), [k for k, v in r["arrays_equal"].items() if not v]
        assert r["summary_missing"] == ["seconds"] and not r["summary_has_seconds"]
        assert set(r["summary_equal"]) == set(r["summary_keys_stage1"]) - {"seconds"}
        assert all(r["summary_equal"].values()), [k for k, v in r["summary_equal"].items() if not v]
        assert r["move"] == dict(kind=transitions.kind_of(*transitions.TRANSITIONS[name]), builder="stage1",
                                 causal=False, command_phase=None, command_t_s=None, rules=[], laws="today")
        assert set(r["gates"]) == {"ik_ok", "join_b", "join_a", "zmp_error"}


def test_legacy_solver_is_stage1s(fast_builds):
    """During a legacy ``generate`` the functions called are Stage 1's own ``initial_state``, ``com_path``,
    ``solve_frames``, ``derivatives``, ``multibody_zmp`` and ``idyn`` (once per pass; the inverse dynamics once),
    never the causal path."""
    for name in ("nominal_to_brisk", "stop"):
        r = fast_builds["legacy"][name]
        assert not r["causal"]
        assert r["calls"] == dict(initial_state=1, com_path=2, preview_run=4, solve_frames=2, derivatives=2,
                                  multibody_zmp=2, idyn=1), name


LEGACY_NAMES = ("start", "stop", "brisk_to_fr05", "enter_walk_0.50_left", "stop_in_place_0.50_right")
#: Full passes, both ways, a fresh model per generation (as ``transitions._job``); each npz through a file object.
LEGACY_CODE = r"""
import hashlib, io, json, sys
import numpy as np
from designlab.stage1 import refs, transitions
from designlab.stage2 import moves

body, run, names = sys.argv[1], sys.argv[2], sys.argv[3:]
recorded = json.loads(open(run + "/transitions.json").read())["transitions"]
plain = lambda x: json.dumps(refs._plain(x), sort_keys=True)

def same(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return bool(x.shape == y.shape and x.dtype == y.dtype and np.array_equal(x, y, equal_nan=x.dtype.kind == "f"))

def sha(arrs):
    buf = io.BytesIO()
    np.savez(buf, **arrs)
    return hashlib.sha256(buf.getvalue()).hexdigest()

out = {}
for name in names:
    a_name, b_name = transitions.TRANSITIONS[name]
    ref_a, ref_b = transitions._load_ref(run, a_name), transitions._load_ref(run, b_name)
    src = refs.source(body)
    g1 = transitions.generate(src, name, ref_a)
    a1, s1 = transitions.arrays(g1), transitions.summary(g1, ref_a, ref_b, src["q_home"])
    src = refs.source(body)
    g2 = moves.generate(src, moves.stage1(body, name, src["H"]), ref_a)
    a2, s2 = moves.arrays(g2), moves.summary(g2, ref_a, ref_b, src["q_home"])
    out[name] = dict(keys=list(a1) == list(a2), arrays_differ=[k for k in a1 if not same(a1[k], a2[k])],
                     summary_differ=[k for k in s1 if k != "seconds" and plain(s1[k]) != plain(s2.get(k))],
                     passes=g2["zmp_iters"], sha256_stage1=sha(a1), sha256_moves=sha(a2),
                     recorded=recorded[name]["sha256"], seconds=[g1["seconds"], g2["seconds"]])
    print(name, out[name]["sha256_moves"] == out[name]["recorded"], out[name]["seconds"], file=sys.stderr, flush=True)
print(json.dumps(out))
"""


@slow
@needs_arrays
def test_legacy_generate_equals_stage1(tmp_path):
    """The same equality at the full 6 passes for a start, the stop, a speed change, a turn entry (exact split) and
    a stop from an in-place gait; each saved npz's sha256 equals the Stage 1 run's record."""
    got = _run(LEGACY_CODE, BODY, str(RUN / BODY), *LEGACY_NAMES, timeout=4 * 3600)
    _log(tmp_path, "test_stage2_moves_legacy.json", got)
    assert set(got) == set(LEGACY_NAMES)
    for name, r in got.items():
        assert r["passes"] == 6 and r["keys"], name
        assert r["arrays_differ"] == [] and r["summary_differ"] == [], name
        assert r["sha256_moves"] == r["sha256_stage1"] == r["recorded"], name


def test_legacy_check_record_is_current():
    """``stage2/sets/legacy_check.json``: both picks complete, 18 of 18 equal, made by today's ``moves.py`` AND
    today's panels' sources (the code the record certifies: ``transitions.py``, ``refs.py``, ``gait.py`` ...), on a
    clean tree, every body by its own run's account.  An edit to ``moves.py`` or to one of the panels' sources fails
    this test until ``python -m designlab.stage2.moves legacy`` is run again for both picks.  The libraries' versions
    are recorded and not compared here: they belong to the machine, not to the tree."""
    from designlab.stage1 import panels

    again = "run again for both picks: python -m designlab.stage2.moves legacy --body <pick> --record stage2/sets/legacy_check.json"
    assert RECORD.exists(), again
    doc = json.loads(RECORD.read_text())
    assert doc["moves_sha256"] == hashlib.sha256((ROOT / "designlab" / "stage2" / "moves.py").read_bytes()).hexdigest(), \
        "moves.py moved: " + again
    assert doc["panel_sources_sha256"] == panels.panel_sources()["sha256"], "the panels' sources moved: " + again
    assert doc["rule"] == moves.LEGACY_RULE and doc["blas_threads"] == 1 and doc["accuracy"] == "r2"
    assert set(doc["bodies"]) == set(look.BODIES)
    for body, b in doc["bodies"].items():
        assert b["complete"] and list(b["names"]) == list(transitions.TRANSITIONS), body
        assert all(e["equal"] and e["sha256"] == e["recorded"] for e in b["names"].values()), body
        assert b["equal"] == b["of"] == 18 and set(b["passes"].values()) == {6}
        stage1 = json.loads((RUN / body / "transitions.json").read_text())["transitions"]
        assert {n: e["recorded"] for n, e in b["names"].items()} == {n: stage1[n]["sha256"] for n in stage1}
        # the body's own provenance: every run that checked some of its names, each name in exactly one run
        assert b["checked"] and sorted(n for c in b["checked"] for n in c["names"]) == sorted(transitions.TRANSITIONS)
        for c in b["checked"]:
            assert c["dirty"] is False and c["dirty_paths"] == [] and len(c["commit"]) == 40 and c["at"], (body, c)
    assert doc["dirty"] is False and doc["dirty_paths"] == [] and len(doc["commit"]) == 40
    assert set(doc["versions"]) == set(panels.library_versions()) and all(doc["versions"].values())


# ------------------------------------------------------------------------------------------------
# 2. The causal builder and the stride rule
# ------------------------------------------------------------------------------------------------
def test_rules_and_constants():
    assert moves.STRIDE_GRID == (1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60)
    assert (moves.TWIST_KINEMATIC, moves.TWIST_FINAL, moves.STEPS_DOWN_MAX) == (0.85, 0.87, 2)
    assert moves.CAUSAL_FADE == (0.5, 1.0) and moves.RAMP_S == 0.375
    assert (moves.CONSISTENCY_LIMIT_M, moves.ZMP_HISTORY_LIMIT_M, moves.CAUSAL_COP_MARGIN_M) == (2.5e-3, 4.0e-3, 0.010)
    assert moves.JOINT_AT_COMMAND_RAD == 1e-3
    assert (moves.TURN_HEADING_MIN_DEG, moves.TURN_CLOSURE_RAD, moves.KAPPA_TOL) == (80.0, 1e-9, 1e-6)
    assert (moves.STOP_TOUCHDOWNS, moves.STOP_ACCEL_LP_LIMIT_M_S2, moves.ACCEL_FS) == (2, 1.5, 500.0)
    assert moves.ACCEL_HISTORY_S == 3.0  # not a gate constant; the reading does not move with it from 1 s on
    assert (transitions.G3_COM_MM, transitions.G3_ZMP_MM) == (1.0, 2.0)  # Stage 1's, which the gates read
    assert (transitions.STOP_SPEED_FRAC, transitions.STOP_WITHIN_STRIDES) == (0.05, 1.5)
    assert set(moves.RULES) == {"S2A-causal", "S2A-rule-F", "S2A-fade", "S2A-ramp", "S2A-appendage", "S2A-stride",
                                "S2A-gates"}
    where = "constants set on the 14 Nov prototypes of V0 on 75-M-H-10, before any closed-loop run"
    assert where in moves.RULES["S2A-gates"] and where in moves.RULES["S2A-stride"]
    # the two causal ZMP limits, each on its own frames (the lead's ruling of 15 Nov, STAGE2_75_PLAN.md A7), word for word
    assert ("consistency error over the first stride after the command at most 2.5 mm; ZMP error from one stride after "
            "it at most 4.0 mm (windows fixed on 15 Nov after the scratch builds of V0, before any set was frozen and "
            "before any closed-loop episode on a Stage 2 set: STAGE2_75_PLAN.md A7)") in moves.RULES["S2A-gates"]
    for text in ("at least 80 deg", "within 1e-09 rad", "thigh twist at most 0.87", "exactly 2 touchdowns",
                 "within 1.5 strides", "within 1.5 m/s^2", "at least 10 mm inside the pad", "joints within 0.001 rad"):
        assert text in moves.RULES["S2A-gates"], text
    assert "under 0.05 mm of CoM" in moves.RULES["S2A-causal"] and "exactly on one grid" in moves.RULES["S2A-causal"]
    assert "0.375 s of reference time" in moves.RULES["S2A-ramp"] and "(T(p) / T_walk)^2" in moves.RULES["S2A-appendage"]
    x = np.linspace(0.0, 1.0, 200001)
    assert moves.RAMP_SQ_INTEGRAL == pytest.approx(float(np.trapezoid(transitions._ramp(x) ** 2, x)), abs=1e-10)
    assert moves.TODAY.name == "today" and moves.TODAY.solver == "stage1"
    assert all(getattr(moves.TODAY, k) is None for k in ("trunk_roll", "foot", "zmp_point", "period_factor"))


def test_a_move_records_its_identity_without_functions(v0):
    mv = moves.turn("turn90_pi4_left", v0["a"], math.pi / 2, math.pi / 4, 0.8)
    d = mv.definition()
    assert json.loads(json.dumps(d, allow_nan=False)) == d  # JSON-plain: no function object, no NaN
    assert d["builder"] == "turn" and d["causal"] and d["laws"] == "today"
    assert d["args"] == dict(angle=math.pi / 2, rate=math.pi / 4, stride_factor=0.8, ramp_s=0.375)
    assert d["specs"]["a"] == d["specs"]["b"] and d["specs"]["a"]["name"] == "v0.4" and d["specs"]["a"]["H"] == v0["H"]
    assert len(d["specs"]["a"]["spec"]) == len(fields(gait.GaitSpec)) and d["specs"]["a"]["split"] == "statics"
    assert d["rules"] == ["S2A-causal", "S2A-rule-F", "S2A-fade", "S2A-ramp", "S2A-appendage", "S2A-stride", "S2A-gates"]
    stop = moves.stop_two_step("stop2_v0.4", v0["a"])
    assert stop.definition()["specs"]["b"] == dict(name="stand", spec=None, H=None, yaw_rate=0.0, in_place=False,
                                                   split="statics")
    assert stop.causal and stop.schedule.command == -0.5 and stop.schedule.window == (-0.5, 3)
    assert not stop.schedule.rule_f and stop.schedule.lift_max == 0.0 and stop.args == {}


def test_incremental_preview_equals_preview_run():
    """On one ZMP sequence the incremental form is ``transitions.preview_run``'s controller (CoM to 1e-12 m); with
    two sequences, nothing before the switch depends on the second."""
    gains = transitions.preview_gains(0.005, 4.5)
    rng = np.random.default_rng(0)
    k = np.arange(3000)
    z = np.cumsum(rng.normal(0.0, 1e-3, len(k))) + 0.05 * np.sin(0.02 * k)
    x0 = np.array([z[0], 0.1, 0.0])
    xs, us = transitions.preview_run(gains, z, x0)
    for k_switch in (0, 700, len(z)):
        xs2, us2 = moves.preview_run_switch(gains, z, z, x0, k_switch)
        assert xs2.shape == xs.shape and np.abs(xs2[:, 0] - xs[:, 0]).max() < 1e-12
        assert np.abs(xs2 - xs).max() < 1e-9 and np.abs(us2 - us).max() < 1e-6 * np.abs(us).max()
    other = z + 0.03 * np.clip((k - 700) / 300.0, 0.0, 1.0)  # a second sequence that leaves the first after step 700
    xa, ua = moves.preview_run_switch(gains, z, other, x0, 700)
    xb, ub = moves.preview_run_switch(gains, z, z, x0, 700)
    assert np.array_equal(xa[:701], xb[:701]) and np.array_equal(ua[:700], ub[:700])
    assert np.abs(xa[900:, 0] - xb[900:, 0]).max() > 1e-3  # and the new information is followed after it


def _stop_and_turn(a: moves.GaitX, f: float = 0.8):
    """Two different commanded plans at the same command phase: the two-touchdown stop (commanded at the right foot's
    lift-off, phase -0.5) and the quick right turn commanded at that same lift-off."""
    return (moves.stop_two_step("stop2", a),
            moves._turn("turn90_pi4_right", a, math.pi / 2, -math.pi / 4, f, p0=-0.5))


def test_nothing_before_the_command_knows(v0):
    """From one V0 0.40 walk, every plan array before the command frame is ``array_equal`` between the stop, the
    turn and the walk that continues; from the command frame on the plans differ."""
    stop, turn = _stop_and_turn(v0["a"])
    ps, pt = moves.plan(stop, v0["geo"]), moves.plan(turn, v0["geo"])
    pw = moves.plan(moves.continuing_walk(stop), v0["geo"])
    k = ps.out.start
    assert k == pt.out.start == pw.out.start == 160 and ps.phase[k] == pt.phase[k] == -0.5 and ps.t[k] == pt.t[k] == 0.0
    for key in PLAN_ARRAYS:
        x, y, w = getattr(ps, key), getattr(pt, key), getattr(pw, key)
        assert _same(x[:k], y[:k]) and _same(x[:k], w[:k]), key
    assert _same(ps.foot_pos[k], pt.foot_pos[k]) and _same(ps.zmp[k], pt.zmp[k])  # the command frame itself
    assert not _same(ps.zmp[k : k + 400], pt.zmp[k : k + 400]) and not _same(pt.heading[k : k + 400], pw.heading[k : k + 400])
    # the first reference the correction starts from, and the continuing walk's, are plans plus the walk's own
    # correction by frame: equal before the command whatever the walk's correction is
    assert moves.continuing_walk(turn).schedule.window == moves.continuing_walk(stop).schedule.window == (-0.5, 5)


@needs_arrays
def test_nothing_before_the_command_knows_com_path():
    """The CoM path before the command frame, and at it, is ``array_equal`` between the stop and the turn from the
    same walk (Stage 1's nominal walk on 75-M-H-10 and its saved reference) when the two preview controllers run on
    the same grid; on their own grids (the turn's frames are closer in time, and the grid is the smallest frame
    interval) the two paths agree to 0.05 mm."""
    src = _src(BODY75)
    a = moves.stage1_gait(BODY75, "nominal", src["H"])
    ref = transitions._load_ref(RUN / BODY75, "nominal")
    stop, turn = _stop_and_turn(a)
    cs, ct = moves._causal_setup(src, stop, ref), moves._causal_setup(src, turn, ref)
    k = cs["k_cmd"]
    assert k == ct["k_cmd"] and _same(cs["x0"], ct["x0"]) and _same(cs["zref0"][:k], ct["zref0"][:k])
    assert _same(cs["zpre_t"], ct["zpre_t"]) and _same(cs["zpre"], ct["zpre"])
    dts = [float(np.min(np.diff(c["pl"].t))) for c in (cs, ct)]
    assert dts[1] < 0.85 * dts[0]
    com = [moves.com_path_causal(c["pl"], c["zref0"], c["zpre_t"], c["zpre"], c["x0"], c["geo"].omega, k, dt=dts[1])
           for c in (cs, ct)]
    assert _same(com[0][0][: k + 1], com[1][0][: k + 1]) and _same(com[0][1][:k], com[1][1][:k])
    assert np.abs(com[0][0][k + 40 : k + 400] - com[1][0][k + 40 : k + 400]).max() > 1e-3  # the plans do differ after it
    own = [moves.com_path_causal(c["pl"], c["zref0"], c["zpre_t"], c["zpre"], c["x0"], c["geo"].omega, k)[0]
           for c in (cs, ct)]
    assert np.abs(own[0][: k + 1] - own[1][: k + 1]).max() < 5e-5
    # and the warm-up stride is the walk's own LIPM path
    walk = moves.walk_frames(ref, cs["pl"].phase[: k + 1], cs["pl"].first, "com_ref")
    assert np.abs(own[0][: k + 1] - walk).max() < 5e-5 and np.abs(own[1][: k + 1] - walk).max() < 5e-5


def test_null_command_reproduces_the_walk(fast_builds):
    """A walk commanded, causally, to walk on: every output frame within the G3 limits of the periodic walk (model
    CoM 1.0 mm, multibody ZMP 2.0 mm; one solve).  What is left to "excuse" is the preview controller's ordinary
    error on its own walk."""
    r = fast_builds["null"]
    assert r["frames"] == 320 and r["ik_ok"] and len(r["consistency"]) == 1
    assert r["com_mm"] <= transitions.G3_COM_MM and r["zmp_mm"] <= transitions.G3_ZMP_MM, r
    assert r["join_b"]["com_ok"] and r["join_b"]["zmp_ok"] and r["joint_rad"] < 1e-3, r
    assert r["transient"][0] < 3e-3, r


def test_rule_f_keeps_committed_footprints(v0):
    """Every stance planted before the command has the walk's footprint exactly (position and yaw differences 0.0);
    swings that lift at or after it land on the commanded path; without the rule the stance that is on the ground
    at the command would have moved."""
    a, geo = v0["a"], v0["geo"]
    turn = moves.turn("turn90_pi4_left", a, math.pi / 2, math.pi / 4, 0.8)
    cmd = turn.schedule.command
    assert turn.schedule.rule_f and cmd == 0.0
    pl, walk = moves.plan(turn, geo), moves.plan(moves.continuing_walk(turn), geo)
    landed = {(s[0], round(s[2], 9)): s[5] for s in walk.swings}
    before = after = 0
    for side, lift_off, touchdown, _lift, _f0, f1 in pl.swings:
        if touchdown < cmd - 1e-9:  # the stance this swing lands in began before the command
            w = landed[(side, round(touchdown, 9))]
            assert np.abs(f1[0] - w[0]).max() == 0.0 and f1[1] - w[1] == 0.0, (side, touchdown)
            before += 1
        else:
            assert lift_off >= cmd - 1e-9  # no swing is in the air at a lift-off
            P, th = pl.pose_fn(0.5 * (touchdown + (lift_off + 1.0)))  # mid-stance of the stance that follows
            c, s = math.cos(th), math.sin(th)
            want = P + np.array([[c, -s], [s, c]]) @ (geo.feet0[side][:2] - geo.origin)
            assert f1[0] == pytest.approx(want, abs=1e-15) and f1[1] == th, (side, touchdown)
            after += 1
    assert before >= 4 and after >= 12
    k = pl.out.start
    assert _same(pl.foot_pos[: k + 1], walk.foot_pos[: k + 1]) and _same(pl.foot_yaw[: k + 1], walk.foot_yaw[: k + 1])
    # the right foot stands through the command (stance from 0.5 - duty to 0.5): the rule keeps it where the walk put it
    j = transitions.SIDES.index("right")
    free = moves.plan(replace(turn, schedule=replace(turn.schedule, rule_f=False)), geo)
    assert not pl.swing[k, j] and np.abs(free.foot_pos[k + 8, j] - walk.foot_pos[k + 8, j]).max() > 1e-4
    assert _same(pl.foot_pos[k + 8, j], walk.foot_pos[k + 8, j])
    with pytest.raises(ValueError, match="command phase"):
        moves.plan(replace(turn, schedule=replace(turn.schedule, command=None)), geo)


def test_command_must_be_a_lift_off(v0):
    """``plan`` raises ``ValueError`` for a causal schedule with command phase 0.2; it accepts 0.0, 0.5 and -0.5."""
    a, g, geo = v0["a"], v0["g"], v0["geo"]

    def commanded(c):
        return moves.Move("m", a, a, moves.walk_schedule(g, (c, math.ceil(c) + 1), command=c, rule_f=True), "change", {},
                          causal=True)

    for c in (0.0, 0.5, -0.5):
        pl = moves.plan(commanded(c), geo)
        assert pl.phase[pl.out.start] == c and pl.t[pl.out.start] == 0.0
    for c in (0.2, 0.25, -0.5 + 1 / 160):
        with pytest.raises(ValueError, match="lift-off"):
            moves.plan(commanded(c), geo)
    with pytest.raises(ValueError, match="lift-off of its inside foot"):
        moves._turn("t", a, math.pi / 2, math.pi / 4, 0.8, p0=0.5)  # a left turn at a right lift-off


def _symmetric(geo: transitions.Geometry) -> transitions.Geometry:
    """The geometry made exactly mirror-symmetric about the path's x axis (the model is, to its solver's tolerance)."""
    oy = float(geo.origin[1])
    L, R = geo.feet0["left"], geo.feet0["right"]
    x, y, z = 0.5 * (L[0] + R[0]), 0.5 * ((L[1] - oy) - (R[1] - oy)), 0.5 * (L[2] + R[2])
    px = 0.5 * (geo.pad_c["left"][0] + geo.pad_c["right"][0])
    py = 0.5 * (geo.pad_c["left"][1] - geo.pad_c["right"][1])
    assert abs(L[1] - oy - y) < 1e-6 and abs(L[0] - x) < 1e-6 and abs(geo.pad_c["left"][1] - py) < 1e-6
    return transitions.Geometry(feet0=dict(left=np.array([x, oy + y, z]), right=np.array([x, oy - y, z])),
                                pad_c=dict(left=np.array([px, py]), right=np.array([px, -py])), origin=geo.origin.copy(),
                                com0=np.array([geo.com0[0], oy, geo.com0[2]]), root0=geo.root0.copy(),
                                quat0=geo.quat0.copy(), omega=geo.omega)


def test_right_turn_mirrors_left(v0):
    """Plan level: foot positions, yaw, ZMP target and heading of the right turn are the left turn's mirrored (the
    feet swapped, half a stride later in phase), to 1e-12."""
    geo = _symmetric(v0["geo"])
    left = moves.plan(moves.turn("l", v0["a"], math.pi / 2, math.pi / 4, 0.8), geo)
    right = moves.plan(moves.turn("r", v0["a"], math.pi / 2, -math.pi / 4, 0.8), geo)
    oy = geo.origin[1]
    n = min(len(left.phase), len(right.phase))  # the common computed frames, from the warm-up stride on
    assert left.out.start == right.out.start == 160 and n == 1080 and right.n_out == 880

    def mirror(xy):
        return np.stack([xy[..., 0], 2 * oy - xy[..., 1]], axis=-1)

    assert right.phase[:n] - left.phase[:n] == pytest.approx(0.5, abs=1e-12)
    assert right.t[:n] == pytest.approx(left.t[:n], abs=1e-12)
    assert right.heading[:n] == pytest.approx(-left.heading[:n], abs=1e-12)
    assert right.zmp[:n] == pytest.approx(mirror(left.zmp[:n]), abs=1e-12)
    assert right.path[:n] == pytest.approx(mirror(left.path[:n]), abs=1e-12)
    for j in (0, 1):
        assert right.foot_pos[:n, j, :2] == pytest.approx(mirror(left.foot_pos[:n, 1 - j, :2]), abs=1e-12)
        assert right.foot_pos[:n, j, 2] == pytest.approx(left.foot_pos[:n, 1 - j, 2], abs=1e-12)
        assert right.foot_yaw[:n, j] == pytest.approx(-left.foot_yaw[:n, 1 - j], abs=1e-12)
        assert np.array_equal(right.swing[:n, j], left.swing[:n, 1 - j])
    assert abs(left.heading[left.out][-1]) == pytest.approx(math.pi / 2, abs=1e-9) and np.ptp(left.heading[:160]) == 0.0


@pytest.mark.parametrize("which,f,ramp_strides", [("V0", 0.8, 0.484), ("F25F9cl", 0.65, 0.458)])
def test_turn_closes_and_ramps_last_0_375_s(v0, which, f, ramp_strides):
    """``kappa`` within 1e-6 of 1, total heading within 1e-9 rad of pi/2; both ramps last 0.375 s of reference time
    for V0 (stride x0.8) and for the candidate's timing (x0.65); the reference's heading at the bar's 2.0 s is at
    least 80 deg on the nominal commanded clock and on the slowest."""
    src, H = v0["src"], v0["H"]
    a = v0["a"] if which == "V0" else moves.GaitX.of("v0.4", _f25f9cl_timing(H), H)
    g = a.legacy()
    geo = transitions.Geometry.of(src["M"], src["q_home"], g.crouch)
    readings = []
    for sign in (1.0, -1.0):
        mv = moves.turn("turn90_pi4", a, math.pi / 2, sign * math.pi / 4, f)
        d = moves.turn_design(g, **mv.args)
        assert abs(d["kappa"] - 1.0) < moves.KAPPA_TOL and d["ramp_s"] == moves.RAMP_S == 0.375
        assert d["ramp_strides"] == pytest.approx(ramp_strides, abs=1e-3) and d["period_in_turn_s"] == f * g.period
        assert d["ramp_strides"] * (g.period + d["period_in_turn_s"]) / 2 == pytest.approx(0.375, abs=1e-12)
        assert d["p0"] == (0.0 if sign > 0 else 0.5) and d["direction"] == d["first_swing"] == ("left" if sign > 0 else "right")
        assert d["hold_strides"] > 0 and mv.schedule.window == (d["p0"], math.ceil(d["p1"] + d["ramp_strides"]) + 2)
        pl = moves.plan(mv, geo)
        o = pl.out
        t, h = pl.t[o] - pl.t[o][0], pl.heading[o]
        assert abs(sign * (h[-1] - h[0]) - math.pi / 2) <= moves.TURN_CLOSURE_RAD
        assert np.ptp(h[int(round((d["p1"] + d["ramp_strides"] - d["p0"]) * 160)) + 1 :]) == 0.0  # closed before the lead-out
        for p in (d["p0"], d["p1"]):  # both ramps, in reference time
            assert np.interp(p + d["ramp_strides"], pl.phase, pl.t) - np.interp(p, pl.phase, pl.t) == pytest.approx(0.375, abs=1e-6)
        hold = (pl.phase > d["p0"] + d["ramp_strides"]) & (pl.phase < d["p1"])
        assert np.diff(pl.t)[hold[:-1] & hold[1:]] == pytest.approx(f * g.period / 160, rel=1e-9)
        assert np.diff(pl.t)[:159] == pytest.approx(g.period / 160, rel=1e-12)
        r = moves.turn_reference(t, h, math.pi / 2, sign * math.pi / 4)
        assert r["at_s"] == 2.0 and r["command_rate"] == 1.05 and r["slowest_clock"] == 0.95
        assert r["heading_at_bar_deg"] >= 80.0 and r["heading_at_slowest_clock_deg"] >= 80.0, r
        assert r["heading_at_slowest_clock_deg"] < r["heading_at_ref_clock_deg"] < r["heading_at_bar_deg"] < 90.0
        assert 1.5 < r["time_to_80deg_bar_s"] < 2.0
        readings.append((r["heading_at_bar_deg"], r["heading_at_slowest_clock_deg"]))
    assert readings[0] == pytest.approx(readings[1], abs=1e-9)  # left and right read the same
    with pytest.raises(ValueError, match="hold would be negative"):
        moves.turn("t", a, 0.2, math.pi / 4, f)  # 0.2 rad at pi/4 rad/s: shorter than its two ramps


def test_appendage_rule(v0):
    """In a turn at stride x0.8 the planned tail and neck amplitudes at the hold are 0.64 of the walk's; in the walk
    before the command they are the walk's."""
    a, g = v0["a"], v0["g"]
    mv = moves.turn("turn90_pi4_left", a, math.pi / 2, math.pi / 4, 0.8)
    assert "S2A-appendage" in mv.rules
    d = moves.turn_design(g, **mv.args)
    pl = moves.plan(mv, v0["geo"])
    p = pl.phase
    tail_wave, neck_wave = np.sin(2 * np.pi * p + np.pi / 2), np.sin(4 * np.pi * p)
    hold = (p >= d["p0"] + d["ramp_strides"]) & (p <= d["p1"])
    before = p < d["p0"]
    after = p >= d["p1"] + d["ramp_strides"]
    assert hold.sum() > 300 and before.sum() == 160 and after.sum() > 300
    assert pl.tail[hold] == pytest.approx(0.64 * g.tail_amp * tail_wave[hold], abs=1e-15)
    assert pl.neck[hold] == pytest.approx(0.64 * g.head_amp * neck_wave[hold], abs=1e-15)
    for sel in (before, after):
        assert np.array_equal(pl.tail[sel], 1.0 * g.tail_amp * tail_wave[sel])
        assert np.array_equal(pl.neck[sel], 1.0 * g.head_amp * neck_wave[sel])
    assert np.abs(pl.tail).max() <= abs(g.tail_amp) and np.abs(pl.neck).max() <= abs(g.head_amp)
    # the stop keeps the walk's amplitudes until its own amplitude ramp, which starts at the last lift-off (phase 0)
    stop = moves.plan(moves.stop_two_step("stop2", a), v0["geo"])
    sel = stop.phase <= 0.0
    assert np.array_equal(stop.tail[sel], 1.0 * g.tail_amp * np.sin(2 * np.pi * stop.phase[sel] + np.pi / 2))
    assert stop.tail[-1] == 0.0 and stop.neck[-1] == 0.0


def test_causal_fade_window(v0):
    """``tau_c`` is zero before the command and from one stride after it; ``post_window_zmp_error_m`` is read over
    p >= c + 1 only and the consistency error over p < c + 1 only (the two limits each read their own frames; the
    all-frames value is the larger of the two); in between the loop corrects the multibody ZMP less the faded
    transient."""
    stop = moves.stop_two_step("stop2", v0["a"])
    pl = moves.plan(stop, v0["geo"])
    c, o, k = -0.5, pl.out, pl.out.start
    rng = np.random.default_rng(1)
    K = len(pl.phase)
    zref = pl.zmp + rng.normal(0.0, 2e-3, (K, 2))
    zct = zref + rng.normal(0.0, 5e-3, (K, 2))
    zmb = pl.zmp + rng.normal(0.0, 3e-3, (K, 2))
    zmb[:2] = zmb[-2:] = np.nan  # no acceleration at the ends of the computed window
    e = moves.causal_errors(pl, c, zmb, zct, zref)
    fade = moves.causal_fade(pl.phase, c)
    full, none = pl.phase <= c + 0.5, pl.phase >= c + 1.0
    assert np.all(fade[full] == 1.0) and np.all(fade[none] == 0.0) and np.all(np.diff(fade) <= 0.0)
    assert fade[np.argmin(np.abs(pl.phase - (c + 0.75)))] == pytest.approx(0.5)
    tau = e["tau_c"]
    assert np.all(tau[:k] == 0.0) and np.all(tau[none] == 0.0) and np.all(e["e"][:k] == 0.0)
    cmd_on = np.arange(K) >= k
    assert np.array_equal(tau[full & cmd_on], (zct - zref)[full & cmd_on])
    err = np.nan_to_num(zmb - pl.zmp)
    assert np.array_equal(e["e"][none], err[none])  # one stride after the command: Stage 1's error
    assert np.array_equal(e["e"][full & cmd_on], np.nan_to_num(zmb - pl.zmp - (zct - zref))[full & cmd_on])
    norm = np.linalg.norm(err, axis=1)
    out_frames = np.zeros(K, bool)
    out_frames[o] = True
    post = out_frames & (pl.phase >= c + 1.0 - 1e-9)
    first = out_frames & ~post  # the first stride after the command: frames k .. k + 159
    assert np.flatnonzero(first)[[0, -1]].tolist() == [k, k + 159] and np.flatnonzero(post)[0] == k + 160
    assert e["post_window"] == norm[post].max()
    assert e["target"] == norm[out_frames].max() and e["post_window"] <= e["target"]
    assert e["transient"] == np.linalg.norm(tau[o], axis=1).max() > 0
    en = np.linalg.norm(e["e"], axis=1)
    assert e["consistency"] == en[first].max() and e["consistency_all"] == en[out_frames].max()
    assert e["consistency_all"] == max(e["consistency"], e["post_window"])  # after the fade e is the ZMP error itself
    # a spike of error inside the first stride shows in the consistency error and not in the post window; one
    # after it shows in the post window (and in the all-frames value) and not in the consistency error
    spike = zmb.copy()
    spike[k + 100] += 0.5
    got = moves.causal_errors(pl, c, spike, zct, zref)
    assert got["post_window"] == e["post_window"] and got["consistency"] > 0.4 and got["consistency_all"] > 0.4
    spike = zmb.copy()
    spike[k + 200] += 0.5
    got = moves.causal_errors(pl, c, spike, zct, zref)
    assert got["post_window"] > 0.4 and got["consistency"] == e["consistency"] and got["consistency_all"] > 0.4
    for edge, in_first in ((k + 159, True), (k + 160, False)):  # the boundary frame belongs to the post window
        spike = zmb.copy()
        spike[edge] += 0.5
        got = moves.causal_errors(pl, c, spike, zct, zref)
        assert (got["consistency"] > 0.4) is in_first and (got["post_window"] > 0.4) is not in_first


def test_pick_stride_factor(v0, monkeypatch):
    """With ``kinematic`` stubbed by a table: the largest f that passes the twist rule AND leaves every other gated
    joint 0.10 rad is chosen (A11 as amended on 15 Nov); both directions must pass; every f tried is recorded with
    both directions' twist use and tightest other joint; a reading the pass does not give is not passed; no f gives
    ``chosen: None``; the full-build step-down (twist over 0.87, an IK failure, or under 0.10 rad on another joint)
    happens at most twice; "over a leg drive" and "a1_over" are told apart by the caller's flags."""
    a = v0["a"]
    table = {}
    calls = []

    def fake_kinematic(src, move, ref_a):
        f, direction = move.args["stride_factor"], "left" if move.args["rate"] > 0 else "right"
        assert move.builder == "turn" and move.causal and move.args["angle"] == math.pi / 2 and move.args["ramp_s"] == 0.375
        calls.append((f, direction))
        use, ik_ok, margin = table[(f, direction)]
        out = dict(ik_ok=ik_ok, ik_fail_frames=0 if ik_ok else 7,
                   range_use=dict(left_hip_yaw=use if direction == "left" else use - 0.05,
                                  right_hip_yaw=use if direction == "right" else use - 0.05, left_knee=0.99))
        if margin is not None:  # None: a pass that gives no reading of the other joints
            out["tightest_other"] = dict(joint=f"{direction}_hip_roll", end="min", margin_rad=margin)
        return out

    monkeypatch.setattr(moves, "kinematic", fake_kinematic)
    monkeypatch.setattr(moves, "steady_turn_kinematic", lambda src, a, rate, f: dict(available=True, f=f))
    assert moves.OTHER_JOINT_MARGIN_RAD == 0.10  # A12's own limit (clearance.RANGE_MARGIN_RAD), not a new constant
    for f in moves.STRIDE_GRID:
        table[(f, "left")] = table[(f, "right")] = (0.80, True, 0.20)
    table[(1.0, "left")] = table[(1.0, "right")] = (1.0, False, 0.20)  # pinned, with IK failures
    table[(0.95, "left")], table[(0.95, "right")] = (0.84, False, 0.20), (0.84, True, 0.20)  # IK fails on one side
    table[(0.90, "left")], table[(0.90, "right")] = (0.84, True, 0.20), (0.851, True, 0.20)  # one direction over 0.85
    table[(0.85, "left")], table[(0.85, "right")] = (0.85, True, 0.10), (0.85, True, 0.10)  # at both limits: passes
    rec = moves.pick_stride_factor(None, a, None, math.pi / 2, math.pi / 4)
    assert rec["chosen"] == 0.85 and rec["steps_down"] == 0 and rec["outcome"] == "picked"
    assert [r["f"] for r in rec["tried"]] == [1.0, 0.95, 0.9, 0.85]
    assert calls == [(f, d) for f in (1.0, 0.95, 0.9, 0.85) for d in ("left", "right")]  # both directions, every f
    assert [r["pass_"] for r in rec["tried"]] == [False, False, False, True]
    assert [r["ik_ok"] for r in rec["tried"]] == [False, False, True, True]
    assert rec["tried"][2]["left"] == 0.84 and rec["tried"][2]["right"] == 0.851
    assert rec["tried"][0]["ik_fail_frames"] == dict(left=7, right=7)
    for r in rec["tried"]:  # both directions' tightest other joint, by name, end and margin
        assert r["tightest_other"] == {d: dict(joint=f"{d}_hip_roll", end="min", margin_rad=table[(r["f"], d)][2])
                                       for d in ("left", "right")}
    assert rec["steady_turn_kinematic"] == dict(available=True, f=0.85)
    assert rec["rule"] == "S2A-stride" and rec["text"] == moves.RULES["S2A-stride"]
    assert (rec["grid"], rec["limit_kinematic"], rec["limit_final"], rec["limit_other_joints_rad"],
            rec["steps_down_max"]) == (list(moves.STRIDE_GRID), 0.85, 0.87, 0.10, 2)
    for text in ("at most 0.85 of the usable thigh-twist range", "every other gated joint at least 0.1 rad",
                 "at most 0.87 and the same 0.1 rad", "STAGE2_75_PLAN.md A12", "amended on 15 Nov", "no limit moved"):
        assert text in rec["text"], text
    json.dumps(rec, allow_nan=False)
    # --- the joint-range half of the rule (A11 as amended): the twist alone no longer picks
    calls.clear()
    for f in moves.STRIDE_GRID:
        table[(f, "left")] = table[(f, "right")] = (0.80, True, 0.20)
    table[(1.0, "left")], table[(1.0, "right")] = (0.80, True, 0.0999), (0.80, True, 0.20)  # one direction under 0.10
    table[(0.95, "left")], table[(0.95, "right")] = (0.80, True, 0.20), (0.80, True, -0.01)  # outside its range
    table[(0.90, "left")], table[(0.90, "right")] = (0.80, True, None), (0.80, True, 0.20)  # not read: not passed
    table[(0.85, "left")], table[(0.85, "right")] = (0.86, True, 0.30), (0.80, True, 0.30)  # the twist still gates
    table[(0.80, "left")], table[(0.80, "right")] = (0.80, True, 0.101), (0.80, True, 0.10)
    by_range = moves.pick_stride_factor(None, a, None, math.pi / 2, math.pi / 4)
    assert by_range["chosen"] == 0.8 and [r["pass_"] for r in by_range["tried"]] == [False] * 4 + [True]
    assert all(r["ik_ok"] for r in by_range["tried"]) and len(calls) == 10
    assert by_range["tried"][0]["tightest_other"]["left"]["margin_rad"] == 0.0999
    assert by_range["tried"][2]["tightest_other"] == dict(
        left=None, right=dict(joint="right_hip_roll", end="min", margin_rad=0.20))
    # the two limits are arguments with the registered defaults
    loose = moves.pick_stride_factor(None, a, None, math.pi / 2, math.pi / 4, other_margin=0.05)
    assert loose["chosen"] == 1.0 and loose["limit_other_joints_rad"] == 0.05
    for f in moves.STRIDE_GRID:
        table[(f, "left")] = table[(f, "right")] = (0.80, True, 0.20)
    table[(1.0, "left")] = table[(1.0, "right")] = (1.0, False, 0.20)
    table[(0.95, "left")], table[(0.95, "right")] = (0.84, False, 0.20), (0.84, True, 0.20)
    table[(0.90, "left")], table[(0.90, "right")] = (0.84, True, 0.20), (0.851, True, 0.20)
    table[(0.85, "left")], table[(0.85, "right")] = (0.85, True, 0.10), (0.85, True, 0.10)
    # --- the full build: within 0.87 and 0.10 rad it stands; the caller's flags tell the two Stage 0 outcomes apart
    ok = moves.step_down(rec, dict(ik_ok=True, hip_yaw_range_use=0.87, tightest_other_rad=0.10))
    assert ok["outcome"] == "frozen" and ok["chosen"] == 0.85 and ok["a1_over"] is False and ok["steps_down"] == 0
    assert ok["builds"] == [dict(f=0.85, ik_ok=True, hip_yaw_range_use=0.87, tightest_other_rad=0.10)]
    a1 = moves.step_down(rec, dict(ik_ok=True, hip_yaw_range_use=0.86, tightest_other_rad=0.2, a1_over=True))
    assert a1["outcome"] == "frozen" and a1["a1_over"] is True and a1["chosen"] == 0.85
    leg = moves.step_down(rec, dict(ik_ok=True, hip_yaw_range_use=0.86, tightest_other_rad=0.2, over_leg_drive=True,
                                    a1_over=True))
    assert leg["outcome"] == "none" and leg["chosen"] is None and "leg drive" in leg["reason"]
    # over 0.87, an IK failure, or under 0.10 rad on another joint: one factor down the grid, at most twice in all
    one = moves.step_down(rec, dict(ik_ok=True, hip_yaw_range_use=0.871, tightest_other_rad=0.2))
    assert (one["outcome"], one["chosen"], one["steps_down"]) == ("rebuild", 0.8, 1) and rec["steps_down"] == 0
    two = moves.step_down(one, dict(ik_ok=False, hip_yaw_range_use=0.80, tightest_other_rad=0.2))
    assert (two["outcome"], two["chosen"], two["steps_down"]) == ("rebuild", 0.75, 2)
    three = moves.step_down(two, dict(ik_ok=True, hip_yaw_range_use=0.88, tightest_other_rad=0.2))
    assert (three["outcome"], three["chosen"], three["steps_down"]) == ("none", None, 2)
    assert [b["f"] for b in three["builds"]] == [0.85, 0.8, 0.75] and "no step-down is left" in three["reason"]
    assert moves.step_down(two, dict(ik_ok=True, hip_yaw_range_use=0.5, tightest_other_rad=0.2))["outcome"] == "frozen"
    with pytest.raises(ValueError, match="nothing was built"):
        moves.step_down(three, dict(ik_ok=True, hip_yaw_range_use=0.5, tightest_other_rad=0.2))
    by_margin = moves.step_down(rec, dict(ik_ok=True, hip_yaw_range_use=0.5, tightest_other_rad=0.0999))
    assert (by_margin["outcome"], by_margin["chosen"], by_margin["steps_down"]) == ("rebuild", 0.8, 1)
    last_margin = moves.step_down(two, dict(ik_ok=True, hip_yaw_range_use=0.5, tightest_other_rad=0.05))
    assert last_margin["outcome"] == "none" and "joint-range rule (0.1 rad" in last_margin["reason"]
    # the built reference's reading of the other joints is required: without it the rule cannot be read
    with pytest.raises(ValueError, match="tightest_other_rad"):
        moves.step_down(rec, dict(ik_ok=True, hip_yaw_range_use=0.5))
    # ... except on a record made before the amendment (no limit_other_joints_rad): the twist alone, as it was
    older = {k: v for k, v in rec.items() if k != "limit_other_joints_rad"}
    assert moves.step_down(older, dict(ik_ok=True, hip_yaw_range_use=0.87))["outcome"] == "frozen"
    assert moves.step_down(older, dict(ik_ok=True, hip_yaw_range_use=0.871))["outcome"] == "rebuild"
    # no factor of the grid passes: nothing is chosen, every f is recorded, no steady reading
    calls.clear()
    for key in table:
        table[key] = (0.90, True, 0.20)
    none = moves.pick_stride_factor(None, a, None, math.pi / 2, math.pi / 4)
    assert none["chosen"] is None and none["outcome"] == "none" and none["steady_turn_kinematic"] is None
    assert [r["f"] for r in none["tried"]] == list(moves.STRIDE_GRID) and len(calls) == 18
    assert "0.85" in none["reason"] and "0.1 rad" in none["reason"]
    # the last factor of the grid has no step-down
    for key in table:
        table[key] = (0.90, True, 0.20) if key[0] > 0.6 else (0.80, True, 0.20)
    last = moves.pick_stride_factor(None, a, None, math.pi / 2, math.pi / 4)
    assert last["chosen"] == 0.6
    assert moves.step_down(last, dict(ik_ok=True, hip_yaw_range_use=0.9, tightest_other_rad=0.2))["outcome"] == "none"


def test_tightest_other_reads_the_usable_ranges(monkeypatch):
    """``tightest_other``: the least margin over every joint but the two thigh twists and every end but one a joint
    rests on by design, against ``plant.usable_ranges`` at full precision (negative outside the range)."""
    from types import SimpleNamespace

    from designlab import statics
    from designlab.stage1 import plant

    names = ["left_hip_yaw", "left_hip_roll", "jaw", "tail_yaw"]
    usable = dict(left_hip_yaw=(-0.3, 0.3), left_hip_roll=(-0.45, 0.2618), jaw=(0.0, 0.5), tail_yaw=(-0.45, 0.45))
    monkeypatch.setattr(plant, "usable_ranges", lambda body: usable)
    monkeypatch.setattr(statics, "jid", lambda m, jn: names.index(jn))
    src = dict(body="B", M=SimpleNamespace(m=SimpleNamespace(jnt_qposadr=np.arange(len(names)))))
    q = np.zeros((5, len(names)))
    q[:, 0] = [-0.29, 0.0, 0.1, 0.2, 0.299]  # the twist: at its limits, never read here
    q[:, 1] = [-0.30, -0.1, 0.0, 0.1, 0.20]
    q[:, 3] = [-0.2, 0.0, 0.1, 0.2, 0.30]
    got = moves.tightest_other(src, q)
    assert got["joint"] == "left_hip_roll" and got["end"] == "max" and got["margin_rad"] == pytest.approx(0.0618)
    q[:, 2] = 0.0  # the jaw rests on its lower stop (clearance.REST_ON_STOP): that end is not read, its other end is
    assert moves.tightest_other(src, q)["joint"] == "left_hip_roll"
    q[:, 2] = [0.0, 0.1, 0.2, 0.3, 0.46]
    got = moves.tightest_other(src, q)
    assert (got["joint"], got["end"]) == ("jaw", "max") and got["margin_rad"] == pytest.approx(0.04)
    q[:, 3] = [-0.5, 0.0, 0.1, 0.2, 0.30]  # outside its range: a negative margin
    got = moves.tightest_other(src, q)
    assert (got["joint"], got["end"]) == ("tail_yaw", "min") and got["margin_rad"] == pytest.approx(-0.05)
    monkeypatch.setattr(plant, "usable_ranges", lambda body: dict(left_hip_yaw=(-0.3, 0.3)))
    assert moves.tightest_other(src, q) is None


def _assert_gates_are_the_rule(s: dict) -> None:
    """Every gate of a causal move's entry is its rule's comparison on the entry's own numbers, with the
    pre-registered limits written out here: a flipped comparison or a moved limit in ``moves.summary`` fails."""
    g, ref, jb, jc = s["gates"], s["reference"], s["join_b"], s["join_command"]
    assert g["ik_ok"] == s["ik_ok"] == (s["ik_fail_frames"] == 0)
    assert g["join_b"] == (jb["com_max_mm"] <= 1.0 and jb["zmp_max_mm"] <= 2.0)
    assert g["join_command"] == jc["ok"] == (jc["warmup_com_max_mm"] <= 1.0 and jc["at_command"]["com_mm"] <= 1.0
                                             and jc["at_command"]["joint_rad"] <= 1e-3
                                             and jc["warmup_zmp_max_mm"] <= 2.0)
    assert g["zmp_consistency"] == (s["zmp_consistency_history_m"][-1] <= 2.5e-3)
    assert g["post_window_zmp"] == (s["post_window_zmp_error_m"][-1] <= 4.0e-3)
    assert g["cop_margin_single"] == (s["cop_margin_single_min"]["m"] >= 0.010)
    extra = set()
    if s["move"]["builder"] == "stop_two_step":
        fwd = ref["forward_stopped_strides"]
        assert g["touchdowns_after_command"] == (ref["touchdowns_after_command"] == 2 == len(ref["touchdown_s"]))
        assert g["stopped_within"] == (fwd is not None and fwd <= 1.5)
        assert g["accel_lp1hz"] == (ref["accel_forward_lp1hz_peak_m_s2"] <= 1.5)
        extra = {"touchdowns_after_command", "stopped_within", "accel_lp1hz"}
    if s["move"]["builder"] == "turn":
        assert g["heading_total"] == (abs(abs(s["turn"]["heading_total_rad"]) - s["turn"]["angle_rad"]) <= 1e-9)
        assert g["heading_at_bar"] == (ref["heading_at_bar_deg"] >= 80.0)
        assert g["heading_at_slowest_clock"] == (ref["heading_at_slowest_clock_deg"] >= 80.0)
        assert g["thigh_twist"] == (s["hip_yaw_range_use"] <= 0.87)
        extra = {"heading_total", "heading_at_bar", "heading_at_slowest_clock", "thigh_twist"}
    assert set(g) == {"ik_ok", "join_b", "join_command", "zmp_consistency", "post_window_zmp",
                      "cop_margin_single"} | extra
    assert s["pass_"] == all(g.values())
    # the all-frames consistency error is recorded beside the two gated windows, gated by nothing: the larger of them
    assert s["zmp_consistency_all_m"] == s["zmp_consistency_all_history_m"][-1]
    assert s["zmp_consistency_all_m"] == max(s["zmp_consistency_history_m"][-1], s["post_window_zmp_error_m"][-1])
    assert len(s["zmp_consistency_all_history_m"]) == len(s["zmp_consistency_history_m"])


def test_turn_summary_and_gates(fast_builds):
    """The quick right turn from Stage 1's nominal walk on 60-N-T-10, solved once (no correction pass): the entry's
    turn readings, and every gate the rule's comparison on the entry's own numbers.  Without a correction the two
    ZMP limits fail, and on this body at stride x0.8 so does the thigh twist; the heading gates and the CoP margin
    hold; the move does not pass."""
    s = fast_builds["turn"]["summary"]
    assert fast_builds["turn"]["frames"] == 880 and s["kind"] == "walk" and s["split"] == "exact"
    assert s["move"] == dict(kind="turn", builder="turn", causal=True, command_phase=0.5, command_t_s=0.0,
                             rules=["S2A-causal", "S2A-rule-F", "S2A-fade", "S2A-ramp", "S2A-appendage",
                                    "S2A-stride", "S2A-gates"], laws="today")
    assert s["join_a"] is None and s["stop"] is None and s["ik_ok"]
    _assert_gates_are_the_rule(s)
    g = s["gates"]
    assert g["heading_total"] and g["heading_at_bar"] and g["heading_at_slowest_clock"] and g["cop_margin_single"]
    assert not g["zmp_consistency"] and not g["post_window_zmp"] and not g["thigh_twist"] and not s["pass_"]
    t, ref = s["turn"], s["reference"]
    assert t["direction"] == t["first_swing"] == "right" and t["stride_factor"] == 0.8 and t["ramp_s"] == 0.375
    assert abs(t["kappa"] - 1.0) < 1e-6 and t["heading_total_rad"] == pytest.approx(-math.pi / 2, abs=1e-9)
    assert ref["at_s"] == 2.0 and 80.0 <= ref["heading_at_slowest_clock_deg"] < ref["heading_at_bar_deg"] < 90.0
    # the walk's own left step, landed before the command (a right lift-off), is labelled; every later step follows it
    assert [(st["side"], st["lift_off_phase"], st["after_command"]) for st in s["steps"][:3]] == [
        ("left", 0.0, False), ("right", 0.5, True), ("left", 1.0, True)]
    assert all(st["after_command"] for st in s["steps"][1:])
    # a turn at constant path speed has next to no 1 Hz forward acceleration once the filter is given the walk on
    # both sides (the first build read 0.3 m/s^2 here: its file began mid-walk with no history)
    assert ref["accel_filter"]["before"] == ref["accel_filter"]["after"] == "the periodic stride of nominal"
    assert ref["accel_forward_lp1hz_peak_m_s2"] < 0.15 < 1.0 < ref["accel_forward_raw_peak_m_s2"]


def test_legacy_steps_have_no_new_key(fast_builds):
    """A move of Stage 1's construction keeps Stage 1's list of steps, key for key (``after_command`` is a causal
    move's): the summary equality above depends on it."""
    for name in ("nominal_to_brisk", "stop"):
        assert fast_builds["legacy"][name]["summary_equal"]["steps"] is True


def test_accuracy_inside_and_after(fast_builds):
    """``generate`` (both solve paths) and ``kinematic`` run at round two's accuracy and leave the process at the
    level they found: round one's, in a process that never set it.  ``kinematic`` is ``generate``'s first solve."""
    acc = fast_builds["accuracy"]
    assert acc["start"] == "r1" and acc["inside"] == ["r2"] and acc["end"] == "r1"
    assert acc["after"] == {"generate nominal_to_brisk": "r1", "generate stop": "r1", "generate null_command": "r1",
                            "kinematic stop": "r1"}
    assert fast_builds["kinematic"] == dict(ik_ok=True, frames=560, same_q=True)


def test_accuracy_is_put_back(monkeypatch):
    """``generate``, ``kinematic`` and the legacy check's job set round two's accuracy for their own duration and
    put the process's level back, also when they raise and whatever the level was (the solvers are stubbed: no
    arrays are built).  A level left set makes a later caller in the same process build round one's walks at r2."""
    H = _src(BODY)["H"]
    mv = moves.stage1(BODY, "start", H)
    stop = moves.stop_two_step("stop2", moves.stage1_gait(BODY, "nominal", H))
    seen = []

    def stub(*args, **kw):
        seen.append(gait.ACCURACY)
        return dict(zmp_iters=6)

    def boom(*args, **kw):
        seen.append(gait.ACCURACY)
        raise RuntimeError("stopped here")

    was = gait.ACCURACY  # another test file may have left a level set: this test puts back what it found
    try:
        for level in ("r1", "r2"):
            gait.set_accuracy(level)
            for target, call in (("_generate_stage1", lambda: moves.generate({}, mv, None)),
                                 ("_generate_causal", lambda: moves.generate({}, stop, None)),
                                 ("_kinematic", lambda: moves.kinematic({}, mv, None))):
                seen.clear()
                monkeypatch.setattr(moves, target, stub)
                assert call() == dict(zmp_iters=6) and seen == ["r2"] and gait.ACCURACY == level, (target, level)
                monkeypatch.setattr(moves, target, boom)
                with pytest.raises(RuntimeError, match="stopped here"):
                    call()
                assert seen == ["r2", "r2"] and gait.ACCURACY == level, (target, level)
        # the legacy check's job: one transition from the stand (no saved reference needed), its solve stubbed
        gait.set_accuracy("r1")
        for var in moves.THREAD_VARS:
            monkeypatch.setenv(var, "1")
        monkeypatch.setattr(refs, "source", lambda body: dict(H=H))
        monkeypatch.setattr(moves, "generate", stub)
        monkeypatch.setattr(moves, "arrays", lambda g: dict(x=np.zeros(3)))
        seen.clear()
        r = moves._legacy_job((BODY, "start", str(RUN)))
        assert seen == ["r2"] and r["accuracy"] == "r2" and gait.ACCURACY == "r1"
        assert (r["body"], r["name"], r["passes"], r["equal"]) == (BODY, "start", 6, False) and len(r["sha256"]) == 64
        monkeypatch.setattr(moves, "generate", boom)
        with pytest.raises(RuntimeError, match="stopped here"):
            moves._legacy_job((BODY, "start", str(RUN)))
        assert gait.ACCURACY == "r1"
    finally:
        gait.set_accuracy(was)


def test_legacy_record_keeps_each_bodys_provenance(tmp_path, monkeypatch):
    """The record's commit and tree state belong to the run that checked a name: each body keeps its own under
    ``checked`` (one entry per run, a name checked again moving to the later run), so a body checked on a dirty tree
    does not read as clean after another body is run on a clean one; the top of the record is the last run's.  A
    record made by other code is not merged into.  (The job is stubbed: nothing is generated.)"""
    for var in moves.THREAD_VARS:
        monkeypatch.setenv(var, "1")
    state = {}
    monkeypatch.setattr(moves, "_git_state", lambda leave_out: dict(state, dirty_paths=list(state["dirty_paths"])))
    monkeypatch.setattr(moves, "_legacy_line", lambda r: None)
    monkeypatch.setattr(moves, "_legacy_job", lambda job: dict(
        body=job[0], name=job[1], sha256="a" * 64, recorded="a" * 64, equal=True, passes=6, accuracy="r2", seconds=0.0))
    record = tmp_path / "sets" / "legacy_check.json"
    names = list(transitions.TRANSITIONS)
    b75, b60 = "75-M-H-10", "60-N-T-10"

    def run(commit, dirty, bodies, which):
        state.update(commit=commit * 40, dirty=bool(dirty), dirty_paths=[" M designlab/gait.py"] if dirty else [])
        doc = moves.legacy_check(bodies, which, record=record)
        on_disk = json.loads(record.read_text())
        assert {k: v for k, v in doc.items() if k != "checked"} == on_disk and not list(tmp_path.rglob("*.tmp"))
        assert doc["checked"] == [dict(body=b, name=n, equal=True) for b in bodies for n in (which or names)]
        return on_disk

    def runs(doc, body):
        return [(c["commit"][0], c["dirty"], c["names"]) for c in doc["bodies"][body]["checked"]]

    doc = run("1", True, [b75], names[:5])  # the first five names of one body, on a dirty tree
    assert runs(doc, b75) == [("1", True, names[:5])] and not doc["bodies"][b75]["complete"]
    assert doc["dirty"] is True and doc["commit"] == "1" * 40 and doc["bodies"][b75]["checked"][0]["at"] == doc["at"]
    assert doc["bodies"][b75]["checked"][0]["dirty_paths"] == [" M designlab/gait.py"]
    doc = run("2", False, [b60], None)  # the other body, complete, on a clean tree
    assert doc["dirty"] is False and doc["commit"] == "2" * 40  # the top of the record: the last run's
    assert runs(doc, b60) == [("2", False, names)] and doc["bodies"][b60]["complete"]
    assert runs(doc, b75) == [("1", True, names[:5])]  # and the first body is still what its own run saw
    doc = run("3", False, [b75], names[3:])  # the rest of the first body, two of its names again
    assert runs(doc, b75) == [("1", True, names[:3]), ("3", False, names[3:])] and doc["bodies"][b75]["complete"]
    assert (doc["bodies"][b75]["equal"], doc["bodies"][b75]["of"]) == (18, 18)
    assert list(doc["bodies"][b75]["names"]) == names and runs(doc, b60) == [("2", False, names)]
    doc = run("4", False, [b75], names[:3])  # the three names of the dirty run again: no dirty entry is left
    assert runs(doc, b75) == [("3", False, names[3:]), ("4", False, names[:3])]
    assert all(c["dirty"] is False for b in doc["bodies"].values() for c in b["checked"])
    monkeypatch.setattr(moves, "LEGACY_RULE", "another rule")  # a record made under other code or rules
    doc = run("5", False, [b60], names[:1])
    assert set(doc["bodies"]) == {b60} and runs(doc, b60) == [("5", False, names[:1])] and doc["rule"] == "another rule"


def test_second_difference_is_on_the_frames_own_grid():
    """Three-point second differences at the frames' own times: exact for a quadratic on a non-uniform grid, the
    plain second difference on a uniform one; the last frame takes its neighbour's value, and so does the first
    unless the frame before it is given."""
    rng = np.random.default_rng(3)
    t = np.cumsum(rng.uniform(0.004, 0.006, 400))
    quad = np.column_stack([0.3 - 0.7 * t + 1.25 * t**2, 2.0 * t - 0.4 * t**2])
    acc = moves.second_difference(t, quad)
    assert acc.shape == (400, 2) and acc == pytest.approx(np.tile([2.5, -0.8], (400, 1)), abs=1e-6)
    x = np.sin(3.0 * t)
    acc = moves.second_difference(t, x)
    assert acc.shape == (400,) and acc[0] == acc[1] and acc[-1] == acc[-2]
    assert acc[1:-1] == pytest.approx(-9.0 * x[1:-1], abs=0.03)  # first order in the difference of the two intervals
    before = t[0] - 0.005
    acc2 = moves.second_difference(t, x, prev=(before, math.sin(3.0 * before)))
    assert np.array_equal(acc2[1:], acc[1:]) and acc2[0] != acc[0] and acc2[0] == pytest.approx(-9.0 * x[0], abs=0.02)
    tu = 0.005378818345924685 * np.arange(300)
    xu = np.sin(7.0 * tu)
    plain = (xu[2:] - 2.0 * xu[1:-1] + xu[:-2]) / tu[1] ** 2
    assert moves.second_difference(tu, xu)[1:-1] == pytest.approx(plain, rel=1e-9, abs=1e-9)
    with pytest.raises(ValueError, match="three frames"):
        moves.second_difference(t[:2], x[:2])
    with pytest.raises(ValueError, match="increase"):
        moves.second_difference(t[::-1], x)


def test_forward_acceleration_reads_no_resampling_comb():
    """A CoM path with a known forward acceleration (peak 1.9 m/s^2), sampled at a walk's frame interval of 5.38 ms
    (no multiple of the filter's 2 ms): the unfiltered reading is the true acceleration at the frames.  The first
    build's way (the path interpolated to 500 Hz, then differentiated twice) reads the knots of its interpolation, a
    quarter and more above the truth; it is rebuilt here to show the size."""
    h = 0.005378818345924685
    t = h * np.arange(600)
    w = 2.0 * math.pi * 1.7
    x = 0.4 * t + (1.9 / w**2) * np.sin(w * t)
    true = -1.9 * np.sin(w * t)
    tu, lp, raw = moves.forward_acceleration(t, np.column_stack([x, 0.0 * x]), np.zeros(len(t)))
    assert raw.shape == t.shape and np.abs(raw[1:-1] - true[1:-1]).max() < 2e-3
    assert np.abs(raw).max() == pytest.approx(1.9, rel=2e-3) and raw[0] == raw[1] and raw[-1] == raw[-2]
    xu = np.interp(tu, t, x)
    comb = np.gradient(np.gradient(xu, tu), tu)
    assert np.abs(comb[5:-5]).max() > 1.25 * 1.9
    # the uniform grid: 500 Hz from the first frame, within the move's own frames
    assert tu[0] == t[0] and np.diff(tu) == pytest.approx(1.0 / moves.ACCEL_FS, abs=1e-12) and lp.shape == tu.shape
    assert tu[-1] <= t[-1] < tu[-1] + 1.0 / moves.ACCEL_FS
    # along the heading: the same path turned by 0.3 rad reads the same along its own heading, nothing across it
    turned = np.column_stack([math.cos(0.3) * x, math.sin(0.3) * x])
    assert moves.forward_acceleration(t, turned, np.full(len(t), 0.3))[2] == pytest.approx(raw, abs=1e-9)
    assert np.abs(moves.forward_acceleration(t, turned, np.full(len(t), 0.3 + math.pi / 2))[2]).max() < 1e-9


def test_forward_acceleration_filter_has_the_tracks_history(monkeypatch):
    """A file cut out of a periodic walk in mid-stride: with the walk's own acceleration before its first frame and
    after its last, the 1 Hz reading over the file's frames is that of the endless walk (to 5e-4 m/s^2 with the 3 s
    of history; to 0.01 with 1 s); given nothing (rest on both sides) it is wrong at both ends by half the walk's
    own reading.  Both peaks are read over the file's own frames only, and rest is a history of zeros."""
    from designlab.stage1 import metrics

    n, K, k0, fs = 160, 368, 37, moves.ACCEL_FS
    h = 0.86 / n

    def walk(k):  # zero-mean, periodic over n frames: the stride's own rhythm (1.16 Hz) and the step's
        p = 2.0 * math.pi * np.asarray(k, float) / n
        return 1.5 * np.sin(p + 0.4) + 0.6 * np.sin(2.0 * p)

    a = walk(k0 + np.arange(K))
    x = np.zeros(K + 2)  # frames -1 .. K: positions whose second difference is the walk's acceleration exactly
    x[1] = 0.4 * h
    for k in range(K):
        x[k + 2] = 2.0 * x[k + 1] - x[k] + h * h * walk(k0 + k)
    t = h * np.arange(K)
    com, heading, prev = np.column_stack([x[1 : K + 1], np.zeros(K)]), np.zeros(K), (-h, [x[0], 0.0])
    m = int(math.ceil(moves.ACCEL_HISTORY_S / h))
    before, after = (walk(k0 + np.arange(-m, 0)), h), (walk(k0 + K + np.arange(m)), h)
    tu, lp, raw = moves.forward_acceleration(t, com, heading, before=before, after=after, prev=prev)
    assert raw[:-1] == pytest.approx(a[:-1], abs=1e-6) and raw[-1] == raw[-2]  # the first frame from the one before
    # the endless walk, by hand: 20 strides on each side, sampled at the same instants
    M = 20 * n
    ts, xs = h * np.arange(-M, K + M), walk(k0 + np.arange(-M, K + M))
    grid = np.arange(-int(M * h * fs), int((K + M) * h * fs)) / fs
    endless = metrics.lowpass(np.interp(grid, ts, xs), fs)[np.searchsorted(grid, tu - 1e-9)]
    assert np.abs(endless).max() > 0.5 and len(tu) == int(t[-1] * fs + 1e-9) + 1
    assert lp == pytest.approx(endless, abs=5e-4)
    bare = moves.forward_acceleration(t, com, heading)  # nothing given: at rest before and after
    assert np.abs(bare[1] - endless)[[0, -1]].min() > 0.3
    assert np.array_equal(bare[0], tu) and np.array_equal(bare[2][1:], raw[1:])
    zeros = (np.zeros(int(moves.ACCEL_HISTORY_S * fs)), 1.0 / fs)
    assert np.array_equal(moves.forward_acceleration(t, com, heading, before=zeros, after=zeros)[1], bare[1])
    # a longer series than the history is cut to it, nearest the file; a shorter one is taken whole
    long_b, long_a = (walk(k0 + np.arange(-3 * m, 0)), h), (walk(k0 + K + np.arange(3 * m)), h)
    assert np.array_equal(moves.forward_acceleration(t, com, heading, before=long_b, after=long_a, prev=prev)[1], lp)
    monkeypatch.setattr(moves, "ACCEL_HISTORY_S", 1.0)
    one = moves.forward_acceleration(t, com, heading, before=before, after=after, prev=prev)
    assert np.array_equal(one[0], tu) and one[1] == pytest.approx(lp, abs=1e-2)
    assert np.abs(one[1] - lp).max() > 1e-3  # 1 s is not enough for the third digit at the file's ends; 3 s is


def test_walk_forward_acceleration():
    """A periodic gait's forward acceleration from its saved model CoM: second differences across the stride
    boundary, along the stride's own heading; periodic in the gait phase.  A steady turn on a circle at constant
    speed has none (its acceleration is across the heading)."""
    n, T = 160, 0.8
    dt, k = T / n, np.arange(n)
    adv = 0.32
    x = adv * k / n + 0.004 * np.sin(4.0 * math.pi * k / n)
    y = 0.01 * np.sin(2.0 * math.pi * k / n)

    def ref_of(name, xy, motion):
        return refs.Reference(name, dict(com=np.column_stack([xy, np.full(n, 0.3)]), t=dt * k, dt=np.array(dt),
                                         stride_motion=np.array(motion, float)))

    straight = ref_of("straight", np.column_stack([x, y]), [0.0, 0.0, 0.0, adv, 0.0])
    af = moves.walk_forward_acceleration(straight, k / n)
    w = 4.0 * math.pi / T
    assert af == pytest.approx(-0.004 * w**2 * np.sin(4.0 * math.pi * k / n), abs=2e-3 * 0.004 * w**2)  # all n frames
    assert np.array_equal(moves.walk_forward_acceleration(straight, k / n - 3.0), af)
    assert np.array_equal(moves.walk_forward_acceleration(straight, [1.0, 2.5, -0.5]), af[[0, 80, 80]])
    phi, R, C = 0.5678, 0.9, np.array([0.2, 0.9])

    def circle(theta):
        return C + R * np.column_stack([np.sin(theta), -np.cos(theta)])

    steady = ref_of("steady", circle(phi * k / n), [phi, C[0], C[1], C[0], C[1]])
    assert np.abs(moves.walk_forward_acceleration(steady, k / n)).max() < 1e-8
    surge = ref_of("surge", circle(phi * k / n + 0.02 * np.sin(2.0 * math.pi * k / n)), [phi, C[0], C[1], C[0], C[1]])
    want = -R * 0.02 * (2.0 * math.pi / T) ** 2 * np.sin(2.0 * math.pi * k / n)
    assert moves.walk_forward_acceleration(surge, k / n) == pytest.approx(want, abs=0.03) and np.abs(want).max() > 1.0


def test_acceleration_history_per_end(monkeypatch):
    """What the 1 Hz filter is given at each end of a move, and that the entry says which: rest for the stand; the
    gait's periodic stride where its reference is one saved stride; the look solver's own reader where it has one;
    else the move's own stride at that end, repeated (the warm-up stride of a causal move, the first or last output
    stride otherwise).  On a move that is the walk itself the three walking histories read the same to 0.01 m/s^2,
    and rest reads otherwise.  (Synthetic frames: no model.)"""
    import types

    n, K, h = 160, 480, 0.86 / 160

    def walk(k):
        p = 2.0 * math.pi * np.asarray(k, float) / n
        return 1.5 * np.sin(p + 0.4) + 0.6 * np.sin(2.0 * p)

    x = np.zeros(K + n + 2)  # frames -n .. K + 1 of a straight walk whose forward acceleration is walk(frame)
    x[1] = 0.4 * h
    for i in range(len(x) - 2):
        x[i + 2] = 2.0 * x[i + 1] - x[i] + h * h * walk(i + 1 - n)

    def xyz(k):
        k = np.asarray(k)
        return np.column_stack([x[k + n], np.zeros(len(k)), np.full(len(k), 0.3)])

    adv = float(x[2 * n] - x[n])
    assert x[2 * n : 3 * n] - x[n : 2 * n] == pytest.approx(adv, abs=1e-9)  # periodic but for the stride's advance
    one = refs.Reference("w", dict(com=xyz(np.arange(n)), t=h * np.arange(n), dt=np.array(h),
                                   stride_motion=np.array([0.0, 0.0, 0.0, adv, 0.0])))
    walking, standing = types.SimpleNamespace(stand=False), types.SimpleNamespace(stand=True)

    def entry(ref_a, ref_b, a=walking, b=walking, causal=True, laws=moves.TODAY):
        pl = types.SimpleNamespace(out=slice(n, n + K), t=h * np.arange(-n, K + 40), heading=np.zeros(n + K + 40), a=a,
                                   b=b, first=0.0, last=3)
        g = dict(plan=pl, com=xyz(np.arange(K)), move=types.SimpleNamespace(laws=laws))
        if causal:
            g["warmup"] = dict(t=h * np.arange(-n, 1), com=xyz(np.arange(-n, 1)))
        r = moves._accel_reading(g, ref_a, ref_b)
        return r["accel_forward_lp1hz_peak_m_s2"], r["accel_filter"]["before"], r["accel_filter"]["after"], r

    stride = "the periodic stride of w"
    exact, before, after, r = entry(one, one)
    assert (before, after) == (stride, stride) and 0.5 < exact < 0.6
    assert r["accel_forward_raw_peak_m_s2"] == pytest.approx(np.abs(walk(np.arange(n))).max(), abs=1e-6)
    assert "frame before the command" in r["accel_filter"]["of"]
    window = types.SimpleNamespace(name="a look window", step=h)  # a reference that is not one saved stride
    own, before, after, _ = entry(window, None)
    assert before == "the move's own warm-up stride, repeated (the gait's reference is not one saved stride)"
    assert after == "the move's own last output stride, repeated (no reference of the gait was given)"
    assert own == pytest.approx(exact, abs=1e-2)
    own, before, after, r = entry(None, window, causal=False)
    assert before == "the move's own first output stride, repeated (no reference of the gait was given)"
    assert after == "the move's own last output stride, repeated (the gait's reference is not one saved stride)"
    assert own == pytest.approx(exact, abs=1e-2) and "takes its neighbour's value" in r["accel_filter"]["of"]
    rest, before, after, _ = entry(None, None, a=standing, b=standing)
    assert (before, after) == ("at rest (the stand)", "at rest (the stand)") and abs(rest - exact) > 0.02
    assert entry(one, one, a=standing, b=standing)[0] == rest  # the stand is rest whatever is handed in
    assert entry(one, None, b=standing)[1:3] == (stride, "at rest (the stand)")
    # the look solver: its own reader where it has one (handed the reference and the phases), else as above
    look_laws = replace(moves.TODAY, name="look", solver="look")
    seen = []

    def reader(ref, phases):
        seen.append((ref, np.asarray(phases)))
        return walk(np.round(np.asarray(phases) * n))

    monkeypatch.setitem(sys.modules, "designlab.stage2.moves_look", types.SimpleNamespace(walk_forward_acceleration=reader))
    got, before, after, _ = entry(window, window, laws=look_laws)
    assert (before, after) == ("the periodic stride of a look window",) * 2 and got == pytest.approx(exact, abs=1e-9)
    m = math.ceil(moves.ACCEL_HISTORY_S / h)
    assert [s[0] for s in seen] == [window, window] and [len(s[1]) for s in seen] == [m, m]
    assert seen[0][1][-1] == pytest.approx(-1.0 / n) and seen[1][1][0] == 3.0  # up to one frame before; from the end on
    monkeypatch.setitem(sys.modules, "designlab.stage2.moves_look", types.SimpleNamespace())
    assert entry(window, window, laws=look_laws)[1].startswith("the move's own warm-up stride, repeated")
    assert entry(one, one, laws=look_laws)[1:3] == (stride, stride)


def test_touchdowns_are_counted_in_the_files_frames():
    """A touchdown is a foot in swing at one frame and in stance at the next, timed at that first stance frame; a
    swing still in the air at the last frame, and a foot that never leaves the ground, are none."""
    t = 0.005 * np.arange(12)
    sw = np.zeros((12, 2), bool)
    assert moves.touchdowns(sw, t) == []
    sw[0:3, 1] = True  # the right foot is in swing at the first frame (the command at its lift-off) and lands at 3
    sw[5:8, 0] = True  # the left foot swings over frames 5 to 7 and lands at 8
    assert moves.touchdowns(sw, t) == [(t[3], "right"), (t[8], "left")]
    sw[10:, 1] = True  # still in the air at the last frame
    assert moves.touchdowns(sw, t) == [(t[3], "right"), (t[8], "left")]
    sw[9, 0] = True  # a third, one frame long
    assert moves.touchdowns(sw, t) == [(t[3], "right"), (t[8], "left"), (t[10], "left")]


def test_acceleration_reading_of_a_built_stop(fast_builds):
    """The entry's acceleration readings are ``forward_acceleration``'s with the walk before the command and rest
    after the stop as the filter's history and the computed frame before the command for the first frame's second
    difference: there the reading is the walk's own (a file read alone starts with its neighbour's value and a
    filter that knows no walk)."""
    r = fast_builds["stop"]
    ref, full, bare = r["summary"]["reference"], r["full"], r["bare"]
    f = ref["accel_filter"]
    assert (f["hz"], f["order"], f["zero_phase"], f["fs_hz"], f["history_s"]) == (1.0, 2, True, 500.0, 3.0)
    assert f["before"] == "the periodic stride of nominal" and f["after"] == "at rest (the stand)"
    assert "frames' own times" in f["of"] and "frame before the command" in f["of"] and "own frames only" in f["read_over"]
    assert ref["accel_forward_lp1hz_peak_m_s2"] == full["lp_peak"] and ref["accel_forward_raw_peak_m_s2"] == full["raw_peak"]
    assert 0.0 < ref["accel_forward_lp1hz_peak_t_s"] < full["duration_s"] and 0.0 < ref["accel_forward_raw_peak_t_s"]
    assert ref["accel_forward_lp1hz_peak_m_s2"] < ref["accel_forward_raw_peak_m_s2"]
    assert r["warmup_t"][1] == pytest.approx(-r["frame_s"], rel=1e-9) and r["warmup_t"][2] == 0.0
    assert r["warmup_t"][0] == pytest.approx(-r["summary"]["gaits"]["a"]["period"], rel=1e-9)
    # the command frame: with the frame before it, the walk's own acceleration there; alone, the neighbour's value
    assert abs(full["raw0"] - r["walk_accel_at_command"]) < 0.05 < abs(bare["raw0"] - r["walk_accel_at_command"])
    assert bare["lp_peak"] != full["lp_peak"] and abs(bare["lp0"] - full["lp0"]) > 0.05
    assert full["samples"] == int(full["duration_s"] * 500.0 + 1e-9) + 1 and full["last_sample_s"] <= full["duration_s"]


def test_causal_join_a_is_null(fast_builds):
    """The summary of a causal move has ``join_a`` and ``stop`` None, ``join_command`` with ``at_command.zmp_mm``,
    the histories, the reference's own readings, and no wall time; every gate is its rule's comparison."""
    r = fast_builds["stop"]
    s = r["summary"]
    assert r["solves"] == 1 and r["frames"] == 560 and len(r["array_keys"]) == 27
    assert s["join_a"] is None and s["stop"] is None and "seconds" not in s
    assert s["move"] == dict(kind="stop", builder="stop_two_step", causal=True, command_phase=-0.5, command_t_s=0.0,
                             rules=["S2A-causal", "S2A-fade", "S2A-gates"], laws="today")
    jc = s["join_command"]
    assert isinstance(jc["at_command"]["zmp_mm"], float) and "not gated" in jc["at_command"]["zmp_note"]
    assert jc["frames"] == 161 and jc["warmup_com_max_mm"] < 1.0 and jc["at_command"]["joint_rad"] < 1e-3
    for key in ("zmp_error_history_m", "zmp_consistency_history_m", "zmp_consistency_all_history_m",
                "causal_transient_m", "post_window_zmp_error_m"):
        assert len(s[key]) == 1 and s[key][0] > 0
    # the all-frames consistency error is recorded beside the two gated windows, and is the larger of them
    assert s["zmp_consistency_all_m"] == s["zmp_consistency_all_history_m"][-1]
    assert s["zmp_consistency_all_m"] == max(s["zmp_consistency_history_m"][-1], s["post_window_zmp_error_m"][-1])
    assert s["output_phase"] == [-0.5, 3] and s["kind"] == "stop" and s["to"] == "stand"
    assert set(s["cop_margin_single_min"]) == {"m", "frame", "foot"}
    ref = s["reference"]
    duty, period = s["gaits"]["a"]["duty"], s["gaits"]["a"]["period"]
    # the touchdowns are counted in the file's own frames; the plan's scheduled ones are recorded beside them
    assert ref["touchdowns_after_command"] == 2 and ref["touchdown_foot"] == ["right", "left"]
    assert [list(x) for x in r["file_touchdowns"]] == [[t, f] for t, f in zip(ref["touchdown_s"], ref["touchdown_foot"])]
    assert ref["touchdowns_scheduled"] == 2 and ref["touchdown_phase"] == pytest.approx([0.5 - duty, 1.0 - duty])
    assert ref["touchdown_scheduled_s"] == pytest.approx([(1.0 - duty) * period, (1.5 - duty) * period])
    for built, planned in zip(ref["touchdown_s"], ref["touchdown_scheduled_s"], strict=True):
        assert -1e-9 <= built - planned < r["frame_s"]  # the first stance frame at or after the scheduled instant
    assert "file's own frames" in ref["touchdown_rule"]
    assert ref["margin_before_command_s"] == pytest.approx((duty - 0.5) * period)
    assert ref["stage1_stop"]["command_phase"] == pytest.approx(transitions.stop_command_phase(duty))
    assert "Stage 1's convention" in ref["stage1_stop"]["command_phase_is"]
    # Stage 1's list of steps begins with the walk's own left step, which landed before the command: labelled
    assert [(st["side"], st["lift_off_phase"], st["after_command"]) for st in s["steps"]] == [
        ("left", -1.0, False), ("right", -0.5, True), ("left", 0.0, True)]
    assert sum(st["after_command"] for st in s["steps"]) == ref["touchdowns_after_command"]
    assert set(s["gates"]) == {"ik_ok", "join_b", "join_command", "zmp_consistency", "post_window_zmp",
                               "cop_margin_single", "touchdowns_after_command", "stopped_within", "accel_lp1hz"}
    assert s["pass_"] == all(s["gates"].values()) and s["gates"]["touchdowns_after_command"]
    assert isinstance(jc["ok"], bool) and s["gates"]["join_command"] == jc["ok"]
    _assert_gates_are_the_rule(s)
    # one solve, no correction: 8.8 mm in the first stride (over 2.5: this gate's False branch), 3.4 mm after it
    assert not s["gates"]["zmp_consistency"] and s["gates"]["post_window_zmp"] and not s["pass_"]
    assert s["gates"]["cop_margin_single"] and s["gates"]["stopped_within"] and s["gates"]["accel_lp1hz"]
    assert s["definition"]["builder"] == "stop_two_step" and s["hip_yaw_range_use"] < 0.1 and s["ik_fail_frames"] == 0
    # JSON-plain; the only NaN is where Stage 1's own entries have it: the stand's period and duty
    stand = s["gaits"]["b"]
    assert stand["name"] == "stand" and math.isnan(stand["period"]) and math.isnan(stand["duty"])
    json.dumps(dict(s, gaits=dict(a=s["gaits"]["a"])), allow_nan=False)


#: The scratch build of the bar's two commanded moves from V0's 0.40 m/s walk on 75-M-H-10: the walk as a set builds
#: it (round two's accuracy, inverse dynamics in the same block), the stride rule's kinematic pick, then the
#: two-touchdown stop and the quick right turn at the picked factor with all passes.
SCRATCH_CODE = r"""
import json, math, sys, time
import numpy as np
from designlab import gait
from designlab.stage1 import refs
from designlab.stage2 import look, moves

body = sys.argv[1]
src = refs.source(body)
H, M, q_home = src["H"], src["M"], src["q_home"]
spec = look.speed_spec(H)
t0 = time.perf_counter()
with look.accuracy("r2"):
    tr = gait.generate(M, q_home, H, spec)
    ref = refs.Reference("v0.4", refs.arrays(src, dict(traj=tr, idr=gait.inverse_dynamics(M, tr))))
    a = moves.GaitX.of("v0.4", spec, H)
    print(f"walk {time.perf_counter() - t0:.0f} s", file=sys.stderr, flush=True)
    pick = moves.pick_stride_factor(src, a, ref, math.pi / 2, math.pi / 4)
    print(f"pick {pick['chosen']} {time.perf_counter() - t0:.0f} s", file=sys.stderr, flush=True)
    out = dict(pick=pick, walk=dict(period_s=float(tr["info"]["period"]), speed_m_s=float(tr["info"]["speed"]),
                                    ik_ok=bool(tr["ik_ok"])), seconds={})
    for mv, ref_b in ((moves.stop_two_step("stop2_v0.4", a), None),
                      (moves.turn("turn90_pi4_right", a, math.pi / 2, -math.pi / 4, pick["chosen"]), ref)):
        g = moves.generate(src, mv, ref, verbose=True)
        out[mv.name] = moves.summary(g, ref, ref_b, q_home)
        out["seconds"][mv.name] = g["seconds"]
        out[mv.name + "/kappa"] = g["kappa"]
        assert gait.ACCURACY == "r2"
print(json.dumps(out))
"""


@slow
def test_stop_and_turn_from_v0(tmp_path):
    """A scratch build of ``stop2_v0.4`` and ``turn90_pi4_right`` from the V0 0.40 walk on 75-M-H-10.  The stop
    passes every gate of the design's sections 3.3 to 3.5 (two touchdowns).  The kinematic pick is the largest
    stride factor within the twist rule that also leaves every other gated joint 0.10 rad (A11 as amended on 15 Nov):
    by the twist alone it would be 0.80, where the inside leg's hip roll keeps under 0.10 rad, so the pick is lower
    (0.70).  The turn built there holds the 80 deg rule, the twist, the 0.10 rad, the CoP margin and its join to the
    walk, and is outside two of A7's gates, as first built on 15 Nov: the warm-up stride's multibody ZMP against the
    walk's (2.26 mm against 2.0 mm) and the consistency error of the first stride after the command (2.63 mm against
    2.5 mm).  That is a result of the rules as registered (the turn has no reference and goes to the owner), pinned
    here as it was found; no constant is changed for it.

    Every reading is checked first and the gates last, in one assertion that names each failed gate of both moves,
    so that one failed gate cannot hide the other readings."""
    got = _run(SCRATCH_CODE, BODY75, timeout=4 * 3600)
    _log(tmp_path, "test_stage2_moves_scratch.json", got)
    pick = got["pick"]
    tried = pick["tried"]
    f = pick["chosen"]
    assert f is not None and f < 0.8, tried  # the twist alone picked 0.80; the hip roll takes it lower
    assert [r["f"] for r in tried] == [x for x in moves.STRIDE_GRID if x >= f] and tried[-1]["f"] == f
    assert [r["pass_"] for r in tried] == [False] * (len(tried) - 1) + [True]
    assert pick["limit_other_joints_rad"] == moves.OTHER_JOINT_MARGIN_RAD == 0.10

    def within(r):  # the rule, written out on a row's own readings
        return bool(r["ik_ok"] and max(r["left"], r["right"]) <= 0.85
                    and min(r["tightest_other"][d]["margin_rad"] for d in ("left", "right")) >= 0.10)

    assert [within(r) for r in tried] == [r["pass_"] for r in tried], tried
    at_08 = next(r for r in tried if r["f"] == 0.8)  # within the twist rule, out by the hip roll alone
    assert at_08["ik_ok"] and max(at_08["left"], at_08["right"]) <= moves.TWIST_KINEMATIC
    assert {d: at_08["tightest_other"][d]["joint"] for d in ("left", "right")} == dict(left="left_hip_roll",
                                                                                         right="right_hip_roll")
    assert max(at_08["tightest_other"][d]["margin_rad"] for d in ("left", "right")) < 0.10
    names = ("stop2_v0.4", "turn90_pi4_right")
    stop, turn = (got[n] for n in names)
    # --- the readings: what each entry holds, and that every gate is its rule's comparison on the entry's numbers
    for s in (stop, turn):
        assert s["join_a"] is None and s["stop"] is None and s["move"]["causal"] and s["move"]["command_t_s"] == 0.0
        for key in ("zmp_error_history_m", "zmp_consistency_history_m", "zmp_consistency_all_history_m",
                    "causal_transient_m", "post_window_zmp_error_m"):
            assert len(s[key]) == 7, key
        _assert_gates_are_the_rule(s)
        assert s["reference"]["accel_filter"]["before"] == "the periodic stride of v0.4"
    ref = stop["reference"]
    assert ref["touchdown_foot"] == ["right", "left"] and ref["accel_filter"]["after"] == "at rest (the stand)"
    assert ref["touchdowns_scheduled"] == 2 and len(ref["touchdown_scheduled_s"]) == 2
    for built, planned in zip(ref["touchdown_s"], ref["touchdown_scheduled_s"], strict=True):
        # V0's duty puts no scheduled touchdown on a frame: the file's time is its own first stance frame, after it
        assert 0.0 < built - planned < stop["duration_s"] / (stop["frames"] - 1)
    assert [st["after_command"] for st in stop["steps"]] == [False, True, True] and stop["move"]["command_phase"] == -0.5
    assert ref["accel_forward_lp1hz_peak_m_s2"] < ref["accel_forward_raw_peak_m_s2"]
    assert 0.0 < ref["accel_forward_raw_peak_t_s"] < stop["duration_s"]
    ref = turn["reference"]
    assert ref["accel_filter"]["after"] == "the periodic stride of v0.4" and ref["at_s"] == 2.0
    assert abs(turn["turn"]["kappa"] - 1.0) < 1e-6 and turn["turn"]["stride_factor"] == f
    assert turn["turn"]["direction"] == "right" and turn["move"]["command_phase"] == 0.5 and turn["split"] == "exact"
    assert turn["turn"]["ramp_s"] == 0.375 and got["turn90_pi4_right/kappa"] == turn["turn"]["kappa"]
    assert [st["after_command"] for st in turn["steps"][:2]] == [False, True]
    # --- the gates: one assertion for both moves (the turn's two failing gates are the result found on 15 Nov)
    failed = {n: sorted(k for k, v in got[n]["gates"].items() if not v) for n in names}
    assert failed == {"stop2_v0.4": [], "turn90_pi4_right": ["join_command", "zmp_consistency"]}, failed
    assert stop["pass_"] and not turn["pass_"]
    # --- and what passing means, with the limits written out
    for s in (stop, turn):
        assert s["ik_ok"] and s["ik_fail_frames"] == 0
        assert s["join_b"]["com_max_mm"] <= 1.0 and s["join_b"]["zmp_max_mm"] <= 2.0, s["join_b"]
        jc = s["join_command"]
        assert jc["warmup_com_max_mm"] <= 1.0 and jc["at_command"]["com_mm"] <= 1.0 and jc["at_command"]["joint_rad"] <= 1e-3
        assert s["post_window_zmp_error_m"][-1] <= 4.0e-3, s["post_window_zmp_error_m"]  # from one stride after it
        assert s["cop_margin_single_min"]["m"] >= 0.010, s["cop_margin_single_min"]
    jc = stop["join_command"]
    assert jc["ok"] and jc["warmup_zmp_max_mm"] <= 2.0, jc
    assert stop["zmp_consistency_history_m"][-1] <= 2.5e-3, stop["zmp_consistency_history_m"]  # the first stride
    # the turn at the picked factor: outside the warm-up ZMP limit and the first stride's consistency limit
    jc = turn["join_command"]
    assert not jc["ok"] and jc["warmup_zmp_max_mm"] == pytest.approx(2.261, rel=0.05) and jc["warmup_zmp_max_mm"] > 2.0
    assert turn["zmp_consistency_history_m"][-1] == pytest.approx(2.634e-3, rel=0.05) > 2.5e-3
    assert turn["hip_yaw_range_use"] <= moves.TWIST_FINAL and turn["gates"]["thigh_twist"]
    ref = stop["reference"]
    assert ref["touchdowns_after_command"] == 2 and ref["forward_stopped_strides"] <= 1.5
    assert ref["accel_forward_lp1hz_peak_m_s2"] <= 1.5
    ref = turn["reference"]
    assert ref["heading_at_bar_deg"] >= 80.0 and ref["heading_at_slowest_clock_deg"] >= 80.0
    assert abs(abs(turn["turn"]["heading_total_rad"]) - math.pi / 2) <= 1e-9
    assert turn["hip_yaw_range_use"] <= 0.87
