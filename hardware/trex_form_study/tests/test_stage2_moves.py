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
   logic on a stubbed kinematic pass, a causal move's summary; with STAGE2_SLOW=1 a scratch build of the two-touchdown
   stop and of the quick right turn from V0's 0.40 m/s walk on 75-M-H-10.

Every test that builds arrays does so in a subprocess with one BLAS thread; tests that need the git-ignored Stage 1
arrays skip without them (``tools/restore_inputs.py``).
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
#: (c) the two-touchdown stop from the nominal walk, solved once (no correction), and its summary.
FAST_CODE = r"""
import json, sys
from dataclasses import replace
import numpy as np
from designlab.stage1 import refs, transitions
from designlab.stage2 import moves

body, run = sys.argv[1], sys.argv[2]
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
            return f(*args, **kw)
        setattr(mod, fname, w)
    for fname in ("initial_state", "com_path", "solve_frames", "derivatives", "multibody_zmp", "idyn", "preview_run"):
        counted(transitions, fname)
    for fname in ("com_path_causal", "preview_run_switch", "_causal_setup", "_generate_causal"):
        counted(moves, fname)
    g2 = moves.generate(src, mv, ref_a)
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
out["stop"] = dict(summary=moves.summary(g, ref, None, q_home), array_keys=list(moves.arrays(g)),
                   frames=int(g["plan"].n_out), solves=len(g["zmp_error_history_m"]))
print(json.dumps(out))
"""


@pytest.fixture(scope="module")
def fast_builds() -> dict:
    if not ARRAYS:
        pytest.skip("the git-ignored Stage 1 arrays are not here (tools/restore_inputs.py --group picks-arrays)")
    return _run(FAST_CODE, BODY, str(RUN / BODY), timeout=3600)


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
def test_legacy_generate_equals_stage1():
    """The same equality at the full 6 passes for a start, the stop, a speed change, a turn entry (exact split) and
    a stop from an in-place gait; each saved npz's sha256 equals the Stage 1 run's record."""
    got = _run(LEGACY_CODE, BODY, str(RUN / BODY), *LEGACY_NAMES, timeout=4 * 3600)
    assert set(got) == set(LEGACY_NAMES)
    for name, r in got.items():
        assert r["passes"] == 6 and r["keys"], name
        assert r["arrays_differ"] == [] and r["summary_differ"] == [], name
        assert r["sha256_moves"] == r["sha256_stage1"] == r["recorded"], name


def test_legacy_check_record_is_current():
    """``stage2/sets/legacy_check.json``: both picks complete, 18 of 18 equal, made by today's ``moves.py`` (an edit
    to ``moves.py`` fails this test until ``python -m designlab.stage2.moves legacy`` is run again for both picks)."""
    assert RECORD.exists(), "run: python -m designlab.stage2.moves legacy --body <pick> --record stage2/sets/legacy_check.json"
    doc = json.loads(RECORD.read_text())
    assert doc["rule"] == moves.LEGACY_RULE and doc["blas_threads"] == 1 and doc["accuracy"] == "r2"
    assert set(doc["bodies"]) == set(look.BODIES)
    for body, b in doc["bodies"].items():
        assert b["complete"] and list(b["names"]) == list(transitions.TRANSITIONS), body
        assert all(e["equal"] and e["sha256"] == e["recorded"] for e in b["names"].values()), body
        assert b["equal"] == b["of"] == 18 and set(b["passes"].values()) == {6}
        stage1 = json.loads((RUN / body / "transitions.json").read_text())["transitions"]
        assert {n: e["recorded"] for n, e in b["names"].items()} == {n: stage1[n]["sha256"] for n in stage1}
    assert doc["moves_sha256"] == hashlib.sha256((ROOT / "designlab" / "stage2" / "moves.py").read_bytes()).hexdigest()
    assert len(doc["panel_sources_sha256"]) == 64 and doc["commit"] and doc["versions"]["numpy"]


# ------------------------------------------------------------------------------------------------
# 2. The causal builder and the stride rule
# ------------------------------------------------------------------------------------------------
def test_rules_and_constants():
    assert moves.STRIDE_GRID == (1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60)
    assert (moves.TWIST_KINEMATIC, moves.TWIST_FINAL, moves.STEPS_DOWN_MAX) == (0.85, 0.87, 2)
    assert moves.CAUSAL_FADE == (0.5, 1.0) and moves.RAMP_S == 0.375
    assert (moves.CONSISTENCY_LIMIT_M, moves.ZMP_HISTORY_LIMIT_M, moves.CAUSAL_COP_MARGIN_M) == (2.5e-3, 4.0e-3, 0.010)
    assert moves.JOINT_AT_COMMAND_RAD == 1e-3
    assert set(moves.RULES) == {"S2A-causal", "S2A-rule-F", "S2A-fade", "S2A-ramp", "S2A-appendage", "S2A-stride",
                                "S2A-gates"}
    where = "constants set on the 14 Nov prototypes of V0 on 75-M-H-10, before any closed-loop run"
    assert where in moves.RULES["S2A-gates"] and where in moves.RULES["S2A-stride"]
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
    p >= c + 1 only; in between the loop corrects the multibody ZMP less the faded transient."""
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
    assert e["post_window"] == norm[out_frames & (pl.phase >= c + 1.0 - 1e-9)].max()
    assert e["target"] == norm[out_frames].max() and e["post_window"] <= e["target"]
    assert e["transient"] == np.linalg.norm(tau[o], axis=1).max() > 0
    assert e["consistency"] == np.linalg.norm(e["e"][o], axis=1).max()
    # a spike of ZMP error before the post window does not show in it; one inside it does
    spike = zmb.copy()
    spike[k + 100] += 0.5
    assert moves.causal_errors(pl, c, spike, zct, zref)["post_window"] == e["post_window"]
    spike[k + 200] += 0.5
    assert moves.causal_errors(pl, c, spike, zct, zref)["post_window"] > 0.4


def test_pick_stride_factor(v0, monkeypatch):
    """With ``kinematic`` stubbed by a table: the largest passing f is chosen; both directions must pass; every f
    tried is recorded; no f gives ``chosen: None``; the full-build step-down happens at most twice; "over a leg
    drive" and "a1_over" are told apart by the caller's flags."""
    a = v0["a"]
    table = {}
    calls = []

    def fake_kinematic(src, move, ref_a):
        f, direction = move.args["stride_factor"], "left" if move.args["rate"] > 0 else "right"
        assert move.builder == "turn" and move.causal and move.args["angle"] == math.pi / 2 and move.args["ramp_s"] == 0.375
        calls.append((f, direction))
        use, ik_ok = table[(f, direction)]
        return dict(ik_ok=ik_ok, ik_fail_frames=0 if ik_ok else 7,
                    range_use=dict(left_hip_yaw=use if direction == "left" else use - 0.05,
                                   right_hip_yaw=use if direction == "right" else use - 0.05, left_knee=0.99))

    monkeypatch.setattr(moves, "kinematic", fake_kinematic)
    monkeypatch.setattr(moves, "steady_turn_kinematic", lambda src, a, rate, f: dict(available=True, f=f))
    for f in moves.STRIDE_GRID:
        table[(f, "left")] = table[(f, "right")] = (0.80, True)
    table[(1.0, "left")] = table[(1.0, "right")] = (1.0, False)  # pinned, with IK failures
    table[(0.95, "left")], table[(0.95, "right")] = (0.84, False), (0.84, True)  # within the twist, IK fails on one side
    table[(0.90, "left")], table[(0.90, "right")] = (0.84, True), (0.851, True)  # one direction over 0.85
    table[(0.85, "left")], table[(0.85, "right")] = (0.85, True), (0.85, True)  # at the limit: passes
    rec = moves.pick_stride_factor(None, a, None, math.pi / 2, math.pi / 4)
    assert rec["chosen"] == 0.85 and rec["steps_down"] == 0 and rec["outcome"] == "picked"
    assert [r["f"] for r in rec["tried"]] == [1.0, 0.95, 0.9, 0.85]
    assert calls == [(f, d) for f in (1.0, 0.95, 0.9, 0.85) for d in ("left", "right")]  # both directions, every f
    assert [r["pass_"] for r in rec["tried"]] == [False, False, False, True]
    assert [r["ik_ok"] for r in rec["tried"]] == [False, False, True, True]
    assert rec["tried"][2]["left"] == 0.84 and rec["tried"][2]["right"] == 0.851
    assert rec["tried"][0]["ik_fail_frames"] == dict(left=7, right=7)
    assert rec["steady_turn_kinematic"] == dict(available=True, f=0.85)
    assert rec["rule"] == "S2A-stride" and rec["text"] == moves.RULES["S2A-stride"]
    assert (rec["grid"], rec["limit_kinematic"], rec["limit_final"], rec["steps_down_max"]) == (
        list(moves.STRIDE_GRID), 0.85, 0.87, 2)
    json.dumps(rec, allow_nan=False)
    # the full build: within 0.87 it stands; the caller's flags tell the two Stage 0 outcomes apart
    ok = moves.step_down(rec, dict(ik_ok=True, hip_yaw_range_use=0.87))
    assert ok["outcome"] == "frozen" and ok["chosen"] == 0.85 and ok["a1_over"] is False and ok["steps_down"] == 0
    a1 = moves.step_down(rec, dict(ik_ok=True, hip_yaw_range_use=0.86, a1_over=True))
    assert a1["outcome"] == "frozen" and a1["a1_over"] is True and a1["chosen"] == 0.85
    leg = moves.step_down(rec, dict(ik_ok=True, hip_yaw_range_use=0.86, over_leg_drive=True, a1_over=True))
    assert leg["outcome"] == "none" and leg["chosen"] is None and "leg drive" in leg["reason"]
    # over 0.87 (or an IK failure): one factor down the grid, at most twice in all
    one = moves.step_down(rec, dict(ik_ok=True, hip_yaw_range_use=0.871))
    assert (one["outcome"], one["chosen"], one["steps_down"]) == ("rebuild", 0.8, 1) and rec["steps_down"] == 0
    two = moves.step_down(one, dict(ik_ok=False, hip_yaw_range_use=0.80))
    assert (two["outcome"], two["chosen"], two["steps_down"]) == ("rebuild", 0.75, 2)
    three = moves.step_down(two, dict(ik_ok=True, hip_yaw_range_use=0.88))
    assert (three["outcome"], three["chosen"], three["steps_down"]) == ("none", None, 2)
    assert [b["f"] for b in three["builds"]] == [0.85, 0.8, 0.75] and "no step-down is left" in three["reason"]
    assert moves.step_down(two, dict(ik_ok=True, hip_yaw_range_use=0.5))["outcome"] == "frozen"
    with pytest.raises(ValueError, match="nothing was built"):
        moves.step_down(three, dict(ik_ok=True, hip_yaw_range_use=0.5))
    # no factor of the grid passes: nothing is chosen, every f is recorded, no steady reading
    calls.clear()
    for key in table:
        table[key] = (0.90, True)
    none = moves.pick_stride_factor(None, a, None, math.pi / 2, math.pi / 4)
    assert none["chosen"] is None and none["outcome"] == "none" and none["steady_turn_kinematic"] is None
    assert [r["f"] for r in none["tried"]] == list(moves.STRIDE_GRID) and len(calls) == 18
    # the last factor of the grid has no step-down
    for key in table:
        table[key] = (0.90, True) if key[0] > 0.6 else (0.80, True)
    last = moves.pick_stride_factor(None, a, None, math.pi / 2, math.pi / 4)
    assert last["chosen"] == 0.6
    assert moves.step_down(last, dict(ik_ok=True, hip_yaw_range_use=0.9))["outcome"] == "none"


def test_causal_join_a_is_null(fast_builds):
    """The summary of a causal move has ``join_a`` and ``stop`` None, ``join_command`` with ``at_command.zmp_mm``,
    the three histories, the reference's own readings, and no wall time."""
    r = fast_builds["stop"]
    s = r["summary"]
    assert r["solves"] == 1 and r["frames"] == 560 and len(r["array_keys"]) == 27
    assert s["join_a"] is None and s["stop"] is None and "seconds" not in s
    assert s["move"] == dict(kind="stop", builder="stop_two_step", causal=True, command_phase=-0.5, command_t_s=0.0,
                             rules=["S2A-causal", "S2A-fade", "S2A-gates"], laws="today")
    jc = s["join_command"]
    assert isinstance(jc["at_command"]["zmp_mm"], float) and "not gated" in jc["at_command"]["zmp_note"]
    assert jc["frames"] == 161 and jc["warmup_com_max_mm"] < 1.0 and jc["at_command"]["joint_rad"] < 1e-3
    for key in ("zmp_error_history_m", "zmp_consistency_history_m", "causal_transient_m", "post_window_zmp_error_m"):
        assert len(s[key]) == 1 and s[key][0] > 0
    assert s["output_phase"] == [-0.5, 3] and s["kind"] == "stop" and s["to"] == "stand"
    assert set(s["cop_margin_single_min"]) == {"m", "frame", "foot"}
    ref = s["reference"]
    assert ref["touchdowns_after_command"] == 2 and len(ref["touchdown_s"]) == 2
    duty = s["gaits"]["a"]["duty"]
    assert ref["touchdown_phase"] == pytest.approx([0.5 - duty, 1.0 - duty])
    assert ref["margin_before_command_s"] == pytest.approx((duty - 0.5) * s["gaits"]["a"]["period"])
    assert ref["stage1_stop"]["command_phase"] == pytest.approx(transitions.stop_command_phase(duty))
    assert "Stage 1's convention" in ref["stage1_stop"]["command_phase_is"]
    assert ref["accel_forward_lp1hz_peak_m_s2"] <= ref["accel_forward_raw_peak_m_s2"]
    assert set(s["gates"]) == {"ik_ok", "join_b", "join_command", "zmp_consistency", "post_window_zmp",
                               "cop_margin_single", "touchdowns_after_command", "stopped_within", "accel_lp1hz"}
    assert s["pass_"] == all(s["gates"].values()) and s["gates"]["touchdowns_after_command"]
    assert isinstance(jc["ok"], bool) and s["gates"]["join_command"] == jc["ok"]
    assert s["definition"]["builder"] == "stop_two_step" and s["hip_yaw_range_use"] < 0.1 and s["ik_fail_frames"] == 0
    json.dumps(s, allow_nan=False)


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
def test_stop_and_turn_from_v0():
    """A scratch build of ``stop2_v0.4`` and ``turn90_pi4_right`` from the V0 0.40 walk on 75-M-H-10 passes every
    gate of the design's sections 3.3 to 3.5 (two touchdowns, the 80 deg rule); the kinematic pick is 0.80."""
    got = _run(SCRATCH_CODE, BODY75, timeout=4 * 3600)
    log = ROOT / "logs" / "test_stage2_moves_scratch.json"  # logs/ is git-ignored: the numbers, for the report
    log.parent.mkdir(exist_ok=True)
    log.write_text(json.dumps(got, indent=1))
    pick = got["pick"]
    assert pick["chosen"] == 0.8, pick["tried"]
    assert [r["f"] for r in pick["tried"]] == [1.0, 0.95, 0.9, 0.85, 0.8]
    assert max(pick["tried"][-1]["left"], pick["tried"][-1]["right"]) <= moves.TWIST_KINEMATIC
    stop, turn = got["stop2_v0.4"], got["turn90_pi4_right"]
    for s in (stop, turn):
        assert s["ik_ok"] and s["ik_fail_frames"] == 0 and s["join_a"] is None and s["stop"] is None
        assert s["join_b"]["com_max_mm"] <= 1.0 and s["join_b"]["zmp_max_mm"] <= 2.0, s["join_b"]
        jc = s["join_command"]
        assert jc["ok"] and jc["warmup_com_max_mm"] <= 1.0 and jc["warmup_zmp_max_mm"] <= 2.0, jc
        assert jc["at_command"]["com_mm"] <= 1.0 and jc["at_command"]["joint_rad"] <= 1e-3
        assert len(s["zmp_consistency_history_m"]) == 7
        assert s["zmp_consistency_history_m"][-1] <= moves.CONSISTENCY_LIMIT_M, s["zmp_consistency_history_m"]
        assert s["post_window_zmp_error_m"][-1] <= moves.ZMP_HISTORY_LIMIT_M, s["post_window_zmp_error_m"]
        assert s["cop_margin_single_min"]["m"] >= moves.CAUSAL_COP_MARGIN_M, s["cop_margin_single_min"]
        assert all(s["gates"].values()) and s["pass_"], s["gates"]
    ref = stop["reference"]
    assert ref["touchdowns_after_command"] == 2 and ref["forward_stopped_strides"] <= 1.5
    assert ref["accel_forward_lp1hz_peak_m_s2"] <= moves.STOP_ACCEL_LP_LIMIT_M_S2
    ref = turn["reference"]
    assert ref["heading_at_bar_deg"] >= 80.0 and ref["heading_at_slowest_clock_deg"] >= 80.0
    assert abs(abs(turn["turn"]["heading_total_rad"]) - math.pi / 2) <= 1e-9
    assert abs(turn["turn"]["kappa"] - 1.0) < 1e-6 and turn["turn"]["stride_factor"] == 0.8
    assert turn["turn"]["direction"] == "right" and turn["move"]["command_phase"] == 0.5
    assert turn["hip_yaw_range_use"] <= moves.TWIST_FINAL and turn["split"] == "exact"
