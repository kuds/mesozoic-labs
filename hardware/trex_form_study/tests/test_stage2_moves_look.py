"""Tests for the look inside moves (``designlab/stage2/moves_look.py``) and the look sets' definitions
(``designlab/stage2/sets.py``: ``LOOK_SETS``, their spec functions, their items).

Run from hardware/trex_form_study:  python -m pytest tests/test_stage2_moves_look.py

1. Today's laws: a move between two of today's walks takes ``moves.TODAY``; the look's laws and solver at the default
   fields give ``moves.TODAY``'s plan arrays, and its generated arrays (a start, a change, the stop and the turn, one
   solve each way).
2. One check per rule on the plan's arrays: L1 duty, L2 tail, L3 settle dip, L4 lift peak, L5 bob, L6 trunk roll,
   L7/L8 the variation as a warp of time on the shared strides, L10 flat feet per stance as the lead settled it on
   16 Nov 2026 (a causal move, a change, a start, the stop), the instant in ``move.command_phase``, L13 the turn in
   place's own flat feet; causality with the look's laws.
3. Windows of K strides: reading one at any phase; the walk in Stage 1's format; a causal move's warm-up and command
   frame against the window (J1); a null command through a whole window boundary (J1, J2, L7, L8); a periodic
   reference cut from a long plan.
4. The look sets' definitions: the walks' specs at the frozen rungs are the ladder's; the items, their names and
   their identities; the registry beside ``sets.SETS``; the turn in place's optional group.

Every test that builds arrays does so in a subprocess with one BLAS thread.  The plan-level tests use a window made
of the spec alone (its times, its step deviations and two smooth variation signals: no frame is solved).
"""

from __future__ import annotations

import inspect
import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import designlab  # noqa: E402,F401  (selects the GL backend before mujoco is imported)
from designlab import gait  # noqa: E402
from designlab.stage1 import control, refs, transitions  # noqa: E402
from designlab.stage2 import look, moves, s2io, setio, sets  # noqa: E402
from designlab.stage2 import moves_look as ml  # noqa: E402

RUN = ROOT / "stage1" / "runs" / "2026-11-08"
BODY = "75-M-H-10"
PY = sys.executable
N = moves.N_FRAMES
PLAN_ARRAYS = ("phase", "t", "foot_pos", "foot_yaw", "swing", "swing_u", "zmp", "heading", "path", "root_z", "neck",
               "tail")
ARRAYS = (RUN / BODY / "refs" / "nominal.npz").exists() and (RUN / BODY / "refs" / "brisk.npz").exists()
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


def _log(path: Path, doc: dict) -> None:
    """A fixture's numbers for a reader, beside its own files in pytest's temporary directory (nothing is written
    into the study)."""
    path.write_text(json.dumps(doc, indent=1))


def _same(x, y) -> bool:
    x, y = np.asarray(x), np.asarray(y)
    return x.shape == y.shape and bool(np.array_equal(x, y, equal_nan=x.dtype.kind == "f"))


def _spec_from(d: dict) -> gait.GaitSpec:
    d = dict(d)
    d["var_band"] = tuple(d["var_band"])
    return gait.GaitSpec(**d)


def _ladder_spec(speed: str) -> gait.GaitSpec:
    """The look ladder's V6 spec of 75-M-H-10 at a speed key, as committed (``stage2/look/look_refs.json``)."""
    return _spec_from(json.loads((ROOT / "stage2" / "look" / "look_refs.json").read_text())
                      ["bodies"][BODY][speed]["V6"]["spec"])


def _one_stride(spec: gait.GaitSpec) -> gait.GaitSpec:
    """A V6 spec without F14: one stride, no step-time, gaze or tail variation (the ladder's V5)."""
    d = gait.LOOK_DEFAULTS
    return replace(spec, strides=1, phase_jitter=d["phase_jitter"], asym=d["asym"], gaze_var=d["gaze_var"],
                   tail_var=d["tail_var"])


def _fake_window(name: str, spec: gait.GaitSpec, H: float) -> ml.Window:
    """A window of the spec with no solved frame: its times, its step deviations (from the spec's seed, as the
    generator draws them) and two smooth signals standing in for the slow variations."""
    info = spec.describe(H)
    K, T = int(spec.strides), float(info["period"])
    t = np.arange(N * K) * (T / N)
    arrays = dict(t=t, dt=np.array(T / N), stride_motion=np.array([0.0, 0.0, 0.0, K * float(info["stride"]), 0.0]),
                  window_strides=np.array(K), tail_var=np.sin(2 * np.pi * t / (K * T)),
                  gaze_var=np.cos(4 * np.pi * t / (K * T)))
    return ml.Window(name, arrays, spec, H)


@pytest.fixture(scope="module")
def ctx() -> dict:
    """75-M-H-10 with the ladder's V6 walks as moves see them (plan level: no frame is solved)."""
    src = look.source(BODY)
    H = src["H"]
    sn, sv = _ladder_spec("nominal"), _ladder_spec("v040")
    out = dict(src=src, H=H, sn=sn, sv=sv, nominal=moves.GaitX.of("nominal", sn, H), v04=moves.GaitX.of("v0.4", sv, H),
               wn=_fake_window("nominal", sn, H), wv=_fake_window("v0.4", sv, H))
    out["n1"] = moves.GaitX.of("nominal", _one_stride(sn), H)
    out["v1"] = moves.GaitX.of("v0.4", _one_stride(sv), H)
    out["geo"] = transitions.Geometry.of(src["M"], src["q_home"], sn.crouch * H)
    return out


# ------------------------------------------------------------------------------------------------
# 1. Today's laws
# ------------------------------------------------------------------------------------------------
def test_rules_are_recorded(ctx):
    assert set(ml.LOOK_RULE_IDS) | set(ml.JOIN_RULE_IDS) | {"S2L-periodic", "S2L-L13"} == set(ml.RULES)
    for key, text in {**ml.RULES, **sets.LOOK_RULES}.items():
        assert key.startswith("S2L-") and len(text) > 40
    assert "S2L-L9" not in ml.RULES  # rolling contact inside a move is not built (L11)
    assert "ends after the instant and no later than the end of the move's last ramp is flat" in ml.RULES["S2L-L10"]
    assert "S2L-L10" not in ml.INPLACE_RULE_IDS and "S2L-L13" in ml.INPLACE_RULE_IDS
    bd = ml.turn(ctx["src"], "turn90_pi4_left", ctx["v04"], ctx["wv"], math.pi / 2, math.pi / 4, 0.8)
    assert bd.move.laws.name == "look" and bd.move.laws.solver == "look"
    assert set(ml.LOOK_RULE_IDS + ml.JOIN_RULE_IDS) <= set(bd.move.rules) and "S2A-appendage" in bd.move.rules
    d = bd.move.definition()
    assert d["laws"] == "look" and json.dumps(d)  # no function object in the identity
    lk = bd.look()
    assert lk["a_phase"] == ml.LEAVE_PHASE["turn_left"] == 2.0 and lk["window_a"]["strides"] == ctx["sv"].strides
    assert lk["a_time_s"] == pytest.approx(float(ctx["wv"].time(2.0))) and json.dumps(lk)


def test_todays_walks_take_todays_laws(ctx):
    """No look field active and no window bound: ``moves.TODAY`` itself.  With ``force`` the look's laws and solver:
    every plan array is today's, no trunk roll, flat feet."""
    src, H = ctx["src"], ctx["H"]
    nominal, brisk = (moves.GaitX.of(n, s, H) for n, s in (("nominal", refs.walk_specs(BODY)["nominal"]),
                                                            ("brisk", refs.walk_specs(BODY)["brisk"])))
    geo = transitions.Geometry.of(src["M"], src["q_home"], nominal.legacy().crouch)
    assert ml.laws_for(nominal, brisk) is moves.TODAY and ml.laws_for(moves.GaitX.stand(), nominal) is moves.TODAY
    assert ml.bind_move(src, moves.start("start", nominal), None, None).move.laws is moves.TODAY
    cases = [moves.start("start", nominal), moves.change("c", nominal, brisk), moves.stop_two_step("stop2", nominal),
             moves.turn("turn", nominal, math.pi / 2, -0.5, 0.9), moves.stop_one_step("stop1", brisk)]
    for mv in cases:
        forced = replace(mv, laws=ml.laws_for(mv.a, mv.b, force=True))
        assert forced.laws.name == "look" and forced.laws.solver == "look"
        p0, p1 = moves.plan(mv, geo), moves.plan(forced, geo)
        for key in PLAN_ARRAYS:
            assert _same(getattr(p0, key), getattr(p1, key)), (mv.name, key)
        assert p1.extras["trunk_roll"] is None and p1.extras["foot"]["flat"] and not p1.extras["foot"]["rolling"].any()


#: Run in a subprocess on 75-M-H-10 with the Stage 1 run's references (one BLAS thread, accuracy r2), every solve
#: once (no correction pass):
#: (a) a start, a change, the two-touchdown stop and a turn between today's walks through ``moves.TODAY`` and through
#:     the look's laws and solver at the default fields: are the files' arrays equal?
#: (b) a look walk of two strides (the ladder's V6 at 0.40 m/s with ``strides=2`` and two correction passes) through
#:     ``moves_look.walk_arrays``; the stop and a turn commanded at the same phase from it; a null command across the
#:     window's boundary; the warm-ups against the window.
BUILD_CODE = r"""
import json, math, sys
from dataclasses import replace
import numpy as np
from designlab.stage1 import refs, transitions
from designlab.stage2 import look, moves, s2io
from designlab.stage2 import moves_look as ml
from designlab import gait

body, run, tmp = sys.argv[1], sys.argv[2], sys.argv[3]
out = {}
src = look.source(body)
H, q_home = src["H"], src["q_home"]
passes = lambda g, n: replace(g, spec=replace(g.spec, zmp_iters=n))

with s2io.r2():
    # (a) today's walks: today's laws against the look's at the default fields
    if run != "-":
        nominal = passes(moves.stage1_gait(body, "nominal", H), 0)
        brisk = passes(moves.stage1_gait(body, "brisk", H), 0)
        ref = transitions._load_ref(__import__("pathlib").Path(run), "nominal")
        eq = {}
        for mv, ra in ((moves.start("start", nominal), None), (moves.change("c", nominal, brisk), ref),
                       (moves.stop_two_step("stop2", nominal), ref),
                       (moves.turn("turn", nominal, math.pi / 2, -0.5, 0.9), ref)):
            a0 = moves.arrays(moves.generate(src, mv, ra))
            g1 = moves.generate(src, replace(mv, laws=ml.laws_for(mv.a, mv.b, force=True)), ra)
            a1 = moves.arrays(g1)
            keys = list(a0)
            eq[mv.name] = dict(keys_first=list(a1)[: len(keys)] == keys, extra_keys=list(a1)[len(keys):],
                               differing=[k for k in keys if not np.array_equal(a0[k], a1[k], equal_nan=a0[k].dtype.kind == "f")],
                               frames=int(len(a0["t"])))
        out["defaults"] = eq

    # (b) a look walk of two strides, and moves that leave and enter it
    _d = dict(json.load(open("stage2/look/look_refs.json"))["bodies"][body]["v040"]["V6"]["spec"])
    _d["var_band"] = tuple(_d["var_band"])
    spec = replace(gait.GaitSpec(**_d), strides=2, zmp_iters=2)
    s1, la, parts = ml.walk_arrays(src, spec)
    path = tmp + "/w2.npz"
    sha = s2io.write_npz(path, s1)
    out["walk"] = dict(file=path, sha256=sha, frames=int(parts["frames"]), strides=int(parts["strides"]),
                       keys=list(s1), stage1_keys=list(ml.STAGE1_KEYS), look_keys=list(la),
                       q_equal=all(bool(np.array_equal(s1[k], la[k])) for k in ("q", "qd", "qdd")),
                       tau_finite=bool(np.all(np.isfinite(s1["tau"]))), rolling_frames=int(parts["rolling_frames"]),
                       split_failed=int(parts["split_failed_frames"]), ik_ok=bool(np.asarray(la["ik_ok"])),
                       period=float(parts["info"]["period"]), window_s=float(parts["info"]["window_s"]),
                       spec=look._plain(spec.__dict__))
    w = ml.Window.load(path, spec, H, "w2")
    a = moves.GaitX.of("w2", replace(spec, zmp_iters=0), H)

    def warm(g, view):
        wu, pl = g["warmup"], g["plan"]
        ph = wu["phase"]
        com = view.frames("com", ph)
        q = view.frames("q", ph)
        ex = pl.extras
        k = len(ph)
        return dict(com_mm=float(1e3 * np.linalg.norm(wu["com"] - com, axis=1).max()),
                    joint_rad=float(np.abs(wu["q"][:, 7:] - q[:, 7:]).max()),
                    pitch=float(np.abs(ex["foot"]["pitch"][:k] - w.at("foot_pitch", view.phi(ph))).max()),
                    pitch_peak=float(ex["foot"]["pitch"][:k].max()),
                    neck=float(np.abs(pl.neck[:k] - w.at("neck", view.phi(ph))).max()),
                    tail=float(np.abs(pl.tail[:k] - w.at("tail", view.phi(ph))).max()),
                    roll=float(np.abs(ex["trunk_roll"][:k] - w.at("trunk_roll", view.phi(ph))).max()),
                    root_z=float(np.abs(pl.root_z[:k] - view.frames("q", ph)[:, 2]).max()))

    bd = ml.stop_two_step(src, "stop2", a, w)
    g = moves.generate(src, bd.move, bd.ref_a)
    e = moves.summary(g, bd.ref_a, bd.ref_b, q_home)
    arr = moves.arrays(g)
    out["stop"] = dict(join_command=e["join_command"], join_events=ml.join_events(g, bd.ref_a), warm=warm(g, bd.ref_a),
                       gates=e["gates"], look=bd.look(), keys=list(arr), rules=e["move"]["rules"],
                       touchdowns=e["reference"]["touchdowns_after_command"],
                       pitch_after=float(np.abs(arr["foot_pitch"][1:][~arr["swing"][1:]]).max()),
                       rolling_after=bool(arr["rolling"][1:].any()), a_time_s=bd.ref_a.a_time_s)
    # the window's own forward acceleration (the history of the move's 1 Hz reading) against second differences of
    # its frames; the move's entry with the look's keys
    cw = np.asarray(w.arrays["com"], float)[:, 0]
    fd = (cw[2:] - 2 * cw[1:-1] + cw[:-2]) / float(w.arrays["dt"]) ** 2
    got = ml.walk_forward_acceleration(bd.ref_a, np.asarray(w.arrays["phase"], float)[1:-1] - bd.ref_a.offset)
    e = ml.entry_extra(bd, g, e)
    out["stop"].update(accel_err=float(np.abs(got - fd).max()), accel_peak=float(np.abs(fd).max()),
                       accel_before=e["reference"]["accel_filter"]["before"], command_phase=e["move"]["command_phase"],
                       rolling_frames=e["rolling_frames"], rolling_split_failed=e["rolling_split_failed_frames"],
                       gates2=e["gates"], pass_=e["pass_"])
    cs = moves._causal_setup(src, bd.move, bd.ref_a)
    tb = ml.bind_move(src, moves._turn("turn", a, math.pi / 2, -math.pi / 4, 0.8, p0=-0.5), w, w)
    ct = moves._causal_setup(src, tb.move, tb.ref_a)
    k = cs["k_cmd"]
    dts = min(float(np.min(np.diff(c["pl"].t))) for c in (cs, ct))
    com = [moves.com_path_causal(c["pl"], c["zref0"], c["zpre_t"], c["zpre"], c["x0"], c["geo"].omega, k, dt=dts)
           for c in (cs, ct)]
    same = lambda x, y: bool(np.array_equal(x, y, equal_nan=True))
    out["causal"] = dict(k=int(k), k_turn=int(ct["k_cmd"]), x0=same(cs["x0"], ct["x0"]),
                         zref0=same(cs["zref0"][:k], ct["zref0"][:k]), zpre=same(cs["zpre"], ct["zpre"]),
                         zpre_t=same(cs["zpre_t"], ct["zpre_t"]),
                         com_before=same(com[0][0][: k + 1], com[1][0][: k + 1]),
                         differ_after=float(np.abs(com[0][0][k + 40 : k + 400] - com[1][0][k + 40 : k + 400]).max()),
                         plan={key: same(getattr(cs["pl"], key)[:k], getattr(ct["pl"], key)[:k])
                               for key in ("phase", "t", "foot_pos", "foot_yaw", "swing", "zmp", "heading", "root_z",
                                           "neck", "tail")},
                         pitch=same(cs["pl"].extras["foot"]["pitch"][: k + 1], ct["pl"].extras["foot"]["pitch"][: k + 1]),
                         roll=same(cs["pl"].extras["trunk_roll"][:k], ct["pl"].extras["trunk_roll"][:k]),
                         walk_com_mm=float(1e3 * np.abs(com[0][0][: k + 1] - bd.ref_a.frames("com_ref", cs["pl"].phase[: k + 1])).max()))

    # a null command across the window's boundary: the walk commanded, causally, to walk on (window b = window a)
    null = moves.Move("null", a, a, moves.walk_schedule(a.legacy(), (0.0, 2), command=0.0, rule_f=True), "change", {},
                      causal=True)
    nb = ml.bind_move(src, null, w, w, a_phase=0.0)
    g = moves.generate(src, nb.move, nb.ref_a)
    pl = g["plan"]
    ph = pl.phase[pl.out]
    env = pl.extras["foot"]["var_env"][pl.out]
    out["null"] = dict(frames=int(pl.n_out), ik_ok=bool(g["ik_ok"]),
                       com_mm=float(1e3 * np.linalg.norm(g["com"] - nb.ref_a.frames("com", ph), axis=1).max()),
                       zmp_mm=float(1e3 * np.nanmax(np.linalg.norm(g["zmp_mb"] - nb.ref_a.frames("zmp_mb", ph), axis=1))),
                       zmp_p90_mm=float(1e3 * np.nanpercentile(np.linalg.norm(g["zmp_mb"] - nb.ref_a.frames("zmp_mb", ph), axis=1), 90)),
                       joint_rad=float(np.abs(g["q"][:, 7:] - nb.ref_a.frames("q", ph)[:, 7:]).max()),
                       env_sum=[float(env.sum(axis=1).min()), float(env.sum(axis=1).max())],
                       t_err=float(np.abs((pl.t[pl.out] - pl.t[pl.out][0]) - (w.time(ph) - w.time(ph[0]))).max()),
                       join_b=ml.compare(g, nb.ref_b, "b", q_home), transient=g["causal_transient_m"],
                       consistency=g["zmp_consistency_history_m"],
                       pitch=float(np.abs(pl.extras["foot"]["pitch"][pl.out] - w.at("foot_pitch", nb.ref_a.phi(ph))).max()),
                       pitch_peak=float(pl.extras["foot"]["pitch"][pl.out].max()),
                       pitch_low=float(pl.extras["foot"]["pitch"][pl.out].min()),
                       pitch_last=float(np.abs(pl.extras["foot"]["pitch"][pl.out][-1] - w.at("foot_pitch", nb.ref_a.phi(ph[-1:]))[0]).max()),
                       rolling_same=float(np.mean(((pl.extras["foot"]["pitch"][pl.out] > 1e-4) & ~pl.swing[pl.out])
                                                  == ((w.at("foot_pitch", nb.ref_a.phi(ph)) > 1e-4) & ~pl.swing[pl.out]))),
                       rolling_frames=int(pl.extras["foot"]["rolling"][pl.out].any(axis=1).sum()))
print(json.dumps(refs._plain(out)))
"""


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> dict:
    tmp = tmp_path_factory.mktemp("look_moves")
    got = _run(BUILD_CODE, BODY, str(RUN / BODY) if ARRAYS else "-", str(tmp), timeout=3600)
    _log(tmp / "built.json", got)
    return got


@needs_arrays
def test_default_fields_give_todays_arrays(built):
    """A start, a change, the stop and the turn between today's walks, generated with the look's laws and solver at
    the default fields: every array of ``transitions.arrays`` is ``array_equal`` to today's, in the same order; the
    look keys come after them."""
    for name, r in built["defaults"].items():
        assert r["keys_first"] and r["differing"] == [], (name, r)
        assert r["extra_keys"] == ["foot_pitch", "rolling", "rolling_contact", "trunk_roll", "bob", "var_env",
                                   "period_factor"], name


# ------------------------------------------------------------------------------------------------
# 2. One check per rule, on the plan's arrays
# ------------------------------------------------------------------------------------------------
def _change(ctx, one_stride: bool = True):
    a, b = (ctx["n1"], ctx["v1"]) if one_stride else (ctx["nominal"], ctx["v04"])
    wa, wb = (None, None) if one_stride else (ctx["wn"], ctx["wv"])
    bd = ml.change(ctx["src"], "nominal_to_v0.4", a, b, wa, wb)
    return bd, moves.plan(bd.move, ctx["geo"])


def test_l1_duty_is_describes_mixed_by_the_ramp(ctx):
    """F3: each swing's duty is ``describe(H)["duty"]`` of the two gaits mixed by the ramp at its lift-off."""
    bd, pl = _change(ctx)
    da, db = ctx["sn"].describe(ctx["H"])["duty"], ctx["sv"].describe(ctx["H"])["duty"]
    assert da != ctx["sn"].duty and db != ctx["sv"].duty and pl.a.duty == da and pl.b.duty == db
    seen = set()
    for _side, lo, td, *_ in pl.swings:
        w = float(transitions._ramp(lo))
        assert 1.0 - (td - lo) == pytest.approx(da + (db - da) * w, abs=1e-12)
        seen.add(round(w, 3))
    assert 0.0 in seen and 1.0 in seen and 0.5 in seen


def test_l2_tail_amplitude_and_phase_at_both_ends(ctx):
    """F9: on the first and the last stride of a change the tail's fitted amplitude and phase are the two gaits'."""
    bd, pl = _change(ctx)
    for sl_, spec in ((slice(0, N), ctx["sn"]), (slice(len(pl.phase) - N - 40, len(pl.phase) - 40), ctx["sv"])):
        A, psi = look._fit_phase(pl.phase[sl_], pl.tail[sl_])
        assert A == pytest.approx(spec.tail_amp, abs=1e-9)
        assert math.remainder(psi - spec.tail_phase, 2 * math.pi) == pytest.approx(0.0, abs=1e-9)
    assert ctx["sn"].tail_phase != ctx["sv"].tail_phase  # the two ends do differ
    # a stand has no tail swing
    st = moves.plan(ml.start(ctx["src"], "start", ctx["n1"], None).move, ctx["geo"])
    assert np.abs(st.tail[st.phase < -0.25]).max() == 0.0


def test_l3_settle_dip_follows_each_touchdown(ctx):
    """F10: the neck dips by ``head_settle`` after each touchdown of the plan, deepest ``settle_lag`` after it; in a
    start the dip grows with the amplitude ramp."""
    bd, pl = _change(ctx)
    spec = ctx["sn"]
    assert spec.head_mode == "stabilise" and spec.settle_lag == ctx["sv"].settle_lag
    n = 0
    for _side, _lo, td, *_ in pl.swings:
        if not pl.phase[0] + 0.1 < td < pl.phase[-1] - 0.3:
            continue
        t_td = float(np.interp(td, pl.phase, pl.t))
        sel = np.flatnonzero((pl.t >= t_td) & (pl.t <= t_td + 2 * spec.settle_lag))
        k = sel[np.argmin(pl.neck[sel])]
        assert pl.t[k] - t_td == pytest.approx(spec.settle_lag, abs=1.01 * float(np.max(np.diff(pl.t))))
        assert pl.neck[k] == pytest.approx(-spec.head_settle, rel=0.02)
        assert np.abs(pl.neck[(pl.t > t_td - 0.05) & (pl.t < t_td)]).max() == 0.0  # level before the touchdown
        n += 1
    assert n >= 10
    st = moves.plan(ml.start(ctx["src"], "start", ctx["n1"], None).move, ctx["geo"])
    assert np.abs(st.neck[st.phase < 0.2]).max() == 0.0 and st.neck.min() == pytest.approx(-spec.head_settle, rel=0.02)


def test_l4_lift_peak_at_each_swing(ctx):
    """F4: each swing's lift peaks at ``lift_peak`` of the swing."""
    bd, pl = _change(ctx)
    lp = ctx["sn"].lift_peak
    assert lp == ctx["sv"].lift_peak == 0.375
    n = 0
    for j in range(2):
        z = pl.foot_pos[:, j, 2]
        runs = np.flatnonzero(np.diff(np.concatenate([[0], pl.swing[:, j].astype(int), [0]])))
        for a, b in zip(runs[::2], runs[1::2], strict=True):
            if a == 0 or b == len(z):
                continue
            k = a + int(np.argmax(z[a:b]))
            assert pl.swing_u[k, j] == pytest.approx(lp, abs=1.5 / (b - a))
            n += 1
    assert n >= 10


def _bob_of(spec: gait.GaitSpec, H: float, phase: np.ndarray) -> np.ndarray:
    """The periodic walk's dip (m) at gait phases, by the generator's own expressions (``gait._generate_look``)."""
    info = spec.describe(H)
    tau = np.mod(phase - (0.5 - info["duty"] / 2) - spec.bob_phase, 0.5) / 0.5
    drop = None if spec.bob_drop is None else spec.bob_drop / (0.5 * info["period"])
    return spec.bob * H * gait.bob_profile(tau, gait.bob_trough(spec, info["duty"]), drop)


def test_l5_bob_is_the_periodic_walks_at_both_ends(ctx):
    """F1/F2: on the first and last strides of a change the pelvis dip is each gait's own; the trough follows each
    touchdown by ``bob_skew`` of the stance; a start has no dip before its first swing."""
    bd, pl = _change(ctx)
    H, geo = ctx["H"], ctx["geo"]
    dip = pl.extras["foot"]["bob"]
    first, last = slice(0, N), slice(len(pl.phase) - N - 40, len(pl.phase) - 40)
    assert np.abs(dip[first] - _bob_of(ctx["sn"], H, pl.phase[first])).max() < 1e-12
    assert np.abs(dip[last] - _bob_of(ctx["sv"], H, pl.phase[last])).max() < 1e-12
    assert np.abs(pl.root_z - (geo.root0[2] - ctx["sn"].crouch * H - dip)).max() < 1e-12  # amp is 1, one crouch
    assert dip[first].max() == pytest.approx(ctx["sn"].bob * H, rel=1e-3)
    d = ctx["sn"].describe(H)["duty"]
    td = -2.0 + 1.0 - d  # the left foot's first touchdown of the output
    k = np.flatnonzero((pl.phase > td) & (pl.phase < td + 0.45))
    assert pl.phase[k[np.argmax(dip[k])]] - td == pytest.approx(ctx["sn"].bob_skew * d, abs=1.0 / N)
    st = moves.plan(ml.start(ctx["src"], "start", ctx["n1"], None).move, ctx["geo"])
    assert np.abs(st.extras["foot"]["bob"][st.phase < 0.15]).max() == 0.0 and st.extras["foot"]["bob"].max() > 0


def test_l6_trunk_roll_toward_the_stance_leg(ctx):
    """F6: the trunk's roll is the walk's law in phase: at the left foot's mid-stance it is -roll_amp, at the right
    foot's +roll_amp (the generator's sign: toward the stance leg); it fades with a start's amplitude."""
    bd, pl = _change(ctx)
    roll = pl.extras["trunk_roll"]
    for sl_, spec in ((slice(0, N), ctx["sn"]), (slice(len(pl.phase) - N - 40, len(pl.phase) - 40), ctx["sv"])):
        d = spec.describe(ctx["H"])["duty"]
        ph = pl.phase[sl_]
        want = -spec.roll_amp * np.cos(2 * np.pi * (ph - (1.0 - d / 2) - spec.roll_phase))
        assert np.abs(roll[sl_] - want).max() < 1e-12
        left_mid = np.argmin(np.abs(np.mod(ph, 1.0) - (1.0 - d / 2)))
        right_mid = np.argmin(np.abs(np.mod(ph, 1.0) - (0.5 - d / 2)))
        assert roll[sl_][left_mid] == pytest.approx(-spec.roll_amp, rel=1e-3)
        assert roll[sl_][right_mid] == pytest.approx(spec.roll_amp, rel=1e-3)
        assert not pl.swing[sl_][left_mid, 0] and not pl.swing[sl_][right_mid, 1]
    st = moves.plan(ml.start(ctx["src"], "start", ctx["n1"], None).move, ctx["geo"])
    assert np.abs(st.extras["trunk_roll"][st.phase <= -0.25]).max() == 0.0


def test_l7_l8_variation_is_a_warp_of_time_on_the_shared_strides(ctx):
    """F14 in a turn from a window of K strides: the envelope is 1 on the warm-up, 0 from one stride after the
    command to two strides before the end and 1 on the last stride; where it is 1 the plan's times are the window's
    own; where it is 0 they are the unvaried plan's; the path, the heading and every footprint are the unvaried
    plan's in phase; the last computed stride lasts the walk's period."""
    src, a, w, geo = ctx["src"], ctx["v04"], ctx["wv"], ctx["geo"]
    assert w.varies and np.abs(w.e).max() > 0.02 and abs(w.e.sum()) < 1e-12
    bd = ml.turn(src, "turn90_pi4_left", a, w, math.pi / 2, math.pi / 4, 0.8)
    pl = moves.plan(bd.move, geo)
    pn = moves.plan(moves.turn("turn90_pi4_left", a, math.pi / 2, math.pi / 4, 0.8), geo)  # today's laws: no warp
    c, last = bd.bind.command, bd.bind.last
    env = pl.extras["foot"]["var_env"]
    ph = pl.phase
    assert np.all(env[ph <= c, 0] == 1.0) and np.all(env[ph >= c + 1, 0] == 0.0)
    assert np.all(env[ph <= last - 2, 1] == 0.0) and np.all(env[ph >= last - 1, 1] == 1.0)
    assert np.all(np.diff(env[(ph > c) & (ph < c + 1), 0]) < 0)
    # geometry in phase: the unvaried plan's
    for key in ("foot_pos", "foot_yaw", "heading", "path", "zmp", "swing"):
        x, y = getattr(pl, key).astype(float), getattr(pn, key).astype(float)
        if key == "foot_pos":  # the lift's shape is the look's (L4): the footprints and the swing's path in x, y
            x, y = x[..., :2], y[..., :2]
        assert float(np.abs(x - y).max()) < 1e-9, key
    assert abs(abs(pl.heading[pl.out][-1] - pl.heading[pl.out][0]) - math.pi / 2) < moves.TURN_CLOSURE_RAD
    # time: the window's on the warm-up and on the last stride, the unvaried plan's between the fades
    va, vb = bd.ref_a, bd.ref_b
    k = pl.out.start
    assert np.abs((pl.t[: k + 1] - pl.t[k]) - (w.time(va.phi(ph[: k + 1])) - w.time(va.phi(c)))).max() < 1e-7
    m = (ph >= last - 1) & (ph <= last)
    assert np.abs((pl.t[m] - pl.t[m][0]) - (w.time(vb.phi(ph[m])) - w.time(vb.phi(ph[m][0])))).max() < 1e-7
    mid = (ph >= c + 1) & (ph <= last - 2)
    assert np.abs(np.diff(pl.t[mid]) - np.diff(pn.t[mid])).max() < 1e-12
    assert np.abs(np.diff(pl.t[: k + 1]) - np.diff(pn.t[: k + 1])).max() > 1e-5  # the warm-up is warped
    # the preview's periodic continuation: the last computed stride lasts T
    T = a.legacy().period
    assert (pl.t[-N] + T) - pl.t[-1] == pytest.approx(pl.t[-1] - pl.t[-2], rel=0.05)
    assert vb.offset == w.K - last and va.offset == 2.0 and va.a_phase == 2.0
    # slow variations ride the same envelopes
    one = moves.plan(ml.turn(src, "t", ctx["v1"], None, math.pi / 2, math.pi / 4, 0.8).move, geo)
    d_tail = np.abs(pl.tail - one.tail)
    assert d_tail[mid].max() < 1e-12 and d_tail[: k + 1].max() > 1e-3


def _stances(pl, j: int) -> list:
    """(touchdown or None, lift-off) of every stance of foot ``j`` that ends at a lift-off of the plan, in order."""
    side = transitions.SIDES[j]
    sw = sorted((s for s in pl.swings if s[0] == side), key=lambda s: s[1])
    return [(sw[i - 1][2] if i else None, s[1]) for i, s in enumerate(sw)]


def _stance_frames(pl, j: int, td, lo) -> np.ndarray:
    ph = pl.phase
    return (~pl.swing[:, j]) & (ph < lo - 1e-9) & ((ph >= td - 1e-9) if td is not None else True)


def _check_l10(pl, foot, p_in: float, p_out: float, toes: bool = True) -> dict:
    """L10 as settled, read on a plan: every stance that ends after ``p_in`` and no later than ``p_out`` has
    ``foot_pitch`` 0 and ``rolling`` false through it; counts the stances of each kind that the plan holds whole."""
    pitch, rolling = foot["pitch"], foot["rolling"]
    n = dict(before=0, flat=0, after=0)
    for j in (0, 1):
        for td, lo in _stances(pl, j):
            sel = _stance_frames(pl, j, td, lo)
            if not sel.any() or lo > pl.phase[-1]:
                continue
            if p_in + 1e-9 < lo <= p_out + 1e-9:
                assert not rolling[sel, j].any() and np.abs(pitch[sel, j]).max() == 0.0, (j, td, lo)
                n["flat"] += 1
            elif td is not None and td >= pl.phase[0]:
                n["before" if lo <= p_in + 1e-9 else "after"] += 1
                if toes:  # the walk's own law: the foot rolls before this lift-off
                    assert rolling[sel, j].any() and pitch[sel, j].max() > 0.02, (j, td, lo)
    return n


def test_l10_flat_feet_per_stance_as_settled(ctx):
    """Rule L10 as the lead settled it on 16 Nov 2026 (STAGE2_TRACKB_PLAN.md L10; rulings 3 and 6): a stance is flat
    when it ends after the move's instant and no later than the end of the move's last ramp.  A causal move (the quick
    turn): the frames up to the command carry the walk's toe-off, the stance that ends at the command is not flat,
    the swing that lifts there carries its pitch to zero and has no toes-up reach; a stance that ends after the last
    ramp has the walk's toe-off again, the one that began before that ramp's end included.  The stop: flat from the
    command to the stand."""
    src, a, w, geo = ctx["src"], ctx["v04"], ctx["wv"], ctx["geo"]
    assert ctx["sv"].toe_off > 0 and ctx["sv"].toe_reach > 0
    bd = ml.turn(src, "turn90_pi4_left", a, w, math.pi / 2, math.pi / 4, 0.8)
    pl = moves.plan(bd.move, geo)
    foot, ph, k = pl.extras["foot"], pl.phase, pl.out.start
    pitch, rolling = foot["pitch"], foot["rolling"]
    c, p_out, last = bd.bind.command, bd.bind.p_out, bd.bind.last
    assert bd.bind.instant == bd.bind.p_in == c == 0.0 and c < p_out < last - 1
    rolls = json.loads((ROOT / "stage2" / "look" / "look_refs.json").read_text())["bodies"][BODY]["v040"]["V6"]["rolls"]
    rec = next(r for r in rolls["left"] if r["lift_off_phase"] == bd.ref_a.a_phase)
    assert pitch[k, 0] == pytest.approx(rec["amplitude"], abs=1e-7) and rec["amplitude"] > 0.03  # the walk's, at the command
    assert rolling[:k, 0].any() and rolling[:k, 1].any()  # both feet roll in the walk before the command
    n = _check_l10(pl, foot, c, p_out)
    assert n["flat"] >= 4 and n["before"] >= 1 and n["after"] >= 2, n
    sw = pl.swing[:, 0] & (ph > c + 1e-9) & (ph < c + 1.0 - a.legacy().duty)  # the swing that lifts at the command
    assert np.all(np.diff(pitch[sw, 0]) <= 0) and pitch[sw, 0][-1] == 0.0 and pitch[sw, 0][0] > 0.03
    assert np.abs(pitch[sw & (pl.swing_u[:, 0] >= 0.5), 0]).max() == 0.0  # no toes-up reach in that swing
    # the stance that ends first after the last ramp began before that ramp ended, and it rolls (the settled wording;
    # as first committed it was flat)
    late = sorted((lo, td, j) for j in (0, 1) for td, lo in _stances(pl, j) if lo > p_out + 1e-9)[0]
    assert late[1] < p_out < late[0] and rolling[_stance_frames(pl, late[2], late[1], late[0]), late[2]].any()
    # a swing of the changing gait has no toes-up reach; a swing that leaves a stance of the walk's law has it
    for j in (0, 1):
        for td, lo in _stances(pl, j):
            air = pl.swing[:, j] & (ph >= lo - 1e-9) & (ph < lo + 1.0 - a.legacy().duty) & (pl.swing_u[:, j] > 0.5)
            if not air.any() or lo < ph[0]:
                continue
            reach = pitch[air, j].min() < -0.01
            assert reach == (lo > p_out + 1e-9 or lo < c - 1e-9), (j, lo, reach)
    end = (ph >= last - 1) & (ph < last)
    assert rolling[end, 0].any() and rolling[end, 1].any() and pitch[end].min() < -0.01  # the walk's law at the end
    assert not rolling[np.sum(pl.swing, axis=1) == 1].any()
    # the stop: the stance that ends at the command rolls up to it; flat from there to the stand
    bs = ml.stop_two_step(src, "stop2_v0.4", a, w)
    ps = moves.plan(bs.move, geo)
    fs, ks = ps.extras["foot"], ps.out.start
    assert bs.bind.instant == -0.5 and bs.bind.p_out == math.inf
    assert fs["pitch"][ks, 1] > 0.03 and fs["rolling"][:ks].any() and not fs["rolling"][ks + 1 :].any()
    assert np.abs(fs["pitch"][ks + 1 :][~ps.swing[ks + 1 :]]).max() == 0.0
    assert _check_l10(ps, fs, -0.5, math.inf)["after"] == 0
    # V6noF5: no pitch anywhere
    nof5 = moves.GaitX.of("v0.4", sets.strip_f5(ctx["sv"]), ctx["H"])
    pn = moves.plan(ml.turn(src, "t", nof5, _fake_window("v0.4", nof5.spec, ctx["H"]), math.pi / 2, math.pi / 4, 0.8).move, geo)
    assert pn.extras["foot"]["flat"] and not pn.extras["foot"]["rolling"].any()


def test_l10_before_the_instant_of_a_change_the_feet_follow_the_walk_it_leaves(ctx):
    """A change (``nominal_to_v0.4``, one stride of ramp from phase 0): its instant is the start of its ramp.  Every
    stance that ends at or before the instant has the lift-off ``foot_pitch`` and ``rolling`` frames that gait a's
    spec gives on the move's own step times (the roll's start by ``gait.rolling_entry`` of gait a, its amplitude
    ``gait.roll_amplitude``, the pitch ``gait.toe_off_bump``); the stance that ends at the instant itself is one of
    them (not flat); the stances that end in (0, 1] are flat, the left one that ends at the ramp's end included; the
    right stance that ends at 1.5 follows gait b's law; the move's last frames carry gait b's foot pitch."""
    src, geo = ctx["src"], ctx["geo"]
    a, b = ctx["nominal"], ctx["v04"]
    bd = ml.change(src, "nominal_to_v0.4", a, b, ctx["wn"], ctx["wv"])
    assert bd.bind.instant == bd.bind.p_in == 0.0 and bd.bind.p_out == 1.0 and bd.move.schedule.ramps == dict(all=[0.0, 1.0])
    pl = moves.plan(bd.move, geo)
    foot, ph, ctxp = pl.extras["foot"], pl.phase, pl.ctx
    pitch, rolling = foot["pitch"], foot["rolling"]
    n = _check_l10(pl, foot, 0.0, 1.0)
    assert n["flat"] == 2 and n["before"] >= 4 and n["after"] >= 4, n
    sa, sb = a.spec, b.spec
    factor = ml.period_factor_of(bd.bind)
    for j, side in enumerate(transitions.SIDES):
        for td, lo in _stances(pl, j):
            if td is None or td < ph[0] or lo > 0.0 + 1e-9:
                continue
            # gait a's law on the move's own step times
            d = float(ctxp.fn["duty"](lo))
            ent = bd.bind.entry["a"][side]
            f_r = max(lo - (1.0 - sa.toe_off_start) * d, lo - (1.0 - ent["a"]) * (d - 0.5))
            nxt = next(s for s in pl.swings if s[0] == side and abs(s[1] - lo) < 1e-9)
            t_r, t_lo, t_e = (ml._time_of(ctxp, f, factor) for f in (f_r, lo, lo + 0.5 * (nxt[2] - lo)))
            want = gait.toe_off_bump(pl.t, t_r, t_lo, t_e, gait.roll_amplitude(sa, t_r, t_lo))[0]
            sel = _stance_frames(pl, j, td, lo)
            assert np.allclose(pitch[sel, j], want[sel], atol=1e-12) and want[sel].max() > 0.02, (side, lo)
            assert np.array_equal(rolling[sel, j], want[sel] > 1e-9), (side, lo)
    k0 = int(np.argmin(np.abs(ph - 0.0)))
    assert pitch[k0, 0] > 0.02  # the left foot's toe-off at the instant itself: the stance that ends there is not flat
    right_15 = _stance_frames(pl, 1, next(td for td, lo in _stances(pl, 1) if abs(lo - 1.5) < 1e-9), 1.5)
    assert rolling[right_15, 1].any()  # it began inside the ramp and ends after it: gait b's law
    left_1 = _stance_frames(pl, 0, next(td for td, lo in _stances(pl, 0) if abs(lo - 1.0) < 1e-9), 1.0)
    assert not rolling[left_1, 0].any() and np.abs(pitch[left_1, 0]).max() == 0.0  # ends at the ramp's end: flat
    # the last output stride carries gait b's own roll amplitude at each lift-off (the walk's first frame follows it)
    o = pl.out
    kl = o.stop - 1
    rb = json.loads((ROOT / "stage2" / "look" / "look_refs.json").read_text())["bodies"][BODY]["v040"]["V6"]["rolls"]
    assert pitch[kl, 0] > 0.02 and rolling[kl, 0]  # the left foot rolls toward its lift-off at the window's end
    assert abs(pitch[kl + 1, 0] - next(r for r in rb["left"] if r["lift_off_phase"] == 0.0)["amplitude"]) < 2e-3
    assert sb.toe_off > 0 and bd.look()["instant_phase"] == 0.0 and bd.look()["flat_until_phase"] == 1.0


def test_l10_a_start_is_flat_until_its_ramps_are_over_and_holds_its_instant(ctx):
    """A start has no frame before its instant but the stand: every stance that ends no later than the end of its
    last ramp (phase 1) is flat, the stand's included; the right stance that ends at 1.5 rolls.  Its instant, the
    start of its first ramp (Stage 1's weight shift, ``-transitions.SHIFT_STRIDES``), is what the entry's
    ``move.command_phase`` holds, and a change's is 0 (``entry_extra``)."""
    src, geo = ctx["src"], ctx["geo"]
    bd = ml.start(src, "start", ctx["nominal"], ctx["wn"])
    assert bd.bind.instant == -transitions.SHIFT_STRIDES == min(v[0] for v in bd.move.schedule.ramps.values())
    assert bd.bind.p_in == -math.inf and bd.bind.p_out == 1.0
    st = moves.plan(bd.move, geo)
    foot = st.extras["foot"]
    assert not foot["rolling"][st.phase <= 1.0].any() and np.abs(foot["pitch"][st.phase <= 1.0][~st.swing[st.phase <= 1.0]]).max() == 0.0
    n = _check_l10(st, foot, -math.inf, 1.0)
    assert n["flat"] == 3 and n["before"] == 0 and n["after"] >= 2, n
    r15 = _stance_frames(st, 1, next(td for td, lo in _stances(st, 1) if abs(lo - 1.5) < 1e-9), 1.5)
    assert foot["rolling"][r15, 1].any()
    # entry_extra: the instant for a move of Stage 1's construction; a causal move's command is left as it is
    g = dict(plan=st)
    e = ml.entry_extra(bd, g, dict(move=dict(command_phase=None, causal=False, command_t_s=None), gates=dict(ik_ok=True),
                                   pass_=True))
    assert e["move"]["command_phase"] == -0.25 and e["move"]["causal"] is False and e["move"]["command_t_s"] is None
    assert e["look"]["instant_phase"] == -0.25 and e["look"]["a_windows"] is None and e["join_events"] is None
    assert e["rolling_frames"] > 0 and e["rolling_split_failed_frames"] == 0 and e["gates"]["rolling_split"] is True
    foot["split_failed_frames"] = [3, 4]
    e2 = ml.entry_extra(bd, g, dict(move=dict(command_phase=None), gates=dict(ik_ok=True), pass_=True))
    assert e2["gates"]["rolling_split"] is False and e2["pass_"] is False and e2["rolling_split_failed_frames"] == 2
    del foot["split_failed_frames"]
    bc = ml.change(src, "nominal_to_v0.4", ctx["nominal"], ctx["v04"], ctx["wn"], ctx["wv"])
    pc = moves.plan(bc.move, geo)
    ec = ml.entry_extra(bc, dict(plan=pc), dict(move=dict(command_phase=None), gates={}, pass_=True))
    assert ec["move"]["command_phase"] == 0.0
    bt = ml.turn(src, "turn90_pi4_right", ctx["v04"], ctx["wv"], math.pi / 2, -math.pi / 4, 0.8)
    pt = moves.plan(bt.move, geo)
    et = ml.entry_extra(bt, dict(plan=pt, causal=False), dict(move=dict(command_phase=0.5), gates={}, pass_=True))
    assert et["move"]["command_phase"] == 0.5 and et["look"]["a_phase"] == 2.5
    assert et["look"]["a_windows"] == pytest.approx(float(ctx["wv"].time(2.5)) / ctx["wv"].W)


def test_l13_the_turn_in_place_has_flat_feet_in_every_stride(ctx):
    """The turn in place (STAGE2_TRACKB_PLAN.md L13 as settled on 16 Nov 2026): flat feet as its own law, in every
    stride, its start and its stop included: ``foot_pitch`` 0 and ``rolling`` false in every computed frame of the
    long plan (the start and the hold strides) and of the stop; its entries name L13 and not L10."""
    src, geo, H = ctx["src"], ctx["geo"], ctx["H"]
    for sign in (1.0, -1.0):
        b = moves.GaitX.of("inplace", sets.inplace_spec("V6", BODY, H), H, yaw_rate=sign * 0.5, in_place=True)
        assert b.spec.toe_off == 0 and b.spec.toe_reach == 0 and b.spec.strides == 1 and gait.look_active(b.spec)
        bd = ml.hold_move(src, "start_inplace", b, sets.INPLACE_HOLD)
        bs = ml.bind_move(src, moves.stop_one_step("stop_inplace", b), None, None)
        for bound in (bd, bs):
            pl = moves.plan(bound.move, geo)
            foot = pl.extras["foot"]
            assert foot["flat"] and not foot["rolling"].any() and np.abs(foot["pitch"]).max() == 0.0
            assert np.all(foot["var_env"] == 0.0) and bound.move.laws.period_factor is None  # no F14 (cut 4)
            assert "S2L-L13" in bound.move.rules and "S2L-L10" not in bound.move.rules
            assert bound.look()["rules"] == list(ml.INPLACE_RULE_IDS) and "S2L-L13" in bound.look()["feet"]
    # every other move of a look set names L10
    for bound in (ml.start(src, "start", ctx["nominal"], ctx["wn"]), ml.stop_one_step(src, "s", ctx["v04"], ctx["wv"])):
        assert "S2L-L10" in bound.move.rules and "S2L-L13" not in bound.move.rules


def test_nothing_before_the_command_knows_with_the_look(ctx):
    """From one look walk, every plan array, the foot pitch and the trunk roll before the command frame are
    ``array_equal`` between the stop and a turn commanded at the same phase; the continuing walk's times, ZMP target
    and pose laws are the same there too (its ZMP points stand on the window's own footprints)."""
    src, a, w, geo = ctx["src"], ctx["v04"], ctx["wv"], ctx["geo"]
    stop = ml.stop_two_step(src, "stop2", a, w)
    turn = ml.bind_move(src, moves._turn("turn", a, math.pi / 2, -math.pi / 4, 0.8, p0=-0.5), w, w)
    assert stop.ref_a.offset == turn.ref_a.offset == 3.0 and stop.ref_a.a_phase == turn.ref_a.a_phase == 2.5
    ps, pt = moves.plan(stop.move, geo), moves.plan(turn.move, geo)
    pw = moves.plan(moves.continuing_walk(stop.move), geo)
    k = ps.out.start
    assert k == pt.out.start == pw.out.start == N and ps.phase[k] == -0.5 and ps.t[k] == pt.t[k] == 0.0
    for key in PLAN_ARRAYS:
        assert _same(getattr(ps, key)[:k], getattr(pt, key)[:k]), key
    for key in ("pitch", "rolling", "contact_offset"):
        assert _same(ps.extras["foot"][key][: k + 1], pt.extras["foot"][key][: k + 1]), key
    assert _same(ps.extras["trunk_roll"][:k], pt.extras["trunk_roll"][:k])
    for key in ("t", "root_z", "neck", "tail", "swing"):
        assert _same(getattr(ps, key)[:k], getattr(pw, key)[:k]), key
    assert np.abs(ps.zmp[:k] - pw.zmp[:k]).max() < 1e-9
    assert np.abs(pw.zmp[k + 300 : k + 600] - ps.zmp[k + 300 : k + 600]).max() > 1e-3  # after it they part
    # the continuing walk keeps the window's variation; the commanded plans lose it over one stride
    assert np.all(pw.extras["foot"]["var_env"][:, 0] == 1.0) and ps.extras["foot"]["var_env"][-1, 0] == 0.0
    with pytest.raises(ValueError, match="lift-off"):
        moves.plan(replace(stop.move, schedule=replace(stop.move.schedule, command=0.2, window=(0.2, 3))), geo)


# ------------------------------------------------------------------------------------------------
# 3. Windows of K strides
# ------------------------------------------------------------------------------------------------
def test_window_reads_at_any_phase(ctx):
    """A window's time is the warp's; its LIPM path is read spectrally (value, rate and acceleration of a band-
    limited path at an off-frame phase); a view shifts the window into the move's plan."""
    spec, H = ctx["sv"], ctx["H"]
    w0 = ctx["wv"]
    K, T, W = w0.K, w0.T, w0.W
    t = np.asarray(w0.arrays["t"], float)
    f = 2 * np.pi * 3 / W
    arrays = dict(w0.arrays, com_ref=np.column_stack([w0.advance * t / W + 0.01 * np.sin(f * t), 0.02 * np.cos(2 * f * t)]),
                  zmp_ref=np.zeros((len(t), 2)), com=np.column_stack([w0.advance * t / W, 0 * t, 0.5 + 0 * t]),
                  phase=w0.warp.phase(t))
    w = ml.Window("w", arrays, spec, H)
    assert w.K == K == spec.strides and w.frames == N * K and float(w.time(K)) == pytest.approx(W)
    assert float(w.time(1.0)) != pytest.approx(T, abs=1e-4)  # a lift-off inside the window is off the uniform schedule
    phi = 3.3
    tt = float(w.time(phi))
    x = w.com_state(phi)
    assert x[0] == pytest.approx([w.advance * tt / W + 0.01 * math.sin(f * tt), 0.02 * math.cos(2 * f * tt)], abs=1e-12)
    assert x[1] == pytest.approx([w.advance / W + 0.01 * f * math.cos(f * tt), -0.04 * f * math.sin(2 * f * tt)], abs=1e-10)
    assert x[2] == pytest.approx([-0.01 * f * f * math.sin(f * tt), -0.08 * f * f * math.cos(2 * f * tt)], abs=1e-9)
    assert np.abs(w.at("com", w.arrays["phase"][5:50]) - arrays["com"][5:50]).max() < 1e-12
    assert w.at("com", [phi + K])[0, 0] == pytest.approx(w.at("com", [phi])[0, 0] + w.advance)  # the next window
    omega = 4.0
    d = w.correction(omega)  # com_ref - com_ref'' / omega^2 - zmp_ref
    want = arrays["com_ref"] - np.column_stack([-0.01 * f * f * np.sin(f * t), -0.08 * f * f * np.cos(2 * f * t)]) / omega**2
    assert np.abs(d - want).max() < 1e-9
    v = ml.View(w, 3.0, -0.5, command=-0.5, a_phase=2.5)
    assert v.n == N and v.a_time_s == pytest.approx(float(w.time(2.5))) and v.phi(-0.5) == 2.5
    assert v.frames("com", [-0.5])[0, 0] == pytest.approx(w.at("com", [2.5])[0, 0] - 2.5 * w.stride)
    c = ml.walk_correction(v, omega)
    kk = np.round((-1.5 + np.arange(400) / N) * N).astype(int) % N
    assert np.abs(c[kk] - w.correction_at(omega, 1.5 + np.arange(400) / N)).max() == 0.0  # by position, not by stride
    with pytest.raises(ml.LookError, match="computed frames"):
        c[np.arange(N)]
    with pytest.raises(ml.LookError, match="whole number"):
        ml.View(w, 2.5, 0.0)
    with pytest.raises(ml.LookError, match="strides"):
        ml.Window("w", dict(arrays, window_strides=np.array(K + 1)), spec, H)


def test_k_stride_walk_in_stage1_format(built):
    """The walk file: Stage 1's reference keys first, in ``refs.arrays``' order, then ``window_strides`` and every
    look key; the frames are the look-format file's; the feedforward is finite, with F5's rolling frames split."""
    r = built["walk"]
    assert r["keys"][: len(ml.STAGE1_KEYS)] == list(ml.STAGE1_KEYS) == r["stage1_keys"]
    assert r["keys"][len(ml.STAGE1_KEYS)] == "window_strides" and "phase" in r["keys"] and "toe_edge" in r["keys"]
    assert set(r["look_keys"]) <= set(r["keys"])
    assert r["frames"] == 2 * N and r["strides"] == 2 and r["q_equal"] and r["tau_finite"] and r["ik_ok"]
    assert r["rolling_frames"] > 0 and r["split_failed"] == 0
    ref = refs.Reference.load(r["file"])
    assert ref.n == 2 * N and int(ref.window_strides) == 2 and ref.period == pytest.approx(r["window_s"])
    assert ref.period / int(ref.window_strides) == pytest.approx(r["period"])
    assert float(ref.stride_motion[3]) == pytest.approx(2 * _spec_from(r["spec"]).describe(look.source(BODY)["H"])["stride"])
    assert s2io.sha256_file(r["file"]) == r["sha256"]


@pytest.mark.skipif("window_strides" not in inspect.getsource(control.Track.add_periodic),
                    reason="control.Track.add_periodic does not read window_strides here (work package G's edit)")
def test_k_stride_walk_on_a_track(built):
    """A K-stride walk file on a ``control.Track``: a plan item counts windows; the feedback's stride period is the
    stride's."""
    from designlab.stage1 import plant

    r = built["walk"]
    ref = refs.Reference.load(r["file"])
    p = plant.build(BODY, "hard", settle=False)
    track = control.Track(p, control.omega0(p, ref.q_home)).add_periodic(ref, 1.5)
    seg = track.segments[0]
    assert seg.stride_s == pytest.approx(r["period"]) and seg.period == pytest.approx(r["window_s"])
    assert seg.duration == pytest.approx(1.5 * r["window_s"]) and seg.n == 2 * N
    assert set(np.round(seg.info["period"], 12)) == {round(r["period"], 12)}


def test_a_causal_move_leaves_a_window_as_the_walk(built):
    """J1 on a solved window (two strides of the ladder's V6 at 0.40 m/s): the stop's warm-up and command frame are
    the window's (model CoM, joints, and every pose law); the frames after the command have flat stance feet."""
    r = built["stop"]
    jc, wm = r["join_command"], r["warm"]
    assert jc["warmup_com_max_mm"] <= transitions.G3_COM_MM and jc["at_command"]["com_mm"] <= transitions.G3_COM_MM
    assert jc["at_command"]["joint_rad"] <= moves.JOINT_AT_COMMAND_RAD and jc["warmup_ik_ok"]
    assert wm["com_mm"] < 0.2 and wm["joint_rad"] < 1e-3, wm
    assert wm["tail"] < 1e-6 and wm["roll"] < 1e-6 and wm["neck"] < 5e-6 and wm["root_z"] < 1e-5, wm
    assert wm["pitch_peak"] > 0.03 and wm["pitch"] < 0.02 * wm["pitch_peak"], wm  # the window's file, splined
    assert r["touchdowns"] == 2 and r["pitch_after"] == 0.0 and not r["rolling_after"]
    assert r["look"]["a_phase"] == 2.5 and r["a_time_s"] == pytest.approx(r["look"]["a_time_s"])
    assert set(ml.LOOK_RULE_IDS) <= set(r["rules"]) and "S2A-causal" in r["rules"]
    assert r["keys"][-7:] == ["foot_pitch", "rolling", "rolling_contact", "trunk_roll", "bob", "var_env", "period_factor"]
    ev = r["join_events"]  # the ZMP reading apart from the plan's events: a reading, never a gate
    assert ev["frames_over_limit"] >= ev["frames_over_limit_away_from_events"] and ev["zmp_max_mm"] == pytest.approx(
        jc["warmup_zmp_max_mm"])
    assert ev["limit_mm"] == jc["limits"]["zmp_mm"] == moves.WARMUP_ZMP_LIMIT_MM


def test_join_events_reads_join_commands_limit_and_no_event_across_the_warmups_ends(monkeypatch):
    """``join_events`` (a reading beside ``join_command``, never a gate) on a warm-up made by hand: it counts the
    frames over ``join_command``'s own warm-up limit (``moves.WARMUP_ZMP_LIMIT_MM``, A7 as amended), on the gate's
    frames (from the first with an acceleration to two before the command), and a frame is "at an event" only for a
    planned lift-off or touchdown, or a frame of the warm-up itself on which a stance foot rolls, within
    ``EVENT_FRAMES`` of it: a foot that rolls on the warm-up's last frame marks none of its first frames (no wrap)."""
    n = 60
    ph = -1.0 + np.arange(n) / N
    mm = np.full(n, 0.5)
    mm[:2] = np.nan  # (as a move's own: no acceleration on the first two computed frames)
    mm[2], mm[3] = 3.0, 5.0  # over transitions.G3_ZMP_MM (2.0 mm) and under the gate's 4.0 mm; over both
    zmp = np.zeros((n, 2))
    zmp[:, 0] = 1e-3 * mm
    rolling = np.zeros((n + 40, 2), bool)
    rolling[n - 1, 0] = True  # a stance foot rolls on the warm-up's last frame only
    pl = SimpleNamespace(first=-1.0, swings=[(0, -9.0, -8.6)], extras=dict(foot=dict(rolling=rolling)))
    g = dict(causal=True, warmup=dict(phase=ph, zmp_mb=zmp), plan=pl)
    monkeypatch.setattr(ml, "_is_view", lambda ref: True)
    monkeypatch.setattr(ml, "walk_frames", lambda ref, phases, first, key: np.zeros((len(phases), 2)))
    ev = ml.join_events(g, object())
    assert ev["limit_mm"] == moves.WARMUP_ZMP_LIMIT_MM == 4.0
    assert ev["frames"] == n - 4 and ev["zmp_max_mm"] == pytest.approx(5.0)
    assert ev["frames_over_limit"] == 1 and ev["phases_over_limit"] == [pytest.approx(ph[3])]
    # the rolling frame marks frames n-4 to n-1, of which n-4 and n-3 are the gate's; frames 0 to 2 are not near it
    assert ev["event_frames"] == 2 and ev["frames_over_limit_away_from_events"] == 1
    assert ev["zmp_max_mm_away_from_events"] == pytest.approx(5.0)


def test_the_windows_own_acceleration_is_the_moves_filter_history(built):
    """``moves_look.walk_forward_acceleration`` (the look solver's reader that ``moves`` takes for the history of a
    move's 1 Hz acceleration reading) is the window's own CoM acceleration, and the stop's reading says so; the
    stop's entry keeps its command phase."""
    r = built["stop"]
    assert r["accel_err"] < 0.05 * r["accel_peak"] and r["accel_peak"] > 0.1, r
    assert r["accel_before"] == "the periodic stride of w2"
    # its file begins at the command, the lift-off frame of the stance that rolled: no stance rolls after it, so
    # the entry has no rolling-split gate
    assert r["command_phase"] == -0.5 and r["rolling_frames"] == 0 and r["rolling_split_failed"] == 0
    assert "rolling_split" not in r["gates2"] and r["pitch_after"] == 0.0 and r["rolling_after"] is False


def test_nothing_before_the_command_knows_on_a_solved_window(built):
    """The start state, the first references, the continuing walk and the CoM path up to the command frame are
    ``array_equal`` between the stop and a turn commanded at the same phase of the same window; the warm-up CoM path
    is the window's own LIPM path."""
    r = built["causal"]
    assert r["k"] == r["k_turn"] == N and r["x0"] and r["zref0"] and r["zpre"] and r["zpre_t"] and r["com_before"], r
    assert all(r["plan"].values()) and r["pitch"] and r["roll"], r
    assert r["differ_after"] > 1e-3 and r["walk_com_mm"] < 0.1, r


def test_null_command_reproduces_the_window(built):
    """A look walk commanded, causally, to walk on across its window's boundary (window a faded out as window b
    fades in: the whole variation throughout): the plan's times are the window's, and every output frame is within
    the G3 limits of the window's CoM; one solve.  (The stance on the ground at the command is flat by L10, so the
    joints differ there by the toe-off's pitch.)"""
    r = built["null"]
    assert r["frames"] == 2 * N and r["ik_ok"] and r["t_err"] < 1e-7
    assert r["env_sum"][0] == pytest.approx(1.0, abs=1e-12) and r["env_sum"][1] == pytest.approx(1.0, abs=1e-12)
    assert r["com_mm"] <= transitions.G3_COM_MM and r["join_b"]["com_ok"], r
    assert r["zmp_p90_mm"] <= transitions.G3_ZMP_MM, r
    # no ramp: every stance follows the walk's own law (L10), so the move's foot pitch is the window's, toe-off and
    # toes-up reach, up to its last frame (the walk's first frame follows it, foot pitch included)
    assert r["pitch_peak"] > 0.03 and r["pitch_low"] < -0.01
    assert r["pitch"] < 1e-3 and r["pitch_last"] < 1e-4 and r["joint_rad"] < 1e-3, r
    assert r["rolling_same"] > 0.995 and r["rolling_frames"] > 20, r  # (stance frames pitched over 1e-4 rad, both)


def test_periodicity_gate_and_cut(ctx):
    """A periodic reference cut from a long plan: the gate maps the strides before and after onto the stride by the
    plan's own stride motion; the cut is in a reference's own frame (home origin, heading 0 at its first frame)."""
    src, geo, H = ctx["src"], ctx["geo"], ctx["H"]
    spec = sets.strip_f5(_one_stride(ctx["sn"]))
    assert spec == sets.inplace_spec("V6", BODY, H) == sets.strip_f14(sets.strip_f5(ctx["sn"]))
    b = moves.GaitX.of("inplace_0.5_left", spec, H, yaw_rate=0.5, in_place=True)
    bd = ml.hold_move(src, "start_inplace_0.5_left", b, 4)
    assert bd.move.schedule.window == (-2, 7) and bd.move.laws.name == "look" and bd.move.args == dict(hold_strides=4)
    pl = moves.plan(bd.move, geo)
    assert pl.extras["foot"]["flat"] and np.all(pl.extras["foot"]["var_env"] == 0.0)
    # a stand-in for the generated plan: exactly periodic frames carried by the plan's own stride motion
    o, n_out = pl.out, pl.n_out
    T = b.legacy().period
    th = 0.5 * T
    ph = pl.phase[o]
    k = np.arange(n_out)
    q = np.zeros((n_out, 7 + 14))
    q[:, 7:] = 0.1 * np.sin(2 * np.pi * ph)[:, None]
    P = np.array([pl.pose_fn(p)[0] for p in ph])
    hd = pl.heading[o]
    com = np.column_stack([P[:, 0] + 0.01 * np.cos(hd), P[:, 1] + 0.01 * np.sin(hd), np.full(n_out, 0.5)])
    g = dict(plan=pl, geo=geo, q=q, com=com, zmp_mb=com[:, :2].copy())
    per = ml.periodicity(g, 5.0)
    assert per["ok"] and per["com_mm"] < 1e-9 and per["zmp_mm"] < 1e-9 and per["joint_rad"] < 1e-12
    assert pl.pose_fn(6.0)[1] - pl.pose_fn(5.0)[1] == pytest.approx(th)
    g2 = dict(g, com=com + np.outer(k / n_out, [0.004, 0.0, 0.0]))
    bad = ml.periodicity(g2, 5.0)
    assert not bad["ok"] and bad["com_mm"] > ml.PERIODIC_COM_MM and bad["limits"]["com_mm"] == 1.0
    with pytest.raises(ml.LookError, match="no output stride"):
        ml.periodicity(g, 6.0)


# ------------------------------------------------------------------------------------------------
# 4. The look sets' definitions
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("body,H", [("75-M-H-10", 0.5018535496646683), ("60-N-T-10", 0.40468919988052376)])
def test_set_walks_at_the_frozen_rungs_are_the_ladders(body, H):
    """V6's nominal and 0.40 m/s walks are the look ladder's V6 rungs, spec for spec (their look-format files are
    then held to the ladder's hashes where the set is built: ``ladder_twins``); the other speeds follow rule A2's walk
    through the same chain; V6noF5 differs from V6 in F5's fields only; a head body walks its pick's look."""
    recs = json.loads((ROOT / "stage2" / "look" / "look_refs.json").read_text())["bodies"][body]
    tps = json.loads((ROOT / "stage2" / "look" / "tail_phase.json").read_text())["bodies"][body]
    frz = json.loads((ROOT / "stage2" / "look" / "L1_FREEZE.json").read_text())["rungs"][body]
    assert sets.body_facts(body)[0] == H
    for name, key, v in (("nominal", "nominal", None), ("v0.4", "v040", 0.40)):
        tp = tps[key]["tail_phase_F9_rad"]
        assert sets.tail_phase(body, v) == sets.tail_phase(body + "-X", v) == tp and sets.KNOTS[name] == key
        s = sets.v6_spec(body, H, v)
        assert look._plain(s.__dict__) == recs[key]["V6"]["spec"] and s == look.ladder(body, H, key, tp)["V6"]
        assert s == sets.LOOK_SETS["V6"].spec_of(body, H, v) == sets.v6_spec(body + "-X", H, v)
        twin = sets.ladder_twin(body, s)
        assert twin["sha256"] == frz[key]["sha256"] == recs[key]["V6"]["sha256"] and twin["variant"] == "V6"
        assert sets.twin_owed(sets.LOOK_SETS["V6"], body, name, walk=True) == dict(stage1=False, ladder=True)
        assert sets.twin_owed(sets.LOOK_SETS["V6noF5"], body, name, walk=True) == dict(stage1=False, ladder=False)
        n5 = sets.LOOK_SETS["V6noF5"].spec_of(body, H, v)
        diff = {k for k in s.__dict__ if getattr(s, k) != getattr(n5, k)}
        assert diff == {"toe_off", "toe_reach", "toe_off_rate", "toe_off_accel"} and n5.toe_off == 0 and n5.lift_peak == 0.375
        assert sets.ladder_twin(body, n5) is None
    for v in setio.SPEEDS:
        v0 = sets.v0_spec(body, H, v)
        s = sets.look_chain(body, H, v0, -1.7)
        d = s.describe(H)
        assert d["speed"] == pytest.approx(v) and gait.look_active(s) and s.zmp_iters == look.LOOK_ZMP_ITERS
        assert s.strides == look.window_strides(s, H) and s.strides * d["period"] >= look.CLIP_S
        assert gait.DUTY_RANGE[0] <= d["duty"] <= gait.DUTY_RANGE[1] and s.swing_time == look.swing_time(H)
        assert not gait.look_active(v0) and s.stride_rel == v0.stride_rel and s.lift_frac == v0.lift_frac
        assert sets.LOOK_SETS["V6"].label_of(body, H, v) == ("new" if v > look.envelope.FAST.describe(H)["speed"] else None)
        v4 = sets.look_chain(body, H, v0, None, upto="V4")
        assert v4.tail_phase == v0.tail_phase and v4.strides == 1 and v4.roll_amp == s.roll_amp
    with pytest.raises(sets.SetError, match="need F9's tail phase"):
        sets.look_chain(body, H, sets.v0_spec(body, H, 0.3), None)
    assert sets.base_body(body + "-X") == body == sets.base_body(body)
    with pytest.raises(sets.SetError, match="no F9 tail phase"):
        sets.tail_phase("75-M-H-10-J", None)  # (a body the look ladder does not hold)


def test_the_look_sets_stand_beside_the_three_sets():
    """``sets.SETS`` is the three sets of round A, as its own tests hold it; the look sets are ``LOOK_SETS``, and
    every function of ``sets`` takes either (``_set``).  A look set's entries are made by ``moves_look.py`` too."""
    assert list(sets.SETS) == ["V0", "F25F9", "F25F9cl"] and list(sets.LOOK_SETS) == ["V6", "V6noF5"]
    for name, st in sets.LOOK_SETS.items():
        assert sets._set(name) is st and st.look and st.name == name and st.margin_rule is None and not st.stage1_twins
        assert sets.modules_of(st) == sets.MODULES + ("moves_look.py",)
        assert set(sets.modules_for(name)) - set(sets.modules_now()) == {"designlab/stage2/moves_look.py"}
    assert sets.modules_for("V0") == sets.modules_now()
    assert not any(st.look for st in sets.SETS.values()) and sets.modules_of(sets.SETS["V0"]) == sets.MODULES
    assert sets.LOOK_SETS["V6"].ladder_twins == ("nominal", "v0.4") and sets.LOOK_SETS["V6noF5"].ladder_twins == ()
    assert sets.LOOK_SETS["V6"].features == ("F1", "F2", "F3", "F4", "F5", "F6", "F9", "F10", "F14")
    assert sets.LOOK_SETS["V6noF5"].features == ("F1", "F2", "F3", "F4", "F6", "F9", "F10", "F14")
    with pytest.raises(sets.SetError, match="not a reference set"):
        sets._set("V7")
    assert ml.LEAVE_PHASE == dict(turn_left=2.0, turn_right=2.5, stop_two_step=2.5)
    assert (sets.INPLACE_RATE, sets.INPLACE_HOLD, sets.INPLACE_CUT) == (0.5, 4, 2)


def _tail_phases_here(body: str) -> bool:
    doc = json.loads(sets.SET_TAIL.read_text()) if sets.SET_TAIL.is_file() else {}
    return all(n in (doc.get("bodies") or {}).get(body, {}) for n in ("v0.3", "v0.35", "v0.45", "v0.5"))


@pytest.mark.parametrize("set_name", ["V6", "V6noF5"])
@pytest.mark.parametrize("body", ["75-M-H-10", "60-N-T-10", "75-M-H-10-X", "60-N-T-10-X"])
def test_set_items_and_names(set_name, body):
    """A look set's items: the 22 of every set, in the same order, with the look's laws in every move's identity,
    then the turn in place's two optional groups of three (the gait, its start, its stop)."""
    if not _tail_phases_here(sets.base_body(body)):
        pytest.skip(f"stage2/sets/V6/tail_phase.json holds no phases of {sets.base_body(body)}")
    its = sets.items(set_name, body)
    H, model = sets.body_facts(body)
    names = [i.name for i in its]
    assert names[:22] == [i.name for i in sets.items("V0", body)] and len(names) == len(set(names)) == 28
    assert names[22:] == ["inplace_0.5_left", "start_inplace_0.5_left", "stop_inplace_0.5_left",
                          "inplace_0.5_right", "start_inplace_0.5_right", "stop_inplace_0.5_right"]
    assert [i.kind for i in its[22:]] == list(sets.LOOK_KINDS) * 2 and all(i.optional for i in its[22:])
    assert not any(i.optional for i in its[:22]) and [i.kind for i in its[:22]] == [i.kind for i in sets.items("V0", body)]
    groups = sets.groups(its)
    assert len(groups) == 20 and list(groups)[18:] == ["inplace_0.5_left", "inplace_0.5_right"]
    specs = sets.walk_specs(set_name, body)
    by = {i.name: i for i in its}
    for i in its:
        json.dumps(i.definition, allow_nan=False)
        assert i.definition["kind"] == i.kind and i.definition["body_model_sha256"] == model
        if i.kind == "walk":
            assert i.definition["generator"] == sets.LOOK_WALK_GENERATOR
            assert i.definition["spec"] == json.loads(json.dumps(look._plain(specs[i.name].__dict__)))
            assert gait.look_active(specs[i.name]) and specs[i.name].strides >= 9
            assert (specs[i.name].toe_off > 0) == (set_name == "V6")
        else:
            assert i.definition["laws"] == "look"
            want = "S2L-L13" if i.kind in sets.LOOK_KINDS else "S2L-L10"
            assert want in i.definition["rules"] and "S2L-L9" not in i.definition["rules"]
            assert ("S2L-L10" in i.definition["rules"]) != (i.kind in sets.LOOK_KINDS)
        if "inplace" not in i.name:
            assert setio.parse_name(i.name)["name"] == i.name
        else:
            assert not i.name.startswith("stop_in_place") and i.name != "stop"  # Stage 1's own stop-event names
            assert sets.inplace_of(i.name)["group"] == i.group and i.depends == ("nominal",)
    # a move's identity is the bound move's (the builder binds it to the windows; nothing of the identity changes)
    assert by["stop2_v0.4"].definition["leave_phase"] == 2.5 == by["turn90_1_right"].definition["leave_phase"]
    assert by["turn90_pi4_left"].definition["leave_phase"] == 2.0 and "leave_phase" not in by["start"].definition
    assert by["start"].definition["lead_in"]["start"]["laws"] == "look"
    g = by["inplace_0.5_right"].definition
    assert g["rate_rad_s"] == -0.5 and g["hold_strides"] == 4 and g["cut_hold_stride"] == 2
    assert g["gait"]["in_place"] is True and g["gait"]["spec"]["toe_off"] == 0 and g["gait"]["spec"]["strides"] == 1
    assert g["periodic"] == dict(com_mm=1.0, zmp_mm=2.0, joint_rad=1e-3)
    assert sets.inplace_names("left") == dict(ref="inplace_0.5_left", start="start_inplace_0.5_left",
                                              stop="stop_inplace_0.5_left")
    assert sets.inplace_of("start_inplace_0.5_right") == dict(role="start", direction="right", group="inplace_0.5_right")
    assert sets.inplace_of("start") is None
    if body.endswith("-X"):  # a head body's walks are its pick's look on its own model
        pick = sets.walk_specs(set_name, sets.base_body(body))
        assert specs == pick and model != sets.body_facts(sets.base_body(body))[1]


def test_a_bound_moves_identity_is_the_items(ctx):
    """``moves_look.definition_of`` (what ``sets.items`` holds, made without a body or a window) is the definition of
    the move once it is bound to the look's laws and its windows."""
    src = ctx["src"]
    n, v, wn, wv = ctx["nominal"], ctx["v04"], ctx["wn"], ctx["wv"]
    cases = [(moves.start("start", n), None, wn), (moves.change("nominal_to_v0.4", n, v), wn, wv),
             (moves.change("nominal_to_v0.4", n, v, ramp_strides=2.0), wn, wv),
             (moves.stop_two_step("stop2_v0.4", v), wv, None), (moves.stop_one_step("stop1_v0.4", v), wv, None),
             (moves.turn("turn90_pi4_left", v, math.pi / 2, math.pi / 4, 0.8), wv, wv),
             (moves.turn("turn90_0.5_right", v, math.pi / 2, -0.5, 1.0), wv, wv)]
    for mv, wa, wb in cases:
        bd = ml.bind_move(src, mv, wa, wb)
        d = ml.definition_of(mv)
        lp = d.pop("leave_phase", None)
        assert d == bd.move.definition() and d["laws"] == "look", mv.name
        assert lp == bd.look()["a_phase"] and (lp is not None) == mv.causal, mv.name


# ------------------------------------------------------------------------------------------------
# 5. The set builder's side of a look set (``sets``): records, names, optional items
# ------------------------------------------------------------------------------------------------
needs_phases = pytest.mark.skipif(not _tail_phases_here(BODY), reason="stage2/sets/V6/tail_phase.json holds no phases "
                                                                      f"of {BODY}")


@needs_phases
def test_a_look_sets_records_hold_the_looks_rules_and_no_cut():
    """A new look set's three records: the builder's rules with the look's beside them, the look's laws' rules in
    ``transitions.json``, and ``set.json`` ``look``: join rule J1 as designed, no cut in force, no review named (the
    lead records it before a cut: STAGE2_TRACKB_PLAN.md L12).  A set of today's laws has none of this."""
    header = dict(model=dict(file_sha256="x"), H_m=0.5, q_home=[0.0], joints=["j"])
    for name in sets.LOOK_SETS:
        docs = sets._new_docs(name, BODY, header)
        sd, td, rd = docs["set.json"], docs["transitions.json"], docs["refs.json"]
        assert set(sets.RULES) | set(sets.LOOK_RULES) == set(sd["rules"]) and sd["role"].startswith("look")
        assert sd["look"]["cut"] is None and sd["look"]["join_review"] is None and sd["look"]["pick"] == BODY
        assert sd["look"]["leave_phase"] == dict(turn_left=2.0, turn_right=2.5, stop_two_step=2.5)
        assert set(ml.RULES) <= set(td["rules"]) and "S2L-moves" in td["rules"] and "S2A-gates" in td["rules"]
        assert "S2L-format" in rd["rules"] and "rolling contact" in rd["feedforward"]
        assert sd["freezes"] == [] and sd["blocked"] is None and sd["lead_in"] is None and sd["crouch_frac"] == 0.03
        json.dumps(docs, allow_nan=False)
    v0 = sets._new_docs("V0", BODY, header)
    assert "look" not in v0["set.json"] and not any(k.startswith("S2L-") for d in v0.values() for k in d["rules"])
    committed = ROOT / "stage2" / "sets" / "V0" / BODY / "set.json"
    if committed.is_file():  # the frozen V0 set's rules are the ones a new V0 set would get: nothing of them moved
        assert json.loads(committed.read_text())["rules"] == v0["set.json"]["rules"]
    assert sets._new_docs("V6", BODY + "-X", dict(header))["set.json"]["look"]["pick"] == BODY


@needs_phases
def test_the_turn_in_places_names_and_optional_items_in_verify_and_status(tmp_path):
    """The turn in place's names are not ``setio``'s: ``sets`` checks them by role.  Its three references of a
    direction are optional: a set that was never built lists them as optional and not as missing."""
    its = {i.name: i for i in sets.items("V6", BODY)}
    bad: list = []
    sets._check_name("inplace_0.5_left", "walk", {}, its["inplace_0.5_left"], bad)
    sets._check_name("start_inplace_0.5_left", "move", {}, its["start_inplace_0.5_left"], bad)
    sets._check_name("stop_inplace_0.5_right", "move", {}, its["stop_inplace_0.5_right"], bad)
    assert bad == []
    sets._check_name("inplace_0.5_left", "move", {}, its["inplace_0.5_left"], bad)  # the gait among the moves
    sets._check_name("stop_inplace_0.5_right", "walk", {}, its["stop_inplace_0.5_right"], bad)
    sets._check_name("start_inplace_0.5_left", "move", {}, None, bad)  # no item of it
    assert len(bad) == 3 and all("a name of the turn in place" in x for x in bad)
    st = sets.status("V6", BODY, tmp_path)
    assert st["built"] is False and len(st["missing"]) == 22 and "inplace_0.5_left" not in st["missing"]
    assert st["items"]["stop_inplace_0.5_right"].startswith("not built (optional") and st["items"]["start"] == "missing"
    with pytest.raises(sets.SetError, match="no item or group"):
        sets.build("V6", BODY, tmp_path, only=["inplace_1_left"], procs=0)


def test_the_turn_in_places_gait_is_verified_against_l13():
    """``sets._verify_inplace``: 0.5 rad/s signed, speed 0, one stride, periodic within its gate, not blocked at
    Stage 0, the set's crouch (STAGE2_TRACKB_PLAN.md L13's check)."""
    good = dict(pass_=True, periodicity=dict(ok=True), stage0=dict(blocks_freeze=[]), yaw_rate_rad_s=-0.5, speed_m_s=0.0,
                strides=1, crouch_frac=0.03)
    sd = dict(crouch_frac=0.03)
    bad: list = []
    sets._verify_inplace("inplace_0.5_right", good, sd, bad)
    assert bad == []
    for edit in (dict(periodicity=dict(ok=False)), dict(yaw_rate_rad_s=0.35), dict(speed_m_s=0.1), dict(strides=9),
                 dict(stage0=dict(blocks_freeze=[dict(limit="knee speed", value=1.1)])), dict(crouch_frac=0.05),
                 dict(pass_=False)):
        got: list = []
        sets._verify_inplace("inplace_0.5_right", dict(good, **edit), sd, got)
        assert len(got) == 1, edit


def test_the_stride_pick_on_look_turns_keeps_the_rules_record(ctx, monkeypatch):
    """``moves_look.pick_stride_factor`` is ``moves.pick_stride_factor``'s rule on the bound turns: the same record
    keys, the joint-range reading of every other joint included (A11 and A12 as amended), each kinematic pass given
    the turn bound to the look's laws and the walk's window."""
    seen = []

    def kinematic(src, mv, ref_a):
        seen.append((mv.name, mv.laws.name, mv.args["stride_factor"], type(ref_a).__name__))
        f, left = mv.args["stride_factor"], mv.args["rate"] > 0
        return dict(range_use=dict(left_hip_yaw=0.79 + 0.1 * f if left else 0.5, right_hip_yaw=0.6), ik_ok=True,
                    ik_fail_frames=0,
                    tightest_other=dict(joint="left_knee", end="max", margin_rad=0.04 if f > 0.9 else 0.06))

    monkeypatch.setattr(moves, "kinematic", kinematic)
    monkeypatch.setattr(moves, "steady_turn_kinematic", lambda src, a, rate, f: dict(available=False, reason="look"))
    rec = ml.pick_stride_factor(ctx["src"], ctx["v04"], ctx["wv"], math.pi / 2, math.pi / 4, grid=(1.0, 0.95, 0.9, 0.5))
    # f 1.0: twist 0.89 over 0.85 and the other joint under 0.05; 0.95: twist 0.885 over; 0.9: 0.88 over; 0.5 passes
    assert [r["f"] for r in rec["tried"]] == [1.0, 0.95, 0.9, 0.5] and rec["chosen"] == 0.5 and rec["outcome"] == "picked"
    assert [r["pass_"] for r in rec["tried"]] == [False, False, False, True]
    assert rec["tried"][0]["tightest_other"]["left"] == dict(joint="left_knee", end="max", margin_rad=0.04)
    assert rec["limit_other_joints_rad"] == moves.OTHER_JOINT_MARGIN_RAD and rec["rule"] == "S2A-stride"
    assert all(s[1] == "look" and s[3] == "View" for s in seen) and len(seen) == 8
    want = {"rule", "text", "grid", "limit_kinematic", "limit_final", "limit_other_joints_rad", "steps_down_max",
            "angle_rad", "rate_rad_s", "tried", "chosen", "steady_turn_kinematic", "steps_down", "builds", "outcome",
            "reason"}
    assert set(rec) == want
    none = ml.pick_stride_factor(ctx["src"], ctx["v04"], ctx["wv"], math.pi / 2, math.pi / 4, grid=(1.0,))
    assert none["chosen"] is None and none["outcome"] == "none" and "every other gated joint" in none["reason"]
    # moves.step_down reads the record as it reads its own
    down = moves.step_down(rec, dict(ik_ok=True, hip_yaw_range_use=0.80, tightest_other_rad=0.07))
    assert down["outcome"] == "frozen"
