"""Tests for the Stage 2 look generator: ``gait.GaitSpec``'s look fields (designlab/gait.py) and the look ladder
(designlab/stage2/look.py; STAGE2_PLAN.md section 4 A, research_look.md sections 3.1 and 4.1).

Run from hardware/trex_form_study:  python -m pytest tests/test_stage2_look.py -q

1. G2: with every look field at its default, all of Stage 1's references of both bodies (the walks nominal, brisk, fr05
   and carpet, and the six turns ``turning.generate`` makes) regenerate bit for bit (each npz file's sha256 as
   ``stage1/runs/2026-11-08/<body>/refs.json`` records it), built as ``stage1.refs.prepare`` builds them (round two's
   accuracy) in subprocesses with one BLAS thread, as the Stage 1 runner (with several threads the inverse dynamics'
   double-support QP differs in the 15th digit).  ``turning.generate`` refuses a look spec.
2. Each feature measured on generated walks (60-N-T-10, nominal), from the frames' geometry: bob amplitude, its fall,
   rise and share after the touchdown (F2); duty, swing time and cadence from the detected floor contact; lift peak and
   the clearance screen's swing gate; the toe-off's roll on the toe edge, the transition arc and the claw tip with no
   part of the foot below the floor; trunk roll; tail and head phase; stride; the variation's bounds and seeds; the ZMP
   correction's convergence.  The F5-aware inverse dynamics (the rolling CoP on its contact; every other frame
   ``gait.inverse_dynamics``'s exactly) and the Stage 0 price.  On 75-M-H-10, a coarse animal-pace walk with every
   feature (F13's stride); both bodies' markers and the rigid foot's claw geometry.
3. Every rung of the committed ladder (``stage2/look/look_refs.json``): IK on every frame, the ZMP inside the support,
   the correction's convergence against V0's, the measured features against the spec, the Stage 0 gates (the swing
   clearance of 0.04 H and the walking gates on every rung; drive ratios at most 1 or the rung marked over its limit
   under the plan's rule), the flags for the owner; where the git-ignored arrays are here, each rung's measures
   recomputed from them; one rung regenerated to its recorded sha256 with its price recomputed (every rung with
   STAGE2_SLOW=1); the tail-phase checks (``tail_phase.json``) and F9's phases; the V0 price against the round-two
   envelope screen.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, fields, replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import designlab  # noqa: E402,F401  (selects the GL backend before mujoco is imported)
from designlab import clearance, envelope, gait, sizing, statics, turning  # noqa: E402
from designlab.stage1 import plant  # noqa: E402
from designlab.stage2 import look  # noqa: E402

RUN = ROOT / "stage1" / "runs" / "2026-11-08"
LOOK = ROOT / "stage2" / "look"
BODY = "60-N-T-10"
#: The bodies' hip heights H (m), as their records rebuild them (``test_markers_on_both_bodies`` checks them).
H_BODY = {"60-N-T-10": 0.40468919988052376, "75-M-H-10": 0.5018535496646683}
#: Every Stage 1 reference of a body (``stage1.refs.reference_names``): the walks and the turning screen's turns.
G2_NAMES = ("nominal", "brisk", "fr05", "carpet", "walk_0.35_left", "walk_0.35_right", "in_place_0.50_left",
            "in_place_0.50_right", "walk_0.50_left", "walk_0.50_right")
PY = sys.executable
slow = pytest.mark.skipif(not os.environ.get("STAGE2_SLOW"), reason="every variant regenerated; set STAGE2_SLOW=1")


def _env() -> dict:
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[var] = "1"
    return env


def _record(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


# ------------------------------------------------------------------------------------------------
# 1. G2: today's walk, bit for bit
# ------------------------------------------------------------------------------------------------
def test_look_fields_default_to_todays_walk():
    """The look fields' defaults are today's walk: none of the study's walks is a look walk, describe() is unchanged,
    and the tail law's default phase is today's pi / 2."""
    names = {f.name for f in fields(gait.GaitSpec)}
    assert set(gait.LOOK_FIELDS) <= names and set(gait.LOOK_DEFAULTS) == set(gait.LOOK_FIELDS)
    old = ("froude", "duty", "lift_frac", "head_amp", "tail_amp", "sway_gain", "stride_rel", "crouch", "zmp_iters",
           "zmp_gain", "n")
    assert [f.name for f in fields(gait.GaitSpec)][: len(old)] == list(old)  # the old fields first, unchanged
    for spec in (gait.GaitSpec(), sizing.NOMINAL, sizing.BRISK, envelope.FAST):
        assert not gait.look_active(spec)
        d = spec.describe(0.4)
        assert d["duty"] == spec.duty and d["period"] == spec.stride_rel * 0.4 / np.sqrt(spec.froude * gait.G * 0.4)
    assert gait.GaitSpec().tail_phase == np.pi / 2 and gait.GaitSpec().head_mode == "nod"
    for f in ("bob", "strides", "bob_drop", "toe_off_rate", "toe_off_accel"):
        val = {"bob": 0.01, "strides": 2, "bob_drop": 0.1, "toe_off_rate": 3.0, "toe_off_accel": 200.0}[f]
        assert gait.look_active(replace(sizing.NOMINAL, **{f: val})), f


def test_a_default_spec_read_back_from_json_stays_todays_walk():
    """JSON turns ``var_band``'s tuple into a list; a default spec rebuilt from a record must not become a look walk."""
    for spec in (sizing.NOMINAL, sizing.BRISK, envelope.FAST):
        back = gait.GaitSpec(**json.loads(json.dumps(asdict(spec))))
        assert isinstance(back.var_band, list) and not gait.look_active(back)
        assert not gait.look_active(replace(spec, var_band=np.array([0.05, 0.2])))
        assert gait.look_active(replace(spec, var_band=[0.05, 0.3])) and gait.look_active(replace(spec, var_band=[0.05]))


def test_default_fields_take_todays_code_path(monkeypatch):
    """A walk with every look field at its default never reaches the look generator."""
    def refuse(*a, **k):
        raise AssertionError("the look path ran for today's walk")

    monkeypatch.setattr(gait, "_generate_look", refuse)
    src = _src()
    tr = gait.generate(src["M"], src["q_home"], src["H"], replace(sizing.NOMINAL, n=20, zmp_iters=0))
    assert "look" not in tr and len(tr["q"]) == 20


def test_turning_generate_refuses_a_look_spec():
    """The turn path reads today's duty, nod and tail law: a look spec would silently lose its features, so it is
    refused (the look's turns are deferred: ``look.DEFERRED``); today's specs pass (G2 below shows them unchanged)."""
    src = _src()
    with pytest.raises(ValueError, match="look"):
        turning.generate(src["M"], src["q_home"], src["H"], replace(sizing.NOMINAL, bob=0.01, n=20))
    tr = turning.generate(src["M"], src["q_home"], src["H"], replace(sizing.NOMINAL, n=20, zmp_iters=0), yaw_rate=0.35)
    assert len(tr["q"]) == 20
    assert "turns" in look.DEFERRED and "starts_and_stops" in look.DEFERRED


@pytest.fixture(scope="module")
def g2_hashes():
    """``look.g2_hashes`` of both bodies over every Stage 1 reference, run at once in two subprocesses with one BLAS
    thread each."""
    names = [a for nm in G2_NAMES for a in ("--name", nm)]
    procs = {b: subprocess.Popen([PY, "-m", "designlab.stage2.look", "g2", "--body", b, *names], cwd=ROOT, env=_env(),
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for b in look.BODIES}
    out = {}
    for b, p in procs.items():
        so, se = p.communicate(timeout=1800)
        assert p.returncode == 0, se[-3000:]
        out[b] = json.loads(so.strip().splitlines()[-1])
    return out


@pytest.mark.parametrize("body", look.BODIES)
def test_g2_stage1_references_regenerate_bit_for_bit(body, g2_hashes):
    """Gate G2: every Stage 1 reference of both bodies (the walks, and the turns of the guarded ``turning.generate``),
    at round two's accuracy (refs.ACCURACY), regenerates to the sha256 Stage 1 recorded, and every record leaf still
    matches."""
    rec = json.loads((RUN / body / "refs.json").read_text())
    assert rec["accuracy"] == "r2" and set(G2_NAMES) == set(rec["references"])
    for name in G2_NAMES:
        got = g2_hashes[body][name]
        assert got["sha256"] == rec["references"][name]["sha256"], (body, name)
        assert got["matches_records"], (body, name)


# ------------------------------------------------------------------------------------------------
# Unit pieces of the look features
# ------------------------------------------------------------------------------------------------
def test_lift_profile_peaks_early_and_keeps_the_middle_of_the_swing_high():
    """F4: the peak where asked, zero speed and today's acceleration at lift-off and touchdown, and (finding: the old
    warp kept 0.28 of the lift at 75 % of the swing, under the clearance screen's 0.04 H with a 0.10 H lift) at least
    0.45 of the lift over the middle half of the swing for a peak at 35-40 %."""
    s = np.linspace(0.0, 1.0, 200001)
    assert np.array_equal(gait.lift_profile(s), np.sin(np.pi * s) ** 2)  # today's lift, exactly
    for peak in (0.3, 0.35, 0.375, 0.40, 0.6):
        y = gait.lift_profile(s, peak)
        assert s[np.argmax(y)] == pytest.approx(peak, abs=1e-4) and y.max() == pytest.approx(1.0)
        assert y[0] == 0.0 and abs(y[-1]) < 1e-20 and np.gradient(y, s)[[0, -1]] == pytest.approx([0, 0], abs=1e-3)
        acc = np.gradient(np.gradient(y, s), s)
        assert acc[[2, -3]] == pytest.approx([2 * np.pi**2] * 2, rel=2e-2)  # today's lift-off and touchdown acceleration
        assert np.all(np.diff(gait._lift_warp(s, peak)) > 0)  # a monotone warp
        if 0.35 <= peak <= 0.40:
            mid = (s >= 0.25) & (s <= 0.75)
            assert y[mid].min() >= 0.45 and y[mid].min() == pytest.approx(gait.lift_profile(0.75, peak), abs=1e-4)
            up, down = s <= peak, s >= peak
            rate = np.gradient(y, s)
            assert rate[up].max() > np.pi > -rate[down].min()  # rises quicker than today's, descends slower
    with pytest.raises(ValueError):
        gait.lift_profile(s, 0.2)  # the warp would not be monotone


def test_bob_profile_holds_drops_and_rises():
    """F2 (finding: 72-81 % of the old dip came before the touchdown): with ``drop``, the crest held until ``drop``
    before the trough, a min-jerk drop and a min-jerk rise, C2 throughout; without it, the smooth warped cosine."""
    tau = np.linspace(0.0, 1.0, 100001)
    for trough, drop in ((0.55, 0.2), (0.485, 0.196), (0.58, 0.275)):
        y = gait.bob_profile(tau, trough, drop)
        start = trough - drop
        assert np.all(y[tau <= start] == 0.0) and y.max() == pytest.approx(1.0)
        assert tau[np.argmax(y)] == pytest.approx(trough, abs=1e-4)
        assert np.all(np.diff(y[(tau > start) & (tau <= trough)]) >= 0) and np.all(np.diff(y[tau > trough]) <= 0)
        d1 = np.gradient(y, tau)
        d2 = np.gradient(d1, tau)
        assert np.abs(np.diff(d1)).max() < 1e-2 and np.abs(np.diff(d2)).max() < 5.0  # no jump in rate or curvature
        assert (1 - trough) > drop  # the rise is slower than the drop here
    for trough in (0.45, 0.5, 0.58):
        y = gait.bob_profile(tau, trough)
        assert tau[np.argmax(y)] == pytest.approx(trough, abs=1e-4) and y[0] == 0.0 and y.max() == pytest.approx(1.0)
    spec = replace(sizing.NOMINAL, bob_skew=0.125)
    assert gait.bob_trough(spec, 0.6) == pytest.approx(0.4 + 0.15) and gait.bob_trough(sizing.NOMINAL, 0.6) == 0.5


def test_toe_off_bump_peaks_at_lift_off_and_the_caps():
    """F5 (finding: at the faster rungs the old roll peaked 65-76 ms into the swing, a heel flick in the air): the pitch
    rises to its peak at the lift-off and only falls after it; its amplitude is capped by the roll's mean rate and peak
    acceleration on the floor."""
    t = np.linspace(-1.0, 1.0, 20001)
    th, pk = gait.toe_off_bump(t, -0.3, 0.0, 0.2, 0.4)
    assert pk == 0.0 and th.max() == pytest.approx(0.4) and t[np.argmax(th)] == pytest.approx(0.0, abs=1e-4)
    assert np.all(np.diff(th[t > 0.0]) <= 1e-15) and th[t <= -0.3].max() == 0.0 and th[t >= 0.2].max() == 0.0
    th, pk = gait.toe_off_bump(t, -0.05, 0.0, 0.2, 0.1)
    assert pk == 0.0 and t[np.argmax(th)] == pytest.approx(0.0, abs=1e-4)
    with pytest.raises(ValueError):
        gait.toe_off_bump(t, 0.0, 0.0, 0.2, 0.1)
    spec = replace(sizing.NOMINAL, toe_off=0.30, toe_off_rate=3.4, toe_off_accel=220.0)
    for D in (0.0934, 0.96 * 0.0934, 0.105):  # the nominal walks' rolls, with F14's jitter: the full toe-off
        assert gait.roll_amplitude(spec, 0.0, D) == pytest.approx(0.30)
    assert gait.roll_amplitude(spec, 0.0, 0.023) == pytest.approx(220.0 * 0.023**2 / gait.MIN_JERK_ACC)
    assert gait.roll_amplitude(replace(spec, toe_off_accel=None), 0.0, 0.023) == pytest.approx(3.4 * 0.023)
    u = np.linspace(0.0, 1.0, 100001)
    assert gait.MIN_JERK_ACC == pytest.approx(np.max(np.gradient(np.gradient(gait._min_jerk(u), u), u)), rel=1e-4)


def test_warp_inverts_and_closes_the_window():
    rng = np.random.default_rng(1)
    spec = replace(sizing.NOMINAL, strides=4, phase_jitter=0.015, asym=0.025)
    e = gait.step_deviations(spec, rng)
    assert len(e) == 8 and abs(e.sum()) < 1e-15 and np.max(np.abs(e)) <= 0.015 + 0.0125 + 1e-15
    w = gait._Warp(1.3, 4, e)
    phi = np.linspace(-1.0, 5.0, 3001)
    assert w.phase(w.time(phi)) == pytest.approx(phi, abs=1e-12)
    assert w.time(4.0) == pytest.approx(4 * 1.3, abs=1e-12) and w.time(0.0) == 0.0
    steps = np.diff(w.time(np.arange(9) / 2))
    assert steps == pytest.approx(0.65 * (1 + e), abs=1e-12)  # step j lasts T / 2 (1 + e_j)
    assert np.all(w.rate(phi) > 0)


def test_variation_is_seeded_and_bounded():
    spec = replace(sizing.NOMINAL, strides=10, phase_jitter=0.015, asym=0.025)
    a = gait.step_deviations(spec, np.random.default_rng(np.random.SeedSequence(2026).spawn(3)[0]))
    b = gait.step_deviations(spec, np.random.default_rng(np.random.SeedSequence(2026).spawn(3)[0]))
    c = gait.step_deviations(spec, np.random.default_rng(np.random.SeedSequence(7).spawn(3)[0]))
    assert np.array_equal(a, b) and not np.allclose(a, c)
    assert np.max(np.abs(a - a.mean())) <= 0.015 + 0.0125 + 1e-12
    t = np.linspace(0.0, 10.2, 1632, endpoint=False)
    g, f = gait.slow_variation(np.random.default_rng(3), 10.2, (0.05, 0.2), t)
    assert f == pytest.approx([1 / 10.2, 2 / 10.2]) and np.abs(g).max() <= 1.0 + 1e-12
    assert np.abs(g).max() > 0.99
    g2, f2 = gait.slow_variation(np.random.default_rng(3), 3.0, (0.05, 0.2), t)  # no harmonic in the band
    assert f2 == [] and not g2.any()


def test_speed_spec_is_the_speed_change_at_0_40():
    """0.40 m/s: the gait brisk_to_fr05 passes through at 0.40 m/s (period, duty, lift mixed with the speed's weight)."""
    for body, H, w, T, stride, duty in (("60", H_BODY["60-N-T-10"], 0.651, 0.751, 0.742, 0.567),
                                        ("75", H_BODY["75-M-H-10"], 0.338, 0.861, 0.686, 0.573)):
        s = look.speed_spec(H)
        d = s.describe(H)
        assert d["speed"] == pytest.approx(0.40, abs=1e-12)
        assert d["period"] == pytest.approx(T, abs=1e-3) and s.stride_rel == pytest.approx(stride, abs=1e-3)
        assert s.duty == pytest.approx(duty, abs=1e-3) and s.lift_frac == pytest.approx(0.11)
        a, b = sizing.BRISK.describe(H), envelope.FAST.describe(H)
        ww = (0.40 - a["speed"]) / (b["speed"] - a["speed"])
        assert ww == pytest.approx(w, abs=1e-3) and d["period"] == pytest.approx(a["period"] + ww * (b["period"] - a["period"]))
        assert not gait.look_active(s), body


def test_ladder_values_and_one_change_per_rung():
    H = H_BODY[BODY]
    lad = look.ladder(BODY, H, "nominal", tail_phase=-1.8)
    v0, v6 = lad["V0"], lad["V6"]
    assert v0 is sizing.NOMINAL
    assert lad["V1"].bob == pytest.approx(0.75 * (0.6 * 0.55 / 2) ** 2 / (2 * 0.97)) == pytest.approx(0.0105, abs=2e-4)
    assert (lad["V1"].bob_skew, lad["V1"].bob_drop) == (0.125, 0.10)
    assert lad["V1B50"].bob == pytest.approx(lad["V1"].bob / 0.75 * 0.5) and lad["V1B100"].bob == pytest.approx(lad["V1"].bob / 0.75)
    assert look.ladder(BODY, H, "fr05")["V1"].bob == pytest.approx(0.0195, abs=2e-4)  # research: 0.013-0.026 H
    assert lad["V2"].swing_time == pytest.approx(0.32, abs=2e-3) and lad["V2"].describe(H)["duty"] == pytest.approx(0.69, abs=0.005)
    assert look.ladder("75-M-H-10", H_BODY["75-M-H-10"], "nominal")["V2"].swing_time == pytest.approx(0.356, abs=1e-3)
    v3 = lad["V3"]
    assert (v3.lift_peak, v3.toe_off, v3.toe_off_rate, v3.toe_off_accel, v3.toe_off_start, v3.toe_reach) == (
        0.375, 0.30, 3.4, 220.0, 0.825, 0.075)
    assert (lad["V3T"].toe_off, lad["V3T"].toe_off_rate, lad["V3T"].toe_off_accel) == (0.40, 4.5, 290.0)
    assert lad["V4"].roll_amp == pytest.approx(math.radians(1.5)) and lad["V4L"].roll_amp == -lad["V4"].roll_amp
    assert (lad["V5"].tail_amp, lad["V5"].tail_phase, lad["V5"].head_mode) == (0.125, -1.8, "stabilise")
    assert (lad["V5"].head_gain, lad["V5"].head_settle, lad["V5"].settle_lag) == (0.9, 0.015, 0.075)
    assert (v6.phase_jitter, v6.asym, v6.var_band, v6.stride_rel) == (0.015, 0.025, (0.05, 0.2), 0.55)
    assert v6.strides * v6.describe(H)["period"] >= look.CLIP_S > (v6.strides - 1) * v6.describe(H)["period"]
    assert lad["V6S"].zmp_inset_y == 0.010 and "V5F13" not in lad
    # F13 at 75 cm's animal pace: V5F13 (F13's stride, its own tail phase), V6 = V5F13 + F14; without that phase, none
    l75 = look.ladder("75-M-H-10", H_BODY["75-M-H-10"], "fr05", tail_phase=0.0, tail_phase_f13=-1.0)
    assert l75["V5F13"].stride_rel == 0.875 and l75["V5F13"].tail_phase == -1.0 and l75["V5"].tail_phase == 0.0
    assert l75["V6"].stride_rel == 0.875 and l75["V6"].tail_phase == -1.0
    assert not {"V5F13", "V6", "V6S"} & set(look.ladder("75-M-H-10", H_BODY["75-M-H-10"], "fr05", tail_phase=0.0))
    assert look.ladder(BODY, H, "fr05", tail_phase=0.0)["V6"].stride_rel == 0.80
    allowed = {"bob", "bob_skew", "bob_drop", "swing_time", "lift_peak", "toe_off", "toe_off_rate", "toe_off_accel",
               "toe_reach", "roll_amp", "tail_amp", "tail_phase", "head_mode", "head_settle", "phase_jitter", "asym",
               "gaze_var", "tail_var", "noise_seed", "strides", "stride_rel"}
    chains = [("V0", "V1", {"zmp_iters"}), ("V1", "V2", set()), ("V2", "V3", set()), ("V3", "V4", set()),
              ("V4", "V5", set()), ("V5", "V6", set())]
    for a, b, extra in chains:  # one feature group per rung; V1 also raises the correction passes (checks.zmp_passes)
        changed = {k for k, x in asdict(lad[b]).items() if asdict(lad[a])[k] != x}
        assert changed and changed <= allowed | extra, (a, b, changed)
    changed = {k for k, x in asdict(l75["V6"]).items() if asdict(l75["V5F13"])[k] != x}
    assert changed <= {"phase_jitter", "asym", "gaze_var", "tail_var", "noise_seed", "strides"}  # V6 adds F14 alone
    assert lad["V1"].zmp_iters == look.LOOK_ZMP_ITERS == 12 and v0.zmp_iters == 6
    assert look.features_of("V6", "75-M-H-10", "fr05")[-1] == "F14" and "F13" in look.features_of("V6", "75-M-H-10", "fr05")
    assert "F13" not in look.features_of("V6", BODY, "fr05")


# ------------------------------------------------------------------------------------------------
# 2. Features measured on generated walks (60-N-T-10, nominal)
# ------------------------------------------------------------------------------------------------
_SRC = {}


def _src(body: str = BODY) -> dict:
    if body not in _SRC:
        _SRC[body] = look.source(body)
    return _SRC[body]


def _tail_phase(body: str, speed: str, key: str = "tail_phase_F9_rad"):
    doc = _record(LOOK / "tail_phase.json")
    try:
        return doc["bodies"][body][speed][key]
    except (TypeError, KeyError):
        return -math.pi / 2


@pytest.fixture(scope="module")
def walks():
    """V0 (coarse), V1, V3, V5 at the nominal walk and a three-stride F14 window (coarse): trajectory, arrays,
    measures."""
    src = _src()
    lad = look.ladder(BODY, src["H"], "nominal", tail_phase=_tail_phase(BODY, "nominal"))
    specs = dict(V0=replace(lad["V0"], n=80), V1=lad["V1"], V3=lad["V3"], V5=lad["V5"],
                 V6w=replace(lad["V6"], strides=3, n=80))
    out = {}
    for k, s in specs.items():
        g = look.generate(src, s)
        a = look.arrays(src, g)
        out[k] = dict(spec=s, g=g, a=a, m=look.measure(a, g["info"], src["H"]))
    return out


def test_bob_amplitude_and_weight_acceptance(walks):
    """F1/F2 from the hip-roll axes' height: the bob's depth, its trough 12.5 % of the stance after each touchdown, a
    quick dip (about bob_drop) and a slow rise, nearly all of the dip after the touchdown at the nominal walk, the
    crest at mid single support (today's height)."""
    H = _src()["H"]
    for k in ("V1", "V3", "V5"):
        w = walks[k]
        m = w["m"]
        assert m["bob_pp_H"] == pytest.approx(w["spec"].bob, rel=0.01), k
        assert m["bob_trough_frac_stance"] == pytest.approx(0.125, abs=0.01), k
        assert np.ptp(w["g"]["q"][:, 2]) == pytest.approx(w["spec"].bob * H, rel=2e-3), k  # the pelvis carries it
        assert m["bob_fall_s"] == pytest.approx(0.88 * w["spec"].bob_drop, abs=0.012), k  # 2 % of the depth to trough
        assert m["bob_rise_s"] > 2.0 * m["bob_fall_s"], k
        assert m["bob_share_after_touchdown"] > 0.9, k
    v1 = walks["V1"]
    phase, z = v1["a"]["phase"], v1["a"]["hip_mid"][:, 2]
    duty = v1["g"]["info"]["duty"]
    mid = np.mod(np.array([0.5 - duty / 2, 1 - duty / 2]), 1.0)  # mid single support: today's height
    k = [int(np.argmin(np.abs(phase - p))) for p in mid]
    assert z[k] == pytest.approx(z.max(), abs=2e-5)
    assert walks["V0"]["m"]["bob_pp_H"] < 1e-6 and walks["V0"]["m"]["mechanical_tells"]["constant_hip_height"]
    assert walks["V1"]["m"]["bob_share_of_enforced"] == pytest.approx(0.75, rel=1e-3)
    assert walks["V3"]["m"]["bob_share_of_enforced"] == pytest.approx(0.57, abs=0.01)  # F3's longer stance


def test_duty_swing_time_and_cadence_from_the_floor_contact(walks):
    """Duty, swing time and cadence measured from each foot's detected floor contact (its lowest point over every
    foot geom), not from the gait's flags: within a frame of the plan at each event, F3's fixed swing, alternating
    touchdowns at 2 / T."""
    for k in ("V0", "V1", "V3", "V6w"):
        w = walks[k]
        info, m, a = w["g"]["info"], w["m"], w["a"]
        n, dt = len(a["t"]), float(a["dt"])
        assert m["duty"] == pytest.approx([info["duty"]] * 2, abs=0.02 if w["spec"].asym else 2.5 / n), k
        assert m["swing_time_s"] == pytest.approx((1 - info["duty"]) * info["period"], abs=1.5 * dt), k
        assert m["swings_detected"] == [w["spec"].strides] * 2 and m["touchdowns_alternate"], k
        assert m["cadence_steps_s"] == pytest.approx(2 / info["period"], rel=1e-9), k
    v3 = walks["V3"]
    assert v3["g"]["info"]["duty"] == pytest.approx(1 - v3["spec"].swing_time / v3["g"]["info"]["period"]) == pytest.approx(0.6865, abs=1e-3)
    assert v3["m"]["swing_time_s"] == pytest.approx(v3["spec"].swing_time, abs=1.5 * float(v3["a"]["dt"]))
    assert walks["V1"]["g"]["info"]["duty"] == 0.6  # V1 keeps the fixed duty


def test_lift_peak_and_the_clearance_screens_swing_gate(walks):
    """F4: the lowest pad point peaks at 37.5 % of the swing, and the round-two clearance screen's swing gate (lowest
    pad point over the middle half of each swing at least 0.04 H) passes (finding: the old warp left 0.030 H)."""
    assert walks["V3"]["m"]["lift_peak_frac_swing"] == pytest.approx(0.375, abs=0.005)
    assert walks["V1"]["m"]["lift_peak_frac_swing"] == pytest.approx(0.5, abs=0.005)
    a = walks["V3"]["a"]
    assert np.max(a["pad_floor_z"][a["swing"]]) == pytest.approx(walks["V3"]["g"]["info"]["lift"], rel=5e-3)
    src = _src()
    tr = look.trajectory(a, walks["V3"]["g"]["info"])
    cl = look._clearance_gates(src, tr)
    assert cl["gates"]["swing_clearance"] and cl["mid_swing_min_frac_H"] >= clearance.SWING_MIN_FRAC
    assert cl["walking_gates"], cl["gates"]


def test_toe_off_rolls_on_the_real_sole_and_never_below_the_floor(walks):
    """F5 (findings: the old roll took the claw 6.5 mm below the floor): the foot pitches heel-up to its peak at the
    lift-off, about the toe pad's front edge (in place), then over the transition arc on to the claw tip (rolling,
    no slip); IK honours the pitch; no part of the foot goes below the floor; the ZMP stays inside the actual contact;
    the toes rise before a flat landing."""
    w = walks["V3"]
    a, g, s = w["a"], w["g"], w["spec"]
    gs = _src()["geometry"]
    m = w["m"]
    assert np.abs(a["foot_pitch_solved"] - a["foot_pitch"]).max() < 1e-4  # IK honours the target pitch
    rolls = [r for side in ("left", "right") for r in g["look"]["rolls"][side] if 0 <= r["t_lift_off"] < g["t"][-1]]
    assert all(r["amplitude"] == pytest.approx(0.30) for r in rolls)  # the nominal walk reaches toe_off
    assert m["toe_off_at_lift_off_rad"] == pytest.approx(0.30, abs=2e-3) == pytest.approx(m["toe_off_peak_rad"], abs=2e-3)
    assert abs(m["toe_off_peak_after_lift_off_s"]) < 0.5 * float(a["dt"])  # the peak at the lift-off
    assert all(r["start_frac_stance"] >= s.toe_off_start - 1e-12 for r in rolls)
    assert m["claw_roll_frames"] > 0 and m["foot_below_floor_max_mm"] < 1e-3 and m["toe_geoms_below_floor_frames"] == 0
    rol = a["rolling"]
    assert rol.any() and m["rolling_pivot_height_max_mm"] < 1e-3  # the contact stays on the floor
    for j, side in enumerate(("left", "right")):
        cl = gs[side]["claw"]
        runs = np.split(np.flatnonzero(rol[:, j]), np.flatnonzero(np.diff(np.flatnonzero(rol[:, j])) > 1) + 1)
        for r in runs:
            edge = r[a["foot_pitch"][r, j] <= cl["arc"]["start_pitch"]]
            assert len(edge) and np.ptp(a["toe_pivot"][edge, j, :2], axis=0).max() < 1e-4  # the edge in place
            th = a["foot_pitch"][r, j]
            x = a["toe_pivot"][r, j, 0] - a["toe_pivot"][edge[0], j, 0]
            rolled = np.array([gait.roll_offset(gs[side], float(t))[1][0][0] - gs[side]["toe_pivot"][0] for t in th])
            assert x == pytest.approx(rolled, abs=1e-4)  # past the edge the contact advances by the length rolled
            assert np.all(np.diff(th) > 0)  # rising to the lift-off: the heel lifts first
    assert np.min(a["zmp_ref_margin"][rol.any(axis=1)]) >= s.toe_off_margin - 1e-6  # inside the actual contact
    assert np.min(a["zmp_margin"]) > 0.0
    assert m["foot_pitch_min_rad"] == pytest.approx(-s.toe_reach, abs=1e-3)  # toes-up reach before landing
    for j in range(2):
        td = a["touchdown_t"][j][np.isfinite(a["touchdown_t"][j])]
        assert np.abs(np.interp(td, a["t"], a["foot_pitch"][:, j])).max() < 1e-5  # flat landing (between frames)
    assert walks["V1"]["m"]["toe_off_peak_rad"] < 1e-9 and walks["V1"]["m"]["mechanical_tells"]["flat_feet_at_lift_off"]


def test_rigid_foot_claw_geometry():
    """The claw reaches the floor first as the rigid foot pitches about its toe edge (``look.toe_clearance``, by IK,
    agrees with ``gait.foot_geometry``'s analytic touch pitch); the transition arc joins the edge and the claw tip
    tangentially (C1 offsets) with the real foot at most 0.3 mm above the floor and never below it; pivoting about the
    edge alone would take the claw below the floor (the finding the arc fixes)."""
    for body in look.BODIES:
        src = _src(body)
        gs = src["geometry"]["left"]
        cl = gs["claw"]
        assert cl["geom"].endswith("claw_0") and 0.05 < cl["touch_pitch"] < 0.10
        tc = look.toe_clearance(src, pitch=0.30)
        assert tc["clear_pitch_rad"] == pytest.approx(cl["touch_pitch"], abs=2e-3) and tc["deepest"].endswith("claw_0")
        assert tc["deepest_mm"] < -3.0  # about the edge alone, at the ladder's 0.30 rad
        arc = cl["arc"]
        assert arc["start_pitch"] < cl["touch_pitch"] < arc["end_pitch"] and arc["radius"] > cl["radius"]
        assert -1e-9 <= arc["hover_min_m"] and arc["hover_max_m"] < 3e-4
        th = np.linspace(0.0, 0.45, 4501)
        off = np.array([gait.roll_offset(gs, float(t))[0] for t in th])
        d1 = np.diff(off, axis=0) / np.diff(th)[:, None]
        assert np.abs(np.diff(d1, axis=0)).max() < 1e-4  # no corner: the velocity never jumps


def test_trunk_roll_banks_toward_the_stance_leg(walks):
    w = walks["V5"]
    assert w["m"]["trunk_roll_amp_rad"] == pytest.approx(w["spec"].roll_amp, rel=1e-4)  # sampled at the frames
    assert w["m"]["trunk_roll_toward_stance"]
    phase, roll = w["a"]["phase"], w["g"]["look"]["trunk_roll"]
    duty = w["g"]["info"]["duty"]
    k = int(np.argmin(np.abs(np.mod(phase, 1.0) - (1 - duty / 2))))  # the left foot's mid-stance
    assert roll[k] == pytest.approx(-w["spec"].roll_amp, abs=1e-5)  # negative about the trunk's x: left side down
    assert walks["V3"]["m"]["trunk_roll_amp_rad"] < 1e-12


def test_tail_and_head_phase(walks):
    """F9: the tail at its phase, its tip moving toward the stance side in every swing (today's moves toward the
    swinging leg's side: research Bottom line 4); F10: no nod, the settle dip settle_lag after each touchdown, a
    steadier head in the world."""
    w, w0 = walks["V5"], walks["V0"]
    s = w["spec"]
    assert w["m"]["tail_amp_rad"] == pytest.approx(s.tail_amp, rel=1e-6)
    assert math.cos(w["m"]["tail_phase_rad"] - s.tail_phase) == pytest.approx(1.0, abs=1e-9)
    assert w["m"]["tail_tip_toward_stance_in_swing"] == 1.0
    assert w0["m"]["tail_phase_rad"] == pytest.approx(math.pi / 2, abs=1e-9)
    assert w0["m"]["tail_tip_toward_stance_in_swing"] == 0.0
    lk = w["g"]["look"]
    assert np.abs(lk["neck"] - (-s.head_settle * lk["head_settle"] + s.gaze_var * lk["gaze_var"])).max() < 1e-15
    assert w["m"]["settle_lag_s"] == pytest.approx(s.settle_lag, abs=1e-3)
    assert w["m"]["settle_depth_rad"] == pytest.approx(s.head_settle, rel=1e-3)  # sampled at the frames
    assert w0["m"]["nod_amp_rad"] == pytest.approx(0.05, rel=1e-6) and w["m"]["nod_amp_rad"] < 0.005
    assert w["m"]["head_world_pitch_rms_deg"] < 0.2 * w0["m"]["head_world_pitch_rms_deg"]


def test_stride_from_the_footprints(walks):
    for k in ("V0", "V5", "V6w"):
        w = walks[k]
        assert w["m"]["stride_H"] == pytest.approx(w["spec"].stride_rel, rel=1e-4), k  # touchdowns between frames


def test_variation_window_is_bounded_seeded_and_periodic(walks):
    w = walks["V6w"]
    s, g, a = w["spec"], w["g"], w["a"]
    e = np.array(g["info"]["step_deviations"])
    bound = s.phase_jitter + s.asym / 2
    assert np.max(np.abs(e)) == pytest.approx(bound, abs=0.004) and np.max(np.abs(e)) <= bound + 1e-12
    assert w["m"]["step_time_dev_max"] == pytest.approx(np.max(np.abs(e)), abs=1e-4)  # measured from the events
    assert w["m"]["step_time_dev_std"] > 0.005 and not w["m"]["mechanical_tells"]["identical_steps"]
    assert w["m"]["step_interval_spread_s"] > 0.01  # the detected touchdowns are not evenly spaced
    rng = np.random.default_rng(np.random.SeedSequence(s.noise_seed).spawn(3)[0])
    assert np.array_equal(gait.step_deviations(s, rng), e)  # seeded
    T, K = g["info"]["period"], s.strides
    assert len(a["t"]) == K * s.n and a["phase"][0] == 0.0 and a["phase"][-1] < K
    assert g["info"]["window_s"] == pytest.approx(K * T)
    assert np.array_equal(a["stride_motion"], [0.0, 0.0, 0.0, K * g["info"]["stride"], 0.0])
    assert g["ik_ok"] and np.min(a["zmp_margin"]) > 0


def test_zmp_correction_converges_with_the_look_features(walks):
    """The multibody-ZMP correction passes still converge with the bob, the toe-off and the roll: the error falls to
    about today's millimetre or below, the last pass within 0.05 mm of the best (V3's falls 50 -> 0.56 mm with one
    bump of 0.2 mm on the way)."""
    for k in ("V1", "V3", "V5"):
        h = walks[k]["g"]["info"]["zmp_error_history_m"]
        assert h[-1] < 2e-3 and h[-1] < 0.1 * h[0], (k, h)
        assert h[-1] <= min(h) + 5e-5 and h[len(h) // 2] < 0.2 * h[0], (k, h)


def test_look_inverse_dynamics_holds_the_rolling_cop_on_its_contact(walks):
    """Finding: Stage 0's double-support split keeps a heel-up foot's CoP in its flat pad box, 50-85 mm behind its toe
    edge, under a heel in the air.  ``look_inverse_dynamics`` holds it on the toe edge, transition arc or claw tip on
    every rolling frame, the split solving; every other frame is ``gait.inverse_dynamics``'s bit for bit."""
    w = walks["V3"]
    a, info = w["a"], w["g"]["info"]
    M = _src()["M"]
    tr = look.trajectory(a, info)
    assert tr["ik_ok"] is True and tr["ik_fail_frames"] == []
    with look.accuracy("r2"):
        idr = look.look_inverse_dynamics(M, tr, a["rolling"], a["toe_edge"])
        ref = gait.inverse_dynamics(M, tr)  # the screen's call
    rol = np.asarray(a["rolling"], bool)
    other = ~rol.any(axis=1)
    assert np.array_equal(idr["tau"][other], ref["tau"][other]) and np.array_equal(idr["qd"], a["qd"])
    assert np.array_equal(idr["cop_margin"][other], ref["cop_margin"][other], equal_nan=True)
    assert not idr["split_failed_frames"] and np.nanmax(idr["rolling_cop_off_line"]) < 1e-9
    assert np.isfinite(idr["rolling_cop_off_line"][rol]).all()
    edge = rol & (a["foot_pitch"] <= _src()["geometry"]["left"]["claw"]["arc"]["start_pitch"])
    assert np.nanmin(idr["rolling_cop_along"][edge]) >= -1e-9  # between the edge's ends
    # the old split's CoP for the same frames lies far behind the toe edge (the finding)
    k = int(np.flatnonzero(rol.any(axis=1))[0])
    j = int(np.flatnonzero(rol[k])[0])
    side = ("left", "right")[j]
    with look.accuracy("r2"):
        _, wr = statics.contact_torques(M, tr["q"][k], ref["qd"][k], ref["qdd"][k], support=tr["phases"][k])
    M.forward(tr["q"][k])
    assert a["toe_pivot"][k, j, 0] - statics.cop(M, side, wr[side])[0] > 0.02
    bare = {kk: v for kk, v in a.items() if kk not in ("ik_ok", "ik_fail_frames")}
    with pytest.raises(KeyError):
        look.trajectory(bare, info)  # arrays that do not record the IK's result are refused


def test_price_of_a_rung(walks):
    """The Stage 0 price of a rung: the drive ratios are ``envelope._ratios`` of the F5-aware inverse dynamics against
    the record's drives at its bus; the contact and clearance blocks; the verdict follows the limits."""
    w = walks["V3"]
    src = _src()
    pr = look.price(src, w["a"], w["g"]["info"])
    M = src["M"]
    tr = look.trajectory(w["a"], w["g"]["info"])
    with look.accuracy("r2"):
        idr = look.look_inverse_dynamics(M, tr, w["a"]["rolling"], w["a"]["toe_edge"])
    r = envelope._ratios(src["design"], src["record"]["selection"]["bus"],
                         sizing._motor_space(M.names, idr["tau"], idr["omega"], src["layout"]))
    for g in sizing.GROUPS:
        for kk in ("cont_ratio", "peak_ratio", "speed_ratio"):
            assert pr["drives"][g][kk] == pytest.approx(r[g][kk], rel=1e-12), (g, kk)
            assert pr["drives"][g]["actuator"] == src["design"].act(g).id
    worst = max(v["worst"] for v in r.values())
    assert pr["worst_ratio"] == pytest.approx(worst)
    assert pr["within_limits"] == (not pr["over"]) and (pr["verdict"] == "within the Stage 0 limits") == pr["within_limits"]
    assert all(v["cont_ratio"] <= 1 and v["peak_ratio"] <= 1 and v["speed_ratio"] <= 1 for v in r.values()) == (
        not any(o["limit"].split()[0] in sizing.GROUPS for o in pr["over"]))
    c = pr["contact"]
    assert c["rolling_frames"] > 0 and c["split_failed_frames"] == 0 and c["rolling_cop_off_contact_max_mm"] < 1e-6
    assert 0.0 < c["total_normal_frac_weight"][0] < 1.0 < c["total_normal_frac_weight"][1]
    assert pr["clearance"]["gates"]["swing_clearance"] and pr["passive"]["tail"]["damping"] == 0.2


def test_hybrid_animal_pace_with_every_feature():
    """75-M-H-10 at the animal pace with every feature (V6, with F13's 0.875 H stride) on a coarse two-stride window: IK
    on every frame, the ZMP inside its support, the correction converging, and each feature where its spec puts it;
    F5 reduced to what its caps allow in the short double support."""
    body = "75-M-H-10"
    src = _src(body)
    lad = look.ladder(body, src["H"], "fr05", tail_phase=_tail_phase(body, "fr05"),
                      tail_phase_f13=_tail_phase(body, "fr05", "tail_phase_F9_F13_rad"))
    spec = replace(lad["V6"], strides=2, n=80)
    assert spec.stride_rel == 0.875 and lad["V5"].stride_rel == 0.80  # F13 comes with V5F13 / V6, at 75 cm only
    g = look.generate(src, spec)
    a = look.arrays(src, g)
    m = look.measure(a, g["info"], src["H"])
    info = g["info"]
    h = info["zmp_error_history_m"]  # (80 frames a stride: coarser differences than the ladder's 160)
    assert g["ik_ok"] and m["zmp_margin_min_mm"] > 0.0 and h[-1] < 4e-3 and h[-1] < 0.1 * h[0], h
    assert info["duty"] == pytest.approx(1.0 - spec.swing_time / info["period"]) == pytest.approx(0.598, abs=2e-3)
    assert m["stride_H"] == pytest.approx(0.875, rel=1e-4) and m["bob_pp_H"] == pytest.approx(spec.bob, rel=0.02)
    assert m["lift_peak_frac_swing"] == pytest.approx(0.375, abs=0.01)
    rolls = [r for s in ("left", "right") for r in g["look"]["rolls"][s]]
    for r in rolls:
        assert r["amplitude"] == pytest.approx(gait.roll_amplitude(spec, r["t_start"], r["t_lift_off"]))
        assert r["amplitude"] < 0.1 and r["ground_roll_s"] < 0.05  # the short double support caps it
    assert m["toe_off_ground_roll_s"] > 0.0 and m["foot_below_floor_max_mm"] < 1e-3
    assert m["trunk_roll_amp_rad"] == pytest.approx(spec.roll_amp, rel=1e-3) and m["trunk_roll_toward_stance"]
    # no metronome nod: what is left at the step frequency is F10's settle dip (0.015 rad, twice a stride)
    assert m["nod_amp_rad"] < 0.01 and not m["mechanical_tells"]["metronome_nod"]
    assert m["step_time_dev_max"] <= spec.phase_jitter + spec.asym / 2 + 1e-4
    assert np.abs(a["foot_pitch_solved"] - a["foot_pitch"]).max() < 1e-4


@pytest.mark.parametrize("body", look.BODIES)
def test_markers_on_both_bodies(body):
    """The point-light markers resolve on both bodies (the Hybrid's tail cover is a root and segments, the Tapered's
    one mesh) and sit where their names say at home."""
    src = _src(body)
    assert src["H"] == H_BODY[body]
    M = src["M"]
    M.forward(src["q_home"])
    d = M.d
    p = {nm: d.xpos[b] + d.xmat[b].reshape(3, 3) @ loc for nm, b, loc in src["markers"]}
    assert list(p) == list(look.POINTS)
    assert p["snout"][0] > p["eye"][0] > p["neck_base"][0] > p["hip_l"][0] > p["tail_mid"][0] > p["tail_tip"][0]
    assert p["tail_tip"][0] < -1.5 * src["H"] and abs(p["tail_tip"][1]) < 1e-3  # far behind, on the midline
    for s in ("l", "r"):
        assert p[f"toe_{s}"][0] > p[f"mtp_{s}"][0] and 0.0 < p[f"toe_{s}"][2] < p[f"mtp_{s}"][2] < p[f"knee_{s}"][2]
    assert p["hip_l"][1] == pytest.approx(-p["hip_r"][1]) and p["hip_l"][1] > 0


def test_markers_and_arrays_for_the_screen_and_renders(walks):
    a = walks["V5"]["a"]
    n = len(a["t"])
    assert a["points"].shape == (n, len(look.POINTS), 3) and list(a["point_names"]) == list(look.POINTS)
    i = list(look.POINTS)
    assert np.all(a["points"][:, i.index("snout"), 0] > a["points"][:, i.index("eye"), 0])
    assert np.all(a["points"][:, i.index("tail_tip"), 0] < a["points"][:, i.index("tail_mid"), 0])
    assert a["support_hull"].shape[0] == n and np.array_equal(a["support"], ~a["swing"])
    assert a["toe_edge"].shape == (n, 2, 2, 3)  # a rolling foot's support: its contact, on the floor while it rolls
    rol = a["rolling"]
    assert np.abs(a["toe_edge"][rol][:, :, 2]).max() < 1e-6
    assert np.allclose(a["toe_edge"].mean(axis=2), a["toe_pivot"], atol=1e-12)
    assert bool(a["ik_ok"]) and a["ik_fail_frames"].shape == (0,)


# ------------------------------------------------------------------------------------------------
# 3. The committed ladder
# ------------------------------------------------------------------------------------------------
LADDER = _record(LOOK / "look_refs.json")
ENTRIES = [] if LADDER is None else [(b, s, v) for b, d in LADDER["bodies"].items() for s, e in d.items() for v in e]
needs_ladder = pytest.mark.skipif(LADDER is None, reason="stage2/look/look_refs.json not generated")


def _rungs(body: str, speed: str) -> set:
    rungs = set(look.VARIANTS + look.EXTRAS)
    return rungs if look.f13_applies(body, speed) else rungs - {"V5F13"}


@needs_ladder
def test_the_ladder_has_every_rung():
    for b in look.BODIES:
        for s in look.SPEEDS:
            assert set(LADDER["bodies"][b][s]) == _rungs(b, s), (b, s)
    assert LADDER["deferred"] == look.DEFERRED and LADDER["limit_rule"] == look.LIMIT_RULE


@needs_ladder
@pytest.mark.parametrize("body,speed,variant", ENTRIES)
def test_every_variant_solves_and_keeps_the_zmp_inside_its_support(body, speed, variant):
    e = LADDER["bodies"][body][speed][variant]
    m, spec = e["measured"], e["spec"]
    assert e["ik_ok"] and e["ik_fail_frames"] == 0
    assert e["zmp_margin_min_mm"] > 0.0 and m["zmp_margin_min_mm"] > 0.0  # the generator's, and from the saved hull
    # the multibody-ZMP correction converges with the features, to about today's error at this body and speed (V0 is
    # Stage 1's reference there: 1.3-2.6 mm after its 6 passes; the look rungs run 12)
    h = e["info"]["zmp_error_history_m"]
    v0 = LADDER["bodies"][body][speed]["V0"]["zmp_error_final_mm"]
    assert h[-1] < 0.1 * h[0] and e["zmp_error_final_mm"] <= 1.25 * v0 and e["zmp_error_final_mm"] < 3.5, (h, v0)
    H = H_BODY[body]
    n = e["frames"]
    assert m["bob_pp_H"] == pytest.approx(spec["bob"], abs=2e-4)
    if spec["bob_skew"] is not None:
        assert m["bob_trough_frac_stance"] == pytest.approx(spec["bob_skew"], abs=0.02)
        assert m["bob_rise_s"] > m["bob_fall_s"] and m["bob_share_after_touchdown"] > 0.45
    assert m["duty"] == pytest.approx([e["info"]["duty"]] * 2, abs=0.02 if spec["asym"] else 2.5 / spec["n"])
    assert m["swings_detected"] == [spec["strides"]] * 2 and m["touchdowns_alternate"]
    assert m["lift_peak_frac_swing"] == pytest.approx(spec["lift_peak"], abs=0.01)
    assert m["trunk_roll_amp_rad"] == pytest.approx(abs(spec["roll_amp"]), rel=1e-3, abs=1e-9)
    if spec["roll_amp"]:
        assert m["trunk_roll_toward_stance"] == (spec["roll_amp"] > 0)
    assert math.cos(m["tail_phase_rad"] - spec["tail_phase"]) == pytest.approx(1.0, abs=1e-4)
    assert m["stride_H"] == pytest.approx(spec["stride_rel"], rel=1e-4)
    assert m["step_time_dev_max"] <= spec["phase_jitter"] + spec["asym"] / 2 + 1e-4
    assert n == spec["n"] * spec["strides"] and e["info"]["window_s"] >= (look.CLIP_S if spec["strides"] > 1 else 0)
    assert e["info"]["speed"] == pytest.approx({"nominal": math.sqrt(0.012 * gait.G * H), "v040": 0.40,
                                                "fr05": math.sqrt(0.05 * gait.G * H)}[speed])
    if spec["toe_off"]:  # F5: a roll on the floor peaking at the lift-off, within its caps, no heel flick
        assert m["toe_off_ground_roll_s"] > 0.0 and 0.0 < m["toe_off_at_lift_off_rad"] <= spec["toe_off"] + 1e-6
        assert abs(m["toe_off_peak_after_lift_off_s"]) < 0.004
        assert m["toe_off_peak_rad"] == pytest.approx(m["toe_off_at_lift_off_range_rad"][1], abs=1e-3)
        for side in ("left", "right"):
            for r in e["rolls"][side]:
                assert r["amplitude"] <= min(spec["toe_off"], spec["toe_off_rate"] * r["ground_roll_s"],
                                             spec["toe_off_accel"] * r["ground_roll_s"] ** 2 / gait.MIN_JERK_ACC) + 1e-12
        assert m["foot_pitch_min_rad"] == pytest.approx(-spec["toe_reach"], abs=2e-3)
    else:  # flat feet: nothing pitches, and the toes and claws never reach the floor
        assert m["toe_off_ground_roll_s"] is None and m["toe_geoms_below_floor_max_mm"] == 0.0
    assert m["foot_below_floor_max_mm"] < 1e-3  # no part of the foot below the floor (finding: the claw went 6.5 mm)
    assert m["rolling_pivot_height_max_mm"] is None or m["rolling_pivot_height_max_mm"] < 1e-3
    assert m["stance_pad_height_max_mm"] < 0.1  # flat stance feet stay on the floor


@needs_ladder
@pytest.mark.parametrize("body,speed,variant", ENTRIES)
def test_every_rung_is_priced_against_the_stage0_gates(body, speed, variant):
    """Findings 1-3: every rung carries its Stage 0 price.  The clearance screen's swing gate (0.04 H) and walking
    gates pass on every rung; the rolling CoP sits on its contact; the split solved; every drive ratio is at most 1 or
    the rung is marked over that limit, with the plan's rule in its verdict and flags."""
    e = LADDER["bodies"][body][speed][variant]
    pr = e["price"]
    cl = pr["clearance"]
    assert cl["gates"]["swing_clearance"] is True and cl["mid_swing_min_frac_H"] >= clearance.SWING_MIN_FRAC
    assert cl["walking_gates"] is True, cl["gates"]
    assert cl["tightest_joint"][1] >= clearance.RANGE_MARGIN_RAD  # usable joint ranges
    c = pr["contact"]
    assert c["split_failed_frames"] == 0
    assert c["single_support_cop_margin_min_mm"] > 1e3 * look.COP_RULE_M
    if e["spec"]["toe_off"]:
        assert c["rolling_frames"] > 0 and c["rolling_cop_off_contact_max_mm"] < 1e-6
    over = {(g, k) for g, v in pr["drives"].items() for k in ("cont_ratio", "peak_ratio", "speed_ratio") if v[k] > 1.0}
    listed = {(o["limit"].split()[0], o["limit"].split()[1] + "_ratio") for o in pr["over"]
              if o["limit"].split()[0] in sizing.GROUPS}
    assert over == listed and pr["within_limits"] == (not pr["over"])
    if pr["over"]:
        assert look.LIMIT_RULE in pr["verdict"] and pr["verdict"] in e["flags"]
    else:
        assert pr["verdict"] == "within the Stage 0 limits"
    assert pr["drives_at"]["bus_v"] == plant.record(body)["selection"]["bus"]  # the record's drives at its bus


@needs_ladder
def test_v0_price_reproduces_the_round_two_envelope_screen():
    """V0 at the animal pace priced here (gait.generate at r2) against the round-two envelope screen's recorded Froude
    0.05 walk (turning.generate at yaw 0): the knee's speed ratio, the binding limit, to 0.2 %."""
    for body, case in (("60-N-T-10", "narrow_carpet_60"), ("75-M-H-10", "feet_75")):
        env = json.loads((ROOT / "round2" / "refine" / case / "review" / "screening" / f"{body}_envelope.json").read_text())
        rec = env["fast_walk"]["groups"]
        ours = LADDER["bodies"][body]["fr05"]["V0"]["price"]["drives"]
        for g in ("knee", "hip_pitch"):
            assert ours[g]["actuator"] == rec[g]["actuator"]
            assert ours[g]["speed_ratio"] == pytest.approx(rec[g]["speed_ratio"], rel=2e-3), (body, g)


@needs_ladder
def test_flags_for_the_owner():
    """Findings 4 and 10: the nominal V2+ rungs (duty 0.69) carry owner decision 2; the faster rungs' F5 carries its
    reduction; a claw roll carries the hardware question; F5 at the nominal walk reaches the research's 0.30 rad."""
    for b in look.BODIES:
        for s in look.SPEEDS:
            for v, e in LADDER["bodies"][b][s].items():
                fl = " | ".join(e["flags"])
                needs_d2 = e["spec"]["swing_time"] is not None and s == "nominal"
                assert ("owner decision 2" in fl) == needs_d2, (b, s, v)
                if e["spec"]["toe_off"]:
                    reduced = e["measured"]["toe_off_at_lift_off_rad"] < 0.3 - 1e-6
                    assert ("F5 reduced" in fl) == reduced, (b, s, v)
                    assert reduced == (s != "nominal"), (b, s, v)
                    assert ("CLAW_QUESTION" in fl) == (e["measured"]["claw_roll_frames"] > 0), (b, s, v)
    assert LADDER["values"]["F3 swing_time"]["owner_decision_2"]


@needs_ladder
def test_checks_zmp_passes_and_bob_drift():
    """Finding: V1 also raises the correction passes; V0 at 12 passes against V0 at 6 moves the CoM path by well under
    a millimetre.  The bob's share of the walk's own enforced shortening is recorded per rung."""
    for b in look.BODIES:
        for s in look.SPEEDS:
            z = LADDER["checks"][b][s]["zmp_passes"]
            assert z["passes"] == [6, 12] and z["com_ref_max_diff_mm"] < 1.0 and z["root_xyz_max_diff_mm"] < 1.0
            assert z["zmp_error_final_mm"][1] <= z["zmp_error_final_mm"][0] + 1e-9
            for v in ("V1B50", "V1", "V1B100"):
                share = LADDER["bodies"][b][s][v]["measured"]["bob_share_of_enforced"]
                assert share == pytest.approx({"V1B50": 0.5, "V1": 0.75, "V1B100": 1.0}[v], rel=2e-3), (b, s, v)
    assert LADDER["bodies"]["60-N-T-10"]["nominal"]["V2"]["measured"]["bob_share_of_enforced"] == pytest.approx(0.57, abs=0.01)


@needs_ladder
def test_tail_check_confirms_the_research_reading_and_sets_f9():
    """Research Bottom line 4: today's tail (phase pi/2) raises the stance pad's single-support yaw moment above the
    no-tail walk, the opposite phase lowers it; F9's phase is the minimiser found on V4 (and on V4 with F13's stride
    where F13 applies), and the later rungs carry it."""
    doc = _record(LOOK / "tail_phase.json")
    for b in look.BODIES:
        for s in look.SPEEDS:
            r = doc["bodies"][b][s]
            v0, v4 = r["V0"], r["V4"]
            assert v0["today_adds_to_leg_yaw_momentum"] and v0["flipped_lowers_peak"], (b, s)
            assert v0["predicted_peak_nm"]["today"] > v0["no_tail"]["peak_nm"] > v0["predicted_peak_nm"]["flipped"]
            checks = [v0, v4] + ([r["V4F13"]] if look.f13_applies(b, s) else [])
            for c in checks:
                assert c["fit_max_residual_nm"] < 0.01 and c["no_tail_vs_fit_mean_nm"] < 0.01
                assert c["confirmed"]["peak_nm"] == pytest.approx(c["predicted_peak_nm"]["best"], abs=0.01)
                assert c["confirmed"]["peak_nm"] <= c["predicted_peak_nm"]["today"]
                assert c["confirmed"]["mu_req_max"] < c["runs"][f"{0.5 * math.pi:.6f}"]["mu_req_max"], (b, s)
            assert r["tail_phase_F9_rad"] == v4["best_phase_rad"]
            rungs = LADDER["bodies"][b][s]
            assert rungs["V5"]["spec"]["tail_phase"] == v4["best_phase_rad"]
            if look.f13_applies(b, s):
                assert r["tail_phase_F9_F13_rad"] == r["V4F13"]["best_phase_rad"]
                for v in ("V5F13", "V6", "V6S"):
                    assert rungs[v]["spec"]["tail_phase"] == r["V4F13"]["best_phase_rad"], (b, s, v)
            else:
                for v in ("V6", "V6S"):
                    assert rungs[v]["spec"]["tail_phase"] == v4["best_phase_rad"], (b, s, v)
        # the check's walk at today's phase is the turning screen's recorded straight run (the nominal walk)
        tsc = doc["bodies"][b]["nominal"]["V0"]["turning_screen"]
        assert abs(tsc["today_peak_rel"]) < 0.01 and abs(tsc["today_mu_req_rel"]) < 0.01, (b, tsc)


@needs_ladder
def test_body_info_records_the_claw():
    """F5's rigid-foot geometry and the hardware question are recorded per body."""
    for b in look.BODIES:
        bi = LADDER["body_info"][b]
        assert bi["H_m"] == H_BODY[b] and bi["claw_question"] == look.CLAW_QUESTION
        for s in ("left", "right"):
            cl = bi["feet"][s]["claw"]
            assert cl["geom"].endswith("claw_0") and cl["arc"]["hover_max_m"] < 3e-4
        tc = bi["toe_clearance"]
        assert tc["clear_pitch_rad"] == pytest.approx(bi["feet"]["left"]["claw"]["touch_pitch"], abs=2e-3)
        assert tc["deepest_mm"] < 0.0  # about the pad edge alone, the claw would go below the floor


_ARRAYS_HERE = LADDER is not None and all((LOOK / LADDER["bodies"][b][s][v]["file"]).exists() for b, s, v in ENTRIES)


@needs_ladder
@pytest.mark.skipif(not _ARRAYS_HERE, reason="the git-ignored arrays are not here")
def test_recorded_measures_recompute_from_the_arrays():
    """Where the git-ignored arrays are here: each is the file the ladder recorded, and its measures recompute from it
    (the ZMP margin from the saved hull, the duty and swing time from the floor contact)."""
    for b, s, v in ENTRIES:
        e = LADDER["bodies"][b][s][v]
        p = LOOK / e["file"]
        assert hashlib.sha256(p.read_bytes()).hexdigest() == e["sha256"], (b, s, v)
        with np.load(p, allow_pickle=False) as z:
            arrs = {k: z[k] for k in z.files}
        m = look.measure(arrs, dict(e["info"], look=e["spec"]), H_BODY[b], crouch=e["spec"]["crouch"])
        for k in ("zmp_margin_min_mm", "duty", "swing_time_s", "cadence_steps_s", "bob_pp_H", "toe_off_at_lift_off_rad",
                  "foot_below_floor_max_mm", "lift_peak_frac_swing"):
            assert m[k] == pytest.approx(e["measured"][k], rel=1e-9, abs=1e-12), (b, s, v, k)
        assert m["zmp_margin_min_mm"] == pytest.approx(e["zmp_margin_min_mm"], abs=1e-6)


def _regenerate(body: str, speed: str, variant: str) -> dict:
    """The variant regenerated as ``prepare`` makes it (a subprocess with one BLAS thread): its entry (sha256, measures
    and Stage 0 price recomputed)."""
    code = ("import json, sys, tempfile; from pathlib import Path; from designlab.stage2 import look; "
            f"src = look.source({body!r}); "
            f"tp = json.loads(Path('stage2/look/tail_phase.json').read_text())['bodies'][{body!r}][{speed!r}]; "
            f"spec = look.ladder({body!r}, src['H'], {speed!r}, tp['tail_phase_F9_rad'], "
            "tail_phase_f13=tp.get('tail_phase_F9_F13_rad'))"
            f"[{variant!r}]; "
            "d = tempfile.mkdtemp(); "
            f"e = look.write(src, {speed!r}, {variant!r}, spec, look.generate(src, spec), Path(d)); "
            "print(json.dumps(dict(sha256=e['sha256'], price=e['price'], flags=e['flags'])))")
    r = subprocess.run([PY, "-c", code], cwd=ROOT, env=_env(), capture_output=True, text=True, timeout=3600)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def _same_price(a, b):
    if isinstance(a, dict):
        assert set(a) == set(b)
        for k in a:
            _same_price(a[k], b[k])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            _same_price(x, y)
    elif isinstance(a, float):
        assert a == pytest.approx(b, rel=1e-9, abs=1e-12)
    else:
        assert a == b


@needs_ladder
def test_a_variant_regenerates_to_its_recorded_hash_and_price():
    e = LADDER["bodies"][BODY]["nominal"]["V4"]
    got = _regenerate(BODY, "nominal", "V4")
    assert got["sha256"] == e["sha256"]
    _same_price(got["price"], e["price"])
    assert got["flags"] == e["flags"]


@slow
@needs_ladder
@pytest.mark.parametrize("body,speed,variant", ENTRIES)
def test_every_variant_regenerates_to_its_recorded_hash(body, speed, variant):
    e = LADDER["bodies"][body][speed][variant]
    got = _regenerate(body, speed, variant)
    assert got["sha256"] == e["sha256"]
    _same_price(got["price"], e["price"])
