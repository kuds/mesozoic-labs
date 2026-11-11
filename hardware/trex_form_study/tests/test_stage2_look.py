"""Tests for the Stage 2 look generator: ``gait.GaitSpec``'s look fields (designlab/gait.py) and the look ladder
(designlab/stage2/look.py; STAGE2_PLAN.md section 4 A, research_look.md sections 3.1 and 4.1).

Run from hardware/trex_form_study:  python -m pytest tests/test_stage2_look.py -q

1. G2: with every look field at its default, Stage 1's nominal, brisk and fr05 references of both bodies regenerate bit
   for bit (each npz file's sha256 as ``stage1/runs/2026-11-08/<body>/refs.json`` records it), built as
   ``stage1.refs.prepare`` builds them (round two's accuracy) in subprocesses with one BLAS thread, as the Stage 1
   runner (with several threads the inverse dynamics' double-support QP differs in the 15th digit).
2. Each feature measured on generated walks (60-N-T-10, nominal): bob amplitude and trough time, duty from the swing
   time, lift peak, toe-off pitch at lift-off (IK honouring the pitch about the toe pads' front edge), trunk roll,
   tail and head phase, stride, the variation's bounds and seeds; the ZMP correction's convergence with them.
3. Every variant of the committed ladder (``stage2/look/look_refs.json``): IK on every frame, the ZMP inside the
   support with its recorded margin, the measured features against the spec; one variant regenerated to its recorded
   sha256 (every variant with STAGE2_SLOW=1); the tail-phase check (``tail_phase.json``) and F9's phase.
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
from designlab import envelope, gait, sizing  # noqa: E402
from designlab.stage2 import look  # noqa: E402

RUN = ROOT / "stage1" / "runs" / "2026-11-08"
LOOK = ROOT / "stage2" / "look"
BODY = "60-N-T-10"
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
    assert gait.look_active(replace(sizing.NOMINAL, bob=0.01)) and gait.look_active(replace(sizing.NOMINAL, strides=2))


def test_default_fields_take_todays_code_path(monkeypatch):
    """A walk with every look field at its default never reaches the look generator."""
    def refuse(*a, **k):
        raise AssertionError("the look path ran for today's walk")

    monkeypatch.setattr(gait, "_generate_look", refuse)
    src = _src()
    tr = gait.generate(src["M"], src["q_home"], src["H"], replace(sizing.NOMINAL, n=20, zmp_iters=0))
    assert "look" not in tr and len(tr["q"]) == 20


@pytest.fixture(scope="module")
def g2_hashes():
    """``look.g2_hashes`` of both bodies, run at once in two subprocesses with one BLAS thread each."""
    procs = {b: subprocess.Popen([PY, "-m", "designlab.stage2.look", "g2", "--body", b], cwd=ROOT, env=_env(),
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for b in look.BODIES}
    out = {}
    for b, p in procs.items():
        so, se = p.communicate(timeout=900)
        assert p.returncode == 0, se[-3000:]
        out[b] = json.loads(so.strip().splitlines()[-1])
    return out


@pytest.mark.parametrize("body", look.BODIES)
def test_g2_stage1_references_regenerate_bit_for_bit(body, g2_hashes):
    """Gate G2: nominal, brisk and fr05 of both bodies, at round two's accuracy (refs.ACCURACY), regenerate to the
    sha256 Stage 1 recorded, and every record leaf still matches."""
    rec = json.loads((RUN / body / "refs.json").read_text())
    assert rec["accuracy"] == "r2"
    for name in ("nominal", "brisk", "fr05"):
        got = g2_hashes[body][name]
        assert got["sha256"] == rec["references"][name]["sha256"], (body, name)
        assert got["matches_records"], (body, name)


# ------------------------------------------------------------------------------------------------
# Unit pieces of the look features
# ------------------------------------------------------------------------------------------------
def test_profiles_put_their_extremes_where_asked():
    s = np.linspace(0.0, 1.0, 200001)
    assert np.array_equal(gait.lift_profile(s), np.sin(np.pi * s) ** 2)  # today's lift, exactly
    for peak in (0.3, 0.375, 0.6):
        y = gait.lift_profile(s, peak)
        assert s[np.argmax(y)] == pytest.approx(peak, abs=1e-4) and y.max() == pytest.approx(1.0)
        assert y[0] == 0.0 and abs(y[-1]) < 1e-20 and np.gradient(y, s)[[0, -1]] == pytest.approx([0, 0], abs=1e-3)
    for trough in (0.45, 0.5, 0.58):
        y = gait.bob_profile(s, trough)
        assert s[np.argmax(y)] == pytest.approx(trough, abs=1e-4) and y[0] == 0.0 and y.max() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        gait.lift_profile(s, 0.2)  # the warp would not be monotone
    spec = replace(sizing.NOMINAL, bob_skew=0.125)
    assert gait.bob_trough(spec, 0.6) == pytest.approx(0.4 + 0.15) and gait.bob_trough(sizing.NOMINAL, 0.6) == 0.5


def test_toe_off_bump_peaks_at_lift_off_unless_the_stance_part_is_short():
    t = np.linspace(-1.0, 1.0, 20001)
    th, pk = gait.toe_off_bump(t, -0.3, 0.0, 0.2, 0.4)  # stance part 0.3 of 0.5: the peak at the lift-off
    assert pk == 0.0 and th.max() == pytest.approx(0.4) and t[np.argmax(th)] == pytest.approx(0.0, abs=1e-4)
    th, pk = gait.toe_off_bump(t, -0.05, 0.0, 0.2, 0.4)  # stance part 0.05 of 0.25: the peak in early swing
    assert pk == pytest.approx(0.075) and th[t <= -0.05].max() == 0.0 and th[t >= 0.2].max() == 0.0
    rate = np.gradient(th, t)
    assert np.abs(rate).max() == pytest.approx(1.875 * 0.4 / 0.125, rel=1e-3)  # rise and fall of the same length


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
    for body, H, w, T, stride, duty in (("60", 0.40468919988052376, 0.651, 0.751, 0.742, 0.567),
                                        ("75", 0.502, 0.338, 0.861, 0.686, 0.573)):
        s = look.speed_spec(H)
        d = s.describe(H)
        assert d["speed"] == pytest.approx(0.40, abs=1e-12)
        assert d["period"] == pytest.approx(T, abs=1e-3) and s.stride_rel == pytest.approx(stride, abs=1e-3)
        assert s.duty == pytest.approx(duty, abs=1e-3) and s.lift_frac == pytest.approx(0.11)
        a, b = sizing.BRISK.describe(H), envelope.FAST.describe(H)
        ww = (0.40 - a["speed"]) / (b["speed"] - a["speed"])
        assert ww == pytest.approx(w, abs=1e-3) and d["period"] == pytest.approx(a["period"] + ww * (b["period"] - a["period"]))
        assert not gait.look_active(s), body


def test_ladder_values_are_the_research_middles():
    H = 0.40468919988052376
    lad = look.ladder(BODY, H, "nominal", tail_phase=-1.8)
    v0, v6 = lad["V0"], lad["V6"]
    assert v0 is sizing.NOMINAL
    assert lad["V1"].bob == pytest.approx(0.75 * (0.6 * 0.55 / 2) ** 2 / (2 * 0.97)) == pytest.approx(0.0105, abs=2e-4)
    assert look.ladder(BODY, H, "fr05")["V1"].bob == pytest.approx(0.0195, abs=2e-4)  # research: 0.013-0.026 H
    assert lad["V2"].swing_time == pytest.approx(0.32, abs=2e-3) and lad["V2"].describe(H)["duty"] == pytest.approx(0.69, abs=0.005)
    assert look.ladder("75-M-H-10", 0.502, "nominal")["V2"].swing_time == pytest.approx(0.36, abs=5e-3)  # 0.356 s
    assert (lad["V3"].lift_peak, lad["V3"].toe_off, lad["V3"].toe_off_start, lad["V3"].toe_reach) == (0.375, 0.4, 0.825, 0.075)
    assert lad["V4"].roll_amp == pytest.approx(math.radians(1.5)) and lad["V4L"].roll_amp == -lad["V4"].roll_amp
    assert (lad["V5"].tail_amp, lad["V5"].tail_phase, lad["V5"].head_mode) == (0.125, -1.8, "stabilise")
    assert (lad["V5"].head_gain, lad["V5"].head_settle, lad["V5"].settle_lag) == (0.9, 0.015, 0.075)
    assert (v6.phase_jitter, v6.asym, v6.var_band, v6.stride_rel) == (0.015, 0.025, (0.05, 0.2), 0.55)
    assert v6.strides * v6.describe(H)["period"] >= look.CLIP_S > (v6.strides - 1) * v6.describe(H)["period"]
    assert lad["V6S"].zmp_inset_y == 0.010
    assert look.ladder("75-M-H-10", 0.502, "fr05", tail_phase=0.0)["V6"].stride_rel == 0.875  # F13: 75 cm only
    assert look.ladder(BODY, H, "fr05", tail_phase=0.0)["V6"].stride_rel == 0.80
    for v in look.VARIANTS[1:]:  # one feature group per rung
        prev = lad[look.VARIANTS[look.VARIANTS.index(v) - 1]]
        changed = {k for k, x in asdict(lad[v]).items() if asdict(prev)[k] != x}
        assert changed, v
        assert changed <= {"bob", "bob_skew", "swing_time", "lift_peak", "toe_off", "toe_reach", "roll_amp", "tail_amp",
                           "tail_phase", "head_mode", "head_settle", "phase_jitter", "asym", "gaze_var", "tail_var",
                           "noise_seed", "strides", "stride_rel", "zmp_iters"}
    assert lad["V1"].zmp_iters == look.LOOK_ZMP_ITERS == 12 and v0.zmp_iters == 6


# ------------------------------------------------------------------------------------------------
# 2. Features measured on generated walks (60-N-T-10, nominal)
# ------------------------------------------------------------------------------------------------
_SRC = {}


def _src(body: str = BODY) -> dict:
    if body not in _SRC:
        _SRC[body] = look.source(body)
    return _SRC[body]


def _tail_phase(body: str, speed: str):
    doc = _record(LOOK / "tail_phase.json")
    try:
        return doc["bodies"][body][speed]["tail_phase_F9_rad"]
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


def test_bob_amplitude_and_trough(walks):
    H = _src()["H"]
    for k in ("V1", "V3", "V5"):
        w = walks[k]
        assert w["m"]["bob_pp_H"] == pytest.approx(w["spec"].bob, rel=0.01), k  # measured at the hip-roll axes
        assert w["m"]["bob_trough_frac_stance"] == pytest.approx(0.125, abs=0.01), k
        assert np.ptp(w["g"]["q"][:, 2]) == pytest.approx(w["spec"].bob * H, rel=2e-3), k  # the pelvis carries it
    v1 = walks["V1"]
    phase, z = v1["a"]["phase"], v1["a"]["hip_mid"][:, 2]
    duty = v1["g"]["info"]["duty"]
    mid = np.mod(np.array([0.5 - duty / 2, 1 - duty / 2]), 1.0)  # mid single support: today's height
    k = [int(np.argmin(np.abs(phase - p))) for p in mid]
    assert z[k] == pytest.approx(z.max(), abs=2e-5)
    assert walks["V0"]["m"]["bob_pp_H"] < 1e-6 and walks["V0"]["m"]["mechanical_tells"]["constant_hip_height"]


def test_duty_follows_the_swing_time(walks):
    w = walks["V3"]
    info, s = w["g"]["info"], w["spec"]
    T = info["period"]
    assert info["duty"] == pytest.approx(min(max(1 - s.swing_time / T, 0.55), 0.72)) == pytest.approx(0.6865, abs=1e-3)
    assert w["m"]["swing_time_s"] == pytest.approx(s.swing_time, abs=1e-9)
    n = len(w["a"]["t"])
    assert w["m"]["duty"] == pytest.approx([info["duty"]] * 2, abs=1.0 / n)
    assert walks["V1"]["g"]["info"]["duty"] == 0.6  # V1 keeps the fixed duty


def test_lift_peak_timing(walks):
    assert walks["V3"]["m"]["lift_peak_frac_swing"] == pytest.approx(0.375, abs=0.005)
    assert walks["V1"]["m"]["lift_peak_frac_swing"] == pytest.approx(0.5, abs=0.005)
    a = walks["V3"]["a"]
    sw = a["swing"]
    assert np.max(a["pad_floor_z"][sw]) == pytest.approx(walks["V3"]["g"]["info"]["lift"], rel=5e-3)


def test_toe_off_pitches_about_the_toe_pads_front_edge(walks):
    """F5: the foot pitches heel-up about its frontmost toe pad's front bottom edge, which stays on the floor and in
    place while the foot rolls; IK honours the pitch; the roll waits for the ZMP; the toes rise before a flat landing."""
    w = walks["V3"]
    a, g, s = w["a"], w["g"], w["spec"]
    assert np.abs(a["foot_pitch_solved"] - a["foot_pitch"]).max() < 1e-4  # IK honours the target pitch
    assert w["m"]["toe_off_peak_rad"] == pytest.approx(s.toe_off, abs=1e-4)
    rolls = [r for side in ("left", "right") for r in g["look"]["rolls"][side] if 0 <= r["t_lift_off"] < g["t"][-1]]
    assert w["m"]["toe_off_at_lift_off_rad"] == pytest.approx(np.mean([r["at_lift_off"] for r in rolls]), abs=2e-3)
    assert w["m"]["toe_off_at_lift_off_rad"] > 0.3  # the nominal walk's long double support: most of it by lift-off
    assert all(r["start_frac_stance"] >= s.toe_off_start - 1e-12 for r in rolls)
    rol = a["rolling"]
    assert rol.any() and w["m"]["rolling_pivot_height_max_mm"] < 0.1  # the pivot edge stays on the floor ...
    for j in range(2):
        runs = np.split(np.flatnonzero(rol[:, j]), np.flatnonzero(np.diff(np.flatnonzero(rol[:, j])) > 1) + 1)
        for r in runs:
            xy = a["toe_pivot"][r, j, :2]
            assert np.ptp(xy, axis=0).max() < 1e-4  # ... and in place (m)
            assert a["foot_pitch"][r[-1], j] > a["foot_pitch"][r[0], j]  # rising: the heel lifts first
    assert np.min(a["zmp_ref_margin"][rol.any(axis=1)]) >= s.toe_off_margin - 1e-6  # the ZMP is inside the roll's support
    assert np.min(a["zmp_margin"]) > 0.0
    assert w["m"]["foot_pitch_min_rad"] == pytest.approx(-s.toe_reach, abs=1e-3)  # toes-up reach before landing
    for j in range(2):
        td = a["touchdown_t"][j][np.isfinite(a["touchdown_t"][j])]
        assert np.abs(np.interp(td, a["t"], a["foot_pitch"][:, j])).max() < 1e-5  # flat landing (between frames)
    assert walks["V1"]["m"]["toe_off_peak_rad"] < 1e-9 and walks["V1"]["m"]["mechanical_tells"]["flat_feet_at_lift_off"]


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


def test_stride_and_cadence(walks):
    for k in ("V0", "V5", "V6w"):
        w = walks[k]
        assert w["m"]["stride_H"] == pytest.approx(w["spec"].stride_rel, rel=1e-4), k  # touchdowns between frames
        assert w["m"]["cadence_steps_s"] == pytest.approx(2 / w["g"]["info"]["period"], rel=1e-9), k


def test_variation_window_is_bounded_seeded_and_periodic(walks):
    w = walks["V6w"]
    s, g, a = w["spec"], w["g"], w["a"]
    e = np.array(g["info"]["step_deviations"])
    bound = s.phase_jitter + s.asym / 2
    assert np.max(np.abs(e)) == pytest.approx(bound, abs=0.004) and np.max(np.abs(e)) <= bound + 1e-12
    assert w["m"]["step_time_dev_max"] == pytest.approx(np.max(np.abs(e)), abs=1e-4)  # measured from the events
    assert w["m"]["step_time_dev_std"] > 0.005 and not w["m"]["mechanical_tells"]["identical_steps"]
    rng = np.random.default_rng(np.random.SeedSequence(s.noise_seed).spawn(3)[0])
    assert np.array_equal(gait.step_deviations(s, rng), e)  # seeded
    T, K = g["info"]["period"], s.strides
    assert len(a["t"]) == K * s.n and a["phase"][0] == 0.0 and a["phase"][-1] < K
    assert g["info"]["window_s"] == pytest.approx(K * T)
    assert np.array_equal(a["stride_motion"], [0.0, 0.0, 0.0, K * g["info"]["stride"], 0.0])
    assert g["ik_ok"] and np.min(a["zmp_margin"]) > 0


def test_zmp_correction_converges_with_the_look_features(walks):
    """The multibody-ZMP correction passes still converge with the bob, the toe-off and the roll: the error falls
    pass by pass to about today's millimetre."""
    for k in ("V1", "V3", "V5"):
        h = walks[k]["g"]["info"]["zmp_error_history_m"]
        assert h[-1] < 2e-3 and h[-1] < 0.1 * h[0], (k, h)
        assert all(b < a for a, b in zip(h, h[1:], strict=False)), (k, h)


def test_markers_and_arrays_for_the_screen_and_renders(walks):
    a, g = walks["V5"]["a"], walks["V5"]["g"]
    n = len(a["t"])
    assert a["points"].shape == (n, len(look.POINTS), 3) and list(a["point_names"]) == list(look.POINTS)
    i = list(look.POINTS)
    assert np.all(a["points"][:, i.index("snout"), 0] > a["points"][:, i.index("eye"), 0])
    assert np.all(a["points"][:, i.index("tail_tip"), 0] < a["points"][:, i.index("tail_mid"), 0])
    assert a["support_hull"].shape[0] == n and np.array_equal(a["support"], ~a["swing"])
    tr = look.trajectory(a, g["info"])
    with look.accuracy("r2"):  # the exact unwrap, as the arrays' qd
        idr = gait.inverse_dynamics(_src()["M"], tr)  # the screen's call on the reference
    assert np.array_equal(idr["qd"], a["qd"]) and np.isfinite(idr["tau"]).all()


# ------------------------------------------------------------------------------------------------
# 3. The committed ladder
# ------------------------------------------------------------------------------------------------
LADDER = _record(LOOK / "look_refs.json")
ENTRIES = [] if LADDER is None else [(b, s, v) for b, d in LADDER["bodies"].items() for s, e in d.items() for v in e]
needs_ladder = pytest.mark.skipif(LADDER is None, reason="stage2/look/look_refs.json not generated")


@needs_ladder
def test_the_ladder_has_every_rung():
    for b in look.BODIES:
        for s in look.SPEEDS:
            assert set(LADDER["bodies"][b][s]) == set(look.VARIANTS + look.EXTRAS), (b, s)


@needs_ladder
@pytest.mark.parametrize("body,speed,variant", ENTRIES)
def test_every_variant_solves_and_keeps_the_zmp_inside_its_support(body, speed, variant):
    e = LADDER["bodies"][body][speed][variant]
    m, spec = e["measured"], e["spec"]
    assert e["ik_ok"] and e["ik_fail_frames"] == 0
    assert e["zmp_margin_min_mm"] > 0.0 and m["zmp_margin_min_mm"] == pytest.approx(e["zmp_margin_min_mm"])
    assert e["zmp_error_final_mm"] < 2.5
    H = 0.40468919988052376 if body.startswith("60") else 0.502
    n = e["frames"]
    assert m["bob_pp_H"] == pytest.approx(spec["bob"], abs=2e-4)
    if spec["bob_skew"] is not None:
        assert m["bob_trough_frac_stance"] == pytest.approx(spec["bob_skew"], abs=0.02)
    assert m["duty"] == pytest.approx([e["info"]["duty"]] * 2, abs=0.02 if spec["asym"] else 1.0 / spec["n"])
    assert m["lift_peak_frac_swing"] == pytest.approx(spec["lift_peak"], abs=0.01)
    assert m["toe_off_peak_rad"] == pytest.approx(spec["toe_off"], abs=1e-3)
    assert m["trunk_roll_amp_rad"] == pytest.approx(abs(spec["roll_amp"]), rel=1e-3, abs=1e-9)
    if spec["roll_amp"]:
        assert m["trunk_roll_toward_stance"] == (spec["roll_amp"] > 0)
    assert math.cos(m["tail_phase_rad"] - spec["tail_phase"]) == pytest.approx(1.0, abs=1e-4)
    assert m["stride_H"] == pytest.approx(spec["stride_rel"], rel=1e-4)
    assert m["step_time_dev_max"] <= spec["phase_jitter"] + spec["asym"] / 2 + 1e-4
    assert n == spec["n"] * spec["strides"] and e["info"]["window_s"] >= (look.CLIP_S if spec["strides"] > 1 else 0)
    assert e["info"]["speed"] == pytest.approx({"nominal": math.sqrt(0.012 * gait.G * H), "v040": 0.40,
                                                "fr05": math.sqrt(0.05 * gait.G * H)}[speed])


@needs_ladder
def test_tail_check_confirms_the_research_reading_and_sets_f9():
    """Research Bottom line 4: today's tail (phase pi/2) raises the stance pad's single-support yaw moment above the
    no-tail walk, the opposite phase lowers it; F9's phase is the minimiser found on V4, and V5 and V6 carry it."""
    doc = _record(LOOK / "tail_phase.json")
    for b in look.BODIES:
        for s in look.SPEEDS:
            r = doc["bodies"][b][s]
            v0, v4 = r["V0"], r["V4"]
            assert v0["today_adds_to_leg_yaw_momentum"] and v0["flipped_lowers_peak"], (b, s)
            assert v0["predicted_peak_nm"]["today"] > v0["no_tail"]["peak_nm"] > v0["predicted_peak_nm"]["flipped"]
            for c in (v0, v4):
                assert c["fit_max_residual_nm"] < 0.01 and c["no_tail_vs_fit_mean_nm"] < 0.01
                assert c["confirmed"]["peak_nm"] == pytest.approx(c["predicted_peak_nm"]["best"], abs=0.01)
                assert c["confirmed"]["peak_nm"] <= c["predicted_peak_nm"]["today"]
            assert r["tail_phase_F9_rad"] == v4["best_phase_rad"]
            for v in ("V5", "V6", "V6S"):
                assert LADDER["bodies"][b][s][v]["spec"]["tail_phase"] == v4["best_phase_rad"]


def _regenerate(body: str, speed: str, variant: str) -> str:
    """The variant regenerated as ``prepare`` makes it (a subprocess with one BLAS thread), its npz sha256."""
    code = ("import json, sys, tempfile; from pathlib import Path; from designlab.stage2 import look; "
            f"src = look.source({body!r}); "
            f"tp = json.loads(Path('stage2/look/tail_phase.json').read_text())['bodies'][{body!r}][{speed!r}]['tail_phase_F9_rad']; "
            f"spec = look.ladder({body!r}, src['H'], {speed!r}, tp)[{variant!r}]; "
            "d = tempfile.mkdtemp(); "
            f"e = look.write(src, {speed!r}, {variant!r}, spec, look.generate(src, spec), Path(d)); print(e['sha256'])")
    r = subprocess.run([PY, "-c", code], cwd=ROOT, env=_env(), capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-3000:]
    return r.stdout.strip().splitlines()[-1]


@needs_ladder
def test_a_variant_regenerates_to_its_recorded_hash():
    e = LADDER["bodies"][BODY]["nominal"]["V4"]
    assert _regenerate(BODY, "nominal", "V4") == e["sha256"]


@slow
@needs_ladder
@pytest.mark.parametrize("body,speed,variant", ENTRIES)
def test_every_variant_regenerates_to_its_recorded_hash(body, speed, variant):
    assert _regenerate(body, speed, variant) == LADDER["bodies"][body][speed][variant]["sha256"]


@needs_ladder
def test_recorded_arrays_match_their_hashes():
    """Where the git-ignored arrays are here, each is the file the ladder recorded."""
    for b, s, v in ENTRIES:
        e = LADDER["bodies"][b][s][v]
        p = LOOK / e["file"]
        if p.exists():
            assert hashlib.sha256(p.read_bytes()).hexdigest() == e["sha256"], (b, s, v)
