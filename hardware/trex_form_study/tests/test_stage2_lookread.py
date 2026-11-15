"""Tests for the look reading (designlab/stage2/lookread.py; STAGE2_TRACKB_PLAN rules U1, U2, U3, U5 and gate Z2 (b),
(c)).

Run from hardware/trex_form_study (the fit's venv):
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -p no:cacheprovider tests/test_stage2_lookread.py

Fast tests on synthetic series (no model): each band on both sides of its limit, segmentation by the own events,
"not read", too few steps, the window of a bar item.  Tests on a model (kinematics only, no physics and no episode):
the look ladder's V6 rungs played from their own arrays against ``stage2/look/metrics.json`` within gate Z2 (b)'s
limits, the playback through ``bar._Played`` (a rung written as a reference file under pytest's temporary directory,
and Stage 1's own run directory with a real track), the negative controls of gate Z2 (c), and an ``-X`` body.  They
need the round-two model files, the look ladder's arrays and Stage 1's arrays (git-ignored;
``python tools/restore_inputs.py``) and skip without them.  Nothing is written outside ``tmp_path``.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from designlab.stage2 import lookread, motion, readings  # noqa: E402

LOOK = ROOT / "stage2" / "look"
RUN = ROOT / "stage1" / "runs" / "2026-11-08"
BODIES = ("75-M-H-10", "60-N-T-10")
SPEEDS = ("nominal", "v040", "fr05")
RATE, DT, EVERY = 1.05, 0.002, 10


# ------------------------------------------------------------------------------------------------
# A synthetic series (no model)
# ------------------------------------------------------------------------------------------------
def synth(bob=0.008, head_deg=0.5, tail_amp=0.12, tail_phase=-0.28, toe=0.06, T=0.82, shift=0.0, seconds=14.0,
          duty=0.6, last_lift_off=None) -> dict:
    """A walk as ``lookread.series`` would give it: 50 Hz samples over ``seconds``; the left foot lifts at ``shift +
    i T`` and the right half a stride later, each landing ``(1 - duty) T`` after; two bobs of ``bob`` per stride, a
    head pitch of about ``head_deg`` RMS, the tail ``tail_amp sin(2 pi (phase + tail_phase))`` on the phase that is
    0 at the left lift-offs, and each foot's pitch a bump of height ``toe`` at its lift-offs.  ``shift`` moves the
    whole motion (signals and events) on the fixed sample grid.  No event after ``last_lift_off`` (the robot then
    stands)."""
    t = np.arange(int(round(seconds / 0.02)) + 1) * 0.02
    phase = (t - shift) / T
    i = np.arange(-2, int(seconds / T) + 3)
    lo = [shift + T * i, shift + T * (i + 0.5)]
    td = [x + (1.0 - duty) * T for x in lo]
    end = seconds if last_lift_off is None else last_lift_off

    def keep(x, last):
        return x[(x >= 0.0) & (x <= last)]

    pitch = np.column_stack([sum(toe * np.exp(-0.5 * ((t - x) / 0.06) ** 2) for x in lo[j]) for j in (0, 1)])
    return dict(t50=t, hip_mid_z=0.45 + 0.5 * bob * np.cos(4 * np.pi * (phase - 0.1)),
                head_pitch_world=0.05 + math.radians(head_deg) * math.sqrt(2.0) * np.sin(4 * np.pi * phase + 0.7),
                tail=0.01 + tail_amp * np.sin(2 * np.pi * (phase + tail_phase)), foot_pitch=pitch,
                lift_offs=[keep(x, end) for x in lo], touchdowns=[keep(x, end + (1.0 - duty) * T) for x in td])


WIN = (3.0, 13.0)
#: A window of six steps and six lift-offs of ``synth()`` (a step lasts 0.41 s).
SIX = (3.0, 3.0 + 6.6 * 0.41)


def read(ser, win=WIN):
    return lookread.reading(ser, *win)


def test_constants_are_the_plans_and_stage1s():
    from designlab.stage1 import control, panels, tune

    assert lookread.BANDS is motion.BANDS
    assert lookread.BANDS == dict(bob_rel=0.30, head_world_pitch_rms_deg=1.5, tail_phase_cycles=0.1, toe_off_rel=0.70)
    assert lookread.MIN_STEPS == 4 and lookread.NOT_READ_RAD == 0.02
    assert lookread.G == control.G and lookread.RATE == panels.RATE == 1.05
    assert lookread.WALK_WINDOW_S == tune.WALK_WINDOW_S == 10.0 and lookread.TURN_WINDOW_S == readings.HEADING_HOLD_S == 3.0
    assert lookread.POSE_S == tune.LEGS_EVERY * control.FEEDBACK_EVERY * 0.002
    assert set(lookread.RULES) == {"measures", "window", "bands", "reference"}
    assert all("STAGE2_TRACKB_PLAN rule U" in v for v in lookread.RULES.values())


def test_the_module_does_not_import_the_bar_and_names_its_own_file():
    code = ("import sys; import designlab.stage2.lookread as m; "
            "assert 'designlab.stage2.bar' not in sys.modules and 'designlab.stage1.control' not in sys.modules; "
            "print(list(m.sources()))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "['designlab/stage2/lookread.py']"
    import hashlib

    src = lookread.sources()
    assert src == {"designlab/stage2/lookread.py": hashlib.sha256(Path(lookread.__file__).read_bytes()).hexdigest()}


def test_the_four_numbers_of_a_synthetic_walk():
    r = read(synth())
    assert r["window"] == [3.0, 13.0] and r["samples"] == 501
    assert r["steps"] == 23 and r["lift_offs"] == 24 and r["tail"]["strides"] == 11
    assert r["bob_pp_m"] == pytest.approx(0.008, rel=0.02)  # (the range of 50 Hz samples of a 2.4 Hz bob)
    assert r["head_world_pitch_rms_deg"] == pytest.approx(0.5, rel=0.01)
    assert r["tail"]["amp_rad"] == pytest.approx(0.12, abs=1e-9) and r["tail"]["phase_cycles"] == pytest.approx(-0.28, abs=1e-9)
    assert r["toe_off_rad"] == pytest.approx(0.06, rel=0.02)
    # only what lies inside the window is read: the series cut to the window gives the same reading
    s = synth()
    sel = (s["t50"] >= 3.0 - 1e-9) & (s["t50"] <= 13.0 + 1e-9)
    cut = dict(t50=s["t50"][sel], hip_mid_z=s["hip_mid_z"][sel], head_pitch_world=s["head_pitch_world"][sel],
               tail=s["tail"][sel], foot_pitch=s["foot_pitch"][sel],
               lift_offs=[x[(x >= 3.0) & (x <= 13.0)] for x in s["lift_offs"]],
               touchdowns=[x[(x >= 3.0) & (x <= 13.0)] for x in s["touchdowns"]])
    assert read(cut) == r


def test_the_own_phase_is_zero_at_each_left_lift_off_and_linear_between():
    L = np.array([1.0, 2.0, 4.0])
    phi = lookread.own_phase(np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 4.5]), L)
    assert np.isnan(phi[0]) and np.isnan(phi[-1])
    assert phi[1:-1] == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0])
    assert np.isnan(lookread.own_phase(np.array([1.0, 2.0]), np.array([1.5]))).all()
    assert lookread.wrap_cycles(0.55) == pytest.approx(-0.45) and lookread.wrap_cycles(-0.91) == pytest.approx(0.09)
    assert lookread.wrap_cycles(0.5) == -0.5 and lookread.wrap_cycles(0.0) == 0.0


@pytest.mark.parametrize("k, ok", [(0.69, False), (0.71, True), (1.0, True), (1.29, True), (1.31, False)])
def test_the_bob_band_on_both_sides(k, ok):
    c = lookread.compare(read(synth(bob=k * 0.008)), read(synth()))
    assert c["bob_ratio"] == pytest.approx(k, abs=1e-9)
    assert c["terms"] == dict(bob=ok, head=True, tail=True, toe_off=True) and c["ok"] is ok
    assert c["not_read"] == [] and c["too_few_steps"] is False


@pytest.mark.parametrize("deg, ok", [(1.49, True), (1.51, False)])
def test_the_head_band_is_absolute_on_both_sides(deg, ok):
    s = synth()
    mean = s["head_pitch_world"].mean()
    r0 = read(s)["head_world_pitch_rms_deg"]
    own = read(dict(s, head_pitch_world=mean + (s["head_pitch_world"] - mean) * deg / r0))
    assert own["head_world_pitch_rms_deg"] == pytest.approx(deg, abs=1e-6)
    for ref_head in (0.1, 0.5, 3.0):  # not relative: the reference's own head pitch does not enter
        c = lookread.compare(own, read(synth(head_deg=ref_head)))
        assert c["terms"] == dict(bob=True, head=ok, tail=True, toe_off=True) and c["ok"] is ok
        assert c["head_world_pitch_rms_deg"] == own["head_world_pitch_rms_deg"]


@pytest.mark.parametrize("ref_phase", [-0.28, 0.46, -0.46])
@pytest.mark.parametrize("d, ok", [(0.09, True), (0.11, False), (-0.09, True), (-0.11, False)])
def test_the_tail_band_on_both_sides_and_across_the_wrap(ref_phase, d, ok):
    own, ref = read(synth(tail_phase=ref_phase + d)), read(synth(tail_phase=ref_phase))
    assert -0.5 < own["tail"]["phase_cycles"] <= 0.5
    c = lookread.compare(own, ref)
    assert c["tail_phase_diff_cycles"] == pytest.approx(d, abs=1e-9)
    assert c["terms"] == dict(bob=True, head=True, tail=ok, toe_off=True) and c["ok"] is ok
    if ref_phase == 0.46 and d > 0:  # 0.55 and 0.57 cycle are read as -0.45 and -0.43
        assert own["tail"]["phase_cycles"] < -0.4 < 0.4 < ref["tail"]["phase_cycles"]


@pytest.mark.parametrize("k, ok", [(0.69, False), (0.71, True), (1.0, True), (1.6, True)])
def test_the_toe_off_band_on_both_sides(k, ok):
    c = lookread.compare(read(synth(toe=k * 0.06)), read(synth()))
    assert c["toe_off_ratio"] == pytest.approx(k, abs=1e-9)
    assert c["terms"] == dict(bob=True, head=True, tail=True, toe_off=ok) and c["ok"] is ok


def test_segmentation_by_the_own_events_ignores_a_10ms_shift_of_the_whole_motion():
    a, b = read(synth()), read(synth(shift=0.010))
    assert a["steps"] == b["steps"] and a["lift_offs"] == b["lift_offs"]
    assert b["bob_pp_m"] == pytest.approx(a["bob_pp_m"], rel=0.02)
    assert b["head_world_pitch_rms_deg"] == pytest.approx(a["head_world_pitch_rms_deg"], rel=0.01)
    assert b["tail"]["phase_cycles"] == pytest.approx(a["tail"]["phase_cycles"], abs=1e-9)
    assert b["tail"]["amp_rad"] == pytest.approx(a["tail"]["amp_rad"], abs=1e-9)
    assert b["toe_off_rad"] == pytest.approx(a["toe_off_rad"], rel=0.02)
    c = lookread.compare(b, a)
    assert c["ok"] and abs(c["tail_phase_diff_cycles"]) < 1e-9
    # the same tail read on the unshifted walk's lift-offs (a clock that is not the robot's own) is 0.012 cycle off
    s = synth(shift=0.010)
    other = read(dict(s, lift_offs=synth()["lift_offs"]))
    assert other["tail"]["phase_cycles"] - a["tail"]["phase_cycles"] == pytest.approx(-0.010 / 0.82, abs=1e-6)


def test_terms_under_the_threshold_are_not_read_and_pass_labelled():
    # the tail: a reference amplitude under 0.02 rad
    flat = read(synth(tail_amp=0.019))
    c = lookread.compare(read(synth(tail_phase=0.1)), flat)
    assert c["not_read"] == ["tail"] and c["terms"]["tail"] is True and c["ok"] is True
    assert abs(c["tail_phase_diff_cycles"]) > 0.3  # (recorded all the same)
    assert c["reference"]["tail_amp_side"] == "under 0.02 rad" and c["reference"]["tail_amp_rad"] == pytest.approx(0.019)
    read_ = lookread.compare(read(synth(tail_phase=0.1)), read(synth(tail_amp=0.021)))
    assert read_["not_read"] == [] and read_["terms"]["tail"] is False
    assert read_["reference"]["tail_amp_side"] == "at or over 0.02 rad"
    # the toe-off: a reference under 0.02 rad (a set without F5)
    ref = read(synth(toe=0.019))
    assert ref["toe_off_rad"] < 0.02
    c = lookread.compare(read(synth(toe=0.0)), ref)
    assert c["not_read"] == ["toe_off"] and c["terms"]["toe_off"] is True and c["ok"] is True
    assert c["reference"]["toe_off_side"] == "under 0.02 rad" and c["reference"]["toe_off_rad"] == ref["toe_off_rad"]
    ref = read(synth(toe=0.021))
    assert ref["toe_off_rad"] >= 0.02
    c = lookread.compare(read(synth(toe=0.0)), ref)
    assert c["not_read"] == [] and c["terms"]["toe_off"] is False and c["reference"]["toe_off_side"] == "at or over 0.02 rad"
    # both at once, flat feet in the reference exactly
    both = lookread.compare(read(synth()), read(synth(toe=0.0, tail_amp=0.0)))
    assert both["not_read"] == ["tail", "toe_off"] and both["ok"] and both["toe_off_ratio"] is None
    # the threshold itself is read (compared without slack)
    exact = dict(read(synth()), toe_off_rad=0.02, tail=dict(phase_cycles=-0.28, amp_rad=0.02, strides=11))
    assert lookread.compare(read(synth()), exact)["not_read"] == []


def test_too_few_own_steps_fail_bob_tail_and_toe_off_and_are_no_error():
    ref = read(synth(), SIX)
    assert ref["steps"] == 6 and ref["lift_offs"] == 6
    stood = synth(last_lift_off=4.5)  # the robot steps to 4.5 s and stands from then on
    own = read(stood, SIX)
    assert own["steps"] == 3 and own["lift_offs"] == 3
    c = lookread.compare(own, ref)
    assert c["too_few_steps"] is True and c["ok"] is False
    assert c["terms"] == dict(bob=False, head=True, tail=False, toe_off=False)  # the head is still read
    assert c["bob_ratio"] == pytest.approx(1.0, rel=0.03)  # (the numbers of the three steps are recorded)
    assert c["steps"] == dict(own=3, reference=6, needed=4) and c["lift_offs"] == dict(own=3, reference=6, needed=4)
    # the head term is its own: over 1.5 deg it fails beside them
    loud = dict(stood, head_pitch_world=4.0 * stood["head_pitch_world"])
    assert lookread.compare(read(loud, SIX), ref)["terms"]["head"] is False
    # no step at all: nothing can be computed, and it is still no error
    none = read(synth(last_lift_off=1.0))
    assert none["steps"] == 0 and none["lift_offs"] == 0 and none["bob_pp_m"] is None and none["toe_off_rad"] is None
    assert none["tail"] == dict(phase_cycles=None, amp_rad=None, strides=0)
    c = lookread.compare(none, read(synth()))
    assert c["too_few_steps"] is True and c["terms"] == dict(bob=False, head=True, tail=False, toe_off=False)
    assert c["bob_ratio"] is None and c["tail_phase_diff_cycles"] is None and c["toe_off_ratio"] is None
    # too few lift-offs alone (4 steps, 3 lift-offs) is too few as well; 4 and 4 are enough
    four = read(synth(), (3.0, 3.0 + 4.6 * 0.41))
    assert four["steps"] == 4 and four["lift_offs"] == 4
    assert lookread.compare(four, ref)["too_few_steps"] is False and lookread.compare(four, ref)["ok"] is True
    assert lookread.compare(dict(four, lift_offs=3), ref)["too_few_steps"] is True
    # a term the reference leaves unread is listed and is failed all the same when the robot has too few steps
    c = lookread.compare(own, dict(ref, toe_off_rad=0.01))
    assert c["not_read"] == ["toe_off"] and c["terms"]["toe_off"] is False and c["too_few_steps"] is True


def test_a_reference_with_too_few_steps_is_a_definition_fault():
    short = read(synth(), (3.0, 3.0 + 3.6 * 0.41))
    assert short["steps"] == 3
    with pytest.raises(lookread.LookError, match="reference's own window holds 3 steps"):
        lookread.compare(read(synth()), short)
    with pytest.raises(lookread.LookError, match="rule U5"):
        lookread.compare(short, short)
    with pytest.raises(lookread.LookError, match="3 lift-offs"):
        lookread.compare(read(synth()), dict(read(synth()), lift_offs=3))


def test_a_missing_own_value_fails_its_term():
    ref, own = read(synth()), read(synth())
    assert lookread.compare(dict(own, bob_pp_m=None), ref)["terms"] == dict(bob=False, head=True, tail=True, toe_off=True)
    assert lookread.compare(dict(own, head_world_pitch_rms_deg=float("nan")), ref)["terms"]["head"] is False
    assert lookread.compare(dict(own, tail=dict(phase_cycles=None, amp_rad=None)), ref)["terms"]["tail"] is False
    assert lookread.compare(dict(own, toe_off_rad=None), ref)["terms"]["toe_off"] is False
    # a reference without a bob has no ratio: the term fails (the plan names no "not read" for the bob)
    c = lookread.compare(own, dict(ref, bob_pp_m=0.0))
    assert c["bob_ratio"] is None and c["terms"]["bob"] is False


# ------------------------------------------------------------------------------------------------
# The window of a bar item (attributes only)
# ------------------------------------------------------------------------------------------------
def _rec(t_end, t_ref_rate=1.05, dt=0.002):
    t = np.arange(int(round(t_end / 0.01)) + 1) * 0.01

    def sim_time(t_ref):
        return None if t_ref > t_ref_rate * t[-1] + 1e-12 else float(t_ref / t_ref_rate)

    return types.SimpleNamespace(t=t, rows=len(t), dt=dt, sim_time=sim_time)


def _built(judge, seconds=20.0, command=None, at_s=None, cut_from=None):
    resolved = dict(at_s=at_s)
    if cut_from is not None:
        resolved["cut_from_s"] = cut_from
    return types.SimpleNamespace(cond=types.SimpleNamespace(judge=judge), seconds=seconds, resolved=resolved,
                                 command=None if command is None else dict(ref_s=command))


def test_the_window_of_a_walk_item_is_its_judged_last_10s():
    assert lookread.window(_built("bar_walk"), _rec(19.99)) == pytest.approx((9.99, 19.99))
    assert lookread.window(_built("bar_walk"), _rec(19.98)) == pytest.approx((9.98, 19.98))  # (the bar's slack: two updates)
    assert lookread.window(_built("bar_walk"), _rec(19.97)) is None  # the trace ends before the planned end (a fall)
    assert lookread.window(_built("bar_walk", seconds=12.0, cut_from=20.0), _rec(11.99)) is None  # a cut episode
    assert lookread.window(_built("bar_walk"), types.SimpleNamespace(t=np.zeros(0), rows=0, dt=0.002)) is None


def test_the_window_of_a_turn_item_is_the_3s_after_the_judged_time():
    b = _built("bar_turn", command=8.4, at_s=2.0)
    assert lookread.window(b, _rec(13.2)) == pytest.approx((10.0, 13.0))
    assert lookread.window(b, _rec(13.0)) == pytest.approx((10.0, 13.0))
    assert lookread.window(b, _rec(12.99)) is None  # the trace does not span the window
    assert lookread.window(b, _rec(7.0)) is None  # the reference clock never reached the command
    slow = lookread.window(b, _rec(14.0, t_ref_rate=1.0))  # a slower clock: the command comes later
    assert slow == pytest.approx((10.4, 13.4))
    assert lookread.window(_built("bar_turn", command=8.4, at_s=math.pi / 2 / 0.35), _rec(20.0)) == pytest.approx(
        (8.0 + math.pi / 0.7, 11.0 + math.pi / 0.7))


def test_nudge_and_stop_items_and_stock_conditions_have_no_window():
    for judge in ("bar_nudge", "bar_stop", "walk", None):
        assert lookread.window(_built(judge, command=8.4, at_s=2.0), _rec(19.99)) is None
    assert lookread.window(types.SimpleNamespace(cond=types.SimpleNamespace()), _rec(19.99)) is None


# ------------------------------------------------------------------------------------------------
# On a model: kinematics only (no physics, no episode)
# ------------------------------------------------------------------------------------------------
def _plant_module():
    from designlab.stage1 import plant

    return plant


def _have_model(body: str) -> bool:
    try:
        plant = _plant_module()
        d, cid = plant.case_dir(body), plant.candidate_id(body)
        return (d / "models" / f"{cid}.xml").exists() and (d / "models" / f"{cid}.json").exists()
    except Exception:  # noqa: BLE001  (no model record: the same skip)
        return False


def _rung(body, speed, variant="V6") -> Path:
    return LOOK / "refs" / body / speed / f"{variant}.npz"


HAVE = {b: _have_model(b) and all(_rung(b, s).exists() for s in SPEEDS) for b in BODIES}
needs = {b: pytest.mark.skipif(not HAVE[b], reason=(
    f"needs the round-two model of {b} and the look ladder's arrays (python tools/restore_inputs.py --group "
    "picks-models look)")) for b in BODIES}
_PLANTS: dict = {}


def plant_of(body: str):
    if body not in _PLANTS:
        _PLANTS[body] = _plant_module().build(body, "hard", settle=False)
    return _PLANTS[body]


def load(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def rung_arrays(z: dict, w: float, ta_ref: float, tb_ref: float, rate: float = RATE) -> tuple[np.ndarray, np.ndarray]:
    """A kinematic walk from a look-ladder rung's own arrays over the reference times ``[ta_ref, tb_ref]`` at the
    clock ``rate``: its frames interpolated in time and repeated with the stride's advance (50 Hz poses on the
    simulated clock), each foot's load half the weight where the frame at or before the instant flags support
    (every physics step).  Written apart from ``bar._Played``, which a test below sets beside it."""
    q, step, n, sm, sup = z["q"], float(z["dt"]), len(z["t"]), z["stride_motion"], z["support"]
    steps = int(math.floor((tb_ref - ta_ref) / (rate * DT) + 1e-9)) + 1

    def frame(k):
        x = np.array(q[k % n], float)
        x[0] += (k // n) * sm[3]
        x[1] += (k // n) * sm[4]
        return x

    poses = []
    for j in range((steps - 1) // EVERY + 1):
        s = (ta_ref + rate * j * EVERY * DT) / step
        k0 = int(math.floor(s + 1e-12))
        a = min(max(s - k0, 0.0), 1.0)
        x = (1.0 - a) * frame(k0) + a * frame(k0 + 1)
        x[3:7] /= np.linalg.norm(x[3:7])
        poses.append(x)
    loads = np.array([np.where(sup[int(math.floor((ta_ref + rate * k * DT) / step + 1e-12)) % n], 0.5 * w, 0.0)
                      for k in range(steps)])
    return np.array(poses), loads


#: The walk item's window on the reference clock: 10 s of simulated time at the nominal commanded clock.
TA_REF, TB_REF = 2.0, 2.0 + 10.0 * RATE


def rung_reading(body, speed, variant="V6", ta_ref=TA_REF, tb_ref=TB_REF) -> dict:
    p = plant_of(body)
    poses, loads = rung_arrays(load(_rung(body, speed, variant)), lookread.weight(p), ta_ref, tb_ref)
    ser = lookread.series(p, poses, loads, DT, EVERY, t0=ta_ref / RATE)
    return lookread.reading(ser, ta_ref / RATE, tb_ref / RATE)


@needs["75-M-H-10"]
def test_the_series_are_the_look_arrays_own_columns_and_leave_the_plant_alone():
    p = plant_of("75-M-H-10")
    z = load(_rung("75-M-H-10", "v040"))
    before = (p.d.qpos.copy(), p.d.time)
    k = np.arange(0, len(z["t"]), 4)
    ser = lookread.series(p, z["q"][k], np.zeros((len(k) * EVERY, 2)), DT, EVERY)
    assert np.array_equal(p.d.qpos, before[0]) and p.d.time == before[1]
    assert ser["t50"] == pytest.approx(np.arange(len(k)) * 0.02) and ser["dt"] == DT and ser["every"] == EVERY
    assert np.allclose(ser["hip_mid_z"], z["hip_mid"][k, 2], atol=1e-12, rtol=0)
    assert np.allclose(ser["head_pitch_world"], z["head_pitch_world"][k], atol=1e-12, rtol=0)
    assert np.allclose(ser["foot_pitch"], z["foot_pitch_solved"][k], atol=1e-9, rtol=0)  # toe down positive
    assert z["foot_pitch_solved"].max() > 0.05 and np.allclose(ser["tail"], z["tail"][k], atol=1e-12, rtol=0)
    assert ser["weight"] == lookread.weight(p) == pytest.approx(float(p.m.body_subtreemass[p.trunk]) * 9.81)
    assert ser["contact"].shape == (len(k) * EVERY, 2) and not ser["contact"].any()
    assert [len(x) for x in ser["lift_offs"] + ser["touchdowns"]] == [0, 0, 0, 0]
    # a foot's pitch does not depend on the heading (a turn item reads it after 90 degrees)
    yaw = math.radians(90.0)
    q = np.array(z["q"][k], float)
    for row in q:
        w0, x0, y0, z0 = row[3:7]
        hw, hz = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
        row[3:7] = (hw * w0 - hz * z0, hw * x0 - hz * y0, hw * y0 + hz * x0, hw * z0 + hz * w0)
    turned = lookread.series(p, q, np.zeros((len(k) * EVERY, 2)), DT, EVERY)
    for key in ("hip_mid_z", "head_pitch_world", "foot_pitch", "tail"):
        assert np.allclose(turned[key], ser[key], atol=1e-12, rtol=0), key
    # arrays that do not fit are refused
    with pytest.raises(lookread.LookError, match="full qpos"):
        lookread.series(p, z["q"][:5, :20], np.zeros((50, 2)), DT, EVERY)
    with pytest.raises(lookread.LookError, match=r"\[steps, 2\]"):
        lookread.series(p, z["q"][:5], np.zeros((50, 3)), DT, EVERY)
    with pytest.raises(lookread.LookError, match="reach past"):
        lookread.series(p, z["q"][:5], np.zeros((39, 2)), DT, EVERY)


@needs["75-M-H-10"]
def test_contact_events_are_the_bars_rule_on_the_loads():
    p = plant_of("75-M-H-10")
    w = lookread.weight(p)
    loads = np.zeros((2000, 2))
    loads[:, 0] = 0.5 * w
    loads[100:400, 1] = 0.5 * w  # the right foot: down at step 100, up at step 400
    loads[1000:1300, 0] = 0.019 * w  # the left foot under 2 % of the weight: in the air
    loads[1500:1505, 0] = 0.0  # 10 ms without load: merged by the 20 ms debounce
    loads[1700:1711, 1] = 0.5 * w  # 22 ms of load: a touchdown and a lift-off
    ser = lookread.series(p, np.tile(p.q_home, (201, 1)), loads, DT, EVERY, t0=5.0)
    assert np.array_equal(ser["contact"], readings.contact_flags(loads, w, DT))
    assert ser["touchdowns"][1] == pytest.approx([5.2, 8.4]) and ser["lift_offs"][1] == pytest.approx([5.8, 8.422])
    assert ser["lift_offs"][0] == pytest.approx([7.0]) and ser["touchdowns"][0] == pytest.approx([7.6])
    assert ser["t50"][0] == 5.0 and ser["t50"][-1] == pytest.approx(9.0) and ser["steps"] == 2000


@pytest.mark.parametrize("body", BODIES)
@pytest.mark.parametrize("speed", SPEEDS)
def test_a_ladder_rungs_playback_reproduces_the_ladders_own_numbers(body, speed):
    """Gate Z2 (b)'s comparison on the rung's own arrays: bob within 5 %, head pitch RMS within 0.1 deg, tail phase
    within 0.01 cycle, toe-off within 0.02 rad and within 10 % of the ladder's number."""
    if not HAVE[body]:
        pytest.skip(needs[body].kwargs["reason"])
    rung = json.loads((LOOK / "metrics.json").read_text())["bodies"][body][speed]["V6"]
    lad = lookread.ladder_numbers(rung)
    assert lad["bob_pp_m"] == pytest.approx(1e-3 * rung["bob"]["per_step_pp_mm"])
    assert lad["tail"]["phase_cycles"] == pytest.approx(rung["tail"]["phase_rad"] / (2 * math.pi))
    r = rung_reading(body, speed)
    assert r["steps"] >= 17 and r["lift_offs"] >= 17 and r["window"] == pytest.approx([TA_REF / RATE, TB_REF / RATE])
    a = lookread.against_ladder(r, lad)
    d = a["differences"]
    assert a["ok"] and a["terms"] == dict(bob=True, head=True, tail=True, toe_off=True), (body, speed, d)
    assert a["limits"] == dict(bob_rel=0.05, head_deg=0.1, tail_cycles=0.01, toe_off_rad=0.02, toe_off_rel=0.10)
    # the margins found on 15 Nov 2026 (the toe-off reads low: a lift-off is the first step without load, and the
    # pitch is interpolated between 50 Hz poses)
    assert abs(d["bob_rel"]) < 0.005 and abs(d["head_deg"]) < 0.02 and abs(d["tail_cycles"]) < 0.002
    assert -0.003 < d["toe_off_rad"] <= 0.0 and -0.08 < d["toe_off_rel"] <= 0.0
    assert r["tail"]["amp_rad"] == pytest.approx(lad["tail"]["amp_rad"], rel=0.02)
    # the same window read 3 s long (a turn item's): at least 4 steps and 4 lift-offs on every rung
    short = rung_reading(body, speed, tb_ref=TA_REF + 3.0 * RATE)
    assert short["steps"] >= 4 and short["lift_offs"] >= 4 and short["tail"]["strides"] >= 2
    # without the toe-off (a set without F5): three terms
    assert lookread.against_ladder(r, lad, toe_off=False)["terms"] == dict(bob=True, head=True, tail=True)
    # and the limits bite: the ladder's numbers moved by more than each limit
    off = dict(lad, bob_pp_m=1.06 * lad["bob_pp_m"], head_world_pitch_rms_deg=lad["head_world_pitch_rms_deg"] + 0.12,
               tail=dict(lad["tail"], phase_cycles=lad["tail"]["phase_cycles"] + 0.012), toe_off_rad=1.15 * lad["toe_off_rad"])
    assert lookread.against_ladder(r, off)["terms"] == dict(bob=False, head=False, tail=False, toe_off=False)


@needs["75-M-H-10"]
def test_a_flat_rung_of_the_ladder_reads_no_bob_and_no_toe_off():
    """V0 (no look feature): a constant hip height and flat feet, so that the toe-off term would not be read."""
    r = rung_reading("75-M-H-10", "v040", "V0")
    assert r["steps"] >= 20 and r["bob_pp_m"] < 1e-3 and abs(r["toe_off_rad"]) < 1e-6
    c = lookread.compare(rung_reading("75-M-H-10", "v040"), r)
    assert "toe_off" in c["not_read"] and c["reference"]["toe_off_side"] == "under 0.02 rad" and c["terms"]["bob"] is False


def _stub_track(name: str, step: float, period: float, strides: int):
    seg = types.SimpleNamespace(kind="periodic", name=name, t0=0.0, duration=strides * period, step=step)
    return types.SimpleNamespace(segments=[seg], t_end=strides * period)


@needs["75-M-H-10"]
def test_playback_through_the_bars_player_is_the_rungs_own_reading(tmp_path):
    pytest.importorskip("designlab.stage2.bar")
    body, speed = "75-M-H-10", "v040"
    p = plant_of(body)
    z = load(_rung(body, speed))
    w = lookread.weight(p)
    (tmp_path / "refs").mkdir()
    np.savez(tmp_path / "refs" / "v0.4.npz", t=z["t"], dt=z["dt"], q=z["q"], support=z["support"],
             normal=np.where(z["support"], 0.5 * w, 0.0), stride_motion=z["stride_motion"], q_home=z["q_home"])
    period = len(z["t"]) * float(z["dt"])
    track = _stub_track("v0.4", float(z["dt"]), period, 3)
    a = lookread.played(p, track, tmp_path, TA_REF, TB_REF)
    poses, loads = rung_arrays(z, w, TA_REF, TB_REF)
    assert a["dt"] == DT and a["every"] == EVERY and a["t0"] == TA_REF / RATE and a["rate"] == RATE
    assert a["poses"].shape == poses.shape == (501, p.m.nq) and a["loads"].shape == loads.shape == (5001, 2)
    assert np.allclose(a["poses"], poses, atol=1e-12, rtol=0)
    assert np.array_equal(a["loads"] > 0.02 * w, loads > 0.02 * w)
    r = lookread.playback(p, track, tmp_path, TA_REF, TB_REF)
    mine = rung_reading(body, speed)
    assert r["window_ref"] == [TA_REF, TB_REF] and r["clock_rate"] == RATE
    assert r["window"] == pytest.approx([TA_REF / RATE, TB_REF / RATE]) and r["samples"] == 501
    assert (r["steps"], r["lift_offs"], r["tail"]["strides"]) == (mine["steps"], mine["lift_offs"], mine["tail"]["strides"])
    for key in ("bob_pp_m", "head_world_pitch_rms_deg", "toe_off_rad"):
        assert r[key] == pytest.approx(mine[key], rel=1e-9, abs=1e-12), key
    assert r["tail"]["phase_cycles"] == pytest.approx(mine["tail"]["phase_cycles"], abs=1e-9)
    assert lookread.compare(r, r)["ok"] and json.dumps(r)  # (a reading is plain JSON)
    # past the track's end the last pose is held (as the bar's reference recorder holds it): no error, fewer steps
    late = lookread.playback(p, track, tmp_path, 3 * period - 2.0, 3 * period + 4.0)
    assert late["steps"] < 6 and late["samples"] == 286
    with pytest.raises(lookread.LookError, match="reference times"):
        lookread.playback(p, track, tmp_path, 5.0, 5.0)


@needs["75-M-H-10"]
def test_playback_of_a_real_track_is_the_bars_reference_recorder():
    """Stage 1's own run directory (kinematic only): the start and the nominal walk on a ``control.Track``."""
    bar = pytest.importorskip("designlab.stage2.bar")
    from designlab.stage1 import control
    from designlab.stage1.refs import Reference

    body = "75-M-H-10"
    run = RUN / body
    if not (run / "refs" / "nominal.npz").exists() or not (run / "transitions" / "start.npz").exists():
        pytest.skip("needs the Stage 1 arrays (python tools/restore_inputs.py --group picks-arrays)")
    p = plant_of(body)
    omega = control.omega0(p, Reference.load(run / "refs" / "nominal.npz").q_home)
    track = control.build_track(p, run, [("transition", "start"), ("ref", "nominal", 14)], omega)
    ta, tb = 7.0, 17.0  # simulated seconds, in the walk (the start ends at 5.68 s of reference time)
    a = lookread.played(p, track, run, RATE * ta, RATE * tb)
    rec = bar.reference_recorder(p, track, tb + 0.002, run, rate=RATE)
    k0 = int(round(ta / DT))
    assert np.allclose(a["loads"], rec.loads[k0 : k0 + len(a["loads"])], atol=1e-9, rtol=0)
    rows = (k0 // 5) + 2 * np.arange(len(a["poses"]))
    assert np.allclose(a["poses"][:, p.qadr], rec.Q[rows], atol=1e-9, rtol=0)
    assert np.allclose(a["poses"][:, 0], rec.col("root_x")[rows], atol=1e-9, rtol=0)
    r = lookread.playback(p, track, run, RATE * ta, RATE * tb)
    td = [t for t, _ in readings.touchdowns(rec.contact(), rec.dt) if ta - 1e-9 <= t <= tb + 1e-9]
    assert r["steps"] == len(td) - 1 >= 17 and r["window"] == pytest.approx([ta, tb])
    # Stage 1's stock walk: a level hip, a tail a quarter cycle after the left lift-off ("today's pi/2"), flat feet
    assert r["bob_pp_m"] < 1e-4 and abs(r["toe_off_rad"]) < 1e-6
    assert r["tail"]["phase_cycles"] == pytest.approx(0.25, abs=0.005) and r["tail"]["amp_rad"] == pytest.approx(0.12, abs=0.005)
    assert "toe_off" in lookread.compare(r, r)["not_read"]
    # a periodic turning gait is never played
    turning = _stub_track("walk_0.35_left", 0.01, 1.0, 3)
    if (run / "refs" / "walk_0.35_left.npz").exists():
        with pytest.raises(lookread.LookError, match="not played: walk_0.35_left: a periodic segment must be a straight"):
            lookread.playback(p, turning, run, 1.0, 2.0)


@needs["75-M-H-10"]
def test_each_degraded_playback_fails_exactly_its_term():
    """Gate Z2 (c) on the 0.40 m/s rung of the 75 cm body, over the walk item's window and over a turn item's."""
    body, speed = "75-M-H-10", "v040"
    p = plant_of(body)
    for tb_ref in (TB_REF, TA_REF + 3.0 * RATE):
        poses, loads = rung_arrays(load(_rung(body, speed)), lookread.weight(p), TA_REF, tb_ref)
        c = lookread.controls(p, poses, loads, DT, EVERY, t0=TA_REF / RATE)
        assert c["ok"] is True and c["window"] == pytest.approx([TA_REF / RATE, tb_ref / RATE])
        same = c["unchanged"]["compare"]
        assert c["unchanged"]["ok"] and same["ok"] and same["not_read"] == []
        assert same["bob_ratio"] == 1.0 and same["tail_phase_diff_cycles"] == 0.0 and same["toe_off_ratio"] == 1.0
        assert set(c["degraded"]) == {"bob_half", "neck_3deg", "tail_late", "flat_feet"}
        g = {k: v["compare"] for k, v in c["degraded"].items()}
        assert [c["degraded"][k]["failed"] for k in ("bob_half", "neck_3deg", "tail_late", "flat_feet")] == [
            ["bob"], ["head"], ["tail"], ["toe_off"]]
        assert g["bob_half"]["bob_ratio"] == pytest.approx(0.5, abs=0.01)
        assert g["neck_3deg"]["head_world_pitch_rms_deg"] == pytest.approx(math.hypot(3.0 / math.sqrt(2.0), same["head_world_pitch_rms_deg"]), abs=0.1)
        assert g["tail_late"]["tail_phase_diff_cycles"] == pytest.approx(-0.2, abs=0.005)
        assert abs(g["flat_feet"]["toe_off_ratio"]) < 1e-6
        for k, v in g.items():  # nothing else moves
            if k != "bob_half":
                assert v["bob_ratio"] == pytest.approx(1.0, abs=1e-9), k
            if k != "tail_late":
                assert abs(v["tail_phase_diff_cycles"]) < 1e-9, k
            if k != "flat_feet":
                assert v["toe_off_ratio"] == pytest.approx(1.0, abs=1e-9), k
        cut = c["cut"]
        assert cut["ok"] and cut["own"]["steps"] == 3 and cut["compare"]["too_few_steps"] is True
        assert cut["compare"]["terms"] == dict(bob=False, head=True, tail=False, toe_off=False)
    with pytest.raises(lookread.LookError, match="degraded cop"):
        lookread.controls(p, poses, loads, DT, EVERY, kinds=("limp",))
    with pytest.raises(lookread.LookError, match="period_s"):
        lookread.degrade(p, poses, "tail_late")
    flat = lookread.series(p, lookread.degrade(p, poses, "flat_feet"), loads, DT, EVERY)
    assert np.abs(flat["foot_pitch"]).max() < 1e-9


@needs["60-N-T-10"]
def test_a_reference_toe_off_under_the_threshold_leaves_the_flat_feet_control_nothing_to_fail():
    """The 60 cm body's fastest ladder rung: its toe-off reads under 0.02 rad (rule U3 names the case), so the term
    is not read there and gate Z2 (c)'s flat-feet control cannot fail; the other three still do."""
    body, speed = "60-N-T-10", "fr05"
    p = plant_of(body)
    poses, loads = rung_arrays(load(_rung(body, speed)), lookread.weight(p), TA_REF, TB_REF)
    c = lookread.controls(p, poses, loads, DT, EVERY, t0=TA_REF / RATE)
    assert 0.017 < c["reference"]["toe_off_rad"] < 0.02
    assert c["unchanged"]["ok"] and c["unchanged"]["compare"]["not_read"] == ["toe_off"]
    assert c["unchanged"]["compare"]["reference"]["toe_off_side"] == "under 0.02 rad"
    assert c["degraded"]["flat_feet"]["failed"] == [] and c["degraded"]["flat_feet"]["ok"] is False and c["ok"] is False
    three = lookread.controls(p, poses, loads, DT, EVERY, t0=TA_REF / RATE, kinds=("bob_half", "neck_3deg", "tail_late"))
    assert three["ok"] is True and set(three["degraded"]) == {"bob_half", "neck_3deg", "tail_late"}
    # the 0.40 m/s rung of the same body is read (0.030 rad) and all four controls hold
    poses, loads = rung_arrays(load(_rung(body, "v040")), lookread.weight(p), TA_REF, TB_REF)
    c = lookread.controls(p, poses, loads, DT, EVERY, t0=TA_REF / RATE)
    assert c["ok"] is True and 0.028 < c["reference"]["toe_off_rad"] < 0.032


@needs["75-M-H-10"]
def test_a_16_joint_body_builds_its_series_by_name():
    """An ``-X`` body (neck yaw and jaw beside the 14 hinges): the plain body's walk with the two extra joints at
    home reads as the plain body's (gate Z2 (b) sets an ``-X`` playback against the plain body's)."""
    body = "75-M-H-10"
    if not _have_model(body + "-X"):
        pytest.skip("needs the -X body's model (python tools/restore_inputs.py --group x)")
    p, px = plant_of(body), plant_of(body + "-X")
    assert len(px.joints) == 16 and px.m.nq == 23 and px.joints.index("tail_yaw") != p.joints.index("tail_yaw")
    poses, loads = rung_arrays(load(_rung(body, "v040")), lookread.weight(p), TA_REF, TB_REF)
    qx = np.tile(px.q_home, (len(poses), 1))
    qx[:, :7] = poses[:, :7]
    for j, name in enumerate(p.joints):
        qx[:, px.qadr[px.joints.index(name)]] = poses[:, p.qadr[j]]
    a = lookread.series(p, poses, loads, DT, EVERY)
    b = lookread.series(px, qx, loads * lookread.weight(px) / lookread.weight(p), DT, EVERY)
    assert np.array_equal(b["tail"], poses[:, p.qadr[p.joints.index("tail_yaw")]])
    for key in ("hip_mid_z", "head_pitch_world", "foot_pitch"):
        assert np.isfinite(b[key]).all() and np.allclose(b[key], a[key], atol=1e-9, rtol=0), key
    ra, rb = lookread.reading(a, 0.0, 10.0), lookread.reading(b, 0.0, 10.0)
    c = lookread.compare(rb, ra)
    assert c["ok"] and c["bob_ratio"] == pytest.approx(1.0, abs=1e-9) and abs(c["tail_phase_diff_cycles"]) < 1e-9
    assert lookread.controls(px, qx, loads * lookread.weight(px) / lookread.weight(p), DT, EVERY)["ok"] is True
