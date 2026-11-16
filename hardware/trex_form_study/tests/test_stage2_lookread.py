"""Tests for the look reading (designlab/stage2/lookread.py; STAGE2_TRACKB_PLAN rules U1, U2, U3, U5 and gate Z2 (b),
(c)).

Run from hardware/trex_form_study (the fit's venv):
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -p no:cacheprovider tests/test_stage2_lookread.py

Fast tests on synthetic series (no model): each band on both sides of its limit, segmentation by the own events,
the bob per step, "not read", too few steps, a reference that lacks a value, the window of a bar item and its
reference times, gate Z2 (b)'s limits one at a time, the rule texts.  Tests on a model (kinematics only, no physics
and no episode): the look ladder's V6 rungs played from their own arrays against ``stage2/look/metrics.json`` within
gate Z2 (b)'s limits, the playback through ``bar._Played`` (a rung written as a reference file under pytest's
temporary directory, and Stage 1's own run directory with a real track), the window and the playback of real bar
items (a frozen 75 cm set's tracks, played, never run), the negative controls of gate Z2 (c), and an ``-X`` body.
They need the round-two model files, the look ladder's arrays, Stage 1's arrays and the frozen sets' arrays
(git-ignored; ``python tools/restore_inputs.py restore``) and skip without them.  Nothing is written outside
``tmp_path``.

Slow (``STAGE2_SLOW=1``; about 5 s): one closed-loop episode of Stage 1's own run directory (the view ``S1``; row
R0, no record), its recorder lent 50 Hz poses as the row wrapper will lend them: what ``series`` takes of a live
recorder.
"""

from __future__ import annotations

import json
import math
import os
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
SLOW = os.environ.get("STAGE2_SLOW") == "1"


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


#: The command each skip names, by the groups it restores (the tool takes a subcommand before ``--group``).
RESTORE = "python tools/restore_inputs.py restore --group"
HINTS = {"ladder": f"{RESTORE} picks-models look", "stage1": f"{RESTORE} picks-arrays", "sets": f"{RESTORE} sets",
         "x": f"{RESTORE} x"}


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


def test_the_rule_texts_mark_what_is_the_modules_own_reading():
    """The texts go into records as rule texts: each opens with the plan's rule and holds, after one mark, the
    sentences that are this module's reading and not the rule's words."""
    mark = lookread.MODULE_READING
    assert mark == "This module's own reading, which is not the rule's wording: "
    own = dict(
        measures=("hip-roll joints' anchors", "elevation of the head body's x axis", "first and last left lift-off",
                  "elevation of the foot body's x axis", "first physics step after the change"),
        window=("did not fall", "an episode that fell", "two updates", "when the trace spans it"),
        bands=("cannot be computed fails its term", "The bob has no such threshold", "reads the phase alone",
               "no band reads it"),
        reference=("bar._Played", "support flags", "lacks one of its own values"))
    rule = dict(
        measures=("rule U1", "own steps", "RMS about its mean", "zero at each of its left lift-offs", "2 % load rule"),
        window=("rule U2", "judged last 10 s", "3 s after t_c + at_s", "Nudge and stop items are not read"),
        bands=("rule U3", "<= 0.3;", "<= 1.5 deg RMS (not relative)", "<= 0.1 cycle", ">= 0.7 x reference",
               "under 0.02 rad", "both then pass, labelled"),
        reference=("rule U5", "nominal commanded clock (1.05)", "at least 4 steps and 4 lift-offs", "too_few_steps",
                   "When the playback itself has fewer, it is an error of the definition"))
    for key, text in lookread.RULES.items():
        assert text.count(mark) == 1, key
        before, after = text.split(mark)
        assert before.startswith("STAGE2_TRACKB_PLAN rule U") and after.endswith("."), key
        for phrase in own[key]:
            assert phrase in after and phrase not in before, (key, phrase)
        for phrase in rule[key]:
            assert phrase in before, (key, phrase)


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


def test_the_bob_is_the_mean_of_the_own_steps_ranges_not_the_windows_range():
    """A hip height that drifts 2 mm/s under the bob: each step's own range holds 0.8 mm of the drift, the whole
    window's range 20 mm of it."""
    s = synth()
    flat = read(s)
    drifting = s["hip_mid_z"] + 0.002 * s["t50"]
    r = read(dict(s, hip_mid_z=drifting))
    sel = (s["t50"] >= 3.0) & (s["t50"] <= 13.0)
    whole = float(drifting[sel].max() - drifting[sel].min())
    assert whole == pytest.approx(0.008 + 0.020, rel=0.03) and whole > 3.0 * flat["bob_pp_m"]
    assert r["steps"] == flat["steps"] == 23 and r["bob_pp_m"] == pytest.approx(flat["bob_pp_m"], rel=0.10)
    assert lookread.compare(r, flat)["terms"]["bob"] is True
    # and it is the mean over the steps, not the largest of them
    big = s["hip_mid_z"].copy()
    step = (s["t50"] > 5.0) & (s["t50"] < 5.4)
    big[step] = 0.45 + 2.0 * (big[step] - 0.45)
    one = read(dict(s, hip_mid_z=big))  # (0.4 s with twice the bob: one or two of the 23 steps)
    assert flat["bob_pp_m"] < one["bob_pp_m"] < 1.1 * flat["bob_pp_m"]


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
    # too few lift-offs alone (4 steps, 3 lift-offs) is too few as well, and so are too few steps alone (3 steps,
    # 4 lift-offs); 4 and 4 are enough
    four = read(synth(), (3.0, 3.0 + 4.6 * 0.41))
    assert four["steps"] == 4 and four["lift_offs"] == 4
    assert lookread.compare(four, ref)["too_few_steps"] is False and lookread.compare(four, ref)["ok"] is True
    for short in (dict(four, lift_offs=3), dict(four, steps=3)):
        c = lookread.compare(short, ref)
        assert c["too_few_steps"] is True and c["terms"] == dict(bob=False, head=True, tail=False, toe_off=False)
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


def test_a_reference_that_lacks_a_value_is_a_definition_fault():
    """A term is left unread only for a reference value under 0.02 rad, never for want of the value: a reference
    with enough steps that holds no finite bob, tail or toe-off is refused."""
    ref, own = read(synth()), read(synth())
    for bad, name in ((dict(ref, bob_pp_m=None), "bob_pp_m"), (dict(ref, toe_off_rad=None), "toe_off_rad"),
                      (dict(ref, toe_off_rad=float("nan")), "toe_off_rad"),
                      (dict(ref, tail=dict(phase_cycles=None, amp_rad=None, strides=0)), r"tail\.amp_rad"),
                      (dict(ref, tail=dict(phase_cycles=None, amp_rad=0.12, strides=1)), r"tail\.phase_cycles"),
                      ({k: v for k, v in ref.items() if k != "tail"}, r"tail\.amp_rad")):
        with pytest.raises(lookread.LookError, match=f"holds no finite .*{name}.*rule U5"):
            lookread.compare(own, bad)
    with pytest.raises(lookread.LookError, match="bob_pp_m, tail.amp_rad, toe_off_rad"):
        lookread.compare(own, dict(steps=6, lift_offs=6))
    # a tail under the threshold needs no phase: it is not read
    c = lookread.compare(own, dict(ref, tail=dict(phase_cycles=None, amp_rad=0.01, strides=1)))
    assert c["not_read"] == ["tail"] and c["terms"]["tail"] is True and c["tail_phase_diff_cycles"] is None
    # the reference's head pitch is not read at all (the band is absolute)
    assert lookread.compare(own, dict(ref, head_world_pitch_rms_deg=None))["ok"] is True


def test_the_tails_amplitude_is_recorded_and_no_band_reads_it():
    ref = read(synth())
    for k in (0.05, 0.5, 1.0, 3.0):
        c = lookread.compare(read(synth(tail_amp=k * 0.12)), ref)
        assert c["tail_amp_ratio"] == pytest.approx(k, abs=1e-9) and c["terms"]["tail"] is True and c["ok"] is True
    c = lookread.compare(dict(read(synth()), tail=dict(phase_cycles=None, amp_rad=None)), ref)
    assert c["tail_amp_ratio"] is None and c["terms"]["tail"] is False
    flat = lookread.compare(read(synth()), read(synth(tail_amp=0.0)))
    assert flat["not_read"] == ["tail"] and (flat["tail_amp_ratio"] is None or flat["tail_amp_ratio"] > 1e6)


# ------------------------------------------------------------------------------------------------
# Gate Z2 (b)'s limits (no model)
# ------------------------------------------------------------------------------------------------
def test_the_ladders_limits_each_bite_alone_and_the_tails_difference_is_wrapped():
    """``against_ladder`` on numbers alone: each limit on both sides, the toe-off's two limits (0.02 rad and 10 % of
    the ladder's number) each failing where the other holds, and a tail phase compared across the wrap at half a
    cycle."""
    lad = dict(bob_pp_m=0.0075, head_world_pitch_rms_deg=0.54, tail=dict(phase_cycles=-0.28, amp_rad=0.12), toe_off_rad=0.30)

    def terms(ladder=lad, **own):
        return lookread.against_ladder(dict(ladder, **own), ladder)

    assert terms()["ok"] and terms()["terms"] == dict(bob=True, head=True, tail=True, toe_off=True)
    for own, term, ok in ((dict(bob_pp_m=0.0075 * 1.049), "bob", True), (dict(bob_pp_m=0.0075 * 1.051), "bob", False),
                          (dict(bob_pp_m=0.0075 * 0.949), "bob", False),
                          (dict(head_world_pitch_rms_deg=0.54 + 0.099), "head", True),
                          (dict(head_world_pitch_rms_deg=0.54 - 0.101), "head", False),
                          (dict(tail=dict(phase_cycles=-0.28 + 0.009, amp_rad=0.12)), "tail", True),
                          (dict(tail=dict(phase_cycles=-0.28 - 0.011, amp_rad=0.12)), "tail", False)):
        a = terms(**own)
        assert a["terms"] == dict(dict(bob=True, head=True, tail=True, toe_off=True), **{term: ok}), own
    # the toe-off on a large ladder number (0.30 rad): 0.025 rad off is 8.3 % and fails by the 0.02 rad limit alone
    a = terms(toe_off_rad=0.30 - 0.025)
    assert abs(a["differences"]["toe_off_rad"]) > 0.02 and abs(a["differences"]["toe_off_rel"]) < 0.10
    assert a["terms"] == dict(bob=True, head=True, tail=True, toe_off=False)
    assert terms(toe_off_rad=0.30 + 0.025)["terms"]["toe_off"] is False
    assert terms(toe_off_rad=0.30 - 0.019)["terms"]["toe_off"] is True
    # on a small one (0.056 rad): 12 % off is 0.0067 rad and fails by the 10 % limit alone
    small = dict(lad, toe_off_rad=0.056)
    a = terms(small, toe_off_rad=0.056 * 0.88)
    assert abs(a["differences"]["toe_off_rad"]) < 0.02 and abs(a["differences"]["toe_off_rel"]) > 0.10
    assert a["terms"] == dict(bob=True, head=True, tail=True, toe_off=False)
    assert terms(small, toe_off_rad=0.056 * 1.12)["terms"]["toe_off"] is False
    assert terms(small, toe_off_rad=0.056 * 0.91)["terms"]["toe_off"] is True
    # left out (a set without F5), neither limit is read
    assert lookread.against_ladder(dict(lad, toe_off_rad=0.0), lad, toe_off=False)["ok"] is True
    # the tail across the wrap: 0.498 and -0.497 cycle are 0.005 cycle apart, 0.49 and -0.49 are 0.02
    for own, other, d, ok in ((-0.497, 0.498, 0.005, True), (0.498, -0.497, -0.005, True), (-0.49, 0.49, 0.02, False)):
        a = lookread.against_ladder(dict(lad, tail=dict(phase_cycles=own, amp_rad=0.12)),
                                    dict(lad, tail=dict(phase_cycles=other, amp_rad=0.12)))
        assert a["differences"]["tail_cycles"] == pytest.approx(d, abs=1e-12) and a["terms"]["tail"] is ok


def test_the_skip_reasons_name_commands_the_restore_tool_takes():
    """Each skip names ``python tools/restore_inputs.py restore --group ...``: the tool's own parser takes it (asked
    with ``--help`` after the arguments, so that nothing is restored), and refuses the form without a subcommand."""
    tool = ROOT / "tools" / "restore_inputs.py"
    for hint in HINTS.values():
        words = hint.split()
        assert words[:3] == ["python", "tools/restore_inputs.py", "restore"] and words[3] == "--group" and words[4:]
        ok = subprocess.run([sys.executable, str(tool), *words[2:], "--help"], cwd=ROOT, capture_output=True, text=True)
        assert ok.returncode == 0 and "usage: python tools/restore_inputs.py restore" in ok.stdout, hint
        bad = subprocess.run([sys.executable, str(tool), *words[3:], "--help"], cwd=ROOT, capture_output=True, text=True)
        assert bad.returncode == 2 and "invalid choice" in bad.stderr, hint
    assert all(HINTS["ladder"] in needs[b].kwargs["reason"] for b in BODIES)
    without_subcommand = "restore_inputs.py " + "--group"  # (the form this file's skips named before)
    assert without_subcommand not in Path(__file__).read_text()


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
    plant_ = types.SimpleNamespace(m=types.SimpleNamespace(opt=types.SimpleNamespace(timestep=0.002)))
    return types.SimpleNamespace(cond=types.SimpleNamespace(judge=judge), seconds=seconds, resolved=resolved, p=plant_,
                                 command=None if command is None else dict(ref_s=command))


def test_the_window_of_a_walk_item_is_its_judged_last_10s():
    assert lookread.window(_built("bar_walk"), _rec(19.99)) == pytest.approx((9.99, 19.99))
    assert lookread.window(_built("bar_walk"), _rec(19.98)) == pytest.approx((9.98, 19.98))  # (the bar's slack: two updates)
    assert lookread.window(_built("bar_walk"), _rec(19.97)) is None  # the trace ends before the planned end (a fall)
    assert lookread.window(_built("bar_walk", seconds=12.0, cut_from=20.0), _rec(11.99)) is None  # a cut episode
    assert lookread.window(_built("bar_walk"), types.SimpleNamespace(t=np.zeros(0), rows=0, dt=0.002)) is None


def test_a_walk_episode_that_fell_has_no_window():
    """The bar judges no walk window after a fall (``bar._judge``), also one inside the last two updates, which the
    test of the planned end lets through.  The fall is the recorder's own ``out`` (``metrics.Recorder.end``)."""
    b = _built("bar_walk")
    for t_end in (19.99, 19.98):
        rec = _rec(t_end)
        assert lookread.window(b, rec) == pytest.approx((t_end - 10.0, t_end))  # (no ``out``: a playback's recorder)
        rec.out = dict(fell=False, fall_time_s=None)
        assert lookread.window(b, rec) == pytest.approx((t_end - 10.0, t_end))
        rec.out = dict(fell=True, fall_time_s=t_end)
        assert lookread.window(b, rec) is None
        rec.out = None
        assert lookread.window(b, rec) is not None
    assert lookread.window_ref(b) is not None  # (the condition's reference times do not depend on an episode)
    # a turn item's window is read whenever the trace spans it, as the bar reads it: also when the robot fell later
    turn, rec = _built("bar_turn", command=8.4, at_s=2.0), _rec(13.2)
    rec.out = dict(fell=True, fall_time_s=13.2)
    assert lookread.window(turn, rec) == pytest.approx((10.0, 13.0))


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
        assert lookread.window_ref(_built(judge, command=8.4, at_s=2.0)) is None
    assert lookread.window(types.SimpleNamespace(cond=types.SimpleNamespace()), _rec(19.99)) is None
    assert lookread.window_ref(types.SimpleNamespace(cond=types.SimpleNamespace())) is None


def test_the_windows_reference_times_are_the_window_on_the_nominal_clock():
    """``window_ref``: from the built condition alone, the reference times ``playback`` takes.  On a trace whose
    clock ran at the nominal rate they are ``RATE`` times the simulated window; on a slower clock the simulated
    window moves and the reference times do not."""
    # a walk item: 20 s are 10000 steps, the last update is step 9995 (19.99 s)
    b = _built("bar_walk")
    assert lookread.window_ref(b) == pytest.approx((RATE * 9.99, RATE * 19.99), abs=1e-12)
    assert lookread.window_ref(b) == pytest.approx(tuple(RATE * x for x in lookread.window(b, _rec(19.99))), abs=1e-12)
    # lengths that are no whole number of updates: 14.1917 s are 7096 steps (last update 14.19 s), 14.1899 s 7095
    assert lookread.window_ref(_built("bar_walk", seconds=14.1917)) == pytest.approx((RATE * 4.19, RATE * 14.19), abs=1e-12)
    assert lookread.window_ref(_built("bar_walk", seconds=14.1899)) == pytest.approx((RATE * 4.18, RATE * 14.18), abs=1e-12)
    assert lookread.window_ref(_built("bar_walk", seconds=12.0, cut_from=20.0)) is None  # a cut episode has no window
    assert lookread.window_ref(_built("bar_walk", seconds=0.0)) is None
    # a turn item: from the command's reference time
    b = _built("bar_turn", command=8.4, at_s=2.0)
    assert lookread.window_ref(b) == pytest.approx((8.4 + RATE * 2.0, 8.4 + RATE * 5.0), abs=1e-12)
    assert lookread.window_ref(b) == pytest.approx(tuple(RATE * x for x in lookread.window(b, _rec(13.2))), abs=1e-9)
    slow = lookread.window(b, _rec(14.0, t_ref_rate=1.0))
    assert slow == pytest.approx((10.4, 13.4)) and lookread.window_ref(b) == pytest.approx((10.5, 13.65), abs=1e-12)
    assert lookread.window_ref(_built("bar_turn", at_s=2.0)) is None  # (no command)


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
    f"needs the round-two model of {b} and the look ladder's arrays ({HINTS['ladder']})")) for b in BODIES}
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
    # every pose needs its load: pose j is physics step j * every, the last load is step steps - 1
    for n, steps in ((5, 39), (5, 40), (6, 50), (1, 0)):
        with pytest.raises(lookread.LookError, match=f"{n} poses every 10 steps reach past the {steps} physics steps"):
            lookread.series(p, z["q"][:n], np.zeros((steps, 2)), DT, EVERY)
    assert len(lookread.series(p, z["q"][:5], np.zeros((41, 2)), DT, EVERY)["t50"]) == 5
    assert len(lookread.series(p, z["q"][:0], np.zeros((0, 2)), DT, EVERY)["t50"]) == 0


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
    ser = lookread.series(p, np.tile(p.q_home, (200, 1)), loads, DT, EVERY, t0=5.0)
    assert np.array_equal(ser["contact"], readings.contact_flags(loads, w, DT))
    assert ser["touchdowns"][1] == pytest.approx([5.2, 8.4]) and ser["lift_offs"][1] == pytest.approx([5.8, 8.422])
    assert ser["lift_offs"][0] == pytest.approx([7.0]) and ser["touchdowns"][0] == pytest.approx([7.6])
    assert ser["t50"][0] == 5.0 and ser["t50"][-1] == pytest.approx(8.98) and ser["steps"] == 2000


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
    # a number that is missing on either side fails its term (no exception)
    gone = lookread.against_ladder(r, dict(lad, tail=dict(phase_cycles=None, amp_rad=None), toe_off_rad=None))
    assert gone["terms"] == dict(bob=True, head=True, tail=False, toe_off=False) and gone["ok"] is False
    assert lookread.against_ladder(dict(r, bob_pp_m=None), lad)["terms"]["bob"] is False


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
    from designlab.stage1 import control
    from designlab.stage1.refs import Reference
    from designlab.stage2 import bar

    body = "75-M-H-10"
    run = RUN / body
    if not (run / "refs" / "nominal.npz").exists() or not (run / "transitions" / "start.npz").exists():
        pytest.skip(f"needs the Stage 1 arrays ({HINTS['stage1']})")
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
def test_the_window_and_the_playback_of_real_bar_items():
    """The bar's own built conditions on a frozen 75 cm set (F25F9, a control set without the look's bob and
    toe-off; kinematic only: its tracks are played, no episode runs).  A recorder filled by the bar's own playback
    at the nominal clock, with the 50 Hz poses of the whole episode, stands for the robot: the window is the bar's,
    its reference times are ``window_ref``'s, and the playback of just those times reads what the whole trace
    reads inside the window."""
    from designlab.stage1 import control, tune
    from designlab.stage2 import bar, setio

    body = "75-M-H-10"
    d = setio.set_dir("F25F9", body)
    if not (d / "refs" / "v0.4.npz").exists() or not (d / "transitions" / "turn90_pi4_left.npz").exists():
        pytest.skip(f"needs the frozen sets' arrays ({HINTS['sets']})")
    docs = setio.load(d)
    conds = {c.key: c for c in bar.conditions(docs, every=True)}
    ctx = bar.BarContext(body, d, {"refs.json": docs["refs.json"], "transitions.json": docs["transitions.json"]},
                         docs["set.json"])
    every = tune.LEGS_EVERY * control.FEEDBACK_EVERY
    assert every == EVERY
    recs = {}
    for key in ("walk/v0.4", "turn/q90_left"):
        b = ctx.built(conds[key])
        rec = recs[key] = bar.reference_recorder(b.p, b.track, b.seconds, d, rate=RATE)
        a = lookread.played(b.p, b.track, d, 0.0, RATE * (rec.steps - 1) * rec.dt)  # the whole episode's poses
        assert np.allclose(a["loads"], rec.loads, atol=1e-9, rtol=0) and len(a["poses"]) == (rec.rows + 1) // 2
        rec.poses = list(a["poses"])  # (as a recorder with legs_every = tune.LEGS_EVERY holds them)
        win = lookread.window(b, rec)
        win_ref = lookread.window_ref(b)
        if key == "walk/v0.4":
            assert win == (float(rec.t[-1]) - 10.0, float(rec.t[-1])) and bar._ran_to_end(b, rec)
        else:
            turn = next(s for s in b.track.segments if s.name == conds[key].move)
            assert b.resolved["at_s"] == 2.0 and b.command["ref_s"] == pytest.approx(turn.t0, abs=1e-9)
            assert win == pytest.approx((turn.t0 / RATE + 2.0, turn.t0 / RATE + 5.0), abs=1e-9)
        assert win_ref == pytest.approx((RATE * win[0], RATE * win[1]), abs=1e-9)
        own = lookread.reading(lookread.series(b.p, rec.poses, rec.loads, rec.dt, every), *win)
        ref = lookread.playback(b.p, b.track, b.track.run_dir, *win_ref)
        assert ref["window"] == pytest.approx(list(win), abs=1e-9) and ref["window_ref"] == list(win_ref)
        assert ref["steps"] >= lookread.MIN_STEPS and ref["lift_offs"] >= lookread.MIN_STEPS  # (what rule U5 relies on)
        assert (own["steps"], own["lift_offs"], own["tail"]["strides"]) == (ref["steps"], ref["lift_offs"], ref["tail"]["strides"])
        c = lookread.compare(own, ref)
        # the two differ by the 50 Hz grid alone (the trace's poses fall on multiples of 20 ms, the playback's on
        # the window's start)
        assert abs(c["tail_phase_diff_cycles"]) < 0.005 and c["tail_amp_ratio"] == pytest.approx(1.0, abs=0.01)
        assert own["head_world_pitch_rms_deg"] == pytest.approx(ref["head_world_pitch_rms_deg"], abs=0.02)
        assert c["terms"]["tail"] is True and c["too_few_steps"] is False
        # this set's walk has a level hip and flat feet: no toe-off to read, and no bob to be within 30 % of
        assert c["not_read"] == ["toe_off"] and c["reference"]["toe_off_side"] == "under 0.02 rad"
        assert ref["bob_pp_m"] < 1e-5 and abs(ref["toe_off_rad"]) < 1e-5
    # a nudge and a stop item of the same set are not read
    for key in ("nudge/forward", "stop/two_step"):
        b = ctx.built(conds[key])
        assert lookread.window(b, recs["walk/v0.4"]) is None and lookread.window_ref(b) is None
    # a trace that ends before the walk's planned end has no window; its reference times stay the condition's
    b = ctx.built(conds["walk/v0.4"])
    rec = recs["walk/v0.4"]
    short = types.SimpleNamespace(t=rec.t[:-3], rows=rec.rows - 3, dt=rec.dt, sim_time=rec.sim_time)
    assert lookread.window(b, short) is None and lookread.window_ref(b) is not None


@pytest.mark.skipif(not SLOW, reason="one closed-loop episode on Stage 1's own run directory (STAGE2_SLOW=1)")
@needs["75-M-H-10"]
def test_a_live_recorders_poses_and_loads_are_what_series_takes():
    """One closed-loop episode on the view ``S1`` (Stage 1's brisk walk, row R0, panel seed 3042; no record): the
    built condition's recorder is lent 50 Hz poses on the instance for the episode, as ``lookbar.LookRow`` will
    lend them, and ``series`` reads the recorder's own arrays."""
    from designlab.stage1 import control, metrics, panels, tune
    from designlab.stage2 import bar, setio

    body = "75-M-H-10"
    run = RUN / body
    if not (run / "refs" / "brisk.npz").exists() or not (run / "transitions" / "start.npz").exists():
        pytest.skip(f"needs the Stage 1 arrays ({HINTS['stage1']})")
    _, docs = setio.stage1_view(RUN, body)
    ctx = bar.BarContext(body, run, docs)
    cond = next(c for c in panels.conditions() if c.key == "walk/brisk")
    row = panels.tracker_rows(body, run, ("R0",))["R0"]
    b = ctx.built(cond)
    b.recorder = lambda: metrics.Recorder(legs_every=tune.LEGS_EVERY)
    try:
        b, out, rec = row.episode(ctx, cond, 3042)
    finally:
        del b.recorder
    assert "recorder" not in vars(b) and not out["fell"]
    assert rec.out["fell"] is False  # (what ``window`` reads of a walk item's recorder: ``metrics.Recorder.end``)
    every = tune.LEGS_EVERY * control.FEEDBACK_EVERY
    assert len(rec.poses) == (rec.rows + 1) // 2 and rec.loads.shape == (rec.steps, 2)
    before = b.p.d.qpos.copy()
    ser = lookread.series(b.p, rec.poses, rec.loads, rec.dt, every)
    assert np.array_equal(b.p.d.qpos, before)
    # the series are on the recorder's own clock and contact
    assert np.allclose(ser["t50"], rec.t[:: tune.LEGS_EVERY], atol=1e-9, rtol=0)
    assert np.array_equal(ser["contact"], rec.contact()) and ser["weight"] == rec.weight
    assert np.array_equal(ser["tail"], rec.Q[:: tune.LEGS_EVERY, b.p.joints.index("tail_yaw")])
    td = readings.touchdowns(rec.contact(), rec.dt)
    assert np.allclose(np.sort(np.concatenate(ser["touchdowns"])), [t for t, _ in td], atol=1e-12, rtol=0)
    assert [len(x) for x in ser["touchdowns"]] == [sum(1 for _, i in td if i == j) for j in (0, 1)]
    # a stock condition is no bar item: it has no window; read here over its last 10 s against the track's playback
    assert lookread.window(b, rec) is None and lookread.window_ref(b) is None
    tb = float(rec.t[-1])
    own = lookread.reading(ser, tb - 10.0, tb)
    ref = lookread.playback(b.p, b.track, run, RATE * (tb - 10.0), RATE * tb)
    c = lookread.compare(own, ref)
    assert ref["steps"] >= 20 and abs(own["steps"] - ref["steps"]) <= 1 and c["too_few_steps"] is False
    assert abs(c["tail_phase_diff_cycles"]) < 0.02 and c["terms"]["tail"] is True
    assert c["tail_amp_ratio"] == pytest.approx(1.0, abs=0.05)
    # Stage 1's stock walk has flat feet and a level hip: the toe-off is not read, and the robot's own are small
    assert c["not_read"] == ["toe_off"] and abs(own["toe_off_rad"]) < 0.01 and own["bob_pp_m"] < 0.003
    assert all(math.isfinite(own[k]) for k in ("bob_pp_m", "head_world_pitch_rms_deg", "toe_off_rad"))


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


@needs["75-M-H-10"]
def test_the_controls_verdict_needs_each_of_its_parts(monkeypatch):
    """Gate Z2 (c)'s ``ok`` on a turn item's window of the same rung, with one part spoiled at a time: a degraded
    copy that fails a second term beside its own is not "exactly its term"; a cut that does not read too few steps;
    an unchanged playback that passes without the exact ratio 1.  Each alone makes ``ok`` False."""
    body, speed = "75-M-H-10", "v040"
    p = plant_of(body)
    poses, loads = rung_arrays(load(_rung(body, speed)), lookread.weight(p), TA_REF, TA_REF + 3.0 * RATE)
    real_compare, real_degrade = lookread.compare, lookread.degrade

    def run():
        return lookread.controls(p, poses, loads, DT, EVERY, t0=TA_REF / RATE)

    def parts(c):
        return (c["unchanged"]["ok"], c["cut"]["ok"], {k: v["ok"] for k, v in c["degraded"].items()}, c["ok"])

    every_kind = dict(bob_half=True, neck_3deg=True, tail_late=True, flat_feet=True)
    assert parts(run()) == (True, True, every_kind, True)

    # (1) the bob's copy also nods: it fails its own term and the head's
    def two(p_, q, kind, *args, **kw):
        out = real_degrade(p_, q, kind, *args, **kw)
        return real_degrade(p_, out, "neck_3deg", *args, **kw) if kind == "bob_half" else out

    monkeypatch.setattr(lookread, "degrade", two)
    c = run()
    assert c["degraded"]["bob_half"]["failed"] == ["bob", "head"] and c["degraded"]["bob_half"]["expected"] == ["bob"]
    assert parts(c) == (True, True, dict(every_kind, bob_half=False), False)
    # ... or fails nothing at all
    monkeypatch.setattr(lookread, "degrade", lambda p_, q, kind, *args, **kw: (
        np.array(q, float) if kind == "tail_late" else real_degrade(p_, q, kind, *args, **kw)))
    c = run()
    assert c["degraded"]["tail_late"]["failed"] == [] and parts(c) == (True, True, dict(every_kind, tail_late=False), False)
    monkeypatch.setattr(lookread, "degrade", real_degrade)

    # (2) the cut to 3 steps is compared by a function that does not label too few steps
    def unlabelled(own, ref):
        c = real_compare(own, ref)
        return dict(c, too_few_steps=False) if own["steps"] == lookread.MIN_STEPS - 1 else c

    monkeypatch.setattr(lookread, "compare", unlabelled)
    c = run()
    assert c["cut"]["own"]["steps"] == 3 and c["cut"]["compare"]["terms"] == dict(bob=False, head=True, tail=False, toe_off=False)
    assert parts(c) == (True, False, every_kind, False)

    # ... or that labels them and fails no term
    def unfailed(own, ref):
        c = real_compare(own, ref)
        return dict(c, terms=dict(c["terms"], toe_off=True)) if own["steps"] == lookread.MIN_STEPS - 1 else c

    monkeypatch.setattr(lookread, "compare", unfailed)
    assert parts(run()) == (True, False, every_kind, False)

    # (3) the unchanged playback passes all four terms, with a ratio or a difference that is not exactly 1 or 0
    for term, key, value in (("bob", "bob_ratio", 1.0 + 1e-12), ("tail", "tail_phase_diff_cycles", 1e-15),
                             ("toe_off", "toe_off_ratio", 1.0 - 1e-12)):
        def near(own, ref, key=key, value=value):
            c = real_compare(own, ref)
            return dict(c, **{key: value}) if own is ref else c

        monkeypatch.setattr(lookread, "compare", near)
        c = run()
        assert c["unchanged"]["compare"]["ok"] is True
        assert c["unchanged"]["exact"] == dict(dict(bob=True, tail=True, toe_off=True), **{term: False}), key
        assert parts(c) == (False, True, every_kind, False), key
    monkeypatch.setattr(lookread, "compare", real_compare)
    assert parts(run()) == (True, True, every_kind, True)


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
        pytest.skip(f"needs the -X body's model ({HINTS['x']})")
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
    # gate Z2 (b)'s form for an -X body: its reading against the plain body's, within the ladder's limits
    g = lookread.against_ladder(rb, ra)
    assert g["ok"] and g["terms"] == dict(bob=True, head=True, tail=True, toe_off=True)
    assert max(abs(v) for v in g["differences"].values()) < 1e-8
    assert lookread.controls(px, qx, loads * lookread.weight(px) / lookread.weight(p), DT, EVERY)["ok"] is True
