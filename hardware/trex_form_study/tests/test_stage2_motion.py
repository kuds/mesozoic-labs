"""Tests for the Stage 2 look ladder's machine metrics and plots (designlab/stage2/motion.py; research_look.md 4.1
items 3 and 5).

Run from hardware/trex_form_study:  python -m pytest tests/test_stage2_motion.py -q

1. The readings on known inputs: SPARC (the published algorithm: invariant to scale, a single bell smoother than two
   submovements) and its periodic reading (a sinusoid -1.95 at any amplitude and frequency, a circle the same, a
   harmonic or a corner lowers it); angular velocity and relative attitude of known rotations; extremes of known
   periodic signals; the crest hold of a sinusoidal bob.
2. On the ladder (git-ignored arrays): today's walk shows every mechanical tell and V6 none of the first four; the
   readings ``look.measure`` also takes equal its; the stabilisation ratio is undefined while the trunk is still.
3. The committed metrics.json: every rung of look_refs.json read, from the arrays it records (sha256), with the
   readings in their ranges.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import designlab  # noqa: E402,F401  (selects the GL backend before mujoco is imported)
from designlab.stage2 import look, motion, screen  # noqa: E402

LOOK = ROOT / "stage2" / "look"
LADDER = json.loads((LOOK / "look_refs.json").read_text()) if (LOOK / "look_refs.json").exists() else None
METRICS = json.loads((LOOK / motion.METRICS_JSON).read_text()) if (LOOK / motion.METRICS_JSON).exists() else None
_F = None if LADDER is None else LOOK / LADDER["bodies"]["60-N-T-10"]["v040"]["V6"]["file"]
needs_arrays = pytest.mark.skipif(_F is None or not _F.exists(), reason="the git-ignored ladder arrays are not here")
needs_metrics = pytest.mark.skipif(METRICS is None or LADDER is None, reason="stage2/look/metrics.json not generated")


def _min_jerk(n=120):
    s = np.linspace(0, 1, n)
    return 30 * s**2 * (1 - s) ** 2  # the minimum-jerk speed profile


# ------------------------------------------------------------------------------------------------
# 1. Readings on known inputs
# ------------------------------------------------------------------------------------------------
def test_sparc_of_a_bell_is_scale_free_and_smoother_than_two_submovements():
    v = _min_jerk()
    fs = 200.0
    one = motion.sparc(v, fs)
    assert one == pytest.approx(-1.40, abs=0.01)  # a minimum-jerk bell (the reference walk's V0 swing reads the same)
    assert motion.sparc(3.7 * v, fs) == pytest.approx(one, abs=1e-12)
    two = motion.sparc(np.concatenate([v, 0.6 * v]), fs)
    assert two < one - 0.3
    slow = motion.sparc(_min_jerk(240), fs)  # the same movement over twice the time
    assert slow == pytest.approx(one, abs=0.01)


def test_sparc_periodic_of_sinusoids_harmonics_and_corners():
    dt = 0.005
    t = np.arange(200) * dt  # one period of 1 s
    sine = np.sin(2 * np.pi * t)[:, None]
    ref = motion.sparc_periodic(sine, dt)["sparc"]
    assert ref == pytest.approx(-1.95, abs=0.01)
    assert motion.sparc_periodic(5 * sine, dt)["sparc"] == pytest.approx(ref, abs=1e-9)
    t2 = np.arange(100) * dt  # a 0.5 s period: the band holds twice the bins below the peak, so the arc's
    # frequency steps differ a little (1 - 1/bins), as in the published reading's normalised band
    assert motion.sparc_periodic(np.sin(2 * np.pi * t2 / 0.5)[:, None], dt)["sparc"] == pytest.approx(ref, abs=0.05)
    circle = np.column_stack([np.sin(2 * np.pi * t), np.cos(2 * np.pi * t), np.zeros_like(t)])
    assert motion.sparc_periodic(circle, dt)["sparc"] == pytest.approx(ref, abs=0.01)
    harm = sine + 0.3 * np.sin(2 * np.pi * 5 * t)[:, None]
    square = np.sign(np.sin(2 * np.pi * t + 1e-9))[:, None]
    h, q = motion.sparc_periodic(harm, dt)["sparc"], motion.sparc_periodic(square, dt)["sparc"]
    assert h < ref - 1.0 and q < h
    assert motion.sparc_periodic(sine, dt)["repeats"] == 20  # tiled to 20 s


def test_angular_velocity_and_relative_attitude_of_known_rotations():
    dt, w = 0.01, 0.7
    n = int(round(2 * np.pi / w / dt))
    ang = 2 * np.pi * np.arange(n) / n  # a whole turn about z, periodic
    q = np.column_stack([np.cos(ang / 2), 0 * ang, 0 * ang, np.sin(ang / 2)])
    om = motion.angular_velocity(q, dt)
    np.testing.assert_allclose(om[:, 2], (2 * np.pi / (n * dt)), rtol=1e-3)
    np.testing.assert_allclose(om[:, :2], 0, atol=1e-12)
    home = np.array([math.cos(0.05), 0.0, math.sin(-0.05), 0.0])  # pitched nose down a little
    roll = np.array([math.cos(0.01), math.sin(0.01), 0.0, 0.0])
    e = motion.relative_euler([motion._qmul(home, roll)], home)[0]
    np.testing.assert_allclose(e, [0.02, 0.0, 0.0], atol=1e-12)


def test_extremes_and_flat_signals_on_a_periodic_window():
    n = 400
    ph = np.arange(n) / n
    arrs = dict(phase=ph)
    sig = {k: np.zeros(n) for k, _, _ in motion.PARTS}
    sig["hip_height"] = 0.4 + 0.003 * np.cos(4 * np.pi * (ph - 0.3))  # maxima at 0.3 and 0.8
    sig["tail_yaw"] = 0.12 * np.sin(2 * np.pi * ph + np.pi / 2)  # maximum at 0, minimum at 0.5
    sig["sway"] = 0.02 * np.sin(2 * np.pi * (ph - 0.3) + np.pi / 2)  # maximum at 0.3: with the hip's first maximum
    ex = motion.extremes(arrs, sig)
    p = ex["parts"]
    np.testing.assert_allclose(p["hip_height"]["max"], [0.3, 0.8], atol=1.5 / n)
    np.testing.assert_allclose(p["hip_height"]["min"], [0.05, 0.55], atol=1.5 / n)
    np.testing.assert_allclose(p["tail_yaw"]["max"], [0.0], atol=1.5 / n)
    np.testing.assert_allclose(p["tail_yaw"]["min"], [0.5], atol=1.5 / n)
    assert p["trunk_roll"]["flat"] and p["trunk_roll"]["max"] == [] and p["left_foot_pitch"]["flat"]
    assert ["hip_height", "sway"] in ex["synchronous_pairs"]  # one of the hip's two maxima per stride, half of them


def test_crest_hold_of_a_sinusoid():
    assert motion.CREST_SINUSOID == pytest.approx(0.1436, abs=1e-4)
    th = np.linspace(0, 2 * np.pi, 100000, endpoint=False)
    z = np.cos(th)
    assert np.mean(z >= z.max() - motion.CREST_BAND * np.ptp(z)) == pytest.approx(motion.CREST_SINUSOID, abs=1e-3)


def test_bird_curves():
    b = motion.bird_curves(0.405, 0.05, "fr05")
    assert b["trex_cadence_steps_s"] == pytest.approx(1.83, abs=0.005)  # research Bottom line 5
    assert b["tinamou_stride_H"] == pytest.approx(0.98, abs=0.005)  # research section 2
    assert b["bird_swing_time_s"] == pytest.approx(0.32, abs=0.01)  # research F3: about 1.6 sqrt(H / g), 0.32 s
    assert "trex_cadence_steps_s" not in motion.bird_curves(0.502, 0.012, "nominal")
    assert motion.bird_curves(0.502, 0.05, "fr05")["trex_cadence_steps_s"] == pytest.approx(1.64, abs=0.005)


# ------------------------------------------------------------------------------------------------
# 2. On the ladder
# ------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ladder_060():
    if _F is None or not _F.exists():
        pytest.skip("the git-ignored ladder arrays are not here")
    out = {}
    for v in ("V0", "V5", "V6"):
        e = LADDER["bodies"]["60-N-T-10"]["v040"][v]
        out[v] = (e, motion.metrics("60-N-T-10", "v040", v, e, screen.load_arrays(e)))
    return out


@needs_arrays
def test_todays_walk_shows_every_mechanical_tell_and_v6_none(ladder_060):
    m0, m6 = ladder_060["V0"][1], ladder_060["V6"][1]
    t0, t6 = m0["mechanical_tells"], m6["mechanical_tells"]
    assert t0["constant_hip_height"] and t0["flat_feet_at_lift_off"] and t0["identical_steps"] and t0["metronome_nod"]
    assert not (t6["constant_hip_height"] or t6["flat_feet_at_lift_off"] or t6["identical_steps"] or t6["metronome_nod"])
    assert m0["bob"]["pp_mm"] < 1e-6 and m0["bob"]["crest_hold_share"] is None
    assert m0["trunk"]["roll_rms_deg"] < 1e-9 and m6["trunk"]["roll_rms_deg"] == pytest.approx(1.5 / math.sqrt(2), abs=0.01)
    assert m0["trunk"]["pitch_rms_deg"] < 1e-9 and m6["trunk"]["pitch_rms_deg"] < 1e-9  # F8: no trunk pitch
    assert m0["head"]["stabilisation_ratio"] is None and m0["head"]["trunk_angular_rate_rms_deg_s"] < 1e-9
    assert m6["head"]["stabilisation_ratio"] > 1.0  # the head rolls with the trunk (no neck roll) and settles
    assert m0["tail"]["phase_from_today_cycles"] == pytest.approx(0.0, abs=1e-6)
    assert abs(m6["tail"]["phase_from_today_cycles"]) > 0.4 and m6["tail"]["tip_toward_stance_in_swing"] == 1.0
    assert m0["step_times"]["cv"] < 1e-6 and 0.005 < m6["step_times"]["cv"] < 0.03 and m6["step_times"]["alternate"]
    assert m6["head"]["world_pitch_rms_deg"] < m0["head"]["world_pitch_rms_deg"]


@needs_arrays
def test_readings_shared_with_look_measure_equal_its(ladder_060):
    for v, (e, m) in ladder_060.items():
        rec = e["measured"]
        assert m["gait"]["swing_time_s"] == pytest.approx(rec["swing_time_s"], rel=1e-12)
        assert m["gait"]["duty"] == pytest.approx(rec["duty"], rel=1e-12)
        assert m["toe_off"]["at_lift_off_rad"] == pytest.approx(rec["toe_off_at_lift_off_rad"], rel=1e-12, abs=1e-15)
        assert m["tail"]["phase_rad"] == pytest.approx(rec["tail_phase_rad"], rel=1e-12)
        assert m["head"]["world_pitch_rms_deg"] == pytest.approx(rec["head_world_pitch_rms_deg"], rel=1e-9)
        assert m["bob"]["pp_H"] == pytest.approx(rec["bob_pp_H"], rel=1e-12, abs=1e-15)
        assert m["rung_sha256"] == e["sha256"], v


@needs_arrays
def test_feet_sparc_reads_each_swing(ladder_060):
    for _, m in ladder_060.values():
        f = m["smoothness"]["foot"]
        assert f["left"]["swings"] == f["right"]["swings"] == m["window"]["strides"]
        assert -2.5 < f["lowest"] <= f["mean"] < -1.2 and -2.5 < f["fixed_band_mean"] < -1.2
        assert -8 < m["smoothness"]["head"]["fixed_band"] < -1.9
    # F4's swing puts a sidelobe near 9 Hz at about 5 % (the published reading's threshold): over the fixed band the
    # feet's reading hardly moves along the ladder
    fb = [m["smoothness"]["foot"]["fixed_band_mean"] for _, m in ladder_060.values()]
    assert np.ptp(fb) < 0.1


@needs_arrays
def test_plots_are_written(tmp_path, ladder_060):
    arrs = {v: screen.load_arrays(e) for v, (e, _) in ladder_060.items()}
    motion.path_plot("t", [("V0", arrs["V0"], motion.GREY), ("V6", arrs["V6"], motion.BLUE)], tmp_path / "p.png")
    motion.phase_chart("t", [(v, arrs[v], m) for v, (_, m) in ladder_060.items()], tmp_path / "c.png")
    from PIL import Image

    for f in ("p.png", "c.png"):
        im = Image.open(tmp_path / f)
        assert im.size[0] > 600 and im.size[1] > 500


# ------------------------------------------------------------------------------------------------
# 3. The committed metrics.json
# ------------------------------------------------------------------------------------------------
@needs_metrics
def test_metrics_json_reads_every_rung_from_its_recorded_arrays():
    for b, speeds in LADDER["bodies"].items():
        for s, entries in speeds.items():
            for v, e in entries.items():
                m = METRICS["bodies"][b][s][v]
                assert m["rung_sha256"] == e["sha256"]
                assert m["features"] == e["features"]
                assert m["gait"]["swing_time_s"] == pytest.approx(e["measured"]["swing_time_s"], rel=1e-9)
                assert 0.5 < min(m["gait"]["duty"]) and max(m["gait"]["duty"]) < 0.75
                assert m["sway"]["com_pp_per_stride_max_H"] < 0.2
                assert -2.5 < m["smoothness"]["foot"]["mean"] < -1.2 and -8 < m["smoothness"]["head"]["sparc"] < -1.9
                tells = m["mechanical_tells"]
                assert tells["constant_hip_height"] == (v == "V0")
                assert tells["identical_steps"] == (v not in ("V6", "V6S"))
    assert METRICS["sparc"]["fc_hz"] == 10.0 and METRICS["bands_closed_loop"]["head_world_pitch_rms_deg"] == 1.5
    assert set(METRICS["v040"]["per_body"]) == set(look.BODIES)


@needs_metrics
def test_renders_md_regenerates_from_the_json_files():
    clips_doc = json.loads((LOOK / "clips.json").read_text()) if (LOOK / "clips.json").exists() else None
    text = motion.markdown(METRICS, clips_doc)
    assert (LOOK / motion.RENDERS_MD).read_text() == text
    for b in look.BODIES:
        for s in look.SPEEDS:
            assert f"### {motion.screen.BODY_LABEL[b]}, {motion.screen.SPEED_LABEL[s]}" in text
    if clips_doc is not None:
        assert all(f"`{e['file']}`" in text for e in clips_doc["clips"])


def test_summary_cells():
    m = dict(bob=dict(pp_mm=6.93), step_times=dict(cv=0.0149), gait=dict(duty=[0.57, 0.58]),
             stage0=dict(within_limits=False), head=dict(stabilisation_ratio=None))
    assert motion._cell(motion._get(m, ("bob", "pp_mm")), 1) == "6.9"
    assert motion._cell(motion._get(m, ("step_times", "cv")), 2, ("step_times", "cv")) == "1.49"
    assert motion._cell(motion._get(m, ("gait", "duty", 0)), 3) == "0.570"
    assert motion._cell(motion._get(m, ("stage0", "within_limits")), None) == "**over**"
    assert motion._cell(motion._get(m, ("head", "stabilisation_ratio")), 2) == "-"
    assert motion._cell(motion._get(m, ("tail", "phase_rad")), 2) == "-"
