"""Unit tests of gait_core on synthetic data: gait classification, support-polygon margins, debounced footfalls."""

import math

import numpy as np
import pytest

import gait_core as gc

FEET = ("right_fore", "left_hind", "right_hind")


def offsets(rf, lh, rh):
    return dict(zip(FEET, (rf, lh, rh)))


# ---------------------------------------------------------------- gait classification
@pytest.mark.parametrize("phases, cls, detail", [
    ((0.5, 0.5, 0.0), "trot", "trot"),
    ((0.5, 0.0, 0.5), "pace", "pace"),
    ((0.0, 0.5, 0.5), "bound", "bound"),
    ((0.5, 0.75, 0.25), "walk", "walk (lateral sequence)"),
    ((0.5, 0.25, 0.75), "walk", "walk (diagonal sequence)"),
    ((0.0, 0.0, 0.0), "other", "pronk"),
])
def test_ideal_patterns(phases, cls, detail):
    g = gc.classify_gait(offsets(*phases))
    assert g["class"] == cls and g["detail"] == detail and g["distance"] == pytest.approx(0.0)


def test_trot_within_tolerance_and_wraparound():
    # RH at 0.97 is 0.03 cycles from 0 across the wrap; LH 0.4 is 0.1 off
    g = gc.classify_gait(offsets(0.55, 0.40, 0.97))
    assert g["class"] == "trot"
    assert g["distance"] == pytest.approx(0.10)


def test_outside_tolerance_is_other_with_nearest():
    g = gc.classify_gait(offsets(0.43, 0.36, 0.95))  # pipe1-like: LH 0.14 off the trot template
    assert g["class"] == "other"
    assert g["nearest"] == "trot"
    assert g["distance"] == pytest.approx(0.14)


def test_tolerance_boundary_does_not_overlap():
    # LH halfway between trot (0.5) and the diagonal-sequence walk (0.25): exactly one template can be within 0.125
    for lh in np.linspace(0.0, 1.0, 81):
        g = gc.classify_gait(offsets(0.5, lh, (lh + 0.5) % 1.0))
        within = [n for n, t in gc.GAIT_TEMPLATES.items()
                  if max(gc.circ_dist(o, t[f]) for f, o in zip(FEET, (0.5, lh, (lh + 0.5) % 1.0))) < 0.125]
        assert len(within) <= 1


def test_unlocked_phase_is_other():
    g = gc.classify_gait(offsets(0.5, 0.5, 0.0), {"right_fore": 0.9, "left_hind": 0.3, "right_hind": 0.9})
    assert g["class"] == "other" and "not locked" in g["detail"] and "LH" in g["detail"]


def test_missing_offset_is_other():
    assert gc.classify_gait(offsets(0.5, float("nan"), 0.0))["class"] == "other"


def test_phase_offsets_and_circular_mean_from_touchdowns():
    lf = np.arange(0, 1000, 100)  # stride 100 samples
    rf = lf + 50
    rh = lf + 97  # lands 0.97 of a stride after LF, i.e. 0.03 before the next LF
    ph_rf = gc.phase_offsets(lf, rf)
    ph_rh = gc.phase_offsets(lf, rh)
    assert np.allclose(ph_rf, 0.5) and len(ph_rf) == 9
    mu, R = gc.circular_mean(np.r_[ph_rh, 0.01, 0.02])  # straddles the wrap
    assert R > 0.95 and gc.circ_dist(mu, 0.0) < 0.03


def test_diagonal_timing_of_a_perfect_trot_and_an_uneven_one():
    d = gc.diagonal_timing(offsets(0.5, 0.5, 0.0))
    assert d["limb_phase_left"] == pytest.approx(0.5) and d["limb_phase_right"] == pytest.approx(0.5)
    assert d["fore_after_diagonal_hind_LF_RH"] == pytest.approx(0.0)
    assert d["pair_interval_LFRH_to_RFLH"] == pytest.approx(0.5)
    u = gc.diagonal_timing(offsets(0.43, 0.37, 0.95))
    assert u["fore_after_diagonal_hind_LF_RH"] == pytest.approx(0.05)
    assert u["fore_after_diagonal_hind_RF_LH"] == pytest.approx(0.06)
    assert u["limb_phase_left"] == pytest.approx(0.63)


def test_symmetry_index():
    assert gc.symmetry_index(0.5, 0.5) == 0.0
    assert gc.symmetry_index(0.4, 0.6) == pytest.approx(40.0)
    assert gc.symmetry_index(None, 1.0) is None
    assert gc.symmetry_index(0.0, 0.0) is None


# ---------------------------------------------------------------- support polygon and margins
SQUARE = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)


@pytest.mark.parametrize("p, m", [((0.5, 0.5), 0.5), ((0.9, 0.5), 0.1), ((0.5, 0.05), 0.05), ((1.2, 0.5), -0.2),
                                  ((0.5, -0.3), -0.3), ((1.0, 0.5), 0.0)])
def test_margin_inside_and_outside_a_square(p, m):
    assert gc.support_margin(SQUARE, p) == pytest.approx(m)


def test_margin_beyond_a_corner_is_the_most_violated_edge():
    # Euclidean distance to the corner (1, 1) is 0.5; the shared signed_margin reports the larger edge excess, 0.4
    assert gc.support_margin(SQUARE, (1.3, 1.4)) == pytest.approx(-0.4)


def test_margin_is_order_independent_and_degenerate_support_is_nan():
    shuffled = SQUARE[[2, 0, 3, 1]]
    assert gc.support_margin(shuffled, (0.5, 0.5)) == pytest.approx(0.5)
    assert math.isnan(gc.support_margin(np.array([[0, 0], [1, 1]]), (0.5, 0.5)))


def test_sole_corners_and_margins_over_time_for_two_diagonal_feet():
    size = np.array([0.044, 0.032, 0.0072])
    eye = np.eye(3).ravel()
    yaw90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float).ravel()
    c = gc.sole_bottom_corners(np.array([0.0, 0.0, 0.0072]), eye, size)
    assert c.shape == (4, 3)
    assert np.allclose(np.sort(c[:, 0]), [-0.044, -0.044, 0.044, 0.044]) and np.allclose(c[:, 2], 0.0)
    r = gc.sole_bottom_corners(np.array([0.0, 0.0, 0.0072]), yaw90, size)
    assert np.allclose(np.sort(np.abs(r[:, 0])), [0.032] * 4) and np.allclose(np.sort(np.abs(r[:, 1])), [0.044] * 4)
    # four feet at the corners of a 0.6 x 0.3 m rectangle; three samples: two diagonal feet, all four, none
    centres = np.array([[0.3, 0.15, 0.0072], [0.3, -0.15, 0.0072], [-0.3, 0.15, 0.0072], [-0.3, -0.15, 0.0072]])
    corners = gc.sole_bottom_corners(centres, np.tile(eye, (4, 1)), size)[None].repeat(3, 0)[..., :2]
    down = np.array([[1, 0, 0, 1], [1, 1, 1, 1], [0, 0, 0, 0]], dtype=bool)  # LF + RH, all, none
    com = np.zeros((3, 2))
    off = np.tile([0.0, 0.1], (3, 1))
    m_com, m_off = gc.margins_over_time(corners, down, com, off)
    assert m_com[0] > 0  # the centre lies on the LF-RH diagonal strip
    assert m_off[0] < 0  # 0.1 m to the left of it does not
    assert m_com[1] == pytest.approx(0.15 + 0.032)  # all four feet: nearest edge is the side
    assert math.isnan(m_com[2])


# ---------------------------------------------------------------- debounced footfalls
def test_debounce_fills_short_gaps_only_after_first_landing():
    raw = np.array([0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0], dtype=bool)
    deb = gc.debounce_contact(raw, 3)
    # the leading unloaded run stays (first landing), the 2-sample gap is filled, the 4-sample gap is kept, the
    # trailing 1-sample gap is filled (the env has not yet counted the foot as lifted)
    assert deb.astype(int).tolist() == [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1]
    td = gc.touchdowns(deb)
    assert td.tolist() == [13]  # the first landing at 3 is not a touchdown
    assert gc.liftoffs(deb).tolist() == [9]


def test_debounce_keeps_a_short_blip_after_a_real_lift():
    raw = np.r_[np.ones(20), np.zeros(15), np.ones(2), np.zeros(15), np.ones(20)].astype(bool)
    deb = gc.debounce_contact(raw, 10)
    assert gc.touchdowns(deb).tolist() == [35, 52]  # the 2-sample blip counts, as it does for the env


def test_step_contact_merges_rebounds_and_drops_blips():
    stance, rebound, swing, blip = np.ones(200), np.zeros(25), np.zeros(150), np.ones(30)
    raw = np.r_[np.zeros(5), stance, rebound, stance, swing[:60], blip, swing[60:], stance].astype(bool)
    stp = gc.step_contact(raw, 10, 50)
    # one step-level touchdown: the rebound (25 samples) is merged, the 30-sample blip inside the swing is dropped
    assert gc.touchdowns(stp).tolist() == [5 + 200 + 25 + 200 + 150 + 30]
    assert stp[5 + 200 + 10]  # inside the merged rebound
    assert not stp[5 + 425 + 70]  # inside the dropped blip
    # the env's 10 ms rule alone counts the rebound and the blip as touchdowns
    assert len(gc.touchdowns(gc.debounce_contact(raw, 10))) == 3


def test_touchdown_count_equals_the_training_evaluation_rule():
    gm = pytest.importorskip("gait_metrics")  # rl/gait_metrics.py (imports the current trike_env)
    rng = np.random.default_rng(1)
    for _ in range(20):
        n, start = int(rng.integers(200, 2000)), int(rng.integers(0, 150))
        # bursts of contact and gaps of mixed lengths, starting unloaded like the env (soles 2 mm up)
        x = np.zeros((n, 4), dtype=bool)
        for f in range(4):
            i = int(rng.integers(1, 15))
            while i < n:
                on, off = int(rng.integers(1, 60)), int(rng.integers(1, 30))
                x[i:i + on, f] = True
                i += on + off
        ours = gc.touchdown_count(x, 10, start)
        theirs = gm.debounced_touchdowns(x, 10, start)
        assert ours.tolist() == theirs.tolist()


def test_segments_exclude_runs_touching_the_ends():
    x = np.array([1, 1, 0, 0, 1, 1, 1, 0, 1], dtype=bool)
    assert gc.segments(x, 0, None, True) == [(4, 7)]
    assert gc.segments(x, 0, None, False) == [(2, 4), (7, 8)]
    assert gc.segments(x, 3, None, False) == [(7, 8)]


def test_stride_resample():
    sig = np.arange(1000, dtype=float)[:, None].repeat(2, 1)
    prof = gc.stride_resample(sig, np.array([100, 200, 400]), 11)
    assert prof.shape == (2, 11, 2)
    assert np.allclose(prof[0, :, 0], np.linspace(100, 200, 11))
    assert np.allclose(prof[1, :, 1], np.linspace(200, 400, 11))
