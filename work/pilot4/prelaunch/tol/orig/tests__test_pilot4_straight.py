"""Tests of pilot4_straight.py (pytest; walking pilot 4 scratch, never merged), as v3's TOOL PLAN lists them.

* Pure (no tree): synthetic heading traces with the plan's exact expectations (constant, drift, sine weave, the 1.01-s
  weave, the window edges, the wrap past +-180 deg, a fall); the percentile rule with inf entries; the bars at their
  edges; the counting rule, the movement condition and NOT READ below 30; the turn's mechanics on a stand-in MjData
  (a 0-deg turn writes back every value unchanged; +-15 deg turns the quaternion, the linear velocity and the prey);
  the refusals (seed, baseline, env, existing outputs) and the strict-JSON record.
* Tree (skipped when the pilot-4 tree or s42's 8M parent is absent): the probe's own yaw formula equals the tool's copy;
  the baseline mode on the first two recovery resets through iso.py: states 0..650 of every nudged episode equal the
  panel episode's exactly, a 0-deg nudge reproduces the panel episode exactly, the state-651 observation differs from
  the un-nudged one beyond 1e-3 only in obs[60:63], the command-blind walker's nudged heading at state 1000 is the
  panel heading + angle within 1e-5 deg, the pass check at state 650, the prepared walker acts as the blind parent.
* Validation records (skipped unless STRAIGHT_VAL names the folder holding them): the runs on pilot 3's 8M cells
  (pilot 3's tree, --validation) reproduce probe.json's yaw_change_deg to 1e-12 and the draft's numbers; the entry
  baseline reproduces v3's entry rows and (c) baseline.
Run: python -I -B -m pytest -p no:cacheprovider tests/test_pilot4_straight.py (nothing here trains); the tree tests are
named test_tree_*, the record tests test_valrec_* (-k "not tree_ and not valrec_" keeps the pure ones).
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import pilot4_straight as ps  # noqa: E402

TREE = Path(os.environ.get("STRAIGHT_TREE", str(TOOLS / "tree")))
WORK = TOOLS.parents[1]
P3_RUNS = Path(os.environ.get("STRAIGHT_P3_RUNS", str(WORK / "pilot3" / "runs" / "final")))
P3_CELLS = Path(os.environ.get("STRAIGHT_P3_CELLS", str(WORK / "pilot3" / "judging" / "OUT")))
PROBE = TREE / "docs" / "investigations" / "gait_2026_09" / "gait_probe.py"
DT = 0.01


def deg_trace(fn, T=1000):
    """psi in radians over states 0..T from a function of time in seconds returning degrees."""
    t = np.arange(T + 1) * DT
    return np.radians(np.array([fn(x) for x in t], dtype=np.float64))


def gain(P):
    """The 101-state mean's gain on a sine of period P seconds (v3 TOOL PLAN)."""
    return math.sin(1.01 * math.pi / P) / (101 * math.sin(0.01 * math.pi / P))


# ================================================================ synthetic traces (exact expectations)


def test_a_constant_heading_has_no_deviation():
    m = ps.episode_metrics(deg_trace(lambda t: 0.0), 0.0, 1000)
    assert m["full"] and m["D_deg"] == 0.0 and m["W0_deg"] == 0.0 and m["R_deg"] == 0.0
    assert m["yaw_from_psi_deg"] == 0.0 and m["last_mean_error_deg"] == 0.0
    m = ps.episode_metrics(deg_trace(lambda t: 7.25), math.radians(7.25), 1000)
    assert abs(m["D_deg"]) < 1e-12 and abs(m["W0_deg"]) < 1e-12 and abs(m["R_deg"]) < 1e-12


@pytest.mark.parametrize("r", [1.0, -1.3, 0.5])
def test_a_drift_of_r_deg_per_s_reads_d_9_5_r_at_state_950(r):
    m = ps.episode_metrics(deg_trace(lambda t: r * t), 0.0, 1000)
    assert m["D_deg"] == pytest.approx(9.5 * abs(r), abs=1e-9) and m["D_k"] == 950
    assert m["W0_deg"] == pytest.approx(10.0 * abs(r), abs=1e-9) and m["W0_state"] == 1000
    assert m["R_deg"] == pytest.approx(7.99 * abs(r), abs=1e-9)  # m_950 - m_151
    assert m["yaw_from_psi_deg"] == pytest.approx(8.99 * r, abs=1e-9)  # psi[1000] - psi[101]
    assert m["last_mean_error_deg"] == pytest.approx(9.5 * r, abs=1e-9)


def test_the_gain_of_the_101_state_mean():
    assert gain(2.0) == pytest.approx(0.630, abs=5e-4) and gain(4.0) == pytest.approx(0.898, abs=5e-4)


@pytest.mark.parametrize("P", [2.0, 3.0, 4.0, 6.0, 8.0])
@pytest.mark.parametrize("A", [10.0, 25.0])
def test_a_sine_weave_reads_d_g_a_and_r_2_g_a(P, A):
    t_peak = 5.0  # the 1-s mean peaks on state 500; its trough P/2 later, still inside k = 151..950 for P <= 8 s
    m = ps.episode_metrics(deg_trace(lambda t: A * math.sin(2 * math.pi * (t - t_peak) / P + math.pi / 2)), 0.0, 1000)
    assert m["D_deg"] == pytest.approx(gain(P) * A, abs=1e-9)
    assert m["R_deg"] == pytest.approx(2 * gain(P) * A, abs=1e-9)
    assert m["W0_deg"] == pytest.approx(A, abs=1e-9)  # a sample sits on every peak


@pytest.mark.parametrize("A", [5.0, 18.5, 40.0])
def test_a_1_01_s_weave_reads_d_0_and_w0_a(A):
    m = ps.episode_metrics(deg_trace(lambda t: A * math.sin(2 * math.pi * (t - 2.0) / 1.01 + math.pi / 2)), 0.0, 1000)
    assert m["D_deg"] < 1e-9
    assert m["W0_deg"] == pytest.approx(A, abs=1e-9)


def spike(state, size=101.0, T=1000):
    psi = np.zeros(T + 1)
    psi[state] = size
    return np.radians(psi)


def test_the_window_edges():
    m = ps.episode_metrics(spike(100), 0.0, 1000)  # state 100 is outside the window and outside every m_k
    assert m["D_deg"] == 0.0 and m["W0_deg"] == 0.0 and m["yaw_from_psi_deg"] == 0.0
    m = ps.episode_metrics(spike(101), 0.0, 1000)  # the window's first state: W0, m_151 only, and the yaw
    assert m["W0_deg"] == pytest.approx(101.0) and m["W0_state"] == 101
    assert m["D_deg"] == pytest.approx(1.0) and m["D_k"] == 151
    assert m["yaw_from_psi_deg"] == pytest.approx(-101.0)
    m = ps.episode_metrics(spike(1000), 0.0, 1000)  # the last state: W0, m_950 only, and the yaw
    assert m["W0_state"] == 1000 and m["D_k"] == 950 and m["D_deg"] == pytest.approx(1.0)
    assert m["yaw_from_psi_deg"] == pytest.approx(101.0)
    m = ps.episode_metrics(spike(201), 0.0, 1000)  # inside m_151 (its last state) and m_152..m_251
    assert m["D_deg"] == pytest.approx(1.0) and m["D_k"] == 151
    ks, means = ps.one_s_means(np.degrees(spike(201)), 1000)
    assert ks[0] == 151 and ks[-1] == 950 and len(ks) == 800
    assert np.count_nonzero(means) == 101  # k = 151..251


def quat_of_heading(deg):
    a = math.radians(deg)
    return np.array([math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)])


def test_the_wrap_past_180_deg():
    heading = [179.0 + 1.0 * k * DT for k in range(1001)]  # crosses +180 at 1 s; raw atan2 jumps to -180
    raw = [ps.yaw_from_quat(quat_of_heading(h)) for h in heading]
    assert min(raw) < -170 * math.pi / 180  # the raw yaw did wrap
    psi = ps.unwrap_heading(raw, math.radians(179.0))
    m = ps.episode_metrics(psi, math.radians(179.0), 1000)
    assert m["D_deg"] == pytest.approx(9.5, abs=1e-6) and m["W0_deg"] == pytest.approx(10.0, abs=1e-6)
    # the commanded heading given on the other side of the cut: psi starts on psi_cmd's branch
    psi = ps.unwrap_heading([math.radians(179.9)] * 1001, math.radians(-179.9))
    assert math.degrees(psi[0]) == pytest.approx(-180.1, abs=1e-9)
    m = ps.episode_metrics(psi, math.radians(-179.9), 1000)
    assert m["D_deg"] == pytest.approx(0.2, abs=1e-9) and m["W0_deg"] == pytest.approx(0.2, abs=1e-9)
    # a heading already on psi_cmd's branch is not touched (no arithmetic)
    raw = np.radians(np.linspace(-3.0, 4.0, 1001))
    assert np.array_equal(ps.unwrap_heading(raw, 0.0), np.unwrap(raw))


def test_a_fall_has_no_d():
    m = ps.episode_metrics(deg_trace(lambda t: 3.0 * t, T=600), 0.0, 600)
    assert not m["full"] and m["D_deg"] is None and m["W0_deg"] is None and m["R_deg"] is None
    assert m["yaw_from_psi_deg"] == pytest.approx(3.0 * (6.0 - 1.01), abs=1e-9)
    m = ps.episode_metrics(deg_trace(lambda t: 0.0, T=105), 0.0, 105)
    assert m["yaw_from_psi_deg"] is None  # the probe forms no yaw_change_deg below 110 steps
    assert ps.part_a([])["verdict"] == "NOT READ" and ps.part_b([], [])["verdict"] == "NOT READ"


def test_the_crab_and_cross_track_report():
    T = 1000
    xy = np.stack([np.arange(T + 1) * 0.025, np.arange(T + 1) * 0.025 * math.tan(math.radians(4.0))], axis=1)
    m = ps.episode_metrics(np.zeros(T + 1), 0.0, T, xy)  # heading 0, course +4 deg: a crab of +4 deg
    assert m["crab_median_deg"] == pytest.approx(4.0, abs=1e-9)
    assert m["end_cross_track_m"] == pytest.approx(25.0 * math.tan(math.radians(4.0)))


# ================================================================ the percentile rule


def forty(n_inf, n=40):
    return [0.1 * i for i in range(n - n_inf)] + [math.inf] * n_inf


def test_one_inf_of_40_passes_a_p95_bar_and_two_fail_it():
    assert ps.percentile(forty(0), 95) == pytest.approx(np.percentile(forty(0), 95))
    p1 = ps.percentile(forty(1), 95)
    assert math.isfinite(p1) and p1 == pytest.approx(np.percentile([0.1 * i for i in range(39)] + [99.0], 95))
    assert ps.percentile(forty(2), 95) == math.inf
    assert ps.percentile(forty(3), 95) == math.inf  # numpy: nan (both neighbours inf)


def test_n_30_with_2_inf_reads_inf_where_numpy_returns_nan():
    vals = forty(2, n=30)
    with np.errstate(invalid="ignore"):
        assert math.isnan(float(np.percentile(vals, 95)))  # position 27.55: the upper neighbour inf, fraction 0.55
    assert ps.percentile(vals, 95) == math.inf


def test_an_integer_position_with_an_inf_above_reads_inf():
    # pinned (text issue): numpy interpolates at fraction 0 too, inf * 0 = nan, read inf, although the order statistic
    # at the position is finite (n 41 with 2 inf: p95 at position 38; n 31 with 15 inf: the median at position 15)
    assert ps.percentile(forty(2, n=41), 95) == math.inf
    assert ps.percentile(forty(15, n=31), 50) == math.inf
    assert ps.percentile(forty(14, n=31), 50) == pytest.approx(1.5)  # x[15] (17 finite: 0.0..1.6), x[16] = 1.6


def test_the_median_is_the_50th_percentile_and_inf_sorts_last():
    assert ps.percentile([3.0, math.inf, 1.0, 2.0], 50) == pytest.approx(2.5)
    assert ps.percentile(forty(20), 50) == math.inf
    assert ps.percentile(forty(19), 50) == pytest.approx((1.9 + 2.0) / 2)
    assert ps.percentile([], 50) is None
    with pytest.raises(ValueError):
        ps.percentile([1.0, math.nan], 50)


# ================================================================ the bars at their edges


def test_a_bars_at_their_edges():
    assert ps.part_a([5.0] * 40)["verdict"] == "PASS"
    assert ps.part_a([-5.0] * 40)["verdict"] == "PASS"
    assert ps.part_a([5.0 + 1e-9] * 40)["verdict"] == "FAIL"
    edge = [0.0] * 21 + [15.0] * 19  # median 0, p95 exactly 15
    a = ps.part_a(edge)
    assert a["p95_abs_deg"] == 15.0 and a["verdict"] == "PASS"
    assert ps.part_a([0.0] * 21 + [-15.0 - 1e-9] * 19)["verdict"] == "FAIL"  # |yaw| is what p95 reads
    assert ps.A_MEDIAN_ABS_MAX_DEG == 5.0 and ps.A_P95_ABS_MAX_DEG == 15.0


def test_b_bars_at_their_edges():
    assert ps.part_b([11.5] * 40, [18.5] * 40)["verdict"] == "PASS"
    assert ps.part_b([11.5 + 1e-9] * 40, [18.5] * 40)["verdict"] == "FAIL"
    assert ps.part_b([11.5] * 40, [18.5 + 1e-9] * 40)["verdict"] == "FAIL"
    b = ps.part_b([1.0] * 38 + [30.0] * 2, [2.0] * 40)  # 2 of 40 far out: p95 interpolates past the bar
    assert b["D_p95_deg"] == pytest.approx(1.0 + 0.05 * 29.0) and b["verdict"] == "PASS"
    assert ps.part_b([1.0] * 37 + [30.0] * 3, [2.0] * 40)["verdict"] == "FAIL"
    assert ps.B_D_P95_MAX_DEG == 11.5 and ps.B_W0_P95_MAX_DEG == 18.5
    with pytest.raises(ValueError):
        ps.part_b([1.0], [1.0, 2.0])


def creading(r, angle=15.0, share=0.0):
    return {"counting": True, "reached_nudge": True, "passed_prey_at_650": True, "r_deg": r, "angle_deg": angle,
            "share_removed": share, "r_reason": None if math.isfinite(r) else "ended before state 1000 (T = 900)"}


def test_c_bars_at_their_edges_and_not_read_below_30():
    edge = [creading(0.0)] * 21 + [creading(11.5)] * 19  # median 0, p95 exactly 11.5
    c = ps.part_c(edge)
    assert c["p95_r_deg"] == 11.5 and c["verdict"] == "PASS"
    assert ps.part_c([creading(5.0)] * 40)["verdict"] == "PASS"
    assert ps.part_c([creading(5.0 + 1e-9)] * 40)["verdict"] == "FAIL"
    assert ps.part_c([creading(0.0)] * 21 + [creading(11.5 + 1e-9)] * 19)["verdict"] == "FAIL"
    assert ps.part_c([creading(1.0)] * 38 + [creading(math.inf)] * 2)["verdict"] == "FAIL"  # 2 inf of 40
    assert ps.part_c([creading(1.0)] * 39 + [creading(math.inf)] * 1)["verdict"] == "PASS"
    out = [{"counting": False, "reached_nudge": True, "passed_prey_at_650": False, "r_deg": None, "angle_deg": 15.0,
            "share_removed": None, "r_reason": "prey not yet passed at state 650"}]
    c = ps.part_c([creading(1.0)] * 29 + out * 11)
    assert c["verdict"] == "NOT READ" and c["n_counting"] == 29 and c["left_out_not_yet_passed"] == 11
    assert "prey not yet passed at 6.5 s in 11 of 40" in c["why"]
    assert ps.part_c([creading(1.0)] * 30 + out * 10)["verdict"] == "PASS"
    assert ps.C_MIN_COUNTING == 30 and ps.C_MEDIAN_MAX_DEG == 5.0 and ps.C_P95_MAX_DEG == 11.5


def test_straight_needs_three_passes_and_a_not_read_c_is_not_a_pass():
    p, f, n = {"verdict": "PASS"}, {"verdict": "FAIL"}, {"verdict": "NOT READ"}
    assert ps.straight_verdict(p, p, p) == "PASS"
    assert ps.straight_verdict(p, p, n) == "FAIL"
    assert ps.straight_verdict(p, f, p) == "FAIL" and ps.straight_verdict(f, p, p) == "FAIL"


# ================================================================ the recovery reading (counting rule, movement, r)


def recovery_ep(T=1000, err_after=0.0, x650=10.0, prey_x=8.0, speed=2.5, turned=True, cmd=0.0):
    psi = np.zeros(T + 1)
    psi[ps.NUDGE_STATE + 1:] = err_after
    x = np.arange(T + 1) * speed * DT
    x = x - x[min(ps.NUDGE_STATE, T)] + x650
    xy = np.stack([x, np.zeros(T + 1)], axis=1)
    return {"T": T, "turned": turned, "psi_rad": np.radians(psi) + cmd, "psi_cmd_rad": cmd, "root_xy": xy,
            "prey_x_at_turn": prey_x}


def test_the_pass_check_at_state_650():
    r = ps.recovery_reading(recovery_ep(x650=10.0, prey_x=8.0), None, 15.0)
    assert r["counting"] and r["passed_prey_at_650"] and r["pelvis_minus_prey_x_650_m"] == pytest.approx(2.0)
    r = ps.recovery_reading(recovery_ep(x650=8.0, prey_x=8.0), None, 15.0)  # level is not past
    assert not r["counting"] and r["passed_prey_at_650"] is False
    assert r["r_reason"] == "prey not yet passed at state 650"
    r = ps.recovery_reading(recovery_ep(T=640, turned=False), None, 15.0)
    assert not r["counting"] and not r["reached_nudge"] and r["r_reason"] == "ended before the nudge"
    c = ps.part_c([ps.recovery_reading(recovery_ep(T=90, turned=False), None, s) for s in (15.0, -15.0)]
                  + [ps.recovery_reading(recovery_ep(), None, 15.0)] * 38)
    assert c["n_counting"] == 38 and c["left_out_ended_before_nudge"] == 2 and c["verdict"] == "PASS"


def test_r_is_the_m950_error_and_inf_after_a_fall_or_without_movement():
    r = ps.recovery_reading(recovery_ep(err_after=-3.25), None, 15.0)
    assert r["r_deg"] == pytest.approx(3.25) and r["m_read_error_deg"] == pytest.approx(-3.25)
    r = ps.recovery_reading(recovery_ep(err_after=-3.25, cmd=0.4), None, 15.0)  # read against psi_cmd
    assert r["r_deg"] == pytest.approx(3.25)
    r = ps.recovery_reading(recovery_ep(T=999), None, 15.0)
    assert r["counting"] and r["r_deg"] == math.inf and r["r_reason"].startswith("ended before state 1000")
    r = ps.recovery_reading(recovery_ep(speed=0.34 / 3.5), None, 15.0)  # 0.34 m in 3.5 s
    assert r["r_deg"] == math.inf and r["r_reason"].startswith("pelvis moved")
    ep = recovery_ep(prey_x=-1.0)
    ep["root_xy"] = np.zeros((1001, 2))
    ep["root_xy"][1000] = (0.35, 0.0)  # exactly 0.35 m net between states 650 and 1000: not below the bar
    r = ps.recovery_reading(ep, None, 15.0)
    assert r["moved_650_1000_m"] == 0.35 and math.isfinite(r["r_deg"])
    ep["root_xy"][1000] = (0.21, 0.28)  # also 0.35 m, sideways: the net horizontal displacement is what counts
    assert ps.recovery_reading(ep, None, 15.0)["moved_650_1000_m"] == pytest.approx(0.35, abs=1e-15)
    assert ps.C_MIN_NET_MOVE_M == 0.35


def test_r_reads_m950_states_900_to_1000_only():
    """Fix stage (review: a read at m_900 survived the pure tests): only states 900..1000 hold the error, and the
    states just outside the window hold another; r is the window's error exactly."""
    ep = recovery_ep()
    psi = np.zeros(1001)
    psi[900:] = 4.0       # only m_950's window holds this error
    psi[899] = 101.0      # the state just before the window
    psi[651:899] = -50.0  # the rest after the nudge
    ep["psi_rad"] = np.radians(psi)
    r = ps.recovery_reading(ep, None, 15.0)
    assert r["r_deg"] == pytest.approx(4.0, abs=1e-12) and r["m_read_error_deg"] == pytest.approx(4.0, abs=1e-12)
    assert ps.READ_K == 950 and ps.MEAN_HALF == 50


def test_the_share_removed_against_the_unturned_twin():
    twin = recovery_ep(err_after=-10.0)
    keep = ps.recovery_reading(recovery_ep(err_after=5.0), twin, 15.0)  # kept the offset: e_n = e_t + 15
    assert keep["share_removed"] == pytest.approx(0.0)
    back = ps.recovery_reading(recovery_ep(err_after=-10.0), twin, 15.0)  # back on the twin's heading
    assert back["share_removed"] == pytest.approx(1.0)
    minus = ps.recovery_reading(recovery_ep(err_after=-25.0), twin, -15.0)
    assert minus["share_removed"] == pytest.approx(0.0)
    c = ps.part_c([keep] * 20 + [minus] * 20)
    assert c["share_removed_median"] == {"plus": pytest.approx(0.0), "minus": pytest.approx(0.0)}


def test_the_reset_variant_reads_m500_and_counts_every_episode():
    ep = recovery_ep(err_after=0.0)
    ep["psi_rad"] = np.radians(np.full(1001, 2.0))
    ep["turned"] = True
    r = ps.recovery_reading(ep, None, 15.0, read_k=ps.RESET_READ_K, pass_rule=False, movement=False)
    assert r["counting"] and r["r_deg"] == pytest.approx(2.0)
    r = ps.recovery_reading(recovery_ep(T=549), None, 15.0, read_k=500, pass_rule=False, movement=False)
    assert r["r_deg"] == math.inf
    s = ps.variant_summary([r, ps.recovery_reading(ep, None, 15.0, read_k=500, pass_rule=False, movement=False)])
    assert s["infinite"] == 1 and s["n_counting"] == 2


# ================================================================ the turn's mechanics


class FakeData:
    def __init__(self, rng):
        q = rng.normal(size=4)
        self.qpos = np.concatenate([rng.normal(size=3), q / np.linalg.norm(q), rng.normal(size=10)])
        self.qvel = rng.normal(size=16)
        self.mocap_pos = rng.normal(size=(1, 3)) * 10

    def copy(self):
        d = FakeData.__new__(FakeData)
        d.qpos, d.qvel, d.mocap_pos = self.qpos.copy(), self.qvel.copy(), self.mocap_pos.copy()
        return d


def euler(q):
    w, x, y, z = q
    return (math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)), math.asin(max(-1, min(1, 2 * (w * y - z * x)))),
            ps.yaw_from_quat(q))


@pytest.mark.parametrize("turn_prey", [True, False])
def test_a_0_deg_turn_writes_every_value_back_unchanged(turn_prey):
    rng = np.random.default_rng(0)
    for _ in range(50):
        d = FakeData(rng)
        before = d.copy()
        ps.apply_turn(d, 0.0, turn_prey)
        for name in ("qpos", "qvel", "mocap_pos"):
            assert getattr(d, name).tobytes() == getattr(before, name).tobytes()


@pytest.mark.parametrize("angle", [15.0, -15.0, 90.0])
def test_the_turn_rotates_heading_velocity_and_prey_about_the_root(angle):
    rng = np.random.default_rng(1)
    for _ in range(20):
        d = FakeData(rng)
        d.qpos[3:7] = quat_mul_tilt(rng)
        before = d.copy()
        ps.apply_turn(d, angle, True)
        a = math.radians(angle)
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        r0, p0 = before.qpos[0:2], before.mocap_pos[0, 0:2]
        assert np.array_equal(d.qpos[0:3], before.qpos[0:3])  # the root does not move
        assert np.array_equal(d.qpos[7:], before.qpos[7:]) and np.array_equal(d.qvel[2:], before.qvel[2:])
        assert np.allclose(d.qvel[0:2], R @ before.qvel[0:2], atol=1e-12, rtol=0)
        assert np.allclose(d.mocap_pos[0, 0:2], r0 + R @ (p0 - r0), atol=1e-12, rtol=0)
        assert d.mocap_pos[0, 2] == before.mocap_pos[0, 2]
        e0, e1 = euler(before.qpos[3:7]), euler(d.qpos[3:7])
        assert ps.wrap_rad(e1[2] - e0[2] - a) == pytest.approx(0.0, abs=1e-12)  # + is counter-clockwise from above
        assert e1[0] == pytest.approx(e0[0], abs=1e-12) and e1[1] == pytest.approx(e0[1], abs=1e-12)
        d2 = before.copy()
        ps.apply_turn(d2, angle, False)
        assert np.array_equal(d2.mocap_pos, before.mocap_pos)  # the prey-in-place variant


def quat_mul_tilt(rng):
    """A pelvis attitude with some roll and pitch."""
    roll, pitch, yaw = rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), rng.uniform(-math.pi, math.pi)
    qx = np.array([math.cos(roll / 2), math.sin(roll / 2), 0, 0])
    qy = np.array([math.cos(pitch / 2), 0, math.sin(pitch / 2), 0])
    return ps.quat_mul(ps.quat_z(yaw), ps.quat_mul(qy, qx))


# ================================================================ refusals and the record


def p3_cell(seed=42, steps=8000000):
    cell = P3_CELLS / f"s{seed}" / str(steps)
    if not (cell / "binding.json").is_file():
        pytest.skip("pilot 3's judged cells are not here")
    binding = json.loads((cell / "binding.json").read_text())
    doc = json.loads((cell / "probe.json").read_text())
    return cell, binding, doc["meta"], doc["episodes"]


def test_a_pilot3_cell_is_refused_without_validation_and_read_with_it():
    _, binding, meta, rows = p3_cell()
    kw = dict(seed=42, run=binding["run"], head=binding["head"], dirty="")
    assert ps.seed_problems(binding, meta, len(rows), validation=True, **kw) == []
    problems = ps.seed_problems(binding, meta, len(rows), validation=False, **kw)
    assert problems and all("another run" in p for p in problems)


def synthetic_binding(tmp_path, run_name="pilot4_walk_s42", **over):
    run = tmp_path / run_name
    models = run / "03_locomotion" / "models"
    models.mkdir(parents=True)
    ck, vn = models / "stage2_2000000_steps.zip", models / "stage2_vecnormalize_2000000_steps.pkl"
    ck.write_bytes(b"zip")
    vn.write_bytes(b"pkl")
    binding = {"schema": "walking-pilot4-judge/v1", "seed": 42, "steps": 2000000, "head": "H", "run": str(run),
               "checkpoint": str(ck), "checkpoint_sha256": ps.sha256_file(ck), "vecnorm": str(vn),
               "vecnorm_sha256": ps.sha256_file(vn), "off_schedule": False, "validation_only": False,
               "expected_budget": 2000000, **over}
    meta = {"seed_scheme": "panel", "seed_base": 3042, "episodes": 40, "settle_steps_gait_window": 100,
            "checkpoint_sha256": binding["checkpoint_sha256"]}
    return run, binding, meta


def test_seed_problems_accepts_a_pilot4_cell_and_lists_every_problem(tmp_path):
    run, binding, meta = synthetic_binding(tmp_path)
    assert ps.seed_problems(binding, meta, 40, seed=42, run=run, head="H", dirty="") == []
    assert ps.seed_file_problems(binding) == []
    b = dict(binding, off_schedule=True, off_schedule_reason="resumed off-schedule", validation_only=True)
    problems = ps.seed_problems(b, meta, 40, seed=52, run=run, head="G", dirty=" M environments/x.py")
    text = " | ".join(problems)
    for needle in ("not --seed 52", "this tree is G", "local edits", "off-schedule", "validation-only",
                   "not pilot4_walk_s52"):
        assert needle in text
    assert ps.seed_problems(dict(binding, validation_only=True), meta, 40, seed=42, run=run, head="H", dirty="",
                            validation=True) == []


def test_a_cell_from_another_run_is_refused(tmp_path):
    run, binding, meta = synthetic_binding(tmp_path)
    other, _, _ = synthetic_binding(tmp_path / "x", run_name="pilot4_walk_s42_b")
    problems = ps.seed_problems(binding, meta, 40, seed=42, run=other, head="H", dirty="")
    assert any("not --run" in p for p in problems) and any("is not in" in p for p in problems)
    run52, b52, m52 = synthetic_binding(tmp_path / "y", run_name="pilot4_walk_s52")
    assert any("not pilot4_walk_s42" in p for p in ps.seed_problems(dict(b52, seed=42), m52, 40, seed=42, run=run52,
                                                                     head="H", dirty=""))
    p3run, b3, m3 = synthetic_binding(tmp_path / "z", run_name="pilot3_walk_s42")
    assert any("another run" in p for p in ps.seed_problems(b3, m3, 40, seed=42, run=p3run, head="H", dirty=""))
    assert ps.seed_problems(b3, m3, 40, seed=42, run=p3run, head="H", dirty="", validation=True) == []
    Path(binding["checkpoint"]).write_bytes(b"changed")
    assert ps.seed_file_problems(binding) and "no longer the judged file" in ps.seed_file_problems(binding)[0]


def test_a_probe_json_that_is_not_the_fixed_panel_is_refused(tmp_path):
    run, binding, meta = synthetic_binding(tmp_path)
    for bad, needle in (({"seed_scheme": "vecenv"}, "not panel"), ({"seed_base": 3000}, "not panel"),
                        ({"episodes": 20}, "not 40"), ({"settle_steps_gait_window": 200}, "not 100"),
                        ({"checkpoint_sha256": "sha256:0"}, "another checkpoint")):
        problems = ps.seed_problems(binding, dict(meta, **bad), 40, seed=42, run=run, head="H", dirty="")
        assert any(needle in p for p in problems), (bad, problems)
    assert any("not 40" in p for p in ps.seed_problems(binding, meta, 39, seed=42, run=run, head="H", dirty=""))


def test_a_probe_json_with_an_env_kwargs_drift_is_refused(tmp_path):
    """Fix stage (review: a run whose recorded env kwargs differ from the tree's was read): probe.json's own drift
    record must be empty (or absent, in a probe.json written without the run's record); also under --validation."""
    run, binding, meta = synthetic_binding(tmp_path)
    assert ps.seed_problems(binding, dict(meta, env_kwargs_drift_vs_recorded={}), 40, seed=42, run=run, head="H",
                            dirty="") == []
    drift = {"forward_vel_weight": {"toml_main": 2.0, "recorded_run": 3.0}}
    for validation in (False, True):
        problems = ps.seed_problems(binding, dict(meta, env_kwargs_drift_vs_recorded=drift), 40, seed=42, run=run,
                                    head="H", dirty="", validation=validation)
        assert problems == ["probe.json's env-kwargs drift record is not empty: the run's recorded env kwargs differ "
                            "from the env the cell was judged in ({\"forward_vel_weight\": {\"recorded_run\": 3.0, "
                            "\"toml_main\": 2.0}})"], problems


def test_the_crab_note_of_the_look_s_figure():
    assert ps.crab_note({"report_crab_median_deg": 3.04, "report_end_cross_track_median_m": -1.2}) == (
        "report only: median crab +3.0 deg (course minus heading), median end cross-track -1.20 m")
    assert ps.crab_note({"verdict": "NOT READ", "n": 0}) is None


def test_the_training_guard_blocks_backward_step_and_learn():
    """Fix stage (review: dropping a guard went unnoticed): the guard, installed in a fresh interpreter, lists and
    blocks the optimizer steps, backward and SB3's learn and train."""
    code = textwrap.dedent(f'''\
        import sys
        sys.path.insert(0, {str(TOOLS)!r})
        import pilot4_straight as ps
        patched = set(ps.install_no_training_guard())
        need = {{"torch.Tensor.backward", "torch.autograd.backward", "torch.optim.Adam.step", "torch.optim.SGD.step",
                "BaseAlgorithm.learn", "OnPolicyAlgorithm.learn", "PPO.learn", "PPO.train"}}
        missing = sorted(need - patched)
        import torch
        from stable_baselines3 import PPO
        blocked = []
        for name, fn, entry in (
                ("backward", lambda: torch.zeros(1, requires_grad=True).sum().backward(), "torch.Tensor.backward"),
                ("adam", lambda: torch.optim.Adam([torch.zeros(1, requires_grad=True)]).step(), "torch.optim.Adam.step"),
                ("learn", lambda: PPO.learn(None, total_timesteps=1), "PPO.learn"),
                ("train", lambda: PPO.train(None), "PPO.train")):
            try:
                fn()
            except RuntimeError as exc:  # each entry point's own guard, not one further down (autograd.backward)
                blocked.append(name if str(exc).endswith("training entry point blocked: " + entry) else name + "?")
        print(missing, blocked)
        sys.exit(0 if not missing and blocked == ["backward", "adam", "learn", "train"] else 1)
        ''')
    r = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr


def test_baseline_refuses_another_seed_s_parent_and_a_changed_pair():
    z42, p42 = ps.PARENT_PAIRS[42]
    z52, p52 = ps.PARENT_PAIRS[52]
    assert ps.baseline_problems(42, z42, p42, head="H", dirty="", expect_head=None) == []
    p = ps.baseline_problems(42, z52, p42, head="H", dirty="", expect_head=None)
    assert p == [f"the checkpoint is {z52}, not s42's 8M parent {z42}"]  # s52's 8M zip offered as s42's parent
    p = ps.baseline_problems(42, z42, p52, head="H", dirty="", expect_head=None)
    assert p == [f"the sidecar is {p52}, not s42's 8M parent sidecar {p42}"]
    assert any("no pilot-3 8M parent" in p for p in ps.baseline_problems(7, z42, p42, head="H", dirty="",
                                                                          expect_head=None))
    p = ps.baseline_problems(42, z42, p42, head="H", dirty=" M x", expect_head="G")
    assert any("local edits" in x for x in p) and any("--expect-head G" in x for x in p)
    assert z42.endswith("41fc0b0cd7167ab25a2567779b89f431b829f6a2bae4c72fa3c89192732e87c4")


def test_the_env_must_be_the_pilot4_env_unless_validation():
    good = {"command_mode": "heading", "prey_lateral_range": (0.0, 0.0), "walk_heading_price_weight": 0.2,
            "walk_heading_price_scale_rad": 0.5}
    assert ps.env_problems(good, False) == []
    for k, v in (("command_mode", "none"), ("prey_lateral_range", (-2.0, 2.0)), ("walk_heading_price_weight", 0.0),
                 ("walk_heading_price_scale_rad", 1.0)):
        assert ps.env_problems(dict(good, **{k: v}), False) and ps.env_problems(dict(good, **{k: v}), True) == []


def run_tool(args, tmp_path, cwd=None, env_extra=None):
    env = dict(os.environ, MUJOCO_GL="disable", OMP_NUM_THREADS="1", TMPDIR=str(tmp_path),
               MPLCONFIGDIR=str(tmp_path / "mpl"), **(env_extra or {}))
    return subprocess.run([sys.executable, "-I", "-B", *args], cwd=cwd or tmp_path, env=env, capture_output=True,
                          text=True, timeout=1800)


def test_every_mode_refuses_an_existing_output(tmp_path):
    for name in ("out.json", "out.png", "out.npz"):
        target = tmp_path / name
        target.write_text("keep")
        common = ["--json", str(tmp_path / "out.json"), "--plot", str(tmp_path / "out.png"), "--traces",
                  str(tmp_path / "out.npz")]
        for mode in (["seed", "--seed", "42", "--run", "R", "--cell", "C"],
                     ["baseline", "--seed", "42", "--checkpoint", "Z", "--vecnorm", "P"]):
            r = run_tool([str(TOOLS / "pilot4_straight.py"), *mode, *common], tmp_path)
            assert r.returncode == 2 and "refusing (nothing written)" in r.stdout and name in r.stdout
        assert target.read_text() == "keep"
        target.unlink()


def fake_cell(tmp_path, **over):
    run, binding, meta = synthetic_binding(tmp_path, **over)
    cell = tmp_path / "OUT" / "s42" / "2000000"
    cell.mkdir(parents=True)
    (cell / "binding.json").write_text(json.dumps(binding))
    (cell / "probe.json").write_text(json.dumps({"meta": meta, "episodes": [{}] * 40}))
    return run, cell


@pytest.mark.parametrize("over,dirty,needle", [
    ({}, " M environments/shared/base_env.py", "local edits"),
    ({"off_schedule": True, "off_schedule_reason": "x"}, "", "off-schedule"),
    ({"validation_only": True}, "", "validation-only"),
    ({"head": "OTHER"}, "", "judged at OTHER"),
])
def test_seed_main_refuses_before_loading_anything(tmp_path, monkeypatch, capsys, over, dirty, needle):
    run, cell = fake_cell(tmp_path, **over)
    monkeypatch.setattr(ps, "tree_identity", lambda: (tmp_path, "env", "H", dirty))
    monkeypatch.setattr(ps, "build_context", lambda *a, **k: pytest.fail("loaded although refused"))
    out = tmp_path / "s42.json"
    rc = ps.main(["seed", "--seed", "42", "--run", str(run), "--cell", str(cell), "--json", str(out)])
    assert rc == 2 and not out.exists() and needle in capsys.readouterr().out


def test_seed_main_refuses_a_cell_from_another_run(tmp_path, monkeypatch, capsys):
    run, cell = fake_cell(tmp_path)
    other = tmp_path / "elsewhere" / "pilot4_walk_s42"
    other.mkdir(parents=True)
    monkeypatch.setattr(ps, "tree_identity", lambda: (tmp_path, "env", "H", ""))
    monkeypatch.setattr(ps, "build_context", lambda *a, **k: pytest.fail("loaded although refused"))
    out = tmp_path / "s42.json"
    rc = ps.main(["seed", "--seed", "42", "--run", str(other), "--cell", str(cell), "--json", str(out)])
    assert rc == 2 and not out.exists() and "another run" in capsys.readouterr().out


def test_the_record_is_strict_json_with_inf_as_a_string():
    rec = ps.jsonable({"r": [1.0, math.inf, -math.inf], "n": np.int64(3), "b": np.bool_(True), "x": np.float32(0.5),
                       "t": (1, 2)})
    assert rec == {"r": [1.0, "inf", "-inf"], "n": 3, "b": True, "x": 0.5, "t": [1, 2]}
    json.dumps(rec, allow_nan=False)
    with pytest.raises(ValueError):
        ps.jsonable({"r": math.nan})


def test_the_constants_are_v3_s():
    c = ps.constants()
    assert c["RECOVERY_SEEDS"] == tuple(range(3042, 3062)) and c["NUDGE_ANGLES_DEG"] == (15.0, -15.0)
    assert (c["NUDGE_STATE"], c["READ_K"], c["RESET_READ_K"], c["K_FIRST"], c["MEAN_HALF"]) == (650, 950, 500, 151, 50)
    assert (c["PANEL_SEED_BASE"], c["PANEL_EPISODES"], c["HORIZON"], c["WINDOW_FIRST_STATE"]) == (3042, 40, 1000, 101)
    assert c["OBS651_TOL"] == 1e-3 and c["TWIN_HEADING_TOL_DEG"] == 1e-5 and c["YAW_REPRO_TOL_DEG"] == 1e-12


# ================================================================ against the tree


def load_probe_module():
    if not PROBE.is_file():
        pytest.skip("no pilot-4 tree")
    spec = importlib.util.spec_from_file_location("gait_probe_for_test", PROBE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_tool_s_yaw_is_the_probe_s_yaw():
    probe = load_probe_module()
    rng = np.random.default_rng(2)
    for _ in range(2000):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        assert ps.yaw_from_quat(q) == probe._quat_to_euler(q)[2]


PARENT42 = P3_RUNS / "pilot3_walk_s42" / "03_locomotion" / "models"


@pytest.fixture(scope="module")
def baseline_two_resets(tmp_path_factory):
    if not (TREE / "environments" / "__init__.py").is_file() or not (PARENT42 / "stage2_8000000_steps.zip").is_file():
        pytest.skip("no pilot-4 tree or no s42 8M parent")
    tmp = tmp_path_factory.mktemp("baseline")
    out = tmp / "b.json"
    r = run_tool([str(TOOLS / "iso.py"), str(TOOLS / "pilot4_straight.py"), "baseline", "--seed", "42",
                  "--checkpoint", str(PARENT42 / "stage2_8000000_steps.zip"),
                  "--vecnorm", str(PARENT42 / "stage2_vecnormalize_8000000_steps.pkl"), "--json", str(out),
                  "--zero-nudge-check", "--limit-resets", "2"], tmp, cwd=TREE, env_extra={"ISO_WT": str(TREE)})
    (tmp / "stdout.txt").write_text(r.stdout + "\n--- stderr\n" + r.stderr)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return json.loads(out.read_text())


def test_tree_baseline_reports_the_crab_and_the_cross_track(baseline_two_resets):
    """Fix stage, report only: (b) carries the medians of the per-episode crab and end cross-track over the episodes
    (a) and (b) read."""
    b = baseline_two_resets["parts"]["b"]
    used = [e for e in baseline_two_resets["panel"]["episodes"] if e["used"] and e["full"]]
    assert used and b["report_crab_median_deg"] == ps.percentile([e["crab_median_deg"] for e in used], 50)
    assert b["report_end_cross_track_median_m"] == ps.percentile([e["end_cross_track_m"] for e in used], 50)


def test_tree_baseline_states_0_650_of_every_nudged_episode_equal_the_panel_episode(baseline_two_resets):
    ch = baseline_two_resets["checks"]["prefix"]
    assert ch["episodes"] == 8 and ch["equal"] == 8 and ch["refused"] == []  # 2 resets x (c, prey in place) x 2 signs


def test_tree_baseline_a_0_deg_nudge_reproduces_the_panel_episode_exactly(baseline_two_resets):
    ch = baseline_two_resets["checks"]["zero_nudge"]
    assert ch["run"] and ch["episodes"] == 2 and ch["equal"] == 2 and ch["bad"] == []


def test_tree_baseline_the_command_blind_walker_keeps_the_offset_exactly(baseline_two_resets):
    tw = baseline_two_resets["checks"]["turned_twin"]
    assert tw["applies"] and tw["episodes"] == 4 and tw["bad"] == []
    assert tw["max_abs_dev_deg"] <= ps.TWIN_HEADING_TOL_DEG
    for e in baseline_two_resets["recovery"]["episodes"]:
        assert e["share_removed"] == pytest.approx(0.0, abs=1e-6)  # share removed 0 by construction


def test_tree_baseline_the_state_651_observation_shows_the_offset_only_in_the_command_inputs(baseline_two_resets):
    o = baseline_two_resets["checks"]["obs651"]
    assert o["episodes"] == 4 and o["only_command_channels"] == 4 and set(o["channels_seen"]) <= {60, 61, 62}
    assert 60 in o["channels_seen"] and 62 in o["channels_seen"]
    pip = baseline_two_resets["recovery"]["prey_in_place"]["episodes"]
    assert all({56, 57} & set(e["obs651_channels_beyond_tol_vs_twin"]) for e in pip)  # the prey block shows it there


def test_tree_baseline_the_pass_check_and_the_reading(baseline_two_resets):
    panel = {e["reset_seed"]: e for e in baseline_two_resets["panel"]["episodes"]}
    for e in baseline_two_resets["recovery"]["episodes"]:
        assert e["reached_nudge"] and e["passed_prey_at_650"] and e["counting"]
        assert e["pelvis_minus_prey_x_650_m"] > 0
        twin_err = panel[e["reset_seed"]]
        assert e["twin_m_read_error_deg"] is not None and twin_err["T"] == 1000
        assert e["r_deg"] == pytest.approx(abs(e["twin_m_read_error_deg"] + e["angle_deg"]), abs=1e-5)
    p = baseline_two_resets["parts"]
    assert p["straight"] == "NOT READ" and all(p[k]["verdict"] == "NOT READ" for k in "abc")
    assert baseline_two_resets["partial"] is True


def test_tree_baseline_the_prepared_walker_acts_as_the_command_blind_parent(baseline_two_resets):
    assert baseline_two_resets["checks"]["action_delta_vs_blind"]["max"] == 0.0
    prep = baseline_two_resets["preparation"]["assert_command_blind"]
    assert prep["max_action_delta"] == 0.0 and prep["max_value_delta"] == 0.0
    assert prep["prepared"] == ["mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight"]
    assert baseline_two_resets["env"]["pilot4_env"] is True
    assert baseline_two_resets["psi_cmd"]["values_deg"] == [0.0]
    assert baseline_two_resets["psi_cmd"]["info_check_available"]
    assert baseline_two_resets["psi_cmd"]["info_check_max_abs_diff_rad"] <= ps.INFO_HEADING_TOL_RAD


# ================================================================ the validation records (STRAIGHT_VAL)

VAL = Path(os.environ["STRAIGHT_VAL"]) if os.environ.get("STRAIGHT_VAL") else None
SEEDS = (42, 52, 60)


def val_record(name):
    """A validation record, which must have been written by this very tool (fix stage: a record test otherwise keeps
    passing after any edit of pilot4_straight.py)."""
    if VAL is None or not (VAL / name).is_file():
        pytest.skip(f"no validation record {name} (STRAIGHT_VAL)")
    rec = json.loads((VAL / name).read_text())
    want = ps.sha256_file(TOOLS / "pilot4_straight.py")
    assert rec["tool_sha256"] == want, f"{name} was written by {rec['tool_sha256']}, not this tool ({want}): re-run it"
    return rec


def first20(rec):
    return [e for e in rec["panel"]["episodes"] if e["reset_seed"] < 3062]


def r1(x):
    return round(float(x) + 0.0, 1)


def close(x, want, tol=0.051):
    return abs(float(x) - want) <= tol


# v3 "Evidence for the bars" (resets 3042-3061; yaw over T >= 110, D/R/W0 over full-length episodes)
P3_TABLE = {42: (20, -24.3, 36.9, 32.2, 16.9, 41.7, 23.3, 33.1, 37.2),
            52: (19, -46.6, 52.2, 44.0, 34.9, 45.0, 41.0, 44.4, 49.4),
            60: (20, -22.9, 55.1, 52.2, 26.0, 60.8, 37.1, 47.9, 58.1)}
ENTRY_TABLE = {42: (20, -25.3, 31.9, 27.2, 17.0, 35.9, 22.4, 26.6, 31.3),
               52: (19, -46.6, 53.9, 44.8, 32.8, 45.6, 39.0, 44.8, 49.4),
               60: (20, -29.6, 45.6, 43.1, 31.3, 45.2, 38.3, 45.2, 48.4)}
ENTRY_C = {42: (40, 22.8, 40.2, 18, 12), 52: (38, 34.8, 55.4, 2, 0), 60: (40, 29.1, 52.7, 3, 1)}


def table_row(rec):
    eps = first20(rec)
    full = [e for e in eps if e["full"]]
    yaws = [e["yaw_change_deg"] for e in eps if e["yaw_change_deg"] is not None]
    D = [e["D_deg"] for e in full]
    R = [e["R_deg"] for e in full]
    W0 = [e["W0_deg"] for e in full]
    return (len(full), ps.percentile(yaws, 50), ps.percentile([abs(y) for y in yaws], 95), ps.percentile(D, 95),
            ps.percentile(D, 50), max(D), ps.percentile(R, 50), ps.percentile(R, 95), ps.percentile(W0, 95))


@pytest.mark.parametrize("seed", SEEDS)
def test_valrec_pilot3_cells_reproduce_yaw_change_deg_and_the_draft_s_rows(seed):
    rec = val_record(f"p3val_s{seed}.json")
    assert rec["validation"] and rec["panel"]["all_rolled_match"] and rec["panel"]["mismatched"] == []
    assert rec["panel"]["yaw_repro_max_abs_diff_deg"] <= 1e-12
    assert all(e["yaw_abs_diff_deg"] <= 1e-12 for e in rec["panel"]["episodes"] if e["yaw_abs_diff_deg"] is not None)
    row = table_row(rec)
    want = P3_TABLE[seed]
    assert row[0] == want[0]
    assert all(close(x, w) for x, w in zip(row[1:], want[1:], strict=True)), (row, want)


@pytest.mark.parametrize("seed", SEEDS)
def test_valrec_entry_baseline_reproduces_v3_s_entry_rows_and_c(seed):
    rec = val_record(f"baseline_s{seed}.json")
    assert not rec["partial"] and rec["mode"] == "baseline"
    row = table_row(rec)
    want = ENTRY_TABLE[seed]
    assert row[0] == want[0]
    assert all(close(x, w) for x, w in zip(row[1:], want[1:], strict=True)), (row, want)
    c = rec["parts"]["c"]
    n, med, p95, w115, w5 = ENTRY_C[seed]
    assert c["n_counting"] == n and close(c["median_r_deg"], med) and close(c["p95_r_deg"], p95)
    assert c["within_11_5"] == w115 and c["within_5"] == w5 and c["verdict"] == "FAIL"
    assert rec["parts"]["a"]["verdict"] == "FAIL" and rec["parts"]["b"]["verdict"] == "FAIL"
    assert rec["parts"]["straight"] == "FAIL"
    ch = rec["checks"]
    assert ch["prefix"]["refused"] == [] and ch["prefix"]["equal"] == ch["prefix"]["episodes"]
    assert ch["zero_nudge"]["equal"] == ch["zero_nudge"]["episodes"] == 20  # a fall before 650 is compared whole
    assert ch["obs651"]["only_command_channels"] == ch["obs651"]["episodes"] == n
    # v3's tolerance (1e-5 deg at 10 s) is applied literally: an episode beyond it is flagged (exit 1). Measured on the
    # 40-episode panels: s42 flags 3 of 40 (largest 2.7e-5 deg), s52 and s60 none (text issue: the 1e-5 came from
    # 15 episodes); every deviation stays far below the 15-deg offset, so the walker keeps the offset.
    tw = ch["turned_twin"]
    assert tw["episodes"] == n and all(abs(b["dev_deg"]) > ps.TWIN_HEADING_TOL_DEG for b in tw["bad"])
    assert tw["max_abs_dev_deg"] < 1e-4 and len(tw["bad"]) == {42: 3, 52: 0, 60: 0}[seed]
    assert ch["action_delta_vs_blind"]["max"] == 0.0
    assert rec["labels"]["seed_outcome"] == "WALKS"


# v3 "Report only", course against heading (review_text P5: the first 20 resets' full-length episodes): the per-episode
# median crab min / median / max (deg) and the end cross-track min / median / max (m)
P5 = {42: ((3.9, 4.7, 5.7), (-6.21, -0.21, 2.75)), 52: ((-1.8, -1.5, -0.7), (-7.54, -3.92, -0.79)),
      60: ((1.7, 2.3, 3.0), (-4.84, -2.81, -1.41))}


@pytest.mark.parametrize("seed", SEEDS)
def test_valrec_entry_baseline_reports_the_crab_and_reproduces_v3_s_course_numbers(seed):
    """Fix stage: (b)'s report-only crab and cross-track medians are over the episodes (a) and (b) read (all 40
    resets); over the first 20 resets' full-length episodes the per-episode values reproduce v3's P5 numbers."""
    rec = val_record(f"baseline_s{seed}.json")
    used = [e for e in rec["panel"]["episodes"] if e["used"] and e["full"]]
    b = rec["parts"]["b"]
    assert b["report_crab_median_deg"] == ps.percentile([e["crab_median_deg"] for e in used], 50)
    assert b["report_end_cross_track_median_m"] == ps.percentile([e["end_cross_track_m"] for e in used], 50)
    f20 = [e for e in first20(rec) if e["full"]]
    crab = sorted(e["crab_median_deg"] for e in f20)
    cross = sorted(e["end_cross_track_m"] for e in f20)
    want_crab, want_cross = P5[seed]
    assert (r1(crab[0]), r1(float(np.median(crab))), r1(crab[-1])) == want_crab
    assert tuple(round(float(x), 2) for x in (cross[0], np.median(cross), cross[-1])) == want_cross
