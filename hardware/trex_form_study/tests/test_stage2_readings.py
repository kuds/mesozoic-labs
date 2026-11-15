"""Tests for the bar's readings (designlab/stage2/readings.py): the round-A design's section 4.4 table, row by row, on
synthetic arrays.

Run from hardware/trex_form_study (the fit's venv):  python -m pytest tests/test_stage2_readings.py -q
All fast.  ``readings`` itself loads neither MuJoCo nor Stage 1 (checked in a child process); the two tests that
compare it with Stage 1's own filter, contact and constants import Stage 1.

Arrays as the recorder gives them: physics steps of 2 ms, update rows every fifth step (100 Hz), a row's time the
start of its step.  H is 75-M-H-10's hip height.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from designlab.stage2 import readings as R  # noqa: E402

H = 0.501854
DT = 0.002
EVERY = 5  # physics steps per update row
LIFT = 0.081 * H
SWING_S = 0.350
SWING_STEPS = int(round(SWING_S / DT))  # 175
#: Half the planned swing on the simulated clock (the bar's rule for a 0.350 s shortest swing).
AIR_GAP_MAX_S = 0.5 * SWING_S / R.RATE
#: How far the own-swing reading can move with the grid on this profile.  The design's table says 1e-6 m; linear
#: interpolation between 10 ms rows does not reach that here: at 25 % and 75 % of a sin^2 swing the curvature is zero
#: and the cubic term leaves up to f''' h^3 / (36 sqrt 3) = 1.9e-6 m at an edge (f''' = LIFT (2 pi / D)^3 / 2, h the
#: row interval), so two grid offsets can differ by twice that, 3.8e-6 m.  Measured: 3.6e-6 m; rows alone: 2.4 mm.
EDGE_TOL_M = LIFT * (2 * math.pi / SWING_S) ** 3 / 2 * (EVERY * DT) ** 3 / (18 * math.sqrt(3))


def episode(own, ref, seconds=8.0):
    """Synthetic recorder arrays.  ``ref``: the reference's swings as (foot, lift-off step, steps); ``own``: the
    simulated feet's air runs as (foot, lift-off step, steps, low(u) -> m, u in (0, 1) the run's progress).  Returns
    t_u, low [rows, 2], contact [steps, 2], ref_swing [rows], ref_u [rows] (the reference swing's progress)."""
    n = int(round(seconds / DT))
    t_u = np.arange(0, n, EVERY) * DT
    contact = np.ones((n, 2), bool)
    low = np.zeros((len(t_u), 2))
    ref_swing = -np.ones(len(t_u))
    ref_u = np.zeros(len(t_u))
    for foot, a, steps in ref:
        u = (t_u - a * DT) / (steps * DT)
        m = (u >= -1e-12) & (u < 1 - 1e-12)
        ref_swing[m], ref_u[m] = foot, u[m]
    for foot, a, steps, shape in own:
        contact[a : a + steps, foot] = False
        u = (t_u - a * DT) / (steps * DT)
        m = (u > 0) & (u < 1)
        low[m, foot] = shape(u[m])
    return t_u, low, contact, ref_swing, ref_u


def sin2(lift=LIFT):
    return lambda u: lift * np.sin(np.pi * u) ** 2


def walk(feet=(0,), delay=0, shift=0, count=6, period_s=1.0, first_s=1.0, shape=None):
    """``count`` swings per foot, a stride apart (the right foot half a stride after the left); the simulated foot
    is the reference delayed by ``delay`` physics steps; ``shift`` moves both by that many steps."""
    ref, own = [], []
    for foot in feet:
        for j in range(count):
            a = int(round((first_s + (j + 0.5 * foot) * period_s) / DT)) + shift
            ref.append((foot, a, SWING_STEPS))
            own.append((foot, a + delay, SWING_STEPS, shape or sin2()))
    return own, ref


def own_swing(arrays, ta=0.5, tb=7.9, air_gap_max_s=AIR_GAP_MAX_S):
    t_u, low, contact, ref_swing, _ = arrays
    return R.toe_clearance_own_swing(t_u, low, contact, DT, ref_swing, ta, tb, air_gap_max_s)


def reference_clock_reading(arrays, ta=0.5, tb=7.9):
    """Stage 1's reading (``metrics.window``): the lowest point of the reference's swing foot over the rows of the
    middle half of each reference swing."""
    t_u, low, _, ref_swing, ref_u = arrays
    sel = (t_u >= ta) & (t_u <= tb) & (ref_u >= 0.25) & (ref_u <= 0.75)
    return min(float(low[sel & (ref_swing == i), i].min()) for i in (0, 1) if (sel & (ref_swing == i)).any())


# ------------------------------------------------------------------------------------------------
# (a) Toe clearance over each foot's own swing
# ------------------------------------------------------------------------------------------------
def test_own_swing_clearance_ignores_a_delay():
    """A sin^2 lift of 0.081 H over a 0.350 s swing; the simulated foot and its contact are the reference's, k = 0..5
    physics steps late, while ``ref_swing`` stays on the reference's clock.  The own-swing reading is the same for
    every k (to ``EDGE_TOL_M``; the design's table says 1e-6 m, see the constant); the reference-clock reading falls
    through 0.04 H, from 0.0423 H at k = 0 to 0.0351 H at k = 5."""
    own_readings, ref_readings = [], []
    for k in range(6):
        arrays = episode(*walk(delay=k))
        r = own_swing(arrays)
        assert r["own_swings"] == [6, 0] and r["ref_swings"] == [6, 0]
        assert r["interrupted_swings"] == 0 and r["air_gaps"] == 0 and r["count_ok"]
        assert [row[4] for row in r["swings"]] == ["swing"] * 6
        assert R.toe_clearance_own_swing_ok(r, H)
        own_readings.append(r["toe_clearance_own_swing_m"])
        ref_readings.append(reference_clock_reading(arrays))
    assert max(own_readings) - min(own_readings) <= EDGE_TOL_M < 4e-6
    assert max(abs(x - 0.5 * LIFT) for x in own_readings) <= 0.5 * EDGE_TOL_M  # sin^2 is half its lift at the edges
    assert own_readings[0] == pytest.approx(own_readings[5], abs=1e-12)  # 10 ms late: the same place on the grid
    assert ref_readings[0] > R.TOE_CLEARANCE_FRAC_H * H > ref_readings[5]
    assert ref_readings[0] / H == pytest.approx(0.0423, abs=1e-4)
    assert ref_readings[5] / H == pytest.approx(0.0351, abs=1e-4)
    assert all(a > b for a, b in zip(ref_readings[:-1], ref_readings[1:], strict=True))


def test_own_swing_edges_are_interpolated():
    """The same swing with its lift-off moved by 0, 2, 4, 6, 8 ms across the 10 ms grid: the reading stays put (to
    ``EDGE_TOL_M``), where the rows of the middle half alone over-read by up to 2.4 mm."""
    reads, rows_alone = [], []
    for shift in range(5):
        t_u, low, contact, ref_swing, _ = arrays = episode(*walk(shift=shift, count=1))
        r = own_swing(arrays)
        assert r["own_swings"] == [1, 0]
        reads.append(r["toe_clearance_own_swing_m"])
        _, t_lo, t_td, value, cls = r["swings"][0]
        assert cls == "swing" and value == reads[-1]
        assert t_lo == pytest.approx(1.0 + shift * DT, abs=1e-12) and t_td - t_lo == pytest.approx(SWING_S, abs=1e-12)
        mid = (t_u >= t_lo + 0.25 * SWING_S) & (t_u <= t_lo + 0.75 * SWING_S)
        rows_alone.append(float(low[mid, 0].min()))
    assert max(reads) - min(reads) <= EDGE_TOL_M
    over = [x - 0.5 * LIFT for x in rows_alone]
    assert min(over) > 0 and max(over) == pytest.approx(2.4e-3, abs=1e-4)


def test_early_touchdown_fails():
    """A touchdown at 45 % of the planned swing.  The air run is shorter than half the shortest planned swing, but it
    overlaps the reference's swing, so it is a swing, read over its own middle half; the term fails."""
    steps = int(round(0.45 * SWING_STEPS))  # 79 steps: 0.158 s
    assert steps * DT < AIR_GAP_MAX_S

    def tent(u):  # up along the plan, then back down the same way: the foot never gets to mid-swing height
        return LIFT * np.sin(np.pi * np.minimum(u, 1 - u) * steps / SWING_STEPS) ** 2

    a = int(round(1.0 / DT))
    arrays = episode([(0, a, steps, tent)], [(0, a, SWING_STEPS)], seconds=3.0)
    r = own_swing(arrays, ta=0.5, tb=2.9)
    assert r["swings"] == [[0, pytest.approx(1.0), pytest.approx(1.0 + steps * DT), r["toe_clearance_own_swing_m"],
                            "swing"]]
    assert r["own_swings"] == [1, 0] and r["ref_swings"] == [1, 0] and r["air_gaps"] == 0
    expected = LIFT * math.sin(math.pi * 0.25 * steps / SWING_STEPS) ** 2  # the tent at 25 % and 75 % of its own run
    assert r["toe_clearance_own_swing_m"] == pytest.approx(expected, abs=1e-4)  # (the profile curves between rows)
    assert r["toe_clearance_own_swing_m"] < R.TOE_CLEARANCE_FRAC_H * H
    assert not R.toe_clearance_own_swing_ok(r, H)
    # the same short air run wholly inside the reference's stance is an air gap: only the overlap makes it a swing
    late = episode([(0, a + 300, steps, tent)], [(0, a, SWING_STEPS)], seconds=3.0)
    assert own_swing(late, ta=0.5, tb=2.9)["swings"][0][3:] == [None, "air_gap"]


def test_mid_swing_stub_fails():
    """A 30 ms re-contact at mid-swing: two swings in one reference swing, so ``interrupted_swings == 1`` and the
    term fails, whatever the clearance of the two parts."""
    a = int(round(1.0 / DT))
    part = 80  # steps before the stub; 15 steps (30 ms) of contact; 80 steps after
    own, ref = walk(count=3)
    high = lambda u: 0.030 + 0.0 * u  # noqa: E731  (30 mm in the air: the clearance alone would pass)
    own[1] = (0, a + 500, part, high)
    own.insert(2, (0, a + 500 + part + 15, SWING_STEPS - part - 15, high))
    r = own_swing(episode(own, ref, seconds=5.0), ta=0.5, tb=4.9)
    assert r["interrupted_swings"] == 1
    assert r["own_swings"] == [4, 0] and r["ref_swings"] == [3, 0] and r["count_ok"]
    assert [row[4] for row in r["swings"]] == ["swing"] * 4  # both parts are swings: each overlaps the reference's
    assert all(row[2] - row[1] < AIR_GAP_MAX_S for row in r["swings"][1:3])
    assert r["toe_clearance_own_swing_m"] >= R.TOE_CLEARANCE_FRAC_H * H
    assert not R.toe_clearance_own_swing_ok(r, H)
    whole = own_swing(episode(*walk(count=3), seconds=5.0), ta=0.5, tb=4.9)
    assert whole["interrupted_swings"] == 0 and R.toe_clearance_own_swing_ok(whole, H)


def test_stance_chatter_is_an_air_gap():
    """A 30 ms unloading in mid stance is an air gap: counted, not read, the clearance unchanged.  The same
    unloading lasting ``air_gap_max_s`` or more is read as a swing."""
    base = own_swing(episode(*walk()))
    own, ref = walk()
    a = int(round(1.6 / DT))  # mid stance: the swings are at 1.0-1.35 s and 2.0-2.35 s
    graze = lambda u: 0.0005 + 0.0 * u  # noqa: E731  (the unloaded foot half a millimetre up)
    r = own_swing(episode(own + [(0, a, 15, graze)], ref))
    assert r["air_gaps"] == 1 and r["air_gap_longest_s"] == pytest.approx(0.030, abs=1e-12)
    assert r["air_gap_max_s"] == AIR_GAP_MAX_S
    assert r["toe_clearance_own_swing_m"] == base["toe_clearance_own_swing_m"]
    assert r["own_swings"] == base["own_swings"] == [6, 0] and r["interrupted_swings"] == 0
    gap = [row for row in r["swings"] if row[4] == "air_gap"]
    assert gap == [[0, pytest.approx(1.6), pytest.approx(1.63), None, "air_gap"]]
    assert R.toe_clearance_own_swing_ok(r, H)
    long_steps = int(math.ceil(AIR_GAP_MAX_S / DT))  # 84 steps: 0.168 s >= 0.1667 s
    r = own_swing(episode(own + [(0, a, long_steps, graze)], ref))
    assert r["air_gaps"] == 0 and r["own_swings"] == [7, 0]
    assert r["toe_clearance_own_swing_m"] == pytest.approx(0.0005, abs=1e-12)
    assert not R.toe_clearance_own_swing_ok(r, H)
    r = own_swing(episode(own + [(0, a, long_steps - 1, graze)], ref))  # one step shorter: still a gap
    assert r["air_gaps"] == 1 and r["own_swings"] == [6, 0]


def test_swing_count():
    """A window in which one foot never lifts: ``own_swings < ref_swings - 1`` and the term fails.  A swing that
    the window's end cuts is not read and does not fail it."""
    own, ref = walk(feet=(0, 1))
    r = own_swing(episode([s for s in own if s[0] == 0], ref))
    assert r["own_swings"] == [6, 0] and r["ref_swings"] == [6, 6]
    assert not r["count_ok"] and r["toe_clearance_own_swing_m"] >= R.TOE_CLEARANCE_FRAC_H * H
    assert not R.toe_clearance_own_swing_ok(r, H)
    # both feet walk; the window ends 0.2 s into the right foot's last swing, a hop of 1 mm that is never read
    own[-1] = own[-1][:3] + (sin2(0.001),)
    tb = 6.5 + 0.2
    r = own_swing(episode(own, ref), tb=tb)
    assert r["swings"][-1] == [1, pytest.approx(6.5), pytest.approx(6.85), None, "swing_cut"]
    assert r["own_swings"] == [6, 5] and r["ref_swings"] == [6, 5] and r["count_ok"]
    assert r["toe_clearance_own_swing_m"] == pytest.approx(0.5 * LIFT, abs=EDGE_TOL_M)
    assert R.toe_clearance_own_swing_ok(r, H)
    # one swing fewer than the reference's is allowed (a window edge); two fewer is not
    r = own_swing(episode([s for s in own if not (s[0] == 1 and s[1] < 1000)], ref), tb=tb)
    assert r["own_swings"] == [6, 4] and r["count_ok"]
    r = own_swing(episode([s for s in own if not (s[0] == 1 and s[1] < 1500)], ref), tb=tb)
    assert r["own_swings"] == [6, 3] and not r["count_ok"]


def test_air_runs_and_touchdowns():
    c = np.ones(40, bool)
    c[:3] = False  # the trace starts in the air: not a complete run
    c[10:14] = False
    c[20:21] = False
    c[36:] = False  # cut by the trace's end
    assert R.air_runs(c, DT) == [(10, 14), (20, 21)]
    both = np.column_stack([c, np.ones(40, bool)])
    assert R.touchdowns(both, DT) == [(pytest.approx(3 * DT), 0), (pytest.approx(14 * DT), 0),
                                      (pytest.approx(21 * DT), 0)]
    with pytest.raises(ValueError):
        R.air_runs(both, DT)


# ------------------------------------------------------------------------------------------------
# (b) The walk item's line
# ------------------------------------------------------------------------------------------------
def line_case(length, bow, heading=math.radians(30.0), ta=4.0, tb=14.0, lead_offset=0.05):
    """A CoM that covers ``length`` along the heading over [ta, tb] with a sideways bow of ``bow`` (zero at both
    ends, largest in the middle); before the window it comes in along a line ``lead_offset`` to the right."""
    t = np.arange(0, 1401) * 0.01
    h = np.array([math.cos(heading), math.sin(heading)])
    left = np.array([-h[1], h[0]])
    start = np.array([1.5, -0.7])
    s = np.where(t < ta, 0.25 * (t - ta), length * (t - ta) / (tb - ta))
    lat = np.where(t < ta, -lead_offset * (ta - t) / ta, bow * np.sin(np.pi * (t - ta) / (tb - ta)))
    com = start + np.outer(s, h) + np.outer(lat, left)
    return t, com, com[0]


def test_line_reading():
    """A CoM path of 4.2 m with a 0.2 m bow, with a 0.3 m bow, and 3.9 m straight, on a heading of 30 deg: along
    and lateral to 1e-12; ``line`` true, false; ``along`` false; the line is anchored at the window's first update;
    the deviation from the line through the episode's start is reported beside it."""
    heading = math.radians(30.0)
    need = R.WALK_WINDOW_S * 0.40  # 4.0 m at 0.40 m/s
    t, com, c_start = line_case(4.2, 0.2)
    r = R.line_reading(t, com, heading, 4.0, 14.0, c_start)
    assert r["complete"] and r["updates"] == 1001 and r["anchor_s"] == pytest.approx(4.0, abs=1e-12)
    assert r["along_commanded_heading_m"] == pytest.approx(4.2, abs=1e-12)
    assert r["lateral_from_line_max_m"] == pytest.approx(0.2, abs=1e-12)
    assert r["lateral_end_m"] == pytest.approx(0.0, abs=1e-12)
    assert r["along_commanded_heading_m"] >= need and r["lateral_from_line_max_m"] <= R.LINE_MAX_M
    assert r["lateral_from_episode_start_line_max_m"] == pytest.approx(0.25, abs=1e-12)  # 0.05 m lead-in + the bow
    t, com, c_start = line_case(4.2, 0.3)
    r = R.line_reading(t, com, heading, 4.0, 14.0)
    assert r["lateral_from_line_max_m"] == pytest.approx(0.3, abs=1e-12) and r["lateral_from_line_max_m"] > R.LINE_MAX_M
    assert r["lateral_from_episode_start_line_max_m"] is None
    t, com, c_start = line_case(3.9, 0.0)
    r = R.line_reading(t, com, heading, 4.0, 14.0, c_start)
    assert r["along_commanded_heading_m"] == pytest.approx(3.9, abs=1e-12) and r["along_commanded_heading_m"] < need
    assert r["lateral_from_line_max_m"] == pytest.approx(0.0, abs=1e-12)
    # a bow to the right reads the same distance; the end offset is signed, positive to the left
    t, com, _ = line_case(4.2, -0.2)
    assert R.line_reading(t, com, heading, 4.0, 14.0)["lateral_from_line_max_m"] == pytest.approx(0.2, abs=1e-12)
    # the anchor is the first update at or after ta, not an interpolated point
    t, com, _ = line_case(4.2, 0.0)
    r = R.line_reading(t, com, heading, 4.004, 14.0)
    assert r["anchor_s"] == pytest.approx(4.01, abs=1e-12) and r["updates"] == 1000
    assert r["anchor_xy"] == [pytest.approx(com[401, 0]), pytest.approx(com[401, 1])]
    assert r["along_commanded_heading_m"] == pytest.approx(4.2 * (14.0 - 4.01) / 10.0, abs=1e-12)
    # no window, no reading
    r = R.line_reading(t, com, heading, 20.0, 30.0)
    assert not r["complete"] and r["along_commanded_heading_m"] is None


# ------------------------------------------------------------------------------------------------
# (c) The turn item's heading
# ------------------------------------------------------------------------------------------------
T_C = 4.0


def yaw_law(knots, psi_cmd=0.3, sign=1.0, seconds=12.0):
    """Update times and a trunk yaw (rad) that turns, from the command at ``T_C``, through ``knots``: (seconds after
    the command, degrees turned), linear between them and held after the last."""
    t = np.arange(0, int(round(seconds / 0.01)) + 1) * 0.01
    tau, deg = zip(*([(0.0, 0.0)] + list(knots)), strict=True)
    return t, psi_cmd + sign * np.radians(np.interp(t - T_C, tau, deg))


def turn(t, yaw, psi_cmd=0.3, sign=1.0, at_s=2.0):
    return R.turn_reading(t, yaw, T_C, psi_cmd, sign, math.pi / 2, at_s)


def test_turn_reading():
    """A yaw that reaches 82 deg at ``at_s`` and 90 after; one that reaches 79; one that overshoots to 101 deg; a
    right turn; a yaw that crosses +-pi; a 0.5 rad/s turn read at 3.1416 s."""
    # 82 deg at 2.0 s, 90 deg from 2.5 s
    r = turn(*yaw_law([(2.0, 82.0), (2.5, 90.0)]))
    assert r["window_reached"] and r["heading"]
    assert r["heading_error_at_deg"] == pytest.approx(-8.0, abs=1e-9)
    assert r["heading_error_hold_max_deg"] == pytest.approx(8.0, abs=1e-9)
    assert r["turned_own_deg"] == pytest.approx(82.0, abs=1e-9)
    assert r["trunk_heading_at_command_deg"] == pytest.approx(0.0, abs=1e-9)
    # 80 deg is reached 80/82 x 2.0 = 1.951 s after the command: the first update from which it holds is at 1.96 s
    assert r["time_to_within_10deg_s"] == pytest.approx(1.96, abs=1e-9)
    assert (r["at_s"], r["hold_s"], r["tol_deg"], r["angle_deg"]) == (2.0, 3.0, 10.0, pytest.approx(90.0))
    # 79 deg at 2.0 s: one degree short at the judged instant, although it gets there 30 ms later
    r = turn(*yaw_law([(2.0, 79.0), (2.5, 90.0)]))
    assert not r["heading"] and r["heading_error_at_deg"] == pytest.approx(-11.0, abs=1e-9)
    assert r["time_to_within_10deg_s"] == pytest.approx(2.05, abs=1e-9)  # 80 deg at 2.0 + 0.5 / 11 = 2.045 s
    # 90 deg at 2.0 s, then on to 101 deg at 2.5 s and back to 90 at 3.5 s: over the band during the hold
    r = turn(*yaw_law([(2.0, 90.0), (2.5, 101.0), (3.5, 90.0)]))
    assert not r["heading"] and r["heading_error_at_deg"] == pytest.approx(0.0, abs=1e-9)
    assert r["heading_error_hold_max_deg"] == pytest.approx(11.0, abs=1e-9)
    assert r["time_to_within_10deg_s"] == pytest.approx(2.5 + 1.0 / 11.0, abs=0.0101)  # back inside 100 deg
    assert r["time_to_within_10deg_s"] > 2.5
    # a turn that never arrives
    r = turn(*yaw_law([(2.0, 60.0)]))
    assert not r["heading"] and r["time_to_within_10deg_s"] is None
    # the right turn is the mirror: the same verdict, the error and the turn with the other sign
    r = turn(*yaw_law([(2.0, 82.0), (2.5, 90.0)], sign=-1.0), sign=-1.0)
    assert r["heading"] and r["heading_error_at_deg"] == pytest.approx(8.0, abs=1e-9)
    assert r["turned_own_deg"] == pytest.approx(-82.0, abs=1e-9)
    assert r["time_to_within_10deg_s"] == pytest.approx(1.96, abs=1e-9)
    # ... and a left turn's yaw read as a right turn is 172 deg off
    r = turn(*yaw_law([(2.0, 82.0), (2.5, 90.0)]), sign=-1.0)
    assert not r["heading"] and r["heading_error_at_deg"] == pytest.approx(172.0, abs=1e-9)
    # the yaw crosses +-pi: the recorder's wrapped column, unwrapped.  A left turn from 3.0 rad and a right turn
    # from -3.1 rad cross it during the turn; from 3.13 rad the trunk, 0.02 rad to the left, starts beyond it, so the
    # unwrapped yaw starts a whole turn away from the commanded heading
    for psi_cmd, sign, off in ((3.0, 1.0, -0.02), (-3.1, -1.0, 0.02), (3.13, 1.0, 0.02)):
        t, yaw = yaw_law([(2.0, 82.0), (2.5, 90.0)], psi_cmd=psi_cmd, sign=sign)
        wrapped = (yaw + off + math.pi) % (2 * math.pi) - math.pi
        assert np.ptp(wrapped) > 6.0 or abs(wrapped[0] - psi_cmd) > 6.0  # it does wrap
        r = turn(t, np.unwrap(wrapped), psi_cmd=psi_cmd, sign=sign)
        assert r["heading"] and r["heading_error_at_deg"] == pytest.approx(-8.0 * sign + math.degrees(off), abs=1e-9)
        assert r["trunk_heading_at_command_deg"] == pytest.approx(math.degrees(off), abs=1e-9)
        assert r["turned_own_deg"] == pytest.approx(82.0 * sign, abs=1e-9)
    # a 0.5 rad/s turn is read at (pi/2) / 0.5 = 3.1416 s, not at 2.0 s
    at_s = (math.pi / 2) / 0.5
    assert at_s == pytest.approx(3.1416, abs=1e-4)
    t, yaw = yaw_law([(at_s, 82.0), (at_s + 0.5, 90.0)])
    slow = turn(t, yaw, at_s=at_s)
    assert slow["heading"] and slow["at_s"] == at_s
    assert slow["heading_error_at_deg"] == pytest.approx(-8.0, abs=0.02)  # (the law's corner lies between two rows)
    assert not turn(t, yaw, at_s=2.0)["heading"]
    assert turn(t, yaw, at_s=2.0)["heading_error_at_deg"] == pytest.approx(82.0 * 2.0 / at_s - 90.0, abs=1e-9)
    # a trace that ends inside the hold cannot show the term
    t, yaw = yaw_law([(2.0, 82.0), (2.5, 90.0)], seconds=8.0)
    r = turn(t, yaw)
    assert not r["window_reached"] and not r["heading"] and r["heading_error_hold_max_deg"] is None
    assert r["heading_error_at_deg"] == pytest.approx(-8.0, abs=1e-9)
    r = turn(t[:300], yaw[:300])  # it ends before the command
    assert not r["window_reached"] and r["trunk_heading_at_command_deg"] is None


# ------------------------------------------------------------------------------------------------
# (e) The stop item
# ------------------------------------------------------------------------------------------------
STOP_C = 5.0
STRIDE_S = 0.8606 / R.RATE
V_CMD = 0.40 * R.RATE


def stop_case(air, settle_s=1.0, seconds=9.0, residual=0.0, weight=300.0):
    """Loads [steps, 2] for a stop commanded at ``STOP_C``: both feet loaded but for ``air`` = (foot, start s after
    the command, end s after it) spans; a forward speed that ramps from the command speed to ``residual`` between
    0.3 s and ``settle_s`` after the command."""
    n = int(round(seconds / DT))
    loads = np.full((n, 2), 0.5 * weight)
    for foot, a, b in air:
        loads[int(round((STOP_C + a) / DT)) : int(round((STOP_C + b) / DT)), foot] = 0.0
    t_u = np.arange(0, n, EVERY) * DT
    v = np.interp(t_u, [STOP_C + 0.3, STOP_C + settle_s], [V_CMD, residual])
    com = np.column_stack([np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t_u))]), np.zeros(len(v))])
    return t_u, com, np.column_stack([v, np.zeros(len(v))]), loads, weight


def stop(case, contact=None):
    t_u, com, v, loads, weight = case
    contact = R.contact_flags(loads, weight, DT) if contact is None else contact
    return R.stop_reading(t_u, com, v, contact, DT, STOP_C, 0.0, V_CMD, STRIDE_S)


#: The two-touchdown stop on the simulated clock: the left foot (0) lands 60 ms before the command, the right (1)
#: lifts at the command and lands 0.35 s after, the left steps from 0.41 s to 0.76 s.
TWO_STEP = [(0, -0.41, -0.06), (1, -0.9, -0.55), (1, 0.0, 0.35), (0, 0.41, 0.76)]


def test_stop_reading_counts_touchdowns():
    """Contact sequences with 2 and with 3 touchdowns after the command, one exactly at the command step, a 10 ms
    bounce, and a late landing of the earlier step."""
    r = stop(stop_case(TWO_STEP))
    assert r["reached_command"] and r["stopped_by_end"]
    assert r["touchdowns_after_command"] == 2 and r["touchdowns_to_end"] == 2
    assert r["touchdown_table"] == [[1, pytest.approx(STOP_C + 0.35), pytest.approx(0.35)],
                                    [0, pytest.approx(STOP_C + 0.76), pytest.approx(0.76)]]
    assert r["last_touchdown_before_command"] == dict(foot=0, t_s=pytest.approx(STOP_C - 0.06),
                                                      before_command_s=pytest.approx(0.06))
    # the forward speed is under 5 % of 0.42 m/s from 0.3 + 0.95 x 0.7 = 0.965 s on: the next update is at 0.97 s
    assert r["forward_stopped_s"] == pytest.approx(0.97, abs=1e-9)
    assert r["forward_stopped_strides"] == pytest.approx(0.97 / STRIDE_S, abs=1e-9)
    assert r["distance_after_command_m"] == pytest.approx(V_CMD * 0.3 + 0.5 * V_CMD * 0.7, abs=1e-3)
    # a third step before the body has stopped
    r = stop(stop_case(TWO_STEP + [(1, 0.80, 1.10)], settle_s=1.5))
    assert r["touchdowns_after_command"] == 3 and [row[0] for row in r["touchdown_table"]] == [1, 0, 1]
    # ... and one after it has stopped, which is not a touchdown of the stop
    r = stop(stop_case(TWO_STEP + [(1, 1.60, 1.90)]))
    assert r["touchdowns_after_command"] == 2 and r["touchdowns_to_end"] == 3
    # a touchdown exactly at the command step is not after the command
    at_command = [(0, -0.35, 0.0)] + TWO_STEP[1:]
    r = stop(stop_case(at_command))
    assert r["touchdowns_after_command"] == 2
    assert r["last_touchdown_before_command"] == dict(foot=0, t_s=pytest.approx(STOP_C),
                                                      before_command_s=pytest.approx(0.0, abs=1e-12))
    # ... one physics step later it is
    r = stop(stop_case([(0, -0.35, DT)] + TWO_STEP[1:]))
    assert r["touchdowns_after_command"] == 3 and r["touchdown_table"][0][:2] == [0, pytest.approx(STOP_C + DT)]
    # a 10 ms bounce after the right foot's landing: the debounce (20 ms) merges it; the raw flags would count it
    case = stop_case(TWO_STEP + [(1, 0.37, 0.38)])
    r = stop(case)
    assert r["touchdowns_after_command"] == 2
    assert stop(case, contact=case[3] > R.CONTACT_LOAD_FRAC * case[4])["touchdowns_after_command"] == 3
    # the earlier step lands late, 20 ms after the command: it counts, and the last touchdown before the command
    # is then the right foot's, more than half a second back
    late = [(0, -0.41, 0.02)] + TWO_STEP[1:]
    r = stop(stop_case(late))
    assert r["touchdowns_after_command"] == 3 and r["touchdown_table"][0] == [0, pytest.approx(STOP_C + 0.02),
                                                                             pytest.approx(0.02)]
    assert r["last_touchdown_before_command"] == dict(foot=1, t_s=pytest.approx(STOP_C - 0.55),
                                                      before_command_s=pytest.approx(0.55))
    # the speed never settles: not stopped, and the count runs to the end of the trace
    r = stop(stop_case(TWO_STEP + [(1, 1.60, 1.90)], residual=0.1))
    assert not r["stopped_by_end"] and r["forward_stopped_s"] is None and r["forward_stopped_strides"] is None
    assert r["touchdowns_after_command"] == r["touchdowns_to_end"] == 3
    # a trace that ends before the command
    t_u, com, v, loads, weight = stop_case(TWO_STEP, seconds=4.0)
    r = R.stop_reading(t_u, com, v, R.contact_flags(loads, weight, DT), DT, STOP_C, 0.0, V_CMD, STRIDE_S)
    assert not r["reached_command"] and r["touchdowns_after_command"] is None and not r["stopped_by_end"]


def test_stop_acceleration_bound():
    """A step change of 0.42 m/s in forward speed.  The 1 Hz reading cannot exceed the filter's bound, 2.2214 x dv
    = 0.933 m/s^2 (the design's "at most 0.93"): here it reads 0.932.  The raw peak is reported beside it."""
    n = int(round(12.0 / DT))
    t_u = np.arange(0, n, EVERY) * DT
    v = np.where(t_u < STOP_C + 0.5, 0.42, 0.0)
    com = np.column_stack([np.cumsum(v) * 0.01, np.zeros(len(v))])
    r = R.stop_reading(t_u, com, np.column_stack([v, np.zeros(len(v))]), np.ones((n, 2), bool), DT, STOP_C, 0.0, 0.42,
                       0.82)
    bound = math.pi / math.sqrt(2.0) * 0.42  # the zero-phase order-2 Butterworth's impulse peak at 1 Hz, times dv
    assert bound == pytest.approx(0.933, abs=5e-4)
    assert r["com_accel_forward_lp1hz_peak_m_s2"] <= bound
    assert r["com_accel_forward_lp1hz_peak_m_s2"] == pytest.approx(0.93, abs=5e-3)
    assert r["com_accel_forward_lp1hz_peak_m_s2"] < R.ACCEL_MAX_M_S2
    assert r["deceleration_raw_peak_m_s2"] == pytest.approx(0.42 / 0.02, rel=1e-9)  # central differences of the step
    assert r["stopped_by_end"] and r["touchdowns_after_command"] == 0 and r["last_touchdown_before_command"] is None


# ------------------------------------------------------------------------------------------------
# The drive term's split
# ------------------------------------------------------------------------------------------------
def test_drive_ratios_split():
    """``force_sq`` in which only the tail exceeds its sustained rating: ``drive_sustained`` false,
    ``drive_sustained_legs`` true; the worst leg drive and the worst neck or tail drive are named."""
    actuators = ["left_knee_motor", "right_hip_pitch_motor", "neck_pitch_servo", "tail_yaw_servo"]
    sustained = np.array([10.0, 12.0, 2.0, 2.5])
    leg = np.array([True, True, False, False])
    rng = np.random.default_rng(7)
    force = rng.normal(size=(4000, 4))
    force *= np.array([0.5, 0.9, 0.3, 1.2]) * sustained / np.sqrt(np.mean(force[1000:3000] ** 2, axis=0))
    force[:1000] *= 50.0  # outside the span: not read
    r = R.drive_ratios(force**2, sustained, actuators, leg, 1000, 3000)
    assert r["ratios"] == {a: pytest.approx(x, rel=1e-12) for a, x in zip(actuators, (0.5, 0.9, 0.3, 1.2), strict=True)}
    assert r["worst"] == ["tail_yaw_servo", pytest.approx(1.2)]
    assert r["worst_leg"] == ["right_hip_pitch_motor", pytest.approx(0.9)]
    assert r["worst_a1"] == ["tail_yaw_servo", pytest.approx(1.2)]
    assert r["drive_sustained"] is False and r["drive_sustained_legs"] is True and r["steps"] == [1000, 3000]
    direct = np.sqrt(np.mean((force**2)[1000:3000], axis=0)) / sustained  # metrics.window's expression
    assert list(r["ratios"].values()) == [float(x) for x in direct]
    whole = R.drive_ratios(force**2, sustained, actuators, leg, 0, 4000)  # the episode's span: the lead-in counts
    assert not whole["drive_sustained_legs"] and whole["worst"][1] > 10
    legs_only = R.drive_ratios(force**2, sustained, actuators, np.ones(4, bool), 1000, 3000)
    assert legs_only["worst_a1"] is None and legs_only["drive_sustained_legs"] is False
    with pytest.raises(ValueError):
        R.drive_ratios(force**2, sustained, actuators, leg, 3000, 3000)


# ------------------------------------------------------------------------------------------------
# What a reference's own file can say
# ------------------------------------------------------------------------------------------------
def turn_file(rate=math.pi / 4, sign=1.0, ramp=0.375, h0=0.4, speed=0.4):
    """A transition-format file of a 90 degree turn: yaw rate ramping up over ``ramp`` s, held, and ramping down, on
    frames that are not uniform in time; heading by the trapezoid rule, so linear interpolation is exact only at
    the frames."""
    hold = (math.pi / 2) / rate - ramp
    t = np.concatenate([np.linspace(0.0, 1.0, 260, endpoint=False), np.linspace(1.0, 5.0, 700)])
    w = rate * np.interp(t, [0.0, ramp, ramp + hold, 2 * ramp + hold], [0.0, 1.0, 1.0, 0.0])
    heading = h0 + sign * np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(t))])
    vel = speed * np.column_stack([np.cos(heading), np.sin(heading)])
    com = np.vstack([[0.0, 0.0], np.cumsum(0.5 * (vel[1:] + vel[:-1]) * np.diff(t)[:, None], axis=0)])
    return dict(t=t, heading=heading, com=np.column_stack([com, np.full(len(t), 0.45)]))


def test_reference_clock():
    """A transition-format array with a known heading law: ``reference_heading_at(arr, 2.0, 1.05)`` reads the
    heading at 2.1 s of the file's own time; at rate 0.9975 at 1.995 s; always less the first heading."""
    t = np.concatenate([np.linspace(0.0, 1.0, 300, endpoint=False), np.linspace(1.0, 4.0, 500)])
    arr = dict(t=t, heading=0.7 + 0.6 * t)  # 0.6 rad/s from a first heading of 0.7 rad
    assert R.reference_heading_at(arr, 2.0, 1.05) == pytest.approx(math.degrees(0.6 * 2.1), abs=1e-9)
    assert R.reference_heading_at(arr, 2.0, 0.9975) == pytest.approx(math.degrees(0.6 * 1.995), abs=1e-9)
    assert R.reference_heading_at(arr, 2.0, 1.0) == pytest.approx(math.degrees(1.2), abs=1e-9)
    assert R.reference_heading_at(arr, 0.0, 1.05) == 0.0
    assert R.reference_heading_at(dict(arr, heading=-arr["heading"]), 2.0, 1.05) == pytest.approx(
        -math.degrees(0.6 * 2.1), abs=1e-9)
    assert R.reference_heading_at(arr, 100.0, 1.05) == pytest.approx(math.degrees(0.6 * 4.0), abs=1e-9)  # held
    assert (1.0 - R.PHASE_RATE_LIMIT) * R.RATE == pytest.approx(0.9975, abs=1e-15)


def test_reference_readings_of_a_turn():
    arr = turn_file()
    r = R.reference_readings(arr, "turn", dict(rate_rad_s=math.pi / 4))
    assert r["at_s"] == 2.0 and r["heading_total_rad"] == pytest.approx(math.pi / 2, abs=1e-5)

    def turned(t_ref, ramp=0.375, rate=math.pi / 4):  # the law's own heading, deg: ramp up, hold, ramp down from 2.0 s
        if t_ref <= 2.0:
            return math.degrees(rate * (t_ref - 0.5 * ramp))
        return math.degrees(rate * (2.0 - 0.5 * ramp + (t_ref - 2.0) - (t_ref - 2.0) ** 2 / (2 * ramp)))

    assert r["heading_at_bar_deg"] == pytest.approx(turned(2.1), abs=1e-3)
    assert r["heading_at_slowest_clock_deg"] == pytest.approx(turned(1.995), abs=1e-3)
    assert r["heading_at_ref_clock_deg"] == pytest.approx(turned(2.0), abs=1e-3)
    assert r["heading_at_bar_deg"] > r["heading_at_ref_clock_deg"] > r["heading_at_slowest_clock_deg"] > 80.0
    assert r["heading_ok"]
    t80 = 0.5 * 0.375 + math.radians(80.0) / (math.pi / 4)  # reference time at which 80 deg is reached
    assert r["time_to_80deg_bar_s"] == pytest.approx(t80 / R.RATE, abs=1e-4)
    assert set(r) == {"kind", "accel_forward_lp1hz_peak_m_s2", "heading_total_rad", "at_s", "heading_at_bar_deg",
                      "heading_at_slowest_clock_deg", "heading_at_ref_clock_deg", "time_to_80deg_bar_s", "heading_ok"}
    # the right turn reads the same, in its own direction
    right = R.reference_readings(turn_file(sign=-1.0), "turn", dict(at_s=2.0))
    assert right["heading_total_rad"] == pytest.approx(-math.pi / 2, abs=1e-5)
    for key in ("heading_at_bar_deg", "heading_at_slowest_clock_deg", "heading_at_ref_clock_deg", "time_to_80deg_bar_s"):
        assert right[key] == pytest.approx(r[key], abs=1e-9)
    # a slower ramp is behind at the slowest clock: G9's rule fails
    slow = R.reference_readings(turn_file(ramp=0.5), "turn", dict(rate_rad_s=math.pi / 4))
    assert slow["heading_at_slowest_clock_deg"] < 80.0 < slow["heading_at_bar_deg"] and not slow["heading_ok"]
    # a 0.5 rad/s turn is read at 3.1416 s of bar time
    half = R.reference_readings(turn_file(rate=0.5), "turn", dict(rate_rad_s=0.5))
    assert half["at_s"] == pytest.approx(math.pi, abs=1e-12)
    with pytest.raises(ValueError):
        R.reference_readings(arr, "turn")
    # any other move: the forward acceleration only
    assert set(R.reference_readings(arr, "change")) == {"kind", "accel_forward_lp1hz_peak_m_s2"}


def test_reference_readings_of_a_stop():
    """A stop file: the forward speed ramps from 0.40 m/s to rest over [0.3, 1.0] s; two touchdowns after frame 0."""
    t = np.linspace(0.0, 3.0, 601)
    x = np.clip((t - 0.3) / 0.7, 0.0, 1.0)
    v = 0.40 * (1 - (10 * x**3 - 15 * x**4 + 6 * x**5))  # minimum-jerk ramp
    heading = np.full(len(t), math.radians(20.0))
    s = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t))])
    com = np.column_stack([s * math.cos(heading[0]), s * math.sin(heading[0]), np.full(len(t), 0.45)])
    support = np.ones((len(t), 2), bool)
    support[:70, 1] = False  # the right foot is in the air at the command and lands at frame 70
    support[82:150, 0] = False  # the left foot's last step
    walk = dict(speed=0.40, period=0.8606, duty=0.5732)  # an entry's gaits.a
    r = R.reference_readings(dict(t=t, com=com, heading=heading, support=support), "stop", walk)
    assert r["touchdowns_after_command"] == 2 and r["touchdown_feet"] == [1, 0]
    assert r["touchdown_s"] == [pytest.approx(t[70]), pytest.approx(t[150])]
    # below 5 % of 0.40 m/s where the ramp has 5 % left: x = 0.8107 (10 x^3 - 15 x^4 + 6 x^5 = 0.95)
    assert r["forward_stopped_s"] == pytest.approx(0.3 + 0.7 * 0.8107, abs=0.006)
    assert r["forward_stopped_strides"] == pytest.approx(r["forward_stopped_s"] / 0.8606, abs=1e-12)
    peak = 0.40 * 1.875 / 0.7  # a minimum-jerk ramp's largest rate
    assert r["accel_forward_raw_peak_m_s2"] == pytest.approx(peak, rel=2e-3)
    assert 0.5 * peak < r["accel_forward_lp1hz_peak_m_s2"] < peak  # the 1 Hz filter takes the top off
    assert r["accel_forward_lp1hz_peak_m_s2"] <= math.pi / math.sqrt(2.0) * 0.40
    assert r["margin_before_command_s"] == pytest.approx(0.0732 * 0.8606, abs=1e-12)
    same = R.reference_readings(dict(t=t, com=com, heading=heading, support=support), "stop",
                                dict(speed_m_s=0.40, period_s=0.8606))
    assert same["forward_stopped_s"] == r["forward_stopped_s"] and same["margin_before_command_s"] is None
    with pytest.raises(ValueError):
        R.reference_readings(dict(t=t, com=com, heading=heading, support=support), "stop")


# ------------------------------------------------------------------------------------------------
# Against Stage 1, and the module's imports
# ------------------------------------------------------------------------------------------------
def test_lowpass_equals_stage1s():
    """``readings.lowpass`` is ``metrics.lowpass``: random series, at the rows' and the physics steps' rates, and
    the short series the filter cannot take."""
    from designlab.stage1 import metrics

    rng = np.random.default_rng(11)
    for n in (0, 1, 5, 9, 10, 37, 400, 3001):
        x = rng.normal(size=n)
        for fs in (100.0, 500.0):
            assert np.array_equal(R.lowpass(x, fs), metrics.lowpass(x, fs))
    x = rng.normal(size=500)
    assert np.array_equal(R.lowpass(x, 100.0, 2.5, 4), metrics.lowpass(x, 100.0, 2.5, 4))
    assert (R.ACCEL_LP_HZ, R.ACCEL_LP_ORDER) == (metrics.ACCEL_LP_HZ, metrics.ACCEL_LP_ORDER)


def test_contact_and_constants_equal_stage1s():
    """The contact flags are ``metrics.Recorder.contact``'s, and the constants this module restates are Stage
    1's."""
    from designlab import clearance
    from designlab.stage1 import control, metrics, panels

    rng = np.random.default_rng(5)
    for _ in range(20):
        flag = rng.random(600) < rng.choice([0.2, 0.5, 0.8])
        flag = np.repeat(flag, rng.integers(1, 9, size=len(flag)))[:900]
        for k in (1, 3, 10):
            assert np.array_equal(R.debounce(flag, k), metrics.debounce(flag, k))
    loads = np.abs(rng.normal(size=(2000, 2))) * 40.0
    rec = metrics.Recorder.__new__(metrics.Recorder)
    rec.loads, rec.weight, rec.dt = loads, 300.0, DT
    assert np.array_equal(R.contact_flags(loads, 300.0, DT), rec.contact())
    assert (R.CONTACT_LOAD_FRAC, R.DEBOUNCE_S) == (control.CONTACT_LOAD_FRAC, metrics.DEBOUNCE_S)
    assert (R.RATE, R.PHASE_RATE_LIMIT) == (panels.RATE, control.PHASE_RATE_LIMIT)
    assert R.MID_SWING == tuple(clearance.MID_SWING)
    assert EVERY == control.FEEDBACK_EVERY


def test_readings_loads_neither_mujoco_nor_stage1():
    code = ("import sys; import designlab.stage2.readings; "
            "bad = [m for m in sys.modules if m == 'mujoco' or m.startswith('designlab.stage1')]; "
            "assert not bad, bad")
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
