"""The room-temperature option of the thermal model and the demo course (decision D48, 27 Nov 2026).

Left at its default (25 C, the calibration's room) it changes nothing: a stored demo course is reproduced exactly
through ``demo_course.run`` from its stored segments. A warmer room raises the air the windings shed heat to and
leaves each drive's fit alone; refitting at the room instead (``ambient_c``) is the optimistic bound."""

from __future__ import annotations

import json
import sys
from dataclasses import replace

import pytest
from telemetry.thermal import RATING_AMBIENT_C, DriveThermal, ThermalNode, demo_temperatures

from conftest import RL_DIR

sys.path.insert(0, str(RL_DIR / "analysis"))
import thermal_ambient as ta  # noqa: E402

STORED = RL_DIR / "analysis" / "results" / "d22" / "demo" / "fore128-s3" / "demo_course.json"
X4 = DriveThermal("x4", kt_nm_per_a=1.8, r_phase_ohm=0.2, thermal_eq_nm=7.5)
LOAD = [(600.0, 7.5 ** 2), (300.0, 0.0), (1200.0, 5.0 ** 2)]


def _from_stored(monkeypatch, stored):
    """demo_course with its simulated segments replaced by the stored ones."""
    import demo_course

    segs = stored["segments"]
    monkeypatch.setattr(demo_course, "measure_segment",
                        lambda make, command, seed, seconds=20.0: next(
                            s for n, s in segs.items() if demo_course.SEGMENTS[n] == command))
    return demo_course


def test_default_room_reproduces_a_stored_demo_course_exactly(monkeypatch, tmp_path, capsys):
    """The command as the D22 demo courses were run (rl/runs/trike60-d22_LOG.md), with the room left at its
    default: every stored figure (temperatures, traces, battery, loads, minutes) comes back bit for bit, the only
    new key is the room itself, and the printout is the stored one."""
    stored = json.loads(STORED.read_text())
    demo_course = _from_stored(monkeypatch, stored)
    monkeypatch.setattr(demo_course, "policy_factory", lambda ckpt_dir: (None, stored["walker"]))
    assert demo_course.main(["--ckpt-dir", "checkpoints", "--out", str(tmp_path), "--label", "trike60-fore128-s3"]) == 0
    res = json.loads((tmp_path / "demo_course.json").read_text())
    assert set(res) - set(stored) == {"room_c"} and res["room_c"] == RATING_AMBIENT_C == 25.0
    for k in stored:
        assert res[k] == stored[k], k
    printed = capsys.readouterr().out.splitlines()
    assert printed[:-1] == (STORED.parent / "demo_course_printout.txt").read_text().splitlines()[:-1]  # not "wrote"


def test_the_room_option_reaches_the_command_line(monkeypatch, tmp_path, capsys):
    """``--room-c 35`` through demo_course.main: the result records the room, its temperatures are run(room_c=35)'s,
    and the printout and the chart say the room."""
    stored = json.loads(STORED.read_text())
    demo_course = _from_stored(monkeypatch, stored)
    monkeypatch.setattr(demo_course, "policy_factory", lambda ckpt_dir: (None, stored["walker"]))
    assert demo_course.main(["--ckpt-dir", "checkpoints", "--out", str(tmp_path), "--label", "trike60-fore128-s3",
                             "--room-c", "35"]) == 0
    res = json.loads((tmp_path / "demo_course.json").read_text())
    direct = demo_course.run(None, stored["walker"], [tuple(x) for x in stored["loop"]], None, stored["seed0"],
                             room_c=35.0)
    assert res["room_c"] == 35.0 and res["minutes"] == stored["minutes"]
    assert res["temperatures"] == json.loads(json.dumps(direct["temperatures"]))
    hot = max(res["temperatures"], key=lambda j: res["temperatures"][j]["tau_480s"]["peak_c"])
    assert hot == "left_hind_pitch" and 98.0 < res["temperatures"][hot]["tau_480s"]["peak_c"] < 98.6  # 98.3 C
    printed = capsys.readouterr().out
    assert "in a 35 C room (time constant 8 min)" in printed
    assert all(g["temp_c"][0] == 35.0 for g in res["traces"]["groups"].values())


def test_the_optimistic_figure_is_never_below_the_25c_one():
    """Method B can put a drive at or above its rating below its 25 C figure in a warmer room (160 mm seed 0's demo,
    the hottest drive at 1.58 of rating while standing); the optimistic figure is then the stored 25 C one."""
    stored = json.loads((STORED.parents[1] / "fore160-s0" / "demo_course.json").read_text())
    with ta._cached_fits():
        therm, per_joint, _ = ta.thermal_inputs(stored)
        b35 = ta.temperatures(therm, per_joint, 35.0, "B", (480.0,))
        hb = ta.hottest_summary(b35, therm, per_joint, 35.0, "B", (480.0,))
    opt = ta.optimistic_peaks(hb, stored["temperatures"])["tau_480s"]
    s25 = max(v["tau_480s"]["peak_c"] for v in stored["temperatures"].values())
    assert opt["b_below_stored_25c"] and opt["b_peak_c"] < s25 == opt["stored_25c_peak_c"] == opt["optimistic_peak_c"]
    assert 109.8 < s25 < 110.0 and 108.2 < opt["b_peak_c"] < 108.6  # 109.9 and 108.4 C


def test_a_warmer_room_raises_the_sink_and_keeps_the_fit(monkeypatch, tmp_path):
    """Through demo_course.run at 35 C: the same R_th and C as the stored 25 C run, every winding starts at 35 C and
    the hottest ends more than 10 K warmer (the copper's resistance rises with it); the chart says the room."""
    stored = json.loads(STORED.read_text())
    demo_course = _from_stored(monkeypatch, stored)
    res = demo_course.run(None, stored["walker"], [tuple(x) for x in stored["loop"]], None, stored["seed0"],
                          time_constants_s=(480.0,), room_c=35.0)
    assert res["room_c"] == 35.0 and res["minutes"] == stored["minutes"] and res["battery"] == stored["battery"]
    for jn, v in res["temperatures"].items():
        new, old = v["tau_480s"], stored["temperatures"][jn]["tau_480s"]
        assert new["r_th_k_per_w"] == old["r_th_k_per_w"] and new["c_j_per_k"] == old["c_j_per_k"]
        assert new["steady_at_thermal_eq_c"] == old["steady_at_thermal_eq_c"]  # the fit's own, in its own room
        assert 10.0 < new["peak_c"] - old["peak_c"] < 13.0 and 10.0 < new["end_c"] - old["end_c"] < 13.0
    for g in res["traces"]["groups"].values():
        assert g["temp_c"][0] == 35.0
    assert demo_course.chart(res, tmp_path, "35 C") == "demo_temperatures.png"


def _chart_seen(monkeypatch, demo_course, res, out):
    """demo_course.chart's y-limits, legend corner, its traces' end labels, and the traces and end labels that pass
    behind the legend, read off the drawn figure before it is saved."""
    import chart_style as cs

    seen = {}
    orig = cs.save

    def save(fig, path):
        ax = fig.axes[0]
        seen["ylim"] = ax.get_ylim()
        seen["legend_loc"] = ax.get_legend()._loc
        seen["labels"] = {t.get_text(): t.xy[1] for t in ax.texts if t.get_text() in res["traces"]["groups"]}
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        box = ax.get_legend().get_window_extent(r)
        seen["behind_legend"] = [ln.get_label() for ln in ax.get_lines() if not ln.get_label().startswith("_")
                                 and ln.get_transform().transform_path(ln.get_path()).intersects_bbox(box, filled=False)]
        seen["behind_legend"] += [t.get_text() for t in ax.texts if t.get_window_extent(r).overlaps(box)]
        orig(fig, path)

    monkeypatch.setattr(cs, "save", save)
    assert demo_course.chart(res, out, "chart") == "demo_temperatures.png"
    return seen


@pytest.mark.parametrize("run_dir, room, tcs", [
    ("demo/fore160-s0", 35.0, None),  # the hottest winding 123.2 C at 8 min, over the old 110 C top
    ("demo/fore128-s0", 15.0, (480.0,)),  # a room under the old 20 C bottom
    ("demo/fore160-s3", 26.0, (480.0,)),  # a roll trace that settled behind a lower-right legend on a 20 C bottom
    ("demo_walkonly/fore160-s0", None, None),  # the default: its pitch trace ends at 110.2 C, just over the top
])
def test_the_chart_keeps_another_room_in_frame_and_the_25c_chart_as_before(monkeypatch, tmp_path, run_dir, room, tcs):
    """In a room other than 25 C the chart's y-range takes in every trace with 5 C to spare and the legend moves to
    the lower right, with the axis bottom taken low enough that no trace or end label passes behind it, so every
    trace and its end label stays in frame and in view; at 25 C the chart is as before (20-110 C, the legend upper
    left), even where a stored trace runs just over 110 C."""
    stored = json.loads((STORED.parents[2] / run_dir / "demo_course.json").read_text())
    demo_course = _from_stored(monkeypatch, stored)
    minutes = None if stored["minutes_is_one_pack"] else stored["minutes"]
    res = demo_course.run(None, stored["walker"], [tuple(x) for x in stored["loop"]], minutes, stored["seed0"],
                          time_constants_s=tcs, room_c=room)
    assert res["minutes"] == stored["minutes"]
    seen = _chart_seen(monkeypatch, demo_course, res, tmp_path)
    temps = [T for g in res["traces"]["groups"].values() for T in g["temp_c"]]
    assert len(seen["labels"]) == len(res["traces"]["groups"])
    if room is None:
        assert seen["ylim"] == (20.0, 110.0) and seen["legend_loc"] == 2 and max(temps) > 110.0  # 2: upper left
        return
    lo, hi = seen["ylim"]
    assert lo <= min(temps) - 5.0 and hi >= max(temps) + 5.0 and seen["legend_loc"] == 4  # 4: lower right
    assert all(lo <= y <= hi for y in seen["labels"].values()) and seen["behind_legend"] == []
    if room == 35.0:  # the legend clears the traces on the 20 C bottom: no lowering needed
        assert 123.0 < max(temps) < 123.3 and (lo, hi) == (20.0, 130.0)
    elif room == 15.0:
        assert min(temps) == 15.0 and (lo, hi) == (5.0, 110.0)
    else:  # 26 C: on a 20 C bottom the roll trace (about 35 C) and its end label ran behind the legend
        assert min(temps) == 26.0 and (lo, hi) == (10.0, 110.0)


def test_room_none_and_25_are_the_same_and_35_moves_only_the_sink():
    for tc in (240.0, 480.0, 900.0):
        n, n25, n35 = X4.calibrate(tc), X4.calibrate(tc, room_c=25.0), X4.calibrate(tc, room_c=35.0)
        assert (n.r_th_k_per_w, n.c_j_per_k) == (n25.r_th_k_per_w, n25.c_j_per_k) == \
            (n35.r_th_k_per_w, n35.c_j_per_k)
        assert n.room_c is None and n.air_c == n25.air_c == 25.0 and n35.air_c == 35.0
        assert n.run(LOAD) == n25.run(LOAD) and n.steady_state_c(30.0) == n25.steady_state_c(30.0)
        assert all(T == 35.0 for _, T in n35.run([(600.0, 0.0)]))  # no load: the winding sits at the room
        ss25, ss35 = n.steady_state_c(7.5 ** 2), n35.steady_state_c(7.5 ** 2)
        assert 10.0 < ss35 - ss25 < 15.0  # 10 K of room, plus the copper's rise
        assert n35.run([(12 * tc, 7.5 ** 2)])[-1][1] == pytest.approx(ss35, abs=0.2)
        # the cool-down back to 25 C from 80 C in the calibration's room is unchanged; in a 35 C room it ends at 35
        assert n35.run([(8 * tc, 0.0)], start_c=80.0)[-1][1] == pytest.approx(35.0, abs=0.02)
    base = demo_temperatures({"x4": X4}, {"x4": LOAD})
    assert demo_temperatures({"x4": X4}, {"x4": LOAD}, room_c=25.0) == base
    warm = demo_temperatures({"x4": X4}, {"x4": LOAD}, room_c=40.0)["x4"]
    for k, v in warm.items():
        assert v["peak_c"] > base["x4"][k]["peak_c"] + 15.0
        assert v["steady_at_thermal_eq_c"] == base["x4"][k]["steady_at_thermal_eq_c"]
        assert v["cooldown_minutes"] is None  # a 40 C room never cools a winding to 40 C


def test_refitting_at_the_room_is_the_optimistic_bound():
    """Refitting as if the maker's 90 min / 100 C test had started at 35 C needs a lower R_th (about 65/75 of the
    25 C fit's) and gives lower temperatures than the consistent method in the same room."""
    for tc in (240.0, 480.0, 900.0):
        a = X4.calibrate(tc, room_c=35.0)
        b = replace(X4, ambient_c=35.0).calibrate(tc)
        assert b.air_c == 35.0 and b.run([(5400.0, 7.5 ** 2)])[-1][1] == pytest.approx(100.0, abs=0.05)
        assert 0.84 < b.r_th_k_per_w / a.r_th_k_per_w < 0.90
        assert b.run(LOAD)[-1][1] < a.run(LOAD)[-1][1] and b.steady_state_c(30.0) < a.steady_state_c(30.0)
    node = ThermalNode(X4, 5.0, 96.0, room_c=30)
    assert node.air_c == 30.0 and isinstance(node.air_c, float)


def test_the_room_analysis_reproduces_the_stored_25c_figures():
    """thermal_ambient.py's 25 C column is the stored demo course, exactly, for every drive and time constant."""
    stored = json.loads(STORED.read_text())
    with ta._cached_fits():
        therm, per_joint, _ = ta.thermal_inputs(stored)
        for m in ("A", "B"):
            assert ta.check_reproduces(stored["temperatures"], ta.temperatures(therm, per_joint, 25.0, m)) == []
        a35 = ta.temperatures(therm, per_joint, 35.0, "A", (480.0,))
        b35 = ta.temperatures(therm, per_joint, 35.0, "B", (480.0,))
    hot = max(a35, key=lambda j: a35[j]["tau_480s"]["peak_c"])
    assert hot == "left_hind_pitch" and b35[hot]["tau_480s"]["peak_c"] < a35[hot]["tau_480s"]["peak_c"]
