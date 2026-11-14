"""Tests for the ToF speed governor (``stage2/jetson75/tof_governor.py``, ``tof_governor.json``,
``TOF_GOVERNOR.md``; ``JETSON_PLEM_REVIEW.md`` section 5, task 10).

* The committed ``tof_governor.json`` regenerates exactly from the committed records.
* Two governor rows are recomputed here from the raw records with code that shares nothing with the module: plain
  trigonometry for the zone rays, this file's own stride rule and stop floors, a closed form for one row and a
  brute-force scan over speed for the other.
* A longer reaction time, a shorter sight distance or a weaker stop never raises the governor; no step reading asks
  for more than the spec's 1.5 m/s^2; a what-if that changes nothing changes nothing.
* Every input has a kind; every (c) cites a study document, a catalog id or a web source that exists; every (d) says
  whose it is and how it was derived; the cited strings are in the cited documents.
* The tables of ``TOF_GOVERNOR.md`` are the record's, and so are the numbers its brief and its prose quote.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "stage2" / "jetson75"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RUN))

import designlab  # noqa: E402,F401
import tof_governor as tg  # noqa: E402

DOC = json.loads((RUN / "tof_governor.json").read_text())
BODIES = ("75-M-H-10", "60-N-T-10")
STEP_READINGS = ("one_step", "two_steps", "stop_builder_worst", "spec_strides")
G = 9.81


def _record(body: str, key: str) -> dict:
    return json.loads((ROOT / DOC["provenance"]["records"][body][key]["path"]).read_text())


def _terms(mode_hz: float, packet_bytes: int, names: tuple) -> tuple[float, float]:
    """This file's own low and high reaction times (s) for a list of term names."""
    low = dict(frames=1 / mode_hz, i2c=packet_bytes * 9 / 1e6, loop=1 / 1000, walk=1 / 100, delay=0.0,
               planner=0.0, link=0.0)
    high = dict(frames=3 / mode_hz, i2c=2 * packet_bytes * 9 / 4e5, loop=1 / 200, walk=1 / 50, delay=0.020,
                planner=0.200, link=0.005)
    return sum(low[n] for n in names), sum(high[n] for n in names)


def _far_row(body: str, n: int, drop: float) -> list[float]:
    """Far-row zone centres on the foot path, m ahead of the toes, by plain trigonometry (equidistant lens, 45 deg
    down, 12 deg outward), with the sensors lowered by ``drop``."""
    sen = _record(body, "sensors")
    half = sen["tof"]["near_coverage"]["foot_path_half_width_m"]
    toe = sen["model"]["toe_tip_x_m"]
    b_img = 1 - 1 / n  # the far row's centre
    ahead = []
    for name, sgn in (("tof_l", 1.0), ("tof_r", -1.0)):
        ox, oy, oz = sen["tof"][name]["summary"]["site_pos_m"]
        oz -= drop
        yaw, dep = math.radians(12.0 * sgn), math.radians(45.0)
        for k in range(n):
            a_img = -1 + (2 * k + 1) / n
            u, w = a_img * math.radians(22.5), b_img * math.radians(22.5)
            th, ph = math.hypot(u, w), math.atan2(w, u)  # equidistant projection, as designlab/sensors.py
            lx, ly, lz = math.cos(th), -math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph)
            px, pz = lx * math.cos(dep) + lz * math.sin(dep), -lx * math.sin(dep) + lz * math.cos(dep)  # pitch down
            wx, wy = px * math.cos(yaw) - ly * math.sin(yaw), px * math.sin(yaw) + ly * math.cos(yaw)  # yaw outward
            t = -oz / pz
            if abs(oy + wy * t) <= half:
                ahead.append(ox + wx * t - toe)
    return ahead


def _walk(H: float, v: float) -> tuple[float, float]:
    """This file's own (stride, duty) of the reference walk at speed v: period and duty linear in speed between the
    three reference walks, and held outside them."""
    walks = [(math.sqrt(fr * G * H), rel * H, duty) for fr, rel, duty in ((0.012, 0.55, 0.60), (0.025, 0.62, 0.58),
                                                                           (0.05, 0.80, 0.56))]
    periods = [s / sp for sp, s, _ in walks]
    if v <= walks[0][0]:
        return v * periods[0], walks[0][2]
    if v >= walks[2][0]:
        return v * periods[2], walks[2][2]
    k = 0 if v < walks[1][0] else 1
    w = (v - walks[k][0]) / (walks[k + 1][0] - walks[k][0])
    return v * (periods[k] + w * (periods[k + 1] - periods[k])), walks[k][2] + w * (walks[k + 1][2] - walks[k][2])


def test_json_regenerates_exactly():
    assert tg.dumps(tg.compute()) == (RUN / "tof_governor.json").read_text()
    assert tuple(DOC["bodies"]) == BODIES  # the 75 cm body first, the 60 cm body beside it
    assert set(DOC["paths"]) == {"mcu", "rt_core", "host_planner"}
    assert tuple(DOC["modes"]) == ("8x8_15hz", "4x4_60hz", "4x4_30hz")
    assert tuple(DOC["readings"]) == ("spec_decel", "one_foot_hold", "one_step", "two_steps", "stop_builder_worst",
                                      "spec_strides")


@pytest.mark.parametrize("body", BODIES)
def test_far_row_reproduces_the_screen_record(body):
    b = DOC["bodies"][body]
    sen = _record(body, "sensors")
    assert b["zone_reproduction_max_error_m"] < 5e-4  # the record rounds its sensor position to 0.1 mm
    recorded = sen["summary"]["tof"]["tof_l"]["floor_ahead_of_toes_m"][1]
    h8 = b["horizons"]["8x8_15hz"]
    assert h8["screen_record_far_m"] == recorded
    assert h8["home_far_m"] == pytest.approx(recorded, abs=5e-4)
    # the screen's recorded zones give the home-pose range directly
    half = sen["tof"]["near_coverage"]["foot_path_half_width_m"]
    zones = [z["ahead_of_toes_m"] for s in ("tof_l", "tof_r") for z in sen["tof"][s]["zones"]
             if z["row"] == 0 and abs(z["floor_xy"][1]) <= half]
    assert len(zones) == 8
    assert (h8["home_all_m"], h8["home_far_m"]) == pytest.approx((min(zones), max(zones)), abs=5e-4)
    # the main table is at the walking height: the reference walks' crouch below the home pose
    assert b["crouch_m"] == pytest.approx(0.03 * b["H_m"], abs=1e-5)
    assert _record(body, "transitions")["transitions"]["stop"]["gaits"]["a"]["crouch"] == pytest.approx(
        b["crouch_m"], abs=1e-5)
    for mode, h in b["horizons"].items():
        n = DOC["modes"][mode]["zones"]
        assert (h["all_m"], h["far_m"]) == pytest.approx((min(_far_row(body, n, b["crouch_m"])),
                                                           max(_far_row(body, n, b["crouch_m"]))), abs=2e-4)
        assert (h["home_all_m"], h["home_far_m"]) == pytest.approx((min(_far_row(body, n, 0.0)),
                                                                     max(_far_row(body, n, 0.0))), abs=2e-4)
        assert h["all_m"] <= h["ref_m"] <= h["far_m"]
        assert h["all_m"] < h["home_all_m"] and h["far_m"] < h["home_far_m"]  # lower sees less far
        assert h["alt_m"]["standing"] == [h["home_all_m"], h["home_far_m"]]
        for key in ("full_zone", "second_row", "pitch_rms", "pitch_peak"):  # every adverse reading is nearer
            assert h["alt_m"][key][0] < h["all_m"] and h["alt_m"][key][1] < h["far_m"], key
            assert h["alt_m"][key][0] < h["alt_m"][key][1]
        assert h["alt_m"]["pitch_peak"][1] < h["alt_m"]["pitch_rms"][0]
        assert 0.0 < h["far_row_floor_angle_deg"][0] < h["far_row_floor_angle_deg"][1] < 45.0
    # the 4x4 zones are four times the solid angle: nearer centres, and a taller obstacle to fill one
    for m4 in ("4x4_60hz", "4x4_30hz"):
        assert b["horizons"][m4]["far_m"] < h8["far_m"]
        assert b["horizons"][m4]["far_row_fill_height_m"] > h8["far_row_fill_height_m"]
    geometry = {k: v for k, v in b["horizons"]["4x4_60hz"].items()}
    assert geometry == b["horizons"]["4x4_30hz"]  # the rate changes the timing, not what is seen


def test_row_one_recomputed_in_closed_form():
    """75-M-H-10, 8x8 at 15 Hz, the microcontroller path, the one-foot hold: own rays, closed form."""
    body = "75-M-H-10"
    env, cand = _record(body, "envelope"), _record(body, "candidate")
    ahead = _far_row(body, 8, 0.03 * cand["H"])
    assert len(ahead) == 8
    a = env["stopping"]["max_one_foot_decel_ms2"]
    t_low, t_high = _terms(15.0, 316, ("frames", "i2c", "loop", "walk", "delay"))

    def v_of(d, t):
        return a * (-t + math.sqrt(t * t + 2 * d / a))

    cell = DOC["bodies"][body]["governor"]["8x8_15hz"]["mcu"]["one_foot_hold"]
    assert cell["low_m_s"] == pytest.approx(v_of(min(ahead), t_high), abs=1e-3)
    assert cell["high_m_s"] == pytest.approx(v_of(max(ahead), t_low), abs=1e-3)
    total = DOC["reaction"]["8x8_15hz"]["mcu"]["total_low_ref_high_s"]
    assert (total[0], total[2]) == pytest.approx((t_low, t_high), abs=1e-5)
    # the condition itself at the reported speeds; the foot's lead is the shorter distance there
    for v, d, t in ((cell["low_m_s"], min(ahead), t_high), (cell["high_m_s"], max(ahead), t_low)):
        assert d - v * t - v * v / (2 * a) == pytest.approx(0.0, abs=2e-4)
        stride, duty = _walk(cand["H"], v)
        assert 0.5 * duty * stride < v * v / (2 * a)


def test_row_two_recomputed_by_a_scan():
    """60-N-T-10, 4x4 at 60 Hz, the host path, two steps: own rays, own stride rule, brute force over speed."""
    body = "60-N-T-10"
    H = _record(body, "candidate")["H"]
    ahead = _far_row(body, 4, 0.03 * H)
    assert len(ahead) == 4
    hz = DOC["bodies"][body]["horizons"]["4x4_60hz"]
    assert (hz["all_m"], hz["far_m"]) == pytest.approx((min(ahead), max(ahead)), abs=2e-4)
    t_low, t_high = _terms(60.0, 124, ("frames", "i2c", "planner", "link", "loop", "walk", "delay"))

    def stop(v, n):  # n steps, never shorter than the spec's limit allows or than a planted foot's lead
        stride, duty = _walk(H, v)
        return max(n * 0.5 * stride, v * v / (2 * 1.5), 0.5 * duty * stride)

    def scan(d, t, n):
        best, v = 0.0, 0.0
        while v < 4.0:
            if d - v * t - stop(v, n) >= 0.0:
                best = v
            v += 1e-4
        return best

    cell = DOC["bodies"][body]["governor"]["4x4_60hz"]["host_planner"]["two_steps"]
    assert cell["low_m_s"] == pytest.approx(scan(min(ahead), t_high, 2), abs=5e-4)
    assert cell["high_m_s"] == pytest.approx(scan(max(ahead), t_low, 2), abs=5e-4)
    one = DOC["bodies"][body]["governor"]["4x4_60hz"]["host_planner"]["one_step"]  # the capped reading
    assert one["high_m_s"] == pytest.approx(scan(max(ahead), t_low, 1), abs=5e-4)
    assert one["high"]["stop_set_by"] == "spec_decel"
    # this file's stride rule agrees with the module's at the bar's speed and at a ladder rung
    ctx = tg.load_body(body)
    for v in (0.30, 0.40):
        assert tg.stride_m(ctx, v)[0] == pytest.approx(_walk(H, v)[0], rel=1e-9)
        assert tg.foot_lead_m(ctx, v) == pytest.approx(0.5 * _walk(H, v)[1] * _walk(H, v)[0], rel=1e-9)
    assert DOC["bodies"][body]["ledger_at_bar"]["4x4_60hz"]["host_planner"]["stop_m"]["two_steps"] == pytest.approx(
        _walk(H, 0.40)[0], abs=1e-4)


@pytest.mark.parametrize("body", BODIES)
def test_governor_is_monotonic(body):
    ctx = tg.load_body(body)
    for reading in tg.READINGS:
        for d in (0.3, 0.6, 0.9, 1.2):
            prev = None
            for t in (0.0, 0.03, 0.08, 0.17, 0.26, 0.46, 0.9):  # a longer reaction time never raises the speed
                v = tg.governor(ctx, reading, d, t)
                assert v >= 0.0
                assert d - v * t - tg.stop_distance(ctx, reading, v) == pytest.approx(0.0, abs=1e-9)
                if prev is not None:
                    assert v <= prev + 1e-12
                prev = v
        for t in (0.05, 0.3):  # a shorter sight distance never raises it
            vs = [tg.governor(ctx, reading, d, t) for d in (0.2, 0.4, 0.8, 1.1)]
            assert vs == sorted(vs)
        # the distance needed, v t + d_stop(v), rises with speed, so the bisection finds the one crossing
        for t in (DOC["reaction"]["4x4_60hz"]["mcu"]["total_low_ref_high_s"][0], 0.26, 0.46):
            need = [v * t + tg.stop_distance(ctx, reading, v) for v in [0.01 * k for k in range(1, 400)]]
            assert need == sorted(need)
    for d, t in ((0.5, 0.1), (1.0, 0.26), (1.1, 0.46)):  # a weaker stop never raises it
        weaker = dict(ctx, one_foot=0.8 * ctx["one_foot"])
        assert tg.governor(weaker, "one_foot_hold", d, t) <= tg.governor(ctx, "one_foot_hold", d, t)
        assert tg.governor(ctx, "one_foot_hold", d, t) <= tg.governor(ctx, "spec_decel", d, t)  # hold < 1.5 m/s^2
        vs = [tg.governor(ctx, "two_steps", d, t, steps=n) for n in (0.5, 1.0, 2.0, 2.4, 3.0)]
        assert vs == sorted(vs, reverse=True)
        order = [tg.governor(ctx, r, d, t) for r in ("spec_strides", "stop_builder_worst", "two_steps", "one_step",
                                                     "spec_decel")]
        assert order == sorted(order)  # more steps, less speed; and no step reading beats the spec's limit
    assert ctx["one_foot"] < tg.SPEC_DECEL
    # and in the record: ranges are ordered, the host path is never faster, the stress case is the lowest
    b = DOC["bodies"][body]
    for m in DOC["modes"]:
        for p in DOC["paths"]:
            total = DOC["reaction"][m][p]["total_low_ref_high_s"]
            assert total[0] < total[1] < total[2]
            cells = b["governor"][m][p]
            for r in DOC["readings"]:
                c = cells[r]
                assert 0.0 < c["low_m_s"] < c["high_m_s"]
                assert (c["low"]["m_s"], c["high"]["m_s"]) == (c["low_m_s"], c["high_m_s"])
                assert b["governor"][m]["host_planner"][r]["low_m_s"] <= c["low_m_s"]
                assert b["governor"][m]["host_planner"][r]["high_m_s"] <= c["high_m_s"]
                assert b["stress"][m][p]["governor"][r]["governor_m_s"] < c["low_m_s"]
            for end in ("low_m_s", "high_m_s"):
                order = [cells[r][end] for r in ("spec_strides", "stop_builder_worst", "two_steps", "one_step",
                                                 "spec_decel")]
                assert order == sorted(order)
                assert cells["one_foot_hold"][end] <= cells["spec_decel"][end]
        # nothing sourced separates the microcontroller from the real-time core: the two paths give the same range
        assert b["governor"][m]["mcu"] == b["governor"][m]["rt_core"]
        # the host's planner can only add time: the same high end, a lower low end
        assert DOC["reaction"][m]["host_planner"]["total_low_ref_high_s"][0] == DOC["reaction"][m]["mcu"][
            "total_low_ref_high_s"][0]
    lows = [c["low_m_s"] for m in DOC["modes"] for p in DOC["paths"] for c in b["governor"][m][p].values()]
    assert b["lowest_governor_m_s"] == min(lows)
    at = b["lowest_governor_at"]
    assert b["governor"][at["mode"]][at["path"]][at["reading"]]["low_m_s"] == min(lows)
    for r in DOC["readings"]:
        assert b["lowest_by_reading_m_s"][r] == min(b["governor"][m][p][r]["low_m_s"] for m in DOC["modes"]
                                                    for p in DOC["paths"])
    assert b["highest_governor_m_s"] == max(c["high_m_s"] for m in DOC["modes"] for p in DOC["paths"]
                                            for c in b["governor"][m][p].values())


@pytest.mark.parametrize("body", BODIES)
def test_no_step_reading_asks_for_more_than_the_spec_limit(body):
    """A stop counted in steps is never shorter than v^2 / (2 x 1.5 m/s^2): the spec's limit binds every stop."""
    b = DOC["bodies"][body]
    capped = 0
    for m in DOC["modes"]:
        for p in DOC["paths"]:
            cells = b["governor"][m][p]
            for r in STEP_READINGS:
                for end in ("low", "high"):
                    e = cells[r][end]
                    implied = e["m_s"] ** 2 / (2 * e["d_stop_m"])
                    assert implied <= tg.SPEC_DECEL + 2e-3, (m, p, r, end)
                    assert e["implied_decel_ms2"] == pytest.approx(implied, abs=2e-3)
                    assert e["step_over_H"] == pytest.approx(e["step_m"] / b["H_m"], abs=2e-3)
                    assert e["stop_set_by"] in ("reading", "spec_decel")  # a step is longer than the foot's lead
                    if e["stop_set_by"] == "spec_decel":  # then the cell's end is the spec-limit cell's
                        capped += 1
                        assert e["m_s"] == pytest.approx(cells["spec_decel"][end]["m_s"], abs=1e-4)
                    else:
                        n = {"one_step": 1, "two_steps": 2, "stop_builder_worst": 2.4, "spec_strides": 3}[r]
                        assert e["d_stop_m"] == pytest.approx(n * e["step_m"], abs=3e-4)
            for r in ("spec_decel", "one_foot_hold"):  # the deceleration readings stop as they say
                for end in ("low", "high"):
                    e = cells[r][end]
                    a = tg.SPEC_DECEL if r == "spec_decel" else b["one_foot_decel_ms2"]
                    assert e["stop_set_by"] == "reading" and e["implied_decel_ms2"] == pytest.approx(a, abs=2e-3)
            assert cells["two_steps"]["low"]["stop_set_by"] == cells["two_steps"]["high"]["stop_set_by"] == "reading"
            assert cells["one_step"]["high"]["stop_set_by"] == "spec_decel"  # one step from there is too hard a stop
    assert capped > 0
    # the foot's lead matters only at low speed: below every deceleration cell of the governor table
    for r in ("spec_decel", "one_foot_hold"):
        assert b["foot_lead"]["binds_below_m_s"][r] < b["lowest_by_reading_m_s"][r]
        assert b["foot_lead"]["binds_below_m_s"][r] > max(tg.LADDER)  # so it does matter at the bar and the ladder


@pytest.mark.parametrize("body", BODIES)
def test_a_what_if_changes_one_thing(body):
    """Each sight what-if is taken at the midpoint of its own range, as the reference is; no change, no change."""
    ctx = tg.load_body(body)
    for mode in tg.MODES:
        n = tg.MODES[mode]["zones"]
        hz = tg.horizons(ctx, mode)
        assert tg.sight(ctx, n, pitch_deg=0.0) == [hz["all_m"], hz["far_m"]]
        for path in ("mcu", "host_planner"):
            s = tg._sensitivity(ctx, mode, path, "two_steps", hz)
            rec = DOC["bodies"][body]["sensitivity"][mode][path]["two_steps"]
            assert rec["base"]["governor_m_s"] == s["base"]["governor_m_s"]
            t_ref = tg.reaction(mode, path, "ref")
            assert rec["base"]["reaction_s"] == pytest.approx(t_ref, abs=1e-5)
            assert rec["base"]["d_see_m"] == pytest.approx(0.5 * (hz["all_m"] + hz["far_m"]), abs=1e-4)
            base = tg.governor(ctx, "two_steps", hz["ref_m"], t_ref)
            unchanged = tg.governor(ctx, "two_steps", tg._mid(tg.sight(ctx, n, pitch_deg=0.0)), t_ref)
            assert unchanged - base == 0.0
            by = {w["what_if"]: w for w in rec["what_ifs"]}
            for key, w in tg.SIGHT_WHAT_IFS.items():
                own = tg.sight(ctx, n, **w["args"])
                assert by[key]["d_see_m"] == pytest.approx(own, abs=1e-4)
                assert by[key]["governor_m_s"] == pytest.approx(tg.governor(ctx, "two_steps", tg._mid(own), t_ref),
                                                                abs=1e-4)
                assert by[key]["change_m_s"] == pytest.approx(by[key]["governor_m_s"] - rec["base"]["governor_m_s"],
                                                              abs=1.5e-4)
            assert by["standing"]["change_m_s"] > 0 > by["full_zone"]["change_m_s"] > by["pitch_peak"]["change_m_s"]
            assert by["pitch_peak"]["change_m_s"] < by["pitch_rms"]["change_m_s"] < 0
            assert by["standoff"]["change_m_s"] < 0 and by["sensors_take_turns"]["change_m_s"] < 0
            assert set(by) >= {f"stop_{r}" for r in DOC["readings"] if r != "two_steps"}
            for t in rec["terms"]:  # each term's swing is the difference of its two ends
                assert t["swing_m_s"] == pytest.approx(abs(t["governor_m_s"][1] - t["governor_m_s"][0]), abs=1e-4)
                assert t["governor_m_s"][0] <= rec["base"]["governor_m_s"] + 1e-4 <= t["governor_m_s"][1] + 2e-4


@pytest.mark.parametrize("body", BODIES)
def test_ladder_and_bar_ledger_follow_from_the_parts(body):
    b = DOC["bodies"][body]
    stride, duty = _walk(b["H_m"], 0.40)
    lead = 0.5 * duty * stride
    assert b["foot_lead"]["at_bar_m"] == pytest.approx(lead, abs=1e-4)
    for name, (fr, rel, d) in (("nominal", (0.012, 0.55, 0.60)), ("fr05", (0.05, 0.80, 0.56))):
        assert b["walks"][name]["foot_lead_m"] == pytest.approx(0.5 * d * rel * b["H_m"], abs=1e-4)
        assert b["walks"][name]["speed_m_s"] == pytest.approx(math.sqrt(fr * G * b["H_m"]), abs=1e-4)
    for m in DOC["modes"]:
        hz = b["horizons"][m]
        for p in DOC["paths"]:
            t_lo, _, t_hi = DOC["reaction"][m][p]["total_low_ref_high_s"]
            g = b["ledger_at_bar"][m][p]
            assert g["speed_m_s"] == 0.40
            assert g["reaction_m"] == pytest.approx([0.40 * t_lo, 0.40 * t_hi], abs=1e-4)
            assert g["foot_lead_m"] == pytest.approx(lead, abs=1e-4)
            # at the bar's speed a foot already on the ground is farther ahead than either deceleration stop
            assert 0.40**2 / 3.0 < 0.40**2 / (2 * b["one_foot_decel_ms2"]) < lead
            for r in ("spec_decel", "one_foot_hold"):
                assert g["stop_m"][r] == pytest.approx(lead, abs=1e-4) and g["stop_set_by"][r] == "foot_lead"
            assert g["stop_m"]["one_step"] == pytest.approx(0.5 * stride, abs=1e-4)
            for r, n in (("two_steps", 2), ("stop_builder_worst", 2.4), ("spec_strides", 3)):
                assert g["stop_m"][r] == pytest.approx(n * g["stop_m"]["one_step"], abs=3e-4)
                assert g["stop_set_by"][r] == "reading"
            for r in DOC["readings"]:
                worst = hz["all_m"] - 0.40 * t_hi - g["stop_m"][r]
                assert g["spare_m"][r][0] == pytest.approx(worst, abs=3e-4)
                assert b["ladder"][m][p]["0.40"]["worst_spare_m"][r] == pytest.approx(worst, abs=3e-4)
                spare = [b["ladder"][m][p][f"{v:.2f}"]["worst_spare_m"][r] for v in tg.LADDER]
                assert spare == sorted(spare, reverse=True)  # less room at each faster rung
    fits = all(s >= 0 for m in DOC["modes"] for p in DOC["paths"] for row in b["ladder"][m][p].values()
               for s in row["worst_spare_m"].values())
    assert b["ladder_fits_worst_case"] is fits
    # a governor above the ladder's top and room left at every rung say the same thing
    assert (b["lowest_governor_m_s"] > max(tg.LADDER)) is fits
    assert b["lowest_governor_m_s"] > b["fastest_reference_m_s"]


def test_stack_claims_of_the_report():
    """What sections 7 and the brief say of the stacked case against the bar's 0.40 m/s."""
    for body in BODIES:
        st = DOC["bodies"][body]["stress"]
        for m in DOC["modes"]:
            for p in DOC["paths"]:
                g = {r: st[m][p]["governor"][r] for r in DOC["readings"]}
                for r in ("spec_decel", "one_foot_hold", "one_step"):  # the shorter stops keep the bar's speed
                    assert g[r]["governor_m_s"] > tg.BAR_SPEED and not g[r]["below_bar"]
                for r in ("stop_builder_worst", "spec_strides"):  # the longer ones never do
                    assert g[r]["governor_m_s"] < tg.BAR_SPEED and g[r]["below_bar"]
                assert g["two_steps"]["below_bar"] is (g["two_steps"]["governor_m_s"] < tg.BAR_SPEED)
            assert st[m]["host_planner"]["governor"]["two_steps"]["below_bar"]  # never through the host
            if m != "8x8_15hz":  # for a stop of two steps or longer 8x8 is the better mode in the stack
                for p in DOC["paths"]:
                    for r in ("two_steps", "stop_builder_worst", "spec_strides"):
                        assert st["8x8_15hz"][p]["governor"][r]["governor_m_s"] > st[m][p]["governor"][r][
                            "governor_m_s"]
    # two steps on the real-time layer: kept at 75 cm in every mode; at 60 cm in all but 4x4 at 30 Hz
    kept = {(body, m): not DOC["bodies"][body]["stress"][m][p]["governor"]["two_steps"]["below_bar"]
            for body in BODIES for m in DOC["modes"] for p in ("mcu", "rt_core")}
    assert {k for k, v in kept.items() if not v} == {("60-N-T-10", "4x4_30hz")}


def test_stop_reference_is_the_committed_record():
    for body in BODIES:
        s = DOC["bodies"][body]["stop_reference"]
        rec = _record(body, "transitions")["transitions"]["stop"]
        assert rec["from_"] == "nominal" and rec["to"] == "stand" and rec["stop"]["single_step"] is True
        assert s["speed_ramp_phase"] == rec["ramps"]["path"] == [0.0, 0.4]
        assert s["travel_after_command_strides"] == 0.2 and s["worst_phase_steps"] == 2.4
        last = rec["steps"][-1]  # the last step is half a stride, so the path ends 0.2 stride after the command
        assert (last["side"], last["lift_off_phase"]) == ("left", 0.0)
        assert last["length_m"] == pytest.approx(0.5 * rec["gaits"]["a"]["speed"] * rec["gaits"]["a"]["period"],
                                                 rel=1e-6)
        assert "2.4 steps" in DOC["readings"]["stop_builder_worst"]


def test_every_input_has_a_kind_and_every_sourced_fact_a_source():
    web = {w["id"]: w for w in DOC["web_sources"]}
    parts = {p["id"]: p for p in json.loads((ROOT / "designlab/data/parts.json").read_text())["parts"]}
    ids = [i["id"] for i in DOC["inputs"]]
    assert len(ids) == len(set(ids))
    assert set(DOC["kinds"]) == {"a", "b", "c", "d"}
    documents = ("COMPUTE.md", "BEHAVIOR_SPEC.md", "STAGE2_PLAN.md", "STAGE1_PLAN.md", "DECISION_LOG.md",
                 "designlab/data/parts.json", "designlab/build.py", "designlab/sensors.py", "designlab/gait.py",
                 "designlab/sizing.py", "designlab/envelope.py", "designlab/stage2/look.py")
    for i in DOC["inputs"]:
        assert i["kind"] in ("a", "c", "d"), i["id"]  # no closed-loop result (b) is used
        assert i["unit"] and i["source"], i["id"]
        assert "value" in i or ("low" in i and "high" in i), i["id"]
        if i["kind"] == "d":
            assert i["basis"] and ("document's" in i["basis"] or "study's" in i["basis"]), i["id"]
        if i["kind"] in ("a", "c"):
            assert i["source"] not in ("none", "derived"), i["id"]
            cited_docs = [d for d in documents if d in i["source"]]
            cited_web = [w for w in web if w in i["source"]]
            cited_records = [tok for tok in i["source"].split() if tok.endswith(".json") and "/" in tok]
            assert cited_docs or cited_web or cited_records, i["id"]
            for d in cited_docs + cited_records:
                assert (ROOT / d).exists(), (i["id"], d)
            for tok in i["source"].replace(",", " ").split():
                if tok in ("st_vl53l5cx_carrier", "pololu_vl53l8cx"):
                    assert tok in parts
    for w in web.values():
        assert w["url"].startswith("https://") and w["says"] and w["read_how"]
    by_id = {i["id"]: i for i in DOC["inputs"]}
    assert by_id["tof_rate_8x8"]["kind"] == by_id["tof_rate_4x4"]["kind"] == "c"
    assert by_id["mcu_loop"]["kind"] == by_id["walk_loop"]["kind"] == by_id["host_deadline"]["kind"] == "c"
    for d_id in ("tof_frames", "rt_core_loop", "rt_core_walk_loop", "host_planner", "host_link", "servo_delay"):
        assert by_id[d_id]["kind"] == "d", d_id  # a (d) input carries a range
        assert "low" in by_id[d_id] and "high" in by_id[d_id] and by_id[d_id]["low"] < by_id[d_id]["high"], d_id
    assert by_id["i2c_read"]["kind"] == "d" and set(by_id["i2c_read"]["value"]) == set(DOC["modes"])
    for low, ref, high in by_id["i2c_read"]["value"].values():
        assert 0.0 < low < ref < high
    assert by_id["host_planner"]["low"] == 0.0 and by_id["host_planner"]["high"] == by_id["host_deadline"]["value"]
    # every timing term has a kind per path: the walking loop is sourced on the microcontroller only
    for m in DOC["modes"]:
        for p, path in DOC["paths"].items():
            terms = DOC["reaction"][m][p]["terms"]
            assert set(terms) == set(path["terms"])
            for t, term in terms.items():
                assert term["kind"][0] in "cd", (p, t)
                assert term["kind"] == tg.term_kind(t, p)
        assert DOC["reaction"][m]["rt_core"]["terms"]["walk_loop"]["kind"] == "d"
        assert DOC["reaction"][m]["mcu"]["terms"]["walk_loop"]["kind"] == "c"
        assert DOC["reaction"][m]["host_planner"]["terms"]["walk_loop"]["kind"] == "c"
    for t in tg.TERM_LABELS:
        assert DOC["term_kinds"][t][0] in "cd"
    for r in DOC["readings"]:
        assert DOC["reading_kinds"][r][0] in "ac" or DOC["reading_kinds"][r].startswith("count c")
    # the committed records are the ones hashed in the provenance block
    for body in BODIES:
        for rec in DOC["provenance"]["records"][body].values():
            assert tg._sha(ROOT / rec["path"]) == rec["sha256"]
    # only the two ToF catalog entries are hashed, so an edit elsewhere in the catalog leaves the record valid
    prov = DOC["provenance"]
    assert "parts_json_sha256" not in prov
    assert prov["tof_catalog_entries"] == ["st_vl53l5cx_carrier", "pololu_vl53l8cx"]
    assert prov["tof_catalog_entries_sha256"] == tg.tof_catalog_sha()
    own = json.dumps([parts[i] for i in prov["tof_catalog_entries"]], sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(own.encode()).hexdigest() == prov["tof_catalog_entries_sha256"]


def test_cited_strings_are_in_the_cited_documents():
    compute = (ROOT / "COMPUTE.md").read_text()
    for text in ("| CAN loop, IMU, ToF stop, watchdogs | 200–1000 Hz |",
                 "| Locomotion policy (small network) | 50–100 Hz | Trained with an 8 ms command delay | "
                 "Microcontroller |",
                 "planner | 10–15 Hz | ≤ 200 ms from the oldest sample to a motion request",
                 "The real-time layer stays on the microcontroller",
                 "That firmware still has to be written for every option",
                 "Until it exists, a Linux crash does not end in a controlled stop"):
        assert text in compute, text
    spec = (ROOT / "BEHAVIOR_SPEC.md").read_text()
    for text in ("≤ 200 ms from the oldest sensor sample to a motion request", "stop within 1.5 strides",
                 "peak horizontal trunk acceleration ≤ 1.5 m/s²", "Trunk roll and pitch ≤ 4° RMS, ≤ 8° peak",
                 "No contact with static obstacles ≥ 0.10 m tall",
                 "Results do not transfer between computers without re-timing"):
        assert text in spec, text
    plan = (ROOT / "STAGE2_PLAN.md").read_text()
    for text in ("Stop within a step or two", "at most two touchdowns", "deceleration (1 Hz) at most 1.5 m/s²",
                 "0.30–0.50 m/s"):
        assert text in plan, text
    assert '18. **"Stopped within 1.5 strides"**' in (ROOT / "STAGE1_PLAN.md").read_text()
    parts = {p["id"]: p for p in json.loads((ROOT / "designlab/data/parts.json").read_text())["parts"]}
    assert "4x4 or 8x8 zones; 20 mm to 4 m" in parts["st_vl53l5cx_carrier"]["fov"]
    assert "about 1.7 m for the VL53L5CX" in parts["pololu_vl53l8cx"]["fov"]
    assert "15 Hz maximum at 8x8 appeared only in a search snippet and is unverified" in parts[
        "st_vl53l5cx_carrier"]["notes"]
    sensors_py = (ROOT / "designlab/sensors.py").read_text()
    for text in ("NEAR_STRIP = 0.8", "VL53L5CX equidistant (63.6 vs 65 deg)", "exposed"):
        assert text in sensors_py, text
    gait_py = (ROOT / "designlab/gait.py").read_text()
    for text in ("crouch: float = 0.03", "+ duty * stride / 2, False", "q[2] = root0[2] - spec.crouch * H"):
        assert text in gait_py, text
    for body in BODIES:  # the ToF pair is exposed on these bodies: the screen passed no ray through a window
        sen = _record(body, "sensors")
        assert sen["tof"]["tof_l"]["summary"]["window_rays"] == sen["tof"]["tof_r"]["summary"]["window_rays"] == 0.0
    # what section 8 says PR #668 (unverified) gave, in its own words and numbers
    pr = (ROOT / "stage2/jetson_plem/JETSON_PLEM_AND_SPEED.md").read_text()
    for text in ("Chest ToF safe-stop horizon", "worst-case stride-boundary safe speed", "mean-phase safe speed",
                 "`0.485 -> 0.550 m/s`", "`0.540 -> 0.610 m/s`", "`0.710 -> 0.840 m/s`", "`0.815 -> 0.900 m/s`"):
        assert text in pr, text
    pr_py = (ROOT / "stage2/jetson_plem/jetson_plem_study.py").read_text()
    for text in ("safe_stop_speed(0.133 + 0.200, 1.00)", "safe_stop_speed(0.067 + 0.045, 0.50)",
                 "(commit_strides + 0.75) * l_str", "standoff_m = 0.10"):
        assert text in pr_py, text
    assert tg.SPEC_DECEL == 1.5 and tg.BAR_SPEED == 0.40 and tg.CROUCH_FRAC == tg.sizing.NOMINAL.crouch
    assert tg.SPEC_STOP_STRIDES == 1.5 and tg.STEP_READINGS["spec_strides"] == 3.0


def test_report_tables_are_the_record():
    text = (RUN / "TOF_GOVERNOR.md").read_text()
    tables = tg.tables(DOC)
    shown = ("horizons", "terms", "reaction", "governor", "ledger", "ladder", "stress", "inputs",
             "sensitivity_terms.75-M-H-10.8x8_15hz.mcu.two_steps",
             "sensitivity_what_ifs.75-M-H-10.8x8_15hz.mcu.two_steps",
             "sensitivity_terms.75-M-H-10.8x8_15hz.host_planner.two_steps")
    for name in shown:
        for line in tables[name]:
            assert line in text, (name, line[:80])
    table_lines = {line for lines in tables.values() for line in lines}
    hand_made = [ln for ln in text.splitlines() if ln.startswith("| `") and ln not in table_lines]
    assert not hand_made, hand_made[:2]  # every data row of a table in the report is the record's
    for needle in ("tof_governor.json", "tof_governor.py", "JETSON_PLEM_REVIEW.md", "PR #668 (unverified)",
                   "In brief, for the owner", "nothing here is a hardware measurement"):
        assert needle in text, needle
    for w in DOC["web_sources"]:
        assert w["url"] in text, w["id"]
    for word in ("**ToF**", "**ST**", "**H**", "**I2C**", "**CAN**", "**IMU**", "**PR #668**", "**RMS**", "**lux**",
                 "**Jetson**", "**Pi 5**"):  # every abbreviation is explained once, under "Words used here"
        assert word in text.split("Kinds of evidence")[0], word
    assert all(len(ln) <= 120 for ln in text.splitlines() if not ln.startswith("|") and "http" not in ln)


def test_report_prose_numbers_are_the_record():
    """The numbers the brief, the prose and the draft rows quote, formatted here from the record."""
    text = (RUN / "TOF_GOVERNOR.md").read_text()
    flat = " ".join(text.split())
    brief = " ".join(text.split("## In brief, for the owner")[1].split("## 1. The question")[0].split())
    b75, b60 = (DOC["bodies"][b] for b in BODIES)

    def rng(body, mode, path, reading):
        c = DOC["bodies"][body]["governor"][mode][path][reading]
        return f"{c['low_m_s']:.2f}–{c['high_m_s']:.2f}"

    def spare(b, v, reading):
        return min(b["ladder"][m][p][f"{v:.2f}"]["worst_spare_m"][reading] for m in DOC["modes"] for p in DOC["paths"])

    def stack(b, reading, paths):
        vals = [b["stress"][m][p]["governor"][reading]["governor_m_s"] for m in DOC["modes"] for p in paths]
        return f"{min(vals):.2f}–{max(vals):.2f}"

    rt, host = ("mcu", "rt_core"), ("host_planner",)
    sens = b75["sensitivity"]["8x8_15hz"]["mcu"]["two_steps"]
    what = {w["what_if"]: w for w in sens["what_ifs"]}
    term = {t["term"]: t for t in sens["terms"]}
    host_term = {t["term"]: t for t in b75["sensitivity"]["8x8_15hz"]["host_planner"]["two_steps"]["terms"]}
    led = b75["ledger_at_bar"]["8x8_15hz"]

    def minus(x):
        return f"{x:+.2f}".replace("-", "−")

    every = tuple(DOC["paths"])
    st75 = {(r, p): [b75["stress"][m][p]["governor"][r]["governor_m_s"] for m in DOC["modes"]]
            for r in DOC["readings"] for p in every}
    longer = st75["two_steps", "host_planner"] + [v for r in ("stop_builder_worst", "spec_strides") for p in every
                                                  for v in st75[r, p]]
    shorter = [v for r in ("spec_decel", "one_foot_hold", "one_step") for p in every for v in st75[r, p]]
    long_lo, long_hi, short_lo, short_hi = min(longer), max(longer), min(shorter), max(shorter)
    assert long_hi < tg.BAR_SPEED < short_lo
    in_brief = [
        f"the lowest governor is {b75['lowest_governor_m_s']:.2f} m/s at 75 cm and {b60['lowest_governor_m_s']:.2f} "
        "m/s at 60 cm",
        f"({b75['fastest_reference_m_s']:.3f} / {b60['fastest_reference_m_s']:.3f} m/s)",
        f"would allow {rng(BODIES[0], '8x8_15hz', 'mcu', 'two_steps')} m/s",
        f"and {rng(BODIES[0], '8x8_15hz', 'host_planner', 'two_steps')} m/s when the host's planner does",
        f"At 60 cm: {rng(BODIES[1], '8x8_15hz', 'mcu', 'two_steps')} and "
        f"{rng(BODIES[1], '8x8_15hz', 'host_planner', 'two_steps')} m/s",
        f"gives {rng(BODIES[0], '8x8_15hz', 'mcu', 'stop_builder_worst')} and "
        f"{rng(BODIES[0], '8x8_15hz', 'host_planner', 'stop_builder_worst')} m/s at 75 cm",
        f"gives {rng(BODIES[0], '8x8_15hz', 'mcu', 'spec_strides')} and "
        f"{rng(BODIES[0], '8x8_15hz', 'host_planner', 'spec_strides')} m/s "
        f"({rng(BODIES[1], '8x8_15hz', 'mcu', 'spec_strides')} and "
        f"{rng(BODIES[1], '8x8_15hz', 'host_planner', 'spec_strides')} m/s at 60 cm)",
        f"up to {b75['highest_governor_m_s']:.2f} m/s",
        f"see the floor {led['mcu']['d_see_m'][0]:.2f}–{led['mcu']['d_see_m'][1]:.2f} m ahead of the toes",
        f"covers {led['mcu']['reaction_m'][0]:.2f}–{led['mcu']['reaction_m'][1]:.2f} m while it reacts on the "
        f"microcontroller path, or {led['host_planner']['reaction_m'][0]:.2f}–"
        f"{led['host_planner']['reaction_m'][1]:.2f} m on the host path",
        f"covers {min(led['mcu']['stop_m'].values()):.2f}–{max(led['mcu']['stop_m'].values()):.2f} m while it stops",
        f"at least {spare(b75, 0.40, 'two_steps'):.2f} m is left over on any path and in any ToF mode "
        f"({spare(b60, 0.40, 'two_steps'):.2f} m at 60 cm)",
        f"at least {spare(b75, 0.40, 'spec_strides'):.2f} m is left ({spare(b60, 0.40, 'spec_strides'):.2f} m at "
        "60 cm)",
        f"at least {spare(b75, 0.50, 'two_steps'):.2f} m is left with two steps and "
        f"{spare(b75, 0.50, 'spec_strides'):.2f} m with 1.5 strides ({spare(b60, 0.50, 'two_steps'):.2f} and "
        f"{spare(b60, 0.50, 'spec_strides'):.2f} m at 60 cm",
        f"from {b75['governor']['8x8_15hz']['mcu']['two_steps']['low_m_s']:.2f} to "
        f"{b75['governor']['8x8_15hz']['host_planner']['two_steps']['low_m_s']:.2f} m/s",
        f"two-step stop, {sens['base']['governor_m_s']:.2f} m/s at the reference values",
        f"{what['stop_one_foot_hold']['governor_m_s']:.2f} m/s if the stance foot alone brakes the robot",
        f"{what['stop_stop_builder_worst']['governor_m_s']:.2f} m/s with today's stop builder at its worst phase, "
        f"{what['stop_spec_strides']['governor_m_s']:.2f} m/s with the spec's 1.5 strides",
        f"peak limit of 8°: {minus(what['pitch_peak']['change_m_s'])} m/s; by its RMS limit of 4°: "
        f"{minus(what['pitch_rms']['change_m_s'])} m/s",
        f"not trusting the far row of zones: {minus(what['second_row']['change_m_s'])} m/s",
        f"1 to 3: {term['tof_frames']['swing_m_s']:.2f} m/s",
        f"0–200 ms is larger: {host_term['host_planner']['swing_m_s']:.2f} m/s",
        f"falls to {stack(b75, 'two_steps', rt)} m/s on the microcontroller and real-time-core paths and to "
        f"{stack(b75, 'two_steps', host)} m/s on the host path",
        f"to {stack(b75, 'stop_builder_worst', rt)} and {stack(b75, 'stop_builder_worst', host)} m/s; with the "
        f"spec's 1.5 strides, to {stack(b75, 'spec_strides', rt)} and {stack(b75, 'spec_strides', host)} m/s",
        f"to {stack(b75, 'spec_decel', rt)} and {stack(b75, 'spec_decel', host)} m/s",
        f"({b60['stress']['4x4_30hz']['mcu']['governor']['two_steps']['governor_m_s']:.2f} m/s)",
        f"it needs {b75['horizons']['8x8_15hz']['far_row_fill_height_m']:.3f} m",
        f"needs {b75['horizons']['4x4_60hz']['far_row_fill_height_m']:.3f} m to be filled",
    ]
    for needle in in_brief:
        assert needle in brief, needle
    small = max(term[t]["swing_m_s"] for t in ("servo_delay", "i2c_read", "walk_loop", "mcu_loop"))
    assert f"move the governor by about {small:.2f} m/s or less across their whole range" in brief
    assert small < 0.2 * term["tof_frames"]["swing_m_s"]  # the computer's terms are small beside the sensor's
    anywhere = [
        f"{b75['horizons']['8x8_15hz']['all_m']:.3f}–{b75['horizons']['8x8_15hz']['far_m']:.3f} m instead of "
        f"{b75['horizons']['8x8_15hz']['home_all_m']:.3f}–{b75['horizons']['8x8_15hz']['home_far_m']:.3f} m",
        f"{b75['walks']['nominal']['foot_lead_m']:.3f} m in the nominal walk, {b75['foot_lead']['at_bar_m']:.3f} m at "
        f"0.40 m/s and {b75['walks']['fr05']['foot_lead_m']:.3f} m in the Froude 0.05 walk at 75 cm "
        f"({b60['walks']['nominal']['foot_lead_m']:.3f}, {b60['foot_lead']['at_bar_m']:.3f} and "
        f"{b60['walks']['fr05']['foot_lead_m']:.3f} m at 60 cm)",
        f"only below {b75['foot_lead']['binds_below_m_s']['spec_decel']:.2f} m/s at 75 cm and "
        f"{b60['foot_lead']['binds_below_m_s']['spec_decel']:.2f} m/s at 60 cm",
        f"{b75['one_foot_decel_ms2']:.2f} m/s² at 75 cm with the bigger feet, {b60['one_foot_decel_ms2']:.2f} m/s² "
        "at 60 cm",
        f"to {what['pitch_rms']['d_see_m'][0]:.3f}–{what['pitch_rms']['d_see_m'][1]:.3f} m at 4° and to "
        f"{what['pitch_peak']['d_see_m'][0]:.3f}–{what['pitch_peak']['d_see_m'][1]:.3f} m at 8°",
        f"section 6 prices it ({minus(what['standoff']['change_m_s'])} m/s)",
        f"the governor would fall by {abs(what['sensors_take_turns']['change_m_s']):.2f} m/s",
        f"the spec's 1.5 strides ({rng(BODIES[0], '8x8_15hz', 'host_planner', 'spec_strides')} m/s at 75 cm, "
        f"{rng(BODIES[1], '8x8_15hz', 'host_planner', 'spec_strides')} m/s at 60 cm)",
        f"leaves at least {spare(b75, 0.40, 'two_steps'):.2f} m (75 cm) and {spare(b60, 0.40, 'two_steps'):.2f} m "
        f"(60 cm) of its sight distance unused with a two-step stop, and {spare(b75, 0.40, 'spec_strides'):.2f} m and "
        f"{spare(b60, 0.40, 'spec_strides'):.2f} m with the spec's 1.5-stride stop",
        f"leaves at least {spare(b75, 0.50, 'two_steps'):.2f} m (75 cm) and {spare(b60, 0.50, 'two_steps'):.2f} m "
        f"(60 cm) at the top rung with a two-step stop, and {spare(b75, 0.50, 'spec_strides'):.2f} m and "
        f"{spare(b60, 0.50, 'spec_strides'):.2f} m with the spec's 1.5-stride stop",
        "the spec's 1.5 strides " + " / ".join(f"{b['lowest_by_reading_m_s']['spec_strides']:.2f}" for b in (b75, b60)),
        "two steps " + " / ".join(f"{b['lowest_by_reading_m_s']['two_steps']:.2f}" for b in (b75, b60)),
        # the draft finding row
        f"the lowest governor is {b75['lowest_governor_m_s']:.2f} m/s at 75 cm and {b60['lowest_governor_m_s']:.2f} "
        "m/s at 60 cm (the spec's 1.5-stride stop through the host's planner)",
        f"it is {rng(BODIES[0], '8x8_15hz', 'mcu', 'two_steps')} m/s at 75 cm and "
        f"{rng(BODIES[1], '8x8_15hz', 'mcu', 'two_steps')} m/s at 60 cm when the real-time layer decides the stop",
        f"the 75 cm governor is {stack(b75, 'two_steps', rt)} m/s with a two-step stop decided by the real-time layer, "
        f"{long_lo:.2f}–{long_hi:.2f} m/s with a two-step stop through the host or a longer stop on any path, and "
        f"{short_lo:.2f}–{short_hi:.2f} m/s with one step or a stop as hard as the spec allows",
    ]
    for needle in anywhere:
        assert needle in flat, needle
    for b in (b75, b60):  # what the brief and the draft row say of the lowest cell
        assert b["lowest_governor_at"] == dict(mode="4x4_30hz", path="host_planner", reading="spec_strides")
    # PR #668's (unverified) pairs beside this document's host path with 8x8 zones (section 8)
    for body, worst, mean in ((BODIES[0], (0.540, 0.610), (0.815, 0.900)), (BODIES[1], (0.485, 0.550), (0.710, 0.840))):
        g = DOC["bodies"][body]["governor"]["8x8_15hz"]["host_planner"]
        st = DOC["bodies"][body]["stress"]["8x8_15hz"]["host_planner"]["governor"]
        assert worst[0] < g["spec_strides"]["low_m_s"] < worst[1] < g["spec_strides"]["high_m_s"]  # straddles
        assert worst[0] > max(s["governor_m_s"] for r, s in st.items() if r in STEP_READINGS[1:])
        assert g["two_steps"]["low_m_s"] < mean[0] < mean[1] < g["two_steps"]["high_m_s"]  # inside
